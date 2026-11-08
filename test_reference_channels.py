#!/usr/bin/env python3
"""
Tests the REFERENCE_CHANNELS mechanism (currently just "event"): a
read-only track shown alongside whatever the RA chose to review via
--channels, never user-selectable, never seeded with annotations, never
appearing in the exported output. Added after the user's request to
"display the event markers below the chosen signal."

Item 11 (2026-11-08): a mark tied to one row by Shift+click is still saved
for its label's channel, whatever read-only or other row it's tied to (it
used to be left out silently in Step 2), and the summary says so.

Run with: annotate_env\\Scripts\\python.exe test_reference_channels.py
"""

import shutil
import tempfile
from unittest.mock import patch

import mne
import numpy as np

from annotation_io import (
    combine_with_reference_channels,
    compute_event_transitions,
    export_annotations,
    run_stage_b,
    seed_annotations,
    single_row_mark_lines,
)
from channel_config import CHANNELS, FULL_RANGE_CHANNELS, GUIDE_CHANNELS, REFERENCE_CHANNELS
from physio_io import build_raw, compute_scalings, generate_synthetic_demo
from reconcile import _agree_label, export_reconciled

import session_summary_gui  # noqa: E402

# The session-summary dialog is modal; answer "Save" immediately so these
# non-interactive tests never block (tests needing other answers patch it).
session_summary_gui.show_session_summary = session_summary_gui.headless_decision()


def main():
    df, sfreq = generate_synthetic_demo(duration_sec=10)
    assert "event" not in df.columns, "synthetic demo data shouldn't have an event column"

    print("1. combine_with_reference_channels() is a no-op when 'event' isn't in the data...")
    combined = combine_with_reference_channels(CHANNELS, df)
    assert combined == CHANNELS
    assert "event" not in combined
    print("   OK: unchanged when the column is absent (covers every file that predates the PPG/event fix)")

    print("2. combine_with_reference_channels() adds 'event' when it IS present, even with a restricted channel set...")
    df_with_event = df.copy()
    # phase codes like the real physioProcess output: 0 outside task, else 20/30/40/50/65
    rng = np.random.default_rng(0)
    df_with_event["event"] = rng.choice([0, 20, 30, 40, 50, 65], size=len(df_with_event), p=[0.05, 0.19, 0.19, 0.19, 0.19, 0.19])

    restricted = {"ecg": CHANNELS["ecg"]}  # simulating --channels ecg
    combined = combine_with_reference_channels(restricted, df_with_event)
    assert set(combined.keys()) == {"ecg", "event"}
    assert combined["event"] == REFERENCE_CHANNELS["event"]
    print(f"   OK: {list(combined.keys())} -- 'event' shown even though only 'ecg' was requested")

    print("3. build_raw() includes 'event' as a 'stim'-type channel...")
    raw = build_raw(df_with_event, combined, sfreq)
    assert "event" in raw.ch_names
    assert raw.get_channel_types(picks="event")[0] == "stim"
    print(f"   OK: channels={raw.ch_names}, event type={raw.get_channel_types(picks='event')[0]}")

    print("4. compute_scalings() handles 'event' without NaN or crashing...")
    scalings = compute_scalings(df_with_event, combined)
    assert not any(np.isnan(v) for v in scalings.values())
    assert "stim" in scalings
    print(f"   OK: scalings={ {k: round(v, 4) for k, v in scalings.items()} }")

    print("5. seed_annotations() never seeds anything for 'event' (annotation_mode='none')...")
    raw = seed_annotations(raw, df_with_event, combined, {}, sfreq)
    event_annotations = [a for a in raw.annotations if "event" in (a.get("ch_names") or ())]
    assert len(event_annotations) == 0
    print(f"   OK: {len(raw.annotations)} total seeded annotations (all from 'ecg', none from 'event')")

    print("6. export_annotations() produces NO entry for 'event' -- nothing for an RA to correct there...")
    channel_outputs = export_annotations(raw, df_with_event, combined, sfreq)
    assert "event" not in channel_outputs
    assert set(channel_outputs.keys()) == {"ecg"}
    print(f"   OK: exported channels = {list(channel_outputs.keys())} (event correctly excluded)")

    print("7. Full CHANNELS + event together (the common real-world case) all work at once...")
    combined_full = combine_with_reference_channels(CHANNELS, df_with_event)
    raw_full = build_raw(df_with_event, combined_full, sfreq)
    scalings_full = compute_scalings(df_with_event, combined_full)
    assert not any(np.isnan(v) for v in scalings_full.values())
    # Intersected with df_with_event's actual columns, not all of CHANNELS.keys() --
    # channels defined in channel_config.py but absent from this particular file
    # (e.g. sbp/dbp/temperature, not present in synthetic demo data) are correctly
    # excluded by build_raw(), same as any other not-yet-collected channel.
    assert set(raw_full.ch_names) == (set(CHANNELS.keys()) & set(df_with_event.columns)) | {"event"}
    print(f"   OK: channels={raw_full.ch_names}")

    print("8. compute_event_transitions() finds every value-change point, matching a realistic "
          "block-structured task-phase column (0 -> 20 -> 30 -> 40 -> 50 -> 65 -> 0, like real "
          "physioProcess output) -- for raw.plot(events=...)'s read-only, non-interactive, "
          "text-labeled markers (added per the user's 'show the event number' wishlist item)...")
    block_df = df.copy()
    block_event = np.zeros(len(block_df), dtype=int)
    block_len = len(block_df) // 7
    for i, code in enumerate([20, 30, 40, 50, 65]):
        block_event[(i + 1) * block_len:(i + 2) * block_len] = code
    block_df["event"] = block_event

    events_array = compute_event_transitions(block_df, "event", sfreq)
    assert events_array is not None
    assert events_array.shape[1] == 3, "must match mne.find_events()'s (N, 3) shape"
    assert list(events_array[:, 2]) == [20, 30, 40, 50, 65, 0], (
        f"expected the 5 codes then the drop back to 0, got {list(events_array[:, 2])}"
    )
    assert list(events_array[:, 1]) == [0] * len(events_array), "middle column must be 0, per mne's convention"
    assert list(events_array[:, 0]) == [block_len * (i + 1) for i in range(6)], (
        "onset sample indices must exactly match where the value actually changes"
    )
    print(f"   OK: {len(events_array)} transitions found, codes {list(events_array[:, 2])}")

    print("9. compute_event_transitions() returns None when there's nothing to mark...")
    assert compute_event_transitions(df, "event", sfreq) is None, "no 'event' column at all"
    constant_df = df.copy()
    constant_df["event"] = 0
    assert compute_event_transitions(constant_df, "event", sfreq) is None, "constant value, zero transitions"
    print("   OK: None for a missing column and for a column that never changes")

    print("10. run_stage_b() passes the computed events array through to raw.plot() when 'event' "
          "is present, and omits it entirely (no crash, no empty-events edge case) when absent...")
    captured = {}

    def capture_plot_kwargs(self, *args, **kwargs):
        captured.update(kwargs)

    tmp_dir = tempfile.mkdtemp(prefix="reference_channels_stageb_")
    try:
        with patch.object(mne.io.RawArray, "plot", capture_plot_kwargs):
            run_stage_b(block_df, sfreq, {"ecg": CHANNELS["ecg"]}, "hlu", tmp_dir, "event_test", "synthetic")
        assert "events" in captured, "expected raw.plot() to receive an 'events' kwarg"
        assert list(captured["events"][:, 2]) == [20, 30, 40, 50, 65, 0]
        print(f"   OK: raw.plot() received events for {len(captured['events'])} transitions")

        captured.clear()
        with patch.object(mne.io.RawArray, "plot", capture_plot_kwargs):
            run_stage_b(df, sfreq, {"ecg": CHANNELS["ecg"]}, "hlu", tmp_dir, "no_event_test", "synthetic")
        assert "events" not in captured, "no 'event' column -> no 'events' kwarg should be passed at all"
        print("   OK: no 'events' kwarg passed when there's no event column to derive it from")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("11. A mark tied to one row (Shift+click) is saved for its label's channel, whatever row it's "
          "tied to, in Step 2 and Step 3, and the summary says so...")
    tied_df = df_with_event.copy()
    tied_df["emg_cor"] = np.random.default_rng(1).normal(size=len(tied_df))  # real-looking COR, so it has data
    # Every kind of row at once: reviewed ECG with the PPG guide, reviewed COR with its full-range row,
    # reviewed RSP with the EDA guide, and the event row.
    rows = {"ecg": CHANNELS["ecg"], "ppg": GUIDE_CHANNELS["ppg"], "emg_cor": CHANNELS["emg_cor"],
            "emg_cor_full": FULL_RANGE_CHANNELS["emg_cor_full"], "rsp": CHANNELS["rsp"],
            "eda": GUIDE_CHANNELS["eda"], "event": REFERENCE_CHANNELS["event"]}
    tied_raw = build_raw(tied_df, rows, sfreq)
    assert tied_raw.ch_names == list(rows), tied_raw.ch_names
    marks = [  # (onset s, duration s, label, the one row it's tied to, or None for an ordinary mark)
        (0.5, 0.5, "bad_rsp", None),
        (1.5, 0.5, "bad_rsp", "rsp"),            # its own row
        (2.5, 0.5, "bad_rsp", "eda"),            # a guide row
        (3.5, 0.5, "bad_rsp", "ecg"),            # another reviewed channel's row
        (4.5, 0.5, "bad_emg_cor", "emg_cor_full"),  # the full-range row
        (5.5, 0.5, "bad_emg_cor", "event"),      # the event row
        (6.5, 0.3, "peak_ecg", "ppg"),           # a drawn peak tied to the PPG guide row
    ]
    for onset, duration, label, row in marks:
        tied_raw.annotations.append(onset, duration, label, ch_names=[[row] if row else []])
    out = export_annotations(tied_raw, tied_df, rows, sfreq)
    want_rsp = [[500, 1000], [1500, 2000], [2500, 3000], [3500, 4000]]
    assert sorted(out["rsp"]["bad_segments"]) == want_rsp, out["rsp"]["bad_segments"]
    assert sorted(out["emg_cor"]["bad_segments"]) == [[4500, 5000], [5500, 6000]], out["emg_cor"]["bad_segments"]
    assert len(out["ecg"]["indices"]) == 1 and 6500 <= out["ecg"]["indices"][0] <= 6900, out["ecg"]["indices"]
    assert set(out) == {"ecg", "emg_cor", "rsp"}, "read-only rows never get an entry of their own"
    lines = single_row_mark_lines(tied_raw)
    assert lines == ["6 mark(s) tied to one row (Shift+click): saved as ordinary marks covering every row"], lines
    plain_raw = build_raw(tied_df, rows, sfreq)
    plain_raw.annotations.append(0.5, 0.5, "bad_rsp")
    assert single_row_mark_lines(plain_raw) == [], "no line when no mark is tied to a row"
    print(f"   OK (Step 2): all {len(marks)} marks saved (rsp {len(want_rsp)}, emg_cor 2, ecg 1); summary: {lines[0]}")

    step3_raw = build_raw(tied_df, rows, sfreq)
    step3_raw.annotations.append(2.5, 0.5, _agree_label("bad_rsp"), ch_names=[["eda"]])
    step3_raw.annotations.append(4.5, 0.5, _agree_label("bad_emg_cor"), ch_names=[["emg_cor_full"]])
    step3 = export_reconciled(step3_raw, tied_df, rows, sfreq, "aa", "bb")
    assert step3["rsp"]["bad_segments"] == [[2500, 3000]], step3["rsp"]
    assert step3["emg_cor"]["bad_segments"] == [[4500, 5000]], step3["emg_cor"]
    assert len(single_row_mark_lines(step3_raw)) == 1
    print("   OK (Step 3): tied marks saved the same way, and the same summary line")

    print("\nALL REFERENCE CHANNEL TESTS PASSED.")


if __name__ == "__main__":
    main()
