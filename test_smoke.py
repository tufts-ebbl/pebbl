#!/usr/bin/env python3
"""
Non-interactive smoke test for the annotation plumbing (data load -> Raw
construction -> seeding -> simulated RA edits -> export -> JSON round-trip).

This does NOT test the interactive click/drag behavior of MNE's browser
itself (that requires an actual human at the keyboard) -- it verifies
everything around it: that a real RA session would produce a well-formed
JSON with correctly snapped peak indices and correctly bounded bad segments.

Run with: annotate_env\\Scripts\\python.exe test_smoke.py
"""

import glob
import io
import json
import os
import shutil
import tempfile
from contextlib import redirect_stdout
from unittest.mock import patch

import mne
import numpy as np

from annotation_io import (
    _merge_warning_threshold,
    check_interval_regularity,
    export_annotations,
    load_saved_annotations,
    run_stage_b,
    save_annotation_json,
    seed_annotations,
)
from channel_config import CHANNELS
from physio_io import (
    build_raw,
    compute_channel_value_ranges,
    compute_scalings,
    format_channel_value_ranges,
    generate_synthetic_demo,
)

import session_summary_gui  # noqa: E402

# The session-summary dialog is modal; answer "Save" immediately so these
# non-interactive tests never block (tests needing other answers patch it).
session_summary_gui.show_session_summary = session_summary_gui.headless_decision()


def main():
    print("1. Generating short synthetic demo data (10s)...")
    df, sfreq = generate_synthetic_demo(duration_sec=10)
    assert set(["ecg", "rsp", "ppg", "eda", "emg_cor", "emg_zyg"]).issubset(df.columns)
    print(f"   OK: {len(df)} samples at {sfreq} Hz, columns: {list(df.columns)}")

    print("2. Confirming EMG placeholders are literally flat (all zero)...")
    for ch in ["emg_cor", "emg_zyg"]:
        assert (df[ch] == 0).all(), f"{ch} is not exactly zero throughout"
    print("   OK")

    print("3. Building MNE Raw object...")
    raw = build_raw(df, CHANNELS, sfreq)
    assert list(raw.ch_names) == [ch for ch in CHANNELS if ch in df.columns]
    print(f"   OK: channels = {raw.ch_names}, types = {raw.get_channel_types()}")

    print("3a. EDA is MNE's native 'gsr' type, labeled in real microsiemens ('µS'), "
          "not the generic unitless 'misc'/AU or MNE's own default whole-Siemens label -- "
          "the data values themselves are untouched (unit_scalings stays at MNE's default "
          "1.0, since the data is already numerically in µS; only the display STRING "
          "changes). See channel_config.py's DEFAULTS override for why this is needed...")
    assert CHANNELS["eda"]["mne_type"] == "gsr"
    assert raw.get_channel_types(picks="eda")[0] == "gsr"
    assert mne.defaults.DEFAULTS["units"]["gsr"] == "µS", (
        f"expected the microsiemens override to be in effect; got "
        f"{mne.defaults.DEFAULTS['units']['gsr']!r} (MNE's own un-overridden default is 'S')"
    )
    assert mne.defaults.DEFAULTS["scalings"]["gsr"] == 1.0, (
        "the unit_scalings multiplier must stay at 1.0 -- only the label changed, not the data"
    )
    print(f"   OK: eda -> gsr, unit={mne.defaults.DEFAULTS['units']['gsr']!r}, "
          f"unit_scalings={mne.defaults.DEFAULTS['scalings']['gsr']}")

    print("3a2. Every mne_type actually used in this project (ecg/resp/bio/gsr/stim) has a "
          "registered unit -- regression check for a live crash: hovering the crosshair over "
          "the 'event' (stim-type) reference channel called mne_qt_browser's "
          "_get_channel_scaling(), which KeyErrors on any type missing from MNE's own "
          "defaults (stim, like resp before it, was never registered at all)...")
    from mne_qt_browser._utils import _get_channel_scaling
    fake_widget = type("FakeWidget", (), {"mne": type("Mne", (), {
        "butterfly": False, "scalings": {"stim": 1.0}, "unit_scalings": mne.defaults.DEFAULTS["scalings"],
        "scale_factor": 1.0,
    })()})()
    for mne_type in ("ecg", "resp", "bio", "gsr", "stim"):
        assert mne_type in mne.defaults.DEFAULTS["units"], f"{mne_type} missing from units defaults"
        assert mne_type in mne.defaults.DEFAULTS["scalings"], f"{mne_type} missing from scalings defaults"
    _ = _get_channel_scaling(fake_widget, "stim")  # would raise KeyError before the fix
    print("   OK: no KeyError for any in-use channel type, including 'stim'")

    print("3b. Checking display scalings keep EMG placeholders visibly flat...")
    scalings = compute_scalings(df, CHANNELS)
    non_emg_scales = [v for k, v in scalings.items() if k != "emg"]
    median_other = float(np.median(non_emg_scales))
    # EMG's scale should be in the same ballpark as the real channels (not
    # auto-fit to its own tiny amplitude), so the placeholder's actual data
    # ends up a tiny fraction of that scale and renders as flat.
    assert 0.1 * median_other < scalings["emg"] < 10 * median_other, (
        f"EMG scale {scalings['emg']} isn't comparable to other channels' scales "
        f"(median {median_other}) -- it may have been auto-fit to its own tiny "
        f"data again, which would make the placeholder look busy, not flat."
    )
    emg_amplitude_fraction = df["emg_cor"].std() / scalings["emg"]
    assert emg_amplitude_fraction < 0.01, (
        f"EMG placeholder's amplitude is {emg_amplitude_fraction:.4f} of its display "
        f"scale -- too large a fraction to read as an obviously flat line."
    )
    print(f"   OK: emg scale={scalings['emg']:.4g} (median of other channels={median_other:.4g}), "
          f"placeholder amplitude is only {emg_amplitude_fraction:.5f} of that scale")

    print("3c. Checking compute_scalings is NaN-safe (real PPG data was found to be ~91% NaN)...")
    df_nan = df.copy()
    n = len(df_nan)
    df_nan.loc[: int(0.9 * n), "ppg"] = np.nan  # mimic real dropped-signal data
    scalings_nan = compute_scalings(df_nan, CHANNELS)
    assert not any(np.isnan(v) for v in scalings_nan.values()), f"NaN leaked into scalings: {scalings_nan}"
    print(f"   OK: 90%-NaN ppg column -> bio scale = {scalings_nan['bio']:.4g} (no NaN)")

    df_all_nan = df.copy()
    df_all_nan["ppg"] = np.nan
    scalings_all_nan = compute_scalings(df_all_nan, CHANNELS)
    assert not any(np.isnan(v) for v in scalings_all_nan.values())
    assert scalings_all_nan["bio"] == scalings_all_nan["ecg"] or scalings_all_nan["bio"] > 0
    print(f"   OK: fully-NaN ppg column -> borrows a sane scale instead of NaN "
          f"(bio={scalings_all_nan['bio']:.4g})")

    print("3d. Checking a channel with REAL (non-degenerate) data gets its OWN scale, "
          "not a borrowed one -- forward-compat for when real EMG/PPG replaces a placeholder...")
    df_real_emg = df.copy()
    rng = np.random.default_rng(0)
    df_real_emg["emg_cor"] = rng.normal(loc=0.0, scale=123.0, size=len(df_real_emg))  # a "real" EMG-like signal
    scalings_real_emg = compute_scalings(df_real_emg, CHANNELS)
    expected_own_scale = (np.percentile(df_real_emg["emg_cor"], 97.5) - np.percentile(df_real_emg["emg_cor"], 2.5)) / 2
    assert np.isclose(scalings_real_emg["emg"], expected_own_scale, rtol=0.05), (
        f"Expected EMG's own auto-fit scale (~{expected_own_scale:.4g}) once it has real data, "
        f"got {scalings_real_emg['emg']:.4g} -- looks like it's still being borrowed."
    )
    print(f"   OK: real-valued emg_cor data -> its own auto-fit scale ({scalings_real_emg['emg']:.4g}), not borrowed")

    print("4. Seeding point annotations from automated peak columns (eda is segment-mode "
          "now -- see channel_config.py -- so its eda_peaks column is deliberately NOT "
          "seeded here, unlike ecg/ppg). Segment-mode channels present (rsp/eda/emg_cor/"
          "emg_zyg) each get one zero-duration placeholder annotation too, registering "
          "their label so mne-qt-browser's description picker isn't empty on a channel "
          "with no real segments yet...")
    raw = seed_annotations(raw, df, CHANNELS, {}, sfreq)
    n_seeded = len(raw.annotations)
    n_expected_points = int(df["ecg_peaks"].sum() + df["ppg_peaks"].sum())
    # rsp, eda, emg_cor, emg_zyg -- the segment-mode channels present here (rsp since 2026-09-25)
    n_expected_segment_placeholders = 4
    n_expected = n_expected_points + n_expected_segment_placeholders
    assert n_seeded == n_expected, f"seeded {n_seeded} annotations, expected {n_expected}"
    zero_duration_descriptions = sorted(
        ann["description"] for ann in raw.annotations if ann["duration"] == 0
        and ann["description"] not in ("peak_ecg", "peak_rsp", "peak_ppg")
    )
    assert zero_duration_descriptions == ["bad_eda", "bad_emg_cor", "bad_emg_zyg", "bad_rsp"], (
        f"expected exactly one placeholder per present segment-mode channel; got {zero_duration_descriptions}"
    )
    print(f"   OK: {n_seeded} seeded annotations total "
          f"(ecg={df['ecg_peaks'].sum()}, "
          f"ppg={df['ppg_peaks'].sum()}, plus {n_expected_segment_placeholders} segment placeholders: "
          f"{zero_duration_descriptions})")

    print("4b. Confirming seeded annotations are NOT channel-scoped (regression check -- "
          "channel-scoping was a confirmed trigger for a live mne-qt-browser crash "
          "when editing densely-seeded peaks; see HANDOFF.md's debugging history)...")
    scoped = [a for a in raw.annotations if a.get("ch_names")]
    assert not scoped, f"{len(scoped)} seeded annotation(s) are channel-scoped -- this must not regress"
    print("   OK: none of the seeded annotations carry ch_names")

    print("5. Simulating an RA session: delete one auto peak, add one imprecise "
          "click near a true ECG peak, and mark one bad EMG segment...")
    # Simulate deletion by dropping the first seeded annotation
    remaining = raw.annotations[1:]
    raw.set_annotations(remaining)

    # Simulate an RA clicking near (not exactly on) a true ECG peak
    true_ecg_peaks = np.where(df["ecg_peaks"].to_numpy())[0]
    target_peak = true_ecg_peaks[len(true_ecg_peaks) // 2]
    sloppy_click_idx = target_peak - 7  # a few samples off, like an imprecise click

    extra_point = mne.Annotations(
        onset=[sloppy_click_idx / sfreq], duration=[0.0],
        description=[CHANNELS["ecg"]["point_label"]], ch_names=[("ecg",)],
    )
    raw.set_annotations(raw.annotations + extra_point)

    bad_segment = mne.Annotations(
        onset=[2.0], duration=[0.5],
        description=[CHANNELS["emg_cor"]["segment_label"]], ch_names=[("emg_cor",)],
    )
    raw.set_annotations(raw.annotations + bad_segment)
    print(f"   OK: raw now has {len(raw.annotations)} total annotations")

    print("6. Exporting annotations (this is where snap-to-local-max happens)...")
    channel_outputs = export_annotations(raw, df, CHANNELS, sfreq)

    assert target_peak in channel_outputs["ecg"]["indices"], (
        f"Expected the sloppy click at {sloppy_click_idx} to snap to the true peak "
        f"at {target_peak}, but it did not."
    )
    print(f"   OK: sloppy click at index {sloppy_click_idx} correctly snapped to "
          f"true local max at index {target_peak}")

    assert channel_outputs["emg_cor"]["bad_segments"] == [[2000, 2500]], (
        f"Expected [[2000, 2500]], got {channel_outputs['emg_cor']['bad_segments']}"
    )
    print(f"   OK: EMG corrugator bad segment correctly exported as "
          f"{channel_outputs['emg_cor']['bad_segments']} (samples)")

    assert channel_outputs["emg_zyg"]["bad_segments"] == []
    print("   OK: EMG zygomatic (untouched) correctly has no bad segments")

    print("6b. A WIDE click-drag (the RA drags a region AROUND a peak, starting well "
          "before it and ending after it) must find the true peak anywhere WITHIN that "
          "drawn region, not just near its LEFT EDGE (regression check -- confirmed live: "
          "mne_qt_browser stores a drag's onset as its left edge (rgn[0]), never its "
          "center; a fixed-width snap anchored there missed the true peak whenever the "
          "drag was wider than that fixed window, which real single-peak drags commonly "
          "are (documented elsewhere in this file as reaching up to ~350ms) -- found "
          "against real sub-085 data as a systematic ~100-120ms offset on ~25-30 peaks; "
          "see HANDOFF.md)...")
    true_ecg_peaks_2 = np.where(df["ecg_peaks"].to_numpy())[0]
    target_peak_2 = true_ecg_peaks_2[len(true_ecg_peaks_2) // 2 + 5]
    # Drag starts 150ms before the true peak, ends 150ms after -- the true peak sits
    # at the drag's CENTER, well outside the default 50ms snap window if anchored on
    # the drag's onset (left edge) instead.
    drag_start_sample = target_peak_2 - int(0.15 * sfreq)
    wide_drag = mne.Annotations(
        onset=[drag_start_sample / sfreq], duration=[0.30],
        description=[CHANNELS["ecg"]["point_label"]], ch_names=[("ecg",)],
    )
    raw_wide_drag = build_raw(df, CHANNELS, sfreq)
    raw_wide_drag.set_annotations(wide_drag)
    wide_drag_outputs = export_annotations(raw_wide_drag, df, CHANNELS, sfreq)
    assert target_peak_2 in wide_drag_outputs["ecg"]["indices"], (
        f"expected the true peak at {target_peak_2} to be found within the drawn "
        f"[{drag_start_sample}, {drag_start_sample + int(0.30*sfreq)}] drag region; "
        f"got {wide_drag_outputs['ecg']['indices']}"
    )
    print(f"   OK: wide drag starting {int(0.15*sfreq)} samples before the true peak "
          f"still correctly found it at {target_peak_2}")

    print("7. Saving JSON and reading it back...")
    tmp_dir = tempfile.mkdtemp(prefix="annotation_smoke_")
    try:
        out_path = save_annotation_json(tmp_dir, "smoke_test", "xyz", channel_outputs, sfreq, "synthetic")
        assert os.path.basename(out_path) == "smoke_test_annotations_xyz.json"
        with open(out_path) as f:
            reloaded = json.load(f)
        assert reloaded["initials"] == "xyz"
        assert reloaded["channels"]["ecg"]["mode"] == "point"
        assert reloaded["channels"]["emg_cor"]["mode"] == "segment"
        print(f"   OK: round-tripped JSON matches, saved at {out_path}")

        # Confirm a second RA's file for the same run does not clobber the first
        out_path_2 = save_annotation_json(tmp_dir, "smoke_test", "hlu", channel_outputs, sfreq, "synthetic")
        assert os.path.exists(out_path) and os.path.exists(out_path_2)
        print(f"   OK: second RA's file ({os.path.basename(out_path_2)}) coexists "
              f"with the first ({os.path.basename(out_path)})")

        print("8. Testing resume support: re-seeding a FRESH Raw from a previously-saved "
              "session's JSON should exactly reproduce that session's corrections, not the "
              "original automated peaks (regression check -- closing partway through a run "
              "used to silently discard prior edits on re-run; see HANDOFF.md)...")
        saved = load_saved_annotations(tmp_dir, "smoke_test", "xyz")
        assert saved is not None, "expected the JSON saved earlier in this test to be found"
        saved_channels, saved_path = saved
        assert saved_channels["ecg"]["indices"] == channel_outputs["ecg"]["indices"]
        assert saved_channels["emg_cor"]["bad_segments"] == channel_outputs["emg_cor"]["bad_segments"]

        fresh_raw = build_raw(df, CHANNELS, sfreq)
        fresh_raw = seed_annotations(fresh_raw, df, CHANNELS, saved_channels, sfreq)
        resumed_outputs = export_annotations(fresh_raw, df, CHANNELS, sfreq)

        assert resumed_outputs["ecg"]["indices"] == channel_outputs["ecg"]["indices"], (
            "resumed ECG peaks should exactly match the saved session (the earlier deletion "
            "and sloppy-click-turned-precise-peak must both survive), not the original "
            "automated ecg_peaks column"
        )
        assert resumed_outputs["emg_cor"]["bad_segments"] == channel_outputs["emg_cor"]["bad_segments"], (
            "resumed EMG bad segment should exactly match the saved session"
        )
        assert resumed_outputs["emg_zyg"]["bad_segments"] == []
        print(f"   OK: re-seeding from {os.path.basename(saved_path)} exactly reproduced the "
              f"prior session's corrections (not the original automated peaks)")

        print("9. Testing the backup-before-overwrite safety net: re-saving the SAME RA's "
              "file for the SAME run must back up the previous version first, never silently "
              "destroy it (added after a live scare where an RA worried a resumed session had "
              "wiped their prior corrections; see HANDOFF.md)...")
        different_outputs = {
            "ecg": {"mode": "point", "indices": [1, 2, 3]},
            "emg_cor": {"mode": "segment", "bad_segments": []},
        }
        save_annotation_json(tmp_dir, "smoke_test", "xyz", different_outputs, sfreq, "synthetic")
        backups = sorted(glob.glob(os.path.join(tmp_dir, "backups", "smoke_test_annotations_xyz_*.json")))
        assert len(backups) == 1, f"expected exactly one backup of the pre-overwrite file, got {backups}"
        with open(backups[0]) as f:
            backed_up = json.load(f)
        assert backed_up["channels"]["ecg"]["indices"] == channel_outputs["ecg"]["indices"], (
            "the backup must preserve the PREVIOUS session's data, not the new one being written"
        )
        with open(out_path) as f:
            current = json.load(f)
        assert current["channels"]["ecg"]["indices"] == [1, 2, 3], "the live file should now hold the new data"
        print(f"   OK: previous version backed up to {os.path.basename(backups[0])} before being overwritten")

        print("10. Testing the merged-peak warning: a point annotation with a suspiciously "
              "large duration (mne-qt-browser's silent merge-on-overlapping-drag behavior) "
              "must trigger a console warning naming the channel and approximate time (added "
              "after a live incident where a merged peak silently discarded a second one; "
              "see HANDOFF.md)...")
        raw_merge_test = build_raw(df, CHANNELS, sfreq)
        # 0.8s: a real merge of two adjacent peaks spans close to a full R-R
        # interval. 0.2s: a normal, legitimate single-peak click-drag -- confirmed
        # via a real full-run session (see HANDOFF.md) where genuine single peaks'
        # drag widths reached up to ~350ms and must NOT be flagged.
        merged_annotation = mne.Annotations(
            onset=[5.0], duration=[0.8], description=[CHANNELS["ecg"]["point_label"]],
        )
        normal_click_annotation = mne.Annotations(
            onset=[6.0], duration=[0.2], description=[CHANNELS["ecg"]["point_label"]],
        )
        raw_merge_test.set_annotations(merged_annotation + normal_click_annotation)

        captured = io.StringIO()
        with redirect_stdout(captured):
            export_annotations(raw_merge_test, df, CHANNELS, sfreq)
        console_output = captured.getvalue()

        assert "WARNING" in console_output and "merged" in console_output.lower(), (
            "expected a merge warning to be printed for the wide (800ms) annotation"
        )
        assert "ecg" in console_output and "5.00s" in console_output, (
            f"expected the warning to name the channel and ~5.00s onset; got:\n{console_output}"
        )
        assert "6.00s" not in console_output, (
            "a normal single-peak click-drag (200ms) should NOT be flagged"
        )
        print("   OK: the 800ms merged-looking annotation was flagged by name/time; "
              "the normal 200ms click was not")

        print("11. Testing that the merge-warning threshold ADAPTS to a channel's own "
              "typical peak-to-peak interval (regression check -- a single fixed threshold "
              "could miss a real merge for a fast heart rate, since its R-R interval can "
              "shrink close to or below a fixed cutoff; see HANDOFF.md)...")
        df_fast_hr = df.copy()
        fast_rr_samples = int(0.4 * sfreq)  # 150bpm -> 400ms R-R
        fast_peak_indices = list(range(1000, len(df_fast_hr) - 1000, fast_rr_samples))
        assert len(fast_peak_indices) >= 3, "need at least 3 peaks to estimate an interval"
        df_fast_hr["ecg_peaks"] = False
        df_fast_hr.loc[fast_peak_indices, "ecg_peaks"] = True

        threshold = _merge_warning_threshold(df_fast_hr, "ecg_peaks", sfreq)
        assert abs(threshold - 0.2) < 1e-6, f"expected a ~0.2s threshold (half of the 0.4s R-R), got {threshold}"

        raw_fast_hr = build_raw(df_fast_hr, CHANNELS, sfreq)
        # 250ms: below a fixed 400ms threshold (would have been MISSED by the old design),
        # but above this fast heart rate's own adaptive ~200ms threshold.
        would_have_been_missed = mne.Annotations(
            onset=[3.0], duration=[0.25], description=[CHANNELS["ecg"]["point_label"]],
        )
        raw_fast_hr.set_annotations(would_have_been_missed)

        captured = io.StringIO()
        with redirect_stdout(captured):
            export_annotations(raw_fast_hr, df_fast_hr, CHANNELS, sfreq)
        console_output = captured.getvalue()
        assert "WARNING" in console_output and "3.00s" in console_output, (
            f"a 250ms merge should be caught for a 150bpm (400ms R-R) heart rate, even "
            f"though a fixed 400ms threshold would have missed it; got:\n{console_output}"
        )
        print(f"   OK: adaptive threshold ({threshold:.2f}s, half of this fast heart rate's "
              f"400ms R-R) caught a 250ms merge that a fixed 400ms threshold would have missed")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("12. Testing check_interval_regularity(): flags both a long gap (possible missed "
          "peak) and a short gap (possible extra/spurious peak) against a channel's own "
          "median spacing -- the non-interactive stand-in for a live R-R track (wishlist "
          "10c), built after a live-updating version was judged too fragile; see HANDOFF.md)...")
    # A 1.0s-spaced peak train (with small, realistic beat-to-beat jitter so
    # ECG's Berntson criterion has a spread to work from), with one beat
    # removed (a doubled gap = missed peak) and one spurious peak added
    # (a split gap = extra peak). The earlier version of this fixture built
    # its "doubled gap" by inserting a DUPLICATE peak, i.e. a zero-length
    # interval, rather than removing one.
    jitter = [0, 12, -8, 20, -15, 5, -10, 18, -6, 9] * 3
    base_indices = [i * 1000 + jitter[i] for i in range(30)]  # sfreq=1000 -> ~1s apart
    long_gap_indices = base_indices[:10] + base_indices[11:]  # beat 10 missing -> one ~2s gap
    short_gap_indices = sorted(long_gap_indices + [base_indices[20] + 300])  # an extra peak 300 ms after beat 20

    fake_outputs = {
        "ecg": {"mode": "point", "indices": short_gap_indices},
        # A synthetic point-shaped payload for eda (which is actually segment-mode in
        # real use now) -- this exercises check_interval_regularity()'s own exclusion
        # logic (gated on the check_interval_regularity CONFIG FLAG, unset for eda),
        # independent of whatever shape eda's real output happens to be.
        "eda": {"mode": "point", "indices": short_gap_indices},
    }
    captured = io.StringIO()
    with redirect_stdout(captured):
        flagged = check_interval_regularity(fake_outputs, CHANNELS, sfreq=1000)
    console_output = captured.getvalue()

    flagged_channels = {f[0] for f in flagged}
    assert flagged_channels == {"ecg"}, f"eda must never be flagged (irregular spacing is normal there); got {flagged_channels}"
    kinds = sorted(f[4] for f in flagged)
    assert any("long" in k for k in kinds), f"expected a long-gap (missed peak) flag; got {kinds}"
    assert any("short" in k for k in kinds), f"expected a short-gap (extra peak) flag; got {kinds}"
    assert "ecg" in console_output and "eda" not in console_output, (
        "the printed warning must name ecg but never eda"
    )
    print(f"   OK: flagged {len(flagged)} irregular gap(s) for ecg only: {kinds}")

    print("12b. check_interval_regularity()'s console note names a cause appropriate to the "
          "channel actually flagged: 'an ectopic beat' for the cardiac channels (ecg/ppg); rsp is "
          "never interval-checked any more (a segment channel since 2026-09-25)...")
    ppg_fake_outputs = {"ppg": {"mode": "point", "indices": short_gap_indices}}
    captured_ppg = io.StringIO()
    with redirect_stdout(captured_ppg):
        flagged_ppg = check_interval_regularity(ppg_fake_outputs, CHANNELS, sfreq=1000)
    assert flagged_ppg and "ectopic" in captured_ppg.getvalue(), captured_ppg.getvalue()
    rsp_fake_outputs = {"rsp": {"mode": "point", "indices": short_gap_indices}}
    with redirect_stdout(io.StringIO()):
        assert check_interval_regularity(rsp_fake_outputs, CHANNELS, sfreq=1000) == []
    print("   OK: a ppg flag's note mentions an ectopic beat; rsp is not interval-checked")

    print("13. check_interval_regularity() skips a channel with too few peaks to get a "
          "meaningful median (avoids noisy false positives early in a review)...")
    sparse_outputs = {"ecg": {"mode": "point", "indices": [0, 5000, 6000]}}  # only 3 peaks, 2 intervals
    flagged_sparse = check_interval_regularity(sparse_outputs, CHANNELS, sfreq=1000)
    assert flagged_sparse == [], f"expected no flags with too few peaks to estimate a median; got {flagged_sparse}"
    print("   OK: skipped (fewer than the minimum peak count)")

    print("14. save_annotation_json() MERGES with an existing save instead of replacing it "
          "wholesale -- a channel saved in an earlier call must be carried forward untouched "
          "when a LATER call saves a DIFFERENT channel for the same RA/run (confirmed live "
          "bug before this fix: reviewing --channels ppg after already having saved --channels "
          "ecg silently erased the ecg data; see HANDOFF.md)...")
    tmp_dir_2 = tempfile.mkdtemp(prefix="annotation_merge_")
    try:
        ecg_only = {"ecg": {"mode": "point", "indices": [100, 200, 300]}}
        save_annotation_json(tmp_dir_2, "merge_test", "hlu", ecg_only, sfreq, "synthetic")

        ppg_only = {"ppg": {"mode": "point", "indices": [400, 500]}}
        captured = io.StringIO()
        with redirect_stdout(captured):
            save_annotation_json(tmp_dir_2, "merge_test", "hlu", ppg_only, sfreq, "synthetic")
        console_output = captured.getvalue()

        with open(os.path.join(tmp_dir_2, "merge_test_annotations_hlu.json")) as f:
            merged = json.load(f)
        assert set(merged["channels"].keys()) == {"ecg", "ppg"}, (
            f"expected both channels present after the merge, got {list(merged['channels'].keys())}"
        )
        assert merged["channels"]["ecg"] == ecg_only["ecg"], "the earlier ecg save must survive unchanged"
        assert merged["channels"]["ppg"] == ppg_only["ppg"], "the new ppg save must be present as saved"
        assert "ecg" in console_output and "Carrying forward" in console_output, (
            f"expected a console note about carrying ecg forward; got:\n{console_output}"
        )
        print(f"   OK: channels={list(merged['channels'].keys())}, ecg preserved exactly, "
              f"ppg added -- console confirmed: {console_output.strip().splitlines()[-1]!r}")

        print("15. A THIRD save that DOES touch 'ecg' again correctly overwrites just that "
              "channel's entry, still leaving 'ppg' untouched...")
        ecg_updated = {"ecg": {"mode": "point", "indices": [999]}}
        save_annotation_json(tmp_dir_2, "merge_test", "hlu", ecg_updated, sfreq, "synthetic")
        with open(os.path.join(tmp_dir_2, "merge_test_annotations_hlu.json")) as f:
            final = json.load(f)
        assert final["channels"]["ecg"] == ecg_updated["ecg"], "ecg should reflect the NEW save, not the old one"
        assert final["channels"]["ppg"] == ppg_only["ppg"], "ppg must still be untouched from step 14"
        print(f"   OK: ecg updated to {final['channels']['ecg']}, ppg still {final['channels']['ppg']}")
    finally:
        shutil.rmtree(tmp_dir_2, ignore_errors=True)

    print("16. run_stage_b()'s 'Resuming your previous session' message only counts channels "
          "RELEVANT to (and actually seeded into) the CURRENT session -- regression check for a "
          "second bug found alongside the merge fix: it used to sum ALL saved channels "
          "regardless of relevance, misreporting e.g. old ecg counts while resuming an "
          "ppg-only session that never touches ecg at all...")
    tmp_dir_3 = tempfile.mkdtemp(prefix="annotation_resume_count_")
    try:
        df_short, sfreq_short = generate_synthetic_demo(duration_sec=10)
        with patch.object(mne.io.RawArray, "plot", return_value=None):
            with redirect_stdout(io.StringIO()):
                run_stage_b(df_short, sfreq_short, {"ecg": CHANNELS["ecg"]}, "hlu", tmp_dir_3,
                            "resume_count_test", "synthetic")

            captured_2 = io.StringIO()
            with redirect_stdout(captured_2):
                run_stage_b(df_short, sfreq_short, {"ppg": CHANNELS["ppg"]}, "hlu", tmp_dir_3,
                            "resume_count_test", "synthetic")
        resume_line = [l for l in captured_2.getvalue().splitlines() if "Resuming" in l][0]
        assert "0 point annotation(s)" in resume_line, (
            f"expected 0 points reported (no saved ppg data exists yet, and the old ecg "
            f"data is irrelevant to this session) but got: {resume_line!r}"
        )
        print(f"   OK: {resume_line.strip()!r}")

        print("16b. ...and the channel NOT in the saved file (ppg here) still gets its real "
              "automated peaks seeded, rather than being left blank just because SOME saved "
              "file exists for this run (regression check -- confirmed live: reviewing ppg for "
              "the first time on a run whose only prior save was ecg-only silently produced a "
              "blank ppg, discarding its real auto-detected peaks; see HANDOFF.md)...")
        with open(os.path.join(tmp_dir_3, "resume_count_test_annotations_hlu.json")) as f:
            after_ppg_review = json.load(f)
        n_ppg_peaks_available = int(df_short["ppg_peaks"].sum())
        assert n_ppg_peaks_available > 0, "test data must actually have ppg peaks to make this check meaningful"
        assert after_ppg_review["channels"]["ppg"]["indices"] != [], (
            "ppg should have been auto-seeded from its ppg_peaks column, not left empty, "
            "even though this run's only prior save was ecg-only"
        )
        assert after_ppg_review["channels"]["ecg"]["indices"] != [], (
            "the earlier ecg save must still survive the merge, untouched by this ppg-only session"
        )
        print(f"   OK: ppg seeded+saved with {len(after_ppg_review['channels']['ppg']['indices'])} peaks "
              f"(of {n_ppg_peaks_available} available), ecg still preserved from the prior session")
    finally:
        shutil.rmtree(tmp_dir_3, ignore_errors=True)

    print("17. export_annotations()'s default snap_window_samples scales with sfreq instead of a "
          "fixed 50-sample constant (regression check -- the user pointed out that 50 samples is "
          "only ~50ms at 1000 Hz, but would be 100ms at 500 Hz or 25ms at 2000 Hz -- a different "
          "real-world tolerance despite the same nominal sample count)...")
    from annotation_io import DEFAULT_SNAP_WINDOW_SEC
    for test_sfreq in (500, 1000, 2000):
        df_scaled, _ = generate_synthetic_demo(duration_sec=10, sfreq=test_sfreq)
        raw_scaled = build_raw(df_scaled, CHANNELS, test_sfreq)
        seed_annotations(raw_scaled, df_scaled, CHANNELS, {}, test_sfreq)
        outputs_scaled = export_annotations(raw_scaled, df_scaled, CHANNELS, test_sfreq)
        expected_samples = round(DEFAULT_SNAP_WINDOW_SEC * test_sfreq)
        for ch_key, out in outputs_scaled.items():
            if out.get("mode") == "point":
                assert out["snap_window_samples"] == expected_samples, (
                    f"at {test_sfreq} Hz, {ch_key} expected snap_window_samples={expected_samples}, "
                    f"got {out['snap_window_samples']}"
                )
    print(f"   OK: snap_window_samples scales as 25/50/100 samples at 500/1000/2000 Hz "
          f"(all = {DEFAULT_SNAP_WINDOW_SEC * 1000:.0f}ms)")

    print("18. compute_channel_value_ranges()/format_channel_value_ranges(): only SEGMENT-mode "
          "channels (rsp/eda/emg_cor/emg_zyg here) get a real min/mean/max, not point-mode ecg/ppg "
          "-- see channel_config.py's module docstring for why this dividing line was chosen "
          "(MNE's remove_dc-relative crosshair can't show absolute values, and only segment-mode "
          "channels' absolute baseline is the actual thing an RA needs to judge)...")
    ranges = compute_channel_value_ranges(df, CHANNELS)
    assert set(ranges.keys()) == {"rsp", "eda", "emg_cor", "emg_zyg"}, (
        f"expected only the segment-mode channels present in this data; got {set(ranges.keys())}"
    )
    lo, mean, hi = ranges["eda"]
    assert lo == df["eda"].min() and hi == df["eda"].max() and abs(mean - df["eda"].mean()) < 1e-9, (
        f"eda's computed range doesn't match its actual data: got ({lo}, {mean}, {hi})"
    )
    lines = format_channel_value_ranges(ranges, CHANNELS)
    assert len(lines) == 4 and all("RSP" in l or "EDA" in l or "EMG_Corrugator" in l or "EMG_Zygomatic" in l
                                   for l in lines)
    print(f"   OK: {lines}")

    print("18b. run_stage_b()'s startup banner prints these same ranges to the console, only for "
          "segment-mode channels actually being reviewed...")
    tmp_dir_4 = tempfile.mkdtemp(prefix="value_range_banner_")
    try:
        with patch.object(mne.io.RawArray, "plot", return_value=None):
            captured_3 = io.StringIO()
            with redirect_stdout(captured_3):
                run_stage_b(df, sfreq, {"ecg": CHANNELS["ecg"], "eda": CHANNELS["eda"]}, "hlu", tmp_dir_4,
                            "value_range_test", "synthetic")
        banner_output = captured_3.getvalue()
        assert "Real value ranges" in banner_output and "EDA:" in banner_output, (
            f"expected the value-range banner to mention EDA; got:\n{banner_output}"
        )
        assert "ECG" not in banner_output.split("Real value ranges")[1].split("Close the plot")[0], (
            "ECG is point-mode and must not appear in the value-range banner"
        )
        print("   OK: banner mentions EDA's real range, not ECG's")
    finally:
        shutil.rmtree(tmp_dir_4, ignore_errors=True)

    print("\nALL SMOKE TESTS PASSED.")


if __name__ == "__main__":
    main()
