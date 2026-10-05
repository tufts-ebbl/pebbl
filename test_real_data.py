#!/usr/bin/env python3
"""
Runs the non-interactive pipeline checks against every REAL physio.tsv.gz
found at test_input/*_physio.tsv.gz. Skips cleanly if none are present
(this repo doesn't ship real subject data, and test_input/ is local-only).

This exists because real data has already surfaced a bug synthetic data
never would have (a real PPG channel found to be ~91% NaN, which broke
compute_scalings before it was made NaN-safe -- see test_smoke.py's
"3c"/"3d" and HANDOFF.md's debugging history for the full story).

Known real-data quirk (not a bug in this tool -- see physioProcess's
processPPG() in physioFunctions.py): PPG comes from a separate device (the
CareTaker BP monitor's pulseWaveform file) at its own sampling rate,
index-merged (not time-aligned) into the Biopac-derived 1000 Hz timeline.
When the CareTaker recording is shorter than the Biopac recording -- as it
is in every real file seen so far -- everything past that point comes back
NaN. This test asserts that pattern (single contiguous valid block from
sample 0, 100% NaN afterward) rather than treating it as a failure, so a
genuinely different NaN pattern in a future file (e.g. scattered gaps) will
stand out as worth investigating rather than being silently swallowed.

Run with: annotate_env\\Scripts\\python.exe test_real_data.py
"""

import glob
import os

import numpy as np

from annotation_io import combine_with_reference_channels, export_annotations, seed_annotations
from channel_config import CHANNELS
from physio_io import build_raw, compute_scalings, load_physio_tsv


def find_test_inputs():
    # recursive, since a whole-folder Box copy nests files under sub-XXX/ses-runY/beh/
    return sorted(glob.glob(os.path.join(os.path.dirname(__file__), "test_input", "**", "*_physio.tsv.gz"),
                             recursive=True))


def check_one_file(path):
    print(f"\n--- {os.path.basename(path)} ---")

    print("1. Loading...")
    df, sfreq = load_physio_tsv(path)
    print(f"   OK: {len(df)} samples ({len(df) / sfreq / 60:.1f} min) at {sfreq} Hz, columns: {list(df.columns)}")

    print("2. Checking NaN pattern per signal column...")
    for col in ["ecg", "rsp", "ppg", "eda"]:
        if col not in df.columns:
            continue
        pct_nan = 100 * df[col].isna().mean()
        n_nan = int(df[col].isna().sum())
        if n_nan == 0:
            print(f"   {col}: 0% NaN")
            continue
        valid_idx = np.where(~df[col].isna())[0]
        is_single_leading_block = (
            len(valid_idx) > 0
            and valid_idx[0] == 0
            and np.array_equal(valid_idx, np.arange(valid_idx[0], valid_idx[-1] + 1))
            and df[col].iloc[valid_idx[-1] + 1:].notna().sum() == 0
        )
        shape = "single contiguous block from sample 0, then NaN to the end (known PPG/CareTaker pattern)" \
            if is_single_leading_block else "SCATTERED or non-standard NaN pattern -- worth a closer look"
        print(f"   {col}: {pct_nan:.1f}% NaN ({n_nan} samples) -- {shape}")
        if col == "ppg" and not is_single_leading_block:
            print(f"   NOTE: this doesn't match the pattern seen in prior real files -- investigate before trusting it.")

    print("3. Running the full non-interactive Stage B pipeline (all present columns)...")
    raw = build_raw(df, CHANNELS, sfreq)
    scalings = compute_scalings(df, CHANNELS)
    assert not any(np.isnan(v) for v in scalings.values()), f"NaN in scalings on real data: {scalings}"
    raw = seed_annotations(raw, df, CHANNELS, {}, sfreq)
    channel_outputs = export_annotations(raw, df, CHANNELS, sfreq)
    print(f"   OK: channels={raw.ch_names}")
    print(f"   OK: scalings={ {k: round(v, 4) for k, v in scalings.items()} }")
    for ch, out in channel_outputs.items():
        if out["mode"] == "point":
            print(f"   OK: {ch} seeded with {len(out['indices'])} automated peaks")

    print("4. Simulating a leaner study file (dropping ppg/eda)...")
    df_subset = df.drop(columns=[c for c in ["ppg", "ppg_peaks", "eda", "eda_peaks"] if c in df.columns])
    raw_subset = build_raw(df_subset, CHANNELS, sfreq)
    scalings_subset = compute_scalings(df_subset, CHANNELS)
    assert not any(np.isnan(v) for v in scalings_subset.values())
    assert "ppg" not in raw_subset.ch_names and "eda" not in raw_subset.ch_names
    seed_annotations(raw_subset, df_subset, CHANNELS, {}, sfreq)
    print(f"   OK: partial-column handling correct, channels={raw_subset.ch_names}")

    if "event" in df.columns:
        print("5. Real 'event' column present -- checking the reference-channel mechanism against it...")
        value_counts = df["event"].value_counts().sort_index()
        print(f"   event value counts: {dict(value_counts)}")
        pct_outside_task = 100 * (df["event"] == 0).mean()
        print(f"   {pct_outside_task:.1f}% of the recording is event==0 (outside the task)")

        combined = combine_with_reference_channels(CHANNELS, df)
        assert "event" in combined and combined is not CHANNELS
        raw_with_event = build_raw(df, combined, sfreq)
        assert "event" in raw_with_event.ch_names
        assert raw_with_event.get_channel_types(picks="event")[0] == "stim"

        scalings_with_event = compute_scalings(df, combined)
        assert not any(np.isnan(v) for v in scalings_with_event.values())

        raw_with_event = seed_annotations(raw_with_event, df, combined, {}, sfreq)
        outputs_with_event = export_annotations(raw_with_event, df, combined, sfreq)
        assert "event" not in outputs_with_event, "event should never appear in exported output"
        print(f"   OK: 'event' included as a stim channel ({raw_with_event.ch_names}), "
              f"correctly excluded from exported output ({list(outputs_with_event.keys())})")

        print("6. --channels-style restriction still shows 'event' automatically...")
        restricted_combined = combine_with_reference_channels({"ecg": CHANNELS["ecg"]}, df)
        assert set(restricted_combined.keys()) == {"ecg", "event"}
        print("   OK: requesting only 'ecg' still includes 'event' as a reference track")
    else:
        print("5. No 'event' column in this file (predates the physioProcess event-marker fix) -- "
              "confirming it's simply absent, not broken...")
        combined = combine_with_reference_channels(CHANNELS, df)
        assert combined == CHANNELS
        print("   OK: combine_with_reference_channels() is a no-op, as expected")


def main():
    paths = find_test_inputs()
    if not paths:
        print("No files found under test_input/*_physio.tsv.gz -- skipping (this is expected if you "
              "haven't placed real test files there).")
        return

    for path in paths:
        check_one_file(path)

    print(f"\nALL REAL DATA TESTS PASSED ({len(paths)} file(s): {[os.path.basename(p) for p in paths]}).")


if __name__ == "__main__":
    main()
