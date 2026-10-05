#!/usr/bin/env python3
"""
Confirms the pipeline works when a physio.tsv.gz has only a SUBSET of the
"full" 8-column set (ecg/rsp/ppg/eda + their peaks) -- e.g. a study that
doesn't collect PPG, or one that only collects ECG. Also checks the
opposite direction: a file with EXTRA real EMG columns beyond today's
placeholders works the same way, since build_raw/compute_scalings/
seed_annotations/export_annotations all key off "is this channel
actually present in the DataFrame", not a fixed expected column list.

Run with: annotate_env\\Scripts\\python.exe test_column_subsets.py
"""

import numpy as np
import pandas as pd

from annotation_io import export_annotations, seed_annotations
from channel_config import CHANNELS
from physio_io import build_raw, compute_scalings, generate_synthetic_demo


def run_full_pipeline(df, sfreq, channel_configs, label):
    """Exercises every non-interactive step of Stage B's pipeline for a given column set."""
    raw = build_raw(df, channel_configs, sfreq)
    expected_channels = [ch for ch in channel_configs if ch in df.columns]
    assert list(raw.ch_names) == expected_channels, f"[{label}] {raw.ch_names} != {expected_channels}"

    scalings = compute_scalings(df, channel_configs)
    assert not any(np.isnan(v) for v in scalings.values()), f"[{label}] NaN in scalings: {scalings}"

    raw = seed_annotations(raw, df, channel_configs, {}, sfreq)

    channel_outputs = export_annotations(raw, df, channel_configs, sfreq)
    assert set(channel_outputs.keys()) == set(expected_channels), (
        f"[{label}] export_annotations returned {list(channel_outputs.keys())}, expected {expected_channels}"
    )
    return raw, scalings, channel_outputs


def main():
    df, sfreq = generate_synthetic_demo(duration_sec=10)

    print("1. Full 6-channel set (baseline)...")
    raw, scalings, outputs = run_full_pipeline(df, sfreq, CHANNELS, "full")
    print(f"   OK: channels={raw.ch_names}")

    print("2. Study without PPG or EDA (e.g. only collected ECG + RSP + EMG)...")
    df_no_ppg_eda = df.drop(columns=["ppg", "ppg_peaks", "eda", "eda_peaks"])
    raw, scalings, outputs = run_full_pipeline(df_no_ppg_eda, sfreq, CHANNELS, "no ppg/eda")
    assert "ppg" not in raw.ch_names and "eda" not in raw.ch_names
    assert set(scalings.keys()) == {"ecg", "resp", "emg"}
    print(f"   OK: channels={raw.ch_names}, scalings cover only present types: {list(scalings.keys())}")

    print("3. ECG-only file (the minimal realistic case)...")
    df_ecg_only = df[["ecg", "ecg_peaks"]].copy()
    raw, scalings, outputs = run_full_pipeline(df_ecg_only, sfreq, CHANNELS, "ecg only")
    assert list(raw.ch_names) == ["ecg"]
    assert list(scalings.keys()) == ["ecg"]
    assert list(outputs.keys()) == ["ecg"]
    print(f"   OK: channels={raw.ch_names}")

    print("4. A file with a channel NOT in channel_config.py at all (e.g. a future new signal) "
          "is silently ignored rather than crashing...")
    df_extra_col = df.copy()
    df_extra_col["ambient_temperature"] = 22.0  # not in CHANNELS -- should just be ignored
    raw, scalings, outputs = run_full_pipeline(df_extra_col, sfreq, CHANNELS, "extra unknown column")
    assert "ambient_temperature" not in raw.ch_names
    print(f"   OK: unrecognized column ignored cleanly, channels={raw.ch_names}")

    print("5. Real (non-placeholder) EMG data alongside a missing PPG column simultaneously...")
    df_mixed = df.drop(columns=["ppg", "ppg_peaks"]).copy()
    rng = np.random.default_rng(1)
    df_mixed["emg_cor"] = rng.normal(loc=0.0, scale=87.0, size=len(df_mixed))  # "real" EMG this time
    raw, scalings, outputs = run_full_pipeline(df_mixed, sfreq, CHANNELS, "real emg, no ppg")
    assert "ppg" not in raw.ch_names
    assert "emg_cor" in raw.ch_names
    own_scale = (np.percentile(df_mixed["emg_cor"], 97.5) - np.percentile(df_mixed["emg_cor"], 2.5)) / 2
    assert np.isclose(scalings["emg"], own_scale, rtol=0.05), (
        f"expected emg's own scale ({own_scale:.4g}) since it's real data now, got {scalings['emg']:.4g}"
    )
    print(f"   OK: missing ppg + real emg handled together correctly, emg scale={scalings['emg']:.4g}")

    print("6. A file WITH sbp/dbp/finger_temperature (e.g. a future CareTaker-BP-enabled or "
          "finger-temp-probe study) -- all three are segment-mode (like EMG: RAs flag bad/"
          "untrustworthy stretches, no peak-picking), and eda is now segment-mode too (the "
          "user doesn't expect RAs to correct individual SCR peaks)...")
    df_with_new = df.copy()
    df_with_new["sbp"] = 120.0
    df_with_new["dbp"] = 80.0
    df_with_new["finger_temperature"] = 33.5
    raw, scalings, outputs = run_full_pipeline(df_with_new, sfreq, CHANNELS, "sbp/dbp/finger_temperature")
    for ch_key in ("sbp", "dbp", "finger_temperature", "eda"):
        assert ch_key in raw.ch_names, f"{ch_key} should be present"
        assert CHANNELS[ch_key]["annotation_mode"] == "segment", f"{ch_key} should be segment-mode"
        assert outputs[ch_key] == {"mode": "segment", "bad_segments": []}, (
            f"{ch_key} should export like EMG (empty bad_segments, no auto-seeded peaks): {outputs[ch_key]}"
        )
    print(f"   OK: channels={raw.ch_names}, sbp/dbp/finger_temperature/eda all segment-mode "
          f"with no auto-seeded annotations")

    print("\nALL COLUMN SUBSET TESTS PASSED.")


if __name__ == "__main__":
    main()
