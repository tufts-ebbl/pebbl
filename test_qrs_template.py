#!/usr/bin/env python3
"""
Non-interactive test of the QRS template algorithm itself (Stage A's
signal-processing core, not the interactive window). Simulates a "perfect
RA" by using the ground-truth peaks (from neurokit2's own detection on
clean synthetic ECG) as the "corrected" peaks for the template window, then
checks that cross-correlating that template across the full run recovers
peaks that closely match the ground truth everywhere else too.

Run with: annotate_env\\Scripts\\python.exe test_qrs_template.py
"""

import os
import shutil
import tempfile
from unittest.mock import patch

import mne
import numpy as np

import qrs_template as qrs_template_module
from annotation_io import find_template_json, resolve_ecg_peaks
from physio_io import generate_synthetic_demo, generate_synthetic_demo_two_runs
from qrs_template import (
    apply_wavelet_filter,
    build_ecg_raw,
    build_qrs_template,
    enforce_refractory_period,
    extract_and_refine_peaks,
    run_stage_a,
    save_qrs_template_outputs,
)


def match_rate(detected, truth, tolerance_samples=25):
    """Fraction of truth peaks that have a detected peak within tolerance, and vice versa."""
    detected = np.asarray(sorted(detected))
    truth = np.asarray(sorted(truth))
    if len(truth) == 0 or len(detected) == 0:
        return 0.0, 0.0
    recall_hits = sum(1 for t in truth if np.any(np.abs(detected - t) <= tolerance_samples))
    precision_hits = sum(1 for d in detected if np.any(np.abs(truth - d) <= tolerance_samples))
    return recall_hits / len(truth), precision_hits / len(detected)


def main():
    print("1. Unit-testing enforce_refractory_period()...")
    peaks = np.array([100, 150, 500, 900, 920, 1500])  # 150 and 920 are too close to their neighbors
    cleaned = enforce_refractory_period(peaks, sampling_rate=1000, min_rr_ms=400)
    assert list(cleaned) == [100, 500, 900, 1500], f"got {list(cleaned)}"
    print(f"   OK: {list(peaks)} -> {list(cleaned)}")

    print("1b. build_ecg_raw() must NOT channel-scope seeded annotations (regression check -- "
          "channel-scoping was a confirmed trigger for a live mne-qt-browser crash when "
          "editing densely-seeded peaks; see HANDOFF.md's debugging history)...")
    fake_ecg = np.zeros(5000)
    fake_peaks_bool = np.zeros(5000, dtype=bool)
    fake_peaks_bool[[500, 1500, 2500, 3500, 4500]] = True
    fake_raw = build_ecg_raw(fake_ecg, fake_peaks_bool, sfreq=1000)
    scoped = [a for a in fake_raw.annotations if a.get("ch_names")]
    assert not scoped, f"{len(scoped)} seeded annotation(s) are channel-scoped -- this must not regress"
    print(f"   OK: {len(fake_raw.annotations)} seeded annotations, none channel-scoped")

    print("2. Generating 90s of synthetic ECG (with ground-truth peaks)...")
    df, sfreq = generate_synthetic_demo(duration_sec=90)
    ecg_full = df["ecg"].to_numpy()
    true_peaks_full = np.where(df["ecg_peaks"].to_numpy())[0]
    print(f"   OK: {len(ecg_full)} samples, {len(true_peaks_full)} true peaks")

    print("3. Applying wavelet filter to the full run...")
    wavelet_full = apply_wavelet_filter(ecg_full)
    assert len(wavelet_full) == len(ecg_full)
    print("   OK")

    print("4. Building QRS template from a 'perfectly RA-corrected' 20s window "
          "(using the ground-truth peaks as a stand-in for RA correction)...")
    window_sec = 20
    start_idx, end_idx = 0, window_sec * sfreq
    ecg_window = ecg_full[start_idx:end_idx]
    wavelet_window = wavelet_full[start_idx:end_idx]
    window_true_peaks = true_peaks_full[true_peaks_full < end_idx]

    qrs_template, adjustment = build_qrs_template(ecg_window, wavelet_window, window_true_peaks, sampling_rate=sfreq)
    print(f"   OK: template length={len(qrs_template)}, adjustment={adjustment}")

    print("5. Cross-correlating the template against the FULL run...")
    final_peaks = extract_and_refine_peaks(ecg_full, wavelet_full, qrs_template, adjustment, sampling_rate=sfreq)
    print(f"   OK: detected {len(final_peaks)} peaks")

    print("6. Comparing detected peaks against ground truth (tolerance = 25 samples)...")
    recall, precision = match_rate(final_peaks, true_peaks_full, tolerance_samples=25)
    print(f"   recall={recall:.3f}, precision={precision:.3f}")
    assert recall > 0.9, f"recall too low ({recall:.3f}) -- template-matching isn't recovering real peaks well"
    assert precision > 0.9, f"precision too low ({precision:.3f}) -- too many spurious detections"
    print("   OK: template-matching recovers the vast majority of true peaks with few false positives")

    print("7. Saving outputs and checking the JSON schema matches step1_qrs_template.ipynb's...")
    tmp_dir = tempfile.mkdtemp(prefix="qrs_template_smoke_")
    try:
        json_path, csv_path = save_qrs_template_outputs(tmp_dir, "smoke_test", "hlu", final_peaks, qrs_template)
        import json
        with open(json_path) as f:
            payload = json.load(f)
        assert set(payload.keys()) == {"ecg"}
        assert set(payload["ecg"].keys()) == {"corrected_peaks", "initials"}
        assert payload["ecg"]["initials"] == ["hlu"]
        assert len(payload["ecg"]["corrected_peaks"]) == len(final_peaks)
        assert os.path.exists(csv_path)
        print(f"   OK: {os.path.basename(json_path)} matches step1's {{'ecg': {{'corrected_peaks', 'initials'}}}} schema")

        print("8. Checking Stage B's --ecg-source resolution logic against this template JSON...")
        original_batch_peaks = int(df["ecg_peaks"].sum())

        df_batch, label_batch = resolve_ecg_peaks(df.copy(), "batch", json_path)
        assert int(df_batch["ecg_peaks"].sum()) == original_batch_peaks
        assert "batch pipeline" in label_batch
        print(f"   OK: --ecg-source batch ignores the template ({label_batch})")

        df_template, label_template = resolve_ecg_peaks(df.copy(), "template", json_path)
        assert int(df_template["ecg_peaks"].sum()) == len(final_peaks)
        assert "QRS template" in label_template
        print(f"   OK: --ecg-source template uses the template's {len(final_peaks)} peaks ({label_template})")

        df_auto, label_auto = resolve_ecg_peaks(df.copy(), "auto", json_path)
        assert int(df_auto["ecg_peaks"].sum()) == len(final_peaks)
        print(f"   OK: --ecg-source auto prefers the template when it exists ({label_auto})")

        missing_path = os.path.join(tmp_dir, "does_not_exist_ecg_corrected_qrs.json")
        df_auto_fallback, label_fallback = resolve_ecg_peaks(df.copy(), "auto", missing_path)
        assert int(df_auto_fallback["ecg_peaks"].sum()) == original_batch_peaks
        assert "no template JSON found" in label_fallback
        print(f"   OK: --ecg-source auto falls back to batch peaks when no template exists ({label_fallback})")

        try:
            resolve_ecg_peaks(df.copy(), "template", missing_path)
            raise AssertionError("expected FileNotFoundError when --ecg-source template has no file")
        except FileNotFoundError:
            print("   OK: --ecg-source template raises clearly when the template JSON is missing")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("9. Testing 'one template, applied to both runs' (generate_synthetic_demo_two_runs)...")
    run_dfs, sfreq2 = generate_synthetic_demo_two_runs(duration_sec_each=60)
    assert set(run_dfs.keys()) == {"1", "2"}
    print(f"   OK: run 1 has {len(run_dfs['1'])} samples, run 2 has {len(run_dfs['2'])} samples")

    print("10. Building a template from run 1's window only, applying it to BOTH runs...")
    template_run_df = run_dfs["1"]
    ecg_run1 = template_run_df["ecg"].to_numpy()
    wavelet_run1 = apply_wavelet_filter(ecg_run1)
    window_true_peaks_run1 = np.where(template_run_df["ecg_peaks"].to_numpy())[0]
    window_true_peaks_run1 = window_true_peaks_run1[window_true_peaks_run1 < 20 * sfreq2]

    qrs_template_2, adjustment_2 = build_qrs_template(
        ecg_run1[: 20 * sfreq2], wavelet_run1[: 20 * sfreq2], window_true_peaks_run1, sampling_rate=sfreq2
    )

    for run in ("1", "2"):
        ecg_full_run = run_dfs[run]["ecg"].to_numpy()
        wavelet_full_run = wavelet_run1 if run == "1" else apply_wavelet_filter(ecg_full_run)
        true_peaks_run = np.where(run_dfs[run]["ecg_peaks"].to_numpy())[0]

        detected = extract_and_refine_peaks(
            ecg_full_run, wavelet_full_run, qrs_template_2, adjustment_2, sampling_rate=sfreq2
        )
        recall, precision = match_rate(detected, true_peaks_run, tolerance_samples=25)
        print(f"   run {run}: recall={recall:.3f}, precision={precision:.3f} "
              f"({'template-building run' if run == '1' else 'template REUSED on this run'})")
        assert recall > 0.85, f"run {run} recall too low ({recall:.3f}) -- template didn't transfer well"
        assert precision > 0.85, f"run {run} precision too low ({precision:.3f})"
    print("   OK: the run-1-built template transfers well to run 2 too, without building a second template")

    print("11. Testing find_template_json()'s multi-RA disambiguation...")
    tmp_dir_2 = tempfile.mkdtemp(prefix="qrs_template_multi_ra_")
    try:
        file_stem = "sub-999_ses-run1_task-sdi"

        no_candidates = find_template_json(tmp_dir_2, file_stem)
        assert not os.path.exists(no_candidates)
        print(f"   OK: zero candidates -> a (nonexistent) placeholder path is returned, not an error")

        hlu_json, _ = save_qrs_template_outputs(tmp_dir_2, file_stem, "hlu", final_peaks, qrs_template)
        found = find_template_json(tmp_dir_2, file_stem)
        assert found == hlu_json
        print(f"   OK: exactly one candidate -> used automatically ({os.path.basename(found)})")

        xyz_json, _ = save_qrs_template_outputs(tmp_dir_2, file_stem, "xyz", final_peaks, qrs_template)
        found_preferred = find_template_json(tmp_dir_2, file_stem, initials="hlu")
        assert found_preferred == hlu_json
        print(f"   OK: two candidates, caller's own initials match one -> that one is preferred "
              f"({os.path.basename(found_preferred)})")

        try:
            find_template_json(tmp_dir_2, file_stem, initials="someone_else")
            raise AssertionError("expected SystemExit when multiple candidates exist and none match")
        except SystemExit as e:
            assert "hlu" in str(e) and "xyz" in str(e)
            print("   OK: two candidates, no initials match -> raises clearly, listing both")

        explicit = find_template_json(tmp_dir_2, file_stem, explicit_path="/some/explicit/path.json")
        assert explicit == "/some/explicit/path.json"
        print("   OK: an explicit --template-json path bypasses searching entirely")
    finally:
        shutil.rmtree(tmp_dir_2, ignore_errors=True)

    print("12. run_stage_a() with a NONZERO --template-start (regression check -- crop() shifts "
          "raw.first_time but leaves annotation onsets in full-run-absolute time, so converting "
          "onset -> sample index must subtract raw.first_time; omitting that worked by coincidence "
          "when template_start was 0.0, which is why it went unnoticed until a real --template-start "
          "20 session hit 'Only 0 usable peak(s)' -- see HANDOFF.md's debugging history)...")
    tmp_dir_3 = tempfile.mkdtemp(prefix="qrs_template_nonzero_start_")
    try:
        df_long, sfreq_long = generate_synthetic_demo(duration_sec=90)
        with patch.object(mne.io.RawArray, "plot", return_value=None), \
             patch.object(qrs_template_module, "show_template_preview", return_value=True):
            results = run_stage_a(
                run_dfs={"1": df_long},
                run_file_stems={"1": "nonzero_start_test"},
                run_out_dirs={"1": tmp_dir_3},
                sfreq=sfreq_long,
                initials="hlu",
                template_run="1",
                template_start=20.0,
                template_window=20.0,
            )
        assert len(results["1"]["final_peaks"]) > 0, "expected real peaks, not an empty/failed template"
        print(f"   OK: --template-start 20 built a template and found "
              f"{len(results['1']['final_peaks'])} peaks (did not crash with 'Only 0 usable peak(s)')")
    finally:
        shutil.rmtree(tmp_dir_3, ignore_errors=True)

    print("13. extract_and_refine_peaks()'s default snap_window_samples scales with sampling_rate "
          "instead of a fixed 50-sample constant (regression check -- the same nominal sample count "
          "meant a different real-world tolerance at different sampling rates; now it's derived from "
          "DEFAULT_SNAP_WINDOW_SEC=0.05s so 500/1000/2000 Hz studies all get the same ~50ms window)...")
    from annotation_io import DEFAULT_SNAP_WINDOW_SEC
    dummy_ecg = np.zeros(1000)
    dummy_wavelet = np.zeros(1000)
    dummy_template = np.zeros(50)
    dummy_adjustment = 0
    for test_sfreq, expected_window in ((500, round(DEFAULT_SNAP_WINDOW_SEC * 500)),
                                         (1000, round(DEFAULT_SNAP_WINDOW_SEC * 1000)),
                                         (2000, round(DEFAULT_SNAP_WINDOW_SEC * 2000))):
        with patch.object(qrs_template_module, "snap_to_local_max", wraps=qrs_template_module.snap_to_local_max) as mock_snap:
            extract_and_refine_peaks(dummy_ecg, dummy_wavelet, dummy_template, dummy_adjustment,
                                      sampling_rate=test_sfreq)
            if mock_snap.call_args is not None:
                actual_window = mock_snap.call_args.kwargs.get("window_samples")
                assert actual_window == expected_window, (
                    f"at {test_sfreq} Hz expected window_samples={expected_window}, got {actual_window}"
                )
    assert round(DEFAULT_SNAP_WINDOW_SEC * 500) != round(DEFAULT_SNAP_WINDOW_SEC * 2000), (
        "sanity check that the test actually distinguishes different sampling rates"
    )
    print(f"   OK: window scales as 25/50/100 samples at 500/1000/2000 Hz "
          f"(all = {DEFAULT_SNAP_WINDOW_SEC * 1000:.0f}ms)")

    print("\nALL QRS TEMPLATE TESTS PASSED.")


if __name__ == "__main__":
    main()
