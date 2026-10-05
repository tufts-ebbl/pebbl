#!/usr/bin/env python3
"""
Tests for Stage 3 (reconcile.py): diffing two RAs' (or an RA's vs. a
gold-standard reviewer's) saved annotation JSONs for the same run.

Run with: annotate_env\\Scripts\\python.exe test_reconcile.py
"""

import glob
import io
import json
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from unittest.mock import patch

import mne
import numpy as np

import physio_review
from annotation_io import save_annotation_json
from channel_config import CHANNELS
from physio_io import generate_synthetic_demo
from reconcile import (
    build_reconciliation_raw,
    export_reconciled,
    match_points,
    match_segments,
    run_stage_c,
)

import session_summary_gui  # noqa: E402

# The session-summary dialog is modal; answer "Save" immediately so these
# non-interactive tests never block (tests needing other answers patch it).
session_summary_gui.show_session_summary = session_summary_gui.headless_decision()


def main():
    print("1. match_points(): exact/near duplicates agree, unmatched ones fall to only_a/only_b, "
          "global greedy matching (order-independent) resolves ambiguous close pairs correctly...")
    # A: 100, 500 (close to two different B's, but should each get its own best match)
    # B: 110, 490, 900 (900 has no match at all in A)
    agreed, only_a, only_b = match_points([100, 500], [110, 490, 900], tolerance_samples=50)
    assert agreed == [(100, 110), (500, 490)], f"got {agreed}"
    assert only_a == [], f"got {only_a}"
    assert only_b == [900], f"got {only_b}"
    print(f"   OK: agreed={agreed}, only_a={only_a}, only_b={only_b}")

    print("2. match_points(): a gap beyond tolerance never matches, even if it's the closest option...")
    agreed, only_a, only_b = match_points([1000], [1200], tolerance_samples=50)
    assert agreed == [] and only_a == [1000] and only_b == [1200]
    print("   OK: 200-sample gap with a 50-sample tolerance correctly stays unmatched")

    print("3. match_points(): empty inputs on either side don't crash...")
    assert match_points([], [100, 200], tolerance_samples=50) == ([], [], [100, 200])
    assert match_points([100, 200], [], tolerance_samples=50) == ([], [100, 200], [])
    assert match_points([], [], tolerance_samples=50) == ([], [], [])
    print("   OK")

    print("4. match_segments(): spans overlapping by at least half their combined span agree "
          "(shown as their union); smaller overlaps and non-overlapping spans don't...")
    agreed, only_a, only_b = match_segments([[100, 200], [500, 600]], [[120, 210], [900, 1000]])
    assert agreed == [([100, 200], [120, 210])], f"got {agreed}"  # overlap 80 / span 110 = 0.73
    assert only_a == [[500, 600]], f"got {only_a}"
    assert only_b == [[900, 1000]], f"got {only_b}"
    print(f"   OK: agreed={agreed}, only_a={only_a}, only_b={only_b}")
    # Overlap 50 / span 150 = 0.33: no longer "agreed" (audit M2) -- unless the
    # old any-overlap rule is requested explicitly.
    agreed_small, only_a_small, only_b_small = match_segments([[100, 200]], [[150, 250]])
    assert agreed_small == [] and only_a_small == [[100, 200]] and only_b_small == [[150, 250]]
    assert match_segments([[100, 200]], [[150, 250]], min_overlap_fraction=0)[0] == [([100, 200], [150, 250])]
    print("   OK: a one-third overlap is shown as two separate marks; min_overlap_fraction=0 restores any-overlap")

    print("5. build_reconciliation_raw(): seeds three distinct, correctly-labeled descriptions "
          "per point channel -- agree (at the matched reviewer sample with the higher signal), only_A, only_B...")
    df, sfreq = generate_synthetic_demo(duration_sec=10)
    outputs_a = {"ecg": {"mode": "point", "indices": [1000, 5000]}}  # 5000 has no match -> only_a
    outputs_b = {"ecg": {"mode": "point", "indices": [1020, 8000]}}  # 1020 matches 1000 -> agree; 8000 -> only_b
    raw, diff_summary = build_reconciliation_raw(
        df, {"ecg": CHANNELS["ecg"]}, sfreq, outputs_a, outputs_b, "ra1", "ra2", tolerance_samples=50
    )
    descriptions = sorted(a["description"] for a in raw.annotations)
    assert descriptions == ["peak_ecg_added", "peak_ecg_agree", "peak_ecg_only_ra1", "peak_ecg_only_ra2"], f"got {descriptions}"
    # "peak_ecg_added" is a zero-duration placeholder that registers the label for marks the
    # reconciler adds (audit H4); it is never exported itself.
    assert [(a["onset"], a["duration"]) for a in raw.annotations if a["description"] == "peak_ecg_added"] == [(0.0, 0.0)]
    agree_ann = [a for a in raw.annotations if a["description"] == "peak_ecg_agree"][0]
    # Since 2026-09-26: seeded at whichever reviewer's sample sits higher on the
    # signal, not the midpoint (ppg-plan §3.1 #2), so export can keep it exactly.
    higher = 1000 if df["ecg"].iloc[1000] >= df["ecg"].iloc[1020] else 1020
    assert int(round(agree_ann["onset"] * sfreq)) == higher, (
        "the agreed mark's seeded position must be the matched reviewer sample with the higher signal"
    )
    assert diff_summary["ecg"] == {"agreed": 1, "only_ra1": 1, "only_ra2": 1}
    print(f"   OK: descriptions={descriptions}, diff_summary={diff_summary}")

    print("6. build_reconciliation_raw(): a channel only ONE side reviewed is silently skipped "
          "(nothing to compare)...")
    outputs_a_partial = {"ecg": outputs_a["ecg"], "ppg": {"mode": "point", "indices": [200]}}
    outputs_b_partial = {"ecg": outputs_b["ecg"]}  # no "ppg" at all
    raw2, diff_summary2 = build_reconciliation_raw(
        df, {"ecg": CHANNELS["ecg"], "ppg": CHANNELS["ppg"]}, sfreq,
        outputs_a_partial, outputs_b_partial, "ra1", "ra2",
    )
    assert set(diff_summary2.keys()) == {"ecg"}, f"got {diff_summary2.keys()}"
    print(f"   OK: only 'ecg' reconciled, 'ppg' skipped since ra2 never reviewed it")

    print("7. export_reconciled(): collapses whichever marks the reviewer LEFT (regardless of "
          "which of the 3 categories they started in) into one plain peak list, snapped to the "
          "true local max -- simulating a reviewer who deletes one only_ra2 mark and leaves the "
          "rest untouched...")
    raw3, _ = build_reconciliation_raw(
        df, {"ecg": CHANNELS["ecg"]}, sfreq, outputs_a, outputs_b, "ra1", "ra2", tolerance_samples=50
    )
    # Simulate the reviewer rejecting the only_ra2 mark (at 8000) by removing it.
    kept = [a for a in raw3.annotations if not (a["description"] == "peak_ecg_only_ra2")]
    import mne as mne_module
    raw3.set_annotations(mne_module.Annotations(
        onset=[a["onset"] for a in kept], duration=[a["duration"] for a in kept],
        description=[a["description"] for a in kept],
    ))
    reconciled_outputs = export_reconciled(raw3, df, {"ecg": CHANNELS["ecg"]}, sfreq, "ra1", "ra2")
    assert len(reconciled_outputs["ecg"]["indices"]) == 2, (
        f"expected 2 remaining peaks (agree + only_ra1, only_ra2 rejected), "
        f"got {reconciled_outputs['ecg']['indices']}"
    )
    print(f"   OK: final reconciled peaks = {reconciled_outputs['ecg']['indices']}")

    print("8. Full CLI end-to-end: two synthetic RA sessions (real peaks + a deliberate "
          "disagreement), then --stage 3 reconciles them, and the console diff summary + "
          "saved *_annotations_reconciled.json reflect the disagreement correctly...")
    tmp_dir = tempfile.mkdtemp(prefix="reconcile_e2e_")
    try:
        def run_main(argv):
            with patch.object(sys, "argv", ["physio_review.py"] + argv):
                physio_review.main()

        # RA "aaa" reviews run 1 untouched (auto-seeded peaks only).
        with patch.object(mne.io.RawArray, "plot", return_value=None):
            run_main(["--synthetic", "--duration", "30", "--run", "1", "--initials", "aaa",
                      "--out-dir", tmp_dir, "--stage", "2", "--channels", "ecg", "--ecg-source", "batch"])

        # RA "bbb" reviews the SAME run, but deletes the first seeded peak (a genuine disagreement).
        def bbb_plot(self, *args, **kwargs):
            # The first ECG peak mark (ECG sessions also carry the bad_ecg label's placeholder since 2026-09-27).
            self.annotations.delete(next(i for i, a in enumerate(self.annotations) if a["description"] == "peak_ecg"))

        with patch.object(mne.io.RawArray, "plot", bbb_plot):
            run_main(["--synthetic", "--duration", "30", "--run", "1", "--initials", "bbb",
                      "--out-dir", tmp_dir, "--stage", "2", "--channels", "ecg", "--ecg-source", "batch"])

        with open(glob.glob(os.path.join(tmp_dir, "*_annotations_aaa.json"))[0]) as f:
            aaa_indices = set(json.load(f)["channels"]["ecg"]["indices"])
        with open(glob.glob(os.path.join(tmp_dir, "*_annotations_bbb.json"))[0]) as f:
            bbb_indices = set(json.load(f)["channels"]["ecg"]["indices"])
        expected_only_aaa = aaa_indices - bbb_indices
        assert len(expected_only_aaa) == 1, f"expected exactly 1 disagreement, got {expected_only_aaa}"

        captured = {}

        def capture_reconcile_plot(self, *args, **kwargs):
            captured["descriptions"] = sorted(a["description"] for a in self.annotations)

        with patch.object(mne.io.RawArray, "plot", capture_reconcile_plot), \
             patch("builtins.input", side_effect=AssertionError("no terminal prompt expected")):
            run_main(["--synthetic", "--duration", "30", "--run", "1", "--initials", "hlu",
                      "--out-dir", tmp_dir, "--stage", "3", "--compare-initials", "aaa,bbb",
                      "--channels", "ecg"])

        assert "peak_ecg_only_aaa" in captured["descriptions"], captured["descriptions"]
        assert "peak_ecg_agree" in captured["descriptions"], captured["descriptions"]
        assert "peak_ecg_only_bbb" not in captured["descriptions"], (
            "bbb only DELETED a peak relative to aaa -- there should be no peak unique to bbb"
        )

        reconciled_path = glob.glob(os.path.join(tmp_dir, "*_annotations_reconciled.json"))[0]
        with open(reconciled_path) as f:
            reconciled = json.load(f)
        # The reconciler is now recorded (audit M2; it used to be initials: null).
        assert reconciled["initials"] == "hlu" and reconciled["reconciled_by"] == "hlu", reconciled["initials"]
        assert set(reconciled["reconciled_from"]) == {"aaa", "bbb"}
        ecg_agreement = reconciled["agreement"]["ecg"]
        assert ecg_agreement["only_a"] == 1 and ecg_agreement["only_b"] == 0, ecg_agreement
        assert 0 < ecg_agreement["percent_agreement"] < 100, ecg_agreement
        # The reviewer made no further edits in this test (raw.plot mocked to just capture),
        # so everything from both categories should have been exported.
        assert set(reconciled["channels"]["ecg"]["indices"]) >= expected_only_aaa, (
            "the disagreement (only in aaa) should survive into the reconciled output "
            "since the mocked plot didn't reject it"
        )
        print(f"   OK: reconciled file at {os.path.basename(reconciled_path)}, "
              f"reconciled_from={reconciled['reconciled_from']}, "
              f"{len(reconciled['channels']['ecg']['indices'])} final ecg peaks")

        print("9. --stage 3 errors clearly when one side has never actually reviewed this run...")
        try:
            run_main(["--synthetic", "--duration", "30", "--run", "1", "--initials", "hlu",
                      "--out-dir", tmp_dir, "--stage", "3", "--compare-initials", "aaa,zzz"])
            raise AssertionError("expected SystemExit for a missing comparison file")
        except SystemExit as e:
            assert "zzz" in str(e)
            print(f"   OK: {e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("10. build_reconciliation_raw()/export_reconciled()'s default tolerance/snap window "
          "scales with sfreq instead of a fixed 50-sample constant (regression check -- same "
          "reasoning as qrs_template.py's snap_window_samples fix: a fixed sample count means a "
          "different real-world tolerance at a different sampling rate)...")
    from annotation_io import DEFAULT_SNAP_WINDOW_SEC
    for test_sfreq in (500, 1000, 2000):
        expected_samples = round(DEFAULT_SNAP_WINDOW_SEC * test_sfreq)
        df_scaled, _ = generate_synthetic_demo(duration_sec=10, sfreq=test_sfreq)
        outputs_a_scaled = {"ecg": {"mode": "point", "indices": [1000]}}
        # Place B's "near-duplicate" peak exactly at the edge of the expected tolerance:
        # 1 sample INSIDE should agree, and this same offset must behave differently
        # at a different sfreq if the code were still using a fixed 50 (proving the
        # value actually comes from sfreq, not a leftover constant).
        offset = expected_samples - 1
        outputs_b_scaled = {"ecg": {"mode": "point", "indices": [1000 + offset]}}
        raw_scaled, diff_scaled = build_reconciliation_raw(
            df_scaled, {"ecg": CHANNELS["ecg"]}, test_sfreq, outputs_a_scaled, outputs_b_scaled, "ra1", "ra2"
        )
        assert diff_scaled["ecg"]["agreed"] == 1, (
            f"at {test_sfreq} Hz, an offset of {offset} samples (1 inside the expected "
            f"{expected_samples}-sample tolerance) should have agreed; got {diff_scaled['ecg']}"
        )
        reconciled_scaled = export_reconciled(raw_scaled, df_scaled, {"ecg": CHANNELS["ecg"]}, test_sfreq, "ra1", "ra2")
        assert reconciled_scaled["ecg"]["snap_window_samples"] == expected_samples, (
            f"at {test_sfreq} Hz expected snap_window_samples={expected_samples}, "
            f"got {reconciled_scaled['ecg']['snap_window_samples']}"
        )
    print(f"   OK: tolerance/snap window scales as 25/50/100 samples at 500/1000/2000 Hz "
          f"(all = {DEFAULT_SNAP_WINDOW_SEC * 1000:.0f}ms)")

    print("\nALL RECONCILE TESTS PASSED.")


if __name__ == "__main__":
    main()
