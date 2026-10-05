#!/usr/bin/env python3
"""
Regression tests for the fixes made after the 2026-09-23 audit
(multisignal-annotation_audit_20260923-1901.md):
    C1  Step 1 template detection no longer collapses above ~100 bpm.
    C2  An interrupted session still pushes its saved work to Box, and a
        leftover local copy with unpushed work is never silently overwritten.
    C3  A peak added by click-drag is resolved within the drawn region in
        Step 1 and Step 3, not at the drag's left edge.
    Session-summary dialog: discard writes nothing, "go back" reopens the
        viewer, an unchanged re-save is skipped, and the finished status is
        recorded; the tracking-spreadsheet reminder is the last thing printed.

Every check compares against an answer known before the code runs.
Tolerances: a detected/exported peak matches a true one within +/-20 ms.

Run with: annotate_env\\Scripts\\python.exe test_audit_fixes.py
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
from unittest.mock import patch

import mne
import neurokit2 as nk
import numpy as np
import pandas as pd

import box_sync
import physio_review
import processing_log
import qrs_template
import session_summary_gui
from annotation_io import SessionResult, run_stage_b
from channel_config import CHANNELS
from physio_io import build_run_path, generate_synthetic_demo
from reconcile import _agree_label, export_reconciled

session_summary_gui.show_session_summary = session_summary_gui.headless_decision()

FS = 1000
TOL = 20  # samples at 1000 Hz = 20 ms


def match(detected, true, tol=TOL):
    """(recall, precision) with one-to-one matching within tol samples."""
    detected = np.sort(np.asarray(detected))
    used, tp = set(), 0
    for t in true:
        j = np.searchsorted(detected, t)
        cands = [k for k in (j - 1, j) if 0 <= k < len(detected) and k not in used and abs(detected[k] - t) <= tol]
        if cands:
            used.add(min(cands, key=lambda k: abs(detected[k] - t)))
            tp += 1
    return tp / len(true), tp / max(len(detected), 1)


def simulated_ecg(hr, duration, seed):
    ecg = nk.ecg_simulate(duration=duration, sampling_rate=FS, heart_rate=hr, random_state=seed)
    signals, _ = nk.ecg_process(ecg, sampling_rate=FS)
    return np.asarray(ecg), np.where(signals["ECG_R_Peaks"] == 1)[0]


def template_detect(ecg, true):
    wav = qrs_template.apply_wavelet_filter(ecg)
    window = true[true < 20 * FS]
    template, adj = qrs_template.build_qrs_template(ecg[:20 * FS], wav[:20 * FS], window, sampling_rate=FS)
    rr = float(np.median(np.diff(window)))
    return qrs_template.extract_and_refine_peaks(ecg, wav, template, adj, sampling_rate=FS, reference_rr_ms=rr)


def test_c1_heart_rates():
    print("C1. Template detection holds recall/precision >= 0.98 from 60 to 180 bpm, and when heart "
          "rate changes within a run (the old fixed 600 ms spacing gave recall 0.46 at 110 bpm)...")
    for hr in (60, 110, 150, 180):
        ecg, true = simulated_ecg(hr, 60, 2024)
        recall, precision = match(template_detect(ecg, true), true)
        assert recall >= 0.98 and precision >= 0.98, (hr, recall, precision)
        print(f"   OK: {hr} bpm recall {recall:.3f} precision {precision:.3f}")
    for label, (hr1, hr2) in {"65->130": (65, 130), "130->70": (130, 70)}.items():
        e1, t1 = simulated_ecg(hr1, 40, 2024)
        e2, t2 = simulated_ecg(hr2, 40, 2025)
        ecg, true = np.concatenate([e1, e2]), np.concatenate([t1, t2 + len(e1)])
        recall, precision = match(template_detect(ecg, true), true)
        assert recall >= 0.98 and precision >= 0.98, (label, recall, precision)
        print(f"   OK: {label} bpm within one run: recall {recall:.3f} precision {precision:.3f}")


def test_c3_stage1_drag():
    print("C3a. Step 1: window peaks added by click-drag (150 ms before the R wave, 300 ms wide) land "
          "on the true R peak, and the resulting template still detects the whole run...")
    ecg, true = simulated_ecg(70, 60, 2024)
    col = np.zeros(len(ecg), bool)
    col[true] = True
    df = pd.DataFrame({"ecg": ecg, "ecg_peaks": col})
    window_true = true[true < 20 * FS]
    captured = {}
    original_build = qrs_template.build_qrs_template

    def spy_build(raw_window, wav_window, peaks, sampling_rate=1000):
        captured["peaks"] = np.asarray(peaks)
        return original_build(raw_window, wav_window, peaks, sampling_rate=sampling_rate)

    def drag_every_peak(self, *args, **kwargs):
        onsets = window_true / FS - 0.150
        self.set_annotations(mne.Annotations(onset=onsets, duration=[0.300] * len(onsets),
                                             description=["peak_ecg"] * len(onsets)))

    tmp = tempfile.mkdtemp()
    try:
        with patch.object(mne.io.BaseRaw, "plot", drag_every_peak), \
             patch.object(qrs_template, "show_template_preview", return_value=(True, "")), \
             patch.object(qrs_template, "build_qrs_template", spy_build), \
             contextlib.redirect_stdout(io.StringIO()):
            results = qrs_template.run_stage_a({"1": df}, {"1": "x"}, {"1": tmp}, FS, "zz")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    offsets = [int(captured["peaks"][np.argmin(abs(captured["peaks"] - t))] - t) for t in window_true]
    assert max(abs(o) for o in offsets) <= 2, offsets
    recall, precision = match(results["1"]["final_peaks"], true)
    assert recall >= 0.98 and precision >= 0.98, (recall, precision)
    print(f"   OK: max window-peak offset {max(abs(o) for o in offsets)} samples (was ~170 ms); "
          f"run recall {recall:.3f} precision {precision:.3f}")


def test_c3_stage3_drag():
    print("C3b. Step 3: a peak the reconciler ADDS by click-drag is saved at the true peak "
          "(was 173 ms early)...")
    ecg, true = simulated_ecg(70, 30, 2024)
    raw = mne.io.RawArray(ecg[None, :], mne.create_info(["ecg"], FS, ["ecg"]), verbose=False)
    t0 = int(true[10])
    raw.set_annotations(mne.Annotations(onset=[t0 / FS - 0.150], duration=[0.300],
                                        description=[_agree_label("peak_ecg")]))
    out = export_reconciled(raw, pd.DataFrame({"ecg": ecg}), {"ecg": CHANNELS["ecg"]}, FS, "a", "b")
    assert abs(out["ecg"]["indices"][0] - t0) <= 2, (out["ecg"]["indices"], t0)
    print(f"   OK: exported {out['ecg']['indices'][0]} vs true {t0}")


def make_fake_real_subject(box_path, subject, run, duration_sec=20):
    df, sfreq = generate_synthetic_demo(duration_sec=duration_sec)
    tsv_path = build_run_path(box_path, subject, run)
    os.makedirs(os.path.dirname(tsv_path), exist_ok=True)
    df.to_csv(tsv_path, sep="\t", index=False, header=False, compression="gzip")
    with open(tsv_path.replace("_physio.tsv.gz", "_physio.json"), "w") as f:
        json.dump({"Columns": list(df.columns), "SamplingFrequency": sfreq}, f)
    return tsv_path


def run_main(argv):
    with patch.object(sys, "argv", ["physio_review.py"] + argv):
        physio_review.main()


def test_c2_interrupted_session_still_pushes():
    print("C2a. Ctrl-C after the annotation save (e.g. at a prompt) still pushes the saved file to Box, "
          "and the tracker reminder is still the last thing printed...")
    tmp = tempfile.mkdtemp()
    try:
        box, local = os.path.join(tmp, "box"), os.path.join(tmp, "local")
        os.makedirs(local)
        make_fake_real_subject(box, "071", "1")
        target = "sub-071_ses-run1_task-sdi_annotations_zz.json"

        def save_then_interrupt(args, real_box_path, info):
            path = os.path.join(args.box_path, "sub-071", "ses-run1", "beh", target)
            with open(path, "w") as f:
                json.dump({"version": "saved-this-session"}, f)
            info.update(stage="2", run="1", channels=["ecg"], saved=True, finished=True)
            raise KeyboardInterrupt

        out = io.StringIO()
        with patch.object(physio_review, "_run_session", save_then_interrupt), \
             patch.object(processing_log, "LOCAL_SNAPSHOT_DIR", tmp), contextlib.redirect_stdout(out):
            try:
                run_main(["--box-path", box, "--local-path", local, "--subject", "071", "--run", "1",
                          "--initials", "zz", "--stage", "2"])
                raise AssertionError("expected a non-zero exit after Ctrl-C")
            except SystemExit as e:
                assert e.code == 130, e.code
        with open(os.path.join(box, "sub-071", "ses-run1", "beh", target)) as f:
            assert json.load(f)["version"] == "saved-this-session"
        assert not os.path.exists(os.path.join(local, "sub-071")), "local copy should be cleaned up after the push"
        lines = [l for l in out.getvalue().splitlines() if l.strip()]
        assert lines[-1].startswith("=====") and "TRACKING SPREADSHEET" in "\n".join(lines[-6:]), lines[-6:]
        print("   OK: saved file reached Box, local copy removed, reminder printed last")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_c2_leftover_local_copy():
    print("C2b. A leftover local copy with unpushed work is detected before copy-down; declining stops "
          "without touching it, accepting pushes it first...")
    tmp = tempfile.mkdtemp()
    try:
        box, local = os.path.join(tmp, "box"), os.path.join(tmp, "local")
        beh = os.path.join(box, "sub-001", "ses-run1", "beh")
        os.makedirs(beh)
        box_file = os.path.join(beh, "sub-001_ses-run1_task-sdi_annotations_zz.json")
        with open(box_file, "w") as f:
            json.dump({"version": "box-old"}, f)
        box_sync.copy_subject_tree_to_local(box, "001", local)
        local_file = os.path.join(local, "sub-001", "ses-run1", "beh", os.path.basename(box_file))
        assert box_sync.find_unpushed_local_files(box, "001", local) == [], "an untouched copy has nothing unpushed"
        time.sleep(2.5)
        with open(local_file, "w") as f:
            json.dump({"version": "local-unpushed"}, f)
        unpushed = box_sync.find_unpushed_local_files(box, "001", local)
        assert unpushed == [os.path.relpath(local_file, local)], unpushed

        with patch("builtins.input", return_value="n"), contextlib.redirect_stdout(io.StringIO()):
            try:
                physio_review._handle_leftover_local_copy(box, "001", local)
                raise AssertionError("expected SystemExit when the RA declines")
            except SystemExit:
                pass
        with open(local_file) as f:
            assert json.load(f)["version"] == "local-unpushed", "declining must not touch the local copy"
        with open(box_file) as f:
            assert json.load(f)["version"] == "box-old"

        with patch("builtins.input", return_value="y"), contextlib.redirect_stdout(io.StringIO()):
            physio_review._handle_leftover_local_copy(box, "001", local)
        with open(box_file) as f:
            assert json.load(f)["version"] == "local-unpushed", "accepting must push the leftover file to Box"
        assert box_sync.find_unpushed_local_files(box, "001", local) == []
        print("   OK: detected; 'n' left everything untouched; 'y' pushed it to Box")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_session_summary_dialog():
    print("D. Session summary: count_changes, discard, go back, unchanged re-save, finished status...")
    cfg = {"ecg": CHANNELS["ecg"], "eda": CHANNELS["eda"]}
    changes = session_summary_gui.count_changes(
        {"ecg": {"mode": "point", "indices": [100, 1100, 2100]}, "eda": {"mode": "segment", "bad_segments": []}},
        {"ecg": {"mode": "point", "indices": [110, 2100, 3100]}, "eda": {"mode": "segment", "bad_segments": [[5, 9]]}},
        cfg, 50)
    # A 10-sample nudge is a MOVE (not an add + remove), and it counts as a
    # change (ppg-plan §3.1 #3: until 2026-09-26 it counted as nothing).
    assert changes == {"ecg": {"added": 1, "removed": 1, "moved": 1}, "eda": {"added": 1, "removed": 0}}, changes
    assert session_summary_gui.total_changes(changes) == 4
    print("   OK: count_changes (a 10-sample nudge is one move; one removed, one added, one segment)")

    df, sfreq = generate_synthetic_demo(duration_sec=20)
    tmp = tempfile.mkdtemp()
    ecg_only = {"ecg": CHANNELS["ecg"]}
    out_path = os.path.join(tmp, "t_annotations_zz.json")
    quiet = contextlib.redirect_stdout(io.StringIO())
    try:
        with patch.object(mne.io.RawArray, "plot", return_value=None), \
             patch.object(session_summary_gui, "show_session_summary",
                          session_summary_gui.headless_decision(action=session_summary_gui.DISCARD)), quiet:
            result = run_stage_b(df, sfreq, ecg_only, "zz", tmp, "t", "synthetic")
        assert result.discarded and not result.saved and not os.path.exists(out_path)
        print("   OK: Discard wrote nothing")

        answers = iter([session_summary_gui.SessionDecision(session_summary_gui.BACK),
                        session_summary_gui.SessionDecision(session_summary_gui.SAVE, comment="c", finished=True)])
        plot_calls = []
        with patch.object(mne.io.RawArray, "plot", lambda self, *a, **k: plot_calls.append(1)), \
             patch.object(session_summary_gui, "show_session_summary", lambda *a, **k: next(answers)), \
             contextlib.redirect_stdout(io.StringIO()):
            result = run_stage_b(df, sfreq, ecg_only, "zz", tmp, "t", "synthetic")
        assert len(plot_calls) == 2, "Go back must reopen the viewer"
        assert result.saved and result.finished and result.comment == "c"
        with open(out_path) as f:
            assert json.load(f)["channels"]["ecg"]["status"] == "complete"
        print("   OK: Go back reopened the viewer; Save recorded status 'complete'")

        mtime = os.path.getmtime(out_path)
        with patch.object(mne.io.RawArray, "plot", return_value=None), \
             patch.object(session_summary_gui, "show_session_summary",
                          session_summary_gui.headless_decision(finished=True)), \
             contextlib.redirect_stdout(io.StringIO()):
            result = run_stage_b(df, sfreq, ecg_only, "zz", tmp, "t", "synthetic")
        assert not result.saved and os.path.getmtime(out_path) == mtime
        assert not os.path.exists(os.path.join(tmp, "backups")) or len(os.listdir(os.path.join(tmp, "backups"))) == 0
        print("   OK: an unchanged session with an unchanged status wrote nothing (no new backup)")

        with patch.object(mne.io.RawArray, "plot", return_value=None), \
             patch.object(session_summary_gui, "show_session_summary",
                          session_summary_gui.headless_decision(finished=False)), \
             contextlib.redirect_stdout(io.StringIO()):
            result = run_stage_b(df, sfreq, ecg_only, "zz", tmp, "t", "synthetic")
        with open(out_path) as f:
            assert result.saved and json.load(f)["channels"]["ecg"]["status"] == "in_progress"
        print("   OK: changing only the finished status still saves it")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_tracker_reminder_text():
    print("R. Tracker reminder wording by outcome...")
    base = {"synthetic": False, "subject": "085", "run": "2", "channels": ["ecg"], "stage": "2",
            "saved": True, "finished": True, "discarded": False, "ended_early": False}
    # The Google sheet's per-channel dropdowns are "in progress" / "finished" (HLU, 2026-09-27): no dates.
    finished = physio_review._tracker_reminder_text(base)
    assert 'to "finished"' in finished and "ecg column" in finished and "sub-085, run 2" in finished, finished
    assert "/" not in finished.replace("in progress", ""), "no date is asked for"
    assert '"in progress"' in physio_review._tracker_reminder_text({**base, "finished": False})
    assert "reconciler row" in physio_review._tracker_reminder_text({**base, "stage": "3"})
    assert "part of ECG" in physio_review._tracker_reminder_text({**base, "stage": "1"})
    start = physio_review._tracker_start_text(base)
    assert '"in progress"' in start and "ecg column" in start and "sub-085, run 2" in start, start
    assert physio_review._tracker_start_text({**base, "synthetic": True}) is None
    assert "discarded" in physio_review._tracker_reminder_text({**base, "saved": False, "discarded": True})
    assert "ended before anything was saved" in physio_review._tracker_reminder_text(
        {**base, "saved": False, "ended_early": True})
    assert physio_review._tracker_reminder_text({**base, "synthetic": True}) is None
    print(f"   OK: e.g. {finished!r}")


def test_initials_rule_everywhere():
    print("Initials: physio_annotate.py and qrs_template_stage.py lower-case and check them like physio_review.py...")
    import physio_annotate
    import qrs_template_stage
    tmp = tempfile.mkdtemp()
    try:
        seen = {}
        with patch.object(sys, "argv", ["physio_annotate.py", "--synthetic", "--duration", "20", "--initials", " HLU ",
                                        "--out-dir", tmp, "--ecg-source", "batch", "--no-ppg-guide"]), \
             patch.object(physio_annotate, "run_stage_b", lambda *a, **k: seen.setdefault("initials", a[3])), \
             contextlib.redirect_stdout(io.StringIO()):
            physio_annotate.main()
        assert seen["initials"] == "hlu", seen
        for script, argv in ((physio_annotate, ["physio_annotate.py", "--synthetic", "--initials", "sub-003"]),
                             (qrs_template_stage, ["qrs_template_stage.py", "--synthetic", "--initials", "h"])):
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                try:
                    script.main()
                    raise AssertionError(f"{argv} accepted")
                except SystemExit as e:
                    assert "isn't valid initials" in str(e), e
        print("   OK: ' HLU ' -> 'hlu'; 'sub-003' and 'h' refused with the plain message")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_c1_heart_rates()
    test_c3_stage1_drag()
    test_c3_stage3_drag()
    test_c2_interrupted_session_still_pushes()
    test_c2_leftover_local_copy()
    test_session_summary_dialog()
    test_tracker_reminder_text()
    test_initials_rule_everywhere()
    print("\nALL AUDIT-FIX TESTS PASSED.")
