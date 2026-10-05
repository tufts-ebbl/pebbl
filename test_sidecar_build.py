#!/usr/bin/env python3
"""
Known-answer tests for the 2026-09-26 build (multisignal-annotation_ppg-plan_20260925-2327.md
§4a, and the sidecar plan's Phases 1 and 2):

  4a  #1 Step 3 merges the reconciled file across sessions (ECG then PPG keeps both)
      #2 Step 3 keeps seeded marks exactly; only reconciler-drawn marks snap
      #3 a small move on a resumed, already-complete file is a change and is saved
      #4 seeded marks never share an exact start time, yet export to the same sample
      #5 many unusual-gap flags collapse to one line; merges stay first
      #6 marks with a label no channel uses are reported
      dialog: blocking lines disable Save
      #8 a sidecar with a UTF-8 byte-order mark loads
  P1  RSP is a segment channel; an old point-mode RSP review is kept as
      rsp_points_legacy, never seeds the session, and Step 3 refuses a mode mismatch
  P2  sub-001's MachineQC pre-fills exactly its 12 SBP spans and 1 RSP span;
      a sample-count mismatch or an old sidecar pre-fills nothing; the saved
      file records the seed

Run with: annotate_env\\Scripts\\python.exe test_sidecar_build.py
"""

import contextlib
import io
import json
import os
import shutil
import tempfile
from unittest.mock import patch

import mne
import numpy as np

import physio_review
import session_summary_gui
from annotation_io import (
    ONSET_NUDGE_SAMPLES,
    _format_warning_lines,
    export_annotations,
    finish_session,
    run_stage_b,
    seed_annotations,
    unknown_label_lines,
)
from channel_config import CHANNELS
from physio_io import build_raw, generate_synthetic_demo, load_physio_tsv, machine_qc_segments, read_sidecar
from reconcile import build_reconciliation_raw, export_reconciled, save_reconciled_json

session_summary_gui.show_session_summary = session_summary_gui.headless_decision(finished=True)

HERE = os.path.dirname(os.path.abspath(__file__))
# The run-7 fixture test_input/sub-001_* was retired on 2026-09-26 (HLU); the
# run-8 copy in test_input/run8/ is used when it's absent. Its ppg/ppg_peaks
# hashes equal the published vectors, and the version-dependent assertions
# check which one they got.
SUB001 = next((p for p in (os.path.join(HERE, "test_input", "sub-001_ses-run1_task-sdi_physio.tsv.gz"),
                           os.path.join(HERE, "test_input", "run8", "sub-001_ses-run1_task-sdi_physio.tsv.gz"))
               if os.path.exists(p)), os.path.join(HERE, "test_input", "sub-001_ses-run1_task-sdi_physio.tsv.gz"))
SUB085 = os.path.join(HERE, "test_input", "sub-085_ses-run2_task-sdi_physio.tsv.gz")
QUIET = contextlib.redirect_stdout


def test_reconciled_merge():
    print("4a#1. Step 3 merges: reconciling ecg, then ppg, in separate sessions keeps both...")
    tmp = tempfile.mkdtemp()
    try:
        ecg_out = {"ecg": {"mode": "point", "indices": [100, 900], "snap_window_samples": 50, "status": "complete"}}
        ppg_out = {"ppg": {"mode": "point", "indices": [320, 1120], "snap_window_samples": 50, "status": "complete"}}
        with QUIET(io.StringIO()):
            save_reconciled_json(tmp, "stem", ecg_out, 1000, "src", "aa", "bb", reconciler="hl",
                                 agreement={"ecg": {"agreed": 2}})
        with open(os.path.join(tmp, "stem_annotations_reconciled.json")) as f:
            ecg_saved = json.load(f)["channels"]["ecg"]
        out = io.StringIO()
        with QUIET(out):
            save_reconciled_json(tmp, "stem", ppg_out, 1000, "src", "cc", "dd", reconciler="xy",
                                 agreement={"ppg": {"agreed": 2}})
        with open(os.path.join(tmp, "stem_annotations_reconciled.json")) as f:
            final = json.load(f)
        assert set(final["channels"]) == {"ecg", "ppg"}, final["channels"].keys()
        assert final["channels"]["ecg"] == ecg_saved, "the ECG reconciliation must be carried forward unchanged"
        assert final["channels"]["ecg"]["reconciled_by"] == "hl" and final["channels"]["ppg"]["reconciled_by"] == "xy"
        assert set(final["agreement"]) == {"ecg", "ppg"} and final["schema_version"] == 2
        assert "Carrying forward" in out.getvalue()
        print("   OK: both channels present, ECG entry identical, per-channel reconciler recorded")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_step3_keeps_seeded_marks():
    print("4a#2. Step 3 export keeps untouched marks exactly (even off-maximum); drawn marks snap...")
    df, sfreq = generate_synthetic_demo(duration_sec=10)
    peaks = np.where(df["ecg_peaks"].to_numpy())[0]
    off_max = int(peaks[2] + 30)  # a reviewer mark 30 ms off the R peak (not a local maximum)
    a = {"ecg": {"mode": "point", "indices": [int(peaks[1]), off_max]}}
    b = {"ecg": {"mode": "point", "indices": [int(peaks[1]) + 4]}}
    raw, _ = build_reconciliation_raw(df, {"ecg": CHANNELS["ecg"]}, sfreq, a, b, "aa", "bb", tolerance_samples=50)
    out = export_reconciled(raw, df, {"ecg": CHANNELS["ecg"]}, sfreq, "aa", "bb")
    signal = df["ecg"].to_numpy()
    expected_agree = int(peaks[1]) if signal[peaks[1]] >= signal[peaks[1] + 4] else int(peaks[1]) + 4
    assert out["ecg"]["indices"] == sorted([expected_agree, off_max]), out["ecg"]["indices"]
    # Now the reconciler draws a new mark around peaks[4] (a drag): it snaps to the maximum.
    drawn_onset = (peaks[4] - 60) / sfreq
    raw.set_annotations(raw.annotations + mne.Annotations([drawn_onset], [0.12], ["peak_ecg_added"]))
    out2 = export_reconciled(raw, df, {"ecg": CHANNELS["ecg"]}, sfreq, "aa", "bb")
    assert int(peaks[4]) in out2["ecg"]["indices"] and off_max in out2["ecg"]["indices"], out2["ecg"]["indices"]
    print(f"   OK: off-maximum mark kept at {off_max}; agreed pair at {expected_agree}; drawn mark snapped")


def test_small_move_is_saved():
    print("4a#3. A 40-sample move on a resumed, complete file counts as a change and is saved...")
    changes = session_summary_gui.count_changes({"ecg": {"mode": "point", "indices": [1000, 2000]}},
                                                {"ecg": {"mode": "point", "indices": [1040, 2000]}},
                                                {"ecg": CHANNELS["ecg"]}, 50)
    assert changes == {"ecg": {"added": 0, "removed": 0, "moved": 1}}, changes
    n = session_summary_gui.total_changes(changes)
    saved = []
    outputs = {"ecg": {"mode": "point", "indices": [1040, 2000]}}
    with QUIET(io.StringIO()):
        result = finish_session(session_summary_gui.SessionDecision(session_summary_gui.SAVE, finished=True),
                                outputs, n, {"ecg": "complete"}, True, lambda o: saved.append(o) or "p", "p")
    assert n == 1 and result.saved and saved, (n, result)
    lines = session_summary_gui.format_change_lines(changes, {"ecg": CHANNELS["ecg"]})
    assert lines == ["ECG: 0 peaks added, 0 removed, 1 moved"], lines
    print(f"   OK: {lines[0]!r}; the save ran")


def test_unique_onsets():
    print("4a#4. Seeded marks at the same sample get distinct start times but export to that sample...")
    df, sfreq = generate_synthetic_demo(duration_sec=10)
    shared = int(np.where(df["ecg_peaks"].to_numpy())[0][3])
    df.loc[shared, "ppg_peaks"] = True  # an ECG and a PPG peak on one sample
    cfg = {"ecg": CHANNELS["ecg"], "ppg": CHANNELS["ppg"], "eda": CHANNELS["eda"], "emg_cor": CHANNELS["emg_cor"]}
    raw = seed_annotations(build_raw(df, cfg, sfreq), df, cfg, {}, sfreq)
    onsets = list(raw.annotations.onset)
    assert len(set(onsets)) == len(onsets), "every seeded start time must be distinct"
    zero_placeholders = [o for o, d in zip(onsets, raw.annotations.description) if d in ("bad_eda", "bad_emg_cor")]
    assert len(set(zero_placeholders)) == 2 and max(zero_placeholders) < 0.5 / sfreq
    with QUIET(io.StringIO()):
        out = export_annotations(raw, df, cfg, sfreq)
    assert shared in out["ecg"]["indices"] and shared in out["ppg"]["indices"]
    assert ONSET_NUDGE_SAMPLES < 0.5 / 400
    print(f"   OK: {len(onsets)} distinct start times; sample {shared} exported for both ECG and PPG")


def test_warning_lines():
    print("4a#5/#6. Warning order and collapse; unknown labels reported...")
    flags = [("ppg", 10.0 + i, 1.2, 0.85, "sudden change (check this beat)") for i in range(49)]
    flags += [("ecg", 5.0, 1.6, 0.8, "long (possible missed peak)")]
    lines = _format_warning_lines([("ecg", 3.0, 0.9)], flags, CHANNELS, extra_lines=["PPG: check line"])
    assert lines[0].startswith("ECG: possible merged") and lines[1] == "PPG: check line"
    assert any(l.startswith("PPG: 49 unusual gaps") for l in lines) and len(lines) == 4, lines
    df, sfreq = generate_synthetic_demo(duration_sec=10)
    raw = build_raw(df, {"ecg": CHANNELS["ecg"]}, sfreq)
    raw.set_annotations(mne.Annotations([1.0, 2.0, 3.0], [0, 0, 0], ["peak_ecg", "peek_ecg", "peek_ecg"]))
    unknown = unknown_label_lines(raw, {"peak_ecg"})
    assert unknown == ["2 mark(s) labeled 'peek_ecg' will NOT be saved (not a label this session uses) -- "
                       "relabel or delete them"], unknown
    print(f"   OK: {lines}")


def test_blocking_dialog():
    print("dialog. Blocking lines disable Save...")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    dialog = session_summary_gui.SessionSummaryDialog("h", [], [], 0, False, "", blocking_lines=["PPG: two marks"])
    assert not dialog.save_button.isEnabled()
    dialog2 = session_summary_gui.SessionSummaryDialog("h", [], [], 0, False, "")
    assert dialog2.save_button.isEnabled()
    print("   OK")


def test_bom_sidecar():
    print("4a#8. A sidecar written with a UTF-8 byte-order mark loads...")
    tmp = tempfile.mkdtemp()
    try:
        df, sfreq = generate_synthetic_demo(duration_sec=10)
        tsv = os.path.join(tmp, "x_physio.tsv.gz")
        df[["ecg", "ecg_peaks"]].to_csv(tsv, sep="\t", index=False, header=False, compression="gzip")
        with open(os.path.join(tmp, "x_physio.json"), "w", encoding="utf-8-sig") as f:
            json.dump({"SamplingFrequency": 1000, "Columns": ["ecg", "ecg_peaks"],
                       "ecg": {"Units": "mV", "Description": "µ test"}}, f)
        loaded, fs = load_physio_tsv(tsv)
        assert fs == 1000 and list(loaded.columns) == ["ecg", "ecg_peaks"]
        print("   OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_rsp_legacy_and_mode_mismatch():
    print("P1. An old point-mode RSP review: kept as rsp_points_legacy, never seeds, Step 3 refuses...")
    assert CHANNELS["rsp"]["annotation_mode"] == "segment" and CHANNELS["rsp"]["segment_label"] == "bad_rsp"
    tmp = tempfile.mkdtemp()
    try:
        df, sfreq = generate_synthetic_demo(duration_sec=10)
        old = {"schema_version": 1, "initials": "zz", "channels": {
            "rsp": {"mode": "point", "indices": [1000, 5000], "status": "complete"},
            "ecg": {"mode": "point", "indices": [int(i) for i in np.where(df["ecg_peaks"])[0]], "status": "complete"}}}
        path = os.path.join(tmp, "t_annotations_zz.json")
        with open(path, "w") as f:
            json.dump(old, f)
        seen = {}

        def plot(self, *a, **k):
            seen["descriptions"] = sorted(set(self.annotations.description))

        out = io.StringIO()
        with patch.object(mne.io.RawArray, "plot", plot), QUIET(out):
            result = run_stage_b(df, sfreq, {"rsp": CHANNELS["rsp"]}, "zz", tmp, "t", "synthetic")
        assert seen["descriptions"] == ["bad_rsp"], seen  # only the label placeholder: no old breath peaks
        assert "starts fresh" in out.getvalue() and "rsp_points_legacy" in out.getvalue()
        with open(path) as f:
            saved = json.load(f)
        assert saved["channels"]["rsp"]["mode"] == "segment", saved["channels"]["rsp"]
        assert saved["channels"]["rsp_points_legacy"] == old["channels"]["rsp"]
        assert saved["channels"]["ecg"] == old["channels"]["ecg"]
        assert result.saved, "a mode-changed channel must not count as already saved"
        print("   OK: fresh segment review saved; old breath peaks kept under rsp_points_legacy")

        try:
            physio_review._check_modes_match({"rsp": {"mode": "point"}}, {"rsp": {"mode": "segment"}},
                                             {"rsp": CHANNELS["rsp"]}, "aa", "bb")
            raise AssertionError("expected SystemExit")
        except SystemExit as e:
            assert "aa" in str(e) and "bad stretches" in str(e), str(e)
        physio_review._check_modes_match({"rsp": {"mode": "segment"}}, {"rsp": {"mode": "segment"}},
                                         {"rsp": CHANNELS["rsp"]}, "aa", "bb")
        print("   OK: Step 3 refuses a point-mode vs segment-mode RSP comparison, allows segment vs segment")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_machine_qc_prefill():
    print("P2. sub-001: MachineQC pre-fills exactly its SBP spans and 1 RSP span; guards; seed record...")
    if not os.path.exists(SUB001):
        print("   SKIPPED: test_input/sub-001 not present")
        return
    df, sfreq = load_physio_tsv(SUB001)
    meta = read_sidecar(SUB001.replace("_physio.tsv.gz", "_physio.json"))
    qc, note = machine_qc_segments(meta, len(df))
    assert note is None, note
    # The no-reading stretches as parsed must match the fixture's own (12 in run 9; run 10 re-places BP at
    # each heartbeat's time, so the count isn't fixed). From run 9 SBP/DBP also carry "ppg_dropout"
    # stretches, so the total must equal the fixture's BadSegments (checked below).
    for k in ("sbp", "dbp"):
        raw = meta["MachineQC"]["Channels"][k]
        expected = len((raw.get("SegmentsByReason") or {}).get("no_coverage", raw["BadSegments"]))
        no_reading = (qc[k].get("by_reason") or {}).get("no_coverage", qc[k]["segments"])
        assert expected > 0 and len(no_reading) == expected, (k, len(no_reading), expected)
    # The run-7 fixture has no PPG mask (BadSegments []); a run-8 copy has one
    # (run-8 copies belong in test_input/run8/, not over this file).
    ppg_entry = meta["MachineQC"]["Channels"]["ppg"]
    expected_ppg = ppg_entry["BadSegments"] if "SegmentsByReason" in ppg_entry else []
    assert len(qc["rsp"]["segments"]) == 1 and qc["eda"]["segments"] == [], qc
    assert qc["ppg"]["segments"] == expected_ppg, "PPG MachineQC differs from the fixture's version (see test_input/run8/)"
    assert qc["sbp"]["segments"] == meta["MachineQC"]["Channels"]["sbp"]["BadSegments"]

    none, mismatch_note = machine_qc_segments(meta, len(df) - 1)
    assert none == {} and "742090" in mismatch_note, mismatch_note
    old_meta = read_sidecar(SUB085.replace("_physio.tsv.gz", "_physio.json")) if os.path.exists(SUB085) else {}
    none2, old_note = machine_qc_segments(old_meta, 1000)
    assert none2 == {} and "no machine quality check" in old_note

    tmp = tempfile.mkdtemp()
    try:
        cfg = {"sbp": CHANNELS["sbp"], "rsp": CHANNELS["rsp"]}
        out = io.StringIO()
        with patch.object(mne.io.RawArray, "plot", return_value=None), QUIET(out):
            result = run_stage_b(df, sfreq, cfg, "zz", tmp, "sub-001_ses-run1_task-sdi", "sub-001", sidecar=meta)
        with open(result.saved_path) as f:
            saved = json.load(f)
        assert saved["channels"]["sbp"]["bad_segments"] == qc["sbp"]["segments"]
        assert saved["channels"]["rsp"]["bad_segments"] == qc["rsp"]["segments"]
        assert saved["channels"]["rsp"]["bad_segments"][0][1] == 742000, "spans are used exactly as given"
        seed = saved["channels"]["sbp"]["seed"]
        assert seed["source"] == "MachineQC" and seed["segments"] == qc["sbp"]["segments"] and seed["rule"]
        assert seed["pipeline_commit"] == meta["Provenance"]["PipelineCommit"]
        text = out.getvalue()
        pct = 100 * sum(end - start for start, end in qc["rsp"]["segments"]) / len(df)
        assert f"the machine flagged nearly all of this run ({pct:.1f}%)" in text and "KEEP this flag" in text, text
        assert "stops 0.09 s before the end" in text, text
        print("   OK: pre-filled as given; seed recorded; >90% warning and end-gap note printed")

        # Resuming: the saved segments seed the session and the seed record is carried forward.
        seen = {}
        with patch.object(mne.io.RawArray, "plot", lambda self, *a, **k: seen.setdefault(
                "n", sum(d == "bad_sbp" and dur > 0 for d, dur in zip(self.annotations.description,
                                                                          self.annotations.duration)))),                 QUIET(io.StringIO()):
            run_stage_b(df, sfreq, {"sbp": CHANNELS["sbp"]}, "zz", tmp, "sub-001_ses-run1_task-sdi", "sub-001",
                        sidecar=meta)
        assert seen["n"] == len(qc["sbp"]["segments"]), (seen["n"], len(qc["sbp"]["segments"]))
        print("   OK: a resumed session seeds from the saved segments")

        # A mismatched sample count pre-fills nothing, with a note.
        tmp2 = tempfile.mkdtemp(dir=tmp)
        bad_meta = json.loads(json.dumps(meta))
        bad_meta["Provenance"]["NumberOfSamples"] = len(df) + 5
        out2 = io.StringIO()
        with patch.object(mne.io.RawArray, "plot", return_value=None), QUIET(out2):
            result2 = run_stage_b(df, sfreq, {"sbp": CHANNELS["sbp"]}, "zz", tmp2, "s", "s", sidecar=bad_meta)
        with open(result2.saved_path) as f:
            saved2 = json.load(f)
        assert saved2["channels"]["sbp"]["bad_segments"] == [] and "seed" not in saved2["channels"]["sbp"]
        assert "isn't used" in out2.getvalue()
        print("   OK: sample-count mismatch -> blank, with a note")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_checkpoint1_fixes():
    print("CP1. Fixes from the checkpoint-1 review...")
    import practice_certification as pc
    from annotation_io import keep_mode_changed_entries, legacy_channel_key, split_saved_by_mode
    from check_reprocess_alignment import saved_indices
    from reconcile import run_stage_c

    # D1-1: the nudge survives MNE's microsecond rounding above 1000 Hz.
    df2k, fs2k = generate_synthetic_demo(duration_sec=10, sfreq=2000)
    shared = int(np.where(df2k["ecg_peaks"].to_numpy())[0][2])
    df2k.loc[shared, "ppg_peaks"] = True
    cfg = {"ecg": CHANNELS["ecg"], "ppg": CHANNELS["ppg"]}
    raw2k = seed_annotations(build_raw(df2k, cfg, fs2k), df2k, cfg, {}, fs2k)
    assert len(set(raw2k.annotations.onset)) == len(raw2k.annotations.onset), "duplicates came back at 2000 Hz"
    with QUIET(io.StringIO()):
        out2k = export_annotations(raw2k, df2k, cfg, fs2k)
    assert shared in out2k["ecg"]["indices"] and shared in out2k["ppg"]["indices"]
    print("   OK: distinct start times at 2000 Hz after set_annotations")

    # D1-2: a pre-filled segment channel still has its label placeholder.
    df, sfreq = generate_synthetic_demo(duration_sec=10)
    qc = {"eda": {"segments": [[1000, 2000]]}}
    raw = seed_annotations(build_raw(df, {"eda": CHANNELS["eda"]}, sfreq), df, {"eda": CHANNELS["eda"]}, {}, sfreq,
                           machine_qc=qc)
    assert sorted(raw.annotations.duration) == [0.0, 1.0], list(raw.annotations.duration)
    print("   OK: placeholder kept alongside the pre-filled span")

    # SD-7 / D1-3: legacy keys never overwrite, and a mode-less entry is kept too.
    merged = {"rsp_points_legacy": {"mode": "point", "indices": [1]}, "rsp": {"mode": "point", "indices": [2]}}
    kept = keep_mode_changed_entries({"rsp": {"mode": "point", "indices": [2]}}, {"rsp": {"mode": "segment"}}, merged)
    assert kept == [("rsp", "rsp_points_legacy_2", "point")] and merged["rsp_points_legacy_2"]["indices"] == [2]
    assert legacy_channel_key("eda", None) == "eda_unknown_legacy"
    usable, changed = split_saved_by_mode({"eda": {"bad_segments": []}}, {"eda": CHANNELS["eda"]})
    assert changed == ["eda"] and usable == {}
    print("   OK: numbered legacy key; mode-less entry treated as changed")

    # D1-4/D1-5/D1-6/D1-11: malformed MachineQC never crashes; missing list isn't "clean"; notes scoped.
    meta = {"Provenance": {"NumberOfSamples": 1000.0}, "MachineQC": {"Channels": {
        "sbp": {"BadSegments": [[None, 10]]}, "dbp": {"Rule": "x"}, "eda": {"BadSegments": [[10.0, 20]]},
        "rsp": {"BadSegments": [[20, 10]]}}}}
    res, note = machine_qc_segments(meta, 1000)
    assert set(res) == {"eda"} and res["eda"]["segments"] == [[10, 20]], res
    assert "invalid one" in note and all(c in note for c in ("sbp", "dbp", "rsp")), note
    res, note = machine_qc_segments(meta, 1000, only=["eda", "rsp"], labels={"rsp": "RSP"})
    assert set(res) == {"eda"} and "RSP" in note and "sbp" not in note, note
    assert machine_qc_segments({"Provenance": {"NumberOfSamples": "742091.0"}, "MachineQC": {"Channels": {}}},
                               742091)[0] == {}
    print("   OK: malformed spans/counts skip with a scoped note, no crash")

    # SD-5: a segment ending exactly at the file end is not "past the end".
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "a.json")
        with open(path, "w") as f:
            json.dump({"channels": {"sbp": {"mode": "segment", "bad_segments": [[90, 100]]}}}, f)
        assert saved_indices([path]) == 99

        # SD-3 / SD-6: v1 reconciled file keeps its attribution; a mode-changed channel is kept as legacy.
        with open(os.path.join(tmp, "s_annotations_reconciled.json"), "w") as f:
            json.dump({"initials": "hl", "reconciled_by": "hl", "reconciled_from": ["aa", "bb"],
                       "channels": {"ecg": {"mode": "point", "indices": [5]},
                                    "rsp": {"mode": "point", "indices": [7]}}}, f)
        with QUIET(io.StringIO()):
            save_reconciled_json(tmp, "s", {"rsp": {"mode": "segment", "bad_segments": []}}, 1000, "src", "cc", "dd",
                                 reconciler="xy")
        with open(os.path.join(tmp, "s_annotations_reconciled.json")) as f:
            rec = json.load(f)
        assert rec["channels"]["ecg"]["reconciled_by"] == "hl" and rec["channels"]["ecg"]["reconciled_from"] == ["aa", "bb"]
        assert rec["channels"]["rsp_points_legacy"]["indices"] == [7] and rec["channels"]["rsp"]["mode"] == "segment"
        print("   OK: segment end exclusive; v1 attribution backfilled; old RSP consensus kept as legacy")

        # SD-1 / S3-1: Step 3 saves when the result differs from the saved reconciliation, even with 0 edits.
        a = {"ecg": {"mode": "point", "indices": [1000, 2000], "status": "complete"}}
        b = {"ecg": {"mode": "point", "indices": [1000, 2000], "status": "complete"}}
        rdir = os.path.join(tmp, "rec")
        with patch.object(mne.io.RawArray, "plot", return_value=None), QUIET(io.StringIO()):
            r1 = run_stage_c(df, sfreq, {"ecg": CHANNELS["ecg"]}, rdir, "t", "src", a, b, "aa", "bb", reconciler="hl")
            a2 = {"ecg": {"mode": "point", "indices": [1000, 2000, 3000], "status": "complete"}}
            b2 = {"ecg": {"mode": "point", "indices": [1000, 2000, 3000], "status": "complete"}}
            r2 = run_stage_c(df, sfreq, {"ecg": CHANNELS["ecg"]}, rdir, "t", "src", a2, b2, "aa", "bb", reconciler="hl")
            r3 = run_stage_c(df, sfreq, {"ecg": CHANNELS["ecg"]}, rdir, "t", "src", a2, b2, "aa", "bb", reconciler="hl")
        with open(os.path.join(rdir, "t_annotations_reconciled.json")) as f:
            assert json.load(f)["channels"]["ecg"]["indices"] == [1000, 2000, 3000]
        assert r1.saved and r2.saved and not r3.saved, (r1.saved, r2.saved, r3.saved)
        print("   OK: a changed re-reconciliation is saved; an identical one is not")

        # S3-5 / S3-8: Step 3 reports unknown labels and unfinished reviews in the summary.
        seen = {}

        def plot(self, *args, **kwargs):
            self.set_annotations(self.annotations + mne.Annotations([3.0], [0.1], ["peak_ecg"]))

        def summary(heading, change_lines, warning_lines, n, **kw):
            seen["warnings"] = warning_lines
            return session_summary_gui.SessionDecision(session_summary_gui.DISCARD)

        b_unfinished = {"ecg": {"mode": "point", "indices": [1000, 2000], "status": "in_progress"}}
        with patch.object(mne.io.RawArray, "plot", plot), \
             patch.object(session_summary_gui, "show_session_summary", summary), QUIET(io.StringIO()):
            run_stage_c(df, sfreq, {"ecg": CHANNELS["ecg"]}, os.path.join(tmp, "rec2"), "t", "src", a, b_unfinished,
                        "aa", "bb", reconciler="hl")
        assert any("'peak_ecg' will NOT be saved" in w for w in seen["warnings"]), seen
        assert any("bb's review is not marked finished" in w for w in seen["warnings"]), seen
        print("   OK: Step 3 warns about a Step 2 label and an unfinished review")

        # SD-8 / D1-6 / D1-7: practice scoring. (D1-9's key-outside-the-folder check is gone: since v5 the
        # encoded key lives in the practice folder by design, HLU 2026-10-04.)
        practice = os.path.join(tmp, "Practice")
        with QUIET(io.StringIO()):
            pc.make(practice, seed=7)
        key = pc.load_key(pc.key_path_for(practice))
        sbp = key["runs"]["1"]["channels"]["sbp"]
        ff = sbp["machine_qc"]["false_flag"]
        sub = {"initials": "zz", "channels": {"sbp": {"mode": "segment", "bad_segments":
               [[x["start"], x["end"]] for x in sbp["artifacts"]] + [[ff["start"], ff["start"] + 2000]]},
               "rsp": {"mode": "point", "indices": [1]}}}
        _overall, results, lines = pc.score_run(key, sub, "1", ["sbp", "rsp"])
        text = "\n".join(lines)
        assert results == {"sbp": False, "rsp": False}, results
        assert f"FIX  the machine's mark on {ff['type']} at {ff['start'] / 1000:.1f}" in text, text
        assert "33% still marked" in text and "CAN'T SCORE" in text, text
        print("   OK: keeping 2 s of the 6-s false flag fails and is named; mode mismatch can't score")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_checkpoint1_fixes()
    test_reconciled_merge()
    test_step3_keeps_seeded_marks()
    test_small_move_is_saved()
    test_unique_onsets()
    test_warning_lines()
    test_blocking_dialog()
    test_bom_sidecar()
    test_rsp_legacy_and_mode_mismatch()
    test_machine_qc_prefill()
    print("\nALL SIDECAR BUILD TESTS PASSED.")
