#!/usr/bin/env python3
"""
Known-answer tests for the sidecar plan's Phases 1b and 3 and the PPG plan's
Phase 4 (multisignal-annotation_ppg-plan_20260925-2327.md):

  1b  the read-only PPG guide row: shown under ECG in Steps 1-3 exactly as
      stored, never annotated or exported; off with ppg_guide=False; absent in
      a PPG session and when a run has no usable PPG; recorded as guide_shown;
      the Step 1 template is identical with and without it
  3   unit labels and descriptions come from the sidecar
  4b  PPG only on new-pipeline files (lab-staff override); PPG reviewed on its
      own; PPG check lines (off-top markers, markers in machine dropouts,
      two markers on one pulse -- which BLOCKS saving in Step 3)
  4c  per-channel provenance (sample count + the shared content hash) is saved
      in Steps 2 and 3; a changed signal stops a resumed session or Step 3;
      per-session edit history; schema_version 2
  4d  check_reprocess_alignment reports PPG separately
  run-8 decisions (2026-09-26): an empty (all-NaN) channel is dropped from
      the default and refused by name or when handed to Step 2/3 directly;
      the machine's SegmentsByReason shown and kept (malformed ones ignored);
      reviewed_span saved in Steps 2 and 3
  4e  the practice PPG: device dropout with a spurious seed, early-hump traps

Run with: annotate_env\\Scripts\\python.exe test_ppg_build.py
"""

import contextlib
import filecmp
import io
import json
import os
import shutil
import tempfile
from unittest.mock import patch

import mne
import numpy as np
import pandas as pd

import practice_certification as pc
import qrs_template
import session_summary_gui
from annotation_io import combine_with_reference_channels, export_annotations, run_stage_b
from channel_config import CHANNELS, GUIDE_CHANNELS, select_channels
from check_reprocess_alignment import compare_run
from physio_io import build_raw, generate_synthetic_demo, load_physio_tsv, read_sidecar, sidecar_unit_labels
from ppg_checks import markers_off_top, ppg_check_lines, same_beat_lines
from provenance import new_pipeline_ppg
from reconcile import build_reconciliation_raw, export_reconciled, run_stage_c

session_summary_gui.show_session_summary = session_summary_gui.headless_decision(finished=True)

HERE = os.path.dirname(os.path.abspath(__file__))
# The run-7 fixture test_input/sub-001_* was retired on 2026-09-26 (HLU); the
# copy in test_input/run8/ is used when it's absent. Since 2026-10-03 that
# copy is run 10's (physioProcess 1d74b15): sub-001 r1 lost CareTaker pulse
# samples, so its ppg/ppg_peaks changed from run 8/9's vectors (835251095b25...
# / 8b89cd443bb8...) to the run-10 vectors below, confirmed independently by
# the physioProcess maintainer (physioprocess-column-sha256-v1).
SUB001 = next((p for p in (os.path.join(HERE, "test_input", "sub-001_ses-run1_task-sdi_physio.tsv.gz"),
                           os.path.join(HERE, "test_input", "run8", "sub-001_ses-run1_task-sdi_physio.tsv.gz"))
               if os.path.exists(p)), os.path.join(HERE, "test_input", "sub-001_ses-run1_task-sdi_physio.tsv.gz"))
SUB085 = os.path.join(HERE, "test_input", "sub-085_ses-run2_task-sdi_physio.tsv.gz")
SUB001_PPG_SHA = "4ec43a9a88a65503cc0da8f9115ec6788815f480667a9feacfcffef08ff2a893"
QUIET = contextlib.redirect_stdout


def sidecar_of(tsv):
    return read_sidecar(tsv.replace("_physio.tsv.gz", "_physio.json"))


def test_guide_step2_and_3():
    print("1b. The PPG guide row in Steps 2 and 3 (opt-in under ECG since 2026-10-03)...")
    df, sfreq = generate_synthetic_demo(duration_sec=20)
    ecg_only = {"ecg": CHANNELS["ecg"]}
    assert list(combine_with_reference_channels(ecg_only, df, ppg_guide=True)) == ["ecg"], "off unless asked for"
    combined = combine_with_reference_channels(ecg_only, df, ppg_guide=True, ecg_ppg_guide=True)
    assert list(combined) == ["ecg", "ppg"] and combined["ppg"] is GUIDE_CHANNELS["ppg"]
    assert "ppg" not in combine_with_reference_channels(ecg_only, df, ppg_guide=False)
    assert combine_with_reference_channels({"ppg": CHANNELS["ppg"]}, df, ppg_guide=True)["ppg"] is CHANNELS["ppg"]
    nan_df = df.copy()
    nan_df["ppg"] = np.nan
    assert "ppg" not in combine_with_reference_channels(ecg_only, nan_df, ppg_guide=True, ecg_ppg_guide=True)

    raw = build_raw(df, combined, sfreq)
    assert np.array_equal(raw.get_data(picks=["ppg"])[0], df["ppg"].to_numpy(dtype=float)), "shown exactly as stored"
    raw.set_annotations(mne.Annotations([2.0], [0.1], ["peak_ecg"]))
    with QUIET(io.StringIO()):
        out = export_annotations(raw, df, combined, sfreq)
    assert set(out) == {"ecg"}, "the guide is never exported"

    tmp = tempfile.mkdtemp()
    try:
        seen = {}
        with patch.object(mne.io.RawArray, "plot", lambda self, *a, **k: seen.setdefault("names", list(self.ch_names))), \
             QUIET(io.StringIO()) as out_text:
            result = run_stage_b(df, sfreq, ecg_only, "zz", tmp, "g", "synthetic", ecg_ppg_guide=True)
        assert seen["names"] == ["ecg", "ppg"], seen
        assert "PPG (guide)" in out_text.getvalue()
        with open(result.saved_path) as f:
            saved = json.load(f)
        assert saved["channels"]["ecg"]["guide_shown"] is True and "ppg" not in saved["channels"]
        with patch.object(mne.io.RawArray, "plot", return_value=None), QUIET(io.StringIO()):
            run_stage_b(df, sfreq, ecg_only, "yy", tmp, "g", "synthetic", ppg_guide=False)
        with open(os.path.join(tmp, "g_annotations_yy.json")) as f:
            assert json.load(f)["channels"]["ecg"]["guide_shown"] is False

        # Step 3: the guide row is there, never exported, and recorded.
        a = {"ecg": {"mode": "point", "indices": [1000, 2000], "status": "complete"}}
        raw3, _ = build_reconciliation_raw(df, ecg_only, sfreq, a, a, "aa", "bb", ppg_guide=True, ecg_ppg_guide=True)
        assert list(raw3.ch_names) == ["ecg", "ppg"]
        assert set(export_reconciled(raw3, df, ecg_only, sfreq, "aa", "bb")) == {"ecg"}
        with patch.object(mne.io.RawArray, "plot", return_value=None), QUIET(io.StringIO()):
            r = run_stage_c(df, sfreq, ecg_only, os.path.join(tmp, "rec"), "g", "s", a, a, "aa", "bb", reconciler="hl",
                            ecg_ppg_guide=True)
        with open(r.saved_path) as f:
            assert json.load(f)["channels"]["ecg"]["guide_shown"] is True
        print("   OK: shown after ECG exactly as stored, never exported, recorded; off/absent when it should be")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_guide_step1_template_identical():
    print("1b. Step 1: the template is identical with and without the guide row...")
    df, sfreq = generate_synthetic_demo(duration_sec=40)
    peaks = np.where(df["ecg_peaks"].to_numpy())[0]
    outputs = {}
    for guide in (True, False):
        tmp = tempfile.mkdtemp()
        seen = {}

        def plot(self, *a, **k):
            seen["names"] = list(self.ch_names)

        with patch.object(mne.io.RawArray, "plot", plot), \
             patch.object(qrs_template, "show_template_preview", return_value=(True, "")), QUIET(io.StringIO()):
            res = qrs_template.run_stage_a({"1": df}, {"1": "t"}, {"1": tmp}, sfreq, "zz", ecg_ppg_guide=guide)
        outputs[guide] = (tmp, res["1"], seen["names"])
    (tmp_on, res_on, names_on), (tmp_off, res_off, names_off) = outputs[True], outputs[False]
    try:
        assert names_on == ["ecg", "ppg"] and names_off == ["ecg"]
        assert list(res_on["final_peaks"]) == list(res_off["final_peaks"]) and len(peaks) > 0
        assert filecmp.cmp(res_on["csv_path"], res_off["csv_path"], shallow=False), "template CSV must be byte-identical"
        with open(res_on["json_path"]) as f_on, open(res_off["json_path"]) as f_off:
            j_on, j_off = json.load(f_on), json.load(f_off)
        assert j_on["ecg"]["ppg_guide_shown"] is True and j_off["ecg"]["ppg_guide_shown"] is False
        assert j_on["ecg"]["corrected_peaks"] == j_off["ecg"]["corrected_peaks"]
        print(f"   OK: {len(res_on['final_peaks'])} peaks and the template CSV identical; guide recorded")
    finally:
        shutil.rmtree(tmp_on, ignore_errors=True)
        shutil.rmtree(tmp_off, ignore_errors=True)


def test_units_and_descriptions():
    print("3. Units and descriptions from the sidecar...")
    if not os.path.exists(SUB001):
        print("   SKIPPED: no sub-001")
        return
    meta = sidecar_of(SUB001)
    units, warnings = sidecar_unit_labels(meta, CHANNELS)
    assert units == {"ecg": "mV", "resp": "V", "bio": "AU", "gsr": "µS", "misc": "mmHg"}, units
    assert warnings == []
    odd = {"sbp": {"Units": "mmHg"}, "dbp": {"Units": "kPa"}, "eda": {"Units": "furlongs"}}
    _units, odd_warnings = sidecar_unit_labels(odd, CHANNELS)
    assert len(odd_warnings) == 2 and "furlongs" in " ".join(odd_warnings), odd_warnings
    df, sfreq = load_physio_tsv(SUB001)
    tmp = tempfile.mkdtemp()
    try:
        out = io.StringIO()
        with patch.object(mne.io.RawArray, "plot", return_value=None), QUIET(out):
            run_stage_b(df, sfreq, {"rsp": CHANNELS["rsp"]}, "zz", tmp, "s", "s", sidecar=meta)
        assert mne.defaults.DEFAULTS["units"]["resp"] == "V"
        assert meta["rsp"]["Description"] in out.getvalue(), "the channel's Description is shown"
        print("   OK: mapped units (RSP V, EDA µS, PPG AU); conflicts and unknown units warned; description shown")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gate_and_review_alone():
    print("4b. PPG only on new-pipeline files, and only on its own...")
    assert new_pipeline_ppg(None) and new_pipeline_ppg({"Provenance": {"Practice": True}})
    if os.path.exists(SUB001):
        assert new_pipeline_ppg(sidecar_of(SUB001))
    if os.path.exists(SUB085):
        assert not new_pipeline_ppg(sidecar_of(SUB085))
    assert "ppg" not in select_channels(None) and set(select_channels(["ppg"])) == {"ppg"}
    for bad in (["ecg", "ppg"], ["ppg", "eda"]):
        try:
            select_channels(bad)
            raise AssertionError("expected SystemExit")
        except SystemExit as e:
            assert "on its own" in str(e)
    df, sfreq = generate_synthetic_demo(duration_sec=10)
    tmp = tempfile.mkdtemp()
    try:
        try:
            with QUIET(io.StringIO()):
                run_stage_b(df, sfreq, {"ecg": CHANNELS["ecg"], "ppg": CHANNELS["ppg"]}, "zz", tmp, "t", "s")
            raise AssertionError("expected SystemExit")
        except SystemExit as e:
            assert "on its own" in str(e)
        old = {"SamplingFrequency": 1000, "Columns": list(df.columns)}
        try:
            with QUIET(io.StringIO()):
                run_stage_b(df, sfreq, {"ppg": CHANNELS["ppg"]}, "zz", tmp, "t", "s", sidecar=old)
            raise AssertionError("expected SystemExit")
        except SystemExit as e:
            assert "before the PPG timing fix" in str(e)
        with patch.object(mne.io.RawArray, "plot", return_value=None), QUIET(io.StringIO()):
            result = run_stage_b(df, sfreq, {"ppg": CHANNELS["ppg"]}, "zz", tmp, "t", "s", sidecar=old,
                                 allow_old_ppg=True)
        assert result.saved
        print("   OK: old-pipeline PPG refused (override works); PPG never combined; default leaves it out")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_ppg_checks():
    print("4b. PPG check lines and the Step 3 same-beat block...")
    if os.path.exists(SUB001):
        df, sfreq = load_physio_tsv(SUB001)
        seed = np.where(df["ppg_peaks"].to_numpy())[0]
        off = markers_off_top(df["ppg"].to_numpy(), seed, 50)
        assert len(off) == 11, len(off)  # the ppg study's count (ppg-plan §3.1 #2)
        # The crest test (+/-20 ms, 2026-10-04) flags the same 11 slope markers here.
        from ppg_checks import CREST_WINDOW_SEC
        assert markers_off_top(df["ppg"].to_numpy(), seed, round(CREST_WINDOW_SEC * sfreq)) == off
        print(f"   OK: sub-001's machine seed has {len(off)} markers not on a crest of a pulse (same at +/-50 ms)")
    # Two crests 30 ms apart (the higher one later): a mark on either crest passes the crest test; the old
    # +/-50 ms rule flagged the lower one. A mark on the upstroke is caught by both.
    t = np.arange(1000) / 1000.0
    pulse = np.exp(-0.5 * ((t - 0.40) / 0.012) ** 2) + 1.05 * np.exp(-0.5 * ((t - 0.43) / 0.012) ** 2)
    crests = [i for i in range(1, 999) if pulse[i] > 0.5 and pulse[i] >= pulse[i - 1] and pulse[i] >= pulse[i + 1]]
    assert len(crests) == 2, crests
    lower, upstroke = min(crests, key=lambda i: pulse[i]), 370
    assert markers_off_top(pulse, [lower], 50) == [lower] and markers_off_top(pulse, crests, 20) == []
    assert markers_off_top(pulse, [upstroke], 20) == [upstroke]
    lines = ppg_check_lines([lower, upstroke], pulse, 1000, [])
    assert len(lines) == 1 and lines[0].startswith("PPG: 1 marker(s) not on a crest of a pulse (near 0.4 s)"), lines
    print("   OK: either crest of a double-crested pulse passes; a mark on the upstroke is still caught")

    # Option A (HLU, 2026-10-04): a box dragged over the whole top of a two-humped pulse saves the mark on the
    # higher hump (the drawn-peak rule), and every PPG session (fresh or resumed), never ECG, prints how to add one.
    from channel_config import PPG_ADD_PEAK_TEXT
    demo, fs = generate_synthetic_demo(duration_sec=20)
    tt = np.arange(len(demo)) / fs
    humps = np.zeros(len(demo))
    for c in np.arange(1.0, 19.0, 0.9):
        humps += 0.8 * np.exp(-0.5 * ((tt - c) / 0.02) ** 2) + np.exp(-0.5 * ((tt - c - 0.1) / 0.02) ** 2)
    demo["ppg"] = humps
    ppg_only = {"ppg": CHANNELS["ppg"]}
    raw = build_raw(demo, ppg_only, fs)
    raw.set_annotations(mne.Annotations([5.5 - 0.06], [0.22], ["peak_ppg"]))  # over both humps of the 5.5-s pulse
    with QUIET(io.StringIO()):
        drawn = export_annotations(raw, demo, ppg_only, fs)["ppg"]["indices"]
    assert drawn == [int(round(5.6 * fs))], drawn
    tmp = tempfile.mkdtemp()
    try:
        for keys, shown in ((["ppg"], True), (["ppg"], True), (["ecg"], False)):
            out = io.StringIO()
            with patch.object(mne.io.RawArray, "plot", return_value=None), QUIET(out):
                run_stage_b(demo, fs, {k: CHANNELS[k] for k in keys}, "zz", tmp, "h" + keys[0], "s")
            assert (PPG_ADD_PEAK_TEXT in out.getvalue()) is shown, keys
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("   OK: a box over both humps lands on the higher one; the add-a-peak hint shows in PPG sessions only")
    df, sfreq = generate_synthetic_demo(duration_sec=20)
    peaks = list(np.where(df["ppg_peaks"].to_numpy())[0])
    doubled = sorted(peaks + [peaks[5] - 75])
    assert same_beat_lines(peaks, sfreq) == [] and len(same_beat_lines(doubled, sfreq)) == 1
    a = {"ppg": {"mode": "point", "indices": [int(i) for i in doubled], "status": "complete"}}
    captured = {}

    def summary(heading, change_lines, warning_lines, n, **kwargs):
        captured.update(kwargs)
        return session_summary_gui.SessionDecision(session_summary_gui.DISCARD)

    tmp = tempfile.mkdtemp()
    try:
        with patch.object(mne.io.RawArray, "plot", return_value=None), \
             patch.object(session_summary_gui, "show_session_summary", summary), QUIET(io.StringIO()):
            run_stage_c(df, sfreq, {"ppg": CHANNELS["ppg"]}, tmp, "t", "s", a, a, "aa", "bb", reconciler="hl")
        assert captured.get("blocking_lines") and "two markers" in captured["blocking_lines"][0], captured
        print("   OK: two markers on one pulse block saving in Step 3")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_provenance_and_history():
    print("4c. Provenance, the changed-signal stop, edit history...")
    if not os.path.exists(SUB001):
        print("   SKIPPED: no sub-001")
        return
    df, sfreq = load_physio_tsv(SUB001)
    meta = sidecar_of(SUB001)
    tmp = tempfile.mkdtemp()
    try:
        first = np.where(df["ppg_peaks"].to_numpy())[0][0]

        def delete_first(self, *a, **k):
            keep = [i for i, onset in enumerate(self.annotations.onset) if int(round(onset * sfreq)) != first]
            self.set_annotations(self.annotations[keep])

        with patch.object(mne.io.RawArray, "plot", delete_first), QUIET(io.StringIO()):
            result = run_stage_b(df, sfreq, {"ppg": CHANNELS["ppg"]}, "zz", tmp, "sub-001_ses-run1_task-sdi", "s",
                                 sidecar=meta, tsv_path=SUB001)
        with open(result.saved_path) as f:
            saved = json.load(f)
        entry = saved["channels"]["ppg"]
        assert saved["schema_version"] == 2
        assert entry["provenance"] == {"number_of_samples": 742091, "pipeline_commit": meta["Provenance"]["PipelineCommit"],
                                       "signal_sha256": SUB001_PPG_SHA,
                                       "signal_sha256_def": "physioprocess-column-sha256-v1"}, entry["provenance"]
        assert saved["coordinate_space"].endswith("bad_segments are [start, end)")
        history = entry["edit_history"]
        assert len(history) == 1 and history[0]["removed"] == [int(first)] and history[0]["added"] == []
        assert history[0]["started_from"] == "ppg_peaks column" and history[0]["initials"] == "zz"
        print("   OK: provenance = sample count + the shared hash; edit history records the deleted seed")

        # A changed signal stops the resumed session, before any viewer opens.
        saved["channels"]["ppg"]["provenance"]["signal_sha256"] = "0" * 64
        with open(result.saved_path, "w") as f:
            json.dump(saved, f)
        try:
            with patch.object(mne.io.RawArray, "plot", side_effect=AssertionError("viewer must not open")), \
                 QUIET(io.StringIO()):
                run_stage_b(df, sfreq, {"ppg": CHANNELS["ppg"]}, "zz", tmp, "sub-001_ses-run1_task-sdi", "s",
                            sidecar=meta, tsv_path=SUB001)
            raise AssertionError("expected SystemExit")
        except SystemExit as e:
            assert "different version of the file" in str(e) and "changed" in str(e)
        # ...and Step 3 refuses a review made on a different signal.
        try:
            with QUIET(io.StringIO()):
                run_stage_c(df, sfreq, {"ppg": CHANNELS["ppg"]}, tmp, "sub-001_ses-run1_task-sdi", "s",
                            saved["channels"], saved["channels"], "aa", "bb", reconciler="hl", sidecar=meta,
                            tsv_path=SUB001)
            raise AssertionError("expected SystemExit")
        except SystemExit as e:
            assert "Reviewer aa's" in str(e)
        print("   OK: a changed signal stops Step 2 resume and Step 3")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_reprocess_ppg_check():
    print("4d. check_reprocess_alignment reports PPG separately...")
    tmp = tempfile.mkdtemp()
    try:
        df, sfreq = generate_synthetic_demo(duration_sec=200)
        cols = ["ecg", "rsp", "ppg", "eda", "ecg_peaks", "rsp_peaks", "ppg_peaks", "eda_peaks"]

        def write(name, frame):
            path = os.path.join(tmp, f"{name}_physio.tsv.gz")
            frame[cols].to_csv(path, sep="\t", index=False, header=False, compression="gzip")
            with open(path.replace("_physio.tsv.gz", "_physio.json"), "w") as f:
                json.dump({"SamplingFrequency": 1000, "Columns": cols}, f)
            return path

        old = write("old", df)
        same = write("same", df)
        shifted_df = df.copy()
        shifted_df["ppg"] = np.roll(df["ppg"].to_numpy(), 1500)  # PPG 1.5 s later, ECG untouched
        shifted = write("shifted", shifted_df)
        r_same = compare_run(old, same, ppg_work=True)
        r_shift = compare_run(old, shifted, ppg_work=True)
        r_shift_no_work = compare_run(old, shifted, ppg_work=False)
        assert r_same["status"] == "ALIGNED" and r_same["ppg_status"] == "IDENTICAL", r_same
        assert r_shift["status"] == "NEEDS_PPG_CHECK" and r_shift["ppg_status"] == "NEEDS_PPG_CHECK", r_shift
        assert set(r_shift["ppg_lags"]) == {1500}, r_shift["ppg_lags"]
        assert r_shift_no_work["status"] == "ALIGNED", "only runs with saved PPG work are held back"
        print(f"   OK: identical PPG passes; a PPG-only 1.5-s shift is caught (lags {r_shift['ppg_lags']})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_practice_ppg():
    print("4e. Practice PPG (v5): device dropout with a spurious seed and its MachineQC span; marks off a crest "
          "fail...")
    tmp = tempfile.mkdtemp()
    try:
        root = os.path.join(tmp, "practice")
        with QUIET(io.StringIO()):
            pc.make(root, seed=2024)
        key = pc.load_key(pc.key_path_for(root))
        run1 = key["runs"]["1"]
        ppg = run1["channels"]["ppg"]
        drop = run1["channels"]["bad_ppg"]["artifacts"][0]
        assert "dropout" in drop["type"]
        assert not any(drop["start"] - 200 <= s <= drop["end"] + 200 for s in ppg["true_peaks"])
        beh = os.path.join(root, "sub-990", "ses-run1", "beh", "sub-990_ses-run1_task-sdi_physio")
        sidecar = read_sidecar(beh + ".json")
        assert sidecar["MachineQC"]["Channels"]["ppg"]["BadSegments"] == [[drop["start"] - 600, drop["end"] + 600]]
        df = pd.read_csv(beh + ".tsv.gz", sep="\t", header=None, names=pc.COLUMNS, na_values="n/a")
        seeds = np.where(df["ppg_peaks"].to_numpy() == 1)[0]
        assert any(drop["start"] <= s < drop["end"] for s in seeds), "a spurious seed inside the dropout"

        def score(indices):
            sub = {"initials": "zz", "channels": {"ppg": {"mode": "point", "indices": [int(i) for i in indices]}}}
            return pc.score_run(key, sub, "1", ["ppg"])

        assert score(ppg["true_peaks"])[0], "the perfect answer passes"
        i = next(k for k, a in enumerate(ppg["accept"]) if len(a) == 1)
        moved = [s + 60 if k == i else s for k, s in enumerate(ppg["true_peaks"])]
        overall, _results, lines = score(moved)
        text = "\n".join(lines)
        assert not overall and "no mark within 30 ms" in text and "1 extra mark(s)" in text, text
        print("   OK: dropout, its spurious seed and its MachineQC span planted; a mark 60 ms off the crest fails")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_checkpoint2_fixes():
    print("CP2. Fixes from the checkpoint-2 review...")
    import sys
    import physio_review
    import qrs_template_stage
    from annotation_io import ppg_invalid_spans
    from channel_config import present_channels
    from physio_io import apply_unit_labels, machine_qc_notes

    df, sfreq = generate_synthetic_demo(duration_sec=20)
    tmp = tempfile.mkdtemp()
    try:
        # guide_shown: per session in the edit history, and sticky at the channel level.
        ecg_only = {"ecg": CHANNELS["ecg"]}
        peaks = np.where(df["ecg_peaks"].to_numpy())[0]

        def drop_one(k):
            def plot(self, *a, **kw):
                keep = [i for i, o in enumerate(self.annotations.onset)
                        if not (self.annotations.description[i] == "peak_ecg" and int(round(o * sfreq)) == peaks[k])]
                self.set_annotations(self.annotations[keep])
            return plot

        with patch.object(mne.io.RawArray, "plot", drop_one(2)), QUIET(io.StringIO()):
            run_stage_b(df, sfreq, ecg_only, "zz", tmp, "g", "synthetic", ppg_guide=True, ecg_ppg_guide=True)
        with patch.object(mne.io.RawArray, "plot", drop_one(5)), QUIET(io.StringIO()):
            r = run_stage_b(df, sfreq, ecg_only, "zz", tmp, "g", "synthetic", ppg_guide=False)
        with open(r.saved_path) as f:
            ecg = json.load(f)["channels"]["ecg"]
        assert ecg["guide_shown"] is True, "true if ANY saved session had the guide"
        assert [h["guide_shown"] for h in ecg["edit_history"]] == [True, False], ecg["edit_history"]
        print("   OK: guide_shown per session and sticky")

        # Resume count: a saved PPG review isn't counted in an ECG session with the guide row.
        with patch.object(mne.io.RawArray, "plot", return_value=None), QUIET(io.StringIO()):
            run_stage_b(df, sfreq, {"ppg": CHANNELS["ppg"]}, "zz", tmp, "g", "synthetic")
        out = io.StringIO()
        with patch.object(mne.io.RawArray, "plot", return_value=None), QUIET(out):
            run_stage_b(df, sfreq, ecg_only, "zz", tmp, "g", "synthetic", ecg_ppg_guide=True)
        n_ecg = len(ecg["indices"])
        assert f"Resuming your previous session on this run: {n_ecg} point annotation(s)" in out.getvalue(), \
            out.getvalue()[:400]
        print("   OK: the resume count covers only reviewed channels")

        # Reconciled file: edit history, reviewer origin, everyone's guide state.
        a = {"ecg": {"mode": "point", "indices": [int(peaks[1]), int(peaks[3])], "status": "complete",
                     "guide_shown": True}}
        b = {"ecg": {"mode": "point", "indices": [int(peaks[1]), int(peaks[6])], "status": "complete",
                     "guide_shown": False}}
        with patch.object(mne.io.RawArray, "plot", return_value=None), QUIET(io.StringIO()):
            rc = run_stage_c(df, sfreq, ecg_only, os.path.join(tmp, "rec"), "g", "s", a, b, "aa", "bb",
                             reconciler="hl", ppg_guide=False)
        with open(rc.saved_path) as f:
            rec = json.load(f)["channels"]["ecg"]
        assert rec["guide_shown_by"] == {"adjudicator": False, "aa": True, "bb": False} and rec["guide_shown"] is True
        assert rec["reviewer_origin"] == {"agreed": [int(peaks[1])], "only_aa": [int(peaks[3])],
                                          "only_bb": [int(peaks[6])]}, rec["reviewer_origin"]
        assert len(rec["edit_history"]) == 1 and rec["edit_history"][0]["initials"] == "hl"
        print("   OK: the reconciled entry records edit history, reviewer origin and everyone's guide state")

        # Channels the file doesn't have: dropped from the default with a note, refused by name.
        with QUIET(io.StringIO()) as note:
            chosen = present_channels(select_channels(None), df.columns, explicit=False)
        assert "sbp" not in chosen and "finger_temperature" not in chosen and "left out" in note.getvalue()
        try:
            present_channels(select_channels(["sbp"]), df.columns, explicit=True)
            raise AssertionError("expected SystemExit")
        except SystemExit as e:
            assert "no SBP data" in str(e)
        print("   OK: absent channels never logged as reviewed")

        # Standalone Step 1 accepts --no-ppg-guide.
        with patch.object(sys, "argv", ["qrs_template_stage.py", "--synthetic", "--no-ppg-guide"]):
            assert qrs_template_stage.parse_args().no_ppg_guide is True

        # Units: relabel, never rescale; end-gap note printed whatever the flagged fraction.
        apply_unit_labels({"emg": "mV"})
        assert mne.defaults.DEFAULTS["scalings"]["emg"] == 1.0
        notes = machine_qc_notes({"sbp": {"segments": [[700000, 742000]]}}, ["sbp"], 742091, 1000, {"sbp": "SBP"})
        assert any("stops 0.09 s before the end" in n for n in notes), notes
        if os.path.exists(SUB001):
            # SUB001 is the run-7 fixture (715c747): no per-sample mask. A
            # run-8 copy has the mask and so no note (run-8 copies belong in
            # test_input/run8/, not over this file).
            sub001_sidecar = sidecar_of(SUB001)
            _spans, ppg_note = ppg_invalid_spans(sub001_sidecar, 742091)
            has_mask = "SegmentsByReason" in sub001_sidecar["MachineQC"]["Channels"]["ppg"]
            assert (ppg_note is None) if has_mask else ("no per-sample PPG dropout mask yet" in ppg_note), ppg_note
        print("   OK: --no-ppg-guide in qrs_template_stage; relabel keeps scaling 1.0; end-gap note; mask note")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        mne.defaults.DEFAULTS["units"]["emg"] = "µV"
        mne.defaults.DEFAULTS["si_units"]["emg"] = "V"
        mne.defaults.DEFAULTS["scalings"]["emg"] = 1e6


def test_checkpoint2_reprocess():
    print("CP2. Reprocess check: invalidated reviews, CareTaker vitals, PPG removed...")
    from check_reprocess_alignment import compare_run as cr
    from provenance import column_sha256
    tmp = tempfile.mkdtemp()
    try:
        df, sfreq = generate_synthetic_demo(duration_sec=200)
        df["sbp"] = np.repeat(np.arange(len(df) // 1000 + 1) % 7 + 110, 1000)[:len(df)].astype(float)
        cols = ["ecg", "rsp", "ppg", "eda", "sbp", "ecg_peaks", "rsp_peaks", "ppg_peaks", "eda_peaks"]

        def write(name, frame):
            path = os.path.join(tmp, f"{name}_physio.tsv.gz")
            frame[cols].to_csv(path, sep="\t", index=False, header=False, compression="gzip")
            with open(path.replace("_physio.tsv.gz", "_physio.json"), "w") as f:
                json.dump({"SamplingFrequency": 1000, "Columns": cols}, f)
            return path

        old = write("old", df)
        prov = {"ecg": [{"number_of_samples": len(df), "signal_sha256": column_sha256(old, cols, ["ecg"])["ecg"]}]}
        same = write("same", df)
        assert cr(old, same, saved_prov=prov)["status"] == "ALIGNED"
        tweaked = df.copy()
        tweaked["ecg"] = tweaked["ecg"] + 1e-4 * np.random.default_rng(0).normal(size=len(df))
        r = cr(old, write("tweaked", tweaked), saved_prov=prov)
        assert r["status"] == "REVIEW_INVALIDATED" and r["invalidated_channels"] == ["ecg"], r
        moved = df.copy()
        moved["sbp"] = np.roll(df["sbp"].to_numpy(), 1500)
        r2 = cr(old, write("moved", moved), vitals_work=("sbp",))
        assert r2["status"] == "NEEDS_CARETAKER_CHECK", r2["status"]
        assert cr(old, write("moved2", moved))["status"] == "ALIGNED", "only runs with saved SBP/DBP work are held"
        gone = df.copy()
        gone["ppg"] = np.nan
        r3 = cr(old, write("gone", gone), ppg_work=True)
        assert r3["ppg_status"] == "PPG_REMOVED" and r3["status"] == "NEEDS_PPG_CHECK", r3
        print("   OK: a changed reviewed channel -> REVIEW_INVALIDATED; SBP moved -> NEEDS_CARETAKER_CHECK; "
              "PPG gone -> NEEDS_PPG_CHECK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_step3_default_order():
    print("CP2. Step 3 default: PPG is reconciled when the only other shared channel is an old-mode RSP...")
    import sys
    import physio_review
    tmp = tempfile.mkdtemp()
    try:
        df, sfreq = generate_synthetic_demo(duration_sec=30)
        ppg_peaks = [int(i) for i in np.where(df["ppg_peaks"].to_numpy())[0]]
        for who in ("aa", "bb"):
            with open(os.path.join(tmp, f"synthetic_demo_run1_annotations_{who}.json"), "w") as f:
                json.dump({"initials": who, "channels": {
                    "ppg": {"mode": "point", "indices": ppg_peaks, "status": "complete"},
                    "rsp": {"mode": "point", "indices": [1000, 5000], "status": "complete"}}}, f)
        out = io.StringIO()
        with patch.object(sys, "argv", ["physio_review.py", "--synthetic", "--duration", "30", "--run", "1",
                                        "--out-dir", tmp, "--stage", "3", "--compare-initials", "aa,bb",
                                        "--initials", "hl"]), \
             patch.object(mne.io.RawArray, "plot", return_value=None), QUIET(out):
            physio_review.main()
        with open(os.path.join(tmp, "synthetic_demo_run1_annotations_reconciled.json")) as f:
            assert set(json.load(f)["channels"]) == {"ppg", "bad_ppg"}, "PPG is reconciled with its bad stretches"
        assert "skipping RSP" in out.getvalue()
        print("   OK: old-mode RSP skipped, PPG reconciled")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_g2_no_guide_on_old_pipeline():
    print("G2. No PPG guide on old-pipeline files (HLU, 2026-09-26), in Steps 1, 2 and 3...")
    df, sfreq = generate_synthetic_demo(duration_sec=30)
    ecg_only = {"ecg": CHANNELS["ecg"]}
    old = {"SamplingFrequency": 1000, "Columns": list(df.columns)}  # no Provenance: old pipeline
    new = {"SamplingFrequency": 1000, "Columns": list(df.columns), "Provenance": {"CTTiming": {"Pulse": {}}}}
    assert "ppg" not in combine_with_reference_channels(ecg_only, df, ppg_guide=True, sidecar=old, ecg_ppg_guide=True)
    assert "ppg" in combine_with_reference_channels(ecg_only, df, ppg_guide=True, sidecar=new, ecg_ppg_guide=True)
    assert "ppg" in combine_with_reference_channels(ecg_only, df, ppg_guide=True, ecg_ppg_guide=True), \
        "synthetic (no sidecar) keeps it"
    tmp = tempfile.mkdtemp()
    try:
        seen, out = {}, io.StringIO()
        with patch.object(mne.io.RawArray, "plot", lambda self, *a, **k: seen.setdefault("names", list(self.ch_names))), \
             QUIET(out):
            r = run_stage_b(df, sfreq, ecg_only, "zz", tmp, "g", "s", sidecar=old, ecg_ppg_guide=True)
        assert seen["names"] == ["ecg"] and "before the PPG timing fix" in out.getvalue(), (seen, out.getvalue()[:300])
        with open(r.saved_path) as f:
            assert json.load(f)["channels"]["ecg"]["guide_shown"] is False
        a = {"ecg": {"mode": "point", "indices": [1000, 2000], "status": "complete"}}
        raw3, _ = build_reconciliation_raw(df, ecg_only, sfreq, a, a, "aa", "bb", ppg_guide=True, sidecar=old,
                                           ecg_ppg_guide=True)
        assert list(raw3.ch_names) == ["ecg"]
        step1 = {}
        with patch.object(mne.io.RawArray, "plot", lambda self, *a, **k: step1.setdefault("names", list(self.ch_names))), \
             patch.object(qrs_template, "show_template_preview", return_value=(True, "")), QUIET(io.StringIO()):
            qrs_template.run_stage_a({"1": df}, {"1": "t"}, {"1": tmp}, sfreq, "zz", sidecars={"1": old},
                                     ecg_ppg_guide=True)
        assert step1["names"] == ["ecg"], step1
        if os.path.exists(SUB085):
            assert not combine_with_reference_channels(ecg_only, load_physio_tsv(SUB085)[0], ppg_guide=True,
                                                       sidecar=sidecar_of(SUB085), ecg_ppg_guide=True).get("ppg")
        print("   OK: hidden with a note on old-pipeline files; shown on new-pipeline and synthetic data")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_consultation_formats():
    print("Consultation (2026-09-26): no-pulse runs, CTTiming strings, all-NaN PPG, schema tags...")
    from annotation_io import ppg_invalid_spans
    df, sfreq = generate_synthetic_demo(duration_sec=20)
    n = len(df)
    base = {"Provenance": {"NumberOfSamples": n, "CTTiming": {"Pulse": "none (no pulseWaveform reconstruction ran)"}}}
    # A string CTTiming.Pulse is still the NEW pipeline.
    assert new_pipeline_ppg(base)
    assert not new_pipeline_ppg({"Provenance": {"NumberOfSamples": n}})
    no_pulse = {**base, "MachineQC": {"Channels": {"ppg": {"BadSegments": [[0, n]], "Rule": "x",
                                                           "SegmentsByReason": {"no_pulse_data": [[0, n]]}}}}}
    spans, note = ppg_invalid_spans(no_pulse, n)
    assert spans == [[0, n]] and "no usable CareTaker pulse" in note, note
    masked = {**base, "MachineQC": {"Channels": {"ppg": {"BadSegments": [], "Rule": "x",
                                                         "SegmentsByReason": {"zero_run": [], "bridged": [],
                                                                              "window_rule": []}}}}}
    assert ppg_invalid_spans(masked, n) == ([], None), "three (empty) reasons = the mask is present, no note"
    run7 = {**base, "MachineQC": {"Channels": {"ppg": {"BadSegments": [], "Rule": "x"}}}}
    assert "no per-sample PPG dropout mask yet" in ppg_invalid_spans(run7, n)[1]
    # A run whose ppg is all NaN can't be reviewed for PPG, whatever its sidecar says.
    nan_df = df.copy()
    nan_df["ppg"] = np.nan
    tmp = tempfile.mkdtemp()
    try:
        for sidecar in (base, None):
            try:
                with QUIET(io.StringIO()):
                    run_stage_b(nan_df, sfreq, {"ppg": CHANNELS["ppg"]}, "zz", tmp, "t", "s", sidecar=sidecar)
                raise AssertionError("expected SystemExit")
            except SystemExit as e:
                assert "no usable PPG" in str(e), str(e)
        with patch.object(mne.io.RawArray, "plot", return_value=None), QUIET(io.StringIO()):
            r = run_stage_b(df, sfreq, {"eda": CHANNELS["eda"]}, "zz", tmp, "t", "s")
        with open(r.saved_path) as f:
            assert json.load(f)["coordinate_space"].startswith("tsv row index")
        print("   OK: no_pulse_data handled; CTTiming strings count as new pipeline; all-NaN PPG refused; "
              "coordinate space and hash definition stated")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _with_vitals(df, sbp=None, dbp=None):
    """A copy of df with sbp/dbp columns (all NaN unless values are given)."""
    out = df.copy()
    out["sbp"] = np.nan if sbp is None else sbp
    out["dbp"] = np.nan if dbp is None else dbp
    return out


def test_run8_decisions():
    print("Run-8 decisions (2026-09-26): empty channels refused, machine reasons shown, reviewed_span saved...")
    import sys
    import physio_review
    from annotation_io import ppg_invalid_spans
    from channel_config import empty_channels, present_channels
    from physio_io import machine_qc_notes, machine_qc_segments, reason_summary
    df, sfreq = generate_synthetic_demo(duration_sec=20)
    n = len(df)
    no_vitals = _with_vitals(df)
    tmp = tempfile.mkdtemp()
    try:
        # 1. An empty SBP/DBP column: dropped from the all-channels default with a note, refused by name.
        assert empty_channels(no_vitals, ["ecg", "sbp", "dbp", "absent"]) == ["sbp", "dbp"]
        with QUIET(io.StringIO()) as note:
            chosen = present_channels(select_channels(None), no_vitals.columns, explicit=False,
                                      empty=empty_channels(no_vitals, select_channels(None)))
        assert "sbp" not in chosen and "dbp" not in chosen and "ecg" in chosen, list(chosen)
        assert "SBP, DBP data are empty" in note.getvalue(), note.getvalue()
        for requested, expect in ((["sbp"], "no usable SBP"), (["sbp", "dbp"], "no usable SBP, DBP"),
                                  (["ecg", "dbp"], "no usable DBP")):
            try:
                present_channels(select_channels(requested), no_vitals.columns, explicit=True,
                                 empty=empty_channels(no_vitals, requested))
                raise AssertionError(f"expected SystemExit for {requested}")
            except SystemExit as e:
                assert expect in str(e), (requested, str(e))
        # A partly-empty column is NOT empty: its gaps are the machine's flags to review.
        partial = no_vitals.copy()
        partial.loc[:999, "sbp"] = 120.0
        assert empty_channels(partial, ["sbp", "dbp"]) == ["dbp"]
        print("   OK: default drops empty SBP/DBP with a note; naming them is refused with the reason")

        # 2. The backstop in Steps 2 and 3, for callers that skip present_channels().
        for configs in ({"sbp": CHANNELS["sbp"]}, {"eda": CHANNELS["eda"], "dbp": CHANNELS["dbp"]}):
            try:
                with QUIET(io.StringIO()):
                    run_stage_b(no_vitals, sfreq, configs, "zz", tmp, "t", "s")
                raise AssertionError("expected SystemExit")
            except SystemExit as e:
                assert "no usable" in str(e) and "blood pressure" in str(e), str(e)
        seg = {"sbp": {"mode": "segment", "bad_segments": [[0, n]], "status": "complete"}}
        try:
            with QUIET(io.StringIO()):
                run_stage_c(no_vitals, sfreq, {"sbp": CHANNELS["sbp"]}, tmp, "t", "s", seg, seg, "aa", "bb",
                            reconciler="hl")
            raise AssertionError("expected SystemExit")
        except SystemExit as e:
            assert "no usable SBP" in str(e), str(e)
        # PPG keeps its own wording.
        nan_ppg = df.copy()
        nan_ppg["ppg"] = np.nan
        try:
            with QUIET(io.StringIO()):
                run_stage_b(nan_ppg, sfreq, {"ppg": CHANNELS["ppg"]}, "zz", tmp, "t", "s")
            raise AssertionError("expected SystemExit")
        except SystemExit as e:
            assert "no usable PPG" in str(e), str(e)
        print("   OK: Steps 2 and 3 refuse an empty channel even when handed one directly")

        # 3. Step 3's default leaves out a shared channel whose column is empty.
        two_runs = physio_review.generate_synthetic_demo_two_runs

        def with_empty_vitals(duration_sec_each):
            run_dfs, rate = two_runs(duration_sec_each=duration_sec_each)
            return {r: _with_vitals(d) for r, d in run_dfs.items()}, rate

        ecg_peaks = [int(i) for i in np.where(df["ecg_peaks"].to_numpy())[0]][:5]
        for who in ("aa", "bb"):
            with open(os.path.join(tmp, f"synthetic_demo_run1_annotations_{who}.json"), "w") as f:
                json.dump({"initials": who, "channels": {
                    "ecg": {"mode": "point", "indices": ecg_peaks, "status": "complete"},
                    "sbp": {"mode": "segment", "bad_segments": [], "status": "complete"}}}, f)
        out = io.StringIO()
        with patch.object(sys, "argv", ["physio_review.py", "--synthetic", "--duration", "20", "--run", "1",
                                        "--out-dir", tmp, "--stage", "3", "--compare-initials", "aa,bb",
                                        "--initials", "hl", "--no-ppg-guide"]), \
             patch.object(physio_review, "generate_synthetic_demo_two_runs", with_empty_vitals), \
             patch.object(mne.io.RawArray, "plot", return_value=None), QUIET(out):
            physio_review.main()
        with open(os.path.join(tmp, "synthetic_demo_run1_annotations_reconciled.json")) as f:
            reconciled = json.load(f)
        assert set(reconciled["channels"]) == {"ecg", "bad_ecg"}, list(reconciled["channels"])
        assert "skipping SBP" in out.getvalue(), out.getvalue()
        # 4. reviewed_span: the whole file, in the reconciled file too.
        assert reconciled["channels"]["ecg"]["reviewed_span"] == [0, n], reconciled["channels"]["ecg"]
        print("   OK: Step 3's default skips an empty shared channel; the reconciled entry has reviewed_span")

        # 5. SBP/DBP reasons (decision B1): shown in the banner and the summary note, kept in the seed record.
        vitals = _with_vitals(df, sbp=120.0, dbp=80.0)
        union = [[0, 1000], [5000, 5200]]
        by_reason = {"no_coverage": [[0, 1000]], "out_of_range": [], "ordering": [[5000, 5200]]}
        sidecar = {"Provenance": {"NumberOfSamples": n, "PipelineCommit": "run8test", "CTTiming": {}},
                   "MachineQC": {"Channels": {
                       "sbp": {"BadSegments": union, "Rule": "r", "SegmentsByReason": by_reason},
                       "dbp": {"BadSegments": [[5000, 5200]], "Rule": "r",
                               "SegmentsByReason": {"no_coverage": [], "brand_new_reason": [[5000, 5200]]}}}}}
        qc, qc_note = machine_qc_segments(sidecar, n, only=["sbp", "dbp"])
        assert qc_note is None and qc["sbp"]["by_reason"] == by_reason and not qc["sbp"]["by_reason_invalid"]
        assert reason_summary(qc["sbp"]["by_reason"]) == "no CareTaker reading (1), SBP less than 10 above DBP (1)"
        assert reason_summary(qc["dbp"]["by_reason"]) == "brand_new_reason (1)", "an unknown reason shows by name"
        assert reason_summary({"no_vitals_data": [[0, n]]}) == "no CareTaker vitals for this run (1)"
        notes = machine_qc_notes(qc, ["sbp", "dbp"], n, sfreq, {"sbp": "SBP", "dbp": "DBP"})
        assert "SBP: why the machine flagged them: no CareTaker reading (1), SBP less than 10 above DBP (1)." in notes
        captured = {}
        real_format = session_summary_gui.format_change_lines

        def spy(changes, cfgs, seed_notes=None):
            captured.update(seed_notes or {})
            return real_format(changes, cfgs, seed_notes)

        out = io.StringIO()
        with patch.object(session_summary_gui, "format_change_lines", spy), \
             patch.object(mne.io.RawArray, "plot", return_value=None), QUIET(out):
            r = run_stage_b(vitals, sfreq, {"sbp": CHANNELS["sbp"]}, "zz", tmp, "v", "s", sidecar=sidecar)
        assert "why the machine flagged them: no CareTaker reading (1)" in out.getvalue(), out.getvalue()
        assert captured.get("sbp", "").endswith(": no CareTaker reading (1), SBP less than 10 above DBP (1)"), captured
        with open(r.saved_path) as f:
            entry = json.load(f)["channels"]["sbp"]
        assert entry["bad_segments"] == union and entry["seed"]["by_reason"] == by_reason, entry
        assert entry["reviewed_span"] == [0, n], entry
        print("   OK: reasons in the banner, the summary note and the seed record; reviewed_span in Step 2")

        # 6. A malformed breakdown is ignored, never half-used: the flags still pre-fill; no reason line.
        for bad in ({"no_coverage": [[1000, 10]]}, {"no_coverage": "all"}, ["no_coverage"], {"no_coverage": [[0]]}):
            broken = json.loads(json.dumps(sidecar))
            broken["MachineQC"]["Channels"]["sbp"]["SegmentsByReason"] = bad
            qc, _ = machine_qc_segments(broken, n, only=["sbp"])
            assert qc["sbp"]["segments"] == union and qc["sbp"]["by_reason"] is None
            assert qc["sbp"]["by_reason_invalid"] is True, bad
            assert not any("why the machine" in line
                           for line in machine_qc_notes(qc, ["sbp"], n, sfreq, {"sbp": "SBP"})), bad
        # No breakdown at all (run 7) is not "invalid".
        run7 = json.loads(json.dumps(sidecar))
        del run7["MachineQC"]["Channels"]["sbp"]["SegmentsByReason"]
        qc, _ = machine_qc_segments(run7, n, only=["sbp"])
        assert qc["sbp"]["by_reason"] is None and qc["sbp"]["by_reason_invalid"] is False
        # A malformed PPG mask says so, rather than "no mask yet".
        ppg_bad = {"Provenance": {"NumberOfSamples": n, "CTTiming": {}},
                   "MachineQC": {"Channels": {"ppg": {"BadSegments": [], "Rule": "x",
                                                      "SegmentsByReason": {"zero_run": [[9, 3]]}}}}}
        assert "can't be read" in ppg_invalid_spans(ppg_bad, n)[1]
        print("   OK: a malformed breakdown is ignored (flags still pre-fill), and a malformed PPG mask is named")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_run8_review_fixes():
    print("Run-8 review fixes (2026-09-26): entry points end to end, stale Step 3 save, summary seed lines, "
          "coverage wording, PPG lists...")
    import copy
    import sys
    import physio_annotate
    import physio_review
    from annotation_io import print_sidecar_banner
    from physio_io import coverage_only, machine_qc_notes
    from ppg_checks import ppg_check_lines
    df, sfreq = generate_synthetic_demo(duration_sec=20)
    n = len(df)
    tmp = tempfile.mkdtemp()
    try:
        # 1. The Step 2 default through the real entry points leaves out empty SBP/DBP and saves the rest.
        two_runs = physio_review.generate_synthetic_demo_two_runs

        def with_empty_vitals(duration_sec_each):
            run_dfs, rate = two_runs(duration_sec_each=duration_sec_each)
            return {r: _with_vitals(d) for r, d in run_dfs.items()}, rate

        out = io.StringIO()
        step2 = os.path.join(tmp, "step2")
        with patch.object(sys, "argv", ["physio_review.py", "--synthetic", "--duration", "20", "--run", "1",
                                        "--out-dir", step2, "--stage", "2", "--initials", "aa",
                                        "--ecg-source", "batch", "--no-ppg-guide"]), \
             patch.object(physio_review, "generate_synthetic_demo_two_runs", with_empty_vitals), \
             patch.object(mne.io.RawArray, "plot", return_value=None), QUIET(out):
            physio_review.main()
        with open(os.path.join(step2, "synthetic_demo_run1_annotations_aa.json")) as f:
            saved = json.load(f)["channels"]
        assert "sbp" not in saved and "dbp" not in saved and "ecg" in saved, list(saved)
        assert "SBP, DBP data are empty" in out.getvalue()
        standalone = os.path.join(tmp, "standalone")
        out = io.StringIO()
        with patch.object(sys, "argv", ["physio_annotate.py", "--synthetic", "--duration", "20", "--initials", "aa",
                                        "--out-dir", standalone, "--ecg-source", "batch", "--no-ppg-guide"]), \
             patch.object(physio_annotate, "generate_synthetic_demo",
                          lambda duration_sec: (_with_vitals(df), sfreq)), \
             patch.object(mne.io.RawArray, "plot", return_value=None), QUIET(out):
            physio_annotate.main()
        with open(os.path.join(standalone, "synthetic_demo_annotations_aa.json")) as f:
            saved = json.load(f)["channels"]
        assert "sbp" not in saved and "dbp" not in saved and "ecg" in saved, list(saved)
        assert "SBP, DBP data are empty" in out.getvalue()
        # Step 3 naming an empty channel neither reviewer has: the real reason, not "not reviewed by both".
        rec_dir = os.path.join(tmp, "step3")
        os.makedirs(rec_dir)
        for who in ("aa", "bb"):
            with open(os.path.join(rec_dir, f"synthetic_demo_run1_annotations_{who}.json"), "w") as f:
                json.dump({"initials": who, "channels": {
                    "ecg": {"mode": "point", "indices": [1000, 2000], "status": "complete"}}}, f)
        try:
            with patch.object(sys, "argv", ["physio_review.py", "--synthetic", "--duration", "20", "--run", "1",
                                            "--out-dir", rec_dir, "--stage", "3", "--compare-initials", "aa,bb",
                                            "--initials", "hl", "--channels", "sbp"]), \
                 patch.object(physio_review, "generate_synthetic_demo_two_runs", with_empty_vitals), \
                 QUIET(io.StringIO()):
                physio_review.main()
            raise AssertionError("expected SystemExit")
        except SystemExit as e:
            assert "no usable SBP" in str(e), str(e)
        print("   OK: Step 2 default (physio_review, physio_annotate) and Step 3 by name handle empty SBP/DBP")

        # 2. A Step 3 re-run on a reprocessed file with identical marks is re-saved with the new span.
        longer = pd.concat([df, df.iloc[:2000]], ignore_index=True)
        seg = {"eda": {"mode": "segment", "bad_segments": [[2000, 2500]], "status": "complete"}}
        eda = {"eda": CHANNELS["eda"]}
        stale = os.path.join(tmp, "stale")
        with patch.object(mne.io.RawArray, "plot", return_value=None), QUIET(io.StringIO()):
            first = run_stage_c(df, sfreq, eda, stale, "t", "s", copy.deepcopy(seg), copy.deepcopy(seg), "aa", "bb",
                                reconciler="hl")
            again = run_stage_c(longer, sfreq, eda, stale, "t", "s", copy.deepcopy(seg), copy.deepcopy(seg), "aa",
                                "bb", reconciler="hl")
            same = run_stage_c(longer, sfreq, eda, stale, "t", "s", copy.deepcopy(seg), copy.deepcopy(seg), "aa",
                               "bb", reconciler="hl")
        assert first.saved and again.saved and not same.saved, (first.saved, again.saved, same.saved)
        with open(again.saved_path) as f:
            assert json.load(f)["channels"]["eda"]["reviewed_span"] == [0, len(longer)]
        print("   OK: identical marks on a changed file are re-saved; an unchanged re-run still isn't")

        # 3. The summary says what each channel started from, even with no changes.
        vitals = _with_vitals(df, sbp=120.0, dbp=80.0)
        vitals.loc[1000:, ["sbp", "dbp"]] = np.nan  # CareTaker covers 5% of the run
        low = {"no_coverage": [[1000, n]], "out_of_range": [], "ordering": []}
        sidecar = {"Provenance": {"NumberOfSamples": n, "PipelineCommit": "run8test", "CTTiming": {}},
                   "MachineQC": {"Channels": {k: {"BadSegments": [[1000, n]], "Rule": "r", "SegmentsByReason": low}
                                              for k in ("sbp", "dbp")}}}
        seen = {}

        def summary(heading, change_lines, warning_lines, n_changes, **kwargs):
            seen.update(kwargs, warnings=warning_lines, n_changes=n_changes)
            return session_summary_gui.SessionDecision(session_summary_gui.DISCARD)

        out = io.StringIO()
        with patch.object(session_summary_gui, "show_session_summary", summary), \
             patch.object(mne.io.RawArray, "plot", return_value=None), QUIET(out):
            run_stage_b(vitals, sfreq, {"sbp": CHANNELS["sbp"]}, "zz", tmp, "cov", "s", sidecar=sidecar)
        assert seen["n_changes"] == 0 and seen["seed_lines"] == [
            "SBP started from 1 machine-flagged stretch(es): no CareTaker reading (1)"], seen
        from PyQt6.QtWidgets import QApplication, QLabel
        app = QApplication.instance() or QApplication([])
        dialog = session_summary_gui.SessionSummaryDialog("h", ["SBP: 0 bad segments added, 0 removed"], [], 0, False,
                                                          "", seed_lines=seen["seed_lines"])
        texts = [label.text() for label in dialog.findChildren(QLabel)]
        assert any("no CareTaker reading (1)" in t for t in texts), texts
        print("   OK: seed lines (the machine's reasons) reach the summary dialog with no changes")

        # 4. Mostly-uncovered SBP: coverage wording, not the dead-sensor warning (banner and repeat).
        banner = out.getvalue()
        assert "CareTaker has readings for only 5.0% of this run" in banner, banner
        assert "disconnected" not in banner and "dead sensor" not in banner, banner
        assert coverage_only({"by_reason": low, "by_reason_invalid": False})
        assert not coverage_only({"by_reason": {**low, "ordering": [[0, 10]]}, "by_reason_invalid": False})
        assert not coverage_only({"by_reason": None, "by_reason_invalid": False})
        # Without reasons (RSP, EDA, run-7 SBP) the dead-sensor warning stays.
        notes = machine_qc_notes({"rsp": {"segments": [[1000, n]], "by_reason": None}}, ["rsp"], n, sfreq,
                                 {"rsp": "RSP"})
        assert any("sensor may have been disconnected" in line for line in notes), notes

        def clear(self, *args, **kwargs):
            self.set_annotations(mne.Annotations([], [], []))

        seen.clear()
        with patch.object(session_summary_gui, "show_session_summary", summary), \
             patch.object(mne.io.RawArray, "plot", clear), QUIET(io.StringIO()):
            run_stage_b(vitals, sfreq, {"sbp": CHANNELS["sbp"]}, "zz", tmp, "cov2", "s", sidecar=sidecar)
        assert any("no CareTaker reading at all" in w for w in seen["warnings"]), seen["warnings"]
        assert not any("disconnected sensor" in w for w in seen["warnings"]), seen["warnings"]
        print("   OK: coverage-only flags get coverage wording; RSP/EDA keep the dead-sensor warning")

        # 5. PPG check lines: the full lists go to the terminal when the summary shows only three.
        signal = np.sin(np.linspace(0, 40 * np.pi, n))
        indices = [int(i) for i in np.arange(500, n, 1000)]
        with QUIET(io.StringIO()) as term:
            lines = ppg_check_lines(indices, signal, sfreq, [[0, 6000]], 50)
        assert "all listed in the terminal" in lines[0], lines
        assert "(all 6):" in term.getvalue() and "machine-flagged dropouts (1): 0.0-6.0 s" in term.getvalue()
        with QUIET(io.StringIO()) as term:
            quiet_lines = ppg_check_lines(indices, signal, sfreq, [[0, 6000]], 50, print_all=False)
        assert term.getvalue() == "" and "all listed" not in quiet_lines[0]
        with QUIET(io.StringIO()) as term:
            print_sidecar_banner({"ppg": {"Description": "native samples ... zero runs are KEPT"}},
                                 {"ppg": CHANNELS["ppg"]})
        assert "never a flat line" in term.getvalue(), term.getvalue()
        with QUIET(io.StringIO()) as term:
            print_sidecar_banner({"ppg": {"Description": "Finger pulse signal ...", "TechnicalDescription": "..."}},
                                 {"ppg": CHANNELS["ppg"]})
        assert "never a flat line" not in term.getvalue(), "a plain run-8 Description needs no extra line"
        print("   OK: PPG check times listed in full in the terminal; the PPG banner says it's filtered")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bad_ppg_channel():
    print("bad_ppg (HLU, 2026-09-27): PPG's bad stretches, pre-marked from the machine, edited by RAs, never keep peaks...")
    from annotation_io import export_annotations, seed_annotations
    from channel_config import COMPANION_CHANNELS
    df, sfreq = generate_synthetic_demo(duration_sec=20)
    n = len(df)
    df = df.copy()
    df["event"] = 0
    seeds = [int(i) for i in np.flatnonzero(df["ppg_peaks"].to_numpy())]
    span = [seeds[4] - 100, seeds[5] + 100]  # the machine's flag covers two automated peaks
    sidecar = {"Provenance": {"NumberOfSamples": n, "CTTiming": {}},
               "MachineQC": {"Channels": {"ppg": {"BadSegments": [span], "Rule": "r",
                                                  "SegmentsByReason": {"zero_run": [span], "bridged": [],
                                                                       "window_rule": []}}}}}
    seen = {}

    def plot(self, *args, **kwargs):
        seen["names"] = list(self.ch_names)
        seen["ann"] = [(round(a["onset"] * sfreq), round(a["duration"] * sfreq), a["description"])
                       for a in self.annotations]

    tmp = tempfile.mkdtemp()
    try:
        out = io.StringIO()
        with patch.object(mne.io.RawArray, "plot", plot), QUIET(out):
            r = run_stage_b(df, sfreq, {"ppg": CHANNELS["ppg"]}, "zz", tmp, "t", "s", sidecar=sidecar)
        assert seen["names"] == ["ppg", "event"], "no extra row: bad stretches are annotations"
        assert (span[0], span[1] - span[0], "bad_ppg") in seen["ann"], "pre-marked from the machine"
        peak_marks = [o for o, d, desc in seen["ann"] if desc == "peak_ppg"]
        assert seeds[4] not in peak_marks and seeds[5] not in peak_marks, "seeds inside it aren't loaded"
        assert "2 automated peak(s) inside the pre-marked stretches weren't loaded" in out.getvalue(), out.getvalue()
        with open(r.saved_path) as f:
            saved = json.load(f)["channels"]
        assert saved["bad_ppg"]["bad_segments"] == [span] and saved["bad_ppg"]["seed"]["source"] == "MachineQC"
        assert saved["ppg"]["indices"] == [s for s in seeds if s not in (seeds[4], seeds[5])]
        assert saved["ppg"]["edit_history"][0]["seeds_left_out_in_bad_stretches"] == [seeds[4], seeds[5]]
        assert saved["bad_ppg"]["status"] == saved["ppg"]["status"] == "complete"

        # An RA-marked stretch drops the peaks inside it on save, and the summary says so.
        configs = COMPANION_CHANNELS and {"ppg": CHANNELS["ppg"], "bad_ppg": COMPANION_CHANNELS["bad_ppg"]}
        raw = build_raw(df, {"ppg": CHANNELS["ppg"]}, sfreq)
        raw = seed_annotations(raw, df, configs, {}, sfreq)
        new_bad = [seeds[8] - 50, seeds[9] + 50]
        raw.set_annotations(raw.annotations + mne.Annotations([new_bad[0] / sfreq], [(new_bad[1] - new_bad[0]) / sfreq],
                                                              ["bad_ppg"]))
        with QUIET(io.StringIO()):
            exported = export_annotations(raw, df, configs, sfreq)
        assert exported["bad_ppg"]["bad_segments"] == [new_bad]
        assert seeds[8] not in exported["ppg"]["indices"] and seeds[9] not in exported["ppg"]["indices"]
        assert seeds[7] in exported["ppg"]["indices"]

        # The gap check ignores gaps across a bad stretch; an uncovered long gap is still flagged.
        from annotation_io import check_interval_regularity
        keep = [s for s in seeds if s not in (seeds[8], seeds[9])]
        with QUIET(io.StringIO()):
            covered = check_interval_regularity({"ppg": {"mode": "point", "indices": keep},
                                                 "bad_ppg": {"mode": "segment", "bad_segments": [new_bad]}},
                                                configs, sfreq)
            uncovered = check_interval_regularity({"ppg": {"mode": "point", "indices": keep},
                                                   "bad_ppg": {"mode": "segment", "bad_segments": []}}, configs, sfreq)
        near = lambda flags: [f for f in flags if abs(f[1] * sfreq - seeds[7]) < 5]
        assert not near(covered) and near(uncovered), (covered, uncovered)

        # Step 3: reviewers' bad stretches are reconciled; a review saved before bad_ppg existed counts as none.
        seen.clear()
        a = {"ppg": {"mode": "point", "indices": keep, "status": "complete"},
             "bad_ppg": {"mode": "segment", "bad_segments": [new_bad], "status": "complete"}}
        b = {"ppg": {"mode": "point", "indices": keep, "status": "complete"}}
        with patch.object(mne.io.RawArray, "plot", plot), QUIET(io.StringIO()):
            rc = run_stage_c(df, sfreq, {"ppg": CHANNELS["ppg"]}, os.path.join(tmp, "rec"), "t", "s", a, b, "aa", "bb",
                             reconciler="hl", sidecar=sidecar)
        assert seen["names"] == ["ppg", "event"]
        assert any(desc == "bad_ppg_only_aa" for _o, _d, desc in seen["ann"]), seen["ann"]
        with open(rc.saved_path) as f:
            rec = json.load(f)["channels"]
        assert rec["bad_ppg"]["bad_segments"] == [new_bad] and rec["ppg"]["indices"] == keep
        # ECG sessions get no bad_ppg.
        seen.clear()
        with patch.object(mne.io.RawArray, "plot", plot), QUIET(io.StringIO()):
            r2 = run_stage_b(df, sfreq, {"ecg": CHANNELS["ecg"]}, "zz", os.path.join(tmp, "e"), "t", "s",
                             sidecar=sidecar)
        with open(r2.saved_path) as f:
            assert set(json.load(f)["channels"]) == {"ecg", "bad_ecg"}, "ECG gets bad_ecg, not bad_ppg"
        print("   OK: no extra row; pre-marked from MachineQC; seeds inside not loaded (recorded); RA stretches drop "
              "peaks on save; gap check ignores covered gaps only; Step 3 reconciles it; not in ECG sessions")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bad_ecg_channel():
    print("bad_ecg (HLU, 2026-09-27): ECG's bad stretches, blank at the start, RA-marked, never keep peaks...")
    from annotation_io import check_interval_regularity
    df, sfreq = generate_synthetic_demo(duration_sec=20)
    peaks = [int(i) for i in np.flatnonzero(df["ecg_peaks"].to_numpy())]
    bad = [peaks[6] - 50, peaks[7] + 50]
    seen = {}

    def plot(self, *args, **kwargs):
        seen["names"] = list(self.ch_names)
        seen["labels"] = sorted(set(self.annotations.description))
        self.set_annotations(self.annotations + mne.Annotations([bad[0] / sfreq], [(bad[1] - bad[0]) / sfreq],
                                                                ["bad_ecg"]))

    tmp = tempfile.mkdtemp()
    try:
        out = io.StringIO()
        captured = {}

        def summary(heading, change_lines, warning_lines, n_changes, **kwargs):
            captured["warnings"] = warning_lines
            return session_summary_gui.SessionDecision(session_summary_gui.SAVE, finished=True)

        with patch.object(mne.io.RawArray, "plot", plot), \
             patch.object(session_summary_gui, "show_session_summary", summary), QUIET(out):
            r = run_stage_b(df, sfreq, {"ecg": CHANNELS["ecg"]}, "zz", tmp, "t", "s", ppg_guide=False)
        assert seen["names"] == ["ecg"], "no extra row"
        assert "bad_ecg" in seen["labels"], "the label is offered from the start (placeholder)"
        assert "ECG bad stretches (label 'bad_ecg'): mark stretches only where R peaks can't be located" in out.getvalue()
        assert any("ECG: 2 peak(s) inside your bad stretches will be left out" in w for w in captured["warnings"]), \
            captured["warnings"]
        with open(r.saved_path) as f:
            saved = json.load(f)["channels"]
        assert saved["bad_ecg"]["bad_segments"] == [bad] and "seed" not in saved["bad_ecg"], "blank start, no seed"
        assert peaks[6] not in saved["ecg"]["indices"] and peaks[7] not in saved["ecg"]["indices"]
        assert sorted(saved["ecg"]["edit_history"][0]["removed_in_bad_stretches"]) == [peaks[6], peaks[7]]
        # The gap across the bad stretch isn't flagged.
        with QUIET(io.StringIO()):
            flags = check_interval_regularity({"ecg": saved["ecg"], "bad_ecg": saved["bad_ecg"]},
                                              {"ecg": CHANNELS["ecg"], "bad_ecg": {"companion_of": "ecg"}}, sfreq)
        assert not [f for f in flags if abs(f[1] * sfreq - peaks[5]) < 5], flags
        print("   OK: no row; label offered; blank start; peaks inside dropped on save and counted in the summary; "
              "gap across it not flagged")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_guide_rows_2026_09_27():
    print("Guide rows (HLU, 2026-09-27): PPG under SBP/DBP, EDA under RSP, RSP under EDA; none under PPG...")
    from annotation_io import guide_banner_lines
    from channel_config import GUIDE_CONTEXT_TEXT, GUIDE_TEXTS
    df, sfreq = generate_synthetic_demo(duration_sec=20)
    n = len(df)
    df["sbp"] = 120.0 + np.round(np.sin(np.arange(n) / 3000.0) * 5)
    df["dbp"] = 80.0 + np.round(np.sin(np.arange(n) / 3000.0) * 3)

    def rows(keys, **kwargs):
        return list(combine_with_reference_channels({k: CHANNELS[k] for k in keys}, df, ppg_guide=True, **kwargs))

    assert rows(["rsp"]) == ["rsp", "eda"] and rows(["eda"]) == ["eda", "rsp"]
    assert rows(["sbp"]) == ["sbp", "ppg"] and rows(["dbp"]) == ["dbp", "ppg"]
    assert rows(["sbp", "dbp"]) == ["sbp", "dbp", "ppg"], "one PPG row, under the last of them"
    assert rows(["ppg"]) == ["ppg"], "no ECG under PPG (HLU: the ECG is messy; PPG is clean enough on its own)"
    assert rows(["rsp", "eda"]) == ["rsp", "eda"], "both scored: no guides"
    assert rows(["ecg", "rsp"]) == ["ecg", "rsp", "eda"], "PPG under ECG is opt-in (HLU, 2026-10-03)"
    assert rows(["ecg", "rsp"], ecg_ppg_guide=True) == ["ecg", "ppg", "rsp", "eda"]
    combined = combine_with_reference_channels({"rsp": CHANNELS["rsp"]}, df, ppg_guide=True)
    assert combined["eda"] is GUIDE_CHANNELS["eda"] and combined["eda"]["annotation_mode"] == "none"
    assert list(combine_with_reference_channels({"rsp": CHANNELS["rsp"]}, df, ppg_guide=False)) == ["rsp"]

    # HLU's wording: one PPG line for an SBP+DBP session, then the context line.
    lines = guide_banner_lines({"sbp": CHANNELS["sbp"], "dbp": CHANNELS["dbp"]}, df, True)
    assert lines == [GUIDE_TEXTS[("ppg", "sbp")], GUIDE_CONTEXT_TEXT], lines
    assert lines[0].startswith("PPG (guide): the finger pulse the blood pressure comes from.")
    assert guide_banner_lines({"rsp": CHANNELS["rsp"]}, df, True) == [
        "EDA (guide): a rise 1-3 s after a big breath shows the breath was real. No rise means nothing. "
        "Never mark RSP bad because of EDA.", GUIDE_CONTEXT_TEXT]
    assert guide_banner_lines({"eda": CHANNELS["eda"]}, df, True)[0].endswith(
        "Mark EDA bad only where the EDA itself is unusable.")
    assert GUIDE_CONTEXT_TEXT == "Guide rows are context only: mark only what you see on the scored row."
    assert guide_banner_lines({"ppg": CHANNELS["ppg"]}, df, True) == []
    # From run 9 (HLU's decision (b)): when the file's sbp/dbp machine check has "ppg_dropout", the line adds
    # that the machine pre-marks those stretches; without it (run 8 and earlier) the plain line stays.
    from channel_config import GUIDE_PREMARKED_TEXT
    from physio_io import reason_summary
    bp_reasons = {"no_coverage": [], "out_of_range": [], "ordering": [], "ppg_dropout": [[100, 400]]}
    premarked = {"Provenance": {"NumberOfSamples": n, "CTTiming": {"Pulse": {}}}, "MachineQC": {"Channels": {
        k: {"BadSegments": [[100, 400]], "Rule": "r", "SegmentsByReason": bp_reasons} for k in ("sbp", "dbp")}}}
    lines = guide_banner_lines({"sbp": CHANNELS["sbp"], "dbp": CHANNELS["dbp"]}, df, True, premarked)
    assert lines == [GUIDE_TEXTS[("ppg", "sbp")] + " " + GUIDE_PREMARKED_TEXT, GUIDE_CONTEXT_TEXT], lines
    assert lines[0].endswith("aren't real. The machine pre-marks those stretches; check the edges."), lines[0]
    run8_style = {**premarked, "MachineQC": {"Channels": {"sbp": {"BadSegments": [], "Rule": "r", "SegmentsByReason": {
        "no_coverage": [], "out_of_range": [], "ordering": []}}}}}
    assert guide_banner_lines({"sbp": CHANNELS["sbp"]}, df, True, run8_style)[0] == GUIDE_TEXTS[("ppg", "sbp")]
    assert reason_summary(bp_reasons) == "no usable finger pulse (1)"
    assert guide_banner_lines({"rsp": CHANNELS["rsp"]}, df, False) == []

    # Hidden, with the reason: a mostly-flagged (dead) belt; old-pipeline PPG; an empty column.
    dead = {"Provenance": {"NumberOfSamples": n, "CTTiming": {"Pulse": {}}},
            "MachineQC": {"Channels": {"rsp": {"BadSegments": [[1000, n]], "Rule": "r"}}}}
    assert rows(["eda"], sidecar=dead) == ["eda"]
    assert guide_banner_lines({"eda": CHANNELS["eda"]}, df, True, dead) == [
        "RSP (guide): not shown -- the machine check marks most of this run's RSP as unusable, so it can't guide you."]
    live = {**dead, "MachineQC": {"Channels": {"rsp": {"BadSegments": [[1000, 3000]], "Rule": "r"}}}}
    assert rows(["eda"], sidecar=live) == ["eda", "rsp"], "a partly flagged belt is still shown"
    old = {"SamplingFrequency": 1000, "Columns": list(df.columns)}
    assert rows(["sbp"], sidecar=old) == ["sbp"]
    assert "before the PPG timing fix" in guide_banner_lines({"sbp": CHANNELS["sbp"]}, df, True, old)[0]
    no_eda = df.copy()
    no_eda["eda"] = np.nan
    assert list(combine_with_reference_channels({"rsp": CHANNELS["rsp"]}, no_eda, ppg_guide=True)) == ["rsp"]
    assert guide_banner_lines({"rsp": CHANNELS["rsp"]}, no_eda, True) == [
        "EDA (guide): not shown -- this run has no usable EDA."]

    # Step 2: shown, never saved, recorded as guide_shown (per session and per channel).
    tmp = tempfile.mkdtemp()
    try:
        for guide, expected in ((True, ["rsp", "eda"]), (False, ["rsp"])):
            seen, out = {}, io.StringIO()
            with patch.object(mne.io.RawArray, "plot", lambda self, *a, **k: seen.setdefault("names", list(self.ch_names))), \
                 QUIET(out):
                r = run_stage_b(df, sfreq, {"rsp": CHANNELS["rsp"]}, "zy" if guide else "zx", tmp, "g", "s", ppg_guide=guide)
            assert seen["names"] == expected, seen
            assert ("Never mark RSP bad because of EDA" in out.getvalue()) is guide
            with open(r.saved_path) as f:
                saved = json.load(f)["channels"]
            assert set(saved) == {"rsp"}, "the guide is never saved"
            assert saved["rsp"]["guide_shown"] is guide and saved["rsp"]["edit_history"][0]["guide_shown"] is guide
        # A saved EDA review in the same file is untouched by an RSP session that shows EDA as its guide
        # (the guide shares the "eda" key but is never seeded, exported or kept as a legacy entry).
        label = CHANNELS["eda"]["segment_label"]

        def mark_eda(self, *args, **kwargs):
            self.set_annotations(self.annotations + mne.Annotations([3.0], [1.5], [label]))

        with patch.object(mne.io.RawArray, "plot", mark_eda), QUIET(io.StringIO()):
            r_eda = run_stage_b(df, sfreq, {"eda": CHANNELS["eda"]}, "zw", tmp, "g", "s")
        with open(r_eda.saved_path) as f:
            eda_before = json.load(f)["channels"]["eda"]
        assert eda_before["bad_segments"] == [[3000, 4500]], eda_before["bad_segments"]
        seen = {}
        with patch.object(mne.io.RawArray, "plot", lambda self, *a, **k: seen.setdefault("names", list(self.ch_names))), \
             QUIET(io.StringIO()):
            r_rsp = run_stage_b(df, sfreq, {"rsp": CHANNELS["rsp"]}, "zw", tmp, "g", "s")
        assert seen["names"] == ["rsp", "eda"] and r_rsp.saved_path == r_eda.saved_path
        with open(r_rsp.saved_path) as f:
            both = json.load(f)["channels"]
        assert set(both) == {"eda", "rsp"} and both["eda"] == eda_before, sorted(both)
        # Step 3 shows it too.
        a = {"rsp": {"mode": "segment", "bad_segments": [[1000, 2000]], "status": "complete"}}
        raw3, _ = build_reconciliation_raw(df, {"rsp": CHANNELS["rsp"]}, sfreq, a, a, "aa", "bb", ppg_guide=True)
        assert list(raw3.ch_names) == ["rsp", "eda"], raw3.ch_names
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # Real run-8/9 files: sub-001 r1's dead belt hides the RSP guide; sub-085 r1's live belt shows it.
    run8 = os.path.join(HERE, "test_input", "run8")
    for name, expected in (("sub-001_ses-run1", ["eda", "event"]), ("sub-085_ses-run1", ["eda", "rsp", "event"])):
        tsv = os.path.join(run8, f"{name}_task-sdi_physio.tsv.gz")
        if os.path.exists(tsv):
            d, _fs = load_physio_tsv(tsv)
            got = list(combine_with_reference_channels({"eda": CHANNELS["eda"]}, d, ppg_guide=True, sidecar=sidecar_of(tsv)))
            assert got == expected, (name, got)
    print("   OK: rows per session; HLU's wording plus the context line; the BP line adds 'pre-marks' only with "
          "ppg_dropout; hidden with a reason (dead belt, old PPG, "
          "empty); never saved; a saved EDA review untouched by an RSP session; guide_shown recorded; Steps 2 and 3; "
          "real files")


def test_ecg_guide_opt_in():
    print("PPG under ECG is opt-in (HLU, 2026-10-03, option 2), in Steps 1-3 and the form...")
    from annotation_io import guide_banner_lines, guide_shown_by_channel
    from channel_config import GUIDE_CONTEXT_TEXT, GUIDE_OPT_IN, GUIDE_TEXT, GUIDE_TEXTS
    from physio_review import _guide_flags, parse_args
    off, others = GUIDE_OPT_IN[("ppg", "ecg")]["off"], GUIDE_OPT_IN[("ppg", "ecg")]["shown_for_others"]
    assert off.startswith("PPG (guide): not shown under ECG by default") and "--ecg-ppg-guide" in off
    df, sfreq = generate_synthetic_demo(duration_sec=30)
    df["sbp"], df["dbp"] = 120.0, 80.0
    ecg_only = {"ecg": CHANNELS["ecg"]}

    # Banner lines: off by default (with why), the rhythm line when opted in, the "SBP/DBP only" line when
    # the row is on screen for SBP/DBP in a CLI session that also scores ECG.
    assert guide_banner_lines(ecg_only, df, True) == [off]
    assert guide_banner_lines(ecg_only, df, True, ecg_ppg_guide=True) == [GUIDE_TEXT, GUIDE_CONTEXT_TEXT]
    assert guide_banner_lines(ecg_only, df, False) == [], "--no-ppg-guide: nothing at all"
    bp = {k: CHANNELS[k] for k in ("ecg", "sbp", "dbp")}
    assert guide_banner_lines(bp, df, True) == [GUIDE_TEXTS[("ppg", "sbp")], others, GUIDE_CONTEXT_TEXT]
    assert others not in guide_banner_lines(bp, df, True, ecg_ppg_guide=True)
    no_ppg = df.copy()
    no_ppg["ppg"] = np.nan
    assert guide_banner_lines(ecg_only, no_ppg, True) == [], "no usable PPG: no 'off by default' line"
    combined = combine_with_reference_channels(bp, df, ppg_guide=True)
    assert list(combined)[:4] == ["ecg", "sbp", "dbp", "ppg"]
    assert guide_shown_by_channel(bp, combined)["ecg"] is True, "on screen for SBP/DBP: recorded for ECG too"

    tmp = tempfile.mkdtemp()
    try:
        # Step 2 default: no PPG row, the reason in the console, guide_shown false.
        seen, out = {}, io.StringIO()
        with patch.object(mne.io.RawArray, "plot", lambda self, *a, **k: seen.setdefault("names", list(self.ch_names))), \
             QUIET(out):
            r = run_stage_b(df, sfreq, ecg_only, "zz", tmp, "g", "synthetic")
        assert seen["names"] == ["ecg"] and off in out.getvalue(), seen
        with open(r.saved_path) as f:
            assert json.load(f)["channels"]["ecg"]["guide_shown"] is False
        # Step 3 default: the same.
        a = {"ecg": {"mode": "point", "indices": [1000, 2000], "status": "complete"}}
        seen3, out3 = {}, io.StringIO()
        with patch.object(mne.io.RawArray, "plot", lambda self, *a, **k: seen3.setdefault("names", list(self.ch_names))), \
             QUIET(out3):
            rc = run_stage_c(df, sfreq, ecg_only, os.path.join(tmp, "rec"), "g", "s", a, a, "aa", "bb", reconciler="hl")
        assert seen3["names"] == ["ecg"] and off in out3.getvalue()
        with open(rc.saved_path) as f:
            assert json.load(f)["channels"]["ecg"]["guide_shown"] is False
        # Step 1 default: ECG only, the reason printed, ppg_guide_shown false.
        step1, out1 = {}, io.StringIO()
        with patch.object(mne.io.RawArray, "plot", lambda self, *a, **k: step1.setdefault("names", list(self.ch_names))), \
             patch.object(qrs_template, "show_template_preview", return_value=(True, "")), QUIET(out1):
            res = qrs_template.run_stage_a({"1": df}, {"1": "t"}, {"1": tmp}, sfreq, "zz")
        assert step1["names"] == ["ecg"] and off in out1.getvalue()
        with open(res["1"]["json_path"]) as f:
            assert json.load(f)["ecg"]["ppg_guide_shown"] is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # The form's one box -> flags: ECG (or Step 1) ticked = --ecg-ppg-guide; other channels unticked = --no-ppg-guide.
    assert _guide_flags({"stage": "2", "channel": "ecg", "ppg_guide": True}) == ["--ecg-ppg-guide"]
    assert _guide_flags({"stage": "2", "channel": "ecg", "ppg_guide": False}) == []
    assert _guide_flags({"stage": "1", "channel": "ecg", "ppg_guide": True}) == ["--ecg-ppg-guide"]
    assert _guide_flags({"stage": "3", "channel": "ecg", "ppg_guide": False}) == []
    assert _guide_flags({"stage": "2", "channel": "sbp", "ppg_guide": True}) == []
    assert _guide_flags({"stage": "2", "channel": "rsp", "ppg_guide": False}) == ["--no-ppg-guide"]
    assert parse_args(["--synthetic", "--initials", "hlu", "--ecg-ppg-guide"]).ecg_ppg_guide is True
    assert parse_args(["--synthetic", "--initials", "hlu"]).ecg_ppg_guide is False
    print("   OK: off by default under ECG in Steps 1-3 (with the reason), on when asked for; the 'SBP/DBP only' "
          "line in mixed CLI sessions; the form's box maps to the flags")


def test_counter_loss_guide_line():
    print("PPG guide under SBP/DBP: the lost-in-transfer exception line (HLU, 2026-10-03)...")
    from annotation_io import guide_banner_lines
    from channel_config import GUIDE_CONTEXT_TEXT, GUIDE_PREMARKED_TEXT, GUIDE_TEXTS
    df, sfreq = generate_synthetic_demo(duration_sec=20)
    n = len(df)
    df["sbp"] = 120.0
    df["dbp"] = 80.0
    bp = {"BadSegments": [[100, 400]], "Rule": "r",
          "SegmentsByReason": {"no_coverage": [], "out_of_range": [], "ordering": [], "ppg_dropout": [[100, 400]]}}

    def sidecar(counter_loss):
        ppg_reasons = {"zero_run": [[100, 300]], "bridged": [], "window_rule": []}
        if counter_loss is not None:
            ppg_reasons["counter_loss"] = counter_loss
        segments = sorted([[100, 300]] + list(counter_loss or []))
        return {"SamplingFrequency": sfreq, "Provenance": {"NumberOfSamples": n, "CTTiming": {"Pulse": {}}},
                "MachineQC": {"Channels": {"sbp": bp, "dbp": bp, "ppg": {
                    "BadSegments": segments, "Rule": "r", "SegmentsByReason": ppg_reasons}}}}

    bp_keys = {"sbp": CHANNELS["sbp"], "dbp": CHANNELS["dbp"]}
    s = int(sfreq)
    loss = [[2 * s, 3 * s], [5 * s, int(5.5 * s)]]
    lines = guide_banner_lines(bp_keys, df, True, sidecar(loss))
    expected = ("PPG (guide): in 2 stretch(es) (1.5 s in all, at 2.0-3.0 s, 5.0-5.5 s) CareTaker pulse samples were "
                "lost in transfer, so the PPG guide can't be used to check BP there. The BP readings in those "
                "stretches were still measured by the device; the loss affected only the pulse copy sent to the "
                "app. Keep them, except where the machine marked a finger-pulse dropout.")
    assert lines == [GUIDE_TEXTS[("ppg", "sbp")] + " " + GUIDE_PREMARKED_TEXT, expected, GUIDE_CONTEXT_TEXT], lines
    assert guide_banner_lines({"dbp": CHANNELS["dbp"]}, df, True, sidecar(loss))[1] == expected, "DBP alone too"
    # No line: no losses in this run, a run-9 sidecar (no counter_loss key), guides off, no sidecar.
    for quiet in (sidecar([]), sidecar(None)):
        assert lines[1] not in guide_banner_lines(bp_keys, df, True, quiet)
        assert len(guide_banner_lines(bp_keys, df, True, quiet)) == 2
    assert guide_banner_lines(bp_keys, df, False, sidecar(loss)) == []
    assert not any("lost in transfer" in line for line in guide_banner_lines(bp_keys, df, True))
    # PPG under ECG: no exception line (BP only).
    assert not any("lost in transfer" in line for line in guide_banner_lines({"ecg": CHANNELS["ecg"]}, df, True,
                                                                             sidecar(loss)))
    # A guide that isn't shown (PPG mostly flagged) says why, and nothing else about it.
    hidden = sidecar([[0, n - 10]])
    hidden_lines = guide_banner_lines(bp_keys, df, True, hidden)
    assert len(hidden_lines) == 1 and "not shown" in hidden_lines[0], hidden_lines
    # More than GUIDE_EXCEPTION_MAX_LISTED stretches: the first 6 by time, the rest counted.
    many = [[k * s, k * s + s // 10] for k in range(1, 9)]
    line = guide_banner_lines(bp_keys, df, True, sidecar(many))[1]
    assert line.startswith("PPG (guide): in 8 stretch(es) (0.8 s in all, at 1.0-1.1 s, 2.0-2.1 s,"), line
    assert "6.0-6.1 s and 2 more) CareTaker" in line and "7.0-7.1 s" not in line, line
    print("   OK: HLU's wording with the stretches' times, once for SBP+DBP; none without losses, for a run-9 "
          "sidecar, with guides off, under ECG, or when the guide isn't shown; long lists counted")


def test_sbp_short_no_reading_note():
    print("SBP/DBP: short no-reading stretches inside the run get a 'keep them' banner line (HLU, 2026-10-02)...")
    from physio_io import machine_qc_notes, short_no_reading_gaps
    n, fs = 100000, 1000
    spans = [[0, 500], [10000, 10200], [50000, 53000], [99000, n]]
    entry = {"segments": spans, "by_reason": {"no_coverage": spans, "out_of_range": [], "ordering": []}}
    assert short_no_reading_gaps(entry, n, fs) == 1, "only the 0.2-s gap inside the run counts (not edges, not 3 s)"
    lines = machine_qc_notes({"sbp": entry}, ["sbp"], n, fs, {"sbp": "SBP"})
    assert ("SBP: short no-reading stretches (under 2 s) are gaps between CareTaker readings; the trace is blank "
            "there. Keep them (1 in this run).") in lines, lines
    none = {"segments": [[0, 500]], "by_reason": {"no_coverage": [[0, 500]]}}
    assert not any("short no-reading" in l for l in machine_qc_notes({"sbp": none}, ["sbp"], n, fs, {"sbp": "SBP"}))
    assert short_no_reading_gaps({"segments": spans, "by_reason": None}, n, fs) == 0, "no reasons: no line"
    if os.path.exists(SUB001):
        meta = sidecar_of(SUB001)
        from physio_io import machine_qc_segments
        d, _ = load_physio_tsv(SUB001)
        qc, _note = machine_qc_segments(meta, len(d))
        raw_reasons = meta["MachineQC"]["Channels"]["sbp"].get("SegmentsByReason") or {}
        if "no_coverage" in raw_reasons:
            # Recounted independently from the sidecar (8 in run 9; run 10 re-places BP, so not fixed).
            expected = sum(1 for a, b in raw_reasons["no_coverage"] if a > 0 and b < len(d) and (b - a) / 1000 < 2)
            k = short_no_reading_gaps(qc["sbp"], len(d), 1000)
            assert k == expected, (k, expected)
            print(f"   (sub-001 r1: {k} short gaps)")
    print("   OK: counted inside the run only, under 2 s; line shown with the count; absent otherwise")


if __name__ == "__main__":
    test_ecg_guide_opt_in()
    test_counter_loss_guide_line()
    test_sbp_short_no_reading_note()
    test_guide_rows_2026_09_27()
    test_bad_ecg_channel()
    test_bad_ppg_channel()
    test_run8_review_fixes()
    test_run8_decisions()
    test_consultation_formats()
    test_g2_no_guide_on_old_pipeline()
    test_step3_default_order()
    test_checkpoint2_fixes()
    test_checkpoint2_reprocess()
    test_guide_step2_and_3()
    test_guide_step1_template_identical()
    test_units_and_descriptions()
    test_gate_and_review_alone()
    test_ppg_checks()
    test_provenance_and_history()
    test_reprocess_ppg_check()
    test_practice_ppg()
    print("\nALL PPG BUILD TESTS PASSED.")
