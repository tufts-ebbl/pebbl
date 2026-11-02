#!/usr/bin/env python3
"""
Tests the study-profile settings beyond name and task (study_profile.py),
on a synthetic recording laid out like physiopipe's urglab output: columns
ecg, ppg, rsp, eda, emg_cor, emg_cor_env, the four *_peaks and event; event
codes 0 / 50 / 10-15; MachineQC with ecg 'flat', emg_cor 'mains' and ppg
'rail' stretches; no Provenance.CTTiming. Each setting is checked against
PEBBL's defaults on the same file (negative control): the built-in settings,
or, for whole sessions (which need a profile since 2026-11-01), a profile
that sets only the task label (DEFAULTS_PROFILE):

  1. Reading the settings; their built-in values; wrong values refused.
  2. runs: only the profile's runs are looked for.
  3. channels: the default set and the refusal of a channel not offered.
  4. ppg_requires_timing_marker = false: PPG review without CTTiming, and the
     'rail' reason counts as a dropout mask (no "no mask yet" note).
  5. bad_ecg_from_machine_qc: bad_ecg starts from MachineQC's ECG stretches;
     corrugator EMG starts from its own.
  6. event_labels: the viewer gets the names; built in, numbers only.
  7. ecg_seed_snap_ms: Step 1's seed window (15 ms built in, since 2026-10-31).
  8. ECG flip (--ecg-inverted): on an ECG recorded upside down, a drawn R
     peak lands on the true R only when flipped; the saved review records
     it; the defaults refuse it. Reopening it with the other setting asks
     whether to start the ECG over: Cancel stops with nothing changed; yes
     keeps the earlier ECG entries under legacy keys and the other channels
     as they were, in both directions. Another channel's session isn't
     stopped. Step 1 and an old template point the RA the right way.
  9. The form follows the data folder's profile; a ticked flip reaches the
     command line.
 10. check_reprocess_alignment.py uses the profile's task label.
 11. keep_machine_stretches: COR's machine stretches are to be left (the
     opening text and the over-90% warning say so), and deleting or
     shortening one draws a check line before saving; widening doesn't. By
     default COR's stretches are "check each one", with no such line.

Run with: annotate_env\\Scripts\\python.exe test_study_settings.py
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
from unittest.mock import patch

import mne
import numpy as np

import annotation_io
import check_reprocess_alignment
import physio_review
import processing_log
import qrs_template
import session_summary_gui
import study_profile
from channel_config import select_channels
from physio_io import build_run_path, generate_synthetic_demo, machine_spans_not_kept, resolve_subject_run_paths
from provenance import new_pipeline_ppg

QUIET = contextlib.redirect_stdout
PROFILE = """schema_version = 1
name = "Demo urglab"
task = "urglab"
runs = ["1"]
channels = ["ecg", "ppg", "rsp", "eda", "emg_cor"]
practice = false
ppg_requires_timing_marker = false
bad_ecg_from_machine_qc = true
ecg_inversion_option = true
ecg_seed_snap_ms = 0
keep_machine_stretches = ["emg_cor"]
[event_labels]
0 = "off task"
50 = "baseline"
13 = "Urgent tracing"
"""
DEFAULTS_PROFILE = 'schema_version = 1\nname = "urglab defaults"\ntask = "urglab"\n'


def reset():
    study_profile.activate(study_profile.BUILT_IN)
    study_profile.set_ecg_flipped(False)


def write_profile(folder, text=PROFILE):
    with open(os.path.join(folder, study_profile.PROFILE_FILE), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def make_recording(folder, subject, upside_down=False, seconds=30):
    """A urglab-shaped synthetic run 1; returns (tsv path, the upright synthetic data)."""
    df, sfreq = generate_synthetic_demo(duration_sec=seconds)
    n = len(df)
    rng = np.random.default_rng(2024)
    out = df[["ecg", "ppg", "rsp", "eda"]].copy()
    if upside_down:
        out["ecg"] = -out["ecg"]
    out["emg_cor"] = rng.normal(0, 0.02, n)
    out["emg_cor_env"] = np.abs(out["emg_cor"])
    for col in ("ecg_peaks", "ppg_peaks", "rsp_peaks", "eda_peaks"):
        out[col] = df[col].astype(int)
    event = np.zeros(n, dtype=int)
    event[2000:12000] = 50
    for phase, start in enumerate(range(14000, 26000, 2000)):
        event[start:start + 1800] = 10 + phase
    out["event"] = event
    study_profile.activate(study_profile.StudyProfile(name="t", task="urglab", path="x", sha256="x"))
    try:
        tsv = build_run_path(folder, subject, "1")
    finally:
        study_profile.activate(study_profile.BUILT_IN)
    os.makedirs(os.path.dirname(tsv))
    out.to_csv(tsv, sep="\t", index=False, header=False, compression="gzip")
    columns = list(out.columns)
    sidecar = {
        "Columns": columns, "SamplingFrequency": sfreq,
        **{c: {"Units": u} for c, u in (("ecg", "mV"), ("ppg", "V"), ("rsp", "V"), ("eda", "microsiemens"),
                                        ("emg_cor", "mV"), ("emg_cor_env", "mV"))},
        "Provenance": {"NumberOfSamples": n, "PipelineCommit": "synthetic"},
        "MachineQC": {"Channels": {
            "ecg": {"BadSegments": [[1000, 2000]], "SegmentsByReason": {"flat": [[1000, 2000]]}, "Rule": "flat"},
            "emg_cor": {"BadSegments": [[3000, 3500]], "SegmentsByReason": {"mains": [[3000, 3500]]}, "Rule": "emg"},
            "ppg": {"BadSegments": [[4000, 6000]], "SegmentsByReason": {"rail": [[4000, 6000]]}, "Rule": "ppg"},
            "rsp": {"BadSegments": [], "Rule": "rsp"}, "eda": {"BadSegments": [], "Rule": "eda"}}},
    }
    with open(tsv.replace("_physio.tsv.gz", "_physio.json"), "w") as f:
        json.dump(sidecar, f)
    return tsv, df


def run_session(argv, snapshot_dir, plot=None, restart_ecg=False):
    """
    A non-interactive session; returns (exit code, console text, the plot
    call's kwargs). restart_ecg answers "Start the ECG over?" (asked when the
    saved ECG review's flip differs); seen["asked"] records whether it was.
    """
    seen = {}

    def record_plot(self, *args, **kwargs):
        seen.update(kwargs)
        if plot:
            plot(self)

    def answer_restart(saved_flipped):
        seen["asked"] = saved_flipped
        return restart_ecg

    out = io.StringIO()
    code = 0
    with patch.object(sys, "argv", ["physio_review.py", "--no-local-copy", "--initials", "zz"] + argv), \
         patch.object(mne.io.RawArray, "plot", record_plot), \
         patch.object(processing_log, "LOCAL_SNAPSHOT_DIR", snapshot_dir), \
         patch.object(session_summary_gui, "confirm_ecg_restart", answer_restart), \
         patch.object(session_summary_gui, "show_session_summary", session_summary_gui.headless_decision()), \
         patch("builtins.input", side_effect=AssertionError("no terminal prompt expected")), \
         QUIET(out), contextlib.redirect_stderr(out):
        try:
            physio_review.main()
        except SystemExit as e:
            code = e.code
    reset()
    return code, out.getvalue(), seen


def saved(tsv):
    with open(tsv.replace("_physio.tsv.gz", "_annotations_zz.json")) as f:
        return json.load(f)


def refused(text):
    folder = tempfile.mkdtemp()
    try:
        write_profile(folder, text)
        try:
            study_profile.load(folder)
        except SystemExit as e:
            return str(e)
        raise AssertionError("expected the profile to be refused")
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def main():
    tmp = tempfile.mkdtemp()
    snapshot = os.path.join(tmp, "snapshot")
    os.makedirs(snapshot)
    try:
        print("1. Settings read; built-in values; wrong values refused...")
        b = study_profile.BUILT_IN
        assert (b.runs, b.channels, b.practice, b.ppg_requires_timing_marker, b.bad_ecg_from_machine_qc,
                b.ecg_inversion_option, b.ecg_seed_snap_ms, b.event_labels) == (["1", "2"], None, True, True, False,
                                                                                False, 15, {})
        assert b.keep_machine_stretches == []
        folder = os.path.join(tmp, "study")
        os.makedirs(folder)
        write_profile(folder)
        p = study_profile.load(folder)
        assert (p.runs, p.channels, p.practice, p.ppg_requires_timing_marker, p.bad_ecg_from_machine_qc,
                p.ecg_inversion_option, p.ecg_seed_snap_ms) == (["1"], ["ecg", "ppg", "rsp", "eda", "emg_cor"], False,
                                                                False, True, True, 0)
        assert p.event_labels == {0: "off task", 50: "baseline", 13: "Urgent tracing"}
        assert p.keep_machine_stretches == ["emg_cor"]
        head = 'schema_version = 1\nname = "x"\ntask = "x"\n'
        for extra, expected in (('runs = ["a"]\n', "runs must be"), ("runs = []\n", "runs must be"),
                                ('channels = ["xyz"]\n', "doesn't know: xyz"), ('practice = "yes"\n', "true or false"),
                                ("ecg_seed_snap_ms = 300\n", "0 to 200"), ('[event_labels]\nx = "y"\n', "event codes"),
                                ('keep_machine_stretches = "emg_cor"\n', "must be a list"),
                                ('keep_machine_stretches = ["emg_cor", "emg_cor"]\n', "must be a list"),
                                ('keep_machine_stretches = ["ecg"]\n', "aren't bad-stretch channels: ecg"),
                                ('channels = ["ecg"]\nkeep_machine_stretches = ["emg_cor"]\n', "doesn't offer: emg_cor")):
            assert expected in refused(head + extra), (extra, refused(head + extra))
        companion = os.path.join(tmp, "companion")
        os.makedirs(companion)
        write_profile(companion, head + 'channels = ["ecg"]\nkeep_machine_stretches = ["bad_ecg"]\n')
        assert study_profile.load(companion).keep_machine_stretches == ["bad_ecg"], "a companion comes with its parent"
        print("   OK")

        tsv, upright = make_recording(folder, "041")
        print("2-3. runs and channels follow the profile; built in, as before...")
        with QUIET(io.StringIO()):
            assert list(resolve_subject_run_paths(None, None, folder, "041")) == ["1", "2"]
            assert set(select_channels(None)) >= {"sbp", "dbp", "emg_zyg"}
            study_profile.activate(p)
            assert list(resolve_subject_run_paths(None, None, folder, "041")) == ["1"]
            assert set(select_channels(None)) == {"ecg", "rsp", "eda", "emg_cor"}  # PPG is reviewed on its own
            try:
                select_channels(["sbp"])
                raise AssertionError("expected SystemExit")
            except SystemExit as e:
                assert "doesn't review SBP" in str(e), str(e)
        reset()
        print("   OK")

        print("4. PPG without CTTiming: allowed by the profile, 'rail' is a mask; refused built in...")
        assert not new_pipeline_ppg({"Provenance": {}})
        study_profile.activate(p)
        assert new_pipeline_ppg({"Provenance": {}})
        reset()
        code, text, _ = run_session(["--box-path", folder, "--subject", "041", "--run", "1", "--channels", "ppg"],
                                    snapshot)
        assert code in (0, None) and "no per-sample PPG dropout mask" not in text, text[-1500:]
        assert saved(tsv)["channels"]["bad_ppg"]["bad_segments"] == [[4000, 6000]]
        plain = os.path.join(tmp, "plain")
        os.makedirs(plain)
        write_profile(plain, DEFAULTS_PROFILE)
        plain_tsv, _ = make_recording(plain, "041")
        code, text, _ = run_session(["--box-path", plain, "--subject", "041", "--run", "1", "--channels", "ppg"],
                                    snapshot)
        assert code not in (0, None) and "before the PPG timing fix" in text, text[-1500:]
        print("   OK")

        print("5-6. bad_ecg and EMG start from MachineQC; event names reach the viewer; built in, neither...")
        code, text, seen = run_session(["--box-path", folder, "--subject", "041", "--run", "1", "--channels", "ecg",
                                        "--ecg-source", "batch"], snapshot)
        assert code in (0, None), text[-1500:]
        ecg_saved = saved(tsv)["channels"]
        assert ecg_saved["bad_ecg"]["bad_segments"] == [[1000, 2000]], ecg_saved["bad_ecg"]
        assert ecg_saved["bad_ecg"]["seed"]["source"] == "MachineQC"
        assert seen["event_id"] == {"0 off task": 0, "50 baseline": 50, "13 Urgent tracing": 13}, seen.get("event_id")
        code, text, _ = run_session(["--box-path", folder, "--subject", "041", "--run", "1", "--channels", "emg_cor"],
                                    snapshot)
        assert code in (0, None) and saved(tsv)["channels"]["emg_cor"]["bad_segments"] == [[3000, 3500]], text[-1500:]
        code, text, seen = run_session(["--box-path", plain, "--subject", "041", "--run", "1", "--channels", "ecg",
                                        "--ecg-source", "batch"], snapshot)
        assert code in (0, None), text[-1500:]
        assert saved(plain_tsv)["channels"]["bad_ecg"]["bad_segments"] == [], "by default bad_ecg starts blank"
        assert "event_id" not in seen and "Study profile: urglab defaults" in text
        print("   OK")

        print("7. Step 1's seed window: 15 ms built in, the profile's otherwise...")
        assert qrs_template.seed_snap_samples(1000) == 15 and qrs_template.seed_snap_samples(500) == 8  # built in
        study_profile.activate(p)
        assert qrs_template.seed_snap_samples(1000) == 0
        study_profile.activate(study_profile.StudyProfile(name="t", task="t", path="x", sha256="x",
                                                          ecg_seed_snap_ms=50))
        assert qrs_template.seed_snap_samples(1000) == 50 and qrs_template.seed_snap_samples(2000) == 100
        reset()
        print("   OK")

        print("8. ECG recorded upside down: a drawn R lands on the true R only when flipped...")
        beat = int(np.where(upright["ecg_peaks"].to_numpy())[0][5])
        true_r = beat - 20 + int(np.argmax(upright["ecg"].to_numpy()[beat - 20:beat + 20]))

        def draw_r(raw):
            sf = raw.info["sfreq"]
            near = [i for i, a in enumerate(raw.annotations)
                    if a["description"] == "peak_ecg" and abs(a["onset"] * sf - true_r) < 100]
            raw.annotations.delete(near)
            raw.annotations.append((true_r - 30) / sf, 60 / sf, "peak_ecg")

        def nearest(tsv_path):
            idx = np.array(saved(tsv_path)["channels"]["ecg"]["indices"])
            return int(idx[np.argmin(np.abs(idx - true_r))])

        flipped_folder = os.path.join(tmp, "flipped")
        os.makedirs(flipped_folder)
        write_profile(flipped_folder)
        flipped_tsv, _ = make_recording(flipped_folder, "041", upside_down=True)
        base = ["--subject", "041", "--run", "1", "--channels", "ecg", "--ecg-source", "batch"]
        code, text, _ = run_session(["--box-path", flipped_folder] + base + ["--ecg-inverted"], snapshot, plot=draw_r)
        assert code in (0, None) and "shown flipped" in text, text[-1500:]
        assert nearest(flipped_tsv) == true_r and saved(flipped_tsv)["channels"]["ecg"]["inverted"] is True
        code, text, seen = run_session(["--box-path", flipped_folder] + base, snapshot)
        assert seen.get("asked") is True, "the RA is asked whether to start the ECG over"
        assert code not in (0, None) and "made with 'Flip the ECG" in text, text[-1500:]
        assert "You chose not to start the ECG over" in text and saved(flipped_tsv)["channels"]["ecg"]["inverted"]
        cor = ["--subject", "041", "--run", "1", "--channels", "emg_cor"]  # no flip: the form sends it only with ECG
        code, text, seen = run_session(["--box-path", flipped_folder] + cor, snapshot)
        assert code in (0, None) and "asked" not in seen, f"another channel's session isn't stopped: {text[-1500:]}"

        unflipped_folder = os.path.join(tmp, "unflipped")
        os.makedirs(unflipped_folder)
        write_profile(unflipped_folder)
        unflipped_tsv, _ = make_recording(unflipped_folder, "041", upside_down=True)
        code, text, _ = run_session(["--box-path", unflipped_folder] + base, snapshot, plot=draw_r)
        assert code in (0, None), text[-1500:]
        assert nearest(unflipped_tsv) != true_r and "inverted" not in saved(unflipped_tsv)["channels"]["ecg"]

        # The flip forgotten: Step 1 says what's coming; Step 2 starts the ECG over once the RA confirms.
        code, text, _ = run_session(["--box-path", unflipped_folder] + cor, snapshot)
        assert code in (0, None), text[-1500:]
        before = saved(unflipped_tsv)["channels"]
        study_profile.set_ecg_flipped(True)
        out = io.StringIO()
        with QUIET(out):
            qrs_template.save_qrs_template_outputs(os.path.dirname(unflipped_tsv),
                                                   os.path.basename(unflipped_tsv)[:-len("_physio.tsv.gz")], "zz",
                                                   [beat], upright.iloc[:1])
        reset()
        assert "will offer to set that review aside" in out.getvalue(), out.getvalue()
        code, text, seen = run_session(["--box-path", unflipped_folder] + base + ["--ecg-inverted"], snapshot,
                                       plot=draw_r, restart_ecg=True)
        assert code in (0, None) and seen.get("asked") is False and "the ECG starts over" in text, text[-1500:]
        after = saved(unflipped_tsv)["channels"]
        assert after["ecg"]["inverted"] is True and nearest(unflipped_tsv) == true_r
        assert after["ecg_points_legacy"] == before["ecg"] and after["bad_ecg_segments_legacy"] == before["bad_ecg"]
        assert after["emg_cor"] == before["emg_cor"], "other channels are kept as they were"
        # ...and back the other way (an rS ECG flipped by mistake).
        code, text, seen = run_session(["--box-path", flipped_folder] + base, snapshot, restart_ecg=True)
        assert code in (0, None) and seen.get("asked") is True, text[-1500:]
        assert "inverted" not in saved(flipped_tsv)["channels"]["ecg"]
        assert saved(flipped_tsv)["channels"]["ecg_points_legacy"]["inverted"] is True
        # A template made with the other setting points to Step 1.
        template = os.path.join(tmp, "template.json")
        with open(template, "w") as f:
            json.dump({"ecg": {"corrected_peaks": [beat], "initials": ["zz"], "inverted": True}}, f)
        try:
            annotation_io.resolve_ecg_peaks(upright, "template", template)
            raise AssertionError("expected SystemExit")
        except SystemExit as e:
            assert "rebuild your template in Step 1 first" in str(e), str(e)
        code, text, _ = run_session(["--box-path", plain] + base + ["--ecg-inverted"], snapshot)
        assert code not in (0, None) and "doesn't use 'Flip the ECG'" in text, text[-1500:]
        print(f"   OK: flipped, the drawn mark is on the true R ({true_r}); unflipped, it isn't")

        print("9. The form follows the data folder's profile; a ticked flip reaches the command line...")
        from PyQt6.QtWidgets import QApplication
        import simple_gui
        app = QApplication.instance() or QApplication([])  # noqa: F841
        simple_gui.ReviewLauncherDialog.USE_SAVED_SETTINGS = False
        dlg = simple_gui.ReviewLauncherDialog()
        items = lambda combo: [combo.itemText(i) for i in range(combo.count())]  # noqa: E731
        assert items(dlg.run_combo) == ["1", "2"] and not dlg.practice_checkbox.isHidden()
        assert dlg.ecg_flip_checkbox.isHidden() and "sbp" in dlg._channel_keys
        dlg.box_path_edit.setText(folder)
        assert dlg._apply_study_profile()
        assert items(dlg.run_combo) == ["1"] and items(dlg.template_run_combo) == ["1"]
        assert dlg._channel_keys == ["ecg", "rsp", "ppg", "eda", "emg_cor"], dlg._channel_keys
        assert dlg.practice_checkbox.isHidden() and not dlg.ecg_flip_checkbox.isHidden()
        dlg.box_path_edit.setText(plain)
        assert dlg._apply_study_profile() and items(dlg.run_combo) == ["1", "2"] and dlg.ecg_flip_checkbox.isHidden()
        dlg.box_path_edit.setText(folder)
        dlg._apply_study_profile()
        dlg.step2_radio.setChecked(True)
        dlg.channel_combo.setCurrentIndex(dlg._channel_keys.index("ecg"))
        dlg.subject_edit.setText("041")
        dlg.initials_edit.setText("zz")
        dlg.no_local_copy_checkbox.setChecked(True)
        dlg.ecg_flip_checkbox.setChecked(True)
        with patch.object(dlg, "_confirm_initials", return_value=True), patch.object(dlg, "accept"):
            dlg._on_submit()
        assert dlg.result_values.get("ecg_inverted") is True, dlg.result_values
        with patch.object(sys, "argv", ["physio_review.py"]), \
             patch("simple_gui.collect_inputs_via_gui", return_value=dlg.result_values), QUIET(io.StringIO()):
            args = physio_review.parse_args_or_show_gui()
        assert args.ecg_inverted is True and args.box_path == folder
        print("   OK")

        print("10. check_reprocess_alignment.py uses the profile's task label...")
        new_root = os.path.join(tmp, "new")
        shutil.copytree(os.path.join(folder, "sub-041"), os.path.join(new_root, "sub-041"))
        out = io.StringIO()
        with QUIET(out):
            check_reprocess_alignment.main(["--old-root", folder, "--new-root", new_root, "--all"])
        reset()
        lines = [line for line in out.getvalue().splitlines() if line.startswith("sub-041 run 1:")]
        assert "Study profile: Demo urglab" in out.getvalue() and lines, out.getvalue()[-1500:]
        assert "MISSING" not in lines[0].upper(), lines
        # 30-s synthetic files can't fit the checker's 30-s ECG windows plus the lag search: it says so.
        assert "too short to compare" in lines[0] and " Hz)" not in lines[0], lines
        print(f"   OK: {lines[0]}")

        print("11. keep_machine_stretches: leave COR's machine stretches; deleting one is flagged before saving...")
        assert machine_spans_not_kept([[100, 200]], [[100, 200]]) == 0
        assert machine_spans_not_kept([[100, 200]], [[90, 210]]) == 0, "widened"
        assert machine_spans_not_kept([[100, 200]], [[100, 150], [150, 200]]) == 0, "redrawn in two pieces"
        assert machine_spans_not_kept([[100, 200]], [[101, 199]]) == 0, "a sample of rounding at each end"
        assert machine_spans_not_kept([[100, 200]], [[100, 180]]) == 1, "shortened"
        assert machine_spans_not_kept([[100, 200], [300, 400]], [[300, 400]]) == 1, "deleted"

        def fresh(name, profile_text=PROFILE, whole_run=False):
            """A new data folder (so COR starts fresh), optionally with COR flagged over the whole run."""
            new = os.path.join(tmp, name)
            os.makedirs(new)
            write_profile(new, profile_text)
            new_tsv, _ = make_recording(new, "041")
            if whole_run:
                sidecar_path = new_tsv.replace("_physio.tsv.gz", "_physio.json")
                with open(sidecar_path) as f:
                    meta = json.load(f)
                n = meta["Provenance"]["NumberOfSamples"]
                meta["MachineQC"]["Channels"]["emg_cor"] = {"BadSegments": [[0, n]], "Rule": "emg",
                                                            "SegmentsByReason": {"flat_record": [[0, n]]}}
                with open(sidecar_path, "w") as f:
                    json.dump(meta, f)
            return new

        def cor_spans(raw):
            return [i for i, a in enumerate(raw.annotations) if a["description"] == "bad_emg_cor"]

        def delete_cor(raw):
            raw.annotations.delete(cor_spans(raw))

        def widen_cor(raw):  # the machine's [3000, 3500) redrawn as [2900, 3600)
            raw.annotations.delete(cor_spans(raw))
            raw.annotations.append(2900 / raw.info["sfreq"], 700 / raw.info["sfreq"], "bad_emg_cor")

        cor = ["--subject", "041", "--run", "1", "--channels", "emg_cor"]
        removed = "EMG_Corrugator: you removed or shortened 1 of the machine's 1 bad stretch(es)"
        code, text, _ = run_session(["--box-path", fresh("keep_left")] + cor, snapshot)
        assert code in (0, None) and "Leave them: they count in the analysis whatever you do" in text, text[-1500:]
        assert "Leave them; add any it missed" in text and "delete any that are fine" not in text, text[-1500:]
        assert "you removed or shortened" not in text, text[-1500:]
        code, text, _ = run_session(["--box-path", fresh("keep_deleted")] + cor, snapshot, plot=delete_cor)
        assert code in (0, None) and f"CHECK: {removed}" in text, text[-1500:]
        code, text, _ = run_session(["--box-path", fresh("keep_widened")] + cor, snapshot, plot=widen_cor)
        assert code in (0, None) and "you removed or shortened" not in text, text[-1500:]
        code, text, _ = run_session(["--box-path", fresh("keep_whole", whole_run=True)] + cor, snapshot,
                                    plot=delete_cor)
        assert code in (0, None) and "Leave this flag: it counts in the analysis whatever you do" in text, text[-1500:]
        assert "disconnected" not in text and "Save only if you're sure" not in text, text[-1500:]
        assert f"CHECK: {removed}" in text, text[-1500:]
        code, text, _ = run_session(["--box-path", fresh("check_deleted", DEFAULTS_PROFILE)] + cor, snapshot,
                                    plot=delete_cor)
        assert code in (0, None) and "Check each one: delete any that are fine" in text, text[-1500:]
        assert "Leave them" not in text and "you removed or shortened" not in text, text[-1500:]
        print("   OK: left alone, deleted, widened, flagged over the whole run; by default, check each one")
    finally:
        reset()
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nALL STUDY SETTINGS TESTS PASSED.")


if __name__ == "__main__":
    main()
