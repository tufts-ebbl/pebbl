#!/usr/bin/env python3
"""
Tests physio_review.py's orchestration: does it correctly decide when to
run Stage A vs. skip it, and does the whole pipeline (Stage A + Stage B,
or just Stage B) actually produce the right output files?

The interactive raw.plot() call and the template-preview dialog
(show_template_preview()) are both mocked out, so this exercises the
ENTIRE real wiring (data loading -> Stage A's window/template-building/
cross-correlation -> Stage B's seeding/export/save) using only the
auto-seeded peaks, standing in for "the RA didn't change anything." It
does not test the actual click/drag interaction or what the preview looks
like (that needs a human), but it does prove the orchestration and every
non-GUI step around it are correct.

Run with: annotate_env\\Scripts\\python.exe test_physio_review.py
"""

import contextlib
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
import pandas as pd

import physio_review
import processing_log
import practice_certification
import practice_report_gui
import qrs_template
import session_log
from physio_io import build_run_path, generate_synthetic_demo

import session_summary_gui  # noqa: E402

# The session-summary dialog is modal; answer "Save" immediately so these
# non-interactive tests never block (tests needing other answers patch it).
session_summary_gui.show_session_summary = session_summary_gui.headless_decision()


def run_main(argv):
    with patch.object(sys, "argv", ["physio_review.py"] + argv):
        physio_review.main()


def make_fake_real_subject(box_path, subject, run, duration_sec=30):
    """
    Writes a real (non-synthetic-mode) *_physio.tsv.gz + JSON sidecar under
    box_path, using generate_synthetic_demo()'s data -- for tests that need
    to exercise the REAL --box-path/--subject code path (e.g. resolving box
    path/subject, writing to processing_log.csv), which --synthetic mode
    deliberately bypasses entirely.
    """
    df, sfreq = generate_synthetic_demo(duration_sec=duration_sec)
    tsv_path = build_run_path(box_path, subject, run)
    os.makedirs(os.path.dirname(tsv_path), exist_ok=True)

    columns = list(df.columns)
    df.to_csv(tsv_path, sep="\t", index=False, header=False, compression="gzip")

    json_path = tsv_path.replace("_physio.tsv.gz", "_physio.json")
    with open(json_path, "w") as f:
        json.dump({"Columns": columns, "SamplingFrequency": sfreq}, f)

    return tsv_path


def main():
    tmp_dir = tempfile.mkdtemp(prefix="physio_review_smoke_")
    try:
        # raw.plot() mocked out (no GUI); show_template_preview() mocked to
        # auto-approve (added after a user asked "when I'm done with step 1,
        # do I just close the window?", which surfaced that closing without
        # editing wasn't actually a way to reject a bad window -- see
        # HANDOFF.md -- and later replaced with an actual template-preview
        # dialog per a follow-up request).
        with patch.object(mne.io.RawArray, "plot", return_value=None), \
             patch.object(qrs_template, "show_template_preview", return_value=True):

            print("1. Fresh subject, reviewing run 1: no template exists yet -> Stage A should run...")
            with patch.object(physio_review, "run_stage_a", wraps=physio_review.run_stage_a) as spy_a:
                run_main(["--synthetic", "--duration", "30", "--run", "1",
                          "--initials", "hlu", "--out-dir", tmp_dir])
                assert spy_a.called, "Stage A should have run since no template existed yet"
            template_files = sorted(glob.glob(os.path.join(tmp_dir, "*_ecg_corrected_qrs_hlu.json")))
            assert len(template_files) == 2, f"expected templates for both runs, got {template_files}"
            annotation_files = sorted(glob.glob(os.path.join(tmp_dir, "*_annotations_hlu.json")))
            assert len(annotation_files) == 1 and "run1" in annotation_files[0]
            print(f"   OK: Stage A built templates for both runs {[os.path.basename(f) for f in template_files]}, "
                  f"Stage B saved {os.path.basename(annotation_files[0])}")

            print("2. Same subject, reviewing run 2 now: a template ALREADY exists -> Stage A should be SKIPPED...")
            with patch.object(physio_review, "run_stage_a", wraps=physio_review.run_stage_a) as spy_a:
                run_main(["--synthetic", "--duration", "30", "--run", "2",
                          "--initials", "hlu", "--out-dir", tmp_dir])
                assert not spy_a.called, "Stage A should NOT have run -- a template already existed"
            annotation_files = sorted(glob.glob(os.path.join(tmp_dir, "*_annotations_hlu.json")))
            assert len(annotation_files) == 2, f"expected annotations for both runs now, got {annotation_files}"
            print("   OK: Stage A skipped, Stage B ran directly for run 2 using the existing template")

            print("3. --force-template should rebuild even though one already exists...")
            with patch.object(physio_review, "run_stage_a", wraps=physio_review.run_stage_a) as spy_a:
                run_main(["--synthetic", "--duration", "30", "--run", "1", "--initials", "hlu",
                          "--out-dir", tmp_dir, "--force-template"])
                assert spy_a.called, "Stage A should have run because --force-template was passed"
            print("   OK: --force-template forced Stage A to run again")

            print("4. --ecg-source batch should skip Stage A entirely, even on a brand new subject...")
            tmp_dir_2 = tempfile.mkdtemp(prefix="physio_review_batch_", dir=tmp_dir)
            with patch.object(physio_review, "run_stage_a", wraps=physio_review.run_stage_a) as spy_a:
                run_main(["--synthetic", "--duration", "30", "--run", "1", "--initials", "hlu",
                          "--out-dir", tmp_dir_2, "--ecg-source", "batch"])
                assert not spy_a.called, "Stage A should never run under --ecg-source batch"
            template_files_2 = glob.glob(os.path.join(tmp_dir_2, "*_ecg_corrected_qrs_*.json"))
            assert len(template_files_2) == 0, "no template should have been built under --ecg-source batch"
            print("   OK: --ecg-source batch skipped Stage A and built no template")

            print("5. Rejecting the template preview should abort, not silently proceed...")
            tmp_dir_3 = tempfile.mkdtemp(prefix="physio_review_reject_", dir=tmp_dir)
            exit_message = None
            with patch.object(qrs_template, "show_template_preview", return_value=False):
                try:
                    run_main(["--synthetic", "--duration", "30", "--run", "1", "--initials", "hlu",
                              "--out-dir", tmp_dir_3])
                    raise AssertionError("expected SystemExit when the template window is rejected")
                except SystemExit as e:
                    exit_message = str(e)
            assert "not building a template" in exit_message.lower(), exit_message
            rejected_templates = glob.glob(os.path.join(tmp_dir_3, "*_ecg_corrected_qrs_*.json"))
            assert len(rejected_templates) == 0, "no template should exist after rejecting the window"
            print(f"   OK: rejected cleanly ({exit_message!r}), no template file was written")

            print("6. --stage 1 runs ONLY Stage A -- no Stage B, no annotations file, for either run...")
            tmp_dir_4 = tempfile.mkdtemp(prefix="physio_review_stage1_", dir=tmp_dir)
            with patch.object(physio_review, "run_stage_a", wraps=physio_review.run_stage_a) as spy_a, \
                 patch.object(physio_review, "run_stage_b") as spy_b:
                run_main(["--synthetic", "--duration", "30", "--initials", "hlu",
                          "--out-dir", tmp_dir_4, "--stage", "1"])
                assert spy_a.called, "Stage A should run under --stage 1"
                assert not spy_b.called, "Stage B should NEVER run under --stage 1"
            stage1_templates = glob.glob(os.path.join(tmp_dir_4, "*_ecg_corrected_qrs_*.json"))
            assert len(stage1_templates) == 2, f"expected templates for both runs, got {stage1_templates}"
            annotations_after_stage1 = glob.glob(os.path.join(tmp_dir_4, "*_annotations_*.json"))
            assert len(annotations_after_stage1) == 0, "no Stage B output should exist after --stage 1"
            print("   OK: --stage 1 built templates for both runs and never touched Stage B")

            print("7. --stage 2 on a subject with NO template yet: never builds one, uses batch peaks...")
            tmp_dir_5 = tempfile.mkdtemp(prefix="physio_review_stage2_", dir=tmp_dir)
            with patch.object(physio_review, "run_stage_a") as spy_a:
                run_main(["--synthetic", "--duration", "30", "--run", "1", "--initials", "hlu",
                          "--out-dir", tmp_dir_5, "--stage", "2", "--channels", "ppg"])
                assert not spy_a.called, "Stage A should NEVER run under --stage 2"
            stage2_templates = glob.glob(os.path.join(tmp_dir_5, "*_ecg_corrected_qrs_*.json"))
            assert len(stage2_templates) == 0, "no template should have been built under --stage 2"
            stage2_annotations = glob.glob(os.path.join(tmp_dir_5, "*_annotations_*.json"))
            assert len(stage2_annotations) == 1
            print("   OK: --stage 2 reviewed ppg only, using batch peaks, without ever running Stage A")

            print("8. --stage 2 on a subject that ALREADY has a template: uses it (like --ecg-source auto)...")
            with patch.object(physio_review, "run_stage_a") as spy_a:
                run_main(["--synthetic", "--duration", "30", "--run", "1", "--initials", "hlu",
                          "--out-dir", tmp_dir_4, "--stage", "2", "--channels", "ecg"])
                assert not spy_a.called
            print("   OK: --stage 2 reused the existing template without rebuilding it")

            print("9. Resume support: re-running --stage 2 for the same run/initials should seed "
                  "the viewer from the PREVIOUS session's saved corrections, not re-seed from "
                  "scratch (regression: closing partway through a long review used to silently "
                  "discard prior edits when the RA reopened the same run; see HANDOFF.md)...")
            tmp_dir_6 = tempfile.mkdtemp(prefix="physio_review_resume_", dir=tmp_dir)

            def first_session_plot(self, *args, **kwargs):
                # Simulate the RA deleting the very first seeded peak before closing partway through.
                # (Only peak marks: a PPG session also has the bad_ppg label's placeholder since 2026-09-27.)
                first_peak = next(i for i, a in enumerate(self.annotations) if a["description"] == "peak_ppg")
                self.annotations.delete(first_peak)

            # Uses --channels ppg rather than ecg: this test is purely about the
            # resume mechanism (channel-agnostic), and ppg doesn't require a QRS
            # template, so it stays independent of the own-template-required
            # policy tested separately below. (It used rsp until RSP became a
            # segment channel on 2026-09-25.)
            with patch.object(mne.io.RawArray, "plot", first_session_plot):
                run_main(["--synthetic", "--duration", "30", "--run", "1", "--initials", "hlu",
                          "--out-dir", tmp_dir_6, "--stage", "2", "--channels", "ppg"])

            captured = {}

            def second_session_plot(self, *args, **kwargs):
                sfreq_here = self.info["sfreq"]
                captured["indices"] = sorted(int(round(ann["onset"] * sfreq_here)) for ann in self.annotations
                                             if ann["description"] == "peak_ppg")

            with patch.object(mne.io.RawArray, "plot", second_session_plot):
                run_main(["--synthetic", "--duration", "30", "--run", "1", "--initials", "hlu",
                          "--out-dir", tmp_dir_6, "--stage", "2", "--channels", "ppg"])

            saved_path = glob.glob(os.path.join(tmp_dir_6, "*_annotations_hlu.json"))[0]
            with open(saved_path) as f:
                saved_first_session = json.load(f)
            assert captured["indices"] == saved_first_session["channels"]["ppg"]["indices"], (
                "the second session's pre-seeded peaks should exactly match what the first "
                "session saved (i.e. reflect the deletion), not the original automated peaks"
            )
            print("   OK: second session was pre-seeded from the first session's saved "
                  "corrections, not re-seeded from the automated peak columns")

            print("10. processing_log.csv integration: a real (non-synthetic) --box-path run "
                  "logs 'claimed', then 'created QRS template' per run, then a generalized "
                  "'<channels> reviewed -- <status>' entry, each with the RA's comment (typed in the "
                  "template-preview / session-summary dialogs, no longer a terminal prompt) as Notes...")
            box_path = tempfile.mkdtemp(prefix="physio_review_log_", dir=tmp_dir)
            make_fake_real_subject(box_path, "042", "1", duration_sec=30)
            make_fake_real_subject(box_path, "042", "2", duration_sec=30)

            # append_processing_log_entry() also mirrors to LOCAL_SNAPSHOT_DIR, which
            # defaults to the REAL physioCorrection/ folder -- redirected here so this
            # fake box_path never overwrites the user's actual local snapshot file.
            snapshot_dir = tempfile.mkdtemp(prefix="physio_review_snapshot_", dir=tmp_dir)
            with patch.object(processing_log, "LOCAL_SNAPSHOT_DIR", snapshot_dir), \
                 patch.object(qrs_template, "show_template_preview", return_value=(True, "template comment")), \
                 patch.object(session_summary_gui, "show_session_summary",
                              session_summary_gui.headless_decision(comment="review comment", finished=True)), \
                 patch("builtins.input", side_effect=AssertionError("no terminal prompt expected")):
                run_main(["--box-path", box_path, "--subject", "042", "--run", "1",
                          "--initials", "hlu", "--channels", "ecg,eda", "--no-local-copy"])

            log_df = pd.read_csv(os.path.join(box_path, "processing_log.csv"),
                                  dtype=str, keep_default_na=False)
            assert list(log_df["Step"]) == [
                "claimed", "created QRS template", "created QRS template", "ecg, eda reviewed -- finished",
            ], f"unexpected Step sequence: {list(log_df['Step'])}"
            assert set(log_df.loc[log_df["Step"] == "created QRS template", "Run"]) == {"1", "2"}, (
                "Stage A builds a template for EVERY available run -- expect one logged row per run"
            )
            assert log_df.iloc[-1]["Step"] == "ecg, eda reviewed -- finished", (
                "Step B's log entry must list the actually-reviewed channels, not the legacy "
                "ECG-only wording -- a deliberate choice for this generalized tool"
            )
            assert log_df.iloc[-1]["Notes"] == "review comment"
            assert set(log_df.loc[log_df["Step"] == "created QRS template", "Notes"]) == {"template comment"}
            assert log_df.iloc[-1]["Run"] == "1"
            assert all(s.startswith("sub-") for s in log_df["Subject"]), "Subject must be normalized to sub-XXX"
            print(f"   OK: logged Steps {list(log_df['Step'])}, "
                  f"Notes {list(log_df['Notes'])}")

            print("10b. processing_log.csv gets a compact value-range summary in its own Value_ranges "
                  "column for any SEGMENT-mode channel actually reviewed (eda here), with the RA's "
                  "comment alone in Notes...")
            box_path_2b = tempfile.mkdtemp(prefix="physio_review_log2_", dir=tmp_dir)
            make_fake_real_subject(box_path_2b, "043", "1", duration_sec=30)
            snapshot_dir_2b = tempfile.mkdtemp(prefix="physio_review_snapshot2_", dir=tmp_dir)
            with patch.object(processing_log, "LOCAL_SNAPSHOT_DIR", snapshot_dir_2b), \
                 patch.object(session_summary_gui, "show_session_summary",
                              session_summary_gui.headless_decision(comment="eda review comment")):
                run_main(["--box-path", box_path_2b, "--subject", "043", "--run", "1",
                          "--initials", "hlu", "--channels", "eda", "--no-local-copy"])
            log_df_2b = pd.read_csv(os.path.join(box_path_2b, "processing_log.csv"),
                                     dtype=str, keep_default_na=False)
            notes_2b = log_df_2b.iloc[-1]["Notes"]
            ranges_2b = log_df_2b.iloc[-1]["Value_ranges"]
            # Value ranges moved to their own column at the user's request.
            assert notes_2b == "eda review comment", notes_2b
            assert ranges_2b.startswith("EDA:") and "uS" in ranges_2b and "µ" not in ranges_2b, ranges_2b
            print(f"   OK: Notes = {notes_2b!r}; Value_ranges = {ranges_2b!r}")

            print("11. parse_args_or_show_gui() correctly threads the GUI's Step 3 result "
                  "dict ('compare_initials', no 'initials' key) into --stage 3/--compare-initials "
                  "argv, in both synthetic and real mode...")
            with patch.object(sys, "argv", ["physio_review.py"]), \
                 patch("simple_gui.collect_inputs_via_gui", return_value={
                     "synthetic": True, "stage": "3", "channel": "ecg", "compare_initials": "aaa,bbb",
                 }):
                parsed = physio_review.parse_args_or_show_gui()
            assert parsed.stage == "3" and parsed.compare_initials == "aaa,bbb" and parsed.synthetic
            print(f"   OK (synthetic): stage={parsed.stage}, compare_initials={parsed.compare_initials}")

            with patch.object(sys, "argv", ["physio_review.py"]), \
                 patch("simple_gui.collect_inputs_via_gui", return_value={
                     "synthetic": False, "box_path": box_path, "local_path": "/some/local/path",
                     "subject": "042", "run": "1", "stage": "3", "channel": "ecg",
                     "compare_initials": "hlu,xyz",
                 }):
                parsed = physio_review.parse_args_or_show_gui()
            assert parsed.stage == "3" and parsed.compare_initials == "hlu,xyz"
            assert parsed.box_path == box_path and parsed.subject == "042" and parsed.run == "1"
            assert parsed.local_path == "/some/local/path"
            print(f"   OK (real): box_path/local_path/subject/run threaded through alongside compare_initials")

            print("11b. parse_args_or_show_gui() echoes the equivalent CLI command to the terminal "
                  "(so an RA can copy/paste it, editing --run, to redo a similar session for the "
                  "other run without reopening the form each time)...")
            with patch.object(sys, "argv", ["physio_review.py"]), \
                 patch("simple_gui.collect_inputs_via_gui", return_value={
                     "synthetic": False, "box_path": box_path, "local_path": "/some/local/path",
                     "subject": "042", "run": "1", "stage": "2", "channel": "ecg,rsp",
                     "initials": "hlu", "ecg_source": "auto",
                 }):
                captured_echo = io.StringIO()
                with redirect_stdout(captured_echo):
                    parsed = physio_review.parse_args_or_show_gui()
            echo_output = captured_echo.getvalue()
            assert "Equivalent command" in echo_output
            command_line = [l for l in echo_output.splitlines() if "physio_review.py" in l and "python" in l.lower()][0]
            assert '"--run" "1"' in command_line and '"--channels" "ecg,rsp"' in command_line and '"--subject" "042"' in command_line, (
                f"expected the echoed command to contain the GUI's actual values, each individually "
                f"quoted; got: {command_line!r}"
            )
            print(f"   OK: {command_line.strip()}")

            print("11c. ...and every value is quoted, even one with NO whitespace, so a Windows path's "
                  "backslashes survive a copy/paste into bash/Git Bash -- an unquoted backslash before "
                  "an ordinary letter is silently dropped there (regression check: confirmed live -- an "
                  "RA's pasted --box-path lost every backslash and the tool then looked in a nonexistent "
                  "path; see HANDOFF.md)...")
            assert "\\" in box_path, "test fixture must actually contain backslashes for this check to mean anything"
            assert f'"{box_path}"' in command_line, (
                f"expected box_path fully quoted (backslashes intact) in the echoed command; got: {command_line!r}"
            )
            print(f"   OK: box_path ({box_path!r}) appears fully quoted in the echoed command")

            print("12. Full local-copy lifecycle (box_sync.py integration): --local-path copies the "
                  "subject down, the session works against that LOCAL copy, and on success the new "
                  "output is pushed back to --box-path and the local copy is cleaned up by default...")
            box_path_2 = tempfile.mkdtemp(prefix="physio_review_boxsync_box_", dir=tmp_dir)
            local_path_2 = tempfile.mkdtemp(prefix="physio_review_boxsync_local_", dir=tmp_dir)
            make_fake_real_subject(box_path_2, "777", "1", duration_sec=30)

            with patch.object(processing_log, "LOCAL_SNAPSHOT_DIR", snapshot_dir), \
                 patch("builtins.input", return_value=""):
                run_main(["--box-path", box_path_2, "--subject", "777", "--run", "1",
                          "--initials", "hlu", "--channels", "ecg", "--stage", "2",
                          "--ecg-source", "batch", "--local-path", local_path_2])

            pushed_path = os.path.join(box_path_2, "sub-777", "ses-run1", "beh",
                                        "sub-777_ses-run1_task-sdi_annotations_hlu.json")
            assert os.path.exists(pushed_path), "the new annotations file should have been pushed back to Box"
            assert not os.path.exists(os.path.join(local_path_2, "sub-777")), (
                "the local copy's sub-777 subfolder should be cleaned up by default after a "
                "successful push"
            )
            assert os.path.isdir(local_path_2), "the reusable --local-path folder itself must survive"
            log_df_2 = pd.read_csv(os.path.join(box_path_2, "processing_log.csv"),
                                    dtype=str, keep_default_na=False)
            assert list(log_df_2["Step"]) == ["claimed", "ecg reviewed -- not finished"], list(log_df_2["Step"])
            print(f"   OK: pushed to {os.path.basename(pushed_path)}, local sub-777 cleaned up, "
                  f"logged directly to {os.path.basename(box_path_2)} (never local_path)")

            print("13. --keep-local preserves the local copy even after a fully successful push...")
            box_path_3 = tempfile.mkdtemp(prefix="physio_review_boxsync_box2_", dir=tmp_dir)
            local_path_3 = tempfile.mkdtemp(prefix="physio_review_boxsync_local2_", dir=tmp_dir)
            make_fake_real_subject(box_path_3, "778", "1", duration_sec=30)

            with patch.object(processing_log, "LOCAL_SNAPSHOT_DIR", snapshot_dir), \
                 patch("builtins.input", return_value=""):
                run_main(["--box-path", box_path_3, "--subject", "778", "--run", "1",
                          "--initials", "hlu", "--channels", "ecg", "--stage", "2",
                          "--ecg-source", "batch", "--local-path", local_path_3, "--keep-local"])

            assert os.path.exists(os.path.join(box_path_3, "sub-778", "ses-run1", "beh",
                                                "sub-778_ses-run1_task-sdi_annotations_hlu.json"))
            assert os.path.isdir(os.path.join(local_path_3, "sub-778")), (
                "--keep-local should preserve the local copy despite a fully successful push"
            )
            print("   OK: local copy preserved under --keep-local")

            print("14. A failed push confirmation preserves the ENTIRE local copy and warns clearly "
                  "-- nothing is deleted when Box can't be confirmed to have the new file...")
            box_path_4 = tempfile.mkdtemp(prefix="physio_review_boxsync_box3_", dir=tmp_dir)
            local_path_4 = tempfile.mkdtemp(prefix="physio_review_boxsync_local3_", dir=tmp_dir)
            make_fake_real_subject(box_path_4, "779", "1", duration_sec=30)

            with patch.object(processing_log, "LOCAL_SNAPSHOT_DIR", snapshot_dir), \
                 patch("builtins.input", return_value=""), \
                 patch.object(physio_review, "push_changed_files_to_box", return_value=([], ["some/file.json"])):
                run_main(["--box-path", box_path_4, "--subject", "779", "--run", "1",
                          "--initials", "hlu", "--channels", "ecg", "--stage", "2",
                          "--ecg-source", "batch", "--local-path", local_path_4])

            assert os.path.isdir(os.path.join(local_path_4, "sub-779")), (
                "a failed confirmation must preserve the ENTIRE local copy, never delete on failure"
            )
            print("   OK: local copy preserved after a simulated confirmation failure")

            print("15. Per-RA template ownership (combined flow): a SECOND RA reviewing a subject "
                  "someone else already built a template for must build their OWN, never borrow "
                  "the first RA's -- a deliberate reliability choice (sharing a seed would bias "
                  "independent reviews toward agreement, or a shared blind spot, before either RA "
                  "has started; see HANDOFF.md)...")
            tmp_dir_7 = tempfile.mkdtemp(prefix="physio_review_per_ra_", dir=tmp_dir)
            with patch.object(physio_review, "run_stage_a", wraps=physio_review.run_stage_a) as spy_a:
                run_main(["--synthetic", "--duration", "30", "--run", "1", "--initials", "hlu",
                          "--out-dir", tmp_dir_7])
                assert spy_a.called, "hlu should have built a template (none existed yet)"
            with patch.object(physio_review, "run_stage_a", wraps=physio_review.run_stage_a) as spy_a:
                run_main(["--synthetic", "--duration", "30", "--run", "1", "--initials", "rst",
                          "--out-dir", tmp_dir_7])
                assert spy_a.called, (
                    "rst should ALSO have built their own template, even though hlu's already "
                    "exists -- it must never be silently borrowed"
                )
            per_ra_templates = sorted(glob.glob(os.path.join(tmp_dir_7, "*_ecg_corrected_qrs_*.json")))
            per_ra_initials = sorted(os.path.basename(p).rsplit("_", 1)[1].replace(".json", "")
                                      for p in per_ra_templates if "run1" in p)
            assert per_ra_initials == ["hlu", "rst"], (
                f"expected separate run-1 templates for both hlu and rst, got {per_ra_initials}"
            )
            print(f"   OK: both hlu and rst built their own separate run-1 template "
                  f"({[os.path.basename(p) for p in per_ra_templates if 'run1' in p]})")

            print("16. Per-RA template ownership (--stage 2): reviewing ECG without your OWN template "
                  "errors clearly, even when another RA's template for this exact subject/run already "
                  "exists -- no silent fallback to batch peaks, no silent borrowing...")
            # main() now prints the missing-template message itself (no traceback)
            # and exits non-zero, so the tracker reminder can be the last line.
            captured_16 = io.StringIO()
            try:
                with redirect_stdout(captured_16):
                    run_main(["--synthetic", "--duration", "30", "--run", "1", "--initials", "xyz",
                              "--out-dir", tmp_dir_7, "--stage", "2", "--channels", "ecg"])
                raise AssertionError("expected a non-zero exit for xyz's missing own template")
            except SystemExit as e:
                assert e.code == 1, e.code
            exit_message_2 = captured_16.getvalue()
            assert "doesn't exist" in exit_message_2 and "xyz" in exit_message_2, exit_message_2
            print(f"   OK: {exit_message_2}")

            print("17. --channels ppg (no ecg) under --stage 2 needs no template at all, even for an "
                  "RA with no template of their own and no --ecg-source override -- the own-template "
                  "requirement is scoped to sessions that actually review ECG...")
            with patch.object(physio_review, "run_stage_a") as spy_a:
                run_main(["--synthetic", "--duration", "30", "--run", "1", "--initials", "xyz",
                          "--out-dir", tmp_dir_7, "--stage", "2", "--channels", "ppg"])
                assert not spy_a.called
            xyz_annotations = glob.glob(os.path.join(tmp_dir_7, "*_annotations_xyz.json"))
            assert len(xyz_annotations) == 1
            print("   OK: xyz reviewed ppg with no template and no error")

            print("18. Session log (HLU, 2026-10-03): every session is appended to <stem>_annotations_<initials>.log "
                  "next to the annotation file, with a header and a footer; a crash still leaves its log...")
            log_dir = tempfile.mkdtemp(dir=tmp_dir)
            base = ["--synthetic", "--duration", "30", "--run", "1", "--initials", "lgx", "--out-dir", log_dir,
                    "--stage", "2", "--channels", "rsp"]
            with contextlib.redirect_stdout(io.StringIO()):
                run_main(base)
                run_main(base)
            ann = glob.glob(os.path.join(log_dir, "*_annotations_lgx.json"))
            logs = glob.glob(os.path.join(log_dir, "*_annotations_lgx.log"))
            assert len(ann) == 1 and logs == [ann[0][:-len(".json")] + ".log"], (ann, logs)
            text = open(logs[0], encoding="utf-8").read()
            assert text.count("PEBBL session started") == 2 and text.count("Session ended") == 2, text[:400]
            assert "-- lgx" in text and "Step 2 (review), run 1" in text
            assert "Starting Step 2 (review) for run 1" in text, "printed lines are captured"
            assert "Outcome: saved" in text
            assert "--- Session summary (shown in the dialog) ---" in text and "Choice: Save" in text, \
                "the dialog's contents and the RA's choice are logged (log only)"
            with patch.object(physio_review, "run_stage_b", side_effect=RuntimeError("boom")), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                try:
                    run_main(base)
                    raise AssertionError("the crash should end the session")
                except SystemExit:
                    pass
            text = open(logs[0], encoding="utf-8").read()
            assert text.count("PEBBL session started") == 3 and "RuntimeError: boom" in text, text[-600:]
            assert "Outcome: ended early" in text
            assert sys.stdout is sys.__stdout__ or not isinstance(sys.stdout, session_log._Tee), "console restored"
            with contextlib.redirect_stdout(io.StringIO()):
                run_main(["--synthetic", "--duration", "30", "--initials", "lgx", "--out-dir", log_dir, "--stage", "1"])
            step1_logs = sorted(glob.glob(os.path.join(log_dir, "*_annotations_lgx.log")))
            assert len(step1_logs) == 2, step1_logs
            assert all("Step 1 (ECG template)" in open(p, encoding="utf-8").read() for p in step1_logs)
            print("   OK: one .log per RA per run beside the .json; sessions appended with header and footer; "
                  "printed lines and a crash's traceback captured; Step 1 logs every run")

            print("19. Certification practice (HLU, 2026-10-04): --practice on the practice folder; Step 1 isn't "
                  "scored; a saved Step 2 review is scored against the key, the report is shown, and 'Compare' "
                  "opens the key view; Step 3 is refused; no key -> a message, not a crash...")
            practice_root = os.path.join(tmp_dir, "practice")
            with contextlib.redirect_stdout(io.StringIO()):
                practice_certification.make(practice_root, seed=7)
            key = practice_certification.load_key(practice_certification.key_path_for(practice_root))
            snapshot_dir = tempfile.mkdtemp(prefix="physio_review_snapshot_", dir=tmp_dir)
            practice = ["--box-path", practice_root, "--subject", "990", "--initials", "xyz", "--no-local-copy",
                        "--practice"]
            reports, comparisons = [], []

            def fake_report(heading, lines, passed):
                reports.append((heading, list(lines), passed))
                return True  # the trainee clicks "Compare with the answer key"

            with patch.object(processing_log, "LOCAL_SNAPSHOT_DIR", snapshot_dir), \
                 patch.object(practice_report_gui, "show_practice_report", side_effect=fake_report), \
                 patch.object(practice_certification, "show_comparison",
                              side_effect=lambda *a, **k: comparisons.append((a, k))), \
                 patch("builtins.input", side_effect=AssertionError("no terminal prompt expected")):
                with contextlib.redirect_stdout(io.StringIO()):
                    run_main(practice + ["--stage", "1"])
                assert reports == [], "Step 1 is not scored"
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    run_main(practice + ["--stage", "2", "--run", "1", "--channels", "ecg,rsp"])
            assert len(reports) == 1, reports
            heading, lines, passed = reports[0]
            assert "990" in heading and "run 1" in heading and "ecg" in heading and "rsp" in heading, heading
            assert passed is False and lines[-1].startswith("OVERALL: NOT YET"), \
                "the untouched seed has planted errors, so it can't pass"
            assert {l.split(":")[0] for l in lines if l.endswith(("PASS", "NOT YET"))} >= \
                {"ECG", "ECG bad stretches", "RSP"}, "the reviewed channels plus ECG's bad stretches are scored"
            assert "RSP: NOT YET" in lines, "the planted RSP errors are still there"
            assert "OVERALL: NOT YET" in out.getvalue(), "the report is printed too, so the session log keeps it"
            assert len(comparisons) == 1
            args_c = comparisons[0][0]
            assert args_c[2] == key["runs"]["1"] and set(args_c[4]) == {"ecg", "rsp"} and args_c[5] == "xyz"
            assert os.listdir(snapshot_dir) == [], "practice never touches the local snapshot of the real log"
            assert os.path.exists(os.path.join(practice_root, "processing_log.csv")), \
                "practice sessions are logged in the practice folder"

            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    run_main(practice + ["--stage", "3", "--channel", "ecg", "--compare-initials", "xyz,abc"])
                    raise AssertionError("Step 3 should be refused in practice")
                except SystemExit as e:
                    assert "isn't part of certification practice" in str(e), e

            os.rename(practice_certification.key_path_for(practice_root),
                      practice_certification.key_path_for(practice_root) + ".hidden")
            reports.clear()
            out = io.StringIO()
            with patch.object(processing_log, "LOCAL_SNAPSHOT_DIR", snapshot_dir), \
                 patch.object(practice_report_gui, "show_practice_report", side_effect=fake_report), \
                 contextlib.redirect_stdout(out):
                run_main(practice + ["--stage", "2", "--run", "2", "--channels", "eda"])
            assert reports == [] and "no answer key" in out.getvalue(), out.getvalue()[-500:]

            with patch.object(sys, "argv", ["physio_review.py"]), \
                 patch("simple_gui.collect_inputs_via_gui", return_value={
                     "synthetic": False, "box_path": practice_root, "local_path": "", "no_local_copy": True,
                     "subject": "990", "run": "1", "stage": "2", "channel": "ecg", "initials": "xyz",
                     "ecg_source": "auto", "practice": True,
                 }), contextlib.redirect_stdout(io.StringIO()):
                parsed = physio_review.parse_args_or_show_gui()
            assert parsed.practice and parsed.box_path == practice_root and parsed.subject == "990"
            print("   OK: scored after the Step 2 save (NOT YET on the untouched seed), report printed and shown, "
                  "compare opens the run-1 key view; Step 3 refused; missing key reported; the form's box "
                  "becomes --practice; the real log's local snapshot untouched")

        print("\nALL PHYSIO_REVIEW ORCHESTRATION TESTS PASSED.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
