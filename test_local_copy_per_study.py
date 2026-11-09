#!/usr/bin/env python3
"""
Tests the local working folder per study (HLU, 2026-11-08; pebbl-private
pebbl_local-copy-per-study-plan_20261108-1503.md): each session's local copy goes to the study's own folder inside
the RA's local working folder (<local>/<task>, or <local>/<task>-practice for certification practice), so a leftover
copy from one study is never checked against, or pushed into, another study's data folder. A copy left in the old
place (<local>/sub-XXX) is sorted by the task labels in its file names.

Synthetic files in temporary folders only: two data folders, "sdi" (task sdi) and "urglab" (task urglab), each with
a sub-005, and one shared local working folder.

  1. study_local_root and task_labels_in_local_copy.
  2. A urglab leftover with an unpushed review, then an sdi session for sub-005 (a whole Step 2 session): the sdi
     session works in <local>/sdi, never sees the urglab file, and sdi's folder never receives it; the urglab
     leftover is untouched, and the next urglab session offers it and sends it to urglab's folder.
  3. Negative control: the same leftover in the OLD place, checked the old way against sdi's folder, is listed as
     sdi's unpushed work (the misfiling the change prevents).
  4. Old place, this study's copy with an unpushed review: offered; "y" sends it to this study's folder, removes the
     old copy, and the session carries on in <local>/sdi; "n" stops with nothing changed.
  5. Old place, another study's copy: stops before copying anything; both data folders and the old copy are
     byte-identical before and after.
  6. Old place: only physioProcess's files (another study's) are removed and the session carries on; another study's
     files mixed with this study's, or files with no task label, stop; an empty folder is left alone; this study's
     copy with nothing unpushed is removed without a question.
  7. Practice: sdi practice and urglab practice for sub-990 use separate folders, and neither's leftover reaches the
     other's practice folder.
  8. A data folder as the local working folder stops the session before anything is copied or removed (negative
     control: without that check, the old-place step would remove the data folder's own sub-XXX).

Run with: annotate_env\\Scripts\\python.exe test_local_copy_per_study.py
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

import box_sync
import physio_review
import processing_log
import qrs_template
import session_summary_gui
import testdata
from box_sync import (find_unpushed_local_files, study_local_root, task_labels_in_local_copy)
from physio_io import generate_synthetic_demo

# The session-summary dialog is modal; answer "Save" immediately so nothing blocks.
session_summary_gui.show_session_summary = session_summary_gui.headless_decision()

URGLAB_PROFILE = 'schema_version = 1\nname = "urglab-like"\ntask = "urglab"\n'
COPY_PROMPT = "Copy these file(s) to the data folder now"


def run_main(argv):
    with patch.object(sys, "argv", ["physio_review.py"] + argv):
        physio_review.main()


def beh(folder, subject, run="1"):
    return os.path.join(folder, f"sub-{subject}", f"ses-run{run}", "beh")


def make_subject(folder, subject, task, run="1", duration_sec=20):
    """A recording and sidecar named for `task` (what physioProcess writes), in folder/sub-<subject>/..."""
    df, sfreq = generate_synthetic_demo(duration_sec=duration_sec)
    stem = os.path.join(beh(folder, subject, run), f"sub-{subject}_ses-run{run}_task-{task}")
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    df.to_csv(stem + "_physio.tsv.gz", sep="\t", index=False, header=False, compression="gzip")
    with open(stem + "_physio.json", "w") as f:
        json.dump({"Columns": list(df.columns), "SamplingFrequency": sfreq}, f)


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def tree(folder):
    """{relpath: (bytes, mtime)} for every file under folder: equal before and after means nothing changed."""
    found = {}
    for dirpath, _dirnames, filenames in os.walk(folder):
        for name in filenames:
            path = os.path.join(dirpath, name)
            with open(path, "rb") as f:
                found[os.path.relpath(path, folder)] = (f.read(), os.path.getmtime(path))
    return found


def all_names(folder):
    return [name for _d, _s, names in os.walk(folder) for name in names]


class Answers:
    """Stands in for input(): answers the leftover question with `copy_answer`, any other prompt with ""."""

    def __init__(self, copy_answer=None):
        self.copy_answer = copy_answer
        self.asked = []

    def __call__(self, prompt=""):
        if COPY_PROMPT in prompt:
            self.asked.append(prompt)
            if self.copy_answer is None:
                raise AssertionError("the leftover question should not have been asked")
            return self.copy_answer
        return ""


def stub_session(seen, saved_name, subject="005"):
    """In place of _run_session: records where the session works and saves one review file there."""
    def run(args, real_box_path, info):
        seen["box_path"] = args.box_path
        write(os.path.join(beh(args.box_path, subject), saved_name), '{"saved": "this session"}')
        info.update(stage="2", run="1", channels=["ecg"], saved=True, finished=False)
    return run


def not_called(args, real_box_path, info):
    raise AssertionError("the session should have stopped before starting")


@contextlib.contextmanager
def setup():
    tmp = tempfile.mkdtemp(prefix="pebbl_local_per_study_")
    try:
        sdi, urg, local = (os.path.join(tmp, name) for name in ("sdi_data", "urglab_data", "PhysioWorking"))
        for folder in (sdi, urg, local):
            os.makedirs(folder)
        testdata.sdi_profile(sdi)
        write(os.path.join(urg, "pebbl_study.toml"), URGLAB_PROFILE)
        make_subject(sdi, "005", "sdi")
        make_subject(urg, "005", "urglab")
        with patch.object(processing_log, "LOCAL_SNAPSHOT_DIR", tmp):
            yield tmp, sdi, urg, local
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_quietly(argv, answers, session=None):
    """run_main with input() answered; returns (exit code, printed text)."""
    out = io.StringIO()
    patches = [patch("builtins.input", answers), contextlib.redirect_stdout(out)]
    if session is not None:
        patches.append(patch.object(physio_review, "_run_session", session))
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        try:
            run_main(argv)
            code = 0
        except SystemExit as e:
            code = e.code
    return code, out.getvalue()


def sdi_argv(sdi, local, extra=()):
    return ["--box-path", sdi, "--subject", "005", "--run", "1", "--initials", "hlu", "--channels", "ecg",
            "--stage", "2", "--ecg-source", "batch", "--local-path", local, *extra]


def urglab_leftover(urg, local_root):
    """A urglab copy of sub-005 in local_root, with a review that never reached urglab's folder."""
    box_sync.copy_subject_tree_to_local(urg, "005", local_root)
    review = os.path.join(beh(local_root, "005"), "sub-005_ses-run1_task-urglab_annotations_ab.json")
    write(review, '{"review": "urglab, never copied back"}')
    return os.path.relpath(review, local_root)


def test_helpers():
    print("1. study_local_root gives <local>/<task> and <local>/<task>-practice, and refuses a label that isn't "
          "letters and digits; task_labels_in_local_copy reads the labels in the copy's file names...")
    local = os.path.join("C:", os.sep, "PhysioWorking")
    assert study_local_root(local, "sdi") == os.path.join(local, "sdi")
    assert study_local_root(local, "sdi", practice=True) == os.path.join(local, "sdi-practice")
    assert study_local_root(local, "urglab") != study_local_root(local, "sdi")
    for bad in ("", "sub-005", "sdi-practice", "..", "a/b", None):
        try:
            study_local_root(local, bad)
            raise AssertionError(f"{bad!r} should be refused")
        except ValueError:
            pass
    tmp = tempfile.mkdtemp()
    try:
        assert task_labels_in_local_copy(tmp, "005") == set(), "no copy, no labels"
        write(os.path.join(beh(tmp, "005"), "sub-005_ses-run1_task-sdi_physio.json"), "{}")
        write(os.path.join(beh(tmp, "005"), "backups", "sub-005_ses-run1_task-urglab_annotations_ab_20261108_1200.json"),
              "{}")
        write(os.path.join(beh(tmp, "005"), "notes.txt"), "x")
        assert task_labels_in_local_copy(tmp, "005") == {"sdi", "urglab"}, task_labels_in_local_copy(tmp, "005")
        assert len(box_sync.local_copy_files(tmp, "005")) == 3
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("   OK")


def test_two_studies_end_to_end():
    print("2. A urglab leftover with an unpushed review, then a whole sdi Step 2 session for sub-005: the sdi "
          "session works in <local>/sdi and sdi's folder never receives the urglab file; the next urglab session "
          "offers it and sends it to urglab's folder...")
    with setup() as (_tmp, sdi, urg, local), \
            patch.object(mne.io.RawArray, "plot", return_value=None), \
            patch.object(qrs_template, "show_template_preview", return_value=True):
        urg_root = study_local_root(local, "urglab")
        review = urglab_leftover(urg, urg_root)
        leftover_before = tree(urg_root)
        urg_before = tree(urg)

        answers = Answers()  # the leftover question must not come up for sdi
        code, out = run_quietly(sdi_argv(sdi, local), answers)
        assert code == 0, (code, out[-2000:])
        assert not answers.asked, answers.asked
        saved = os.path.join(beh(sdi, "005"), "sub-005_ses-run1_task-sdi_annotations_hlu.json")
        assert os.path.exists(saved), "the sdi review should be in sdi's folder"
        assert not [n for n in all_names(sdi) if "task-urglab" in n], all_names(sdi)
        assert f"to {study_local_root(local, 'sdi')} ..." in out, "the copy goes to <local>/sdi"
        assert not os.path.exists(os.path.join(local, "sdi", "sub-005")), "sdi's copy is removed after the push"
        assert tree(urg_root) == leftover_before, "the urglab leftover is untouched"
        assert tree(urg) == urg_before, "urglab's folder is untouched"
        assert sorted(os.listdir(local)) == ["sdi", "urglab"], os.listdir(local)
        print("   OK: sdi session saved, no question, no task-urglab file in sdi's folder, urglab leftover untouched")

        seen = {}
        answers = Answers("y")
        code, out = run_quietly(["--box-path", urg, "--local-path", local, "--subject", "005", "--run", "1",
                                 "--initials", "ab", "--stage", "2"], answers,
                                stub_session(seen, "sub-005_ses-run1_task-urglab_annotations_ab2.json"))
        assert code == 0, (code, out[-2000:])
        assert len(answers.asked) == 1 and review in out, out[-2000:]
        with open(os.path.join(urg, review)) as f:
            assert "never copied back" in f.read(), "the leftover review now reaches urglab's folder"
        assert seen["box_path"] == urg_root
        assert not [n for n in all_names(sdi) if "task-urglab" in n]
        print("   OK: the urglab session offered the leftover and sent it to urglab's folder")


def test_negative_control():
    print("3. Negative control: the same leftover in the OLD place (<local>/sub-005), checked the old way against "
          "sdi's folder, is listed as sdi's unpushed work; in urglab's own folder inside <local> it isn't...")
    with setup() as (_tmp, sdi, urg, local):
        review = urglab_leftover(urg, local)  # the old place
        assert find_unpushed_local_files(sdi, "005", local) == [review], find_unpushed_local_files(sdi, "005", local)
        shutil.rmtree(os.path.join(local, "sub-005"))
        urglab_leftover(urg, study_local_root(local, "urglab"))
        assert find_unpushed_local_files(sdi, "005", study_local_root(local, "sdi")) == []
    print("   OK: the old way lists the urglab review against sdi's folder; the study folders keep them apart")


def test_old_place_same_study():
    print("4. Old place, this study's copy with an unpushed review: 'n' stops with nothing changed; 'y' sends it "
          "to sdi's folder, removes the old copy, and the session carries on in <local>/sdi...")
    with setup() as (_tmp, sdi, urg, local):
        box_sync.copy_subject_tree_to_local(sdi, "005", local)
        rel = os.path.relpath(os.path.join(beh(local, "005"), "sub-005_ses-run1_task-sdi_annotations_zz.json"), local)
        write(os.path.join(local, rel), '{"review": "sdi, never copied back"}')
        before = (tree(sdi), tree(urg), tree(local))

        answers = Answers("n")
        code, out = run_quietly(sdi_argv(sdi, local), answers, not_called)
        assert code == 1 and len(answers.asked) == 1 and rel in out, (code, out[-2000:])
        assert "Stopped without changing anything" in out
        assert (tree(sdi), tree(urg), tree(local)) == before, "'n' must change nothing"
        print("   OK: 'n' stopped, nothing changed")

        seen = {}
        answers = Answers("y")
        code, out = run_quietly(sdi_argv(sdi, local), answers,
                                stub_session(seen, "sub-005_ses-run1_task-sdi_annotations_hlu.json"))
        assert code == 0, (code, out[-2000:])
        with open(os.path.join(sdi, rel)) as f:
            assert "never copied back" in f.read(), "'y' sends the old review to sdi's folder"
        old_dir = os.path.join(local, "sub-005")
        assert f"Removed the old local copy in {old_dir} (everything in it is in the data folder)." in out, out
        assert not os.path.exists(old_dir)
        assert seen["box_path"] == study_local_root(local, "sdi"), seen
        assert os.path.exists(os.path.join(beh(sdi, "005"), "sub-005_ses-run1_task-sdi_annotations_hlu.json"))
        assert tree(urg) == before[1]
        print("   OK: 'y' pushed it, removed the old copy, and the session worked in <local>/sdi")


def test_old_place_other_study():
    print("5. Old place, another study's copy: the sdi session stops before copying anything; both data folders "
          "and the old copy are byte-identical before and after...")
    with setup() as (_tmp, sdi, urg, local):
        urglab_leftover(urg, local)
        before = (tree(sdi), tree(urg), tree(local))
        code, out = run_quietly(sdi_argv(sdi, local), Answers(), not_called)
        assert code == 1, (code, out[-2000:])
        old_dir = os.path.join(local, "sub-005")
        expected = (f"There's a local copy of sub-005 from another study (its files are named task-urglab) in "
                    f"{old_dir}. It's from before PEBBL kept each study in its own folder, and it may hold work "
                    f"that was never copied back. Nothing has been changed. Start a session for that study's "
                    f"sub-005 to send it back to that study's data folder, or ask the lab staff.")
        assert expected in out, out[-2000:]
        assert (tree(sdi), tree(urg), tree(local)) == before, "nothing may change"
        assert os.listdir(local) == ["sub-005"], "no study folder made, nothing copied"

        seen = {}
        answers = Answers("y")  # urglab's own session then handles it
        code, out = run_quietly(["--box-path", urg, "--local-path", local, "--subject", "005", "--run", "1",
                                 "--initials", "ab", "--stage", "2"], answers,
                                stub_session(seen, "sub-005_ses-run1_task-urglab_annotations_ab2.json"))
        assert code == 0 and len(answers.asked) == 1, (code, out[-2000:])
        assert os.path.exists(os.path.join(beh(urg, "005"), "sub-005_ses-run1_task-urglab_annotations_ab.json"))
        assert not os.path.exists(old_dir)
        assert tree(sdi) == before[0]
    print("   OK: stopped with the message, nothing changed; urglab's session then sent it to urglab's folder")


def test_old_place_other_cases():
    print("6. Old place: physioProcess's files only (another study's) are removed and the session carries on; "
          "mixed studies or no task label stop; an empty folder is left alone; this study's copy with nothing "
          "unpushed is removed without a question...")
    with setup() as (_tmp, sdi, urg, local):
        old_dir = os.path.join(local, "sub-005")
        quiet = io.StringIO

        box_sync.copy_subject_tree_to_local(urg, "005", local)  # urglab's recording and sidecar only
        with patch("builtins.input", Answers()), contextlib.redirect_stdout(quiet()) as out:
            physio_review._handle_old_local_copy(sdi, "005", local, "sdi")
        assert not os.path.exists(old_dir), "physioProcess's files are never RA work: removed"
        assert "it held only copies of recordings, which are never copied back" in out.getvalue(), out.getvalue()
        print("   OK: physioProcess-only copy removed")

        for files, wanted in (
                (["sub-005_ses-run1_task-urglab_annotations_ab.json", "sub-005_ses-run1_task-sdi_annotations_zz.json"],
                 "with files from more than one study (named task-sdi, task-urglab)"),
                (["notes.txt"], "PEBBL can't tell which study its files belong to"),
                # another study's recording beside a file with no label: not physioProcess's files only
                (["sub-005_ses-run1_task-urglab_physio.tsv.gz", "notes.txt"],
                 "from another study (its files are named task-urglab)")):
            for name in files:
                write(os.path.join(beh(local, "005"), name), "x")
            before = (tree(sdi), tree(local))
            with patch("builtins.input", Answers()), contextlib.redirect_stdout(quiet()):
                try:
                    physio_review._handle_old_local_copy(sdi, "005", local, "sdi")
                    raise AssertionError(f"{files} should stop the session")
                except SystemExit as e:
                    assert wanted in str(e.code) and "Nothing has been changed" in str(e.code), e.code
            assert (tree(sdi), tree(local)) == before
            shutil.rmtree(old_dir)
        print("   OK: mixed studies, no label, and another study's file beside an unlabeled one all stop, unchanged")

        os.makedirs(beh(local, "005"))
        with patch("builtins.input", Answers()), contextlib.redirect_stdout(quiet()) as out:
            physio_review._handle_old_local_copy(sdi, "005", local, "sdi")
        assert os.path.isdir(beh(local, "005")) and out.getvalue() == "", "an empty folder: nothing to do"
        shutil.rmtree(old_dir)

        box_sync.copy_subject_tree_to_local(sdi, "005", local)
        stale = os.path.join(beh(local, "005"), "sub-005_ses-run1_task-sdi_annotations_zz.json")
        write(stale, "older")
        write(os.path.join(sdi, os.path.relpath(stale, local)), "newer")  # the data folder's copy is newer
        past = time.time() - 3600
        os.utime(stale, (past, past))
        with patch("builtins.input", Answers()), contextlib.redirect_stdout(quiet()) as out:
            physio_review._handle_old_local_copy(sdi, "005", local, "sdi")
        assert not os.path.exists(old_dir) and "everything in it is in the data folder" in out.getvalue()
        with open(os.path.join(sdi, os.path.relpath(stale, local))) as f:
            assert f.read() == "newer", "the data folder's newer copy is kept"
        print("   OK: empty folder left alone; this study's copy with nothing unpushed removed without a question")


def test_practice():
    print("7. Practice: sdi practice and urglab practice for sub-990 use <local>/sdi-practice and "
          "<local>/urglab-practice, and a urglab practice leftover never reaches sdi's practice folder...")
    with setup() as (tmp, sdi, urg, local):
        sdi_practice, urg_practice = os.path.join(tmp, "sdi_practice"), os.path.join(tmp, "urglab_practice")
        make_subject(sdi_practice, "990", "sdi")
        make_subject(urg_practice, "990", "urglab")
        urg_root = study_local_root(local, "urglab", practice=True)
        box_sync.copy_subject_tree_to_local(urg_practice, "990", urg_root)
        write(os.path.join(beh(urg_root, "990"), "sub-990_ses-run1_task-urglab_annotations_ab.json"), "unpushed")
        leftover_before = tree(urg_root)

        seen = {}
        code, out = run_quietly(["--box-path", sdi_practice, "--study-profile", os.path.join(sdi, "pebbl_study.toml"),
                                 "--practice", "--local-path", local, "--subject", "990", "--run", "1",
                                 "--initials", "hlu", "--stage", "2"], Answers(),
                                stub_session(seen, "sub-990_ses-run1_task-sdi_annotations_hlu.json", subject="990"))
        assert code == 0, (code, out[-2000:])
        assert seen["box_path"] == os.path.join(local, "sdi-practice"), seen
        assert os.path.exists(os.path.join(beh(sdi_practice, "990"), "sub-990_ses-run1_task-sdi_annotations_hlu.json"))
        assert not [n for n in all_names(sdi_practice) if "task-urglab" in n]
        assert tree(urg_root) == leftover_before
        assert sorted(os.listdir(local)) == ["sdi-practice", "urglab-practice"], os.listdir(local)
    print("   OK: separate practice folders; the urglab practice leftover stayed where it was")


def test_data_folder_as_local_folder():
    print("8. A data folder as the local working folder (this study's, or any folder with a pebbl_study.toml) "
          "stops before anything is copied or removed; negative control: without that check, the old-place step "
          "would remove the data folder's own sub-005...")
    with setup() as (_tmp, sdi, urg, local):
        before = (tree(sdi), tree(urg))
        for wrong in (sdi, urg):
            code, out = run_quietly(sdi_argv(sdi, wrong), Answers(), not_called)
            assert code == 1 and f"The local working folder ({wrong}) is a data folder." in out, (code, out[-2000:])
            assert (tree(sdi), tree(urg)) == before, "nothing may change"
        assert os.listdir(local) == []
        print("   OK: both stopped, nothing changed")

        with patch.object(physio_review, "_local_folder_problem", return_value=None), \
                patch("builtins.input", Answers()), contextlib.redirect_stdout(io.StringIO()):
            physio_review._handle_old_local_copy(sdi, "005", sdi, "sdi")
        assert not os.path.exists(os.path.join(sdi, "sub-005")), "the control should show the data being removed"
    print("   OK: without the check, sdi's own sub-005 was removed (in this test's temporary folder)")


def main():
    test_helpers()
    test_two_studies_end_to_end()
    test_negative_control()
    test_old_place_same_study()
    test_old_place_other_study()
    test_old_place_other_cases()
    test_practice()
    test_data_folder_as_local_folder()
    print("\nAll local-copy-per-study checks passed.")


if __name__ == "__main__":
    main()
