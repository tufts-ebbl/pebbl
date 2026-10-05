#!/usr/bin/env python3
"""
Tests ReviewLauncherDialog's LOGIC (field validation, synthetic-mode
toggle, Step 1/Step 2 toggle, channel lock, submit behavior) without
actually showing/clicking the dialog -- .exec() is never called, so
there's no blocking modal event loop. This can't verify what the dialog
looks like (needs a human), but it does verify every decision the form
makes given field values.

Run with: annotate_env\\Scripts\\python.exe test_simple_gui.py
"""

import os
import shutil
import tempfile

from PyQt6.QtWidgets import QApplication

from simple_gui import ReviewLauncherDialog

# Never read or write the real remembered settings (box path, initials) from tests.
ReviewLauncherDialog.USE_SAVED_SETTINGS = False


def main():
    # A QApplication must exist before constructing any QWidget, even though
    # we never call .exec() (no modal event loop, no blocking).
    app = QApplication.instance() or QApplication([])

    print("1. Empty initials -> rejected with an error, dialog not accepted...")
    dlg = ReviewLauncherDialog()
    dlg._on_submit()
    assert dlg.result_values is None
    assert "initials" in dlg.error_label.text().lower()
    print(f"   OK: {dlg.error_label.text()!r}")

    print("2. Synthetic mode checked + initials filled -> accepted, Step 2 (default) with ECG selected...")
    dlg = ReviewLauncherDialog()
    dlg.synthetic_checkbox.setChecked(True)
    dlg.initials_edit.setText("hlu")
    dlg._on_submit()
    assert dlg.result_values == {"synthetic": True, "initials": "hlu", "stage": "2", "channel": "ecg",
                                 "ppg_guide": False}
    print(f"   OK: {dlg.result_values}")

    print("3. Checking synthetic disables the real-file fields (box path/local path/subject/"
          "stage radios/local-copy checkboxes)...")
    dlg = ReviewLauncherDialog()
    assert dlg.box_path_edit.isEnabled() and dlg.subject_edit.isEnabled() and dlg.local_path_edit.isEnabled()
    assert dlg.step1_radio.isEnabled() and dlg.step2_radio.isEnabled()
    assert dlg.no_local_copy_checkbox.isEnabled() and dlg.keep_local_checkbox.isEnabled()
    dlg.synthetic_checkbox.setChecked(True)
    assert not dlg.box_path_edit.isEnabled()
    assert not dlg.subject_edit.isEnabled()
    assert not dlg.local_path_edit.isEnabled()
    assert not dlg.step1_radio.isEnabled() and not dlg.step2_radio.isEnabled()
    assert not dlg.no_local_copy_checkbox.isEnabled() and not dlg.keep_local_checkbox.isEnabled()
    dlg.synthetic_checkbox.setChecked(False)
    assert dlg.box_path_edit.isEnabled() and dlg.subject_edit.isEnabled() and dlg.local_path_edit.isEnabled()
    print("   OK: fields disable/re-enable correctly with the checkbox")

    print("4. Real mode with initials but no box path/subject -> rejected with an error...")
    dlg = ReviewLauncherDialog()
    dlg.initials_edit.setText("hlu")
    dlg._on_submit()
    assert dlg.result_values is None
    assert "box path" in dlg.error_label.text().lower() or "subject" in dlg.error_label.text().lower()
    print(f"   OK: {dlg.error_label.text()!r}")

    print("5. Real mode, Step 2, fully filled in -> accepted with the correct dict...")
    dlg = ReviewLauncherDialog()
    dlg.box_path_edit.setText(r"C:\Users\hlu\Box\DATA\Processed\physioProcessing\derivatives")
    dlg.subject_edit.setText("001")
    dlg.local_path_edit.setText(r"C:\Users\hlu\PhysioWorking")
    dlg.step2_radio.setChecked(True)
    dlg.run_combo.setCurrentText("2")
    channel_index = dlg._channel_keys.index("rsp")
    dlg.channel_combo.setCurrentIndex(channel_index)
    dlg.initials_edit.setText("hlu")
    dlg._on_submit()
    assert dlg.result_values == {
        "ppg_guide": True,
        "practice": False,
        "synthetic": False,
        "box_path": r"C:\Users\hlu\Box\DATA\Processed\physioProcessing\derivatives",
        "local_path": r"C:\Users\hlu\PhysioWorking",
        "subject": "001",
        "no_local_copy": False,
        "keep_local": False,
        "run": "2",
        "initials": "hlu",
        "stage": "2",
        "channel": "rsp",
        "ecg_source": "auto",
    }
    print(f"   OK: {dlg.result_values}")

    print("6. Selecting Step 1 locks the channel to ECG and disables run/channel fields...")
    dlg = ReviewLauncherDialog()
    dlg.step2_radio.setChecked(True)
    channel_index = dlg._channel_keys.index("eda")
    dlg.channel_combo.setCurrentIndex(channel_index)
    assert dlg.run_combo.isEnabled() and dlg.channel_combo.isEnabled()

    dlg.step1_radio.setChecked(True)
    assert not dlg.run_combo.isEnabled()
    assert not dlg.channel_combo.isEnabled()
    assert dlg.channel_combo.currentData() == "ecg", "Step 1 should auto-select ECG"
    print("   OK: Step 1 disabled run/channel and forced the channel to ECG")

    dlg.step2_radio.setChecked(True)
    assert dlg.run_combo.isEnabled() and dlg.channel_combo.isEnabled()
    print("   OK: switching back to Step 2 re-enables run/channel")

    print("7. Real mode, Step 1, fully filled in -> accepted with run=None, channel='ecg', "
          "template_start defaulting to 0.0...")
    dlg = ReviewLauncherDialog()
    dlg.box_path_edit.setText(r"C:\Users\hlu\Box\DATA\Processed\physioProcessing\derivatives")
    dlg.subject_edit.setText("001")
    dlg.local_path_edit.setText(r"C:\Users\hlu\PhysioWorking")
    dlg.step1_radio.setChecked(True)
    dlg.initials_edit.setText("hlu")
    assert dlg.template_start_edit.isEnabled(), "Step 1 should enable the template-start field"
    dlg._on_submit()
    assert dlg.result_values == {
        "ppg_guide": False,
        "practice": False,
        "synthetic": False,
        "box_path": r"C:\Users\hlu\Box\DATA\Processed\physioProcessing\derivatives",
        "local_path": r"C:\Users\hlu\PhysioWorking",
        "subject": "001",
        "no_local_copy": False,
        "keep_local": False,
        "run": None,
        "initials": "hlu",
        "stage": "1",
        "channel": "ecg",
        "template_start": 0.0,
        "template_run": "1",
        "template_window": 20.0,
    }
    print(f"   OK: {dlg.result_values}")

    print("7b. Step 1 with an explicit --template-start value...")
    dlg = ReviewLauncherDialog()
    dlg.box_path_edit.setText(r"C:\Users\hlu\Box\DATA\Processed\physioProcessing\derivatives")
    dlg.subject_edit.setText("001")
    dlg.local_path_edit.setText(r"C:\Users\hlu\PhysioWorking")
    dlg.step1_radio.setChecked(True)
    dlg.initials_edit.setText("hlu")
    dlg.template_start_edit.setText("20")
    dlg._on_submit()
    assert dlg.result_values["template_start"] == 20.0
    print(f"   OK: template_start=20.0 threaded through: {dlg.result_values}")

    print("7c. A non-numeric template-start is rejected with an error, dialog not accepted...")
    dlg = ReviewLauncherDialog()
    dlg.box_path_edit.setText(r"C:\Users\hlu\Box\DATA\Processed\physioProcessing\derivatives")
    dlg.subject_edit.setText("001")
    dlg.local_path_edit.setText(r"C:\Users\hlu\PhysioWorking")
    dlg.step1_radio.setChecked(True)
    dlg.initials_edit.setText("hlu")
    dlg.template_start_edit.setText("not-a-number")
    dlg._on_submit()
    assert dlg.result_values is None
    assert "number" in dlg.error_label.text().lower()
    print(f"   OK: {dlg.error_label.text()!r}")

    print("7d. Step 2 disables the template-start field...")
    dlg = ReviewLauncherDialog()
    dlg.step2_radio.setChecked(True)
    assert not dlg.template_start_edit.isEnabled()
    print("   OK: template-start disabled outside Step 1")

    print("7e. Advanced field enable/disable: --ecg-source only for Step 2, "
          "--template-run/--template-window only for Step 1...")
    dlg = ReviewLauncherDialog()  # Step 2 is checked by default
    assert dlg.ecg_source_combo.isEnabled()
    assert not dlg.template_run_combo.isEnabled() and not dlg.template_window_edit.isEnabled()
    dlg.step1_radio.setChecked(True)
    assert not dlg.ecg_source_combo.isEnabled()
    assert dlg.template_run_combo.isEnabled() and dlg.template_window_edit.isEnabled()
    dlg.step3_radio.setChecked(True)
    assert not dlg.ecg_source_combo.isEnabled()
    assert not dlg.template_run_combo.isEnabled() and not dlg.template_window_edit.isEnabled()
    print("   OK: ecg-source Step-2-only, template-run/window Step-1-only")

    print("7f. Real mode, Step 2, --ecg-source set to 'batch' -> threaded through as 'batch'...")
    dlg = ReviewLauncherDialog()
    dlg.box_path_edit.setText(r"C:\Users\hlu\Box\DATA\Processed\physioProcessing\derivatives")
    dlg.subject_edit.setText("001")
    dlg.local_path_edit.setText(r"C:\Users\hlu\PhysioWorking")
    dlg.initials_edit.setText("hlu")
    batch_index = dlg.ecg_source_combo.findData("batch")
    dlg.ecg_source_combo.setCurrentIndex(batch_index)
    dlg._on_submit()
    assert dlg.result_values["ecg_source"] == "batch"
    print(f"   OK: ecg_source='batch' threaded through: {dlg.result_values}")

    print("7g. Real mode, Step 1, custom --template-run/--template-window -> threaded through...")
    dlg = ReviewLauncherDialog()
    dlg.box_path_edit.setText(r"C:\Users\hlu\Box\DATA\Processed\physioProcessing\derivatives")
    dlg.subject_edit.setText("001")
    dlg.local_path_edit.setText(r"C:\Users\hlu\PhysioWorking")
    dlg.step1_radio.setChecked(True)
    dlg.initials_edit.setText("hlu")
    dlg.template_run_combo.setCurrentText("2")
    dlg.template_window_edit.setText("30")
    dlg._on_submit()
    assert dlg.result_values["template_run"] == "2"
    assert dlg.result_values["template_window"] == 30.0
    print(f"   OK: template_run='2', template_window=30.0 threaded through")

    print("7h. A non-numeric --template-window is rejected with an error, dialog not accepted...")
    dlg = ReviewLauncherDialog()
    dlg.box_path_edit.setText(r"C:\Users\hlu\Box\DATA\Processed\physioProcessing\derivatives")
    dlg.subject_edit.setText("001")
    dlg.local_path_edit.setText(r"C:\Users\hlu\PhysioWorking")
    dlg.step1_radio.setChecked(True)
    dlg.initials_edit.setText("hlu")
    dlg.template_window_edit.setText("not-a-number")
    dlg._on_submit()
    assert dlg.result_values is None
    assert "number" in dlg.error_label.text().lower()
    print(f"   OK: {dlg.error_label.text()!r}")

    print("7i. --no-local-copy: local working folder is no longer required, field disables, "
          "and no_local_copy=True threads through with an empty local_path...")
    dlg = ReviewLauncherDialog()
    dlg.box_path_edit.setText(r"C:\Users\hlu\Box\DATA\Processed\physioProcessing\derivatives")
    dlg.subject_edit.setText("001")
    dlg.initials_edit.setText("hlu")
    dlg.no_local_copy_checkbox.setChecked(True)
    assert not dlg.local_path_edit.isEnabled(), "local path should disable once --no-local-copy is checked"
    assert not dlg.keep_local_checkbox.isEnabled(), "keep-local is meaningless without a local copy"
    dlg._on_submit()
    assert dlg.result_values is not None, f"should not require local_path: {dlg.error_label.text()!r}"
    assert dlg.result_values["no_local_copy"] is True
    assert dlg.result_values["local_path"] == ""
    print(f"   OK: {dlg.result_values}")

    print("7j. --keep-local threads through as True when checked...")
    dlg = ReviewLauncherDialog()
    dlg.box_path_edit.setText(r"C:\Users\hlu\Box\DATA\Processed\physioProcessing\derivatives")
    dlg.subject_edit.setText("001")
    dlg.local_path_edit.setText(r"C:\Users\hlu\PhysioWorking")
    dlg.initials_edit.setText("hlu")
    dlg.keep_local_checkbox.setChecked(True)
    dlg._on_submit()
    assert dlg.result_values["keep_local"] is True
    print(f"   OK: keep_local=True threaded through")

    print("8. Whitespace-only fields are treated as empty...")
    dlg = ReviewLauncherDialog()
    dlg.initials_edit.setText("   ")
    dlg._on_submit()
    assert dlg.result_values is None
    print("   OK: whitespace-only initials rejected")

    print("9. Selecting Step 3 adds two reviewer-initials fields, keeps 'your initials' (now the "
          "RECONCILER, recorded in the reconciled file -- audit M2), and re-enables run/channel...")
    dlg = ReviewLauncherDialog()
    dlg.step3_radio.setChecked(True)
    assert dlg.initials_edit.isEnabled() and "reconciling" in dlg.initials_label.text().lower()
    assert dlg.compare_a_edit.isEnabled() and dlg.compare_b_edit.isEnabled()
    assert dlg.run_combo.isEnabled() and dlg.channel_combo.isEnabled(), "Step 3 needs run+channel, like Step 2"
    print("   OK: reconciler initials + compare-initials fields enabled, run/channel enabled")

    print("10. Real mode, Step 3, missing a comparison initial -> rejected with an error...")
    dlg.box_path_edit.setText(r"C:\Users\hlu\Box\DATA\Processed\physioProcessing\derivatives")
    dlg.subject_edit.setText("001")
    dlg.local_path_edit.setText(r"C:\Users\hlu\PhysioWorking")
    dlg.initials_edit.setText("hlu")
    dlg.compare_a_edit.setText("hlu")
    # compare_b left blank
    dlg._on_submit()
    assert dlg.result_values is None
    assert "both reviewers" in dlg.error_label.text().lower()
    print(f"   OK: {dlg.error_label.text()!r}")

    print("11. Real mode, Step 3, fully filled in -> accepted with compare_initials threaded "
          "through as 'A,B', local_path included, and the reconciler's initials...")
    dlg.compare_b_edit.setText("xyz")
    dlg.run_combo.setCurrentText("1")
    dlg._on_submit()
    assert dlg.result_values == {
        "ppg_guide": False,
        "practice": False,
        "synthetic": False,
        "box_path": r"C:\Users\hlu\Box\DATA\Processed\physioProcessing\derivatives",
        "local_path": r"C:\Users\hlu\PhysioWorking",
        "subject": "001",
        "no_local_copy": False,
        "keep_local": False,
        "run": "1",
        "stage": "3",
        "channel": "ecg",
        "compare_initials": "hlu,xyz",
        "initials": "hlu",
    }
    print(f"   OK: {dlg.result_values}")

    print("11b. Real mode, Step 3, missing local path -> rejected with an error...")
    dlg2 = ReviewLauncherDialog()
    dlg2.step3_radio.setChecked(True)
    dlg2.initials_edit.setText("hlu")
    dlg2.box_path_edit.setText(r"C:\Users\hlu\Box\DATA\Processed\physioProcessing\derivatives")
    dlg2.subject_edit.setText("001")
    dlg2.compare_a_edit.setText("hlu")
    dlg2.compare_b_edit.setText("xyz")
    # local_path left blank
    dlg2._on_submit()
    assert dlg2.result_values is None
    assert "local working folder" in dlg2.error_label.text().lower()
    print(f"   OK: {dlg2.error_label.text()!r}")

    print("12. Synthetic mode, Step 3 -> also works, no box-path/subject needed...")
    dlg = ReviewLauncherDialog()
    dlg.synthetic_checkbox.setChecked(True)
    dlg.step3_radio.setChecked(True)
    dlg.initials_edit.setText("ccc")
    dlg.compare_a_edit.setText("aaa")
    dlg.compare_b_edit.setText("bbb")
    dlg._on_submit()
    assert dlg.result_values == {
        "ppg_guide": False,
        "synthetic": True, "stage": "3", "channel": "ecg", "compare_initials": "aaa,bbb", "initials": "ccc",
    }
    print(f"   OK: {dlg.result_values}")

    print("12b. Initials are lower-cased and must be 2-4 letters; Step 3 rejects comparing a "
          "reviewer with themselves (audit H6)...")
    dlg = ReviewLauncherDialog()
    dlg.synthetic_checkbox.setChecked(True)
    dlg.initials_edit.setText("HLU")
    dlg._on_submit()
    assert dlg.result_values["initials"] == "hlu", dlg.result_values
    for bad in ("sub-003", "h", "hl u", "abcde"):
        dlg = ReviewLauncherDialog()
        dlg.synthetic_checkbox.setChecked(True)
        dlg.initials_edit.setText(bad)
        dlg._on_submit()
        assert dlg.result_values is None and "2-4 letters" in dlg.error_label.text(), (bad, dlg.error_label.text())
    dlg = ReviewLauncherDialog()
    dlg.synthetic_checkbox.setChecked(True)
    dlg.step3_radio.setChecked(True)
    dlg.initials_edit.setText("ccc")
    dlg.compare_a_edit.setText("CC")
    dlg.compare_b_edit.setText("cc")
    dlg._on_submit()
    assert dlg.result_values is None and "different" in dlg.error_label.text().lower()
    print("   OK: 'HLU' -> 'hlu'; 'sub-003', 'h', 'hl u', 'abcde' rejected; 'CC' vs 'cc' rejected as the same person")

    print("13. The Subject field never suggests a subject (RAs follow the Google tracking sheet; HLU 2026-09-27)...")
    dlg = ReviewLauncherDialog()
    tmp_dir = tempfile.mkdtemp(prefix="simple_gui_suggest_")
    try:
        os.makedirs(os.path.join(tmp_dir, "sub-001"))
        os.makedirs(os.path.join(tmp_dir, "sub-002"))
        with open(os.path.join(tmp_dir, "processing_log.csv"), "w") as f:
            f.write("Subject,Run,Initials,Step,Notes,Timestamp\nsub-001,,hlu,claimed,,1/1/2026 9:00\n")
        dlg.box_path_edit.setText(tmp_dir)
        dlg.box_path_edit.editingFinished.emit()
        assert dlg.subject_edit.placeholderText() == "e.g. 001", dlg.subject_edit.placeholderText()
        assert not hasattr(dlg, "_update_subject_suggestion")
        print("   OK: the placeholder stays 'e.g. 001' after a box path is entered")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("14. The guide checkbox (HLU, 2026-09-27) is offered for ECG, SBP, DBP, RSP and EDA, not PPG or EMG...")
    dlg = ReviewLauncherDialog()
    dlg.step2_radio.setChecked(True)
    for key, offered in (("ecg", True), ("sbp", True), ("dbp", True), ("rsp", True), ("eda", True),
                         ("ppg", False), ("emg_cor", False)):
        dlg.channel_combo.setCurrentIndex(dlg._channel_keys.index(key))
        assert dlg.ppg_guide_checkbox.isEnabled() is offered, (key, offered)
    print("   OK")

    print("14b. The PPG under ECG is opt-in (HLU, 2026-10-03): with ECG or Step 1 the box is relabeled and starts "
          "unticked; other guided channels start ticked; a manual tick sticks until ECG/other changes...")
    ecg_label = "Show the PPG under the ECG (read-only; for counting beats only)"
    dlg = ReviewLauncherDialog()
    dlg.step2_radio.setChecked(True)
    dlg.channel_combo.setCurrentIndex(dlg._channel_keys.index("ecg"))
    assert dlg.ppg_guide_checkbox.text() == ecg_label and not dlg.ppg_guide_checkbox.isChecked()
    dlg.channel_combo.setCurrentIndex(dlg._channel_keys.index("sbp"))
    assert dlg.ppg_guide_checkbox.text() == "Show the guide channel (read-only)" and dlg.ppg_guide_checkbox.isChecked()
    dlg.ppg_guide_checkbox.setChecked(False)
    dlg.channel_combo.setCurrentIndex(dlg._channel_keys.index("rsp"))
    assert not dlg.ppg_guide_checkbox.isChecked(), "a manual untick sticks while the choice stays non-ECG"
    dlg.channel_combo.setCurrentIndex(dlg._channel_keys.index("ecg"))
    dlg.ppg_guide_checkbox.setChecked(True)
    dlg.run_combo.setCurrentText("2")
    assert dlg.ppg_guide_checkbox.isChecked(), "a manual tick sticks while ECG stays selected"
    dlg.step1_radio.setChecked(True)
    assert dlg.ppg_guide_checkbox.isChecked() and dlg.ppg_guide_checkbox.text() == ecg_label, "Step 1 is ECG too"
    dlg.step2_radio.setChecked(True)
    dlg.channel_combo.setCurrentIndex(dlg._channel_keys.index("eda"))
    assert dlg.ppg_guide_checkbox.isChecked(), "leaving ECG resets to the other channels' default (on)"
    dlg = ReviewLauncherDialog()
    dlg.step1_radio.setChecked(True)
    assert dlg.ppg_guide_checkbox.text() == ecg_label and not dlg.ppg_guide_checkbox.isChecked()
    print("   OK")

    print("15. New or look-alike initials are confirmed once, in real mode (HLU, 2026-09-27)...")
    tmp_dir = tempfile.mkdtemp(prefix="simple_gui_initials_")
    try:
        with open(os.path.join(tmp_dir, "processing_log.csv"), "w") as f:
            f.write("Subject,Run,Initials,Step,Notes,Timestamp\nsub-001,1,HLU,ecg reviewed,,1/1/2026 9:00\n"
                    "sub-002,1,sub,ecg reviewed,,1/1/2026 9:00\n")

        def filled(initials, box=tmp_dir):
            dlg = ReviewLauncherDialog()
            dlg.box_path_edit.setText(box)
            dlg.subject_edit.setText("001")
            dlg.local_path_edit.setText(os.path.join(tmp_dir, "work"))
            dlg.step2_radio.setChecked(True)
            dlg.initials_edit.setText(initials)
            return dlg

        asked = []

        def answer(value):
            return lambda question: asked.append(question) or value

        dlg = filled("jkl")
        dlg._confirm_initials = answer(False)
        dlg._on_submit()
        assert dlg.result_values is None and "check your initials" in dlg.error_label.text().lower()
        assert "hasn't been used" in asked[-1] and '"jkl"' in asked[-1], asked
        dlg._confirm_initials = answer(True)
        dlg._on_submit()
        assert dlg.result_values and dlg.result_values["initials"] == "jkl", "a new RA says yes once"

        dlg = filled("HLU")  # in the log (any case): not asked
        dlg._confirm_initials = lambda question: (_ for _ in ()).throw(AssertionError(question))
        dlg._on_submit()
        assert dlg.result_values["initials"] == "hlu"

        dlg = filled("sub")  # a look-alike is asked even though an earlier slip put it in the log
        dlg._confirm_initials = answer(False)
        dlg._on_submit()
        assert dlg.result_values is None and "looks like the start of a subject ID" in asked[-1], asked[-1]

        dlg = filled("jkl", box=os.path.join(tmp_dir, "no_log_here"))  # no log to read: not asked
        dlg._confirm_initials = lambda question: (_ for _ in ()).throw(AssertionError(question))
        dlg._on_submit()
        assert dlg.result_values["initials"] == "jkl"

        dlg = ReviewLauncherDialog()  # synthetic practice: never asked
        dlg.synthetic_checkbox.setChecked(True)
        dlg.initials_edit.setText("sub")
        dlg._confirm_initials = lambda question: (_ for _ in ()).throw(AssertionError(question))
        dlg._on_submit()
        assert dlg.result_values["initials"] == "sub"
        print("   OK: new initials asked (No returns to the form, Yes starts); known ones not asked; 'sub' always "
              "asked; no log or synthetic: not asked")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("16. Practice for certification (HLU, 2026-10-04): the box locks subject 990, turns off Step 3, sends "
          "the 'practice' folder next to the derivatives folder; switching it off restores the subject...")
    tmp_dir = tempfile.mkdtemp(prefix="simple_gui_practice_")
    try:
        derivatives = os.path.join(tmp_dir, "derivatives")
        os.makedirs(derivatives)
        dlg = ReviewLauncherDialog()
        dlg.box_path_edit.setText(derivatives)
        dlg.subject_edit.setText("012")
        dlg.local_path_edit.setText(os.path.join(tmp_dir, "work"))
        dlg.initials_edit.setText("hlu")
        dlg.step3_radio.setChecked(True)
        assert dlg.practice_note.isHidden()
        dlg.practice_checkbox.setChecked(True)
        assert dlg.subject_edit.text() == "990" and not dlg.subject_edit.isEnabled()
        assert not dlg.step3_radio.isEnabled() and dlg.step2_radio.isChecked(), "Step 3 off; switched to Step 2"
        assert not dlg.practice_note.isHidden()

        dlg._confirm_initials = lambda question: True
        dlg._on_submit()
        assert dlg.result_values is None and "no practice folder" in dlg.error_label.text().lower(), \
            dlg.error_label.text()
        os.makedirs(os.path.join(tmp_dir, "practice"))
        dlg._on_submit()
        values = dlg.result_values
        assert values["practice"] is True and values["subject"] == "990" and values["stage"] == "2"
        assert os.path.normpath(values["box_path"]) == os.path.join(tmp_dir, "practice"), values["box_path"]
        assert dlg.box_path_edit.text() == derivatives, "the form (and its remembered path) keeps derivatives"

        dlg.practice_checkbox.setChecked(False)
        assert dlg.subject_edit.text() == "012" and dlg.subject_edit.isEnabled() and dlg.step3_radio.isEnabled()
        assert dlg.practice_note.isHidden()

        dlg.synthetic_checkbox.setChecked(True)
        assert not dlg.practice_checkbox.isEnabled(), "practice uses the practice folder, not synthetic data"

        with open(os.path.join(derivatives, "processing_log.csv"), "w") as f:
            f.write("Subject,Run,Initials,Step,Notes,Timestamp\nsub-001,1,HLU,ecg reviewed,,1/1/2026 9:00\n")
        with open(os.path.join(tmp_dir, "practice", "processing_log.csv"), "w") as f:
            f.write("Subject,Run,Initials,Step,Notes,Timestamp\nsub-990,1,abc,claimed,,1/1/2026 9:00\n")
        for initials in ("hlu", "abc"):  # known from the real log, or from earlier practice: not asked
            dlg = ReviewLauncherDialog()
            dlg.box_path_edit.setText(derivatives)
            dlg.local_path_edit.setText(os.path.join(tmp_dir, "work"))
            dlg.practice_checkbox.setChecked(True)
            dlg.initials_edit.setText(initials)
            dlg._confirm_initials = lambda question: (_ for _ in ()).throw(AssertionError(question))
            dlg._on_submit()
            assert dlg.result_values["initials"] == initials
        print("   OK: subject 990 locked, Step 3 off, missing practice folder reported, practice folder sent, "
              "derivatives path kept, subject restored, off with synthetic, initials known from either log")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\nALL SIMPLE_GUI LOGIC TESTS PASSED.")


if __name__ == "__main__":
    main()
