"""
A minimal PyQt6 form shown only when physio_review.py is launched with NO
command-line arguments at all -- collects the same box-path/subject/run/
initials that would otherwise be requested one at a time via console
input() prompts, plus (per user request, after live testing showed the
full 6-channel Stage B view is too busy) an explicit "Step 1 or Step 2"
choice and a single-channel picker for Step 2.

Uses PyQt6 since it's already a dependency (mne-qt-browser needs it) --
no new package required. Any flag on the command line skips this entirely
and uses the normal CLI/prompt behavior, so this is purely an on-ramp for
the simplest case, not a replacement for the CLI.
"""

import os
import re

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from channel_config import CHANNELS, GUIDE_CHANNELS

# The guide box's label and default by context (HLU, 2026-10-03): the PPG
# under ECG is opt-in; the other guide rows are on by default.
GUIDE_BOX_LABELS = {
    "ecg": "Show the PPG under the ECG (read-only; for counting beats only)",
    "other": "Show the guide channel (read-only)",
}
GUIDE_BOX_DEFAULTS = {"ecg": False, "other": True}

# Certification practice (HLU, 2026-10-04).
PRACTICE_BOX_LABEL = "Practice for certification (practice participant 990)"
PRACTICE_NOTE = ("Uses the 'practice' folder next to your Box derivatives folder. Build your ECG template in Step 1 "
                 "first. After you save a Step 2 review, PEBBL scores it and can show your marks next to the answer key.")
PRACTICE_SUBJECT = "990"


def practice_root_for(derivatives_path):
    """The practice folder: a 'practice' folder next to the Box derivatives folder."""
    return os.path.join(os.path.dirname(os.path.normpath(derivatives_path)), "practice")
from processing_log import initials_confirmation_text, known_initials

# Same rule as physio_review.normalize_initials() (audit finding H6).
INITIALS_PATTERN = re.compile(r"^[a-z]{2,4}$")


class ReviewLauncherDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multisignal Physio Review")
        self.setMinimumWidth(640)  # wide enough that no label or option is cut off (2026-10-04)
        self.result_values = None

        layout = QVBoxLayout(self)

        intro = QLabel(
            "Fill in the fields below to review a subject's physio data, or check\n"
            "the box to try it with synthetic demo data instead."
        )
        layout.addWidget(intro)

        self.synthetic_checkbox = QCheckBox("Use synthetic demo data (no real files needed, for testing)")
        self.synthetic_checkbox.stateChanged.connect(self._on_field_changed)
        layout.addWidget(self.synthetic_checkbox)

        # Certification practice (HLU, 2026-10-04): the practice participant
        # sub-990 from the "practice" folder next to the Box derivatives
        # folder; after each Step 2 save, PEBBL scores the review against the
        # answer key there (physio_review.py --practice).
        self.practice_checkbox = QCheckBox(PRACTICE_BOX_LABEL)
        self.practice_checkbox.stateChanged.connect(self._on_field_changed)
        layout.addWidget(self.practice_checkbox)
        self.practice_note = QLabel(PRACTICE_NOTE)
        self.practice_note.setWordWrap(True)
        self.practice_note.setStyleSheet("color: #55616A; margin-left: 22px;")
        layout.addWidget(self.practice_note)
        self._subject_before_practice = ""

        # --- Step 1 vs Step 2 vs Step 3 ---
        stage_row = QHBoxLayout()
        self.step1_radio = QRadioButton("Step 1: Build ECG QRS template")
        self.step2_radio = QRadioButton("Step 2: Review a channel")
        self.step3_radio = QRadioButton("Step 3: Reconcile two reviewers")
        self.step2_radio.setChecked(True)  # the more common day-to-day action
        self.step1_radio.toggled.connect(self._on_field_changed)
        self.step3_radio.toggled.connect(self._on_field_changed)
        stage_row.addWidget(self.step1_radio)
        stage_row.addWidget(self.step2_radio)
        stage_row.addWidget(self.step3_radio)
        layout.addLayout(stage_row)

        form = QFormLayout()

        box_path_row = QHBoxLayout()
        self.box_path_edit = QLineEdit()
        self.box_path_edit.setPlaceholderText(r"C:\Users\you\Box\DATA\Processed\physioProcessing\derivatives")
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._browse_box_path)
        box_path_row.addWidget(self.box_path_edit)
        box_path_row.addWidget(self.browse_button)
        form.addRow(QLabel("Box derivatives path:"), box_path_row)

        self.subject_edit = QLineEdit()
        self.subject_edit.setPlaceholderText("e.g. 001")
        form.addRow(QLabel("Subject:"), self.subject_edit)

        local_path_row = QHBoxLayout()
        self.local_path_edit = QLineEdit()
        self.local_path_edit.setPlaceholderText(r"C:\Users\you\PhysioWorking")
        self.local_browse_button = QPushButton("Browse...")
        self.local_browse_button.clicked.connect(self._browse_local_path)
        local_path_row.addWidget(self.local_path_edit)
        local_path_row.addWidget(self.local_browse_button)
        self.local_path_label = QLabel("Local working folder:")
        form.addRow(self.local_path_label, local_path_row)

        self.run_combo = QComboBox()
        self.run_combo.addItems(["1", "2"])
        self.run_field_label = QLabel("Run to review:")
        form.addRow(self.run_field_label, self.run_combo)

        self.template_start_edit = QLineEdit()
        self.template_start_edit.setPlaceholderText("0")
        self.template_start_label = QLabel("Template window start (s), Step 1 only:")
        form.addRow(self.template_start_label, self.template_start_edit)

        # channel keys in a fixed, friendly order; ECG first since it's the
        # default/Step-1-only channel
        self._channel_keys = list(CHANNELS.keys())
        self.channel_combo = QComboBox()
        for key in self._channel_keys:
            self.channel_combo.addItem(CHANNELS[key]["label"], userData=key)
        self.channel_field_label = QLabel("Channel to review:")
        form.addRow(self.channel_field_label, self.channel_combo)
        # ECG-only options follow the chosen channel (ppg-plan §4b: until
        # 2026-09-26 the picker wasn't connected, so "ECG peak source" stayed
        # live for every channel).
        self.channel_combo.currentIndexChanged.connect(self._on_field_changed)

        self.initials_edit = QLineEdit()
        self.initials_edit.setPlaceholderText("your initials")
        self.initials_label = QLabel("Your initials:")
        form.addRow(self.initials_label, self.initials_edit)

        self.compare_a_edit = QLineEdit()
        self.compare_a_edit.setPlaceholderText("e.g. hlu")
        self.compare_a_label = QLabel("Reviewer A's initials:")
        form.addRow(self.compare_a_label, self.compare_a_edit)

        self.compare_b_edit = QLineEdit()
        self.compare_b_edit.setPlaceholderText("e.g. xyz (or your own, for a gold-standard check)")
        self.compare_b_label = QLabel("Reviewer B's initials:")
        form.addRow(self.compare_b_label, self.compare_b_edit)

        layout.addLayout(form)

        advanced_label = QLabel("Options (optional -- defaults are fine for typical use)")
        advanced_label.setStyleSheet("font-weight: bold; margin-top: 6px;")
        layout.addWidget(advanced_label)

        options_form = QFormLayout()
        self.ecg_source_combo = QComboBox()
        self.ecg_source_combo.addItem("Your own ECG template (built in Step 1)", userData="auto")
        self.ecg_source_combo.addItem("Automatic detection only (no template)", userData="batch")
        self.ecg_source_label = QLabel("ECG peak source, Step 2 only:")
        options_form.addRow(self.ecg_source_label, self.ecg_source_combo)
        # The read-only guide rows: PPG under SBP/DBP, EDA under RSP, RSP under
        # EDA (on by default; HLU, 2026-09-27), and PPG under ECG in Steps 1-3
        # (OFF by default since 2026-10-03, HLU: its delay after the R peak
        # differs from run to run). One box: its label and default follow the
        # ECG/other choice (_sync_guide_default); a manual tick sticks until
        # that choice changes. Only meaningful for guided channels.
        self.ppg_guide_checkbox = QCheckBox(GUIDE_BOX_LABELS["other"])
        self.ppg_guide_checkbox.setChecked(True)
        self._guide_context = None
        options_form.addRow(QLabel(""), self.ppg_guide_checkbox)
        layout.addLayout(options_form)

        # Options that can strand data or change template building are hidden
        # unless lab staff ask for them (audit §5: RAs shouldn't meet them by
        # accident). They still work the same when shown.
        self.lab_staff_checkbox = QCheckBox("Show lab-staff options")
        self.lab_staff_checkbox.stateChanged.connect(self._on_lab_staff_toggled)
        layout.addWidget(self.lab_staff_checkbox)
        self.lab_staff_box = QWidget()
        advanced_form = QFormLayout(self.lab_staff_box)
        advanced_form.setContentsMargins(0, 0, 0, 0)

        self.template_run_combo = QComboBox()
        self.template_run_combo.addItems(["1", "2"])
        self.template_run_label = QLabel("Build template from run, Step 1 only:")
        advanced_form.addRow(self.template_run_label, self.template_run_combo)

        self.template_window_edit = QLineEdit()
        self.template_window_edit.setPlaceholderText("20")
        self.template_window_label = QLabel("Template window length (s), Step 1 only:")
        advanced_form.addRow(self.template_window_label, self.template_window_edit)

        self.no_local_copy_checkbox = QCheckBox("Skip local copy (work directly against the Box path)")
        self.no_local_copy_checkbox.stateChanged.connect(self._on_field_changed)
        advanced_form.addRow(QLabel(""), self.no_local_copy_checkbox)

        self.keep_local_checkbox = QCheckBox("Keep the local copy after pushing back to Box (troubleshooting)")
        advanced_form.addRow(QLabel(""), self.keep_local_checkbox)

        layout.addWidget(self.lab_staff_box)
        self.lab_staff_box.setVisible(False)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #c0392b;")
        layout.addWidget(self.error_label)

        button_row = QHBoxLayout()
        button_row.addStretch()
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        submit_button = QPushButton("Start")
        submit_button.setDefault(True)
        submit_button.clicked.connect(self._on_submit)
        button_row.addWidget(cancel_button)
        button_row.addWidget(submit_button)
        layout.addLayout(button_row)

        self._load_saved_settings()
        self._on_field_changed()

    # --- remembered settings -------------------------------------------------
    # Box path, local working folder and initials are remembered between
    # sessions so RAs don't retype them (audit §5). Tests turn this off.
    USE_SAVED_SETTINGS = True

    def _settings(self):
        return QSettings("EmotionBrainBehaviorLab", "physio_review")

    def _load_saved_settings(self):
        if not self.USE_SAVED_SETTINGS:
            return
        settings = self._settings()
        self.box_path_edit.setText(settings.value("box_path", "", type=str))
        self.local_path_edit.setText(settings.value("local_path", "", type=str))
        self.initials_edit.setText(settings.value("initials", "", type=str))

    def _save_settings(self):
        if not self.USE_SAVED_SETTINGS or self.synthetic_checkbox.isChecked():
            return
        settings = self._settings()
        settings.setValue("box_path", self.box_path_edit.text().strip())
        settings.setValue("local_path", self.local_path_edit.text().strip())
        if self.initials_edit.text().strip():
            settings.setValue("initials", self.initials_edit.text().strip().lower())

    def _on_lab_staff_toggled(self, *_args):
        self.lab_staff_box.setVisible(self.lab_staff_checkbox.isChecked())

    def _set_real_fields_enabled(self, enabled):
        for widget in (self.box_path_edit, self.browse_button, self.subject_edit,
                       self.step1_radio, self.step2_radio, self.step3_radio,
                       self.no_local_copy_checkbox, self.keep_local_checkbox):
            widget.setEnabled(enabled)

    def _sync_guide_default(self, context):
        """Sets the guide box's label and default when the ECG/other choice changes (not on every field change)."""
        if context == self._guide_context:
            return
        self._guide_context = context
        self.ppg_guide_checkbox.setText(GUIDE_BOX_LABELS[context])
        self.ppg_guide_checkbox.setChecked(GUIDE_BOX_DEFAULTS[context])

    def _on_field_changed(self, *_args):
        is_synthetic = self.synthetic_checkbox.isChecked()
        self._set_real_fields_enabled(not is_synthetic)
        self._apply_practice_mode(is_synthetic)

        # The local-copy lifecycle (and so --keep-local/--no-local-copy and
        # the local working folder itself) only applies in real mode, and
        # the local path is meaningless once --no-local-copy is chosen.
        no_local_copy = self.no_local_copy_checkbox.isChecked()
        local_path_enabled = (not is_synthetic) and (not no_local_copy)
        for widget in (self.local_path_edit, self.local_browse_button, self.local_path_label):
            widget.setEnabled(local_path_enabled)
        self.keep_local_checkbox.setEnabled((not is_synthetic) and (not no_local_copy))

        is_step1 = self.step1_radio.isChecked()
        is_step3 = self.step3_radio.isChecked()

        # --ecg-source only affects Step 2's peak-seeding choice. Like
        # --template-start, kept real-mode-only -- synthetic mode is meant
        # to stay a minimal, no-knobs smoke test.
        is_ecg = is_step1 or self.channel_combo.currentData() == "ecg"
        ecg_source_enabled = (not is_synthetic) and (not is_step1) and (not is_step3) and is_ecg
        self.ecg_source_combo.setEnabled(ecg_source_enabled)
        self.ecg_source_label.setEnabled(ecg_source_enabled)
        guided = {k for cfg in GUIDE_CHANNELS.values() for k in cfg["guide_for"]}
        self.ppg_guide_checkbox.setEnabled(is_ecg or is_synthetic or self.channel_combo.currentData() in guided)
        self._sync_guide_default("ecg" if (is_ecg or (is_synthetic and not is_step3)) else "other")

        # --template-run/--template-window only mean anything while Stage A
        # actually runs, which only happens for Step 1 (Step 2 always uses
        # an existing template or the batch pipeline -- see physio_review.py).
        template_fields_enabled = (not is_synthetic) and is_step1
        for widget in (self.template_run_combo, self.template_run_label,
                       self.template_window_edit, self.template_window_label):
            widget.setEnabled(template_fields_enabled)

        # Step 1 always works with ECG and doesn't need a "which run to
        # review" choice (it builds from one run's window and applies the
        # result to every available run) -- lock/hide those accordingly.
        # Step 3 (reconciliation) DOES need both, same as Step 2.
        run_enabled = (not is_synthetic) and (not is_step1)
        self.run_combo.setEnabled(run_enabled)
        self.run_field_label.setEnabled(run_enabled)

        channel_enabled = (not is_synthetic) and (not is_step1)
        if is_step1:
            ecg_index = self._channel_keys.index("ecg")
            self.channel_combo.setCurrentIndex(ecg_index)
        self.channel_combo.setEnabled(channel_enabled)
        self.channel_field_label.setEnabled(channel_enabled)

        # --template-start only means anything for Step 1's template-building window.
        template_start_enabled = (not is_synthetic) and is_step1
        self.template_start_edit.setEnabled(template_start_enabled)
        self.template_start_label.setEnabled(template_start_enabled)

        # Step 3 compares two ALREADY-saved sessions -- there's no single
        # "your initials" for it, so swap that field out for the two
        # reviewer-initials fields it needs instead (works in synthetic
        # mode too, so not gated on is_synthetic).
        # Step 3 still needs the RECONCILER's own initials (recorded in the
        # reconciled file -- audit finding M2), relabeled to say so.
        self.initials_label.setText("Your initials (you are reconciling):" if is_step3 else "Your initials:")
        for widget in (self.compare_a_edit, self.compare_a_label, self.compare_b_edit, self.compare_b_label):
            widget.setEnabled(is_step3)

    def _apply_practice_mode(self, is_synthetic):
        """Practice: subject 990 (locked), no Step 3; not with synthetic data. Switching it off restores the subject."""
        self.practice_checkbox.setEnabled(not is_synthetic)
        practice = self.practice_checkbox.isChecked() and not is_synthetic
        self.practice_note.setVisible(practice)
        was_practice = getattr(self, "_was_practice", False)
        self._was_practice = practice  # set first: changing the Step radios below re-enters this method
        if practice and not was_practice:
            self._subject_before_practice = self.subject_edit.text()
        if practice:
            self.subject_edit.setText(PRACTICE_SUBJECT)
            self.subject_edit.setEnabled(False)
            if self.step3_radio.isChecked():
                self.step2_radio.setChecked(True)
            self.step3_radio.setEnabled(False)
        elif was_practice:
            self.subject_edit.setText(self._subject_before_practice)

    def _browse_box_path(self):
        chosen = QFileDialog.getExistingDirectory(self, "Select Box derivatives folder")
        if chosen:
            self.box_path_edit.setText(chosen)

    def _browse_local_path(self):
        chosen = QFileDialog.getExistingDirectory(self, "Select local working folder")
        if chosen:
            self.local_path_edit.setText(chosen)

    def _confirm_initials(self, question):
        """True if the RA confirms these initials (a Yes/No dialog; tests replace it)."""
        answer = QMessageBox.question(self, "Check your initials", question,
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                      QMessageBox.StandardButton.No)
        return answer == QMessageBox.StandardButton.Yes

    def _on_submit(self):
        if self.step1_radio.isChecked():
            stage = "1"
        elif self.step3_radio.isChecked():
            stage = "3"
        else:
            stage = "2"

        initials = self.initials_edit.text().strip().lower()
        if not initials:
            self.error_label.setText("Please enter your initials.")
            return
        if not INITIALS_PATTERN.match(initials):
            self.error_label.setText("Initials must be 2-4 letters only (e.g. hlu).")
            return
        if stage == "3":
            compare_a = self.compare_a_edit.text().strip().lower()
            compare_b = self.compare_b_edit.text().strip().lower()
            if not compare_a or not compare_b:
                self.error_label.setText("Please enter both reviewers' initials to compare.")
                return
            if not (INITIALS_PATTERN.match(compare_a) and INITIALS_PATTERN.match(compare_b)):
                self.error_label.setText("Reviewers' initials must be 2-4 letters only (e.g. hlu).")
                return
            if compare_a == compare_b:
                self.error_label.setText("The two reviewers to compare must be different people.")
                return
            compare_initials = f"{compare_a},{compare_b}"

        if self.synthetic_checkbox.isChecked():
            if stage == "3":
                self.result_values = {
                    "synthetic": True, "stage": stage, "channel": self.channel_combo.currentData(),
                    "compare_initials": compare_initials, "initials": initials,
                }
            else:
                self.result_values = {"synthetic": True, "initials": initials, "stage": stage, "channel": "ecg"}
            self.result_values["ppg_guide"] = self.ppg_guide_checkbox.isChecked()
            return self.accept()

        box_path = self.box_path_edit.text().strip()
        subject = self.subject_edit.text().strip()
        local_path = self.local_path_edit.text().strip()
        no_local_copy = self.no_local_copy_checkbox.isChecked()
        practice = self.practice_checkbox.isChecked()
        derivatives_path = box_path
        if practice and box_path:
            practice_root = practice_root_for(box_path)
            if not os.path.isdir(practice_root):
                self.error_label.setText(f"There's no practice folder at {practice_root}. Check the Box derivatives "
                                         f"path, or ask the lab staff to set up the practice folder.")
                return
            box_path, subject = practice_root, PRACTICE_SUBJECT

        if not box_path or not subject or (not local_path and not no_local_copy):
            self.error_label.setText(
                "Please fill in the Box path, subject, and local working folder "
                "(or, under lab-staff options, check 'Skip local copy'), or check 'Use synthetic demo data'."
            )
            return

        # Initials name the saved files, so new ones (not yet in the Box
        # processing log) and look-alikes such as "sub" are confirmed once
        # (HLU, 2026-09-27). Real mode only; with no log to read, only the
        # look-alike check applies.
        known = known_initials(box_path)
        if practice:  # initials known from the real log or from earlier practice
            real_known = known_initials(derivatives_path)
            known = None if known is None and real_known is None else (known or set()) | (real_known or set())
        question = initials_confirmation_text(initials, known)
        if question and not self._confirm_initials(question):
            self.error_label.setText("Check your initials, then press Start again.")
            self.initials_edit.setFocus()
            return

        common_values = {
            "ppg_guide": self.ppg_guide_checkbox.isChecked(),
            "practice": practice,
            "synthetic": False,
            "box_path": box_path,
            "local_path": local_path,
            "subject": subject,
            "no_local_copy": no_local_copy,
            "keep_local": self.keep_local_checkbox.isChecked(),
        }

        if stage == "1":
            template_start_text = self.template_start_edit.text().strip()
            try:
                template_start = float(template_start_text) if template_start_text else 0.0
            except ValueError:
                self.error_label.setText("Template window start must be a number (seconds).")
                return
            template_window_text = self.template_window_edit.text().strip()
            try:
                template_window = float(template_window_text) if template_window_text else 20.0
            except ValueError:
                self.error_label.setText("Template window length must be a number (seconds).")
                return
            self.result_values = {
                **common_values,
                "run": None,
                "initials": initials,
                "stage": "1",
                "channel": "ecg",
                "template_start": template_start,
                "template_run": self.template_run_combo.currentText(),
                "template_window": template_window,
            }
            self._save_settings()
            return self.accept()

        if stage == "3":
            self.result_values = {
                **common_values,
                "run": self.run_combo.currentText(),
                "stage": "3",
                "channel": self.channel_combo.currentData(),
                "compare_initials": compare_initials,
                "initials": initials,
            }
            self._save_settings()
            return self.accept()

        self.result_values = {
            **common_values,
            "run": self.run_combo.currentText(),
            "initials": initials,
            "stage": "2",
            "channel": self.channel_combo.currentData(),
            "ecg_source": self.ecg_source_combo.currentData(),
        }
        self._save_settings()
        self.accept()


def collect_inputs_via_gui():
    """
    Shows the launcher dialog and returns a dict of collected values, or
    None if the user closed/canceled the dialog. Always includes "stage"
    ("1", "2", or "3") and "channel" (an internal channel_config.py key,
    "ecg" for Step 1). Real-mode results (any stage) additionally include
    "local_path" -- the local working folder physio_review.py copies this
    subject's files to, works from, then pushes back to Box and cleans up
    automatically (see box_sync.py) -- there's no synthetic-mode
    equivalent, since synthetic data never touches Box at all -- plus
    "no_local_copy" and "keep_local" (both bool, default False; skip/keep
    that local-copy lifecycle -- see the "Advanced options" section).
    Real-mode Step 1 results additionally include "template_start" (float
    seconds, default 0.0), "template_run" ("1" or "2", default "1"), and
    "template_window" (float seconds, default 20.0); real-mode Step 2/3
    results include "run" instead (Step 1 has no "run", since it applies to
    every available run). Real-mode Step 2 results additionally include
    "ecg_source" ("auto" or "batch", default "auto"). Step 3 (reconciliation)
    has no "initials" key at all -- it compares two already-saved sessions,
    not one RA's current one -- and instead includes "compare_initials"
    ("A,B", comma-joined).
    """
    app = QApplication.instance() or QApplication([])
    dialog = ReviewLauncherDialog()
    accepted = dialog.exec() == QDialog.DialogCode.Accepted
    return dialog.result_values if accepted else None
