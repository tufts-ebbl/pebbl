"""
The "session summary" dialog shown when an RA closes the Step 2 (review) or
Step 3 (reconcile) viewer, before anything is saved.

It replaces four separate end-of-session behaviors with one decision point
(audit multisignal-annotation_audit_20260923-1901.md, §9.4):
    - a change summary (peaks added/removed, bad segments added/removed),
    - the merge / irregular-interval warnings that used to appear only in
      the terminal after the window had closed,
    - the optional comment (previously a terminal input() prompt, which
      blocked the push back to Box -- audit finding C2),
    - a "finished" checkbox, recorded as each reviewed channel's status,
and three choices: Save (default), Discard this session's changes (asks to
confirm), or Go back to the viewer (annotations still in place).

count_changes() is pure logic, testable without Qt. Tests replace
show_session_summary with headless_decision(...) so no modal dialog opens.
"""

from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

SAVE = "save"
DISCARD = "discard"
BACK = "back"


@dataclass
class SessionDecision:
    action: str          # SAVE, DISCARD, or BACK
    comment: str = ""
    finished: bool = False


def _match_pairs(indices_a, indices_b, tolerance_samples):
    """One-to-one (a, b) pairs within tolerance (greedy, closest first)."""
    candidates = sorted(
        (abs(a - b), ia, ib)
        for ia, a in enumerate(indices_a)
        for ib, b in enumerate(indices_b)
        if abs(a - b) <= tolerance_samples
    )
    used_a, used_b, pairs = set(), set(), []
    for _dist, ia, ib in candidates:
        if ia not in used_a and ib not in used_b:
            used_a.add(ia)
            used_b.add(ib)
            pairs.append((indices_a[ia], indices_b[ib]))
    return pairs


def count_changes(baseline_outputs, final_outputs, channel_configs, tolerance_samples):
    """
    Compares the annotations a session started with (baseline_outputs) to
    what it ends with (final_outputs), both in export_annotations()'s shape.
    Returns {ch_key: {"added": n, "removed": n, "moved": n}} for every channel
    present in final_outputs ("moved" only for point channels). A point that
    moved by no more than tolerance_samples is "moved", not added and
    removed; segments are compared exactly.

    "moved" counts toward total_changes(). Until 2026-09-26 a move within
    tolerance counted as no change at all, so an RA who nudged peaks by up to
    50 ms on a resumed, already-complete file saw "No changes" and the save
    was skipped (ppg-plan §3.1 #3).
    """
    changes = {}
    for ch_key, final in final_outputs.items():
        base = baseline_outputs.get(ch_key, {})
        if final.get("mode") == "point":
            before, after = list(base.get("indices", [])), list(final.get("indices", []))
            if len(before) * len(after) > 4_000_000:
                # Exhaustive pairing would be too slow for very long runs; a
                # sorted two-pointer match gives the same answer for points
                # that are farther apart than the tolerance, as beats are.
                pairs = _sorted_match_pairs(before, after, tolerance_samples)
            else:
                pairs = _match_pairs(before, after, tolerance_samples)
            moved = sum(1 for a, b in pairs if a != b)
            changes[ch_key] = {"added": len(after) - len(pairs), "removed": len(before) - len(pairs),
                               "moved": moved}
        elif final.get("mode") == "segment":
            before = {tuple(s) for s in base.get("bad_segments", [])}
            after = {tuple(s) for s in final.get("bad_segments", [])}
            changes[ch_key] = {"added": len(after - before), "removed": len(before - after)}
    return changes


def edit_record(baseline_output, final_output, tolerance_samples):
    """
    What one session changed on one channel, for the saved file's per-channel
    "edit_history" (ppg-plan §4c): point channels {"added": [...], "removed":
    [...], "moved": [[from, to], ...]} (a move = matched within tolerance but
    not identical); segment channels {"added": [[s, e], ...], "removed": [...]}.
    """
    base = baseline_output or {}
    if final_output.get("mode") == "point":
        before, after = list(base.get("indices", [])), list(final_output.get("indices", []))
        pairs = (_sorted_match_pairs if len(before) * len(after) > 4_000_000 else _match_pairs)(
            before, after, tolerance_samples)
        matched_before = {a for a, _b in pairs}
        matched_after = {b for _a, b in pairs}
        return {"added": [int(i) for i in after if i not in matched_after],
                "removed": [int(i) for i in before if i not in matched_before],
                "moved": [[int(a), int(b)] for a, b in pairs if a != b]}
    before = {tuple(s) for s in base.get("bad_segments", [])}
    after = {tuple(s) for s in final_output.get("bad_segments", [])}
    return {"added": [list(s) for s in sorted(after - before)], "removed": [list(s) for s in sorted(before - after)]}


def _sorted_match_pairs(indices_a, indices_b, tolerance_samples):
    a, b = sorted(indices_a), sorted(indices_b)
    i = j = 0
    pairs = []
    while i < len(a) and j < len(b):
        if abs(a[i] - b[j]) <= tolerance_samples:
            pairs.append((a[i], b[j]))
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return pairs


def total_changes(changes):
    return sum(c["added"] + c["removed"] + c.get("moved", 0) for c in changes.values())


def format_change_lines(changes, channel_configs, seed_notes=None):
    """
    One plain-language line per channel, e.g. 'ECG: 3 peaks added, 5 removed,
    2 moved'. seed_notes ({ch_key: text}) appends what the channel started
    from, e.g. '(started from 12 machine-flagged stretches)'.
    """
    lines = []
    for ch_key, c in changes.items():
        cfg = channel_configs.get(ch_key, {})
        noun = "peaks" if cfg.get("annotation_mode") == "point" else "bad segments"
        line = f"{cfg.get('label', ch_key)}: {c['added']} {noun} added, {c['removed']} removed"
        if "moved" in c:
            line += f", {c['moved']} moved"
        if seed_notes and seed_notes.get(ch_key):
            line += f" ({seed_notes[ch_key]})"
        lines.append(line)
    return lines


MAX_WARNING_LINES = 12


class SessionSummaryDialog(QDialog):
    def __init__(self, heading, change_lines, warning_lines, n_changes, finished_default, tracker_hint,
                 blocking_lines=None, seed_lines=None):
        super().__init__()
        self.setWindowTitle("Session summary")
        self.setMinimumWidth(560)
        self.decision = SessionDecision(action=SAVE)
        self._n_changes = n_changes
        blocking_lines = list(blocking_lines or [])

        layout = QVBoxLayout(self)
        title = QLabel(heading)
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)

        layout.addWidget(QLabel("Changes this session:" if n_changes else "No changes were made this session."))
        for line in change_lines if n_changes else []:
            layout.addWidget(QLabel(f"    {line}"))
        # With no changes, the change lines (which carry each channel's "started
        # from ...") aren't drawn, so say what the channels started from here:
        # e.g. the machine's reasons for SBP/DBP flags (run-8 review).
        if not n_changes and seed_lines:
            for line in seed_lines:
                layout.addWidget(QLabel(f"    {line}"))

        if blocking_lines:
            # Problems that must be fixed before saving (e.g. Step 3's two PPG
            # marks on one pulse): Save stays disabled while any remain.
            block_label = QLabel("Fix these before you can save (use 'Go back to the viewer'):")
            block_label.setStyleSheet("color: #c0392b; font-weight: bold; margin-top: 6px;")
            layout.addWidget(block_label)
            for line in blocking_lines[:MAX_WARNING_LINES]:
                layout.addWidget(QLabel(f"    {line}"))
            if len(blocking_lines) > MAX_WARNING_LINES:
                layout.addWidget(QLabel(f"    ... and {len(blocking_lines) - MAX_WARNING_LINES} more "
                                        f"(listed in the terminal)."))

        if warning_lines:
            check_label = QLabel("Please check before saving (use 'Go back to the viewer'):")
            check_label.setStyleSheet("color: #b35900; margin-top: 6px;")
            layout.addWidget(check_label)
            for line in warning_lines[:MAX_WARNING_LINES]:
                layout.addWidget(QLabel(f"    {line}"))
            if len(warning_lines) > MAX_WARNING_LINES:
                layout.addWidget(QLabel(f"    ... and {len(warning_lines) - MAX_WARNING_LINES} more "
                                        f"(listed in the terminal)."))

        layout.addWidget(QLabel("Comments about this session (optional):"))
        self.comment_edit = QLineEdit()
        layout.addWidget(self.comment_edit)

        self.finished_checkbox = QCheckBox("I've finished reviewing this for this run")
        self.finished_checkbox.setChecked(finished_default)
        layout.addWidget(self.finished_checkbox)
        if tracker_hint:
            hint = QLabel(tracker_hint)
            hint.setWordWrap(True)
            hint.setStyleSheet("color: #555555;")
            layout.addWidget(hint)

        buttons = QHBoxLayout()
        back_button = QPushButton("Go back to the viewer")
        back_button.clicked.connect(self._on_back)
        discard_button = QPushButton("Discard this session's changes")
        discard_button.clicked.connect(self._on_discard)
        save_button = QPushButton("Save")
        save_button.setDefault(not blocking_lines)
        save_button.setEnabled(not blocking_lines)
        save_button.clicked.connect(self._on_save)
        self.save_button = save_button
        buttons.addWidget(back_button)
        buttons.addStretch()
        buttons.addWidget(discard_button)
        buttons.addWidget(save_button)
        layout.addLayout(buttons)

    def _collect(self, action):
        self.decision = SessionDecision(action=action, comment=self.comment_edit.text().strip(),
                                        finished=self.finished_checkbox.isChecked())

    def _on_back(self):
        self._collect(BACK)
        self.accept()

    def _on_save(self):
        self._collect(SAVE)
        self.accept()

    def _on_discard(self):
        what = f"the {self._n_changes} change(s) made" if self._n_changes else "this session"
        answer = QMessageBox.question(
            self, "Discard changes?",
            f"Discard {what} this session?\n\nYour last saved version (if any) stays exactly as it was.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._collect(DISCARD)
            self.accept()

    def reject(self):
        # Closing the dialog with the window's X (or Esc) must never silently
        # discard work: treat it as "go back to the viewer".
        self._collect(BACK)
        super().reject()


def show_session_summary(heading, change_lines, warning_lines, n_changes, finished_default=False,
                         tracker_hint="", blocking_lines=None, seed_lines=None):
    """
    Shows the dialog modally and returns a SessionDecision. With
    blocking_lines, Save is disabled (the RA can only go back or discard).
    seed_lines ("SBP started from ...") are shown when nothing changed.
    """
    app = QApplication.instance() or QApplication([])
    dialog = SessionSummaryDialog(heading, change_lines, warning_lines, n_changes, finished_default, tracker_hint,
                                  blocking_lines=blocking_lines, seed_lines=seed_lines)
    dialog.exec()
    return dialog.decision


def headless_decision(action=SAVE, comment="", finished=False):
    """
    Returns a stand-in for show_session_summary() that answers immediately,
    for non-interactive tests: e.g.
        session_summary_gui.show_session_summary = session_summary_gui.headless_decision()
    """
    def _decide(*_args, **_kwargs):
        return SessionDecision(action=action, comment=comment, finished=finished)
    return _decide
