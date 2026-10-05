"""
A small PyQt6 dialog showing the QRS template just built from the RA's
corrected window -- individual raw beats overlaid (thin, translucent) plus
the resulting averaged template (bold) -- with Approve / Try a Different
Window buttons, shown BEFORE the template is applied/saved.

This mirrors the beat-segment visualization step1_qrs_template.ipynb
already used (nk.ecg_segment(..., show=True)), but as an explicit
before-you-commit approval gate rather than an after-the-fact display, per
the user's request: "I'd like the user to be able to see the template that
gets created in step 1, ideally before it goes on to save it."

Uses PyQt6 (already a dependency) + matplotlib's Qt-agnostic backend
(matplotlib.backends.backend_qtagg, works with whichever Qt binding is
installed) -- no new package required.
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QApplication, QDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout


class TemplatePreviewDialog(QDialog):
    def __init__(self, raw_beats, template_values, sampling_rate):
        super().__init__()
        self.setWindowTitle("Review QRS Template")
        self.setMinimumSize(700, 500)
        self.approved = False

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            f"{len(raw_beats)} beat(s) from your corrected window (thin gray lines), and the "
            "resulting averaged template (bold black) that will be cross-correlated across the "
            "full recording to detect peaks."
        ))

        figure = Figure(figsize=(7, 5))
        canvas = FigureCanvasQTAgg(figure)
        ax = figure.add_subplot(111)

        time_ms = np.arange(len(template_values)) / sampling_rate * 1000
        for beat in raw_beats:
            n = min(len(beat), len(time_ms))
            ax.plot(time_ms[:n], beat[:n], color="gray", alpha=0.3, linewidth=1)
        ax.plot(time_ms[: len(template_values)], template_values, color="black", linewidth=2.5,
                label="Template (averaged)")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (wavelet-filtered)")
        ax.legend(loc="upper right")
        figure.tight_layout()

        layout.addWidget(canvas)

        # The session comment is collected here, not in the terminal, so an RA
        # who closes the terminal at a comment prompt can't strand unpushed work
        # (audit finding C2).
        layout.addWidget(QLabel("Comments about this template (optional):"))
        self.comment_edit = QLineEdit()
        self.comment_edit.setPlaceholderText("e.g. messy start, used 20-40 s")
        layout.addWidget(self.comment_edit)

        button_row = QHBoxLayout()
        button_row.addStretch()
        reject_button = QPushButton("Try a Different Window")
        reject_button.clicked.connect(self.reject)
        approve_button = QPushButton("Approve Template")
        approve_button.setDefault(True)
        approve_button.clicked.connect(self._on_approve)
        button_row.addWidget(reject_button)
        button_row.addWidget(approve_button)
        layout.addLayout(button_row)

    def _on_approve(self):
        self.approved = True
        self.accept()


def show_template_preview(raw_beats, template_values, sampling_rate):
    """
    Shows the template preview dialog and returns (approved, comment):
    approved is True if the RA approved the template, False if they want to
    try a different window; comment is the optional free text they typed.
    """
    app = QApplication.instance() or QApplication([])
    dialog = TemplatePreviewDialog(raw_beats, template_values, sampling_rate)
    dialog.exec()
    return dialog.approved, dialog.comment_edit.text().strip()
