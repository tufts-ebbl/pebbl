"""
The practice score window (certification, HLU 2026-10-04): after a trainee
saves a Step 2 review of the practice participant, PEBBL scores it against the
answer key and shows the report here, with a button that opens the trainee's
marks next to the answer key in the viewer (view only).
"""

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QApplication, QDialog, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
                             QVBoxLayout)


class PracticeReportDialog(QDialog):
    def __init__(self, heading, lines, passed):
        super().__init__()
        self.setWindowTitle("Practice score")
        self.setMinimumSize(720, 480)
        self.compare = False

        layout = QVBoxLayout(self)
        title = QLabel(heading)
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)
        verdict = QLabel("PASS: well done." if passed else
                         "Not yet: fix the lines marked FIX, then save again. To see where they are, compare your "
                         "marks with the answer key.")
        verdict.setWordWrap(True)
        layout.addWidget(verdict)

        report = QPlainTextEdit("\n".join(lines))
        report.setReadOnly(True)
        report.setFont(QFont("Consolas", 9))
        layout.addWidget(report)

        buttons = QHBoxLayout()
        compare = QPushButton("Compare with the answer key")
        compare.clicked.connect(self._on_compare)
        close = QPushButton("Close")
        close.setDefault(True)
        close.clicked.connect(self.reject)
        buttons.addWidget(compare)
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout.addLayout(buttons)

    def _on_compare(self):
        self.compare = True
        self.accept()


def show_practice_report(heading, lines, passed):
    """Shows the score report; True if the trainee asked to compare with the answer key."""
    app = QApplication.instance() or QApplication([])
    dialog = PracticeReportDialog(heading, lines, passed)
    dialog.exec()
    return dialog.compare
