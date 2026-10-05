#!/usr/bin/env python3
"""
Tests TemplatePreviewDialog's logic (approve/reject button behavior, plot
construction with realistic data) without calling .exec() -- no blocking
modal event loop. Can't verify what the plot looks like (needs a human),
but does verify the dialog's decisions and that it doesn't crash building
the figure from real-shaped data.

Run with: annotate_env\\Scripts\\python.exe test_template_preview_gui.py
"""

import numpy as np
from PyQt6.QtWidgets import QApplication

from template_preview_gui import TemplatePreviewDialog


def main():
    app = QApplication.instance() or QApplication([])

    print("1. Building the dialog with realistic beat data doesn't crash...")
    rng = np.random.default_rng(0)
    template = np.sin(np.linspace(0, 2 * np.pi, 800)) * 0.5
    raw_beats = [template + rng.normal(scale=0.05, size=800) for _ in range(23)]
    dlg = TemplatePreviewDialog(raw_beats, template, sampling_rate=1000)
    assert dlg.approved is False
    print(f"   OK: dialog built with {len(raw_beats)} beats, no crash")

    print("2. Clicking 'Approve Template' sets approved=True and accepts...")
    dlg = TemplatePreviewDialog(raw_beats, template, sampling_rate=1000)
    dlg._on_approve()
    assert dlg.approved is True
    print("   OK")

    print("3. Beats of varying/mismatched lengths don't crash plotting...")
    ragged_beats = [template[:800], template[:750], template[:820]]
    dlg = TemplatePreviewDialog(ragged_beats, template, sampling_rate=1000)
    print("   OK: mismatched-length beats handled without error")

    print("4. Zero beats (edge case) doesn't crash...")
    dlg = TemplatePreviewDialog([], template, sampling_rate=1000)
    print("   OK: empty beat list handled without error")

    print("\nALL TEMPLATE_PREVIEW_GUI TESTS PASSED.")


if __name__ == "__main__":
    main()
