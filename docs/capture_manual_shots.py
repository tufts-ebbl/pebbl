"""
Capture PEBBL screens from SYNTHETIC data for the RA manual (offscreen; no windows appear).

Writes docs/img/form.png, ecg_viewer.png, ecg_flip_rs.png, eda_viewer.png, rsp_hold.png, rsp_clip.png,
rsp_raw_swing.png, summary.png and
practice_report.png, for developers updating the manual. Run from the tool folder, with Windows fonts for Qt:

    QT_QPA_FONTDIR=C:/Windows/Fonts annotate_env/Scripts/python.exe docs/capture_manual_shots.py [form ecg ecgflip eda rsp rspraw summary report]

The RSP pictures show the RSP rule (HLU, 2026-10-05) on synthetic breathing, at times unlike the practice
recordings': a held breath and clipped tops are bad; a sigh and shallow breathing are real.

Then rebuild the PDF and the web page (docs/build_manual_pdf.py) and commit docs/.
"""
import io
import contextlib
import os
import shutil
import sys
import tempfile
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
TOOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the tool folder (docs/..)
sys.path.insert(0, TOOL)
OUT = os.path.join(TOOL, "docs", "img")
os.makedirs(OUT, exist_ok=True)

import mne  # noqa: E402
import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])
from PyQt6.QtGui import QFont  # noqa: E402
app.setFont(QFont("Segoe UI", 9))

import session_summary_gui  # noqa: E402
from annotation_io import run_stage_b  # noqa: E402
from channel_config import CHANNELS  # noqa: E402
from physio_io import generate_synthetic_demo  # noqa: E402
from simple_gui import ReviewLauncherDialog  # noqa: E402

which = sys.argv[1:] or ["form", "ecg", "ecgflip", "eda", "rsp", "rspraw", "summary", "report"]
ORIG_PLOT = mne.io.RawArray.plot
RSP_BAD = {}  # (onset, duration) in seconds of each RSP picture's bad_rsp stretch, set by its edit


def pump(n=5):
    for _ in range(n):
        app.processEvents()


def shot_form():
    import simple_gui
    import study_profile
    dlg = ReviewLauncherDialog()
    dlg.synthetic_checkbox.setChecked(False)
    # An example study's data folder (it doesn't exist here): the line under it shows what a real one would.
    dlg.box_path_edit.setText(r"C:\Users\you\Box\MyStudy\derivatives")
    example = study_profile.StudyProfile(name="mystudy", task="mystudy", path="example", sha256="example")
    with patch.object(simple_gui, "folder_status", return_value=("Study: mystudy", False, example)):
        dlg._apply_study_profile()
    dlg.local_path_edit.setText(r"C:\Users\you\PhysioWorking")
    dlg.subject_edit.setText("001")
    dlg.initials_edit.setText("abc")
    dlg.step2_radio.setChecked(True)
    dlg.run_combo.setCurrentText("1")
    dlg.channel_combo.setCurrentIndex(dlg._channel_keys.index("ecg"))
    dlg.show()
    dlg.adjustSize()
    pump()
    dlg.grab().save(os.path.join(OUT, "form.png"))
    dlg.close()


def viewer_shot(name, channels, start_sec, prepare=None, size=(1280, 720), before=None, edit_df=None):
    def fake_plot(self, *a, **k):
        k.update(block=False, show=False)
        if before:
            before(self)
        fig = ORIG_PLOT(self, *a, **k)
        fig.resize(*size)
        fig.show()
        pump()
        fig._fake_keypress("a")  # annotation mode, as an RA would press it
        pump()
        if prepare:
            prepare(self, fig)
            pump()
        fig.mne.plt.setXRange(start_sec, start_sec + 20, padding=0)
        pump(10)
        fig.grab().save(os.path.join(OUT, name))
        fig.close()
        return fig

    df, sfreq = generate_synthetic_demo(duration_sec=90)
    if edit_df:
        edit_df(df, sfreq)
    tmp = tempfile.mkdtemp()
    try:
        with patch.object(mne.io.RawArray, "plot", fake_plot), \
             patch.object(session_summary_gui, "show_session_summary", session_summary_gui.headless_decision()), \
             contextlib.redirect_stdout(io.StringIO()):
            run_stage_b(df, sfreq, {k: CHANNELS[k] for k in channels}, "abc", tmp, "sub-990_ses-run1_task-sdi",
                        "synthetic demo")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def eda_artifact(df, sfreq):
    # An electrode-lift artifact: the signal drops abruptly, then snaps back.
    a, b = int(24.0 * sfreq), int(25.4 * sfreq)
    eda = df["eda"].to_numpy(dtype=float).copy()
    eda[a:b] -= 0.9 * (eda.max() - eda.min())
    df["eda"] = eda


def mark_bad_eda(raw):
    raw.annotations.append(23.8, 1.8, "bad_eda")


def breaths(rsp, sfreq):
    """The breath tops and troughs (samples), and a typical breath's depth."""
    import neurokit2 as nk
    _, info = nk.rsp_peaks(rsp, sampling_rate=sfreq)
    tops = np.asarray([int(x) for x in info["RSP_Peaks"] if x == x])
    troughs = np.asarray([int(x) for x in info["RSP_Troughs"] if x == x])
    return tops, troughs, float(np.median(rsp[tops]) - np.median(rsp[troughs]))


def taper(length, sfreq, ease_sec=1.5):
    """0 at both ends rising to 1 over ease_sec, for changes that ease in and out."""
    out = np.ones(length)
    edge = min(int(ease_sec * sfreq), length // 3)
    out[:edge] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, edge))
    out[length - edge:] = out[:edge][::-1]
    return out


def rsp_hold_and_sigh(df, sfreq):
    # A sigh (one whole breath, 1.6 times as deep; the demo's EDA rises about 3 s after its top), then a
    # held breath after breathing in: flat for 6.5 s but for a small heartbeat ripple.
    rsp = df["rsp"].to_numpy(dtype=float).copy()
    tops, troughs, depth = breaths(rsp, sfreq)
    gaps = np.diff(troughs) / sfreq
    whole = np.flatnonzero((gaps >= 2.5) & (gaps <= 8.0))
    j = int(whole[np.argmin(np.abs(troughs[whole] / sfreq - 26.5))])
    a, b = troughs[j], troughs[j + 1]
    line = np.linspace(rsp[a], rsp[b], b - a)
    rsp[a:b] = line + (rsp[a:b] - line) * 1.6
    later = tops[tops > b + int(1.0 * sfreq)]
    h0 = int(later[np.argmin(np.abs(later / sfreq - 33.0))])
    h1 = h0 + int(6.5 * sfreq)
    hold = rsp[h0] + 0.02 * depth * np.sin(2 * np.pi * 1.2 * np.arange(h1 - h0) / sfreq)
    ramp = np.linspace(0, 1, int(0.5 * sfreq))
    rsp[h0:h1] = hold
    rsp[h1:h1 + len(ramp)] = hold[-1] * (1 - ramp) + rsp[h1:h1 + len(ramp)] * ramp
    RSP_BAD["hold"] = (h0 / sfreq, (h1 - h0) / sfreq)
    df["rsp"] = rsp


def rsp_clip_and_shallow(df, sfreq):
    # Two deep breaths cut flat at the belt's maximum (just above the run's highest top), from the trough
    # before the first cut top to the trough after the last; then shallow breathing at half depth.
    rsp = df["rsp"].to_numpy(dtype=float).copy()
    tops, troughs, depth = breaths(rsp, sfreq)
    c0, c1 = int(69.0 * sfreq), int(77.5 * sfreq)
    ceiling = float(max(rsp[:c0].max(), rsp[c1:].max())) + 0.03 * depth
    base = float(np.percentile(rsp[c0:c1], 5))
    gain = max(1.6, 1.25 * (ceiling - base) / max(float(np.percentile(rsp[c0:c1], 95)) - base, 1e-9))
    rsp[c0:c1] = np.minimum(base + (rsp[c0:c1] - base) * (1 + (gain - 1) * taper(c1 - c0, sfreq)), ceiling)
    cut = c0 + np.flatnonzero(rsp[c0:c1] >= ceiling)
    k0 = int(troughs[troughs <= cut[0]][-1])
    k1 = int(troughs[troughs >= cut[-1]][0])
    RSP_BAD["clip"] = (k0 / sfreq, (k1 - k0) / sfreq)
    s0, s1 = int(78.5 * sfreq), int(89.0 * sfreq)
    mean = float(np.mean(rsp[s0:s1]))
    rsp[s0:s1] = mean + (rsp[s0:s1] - mean) * (1 - 0.5 * taper(s1 - s0, sfreq))
    df["rsp"] = rsp


def rsp_raw_swing(df, sfreq):
    # The belt as recorded, the row a study with raw_rows = ["rsp_raw"] shows for RSP (HLU, 2026-11-08): the
    # demo's breathing, centred, plus at 48 s a sudden step down about three breaths deep that drifts back over
    # a few seconds (the recorder removes slow levels), as when the belt shifts at a task event. Marked from
    # just before the step to where the drift has mostly gone.
    rsp = df["rsp"].to_numpy(dtype=float)
    _, _, depth = breaths(rsp, sfreq)
    t = np.arange(len(rsp)) / sfreq
    t0 = 48.0
    step = np.where(t >= t0, -3.0 * depth * np.exp(-(t - t0) / 3.0), 0.0)
    df["rsp_raw"] = rsp - float(np.mean(rsp)) + step
    RSP_BAD["swing"] = (t0 - 0.3, 7.3)


def mark_bad_rsp(key):
    return lambda raw: raw.annotations.append(*RSP_BAD[key], "bad_rsp")


def shot_ecgflip():
    """
    Two synthetic ECGs side by side (urglab practice plan, 2026-11-08; HLU: "define rS beat"): one recorded
    upside down (tick Flip the ECG) and an upright one with rS beats (mark the small r). Drawn as plain traces,
    not viewer screenshots: at the viewer's scale the deep S runs off the row and the marks hide the small r.
    Beats from practice_urglab's shapes, 4 s at 70 beats a minute, so no practice recording's times appear.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import practice_urglab as pu
    fs, seconds = 1000, 4.0
    t = np.arange(int(seconds * fs)) / fs
    beats = np.arange(0.45, seconds - 0.4, 60 / 70)
    rng = np.random.default_rng(5)
    panels = [  # the manual's caption explains them (the T waves are the cue)
        ("Recorded upside down: tick Flip the ECG",
         -pu._waves(t, len(t), [(b, pu.UPRIGHT_BEAT) for b in beats])),
        ("Upright, with rS beats: don't flip; mark the small r",
         pu._waves(t, len(t), [(b, pu.RS_BEAT) for b in beats])),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 3.6), dpi=100, sharey=True)
    for ax, (title, ecg) in zip(axes, panels):
        ecg = ecg + rng.normal(0, 0.008, len(t))
        ax.plot(t, ecg, color="#8e1f86", linewidth=1.2)
        for b in beats:
            i = int(round(b * fs))
            ax.plot(t[i], ecg[i] + (0.12 if ecg[i] >= 0 else -0.12), marker="v" if ecg[i] >= 0 else "^",
                    color="#1f5fa8", markersize=8)
        ax.set_title(title, fontsize=12, loc="left")
        ax.set_xlabel("Time (s)")
        ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("ECG (mV)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ecg_flip_rs.png"))
    plt.close(fig)


def shot_summary():
    dlg = session_summary_gui.SessionSummaryDialog(
        "sub-001_ses-run1_task-sdi · ECG · abc",
        ["ECG: 3 peak(s) added, 2 removed", "ECG bad stretches: 1 added"],
        ["ECG: unusual gap near 312.4 s (1640 ms vs. typical 820 ms)"],
        5, False,
        "Tick this only when you've finished. Then set this channel to \"finished\" in the Google tracking "
        "spreadsheet (the terminal will remind you).")
    dlg.show()
    dlg.adjustSize()
    pump()
    dlg.grab().save(os.path.join(OUT, "summary.png"))
    dlg.close()


def shot_report():
    """The practice score window for a run-1 ECG review whose premature-beat mark was deleted."""
    import practice_certification as pc
    import practice_report_gui
    root = tempfile.mkdtemp(prefix="pebbl_practice_shot_")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            pc.make(root, seed=7)
        key = pc.load_key(pc.key_path_for(root))
        run1 = key["runs"]["1"]
        sub = {"initials": "abc", "channels": {
            "ecg": {"mode": "point", "indices": list(run1["channels"]["ecg"]["true_peaks"])},
            "bad_ecg": {"mode": "segment", "bad_segments": [[a["start"], a["end"]]
                                                            for a in run1["channels"]["bad_ecg"]["artifacts"]]}}}
        sub["channels"]["ecg"]["indices"].remove(run1["channels"]["ecg"]["traps"][0]["sample"])
        passed, _results, lines = pc.score_run(key, sub, "1", ["ecg", "bad_ecg"])
        dlg = practice_report_gui.PracticeReportDialog("Practice participant 990, run 1: ecg", lines, passed)
        dlg.setMinimumSize(640, 400)
        dlg.resize(640, 400)
        dlg.show()
        pump()
        dlg.grab().save(os.path.join(OUT, "practice_report.png"))
        dlg.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)


if "form" in which:
    shot_form()
if "report" in which:
    shot_report()
if "ecg" in which:
    viewer_shot("ecg_viewer.png", ["ecg"], 20)
if "ecgflip" in which:
    shot_ecgflip()
if "eda" in which:
    viewer_shot("eda_viewer.png", ["eda"], 20, before=mark_bad_eda, edit_df=eda_artifact)
if "rsp" in which:
    viewer_shot("rsp_hold.png", ["rsp"], 20, before=mark_bad_rsp("hold"), edit_df=rsp_hold_and_sigh)
    viewer_shot("rsp_clip.png", ["rsp"], 68, before=mark_bad_rsp("clip"), edit_df=rsp_clip_and_shallow)
if "rspraw" in which:
    import study_profile
    study_profile.activate(study_profile.StudyProfile(name="mystudy", task="mystudy", path="example",
                                                      sha256="example", raw_rows=["rsp_raw"]))
    try:
        viewer_shot("rsp_raw_swing.png", ["rsp"], 40, before=mark_bad_rsp("swing"), edit_df=rsp_raw_swing)
    finally:
        study_profile.activate(study_profile.BUILT_IN)
if "summary" in which:
    shot_summary()
print("saved:", sorted(os.listdir(OUT)))
