#!/usr/bin/env python3
"""
urglab certification practice: two simulated practice participants with known
answers, in the study's file format, and their answer key (pebbl-private
pebbl_urglab-practice-plan_20261101-2123.md and its addendum A1-A8, approved
by HLU on 2026-11-08).

    sub-990  an upright ECG with rS beats (a small upward r, then a deep S):
             mark the small r; don't tick "Flip the ECG".
    sub-991  an ECG recorded upside down: tick "Flip the ECG".

One run each (ses-run1), 180 s at 1000 Hz, task-urglab, in physiopipe's
format: its 12 columns and units, its sidecar keys, its reason names, and the
study's own RA texts (each column's Description, copied below from
physiopipe's writer-made reference sidecar, fixtures/urglab_reference_sidecar.json;
the answer key records each text's SHA-256, so a new text means a new key
version). practice_certification.py stays the single entry point and scorer:

    practice_certification.py make --out-dir <practice folder> --study-profile <urglab's pebbl_study.toml>

The recordings are never participant data: every signal is built here. COR
reuses the shapes in fixtures/cor_planted_75-140s.npz (physiopipe's planted,
synthetic corrugator; HLU, 2026-11-01: "Yes").

The computer's marks (each sidecar's MachineQC) are written here to imitate
physiopipe's rules (MACHINE_RULES below), including what those rules really
miss: the breathing `flat` mark covers only the middle of a dead belt, `rail`
guards 5 s of clear breathing around a moment at the limit, and the slow drop
and the held breath get no mark at all. physiopipe checks them against its
own rules before HLU's pilot (addendum A7). `snr` and `ecg_unconfirmed` are
always empty: a 180-s file can't trigger them under physiopipe's rules.

What RAs fix, keep and leave (plan answer 2, addendum A2):
  ECG   990: missed beats, starting marks on the S, the rebound after it and
        on T waves (some T waves are taller than the r), a noise burst to
        mark bad_ecg; keep the premature beat and the computer's lead-off
        mark. 991: tick Flip; missed beats, marks on T waves, a noise burst.
  PPG   990: missed pulses, marks on the dicrotic wave and on a slope; a hand
        movement and the weak pulse after the premature beat, both bad_ppg;
        keep the computer's mark where the sensor hit its limit (its 3-s guard
        is neutral, addendum A8). 991: clean (scored, not certified).
  RSP   (judged on rsp_raw; the hidden first and last 15 s and the drift back
        after a step are neutral, addendum A3)
        990: extend the partial `flat` over a noise-only dead belt; add a mark
        on a slow 3.5-V drop at a task event; keep the `swing` on a fast
        drop-and-return; leave a sigh.
        991: add a mark on a held breath; trim `rail` to a 40-ms pop and to
        the belt held at its limit then jumping; leave shallow breathing.
  EDA   990: keep the computer's marks on an electrode lift and a stuck
        sensor; leave real responses. No brief-glitch case (addendum A5).
        991: real responses only (scored, not certified).
  COR   990: mark electrode pops (a train; a single small one is a miss, not
        a fail), a regular repeating pattern, and a perfectly smooth stretch
        that starts right after a spike; leave frowns (at most 10% marked),
        ordinary bursts and a quiet stretch that fades in gradually; keep the
        computer's mark on a frown that reached the sensor limit.
        991: the whole run is a dead channel (`flat_record`): keep it.
"""

import os

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, sosfiltfilt

import practice_certification as pc

FS = pc.FS
DURATION_SEC = 180
N_SAMPLES = DURATION_SEC * FS
SUBJECTS = ("990", "991")
RUN = "1"
TASK = "urglab"
KEY_VERSION = pc.KEY_VERSIONS["urglab"]
FILE_MARGIN_SEC = 5.0  # each practice tsv sits 5 s inside its (imagined) recording file, as physiopipe's writer
#                        records in Provenance.SourceFiles (physiopipe, 2026-11-08: filter_edge counts from the
#                        recording file's ends, so RSP's 20-s filter_edge is the tsv's first and last 15 s)
RSP_EDGE_SEC = 20.0 - FILE_MARGIN_SEC
BELT_LIMIT_V = 10.0
BELT_TAU_SEC = 3.0  # the belt's AC coupling (physiopipe's synthetic file): a lasting step drifts back
DRIFT_SEC = 3 * BELT_TAU_SEC  # the drift back after a step: neutral in scoring (addendum A3)
CRITERIA = dict(pc.PASS_CRITERIA, trap_max_coverage_by_channel={"emg_cor": 0.10})  # HLU, 2026-11-08 (plan 12c)

COLUMNS = ["ecg", "ppg", "rsp", "eda", "emg_cor", "emg_cor_env", "ecg_peaks", "ppg_peaks", "rsp_peaks",
           "eda_peaks", "event", "rsp_raw"]
TECHNICAL_DESCRIPTION = "Practice file, simulated by PEBBL (practice_urglab.py): not participant data."

# The study's texts, word for word, from physiopipe's writer-made reference sidecar
# (fixtures/urglab_reference_sidecar.json: physiopipe d9c276d, urglab physio cc6a468, profile 21078f4f17b8).
# test_urglab_practice.py checks they still match it.
COLUMN_TEXTS = {
    'ecg': ('mV',
        "The heart's electrical signal (ECG), filtered to remove electrical noise from the room. Each sharp "
        'spike is one heartbeat. Stretches where the signal is flat (no heartbeat recorded) are already '
        'marked as bad.'),
    'ppg': ('V',
        "Finger pulse signal, filtered so that each heartbeat's pulse is easy to see. Stretches are already "
        'marked as bad where the pulse is too small to see, the sensor hit its limit, or the pulses do not '
        'follow the heartbeats in the ECG; check each one. Pulse marks inside a bad stretch are not shown: if'
        ' you delete a bad stretch, add the pulse marks there by hand.'),
    'rsp': ('V',
        "Breathing belt around the participant's chest. Review it on rsp_raw, the belt as recorded; this "
        'filtered copy is for analysis. Each rise and fall is one breath. Some stretches are already marked '
        'as bad: where the belt is flat or at its limit, and where it makes a sudden step much faster and '
        'bigger than a breath. Remove a mark where ordinary breathing is clearly visible. Then mark (1) any '
        'stretch with no breathing in it, a flat line or only fine noise, from where the breathing stops to '
        'where it returns, even where part of it is already marked; (2) any other large sudden drop or jump '
        'that is not breathing, often at a task event. If the belt never follows the breathing seen in the '
        'heartbeat, the whole recording is marked bad; leave that mark.'),
    'eda': ('microsiemens',
        'Skin conductance (the sweat response). Rises lasting a few seconds are normal responses. Stretches '
        'are already marked as bad where the signal is out of range, flat, or changes impossibly fast, and in'
        ' the first and last second of each recording; brief glitches are not marked.'),
    'emg_cor': ('mV',
        'Muscle activity over the eyebrow (corrugator, the frown muscle), filtered to keep the muscle signal '
        'and remove electrical noise. Bursts are normal, including large ones and ones with movement on other'
        ' channels; so are quiet stretches and held frowns (activity lasting seconds that builds and fades, '
        'with fine texture after it). Stretches the computer found are already marked: leave them; they count'
        ' whatever you do. Mark as bad: (1) electrode pops: on emg_cor_full, sharp spikes far taller than the'
        ' bursts around them, often in trains; mark from just before the first spike to the end of the '
        'ringing; (2) a regular, repeating pattern lasting several seconds that is not already marked; (3) on'
        ' emg_cor, a stretch that is perfectly smooth (no fine texture), only when it begins right after a '
        'sudden step, a spike or the sudden end of a band; mark it up to the next sudden change or the end of'
        ' the recording. Do not mark a stretch because it is quiet.'),
    'emg_cor_env': ('mV',
        'The size of the eyebrow muscle activity, smoothed, for analysis. Not reviewed.'),
    'ecg_peaks': ('boolean',
        'Heartbeat marks found automatically, used as the starting marks when building the ECG template; the '
        'review tool then finds its own.'),
    'ppg_peaks': ('boolean',
        'Marks at the top of each pulse, found automatically, to be checked and corrected by RAs.'),
    'rsp_peaks': ('boolean',
        'Marks at the top of each breath, found automatically. Not reviewed by RAs, who mark bad stretches of'
        ' the breathing signal instead.'),
    'eda_peaks': ('boolean',
        'Skin-conductance responses found automatically (rises of at least 0.05 microsiemens lasting at least'
        ' half a second). Not reviewed by RAs.'),
    'event': ('code',
        'Task code at each moment. 0 = no code. 50 = resting baseline. Trial codes have two digits: the first'
        ' is the condition (1 = urgent, 2 = not urgent), the second the phase (0 = fixation, 1 = urgency '
        'message, 2 = planning, 3 = tracing, 4 = DR grid, 5 = CT grid). Codes 10, 20 and 50 also appear as '
        'half-second test pulses when the recording is checked.'),
    'rsp_raw': ('V',
        'The breathing belt as recorded, without the breathing filter. Use it to judge large swings: a sudden'
        ' drop or jump much bigger than a breath, often at a task event, then a slow drift back, is not '
        'breathing. Bad stretches marked for breathing cover this row too.'),
}
MACHINE_RULES = {
    'ecg': (
        "Bad where the cleaned ECG is flat (0.5-s windows from each file's first sample with SD < 0.001 mV). "
        'junction: the one-sample join of two files. processing_failed: the channel could not be processed.'),
    'ppg': (
        'Bad where any reason holds: window_rule (pulse too small in a 10-s window), rail (within 3 s of the '
        "sensor limit), ecg_coupling (pulses do not follow the ECG's heartbeats; tested only where the ECG is"
        ' usable). junction and processing_failed as for every channel.'),
    'rsp': (
        'Bad where any reason holds: flat, flat_raw (dead belt, 2-s windows), snr (20-s windows), rail '
        '(within 5 s of the sensor limit), swing (within 1 s of a sudden belt step faster and bigger than a '
        'breath), filter_edge (the first and last 20 s of each recording file, so a flag can end at the file '
        "end), ecg_unconfirmed (the belt does not follow the ECG's breathing pattern; whole trim; judged only"
        ' where the ECG is usable). junction and processing_failed as for every channel.'),
    'eda': (
        'Bad where any reason holds: kleckner (out of range or too fast), ceiling, floor, flat, filter_edge '
        '(the first and last 1 s of each recording file, so a flag can end at the file end), too_short. '
        'junction and processing_failed as for every channel.'),
    'emg_cor': (
        'Bad where any reason holds: clipped and clip_guard (sensor limit, +-1.5 s), nonfinite and its guard,'
        ' mains (electrical noise left after filtering), line (narrowband interference lasting at least 3 of '
        '5 neighbouring 2-s windows), filter_edge (the first and last 1.5 s of each recording file, so a flag'
        ' can end at the file end), flat_record (the whole file: at least 90 % of the 2-s windows in the trim'
        ' are flat, a dead channel), clipped_record (the whole file: at least 90 % of the trim at the sensor '
        'limit), too_short. No reason detects missing electrodes. Pass 2 applies every machine stretch '
        'whatever the RA does. junction and processing_failed as for every channel.'),
}
COORDINATE_SPACE = 'tsv row index at SamplingFrequency; segments are [start, end)'
CODEBOOK = {"bit_order": "msb_first", "baseline_code": 50, "cleared_code": 0, "test_burst_codes": [10, 20, 50],
            "test_burst_pulse_s": 0.5, "test_burst_spacing_s": 1.0, "phase_presented_min_ms": 50.0,
            "conditions": {"1": "Urgent", "2": "Not Urgent"},
            "phases": {"0": "fixation", "1": "urgency_message", "2": "planning", "3": "tracing", "4": "dr_grid",
                       "5": "ct_grid"}}
# Every reason of each channel's module, in physiopipe's order: always present, empty when nothing was found.
REASONS = {
    "ecg": ["flat"],
    "ppg": ["window_rule", "rail", "ecg_coupling"],
    "rsp": ["flat", "flat_raw", "snr", "rail", "swing", "filter_edge", "ecg_unconfirmed"],
    "eda": ["kleckner", "ceiling", "floor", "flat", "filter_edge", "too_short"],
    "emg_cor": ["nonfinite", "nonfinite_guard", "clipped", "clip_guard", "mains", "line", "filter_edge",
                "flat_record", "clipped_record", "too_short"],
}

# The task's timeline, as in physiopipe's synthetic file: 20 s before the task (0), a 60-s baseline (50), then
# 34-s trials alternating urgent (1x) and not urgent (2x), phases 0-5 then 1 s cleared (0).
BASELINE = (20.0, 80.0)
TRIAL_STARTS = (82.0, 116.0, 150.0)
PHASE_SECONDS = (2.0, 3.0, 5.0, 15.0, 4.0, 4.0)  # fixation, urgency message, planning, tracing, DR grid, CT grid
CLEARED_SEC = 1.0

# Where things are, in tsv seconds (hand-placed; at least 3 s apart within a channel; nothing in RSP's hidden
# first and last 15 s). Breathing steps sit at phase changes: planning->tracing at 92, 126 and 160 s,
# tracing->DR grid at 107, 141 and 176 s.
PLANS = {
    "990": {
        "ecg": {"kind": "rS", "heart_rate": 68, "pvc_at": 95.0, "flat": (140.0, 3.0), "noise": (118.0, 4.0),
                "missed": (22.0, 61.0, 150.0), "on_s": (35.0, 128.0), "on_rebound": (88.0,), "on_t": (50.0, 170.0)},
        "ppg": {"movement": (70.0, 4.0), "saturated": (157.0, 1.5), "missed": (30.0, 100.0, 133.0),
                "on_dicrotic": (45.0, 110.0), "on_slope": (125.0,)},
        "rsp": {"dead": (25.0, 65.0), "sigh": 72.0, "fast_swing": 92.0, "slow_drop": 126.0},
        "eda": {"scrs": (20.0, 58.0, 101.0, 146.0, 165.0), "lift": (40.0, 2.5), "stuck": (128.0, 5.0)},
        "emg_cor": {"bursts": (22.0, 140.0), "frown_long": (30.0, 8.0), "pop_single": 45.0, "quiet_fade": 52.0,
                    "frown_fixture": 62.0, "pattern": (75.0, 5.0), "pop_train": 88.0, "dead": (100.0, 6.0),
                    "frown_clipped": (118.0, 6.0)},
    },
    "991": {
        "ecg": {"kind": "inverted", "heart_rate": 62, "pvc_at": None, "flat": (100.0, 2.5), "noise": (60.0, 4.0),
                "missed": (30.0, 90.0, 140.0), "on_s": (), "on_rebound": (), "on_t": (45.0, 120.0)},
        "ppg": {"movement": None, "saturated": None, "missed": (), "on_dicrotic": (), "on_slope": ()},
        "rsp": {"shallow": (25.0, 15.0), "hold": (50.0, 8.0), "limit": 107.0, "pop": 135.0},
        "eda": {"scrs": (30.0, 75.0, 120.0, 160.0), "lift": None, "stuck": None},
        "emg_cor": {"dead_channel": True},
    },
}

COR_FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "cor_planted_75-140s.npz")


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _s(sec):
    return int(round(sec * FS))


def _merge(spans):
    """Sorted, disjoint, half-open [start, end) spans: physiopipe's span convention."""
    out = []
    for a, b in sorted((int(a), int(b)) for a, b in spans if b > a):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _clip_span(a, b, n):
    return [max(0, int(a)), min(n, int(b))]


def _band(x, lo, hi, order=3):
    return sosfiltfilt(butter(order, [lo, hi], btype="bandpass", fs=FS, output="sos"), x)


def _highpass(x, cutoff, order=2):
    return sosfiltfilt(butter(order, cutoff, btype="highpass", fs=FS, output="sos"), x)


def _lowpass(x, cutoff, order=3):
    return sosfiltfilt(butter(order, cutoff, btype="lowpass", fs=FS, output="sos"), x)


def _event_column(n):
    """(event codes per row, Events.Rows) on the task timeline."""
    event = np.zeros(n, dtype=int)
    rows = []

    def row(onset, offset, code, kind, condition=None, phase=None, trial=None, complete=None):
        a, b = _s(onset), min(n, _s(offset))
        if a >= n:
            return
        event[a:b] = code
        margin = _s(FILE_MARGIN_SEC)
        rows.append({"Role": "task", "OnsetRow": a, "OffsetRow": b, "OnsetEventSample": (a + margin) * 10,
                     "OffsetEventSample": (b + margin) * 10, "Code": code, "Kind": kind, "Condition": condition,
                     "Phase": phase, "Trial": trial, "Block": None, "TrialComplete": complete, "Flags": ""})

    row(*BASELINE, 50, "baseline")
    for k, start in enumerate(TRIAL_STARTS):
        condition = 1 if k % 2 == 0 else 2
        complete = start + sum(PHASE_SECONDS) + CLEARED_SEC <= DURATION_SEC
        t = start
        for phase, length in enumerate(PHASE_SECONDS):
            row(t, t + length, 10 * condition + phase, "phase", CODEBOOK["conditions"][str(condition)],
                CODEBOOK["phases"][str(phase)], k + 1, complete)
            t += length
    return event, rows


# --------------------------------------------------------------------------
# Heartbeats, ECG and PPG
# --------------------------------------------------------------------------

# Beat shapes (centre s from the r/R, width s, height mV), Gaussian waves.
RS_BEAT = [(-0.16, 0.022, 0.08), (0.0, 0.009, 0.32), (0.042, 0.010, -1.10), (0.066, 0.009, 0.15),
           (0.30, 0.060, 0.25)]  # P, small r, deep S (42 ms after the r), the rebound (66 ms), T
UPRIGHT_BEAT = [(-0.16, 0.022, 0.12), (-0.022, 0.008, -0.10), (0.0, 0.010, 1.00), (0.026, 0.010, -0.25),
                (0.30, 0.060, 0.30)]  # P, q, R, S, T (991's file holds its negative)
PVC_BEAT = [(0.0, 0.028, 0.75), (0.07, 0.030, -0.45), (0.30, 0.090, -0.40)]  # wide QRS, T the other way
TALL_T = 0.45  # some rS beats' T waves are taller than the r (plan, "What changed" item 2)


def _beat_times(rng, heart_rate, pvc_at):
    times, t = [], 0.35
    while t < DURATION_SEC - 0.6:
        times.append(t)
        t += 60.0 / heart_rate * (1 + 0.03 * np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, 0.015))
    times = np.asarray(times)
    pvc = None
    if pvc_at is not None:  # a premature beat replaces beat k; the next beat keeps its time (compensatory pause)
        pvc = int(np.clip(np.argmin(np.abs(times - pvc_at)), 1, len(times) - 2))
        times[pvc] = times[pvc - 1] + pc.PVC_COUPLING * (times[pvc] - times[pvc - 1])
    return times, pvc


def _waves(t, n, centres_and_shapes):
    x = np.zeros(n)
    for centre, shape in centres_and_shapes:
        lo, hi = max(0, _s(centre - 0.4)), min(n, _s(centre + 0.6))
        tt = t[lo:hi]
        for offset, width, height in shape:
            x[lo:hi] += pc._gauss(tt, centre + offset, width, height)
    return x


def _refine(x, index, half):
    lo, hi = max(0, index - half), min(len(x), index + half + 1)
    return lo + int(np.argmax(x[lo:hi]))


def _build_ecg(rng, plan, n, t, beats, pvc):
    """(file ECG, true r/R samples, the premature beat's sample or None, bad_ecg artifacts, machine flat span)."""
    import neurokit2 as nk
    shapes = []
    for i, b in enumerate(beats):
        if i == pvc:
            shapes.append((b, PVC_BEAT))
        elif plan["kind"] == "rS":
            beat = list(RS_BEAT)
            if rng.random() < 0.25:
                beat[-1] = (0.30, 0.060, TALL_T)
            shapes.append((b, beat))
        else:
            shapes.append((b, UPRIGHT_BEAT))
    clean = _waves(t, n, shapes)
    sign = -1.0 if plan["kind"] == "inverted" else 1.0
    raw = sign * clean + 0.04 * np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, 0.008, n)
    ecg = _highpass(raw, 0.5)  # physiopipe's ECG cleaning starts with a 0.5-Hz high-pass
    upright = sign * ecg  # the ECG as an RA sees it after flipping (991) or as it is (990)

    true_r = []
    for i, b in enumerate(beats):
        idx = _s(b)
        if 300 <= idx < n - 600:
            true_r.append(_refine(upright, idx, 30 if i == pvc else 5))
    pvc_r = _refine(upright, _s(beats[pvc]), 30) if pvc is not None else None

    nlo, nhi = _s(plan["noise"][0]), _s(sum(plan["noise"]))
    burst = rng.normal(0, 1, nhi - nlo)
    ecg[nlo:nhi] += burst / burst.std() * 0.45 + _lowpass(rng.normal(0, 1, nhi - nlo), 3.0) * 1.5
    flo, fhi = _s(plan["flat"][0]), _s(sum(plan["flat"]))
    ecg[flo:fhi] = ecg[flo]  # lead off: dead flat (SD 0 < physiopipe's 0.001 mV)
    true_r = sorted(p for p in true_r if not flo <= p < fhi)  # no heartbeat recorded there
    artifacts = [
        {"type": "heavy noise: R peaks can't be found here (mark it bad_ecg)", "start": nlo, "end": nhi},
        {"type": "the ECG is flat (lead off): the computer's mark is right, keep it", "start": flo, "end": fhi},
    ]
    detected = [int(p) for p in nk.ecg_peaks(ecg, sampling_rate=FS, method="nabian2018")[1]["ECG_R_Peaks"]]
    detected = [_refine(ecg, p, 2) for p in detected if not flo <= p < fhi]  # physiopipe snaps within 2 samples
    return ecg, upright, true_r, pvc_r, artifacts, detected, [flo, fhi]


def _ecg_seed(rng, plan, upright, true_r, pvc_r, detected):
    """The ecg_peaks column: the detector's own marks plus planted mistakes (plan answer 2)."""
    misplaced = [(x, 0.042) for x in plan["on_s"]] + [(x, 0.066) for x in plan["on_rebound"]]
    return pc._seed_with_errors(upright, true_r, detected, plan["missed"], [(x, 0.30) for x in plan["on_t"]],
                                misplaced, [pvc_r] if pvc_r is not None else [], rng)


def _build_ppg(rng, plan, n, t, beats, pvc):
    """Biopac finger pulse (one systolic crest, then the dicrotic wave), 0.25 s after each heartbeat."""
    centres = beats + 0.25 + rng.normal(0, 0.004, len(beats))
    shapes = []
    for i, c in enumerate(centres):
        amp = 0.4 if i == pvc else 1.0 + 0.15 * np.sin(2 * np.pi * 0.25 * c)  # breathing varies the height
        shapes.append((c, [(0.0, 0.07, 0.5 * amp), (0.30, 0.06, 0.14 * amp)]))
    ppg = _waves(t, n, shapes) + 0.05 * np.sin(2 * np.pi * 0.1 * t) + rng.normal(0, 0.005, n)
    artifacts, neutral, rail = [], [], []
    if plan["movement"]:
        m0, m1 = _s(plan["movement"][0]), _s(sum(plan["movement"]))
        ppg[m0:m1] += _lowpass(rng.normal(0, 1, m1 - m0), 2.5) * 1.2 * pc._taper(m1 - m0, 0.3)
        artifacts.append({"type": "hand movement (big, irregular waves): mark it bad_ppg", "start": m0, "end": m1})
    ppg = _band(ppg, 0.5, 8.0)  # physiopipe's 'elgendi' clean
    if plan["saturated"]:
        s0, s1 = _s(plan["saturated"][0]), _s(sum(plan["saturated"]))
        limit = float(ppg[s0:s1].max()) * 0.6
        ppg[s0:s1] = np.minimum(ppg[s0:s1], limit)
        at_limit = s0 + np.flatnonzero(ppg[s0:s1] >= limit)
        a, b = int(at_limit[0]), int(at_limit[-1]) + 1
        rail = _clip_span(a - _s(3.0), b + _s(3.0), n)
        artifacts.append({"type": "the sensor hit its limit (flat tops): the computer's mark is right, keep it",
                          "start": a, "end": b})
        neutral += [[rail[0], a], [b, rail[1]]]  # the rail's 3-s guard: neither required nor penalized (A8)

    tops = []
    for c in centres:
        idx = _s(c)
        if 300 <= idx < n - 600:
            tops.append(_refine(ppg, idx, 40))
    tops = np.asarray(sorted(set(tops)))
    pvc_top = int(tops[np.argmin(np.abs(tops - _s(centres[pvc])))]) if pvc is not None else None
    if pvc_top is not None:  # the weak, odd pulse after the premature beat, trough to the end of its own wave
        j = int(np.searchsorted(tops, pvc_top))
        w0 = int(tops[j - 1] + np.argmin(ppg[tops[j - 1]:pvc_top]))
        w1 = int(min(pvc_top + 0.45 * FS, pvc_top + np.argmin(ppg[pvc_top:tops[j + 1]])))
        artifacts.append({"type": "the weak, odd pulse after the premature beat: mark it bad_ppg", "start": w0,
                          "end": w1})
    detected = list(tops)
    if plan["movement"]:  # the detector also finds 'pulses' in the movement
        m0, m1 = _s(plan["movement"][0]), _s(sum(plan["movement"]))
        crests = pc._local_maxima(ppg, m0, m1)
        detected += [int(c) for c in sorted(crests, key=lambda i: -ppg[i])[:3]]
    seed = pc._seed_with_errors(ppg, tops, detected, plan["missed"], [(x, 0.30) for x in plan["on_dicrotic"]],
                                [(x, -0.06) for x in plan["on_slope"]], [pvc_top] if pvc_top is not None else [],
                                rng)
    return ppg, tops, seed, artifacts, neutral, rail


# --------------------------------------------------------------------------
# The breathing belt (addendum A1)
# --------------------------------------------------------------------------

def _build_belt(rng, plan, n, t, heart_rate):
    """
    rsp_raw: the belt as recorded. Ordinary breathing (about 15 a minute, about 1 V per breath, varying a
    little), through a first-order AC coupling with a 3-s time constant (so a lasting step drifts back to the
    middle), clipped at the +-10 V limit, plus 1 mV of sensor noise. Returns the belt, its cases for the key,
    and the computer's marks by reason.
    """
    rate = 0.25 + 0.02 * np.sin(2 * np.pi * t / 47.0) + 0.01 * _lowpass(rng.normal(0, 1, n), 0.05) / 0.05
    rate = np.clip(rate, 0.18, 0.33)
    gate = np.ones(n)
    freeze = np.zeros(n, dtype=bool)
    artifacts, traps, neutral, reasons = [], [], [], {r: [] for r in REASONS["rsp"]}

    phase0 = 2 * np.pi * np.cumsum(rate) / FS
    if "hold" in plan:  # breathing stops after an in-breath (sin = 1), then picks up where it left off
        h_start, h_len = plan["hold"]
        peaks = np.flatnonzero(np.diff(np.floor((phase0 - np.pi / 2) / (2 * np.pi))) > 0) + 1
        h0 = int(peaks[np.argmin(np.abs(peaks - _s(h_start)))])
        h1 = h0 + _s(h_len)
        freeze[h0:h1] = True
    phase = 2 * np.pi * np.cumsum(np.where(freeze, 0.0, rate)) / FS
    amp = 0.5 * (1 + 0.12 * np.sin(2 * np.pi * t / 31.0)) * (1 + 0.05 * _lowpass(rng.normal(0, 1, n), 0.2) / 0.1)
    breathing = amp * (np.sin(phase) + 0.15 * np.sin(2 * phase))
    cycles = np.flatnonzero(np.diff(np.floor(phase / (2 * np.pi))) > 0) + 1  # each breath starts at sin = 0

    def cycle_at(sec):
        k = int(np.searchsorted(cycles, _s(sec)))
        return int(cycles[k - 1]), int(cycles[k])

    def boundary_after(sec):
        return int(cycles[int(np.searchsorted(cycles, _s(sec)))])

    if "dead" in plan:  # a noise-only dead belt: breathing stops at a breath's end and returns at one's start
        a, b = boundary_after(plan["dead"][0]), boundary_after(plan["dead"][1])
        gate[a:b] = 0.0
        artifacts.append({"type": "no breathing (the belt shows only fine noise): mark it from where the "
                                  "breathing stops to where it returns", "start": a, "end": b})
        # physiopipe's flat (2-s windows, counted from the file's start, so odd tsv seconds) misses about 10 s at
        # the start and 16 s at the end of a noise-only stretch (its synthetic file: 25-75 s -> 35-59 s).
        f0 = int(np.ceil((a / FS + 10.0 - 1) / 2) * 2 + 1)
        f1 = int(np.floor((b / FS - 16.0 - 1) / 2) * 2 + 1)
        if f1 > f0:
            reasons["flat"].append([_s(f0), _s(f1)])
    if "sigh" in plan:
        a, b = cycle_at(plan["sigh"])
        breathing[a:b] *= 2.5
        traps.append({"type": "a sigh (one deep breath): real breathing, leave it", "start": a, "end": b})
    if "shallow" in plan:
        s0, s1 = _s(plan["shallow"][0]), _s(sum(plan["shallow"]))
        gate[s0:s1] = 1 - 0.65 * pc._taper(s1 - s0)
        traps.append({"type": "shallow breathing: real breathing, leave it", "start": s0, "end": s1})
    if "hold" in plan:
        artifacts.append({"type": "a held breath (no breathing): mark it", "start": h0, "end": h1})

    tension = breathing * gate
    if "hold" in plan:  # a faint heartbeat ripple while the breath is held: 'only fine noise'
        tension[h0:h1] += 0.01 * np.sin(2 * np.pi * heart_rate / 60.0 * t[h0:h1])
    steps = np.zeros(n)

    def edge(a, b, height, ramp_sec):
        r = max(1, _s(ramp_sec))
        steps[a:a + r] += height * (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, r)))
        steps[a + r:b] += height
        if b < n:
            steps[b:b + r] += height * (0.5 + 0.5 * np.cos(np.linspace(0, np.pi, r)))

    def swing_mark(a, b):
        reasons["swing"].append(_clip_span(a - _s(1.0), b + _s(1.0), n))

    if "fast_swing" in plan:  # a 4-V drop that returns 0.7 s later, at a phase change
        a, b = _s(plan["fast_swing"]), _s(plan["fast_swing"] + 0.7)
        edge(a, b, -4.0, 0.03)
        swing_mark(a, b)
        artifacts.append({"type": "a sudden 4-V drop and return at a task event: keep the computer's mark",
                          "start": a, "end": b})
    if "slow_drop" in plan:  # 3.5 V over 0.8 s, too slow for the swing rule; it stays, so the belt drifts back
        a, b = _s(plan["slow_drop"]), _s(plan["slow_drop"] + 0.8)
        steps[a:b] += -3.5 * (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, b - a)))
        steps[b:] += -3.5
        artifacts.append({"type": "a large slow drop at a task event (not breathing; the computer missed it): "
                                  "mark it", "start": a, "end": b})
        neutral.append([b, min(n, b + _s(DRIFT_SEC))])
    if "limit" in plan:  # the belt held at its limit about 1.3 s, then a jump past zero
        a, b = _s(plan["limit"]), _s(plan["limit"] + 1.3)
        edge(a, b, -16.0, 0.05)
        swing_mark(a, b)
        artifacts.append({"type": "the belt held at its limit, then a jump: keep the mark there, remove it "
                                  "over the clear breathing around it", "start": a, "end": b + _s(0.8)})
        neutral.append([b + _s(0.8), min(n, b + _s(0.8) + _s(DRIFT_SEC))])
    if "pop" in plan:  # one 40-ms pop to the limit amid ordinary breathing, not at a task event
        a, b = _s(plan["pop"]), _s(plan["pop"] + 0.04)
        steps[a:b] += -14.0
        swing_mark(a, b)
        artifacts.append({"type": "a 40-ms pop to the belt's limit: keep the mark there, remove it over the "
                                  "clear breathing around it", "start": a, "end": b})

    alpha = BELT_TAU_SEC / (BELT_TAU_SEC + 1.0 / FS)
    coupled = lfilter([alpha, -alpha], [1.0, -alpha], tension + steps - (tension + steps)[0])
    belt = np.clip(coupled, -BELT_LIMIT_V, BELT_LIMIT_V) + rng.normal(0, 0.001, n)
    at_limit = np.flatnonzero(np.abs(coupled) >= BELT_LIMIT_V)
    if len(at_limit):  # rail: within 5 s of any sample at the limit
        runs = np.split(at_limit, np.flatnonzero(np.diff(at_limit) > 1) + 1)
        reasons["rail"] = _merge([_clip_span(r[0] - _s(5.0), r[-1] + 1 + _s(5.0), n) for r in runs])
    edge_len = _s(RSP_EDGE_SEC)
    reasons["filter_edge"] = [[0, edge_len], [n - edge_len, n]]
    neutral += [[0, edge_len], [n - edge_len, n]]  # hidden from RAs (urglab hidden_machine_reasons)
    reasons = {r: _merge(spans) for r, spans in reasons.items()}
    breath_tops = np.flatnonzero(np.diff(np.floor((phase - np.pi / 2) / (2 * np.pi))) > 0) + 1
    breath_tops = [int(p) for p in breath_tops if gate[p] > 0.5 and not freeze[p]]
    return belt, breath_tops, artifacts, traps, neutral, reasons


# --------------------------------------------------------------------------
# EDA and COR
# --------------------------------------------------------------------------

def _build_eda(rng, plan, n, t):
    eda = 6.0 + 0.3 * np.sin(2 * np.pi * t / 150) + 0.003 * _lowpass(rng.normal(0, 1, n), 1.0) / 0.03
    traps, artifacts, reasons = [], [], {r: [] for r in REASONS["eda"]}
    scr_tops = []
    for onset in plan["scrs"]:
        amp = rng.uniform(0.3, 0.7)
        tt = np.clip(t - onset, 0, None)
        eda += amp * 1.8 * (np.exp(-tt / 4.0) - np.exp(-tt / 0.75)) * (t >= onset)
        traps.append({"type": "a skin conductance response: real, leave it", "start": _s(onset),
                      "end": _s(onset + 6)})
        scr_tops.append(_s(onset + 1.5))
    eda = _lowpass(eda + rng.normal(0, 0.002, n), 3.0, order=4)  # physiopipe: neurokit's 3-Hz low-pass
    if plan["lift"]:  # an electrode lift: the reading drops below the valid range, then comes back
        l0, l1 = _s(plan["lift"][0]), _s(sum(plan["lift"]))
        eda[l0:l1] = 0.02 + rng.normal(0, 0.001, l1 - l0)
        reasons["kleckner"].append([l0, l1])
        artifacts.append({"type": "electrode lift (out of range): the computer's mark is right, keep it",
                          "start": l0, "end": l1})
    if plan["stuck"]:
        f0, f1 = _s(plan["stuck"][0]), _s(sum(plan["stuck"]))
        eda[f0:f1] = eda[f0]
        reasons["flat"].append([f0, f1])
        artifacts.append({"type": "stuck sensor (flat line): the computer's mark is right, keep it",
                          "start": f0, "end": f1})
    return eda, scr_tops, artifacts, traps, reasons


def _paste(x, start, segment, fade=20):
    """Puts segment into x at start, cross-fading `fade` samples at each end."""
    seg = np.asarray(segment, dtype=float).copy()
    end = start + len(seg)
    w = np.ones(len(seg))
    w[:fade] = np.linspace(0, 1, fade)
    w[-fade:] = np.linspace(1, 0, fade)
    x[start:end] = x[start:end] * (1 - w) + seg * w


def _build_cor(rng, plan, n, t):
    """(emg_cor in mV, artifacts, traps, kept machine stretches, reasons) for one participant."""
    reasons = {r: [] for r in REASONS["emg_cor"]}
    if plan.get("dead_channel"):  # no electrodes attached: the whole run is a dead channel
        cor = rng.normal(0, 1e-5, n)
        reasons["flat_record"] = [[0, n]]
        return cor, [], [], [{"type": "a dead channel (the whole run)", "start": 0, "end": n}], reasons

    z = np.load(COR_FIXTURE)
    fx, f_start = z["emg_cor"].astype(float), float(z["start_s"])
    spans = {str(name): (int((a - f_start) * FS), int((b - f_start) * FS))
             for name, (a, b) in zip(z["event_names"], z["event_spans_s"])}
    burst_shape = fx[slice(*next(v for k, v in spans.items() if k.startswith("ordinary burst")))]
    frown_shape = fx[slice(*next(v for k, v in spans.items() if k.startswith("held frown")))]
    a, b = next(v for k, v in spans.items() if k.startswith("electrode pop"))
    pops_shape = fx[a - 30:b + 30]  # from just before the first spike to the end of the ringing

    texture = _band(rng.normal(0, 1, n), 20.0, 450.0, order=4)
    cor = texture / texture.std() * 0.003
    artifacts, traps, keep = [], [], []

    def frown(start, length, size):
        a, b = _s(start), _s(start + length)
        env = np.ones(b - a)
        rise, fade = _s(1.0), _s(1.5)
        env[:rise] = np.linspace(0, 1, rise) ** 2
        env[-fade:] = np.linspace(1, 0, fade) ** 1.5
        band = _band(rng.normal(0, 1, b - a), 20.0, 450.0, order=4)
        cor[a:b] += band / band.std() * size * env
        return a, b

    for start in plan["bursts"]:
        _paste(cor, _s(start), burst_shape)
        traps.append({"type": "an ordinary burst: real, leave it", "start": _s(start),
                      "end": _s(start) + len(burst_shape)})
    a, b = frown(*plan["frown_long"], 0.03)
    traps.append({"type": "a long held frown (builds and fades): real, leave it", "start": a, "end": b})
    p = _s(plan["pop_single"])  # a single electrode pop with ringing: taller than any burst, but only one spike
    tt = np.arange(_s(0.08)) / FS
    cor[p:p + len(tt)] += 0.8 * np.exp(-tt / 0.004) * np.cos(2 * np.pi * 90 * tt)
    artifacts.append({"type": "a single small electrode pop", "start": p - _s(0.02), "end": p + len(tt),
                      "required": False})
    q0, q1 = _s(plan["quiet_fade"]), _s(plan["quiet_fade"] + 5.0)  # fades gradually: no sudden start
    env = np.ones(q1 - q0)
    ramp = _s(1.5)
    env[:ramp] = np.linspace(1, 0.05, ramp)
    env[ramp:-ramp] = 0.05
    env[-ramp:] = np.linspace(0.05, 1, ramp)
    cor[q0:q1] *= env
    traps.append({"type": "a quiet stretch that fades in gradually (no sudden start): leave it",
                  "start": q0 + ramp, "end": q1 - ramp})
    f0 = _s(plan["frown_fixture"])
    _paste(cor, f0, frown_shape)
    traps.append({"type": "a held frown: real, leave it", "start": f0, "end": f0 + len(frown_shape)})
    r0, r1 = _s(plan["pattern"][0]), _s(sum(plan["pattern"]))  # identical blips every 0.25 s
    blip = 0.04 * np.sin(2 * np.pi * np.arange(_s(0.01)) / _s(0.01))
    for c in range(r0, r1 - len(blip), _s(0.25)):
        cor[c:c + len(blip)] += blip
    artifacts.append({"type": "a regular repeating pattern lasting seconds", "start": r0, "end": r1})
    p0 = _s(plan["pop_train"])
    _paste(cor, p0, pops_shape, fade=5)
    artifacts.append({"type": "a train of electrode pops", "start": p0, "end": p0 + len(pops_shape)})
    d0, d1 = _s(plan["dead"][0]), _s(sum(plan["dead"]))  # a spike, then perfectly smooth until a sudden change
    cor[d0:d1] = 0.0
    cor[d0:d0 + 3] += np.array([0.25, -0.18, 0.06])
    cor[d1:d1 + 3] += np.array([-0.12, 0.10, -0.03])
    artifacts.append({"type": "a perfectly smooth stretch right after a spike (to the next sudden change)",
                      "start": d0, "end": d1})
    c0, c1 = frown(*plan["frown_clipped"], 0.06)
    mid, half = (c0 + c1) // 2, _s(0.5)  # its strongest half second reaches the sensor limit
    peak = _band(rng.normal(0, 1, 2 * half), 20.0, 450.0, order=4)
    cor[mid - half:mid + half] += peak / peak.std() * 0.3 * np.hanning(2 * half)
    limit = 0.5
    clipped = c0 + np.flatnonzero(np.abs(cor[c0:c1]) >= limit)
    cor[c0:c1] = np.clip(cor[c0:c1], -limit, limit)
    k0, k1 = int(clipped[0]), int(clipped[-1]) + 1
    reasons["clipped"] = [[k0, k1]]
    reasons["clip_guard"] = [_clip_span(k0 - _s(1.5), k1 + _s(1.5), n)]
    guard = reasons["clip_guard"][0]
    keep.append({"type": "a frown that reached the sensor limit", "start": guard[0], "end": guard[1]})
    if guard[0] > c0:
        traps.append({"type": "the rest of that frown: real, leave it", "start": c0, "end": guard[0]})
    if guard[1] < c1:
        traps.append({"type": "the rest of that frown: real, leave it", "start": guard[1], "end": c1})
    return cor, artifacts, traps, keep, reasons


def _cor_envelope(cor):
    """Mean rectified COR, centred 100-ms moving average (physiopipe's emg_cor_env, here at 1000 Hz)."""
    k = _s(0.1)
    return np.convolve(np.abs(cor), np.ones(k) / k, mode="same")


# --------------------------------------------------------------------------
# One participant, the sidecar, and the practice folder
# --------------------------------------------------------------------------

def _machine_entry(reasons, n, ch):
    by_reason = {r: _merge(reasons.get(r, [])) for r in REASONS[ch]}
    union = _merge([s for spans in by_reason.values() for s in spans])
    return {"BadSegments": union, "SegmentsByReason": by_reason, "Rule": MACHINE_RULES[ch],
            "ProportionBad": round(sum(b - a for a, b in union) / n, 4)}


def build_subject(seed, subject):
    """(DataFrame, sidecar, run key, ecg_inverted) for one practice participant."""
    plan = PLANS[subject]
    rng = np.random.default_rng(seed)
    n = N_SAMPLES
    t = np.arange(n) / FS
    beats, pvc = _beat_times(rng, plan["ecg"]["heart_rate"], plan["ecg"]["pvc_at"])

    ecg, upright, true_r, pvc_r, ecg_art, detected, flat = _build_ecg(rng, plan["ecg"], n, t, beats, pvc)
    ecg_seed = _ecg_seed(rng, plan["ecg"], upright, true_r, pvc_r, detected)
    ppg, ppg_tops, ppg_seed, ppg_art, ppg_neutral, ppg_rail = _build_ppg(rng, plan["ppg"], n, t, beats, pvc)
    belt, breath_tops, rsp_art, rsp_traps, rsp_neutral, rsp_reasons = _build_belt(
        rng, plan["rsp"], n, t, plan["ecg"]["heart_rate"])
    eda, scr_tops, eda_art, eda_traps, eda_reasons = _build_eda(rng, plan["eda"], n, t)
    cor, cor_art, cor_traps, cor_keep, cor_reasons = _build_cor(rng, plan["emg_cor"], n, t)
    event, rows = _event_column(n)

    df = pd.DataFrame({
        "ecg": ecg, "ppg": ppg, "rsp": _band(belt, 0.07, 1.0, order=2), "eda": eda, "emg_cor": cor,
        "emg_cor_env": _cor_envelope(cor), "ecg_peaks": pc._peaks_column(ecg_seed, n).astype(int),
        "ppg_peaks": pc._peaks_column(ppg_seed, n).astype(int),
        "rsp_peaks": pc._peaks_column(breath_tops, n).astype(int),
        "eda_peaks": pc._peaks_column(scr_tops, n).astype(int), "event": event, "rsp_raw": belt,
    })[COLUMNS]

    machine = {
        "ecg": _machine_entry({"flat": [flat]}, n, "ecg"),
        "ppg": _machine_entry({"rail": [ppg_rail]} if ppg_rail else {}, n, "ppg"),
        "rsp": _machine_entry(rsp_reasons, n, "rsp"),
        "eda": _machine_entry(eda_reasons, n, "eda"),
        "emg_cor": _machine_entry(cor_reasons, n, "emg_cor"),
    }
    sidecar = {"SamplingFrequency": FS, "StartTime": -BASELINE[0], "Columns": COLUMNS}
    for col in COLUMNS:
        units, description = COLUMN_TEXTS[col]
        sidecar[col] = {"Units": units, "Description": description, "TechnicalDescription": TECHNICAL_DESCRIPTION}
    margin = _s(FILE_MARGIN_SEC)
    sidecar["Events"] = {"Codebook": CODEBOOK, "Rows": rows, "Behaviour": None}
    sidecar["MachineQC"] = {"CoordinateSpace": COORDINATE_SPACE, "Channels": machine, "Corrections": {}}
    sidecar["Provenance"] = {
        "Practice": True, "PipelineCommit": "practice", "Dataset": "practice", "Subject": subject,
        "NumberOfSamples": n,
        "SourceFiles": [{"Role": "task", "File": "practice_recording", "TsvRows": [0, n],
                         "FileRows1000Hz": [margin, margin + n], "FileLengthRows1000Hz": n + 2 * margin,
                         "SecondsFromFileStart": FILE_MARGIN_SEC, "SecondsToFileEnd": FILE_MARGIN_SEC}],
        "Junctions": [],
        "ChannelStatus": {ch: ["ok"] for ch in REASONS},
    }
    sidecar["Pass2Inputs"] = {}

    channels = {
        "ecg": {"mode": "point", "true_peaks": [int(p) for p in true_r],
                "traps": ([{"type": "the premature (early) beat: real, keep its mark", "sample": int(pvc_r)}]
                          if pvc_r is not None else [])},
        "bad_ecg": {"mode": "segment", "artifacts": ecg_art, "traps": []},
        "ppg": {"mode": "point", "true_peaks": [int(p) for p in ppg_tops], "traps": []},
        "bad_ppg": {"mode": "segment", "artifacts": ppg_art, "traps": [], "neutral": ppg_neutral},
        "rsp": {"mode": "segment", "artifacts": rsp_art, "traps": rsp_traps, "neutral": rsp_neutral},
        "eda": {"mode": "segment", "artifacts": eda_art, "traps": eda_traps},
        "emg_cor": {"mode": "segment", "artifacts": cor_art, "traps": cor_traps, "must_keep": cor_keep},
    }
    run_key = {"n_samples": n, "channels": channels}
    return df, sidecar, run_key, plan["ecg"]["kind"] == "inverted"


def description_hashes():
    import hashlib
    return {col: hashlib.sha256(text.encode("utf-8")).hexdigest() for col, (_units, text) in COLUMN_TEXTS.items()}


def make(out_dir, seed=None, plain_key_copy=None):
    """Writes sub-990 and sub-991 (run 1) into out_dir and the encoded key to out_dir/practice_key.pebbl."""
    import json
    import secrets
    if seed is None:
        seed = secrets.randbits(31)
    out_dir = os.path.abspath(out_dir)
    key = {"version": KEY_VERSION, "study": TASK, "seed": int(seed), "sampling_rate": FS, "criteria": CRITERIA,
           "descriptions_sha256": description_hashes(), "subjects": {}}
    written = []
    for i, subject in enumerate(SUBJECTS):
        df, sidecar, run_key, inverted = build_subject(int(seed) + 101 * (i + 1), subject)
        stem = f"sub-{subject}_ses-run{RUN}_task-{TASK}"
        beh = os.path.join(out_dir, f"sub-{subject}", f"ses-run{RUN}", "beh")
        os.makedirs(beh, exist_ok=True)
        tsv = os.path.join(beh, f"{stem}_physio.tsv.gz")
        df.to_csv(tsv, sep="\t", index=False, header=False, compression="gzip")
        with open(os.path.join(beh, f"{stem}_physio.json"), "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2)
        key["subjects"][subject] = {"ecg_inverted": inverted, "runs": {RUN: run_key}}
        written.append(tsv)
    with open(pc.key_path_for(out_dir), "w", encoding="utf-8") as f:
        f.write(pc.encode_key(key))
    if plain_key_copy:
        os.makedirs(os.path.dirname(os.path.abspath(plain_key_copy)), exist_ok=True)
        with open(plain_key_copy, "w", encoding="utf-8") as f:
            json.dump(key, f, indent=1)
    for tsv in written:
        print(f"Practice recording: {tsv}")
    print(f"Answer key: {pc.key_path_for(out_dir)} (encoded; PEBBL scores practice reviews with it)")
    return written, pc.key_path_for(out_dir)
