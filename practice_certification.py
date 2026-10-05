#!/usr/bin/env python3
"""
RA certification practice: a simulated participant (sub-990, runs 1 and 2)
with known answers, an answer key, and automatic scoring (key version 5,
HLU 2026-10-04).

    make   -- lab staff, once: writes sub-990 (runs 1 and 2) in the lab's
              usual sub-XXX/ses-runY/beh/*_physio.tsv.gz + .json layout into a
              PRACTICE folder (on Box: a "practice" folder next to
              "derivatives"), and the answer key into that folder too
              (practice_key.pebbl, encoded so it isn't readable at a glance).
    score  -- scores an RA's saved *_annotations_<initials>.json for one run
              and prints a PASS/FAIL report per channel.

RAs normally never run this: the PEBBL form's "Practice for certification"
box opens sub-990 from the practice folder, and after each Step 2 save PEBBL
scores the review, shows the report, and offers to show the RA's marks next
to the answer key (view only). Trainees build their own ECG template in
Step 1 first, as on real data.

The recordings (simulated, never participant data)
  ECG  neurokit2's ECGSYN model (realistic P-QRS-T beats with heart-rate
       variability), made to look like scanner ECG: enlarged T waves, narrow
       gradient spikes, baseline wander, noise, one premature beat, and one
       stretch of heavy noise where R peaks can't be found (to mark bad_ecg).
       The true R peaks come from the clean simulation.
  PPG  CareTaker-shaped pulses (an early systolic hump, the main hump, a
       dicrotic wave), one per true heartbeat, passed through the pipeline's
       0.5-8 Hz band-pass, with a device dropout (a zero run, smoothed into a
       low arc). neurokit2's pulse model lacks CareTaker's two humps, so the
       shape is built here.
  RSP  neurokit2's breathmetrics model, with a sigh, a breath hold, shallow
       breathing and clipped tops (all real: leave them) and two stretches
       where the belt stopped recording (mark them).
  EDA  neurokit2 tonic level and noise, plus skin conductance responses (real:
       leave them) and an electrode lift, motion spikes and a stuck sensor
       (mark them).
  SBP/DBP  one reading per pulse, with blank no-reading stretches, BP inside
       the PPG dropout, a recalibration (held flat) and an implausible spike
       (mark them all).
  As on real files, ppg/rsp/eda/sbp/dbp start with a machine check (the
  sidecar's MachineQC) that is deliberately partly wrong: per channel it
  misses one artifact and falsely flags one clean stretch.

Scoring (by outcome, so any starting marks work: a Step 1 template or the
automatic detections)
  ECG/PPG: every true beat needs one mark within 20 ms (ECG) or 30 ms (PPG)
    of it; for PPG any crest of that pulse counts (either hump; HLU's
    "don't move a mark between humps", 2026-10-04). Beats inside the RA's own
    bad stretches are excused; the bad stretches themselves are scored as the
    bad_ecg/bad_ppg channel. Missed beats and extra marks are listed with
    their times; the premature beat (ECG) and its weak pulse (PPG) must be
    kept.
  Segment channels: every artifact at least 80% marked, no real event or
    machine false flag more than 25% marked, at most 2 s marked outside the
    artifacts (1 s of leeway around each).

Usage (lab staff)
  annotate_env\\Scripts\\python.exe practice_certification.py make --out-dir "<Box>\\DATA\\Processed\\physioProcessing\\practice"
  annotate_env\\Scripts\\python.exe practice_certification.py score --key "<practice folder>\\practice_key.pebbl" --annotations <file> [--channels ecg,bad_ecg]
"""

import argparse
import base64
import json
import os
import re
import secrets
import sys
import zlib

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

FS = 1000
DURATION_SEC = 180
PRACTICE_SUBJECT = "990"
RUNS = ("1", "2")
KEY_VERSION = 5  # 5 (2026-10-04): neurokit2 signals, runs 1 and 2, outcome scoring, key in the practice folder
KEY_FILENAME = "practice_key.pebbl"
KEY_HEADER = "PEBBL practice answer key (encoded; lab staff: practice_certification.py score)\n"

PASS_CRITERIA = {
    "tolerance_ms": {"ecg": 20, "ppg": 30},
    "artifact_min_coverage": 0.80,
    "trap_max_coverage": 0.25,
    "max_seconds_marked_outside": 2.0,
    "artifact_leeway_sec": 1.0,
}

PPG_HUMP_GAP = 0.12        # s between the early and the main systolic hump
PPG_PULSE_DELAY = 0.30     # s from the R peak to the pulse's main hump
PPG_MASK_DILATION = 600    # samples each side, as physioProcess widens zero runs

COLUMNS = ["ecg", "rsp", "ppg", "eda", "sbp", "dbp", "ecg_peaks", "rsp_peaks", "ppg_peaks", "eda_peaks", "event"]
COLUMN_INFO = {
    "ecg": ("mV", "The heart's electrical signal (ECG), filtered to reduce scanner noise. Each tall, sharp spike is "
                  "one heartbeat. Stretches with heavy scanner noise can make beats hard to see."),
    "rsp": ("V", "Breathing signal from the belt around the participant's torso. Each rise and fall is one breath."),
    "ppg": ("arbitrary", "Finger pulse signal from the CareTaker. Where the CareTaker lost the signal, the trace shows "
                         "slow, smooth curves instead of pulses."),
    "eda": ("microsiemens", "Skin conductance (the sweat response). Rises lasting a few seconds are normal responses."),
    "sbp": ("mmHg", "Systolic (upper) blood pressure from the CareTaker finger cuff, one reading per heartbeat. Gaps "
                    "with no reading are blank."),
    "dbp": ("mmHg", "Diastolic (lower) blood pressure from the CareTaker finger cuff, one reading per heartbeat. Gaps "
                    "with no reading are blank."),
    "event": ("code", "Task phase at each moment: 0 = outside the task, 20 = baseline, 30 = reading, 40 = imagery, "
                      "50 = recovery, 65 = ratings."),
}

# Where things are planted, per run (seconds; lengths in seconds). Hand-placed
# so nothing overlaps within a channel and every kind of item appears in both runs.
RUN_PLANS = {
    "1": {
        "heart_rate": 72, "pvc_at": 95.0, "mhd_heavy": (40.0, 25.0),
        "spikes_at": (18.3, 52.7, 77.1, 133.4, 161.9), "ecg_noise": (118.0, 4.0),
        "ecg_missed": (22.0, 61.0, 150.0), "ecg_on_t": (88.0, 170.0), "ecg_misplaced": (140.0,),
        "ppg_flips": (30.0, 65.0, 100.0, 145.0), "ppg_dropout_near": 125.0,
        "ppg_missed": (35.0, 84.0, 156.0), "ppg_on_dicrotic": (50.0, 110.0), "ppg_on_slope": (75.0,),
        "rsp_dead": ((35.0, 6.0), (150.0, 4.0)), "rsp_sigh": 120.0, "rsp_hold": (70.0, 8.0),
        "rsp_shallow": (95.0, 15.0), "rsp_clip": (160.0, 10.0),
        "eda_scrs": (20.0, 58.0, 101.0, 146.0, 165.0), "eda_lift": (40.0, 2.5), "eda_spikes": (85.0, 1.2),
        "eda_flat": (128.0, 5.0),
        "bp_recal": (50.0, 8.0), "bp_spike": (112.0, 1.5), "bp_blank": (30.0, 3.0), "bp_sliver": (75.0, 1.2),
        "bp_false": {"sbp": (150.0, 6.0), "dbp": (165.0, 6.0)},
    },
    "2": {
        "heart_rate": 66, "pvc_at": 60.0, "mhd_heavy": (100.0, 25.0),
        "spikes_at": (12.5, 44.2, 97.8, 141.3, 170.6), "ecg_noise": (30.0, 4.0),
        "ecg_missed": (80.0, 125.0, 160.0), "ecg_on_t": (20.0, 110.0), "ecg_misplaced": (55.0,),
        "ppg_flips": (25.0, 70.0, 120.0, 150.0), "ppg_dropout_near": 85.0,
        "ppg_missed": (40.0, 100.0, 165.0), "ppg_on_dicrotic": (15.0, 130.0), "ppg_on_slope": (55.0,),
        "rsp_dead": ((140.0, 5.0), (45.0, 5.0)), "rsp_sigh": 85.0, "rsp_hold": (110.0, 8.0),
        "rsp_shallow": (20.0, 15.0), "rsp_clip": (160.0, 8.0),
        "eda_scrs": (15.0, 48.0, 95.0, 132.0, 160.0), "eda_lift": (70.0, 2.5), "eda_spikes": (110.0, 1.2),
        "eda_flat": (35.0, 5.0),
        "bp_recal": (120.0, 8.0), "bp_spike": (60.0, 1.5), "bp_blank": (150.0, 3.0), "bp_sliver": (95.0, 1.2),
        "bp_false": {"sbp": (20.0, 6.0), "dbp": (40.0, 6.0)},
    },
}


# --------------------------------------------------------------------------
# Signal construction
# --------------------------------------------------------------------------

def _gauss(t, center, width, amplitude):
    return amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)


def _span(start_sec, length_sec):
    return int(round(start_sec * FS)), int(round((start_sec + length_sec) * FS))


def _nearest(values, target):
    return int(np.argmin(np.abs(np.asarray(values, dtype=float) - target)))


def _refine_max(signal, index, half=15):
    lo, hi = max(0, index - half), min(len(signal), index + half + 1)
    return lo + int(np.argmax(signal[lo:hi]))


def _lowpass(x, cutoff):
    return sosfiltfilt(butter(3, cutoff, btype="lowpass", fs=FS, output="sos"), x)


def _build_ecg(rng, plan, n, t):
    """ECG from neurokit2's ECGSYN model, made scanner-like. Returns (ecg, true R peaks, PVC R, artifacts)."""
    import neurokit2 as nk
    clean = np.asarray(nk.ecg_simulate(duration=DURATION_SEC, sampling_rate=FS, heart_rate=plan["heart_rate"],
                                       heart_rate_std=3, method="ecgsyn", random_state=int(rng.integers(2**31))))[:n]
    _, info = nk.ecg_peaks(clean, sampling_rate=FS, method="neurokit")
    r = [_refine_max(clean, int(p)) for p in info["ECG_R_Peaks"] if 40 <= p < n - 600]
    r = np.asarray(sorted(set(r)))
    ecg = clean.copy()
    baseline = float(np.median(clean))
    r_amp = float(np.median(clean[r])) - baseline

    # A premature (ectopic) beat: remove beat k and put a wide beat 62% of the way from beat k-1,
    # so the next beat keeps its time (a compensatory pause).
    k = max(1, min(len(r) - 2, _nearest(r / FS, plan["pvc_at"])))
    lo, hi = int(r[k] - 0.30 * FS), int(r[k] + 0.45 * FS)
    ecg[lo:hi] = np.linspace(ecg[lo], ecg[hi], hi - lo)
    pvc_t = (r[k - 1] + 0.62 * (r[k] - r[k - 1])) / FS
    plo, phi = int((pvc_t - 0.15) * FS), int((pvc_t + 0.55) * FS)
    tt = t[plo:phi]
    ecg[plo:phi] += (_gauss(tt, pvc_t, 0.022, 1.35 * r_amp) + _gauss(tt, pvc_t + 0.05, 0.03, -0.35 * r_amp)
                     + _gauss(tt, pvc_t + 0.30, 0.06, -0.30 * r_amp))
    pvc_r = _refine_max(ecg, int(round(pvc_t * FS)))
    true_r = np.asarray(sorted([int(x) for i, x in enumerate(r) if i != k] + [pvc_r]))

    # Enlarged T waves (scanner MHD), more so in one stretch.
    heavy_lo, heavy_hi = _span(*plan["mhd_heavy"])
    for i, p in enumerate(true_r[:-1]):
        if p == pvc_r:
            continue
        a, b = int(p + 0.12 * FS), int(min(p + 0.42 * FS, true_r[i + 1] - 0.08 * FS))
        if b <= a + 10:
            continue
        line = np.linspace(ecg[a], ecg[b], b - a)
        gain = 2.4 if heavy_lo <= p < heavy_hi else 1.7
        ecg[a:b] = line + (ecg[a:b] - line) * gain

    # Baseline wander and noise.
    ecg += 0.08 * r_amp * np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, 0.025 * r_amp, n)

    # Narrow gradient spikes, kept at least 0.2 s from any R peak.
    for s in plan["spikes_at"]:
        c = int(s * FS)
        near = true_r[np.argmin(np.abs(true_r - c))]
        if abs(near - c) < 0.2 * FS:
            c = int(near + 0.35 * FS)
        ecg[c - 3:c + 4] += r_amp * 1.15 * (1 - np.abs(np.arange(-3, 4)) / 4)

    # Heavy noise where R peaks can't be found: mark it bad_ecg.
    nlo, nhi = _span(*plan["ecg_noise"])
    burst = rng.normal(0, 1, nhi - nlo)
    burst = burst / np.std(burst) * 0.9 * r_amp + _lowpass(rng.normal(0, 1, nhi - nlo), 3.0) * 2.5 * r_amp
    ecg[nlo:nhi] += burst
    artifacts = [{"type": "heavy scanner noise: R peaks can't be found here (mark it bad_ecg)", "start": nlo,
                  "end": nhi}]
    return ecg, true_r, pvc_r, artifacts


def _seed_with_errors(signal, true_peaks, detected, plan_missed, plan_extra, plan_misplaced, protect, rng):
    """The automatic detections the review starts from: a detector's own output, plus planted mistakes."""
    seed = set(int(x) for x in detected)
    true_peaks = np.asarray(true_peaks)
    used = set(protect)

    def pick(time_sec):
        for i in np.argsort(np.abs(true_peaks / FS - time_sec)):
            if int(true_peaks[i]) not in used:
                used.add(int(true_peaks[i]))
                return int(true_peaks[i])
        raise RuntimeError("no free beat to plant a mistake on")

    def drop_near(sample, tol=40):
        for s in [s for s in seed if abs(s - sample) <= tol]:
            seed.discard(s)

    for time_sec in plan_missed:
        drop_near(pick(time_sec))
    for time_sec, offset in plan_extra:
        seed.add(_refine_max(signal, int(pick(time_sec) + offset * FS), half=40))
    for time_sec, offset in plan_misplaced:
        p = pick(time_sec)
        drop_near(p)
        seed.add(int(p + offset * FS))
    return sorted(seed)


def _peaks_column(indices, n):
    col = np.zeros(n, dtype=bool)
    idx = np.asarray([i for i in indices if 0 <= i < n], dtype=int)
    col[idx] = True
    return col


def _local_maxima(x, lo, hi):
    seg = x[lo:hi]
    return [lo + i for i in range(1, len(seg) - 1) if seg[i] >= seg[i - 1] and seg[i] > seg[i + 1]]


def _build_ppg(rng, plan, n, t, true_r, pvc_r):
    """CareTaker-shaped finger pulse, one per true heartbeat, with a smoothed device dropout."""
    centers = [r / FS + PPG_PULSE_DELAY + rng.normal(0, 0.004) for r in true_r]
    flips = {_nearest(centers, x) for x in plan["ppg_flips"]}
    pvc_i = int(np.where(true_r == pvc_r)[0][0])
    ppg = 0.2 * np.sin(2 * np.pi * 0.1 * t) + rng.normal(0, 0.01, n)
    for i, c in enumerate(centers):
        amp = 0.45 if i == pvc_i else 1.0  # the premature beat ejects less blood
        early = (0.95 if i in flips else 0.5) * amp
        lo, hi = int(max(0, (c - 0.35) * FS)), int(min(n, (c + 0.6) * FS))
        tt = t[lo:hi]
        ppg[lo:hi] += (_gauss(tt, c, 0.04, amp) + _gauss(tt, c - PPG_HUMP_GAP, 0.025, early)
                       + _gauss(tt, c + 0.30, 0.06, 0.35 * amp))
    # The dropout covers one whole pulse and sits in the diastoles on either side.
    kb = _nearest(centers, plan["ppg_dropout_near"])
    drop_lo = int((centers[kb] + 0.35) * FS)
    drop_hi = int((centers[kb + 2] - PPG_HUMP_GAP - 0.30) * FS)
    ppg[drop_lo:drop_hi] = 0.0
    ppg = sosfiltfilt(butter(3, [0.5, 8.0], btype="bandpass", fs=FS, output="sos"), ppg)

    pulses = []  # (center index, top, acceptable crests)
    for i, c in enumerate(centers):
        lo, hi = int(round((c - PPG_HUMP_GAP - 0.05) * FS)), int(round((c + 0.05) * FS))
        if hi >= n or (drop_lo / FS <= c + 0.05 and c - PPG_HUMP_GAP - 0.05 <= drop_hi / FS):
            continue
        top = lo + int(np.argmax(ppg[lo:hi]))
        floor = float(np.min(ppg[lo:hi]))
        height = ppg[top] - floor
        crests = [m for m in _local_maxima(ppg, lo, hi) if ppg[m] - floor >= 0.6 * height]
        pulses.append((i, top, sorted(set(crests + [top]))))
    tops = np.asarray([p[1] for p in pulses])
    accept = [p[2] for p in pulses]
    flip_crest = {p[1]: min(p[2]) for p in pulses if p[0] in flips and len(p[2]) > 1}
    pvc_top = next(p[1] for p in pulses if p[0] == pvc_i)

    # The machine's marks: every pulse top (on the early hump on some beats, as the machine does), plus mistakes.
    detected = [flip_crest.get(top, top) for top in tops]
    extra = [(x, 0.30) for x in plan["ppg_on_dicrotic"]] + [(x, -0.06) for x in plan["ppg_on_slope"]]
    seed = _seed_with_errors(ppg, tops, detected, plan["ppg_missed"], extra[:len(plan["ppg_on_dicrotic"])],
                             [(x, -0.06) for x in plan["ppg_on_slope"]], [pvc_top], rng)
    arc = [i for i in range(drop_lo + 60, drop_hi - 60) if ppg[i] == ppg[i - 50:i + 51].max()]
    if arc:
        seed = sorted(set(seed) | {max(arc, key=lambda i: ppg[i])})  # a machine mark on the dropout's arc
    return ppg, tops, accept, pvc_top, seed, (drop_lo, drop_hi)


def _build_rsp(rng, plan, n, t):
    """Breathing from neurokit2, with real oddities to leave alone and dead-belt stretches to mark."""
    import neurokit2 as nk
    rsp = np.asarray(nk.rsp_simulate(duration=DURATION_SEC, sampling_rate=FS, respiratory_rate=15,
                                     method="breathmetrics", random_state=int(rng.integers(2**31))))[:n]
    _, info = nk.rsp_peaks(rsp, sampling_rate=FS)
    troughs = np.asarray([int(x) for x in info["RSP_Troughs"] if x == x])
    peaks = np.asarray([int(x) for x in info["RSP_Peaks"] if x == x])
    top = float(np.percentile(rsp[peaks], 50))
    bottom = float(np.percentile(rsp[troughs], 50))
    traps = []

    # A sigh: one breath twice as deep.
    j = max(0, min(len(troughs) - 2, _nearest(troughs / FS, plan["rsp_sigh"])))
    a, b = troughs[j], troughs[j + 1]
    line = np.linspace(rsp[a], rsp[b], b - a)
    rsp[a:b] = line + (rsp[a:b] - line) * 2.2
    traps.append({"type": "a sigh (one deep breath): real, leave it", "start": int(a), "end": int(b)})

    # A breath hold after breathing in: level, with a small heartbeat ripple.
    h0 = peaks[_nearest(peaks / FS, plan["rsp_hold"][0])]
    h1 = h0 + int(plan["rsp_hold"][1] * FS)
    hold = rsp[h0] + 0.02 * (top - bottom) * np.sin(2 * np.pi * 1.2 * t[: h1 - h0]) + rng.normal(0, 0.003, h1 - h0)
    ramp = np.linspace(0, 1, 500)
    rsp[h0:h1] = hold
    rsp[h1:h1 + 500] = hold[-1] * (1 - ramp) + rsp[h1:h1 + 500] * ramp
    hold_span = {"type": "a breath hold: real, leave it", "start": int(h0), "end": int(h1)}
    traps.append(hold_span)

    # Shallow breathing and clipped tops: breaths are still visible, so not bad.
    s0, s1 = _span(*plan["rsp_shallow"])
    mean = float(np.mean(rsp[s0:s1]))
    rsp[s0:s1] = mean + (rsp[s0:s1] - mean) * 0.35
    traps.append({"type": "shallow breathing: real, leave it", "start": s0, "end": s1})
    c0, c1 = _span(*plan["rsp_clip"])
    rsp[c0:c1] = np.minimum(rsp[c0:c1], bottom + 0.6 * (top - bottom))
    traps.append({"type": "clipped breath tops: breaths still visible, leave it", "start": c0, "end": c1})

    # The belt stopped recording: a dead-flat line.
    artifacts = []
    for start, length in plan["rsp_dead"]:
        d0, d1 = _span(start, length)
        rsp[d0:d1] = bottom - 0.15 * (top - bottom) + rng.normal(0, 2e-5, d1 - d0)
        artifacts.append({"type": "belt stopped recording (flat line)", "start": d0, "end": d1})
    return rsp, peaks, artifacts, traps, hold_span


def _build_eda(rng, plan, n, t):
    """Skin conductance: neurokit2 tonic level and noise, placed responses, and artifacts to mark."""
    import neurokit2 as nk
    base = np.asarray(nk.eda_simulate(duration=DURATION_SEC, sampling_rate=FS, scr_number=0, drift=-0.003,
                                      noise=0.01, random_state=int(rng.integers(2**31))))[:n]
    eda = 4.5 + (base - base.mean()) * 0.5 + 0.3 * np.sin(2 * np.pi * t / 150)
    traps = []
    for onset in plan["eda_scrs"]:
        amp = rng.uniform(0.3, 0.7)
        tt = np.clip(t - onset, 0, None)
        eda += amp * 1.8 * (np.exp(-tt / 4.0) - np.exp(-tt / 0.75)) * (t >= onset)
        traps.append({"type": "a skin conductance response: real, leave it", "start": int(onset * FS),
                      "end": int((onset + 6) * FS)})
    artifacts = []
    l0, l1 = _span(*plan["eda_lift"])
    eda[l0:l1] = 0.4 + rng.normal(0, 0.005, l1 - l0)
    artifacts.append({"type": "electrode lift (sudden drop)", "start": l0, "end": l1})
    m0, m1 = _span(*plan["eda_spikes"])
    eda[m0:m1] += rng.choice([0, 0, 0, 1.5, -1.2], m1 - m0)
    artifacts.append({"type": "motion spikes", "start": m0, "end": m1})
    f0, f1 = _span(*plan["eda_flat"])
    eda[f0:f1] = eda[f0]
    artifacts.append({"type": "stuck sensor (flat line)", "start": f0, "end": f1})
    return eda, artifacts, traps


def _build_bp(rng, plan, n, pulse_samples, dropout):
    """Beat-by-beat SBP/DBP held between readings, with blank and bad stretches to mark."""
    sbp, dbp = np.full(n, np.nan), np.full(n, np.nan)
    edges = list(pulse_samples) + [n]
    for i, p in enumerate(pulse_samples):
        sec = p / FS
        sbp[p:edges[i + 1]] = round(118 + 4 * np.sin(2 * np.pi * sec / 10) + rng.normal(0, 1.0))
        dbp[p:edges[i + 1]] = round(72 + 2 * np.sin(2 * np.pi * sec / 10) + rng.normal(0, 0.7))
    first = pulse_samples[0]
    lead = (0, first)  # no reading before the first pulse
    out = {}
    for ch, sig, spike in (("sbp", sbp, 185.0), ("dbp", dbp, 25.0)):
        r0, r1 = _span(*plan["bp_recal"])
        sig[r0:r1] = sig[r0]
        k0, k1 = _span(*plan["bp_spike"])
        sig[k0:k1] = spike
        b0, b1 = _span(*plan["bp_blank"])
        sig[b0:b1] = np.nan
        v0, v1 = _span(*plan["bp_sliver"])
        sig[v0:v1] = np.nan
        artifacts = [
            {"type": "no reading at the start (blank): keep it marked", "start": lead[0], "end": lead[1]},
            {"type": "device recalibration (value held flat)", "start": r0, "end": r1},
            {"type": "implausible spike", "start": k0, "end": k1},
            {"type": "no CareTaker reading (blank): keep it marked", "start": b0, "end": b1},
            {"type": "short no-reading gap (blank): keep it marked", "start": v0, "end": v1},
            {"type": "BP inside the finger-pulse dropout: not real, keep it marked", "start": dropout[0],
             "end": dropout[1]},
        ]
        out[ch] = artifacts
    return sbp, dbp, out


def build_run(seed, run):
    """(DataFrame, key for this run, MachineQC block) for one practice run."""
    plan = RUN_PLANS[run]
    rng = np.random.default_rng(seed)
    n = DURATION_SEC * FS
    t = np.arange(n) / FS
    channels = {}

    ecg, true_r, pvc_r, ecg_art = _build_ecg(rng, plan, n, t)
    import neurokit2 as nk
    _, info = nk.ecg_peaks(ecg, sampling_rate=FS, method="neurokit")
    detected = [int(x) for x in info["ECG_R_Peaks"]]
    ecg_seed = _seed_with_errors(ecg, true_r, detected, plan["ecg_missed"], [(x, 0.28) for x in plan["ecg_on_t"]],
                                 [(x, 0.12) for x in plan["ecg_misplaced"]], [pvc_r], rng)
    channels["ecg"] = {"mode": "point", "true_peaks": [int(x) for x in true_r],
                       "traps": [{"type": "the premature (early) beat: real, keep its mark", "sample": int(pvc_r)}]}
    channels["bad_ecg"] = {"mode": "segment", "artifacts": ecg_art, "traps": []}

    ppg, tops, accept, pvc_top, ppg_seed, dropout = _build_ppg(rng, plan, n, t, true_r, pvc_r)
    channels["ppg"] = {"mode": "point", "true_peaks": [int(x) for x in tops],
                       "accept": [[int(c) for c in a] for a in accept],
                       "traps": [{"type": "the weak pulse after the premature beat: real, keep its mark",
                                  "sample": int(pvc_top)}]}
    channels["bad_ppg"] = {"mode": "segment", "traps": [], "artifacts": [
        {"type": "finger-pulse dropout (a low, smooth arc): mark it bad_ppg", "start": dropout[0],
         "end": dropout[1]}]}

    rsp, rsp_peaks, rsp_art, rsp_traps, hold_span = _build_rsp(rng, plan, n, t)
    channels["rsp"] = {"mode": "segment", "artifacts": rsp_art, "traps": rsp_traps}
    eda, eda_art, eda_traps = _build_eda(rng, plan, n, t)
    channels["eda"] = {"mode": "segment", "artifacts": eda_art, "traps": eda_traps}
    pulse_samples = [int(round(c)) for c in tops]
    sbp, dbp, bp_art = _build_bp(rng, plan, n, pulse_samples, dropout)
    channels["sbp"] = {"mode": "segment", "artifacts": bp_art["sbp"], "traps": []}
    channels["dbp"] = {"mode": "segment", "artifacts": bp_art["dbp"], "traps": []}

    event = np.zeros(n, dtype=int)
    for start_s, end_s, code in ((10, 40, 20), (40, 80, 30), (80, 120, 40), (120, 160, 50), (160, 172, 65)):
        event[start_s * FS:end_s * FS] = code

    df = pd.DataFrame({
        "ecg": ecg, "rsp": rsp, "ppg": ppg, "eda": eda, "sbp": sbp, "dbp": dbp,
        "ecg_peaks": _peaks_column(ecg_seed, n), "rsp_peaks": _peaks_column(rsp_peaks, n),
        "ppg_peaks": _peaks_column(ppg_seed, n), "eda_peaks": np.zeros(n, dtype=bool), "event": event,
    })[COLUMNS]
    machine_qc = _plant_machine_qc(channels, plan, n, dropout, hold_span)
    return df, {"n_samples": n, "channels": channels}, machine_qc


def _plant_machine_qc(channels, plan, n, dropout, hold_span):
    """
    A MachineQC block shaped like physioProcess's, deliberately partly wrong:
    per channel it flags some artifacts, MISSES one, and FALSELY flags one
    clean stretch. Records what was planted in each channel's key entry.
    """
    def span(item):
        return [int(item["start"]), int(item["end"])]

    scr = channels["eda"]["traps"][1]
    false_flags = {
        "rsp": ("the breath hold (flagged by mistake)", span(hold_span)),
        "eda": ("a real skin conductance response (flagged by mistake)", span(scr)),
        "sbp": ("clean blood pressure (flagged by mistake)", list(_span(*plan["bp_false"]["sbp"]))),
        "dbp": ("clean blood pressure (flagged by mistake)", list(_span(*plan["bp_false"]["dbp"]))),
    }
    # (indices of the artifacts the machine flags; index of the one it misses)
    flagged = {"rsp": ((0,), 1), "eda": ((0, 1), 2), "sbp": ((0, 1, 3, 4, 5), 2), "dbp": ((0, 1, 3, 4, 5), 2)}
    qc = {}
    for ch, (flag_idx, miss_idx) in flagged.items():
        artifacts = channels[ch]["artifacts"]
        false_type, false_span = false_flags[ch]
        segments = sorted([span(artifacts[i]) for i in flag_idx] + [false_span])
        entry = {"ProportionBad": round(sum(e - s for s, e in segments) / n, 4), "BadSegments": segments,
                 "Rule": "Practice file: a planted machine check that is deliberately partly wrong"}
        if ch in ("sbp", "dbp"):
            arts = artifacts
            entry["SegmentsByReason"] = {
                "no_coverage": [span(arts[0]), span(arts[3]), span(arts[4])],
                "out_of_range": [false_span],
                "ordering": [],
                "ppg_dropout": [span(arts[5])],
            }
            entry["BadSegments"] = sorted([span(arts[i]) for i in flag_idx] + [false_span])
        qc[ch] = entry
        channels[ch]["machine_qc"] = {"flagged": [artifacts[i]["type"] for i in flag_idx],
                                      "missed": artifacts[miss_idx]["type"],
                                      "false_flag": {"type": false_type, "start": false_span[0],
                                                     "end": false_span[1]}}
    ppg_span = [max(0, dropout[0] - PPG_MASK_DILATION), min(n, dropout[1] + PPG_MASK_DILATION)]
    qc["ppg"] = {"ProportionBad": round((ppg_span[1] - ppg_span[0]) / n, 4), "BadSegments": [ppg_span],
                 "SegmentsByReason": {"zero_run": [ppg_span], "bridged": [], "window_rule": [], "counter_loss": []},
                 "Rule": "Practice file: a planted device dropout (zero run, widened 0.6 s each side)"}
    return {"CoordinateSpace": "tsv row index at SamplingFrequency; segments are [start, end)", "Channels": qc}


# --------------------------------------------------------------------------
# Making the practice folder, and the answer key
# --------------------------------------------------------------------------

def key_path_for(practice_root):
    return os.path.join(practice_root, KEY_FILENAME)


def encode_key(key):
    return KEY_HEADER + base64.b64encode(zlib.compress(json.dumps(key).encode("utf-8"), 9)).decode("ascii") + "\n"


def load_key(path):
    """The answer key from practice_key.pebbl (encoded) or a plain .json copy."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if text.startswith(KEY_HEADER):
        return json.loads(zlib.decompress(base64.b64decode(text[len(KEY_HEADER):].strip())).decode("utf-8"))
    return json.loads(text)


def make(out_dir, seed=None, plain_key_copy=None):
    """Writes sub-990 runs 1 and 2 into out_dir and the encoded key to out_dir/practice_key.pebbl."""
    if seed is None:
        seed = secrets.randbits(31)
    out_dir = os.path.abspath(out_dir)
    key = {"version": KEY_VERSION, "seed": int(seed), "subject": PRACTICE_SUBJECT, "sampling_rate": FS,
           "criteria": PASS_CRITERIA, "runs": {}}
    written = []
    for run in RUNS:
        df, run_key, machine_qc = build_run(int(seed) + int(run), run)
        stem = f"sub-{PRACTICE_SUBJECT}_ses-run{run}_task-sdi"
        beh = os.path.join(out_dir, f"sub-{PRACTICE_SUBJECT}", f"ses-run{run}", "beh")
        os.makedirs(beh, exist_ok=True)
        tsv = os.path.join(beh, f"{stem}_physio.tsv.gz")
        df.to_csv(tsv, sep="\t", index=False, header=False, compression="gzip", na_rep="n/a")
        sidecar = {"SamplingFrequency": FS, "StartTime": 0.0, "Columns": COLUMNS,
                   "Description": "Simulated RA certification practice file -- not participant data.",
                   "MachineQC": machine_qc,
                   "Provenance": {"Practice": True, "PipelineCommit": "practice", "NumberOfSamples": len(df),
                                  "CTTiming": {"Pulse": "practice (simulated)"}}}
        for col, (units, description) in COLUMN_INFO.items():
            sidecar[col] = {"Units": units, "Description": description}
        with open(os.path.join(beh, f"{stem}_physio.json"), "w") as f:
            json.dump(sidecar, f, indent=2)
        key["runs"][run] = run_key
        written.append(tsv)
    with open(key_path_for(out_dir), "w", encoding="utf-8") as f:
        f.write(encode_key(key))
    if plain_key_copy:
        os.makedirs(os.path.dirname(os.path.abspath(plain_key_copy)), exist_ok=True)
        with open(plain_key_copy, "w") as f:
            json.dump(key, f, indent=1)
    for tsv in written:
        print(f"Practice recording: {tsv}")
    print(f"Answer key: {key_path_for(out_dir)} (encoded; PEBBL scores practice reviews with it)")
    return written, key_path_for(out_dir)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _times(samples, limit=8):
    shown = ", ".join(f"{s / FS:.1f}" for s in samples[:limit])
    return shown + (" s" if len(samples) <= limit else f" s, and {len(samples) - limit} more")


def match_point_channel(truth, final_indices, tol, bad_stretches=()):
    """
    Pairs marks with true beats. Returns (matches {beat index: mark}, missed
    beat tops, extra marks, excused beat tops). A beat matches the nearest
    unused mark within tol of any of its acceptable positions (PPG: any crest).
    """
    final = sorted(int(x) for x in final_indices)
    accept = truth.get("accept") or [[p] for p in truth["true_peaks"]]
    tops = truth["true_peaks"]
    used, matches, missed, excused = set(), {}, [], []
    for i, positions in enumerate(accept):
        candidates = [(min(abs(m - p) for p in positions), m) for m in final
                      if m not in used and min(abs(m - p) for p in positions) <= tol]
        if candidates:
            mark = min(candidates)[1]
            used.add(mark)
            matches[i] = mark
        elif any(a <= tops[i] < b for a, b in bad_stretches):
            excused.append(tops[i])
        else:
            missed.append(tops[i])
    extra = [m for m in final if m not in used and not any(a <= m < b for a, b in bad_stretches)]
    return matches, missed, extra, excused


def score_point_channel(ch, truth, final_indices, criteria, bad_stretches=()):
    tol = round(criteria["tolerance_ms"][ch] * FS / 1000)
    what = "heartbeat" if ch == "ecg" else "pulse"
    matches, missed, extra, excused = match_point_channel(truth, final_indices, tol, bad_stretches)
    total = len(truth["true_peaks"])
    lines = [f"  {len(matches)} of {total} {what}s marked"
             + (f"; {len(excused)} inside your bad stretches (fine)" if excused else "")]
    if missed:
        lines.append(f"  FIX  {len(missed)} {what}(s) with no mark within {criteria['tolerance_ms'][ch]} ms: "
                     f"{_times(missed)}")
    if extra:
        lines.append(f"  FIX  {len(extra)} extra mark(s) where there's no {what}: {_times(extra)}")
    marked_tops = {truth["true_peaks"][i] for i in matches}
    traps_ok = True
    for trap in truth.get("traps", []):
        kept = trap["sample"] in marked_tops
        traps_ok &= kept
        lines.append(f"  {'OK ' if kept else 'FIX'}  {trap['type']} at {trap['sample'] / FS:.1f} s"
                     f"{'' if kept else ': it has no mark'}")
    ok = not missed and not extra and traps_ok
    return ok, lines


def score_segment_channel(ch, truth, bad_segments, criteria, n_samples):
    marked = np.zeros(n_samples, dtype=bool)
    for start, end in bad_segments:
        marked[max(0, int(start)):min(n_samples, int(end))] = True
    lines, ok = [], True
    leeway = int(criteria["artifact_leeway_sec"] * FS)
    allowed = np.zeros(n_samples, dtype=bool)
    for art in truth["artifacts"]:
        coverage = float(marked[art["start"]:art["end"]].mean()) if art["end"] > art["start"] else 1.0
        hit = coverage >= criteria["artifact_min_coverage"]
        ok &= hit
        lines.append(f"  {'OK ' if hit else 'FIX'}  {art['type']} at {art['start'] / FS:.1f}-{art['end'] / FS:.1f} s: "
                     f"{coverage:.0%} marked bad")
        allowed[max(0, art["start"] - leeway):min(n_samples, art["end"] + leeway)] = True
    for trap in truth.get("traps", []):
        coverage = float(marked[trap["start"]:trap["end"]].mean())
        clean = coverage <= criteria["trap_max_coverage"]
        ok &= clean
        lines.append(f"  {'OK ' if clean else 'FIX'}  {trap['type']} at {trap['start'] / FS:.1f} s: "
                     f"{coverage:.0%} marked bad")
    false_flag = (truth.get("machine_qc") or {}).get("false_flag")
    if false_flag:
        coverage = float(marked[false_flag["start"]:false_flag["end"]].mean())
        removed = coverage <= criteria["trap_max_coverage"]
        ok &= removed
        lines.append(f"  {'OK ' if removed else 'FIX'}  the machine's mark on {false_flag['type']} at "
                     f"{false_flag['start'] / FS:.1f}-{false_flag['end'] / FS:.1f} s: {coverage:.0%} still marked")
    outside = float((marked & ~allowed).sum()) / FS
    within = outside <= criteria["max_seconds_marked_outside"]
    ok &= within
    lines.append(f"  {'OK ' if within else 'FIX'}  {outside:.1f} s marked bad where the signal is fine "
                 f"(limit {criteria['max_seconds_marked_outside']:.1f} s)")
    return ok, lines


CHANNEL_NAMES = {"ecg": "ECG", "bad_ecg": "ECG bad stretches", "ppg": "PPG", "bad_ppg": "PPG bad stretches",
                 "rsp": "RSP", "eda": "EDA", "sbp": "SBP", "dbp": "DBP"}


def run_from_filename(path):
    match = re.search(r"ses-run(\d+)", os.path.basename(path))
    return match.group(1) if match else None


def score_run(key, submission, run, channels=None):
    """(overall, {channel: passed}, report lines) for one run's saved review."""
    if key.get("version") != KEY_VERSION or run not in key.get("runs", {}):
        return False, {}, [f"This answer key (version {key.get('version')}) doesn't match this version of PEBBL "
                           f"(version {KEY_VERSION}) or has no run {run}. Ask the lab staff to make the practice "
                           f"folder again."]
    run_key = key["runs"][run]
    reviewed = submission.get("channels", {})
    wanted = channels or [ch for ch in run_key["channels"] if ch in reviewed]
    criteria = key.get("criteria", PASS_CRITERIA)
    results, lines = {}, [f"Practice score: participant 990, run {run}, initials {submission.get('initials', '?')}"]
    for ch in wanted:
        truth = run_key["channels"].get(ch)
        if truth is None:
            continue
        name = CHANNEL_NAMES.get(ch, ch)
        if ch not in reviewed:
            results[ch] = False
            lines += ["", f"{name}: NOT REVIEWED (nothing saved for it)"]
            continue
        if reviewed[ch].get("mode") and reviewed[ch]["mode"] != truth["mode"]:
            results[ch] = False
            lines += ["", f"{name}: CAN'T SCORE (this review is in an older format; ask the lab staff)"]
            continue
        if truth["mode"] == "point":
            bad = (reviewed.get(f"bad_{ch}") or {}).get("bad_segments", [])
            ok, ch_lines = score_point_channel(ch, truth, reviewed[ch].get("indices", []), criteria, bad)
        else:
            ok, ch_lines = score_segment_channel(ch, truth, reviewed[ch].get("bad_segments", []), criteria,
                                                 run_key["n_samples"])
        results[ch] = ok
        lines += ["", f"{name}: {'PASS' if ok else 'NOT YET'}"] + ch_lines
    overall = bool(results) and all(results.values())
    lines += ["", f"OVERALL: {'PASS' if overall else 'NOT YET'} ({sum(results.values())} of {len(results)} "
                  f"passed)"]
    return overall, results, lines


def score(key_path, annotations_path, channels=None, run=None):
    """Command-line scoring: prints the report and returns (overall, results)."""
    key = load_key(key_path)
    with open(annotations_path) as f:
        submission = json.load(f)
    run = run or run_from_filename(annotations_path) or "1"
    overall, results, lines = score_run(key, submission, run, channels)
    print("\n".join(lines))
    return overall, results


def channels_to_score(reviewed_keys):
    """The key channels a session's reviewed channels are scored on (a peak channel brings its bad stretches)."""
    out = []
    for ch in reviewed_keys:
        out.append(ch)
        if ch in ("ecg", "ppg"):
            out.append(f"bad_{ch}")
    return out


def key_outputs(run_key, submission_channels, channels, criteria=PASS_CRITERIA):
    """
    The answer key as channel outputs, for showing next to the RA's marks.
    A beat the RA marked shows at the RA's mark (so it reads as agreed, even on
    the other PPG hump); a missed beat shows at its top.
    """
    out = {}
    for ch in channels:
        truth = run_key["channels"].get(ch)
        if truth is None:
            continue
        if truth["mode"] == "point":
            tol = round(criteria["tolerance_ms"][ch] * FS / 1000)
            reviewed = submission_channels.get(ch, {})
            bad = (submission_channels.get(f"bad_{ch}") or {}).get("bad_segments", [])
            matches, missed, _extra, _excused = match_point_channel(truth, reviewed.get("indices", []), tol, bad)
            out[ch] = {"mode": "point", "indices": sorted(list(matches.values()) + missed), "status": "complete"}
        else:
            out[ch] = {"mode": "segment", "status": "complete",
                       "bad_segments": [[a["start"], a["end"]] for a in truth["artifacts"]]}
    return out


def show_comparison(df, sfreq, run_key, submission_channels, channels, initials, sidecar=None):
    """
    Opens the viewer with the RA's marks next to the answer key (view only:
    nothing is saved). Labels: <label>_agree, <label>_only_<initials> (a mark
    the key doesn't have) and <label>_only_key (something the RA missed).
    """
    import numpy as np  # noqa: F811 (local import keeps the GUI deps out of make/score)
    from annotation_io import compute_scalings
    from channel_config import CHANNELS, with_companions
    from reconcile import build_reconciliation_raw

    configs = with_companions({ch: CHANNELS[ch] for ch in channels if ch in CHANNELS})
    ra = {ch: submission_channels[ch] for ch in configs if ch in submission_channels}
    key = key_outputs(run_key, submission_channels, list(configs))
    raw, summary = build_reconciliation_raw(df, configs, sfreq, ra, key, initials, "key", ppg_guide=True,
                                            sidecar=sidecar)
    print("\n" + "=" * 70)
    print(f"Your marks ({initials}) next to the answer key -- VIEW ONLY, nothing is saved.")
    print(f"  ..._agree: you and the key agree.  ..._only_{initials}: your mark, not in the key (delete it next "
          f"time).  ..._only_key: something you missed.")
    for ch, counts in summary.items():
        print(f"  {ch}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    print("Close the window when you're done looking.")
    print("=" * 70 + "\n")
    from annotation_io import combine_with_reference_channels
    full = combine_with_reference_channels(configs, df, ppg_guide=True, sidecar=sidecar)
    raw.plot(block=True, scalings=compute_scalings(df, full), duration=20, show=True,
             order=np.arange(len(raw.ch_names)), title=f"Your marks vs the answer key ({initials}) -- view only")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Make or score the RA certification practice participant.")
    sub = parser.add_subparsers(dest="command", required=True)
    make_p = sub.add_parser("make", help="Write sub-990 (runs 1 and 2) and the answer key into a practice folder.")
    make_p.add_argument("--out-dir", required=True,
                        help="The practice folder (on Box: a 'practice' folder next to 'derivatives').")
    make_p.add_argument("--seed", type=int, default=None, help="Random seed (default: a fresh random one).")
    make_p.add_argument("--key-copy", default=None, help="Also write a plain-JSON copy of the key here "
                                                         "(keep it outside the practice folder).")
    score_p = sub.add_parser("score", help="Score an RA's saved annotations against the answer key.")
    score_p.add_argument("--key", required=True, help="practice_key.pebbl (or a plain-JSON copy).")
    score_p.add_argument("--annotations", required=True, help="The RA's *_annotations_<initials>.json.")
    score_p.add_argument("--run", default=None, help="1 or 2 (default: from the file name).")
    score_p.add_argument("--channels", default=None, help="Comma-separated channels to score (default: every "
                                                          "channel in the RA's file).")
    args = parser.parse_args(argv)
    if args.command == "make":
        make(args.out_dir, args.seed, args.key_copy)
        return 0
    channels = [c.strip() for c in args.channels.split(",")] if args.channels else None
    overall, _ = score(args.key, args.annotations, channels, args.run)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
