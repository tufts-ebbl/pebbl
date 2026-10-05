"""
Data loading for the multisignal annotation prototype.

Two sources of data:

1. load_physio_tsv(): reads an existing *_physio.tsv.gz (+ its .json sidecar)
   exactly as already produced by physioProcess/physioBatch.py. Nothing here
   writes to or imports from that pipeline -- it only reads the stable file
   format it already emits.

2. generate_synthetic_demo(): fabricates a physio.tsv.gz-shaped DataFrame for
   testing this tool before real EMG data exists. ECG/RSP/PPG/EDA are
   plausible neurokit2 simulations; the two EMG placeholder channels
   (corrugator, zygomatic) are deliberately zero throughout so an RA never
   mistakes them for real signal worth correcting.

resolve_real_input_path() builds a real physio.tsv.gz path from a Box
derivatives path + subject + run (matching the existing notebooks'
sub-XXX/ses-runY/beh/... convention), prompting interactively for any of
the three not passed in -- mirroring how the notebooks prompt for these
same fields via their ipywidgets form.
"""

import gzip
import json
import os

import mne
import neurokit2 as nk
import numpy as np
import pandas as pd

# Column order/names exactly as written by physioBatch.py's save_tsv_gz() call
REAL_PHYSIO_COLUMNS = [
    "ecg", "rsp", "ppg", "eda",
    "ecg_peaks", "rsp_peaks", "ppg_peaks", "eda_peaks",
]


def resolve_box_path_and_subject(box_path, subject):
    """Prompts for box_path/subject if missing; normalizes subject to e.g. '001'."""
    if not box_path:
        box_path = input("Box derivatives path "
                          "(e.g. C:\\Users\\you\\Box\\DATA\\Processed\\physioProcessing\\derivatives): ").strip()
    if not subject:
        # No "next available subject" suggestion (HLU, 2026-09-27): RAs are
        # assigned files and look them up in the Google tracking spreadsheet.
        subject = input("Subject (e.g. 001): ").strip()
    subject = subject.replace("sub-", "").zfill(3)
    return box_path, subject


def build_run_path(box_path, subject, run):
    """Pure path construction (no prompting, no existence check)."""
    run = str(run).replace("run", "").replace("ses-", "")
    sub_dir = f"sub-{subject}"
    file_stem = f"sub-{subject}_ses-run{run}_task-sdi"
    return os.path.join(box_path, sub_dir, f"ses-run{run}", "beh", f"{file_stem}_physio.tsv.gz")


def resolve_real_input_path(input_path, box_path, subject, run):
    """
    Returns a single real *_physio.tsv.gz path (for Stage B, which always
    reviews exactly one run). If input_path is given, uses it directly.
    Otherwise builds one from box_path/subject/run using the
    sub-XXX/ses-runY/beh/sub-XXX_ses-runY_task-sdi_physio.tsv.gz convention,
    prompting interactively for whichever of box_path/subject/run wasn't
    already passed in.
    """
    if input_path:
        return input_path

    box_path, subject = resolve_box_path_and_subject(box_path, subject)
    if not run:
        run = input("Run (1 or 2): ").strip()
    run = run.replace("run", "").replace("ses-", "")

    resolved_path = build_run_path(box_path, subject, run)
    print(f"Resolved input path: {resolved_path}")
    if not os.path.exists(resolved_path):
        raise SystemExit(
            f"No file found at {resolved_path}\n"
            f"Double-check the Box path, subject ({subject}), and run ({run})."
        )
    return resolved_path


def resolve_subject_run_paths(input_run1, input_run2, box_path, subject):
    """
    Returns {"1": path_or_None, "2": path_or_None} for Stage A, which
    (like step1_qrs_template.ipynb) works per SUBJECT, not per run: it
    builds one QRS template and applies it to every run that exists for
    that subject. Explicit --input-run1/--input-run2 paths take priority;
    otherwise both runs are looked for under box_path/subject, prompting
    for box_path/subject if missing (never for "run" -- both are always
    attempted). At least one run must actually exist on disk.
    """
    if input_run1 or input_run2:
        paths = {
            "1": input_run1 if (input_run1 and os.path.exists(input_run1)) else None,
            "2": input_run2 if (input_run2 and os.path.exists(input_run2)) else None,
        }
    else:
        box_path, subject = resolve_box_path_and_subject(box_path, subject)
        paths = {}
        for run in ("1", "2"):
            candidate = build_run_path(box_path, subject, run)
            paths[run] = candidate if os.path.exists(candidate) else None

    found = {run: p for run, p in paths.items() if p}
    if not found:
        raise SystemExit(
            f"No run 1 or run 2 physio.tsv.gz found for this subject. Checked: {paths}\n"
            f"Double-check the Box path and subject."
        )
    print(f"Found run(s): {list(found.keys())} -> {found}")
    return paths


def sidecar_path_for(tsv_gz_path):
    return tsv_gz_path.replace("_physio.tsv.gz", "_physio.json")


def read_sidecar(json_sidecar_path):
    """
    The sidecar as a dict ({} if the file doesn't exist). Read as UTF-8 with
    an optional byte-order mark: the default (cp1252 on Windows) worked only
    while every sidecar happened to be plain ASCII (ppg-plan §3.1 #8).
    """
    if not json_sidecar_path or not os.path.exists(json_sidecar_path):
        return {}
    with open(json_sidecar_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_physio_tsv(tsv_gz_path, json_sidecar_path=None):
    """
    Reads an existing sub-###_ses-run#_task-sdi_physio.tsv.gz file.

    Returns (df, sfreq) where df has the columns from the JSON sidecar
    (falling back to REAL_PHYSIO_COLUMNS if no sidecar is found).
    """
    if json_sidecar_path is None:
        json_sidecar_path = sidecar_path_for(tsv_gz_path)

    if os.path.exists(json_sidecar_path):
        meta = read_sidecar(json_sidecar_path)
        columns = meta.get("Columns", REAL_PHYSIO_COLUMNS)
        if "SamplingFrequency" not in meta:
            # Previously a silent 1000 Hz default (audit finding M4): a wrong
            # rate would mis-time every peak and window without any error.
            raise SystemExit(f"{json_sidecar_path} has no SamplingFrequency entry. Please ask the lab "
                             f"staff to regenerate this file's sidecar before reviewing it.")
        sfreq = meta["SamplingFrequency"]
    else:
        columns = REAL_PHYSIO_COLUMNS
        sfreq = 1000
        print(f"No JSON sidecar found at {json_sidecar_path}; "
              f"assuming default columns and {sfreq} Hz.")

    df = pd.read_csv(tsv_gz_path, sep="\t", compression="gzip", header=None, names=columns)

    # boolean peak columns are stored as 0/1 in the real files. A missing
    # value means "no peak": astype(bool) alone turned NaN into True, i.e. a
    # false peak (audit finding L3).
    for col in df.columns:
        if col.endswith("_peaks"):
            df[col] = df[col].fillna(0).astype(bool)

    return df, sfreq


MACHINE_QC_FLAGGED_WARNING_FRACTION = 0.9
MACHINE_QC_END_GAP_NOTE_SEC = 5.0


def _as_index(value):
    """An integral JSON number as int (5, 5.0), else None (null, NaN, 5.5, "5", True)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


REASON_LABELS = {
    "no_coverage": "no CareTaker reading",
    "out_of_range": "out of range",
    "ordering": "SBP less than 10 above DBP",
    "no_vitals_data": "no CareTaker vitals for this run",
    "zero_run": "device zero run",
    "bridged": "far from a native CareTaker sample",
    "window_rule": "pulse-quality window rule",
    "no_pulse_data": "no CareTaker pulse data",
    "ppg_dropout": "no usable finger pulse",
    "counter_loss": "CareTaker pulse samples lost in transfer",
}
"""
Plain names for MachineQC SegmentsByReason keys (the complete set, per the
physioProcess maintainer, 2026-09-26): PPG's zero_run / bridged /
window_rule, or no_pulse_data (R1); from run 8, SBP/DBP's no_coverage /
out_of_range / ordering, or no_vitals_data (decision B1); from run 9,
SBP/DBP's ppg_dropout (HLU's decision (b), 2026-09-27: BP inside a PPG
device dropout, plus a recovery margin after it); from run 10, PPG's
counter_loss (pulse rows the CareTaker app lost in transfer, plus the
beat-wide bracket around the splice; present, possibly empty, in every pulse
run; not part of ppg_dropout, since the device kept measuring BP there).
RSP and EDA have none.
An unknown key is shown by its own name.
"""


def _reason_spans(by_reason, n_samples):
    """
    ({reason: [[start, end), ...]} or None, invalid). None with invalid=False
    when there is no breakdown; None with invalid=True when there is one but
    it isn't a dict of lists of valid spans. A malformed breakdown is ignored
    rather than half-used, and never crashes.
    """
    if by_reason is None:
        return None, False
    if not isinstance(by_reason, dict):
        return None, True
    parsed = {}
    for reason, spans in by_reason.items():
        if not isinstance(reason, str) or not isinstance(spans, list):
            return None, True
        out = []
        for span in spans:
            if not (isinstance(span, (list, tuple)) and len(span) == 2):
                return None, True
            start, end = _as_index(span[0]), _as_index(span[1])
            if start is None or end is None or not (0 <= start < end <= n_samples):
                return None, True
            out.append([start, end])
        parsed[reason] = out
    return parsed, False


COVERAGE_REASONS = ("no_coverage", "no_vitals_data")
"""Reasons meaning "no CareTaker reading here": those rows are NaN, blank in the viewer."""


def coverage_only(entry):
    """
    True when the sidecar's (valid) reasons say every flagged stretch is only
    a missing CareTaker reading, so there is no signal there to judge, e.g.
    SBP on a run CareTaker covers for 5% (run-8 review, 2026-09-26).
    """
    by_reason = entry.get("by_reason")
    if not by_reason or entry.get("by_reason_invalid"):
        return False
    given = [reason for reason, spans in by_reason.items() if spans]
    return bool(given) and all(reason in COVERAGE_REASONS for reason in given)


def reason_summary(by_reason):
    """
    'no CareTaker reading (11), out of range (1)': the non-empty reasons and
    their span counts, or '' when there is no breakdown. The counts need not
    add up to the number of bad stretches: those are the reasons' union.
    """
    return ", ".join(f"{REASON_LABELS.get(reason, reason)} ({len(spans)})"
                     for reason, spans in (by_reason or {}).items() if spans)


def machine_qc_segments(meta, n_samples, only=None, labels=None):
    """
    Reads the new physioProcess sidecar's MachineQC block (sidecar plan
    Phase 2) into {channel: {"segments": [[start, end), ...], "rule": text,
    "proportion_bad": float, "by_reason": {...} or None, "by_reason_invalid":
    bool}}, plus a note (None if everything was usable). by_reason is the
    validated SegmentsByReason (see _reason_spans()); by_reason_invalid says
    one was given but is malformed. `only` limits both the result and the note to
    those channel keys (e.g. this session's fresh segment channels); `labels`
    ({key: label}) names channels in the note.

    Guards, each of which means "start blank rather than pre-fill wrongly"
    (never a crash):
      - no MachineQC block (an old sidecar);
      - Provenance.NumberOfSamples missing, not a whole number, or different
        from the tsv's row count (the QC rows would not line up with the data);
      - a channel whose BadSegments is missing or not a list, or holds a span
        that isn't two whole numbers with 0 <= start < end <= n_samples, is
        skipped (other channels still pre-fill) and gets no seed record --
        a missing list is NOT read as "the machine found nothing".
    Spans are used exactly as given: one that stops short of the file end
    is not extended (user's decision (a), 2026-09-25).
    """
    qc = (meta or {}).get("MachineQC")
    if not isinstance(qc, dict) or not isinstance(qc.get("Channels"), dict):
        return {}, ("This file's sidecar has no machine quality check (it was processed before physioProcess "
                    "added one), so bad-segment channels start blank.")
    declared = _as_index(((meta or {}).get("Provenance") or {}).get("NumberOfSamples"))
    if declared is None:
        return {}, ("The sidecar's machine quality check has no usable sample count to verify against, so it "
                    "isn't used and bad-segment channels start blank. Please tell the lab staff.")
    if declared != int(n_samples):
        return {}, (f"The sidecar's machine quality check describes {declared} samples but the data file has "
                    f"{n_samples}, so it isn't used (bad-segment channels start blank). Please tell the lab staff.")
    result, skipped = {}, []
    for ch_key, entry in qc["Channels"].items():
        if only is not None and ch_key not in only:
            continue
        if not isinstance(entry, dict):
            continue
        segments = entry.get("BadSegments")
        parsed = []
        valid = isinstance(segments, list)
        for span in segments if valid else []:
            if not (isinstance(span, (list, tuple)) and len(span) == 2):
                valid = False
                break
            start, end = _as_index(span[0]), _as_index(span[1])
            if start is None or end is None or not (0 <= start < end <= n_samples):
                valid = False
                break
            parsed.append([start, end])
        if not valid:
            skipped.append((labels or {}).get(ch_key, ch_key))
            continue
        by_reason, by_reason_invalid = _reason_spans(entry.get("SegmentsByReason"), n_samples)
        result[ch_key] = {
            "segments": parsed,
            "rule": entry.get("Rule", ""),
            "proportion_bad": entry.get("ProportionBad"),
            "by_reason": by_reason,
            "by_reason_invalid": by_reason_invalid,
        }
    note = None
    if skipped:
        note = (f"The machine quality check for {', '.join(skipped)} is missing its list of bad stretches or has "
                f"an invalid one (outside the file, empty or reversed), so that channel starts blank. Please tell "
                f"the lab staff.")
    return result, note


def machine_qc_flagged_fraction(entry, n_samples):
    return sum(end - start for start, end in entry["segments"]) / n_samples if n_samples else 0.0


# No-reading (no_coverage) stretches inside the run shorter than this are
# slivers between CareTaker readings more than 4 s apart (each reading fills
# only 2 s each side). The banner tells RAs to keep them (HLU, 2026-10-02).
NO_READING_SLIVER_SEC = 2.0


def short_no_reading_gaps(entry, n_samples, sfreq):
    """How many of a channel's no_coverage stretches lie inside the run and last under NO_READING_SLIVER_SEC."""
    spans = (entry.get("by_reason") or {}).get("no_coverage") or []
    return sum(1 for start, end in spans
               if start > 0 and end < n_samples and (end - start) / sfreq < NO_READING_SLIVER_SEC)


def machine_qc_notes(qc_channels, channel_keys, n_samples, sfreq, labels, value_ranges=None, units=None):
    """
    Opening-banner lines for pre-filled channels: how much the machine
    flagged, and when its last span stops short of the file end. When it
    flagged more than 90% of the run (e.g. sub-001's flat RSP belt), the usual
    "check each one" line is replaced by a warning to KEEP the flag unless
    sure: the viewer stretches every channel to fill its row, so a
    disconnected sensor's tiny noise can look like a real signal
    (checkpoint-1 review). value_ranges ({key: (min, mean, max)}) and units
    ({key: unit}) let that warning quote the channel's actual range. When the
    sidecar gives the machine's reasons (SegmentsByReason; SBP/DBP from run
    8), a line says why it flagged them.
    """
    lines = []
    for ch_key in channel_keys:
        entry = qc_channels.get(ch_key)
        if not entry or not entry["segments"]:
            continue
        label = labels.get(ch_key, ch_key)
        fraction = machine_qc_flagged_fraction(entry, n_samples)
        if fraction > MACHINE_QC_FLAGGED_WARNING_FRACTION and coverage_only(entry):
            # Not a dead sensor: the flagged rows are empty (run-8 review).
            lines.append(f"{label}: CareTaker has readings for only {1 - fraction:.1%} of this run; the machine "
                         f"flagged the rest because there is no reading there (it shows blank). Keep those flags "
                         f"and review the recorded part as usual.")
        elif fraction > MACHINE_QC_FLAGGED_WARNING_FRACTION:
            span_text = ""
            if value_ranges and ch_key in value_ranges:
                lo, _mean, hi = value_ranges[ch_key]
                span_text = (f" This run's {label} only ranges from {lo:.3g} to {hi:.3g} "
                             f"{(units or {}).get(ch_key, '')}".rstrip() + ".")
            lines.append(f"{label}: WARNING -- the machine flagged nearly all of this run ({fraction:.1%}) as bad; "
                         f"the sensor may have been disconnected. KEEP this flag unless you are sure the signal is "
                         f"real: the viewer stretches every channel to fill its row, so a dead sensor's tiny noise "
                         f"can look like a normal signal.{span_text} If unsure, ask the lab staff.")
        else:
            lines.append(f"{label}: starts with {len(entry['segments'])} machine-flagged bad stretch(es) "
                         f"({fraction:.1%} of the run). Check each one: remove any that are fine, add any it missed.")
        why = reason_summary(entry.get("by_reason"))
        if why:
            lines.append(f"{label}: why the machine flagged them: {why}.")
        slivers = short_no_reading_gaps(entry, n_samples, sfreq)
        if slivers:
            lines.append(f"{label}: short no-reading stretches (under 2 s) are gaps between CareTaker readings; the "
                         f"trace is blank there. Keep them ({slivers} in this run).")
        gap_sec = (n_samples - max(end for _start, end in entry["segments"])) / sfreq
        if 0 < gap_sec <= MACHINE_QC_END_GAP_NOTE_SEC:
            lines.append(f"{label}: the last flagged stretch stops {gap_sec:.2f} s before the end of the file "
                         f"(the machine check works in whole windows). Extend it if the end is bad too.")
    return lines


SIDECAR_UNIT_LABELS = {
    "mv": "mV",
    "v": "V",
    "microsiemens": "µS",
    "arbitrary": "AU",
    "mmhg": "mmHg",
    "code": "AU",
}
"""
Display labels for the new sidecar's per-column Units (sidecar plan Phase
3). Labels only: the data are never rescaled (see channel_config.py's
DEFAULTS override). "boolean" (peak columns) is never displayed.
"""


def sidecar_unit_labels(meta, channel_configs):
    """
    {mne_type: display unit} from the sidecar's per-column Units, for the
    channels in channel_configs present in the sidecar, plus a list of
    warnings when two channels sharing an MNE type disagree (MNE's unit
    label is per TYPE, so only one can win; the first channel's is kept).
    Unknown unit strings are skipped with a warning.
    """
    units, warnings = {}, []
    for ch_key, cfg in channel_configs.items():
        entry = (meta or {}).get(ch_key)
        if not isinstance(entry, dict) or not entry.get("Units"):
            continue
        raw_unit = str(entry["Units"]).strip()
        if raw_unit.lower() == "boolean":
            continue
        label = SIDECAR_UNIT_LABELS.get(raw_unit.lower())
        if label is None:
            warnings.append(f"{cfg.get('label', ch_key)}: unknown unit '{raw_unit}' in the sidecar; "
                            f"keeping the default label.")
            continue
        mne_type = cfg["mne_type"]
        if mne_type in units and units[mne_type] != label:
            warnings.append(f"{cfg.get('label', ch_key)}: the sidecar says '{label}' but another channel of the "
                            f"same display type says '{units[mne_type]}'; showing '{units[mne_type]}'.")
            continue
        units[mne_type] = label
    return units, warnings


def apply_unit_labels(units):
    """
    Sets MNE's per-type display unit labels, and that type's display scaling
    to 1.0: the data are already in the labeled unit, so they are relabeled,
    never rescaled (channel_config.py's rule). Without the scaling, a type
    channel_config doesn't override (e.g. "emg", whose MNE default multiplies
    by 1e6) would be relabeled but still shown 1,000,000x (checkpoint-2 review).
    """
    for mne_type, label in units.items():
        mne.defaults.DEFAULTS["units"][mne_type] = label
        mne.defaults.DEFAULTS["si_units"][mne_type] = label
        mne.defaults.DEFAULTS["scalings"][mne_type] = 1.0


def sidecar_descriptions(meta, channel_keys):
    """{channel: Description text} for the channels whose sidecar entry has one."""
    return {k: meta[k]["Description"] for k in channel_keys
            if isinstance((meta or {}).get(k), dict) and meta[k].get("Description")}


def generate_synthetic_demo(duration_sec=60, sfreq=1000, heart_rate=70, seed=42):
    """
    Builds a demo DataFrame with the same shape as a real physio.tsv.gz, plus
    two flatline EMG placeholder channels, entirely for exercising this tool
    without touching Box or real subject data.
    """
    # neurokit2's simulators (e.g. rsp_simulate) don't int()-cast duration*sampling_rate
    # internally, and raise a TypeError on a float duration -- force int seconds here.
    duration_sec = int(round(duration_sec))
    n_samples = int(duration_sec * sfreq)

    ecg = nk.ecg_simulate(duration=duration_sec, sampling_rate=sfreq, heart_rate=heart_rate, random_state=seed)
    rsp = nk.rsp_simulate(duration=duration_sec, sampling_rate=sfreq, respiratory_rate=15, random_state=seed)
    ppg = nk.ppg_simulate(duration=duration_sec, sampling_rate=sfreq, heart_rate=heart_rate, random_state=seed)
    eda = nk.eda_simulate(duration=duration_sec, sampling_rate=sfreq, scr_number=int(duration_sec / 15), random_state=seed)

    ecg_signals, _ = nk.ecg_process(ecg, sampling_rate=sfreq)
    ecg_peaks = (ecg_signals["ECG_R_Peaks"] == 1).to_numpy()

    rsp_signals, _ = nk.rsp_process(rsp, sampling_rate=sfreq)
    rsp_peaks = (rsp_signals["RSP_Peaks"] == 1).to_numpy()

    ppg_signals, _ = nk.ppg_process(ppg, sampling_rate=sfreq)
    ppg_peak_col = [c for c in ppg_signals.columns if c.startswith("PPG_Peaks")][0]
    ppg_peaks = (ppg_signals[ppg_peak_col] == 1).to_numpy()

    eda_signals, _ = nk.eda_process(eda, sampling_rate=sfreq)
    eda_peaks = (eda_signals["SCR_Peaks"] == 1).to_numpy()

    # Literal zero, not simulated noise or bursts: nothing here should look
    # like real muscle activity worth reviewing. Safe now that EMG's display
    # scale (compute_scalings) is borrowed from the other channels rather
    # than computed from this channel's own data.
    emg_cor = np.zeros(n_samples)
    emg_zyg = np.zeros(n_samples)

    def fit(arr):
        arr = np.asarray(arr, dtype=float)
        if len(arr) >= n_samples:
            return arr[:n_samples]
        return np.pad(arr, (0, n_samples - len(arr)))

    def fit_bool(arr):
        arr = np.asarray(arr, dtype=bool)
        if len(arr) >= n_samples:
            return arr[:n_samples]
        return np.pad(arr, (0, n_samples - len(arr)), constant_values=False)

    df = pd.DataFrame({
        "ecg": fit(ecg),
        "rsp": fit(rsp),
        "ppg": fit(ppg),
        "eda": fit(eda),
        "ecg_peaks": fit_bool(ecg_peaks),
        "rsp_peaks": fit_bool(rsp_peaks),
        "ppg_peaks": fit_bool(ppg_peaks),
        "eda_peaks": fit_bool(eda_peaks),
        "emg_cor": emg_cor,
        "emg_zyg": emg_zyg,
    })

    return df, sfreq


DEGENERATE_SCALE_FLOOR = 1e-8


def compute_scalings(df, channel_configs):
    """
    Builds a per-channel-TYPE scalings dict for raw.plot(), because MNE's
    'auto'/default scalings assume real physiological units (volts, etc.)
    that our data isn't in.

    Fully data-driven, not hardcoded per channel: every channel's scale is
    a robust (2.5th-97.5th percentile) range computed from its OWN data,
    NaN-safe (real files can have long dropped-signal stretches -- e.g. a
    real PPG channel that's ~91% NaN was seen in actual test data). If a
    channel's own data is degenerate -- entirely NaN, or ~zero variance
    like today's literal-zero EMG placeholder -- it instead borrows the
    median scale of the file's other, non-degenerate channels, so it still
    renders at a sane, comparable row height rather than a NaN/zero scale
    breaking the plot or dwarfing everything else.

    Because this checks the actual data rather than a fixed per-channel
    flag, a channel that's a placeholder TODAY (zero EMG) but carries real
    signal in a FUTURE file (real corrugator/zygomatic EMG, or PPG in a
    study that collects it) automatically gets its own properly auto-fit
    scale the moment real data replaces the placeholder -- no config
    change needed.
    """
    raw_scales = {}  # mne_type -> scale (may be NaN)

    for ch_key, cfg in channel_configs.items():
        if ch_key not in df.columns:
            continue
        mne_type = cfg["mne_type"]
        if mne_type in raw_scales:
            continue  # first channel of a shared type sets the scale (MNE scalings are per-type)

        data = df[ch_key].to_numpy(dtype=float)
        if np.all(np.isnan(data)):
            raw_scales[mne_type] = np.nan
        else:
            raw_scales[mne_type] = (np.nanpercentile(data, 97.5) - np.nanpercentile(data, 2.5)) / 2

    good_scales = [s for s in raw_scales.values() if np.isfinite(s) and s > DEGENERATE_SCALE_FLOOR]
    borrowed = float(np.median(good_scales)) if good_scales else 1.0

    return {
        mne_type: (scale if (np.isfinite(scale) and scale > DEGENERATE_SCALE_FLOOR) else borrowed)
        for mne_type, scale in raw_scales.items()
    }


def compute_channel_value_ranges(df, channel_configs):
    """
    Returns {ch_key: (min, mean, max)} for every SEGMENT-mode channel
    present in df -- see channel_config.py's module docstring for why
    this is scoped to segment-mode specifically (the channels MNE's
    remove_dc-relative crosshair can't show absolute values for, where
    that absolute baseline is also the actual thing an RA needs to judge,
    not just waveform shape). NaN-safe (real CareTaker sbp/dbp data has
    real dropped-signal stretches, same as PPG). A channel that's entirely
    NaN, or not present in df at all, is simply omitted from the result.
    """
    ranges = {}
    for ch_key, cfg in channel_configs.items():
        if cfg.get("annotation_mode") != "segment" or ch_key not in df.columns:
            continue
        series = df[ch_key].dropna()
        if series.empty:
            continue
        ranges[ch_key] = (float(series.min()), float(series.mean()), float(series.max()))
    return ranges


def format_channel_value_ranges(ranges, channel_configs):
    """
    Formats compute_channel_value_ranges()'s output into one human-readable
    line per channel, e.g. "SBP: 62.10-131.40 mmHg (mean 94.64)" -- shared
    by both the Stage B startup console banner and the compact summary
    appended to processing_log.csv's Notes field, so the two stay
    consistent with each other.
    """
    lines = []
    for ch_key, (lo, mean, hi) in ranges.items():
        cfg = channel_configs[ch_key]
        unit = mne.defaults.DEFAULTS["units"].get(cfg["mne_type"], "")
        lines.append(f"{cfg['label']}: {lo:.2f}-{hi:.2f} {unit} (mean {mean:.2f})")
    return lines


def generate_synthetic_demo_two_runs(duration_sec_each=90, sfreq=1000, heart_rate=70, seed=42):
    """
    For testing Stage A's "one template, applied to both runs" workflow:
    generates one continuous synthetic recording and splits it into two
    halves standing in for run 1 and run 2 of the same (synthetic) subject
    -- same underlying QRS morphology, different time content, so a
    template built from one half should transfer meaningfully to the other.
    """
    duration_sec_each = int(round(duration_sec_each))
    full_df, sfreq = generate_synthetic_demo(
        duration_sec=2 * duration_sec_each, sfreq=sfreq, heart_rate=heart_rate, seed=seed
    )
    midpoint = len(full_df) // 2
    df_run1 = full_df.iloc[:midpoint].reset_index(drop=True)
    df_run2 = full_df.iloc[midpoint:].reset_index(drop=True)
    return {"1": df_run1, "2": df_run2}, sfreq


def build_raw(df, channel_configs, sfreq):
    """
    Builds an mne.RawArray containing every channel in channel_configs that
    is actually present as a column in df, tagged with the mne channel type
    declared in channel_configs (channel_config.py).
    """
    available = [ch for ch in channel_configs if ch in df.columns]
    if not available:
        raise ValueError("None of the configured channels were found in the data.")

    data = df[available].to_numpy().T
    ch_types = [channel_configs[ch]["mne_type"] for ch in available]

    info = mne.create_info(ch_names=available, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw
