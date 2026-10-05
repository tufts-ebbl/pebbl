"""
Standalone port of the wavelet-filter + template-cross-correlation ECG peak
detection approach from step1_qrs_template.ipynb -- the approach found
necessary for ECG collected in the MRI scanner (and which also works fine
for clean in-lab ECG).

This is an independent re-implementation, not an import from the notebook
or from physioProcess/physioFunctions.py, so nothing existing is touched.
The one behavioral difference: instead of systole's correct_peaks() for the
final refractory-period cleanup, enforce_refractory_period() below is a
small from-scratch equivalent, so this new environment doesn't need to add
systole as a dependency. It's simpler than systole's fuller ectopic-beat
correction -- worth upgrading later if that turns out to matter.
"""

import json
import shutil
import os
from datetime import datetime

import mne
import neurokit2 as nk
import numpy as np
import pandas as pd
import pywt
from scipy import signal

from annotation_io import DEFAULT_SNAP_WINDOW_SEC, DRAWN_PEAK_CLIMB_SEC, point_sample_from_annotation
from channel_config import GUIDE_OPT_IN, GUIDE_TEXT
from provenance import new_pipeline_ppg
from template_preview_gui import show_template_preview

POINT_LABEL = "peak_ecg"


QRS_BAND_UPPER_HZ = 31.25


def apply_wavelet_filter(ecg_array, wavelet_type="sym4", level=None, sampling_rate=1000):
    """
    Same approach as step1_qrs_template.ipynb's apply_wavelet_filter(): keep
    only the two wavelet detail levels covering roughly 8-31 Hz (the QRS
    band) and reconstruct. The notebook hard-coded levels D5 and D6 of an
    8-level decomposition, which are 15.6-31.25 Hz and 7.8-15.6 Hz ONLY at
    1000 Hz (detail level j spans fs/2^(j+1) to fs/2^j). The levels are now
    chosen from the sampling rate, so the same band is kept at any rate;
    at 1000 Hz this is exactly D5 + D6 of an 8-level decomposition, as
    before (audit finding M4).
    """
    j_high = max(1, int(round(np.log2(sampling_rate / QRS_BAND_UPPER_HZ))))  # 5 at 1000 Hz
    j_low = j_high + 1                                                           # 6 at 1000 Hz
    if level is None:
        level = max(8, j_low + 2)
    coeffs = pywt.wavedec(ecg_array, wavelet_type, level=level)
    coeffs_filtered = [np.zeros_like(c) for c in coeffs]
    # wavedec returns [cA_level, cD_level, ..., cD_1]: detail level j sits at index level - j + 1.
    for j in (j_low, j_high):
        coeffs_filtered[level - j + 1] = coeffs[level - j + 1]
    reference_signal = pywt.waverec(coeffs_filtered, wavelet_type)[: len(ecg_array)]
    return reference_signal


def enforce_refractory_period(peaks_indices, sampling_rate, min_rr_ms=400):
    """
    Keeps only peaks at least min_rr_ms apart, dropping the second of any
    pair that's closer together than that (cleans up double-detections from
    the cross-correlation step). A simpler stand-in for systole's
    correct_peaks(), which does fuller ectopic-beat interpolation.
    """
    peaks_indices = np.sort(np.asarray(peaks_indices))
    if len(peaks_indices) == 0:
        return peaks_indices
    min_samples = int((min_rr_ms / 1000) * sampling_rate)
    valid = [peaks_indices[0]]
    for idx in peaks_indices[1:]:
        if idx - valid[-1] >= min_samples:
            valid.append(idx)
    return np.array(valid)


DEFAULT_REFRACTORY_CAP_MS = 350
DEFAULT_MIN_DISTANCE_CAP_MS = 250
REFRACTORY_FRACTION_OF_RR = 0.6
LOCAL_SHORT_INTERVAL_FRACTION = 0.5
LOCAL_MEDIAN_HALF_WIDTH = 5
DEFAULT_THRESHOLD_WINDOW_SEC = 5.0
"""
Peak-spacing rules for extract_and_refine_peaks() (audit finding C1,
multisignal-annotation_audit_20260923-1901.md). The original fixed
find_peaks(distance=600 ms) made it impossible to detect two beats closer
than 600 ms apart, i.e. any heart rate above 100 bpm lost roughly half its
beats (recall 0.67 at 100 bpm, 0.46 at 110 bpm on synthetic ECG). Simply
lowering that fixed distance nearly doubled detections on real in-scanner
ECG (T waves/artifacts), so the replacement has three parts:
  1. A short minimum distance (<= 250 ms) and a refractory period of
     min(350 ms, 0.6 x the RA-corrected template window's median R-R), where
     a close pair keeps the candidate with the HIGHER template correlation
     (not simply the earlier one).
  2. A local cleanup: any interval shorter than 0.5 x the running median of
     the surrounding 11 intervals drops the weaker peak of that pair, so the
     rule follows heart-rate changes within a run.
  3. A local detection threshold (mean + SD of the correlation over a
     sliding 5-s window) instead of one whole-run threshold.
Validated in audit_20260923/exp_c1_rules.py: recall/precision >= 0.995 at
60-180 bpm and for 65->130 and 130->70 bpm changes within a run (the old
rule: recall 0.42-0.68 at >= 100 bpm). On real in-scanner sub-085 run 2,
compared with the batch peaks (not ground truth), it left fewer long gaps
than the old rule (23-29 vs 28-47) but produced more detections (758-968 vs
713-735; the batch has 745), i.e. more spurious markers for the RA to
delete -- most with a template built from the pre-task 0-20 s window.
"""


def _keep_stronger_peaks(indices, strengths, min_samples):
    """Accepts peaks strongest-first, rejecting any within min_samples of an accepted one."""
    indices = np.asarray(indices)
    accepted = []
    for i in np.argsort(-np.asarray(strengths)):
        if all(abs(indices[i] - a) >= min_samples for a in accepted):
            accepted.append(indices[i])
    return np.sort(np.array(accepted, dtype=int))


def _remove_locally_short_intervals(indices, strength_of, fraction=LOCAL_SHORT_INTERVAL_FRACTION,
                                    half_width=LOCAL_MEDIAN_HALF_WIDTH):
    """
    Repeatedly finds the interval that is shortest relative to the running
    median of its neighbors; if it is below `fraction` of that median, drops
    the weaker (by strength_of) of its two peaks. Stops when none remain.
    """
    from scipy.ndimage import median_filter
    kept = list(np.asarray(indices, dtype=int))
    while len(kept) > 3:
        intervals = np.diff(kept).astype(float)
        local_median = median_filter(intervals, size=2 * half_width + 1, mode="nearest")
        ratios = intervals / local_median
        worst = int(np.argmin(ratios))
        if ratios[worst] >= fraction:
            break
        a, b = kept[worst], kept[worst + 1]
        kept.remove(a if strength_of(a) < strength_of(b) else b)
    return np.array(kept, dtype=int)


def _local_threshold(correlation, window_samples):
    """Mean + SD of the correlation over a centered sliding window."""
    from scipy.ndimage import uniform_filter1d
    mean = uniform_filter1d(correlation, window_samples, mode="nearest")
    mean_sq = uniform_filter1d(correlation * correlation, window_samples, mode="nearest")
    return mean + np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def spacing_from_reference_rr(reference_rr_ms):
    """Returns (min_distance_ms, refractory_ms) for extract_and_refine_peaks(); see the constants above."""
    refractory = DEFAULT_REFRACTORY_CAP_MS
    if reference_rr_ms is not None and np.isfinite(reference_rr_ms) and reference_rr_ms > 0:
        refractory = min(DEFAULT_REFRACTORY_CAP_MS, REFRACTORY_FRACTION_OF_RR * reference_rr_ms)
    return min(DEFAULT_MIN_DISTANCE_CAP_MS, refractory), refractory


def snap_to_local_max(signal_array, indices, window_samples):
    signal_array = np.asarray(signal_array)
    refined = []
    for idx in indices:
        start = max(0, idx - window_samples)
        end = min(len(signal_array), idx + window_samples)
        if end <= start:
            refined.append(idx)
            continue
        window = np.asarray(signal_array[start:end], dtype=float)
        refined.append(idx if np.all(np.isnan(window)) else start + int(np.nanargmax(window)))
    return np.array(sorted(set(refined)))


def filter_safe_peaks(peaks, signal_length, edge_start_sec=0.4, edge_end_sec=0.6, sampling_rate=1000):
    """
    Drops peaks too close to the window edges to avoid NaN-padded epochs.
    The margins are in seconds (formerly fixed 400/600-SAMPLE margins, correct
    only at 1000 Hz -- audit finding M4).
    """
    edge_start, edge_end = edge_start_sec * sampling_rate, edge_end_sec * sampling_rate
    return [p for p in peaks if edge_start < p < (signal_length - edge_end)]


def build_qrs_template(raw_ecg_window, wavelet_filtered_window, ra_corrected_peaks, sampling_rate=1000):
    """
    Given an RA-corrected set of peak indices within a short (e.g. 20s)
    window, builds the averaged QRS template from the wavelet-filtered
    signal, and the sample offset between the template's geometric center
    and where the true R peak actually falls within it (needed later to
    align cross-correlation results, which peak at the alignment point
    between two signals, not necessarily at the R peak itself).

    Mirrors step1_qrs_template.ipynb's on_save_and_push_clicked() template
    construction exactly.
    """
    safe_peaks = filter_safe_peaks(ra_corrected_peaks, len(raw_ecg_window), sampling_rate=sampling_rate)
    if len(safe_peaks) < 2:
        raise ValueError(
            f"Only {len(safe_peaks)} usable peak(s) in this window -- need at least 2 "
            "to build a QRS template. Correct more peaks, or pick a cleaner window."
        )

    epochs = nk.ecg_segment(wavelet_filtered_window, safe_peaks, sampling_rate=sampling_rate, show=False)

    all_beats = [epochs[key]["Signal"].to_numpy() for key in epochs.keys()]
    beat_matrix = np.array(all_beats)
    qrs_template = pd.DataFrame(np.nanmean(beat_matrix, axis=0), columns=["y"])

    center = int(len(qrs_template) / 2)
    r_position = np.where(qrs_template["y"] == np.max(qrs_template["y"][:center]))[0][0]
    adjustment = center - r_position

    return qrs_template, adjustment


def extract_and_refine_peaks(raw_ecg_full, wavelet_filtered_full, qrs_template, adjustment,
                              sampling_rate=1000, min_distance_ms=None, snap_window_samples=None,
                              refractory_min_rr_ms=None, reference_rr_ms=None,
                              threshold_window_sec=DEFAULT_THRESHOLD_WINDOW_SEC):
    """
    Cross-correlates the full-run wavelet-filtered signal against the QRS
    template, keeps correlation peaks above a LOCAL threshold, resolves
    too-close candidates by keeping the better template match, removes
    locally implausible short intervals, aligns by `adjustment`, and snaps
    to the true local max on the raw signal. See the constants block above
    for why (audit finding C1).

    reference_rr_ms: median R-R interval (ms) of the RA-corrected template
    window; sets the refractory period. min_distance_ms/refractory_min_rr_ms
    override the derived values explicitly. threshold_window_sec=None uses
    one whole-run threshold (the original behavior).

    snap_window_samples defaults to DEFAULT_SNAP_WINDOW_SEC converted via
    sampling_rate (not a fixed sample count) -- pass an explicit value
    only to override that.
    """
    if snap_window_samples is None:
        snap_window_samples = round(DEFAULT_SNAP_WINDOW_SEC * sampling_rate)
    derived_distance_ms, derived_refractory_ms = spacing_from_reference_rr(reference_rr_ms)
    if refractory_min_rr_ms is None:
        refractory_min_rr_ms = derived_refractory_ms
    if min_distance_ms is None:
        min_distance_ms = min(derived_distance_ms, refractory_min_rr_ms)

    template_y = qrs_template["y"].to_numpy() if hasattr(qrs_template, "columns") else np.asarray(qrs_template)

    correlation = signal.correlate(wavelet_filtered_full, template_y, mode="same")
    min_distance_samples = max(1, int((min_distance_ms / 1000) * sampling_rate))
    candidates, _ = signal.find_peaks(correlation, distance=min_distance_samples)
    # Ignore positions where the template only partly overlaps the recording
    # (mode="same" zero-pads the ends): these produced spurious detections at
    # sample 0 and in the last fraction of a second.
    half = len(template_y) // 2
    candidates = candidates[(candidates >= half) & (candidates <= len(correlation) - (len(template_y) - half))]
    if threshold_window_sec:
        threshold = _local_threshold(correlation, max(1, int(threshold_window_sec * sampling_rate)))
        candidates = candidates[correlation[candidates] > threshold[candidates]]
    else:
        candidates = candidates[correlation[candidates] > np.mean(correlation) + np.std(correlation)]

    refractory_samples = int((refractory_min_rr_ms / 1000) * sampling_rate)
    candidates = _keep_stronger_peaks(candidates, correlation[candidates], refractory_samples)
    candidates = _remove_locally_short_intervals(candidates, lambda i: correlation[i])

    aligned_peaks = candidates - adjustment
    aligned_peaks = aligned_peaks[(aligned_peaks >= 0) & (aligned_peaks < len(raw_ecg_full))]

    refined_peaks = snap_to_local_max(raw_ecg_full, aligned_peaks, window_samples=snap_window_samples)
    # Two candidates can snap onto the same or nearly the same raw-signal
    # peak; a final refractory pass removes such near-duplicates.
    return enforce_refractory_period(refined_peaks, sampling_rate, min_rr_ms=min_distance_ms)


def save_qrs_template_outputs(output_dir, file_stem, initials, final_peaks, qrs_template, ppg_guide_shown=None):
    """
    Saves outputs with the SAME JSON schema step1_qrs_template.ipynb uses
    (so downstream consumers can't tell whether a given file's *content*
    came from the notebook or from this tool), but with the RA's initials
    appended to the FILENAME (unlike step1, which doesn't) so more than one
    RA can independently build a template for the same subject without
    overwriting each other:
        <file_stem>_ecg_corrected_qrs_<initials>.json  -> {"ecg": {"corrected_peaks": [...], "initials": [...]}}
        <file_stem>_qrs_template_<initials>.csv        -> the averaged template beat
    """
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, f"{file_stem}_ecg_corrected_qrs_{initials}.json")
    csv_path = os.path.join(output_dir, f"{file_stem}_qrs_template_{initials}.csv")
    # Rebuilding a template used to overwrite the previous one with no copy
    # kept (audit finding M7): back both files up first, as Step 2 does.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in (json_path, csv_path):
        if os.path.exists(path):
            backup_dir = os.path.join(output_dir, "backups")
            os.makedirs(backup_dir, exist_ok=True)
            root, ext = os.path.splitext(os.path.basename(path))
            shutil.copy2(path, os.path.join(backup_dir, f"{root}_{stamp}{ext}"))
    annotations_path = os.path.join(output_dir, f"{file_stem}_annotations_{initials}.json")
    if os.path.exists(annotations_path):
        try:
            with open(annotations_path, encoding="utf-8-sig") as f:
                has_ecg = "ecg" in (json.load(f).get("channels") or {})
        except (OSError, UnicodeDecodeError, ValueError, AttributeError):
            has_ecg = False  # an unreadable review file is reported plainly by Step 2
        if has_ecg:
            print(f"NOTE: you already have a saved ECG review for {file_stem}. It is NOT changed by this "
                  f"new template -- your saved peaks are what Step 2 reopens.")
    payload = {"ecg": {"corrected_peaks": [int(p) for p in final_peaks], "initials": [initials]}}
    if ppg_guide_shown is not None:
        # Whether the PPG guide row was on screen while the template window
        # was corrected (ppg-plan §4c, reverse circularity).
        payload["ecg"]["ppg_guide_shown"] = bool(ppg_guide_shown)
    with open(json_path, "w") as f:
        json.dump(payload, f)

    qrs_template.to_csv(csv_path, index=False)

    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")
    return json_path, csv_path


def build_ecg_raw(ecg_full, ecg_peaks_full, sfreq, ppg_full=None):
    """
    Builds a single-channel ECG mne.RawArray, pre-seeded with point
    annotations at ecg_peaks_full. With ppg_full, a second, read-only "ppg"
    row (MNE type "bio") is added as the PPG guide (sidecar plan Phase 1b):
    display only -- the template is still built from the ECG alone, and
    nothing reads that row back.

    Deliberately NOT channel-scoped (no ch_names) -- see the matching note
    in annotation_io.py's seed_annotations() for why: it's redundant
    (there's only one channel here anyway) and it's the confirmed trigger
    for a real mne-qt-browser crash when editing a window with many
    closely-spaced seeded peaks (its interactive region-merge logic has a
    bug specifically for "single-channel" annotations). Don't reintroduce it.
    """
    if ppg_full is None:
        info = mne.create_info(ch_names=["ecg"], sfreq=sfreq, ch_types=["ecg"])
        raw = mne.io.RawArray(ecg_full[np.newaxis, :], info, verbose=False)
    else:
        info = mne.create_info(ch_names=["ecg", "ppg"], sfreq=sfreq, ch_types=["ecg", "bio"])
        raw = mne.io.RawArray(np.vstack([np.asarray(ecg_full, dtype=float), np.asarray(ppg_full, dtype=float)]),
                              info, verbose=False)

    peak_indices = np.where(ecg_peaks_full)[0]
    if len(peak_indices):
        onsets = peak_indices / sfreq
        annotations = mne.Annotations(
            onset=onsets, duration=[0.0] * len(onsets), description=[POINT_LABEL] * len(onsets),
        )
        raw.set_annotations(annotations)
    return raw


def run_stage_a(run_dfs, run_file_stems, run_out_dirs, sfreq, initials,
                 template_run="1", template_start=0.0, template_window=20.0, ppg_guide=True, sidecars=None,
                 ecg_ppg_guide=False):
    """
    The full interactive Stage A session: RA reviews/corrects a short window
    of ONE run's ECG, then that template is cross-correlated against EVERY
    run in run_dfs, saving each run's own outputs. Shared by
    qrs_template_stage.py (standalone) and physio_review.py (unified CLI) so
    neither duplicates this orchestration.

    run_dfs / run_file_stems / run_out_dirs: {run: value} dicts, e.g.
    {"1": df1, "2": df2}. template_run must be a key in run_dfs (callers
    should already have resolved a fallback if the requested run is
    unavailable).

    ppg_guide / ecg_ppg_guide: the PPG guide row under the ECG shows only
    with both (opt-in since 2026-10-03, HLU; channel_config.GUIDE_OPT_IN).

    Returns {run: {"final_peaks": array, "json_path": str, "csv_path": str}}.
    """
    template_df = run_dfs[template_run]
    ecg_full_template_run = template_df["ecg"].to_numpy()
    ecg_peaks_template_run = (
        template_df["ecg_peaks"].to_numpy() if "ecg_peaks" in template_df.columns
        else np.zeros(len(template_df), dtype=bool)
    )

    start_idx = int(round(template_start * sfreq))
    end_idx = start_idx + int(round(template_window * sfreq))
    if end_idx > len(ecg_full_template_run):
        raise SystemExit(f"--template-start {template_start}s + --template-window {template_window}s exceeds "
                          f"run {template_run}'s length ({len(ecg_full_template_run) / sfreq:.1f}s). "
                          f"Pick an earlier start.")

    print(f"Computing wavelet-filtered reference signal for run {template_run} "
          f"({len(ecg_full_template_run) / sfreq:.0f}s)... this takes a moment.")
    wavelet_full_template_run = apply_wavelet_filter(ecg_full_template_run, sampling_rate=sfreq)

    # The PPG guide row (sidecar plan Phase 1b): shown unaltered under the ECG
    # when the run has a usable ppg column; the template never reads it.
    # sidecars: {run: sidecar dict} for real files (None/missing = synthetic).
    # No guide on old-pipeline files (HLU's decision G2, 2026-09-26).
    template_sidecar = (sidecars or {}).get(template_run)
    old_pipeline = not new_pipeline_ppg(template_sidecar)
    ppg_full = None
    usable_ppg = "ppg" in template_df.columns and not template_df["ppg"].isna().all() and not old_pipeline
    opted_out = ppg_guide and not ecg_ppg_guide
    ppg_guide = ppg_guide and ecg_ppg_guide
    if ppg_guide and usable_ppg:
        ppg_full = template_df["ppg"].to_numpy(dtype=float)
    raw = build_ecg_raw(ecg_full_template_run, ecg_peaks_template_run, sfreq, ppg_full=ppg_full)
    raw.crop(tmin=template_start, tmax=template_start + template_window - (1 / sfreq))

    scale = (np.percentile(ecg_full_template_run, 97.5) - np.percentile(ecg_full_template_run, 2.5)) / 2
    scalings = {"ecg": scale}
    if ppg_full is not None:
        window = ppg_full[start_idx:end_idx]
        ppg_scale = (np.nanpercentile(window, 97.5) - np.nanpercentile(window, 2.5)) / 2 if not np.all(
            np.isnan(window)) else 1.0
        scalings["bio"] = ppg_scale if np.isfinite(ppg_scale) and ppg_scale > 0 else 1.0

    print("\n" + "=" * 70)
    print(f"Reviewing run {template_run}'s ECG window [{template_start:.0f}s, {template_start + template_window:.0f}s] "
          f"to build the QRS template.")
    print("Correct peaks here (add missed ones, delete false ones). This template will be applied to "
          f"EVERY available run ({list(run_dfs.keys())}), not just this one.")
    print("Close the window when you're done -- you'll then see a preview of the resulting template "
          "to approve before it's used.")
    if ppg_full is not None:
        print(GUIDE_TEXT)
    elif opted_out and usable_ppg:
        print(GUIDE_OPT_IN[("ppg", "ecg")]["off"])
    elif ppg_guide and old_pipeline and "ppg" in template_df.columns:
        print("PPG (guide): not shown -- this file was processed before the PPG timing fix, so its PPG isn't "
              "reliable enough to guide you.")
    elif ppg_guide:
        print("PPG (guide): not shown -- this run has no usable PPG.")
    print("=" * 70 + "\n")

    raw.plot(block=True, scalings=scalings, duration=template_window, show=True, order=np.arange(len(raw.ch_names)),
              title=f"QRS Template Window (run {template_run}) [{template_start:.0f}s-{template_start + template_window:.0f}s]")

    # raw.annotations store onset in absolute time (relative to the FULL
    # run, i.e. before crop()), not relative to this cropped window's own
    # start -- crop() shifts raw.first_time to template_start but leaves
    # onsets alone. Must subtract raw.first_time to land on the correct
    # sample index within ecg_window/wavelet_window below. Omitting this
    # worked by coincidence whenever template_start was 0.0 (first_time
    # stays 0), which is why it went unnoticed until --template-start was
    # used with a nonzero value.
    ecg_window = ecg_full_template_run[start_idx:end_idx]
    wavelet_window = wavelet_full_template_run[start_idx:end_idx]

    # A peak the RA added by click-drag is resolved within the drawn region,
    # not at the drag's left edge (audit finding C3; see
    # annotation_io.point_sample_from_annotation()).
    # Drawn peaks follow the same rule as Steps 2 and 3 (the box maximum,
    # climbing only if the box cut off the top: Option C, 2026-09-26); peaks
    # the RA left untouched are snapped as before (below).
    drawn_window, seeded_window = [], []
    for ann in raw.annotations:
        if ann["description"] != POINT_LABEL:
            continue
        duration = int(round(ann["duration"] * sfreq))
        point = point_sample_from_annotation(int(round((ann["onset"] - raw.first_time) * sfreq)), duration,
                                             ecg_window, climb_samples=round(DRAWN_PEAK_CLIMB_SEC * sfreq))
        (drawn_window if duration > 0 else seeded_window).append(point)
    window_click_indices = drawn_window + seeded_window
    if not window_click_indices:
        raise SystemExit("No peaks were marked in this window -- nothing to build a template from. "
                          "Start again and correct at least a couple of peaks, or choose a different "
                          "template window start.")

    seeded_snapped = (snap_to_local_max(ecg_window, seeded_window, window_samples=round(DEFAULT_SNAP_WINDOW_SEC * sfreq))
                      if seeded_window else [])
    window_peaks = np.array(sorted({int(p) for p in drawn_window} | {int(p) for p in seeded_snapped}))
    # Median R-R of the RA-corrected window sets the detection refractory
    # period (audit finding C1).
    reference_rr_ms = float(np.median(np.diff(window_peaks)) * 1000 / sfreq) if len(window_peaks) >= 3 else None

    print(f"Building QRS template from {len(window_peaks)} corrected peak(s) in run {template_run}'s window...")
    qrs_template, adjustment = build_qrs_template(ecg_window, wavelet_window, window_peaks, sampling_rate=sfreq)

    # Show the RA the actual resulting template (individual raw beats + the
    # averaged curve) before committing to it -- mirrors the beat-segment
    # visualization step1_qrs_template.ipynb already used, but as a real
    # approve/reject gate rather than an after-the-fact display.
    safe_peaks_for_preview = filter_safe_peaks(window_peaks, len(ecg_window), sampling_rate=sfreq)
    raw_beat_epochs = nk.ecg_segment(ecg_window, safe_peaks_for_preview, sampling_rate=sfreq, show=False)
    raw_beats = [raw_beat_epochs[key]["Signal"].to_numpy() for key in raw_beat_epochs.keys()]

    preview_result = show_template_preview(raw_beats, qrs_template["y"].to_numpy(), sfreq)
    # The real dialog returns (approved, comment); tests may stub a plain bool.
    if isinstance(preview_result, tuple):
        approved, comment = preview_result
    else:
        approved, comment = bool(preview_result), ""
    if not approved:
        next_start = template_start + template_window
        raise SystemExit(
            f"OK -- not building a template from this window. Start Step 1 again and enter "
            f"{next_start:.0f} in 'Template window start (s)' to try the next window "
            f"(command line: --template-start {next_start:.0f})."
        )

    results = {}
    for run, df in run_dfs.items():
        ecg_full = df["ecg"].to_numpy()
        wavelet_full = wavelet_full_template_run if run == template_run else apply_wavelet_filter(ecg_full, sampling_rate=sfreq)

        print(f"Applying the run {template_run} template to run {run} via cross-correlation...")
        final_peaks = extract_and_refine_peaks(ecg_full, wavelet_full, qrs_template, adjustment, sampling_rate=sfreq,
                                               reference_rr_ms=reference_rr_ms)
        print(f"Run {run}: found {len(final_peaks)} peaks across {len(ecg_full) / sfreq:.0f}s.")

        json_path, csv_path = save_qrs_template_outputs(
            run_out_dirs[run], run_file_stems[run], initials, final_peaks, qrs_template,
            ppg_guide_shown=ppg_full is not None,
        )
        results[run] = {"final_peaks": final_peaks, "json_path": json_path, "csv_path": csv_path,
                        "comment": comment}

    return results
