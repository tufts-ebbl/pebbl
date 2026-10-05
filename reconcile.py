"""
Stage 3: reconciling two RAs' (or an RA's vs. a gold-standard reviewer's)
independent annotation JSONs for the same subject/run.

Design, agreed with the user before building:
- Point channels (ecg/rsp/ppg/eda): two peaks within a tolerance derived
  from DEFAULT_SNAP_WINDOW_SEC (annotation_io.py) and this run's sfreq
  (~50ms by default, scaling with sampling rate) are "agreed"; anything
  left over is "only A" or "only B".
- Segment channels (bad_segments): two spans whose overlap is at least half
  their combined span are "agreed" (shown as their union); anything left
  over is "only A"/"only B". (Until 2026-09-23 ANY overlap counted -- see
  MIN_SEGMENT_OVERLAP_FRACTION.)
- The reconciler can add marks neither reviewer made with the
  "<label>_added" description; the saved file records who reconciled
  (reconciled_by) and the two reviewers' agreement statistics.
- The reviewer sees all three categories as distinctly-labeled (and
  therefore distinctly-colored, via mne-qt-browser's per-description
  auto-coloring) annotations in the SAME interactive viewer already used
  for Stage B -- no new UI. Their job: delete any disagreeing mark they
  don't believe, add anything neither RA caught, leave "agreed" marks
  alone. On close, whatever remains (regardless of which of the three
  categories it started in) becomes the final, plain peak/segment list --
  the three-way distinction is a review aid, not part of the output.
- An "agreed" point's final position is whichever of the two RAs' matched
  samples sits higher on the signal (since 2026-09-26; previously their
  midpoint, re-snapped). Only marks the reconciler draws are snapped to the
  local max, exactly as in Step 2; seeded marks keep their sample.
- Output: <file_stem>_annotations_reconciled.json -- same shape as a
  normal RA's file, no initials in the name (a consensus product, not one
  person's pass). The gold-standard-comparison use case needs no separate
  code path: it's the same two-way reconciliation with one "RA" being the
  trainer's own initials.
"""

import contextlib
import io
import json
import os
from datetime import datetime

import mne
import numpy as np


from annotation_io import (
    ANNOTATION_SCHEMA_VERSION,
    COORDINATE_SPACE,
    check_saved_provenance,
    guide_banner_lines,
    guide_shown_by_channel,
    ppg_invalid_spans,
    print_sidecar_banner,
    session_checks_before_viewer,
    DEFAULT_SNAP_WINDOW_SEC,
    STATUS_COMPLETE,
    check_interval_regularity,
    combine_with_reference_channels,
    finish_session,
    keep_mode_changed_entries,
    DRAWN_PEAK_CLIMB_SEC,
    point_sample_from_annotation,
    read_saved_json,
    review_until_decided,
    saved_channel_entries,
    split_saved_by_mode,
    unique_onsets,
    companion_provenance,
    drop_peaks_in_bad_stretches,
    unknown_label_lines,
)
from channel_config import COMPANION_CHANNELS, is_present, with_companions
from channel_config import GUIDE_CHANNELS, REFERENCE_CHANNELS
from physio_io import build_raw, compute_scalings
from session_summary_gui import edit_record
from ppg_checks import ppg_check_lines, same_beat_lines
from provenance import channel_provenance

# Fallback tolerance for match_points() when called without an sfreq-derived
# value (e.g. directly, or in a test). Callers that have sfreq available
# (build_reconciliation_raw/export_reconciled/run_stage_c) instead derive
# their tolerance from DEFAULT_SNAP_WINDOW_SEC so it scales with sampling
# rate -- see annotation_io.DEFAULT_SNAP_WINDOW_SEC for why.
TOLERANCE_SAMPLES = 50


def match_points(indices_a, indices_b, tolerance_samples=TOLERANCE_SAMPLES):
    """
    Globally-greedy nearest-neighbor matching (closest pairs matched first,
    each index used at most once) between two RAs' point indices for the
    same channel, within tolerance_samples. Order-independent, unlike a
    naive "match A's peaks in order" approach, which could leave a peak
    unmatched purely because of iteration order when several candidates
    are within tolerance of each other.

    Returns (agreed, only_a, only_b):
        agreed  -- sorted list of (idx_a, idx_b) matched pairs
        only_a  -- sorted list of indices only in A
        only_b  -- sorted list of indices only in B
    """
    indices_a = sorted(indices_a)
    indices_b = sorted(indices_b)

    candidates = []
    for ia, a in enumerate(indices_a):
        for ib, b in enumerate(indices_b):
            distance = abs(a - b)
            if distance <= tolerance_samples:
                candidates.append((distance, ia, ib))
    candidates.sort(key=lambda c: c[0])

    matched_a, matched_b = set(), set()
    agreed = []
    for _distance, ia, ib in candidates:
        if ia in matched_a or ib in matched_b:
            continue
        matched_a.add(ia)
        matched_b.add(ib)
        agreed.append((indices_a[ia], indices_b[ib]))

    only_a = [indices_a[ia] for ia in range(len(indices_a)) if ia not in matched_a]
    only_b = [indices_b[ib] for ib in range(len(indices_b)) if ib not in matched_b]

    agreed.sort()
    return agreed, sorted(only_a), sorted(only_b)


MIN_SEGMENT_OVERLAP_FRACTION = 0.5
"""
Two bad segments count as "agreed" only if their overlap is at least half
of their combined span (intersection over union >= 0.5). Previously ANY
overlap counted -- a 1-sample overlap between a 1-s and a 60-s segment was
"agreement", shown as their union, which inflated agreement and pulled the
consensus toward the more liberal reviewer (audit finding M2). Pairs that
overlap less than this are shown separately as "only A" / "only B" marks.
"""


def match_segments(segments_a, segments_b, min_overlap_fraction=MIN_SEGMENT_OVERLAP_FRACTION):
    """
    Globally-greedy matching between two RAs' bad-segment spans for the
    same channel: two segments are "agreed" if their overlap is at least
    min_overlap_fraction of their combined span, matched by
    largest-overlap-first so an ambiguous case doesn't get resolved purely
    by iteration order. min_overlap_fraction=0 restores "any overlap".

    Returns (agreed, only_a, only_b):
        agreed  -- list of (seg_a, seg_b) matched pairs, each a [start, end]
        only_a  -- segments only in A
        only_b  -- segments only in B
    """
    def overlap_amount(seg_a, seg_b):
        return min(seg_a[1], seg_b[1]) - max(seg_a[0], seg_b[0])

    candidates = []
    for ia, seg_a in enumerate(segments_a):
        for ib, seg_b in enumerate(segments_b):
            overlap = overlap_amount(seg_a, seg_b)
            union = max(seg_a[1], seg_b[1]) - min(seg_a[0], seg_b[0])
            if overlap > 0 and union > 0 and overlap / union >= min_overlap_fraction:
                candidates.append((-overlap, ia, ib))  # largest overlap first
    candidates.sort(key=lambda c: c[0])

    matched_a, matched_b = set(), set()
    agreed = []
    for _neg_overlap, ia, ib in candidates:
        if ia in matched_a or ib in matched_b:
            continue
        matched_a.add(ia)
        matched_b.add(ib)
        agreed.append((segments_a[ia], segments_b[ib]))

    only_a = [segments_a[ia] for ia in range(len(segments_a)) if ia not in matched_a]
    only_b = [segments_b[ib] for ib in range(len(segments_b)) if ib not in matched_b]

    return agreed, only_a, only_b


def _agree_label(base_label):
    return f"{base_label}_agree"


def _only_label(base_label, who):
    return f"{base_label}_only_{who}"


def _added_label(base_label):
    """Label for a mark the reconciler adds that neither reviewer made."""
    return f"{base_label}_added"


def _agreed_point(idx_a, idx_b, signal):
    """
    Where an agreed pair is seeded: whichever reviewer's own sample sits
    higher on the signal (A's on a tie, or when there's no signal). Until
    2026-09-26 the pair was seeded at its midpoint and EVERY point was then
    re-snapped to the local maximum at export; a machine peak that wasn't a
    local maximum moved to the edge of the snap window, onto neither
    reviewer's point nor a real peak (ppg-plan §3.1 #2). Seeding at a real
    reviewer sample lets export keep untouched marks exactly, as Step 2 does.
    """
    if idx_a == idx_b or signal is None:
        return idx_a
    value_a = signal[idx_a] if 0 <= idx_a < len(signal) else np.nan
    value_b = signal[idx_b] if 0 <= idx_b < len(signal) else np.nan
    if np.isnan(value_a) and np.isnan(value_b):
        return idx_a
    if np.isnan(value_a):
        return idx_b
    return idx_b if (not np.isnan(value_b) and value_b > value_a) else idx_a


def _segments_mask(segments, n_samples):
    mask = np.zeros(n_samples, dtype=bool)
    for start, end in segments:
        mask[max(0, int(start)):min(n_samples, int(end))] = True
    return mask


def agreement_statistics(outputs_a, outputs_b, channel_configs, sfreq, n_samples, tolerance_samples):
    """
    Inter-rater agreement between two reviewers' saved outputs, per shared
    channel, for reporting reliability (audit finding M2):
      point channels   -- agreed / only-A / only-B counts, percent agreement
                          (agreed / all distinct marks), and the median
                          absolute timing difference of agreed pairs (ms);
      segment channels -- agreed / only-A / only-B segment counts, seconds
                          marked by each reviewer, and Cohen's kappa over
                          samples (bad vs. not bad; None when undefined,
                          e.g. neither reviewer marked anything).
    """
    stats = {}
    for ch_key, cfg in channel_configs.items():
        out_a, out_b = outputs_a.get(ch_key), outputs_b.get(ch_key)
        if not out_a or not out_b:
            continue
        if cfg["annotation_mode"] == "point":
            agreed, only_a, only_b = match_points(out_a["indices"], out_b["indices"], tolerance_samples)
            total = len(agreed) + len(only_a) + len(only_b)
            timing = [abs(a - b) * 1000 / sfreq for a, b in agreed]
            stats[ch_key] = {
                "agreed": len(agreed), "only_a": len(only_a), "only_b": len(only_b),
                "percent_agreement": round(100 * len(agreed) / total, 1) if total else None,
                "median_timing_difference_ms": round(float(np.median(timing)), 1) if timing else None,
            }
        elif cfg["annotation_mode"] == "segment":
            agreed, only_a, only_b = match_segments(out_a["bad_segments"], out_b["bad_segments"])
            mask_a = _segments_mask(out_a["bad_segments"], n_samples)
            mask_b = _segments_mask(out_b["bad_segments"], n_samples)
            observed = float(np.mean(mask_a == mask_b))
            pa, pb = float(np.mean(mask_a)), float(np.mean(mask_b))
            expected = pa * pb + (1 - pa) * (1 - pb)
            kappa = round((observed - expected) / (1 - expected), 3) if expected < 1 else None
            stats[ch_key] = {
                "agreed": len(agreed), "only_a": len(only_a), "only_b": len(only_b),
                "seconds_marked_a": round(float(mask_a.sum()) / sfreq, 2),
                "seconds_marked_b": round(float(mask_b.sum()) / sfreq, 2),
                "cohen_kappa_samples": kappa,
            }
    return stats


def format_agreement(stats, label_a, label_b):
    """One compact line, e.g. 'Agreement -- ecg: 97.3% (median 4.0 ms); eda: kappa 0.81'."""
    parts = []
    for ch_key, s in stats.items():
        if "percent_agreement" in s:
            timing = f", median {s['median_timing_difference_ms']} ms" if s["median_timing_difference_ms"] is not None else ""
            parts.append(f"{ch_key}: {s['percent_agreement']}% of peaks agreed ({s['only_a']} only {label_a}, "
                         f"{s['only_b']} only {label_b}{timing})")
        else:
            parts.append(f"{ch_key}: kappa {s['cohen_kappa_samples']} ({s['agreed']} segments agreed, "
                         f"{s['only_a']} only {label_a}, {s['only_b']} only {label_b})")
    return ("Agreement -- " + "; ".join(parts)) if parts else ""


def build_reconciliation_raw(df, channel_configs, sfreq, outputs_a, outputs_b, label_a, label_b,
                              tolerance_samples=None, ppg_guide=False, sidecar=None, ecg_ppg_guide=False):
    """
    Builds a Raw seeded with three-way-labeled annotations per channel
    (agree / only_<label_a> / only_<label_b>) from two RAs' exported
    channel_outputs (the same shape export_annotations() produces). Only
    channels present in BOTH outputs are reconciled -- a channel only one
    RA reviewed has nothing to compare, so it's silently skipped here (the
    caller decides whether to warn about that).

    tolerance_samples=None (the default) derives the matching tolerance from
    DEFAULT_SNAP_WINDOW_SEC and sfreq, so two RAs' clicks count as "agreed"
    within the same real-world time window regardless of sampling rate.

    Returns (raw, diff_summary) -- diff_summary is a
    {channel: {"agreed": n, "only_<label_a>": n, "only_<label_b>": n}} dict
    for printing a session-start overview to the reviewer.
    """
    if tolerance_samples is None:
        tolerance_samples = round(DEFAULT_SNAP_WINDOW_SEC * sfreq)

    full_channel_configs = combine_with_reference_channels(channel_configs, df, ppg_guide=ppg_guide, sidecar=sidecar,
                                                           ecg_ppg_guide=ecg_ppg_guide)
    raw = build_raw(df, full_channel_configs, sfreq)

    onsets, durations, descriptions = [], [], []
    diff_summary = {}

    for ch_key, cfg in channel_configs.items():
        out_a = outputs_a.get(ch_key)
        out_b = outputs_b.get(ch_key)
        if not out_a or not out_b:
            continue
        if cfg["annotation_mode"] in ("point", "segment") and not (
                out_a.get("mode") == out_b.get("mode") == cfg["annotation_mode"]):
            # e.g. an RSP review from when RSP was a point channel (sidecar
            # plan Phase 1); physio_review.py explains this to the RA first.
            raise ValueError(f"Can't reconcile {ch_key}: the reviews' modes ({out_a.get('mode')}, "
                             f"{out_b.get('mode')}) don't match its current mode ({cfg['annotation_mode']}).")

        if cfg["annotation_mode"] == "point":
            agreed, only_a, only_b = match_points(out_a["indices"], out_b["indices"], tolerance_samples)
            signal = df[ch_key].to_numpy(dtype=float) if ch_key in df.columns else None
            for idx_a, idx_b in agreed:
                onsets.append(_agreed_point(idx_a, idx_b, signal) / sfreq)
                durations.append(0.0)
                descriptions.append(_agree_label(cfg["point_label"]))
            for idx_a in only_a:
                onsets.append(idx_a / sfreq)
                durations.append(0.0)
                descriptions.append(_only_label(cfg["point_label"], label_a))
            for idx_b in only_b:
                onsets.append(idx_b / sfreq)
                durations.append(0.0)
                descriptions.append(_only_label(cfg["point_label"], label_b))
            diff_summary[ch_key] = {
                "agreed": len(agreed), f"only_{label_a}": len(only_a), f"only_{label_b}": len(only_b),
            }

        elif cfg["annotation_mode"] == "segment":
            agreed, only_a, only_b = match_segments(out_a["bad_segments"], out_b["bad_segments"])
            for seg_a, seg_b in agreed:
                start = min(seg_a[0], seg_b[0])
                end = max(seg_a[1], seg_b[1])
                onsets.append(start / sfreq)
                durations.append(max(end - start, 1) / sfreq)
                descriptions.append(_agree_label(cfg["segment_label"]))
            for seg in only_a:
                onsets.append(seg[0] / sfreq)
                durations.append(max(seg[1] - seg[0], 1) / sfreq)
                descriptions.append(_only_label(cfg["segment_label"], label_a))
            for seg in only_b:
                onsets.append(seg[0] / sfreq)
                durations.append(max(seg[1] - seg[0], 1) / sfreq)
                descriptions.append(_only_label(cfg["segment_label"], label_b))
            diff_summary[ch_key] = {
                "agreed": len(agreed), f"only_{label_a}": len(only_a), f"only_{label_b}": len(only_b),
            }

    # Register an "<label>_added" description for every reconciled channel,
    # via a zero-duration placeholder, so the reconciler can always label a
    # mark neither reviewer made. mne-qt-browser's description picker lists
    # only descriptions already present; without this, a channel where
    # neither reviewer marked a bad segment offered NO segment label and a
    # newly drawn segment was silently dropped at export (audit finding H4).
    # export_reconciled() ignores the zero-duration placeholder itself.
    for ch_key in diff_summary:
        cfg = channel_configs[ch_key]
        base = cfg["point_label"] if cfg["annotation_mode"] == "point" else cfg["segment_label"]
        onsets.append(0.0)
        durations.append(0.0)
        descriptions.append(_added_label(base))

    if onsets:
        # Distinct start times: see annotation_io.unique_onsets() (ppg-plan §3.1 #4).
        raw.set_annotations(mne.Annotations(onset=unique_onsets(onsets, sfreq), duration=durations,
                                            description=descriptions))

    return raw, diff_summary


def reviewer_origins(raw, channel_configs, sfreq, label_a, label_b):
    """
    {ch_key: {"agreed": [...], "only_<a>": [...], "only_<b>": [...]}}: the
    seeded marks' samples (point channels) or spans (segment channels) by
    category, as build_reconciliation_raw() seeded them. Stored with the
    reconciled channel ("reviewer_origin"), so each final mark's origin can be
    looked up; a final mark in none of the lists was added by the reconciler.
    """
    origins = {}
    for ch_key, cfg in channel_configs.items():
        base = cfg.get("point_label") or cfg.get("segment_label")
        if not base:
            continue
        cats = {_agree_label(base): "agreed", _only_label(base, label_a): f"only_{label_a}",
                _only_label(base, label_b): f"only_{label_b}"}
        entry = {name: [] for name in cats.values()}
        for ann in raw.annotations:
            name = cats.get(ann["description"])
            if name is None:
                continue
            onset = int(round(ann["onset"] * sfreq))
            if cfg["annotation_mode"] == "point":
                entry[name].append(onset)
            else:
                entry[name].append([onset, onset + max(int(round(ann["duration"] * sfreq)), 1)])
        if any(entry.values()):
            origins[ch_key] = {k: sorted(v) for k, v in entry.items()}
    return origins


def export_reconciled(raw, df, channel_configs, sfreq, label_a, label_b, snap_window_samples=None):
    """
    Reads back the reviewer's final annotation state (after they've deleted
    disagreeing marks they rejected and added anything missed) and produces
    a plain per-channel output -- same shape export_annotations() produces
    -- collapsing the three review-time categories (agree/only_a/only_b)
    into one unified list, since that distinction only ever mattered for
    display. Points the reconciler drew are snapped to the true local max
    exactly like a normal Stage B session; seeded marks keep their sample.

    snap_window_samples=None (the default) derives the search window from
    DEFAULT_SNAP_WINDOW_SEC and sfreq, matching export_annotations()'s own
    sampling-rate-independent default.
    """
    if snap_window_samples is None:
        snap_window_samples = round(DEFAULT_SNAP_WINDOW_SEC * sfreq)

    channel_outputs = {}
    drawn_points = {}  # ch_key -> points the reconciler placed by click-drag
    for ch_key, cfg in channel_configs.items():
        if not is_present(ch_key, cfg, raw.ch_names):
            continue
        if cfg["annotation_mode"] == "point":
            channel_outputs[ch_key] = {"mode": "point", "indices": []}
        elif cfg["annotation_mode"] == "segment":
            channel_outputs[ch_key] = {"mode": "segment", "bad_segments": []}

    for ann in raw.annotations:
        onset_sample = int(round(ann["onset"] * sfreq))
        duration_samples = int(round(ann["duration"] * sfreq))

        for ch_key, cfg in channel_configs.items():
            if ch_key not in channel_outputs:
                continue

            is_added = ann["description"] == _added_label(cfg.get("point_label") or cfg.get("segment_label"))
            if is_added and duration_samples <= 0:
                continue  # the zero-duration placeholder that registers the "_added" label
            if cfg["annotation_mode"] == "point" and (is_added or ann["description"] in (
                _agree_label(cfg["point_label"]),
                _only_label(cfg["point_label"], label_a),
                _only_label(cfg["point_label"], label_b),
            )):
                # A peak the reviewer ADDED by click-drag is resolved within
                # the drawn region, not at its left edge (audit finding C3).
                signal = df[ch_key].to_numpy() if ch_key in df.columns else None
                point_sample = point_sample_from_annotation(onset_sample, duration_samples, signal,
                                                            climb_samples=round(DRAWN_PEAK_CLIMB_SEC * sfreq))
                channel_outputs[ch_key]["indices"].append(point_sample)
                if duration_samples > 0:
                    drawn_points.setdefault(ch_key, set()).add(point_sample)

            elif cfg["annotation_mode"] == "segment" and (is_added or ann["description"] in (
                _agree_label(cfg["segment_label"]),
                _only_label(cfg["segment_label"], label_a),
                _only_label(cfg["segment_label"], label_b),
            )):
                channel_outputs[ch_key]["bad_segments"].append(
                    [onset_sample, onset_sample + max(duration_samples, 1)]
                )

    # Points the reconciler drew were resolved above by the same rule as
    # Step 2 (the box maximum, climbing only if the box cut off the top:
    # Option C, 2026-09-26; no second re-snap). Seeded marks -- agreed pairs
    # (seeded at a real reviewer sample, see _agreed_point()) and one-reviewer
    # marks -- keep their exact sample (ppg-plan §3.1 #2).
    for ch_key, cfg in channel_configs.items():
        if ch_key not in channel_outputs or cfg["annotation_mode"] != "point":
            continue
        channel_outputs[ch_key]["indices"] = sorted(set(int(i) for i in channel_outputs[ch_key]["indices"]))
        channel_outputs[ch_key]["snap_window_samples"] = snap_window_samples

    for ch_key, cfg in channel_configs.items():
        if ch_key in channel_outputs and cfg["annotation_mode"] == "segment":
            channel_outputs[ch_key]["bad_segments"] = sorted(channel_outputs[ch_key]["bad_segments"])

    # Bad stretches never keep peaks, as in Step 2 (HLU, 2026-09-27).
    drop_peaks_in_bad_stretches(channel_outputs)
    return channel_outputs


def save_reconciled_json(output_dir, file_stem, channel_outputs, sfreq, source_file, label_a, label_b,
                         reconciler=None, agreement=None):
    """
    Writes <file_stem>_annotations_reconciled.json -- deliberately no
    initials in the name (a consensus product of two reviewers, not one
    person's individual pass). Records who reconciled (`reconciled_by`) and
    the two reviewers' agreement statistics (see agreement_statistics()).
    An existing reconciliation is copied to backups/ first, the same safety
    net save_annotation_json() has (audit finding M2).

    MERGES with an existing reconciled file, like save_annotation_json():
    channels reconciled in THIS session replace their own entries, and every
    other channel (and its agreement statistics) is carried forward. Until
    2026-09-26 this overwrote the file with this session's channels only, so
    reconciling ECG and then any other channel in a separate session (the
    GUI reconciles one channel per session) left the ECG reconciliation
    only in backups/ (ppg-plan §3.1 #1). Each channel entry records who
    reconciled it and from whom, since merged channels can differ.
    """
    import shutil
    from datetime import datetime

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{file_stem}_annotations_reconciled.json")
    merged_channels = {}
    merged_agreement = {}
    for ch_key, output in channel_outputs.items():
        output["reconciled_by"] = reconciler
        output["reconciled_from"] = [label_a, label_b]
    if os.path.exists(out_path):
        backup_dir = os.path.join(output_dir, "backups")
        os.makedirs(backup_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"{file_stem}_annotations_reconciled_{stamp}.json")
        shutil.copy2(out_path, backup_path)
        print(f"NOTE: previous reconciliation backed up to {backup_path}.")
        try:
            existing = read_saved_json(out_path, what="reconciled file")
            existing_channels = saved_channel_entries(existing, out_path, what="reconciled file")
        except SystemExit as problem:
            rescue = os.path.join(output_dir, f"{file_stem}_annotations_reconciled_RESCUED_{stamp}.json")
            with open(rescue, "w") as f:
                json.dump({"schema_version": ANNOTATION_SCHEMA_VERSION, "coordinate_space": COORDINATE_SPACE,
                           "initials": reconciler, "reconciled_by": reconciler, "reconciled_from": [label_a, label_b],
                           "agreement": agreement or {}, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           "source_file": source_file, "sampling_rate": sfreq, "channels": channel_outputs},
                          f, indent=2)
            raise SystemExit(f"{problem} This session's reconciliation was saved separately to {rescue}; give "
                             f"both files to the lab staff.")
        merged_channels = dict(existing_channels)
        merged_agreement = dict(existing.get("agreement", {}))
        # A version-1 file (before 2026-09-26) recorded who reconciled it only
        # at the top level, and it was always written by one session: give its
        # carried-forward channels that attribution before the top level is
        # overwritten with this session's (checkpoint-1 review, SD-3).
        for ch_key, entry in merged_channels.items():
            if ch_key not in channel_outputs and isinstance(entry, dict) and "reconciled_by" not in entry:
                entry["reconciled_by"] = existing.get("reconciled_by", existing.get("initials"))
                entry["reconciled_from"] = existing.get("reconciled_from")
        # A channel whose review mode changed (e.g. an old point-mode RSP
        # consensus) is kept under a legacy key, as in Step 2 (SD-6).
        for ch_key, legacy_key, old_mode in keep_mode_changed_entries(existing_channels, channel_outputs,
                                                                      merged_channels):
            print(f"  The earlier {ch_key} reconciliation (marked as {old_mode or 'unknown'}s) is kept in the file "
                  f"as '{legacy_key}'.")
        carried_forward = [ch for ch in existing_channels if ch not in channel_outputs]
        if carried_forward:
            print(f"  Carrying forward previously reconciled channel(s) not reconciled this session "
                  f"(unchanged): {', '.join(carried_forward)}.")
    merged_channels.update(channel_outputs)
    for ch_key in channel_outputs:
        merged_agreement.pop(ch_key, None)
    merged_agreement.update(agreement or {})

    payload = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "coordinate_space": COORDINATE_SPACE,
        "initials": reconciler,
        "reconciled_by": reconciler,
        "reconciled_from": [label_a, label_b],
        "agreement": merged_agreement,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_file": source_file,
        "sampling_rate": sfreq,
        "channels": merged_channels,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved reconciled annotations to {out_path}")
    return out_path


def run_stage_c(df, sfreq, channel_configs, out_dir, file_stem, source_file,
                 outputs_a, outputs_b, label_a, label_b, tolerance_samples=None, reconciler=None,
                 sidecar=None, ppg_guide=True, tsv_path=None, allow_old_ppg=False, ecg_ppg_guide=False):
    """
    The full interactive Stage C session: builds the three-way-labeled
    Raw, launches the viewer, then shows the session-summary dialog (save /
    discard / go back) and acts on it. Returns a SessionResult.

    tolerance_samples=None (the default) derives the matching tolerance from
    DEFAULT_SNAP_WINDOW_SEC and sfreq -- see build_reconciliation_raw().
    sidecar / ppg_guide / tsv_path / allow_old_ppg / ecg_ppg_guide: as for
    annotation_io.run_stage_b(). With tsv_path, both reviews must have been
    made on the current file's signal (same sample count and content hash),
    and the reconciled file records it -- it is what physioProcess Phase D
    reads (ppg-plan §4c, §5.1).
    """
    if tolerance_samples is None:
        tolerance_samples = round(DEFAULT_SNAP_WINDOW_SEC * sfreq)

    session_checks_before_viewer(channel_configs, sidecar, allow_old_ppg, df=df)
    # Companions ride with their parent (bad_ppg with PPG; HLU, 2026-09-27). A
    # review saved before bad_ppg existed counts as having no bad stretches.
    channel_configs = with_companions(channel_configs)
    for comp_key, comp_cfg in COMPANION_CHANNELS.items():
        if comp_key in channel_configs:
            parent = comp_cfg["companion_of"]
            outputs_a = {comp_key: {"mode": "segment", "bad_segments": [],
                                    "status": (outputs_a.get(parent) or {}).get("status")}, **outputs_a}
            outputs_b = {comp_key: {"mode": "segment", "bad_segments": [],
                                    "status": (outputs_b.get(parent) or {}).get("status")}, **outputs_b}
    reviewed_keys = [k for k, cfg in channel_configs.items() if cfg["annotation_mode"] in ("point", "segment")]
    current_provenance = companion_provenance(channel_provenance(tsv_path, sidecar, reviewed_keys, len(df)))
    check_saved_provenance(outputs_a, current_provenance, channel_configs, who=f"Reviewer {label_a}'s")
    check_saved_provenance(outputs_b, current_provenance, channel_configs, who=f"Reviewer {label_b}'s")
    # A PPG review with no provenance was made before 2026-09-26 on the
    # pre-fix PPG: it can't be reconciled on this file (safety fix).
    for who, out in ((label_a, outputs_a), (label_b, outputs_b)):
        if "ppg" in reviewed_keys and current_provenance.get("ppg") and "ppg" in out \
                and not (out["ppg"] or {}).get("provenance"):
            raise SystemExit(f"Reviewer {who}'s PPG review has no record of the signal it was made on (it was "
                             f"saved before 2026-09-26, when PPG was on the old timing), so it can't be reconciled "
                             f"on this file. {who} needs to redo PPG in Step 2 first. Nothing has been changed.")

    full_channel_configs = combine_with_reference_channels(channel_configs, df, ppg_guide=ppg_guide, sidecar=sidecar,
                                                           ecg_ppg_guide=ecg_ppg_guide)
    guide_shown = guide_shown_by_channel(channel_configs, full_channel_configs)
    raw, diff_summary = build_reconciliation_raw(
        df, channel_configs, sfreq, outputs_a, outputs_b, label_a, label_b, tolerance_samples, ppg_guide=ppg_guide,
        sidecar=sidecar, ecg_ppg_guide=ecg_ppg_guide,
    )
    scalings = compute_scalings(df, full_channel_configs)
    reference_channels_shown = [k for k in REFERENCE_CHANNELS if k in full_channel_configs]
    labels = ", ".join(cfg["label"] for cfg in channel_configs.values())
    agreement = agreement_statistics(outputs_a, outputs_b, channel_configs, sfreq, len(df), tolerance_samples)
    agreement_text = format_agreement(agreement, label_a, label_b)
    if agreement_text:
        print(agreement_text)

    print("\n" + "=" * 70)
    print(f"Reconciling {label_a} vs {label_b} -- channels: {list(diff_summary.keys())}")
    for ch_key, counts in diff_summary.items():
        print(f"  - {ch_key}: {counts}")
    print(f"Launching interactive viewer with channels: {list(raw.ch_names)}")
    print_sidecar_banner(sidecar, channel_configs)
    print(f" - Marks agreed by both reviewers need no action.")
    print(f" - Marks unique to only one reviewer are shown separately (auto-colored by")
    print(f"   MNE per label) -- delete any you don't believe; add anything neither caught.")
    for line in guide_banner_lines(channel_configs, df, ppg_guide, sidecar, ecg_ppg_guide):
        print(f" - {line}")
    for comp_key, comp_cfg in COMPANION_CHANNELS.items():
        if comp_key in channel_configs:
            print(f" - {comp_cfg['label']} (label '{comp_cfg['segment_label']}'): peaks inside the final bad stretches "
                  f"are left out on save.")
    if reference_channels_shown:
        print(f" - {', '.join(reference_channels_shown)}: read-only reference track(s), for context only.")
    print(" - Close the plot window when you're done. A summary will then let you save, discard this")
    print("   session's changes, or go back to the viewer.")
    print("=" * 70 + "\n")

    def show_viewer():
        # order=<declared order>, not MNE's type-based default -- see the matching
        # note in annotation_io.py's run_stage_b() for why (eda/ppg/finger_temperature
        # would otherwise render below the read-only "event" track, ecg/rsp/emg/
        # sbp/dbp above it, due to MNE's own internal channel-type priority list).
        raw.plot(block=True, scalings=scalings, duration=20, show=True,
                  order=np.arange(len(raw.ch_names)),
                  title=f"{file_stem} · {labels} · reconciling {label_a} vs {label_b}")

    existing_path = os.path.join(out_dir, f"{file_stem}_annotations_reconciled.json")
    previous_channels = {}
    if os.path.exists(existing_path):
        # An entry in an older mode (e.g. point-mode RSP) must not count as
        # this channel's saved reconciliation (checkpoint-1, S3-1). An
        # unreadable file stops plainly (2026-09-26 safety fix).
        previous_channels, _mode_changed = split_saved_by_mode(
            saved_channel_entries(read_saved_json(existing_path, what="reconciled file"), existing_path,
                           what="reconciled file"), channel_configs)
    previous_status = {k: previous_channels.get(k, {}).get("status") for k in channel_configs}
    labels = ", ".join(cfg["label"] for cfg in channel_configs.values())

    # Step 3 always starts again from the two reviewers' CURRENT files; an
    # earlier reconciliation of the same channel is not shown. Say so, since
    # saving replaces it (checkpoint-1, S3-2).
    for ch_key in channel_configs:
        earlier = previous_channels.get(ch_key)
        if earlier:
            print(f"NOTE: {channel_configs[ch_key]['label']} was already reconciled"
                  f"{' by ' + earlier['reconciled_by'] if earlier.get('reconciled_by') else ''}"
                  f" ({earlier.get('status', 'no status')}). This session starts again from the two reviews, "
                  f"so earlier reconciliation decisions are not shown; saving replaces that reconciliation (the "
                  f"old one is kept in backups/).")
    # Reconciling a review its reviewer hasn't finished gives a "complete"
    # consensus built on a partial review (checkpoint-1, S3-8).
    unfinished = [f"{channel_configs[ch]['label']}: {who}'s review is not marked finished ({out.get(ch, {}).get('status') or 'no status'})"
                  for ch in channel_configs for who, out in ((label_a, outputs_a), (label_b, outputs_b))
                  if out.get(ch, {}).get("status") != STATUS_COMPLETE]
    for line in unfinished:
        print(f"WARNING: {line}")

    ppg_invalid, ppg_note = ppg_invalid_spans(sidecar, len(df)) if "ppg" in reviewed_keys else ([], None)
    if ppg_note:
        print(f"NOTE: {ppg_note}")
    # What each mark started as, so the reconciled file (the one physioProcess
    # reads) can say which final marks both reviewers made, which only one
    # made, and which the reconciler added (checkpoint-2 review).
    with contextlib.redirect_stdout(io.StringIO()):
        session_baseline = export_reconciled(raw, df, channel_configs, sfreq, label_a, label_b)
    origins = reviewer_origins(raw, channel_configs, sfreq, label_a, label_b)

    def step3_checks(r, outputs):
        lines = unknown_label_lines(r, step3_labels) + unfinished
        if outputs.get("ppg", {}).get("mode") == "point" and "ppg" in df.columns:
            invalid = (outputs.get("bad_ppg") or {}).get("bad_segments", ppg_invalid) if "bad_ppg" in outputs \
                else ppg_invalid
            lines += ppg_check_lines(outputs["ppg"]["indices"], df["ppg"].to_numpy(), sfreq, invalid)
        return lines

    def save(outputs):
        saved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for ch_key in outputs:
            if ch_key in current_provenance:
                outputs[ch_key]["provenance"] = current_provenance[ch_key]
            # What "status" covers: the whole file (annotation_io.STATUS_VALUES).
            outputs[ch_key]["reviewed_span"] = [0, len(df)]
            record = edit_record(session_baseline.get(ch_key), outputs[ch_key], tolerance_samples)
            record.update(saved_at=saved_at, initials=reconciler,
                          started_from=f"the two reviews ({label_a}, {label_b})")
            if ch_key in origins:
                outputs[ch_key]["reviewer_origin"] = origins[ch_key]
            if ch_key in guide_shown:
                # Whether a guide row was on screen for the reconciler and for
                # each reviewer; guide_shown is true if it was for anyone
                # (§4c, reverse circularity; checkpoint-2 review). The saved
                # key stays "adjudicator" (the role's old name) so files saved
                # before 2026-10-04 and Phase D's reader keep matching.
                by = {"adjudicator": guide_shown[ch_key],
                      label_a: outputs_a.get(ch_key, {}).get("guide_shown"),
                      label_b: outputs_b.get(ch_key, {}).get("guide_shown")}
                record["guide_shown"] = guide_shown[ch_key]
                outputs[ch_key]["guide_shown_by"] = by
                outputs[ch_key]["guide_shown"] = any(bool(v) for v in by.values())
            outputs[ch_key]["edit_history"] = [record]
        return save_reconciled_json(out_dir, file_stem, outputs, sfreq, source_file, label_a, label_b,
                                    reconciler=reconciler, agreement=agreement)

    step3_labels = set()
    for cfg in channel_configs.values():
        base = cfg.get("point_label") or cfg.get("segment_label")
        if base:
            step3_labels |= {_agree_label(base), _only_label(base, label_a), _only_label(base, label_b),
                             _added_label(base)}

    decision, channel_outputs, n_changes = review_until_decided(
        raw,
        show_viewer,
        lambda r, _warnings_out: export_reconciled(r, df, channel_configs, sfreq, label_a, label_b),
        lambda outputs: check_interval_regularity(outputs, channel_configs, sfreq),
        channel_configs,
        sfreq,
        heading=f"{file_stem} · {labels} · {reconciler or 'reconciler'} reconciling {label_a} vs {label_b}",
        finished_default=bool(previous_channels) and all(s == STATUS_COMPLETE for s in previous_status.values()),
        # Marks with a label Step 3 doesn't export (e.g. Step 2's "peak_ecg")
        # are reported, as in Step 2 (checkpoint-1, S3-5).
        extra_warnings=step3_checks,
        # Two final PPG marks on one pulse must be fixed before saving: both
        # would otherwise be exported as separate beats (ppg-plan §4b).
        blocking_warnings=lambda _r, outputs: (same_beat_lines(outputs["ppg"]["indices"], sfreq)
                                               if outputs.get("ppg", {}).get("mode") == "point" else []),
    )
    # "Nothing to save" only if the saved reconciliation already holds exactly
    # this result from the same reviewers and reconciler: the viewer's
    # baseline is re-seeded from the reviewers' current files, so "no changes
    # on screen" alone doesn't mean that (checkpoint-1, SD-1).
    same_as_saved = all(
        ch in previous_channels
        and previous_channels[ch].get("indices") == channel_outputs[ch].get("indices")
        and previous_channels[ch].get("bad_segments") == channel_outputs[ch].get("bad_segments")
        and sorted(previous_channels[ch].get("reconciled_from") or []) == sorted([label_a, label_b])
        and previous_channels[ch].get("reconciled_by") == reconciler
        # ...and made on this file's signal, with this file's span: after a
        # reprocess, identical marks must still be re-saved with the current
        # provenance and reviewed_span, or Phase D refuses the channel while
        # the reconciler is told it's saved (run-8 review, 2026-09-26).
        and previous_channels[ch].get("provenance") == current_provenance.get(ch)
        and previous_channels[ch].get("reviewed_span") == [0, len(df)]
        for ch in channel_outputs
    )
    result = finish_session(decision, channel_outputs, n_changes, previous_status, os.path.exists(existing_path),
                            save, existing_path=existing_path, same_as_saved=same_as_saved)
    result.agreement_text = agreement_text
    return result
