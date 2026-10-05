"""
Seeding and exporting RA annotations for the multisignal annotation
prototype. Nothing here imports from or writes to the existing
physioProcess/physioCorrection scripts -- it only reads the same
physio.tsv.gz-shaped DataFrame that physio_io.py produces, and writes new,
independently-named JSON files.
"""

import contextlib
import glob
import io
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import mne
import numpy as np

import session_log
import session_summary_gui
from ppg_checks import ppg_check_lines, same_beat_lines
from provenance import channel_provenance, new_pipeline_ppg, provenance_mismatch
from channel_config import (COMPANION_CHANNELS, GUIDE_CHANNELS, GUIDE_CONTEXT_TEXT, GUIDE_COUNTER_LOSS_TEXT,
                            GUIDE_EXCEPTION_MAX_LISTED, GUIDE_EXCEPTION_REASONS, GUIDE_OPT_IN, GUIDE_PREMARK_REASONS,
                            GUIDE_PREMARKED_TEXT, GUIDE_TEXTS, PPG_ADD_PEAK_TEXT, REFERENCE_CHANNELS, check_review_alone,
                            empty_channels, is_present, no_data_message, with_companions)
from physio_io import (
    apply_unit_labels,
    build_raw,
    compute_channel_value_ranges,
    compute_scalings,
    format_channel_value_ranges,
    MACHINE_QC_FLAGGED_WARNING_FRACTION,
    machine_qc_flagged_fraction,
    machine_qc_notes,
    coverage_only,
    machine_qc_segments,
    reason_summary,
    sidecar_descriptions,
    sidecar_unit_labels,
)


def combine_with_reference_channels(channel_configs, df, ppg_guide=False, sidecar=None, ecg_ppg_guide=False):
    """
    Adds any available REFERENCE_CHANNELS (e.g. "event", a read-only
    annotation_mode="none" track) present in df to channel_configs, so
    Stage B always shows them regardless of which signals the user chose
    to review via --channels. Reference channels are never user-selectable
    and never appear in export_annotations()'s output -- see
    channel_config.py's REFERENCE_CHANNELS docstring for why.

    ppg_guide=True (the name is historical: it now covers every guide row)
    also adds the read-only GUIDE_CHANNELS rows that apply to this session
    -- PPG under ECG or SBP/DBP, EDA under RSP, RSP under EDA -- each right
    after the last scored channel it's for, and only when its own channel
    isn't being scored and it can be shown (see session_guides()). The PPG
    guide needs the new CareTaker timing (sidecar plan Phase 1b; HLU's
    decision G2, 2026-09-26: old-pipeline PPG is multi-humped and
    unreliable). sidecar=None means synthetic data (allowed). The PPG under
    ECG is opt-in (GUIDE_OPT_IN; HLU, 2026-10-03): only with ecg_ppg_guide.
    """
    order = list(channel_configs)
    after = {}
    if ppg_guide:
        for guide_key, (scored, reason) in session_guides(channel_configs, df, sidecar, ecg_ppg_guide).items():
            if reason is None:
                after.setdefault(max(scored, key=order.index), []).append(guide_key)
    combined = {}
    for key, cfg in channel_configs.items():
        combined[key] = cfg
        for guide_key in after.get(key, []):
            combined[guide_key] = GUIDE_CHANNELS[guide_key]
    for key, cfg in REFERENCE_CHANNELS.items():
        if key in df.columns and key not in combined:
            combined[key] = cfg
    return combined


def _inside(i, segments):
    return any(start <= i < end for start, end in segments)


def drop_peaks_in_bad_stretches(channel_outputs):
    """
    Bad stretches never keep peaks (HLU, 2026-09-27): removes, in place, each
    companion's parent peaks that fall inside its bad stretches (e.g. PPG
    peaks inside "bad_ppg"), and returns {parent: [dropped indices]}. Used by
    Step 2's and Step 3's exports, so the summary, the saved file and Phase D
    all see the same result.
    """
    dropped = {}
    for comp_key, comp_cfg in COMPANION_CHANNELS.items():
        parent = comp_cfg["companion_of"]
        out, comp = channel_outputs.get(parent), channel_outputs.get(comp_key)
        if not out or not comp or out.get("mode") != "point":
            continue
        segments = comp.get("bad_segments") or []
        gone = [i for i in out["indices"] if _inside(i, segments)]
        if gone:
            out["indices"] = [i for i in out["indices"] if not _inside(i, segments)]
            dropped[parent] = gone
    return dropped


def companion_provenance(provenance):
    """A companion's provenance is its parent column's (e.g. bad_ppg -> the ppg column), with "column" naming it."""
    for comp_key, comp_cfg in COMPANION_CHANNELS.items():
        parent = comp_cfg["companion_of"]
        if comp_key in provenance and parent in provenance:
            provenance[comp_key] = {**provenance[parent], "column": parent}
    return provenance


def _guide_not_shown_reason(guide_key, df, sidecar=None):
    """None when guide_key's row can be shown for this file, else the console line saying why it isn't."""
    label = GUIDE_CHANNELS[guide_key]["label"]
    name = label.split(" (")[0]
    if guide_key not in df.columns or df[guide_key].isna().all():
        return f"{label}: not shown -- this run has no usable {name}."
    if guide_key == "ppg" and not new_pipeline_ppg(sidecar):
        return (f"{label}: not shown -- this file was processed before the PPG timing fix, so its PPG isn't "
                f"reliable enough to guide you.")
    # A column the machine marks as mostly unusable isn't shown: e.g. a dead
    # RSP belt, which the viewer would stretch to look like fast breathing.
    qc, _note = machine_qc_segments(sidecar, len(df), only=[guide_key]) if sidecar is not None else ({}, None)
    if guide_key in qc and machine_qc_flagged_fraction(qc[guide_key], len(df)) > MACHINE_QC_FLAGGED_WARNING_FRACTION:
        return (f"{label}: not shown -- the machine check marks most of this run's {name} as unusable, so it "
                f"can't guide you.")
    return None


def session_guides(channel_configs, df, sidecar=None, ecg_ppg_guide=False):
    """
    The guide rows that apply to this session: {guide key: (the scored
    channels it goes under, None if it can be shown or else the reason it
    isn't)}. A guide applies when a channel it's for is being scored and its
    own channel isn't (GUIDE_CHANNELS). An opt-in pair (GUIDE_OPT_IN: the PPG
    under ECG) counts only with ecg_ppg_guide.
    """
    guides = {}
    for guide_key, cfg in GUIDE_CHANNELS.items():
        scored = [k for k in cfg["guide_for"] if k in channel_configs
                  and (ecg_ppg_guide or (guide_key, k) not in GUIDE_OPT_IN)]
        if scored and guide_key not in channel_configs:
            guides[guide_key] = (scored, _guide_not_shown_reason(guide_key, df, sidecar))
    return guides


def guide_shown_by_channel(channel_configs, full_channel_configs):
    """{scored channel that can have a guide: whether its guide row is on screen} (recorded as "guide_shown")."""
    return {k: full_channel_configs.get(guide_key) is cfg
            for guide_key, cfg in GUIDE_CHANNELS.items() for k in cfg["guide_for"] if k in channel_configs}


def _guide_text(guide_key, scored_key, df, sidecar=None):
    """A shown guide's line, plus GUIDE_PREMARKED_TEXT when the scored channel's machine check has the reason."""
    text = GUIDE_TEXTS[(guide_key, scored_key)]
    reason = GUIDE_PREMARK_REASONS.get((guide_key, scored_key))
    if reason and sidecar is not None:
        qc, _note = machine_qc_segments(sidecar, len(df), only=[scored_key])
        if reason in ((qc.get(scored_key) or {}).get("by_reason") or {}):
            text += " " + GUIDE_PREMARKED_TEXT
    return text


def _guide_exception_line(guide_key, scored_key, df, sidecar=None):
    """
    The exception line for a shown guide when the GUIDE channel's own machine
    check has the GUIDE_EXCEPTION_REASONS reason (PPG's counter_loss under
    SBP/DBP), naming where the stretches are on the viewer's time axis; else None.
    """
    reason = GUIDE_EXCEPTION_REASONS.get((guide_key, scored_key))
    if not reason or sidecar is None:
        return None
    sfreq = (sidecar or {}).get("SamplingFrequency") or 1000
    qc, _note = machine_qc_segments(sidecar, len(df), only=[guide_key])
    spans = ((qc.get(guide_key) or {}).get("by_reason") or {}).get(reason) or []
    if not spans:
        return None
    total = sum(end - start for start, end in spans) / sfreq
    listed = ", ".join(f"{start / sfreq:.1f}-{end / sfreq:.1f} s" for start, end in spans[:GUIDE_EXCEPTION_MAX_LISTED])
    more = len(spans) - GUIDE_EXCEPTION_MAX_LISTED
    where = f"{total:.1f} s in all, at {listed}" + (f" and {more} more" if more > 0 else "")
    return GUIDE_COUNTER_LOSS_TEXT.format(count=len(spans), where=where)


def guide_banner_lines(channel_configs, df, ppg_guide, sidecar=None, ecg_ppg_guide=False):
    """
    Console lines about this session's guide rows: what each shown one is
    for, or why one isn't shown; and, for an opt-in pair left off (the PPG
    under ECG), that it's off by default or, when the row is on screen for
    another scored channel, not to use it for this one.
    """
    if not ppg_guide:
        return []
    lines, shown = [], False
    guides = session_guides(channel_configs, df, sidecar, ecg_ppg_guide)
    for guide_key, (scored, reason) in guides.items():
        if reason:
            lines.append(reason)
        else:
            shown = True
            lines += list(dict.fromkeys(_guide_text(guide_key, k, df, sidecar) for k in scored))
            lines += list(dict.fromkeys(line for line in (_guide_exception_line(guide_key, k, df, sidecar)
                                                          for k in scored) if line))
    for (guide_key, scored_key), texts in ([] if ecg_ppg_guide else GUIDE_OPT_IN.items()):
        if scored_key not in channel_configs or guide_key in channel_configs:
            continue
        if guide_key in guides and guides[guide_key][1] is None:
            lines.append(texts["shown_for_others"])
        elif guide_key not in guides and _guide_not_shown_reason(guide_key, df, sidecar) is None:
            lines.append(texts["off"])
    if shown:
        lines.append(GUIDE_CONTEXT_TEXT)
    return lines


def compute_event_transitions(df, event_col, sfreq):
    """
    Builds an mne.find_events()-shaped events array ([[sample_idx, 0,
    event_id], ...]) from every point where df[event_col]'s value changes,
    for use with raw.plot(events=...) -- MNE's own built-in mechanism for
    drawing read-only, text-labeled vertical markers (mne_qt_browser's
    EventLine class is explicitly movable=False), so the RA can read the
    actual event/task-phase code directly off the plot instead of only
    seeing "event"'s waveform-like shape. Unlike our own point/segment
    annotations, these can't be accidentally dragged, deleted, or hit the
    click-drag merge bug (see HANDOFF.md's debugging history) -- they're a
    completely different, non-interactive graphics item.

    No event_id/event_color mapping is passed to raw.plot() -- MNE falls
    back to showing the raw numeric code as the label and auto-assigning
    colors, which is exactly what's wanted here (no extra config needed).

    Returns an (N, 3) int array, or None if event_col isn't in df or its
    value never changes (nothing to mark).
    """
    if event_col not in df.columns:
        return None
    values = df[event_col].to_numpy()
    if len(values) < 2:
        return None
    change_indices = np.where(np.diff(values) != 0)[0] + 1
    if len(change_indices) == 0:
        return None
    event_ids = values[change_indices].astype(int)
    return np.column_stack([change_indices, np.zeros_like(change_indices), event_ids]).astype(int)


def find_template_json(out_dir, file_stem, explicit_path=None, initials=None, require_own_initials=False):
    """
    Locates a Stage A (qrs_template_stage.py) output JSON for this file.
    Stage A's outputs are named <file_stem>_ecg_corrected_qrs_<initials>.json
    (one per RA who's built a template for this subject), so unlike a
    single fixed filename, there can be zero, one, or several candidates.

    require_own_initials=True (used by physio_review.py's Stage 2 and its
    combined flow): every RA must build and use their OWN template, in
    full, regardless of who's first to review a given subject -- one RA's
    template is never borrowed to seed another RA's review, even if it's
    the only one that exists. This is a deliberate reliability choice, not
    just a safety net: seeding every RA from a shared template would bias
    their independent corrections toward agreement (or a shared blind
    spot) before they've even started, undermining what Stage 3's
    reconciliation is trying to measure. With this on, no searching/
    disambiguation happens at all -- it just returns the path YOUR
    initials would produce, whether or not it (or anyone else's) exists;
    the caller's own exists-check decides what to do next (see
    resolve_ecg_peaks()'s "template" mode, which raises clearly if it's
    missing). `explicit_path` still short-circuits everything, since an
    explicit override is exactly what physio_annotate.py's standalone
    `--template-json` flag is for.

    require_own_initials=False (physio_annotate.py's default, unchanged):
        - explicit_path given -> use it directly, no searching.
        - exactly one candidate found -> use it, regardless of whose
          initials it's under.
        - multiple candidates AND one matches `initials` -> prefer that one
          (handles the common case of one RA running both stages).
        - multiple candidates, none matching `initials` -> raise clearly,
          listing them, rather than silently guessing which RA's template
          to trust.
        - zero candidates -> return a (deliberately nonexistent) path, so
          the existing "does this path exist" logic in resolve_ecg_peaks()
          still drives the auto-fallback / clear-error behavior unchanged.
    """
    if explicit_path:
        return explicit_path

    if require_own_initials:
        if not initials:
            raise ValueError("find_template_json(require_own_initials=True) needs `initials`.")
        return os.path.join(out_dir, f"{file_stem}_ecg_corrected_qrs_{initials}.json")

    pattern = os.path.join(out_dir, f"{file_stem}_ecg_corrected_qrs_*.json")
    candidates = sorted(glob.glob(pattern))

    if initials:
        preferred = os.path.join(out_dir, f"{file_stem}_ecg_corrected_qrs_{initials}.json")
        if preferred in candidates:
            return preferred

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        raise SystemExit(
            "Multiple QRS templates found for this file, and none match your initials:\n"
            + "\n".join(f"  - {c}" for c in candidates)
            + "\nPick one explicitly with --template-json <path>."
        )

    # No candidates: return a path that's guaranteed not to exist, so
    # resolve_ecg_peaks()'s existing exists-check drives auto-fallback/error.
    return os.path.join(out_dir, f"{file_stem}_ecg_corrected_qrs_none-found.json")


def resolve_ecg_peaks(df, ecg_source, template_json_path):
    """
    Decides which ECG peaks to seed Stage B with, per --ecg-source:
        "batch"    -> always use the batch pipeline's existing ecg_peaks column.
        "template" -> require a Stage A (qrs_template_stage.py) output JSON.
        "auto"     -> use the template JSON if it exists, else fall back to batch.

    Returns (df, source_label). df's "ecg_peaks" column is overwritten in
    place with the template's peaks when the template source is used, so
    the existing seed_annotations() path needs no changes.
    """
    template_exists = os.path.exists(template_json_path)

    if ecg_source == "batch":
        return df, "batch pipeline (physioBatch.py's automated detection)"

    if ecg_source == "template" and not template_exists:
        raise FileNotFoundError(
            f"You haven't built your own ECG template for this file yet (looked for "
            f"{template_json_path}, which doesn't exist). Run Step 1 (Build ECG template) first, "
            f"or choose ECG peak source 'Batch only' (command line: --ecg-source batch)."
        )

    if ecg_source in ("template", "auto") and template_exists:
        payload = read_saved_json(template_json_path, what="ECG template file")
        corrected_peaks = (payload.get("ecg") or {}).get("corrected_peaks") if isinstance(payload.get("ecg"), dict) \
            else None
        if not isinstance(corrected_peaks, list) or not all(
                isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(df) for i in corrected_peaks):
            raise SystemExit(f"The ECG template file {os.path.basename(template_json_path)} has peaks outside this "
                             f"file ({len(df)} samples) or in an unexpected format, so it doesn't fit this file. "
                             f"Nothing has been changed. Rebuild your template (Step 1) or tell the lab staff.")
        df = df.copy()
        df["ecg_peaks"] = False
        df.loc[corrected_peaks, "ecg_peaks"] = True
        return df, f"QRS template ({template_json_path}, {len(corrected_peaks)} peaks)"

    return df, "batch pipeline (physioBatch.py's automated detection; no template JSON found)"


DEFAULT_SNAP_WINDOW_SEC = 0.05
"""
The snap/agreement-tolerance window was originally a hardcoded 50-SAMPLE
constant throughout this project -- correct only at this study's 1000 Hz
sampling rate (=50ms). At a hypothetical future study's 500 Hz, the SAME
"50" would mean 100ms; at 2000 Hz, 25ms -- a materially different
real-world tolerance despite an unchanged number. Expressed here in
SECONDS instead, and converted to a sample count via each call site's own
sfreq, so the real-world window stays constant across studies regardless
of sampling rate. Confirmed as a real, worth-fixing fragility (not just
hypothetical) by the user. snap_to_local_max() itself still takes a plain
sample count -- it's a low-level, sfreq-agnostic utility; callers compute
the right sample count from this constant and their own sfreq.
"""


DRAWN_PEAK_CLIMB_SEC = 0.015
"""
Option C (HLU, 2026-09-26): how far a drawn peak may climb when the RA's
box cut off the top of the wave. Replaces the second re-snap to the highest
sample within +/-50 ms, which could pull a short R wave onto a taller MHD
wave or artifact next to it.
"""
TOP_RIPPLE_SAMPLES = 2
"""A "top" is at least as high as everything within +/- this many samples, so a one-sample ripple doesn't stop a climb."""


def _is_top(signal, i, k=TOP_RIPPLE_SAMPLES):
    lo, hi = max(0, i - k), min(len(signal), i + k + 1)
    return not np.isnan(signal[i]) and signal[i] >= np.nanmax(signal[lo:hi])


def _climb_if_cut(signal, p, start, end, climb_samples):
    """
    p is the highest point inside the box [start, end). If it sits on the box's
    first or last sample and the signal keeps rising just outside the box (the
    box cut off the top), step outward to the first top, at most climb_samples;
    otherwise, or if no top is reached within the cap, return p unchanged.
    """
    n = len(signal)
    rises_right = p == end - 1 and p + 1 < n and signal[p + 1] > signal[p]
    rises_left = p == start and p - 1 >= 0 and signal[p - 1] > signal[p]
    if rises_right and rises_left:  # a one-sample box in a trough: go toward the higher side
        rises_left = signal[p - 1] > signal[p + 1]
        rises_right = not rises_left
    if not (rises_right or rises_left):
        return p
    step = 1 if rises_right else -1
    i = p
    for _ in range(climb_samples):
        nxt = i + step
        if nxt < 0 or nxt >= n or np.isnan(signal[nxt]):
            break
        i = nxt
        if _is_top(signal, i):
            return i
    return p


def point_sample_from_annotation(onset_sample, duration_samples, signal, climb_samples=None):
    """
    Converts one point-style annotation into a sample index. A genuine
    click-drag (duration > 0) returns the true peak WITHIN EXACTLY the region
    the RA drew; an untouched seeded point (duration 0) returns its onset.

    mne_qt_browser stores a drag's onset as its LEFT EDGE (rgn[0]), never its
    center -- confirmed by reading _region_changed() directly. A "drag around
    the peak" gesture starts before the peak, so using the onset alone put
    the true peak outside the later +/-50 ms snap window whenever the drag
    was wider than that (HANDOFF.md §38; ~25-30 real sub-085 peaks shifted
    by ~100-120 ms). This helper is shared by Step 1 (qrs_template.py), Step 2
    (export_annotations below) and Step 3 (reconcile.py) -- originally only
    Step 2 had the fix (audit finding C3).

    With climb_samples (Option C, HLU 2026-09-26; see DRAWN_PEAK_CLIMB_SEC):
    the box maximum is kept when it is inside the box -- it is then a top,
    however short, and a taller wave outside the box never pulls it away;
    only when the box cut off the top (the maximum is on the box's edge and
    the signal keeps rising outside it) does it climb to that wave's top.
    This is the whole rule for drawn peaks: there's no second re-snap.
    """
    if duration_samples > 0 and signal is not None:
        signal = np.asarray(signal, dtype=float)
        start = max(0, onset_sample)
        window_end = min(len(signal), onset_sample + duration_samples)
        signal_slice = signal[start:window_end]
        if len(signal_slice) and not np.all(np.isnan(signal_slice)):
            p = start + int(np.nanargmax(signal_slice))
            if climb_samples:
                p = _climb_if_cut(signal, p, start, window_end, int(climb_samples))
            return p
    return onset_sample


def snap_to_local_max(signal, indices, window_samples=50):
    """
    Refines a list of rough click indices to the true local maximum within
    +/- window_samples in the raw/cleaned signal. Mirrors the same
    convenience systole's Editor gives for free when clicking near a peak,
    and matches the snap_to_local_max() helper already used in
    step1_qrs_template.ipynb.
    """
    signal = np.asarray(signal, dtype=float)
    refined = []
    for idx in indices:
        start = max(0, idx - window_samples)
        end = min(len(signal), idx + window_samples)
        window = signal[start:end]
        # nanargmax, not argmax: np.argmax returns the first NaN's position if
        # the window contains any (audit finding M8).
        if end <= start or np.all(np.isnan(window)):
            refined.append(idx)
            continue
        refined.append(start + int(np.nanargmax(window)))
    return sorted(set(refined))


ONSET_NUDGE_SAMPLES = 0.001
"""
How far apart (in samples) seed_annotations() spreads seeded marks that would
otherwise share an exact start time. mne-qt-browser finds the stored
annotation behind an on-screen mark by exact start time and takes the first
match, so deleting or moving one of two marks at the same time (e.g. an ECG
and a PPG peak on one sample, or the zero-duration label placeholders at
0 s) could change the OTHER one in the saved data, while the screen showed
the right one gone (ppg-plan §3.1 #4, verified in mne-qt-browser 0.7.5). A
nudge of a thousandth of a sample still rounds back to the same sample on
export. The step is never below 1 microsecond: raw.set_annotations() rounds
onsets to whole microseconds, which would erase a smaller nudge above 1000 Hz
(checkpoint-1 review).
"""


def unique_onsets(onsets, sfreq):
    """
    Returns onsets (seconds) with exact duplicates spread by
    max(ONSET_NUDGE_SAMPLES / sfreq, 1 us) each. Raises if so many marks share
    one time that the spread would reach half a sample (it would then export
    to a different sample).
    """
    step = max(ONSET_NUDGE_SAMPLES / sfreq, 1e-6)
    seen = {}
    result = []
    for onset in onsets:
        k = seen.get(onset, 0)
        seen[onset] = k + 1
        if k * step * sfreq >= 0.5:
            raise ValueError(f"{k + 1} annotations share the start time {onset} s; too many to keep apart.")
        result.append(onset + k * step)
    return result


def seed_annotations(raw, df, channel_configs, saved_channels, sfreq, machine_qc=None, exclude_seeds=None):
    """
    Pre-populates the Raw object's annotations for Stage B, decided PER
    CHANNEL rather than for the whole session at once: a channel present
    in saved_channels (this RA's own prior save for this run) uses that
    saved data exactly as before (point indices or bad segments); any
    OTHER point-mode channel -- including every channel on a completely
    fresh session (saved_channels == {}), AND any channel newly added to
    this session's --channels that a prior save never covered -- falls
    back to its automated peak-detection column (auto_peaks_col) instead
    of being left silently blank. A segment-mode channel missing from
    saved_channels starts from machine_qc's spans for it (the sidecar's
    MachineQC bad segments, via physio_io.machine_qc_segments()) when there
    are any, else blank. A saved entry whose mode doesn't match the
    channel's current mode (e.g. an RSP review from when RSP was a point
    channel) must not be passed in saved_channels -- run_stage_b() filters
    those out, so such a channel starts fresh.

    Replaces the former seed_point_annotations()/seed_annotations_from_saved()
    split, which seeded at the whole-session level (EITHER all channels
    from the saved file, OR all channels from auto-detection, never a mix)
    -- confirmed as a real, live data-gap: resuming a run whose saved file
    only covered ecg so far, then reviewing rsp for the first time in that
    same session, silently produced a blank rsp with none of its 159 real
    auto-detected peaks, rather than falling back to them the way a
    genuinely fresh session would have. See HANDOFF.md.

    Deliberately NOT channel-scoped (no ch_names on the mne.Annotations
    below), even though each annotation conceptually belongs to one
    channel. Scoping is unnecessary -- descriptions are already unique per
    channel ("peak_ecg", "peak_rsp", etc.), so export_annotations() can
    (and does) disambiguate by description alone. It's also actively
    harmful: mne-qt-browser's interactive region-editing has a bug for
    channel-scoped ("single-channel") annotations that overlap another
    annotation (mne_qt_browser/_graphic_items.py's AnnotRegion._region_changed()
    warns "Can not combine channel-based annotations with any other
    annotation" and can leave its internal state inconsistent -- observed
    live as a crash in _remove_region()/_get_onset_idx() when editing a
    window with ~20+ closely-spaced seeded ECG peaks, corrupting the
    resulting peak indices). See HANDOFF.md's debugging history for the
    full incident -- don't reintroduce ch_names scoping here.

    exclude_seeds: {point channel: [[start, end), ...]} -- automated peaks
    inside these stretches aren't seeded (e.g. PPG peaks inside the
    pre-marked bad_ppg stretches: bad stretches never keep peaks).
    """
    onsets, durations, descriptions = [], [], []

    for ch_key, cfg in channel_configs.items():
        saved = saved_channels.get(ch_key)

        if saved and cfg["annotation_mode"] == "point" and saved.get("mode") == "point":
            for idx in saved.get("indices", []):
                onsets.append(idx / sfreq)
                durations.append(0.0)
                descriptions.append(cfg["point_label"])
            continue

        if cfg["annotation_mode"] == "segment":
            if not is_present(ch_key, cfg, df.columns):
                continue
            if saved and saved.get("mode") == "segment":
                real_segments = saved.get("bad_segments", [])
            else:
                # A fresh channel starts from the pipeline's MachineQC bad
                # segments when the sidecar has them, exactly as given
                # (sidecar plan Phase 2) -- like ECG/PPG starting from the
                # automated peaks. See physio_io.machine_qc_segments() for
                # the guards that leave it blank instead.
                real_segments = ((machine_qc or {}).get(ch_key) or {}).get("segments", [])
            for start, end in real_segments:
                onsets.append(start / sfreq)
                durations.append(max(end - start, 1) / sfreq)
                descriptions.append(cfg["segment_label"])
            # Always register the channel's label with a zero-duration
            # placeholder, even when real segments are seeded: the RA may
            # delete them all (e.g. a pre-filled machine flag) and choose "Go
            # back to the viewer", and mne-qt-browser rebuilds its description
            # picker strictly from descriptions present in raw.annotations,
            # so the label would vanish (checkpoint-1 review; until
            # 2026-09-26 the placeholder was added only for channels with no
            # segments). Confirmed live originally -- the RA saw no
            # description available reviewing eda. export_annotations()
            # filters this back out by requiring duration > 0, since a real
            # drag-created segment always has nonzero width. See HANDOFF.md.
            onsets.append(0.0)
            durations.append(0.0)
            descriptions.append(cfg["segment_label"])
            continue

        if cfg["annotation_mode"] != "point":
            continue
        auto_col = cfg.get("auto_peaks_col")
        if auto_col is None or auto_col not in df.columns or ch_key not in df.columns:
            continue

        peak_indices = np.where(df[auto_col].to_numpy())[0]
        skip = (exclude_seeds or {}).get(ch_key) or []
        for idx in peak_indices:
            if skip and _inside(int(idx), skip):
                continue
            onsets.append(idx / sfreq)
            durations.append(0.0)
            descriptions.append(cfg["point_label"])

    if onsets:
        annotations = mne.Annotations(onset=unique_onsets(onsets, sfreq), duration=durations,
                                      description=descriptions)
        raw.set_annotations(annotations)

    return raw


def read_saved_json(path, what="saved review file"):
    """
    Reads one of this tool's JSON files, or stops with a plain message naming
    the file (never a Python traceback) when it can't be read -- damaged,
    half-written, hand-edited -- or isn't a JSON object. UTF-8 with or without
    a byte-order mark (2026-09-26 safety fix).
    """
    try:
        with open(path, encoding="utf-8-sig") as f:
            payload = json.load(f)
    except (OSError, UnicodeDecodeError, ValueError) as e:
        raise SystemExit(f"The {what} {os.path.basename(path)} can't be read ({type(e).__name__}: {e}). Nothing "
                         f"has been changed. Please tell the lab staff (an earlier version may be in its "
                         f"backups/ folder).")
    if not isinstance(payload, dict):
        raise SystemExit(f"The {what} {os.path.basename(path)} isn't in the expected format. Nothing has been "
                         f"changed. Please tell the lab staff.")
    return payload


def saved_channel_entries(payload, path, what="saved review file"):
    """The payload's "channels" dict, checked to be a dict of dicts (else a plain stop)."""
    channels = payload.get("channels", {})
    if not isinstance(channels, dict) or not all(isinstance(v, dict) for v in channels.values()):
        raise SystemExit(f"The {what} {os.path.basename(path)} has a damaged 'channels' section. Nothing has been "
                         f"changed. Please tell the lab staff.")
    return channels


def _is_legacy_key(ch_key):
    return bool(re.search(r"_legacy(_\d+)?$", ch_key))


def check_saved_file_fits(payload, channels, path, sfreq=None, n_samples=None, what="saved review file"):
    """
    Stops, plainly, when a saved file doesn't fit the current data file: a
    different sampling_rate, or marks that aren't whole sample numbers inside
    it (point indices in [0, n); segments with 0 <= start < end <= n). Until
    2026-09-26 such marks were dropped or kept silently. Kept "_legacy" entries
    are never loaded, so they aren't checked.
    """
    name = os.path.basename(path)
    rate = payload.get("sampling_rate")
    if sfreq is not None and rate is not None:
        try:
            differs = float(rate) != float(sfreq)
        except (TypeError, ValueError):
            differs = True
        if differs:
            raise SystemExit(f"The {what} {name} was saved at {rate} Hz, but this file is {sfreq} Hz, so its marks "
                             f"wouldn't line up. Nothing has been changed. Please tell the lab staff.")
    if n_samples is None:
        return
    for ch_key, entry in channels.items():
        if _is_legacy_key(ch_key):
            continue
        bad = 0
        for i in entry.get("indices", []) or []:
            if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < n_samples:
                bad += 1
        for seg in entry.get("bad_segments", []) or []:
            ok = (isinstance(seg, (list, tuple)) and len(seg) == 2
                  and all(isinstance(v, int) and not isinstance(v, bool) for v in seg)
                  and 0 <= seg[0] < seg[1] <= n_samples)
            bad += 0 if ok else 1
        if bad:
            raise SystemExit(f"The {what} {name} has {bad} {ch_key} mark(s) outside this file (it has {n_samples} "
                             f"samples) or not whole sample numbers, so it doesn't fit this file. Nothing has been "
                             f"changed. Please tell the lab staff.")


def load_saved_annotations(out_dir, file_stem, initials, sfreq=None, n_samples=None):
    """
    Loads a previously-saved <file_stem>_annotations_<initials>.json for
    this run/RA, if one exists, so a resumed Stage B session can seed from
    the RA's own prior corrections instead of starting over from the
    automated peak-detection columns (which would silently discard any
    peaks they'd deleted/added or bad segments they'd marked in an earlier
    sitting that they closed out of before finishing the whole run).

    Returns (channel_outputs, path) -- channel_outputs has the same shape
    export_annotations() produces -- or None if no saved file exists yet
    (i.e. this is this RA's first session on this run). An unreadable or
    ill-fitting file stops with a plain message (read_saved_json(),
    check_saved_file_fits(); pass sfreq and n_samples to check the fit).
    """
    path = os.path.join(out_dir, f"{file_stem}_annotations_{initials}.json")
    if not os.path.exists(path):
        return None
    payload = read_saved_json(path)
    channels = saved_channel_entries(payload, path)
    # With the current file's sfreq / sample count: a plain stop if the saved
    # file doesn't fit it (2026-09-26 safety fix).
    check_saved_file_fits(payload, channels, path, sfreq=sfreq, n_samples=n_samples)
    return channels, path


def _annotation_channels(annotation, raw_ch_names):
    """mne stores per-annotation ch_names as a tuple; empty tuple means 'all channels'."""
    if not annotation:
        return list(raw_ch_names)
    return list(annotation)


FALLBACK_MERGE_WARNING_SEC = 0.4
MIN_MERGE_WARNING_SEC = 0.15
MERGE_WARNING_FRACTION_OF_INTERVAL = 0.5
"""
mne-qt-browser's click-and-drag annotation tool silently MERGES a new
click-drag with any existing same-description annotation it overlaps or
touches, combining them into one wider region instead of keeping two
separate points (see mne_qt_browser/_widgets.py's mouseDragEvent(),
around the "Add to regions/merge regions" block). Since export_annotations()
only reads a point channel's annotation ONSET (its left edge), a merged
region silently collapses two real peaks into one exported index, with no
error or indication anything went wrong -- confirmed as the root cause of a
live, hard-to-spot data-loss incident (see HANDOFF.md's debugging history).

Calibrating this: a REAL merge of two adjacent peaks produces a duration
close to the interval between them (e.g. the R-R interval for ECG), since
that's the gap the merged region has to span. A single, deliberately-placed
peak from a normal click-and-drag gesture (the documented "drag a small box
around the peak" technique) can easily span 100-350ms on its own --
confirmed against a real full-run session with zero actual lost peaks
(verified via R-R interval analysis) where every legitimately single peak's
drag width fell under 350ms.

A single FIXED threshold can't be right for everyone, though: a fast heart
rate shrinks the R-R interval, and if it drops close to a fixed cutoff, a
real merge could fall UNDER it and go undetected right when dense peaks
(and therefore accidental merges) are most likely. So the threshold is
derived from THIS session's own typical point-to-point interval (via the
automated peak detection, already available before the RA edits anything)
whenever there's enough data to estimate it -- MERGE_WARNING_FRACTION_OF_INTERVAL
(half the typical interval) comfortably clears the ~350ms real-usage ceiling
for a normal resting-range heart rate while still scaling down for a faster
one. MIN_MERGE_WARNING_SEC is a floor so a very fast heart rate (or noisy
automated detection) can't push the threshold low enough to start
false-positiving on normal clicks again. FALLBACK_MERGE_WARNING_SEC is used
when there's no automated-peak baseline to estimate from at all (e.g. too
few automated peaks, or a channel type without one).

An earlier version of this was a single fixed 0.1s constant, calibrated on
a wrong assumption ("a click only produces a few ms of width") -- it
flooded a real session with 100+ false-positive warnings for completely
normal peaks. Don't go back to a single fixed sub-350ms value without new
real-usage evidence (see HANDOFF.md's debugging history).
"""


def _merge_warning_threshold(df, auto_peaks_col, sfreq):
    """
    Estimates this channel's typical point-to-point interval (e.g. the R-R
    interval for ECG) from its automated peak-detection column, and returns
    half of that as the merge-warning threshold -- see the constants above
    for the full rationale. Falls back to FALLBACK_MERGE_WARNING_SEC if
    there's no usable automated baseline (missing column, or too few peaks
    to estimate an interval from).

    For ECG specifically, this ends up estimating heart rate from the
    RA-corrected Stage A (QRS template) peaks rather than the raw batch
    detection whenever one exists: physio_review.py's resolve_ecg_peaks()
    already overwrites df["ecg_peaks"] with the template's cross-correlated
    peaks before Stage B ever runs (except under --ecg-source batch, where
    that's a deliberate opt-out). Since a template-corrected estimate is
    more reliable than the raw automated one -- Stage A exists specifically
    because plain automated detection can be unreliable (e.g. in-scanner
    ECG) -- this makes the threshold more trustworthy for free, with no
    extra plumbing needed here.
    """
    if auto_peaks_col is None or auto_peaks_col not in df.columns:
        return FALLBACK_MERGE_WARNING_SEC

    peak_indices = np.where(df[auto_peaks_col].to_numpy())[0]
    if len(peak_indices) < 3:
        return FALLBACK_MERGE_WARNING_SEC

    median_interval_sec = float(np.median(np.diff(peak_indices))) / sfreq
    return max(MIN_MERGE_WARNING_SEC, MERGE_WARNING_FRACTION_OF_INTERVAL * median_interval_sec)


def export_annotations(raw, df, channel_configs, sfreq, snap_window_samples=None, warnings_out=None):
    """
    Reads back whatever is currently in raw.annotations (after an RA session)
    and converts it into a per-channel dict:
        point channels   -> {"mode": "point", "indices": [sorted ints]}
        segment channels -> {"mode": "segment", "bad_segments": [[start, end], ...]}
    Channels with annotation_mode == "none" (read-only reference tracks
    like "event") are skipped entirely -- there's nothing an RA corrects on
    them, so no entry is created for them at all.

    snap_window_samples defaults to DEFAULT_SNAP_WINDOW_SEC converted via
    this call's own sfreq (not a fixed sample count) -- pass an explicit
    value only to override that.

    Also warns (to console) about any point-channel annotation with a
    suspiciously large duration relative to that channel's own typical
    peak-to-peak interval -- see _merge_warning_threshold()'s docstring for
    why that's a strong signal of two peaks having been silently merged
    into one during editing.
    """
    if snap_window_samples is None:
        snap_window_samples = round(DEFAULT_SNAP_WINDOW_SEC * sfreq)

    channel_outputs = {}
    merge_thresholds = {}
    for ch_key, cfg in channel_configs.items():
        if not is_present(ch_key, cfg, raw.ch_names):
            continue
        if cfg["annotation_mode"] == "point":
            channel_outputs[ch_key] = {"mode": "point", "indices": []}
            merge_thresholds[ch_key] = _merge_warning_threshold(df, cfg.get("auto_peaks_col"), sfreq)
        elif cfg["annotation_mode"] == "segment":
            channel_outputs[ch_key] = {"mode": "segment", "bad_segments": []}

    possible_merges = []
    edited_points = {}  # ch_key -> indices the RA placed by click-drag this session

    for ann in raw.annotations:
        onset_sample = int(round(ann["onset"] * sfreq))
        duration_samples = int(round(ann["duration"] * sfreq))
        ch_names = _annotation_channels(ann.get("ch_names", ()), raw.ch_names)

        for ch_key, cfg in channel_configs.items():
            if not is_present(ch_key, cfg, raw.ch_names) or not is_present(ch_key, cfg, ch_names):
                continue

            if cfg["annotation_mode"] == "point" and ann["description"] == cfg["point_label"]:
                # A click-drag is resolved to the true peak within the drawn
                # region -- see point_sample_from_annotation() for why.
                signal = df[ch_key].to_numpy() if ch_key in df.columns else None
                point_sample = point_sample_from_annotation(onset_sample, duration_samples, signal,
                                                            climb_samples=round(DRAWN_PEAK_CLIMB_SEC * sfreq))
                channel_outputs[ch_key]["indices"].append(point_sample)
                if duration_samples > 0:
                    edited_points.setdefault(ch_key, set()).add(point_sample)
                if ann["duration"] > merge_thresholds[ch_key]:
                    possible_merges.append((ch_key, ann["onset"], ann["duration"]))

            elif (
                cfg["annotation_mode"] == "segment"
                and ann["description"] == cfg["segment_label"]
                and ann["duration"] > 0
            ):
                # duration > 0 excludes seed_annotations()'s own zero-duration
                # placeholder (registers a segment-mode channel's label so
                # mne-qt-browser's description picker isn't empty on a fresh
                # channel) -- a real drag-created bad segment always has
                # nonzero width, so this can only ever exclude the
                # placeholder, never a genuine RA-marked segment.
                channel_outputs[ch_key]["bad_segments"].append(
                    [onset_sample, onset_sample + max(duration_samples, 1)]
                )

    if warnings_out is not None:
        # (ch_key, onset_sec, duration_sec) per possible merge, for the
        # session-summary dialog.
        warnings_out.extend(possible_merges)
    if possible_merges:
        print("\n" + "!" * 70)
        print("WARNING: possible merged/lost peak(s) detected. When you click-drag")
        print("to add a peak too close to an existing one, mne silently merges them")
        print("into a single wider marker instead of keeping both -- only one peak")
        print("ends up saved. Please go back and check these spot(s):")
        for ch_key, onset, duration in sorted(possible_merges, key=lambda x: x[1]):
            print(f"  - {ch_key}: around {onset:.2f}s (marker spans {duration * 1000:.0f}ms, "
                  f"wider than a typical single-peak click) -- verify both peaks are really "
                  f"there; if not, delete this marker and re-add the missing peak(s) separately.")
        print("!" * 70 + "\n")

    # Points the RA placed by click-drag were resolved above (the box maximum,
    # climbing only if the box cut off the top: Option C, 2026-09-26). There
    # is no second re-snap: it could pull a short R wave onto a taller MHD wave
    # or artifact within 50 ms. Seeded points the RA left untouched (duration
    # 0) keep their exact sample (audit finding M8). snap_window_samples stays
    # as the matching tolerance other steps use.
    for ch_key, cfg in channel_configs.items():
        if ch_key not in channel_outputs or cfg["annotation_mode"] != "point":
            continue
        channel_outputs[ch_key]["indices"] = sorted(set(int(i) for i in channel_outputs[ch_key]["indices"]))
        channel_outputs[ch_key]["snap_window_samples"] = snap_window_samples

    for ch_key, cfg in channel_configs.items():
        if ch_key in channel_outputs and cfg["annotation_mode"] == "segment":
            channel_outputs[ch_key]["bad_segments"] = sorted(channel_outputs[ch_key]["bad_segments"])

    # Bad stretches never keep peaks (HLU, 2026-09-27): e.g. PPG peaks inside
    # the final bad_ppg stretches are dropped here, so the summary counts them
    # and the saved file never holds them.
    drop_peaks_in_bad_stretches(channel_outputs)
    return channel_outputs


LONG_INTERVAL_FACTOR = 1.8
SHORT_INTERVAL_FACTOR = 0.5
MIN_PEAKS_FOR_INTERVAL_CHECK = 5
"""
check_interval_regularity()'s thresholds: a post-hoc, non-interactive
stand-in for wishlist item 10c's live R-R-interval track. A live-updating
version was investigated and judged too fragile to build -- it would need
to hook mne-qt-browser's undocumented internals (private redraw methods,
no stable public API for it), unlike everything else in this tool, which
sticks to genuinely public mne.Annotations/raw.plot() behavior. This
covers the same "spot a missed or spurious peak" goal by simply checking,
once, whether any gap between exported peaks is a lot longer or shorter
than that channel's own median gap -- exactly the same by-hand check
already run manually a few times against real sub-085 data.

LONG_INTERVAL_FACTOR/SHORT_INTERVAL_FACTOR match the ad-hoc thresholds
used during that manual analysis (a real run's genuinely normal gaps
varied within a much narrower band than either factor). Below
MIN_PEAKS_FOR_INTERVAL_CHECK peaks, a median is too noisy to be
meaningful, so the check is skipped entirely for that channel.
"""


BERNTSON_MED_QD_FACTOR = 3.32
BERNTSON_SEB_QD_FACTOR = 2.9
"""
Berntson, Quigley, Jang, & Boysen (1990), Psychophysiology 27(5), 586-598,
PMID 2274622 -- constants read from pp. 591-592 of the paper (PDF supplied
by the user, 2026-09-23):
    QD  = quartile deviation (interquartile range / 2) of the subject's own
          successive beat-to-beat differences
    MED = 3.32 * QD                       (maximum expected difference, veridical beats)
    MAD = (median beat - 2.9 * QD) / 3    (minimal artifact difference)
    criterion = (MED + MAD) / 2; a successive difference beyond it is flagged.
Their false-alarm routines (Figs. 6-7) are applied in simplified form: a
flagged LONG interval stays flagged only if splitting it in half yields
intervals within criterion of both neighbors (a missed beat); a flagged
SHORT interval only if adding it to its shorter neighbor does (an extra
beat). The authors caution (p. 597) that the criterion is less effective
when MAD < MED (high heart period variability) and was validated only at
artifact rates up to ~8%.
"""


def berntson_criterion(intervals):
    """Returns (criterion, MED, MAD) in the intervals' own units."""
    diffs = np.diff(intervals)
    q75, q25 = np.percentile(diffs, [75, 25])
    qd = (q75 - q25) / 2
    med = BERNTSON_MED_QD_FACTOR * qd
    mad = (float(np.median(intervals)) - BERNTSON_SEB_QD_FACTOR * qd) / 3
    return (med + mad) / 2, med, mad


def berntson_flags(intervals):
    """
    Returns [(i, kind)] for intervals flagged as likely artifacts, where i
    indexes `intervals` and kind is "long (possible missed peak)" or "short
    (possible extra/spurious peak)". Unresolved flags (neither
    classification passes) are reported as "sudden change (check this beat)".
    """
    intervals = np.asarray(intervals, dtype=float)
    n_int = len(intervals)
    if n_int < MIN_PEAKS_FOR_INTERVAL_CHECK:
        return []
    crit, _med, _mad = berntson_criterion(intervals)
    median_beat = float(np.median(intervals))

    def value(i):
        return intervals[i] if 0 <= i < n_int else None

    def fits(x, *others):
        return all(abs(x - o) <= crit for o in others if o is not None)

    flagged = {}
    for n in range(n_int - 1):
        if abs(intervals[n] - intervals[n + 1]) <= crit:
            continue
        # The more deviant of the pair (relative to the median beat) is the target.
        target = n if abs(intervals[n] - median_beat) >= abs(intervals[n + 1] - median_beat) else n + 1
        if target in flagged:
            continue
        b, prev_b, next_b = intervals[target], value(target - 1), value(target + 1)
        if b > median_beat:
            # LONG BEAT (Fig. 6): halves that fit both neighbors = a missed beat.
            if fits(b / 2, prev_b, next_b):
                flagged[target] = "long (possible missed peak)"
            elif not fits(b, prev_b, next_b):
                flagged[target] = "sudden change (check this beat)"
            # else: a long but plausible beat -- a false alarm, not flagged
        else:
            # SHORT BEAT (Fig. 7): add the target to its SHORTER neighbor; a sum
            # that fits the beats on either side of the merged pair = an extra beat.
            candidates = [(prev_b, target - 1), (next_b, target + 1)]
            candidates = [(v, i) for v, i in candidates if v is not None]
            if not candidates:
                continue
            _v, partner = min(candidates)
            first, last = min(target, partner), max(target, partner)
            if fits(b + intervals[partner], value(first - 1), value(last + 1)):
                flagged[target] = "short (possible extra/spurious peak)"
            elif not fits(b, prev_b, next_b):
                flagged[target] = "sudden change (check this beat)"
    return sorted(flagged.items())


def check_interval_regularity(channel_outputs, channel_configs, sfreq):
    """
    Flags any gap between a channel's exported peaks that's unusually long
    (a likely missed peak) or unusually short (a likely extra/spurious
    peak) relative to that channel's OWN median gap, for every channel
    with channel_config.py's check_interval_regularity=True (currently
    ecg/rsp/ppg, not eda -- see that flag's docstring for why). Prints a
    warning naming the channel, approximate time, and the actual gap
    versus the median, so the RA knows exactly where to look. Purely
    informational -- never modifies channel_outputs.
    """
    flagged = []

    for ch_key, cfg in channel_configs.items():
        if not cfg.get("check_interval_regularity"):
            continue
        output = channel_outputs.get(ch_key)
        if not output or output.get("mode") != "point":
            continue

        indices = sorted(output["indices"])
        if len(indices) < MIN_PEAKS_FOR_INTERVAL_CHECK:
            continue

        intervals = np.diff(indices) / sfreq
        median_interval = float(np.median(intervals))
        if median_interval <= 0:
            continue

        # A gap that crosses one of the channel's bad stretches (e.g. bad_ppg)
        # is intentional: it isn't flagged, and the intervals on either side
        # are checked as separate runs, so what's left are the long gaps no
        # bad stretch explains (HLU, 2026-09-27).
        bad = [seg for comp_key, comp_cfg in COMPANION_CHANNELS.items() if comp_cfg["companion_of"] == ch_key
               for seg in (channel_outputs.get(comp_key) or {}).get("bad_segments", [])]
        crosses = [any(s < indices[i + 1] and e > indices[i] for s, e in bad) for i in range(len(intervals))]

        if cfg.get("interval_check") == "berntson":
            # Cardiac channels: the subject-specific successive-difference
            # criterion of Berntson et al. (1990) -- see berntson_flags().
            run = []
            for i in list(range(len(intervals))) + [None]:
                if i is not None and not crosses[i]:
                    run.append(i)
                    continue
                if len(run) >= MIN_PEAKS_FOR_INTERVAL_CHECK - 1:
                    for j, kind in berntson_flags(intervals[run]):
                        k = run[j]
                        flagged.append((ch_key, indices[k] / sfreq, intervals[k], median_interval, kind))
                run = []
            continue

        for i, interval in enumerate(intervals):
            if crosses[i]:
                continue
            onset_sec = indices[i] / sfreq
            if interval > LONG_INTERVAL_FACTOR * median_interval:
                flagged.append((ch_key, onset_sec, interval, median_interval, "long (possible missed peak)"))
            elif interval < SHORT_INTERVAL_FACTOR * median_interval:
                flagged.append((ch_key, onset_sec, interval, median_interval, "short (possible extra/spurious peak)"))

    if flagged:
        # "An ectopic beat" only makes sense for ecg/ppg (cardiac rhythm) --
        # rsp's analogous real-but-benign cause is a breath-hold/sigh, not a
        # beat. Built from only the channel(s) actually flagged this
        # session rather than one hardcoded cardiac-only phrase, so an
        # rsp-only flag doesn't imply a cardiac cause that can't apply.
        cause_examples = {"ecg": "an ectopic beat", "ppg": "an ectopic beat", "rsp": "a breath-hold or sigh"}
        flagged_channel_keys = {ch_key for ch_key, *_ in flagged}
        examples = [cause_examples[key] for key in ("ecg", "ppg", "rsp") if key in flagged_channel_keys]
        examples = list(dict.fromkeys(examples))  # dedupe (ecg+ppg both flagged -> same phrase once)
        examples_str = " / ".join(examples) if examples else "a natural pause"

        print("\n" + "~" * 70)
        print("NOTE: unusually irregular gap(s) between peaks detected. This can be")
        print(f"a real physiological irregularity ({examples_str}) or a missed/spurious")
        print("peak worth a second look:")
        for ch_key, onset, interval, median_interval, kind in sorted(flagged, key=lambda x: x[1]):
            print(f"  - {ch_key}: around {onset:.2f}s, gap is {kind} "
                  f"({interval * 1000:.0f}ms vs. this channel's typical {median_interval * 1000:.0f}ms).")
        print("~" * 70 + "\n")

    return flagged


ANNOTATION_SCHEMA_VERSION = 2
"""
Top-level "schema_version" of the per-RA and reconciled annotation JSON.
Files without it are version 1 (before 2026-09-26). Version 2 adds, per
channel: "seed" (segment channels pre-filled from MachineQC), "edit_history"
(each saved session's added/removed/moved marks), "provenance" (the sample
count, pipeline commit and a hash of the channel's signal), "guide_shown"
(ECG; since 2026-09-27 also SBP/DBP, RSP and EDA), "reviewed_span" (the rows "status" covers: the whole file; added
2026-09-26 without a version change, since absent means the same), and
legacy "<channel>_points_legacy" entries. The "mode"/"indices"/
"bad_segments" keys every reader relies on are unchanged. See README.md,
"The saved annotation file".
"""


COORDINATE_SPACE = ("tsv row index at sampling_rate (row 0 = the file's first row); point indices are rows; "
                    "bad_segments are [start, end)")
"""
Stated in every saved file (physioProcess maintainer's request, 2026-09-26),
the same convention as the sidecar's MachineQC.
"""

STATUS_VALUES = ("complete", "in_progress")
"""
Every reviewed channel's "status": "complete" (the RA or reconciler ticked
"finished") is the only FINAL value; "in_progress" means saved partway.
physioProcess Phase D ingests only "complete" channels of reconciled files.

"complete" covers the channel's "reviewed_span", [0, N) in tsv rows: the
WHOLE file, pads before and after the task included, because the viewer
always shows the whole file (HLU's decision, 2026-09-26, at the physioProcess
maintainer's request). So on a complete channel, no mark anywhere in the span
means "reviewed, clean", never "not reviewed". An entry without
reviewed_span (saved before 2026-09-26) covers the same whole file.
"""


def legacy_channel_key(ch_key, old_mode, existing=()):
    """
    Where an entry in an older review mode is kept, e.g. "rsp_points_legacy".
    A numbered key ("rsp_points_legacy_2") is used when that name is already
    taken in `existing`, so a second older-mode entry is never dropped.
    """
    kind = {"point": "points", "segment": "segments"}.get(old_mode, "unknown")
    key = f"{ch_key}_{kind}_legacy"
    n = 2
    while key in existing:
        key = f"{ch_key}_{kind}_legacy_{n}"
        n += 1
    return key


def keep_mode_changed_entries(existing_channels, new_channels, merged_channels, also=()):
    """
    For each channel whose review MODE changed between the existing file and
    this save (RSP: breath peaks until 2026-09-25, bad segments since), keeps
    the old entry under a legacy key in merged_channels instead of losing it
    (sidecar plan Phase 1). Returns [(ch_key, legacy_key, old_mode)] kept.
    Used by both save_annotation_json() and reconcile.save_reconciled_json().
    also: channel keys whose old entry is kept the same way even though the
    mode is unchanged -- e.g. a PPG review with no provenance, made on the
    pre-fix PPG (2026-09-26 safety fix).
    """
    kept = []
    for ch_key, new_output in new_channels.items():
        old_output = existing_channels.get(ch_key)
        if not old_output or (new_output.get("mode") == old_output.get("mode") and ch_key not in also):
            continue
        if any(merged_channels.get(k) == old_output for k in merged_channels if k.startswith(f"{ch_key}_")):
            continue  # already kept under a legacy key
        legacy_key = legacy_channel_key(ch_key, old_output.get("mode"), existing=merged_channels)
        merged_channels[legacy_key] = old_output
        kept.append((ch_key, legacy_key, old_output.get("mode")))
    return kept


def save_annotation_json(output_dir, file_stem, initials, channel_outputs, sfreq, source_file, retire=()):
    """
    Writes <file_stem>_annotations_<initials>.json. Deliberately does NOT
    overwrite another RA's file for the same run -- each RA's initials
    produce a separate file so multiple people can independently review the
    same recording.

    MERGES with any existing save for this RA/run rather than replacing it
    wholesale: channels reviewed in THIS session overwrite their own entry,
    but any channel saved in an EARLIER session (e.g. --channels ecg today,
    --channels rsp tomorrow) is carried forward untouched. Confirmed as a
    real bug before this fix -- reviewing a different --channels subset in
    a later session silently erased whatever had been saved for any
    channel not in that session's subset, live-tested and reproduced
    directly (see HANDOFF.md's debugging history).

    Before writing, copies any existing file to a timestamped "backups"
    subfolder first. This is a pure safety net against silent data loss on
    re-save (e.g. if a resumed session ever failed to seed correctly, or
    an RA accidentally closed without the edits they expected) -- it costs
    nothing when everything's working, and means a bad save is always
    recoverable from out_dir/backups/ instead of being gone the moment a
    new save completes.

    retire: channels whose existing entry is kept under a legacy key rather
    than replaced (see keep_mode_changed_entries(also=)). If the existing
    file can't be read, this session's channels are written to a separate
    *_RESCUED_<stamp>.json and the session stops with a plain message, so no
    work is lost and the damaged file isn't overwritten (2026-09-26).
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{file_stem}_annotations_{initials}.json")

    merged_channels = dict(channel_outputs)

    if os.path.exists(out_path):
        backup_dir = os.path.join(output_dir, "backups")
        os.makedirs(backup_dir, exist_ok=True)
        backup_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"{file_stem}_annotations_{initials}_{backup_stamp}.json")
        shutil.copy2(out_path, backup_path)

        try:
            existing_channels = saved_channel_entries(read_saved_json(out_path), out_path)
        except SystemExit as problem:
            rescue = os.path.join(output_dir, f"{file_stem}_annotations_{initials}_RESCUED_{backup_stamp}.json")
            with open(rescue, "w") as f:
                json.dump({"schema_version": ANNOTATION_SCHEMA_VERSION, "coordinate_space": COORDINATE_SPACE,
                           "initials": initials, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           "source_file": source_file, "sampling_rate": sfreq, "channels": channel_outputs},
                          f, indent=2)
            raise SystemExit(f"{problem} This session's work was saved separately to {rescue}; give both files "
                             f"to the lab staff.")
        carried_forward = [ch for ch in existing_channels if ch not in channel_outputs]
        merged_channels = {**existing_channels, **channel_outputs}
        # A channel whose review MODE changed keeps its earlier review under a
        # legacy key rather than losing it (sidecar plan Phase 1).
        for ch_key, legacy_key, old_mode in keep_mode_changed_entries(existing_channels, channel_outputs,
                                                                      merged_channels, also=retire):
            print(f"  Your earlier {ch_key} review (marked as {old_mode or 'unknown'}s) is kept in the file "
                  f"as '{legacy_key}'.")

        print(f"NOTE: {out_path} already exists and will be updated "
              f"(re-saving the same RA's session for this run). Previous "
              f"version backed up to {backup_path}.")
        if carried_forward:
            print(f"  Carrying forward previously-saved channel(s) not reviewed this "
                  f"session (unchanged): {', '.join(carried_forward)}.")

    payload = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "coordinate_space": COORDINATE_SPACE,
        "initials": initials,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_file": source_file,
        "sampling_rate": sfreq,
        "channels": merged_channels,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved annotations to {out_path}")
    return out_path


def split_saved_by_mode(saved_channels, channel_configs):
    """
    (usable, mode_changed): saved channel entries whose mode matches the
    channel's current annotation_mode, and the keys of those that don't
    (e.g. an RSP review saved when RSP was a point channel). A mode-changed
    entry must not seed the session; the channel starts fresh, and
    save_annotation_json() keeps the old entry under a legacy key.
    """
    usable, mode_changed = {}, []
    for ch_key, entry in (saved_channels or {}).items():
        cfg = channel_configs.get(ch_key)
        if cfg and cfg["annotation_mode"] in ("point", "segment") and entry.get("mode") != cfg["annotation_mode"]:
            mode_changed.append(ch_key)
        else:
            usable[ch_key] = entry
    return usable, mode_changed


def _known_labels(channel_configs):
    return {cfg[key] for cfg in channel_configs.values() for key in ("point_label", "segment_label") if cfg.get(key)}


def session_checks_before_viewer(channel_configs, sidecar, allow_old_ppg, df=None):
    """
    Refusals that must happen before any viewer opens, shared by Step 2 and
    Step 3 (ppg-plan §4b): PPG is reviewed on its own and only on files with
    the new CareTaker timing (lab staff can override that); and no channel is
    reviewed whose column is empty -- runs with no CareTaker pulse or vitals,
    or ones that don't cover the task, have all-NaN ppg/sbp/dbp columns
    (maintainer's replies, 2026-09-26).
    """
    reviewed = [k for k, cfg in channel_configs.items()
                if cfg["annotation_mode"] in ("point", "segment") and not cfg.get("companion_of")]
    check_review_alone(reviewed)
    # Any reviewed channel whose column is empty (all NaN) is refused, not
    # only PPG: on the 7 runs with no CareTaker vitals, SBP/DBP used to open
    # with the whole run pre-flagged and a "sensor may have been disconnected"
    # warning, and saving counted it as reviewed (run-8 decisions, 2026-09-26).
    # The entry points' all-channels defaults drop such channels with a note
    # (channel_config.present_channels()); this is the backstop.
    blank = empty_channels(df, reviewed) if df is not None else []
    if blank:
        raise SystemExit(no_data_message(blank))
    if "ppg" in reviewed and not allow_old_ppg and not new_pipeline_ppg(sidecar):
        raise SystemExit("This file was processed before the PPG timing fix (its sidecar has no CareTaker timing "
                         "record), so its PPG can't be reviewed yet. Please tell the lab staff (lab staff: "
                         "--allow-old-ppg overrides this).")


def check_saved_provenance(saved_channels, current_provenance, channel_configs, who="Your saved"):
    """
    Stops the session if a saved channel was made on a different signal than
    the current file's (same sample count and content hash required;
    ppg-plan §4c). Entries saved before 2026-09-26 carry no provenance and
    pass (check_reprocess_alignment.py covers those).
    """
    for ch_key, entry in saved_channels.items():
        if ch_key not in channel_configs:
            continue
        reason = provenance_mismatch(entry.get("provenance"), current_provenance.get(ch_key))
        if reason:
            raise SystemExit(f"{who} {channel_configs[ch_key]['label']} review for this run was made on a different "
                             f"version of the file: {reason}. Nothing has been changed. Please tell the lab staff "
                             f"before continuing.")


PPG_MASK_REASONS = ("zero_run", "bridged", "window_rule")
"""
The per-sample PPG mask's reasons in MachineQC.ppg.SegmentsByReason
(physioProcess R1): zero_run = 8+ consecutive native zeros, widened 0.6 s each
side; bridged = rows more than 17.6 ms from the nearest native sample;
window_rule = the 10-s pulsatility/periodicity rule.
"""


def ppg_invalid_spans(sidecar, n_samples):
    """
    (spans, note) for PPG's machine-invalid stretches (MachineQC.ppg; device
    zero runs and bridged samples once physioProcess writes its per-sample
    mask). The note says when the dropout check isn't available, so "no
    dropout lines" can't be mistaken for "no dropouts" (checkpoint-2 review).
    """
    if sidecar is None:
        return [], None
    qc, note = machine_qc_segments(sidecar, n_samples, only=["ppg"], labels={"ppg": "PPG"})
    entry = qc.get("ppg")
    if entry is None:
        return [], note or "This file's machine check has no PPG entry, so PPG dropouts aren't flagged for you."
    if entry.get("by_reason_invalid"):
        return entry["segments"], ("This file's per-sample PPG dropout mask can't be read, so dropouts aren't "
                                   "flagged for you: look for low, smooth arcs with no clear pulse yourself. Please "
                                   "tell the lab staff.")
    by_reason = entry.get("by_reason") or {}
    # The per-sample mask is present when physioProcess wrote its three
    # reasons (each possibly empty). A run with no usable pulse instead has
    # {"no_pulse_data": [[0, N]]} (maintainer's reply, 2026-09-26).
    if "no_pulse_data" in by_reason:
        note = "The machine check says this run has no usable CareTaker pulse data."
    elif not any(k in by_reason for k in PPG_MASK_REASONS):
        note = ("This file has no per-sample PPG dropout mask yet (physioProcess request R1), so dropouts aren't "
                "flagged for you: look for low, smooth arcs with no clear pulse yourself.")
    return entry["segments"], note


def print_sidecar_banner(sidecar, channel_configs):
    """Unit labels (sidecar plan Phase 3) and each reviewed channel's Description, from the new sidecar."""
    if not sidecar:
        return
    units, unit_warnings = sidecar_unit_labels(sidecar, channel_configs)
    apply_unit_labels(units)
    for line in unit_warnings:
        print(f"NOTE: {line}")
    reviewed = [k for k, cfg in channel_configs.items() if cfg["annotation_mode"] in ("point", "segment")]
    descriptions = sidecar_descriptions(sidecar, reviewed)
    if descriptions:
        print(" - About the channel(s), from the file:")
        for ch_key, text in descriptions.items():
            print(f"     {channel_configs[ch_key]['label']}: {text}")
            if ch_key == "ppg" and "TechnicalDescription" not in sidecar["ppg"]:
                # An older, technical Description says zero runs are kept and
                # gaps are held, but the column is filtered, so neither shows as
                # a flat line (run-8 review). From run 8 the Description is
                # written for RAs (the technical text moved to
                # TechnicalDescription), so this line isn't needed there. Plain
                # words: HLU finds processing jargon confusing for RAs.
                print("       (The pulse is smoothed, so where the CareTaker lost the signal you'll see slow, "
                      "smooth curves, never a flat line.)")


def _started_from(ch_key, cfg, saved_channels, seed_records, seed_label):
    """What a channel's session started from, for its edit history."""
    if ch_key in saved_channels:
        return "saved review"
    if ch_key in seed_records:
        return "MachineQC"
    if ch_key == "ecg" and seed_label:
        return seed_label
    if cfg.get("auto_peaks_col"):
        return f"{cfg['auto_peaks_col']} column"
    return "blank"


def run_stage_b(df, sfreq, channel_configs, initials, out_dir, file_stem, source_file, sidecar=None,
                ppg_guide=True, tsv_path=None, allow_old_ppg=False, seed_label=None, ecg_ppg_guide=False):
    """
    The full interactive Stage B session: builds the multi-channel Raw,
    seeds automated peaks, launches the annotation viewer, then shows the
    session-summary dialog (save / discard / go back; comment; finished)
    and acts on it. Shared by physio_annotate.py (standalone) and
    physio_review.py (unified script). Returns a SessionResult.

    sidecar: the run's sidecar dict (physio_io.read_sidecar()), or None
    (synthetic data). Supplies the MachineQC bad segments that pre-fill fresh
    segment channels (sidecar plan Phase 2), unit labels and descriptions
    (Phase 3), the new-pipeline check for PPG, and PPG's machine-invalid
    stretches (ppg-plan §4b).
    ppg_guide: show the read-only guide rows (default on): PPG under SBP/DBP,
    EDA under RSP, RSP under EDA (HLU, 2026-09-27).
    ecg_ppg_guide: also show the PPG under ECG (Phase 1b), which is opt-in
    since 2026-10-03 (HLU; channel_config.GUIDE_OPT_IN).
    tsv_path: the run's *_physio.tsv.gz, for each reviewed channel's
    provenance (sample count and content hash; ppg-plan §4c). None for
    synthetic data.
    allow_old_ppg: lab-staff override of the new-pipeline check for PPG.
    seed_label: what ECG was seeded from (e.g. the QRS template), recorded in
    its edit history.
    """
    session_checks_before_viewer(channel_configs, sidecar, allow_old_ppg, df=df)
    channel_configs = with_companions(channel_configs)  # e.g. bad_ppg with PPG (HLU, 2026-09-27)
    full_channel_configs = combine_with_reference_channels(channel_configs, df, ppg_guide=ppg_guide, sidecar=sidecar,
                                                           ecg_ppg_guide=ecg_ppg_guide)
    reference_channels_shown = [k for k in REFERENCE_CHANNELS if k in full_channel_configs]
    guide_shown = guide_shown_by_channel(channel_configs, full_channel_configs)

    raw = build_raw(df, full_channel_configs, sfreq)
    reviewed_keys = [k for k, cfg in full_channel_configs.items() if cfg["annotation_mode"] in ("point", "segment")]
    current_provenance = companion_provenance(channel_provenance(tsv_path, sidecar, reviewed_keys, len(df)))

    saved = load_saved_annotations(out_dir, file_stem, initials, sfreq=sfreq, n_samples=len(df))
    saved_channels = {}
    retired = []
    if saved is not None:
        all_saved, saved_path = saved
        saved_channels, mode_changed = split_saved_by_mode(all_saved, full_channel_configs)
        check_saved_provenance(saved_channels, current_provenance, full_channel_configs)
        # A saved PPG review with no provenance was made before 2026-09-26, on
        # the pre-fix PPG timing: never seed it onto this file's PPG. PPG starts
        # fresh and the old entry is kept under a legacy key (safety fix).
        if "ppg" in reviewed_keys and current_provenance.get("ppg") and "ppg" in saved_channels \
                and not saved_channels["ppg"].get("provenance"):
            saved_channels.pop("ppg")
            retired.append("ppg")
            print("NOTE: your earlier PPG review on this run was saved before 2026-09-26, without a record of the "
                  "signal it was made on (PPG was then on the old timing), so it can't be used on this file. PPG "
                  "starts fresh; the earlier review is kept in the file under a name ending in '_legacy' when "
                  "you save.")
        # Only count channels actually relevant to (and seeded into) THIS
        # session -- previously this summed ALL saved channels regardless
        # of whether they were being reviewed now, misreporting e.g. old
        # ecg counts while resuming an rsp-only session that never touches
        # ecg at all. Found alongside the merge-on-save fix above.
        # Reviewed channels only: a guide row shares its channel's key (e.g.
        # "ppg" under ECG) but is never seeded (checkpoint-2 review).
        relevant_saved = {k: v for k, v in saved_channels.items() if k in reviewed_keys}
        n_points = sum(len(v.get("indices", [])) for v in relevant_saved.values() if v.get("mode") == "point")
        n_segments = sum(len(v.get("bad_segments", [])) for v in relevant_saved.values() if v.get("mode") == "segment")
        print(f"\nResuming your previous session on this run: {n_points} point annotation(s) and "
              f"{n_segments} bad segment(s) restored from {os.path.basename(saved_path)}.")
        for ch_key in mode_changed:
            label = full_channel_configs[ch_key]["label"]
            mode = full_channel_configs[ch_key]["annotation_mode"]
            old_mode = all_saved[ch_key].get("mode") or "unknown"
            print(f"NOTE: your earlier {label} review marked {old_mode}s, but {label} is now "
                  f"reviewed by marking {'bad stretches' if mode == 'segment' else 'peaks'}, so {label} starts "
                  f"fresh this session. Your earlier review is kept in the file under a name ending in "
                  f"'_legacy' when you save.")

    # Fresh segment channels start from the pipeline's MachineQC bad
    # segments where the sidecar has them (sidecar plan Phase 2).
    fresh_segment = [k for k, cfg in full_channel_configs.items()
                     if cfg["annotation_mode"] == "segment" and is_present(k, cfg, df.columns)
                     and k not in saved_channels]
    machine_qc, qc_note = {}, None
    labels_by_key = {k: cfg["label"] for k, cfg in full_channel_configs.items()}
    if sidecar is not None and fresh_segment:
        # A companion reads its parent's machine check (bad_ppg <- MachineQC.ppg).
        source = {k: full_channel_configs[k].get("machine_qc_source", k) for k in fresh_segment}
        read, qc_note = machine_qc_segments(sidecar, len(df), only=set(source.values()), labels=labels_by_key)
        machine_qc = {k: read[src] for k, src in source.items() if src in read}
        if qc_note and not any(full_channel_configs[k].get("machine_qc") for k in fresh_segment):
            qc_note = None  # e.g. an EMG-only session: the machine never checks EMG anyway
    mostly_flagged = [k for k, v in machine_qc.items()
                      if machine_qc_flagged_fraction(v, len(df)) > MACHINE_QC_FLAGGED_WARNING_FRACTION]
    commit = ((sidecar or {}).get("Provenance") or {}).get("PipelineCommit")
    # The machine's reasons, when the sidecar gives them (SBP/DBP from run 8,
    # decision B1), go into the seed record and the summary's note too.
    seed_records = {k: {"source": "MachineQC", "rule": v["rule"], "segments": v["segments"],
                        "pipeline_commit": commit,
                        **({"by_reason": v["by_reason"]} if v.get("by_reason") is not None else {})}
                    for k, v in machine_qc.items()}
    seed_notes = {k: f"started from {len(v['segments'])} machine-flagged stretch(es)"
                     + (f": {reason_summary(v.get('by_reason'))}" if reason_summary(v.get("by_reason")) else "")
                  for k, v in machine_qc.items()}
    # PPG's machine-invalid stretches (device zero runs, bridged samples) for
    # the PPG check lines (ppg-plan §4b). PPG is a point channel, so they are
    # never pre-filled.
    ppg_invalid, ppg_note = ppg_invalid_spans(sidecar, len(df)) if "ppg" in reviewed_keys else ([], None)
    # Bad stretches never keep peaks: a fresh parent's automated peaks inside
    # its companion's pre-marked stretches (saved, or the machine's) aren't
    # loaded; the edit history records them (HLU, 2026-09-27).
    exclude_seeds, seeds_left_out = {}, {}
    for comp_key, comp_cfg in COMPANION_CHANNELS.items():
        parent = comp_cfg["companion_of"]
        if comp_key not in full_channel_configs or parent in saved_channels:
            continue
        spans = (saved_channels.get(comp_key) or {}).get("bad_segments") if comp_key in saved_channels \
            else (machine_qc.get(comp_key) or {}).get("segments")
        auto_col = full_channel_configs[parent].get("auto_peaks_col")
        if spans and auto_col in df.columns:
            exclude_seeds[parent] = spans
            seeds_left_out[parent] = [int(i) for i in np.flatnonzero(df[auto_col].to_numpy()) if _inside(int(i), spans)]

    raw = seed_annotations(raw, df, full_channel_configs, saved_channels, sfreq, machine_qc=machine_qc,
                           exclude_seeds=exclude_seeds)

    scalings = compute_scalings(df, full_channel_configs)
    events_array = compute_event_transitions(df, "event", sfreq)

    print("\n" + "=" * 70)
    print(f"Launching interactive annotation viewer with channels: {list(raw.ch_names)}")
    print_sidecar_banner(sidecar, full_channel_configs)
    # Per channel, not per session: a resumed session can still have a
    # channel that's freshly auto-seeded this time (e.g. rsp added to
    # --channels for the first time on a run whose only prior save was
    # ecg-only) -- see seed_annotations()'s docstring for the bug this
    # replaced, where such a channel silently got no message AND no seed.
    freshly_seeded_point = [cfg["label"] for ch_key, cfg in full_channel_configs.items()
                            if cfg["annotation_mode"] == "point" and ch_key not in saved_channels]
    freshly_seeded_segment = [cfg["label"] for ch_key, cfg in full_channel_configs.items()
                              if cfg["annotation_mode"] == "segment" and ch_key not in saved_channels]
    if freshly_seeded_point:
        print(f" - {'/'.join(freshly_seeded_point)} start pre-marked with the automated peak detections.")
        print("   Use annotation mode to add missed peaks or delete false ones.")
        print("   (A peak you add is saved at the highest point inside the box you drag; if the box cuts")
        print("   off the top of the wave, it moves up to the top, at most 15 ms. Draw the box over the peak.)")
    if channel_configs.get("ppg", {}).get("annotation_mode") == "point":
        print(PPG_ADD_PEAK_TEXT)
    prefilled = [full_channel_configs[k]["label"] for k in machine_qc if machine_qc[k]["segments"]]
    blank = [label for label in freshly_seeded_segment if label not in prefilled]
    if prefilled:
        print(f" - {'/'.join(prefilled)} start with the machine's bad stretches pre-marked (its automatic "
              f"quality check). Check each one: delete any that are fine, drag to add any it missed.")
        units_by_key = {k: mne.defaults.DEFAULTS["units"].get(cfg["mne_type"], "")
                        for k, cfg in full_channel_configs.items()}
        for line in machine_qc_notes(machine_qc, list(machine_qc), len(df), sfreq, labels_by_key,
                                     value_ranges=compute_channel_value_ranges(df, full_channel_configs),
                                     units=units_by_key):
            print(f"     {line}")
    if blank:
        print(f" - {'/'.join(blank)} start blank. Drag to mark bad/artifact spans.")
    if qc_note:
        print(f" - NOTE: {qc_note}")
    if ppg_note:
        print(f" - NOTE: {ppg_note}")
    for comp_key, comp_cfg in COMPANION_CHANNELS.items():
        if comp_key not in full_channel_configs:
            continue
        left = len(seeds_left_out.get(comp_cfg["companion_of"], []))
        print(f" - {comp_cfg['label']} (label '{comp_cfg['segment_label']}'): {comp_cfg['banner']}; peaks inside "
              f"them are left out when you save"
              + (f" ({left} automated peak(s) inside the pre-marked stretches weren't loaded)." if left else "."))
    for line in guide_banner_lines(channel_configs, df, ppg_guide, sidecar, ecg_ppg_guide):
        print(f" - {line}")
    value_range_lines = format_channel_value_ranges(
        compute_channel_value_ranges(df, full_channel_configs), full_channel_configs
    )
    if value_range_lines:
        print(" - Real value ranges:")
        for line in value_range_lines:
            print(f"     {line}")
    if reference_channels_shown:
        print(f" - {', '.join(reference_channels_shown)}: read-only reference track(s), shown for "
              f"context only -- not something to correct. 'event'==0 means outside the task.")
        if events_array is not None:
            print(f"   {len(events_array)} labeled event marker(s) show the actual code at each "
                  f"change (press 'e' to toggle them off if they're in the way).")
    print(" - Close the plot window when you're done. A summary will then let you save, discard this")
    print("   session's changes, or go back to the viewer.")
    print("=" * 70 + "\n")

    plot_kwargs = {"events": events_array} if events_array is not None else {}

    labels = ", ".join(full_channel_configs[k]["label"] for k in reviewed_keys)
    # Subject/run, channel(s) and reviewer in the window title, so a wrong-file
    # session is visible while reviewing (audit finding M6).
    window_title = f"{file_stem} · {labels} · reviewer {initials}"
    _warn_if_inverted(df, full_channel_configs)

    def show_viewer():
        # order=<declared order>, not MNE's default (group_by="type", which sorts by
        # a FIXED internal channel-type priority list, e.g. "gsr" sorts after
        # "stim" while "ecg"/"resp" sort before it -- so eda/ppg/finger_temperature
        # would render BELOW the read-only "event" track while ecg/rsp/emg/sbp/dbp
        # render above it, with no relation to this tool's own channel order or to
        # point-vs-segment annotation mode. Confirmed live -- the user noticed eda
        # below event, ecg/rsp above; see HANDOFF.md.
        raw.plot(block=True, scalings=scalings, duration=20, show=True, title=window_title,
                  order=np.arange(len(raw.ch_names)), **plot_kwargs)

    previous_status = {k: saved_channels.get(k, {}).get("status") for k in reviewed_keys}
    value_ranges = compute_channel_value_ranges(df, full_channel_configs)
    tolerance = round(DEFAULT_SNAP_WINDOW_SEC * sfreq)
    with contextlib.redirect_stdout(io.StringIO()):
        session_baseline = export_annotations(raw, df, full_channel_configs, sfreq)

    def save(outputs):
        saved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for ch_key in outputs:
            previous = saved_channels.get(ch_key, {})
            # Each segment-mode channel's real min/mean/max, stored as numbers
            # for downstream scripts (audit §9.2); the log keeps a text copy.
            if ch_key in value_ranges:
                lo, mean, hi = value_ranges[ch_key]
                unit = mne.defaults.DEFAULTS["units"].get(full_channel_configs[ch_key]["mne_type"], "")
                outputs[ch_key]["value_range"] = {"min": lo, "mean": mean, "max": hi, "unit": unit}
            # What each segment channel started from, so RA-versus-machine
            # agreement can be computed later (sidecar plan Phase 2). A resumed
            # channel keeps the record from the session that pre-filled it.
            if ch_key in seed_records:
                outputs[ch_key]["seed"] = seed_records[ch_key]
            elif "seed" in previous:
                outputs[ch_key]["seed"] = previous["seed"]
            # Which signal this was reviewed on: physioProcess Phase D binds
            # annotations to the sample count plus this content hash (§4c, R7).
            if ch_key in current_provenance:
                outputs[ch_key]["provenance"] = current_provenance[ch_key]
            # What "status" covers: the whole file (STATUS_VALUES).
            outputs[ch_key]["reviewed_span"] = [0, len(df)]
            # Per-session edit history: what the RA changed relative to what
            # the session started from, so PAT/PRV can later be re-run
            # excluding RA-added or moved beats (§4c).
            record = session_summary_gui.edit_record(session_baseline.get(ch_key), outputs[ch_key], tolerance)
            record.update(saved_at=saved_at, initials=initials,
                          started_from=_started_from(ch_key, full_channel_configs[ch_key], saved_channels,
                                                     seed_records, seed_label))
            # Whether a guide row was on screen: for ECG the pulse could have
            # shaped which R peaks were marked, so PAT analyses can exclude
            # guided edits (§4c, reverse circularity); likewise for SBP/DBP,
            # RSP and EDA (2026-09-27). Recorded per session in the edit
            # history, and the channel-level flag is true if ANY saved session
            # had the guide (checkpoint-2 review: it used to keep only the
            # latest session's setting).
            if ch_key in guide_shown:
                record["guide_shown"] = guide_shown[ch_key]
                outputs[ch_key]["guide_shown"] = bool(guide_shown[ch_key] or previous.get("guide_shown"))
            # Peaks left out or dropped because they're inside a bad stretch (bad_ppg).
            for comp_key, comp_cfg in COMPANION_CHANNELS.items():
                if comp_cfg["companion_of"] == ch_key and comp_key in outputs:
                    segments = outputs[comp_key].get("bad_segments") or []
                    record["removed_in_bad_stretches"] = [i for i in record.get("removed") or [] if _inside(i, segments)]
                    if seeds_left_out.get(ch_key):
                        record["seeds_left_out_in_bad_stretches"] = seeds_left_out[ch_key]
            outputs[ch_key]["edit_history"] = list(previous.get("edit_history", [])) + [record]
        return save_annotation_json(out_dir, file_stem, initials, outputs, sfreq, source_file, retire=retired)

    known_labels = _known_labels(full_channel_configs)

    def step2_checks(r, outputs):
        lines = unknown_label_lines(r, known_labels)
        # Repeat the "machine flagged nearly all of this run" warning in the
        # summary for as long as the RA has removed most of that flag, so a
        # dead sensor isn't saved as usable without a second look
        # (checkpoint-1 review).
        for ch_key in mostly_flagged:
            kept = sum(e - s for s, e in outputs.get(ch_key, {}).get("bad_segments", [])) / max(len(df), 1)
            if kept >= MACHINE_QC_FLAGGED_WARNING_FRACTION:
                continue
            if coverage_only(machine_qc[ch_key]):
                lines.append(f"{labels_by_key[ch_key]}: you've removed most of the machine's flags, but those "
                             f"stretches have no CareTaker reading at all (they show blank). Keep them flagged.")
            else:
                lines.append(f"{labels_by_key[ch_key]}: the machine flagged nearly all of this run as bad (a "
                             f"disconnected sensor?), and you've removed most of that flag. Save only if you're "
                             f"sure the signal is real.")
        if outputs.get("ppg", {}).get("mode") == "point" and "ppg" in df.columns:
            indices = outputs["ppg"]["indices"]
            # Peaks inside the final bad_ppg stretches are already dropped, so
            # the "inside a dropout" check uses those stretches.
            invalid = (outputs.get("bad_ppg") or {}).get("bad_segments", ppg_invalid) if "bad_ppg" in outputs \
                else ppg_invalid
            lines += ppg_check_lines(indices, df["ppg"].to_numpy(), sfreq, invalid)
            lines += same_beat_lines(indices, sfreq)
        # Peaks on screen inside a bad stretch (bad_ppg, bad_ecg) are dropped on save.
        for comp_key, comp_cfg in COMPANION_CHANNELS.items():
            parent = comp_cfg["companion_of"]
            if comp_key not in outputs or parent not in full_channel_configs:
                continue
            dropped = _peaks_in_bad_on_screen(r, outputs, sfreq, label=full_channel_configs[parent]["point_label"],
                                              companion=comp_key)
            if dropped:
                lines.append(f"{full_channel_configs[parent]['label']}: {dropped} peak(s) inside your bad stretches "
                             f"will be left out when you save (bad stretches never keep peaks).")
        return lines

    decision, channel_outputs, n_changes = review_until_decided(
        raw,
        show_viewer,
        lambda r, warnings_out: export_annotations(r, df, full_channel_configs, sfreq, warnings_out=warnings_out),
        lambda outputs: check_interval_regularity(outputs, full_channel_configs, sfreq),
        full_channel_configs,
        sfreq,
        heading=window_title,
        finished_default=bool(reviewed_keys) and all(previous_status[k] == STATUS_COMPLETE for k in reviewed_keys),
        extra_warnings=step2_checks,
        seed_notes=seed_notes,
    )
    return finish_session(decision, channel_outputs, n_changes, previous_status, saved is not None, save,
                          existing_path=os.path.join(out_dir, f"{file_stem}_annotations_{initials}.json"))


def _peaks_in_bad_on_screen(raw, outputs, sfreq, label="peak_ppg", companion="bad_ppg"):
    """How many visible peak marks (label) sit inside the current companion's bad stretches (dropped on save)."""
    segments = (outputs.get(companion) or {}).get("bad_segments") or []
    if not segments:
        return 0
    return sum(1 for ann in raw.annotations
               if ann["description"] == label and _inside(int(round(ann["onset"] * sfreq)), segments))


def _warn_if_inverted(df, channel_configs):
    """
    Peaks are snapped to local MAXIMA, which assumes an upright waveform. If
    the seeded ECG peaks sit mostly BELOW the signal's median, the recording
    is probably inverted (lead placement) and snapping would misplace them
    (audit finding M8) -- say so before the RA starts.
    """
    for ch_key, cfg in channel_configs.items():
        if ch_key != "ecg" or ch_key not in df.columns or cfg.get("auto_peaks_col") not in df.columns:
            continue
        peaks = np.where(df[cfg["auto_peaks_col"]].to_numpy())[0]
        if len(peaks) < MIN_PEAKS_FOR_INTERVAL_CHECK:
            continue
        signal = df[ch_key].to_numpy(dtype=float)
        if np.nanmedian(signal[peaks]) < np.nanmedian(signal):
            print("WARNING: the ECG peaks sit below the signal's typical level, so this ECG may be "
                  "upside down. Peaks you add are moved to the nearest MAXIMUM, which would be the wrong "
                  "point on an inverted ECG -- please tell the lab staff before reviewing this file.")


STATUS_COMPLETE = "complete"
STATUS_IN_PROGRESS = "in_progress"


@dataclass
class SessionResult:
    """What a Step 2/Step 3 session ended with; returned by run_stage_b()/run_stage_c()."""
    saved_path: Optional[str]
    saved: bool
    discarded: bool
    comment: str = ""
    finished: bool = False
    agreement_text: str = ""  # Step 3 only: the two reviewers' agreement, for the log


MAX_GAP_LINES_PER_CHANNEL = 5
"""
Above this many unusual-gap flags on one channel, the session summary shows
a single count line for that channel instead (the full list is still printed
in the terminal). The dialog shows only 12 warning lines, and PPG alone
gave 49 Berntson flags on sub-001, which pushed every other line out of view
(ppg-plan §3.1 #5).
"""


def _format_warning_lines(merges, flagged, channel_configs, extra_lines=()):
    """
    Session-summary warning lines, most actionable first: possible merges
    (lost peaks), then channel-specific check lines (extra_lines, e.g. PPG
    peaks inside a machine-flagged dropout, or labels that won't be saved),
    then unusual gaps -- listed individually up to MAX_GAP_LINES_PER_CHANNEL,
    else one count line per channel.
    """
    def label(ch_key):
        return channel_configs.get(ch_key, {}).get("label", ch_key)
    lines = [f"{label(ch)}: possible merged/lost peak near {onset:.1f} s"
             for ch, onset, _duration in sorted(merges, key=lambda m: m[1])]
    lines += list(extra_lines)
    by_channel = {}
    for flag in sorted(flagged, key=lambda f: f[1]):
        by_channel.setdefault(flag[0], []).append(flag)
    for ch, flags in by_channel.items():
        if len(flags) > MAX_GAP_LINES_PER_CHANNEL:
            lines.append(f"{label(ch)}: {len(flags)} unusual gaps between peaks, the first near "
                         f"{flags[0][1]:.1f} s (all listed in the terminal)")
            continue
        lines += [f"{label(ch)}: unusual gap near {onset:.1f} s ({interval * 1000:.0f} ms vs. typical "
                  f"{median * 1000:.0f} ms)"
                  for ch, onset, interval, median, _kind in flags]
    return lines


def unknown_label_lines(raw, known_labels):
    """
    One warning line per annotation label that no reviewed channel uses, e.g.
    a typo in the viewer's "Add description" box. Those marks are dropped at
    export; until 2026-09-26 they were dropped without a word
    (ppg-plan §3.1 #6).
    """
    counts = {}
    for description in raw.annotations.description:
        if description not in known_labels:
            counts[description] = counts.get(description, 0) + 1
    return [f"{count} mark(s) labeled '{description}' will NOT be saved (not a label this session uses) -- "
            f"relabel or delete them" for description, count in sorted(counts.items())]


def _dialog_log_lines(heading, change_lines, warning_lines, summary_kwargs, decision):
    """The session-summary dialog as text for the session log: what it showed, and what the RA chose."""
    choice = {session_summary_gui.SAVE: "Save", session_summary_gui.DISCARD: "Discard",
              session_summary_gui.BACK: "Go back to the viewer"}.get(decision.action, str(decision.action))
    lines = ["", "--- Session summary (shown in the dialog) ---", heading]
    lines += [f"  {line}" for line in summary_kwargs.get("seed_lines", [])]
    lines += [f"  {line}" for line in change_lines] or ["  (no changes)"]
    lines += [f"  WARNING: {line}" for line in warning_lines]
    lines += [f"  MUST FIX BEFORE SAVING: {line}" for line in summary_kwargs.get("blocking_lines", [])]
    lines.append(f"Choice: {choice}; {'marked finished' if decision.finished else 'not marked finished'}"
                 + (f"; comment: {decision.comment}" if decision.comment else ""))
    lines.append("---")
    return lines


def review_until_decided(raw, show_viewer, export, check_intervals, channel_configs, sfreq, heading,
                         finished_default=False,
                         tracker_hint="Tick this only when you've finished. Then set this channel to \"finished\" "
                                      "in the Google tracking spreadsheet (the terminal will remind you).",
                         extra_warnings=None, blocking_warnings=None, seed_notes=None):
    """
    Shared Step 2/Step 3 loop: show the viewer, export what's there, show the
    session-summary dialog (session_summary_gui), and repeat while the RA
    chooses "Go back to the viewer". Returns (decision, channel_outputs,
    n_changes). `export(raw, warnings_out)` must return export_annotations()-
    shaped outputs; `check_intervals(outputs)` returns
    check_interval_regularity()-shaped flags. Optional callables, each
    `(raw, outputs) -> [lines]`: extra_warnings adds check lines; any line
    from blocking_warnings disables Save until it's fixed. seed_notes
    ({ch_key: text}) says what each channel started from.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        baseline = export(raw, None)
    tolerance = round(DEFAULT_SNAP_WINDOW_SEC * sfreq)
    while True:
        show_viewer()
        merges = []
        outputs = export(raw, merges)
        flagged = check_intervals(outputs) or []
        changes = session_summary_gui.count_changes(baseline, outputs, channel_configs, tolerance)
        n_changes = session_summary_gui.total_changes(changes)
        extra = extra_warnings(raw, outputs) if extra_warnings else []
        blocking = blocking_warnings(raw, outputs) if blocking_warnings else []
        for line in extra + blocking:
            print(f"CHECK: {line}")
        summary_kwargs = {"blocking_lines": blocking} if blocking else {}
        if seed_notes:
            summary_kwargs["seed_lines"] = [f"{channel_configs.get(k, {}).get('label', k)} {note}"
                                            for k, note in seed_notes.items()]
        change_lines = session_summary_gui.format_change_lines(changes, channel_configs, seed_notes)
        warning_lines = _format_warning_lines(merges, flagged, channel_configs, extra)
        decision = session_summary_gui.show_session_summary(
            heading,
            change_lines,
            warning_lines,
            n_changes,
            finished_default=finished_default,
            tracker_hint=tracker_hint,
            **summary_kwargs,
        )
        # The dialog's contents and the RA's choice go to the session log only
        # (they never appear in the terminal; HLU, 2026-10-03).
        session_log.log_only(_dialog_log_lines(heading, change_lines, warning_lines, summary_kwargs, decision))
        if decision.action != session_summary_gui.BACK:
            return decision, outputs, n_changes
        finished_default = decision.finished
        print("Reopening the viewer with your current (unsaved) annotations...")


def finish_session(decision, channel_outputs, n_changes, previous_status, has_saved_file, save, existing_path,
                   same_as_saved=None):
    """
    Applies the session-summary decision: discard (write nothing), skip a
    save when nothing changed, or record each reviewed channel's status and
    save via `save(channel_outputs)`. Returns a SessionResult.

    same_as_saved: None (Step 2) means "nothing changed" is n_changes == 0,
    which is right there because a resumed channel's baseline IS its saved
    entry. Step 3 passes an explicit bool instead: its baseline is re-seeded
    from the two reviewers' current files, so "no changes on screen" does not
    mean the saved reconciliation already holds this result (checkpoint-1
    review, SD-1).
    """
    if decision.action == session_summary_gui.DISCARD:
        print("Discarded this session's changes -- nothing was saved. Your previous saved version "
              "(if any) is unchanged.")
        return SessionResult(saved_path=None, saved=False, discarded=True, comment=decision.comment)

    status = STATUS_COMPLETE if decision.finished else STATUS_IN_PROGRESS
    for output in channel_outputs.values():
        output["status"] = status

    unchanged = (n_changes == 0) if same_as_saved is None else bool(same_as_saved)
    if has_saved_file and unchanged and all(previous_status.get(k) == status for k in channel_outputs):
        print("No changes this session -- nothing new to save.")
        return SessionResult(saved_path=existing_path, saved=False, discarded=False,
                             comment=decision.comment, finished=decision.finished)

    path = save(channel_outputs)
    return SessionResult(saved_path=path, saved=True, discarded=False, comment=decision.comment,
                         finished=decision.finished)
