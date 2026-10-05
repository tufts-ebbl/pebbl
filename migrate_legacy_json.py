#!/usr/bin/env python3
"""
One-time migration: converts step2_physio_correction.ipynb's old
per-subject/run *_ecg_final.json output (written by the `systole` package's
Editor.save(), shape {"ecg": {"valid", "corrected_peaks", "bad_segments"}})
into this tool's *_annotations_<initials>.json shape (shape {"initials",
"timestamp", "source_file", "sampling_rate", "channels": {"ecg": {"mode":
"point", "indices": [...]}}}) -- so RA work already done via the notebook
isn't orphaned once a study standardizes on this tool.

Confirmed with the user before building (see HANDOFF.md): across this
study's entire real derivatives tree, exactly ONE *_ecg_final.json has a
non-empty "bad_segments" (sub-001, run 1) -- not worth adding real ECG
segment-annotation support to this tool for a single historical case (the
tool has since gained ECG bad stretches, bad_ecg, 2026-09-27, but this
migration is unchanged), so
this script deliberately DROPS "bad_segments" (loudly, not silently) and
carries forward only "corrected_peaks". If that one file's flagged stretch
matters, note it manually (e.g. in processing_log.csv's Notes column)
rather than expecting this script to preserve it.

The old JSON has no "initials" field at all (that convention didn't exist
yet), so this script looks up who did the correction from
processing_log.csv's own history: the LAST "R peaks inspected and
corrected" row (the notebook's fixed Step text) for that subject/run. If
none is found, that file is skipped with a clear warning rather than
guessing -- run with --initials-for <subject>:<run>:<initials> to supply
one manually for cases the log can't resolve (e.g. a missing/garbled log
row).

Read-only by default: prints exactly what it WOULD write, for review,
without touching anything. Pass --apply to actually write. Never touches
or deletes the original *_ecg_final.json.

Usage:
    annotate_env\\Scripts\\python.exe migrate_legacy_json.py --box-path <path>
    annotate_env\\Scripts\\python.exe migrate_legacy_json.py --box-path <path> --apply

--upgrade (2026-09-26, HLU): instead, rewrites each migrated review in the
CURRENT annotation format (schema 2), from the same *_ecg_final.json peaks
and processing_log.csv initials, against the run-8 physio file next to it:
status "in_progress" (an RA reopens it and ticks "finished"), reviewed_span,
the run-8 file's provenance, guide_shown false (the notebook showed ECG
only), an edit_history record computed against the notebook QRS template
the step-2 notebook started from (when that template's stored initials match
the reviewer; else marked unknown), and a "migrated_from" record (source file
and its SHA-256).

Two checks/corrections on the run-8 ECG (HLU, 2026-09-26, after a diagnostic
on the real files):
- TRANSFER CHECK: step1_qrs_template.ipynb snapped template peaks to the
  raw-ECG maximum (a +/-50-sample window, end exclusive), then ran systole's
  correct_peaks(), which inserts missed beats at interval-interpolated
  positions (not snapped). So on an unchanged signal most, not all, template
  peaks are a local maximum within +/-2 samples: 91-98.5% on the real files,
  where the peaks reviewers left unedited were 99.5-100% at the top. A
  changed signal would move nearly all of them. Runs under 90% are flagged
  for a person to look at (e.g. sub-032 r1 at 57%: a template misaligned by
  ~40 ms, whose peaks sit on slopes at the snap window's edge).
  --exclude SUBJECT:RUN leaves a run out entirely.
- 3-SAMPLE SNAP: the notebook's editor stored edited peaks 1-2 samples
  before the R-wave maximum (e.g. sub-003 r1: 322 peaks at +1 sample, IQR +1
  to +1, while unedited peaks were 100% at the top). A peak within 3 samples
  of a local maximum is moved onto it; each original position is kept in
  migrated_from.snapped_to_local_max, and the edit history is computed after
  the snap. Peaks further away (messy stretches) are left exactly as they were.
It never overwrites a review already re-saved in the tool, or one whose peaks
differ from the notebook's; the existing file is backed up first.
    annotate_env\\Scripts\\python.exe migrate_legacy_json.py --box-path <path> --upgrade
    annotate_env\\Scripts\\python.exe migrate_legacy_json.py --box-path <path> --upgrade --apply
"""

import argparse
import glob
import hashlib
import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

from annotation_io import DEFAULT_SNAP_WINDOW_SEC, STATUS_IN_PROGRESS, save_annotation_json
from physio_io import build_run_path, load_physio_tsv, read_sidecar, sidecar_path_for
from ppg_checks import markers_off_top
from provenance import channel_provenance
from session_summary_gui import edit_record

LEGACY_FILENAME_RE = re.compile(r"^(sub-(\d+)_ses-run(\d+)_task-sdi)_ecg_final\.json$")

STEP2_COMPLETION_TEXT = "R peaks inspected and corrected"


def find_legacy_final_jsons(derivatives_root):
    """Recursively finds every *_ecg_final.json under derivatives_root, oldest-path-first."""
    return sorted(glob.glob(os.path.join(derivatives_root, "**", "*_ecg_final.json"), recursive=True))


def parse_legacy_filename(path):
    """Returns (file_stem, subject, run) from a legacy filename, or None if it doesn't match."""
    match = LEGACY_FILENAME_RE.match(os.path.basename(path))
    if not match:
        return None
    file_stem, subject, run = match.groups()
    return file_stem, subject, run


def lookup_step2_timestamp(box_path, subject, run):
    """The Timestamp of the LAST "R peaks inspected and corrected" row for this subject/run, or None."""
    log_path = os.path.join(box_path, "processing_log.csv")
    if not os.path.exists(log_path):
        return None
    log_df = pd.read_csv(log_path, dtype=str, keep_default_na=False)
    if "Timestamp" not in log_df.columns:
        return None
    matches = log_df[(log_df["Subject"] == f"sub-{subject}") & (log_df["Run"] == str(int(run)))
                     & (log_df["Step"] == STEP2_COMPLETION_TEXT)]
    return None if matches.empty else (matches.iloc[-1]["Timestamp"] or None)


def lookup_step2_initials(box_path, subject, run, overrides=None):
    """
    Returns the initials who completed Stage B for this subject/run, per
    the LAST "R peaks inspected and corrected" row in processing_log.csv
    (rows are appended chronologically, so the last match is the most
    recent -- avoids parsing the CSV's own non-zero-padded date format).
    Checks `overrides` (a {(subject, run): initials} dict from
    --initials-for) first. Returns None if nothing resolves either way.
    """
    key = (subject, run)
    if overrides and key in overrides:
        return overrides[key]

    log_path = os.path.join(box_path, "processing_log.csv")
    if not os.path.exists(log_path):
        return None

    log_df = pd.read_csv(log_path, dtype=str, keep_default_na=False)
    matches = log_df[
        (log_df["Subject"] == f"sub-{subject}")
        & (log_df["Run"] == str(int(run)))
        & (log_df["Step"] == STEP2_COMPLETION_TEXT)
    ]
    if matches.empty:
        return None
    return matches.iloc[-1]["Initials"]


def convert_one(legacy_path, box_path, overrides=None):
    """
    Reads one legacy *_ecg_final.json and returns a dict describing the
    planned migration (never writes anything itself -- see migrate()):
        {"file_stem", "subject", "run", "initials", "sfreq", "source_file",
         "indices", "bad_segments", "skip_reason"}
    "skip_reason" is None on success, else a human-readable reason this
    file can't be converted (e.g. no matching processing_log.csv row).
    """
    parsed = parse_legacy_filename(legacy_path)
    if parsed is None:
        return {"legacy_path": legacy_path, "skip_reason": "filename doesn't match the expected pattern"}
    file_stem, subject, run = parsed

    with open(legacy_path) as f:
        legacy_payload = json.load(f)

    if "ecg" not in legacy_payload:
        return {"legacy_path": legacy_path, "skip_reason": f"no 'ecg' key found (has: {list(legacy_payload.keys())})"}

    ecg_entry = legacy_payload["ecg"]
    indices = sorted(int(i) for i in ecg_entry.get("corrected_peaks", []))
    bad_segments = ecg_entry.get("bad_segments")

    initials = lookup_step2_initials(box_path, subject, run, overrides)
    if not initials:
        return {
            "legacy_path": legacy_path, "file_stem": file_stem, "subject": subject, "run": run,
            "skip_reason": (
                f"no '{STEP2_COMPLETION_TEXT}' row found in processing_log.csv for sub-{subject} run {run} "
                f"-- pass --initials-for {subject}:{run}:<initials> to supply one manually"
            ),
        }

    tsv_path = build_run_path(box_path, subject, run)
    if not os.path.exists(tsv_path):
        return {
            "legacy_path": legacy_path, "file_stem": file_stem, "subject": subject, "run": run,
            "skip_reason": f"expected physio.tsv.gz not found at {tsv_path}",
        }
    _df, sfreq = load_physio_tsv(tsv_path)

    return {
        "legacy_path": legacy_path, "file_stem": file_stem, "subject": subject, "run": run,
        "initials": initials, "sfreq": sfreq, "source_file": tsv_path,
        "indices": indices, "bad_segments": bad_segments, "skip_reason": None,
    }


def migrate(derivatives_root, box_path, apply=False, overrides=None):
    """
    Finds and converts every legacy *_ecg_final.json under derivatives_root.
    Prints a report either way; only writes new *_annotations_<initials>.json
    files (via save_annotation_json(), which merges-on-save and backs up
    any existing file first) when apply=True.
    """
    legacy_paths = find_legacy_final_jsons(derivatives_root)
    if not legacy_paths:
        print(f"No *_ecg_final.json files found under {derivatives_root}.")
        return []

    print(f"Found {len(legacy_paths)} legacy file(s) under {derivatives_root}.\n")

    results = []
    for legacy_path in legacy_paths:
        result = convert_one(legacy_path, box_path, overrides)
        results.append(result)

        if result["skip_reason"]:
            print(f"SKIP {legacy_path}\n  reason: {result['skip_reason']}\n")
            continue

        out_dir = os.path.dirname(legacy_path)
        out_path = os.path.join(out_dir, f"{result['file_stem']}_annotations_{result['initials']}.json")
        print(f"{'WRITE' if apply else 'WOULD WRITE'} {out_path}")
        print(f"  from {legacy_path}")
        print(f"  initials={result['initials']!r} (from processing_log.csv), "
              f"sfreq={result['sfreq']}, {len(result['indices'])} ecg peak(s)")
        if result["bad_segments"]:
            print(f"  WARNING: dropping non-empty bad_segments={result['bad_segments']} -- "
                  f"see this script's module docstring for why; note it manually if it matters.")

        if apply:
            channel_outputs = {"ecg": {"mode": "point", "indices": result["indices"]}}
            save_annotation_json(out_dir, result["file_stem"], result["initials"],
                                  channel_outputs, result["sfreq"], result["source_file"])
        print()

    written = sum(1 for r in results if not r["skip_reason"])
    skipped = len(results) - written
    print(f"{'Wrote' if apply else 'Would write'} {written} file(s); skipped {skipped}.")
    if not apply and written:
        print("Re-run with --apply to actually write these files.")
    return results


ON_TOP_WARNING_FRACTION = 0.90
"""Below this share of template peaks at a local maximum on the new file, a run is flagged (transfer check)."""


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _as_indices(values, n_samples):
    """Peak indices from a list of indices, or from a per-sample boolean/0-1 list."""
    values = list(values or [])
    if len(values) == n_samples and set(values) <= {0, 1, True, False}:
        return [int(i) for i in np.flatnonzero(np.asarray(values, dtype=bool))]
    return sorted({int(i) for i in values})


TRANSFER_CHECK_SAMPLES = 2
SNAP_SAMPLES = 3


def _is_local_max(signal, i, k):
    lo, hi = max(0, i - k), min(len(signal), i + k + 1)
    return not np.isnan(signal[i]) and signal[i] >= np.nanmax(signal[lo:hi])


def snap_small(signal, peaks, k=SNAP_SAMPLES):
    """
    (peaks, moves): each peak within k samples of a local maximum (the highest
    sample within +/-k that is itself a local maximum within +/-k) moved onto
    it; moves = [[from, to], ...]. Peaks that land on the same sample are merged.
    """
    out, moves = [], []
    for p in peaks:
        lo, hi = max(0, p - k), min(len(signal), p + k + 1)
        window = signal[lo:hi]
        if np.all(np.isnan(window)):
            out.append(p)
            continue
        q = lo + int(np.nanargmax(window))
        if q != p and _is_local_max(signal, q, k):
            out.append(q)
            moves.append([int(p), int(q)])
        else:
            out.append(p)
    return sorted(set(out)), moves


def template_transfer_check(legacy_path, file_stem, signal, n_samples):
    """
    Share of the folder's notebook template peaks (whoever made it) that are a
    local maximum within +/-TRANSFER_CHECK_SAMPLES on this ECG, or None when
    there's no readable template.
    """
    path = os.path.join(os.path.dirname(legacy_path), f"{file_stem}_ecg_corrected_qrs.json")
    try:
        with open(path) as f:
            peaks = _as_indices(((json.load(f) or {}).get("ecg") or {}).get("corrected_peaks"), n_samples)
    except (OSError, ValueError):
        return None
    peaks = [p for p in peaks if 0 <= p < n_samples]
    if not peaks:
        return None
    return sum(_is_local_max(signal, p, TRANSFER_CHECK_SAMPLES) for p in peaks) / len(peaks)


def find_notebook_seed(legacy_path, file_stem, initials, n_samples):
    """
    (peaks, description) of the notebook QRS template the step-2 notebook
    started from -- <file_stem>_ecg_corrected_qrs.json next to the legacy
    file -- when its stored initials include this reviewer; else (None,
    reason). The notebook named templates without initials, so a later
    Step 1 by someone else could have replaced the one this RA started from.
    """
    path = os.path.join(os.path.dirname(legacy_path), f"{file_stem}_ecg_corrected_qrs.json")
    if not os.path.exists(path):
        return None, "no notebook QRS template for this run"
    try:
        with open(path) as f:
            ecg = (json.load(f) or {}).get("ecg") or {}
    except (OSError, ValueError) as e:
        return None, f"the notebook QRS template can't be read ({e})"
    makers = [str(i) for i in (ecg.get("initials") or [])]
    if initials.lower() not in [m.lower() for m in makers]:
        return None, f"the notebook QRS template was made by {', '.join(makers) or 'someone unrecorded'}, not {initials}"
    return _as_indices(ecg.get("corrected_peaks"), n_samples), f"{os.path.basename(path)}, by {', '.join(makers)}"


def upgrade_one(legacy_path, box_path, overrides=None, upgraded_on=None):
    """
    Plans the schema-2 rewrite of one migrated review (see the module
    docstring's --upgrade). Returns convert_one()'s dict plus "entry" (the new
    channels["ecg"] entry), "out_path", "on_top", "n_off", "seed_note", or a
    "skip_reason". Writes nothing.
    """
    plan = convert_one(legacy_path, box_path, overrides)
    if plan["skip_reason"]:
        return plan
    file_stem, initials, tsv_path = plan["file_stem"], plan["initials"], plan["source_file"]
    df, sfreq = load_physio_tsv(tsv_path)
    n = len(df)
    out_path = os.path.join(os.path.dirname(legacy_path), f"{file_stem}_annotations_{initials}.json")
    raw_peaks = list(plan["indices"])
    indices = sorted(set(raw_peaks))

    def skip(reason):
        return {**plan, "out_path": out_path, "skip_reason": reason}

    if int(round(sfreq)) != 1000:
        return skip(f"the file is {sfreq} Hz; the notebook worked at 1000 Hz")
    outside = [i for i in indices if i < 0 or i >= n]
    if outside:
        return skip(f"{len(outside)} peak(s) outside this file (0-{n - 1})")
    if not indices:
        return skip("the notebook review has no ECG peaks")
    # Only an existing migrated review is rewritten: if processing_log.csv now
    # resolves to other initials, writing a new file under them would be wrong.
    if not os.path.exists(out_path):
        return skip(f"no migrated review {os.path.basename(out_path)} to upgrade (were these the initials used on "
                    f"2026-09-20? see --initials-for)")
    with open(out_path) as f:
        existing = (json.load(f).get("channels") or {}).get("ecg")
    if existing is not None:
        if existing and any(k in existing for k in ("status", "edit_history", "provenance")):
            return skip("this ECG review was already re-saved in the tool (or upgraded); not overwritten")
        if existing and sorted(existing.get("indices", [])) != sorted(raw_peaks):
            return skip("the migrated file's ECG peaks differ from the notebook's; not overwritten -- check it")

    tolerance = int(round(DEFAULT_SNAP_WINDOW_SEC * sfreq))
    signal = df["ecg"].to_numpy(dtype=float)
    transfer = template_transfer_check(legacy_path, file_stem, signal, n)
    notebook_peaks = indices
    indices, snapped = snap_small(signal, notebook_peaks)
    off = markers_off_top(signal, indices, tolerance)
    on_top = 1.0 - len(off) / len(indices)
    seed, seed_note = find_notebook_seed(legacy_path, file_stem, initials, n)
    record = {"saved_at": lookup_step2_timestamp(box_path, plan["subject"], plan["run"]), "initials": initials,
              "guide_shown": False,
              "note": "notebook-era review (step2_physio_correction.ipynb), rewritten in the current format"}
    if seed is not None:
        record["started_from"] = f"notebook QRS template ({seed_note})"
        record.update(edit_record({"mode": "point", "indices": seed}, {"mode": "point", "indices": indices},
                                  tolerance))
    else:
        record["started_from"] = f"notebook (starting peaks unknown: {seed_note})"
        record.update(added=None, removed=None, moved=None)
    sidecar = read_sidecar(sidecar_path_for(tsv_path))
    provenance = channel_provenance(tsv_path, sidecar, ["ecg"], n).get("ecg")
    entry = {
        "mode": "point", "indices": indices, "snap_window_samples": tolerance, "status": STATUS_IN_PROGRESS,
        "reviewed_span": [0, n], "guide_shown": False, "edit_history": [record],
        "migrated_from": {
            "file": os.path.basename(legacy_path), "sha256": _sha256(legacy_path),
            "made_with": "step2_physio_correction.ipynb", "rewritten_on": upgraded_on,
            "template_peaks_at_local_max": None if transfer is None else round(transfer, 4),
            "snapped_to_local_max": snapped,
            "snap_rule": (f"peaks within {SNAP_SAMPLES} samples of a local ECG maximum were moved onto it (the "
                          f"notebook's editor stored edited peaks 1-2 samples early); [from, to] pairs"),
            "merged_by_snap": len(notebook_peaks) - len(indices),
            "peaks_at_top_within_50_ms": round(on_top, 4),
            "duplicate_peaks_removed": len(raw_peaks) - len(notebook_peaks),
            "dropped_bad_segments": plan["bad_segments"] or None,
            "note": ("peaks carried over unchanged to this file, whose ECG timeline matches the file they were "
                     "made on (check_reprocess_alignment: lag 0, same length)"),
        },
    }
    if provenance:
        entry["provenance"] = provenance
    return {**plan, "indices": indices, "sfreq": sfreq, "entry": entry, "out_path": out_path, "on_top": on_top,
            "n_off": len(off), "seed_note": seed_note, "seed_found": seed is not None, "record": record,
            "transfer": transfer, "snapped": snapped}


def upgrade(derivatives_root, box_path, apply=False, overrides=None, exclude=()):
    """
    Plans (and with apply=True, writes) the schema-2 rewrite of every migrated
    review; prints a report. exclude: {(subject, run)} left out entirely.
    """
    legacy_paths = find_legacy_final_jsons(derivatives_root)
    if not legacy_paths:
        print(f"No *_ecg_final.json files found under {derivatives_root}.")
        return []
    upgraded_on = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Found {len(legacy_paths)} notebook ECG review(s) under {derivatives_root}.\n")
    results, flagged, unchecked = [], [], []
    for legacy_path in legacy_paths:
        parsed = parse_legacy_filename(legacy_path)
        if parsed and (parsed[1], parsed[2]) in set(exclude):
            print(f"EXCLUDED {parsed[0]} (--exclude): left as it is\n")
            results.append({"legacy_path": legacy_path, "file_stem": parsed[0], "skip_reason": "excluded"})
            continue
        result = upgrade_one(legacy_path, box_path, overrides, upgraded_on=upgraded_on)
        results.append(result)
        name = result.get("file_stem", os.path.basename(legacy_path))
        if result["skip_reason"]:
            print(f"SKIP {name}\n  reason: {result['skip_reason']}\n")
            continue
        rec = result["record"]
        print(f"{'UPGRADE' if apply else 'WOULD UPGRADE'} {result['out_path']}")
        transfer = result["transfer"]
        low = transfer is not None and transfer < ON_TOP_WARNING_FRACTION
        if transfer is None:
            print("  transfer check: no readable notebook template here, so it can't be checked")
            unchecked.append(name)
        else:
            print(f"  transfer check: {transfer:.1%} of the template's peaks at a local maximum (+/-"
                  f"{TRANSFER_CHECK_SAMPLES} samples) on this file" + (" -- CHECK: the ECG may have changed" if low else ""))
        print(f"  {len(result['indices'])} ECG peak(s) by {result['initials']!r}; {len(result['snapped'])} moved "
              f"1-{SNAP_SAMPLES} samples onto the R-wave maximum; {result['on_top']:.1%} at the top within 50 ms "
              f"(messy stretches lower this)")
        if result["seed_found"]:
            print(f"  started from {result['seed_note']}: {len(rec['added'])} added, {len(rec['removed'])} removed, "
                  f"{len(rec['moved'])} moved")
        else:
            print(f"  what the RA changed is unknown: {result['seed_note']}")
        if result["entry"]["migrated_from"]["dropped_bad_segments"]:
            print(f"  note: the notebook's ECG bad_segments {result['bad_segments']} are not carried over "
                  f"(ECG is point-only)")
        if low:
            flagged.append(name)
        if apply:
            save_annotation_json(os.path.dirname(legacy_path), result["file_stem"], result["initials"],
                                 {"ecg": result["entry"]}, result["sfreq"], result["file_stem"])
        print()
    done = sum(1 for r in results if not r["skip_reason"])
    print(f"{'Upgraded' if apply else 'Would upgrade'} {done} review(s); skipped {len(results) - done}.")
    if flagged:
        print(f"CHECK these (under {ON_TOP_WARNING_FRACTION:.0%} of template peaks at a local maximum; the ECG may "
              f"have changed): {', '.join(flagged)}")
    if unchecked:
        print(f"Not checked (no notebook template): {', '.join(unchecked)}")
    if not apply and done:
        print("Re-run with --apply to write them (each existing file is backed up first).")
    return results


def _parse_overrides(pairs):
    overrides = {}
    for pair in pairs or []:
        try:
            subject, run, initials = pair.split(":")
        except ValueError:
            raise SystemExit(f"--initials-for expects <subject>:<run>:<initials>, got {pair!r}")
        overrides[(subject.replace("sub-", "").zfill(3), run.replace("run", ""))] = initials
    return overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--box-path", required=True,
                         help="Box derivatives folder (e.g. ...\\physioProcessing\\derivatives). Both searched "
                              "for legacy files and used to look up processing_log.csv/physio.tsv.gz files.")
    parser.add_argument("--apply", action="store_true",
                         help="Actually write the converted *_annotations_<initials>.json files. Without this, "
                              "only prints what would be written.")
    parser.add_argument("--upgrade", action="store_true",
                         help="Rewrite the migrated reviews in the current format (schema 2, status in_progress, "
                              "provenance, edit history against the notebook template); see the module docstring.")
    parser.add_argument("--exclude", action="append", metavar="SUBJECT:RUN",
                         help="With --upgrade: leave this subject/run out (e.g. '032:1'). Can be repeated.")
    parser.add_argument("--initials-for", action="append", metavar="SUBJECT:RUN:INITIALS",
                         help="Manually supply initials for a subject/run processing_log.csv can't resolve "
                              "(e.g. '001:1:hlu'). Can be passed multiple times.")
    args = parser.parse_args()

    overrides = _parse_overrides(args.initials_for)
    if args.upgrade:
        exclude = set()
        for pair in args.exclude or []:
            try:
                subject, run = pair.split(":")
            except ValueError:
                raise SystemExit(f"--exclude expects <subject>:<run>, got {pair!r}")
            exclude.add((subject.replace("sub-", "").zfill(3), run.replace("run", "")))
        upgrade(args.box_path, args.box_path, apply=args.apply, overrides=overrides, exclude=exclude)
    else:
        migrate(args.box_path, args.box_path, apply=args.apply, overrides=overrides)


if __name__ == "__main__":
    main()
