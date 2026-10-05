#!/usr/bin/env python3
"""
Checks that reprocessed physio files still line up, sample for sample, with
the files RAs annotated -- BEFORE the new files are copied over the old ones.

Why: RA annotations (*_annotations_<initials>.json), ECG templates
(*_ecg_corrected_qrs*.json) and legacy notebook output (*_ecg_final.json)
store peaks and bad segments as SAMPLE POSITIONS in the *_physio.tsv.gz. If
a reprocess trims or shifts a recording even slightly differently, that saved
work silently points at the wrong samples.

What it compares, per run (old file = the one RAs annotated, e.g. on Box;
new file = the freshly reprocessed one, e.g. in data/processed/bids):
  - number of samples;
  - the time lag between the two ECG traces, estimated independently in
    three 30-s windows (early / middle / late) by normalized
    cross-correlation within +/- --max-lag-sec;
  - how many old ECG peaks have a new ECG peak within 5 samples at that lag
    (informational: the peak detector itself may have changed);
  - whether any saved RA/template index falls beyond the end of the new file;
  - separately, whether the PPG (CareTaker) signal moved: it is timed by its
    own counter fit, which can change between pipeline versions while the ECG
    stays the same (ppg-plan §4d). Identical values pass at once; otherwise
    the lag is estimated in the same three windows within +/- 10 s (on the
    PPG decimated to 100 Hz; it is band-limited below ~8 Hz).

Status per run
  ALIGNED                 lag 0 in every window, same length
  ALIGNED_LENGTH_DIFFERS  lag 0 in every window, but a different length
                          (fine for saved work only if no index is past the end)
  SHIFTED                 the same nonzero lag in every window -- saved indices
                          would need that shift applied
  UNCLEAR                 windows disagree or correlate weakly -- inspect by hand
  NEEDS_PPG_CHECK         the ECG lines up, but the PPG changed and couldn't be
                          confirmed at lag 0 (or is gone), and saved RA work
                          includes PPG
  NEEDS_CARETAKER_CHECK   the ECG lines up, but SBP/DBP (CareTaker vitals, timed
                          by their own fit) changed, and saved work includes them
  REVIEW_INVALIDATED      a saved review records the signal it was made on
                          (provenance: sample count + content hash, since
                          2026-09-26), and the new file's channel differs. The
                          tool will refuse to reopen that review, so don't copy
                          until it's resolved (the RA redoes that channel, or the
                          old file is kept). To redo it, set the old entry aside
                          with retire_channels.py, after the copy and before the
                          RA's next session
  MISSING_NEW / MISSING_OLD  one of the two files isn't there

Read-only: it opens files and writes only the --report CSV (if given).
By default only runs with saved RA work in --old-root are checked; --all
checks every run present in both trees.

Usage
  annotate_env\\Scripts\\python.exe check_reprocess_alignment.py ^
      --old-root "C:\\Users\\you\\Box\\DATA\\Processed\\physioProcessing\\derivatives" ^
      --new-root "C:\\Users\\you\\Documents\\springboard2\\data\\processed\\bids" ^
      --report alignment_report.csv
Exit code 0 if every checked run is ALIGNED (or ALIGNED_LENGTH_DIFFERS with no
saved index past the end), 1 otherwise.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np

from physio_io import build_run_path, load_physio_tsv, read_sidecar, sidecar_path_for
from provenance import column_sha256

RA_WORK_PATTERNS = ("*_annotations_*.json", "*_ecg_corrected_qrs*.json", "*_ecg_final.json")
WINDOW_SEC = 30
WINDOW_POSITIONS = (0.2, 0.5, 0.8)
MIN_CORRELATION = 0.9
PEAK_MATCH_SAMPLES = 5


def find_runs_with_ra_work(old_root):
    """{(subject, run): [paths of saved RA work]} for every run folder holding any."""
    runs = {}
    for pattern in RA_WORK_PATTERNS:
        for path in glob.glob(os.path.join(old_root, "sub-*", "ses-run*", "beh", pattern)):
            if os.sep + "backups" + os.sep in path:
                continue
            m = re.search(r"sub-(\w+)_ses-run(\d+)_", os.path.basename(path))
            if m:
                runs.setdefault((m.group(1), m.group(2)), []).append(path)
    return runs


def find_all_runs(root):
    runs = set()
    for path in glob.glob(os.path.join(root, "sub-*", "ses-run*", "beh", "*_physio.tsv.gz")):
        m = re.search(r"sub-(\w+)_ses-run(\d+)_", os.path.basename(path))
        if m:
            runs.add((m.group(1), m.group(2)))
    return runs


def read_ra_file(path):
    """(payload dict, None) or (None, reason) -- never raises (2026-09-26 safety fix)."""
    try:
        with open(path, encoding="utf-8-sig") as f:
            payload = json.load(f)
    except (OSError, UnicodeDecodeError, ValueError) as e:
        return None, f"{type(e).__name__}: {e}"
    if not isinstance(payload, dict):
        return None, "not a JSON object"
    channels = payload.get("channels", {})
    if not isinstance(channels, dict) or not all(isinstance(v, dict) for v in channels.values()):
        return None, "damaged 'channels' section"
    return payload, None


def unreadable_ra_files(paths):
    """['<name>: <reason>', ...] for the saved RA files that can't be read. Such a run is not OK: until
    2026-09-26 they were skipped silently, which could report a false OK."""
    out = []
    for path in paths:
        _payload, reason = read_ra_file(path)
        if reason:
            out.append(f"{os.path.basename(path)}: {reason}")
    return out


def _ints(values):
    return [v for v in (values if isinstance(values, list) else []) if isinstance(v, int) and not isinstance(v, bool)]


LEGACY_KEY_RE = re.compile(r"_legacy(_\d+)?$")
"""
Set-aside entries (e.g. "rsp_points_legacy", or "sbp_segments_legacy" from
retire_channels.py): the tool never loads them, so they can't be invalidated
or fall past a file's end. Same rule as annotation_io._is_legacy_key (not
imported, to keep this check free of the viewer's dependencies).
"""


def _live_channels(payload):
    """The payload's channel entries the tool would load (legacy keys skipped)."""
    return {k: v for k, v in (payload.get("channels") or {}).items() if not LEGACY_KEY_RE.search(k)}


def saved_indices(paths):
    """Largest sample index referenced by any saved RA/template file (None if none)."""
    largest = None
    for path in paths:
        payload, reason = read_ra_file(path)
        if reason:
            continue  # reported by unreadable_ra_files()
        values = []
        for ch in _live_channels(payload).values():
            values += _ints(ch.get("indices", []))
            # Segment ends are exclusive ([start, end)): the last sample a
            # segment covers is end - 1. Using end itself flagged every span
            # ending exactly at the file end (common since MachineQC pre-fill)
            # as "past the end" (checkpoint-1 review, SD-5).
            values += [seg[1] - 1 for seg in (ch.get("bad_segments") or []) if isinstance(seg, (list, tuple))
                       and len(seg) == 2 and len(_ints(list(seg))) == 2]
        ecg = payload.get("ecg")
        if isinstance(ecg, dict):
            values += _ints(ecg.get("corrected_peaks", []) or [])
            values += _ints(ecg.get("bad_segments", []) or [])
        if values:
            largest = max(values) if largest is None else max(largest, max(values))
    return largest


def saved_provenance(paths):
    """{channel: [provenance dict, ...]} from the saved files' channel entries that record one."""
    found = {}
    for path in paths:
        payload, reason = read_ra_file(path)
        if reason:
            continue  # reported by unreadable_ra_files()
        for ch_key, entry in _live_channels(payload).items():
            if isinstance(entry, dict) and isinstance(entry.get("provenance"), dict):
                found.setdefault(ch_key, []).append(entry["provenance"])
    return found


def invalidated_channels(new_path, new_len, provenance_by_channel):
    """
    The channels whose saved provenance doesn't match the new file: a different
    sample count, or a different content hash (provenance.column_sha256, the
    definition shared with physioProcess). These reviews will be refused by
    the tool after a copy, so the run must not pass (checkpoint-2 review).
    """
    if not provenance_by_channel:
        return []
    columns = read_sidecar(sidecar_path_for(new_path)).get("Columns") or []
    # A companion entry (bad_ppg) is bound to its parent's column ("column": "ppg").
    column_of = {ch_key: (entries[0].get("column") or ch_key) for ch_key, entries in provenance_by_channel.items()
                 if entries}
    hashes = column_sha256(new_path, columns, wanted=sorted({c for c in column_of.values() if c in columns}))
    bad = []
    for ch_key, entries in provenance_by_channel.items():
        for entry in entries:
            column = entry.get("column") or ch_key
            if entry.get("number_of_samples") != new_len or (
                    entry.get("signal_sha256") and entry.get("signal_sha256") != hashes.get(column)):
                bad.append(ch_key)
                break
    return sorted(bad)


CARETAKER_VITALS = ("sbp", "dbp")


def compare_vitals(old_df, new_df, channels):
    """'IDENTICAL' or 'CHANGED' for the CareTaker vitals channels asked about (NaN-aware)."""
    for ch_key in channels:
        in_old, in_new = ch_key in old_df.columns, ch_key in new_df.columns
        if not (in_old and in_new):
            if in_old or in_new:
                return "CHANGED"
            continue
        old_v, new_v = old_df[ch_key].to_numpy(dtype=float), new_df[ch_key].to_numpy(dtype=float)
        if len(old_v) != len(new_v) or not np.array_equal(old_v, new_v, equal_nan=True):
            return "CHANGED"
    return "IDENTICAL"


def saved_channel_keys(paths):
    """Every channel key named in the saved RA/reconciled files' "channels"."""
    keys = set()
    for path in paths:
        payload, reason = read_ra_file(path)
        if not reason:
            keys |= set(_live_channels(payload).keys())
    return keys


PPG_MAX_LAG_SEC = 10.0
PPG_DECIMATE = 10


def compare_ppg(old_df, new_df, fs):
    """
    ("NO_PPG" | "IDENTICAL" | "ALIGNED" | "NEEDS_PPG_CHECK", lags in samples at
    fs or None, correlations or None).
    """
    def usable(frame):
        return "ppg" in frame.columns and not frame["ppg"].isna().all()

    if not usable(old_df) and not usable(new_df):
        return "NO_PPG", None, None
    if usable(old_df) != usable(new_df):
        # PPG present on one side only, e.g. the new pipeline found no usable
        # CareTaker pulse (checkpoint-2 review: this used to pass silently).
        return "PPG_REMOVED" if usable(old_df) else "PPG_ADDED", None, None
    old_ppg, new_ppg = old_df["ppg"].to_numpy(dtype=float), new_df["ppg"].to_numpy(dtype=float)
    if len(old_ppg) == len(new_ppg) and np.array_equal(old_ppg, new_ppg, equal_nan=True):
        return "IDENTICAL", [0, 0, 0], None
    step = PPG_DECIMATE
    old_d, new_d = old_ppg[::step], new_ppg[::step]
    fs_d = fs / step
    n = min(len(old_d), len(new_d))
    lags, correlations = [], []
    for position in WINDOW_POSITIONS:
        lag, r = estimate_lag(old_d, new_d, int(position * n), int(WINDOW_SEC * fs_d), int(PPG_MAX_LAG_SEC * fs_d))
        if lag is not None:
            lags.append(int(lag) * step)
            correlations.append(round(r, 3))
    ok = len(lags) == len(WINDOW_POSITIONS) and set(lags) == {0} and min(correlations) >= MIN_CORRELATION
    return ("ALIGNED" if ok else "NEEDS_PPG_CHECK"), lags, correlations


def _zscore(x):
    x = np.nan_to_num(np.asarray(x, dtype=float))
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x * 0


def estimate_lag(old_signal, new_signal, center, window, max_lag):
    """
    Lag (samples) that best aligns new[center-w/2 : center+w/2] with old, as
    new_index - old_index, and the normalized correlation at that lag.
    """
    half = window // 2
    lo, hi = center - half, center + half
    if lo - max_lag < 0 or hi + max_lag > len(old_signal) or hi > len(new_signal):
        return None, None
    segment = _zscore(new_signal[lo:hi])
    best_lag, best_r = None, -np.inf
    reference = np.nan_to_num(np.asarray(old_signal[lo - max_lag:hi + max_lag], dtype=float))
    # Correlate the new window against every shifted old window in one pass.
    corr = np.correlate(reference - reference.mean(), segment, mode="valid")
    for k, value in enumerate(corr):
        old_start = lo - max_lag + k
        old_window = reference[k:k + (hi - lo)]
        sd = old_window.std()
        r = value / (len(segment) * sd) if sd > 0 else -np.inf
        if r > best_r:
            best_r, best_lag = r, lo - old_start
    return best_lag, float(best_r)


def compare_run(old_path, new_path, sfreq_expected=None, max_lag_sec=2.0, saved_max_index=None, ppg_work=False,
                saved_prov=None, vitals_work=()):
    result = {"old_file": old_path, "new_file": new_path}
    if not os.path.exists(new_path):
        return {**result, "status": "MISSING_NEW"}
    if not os.path.exists(old_path):
        return {**result, "status": "MISSING_OLD"}
    old_df, old_fs = load_physio_tsv(old_path)
    new_df, new_fs = load_physio_tsv(new_path)
    result.update(n_old=len(old_df), n_new=len(new_df), fs_old=old_fs, fs_new=new_fs)
    if old_fs != new_fs:
        return {**result, "status": "UNCLEAR", "note": "sampling rates differ"}
    fs = new_fs
    window, max_lag = int(WINDOW_SEC * fs), int(max_lag_sec * fs)
    old_ecg, new_ecg = old_df["ecg"].to_numpy(), new_df["ecg"].to_numpy()
    n = min(len(old_ecg), len(new_ecg))
    lags, correlations = [], []
    for position in WINDOW_POSITIONS:
        lag, r = estimate_lag(old_ecg, new_ecg, int(position * n), window, max_lag)
        if lag is not None:
            lags.append(int(lag))
            correlations.append(round(r, 3))
    result.update(lags=lags, correlations=correlations)

    if len(lags) and len(set(lags)) == 1 and min(correlations) >= MIN_CORRELATION:
        lag = lags[0]
        old_peaks = np.where(old_df["ecg_peaks"].to_numpy())[0] + lag
        new_peaks = np.where(new_df["ecg_peaks"].to_numpy())[0]
        if len(old_peaks) and len(new_peaks):
            nearest = np.abs(new_peaks[np.clip(np.searchsorted(new_peaks, old_peaks), 0, len(new_peaks) - 1)] - old_peaks)
            before = np.abs(new_peaks[np.clip(np.searchsorted(new_peaks, old_peaks) - 1, 0, len(new_peaks) - 1)] - old_peaks)
            result["ecg_peak_agreement"] = round(float(np.mean(np.minimum(nearest, before) <= PEAK_MATCH_SAMPLES)), 3)
        if lag != 0:
            status = "SHIFTED"
        elif len(old_df) == len(new_df):
            status = "ALIGNED"
        else:
            status = "ALIGNED_LENGTH_DIFFERS"
        result["lag_samples"] = lag
    else:
        status = "UNCLEAR"
    if saved_max_index is not None:
        result["max_saved_index"] = int(saved_max_index)
        result["saved_index_past_end"] = bool(saved_max_index >= len(new_df))
    ppg_status, ppg_lags, ppg_correlations = compare_ppg(old_df, new_df, fs)
    result.update(ppg_status=ppg_status, ppg_lags=ppg_lags, ppg_correlations=ppg_correlations)
    aligned = status in ("ALIGNED", "ALIGNED_LENGTH_DIFFERS")
    if ppg_work and ppg_status in ("NEEDS_PPG_CHECK", "PPG_REMOVED") and aligned:
        status = "NEEDS_PPG_CHECK"
    if vitals_work:
        result["caretaker_vitals_status"] = compare_vitals(old_df, new_df, vitals_work)
        if result["caretaker_vitals_status"] == "CHANGED" and status in ("ALIGNED", "ALIGNED_LENGTH_DIFFERS"):
            status = "NEEDS_CARETAKER_CHECK"
    result["invalidated_channels"] = invalidated_channels(new_path, len(new_df), saved_prov or {})
    if result["invalidated_channels"]:
        status = "REVIEW_INVALIDATED"
    return {**result, "status": status}


def run_ok(result):
    if result["status"] == "ALIGNED":
        return True
    return result["status"] == "ALIGNED_LENGTH_DIFFERS" and not result.get("saved_index_past_end", False)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check reprocessed physio files still align with annotated ones.")
    parser.add_argument("--old-root", required=True, help="Derivatives tree RAs annotated (e.g. on Box).")
    parser.add_argument("--new-root", required=True, help="Freshly reprocessed tree (e.g. data/processed/bids).")
    parser.add_argument("--all", action="store_true", help="Check every run in both trees, not only runs with RA work.")
    parser.add_argument("--max-lag-sec", type=float, default=2.0,
                        help="How far (seconds) the ECG lag search looks each way (default: 2).")
    parser.add_argument("--report", default=None, help="Optional CSV report path.")
    args = parser.parse_args(argv)

    # A mistyped folder must not read as "nothing to check" (HLU, 2026-09-27:
    # a placeholder C:\Users\you\... path printed "No runs to check").
    for flag, path in (("--old-root", args.old_root), ("--new-root", args.new_root)):
        if not os.path.isdir(path):
            raise SystemExit(f"{flag} folder not found: {path}\nCheck the path (for example the user name in "
                             f"C:\\Users\\<name>\\Box\\...). Nothing was checked.")

    work = find_runs_with_ra_work(args.old_root)
    runs = sorted(find_all_runs(args.old_root) & find_all_runs(args.new_root) | set(work)) if args.all \
        else sorted(work)
    if not runs:
        print(f"No runs to check: no saved RA work was found under --old-root ({args.old_root}). If you expected "
              f"some, check that this is the Box derivatives folder (the one holding the sub-XXX folders).")
        return 0

    rows = []
    for subject, run in runs:
        old_path = build_run_path(args.old_root, subject, run)
        new_path = build_run_path(args.new_root, subject, run)
        paths = work.get((subject, run), [])
        keys = saved_channel_keys(paths)
        result = compare_run(old_path, new_path, max_lag_sec=args.max_lag_sec,
                             saved_max_index=saved_indices(paths), ppg_work="ppg" in keys,
                             saved_prov=saved_provenance(paths),
                             vitals_work=tuple(k for k in CARETAKER_VITALS if k in keys))
        result.update(subject=subject, run=run, ra_files=len(work.get((subject, run), [])))
        unreadable = unreadable_ra_files(paths)
        if unreadable:
            result.update(status="UNREADABLE_RA_FILE", unreadable_ra_files=unreadable)
        rows.append(result)
        extra = ""
        if "lag_samples" in result:
            extra = f"lag {result['lag_samples']} samples, windows r={result['correlations']}"
        elif result.get("lags"):
            extra = f"window lags {result['lags']}, r={result['correlations']}"
        if "n_old" in result:
            extra += f"; samples old {result['n_old']} / new {result['n_new']}"
        if result.get("saved_index_past_end"):
            extra += "; a saved index is PAST THE END of the new file"
        if result.get("note"):
            extra += f"; {result['note']} ({result.get('fs_old')} vs {result.get('fs_new')} Hz)"
        if result.get("unreadable_ra_files"):
            extra += f"; saved RA file(s) can't be read: {'; '.join(result['unreadable_ra_files'])}"
        if result.get("invalidated_channels"):
            extra += f"; saved review(s) of {', '.join(result['invalidated_channels'])} made on a different signal"
        if result.get("caretaker_vitals_status") == "CHANGED":
            extra += "; SBP/DBP changed"
        if result.get("ppg_status") not in (None, "NO_PPG"):
            extra += f"; PPG {result['ppg_status']}"
            if result.get("ppg_lags") and result["ppg_status"] != "IDENTICAL":
                extra += f" (lags {result['ppg_lags']}, r={result['ppg_correlations']})"
        print(f"sub-{subject} run {run}: {result['status']}  {extra}".rstrip())

    if args.report:
        fields = ["subject", "run", "status", "lag_samples", "lags", "correlations", "n_old", "n_new",
                  "ecg_peak_agreement", "ppg_status", "ppg_lags", "ppg_correlations", "caretaker_vitals_status",
                  "invalidated_channels", "unreadable_ra_files", "note", "fs_old", "fs_new", "ra_files",
                  "max_saved_index",
                  "saved_index_past_end", "old_file", "new_file"]
        with open(args.report, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Report written to {args.report}")

    bad = [r for r in rows if not run_ok(r)]
    bad_names = ", ".join("sub-{} run {}".format(r["subject"], r["run"]) for r in bad)
    print(f"\n{len(rows) - len(bad)} of {len(rows)} run(s) OK." +
          ("" if not bad else f" Do NOT copy the new files over these until resolved: {bad_names}"))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
