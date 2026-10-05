"""
Integration with the existing processing_log.csv convention from
step1_qrs_template.ipynb/step2_physio_correction.ipynb -- lets sessions
from this tool show up in the SAME shared log the notebooks already write
to (<box_path>/processing_log.csv, alongside the sub-XXX folders), using
the same Subject/Run/Initials/Notes/Timestamp columns.

The one deliberate difference is the "Step" text for a review session:
the notebooks always log the fixed, ECG-specific "R peaks inspected and
corrected", but this tool's Stage B can review several channels in one
sitting, so it logs a generalized "<ch1>, <ch2>, ... reviewed" string
listing whatever was actually reviewed that session (a deliberate choice,
made explicitly over keeping the legacy text -- see HANDOFF.md).
"""

import csv
import glob
import os
import shutil
from datetime import datetime

import pandas as pd

LOG_COLUMNS = ["Subject", "Run", "Initials", "Step", "Notes", "Timestamp", "Value_ranges"]

# Matches the existing notebooks' own convention EXACTLY: the actual
# processing_log.csv writes go straight to Box (it's the shared source of
# truth -- as soon as anyone else processes data, a local copy is stale by
# definition), but a local READ-ONLY snapshot gets refreshed after every
# write, named specifically to make clear it's a snapshot, not the live
# file. Deliberately the SAME path/filename step1_qrs_template.ipynb/
# step2_physio_correction.ipynb already use (one level up from this
# multisignal_annotation/ folder) -- confirmed with the user this is where
# theirs already lands -- so there's one shared, most-recent snapshot
# regardless of whether the notebooks or this tool wrote last, not a
# second, redundant one specific to this tool.
#
# A MODULE-LEVEL variable (not a hardcoded path inline) specifically so
# tests can redirect it via `unittest.mock.patch("processing_log.
# LOCAL_SNAPSHOT_DIR", tmp_dir)` -- without that, every test that exercises
# append_processing_log_entry() against a fake/temp box_path would still
# overwrite the user's REAL local snapshot with test garbage, since the
# snapshot destination doesn't depend on box_path at all.
#
# Installed copies (HLU, 2026-10-05) put it in the user's home folder
# instead. On a lab computer, PEBBL is one shared clone of the public repo
# in C:\Users\Public\Downloads\pebbl, and the folder above it is readable by
# every account. The log holds participant IDs, RA initials and comments,
# so the copy has to stay in each user's own folder. A personal install in
# ~/pebbl already wrote to the home folder, so nothing changes for it.
def default_snapshot_dir(tool_dir=None):
    """
    Where the read-only snapshot goes. An installed copy (the tool folder is
    its own git clone, as pebbl_launcher.py checks) uses the user's home
    folder. A development checkout (this folder inside physioCorrection/)
    keeps the notebooks' place, the folder above it.
    """
    tool_dir = tool_dir or os.path.dirname(os.path.abspath(__file__))
    if os.path.isdir(os.path.join(tool_dir, ".git")):
        return os.path.expanduser("~")
    return os.path.dirname(tool_dir)


LOCAL_SNAPSHOT_DIR = default_snapshot_dir()
LOCAL_SNAPSHOT_FILENAME = "processing_log_READONLY_SNAPSHOT.csv"


def _log_path(box_path):
    return os.path.join(box_path, "processing_log.csv")


# Entries that look like initials but aren't: "sub", from a subject ID
# (sub-001) typed into the initials box, which HLU has seen happen. Always
# confirmed, even if an earlier slip put it in the log (HLU, 2026-09-27).
# The run is picked from a list, so it can't end up here; other slips are
# caught by the never-used-before question.
LOOKALIKE_INITIALS = ("sub",)


def known_initials(box_path):
    """
    The lower-cased initials already in <box_path>/processing_log.csv, or
    None when there's no log to read (no Box path, no log yet, or it can't be
    read). Read-only: it never writes the log or the local snapshot.
    """
    log_path = _log_path(box_path) if box_path else None
    if not log_path or not os.path.exists(log_path):
        return None
    try:
        log_df = pd.read_csv(log_path, dtype=str, keep_default_na=False)
    except Exception:
        return None
    if "Initials" not in log_df.columns:
        return None
    return {value.strip().lower() for value in log_df["Initials"] if value.strip()}


def initials_confirmation_text(initials, known):
    """
    The question to ask once before these initials are used (they name the
    RA's saved files), or None. Asked for a look-alike (LOOKALIKE_INITIALS),
    and for initials the log has never seen; known=None (no log) skips the
    second check.
    """
    if initials in LOOKALIKE_INITIALS:
        return (f'"{initials}" looks like the start of a subject ID (as in sub-001), not your initials.\n\n'
                f"Click No to fix it, or Yes if these really are your initials.")
    if known is not None and initials not in known:
        return (f'"{initials}" hasn\'t been used in this study\'s processing log before.\n\n'
                f"If this is your first time using the tool, that's expected: click Yes.\n"
                f"If you've used it before, click No and check your initials: they name your saved files.")
    return None


def _update_local_snapshot(log_path):
    """
    Best-effort only: the actual log write to Box has already succeeded by
    the time this runs, so a failed snapshot copy (e.g. permissions, the
    local folder not existing in some other checkout) is silently skipped
    rather than raised -- it's a convenience mirror, not the source of truth.
    """
    try:
        os.makedirs(LOCAL_SNAPSHOT_DIR, exist_ok=True)
        shutil.copy(log_path, os.path.join(LOCAL_SNAPSHOT_DIR, LOCAL_SNAPSHOT_FILENAME))
    except OSError:
        pass


def _now_stamp():
    # Matches the existing log's "M/D/YYYY H:MM" style (no zero-padding).
    # Built by hand rather than with strftime("%#m/...") -- the "#" flag is
    # Windows-only, and RAs include Mac users (audit finding M5).
    now = datetime.now()
    return f"{now.month}/{now.day}/{now.year} {now.hour}:{now.minute:02d}"


def _read_header(log_path):
    with open(log_path, newline="", encoding="utf-8-sig") as f:
        return next(csv.reader(f), [])


def _append_row(log_path, row):
    """
    Appends ONE line to an existing log instead of re-reading and rewriting
    the whole file (audit finding H5): the rewrite left a window in which two
    RAs saving at nearly the same moment on Box Drive could erase each
    other's row. Columns are written in the file's own header order; if the
    header lacks a column this tool writes (e.g. Value_ranges on a log that
    predates it), the file is rewritten ONCE with that column added.
    """
    header = _read_header(log_path)
    if not all(col in header for col in LOG_COLUMNS):
        existing = pd.read_csv(log_path, dtype=str, keep_default_na=False)
        combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        combined.to_csv(log_path, index=False)
        return
    needs_newline = False
    with open(log_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            needs_newline = f.read(1) not in (b"\n", b"\r")
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        csv.writer(f).writerow([row.get(col, "") for col in header])


def _subject_id(subject):
    """Normalizes to 'sub-XXX' regardless of whether 'sub-' was already included."""
    subject = str(subject)
    return subject if subject.startswith("sub-") else f"sub-{subject}"


def append_processing_log_entry(box_path, subject, run, initials, step, notes="", value_ranges="", snapshot=True):
    """
    Appends one row to <box_path>/processing_log.csv, matching the exact
    column shape the existing notebooks already write, plus a Value_ranges
    column (min-max and mean of each segment-mode channel reviewed; kept out
    of the RA's free-text Notes per the user's request). Never raises: by
    the time this is called, the RA's actual annotation output has already
    been saved separately, so a logging hiccup (most commonly the file
    being open in Excel) shouldn't threaten that work -- this prints a
    clear message and moves on instead of crashing the session.
    """
    log_path = _log_path(box_path)
    run_value = "" if run in (None, "") else str(run).replace("run", "").replace("ses-", "")

    row = {
        "Subject": _subject_id(subject),
        "Run": run_value,
        "Initials": initials,
        "Step": step,
        "Notes": notes or "",
        "Timestamp": _now_stamp(),
        "Value_ranges": value_ranges or "",
    }

    try:
        if os.path.exists(log_path):
            _append_row(log_path, row)
        else:
            pd.DataFrame([row], columns=LOG_COLUMNS).to_csv(log_path, index=False)
        print(f"Logged to {log_path}: {_subject_id(subject)}, run={run_value or 'N/A'}, {step!r}")
        if snapshot:  # off for certification practice: the snapshot mirrors the REAL log only
            _update_local_snapshot(log_path)
    except PermissionError:
        print(f"\nNOTE: could not write to {log_path} -- it's likely open in Excel or another "
              f"program. Close it if you'd like this entry recorded; your actual annotation "
              f"work above was already saved separately and is NOT affected by this.")
    except Exception as exc:  # noqa: BLE001 -- see docstring: a log hiccup must never end the session
        # Anything else (Box Drive offline, a malformed CSV, ...) previously
        # crashed the session AFTER the annotation save but BEFORE the push
        # back to Box (audit finding C2).
        print(f"\nNOTE: could not write to {log_path} ({type(exc).__name__}: {exc}). Your actual "
              f"annotation work was already saved separately and is NOT affected by this.")


def find_next_available_subject(box_path):
    """
    NOT USED by the tool since 2026-09-27 (HLU): RAs are assigned files and
    follow the Google tracking spreadsheet, so no subject is suggested. Kept,
    with its tests, in case that changes.

    Mirrors step1_qrs_template.ipynb's "next available subject" autodetect:
    lists every sub-XXX folder directly under box_path, reads
    processing_log.csv (if it exists) to find every subject with ANY log
    entry at all (claimed OR further along), and returns the first
    sub-XXX folder with no such entry. Returns None if box_path doesn't
    exist, no sub-XXX folders are found, or every one found is already
    claimed -- callers should treat None as "nothing to suggest", not an
    error.
    """
    if not box_path or not os.path.isdir(box_path):
        return None

    all_subject_dirs = sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(box_path, "sub-*"))
        if os.path.isdir(p)
    )
    if not all_subject_dirs:
        return None

    log_path = _log_path(box_path)
    if not os.path.exists(log_path):
        return all_subject_dirs[0]

    log_df = pd.read_csv(log_path, dtype=str, keep_default_na=False)
    claimed = set(log_df["Subject"].astype(str)) if "Subject" in log_df.columns else set()

    for sub_dir in all_subject_dirs:
        if sub_dir not in claimed:
            return sub_dir
    return None
