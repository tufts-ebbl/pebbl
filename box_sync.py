"""
Automates the existing notebooks' full local-copy lifecycle for
physio_review.py: copy a subject's Box-hosted files down to an RA-chosen
local folder (--local-path, prompted for explicitly just like --box-path
-- RAs already do this manually today, so this keeps the same two-folder
mental model rather than hiding a temp directory from them), do the
session's actual work against that local copy, push back whatever files
this session created or changed, confirm each one landed on Box, and only
then delete THAT SUBJECT's local subfolder (never the --local-path folder
itself, which an RA typically reuses across sessions) -- so RA laptops
never retain a persistent copy of subject data. Modeled directly on
step1_qrs_template.ipynb/step2_physio_correction.ipynb's own workflow
(copy locally -> work -> push to Box -> confirm -> delete local, but never
the processing_log snapshot), confirmed piece-by-piece with the user
before building.

This is a safe fit with everything else in this tool because
load_physio_tsv() already reads the whole recording into memory in ONE
call -- the long interactive session that follows never touches the
source file again, so working from a local copy changes nothing about
correctness, only where the bytes physically sit while editing happens.

"Confirming" a push only checks that the Box Drive mount has the file with
a fresh modification time -- it confirms the LOCAL WRITE succeeded, not
that Box's cloud servers have finished syncing it (that isn't practically
checkable from plain file I/O; a false confirmation there is possible in
principle, just not something this tool can detect). If confirmation
fails for ANY file, NOTHING is deleted -- the local copy is left in place
and a clear message says exactly where, since cleanup is irreversible.
"""

import os
import shutil
import time

CONFIRM_TOLERANCE_SECONDS = 300


def _subject_dir_name(subject):
    return subject if str(subject).startswith("sub-") else f"sub-{subject}"


UNPUSHED_MTIME_TOLERANCE_SECONDS = 2

PIPELINE_FILE_SUFFIXES = ("_physio.tsv.gz", "_physio.json")
"""
physioProcess's own output files. This tool never changes them, so they are
never pushed back to Box: a leftover local copy from before a reprocess
holds the OLD files, and pushing one would overwrite Box's current file
(2026-09-26 safety fix, after run 8).
"""


def is_pipeline_file(relpath):
    return os.path.basename(relpath).endswith(PIPELINE_FILE_SUFFIXES)


def find_unpushed_local_files(box_path, subject, local_root, tolerance_seconds=UNPUSHED_MTIME_TOLERANCE_SECONDS):
    """
    Looks for a leftover local_root/sub-<subject>/ from an earlier session
    that never finished pushing back to Box (e.g. the session was
    interrupted, crashed, or a push couldn't be confirmed). Returns the
    relpaths (relative to local_root, sorted) of every local file that is
    either missing on Box or newer on the local side than on Box -- i.e.
    work that exists ONLY locally.

    Must be checked BEFORE copy_subject_tree_to_local(), which overwrites
    local files with Box's copies: without this check, the next session's
    copy-down silently replaced unpushed local work with Box's older
    version (audit finding C2). A file copied down and never touched keeps
    Box's mtime exactly (copy2); a file pushed back gets a NEWER Box mtime
    (plain copy); so "local newer than Box" can only mean an unpushed local
    change. physioProcess's own files (is_pipeline_file) are never listed:
    the tool doesn't change them, so a "newer" local one is never RA work.
    """
    subject_dir = os.path.join(local_root, _subject_dir_name(subject))
    if not os.path.isdir(subject_dir):
        return []
    unpushed = []
    for dirpath, _dirnames, filenames in os.walk(subject_dir):
        for filename in filenames:
            local_file = os.path.join(dirpath, filename)
            relpath = os.path.relpath(local_file, local_root)
            if is_pipeline_file(relpath):
                continue
            box_file = os.path.join(box_path, relpath)
            if (not os.path.exists(box_file)
                    or os.path.getmtime(local_file) > os.path.getmtime(box_file) + tolerance_seconds):
                unpushed.append(relpath)
    return sorted(unpushed)


def copy_subject_tree_to_local(box_path, subject, local_root):
    """
    Copies box_path/sub-<subject>/ into local_root/sub-<subject>/,
    preserving modification times (shutil.copy2) so find_changed_files()
    can later tell a genuinely-rewritten file from one that's simply been
    copied around. Tolerant of a partially- or non-existent source tree
    (a brand-new subject, or only one run collected so far).

    Returns a {relpath: mtime} snapshot of everything copied -- relpath is
    relative to local_root -- the "before" state to diff against later.
    """
    source_dir = os.path.join(box_path, _subject_dir_name(subject))
    dest_dir = os.path.join(local_root, _subject_dir_name(subject))

    snapshot = {}
    if not os.path.isdir(source_dir):
        os.makedirs(dest_dir, exist_ok=True)
        return snapshot

    for dirpath, _dirnames, filenames in os.walk(source_dir):
        rel_dir = os.path.relpath(dirpath, box_path)
        local_dirpath = os.path.join(local_root, rel_dir)
        os.makedirs(local_dirpath, exist_ok=True)
        for filename in filenames:
            src_file = os.path.join(dirpath, filename)
            dst_file = os.path.join(local_dirpath, filename)
            shutil.copy2(src_file, dst_file)
            snapshot[os.path.relpath(dst_file, local_root)] = os.path.getmtime(dst_file)

    return snapshot


def find_changed_files(local_root, subject, before_snapshot):
    """
    Walks local_root/sub-<subject>/ as it stands NOW and returns the
    relpaths (relative to local_root, sorted) of every file that's new or
    has a different modification time than before_snapshot recorded --
    i.e. everything this session actually created or rewrote. Files never
    get deleted by this tool's own saves, so there's no "removed" case to
    handle here.
    """
    subject_dir = os.path.join(local_root, _subject_dir_name(subject))
    if not os.path.isdir(subject_dir):
        return []

    changed = []
    for dirpath, _dirnames, filenames in os.walk(subject_dir):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            relpath = os.path.relpath(full_path, local_root)
            if is_pipeline_file(relpath):
                continue  # never pushed back (see PIPELINE_FILE_SUFFIXES)
            if before_snapshot.get(relpath) != os.path.getmtime(full_path):
                changed.append(relpath)
    return sorted(changed)


def push_changed_files_to_box(local_root, box_path, changed_relpaths, tolerance_seconds=CONFIRM_TOLERANCE_SECONDS):
    """
    Copies each changed_relpaths entry from local_root to the SAME
    relative path under box_path, then confirms it landed by checking the
    Box destination exists with a modification time within
    tolerance_seconds of right now -- see this module's docstring for
    exactly what that confirms and what it can't.

    Returns (confirmed, failed) -- both sorted lists of relpaths.
    physioProcess's own files are never copied, whatever the caller passes.
    """
    confirmed, failed = [], []
    for relpath in changed_relpaths:
        if is_pipeline_file(relpath):
            continue
        src = os.path.join(local_root, relpath)
        dst = os.path.join(box_path, relpath)
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            # Plain copy (not copy2): the destination's mtime must reflect
            # THIS write, so the "is it fresh?" check below is meaningful.
            # copy2 would preserve the local file's own (possibly old, from
            # earlier in this session) mtime and cause a false failure here.
            shutil.copy(src, dst)
            if os.path.exists(dst) and abs(time.time() - os.path.getmtime(dst)) <= tolerance_seconds:
                confirmed.append(relpath)
            else:
                failed.append(relpath)
        except OSError:
            failed.append(relpath)
    return sorted(confirmed), sorted(failed)


def cleanup_local_copy(local_root, subject):
    """
    Deletes local_root/sub-<subject>/ only -- NOT local_root itself, since
    that's a folder the RA typically reuses session after session (an
    explicit --local-path they enter, matching the existing notebooks'
    own local-path prompt, not a one-off temp directory this tool
    invented). Best-effort: never raises.
    """
    shutil.rmtree(os.path.join(local_root, _subject_dir_name(subject)), ignore_errors=True)
