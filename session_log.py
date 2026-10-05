"""
Session log (HLU, 2026-10-03): everything physio_review.py prints during a
session -- the equivalent command, banners, notes, warnings, the session
summary's lines, errors with their tracebacks -- is also written to

    <run folder>/<file_stem>_annotations_<initials>.log

next to that RA's <file_stem>_annotations_<initials>.json (same name, so the
two sort side by side). One file per RA per subject-run, holding every Step
1, 2 and 3 session that RA ran on that run; each session is appended under a
header and closed with a footer saying how it ended. The reconciler's Step 3
sessions go in the reconciler's own log.

Capture starts when the program starts (so nothing printed early is lost)
and is held in memory until the session knows the initials and run folder;
then the log file is opened in append mode, gets the header plus everything
so far, and from then on each line is written as it is printed, so a session
that crashes or is closed still leaves its log behind. The file sits in the
same folder as the annotation file, so the local-copy lifecycle pushes it to
Box with it (or a leftover local copy's next push does). The console output
itself is unchanged.
"""

import io
import os
import sys
from datetime import datetime

RULE = "=" * 78

_ACTIVE = None
"""The SessionLog that is capturing right now (set by start(), cleared by close())."""


def log_only(lines):
    """
    Writes lines to the active session log only, not the console: e.g. what
    the save dialog showed and what the RA chose there, which never appear in
    the terminal. Does nothing when no session log is active.
    """
    if _ACTIVE is not None:
        _ACTIVE.record("\n".join(lines) + "\n")


def session_log_path(out_dir, file_stem, initials):
    """The RA's log for this run: same name as their annotation file, ending .log."""
    return os.path.join(out_dir, f"{file_stem}_annotations_{initials}.log")


class _Tee:
    """Writes to the real stream and to the session log; everything else is the real stream's."""

    def __init__(self, stream, log):
        self._stream = stream
        self._log = log

    def write(self, text):
        written = self._stream.write(text)
        self._log.record(text)
        return len(text) if written is None else written

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


class SessionLog:
    """
    start() begins capturing stdout and stderr; attach() opens one or more log
    files (writing the header and everything captured so far); close() writes
    the footer, closes the files and restores the streams. close() is safe to
    call more than once, and works when nothing was ever attached.
    """

    def __init__(self):
        self.started = datetime.now()
        self.paths = []
        self._buffer = io.StringIO()
        self._files = []
        self._saved_streams = None

    def start(self):
        global _ACTIVE
        _ACTIVE = self
        if self._saved_streams is None:
            self._saved_streams = (sys.stdout, sys.stderr)
            sys.stdout = _Tee(sys.stdout, self)
            sys.stderr = _Tee(sys.stderr, self)

    def record(self, text):
        self._buffer.write(text)
        for f in self._files:
            try:
                f.write(text)
            except OSError:
                pass  # a log that can't be written must never stop the review

    def flush(self):
        for f in self._files:
            try:
                f.flush()
            except OSError:
                pass

    def attach(self, targets, initials, header_lines=()):
        """
        Opens <out_dir>/<file_stem>_annotations_<initials>.log for each
        (out_dir, file_stem) in targets (each path once), writing a header
        and everything printed so far. Returns the newly opened paths. A log
        that can't be opened is reported once and skipped.
        """
        opened = []
        header = "\n".join([RULE, f"PEBBL session started {self.started:%Y-%m-%d %H:%M:%S} -- {initials}",
                            *header_lines, RULE]) + "\n"
        for out_dir, file_stem in targets:
            path = session_log_path(out_dir, file_stem, initials)
            if path in self.paths:
                continue
            try:
                os.makedirs(out_dir, exist_ok=True)
                f = open(path, "a", encoding="utf-8")
                f.write(header + self._buffer.getvalue())
                f.flush()
            except OSError as e:
                print(f"NOTE: couldn't write the session log {path} ({e}); the review itself is unaffected.")
                continue
            self._files.append(f)
            self.paths.append(path)
            opened.append(path)
        return opened

    def close(self, footer_lines=()):
        global _ACTIVE
        if _ACTIVE is self:
            _ACTIVE = None
        if self._saved_streams is not None:
            sys.stdout, sys.stderr = self._saved_streams
            self._saved_streams = None
        if not self._files:
            return
        footer = "\n".join([RULE, f"Session ended {datetime.now():%Y-%m-%d %H:%M:%S}", *footer_lines, RULE]) + "\n\n"
        for f in self._files:
            try:
                f.write("\n" + footer)
                f.close()
            except OSError:
                pass
        self._files = []


def footer_lines(info, exit_code):
    """How the session ended, from physio_review.main()'s info dict."""
    step = {"1": "Step 1 (ECG template)", "2": "Step 2 (review)", "3": "Step 3 (reconcile)"}.get(
        str(info.get("stage")), "Step not reached")
    run = info.get("run")
    channels = ", ".join(info.get("channels") or []) or "none"
    if info.get("ended_early"):
        outcome = f"ended early (exit code {exit_code})"
    elif info.get("discarded"):
        outcome = "discarded (nothing saved)"
    elif info.get("saved"):
        outcome = "saved, marked finished" if info.get("finished") else "saved, still in progress"
    else:
        outcome = "nothing saved"
    return [f"{step}" + (f", run {run}" if run else "") + f"; channels: {channels}", f"Outcome: {outcome}"]
