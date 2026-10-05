"""
Which signal an annotation was made on (ppg-plan §4c; physioProcess request
R7). Each reviewed channel's saved entry records:

    "provenance": {"number_of_samples": int,      # tsv rows
                   "pipeline_commit": str | None,  # sidecar Provenance.PipelineCommit (for information)
                   "signal_sha256": str | None}    # column_sha256() below

physioProcess Phase D binds annotations to number_of_samples plus the
reviewed channel's content hash, NOT to the pipeline commit, so a rerun that
leaves a channel's values unchanged (e.g. a sidecar-only fix) keeps that
channel's annotations valid, while any change to a value invalidates them
(maintainer's reply, 2026-09-26, R7).

column_sha256() implements the maintainer's byte-exact definition,
physioProcess/phase_c_20260925/physioprocess_column-hash-reference_20260926.py:
  1. decompress the *_physio.tsv.gz (no header row);
  2. the channel's 0-based column index comes from the sidecar "Columns";
  3. for every line, in file order, remove the line terminator ("\\r\\n" or
     "\\n"), split on "\\t", and take that field's RAW BYTES as written (an
     empty field -- a missing value -- is b"");
  4. SHA-256 over the concatenation of (field bytes + b"\\n") for every line;
  5. lowercase hex.
The line-terminator style doesn't affect the hash; any change to a written
value (including float formatting) does. test_provenance.py checks the
maintainer's test vectors for sub-001 run 1.
"""

import gzip
import hashlib


HASH_DEFINITION = "physioprocess-column-sha256-v1"
"""
Names the hash definition above, stored beside each signal_sha256 so files
stay unambiguous if the definition ever changes (physioProcess maintainer's
request, 2026-09-26). An entry without it was made with v1.
"""


def column_sha256(tsv_gz_path, columns, wanted=None):
    """{column: hex digest} for every column in `wanted` (default: all of `columns`)."""
    wanted = list(columns) if wanted is None else [c for c in wanted if c in columns]
    positions = {c: columns.index(c) for c in wanted}
    hashes = {c: hashlib.sha256() for c in wanted}
    with gzip.open(tsv_gz_path, "rb") as f:
        for line in f:
            if line.endswith(b"\r\n"):
                line = line[:-2]
            elif line.endswith(b"\n"):
                line = line[:-1]
            fields = line.split(b"\t")
            for c, h in hashes.items():
                h.update(fields[positions[c]] + b"\n")
    return {c: h.hexdigest() for c, h in hashes.items()}


def channel_provenance(tsv_gz_path, sidecar, channel_keys, n_samples):
    """
    {channel: provenance dict} for the reviewed channels present in the
    sidecar's Columns. Returns {} for synthetic data (no file) and leaves
    signal_sha256 None for a channel the sidecar doesn't list.
    """
    if not tsv_gz_path or not sidecar:
        return {}
    columns = sidecar.get("Columns") or []
    commit = (sidecar.get("Provenance") or {}).get("PipelineCommit")
    hashes = column_sha256(tsv_gz_path, columns, wanted=[k for k in channel_keys if k in columns])
    return {k: {"number_of_samples": int(n_samples), "pipeline_commit": commit, "signal_sha256": hashes.get(k),
                "signal_sha256_def": HASH_DEFINITION}
            for k in channel_keys}


def new_pipeline_ppg(sidecar):
    """
    Whether this file comes from the pipeline with the PPG timing fix
    (ppg-plan §4b, D3): physioProcess writes Provenance.CTTiming in every
    sidecar since commit 33004c0, which includes the fix (maintainer,
    2026-09-25). Old-pipeline PPG is unusable (sub-085: ~1.4 automated peaks
    per heartbeat). The practice file carries Provenance.Practice.
    sidecar=None means synthetic demo data, which is always allowed.

    CTTiming.Pulse is a dict only when a pulse-timing reconstruction ran;
    otherwise it is a "none (...)" string (e.g. a run with no CareTaker pulse
    file). That string still means the NEW pipeline, so the test is on
    CTTiming being present, not on its type (maintainer's reply, 2026-09-26).
    Whether there is usable PPG to review is a separate test on the data
    (annotation_io.session_checks_before_viewer).
    """
    if sidecar is None:
        return True
    provenance = sidecar.get("Provenance") or {}
    if provenance.get("Practice") is True:
        return True
    return "CTTiming" in provenance


def provenance_mismatch(saved_provenance, current_provenance):
    """
    A plain-language reason when a saved channel's provenance says it was
    made on a different signal than the current file's, else None. A saved
    entry without provenance (made before 2026-09-26) or a current file
    without it (synthetic data) can't be checked here, so returns None;
    check_reprocess_alignment.py covers those.
    """
    if not saved_provenance or not current_provenance:
        return None
    if saved_provenance.get("number_of_samples") != current_provenance.get("number_of_samples"):
        return (f"the file now has {current_provenance.get('number_of_samples')} samples, but the saved review was "
                f"made on {saved_provenance.get('number_of_samples')}")
    saved_def = saved_provenance.get("signal_sha256_def", HASH_DEFINITION)
    current_def = current_provenance.get("signal_sha256_def", HASH_DEFINITION)
    if saved_def != current_def:
        return (f"the saved review's signal hash uses a different definition ({saved_def}) than this version of the "
                f"tool ({current_def}), so it can't be checked")
    saved_hash, current_hash = saved_provenance.get("signal_sha256"), current_provenance.get("signal_sha256")
    if saved_hash and current_hash and saved_hash != current_hash:
        return "the channel's signal values have changed since the review was saved (the file was reprocessed)"
    return None
