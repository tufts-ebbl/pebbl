#!/usr/bin/env python3
"""
Tests provenance.py's per-channel content hash against the byte-exact
definition shared with physioProcess (request R7; reference script
physioProcess/phase_c_20260925/physioprocess_column-hash-reference_20260926.py):

  - known answers: sub-001 run 1's ppg and ppg_peaks hash to the maintainer's
    published test vectors (run 10, 1d74b15: 4ec43a9a88a6...2a893,
    1f0579c29e51...43263, confirmed by the maintainer from both the bids tree
    and this copy on 2026-10-03; runs 8/9 were 835251095b25...9030f,
    8b89cd443bb8...0f44c);
  - CRLF and LF line endings give the same hash (negative control for the
    line-terminator rule);
  - changing one written value, even only its formatting ("1.0" -> "1"),
    changes the hash; an empty field (a missing value) hashes as b"";
  - provenance_mismatch() flags a changed sample count or hash, and stays
    silent when either side has no provenance.

Run with: annotate_env\\Scripts\\python.exe test_provenance.py
"""

import gzip
import hashlib
import json
import os
import shutil
import tempfile

from provenance import channel_provenance, column_sha256, provenance_mismatch

HERE = os.path.dirname(os.path.abspath(__file__))
# The run-7 fixture test_input/sub-001_* was retired on 2026-09-26 (HLU); the
# copy in test_input/run8/ is used when it's absent. Since 2026-10-03 that
# copy is run 10's (physioProcess 1d74b15): sub-001 r1 lost CareTaker pulse
# samples, so its ppg/ppg_peaks changed from run 8/9's vectors (835251095b25...
# / 8b89cd443bb8...) to the run-10 vectors below, confirmed independently by
# the physioProcess maintainer (physioprocess-column-sha256-v1).
SUB001 = next((p for p in (os.path.join(HERE, "test_input", "sub-001_ses-run1_task-sdi_physio.tsv.gz"),
                           os.path.join(HERE, "test_input", "run8", "sub-001_ses-run1_task-sdi_physio.tsv.gz"))
               if os.path.exists(p)), os.path.join(HERE, "test_input", "sub-001_ses-run1_task-sdi_physio.tsv.gz"))
SUB001_VECTORS = {
    "ppg": "4ec43a9a88a65503cc0da8f9115ec6788815f480667a9feacfcffef08ff2a893",
    "ppg_peaks": "1f0579c29e5108cd524f171446dc7eca95859f00e7f6c8e11ecb246dfb443263",
}


def write_gz(path, text_bytes):
    with gzip.open(path, "wb") as f:
        f.write(text_bytes)


def main():
    print("1. sub-001 run 1 matches the physioProcess maintainer's test vectors...")
    if os.path.exists(SUB001):
        with open(SUB001.replace("_physio.tsv.gz", "_physio.json"), encoding="utf-8-sig") as f:
            sidecar = json.load(f)
        hashes = column_sha256(SUB001, sidecar["Columns"], wanted=["ppg", "ppg_peaks"])
        assert hashes["ppg"].startswith("4ec43a9a88a6") and hashes["ppg"].endswith("2a893"), hashes
        assert hashes["ppg_peaks"].startswith("1f0579c29e51") and hashes["ppg_peaks"].endswith("43263"), hashes
        assert hashes == SUB001_VECTORS, hashes
        prov = channel_provenance(SUB001, sidecar, ["ppg"], 742091)
        assert prov["ppg"] == {"number_of_samples": 742091, "pipeline_commit": sidecar["Provenance"]["PipelineCommit"],
                               "signal_sha256": SUB001_VECTORS["ppg"],
                               "signal_sha256_def": "physioprocess-column-sha256-v1"}, prov
        print(f"   OK: ppg {hashes['ppg'][:12]}...{hashes['ppg'][-5:]}, "
              f"ppg_peaks {hashes['ppg_peaks'][:12]}...{hashes['ppg_peaks'][-5:]}")
    else:
        print("   SKIPPED: test_input/sub-001 not present")

    tmp = tempfile.mkdtemp()
    try:
        print("2. The definition itself: line endings don't matter; values and formatting do; empty = b''...")
        cols = ["a", "b"]
        lf, crlf = os.path.join(tmp, "lf.tsv.gz"), os.path.join(tmp, "crlf.tsv.gz")
        write_gz(lf, b"1.0\t\n2.5\t1\n")
        write_gz(crlf, b"1.0\t\r\n2.5\t1\r\n")
        h_lf, h_crlf = column_sha256(lf, cols), column_sha256(crlf, cols)
        assert h_lf == h_crlf, (h_lf, h_crlf)
        assert h_lf["a"] == hashlib.sha256(b"1.0\n2.5\n").hexdigest()
        assert h_lf["b"] == hashlib.sha256(b"\n1\n").hexdigest(), "an empty field hashes as the empty byte string"
        reformatted = os.path.join(tmp, "fmt.tsv.gz")
        write_gz(reformatted, b"1\t\n2.5\t1\n")
        assert column_sha256(reformatted, cols)["a"] != h_lf["a"] and column_sha256(reformatted, cols)["b"] == h_lf["b"]
        print("   OK")

        print("3. provenance_mismatch()...")
        saved = {"number_of_samples": 10, "signal_sha256": "x"}
        assert provenance_mismatch(saved, {"number_of_samples": 10, "signal_sha256": "x"}) is None
        assert "samples" in provenance_mismatch(saved, {"number_of_samples": 11, "signal_sha256": "x"})
        assert "changed" in provenance_mismatch(saved, {"number_of_samples": 10, "signal_sha256": "y"})
        assert provenance_mismatch(None, saved) is None and provenance_mismatch(saved, None) is None
        # The hash definition is tagged; an entry without the tag is v1; a different tag can't be compared.
        tagged = {**saved, "signal_sha256_def": "physioprocess-column-sha256-v1"}
        assert provenance_mismatch(saved, tagged) is None
        assert "different definition" in provenance_mismatch({**saved, "signal_sha256_def": "v2"}, tagged)
        assert channel_provenance(None, {}, ["ppg"], 5) == {}
        print("   OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL PROVENANCE TESTS PASSED.")


if __name__ == "__main__":
    main()
