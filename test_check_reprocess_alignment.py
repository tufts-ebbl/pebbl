#!/usr/bin/env python3
"""
Tests check_reprocess_alignment.py against cases whose answer is known:
identical files -> ALIGNED; the new file shifted by a known number of samples
-> SHIFTED with exactly that lag; a shorter new file -> ALIGNED_LENGTH_DIFFERS,
flagged when a saved index falls past its end; a missing new file ->
MISSING_NEW; only runs with saved RA work are checked by default, and the
exit code is 1 when anything isn't OK.

Run with: annotate_env\\Scripts\\python.exe test_check_reprocess_alignment.py
"""

import contextlib
import io
import json
import os
import shutil
import tempfile

import check_reprocess_alignment as cra
from physio_io import build_run_path, generate_synthetic_demo


def write_run(root, subject, run, df, sfreq):
    path = build_run_path(root, subject, run)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep="\t", index=False, header=False, compression="gzip")
    with open(path.replace("_physio.tsv.gz", "_physio.json"), "w") as f:
        json.dump({"Columns": list(df.columns), "SamplingFrequency": sfreq}, f)
    return path


def write_ra_file(root, subject, run, indices):
    folder = os.path.dirname(build_run_path(root, subject, run))
    with open(os.path.join(folder, f"sub-{subject}_ses-run{run}_task-sdi_annotations_zz.json"), "w") as f:
        json.dump({"initials": "zz", "channels": {"ecg": {"mode": "point", "indices": indices}}}, f)


def main():
    tmp = tempfile.mkdtemp(prefix="align_check_")
    try:
        old_root, new_root = os.path.join(tmp, "old"), os.path.join(tmp, "new")
        df, fs = generate_synthetic_demo(duration_sec=150, seed=7)
        n = len(df)

        write_run(old_root, "001", "1", df, fs)            # identical
        write_run(new_root, "001", "1", df, fs)
        write_ra_file(old_root, "001", "1", [1000, 2000])

        write_run(old_root, "002", "1", df, fs)            # new file = old shifted: 250 samples trimmed off the front
        write_run(new_root, "002", "1", df.iloc[250:].reset_index(drop=True), fs)
        write_ra_file(old_root, "002", "1", [1000])

        write_run(old_root, "003", "1", df, fs)            # new file 30 s shorter at the end
        write_run(new_root, "003", "1", df.iloc[:n - 30000].reset_index(drop=True), fs)
        write_ra_file(old_root, "003", "1", [n - 5000])    # a saved index past the new end

        write_run(old_root, "004", "1", df, fs)            # no new file
        write_ra_file(old_root, "004", "1", [1000])

        write_run(old_root, "005", "1", df, fs)            # no RA work: skipped by default
        write_run(new_root, "005", "1", df, fs)

        print("1. Only runs with saved RA work are checked by default...")
        assert sorted(cra.find_runs_with_ra_work(old_root)) == [("001", "1"), ("002", "1"), ("003", "1"), ("004", "1")]
        print("   OK")

        print("2. Known cases get the known status...")
        expected = {"001": ("ALIGNED", 0), "002": ("SHIFTED", -250), "003": ("ALIGNED_LENGTH_DIFFERS", 0),
                    "004": ("MISSING_NEW", None)}
        work = cra.find_runs_with_ra_work(old_root)
        for subject, (status, lag) in expected.items():
            result = cra.compare_run(build_run_path(old_root, subject, "1"), build_run_path(new_root, subject, "1"),
                                     saved_max_index=cra.saved_indices(work[(subject, "1")]))
            assert result["status"] == status, (subject, result)
            if lag is not None:
                assert result["lag_samples"] == lag, (subject, result)
            print(f"   OK: sub-{subject} -> {result['status']} (lag {result.get('lag_samples')}, r={result.get('correlations')})")
        assert cra.compare_run(build_run_path(old_root, "003", "1"), build_run_path(new_root, "003", "1"),
                               saved_max_index=n - 5000)["saved_index_past_end"] is True
        print("   OK: sub-003's saved index past the new end is flagged")

        print("2b. Negative control: an UNRELATED recording (different seed) is UNCLEAR, never ALIGNED; "
              "a slightly re-filtered ECG (added noise) still reads as ALIGNED...")
        other, _ = generate_synthetic_demo(duration_sec=150, seed=99)
        write_run(new_root, "006", "1", other, fs)
        write_run(old_root, "006", "1", df, fs)
        unrelated = cra.compare_run(build_run_path(old_root, "006", "1"), build_run_path(new_root, "006", "1"))
        assert unrelated["status"] == "UNCLEAR", unrelated
        import numpy as np
        noisy = df.copy()
        noisy["ecg"] = noisy["ecg"] + np.random.default_rng(2024).normal(0, 0.05 * noisy["ecg"].std(), n)
        write_run(new_root, "007", "1", noisy, fs)
        write_run(old_root, "007", "1", df, fs)
        assert cra.compare_run(build_run_path(old_root, "007", "1"), build_run_path(new_root, "007", "1"))["status"] == "ALIGNED"
        print(f"   OK: unrelated -> UNCLEAR (r={unrelated['correlations']}); noisy copy -> ALIGNED")

        print("3. The command line reports, writes a CSV, and exits 1 when anything isn't OK...")
        report = os.path.join(tmp, "report.csv")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cra.main(["--old-root", old_root, "--new-root", new_root, "--report", report])
        assert code == 1 and os.path.exists(report), out.getvalue()
        assert "1 of 4 run(s) OK" in out.getvalue() and "sub-002 run 1" in out.getvalue(), out.getvalue()
        with contextlib.redirect_stdout(io.StringIO()):
            assert cra.main(["--old-root", os.path.join(tmp, "old"), "--new-root", new_root, "--all"]) == 1
            # A folder that doesn't exist stops with a plain message; it isn't "nothing to check" (2026-09-27).
            for bad in (["--old-root", os.path.join(tmp, "no_such_folder"), "--new-root", new_root],
                        ["--old-root", os.path.join(tmp, "old"), "--new-root", os.path.join(tmp, "no_such_folder")]):
                try:
                    cra.main(bad)
                    raise AssertionError(f"{bad} accepted")
                except SystemExit as e:
                    assert "folder not found" in str(e) and "Nothing was checked" in str(e), e
        print("   OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL REPROCESS ALIGNMENT TESTS PASSED.")


if __name__ == "__main__":
    main()
