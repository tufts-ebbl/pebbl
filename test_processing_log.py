#!/usr/bin/env python3
"""
Non-interactive tests for processing_log.py: appending entries to
processing_log.csv in the same shape the existing notebooks already write,
suggesting the next unclaimed subject, and mirroring to a local read-only
snapshot (matching the notebooks' own processing_log_READONLY_SNAPSHOT.csv
convention).

Run with: annotate_env\\Scripts\\python.exe test_processing_log.py
"""

import os
import shutil
import tempfile
from unittest.mock import patch

import pandas as pd

import processing_log
from processing_log import append_processing_log_entry, find_next_available_subject


def main():
    tmp_dir = tempfile.mkdtemp(prefix="processing_log_test_")
    # append_processing_log_entry() also mirrors to LOCAL_SNAPSHOT_DIR
    # (matching the real notebooks' processing_log_READONLY_SNAPSHOT.csv
    # convention). That constant defaults to the REAL physioCorrection/
    # folder -- redirected here for this whole test run so these fake/temp
    # box_path calls never touch the user's actual local snapshot file.
    snapshot_dir = tempfile.mkdtemp(prefix="processing_log_snapshot_", dir=tmp_dir)
    try:
        with patch.object(processing_log, "LOCAL_SNAPSHOT_DIR", snapshot_dir):
            print("1. Appending a 'claimed' entry to a brand-new processing_log.csv...")
            append_processing_log_entry(tmp_dir, "085", None, "hlu", "claimed")
            log_path = os.path.join(tmp_dir, "processing_log.csv")
            assert os.path.exists(log_path)
            df = pd.read_csv(log_path, dtype=str, keep_default_na=False)
            assert list(df.columns) == ["Subject", "Run", "Initials", "Step", "Notes", "Timestamp", "Value_ranges"]
            assert len(df) == 1
            assert df.iloc[0]["Subject"] == "sub-085", "bare subject id must be normalized to 'sub-085'"
            assert df.iloc[0]["Run"] == "", "a 'claimed' entry (run=None) must leave Run blank, not 'None'/'nan'"
            print(f"   OK: {df.iloc[0].to_dict()}")

            print("2. Appending a run-specific entry (already-prefixed 'sub-085') appends, doesn't clobber...")
            append_processing_log_entry(tmp_dir, "sub-085", "1", "hlu", "created QRS template", "Shifted 20s")
            df = pd.read_csv(log_path, dtype=str, keep_default_na=False)
            assert len(df) == 2
            assert df.iloc[1]["Subject"] == "sub-085", "already-prefixed subject shouldn't become 'sub-sub-085'"
            assert df.iloc[1]["Run"] == "1"
            assert df.iloc[1]["Notes"] == "Shifted 20s"
            print(f"   OK: {df.iloc[1].to_dict()}")

            print("3. A generalized multi-channel Step B entry logs the channels reviewed...")
            append_processing_log_entry(tmp_dir, "085", "1", "hlu", "ecg, rsp, emg_cor reviewed")
            df = pd.read_csv(log_path, dtype=str, keep_default_na=False)
            assert df.iloc[2]["Step"] == "ecg, rsp, emg_cor reviewed"
            assert df.iloc[2]["Notes"] == "", "no comment given -> blank Notes, not 'nan'"
            print(f"   OK: {df.iloc[2]['Step']!r}")

            print("3a. A later write APPENDS one line -- earlier bytes are left exactly as they were "
                  "(audit H5: no whole-file rewrite for another RA's save to race with) -- and value "
                  "ranges go in their own Value_ranges column, not Notes...")
            before = open(log_path, "rb").read()
            append_processing_log_entry(tmp_dir, "085", "1", "hlu", "eda reviewed -- finished", "ok",
                                        value_ranges="EDA: 0.21-0.89 uS (mean 0.82)")
            after = open(log_path, "rb").read()
            assert after.startswith(before), "an append must not rewrite earlier rows"
            df = pd.read_csv(log_path, dtype=str, keep_default_na=False)
            assert df.iloc[3]["Notes"] == "ok" and df.iloc[3]["Value_ranges"] == "EDA: 0.21-0.89 uS (mean 0.82)"
            print("   OK: appended in place; Notes and Value_ranges kept separate")

            print("3a2. A log that predates the Value_ranges column gains it once, keeping every old row...")
            legacy_dir = tempfile.mkdtemp(prefix="processing_log_legacy_", dir=tmp_dir)
            legacy_path = os.path.join(legacy_dir, "processing_log.csv")
            pd.DataFrame([{"Subject": "sub-001", "Run": "1", "Initials": "jkl", "Step": "R peaks inspected and corrected",
                           "Notes": "old", "Timestamp": "1/2/2026 9:05"}]).to_csv(legacy_path, index=False)
            # Separate snapshot dir: the snapshot mirrors whichever log was written
            # last, and step 3b below checks it against the main log.
            with patch.object(processing_log, "LOCAL_SNAPSHOT_DIR", legacy_dir):
                append_processing_log_entry(legacy_dir, "001", "2", "hlu", "ecg reviewed -- not finished")
            legacy_df = pd.read_csv(legacy_path, dtype=str, keep_default_na=False)
            assert list(legacy_df["Step"]) == ["R peaks inspected and corrected", "ecg reviewed -- not finished"]
            assert "Value_ranges" in legacy_df.columns and legacy_df.iloc[0]["Value_ranges"] == ""
            print("   OK: old row kept, column added")

            print("3b. Every append_processing_log_entry() call mirrors the box-hosted log to a "
                  "local READONLY snapshot (matching step1/step2's own processing_log_READONLY_"
                  "SNAPSHOT.csv convention) -- the snapshot's content must match the box copy "
                  "exactly after each write...")
            snapshot_path = os.path.join(snapshot_dir, "processing_log_READONLY_SNAPSHOT.csv")
            assert os.path.exists(snapshot_path), "snapshot should have been created by step 1's call already"
            box_content = open(log_path).read()
            snapshot_content = open(snapshot_path).read()
            assert box_content == snapshot_content, "snapshot must be an exact mirror of the box-hosted log"
            print(f"   OK: {os.path.basename(snapshot_path)} exactly mirrors {os.path.basename(log_path)}")

            print("4. find_next_available_subject(): no processing_log.csv yet -> first sub-XXX folder found...")
            tmp_dir_2 = tempfile.mkdtemp(prefix="processing_log_next_", dir=tmp_dir)
            os.makedirs(os.path.join(tmp_dir_2, "sub-003"))
            os.makedirs(os.path.join(tmp_dir_2, "sub-001"))
            suggested = find_next_available_subject(tmp_dir_2)
            assert suggested == "sub-001", f"expected the alphabetically-first folder, got {suggested}"
            print(f"   OK: {suggested}")

            print("5. find_next_available_subject(): skips any subject with ANY log entry at all...")
            append_processing_log_entry(tmp_dir_2, "001", None, "hlu", "claimed")
            suggested = find_next_available_subject(tmp_dir_2)
            assert suggested == "sub-003", f"expected sub-003 (sub-001 is claimed), got {suggested}"
            print(f"   OK: {suggested}")

            print("6. find_next_available_subject(): every found subject already claimed -> None...")
            append_processing_log_entry(tmp_dir_2, "003", None, "hlu", "claimed")
            suggested = find_next_available_subject(tmp_dir_2)
            assert suggested is None
            print("   OK: None (nothing left to suggest)")

            print("7. find_next_available_subject(): a nonexistent box_path returns None, not an error...")
            assert find_next_available_subject(os.path.join(tmp_dir, "does-not-exist")) is None
            print("   OK")

            print("8. The snapshot updates again after a LATER write elsewhere makes the box copy "
                  "more current (simulating someone else processing data)...")
            append_processing_log_entry(tmp_dir_2, "999", None, "xyz", "claimed")  # a DIFFERENT box_path
            other_log_path = os.path.join(tmp_dir_2, "processing_log.csv")
            snapshot_content_after = open(snapshot_path).read()
            assert snapshot_content_after == open(other_log_path).read(), (
                "the ONE shared local snapshot should reflect whichever box_path was written to most "
                "recently, exactly like the real notebooks sharing a single snapshot file"
            )
            assert snapshot_content_after != box_content, "snapshot should have changed, not stayed stale"
            print("   OK: snapshot reflects the most recently written box-hosted log")

        print("9. The snapshot's default place (HLU, 2026-10-05): the user's home folder for an installed "
              "copy (its own git clone, e.g. the shared lab install), the folder above for a development "
              "checkout...")
        install = os.path.join(tmp_dir, "Public", "Downloads", "pebbl")
        os.makedirs(os.path.join(install, ".git"))
        assert processing_log.default_snapshot_dir(install) == os.path.expanduser("~"), \
            "never the shared folder above a lab install"
        dev = os.path.join(tmp_dir, "physioCorrection", "multisignal_annotation")
        os.makedirs(dev)
        assert processing_log.default_snapshot_dir(dev) == os.path.join(tmp_dir, "physioCorrection")
        here = os.path.dirname(os.path.abspath(processing_log.__file__))
        assert processing_log.LOCAL_SNAPSHOT_DIR == processing_log.default_snapshot_dir(here)
        print("   OK: home folder for an install; the folder above for a development checkout")

        print("\nALL PROCESSING_LOG TESTS PASSED.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
