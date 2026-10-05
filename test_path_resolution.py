#!/usr/bin/env python3
"""
Tests resolve_real_input_path(): direct --input passthrough, path
construction from box-path/subject/run (with normalization), interactive
prompting for whichever pieces are missing, and a clear error when the
resulting file doesn't exist.

Run with: annotate_env\\Scripts\\python.exe test_path_resolution.py
"""

import os
import shutil
import tempfile
from unittest.mock import patch

from physio_io import resolve_real_input_path, resolve_subject_run_paths


def make_fake_box_tree(root, subject="001", run="1"):
    """Creates <root>/sub-{subject}/ses-run{run}/beh/sub-{subject}_ses-run{run}_task-sdi_physio.tsv.gz"""
    beh_dir = os.path.join(root, f"sub-{subject}", f"ses-run{run}", "beh")
    os.makedirs(beh_dir, exist_ok=True)
    file_path = os.path.join(beh_dir, f"sub-{subject}_ses-run{run}_task-sdi_physio.tsv.gz")
    with open(file_path, "wb") as f:
        f.write(b"")
    return file_path


def main():
    tmp_dir = tempfile.mkdtemp(prefix="path_resolution_smoke_")
    try:
        print("1. --input passed directly bypasses box-path/subject/run entirely...")
        result = resolve_real_input_path("/some/explicit/path.tsv.gz", None, None, None)
        assert result == "/some/explicit/path.tsv.gz"
        print("   OK")

        print("2. box-path + subject + run (already zero-padded) construct the correct path, no prompting...")
        expected_path = make_fake_box_tree(tmp_dir, subject="001", run="1")
        with patch("builtins.input") as mock_input:
            result = resolve_real_input_path(None, tmp_dir, "001", "1")
            mock_input.assert_not_called()
        assert result == expected_path, f"expected {expected_path}, got {result}"
        print(f"   OK: {result}")

        print("3. Subject/run normalization: 'sub-5' and 'run2' -> sub-005/ses-run2...")
        expected_path_2 = make_fake_box_tree(tmp_dir, subject="005", run="2")
        result = resolve_real_input_path(None, tmp_dir, "sub-5", "run2")
        assert result == expected_path_2, f"expected {expected_path_2}, got {result}"
        print(f"   OK: {result}")

        print("4. Missing subject and run prompts for exactly those two, in order...")
        with patch("builtins.input", side_effect=["001", "1"]) as mock_input:
            result = resolve_real_input_path(None, tmp_dir, None, None)
        assert result == expected_path, f"expected {expected_path}, got {result}"
        assert mock_input.call_count == 2
        print("   OK: prompted for subject then run (box-path was already provided, so not prompted)")

        print("5. Nothing provided at all prompts for all three, in order (box-path, subject, run)...")
        with patch("builtins.input", side_effect=[tmp_dir, "001", "1"]) as mock_input:
            result = resolve_real_input_path(None, None, None, None)
        assert result == expected_path
        assert mock_input.call_count == 3
        print("   OK: prompted for box-path, subject, run")

        print("6. A well-formed but nonexistent file raises a clear SystemExit...")
        try:
            resolve_real_input_path(None, tmp_dir, "999", "1")
            raise AssertionError("expected SystemExit for a nonexistent subject/run")
        except SystemExit as e:
            assert "999" in str(e) and "No file found" in str(e)
            print(f"   OK: {e}")

        print("7. resolve_subject_run_paths finds both runs when both exist, no prompting for run...")
        subj2_run1 = make_fake_box_tree(tmp_dir, subject="010", run="1")
        subj2_run2 = make_fake_box_tree(tmp_dir, subject="010", run="2")
        with patch("builtins.input") as mock_input:
            paths = resolve_subject_run_paths(None, None, tmp_dir, "010")
            mock_input.assert_not_called()
        assert paths == {"1": subj2_run1, "2": subj2_run2}, paths
        print(f"   OK: {paths}")

        print("8. resolve_subject_run_paths handles only run 1 existing (run 2 not yet collected)...")
        subj3_run1 = make_fake_box_tree(tmp_dir, subject="011", run="1")
        paths = resolve_subject_run_paths(None, None, tmp_dir, "011")
        assert paths == {"1": subj3_run1, "2": None}, paths
        print(f"   OK: {paths}")

        print("9. resolve_subject_run_paths errors clearly when NEITHER run exists...")
        try:
            resolve_subject_run_paths(None, None, tmp_dir, "999")
            raise AssertionError("expected SystemExit when neither run exists")
        except SystemExit as e:
            assert "No run 1 or run 2" in str(e)
            print(f"   OK: {e}")

        print("10. resolve_subject_run_paths prompts for box-path/subject when omitted, never for run...")
        with patch("builtins.input", side_effect=[tmp_dir, "010"]) as mock_input:
            paths = resolve_subject_run_paths(None, None, None, None)
        assert paths == {"1": subj2_run1, "2": subj2_run2}, paths
        assert mock_input.call_count == 2
        print("   OK: prompted only for box-path and subject (2 prompts), not run")

        print("11. resolve_subject_run_paths respects explicit --input-run1/--input-run2 paths directly...")
        with patch("builtins.input") as mock_input:
            paths = resolve_subject_run_paths(subj2_run1, subj2_run2, None, None)
            mock_input.assert_not_called()
        assert paths == {"1": subj2_run1, "2": subj2_run2}, paths
        print("   OK: explicit paths bypass box-path/subject entirely")

        print("\nALL PATH RESOLUTION TESTS PASSED.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
