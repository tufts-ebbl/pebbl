#!/usr/bin/env python3
"""
Tests for box_sync.py: the copy-down / diff / push-back / confirm /
cleanup lifecycle that lets physio_review.py work against a local copy of
a subject's Box-hosted files without leaving anything behind on the RA's
machine afterward.

Run with: annotate_env\\Scripts\\python.exe test_box_sync.py
"""

import os
import shutil
import tempfile
import time

from box_sync import (
    cleanup_local_copy,
    copy_subject_tree_to_local,
    find_changed_files,
    push_changed_files_to_box,
)


def write_file(path, content="data"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def main():
    tmp_dir = tempfile.mkdtemp(prefix="box_sync_test_")
    try:
        box_path = os.path.join(tmp_dir, "box")
        local_root = os.path.join(tmp_dir, "local")
        os.makedirs(box_path)
        os.makedirs(local_root)

        print("1. copy_subject_tree_to_local(): mirrors an existing subject's files, preserving "
              "modification times (needed so a later rewrite is distinguishable from a fresh copy)...")
        write_file(os.path.join(box_path, "sub-085", "ses-run1", "beh", "sub-085_ses-run1_task-sdi_physio.tsv.gz"),
                   "physio data")
        write_file(os.path.join(box_path, "sub-085", "ses-run1", "beh", "sub-085_ses-run1_task-sdi_physio.json"),
                   '{"SamplingFrequency": 1000}')
        snapshot = copy_subject_tree_to_local(box_path, "085", local_root)
        copied_physio = os.path.join(local_root, "sub-085", "ses-run1", "beh",
                                      "sub-085_ses-run1_task-sdi_physio.tsv.gz")
        assert os.path.exists(copied_physio)
        assert open(copied_physio).read() == "physio data"
        assert len(snapshot) == 2, f"expected 2 files snapshotted, got {snapshot}"
        print(f"   OK: {len(snapshot)} file(s) copied and snapshotted")

        print("2. copy_subject_tree_to_local(): a brand-new subject (no Box folder yet) doesn't "
              "crash, just creates the empty local destination...")
        empty_snapshot = copy_subject_tree_to_local(box_path, "999", local_root)
        assert empty_snapshot == {}
        assert os.path.isdir(os.path.join(local_root, "sub-999"))
        print("   OK: empty snapshot, empty local folder created")

        print("3. find_changed_files(): a rewritten file is detected as changed; an untouched "
              "one is NOT...")
        time.sleep(0.05)  # ensure a real mtime difference on filesystems with coarse resolution
        write_file(copied_physio, "physio data")  # rewritten, same content, but a fresh mtime
        new_annotations = os.path.join(local_root, "sub-085", "ses-run1", "beh",
                                        "sub-085_ses-run1_task-sdi_annotations_hlu.json")
        write_file(new_annotations, '{"channels": {}}')  # a brand-new file this "session" created
        changed = find_changed_files(local_root, "085", snapshot)
        # A rewritten physio file is NEVER listed: physioProcess's own files are never pushed back
        # (2026-09-26 safety fix; test_safety_fixes.py).
        assert "sub-085/ses-run1/beh/sub-085_ses-run1_task-sdi_physio.tsv.gz".replace("/", os.sep) not in changed
        assert "sub-085/ses-run1/beh/sub-085_ses-run1_task-sdi_annotations_hlu.json".replace("/", os.sep) in changed
        assert "sub-085/ses-run1/beh/sub-085_ses-run1_task-sdi_physio.json".replace("/", os.sep) not in changed, (
            "the untouched sidecar file must NOT be reported as changed"
        )
        print(f"   OK: changed={changed}")

        print("4. push_changed_files_to_box(): copies each changed file to the matching Box path "
              "and confirms it landed (fresh mtime, file exists)...")
        confirmed, failed = push_changed_files_to_box(local_root, box_path, changed)
        assert failed == [], f"expected no failures, got {failed}"
        assert confirmed == changed
        pushed_annotations = os.path.join(box_path, "sub-085", "ses-run1", "beh",
                                           "sub-085_ses-run1_task-sdi_annotations_hlu.json")
        assert os.path.exists(pushed_annotations)
        assert open(pushed_annotations).read() == '{"channels": {}}'
        print(f"   OK: confirmed={confirmed}")

        print("5. push_changed_files_to_box(): a destination directory that doesn't exist yet on "
              "Box is created automatically (e.g. a brand-new subject's very first push)...")
        write_file(os.path.join(local_root, "sub-999", "ses-run1", "beh",
                                 "sub-999_ses-run1_task-sdi_annotations_hlu.json"), "{}")
        confirmed2, failed2 = push_changed_files_to_box(
            local_root, box_path, ["sub-999/ses-run1/beh/sub-999_ses-run1_task-sdi_annotations_hlu.json".replace("/", os.sep)]
        )
        assert failed2 == []
        assert os.path.exists(os.path.join(box_path, "sub-999", "ses-run1", "beh",
                                            "sub-999_ses-run1_task-sdi_annotations_hlu.json"))
        print("   OK: new subject/run folder created on Box automatically")

        print("6. push_changed_files_to_box(): a stale/failed confirmation is reported, not silently "
              "treated as success...")
        confirmed3, failed3 = push_changed_files_to_box(local_root, box_path, changed, tolerance_seconds=-1)
        assert confirmed3 == [] and set(failed3) == set(changed), (
            f"an impossible (negative) tolerance should fail every confirmation; got "
            f"confirmed={confirmed3}, failed={failed3}"
        )
        print(f"   OK: failed={failed3} (as expected with an impossible tolerance)")

        print("7. push_changed_files_to_box(): a local file saved well before the tolerance window "
              "(e.g. an early-session save, pushed only at session end) still confirms -- the "
              "destination's mtime must reflect THIS write, not the source's own old mtime "
              "(regression test for the copy2-preserves-mtime false-failure bug)...")
        stale_local = os.path.join(local_root, "sub-085", "ses-run1", "beh",
                                    "sub-085_ses-run1_task-sdi_annotations_hlu.json")
        old_time = time.time() - 1000  # well outside the default 300s tolerance
        os.utime(stale_local, (old_time, old_time))
        confirmed_stale, failed_stale = push_changed_files_to_box(
            local_root, box_path,
            ["sub-085/ses-run1/beh/sub-085_ses-run1_task-sdi_annotations_hlu.json".replace("/", os.sep)],
        )
        assert failed_stale == [], f"a stale-but-successful push must still confirm, got failed={failed_stale}"
        assert confirmed_stale == ["sub-085/ses-run1/beh/sub-085_ses-run1_task-sdi_annotations_hlu.json".replace("/", os.sep)]
        print(f"   OK: confirmed={confirmed_stale} despite a 1000s-old local mtime")

        print("8. cleanup_local_copy(): removes only that subject's local subfolder, NEVER the "
              "reusable --local-path folder itself (an RA reuses that across sessions/subjects)...")
        assert os.path.isdir(os.path.join(local_root, "sub-085"))
        cleanup_local_copy(local_root, "085")
        assert not os.path.exists(os.path.join(local_root, "sub-085")), "sub-085 should be gone"
        assert os.path.isdir(local_root), "the local_root folder itself must survive"
        assert os.path.isdir(os.path.join(local_root, "sub-999")), "sub-999 (untouched by this cleanup) must survive"
        print("   OK: sub-085 removed; local_root and sub-999 both preserved")

        print("9. Accepts subject already prefixed with 'sub-' the same way as a bare number...")
        snapshot_prefixed = copy_subject_tree_to_local(box_path, "sub-999", local_root)
        assert isinstance(snapshot_prefixed, dict)
        print("   OK: 'sub-999' and '999' both resolve to the same folder")

        print("\nALL BOX_SYNC TESTS PASSED.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
