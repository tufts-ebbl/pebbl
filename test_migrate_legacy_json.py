#!/usr/bin/env python3
"""
Tests migrate_legacy_json.py: converting step2_physio_correction.ipynb's
old *_ecg_final.json output into this tool's *_annotations_<initials>.json
shape, using synthetic fixtures standing in for a real Box derivatives tree
(never touches the real Box path). Also --upgrade (2026-09-26): the migrated
reviews rewritten in the current format -- status in_progress, reviewed_span,
provenance of the new file, edit history against the notebook template,
migrated_from -- and resumable in Step 2 with the provenance check passing.

Run with: annotate_env\\Scripts\\python.exe test_migrate_legacy_json.py
"""

import json
import os
import shutil
import tempfile

from migrate_legacy_json import convert_one, lookup_step2_initials, migrate, parse_legacy_filename
from physio_io import build_run_path, generate_synthetic_demo


def make_fake_subject(box_path, subject, run, sfreq=1000, duration_sec=10):
    """Writes a real-shaped *_physio.tsv.gz + JSON sidecar, mirroring physioBatch.py's output."""
    df, actual_sfreq = generate_synthetic_demo(duration_sec=duration_sec, sfreq=sfreq)
    tsv_path = build_run_path(box_path, subject, run)
    os.makedirs(os.path.dirname(tsv_path), exist_ok=True)
    columns = list(df.columns)
    df.to_csv(tsv_path, sep="\t", index=False, header=False, compression="gzip")
    json_path = tsv_path.replace("_physio.tsv.gz", "_physio.json")
    with open(json_path, "w") as f:
        json.dump({"Columns": columns, "SamplingFrequency": actual_sfreq}, f)
    return tsv_path, actual_sfreq


def write_legacy_final_json(box_path, subject, run, corrected_peaks, bad_segments=None, valid=True):
    file_stem = f"sub-{subject}_ses-run{run}_task-sdi"
    out_dir = os.path.dirname(build_run_path(box_path, subject, run))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{file_stem}_ecg_final.json")
    with open(path, "w") as f:
        json.dump({"ecg": {"valid": valid, "corrected_peaks": corrected_peaks, "bad_segments": bad_segments}}, f)
    return path


def write_processing_log(box_path, rows):
    """rows: list of (subject_with_prefix, run, initials, step) tuples."""
    lines = ["Subject,Run,Initials,Step,Notes,Timestamp"]
    for subject, run, initials, step in rows:
        lines.append(f"{subject},{run},{initials},{step},,1/1/2026 9:00")
    with open(os.path.join(box_path, "processing_log.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")


def test_upgrade(tmp_dir):
    import contextlib
    import hashlib
    import io
    from unittest.mock import patch

    import mne
    import numpy as np

    import session_summary_gui
    from annotation_io import run_stage_b
    from channel_config import CHANNELS
    from migrate_legacy_json import upgrade
    from physio_io import load_physio_tsv, read_sidecar, sidecar_path_for
    from ppg_checks import markers_off_top
    from provenance import column_sha256

    quiet = contextlib.redirect_stdout
    print("10. --upgrade: the migrated review rewritten in the current format...")
    box = tempfile.mkdtemp(prefix="upgrade_", dir=tmp_dir)
    stems = {}
    for subject in ("001", "002", "003"):
        tsv, _ = make_fake_subject(box, subject, "1", duration_sec=20)
        stems[subject] = (tsv, f"sub-{subject}_ses-run1_task-sdi")
    df, sfreq = load_physio_tsv(stems["001"][0])
    candidates = [int(i) for i in np.flatnonzero(df["ecg_peaks"].to_numpy())]
    off = set(markers_off_top(df["ecg"].to_numpy(dtype=float), candidates, 50))
    peaks = [i for i in candidates if i not in off]
    sig = df["ecg"].to_numpy(dtype=float)
    from migrate_legacy_json import _is_local_max
    humps = [i for i in range(peaks[3] + 150, peaks[3] + 500) if _is_local_max(sig, i, 2)]
    false_peak = max(humps, key=lambda i: sig[i])  # a T-wave maximum: a local maximum, but not an R peak
    seed = sorted(peaks[:-2] + [false_peak])
    write_processing_log(box, [(f"sub-{s}", "1", "hlu", "R peaks inspected and corrected") for s in stems])

    def put_template(subject, peak_list, initials):
        out_dir = os.path.dirname(build_run_path(box, subject, "1"))
        with open(os.path.join(out_dir, f"{stems[subject][1]}_ecg_corrected_qrs.json"), "w") as f:
            json.dump({"ecg": {"corrected_peaks": peak_list, "initials": initials}}, f)

    put_template("001", seed, ["hlu"])
    early = sorted(set(peaks) - set(peaks[5:8])) + [peaks[5] - 1, peaks[6] - 2, peaks[7] - 1]  # editor artifact
    legacy_001 = write_legacy_final_json(box, "001", "1", sorted(early), bad_segments=[100, 200])
    shifted = sorted(peaks[:-1] + [peaks[-1] + 20])  # one peak well off the top: not snapped
    put_template("002", [p + 1 for p in seed], ["jkl"])  # someone else's template, off its maxima: flagged
    write_legacy_final_json(box, "002", "1", shifted)
    write_legacy_final_json(box, "003", "1", peaks)
    with quiet(io.StringIO()):
        migrate(box, box, apply=True)                # the 2026-09-20 state on Box
    out_dir = os.path.dirname(build_run_path(box, "003", "1"))
    path_003 = os.path.join(out_dir, f"{stems['003'][1]}_annotations_hlu.json")
    with open(path_003) as f:
        tampered = json.load(f)
    tampered["channels"]["ecg"]["indices"] = peaks[:-1]
    with open(path_003, "w") as f:
        json.dump(tampered, f)

    path_001 = os.path.join(os.path.dirname(legacy_001), f"{stems['001'][1]}_annotations_hlu.json")
    before = open(path_001, "rb").read()
    with quiet(io.StringIO()) as out:
        upgrade(box, box, apply=False)
    assert open(path_001, "rb").read() == before, "a dry run must not write"
    assert "Would upgrade 2 review(s); skipped 1" in out.getvalue(), out.getvalue()
    assert "differ from the notebook's" in out.getvalue()
    print("   OK: the dry run writes nothing; a migrated file whose peaks were changed is skipped")

    with quiet(io.StringIO()) as out:
        upgrade(box, box, apply=True)
    report = out.getvalue()
    with open(path_001) as f:
        saved = json.load(f)
    ecg = saved["channels"]["ecg"]
    assert saved["schema_version"] == 2 and "coordinate_space" in saved and saved["initials"] == "hlu"
    assert ecg["indices"] == peaks and ecg["status"] == "in_progress" and ecg["reviewed_span"] == [0, len(df)]
    assert ecg["guide_shown"] is False and ecg["snap_window_samples"] == 50
    sidecar = read_sidecar(sidecar_path_for(stems["001"][0]))
    assert ecg["provenance"]["signal_sha256"] == column_sha256(stems["001"][0], sidecar["Columns"], wanted=["ecg"])["ecg"]
    [record] = ecg["edit_history"]
    assert record["started_from"].startswith("notebook QRS template") and record["saved_at"] == "1/1/2026 9:00"
    assert sorted(record["added"]) == peaks[-2:] and record["removed"] == [false_peak] and record["moved"] == []
    moved_from = ecg["migrated_from"]
    assert moved_from["sha256"] == hashlib.sha256(open(legacy_001, "rb").read()).hexdigest()
    assert sorted(moved_from["snapped_to_local_max"]) == sorted([[peaks[5] - 1, peaks[5]], [peaks[6] - 2, peaks[6]],
                                                                 [peaks[7] - 1, peaks[7]]]), moved_from
    assert moved_from["template_peaks_at_local_max"] == 1.0 and moved_from["peaks_at_top_within_50_ms"] == 1.0
    assert moved_from["dropped_bad_segments"] == [100, 200]
    assert os.listdir(os.path.join(os.path.dirname(path_001), "backups")), "the migrated file is backed up first"
    assert os.path.exists(legacy_001), "the notebook file is never touched"
    rec_002 = json.load(open(os.path.join(os.path.dirname(build_run_path(box, "002", "1")),
                                          f"{stems['002'][1]}_annotations_hlu.json")))["channels"]["ecg"]["edit_history"][0]
    assert "unknown" in rec_002["started_from"] and rec_002["added"] is None, rec_002
    ecg_002 = json.load(open(os.path.join(os.path.dirname(build_run_path(box, "002", "1")),
                                          f"{stems['002'][1]}_annotations_hlu.json")))["channels"]["ecg"]
    assert peaks[-1] + 20 in ecg_002["indices"], "a peak 20 samples off is left as the reviewer placed it"
    assert "CHECK: the ECG may have changed" in report and "CHECK these" in report, report
    flagged_line = [l for l in report.splitlines() if l.startswith("CHECK these")][0]
    assert "sub-002" in flagged_line and "sub-001" not in flagged_line, flagged_line
    print("   OK: schema 2, in_progress, reviewed_span, provenance of the new file, edit history against the "
          "notebook template (or unknown), migrated_from; editor-shifted peaks snapped (originals kept); a "
          "template off its maxima flags the run")

    with quiet(io.StringIO()) as out:
        upgrade(box, box, apply=True)
    assert "already re-saved in the tool (or upgraded)" in out.getvalue() and "Upgraded 0 review(s)" in out.getvalue()
    print("   OK: re-running never overwrites an upgraded review")
    from migrate_legacy_json import upgrade_one
    wrong = upgrade_one(legacy_001, box, overrides={("001", "1"): "zzz"})
    assert "no migrated review" in wrong["skip_reason"], wrong["skip_reason"]
    assert not os.path.exists(os.path.join(os.path.dirname(legacy_001), f"{stems['001'][1]}_annotations_zzz.json"))
    print("   OK: other initials than the migrated file's are refused, never written as a new file")
    with quiet(io.StringIO()) as out:
        results = upgrade(box, box, apply=False, exclude={("002", "1")})
    assert "EXCLUDED sub-002_ses-run1_task-sdi" in out.getvalue()
    assert [r["skip_reason"] for r in results if r.get("file_stem", "").startswith("sub-002")] == ["excluded"]
    print("   OK: --exclude leaves a run out")

    seen = {}

    def summary(heading, change_lines, warning_lines, n_changes, **kwargs):
        seen["n_changes"] = n_changes
        return session_summary_gui.SessionDecision(session_summary_gui.DISCARD)

    with patch.object(mne.io.RawArray, "plot", return_value=None), \
         patch.object(session_summary_gui, "show_session_summary", summary), quiet(io.StringIO()) as out:
        run_stage_b(df, sfreq, {"ecg": CHANNELS["ecg"]}, "hlu", os.path.dirname(path_001), stems["001"][1],
                    stems["001"][1], sidecar=sidecar, tsv_path=stems["001"][0])
    assert seen["n_changes"] == 0 and f"{len(peaks)} point annotation(s)" in out.getvalue(), out.getvalue()
    print("   OK: the upgraded review resumes in Step 2 with every peak and the provenance check passing")


def main():
    print("1. parse_legacy_filename(): extracts (file_stem, subject, run) from the real naming "
          "convention, returns None for anything that doesn't match...")
    parsed = parse_legacy_filename("/some/path/sub-001_ses-run1_task-sdi_ecg_final.json")
    assert parsed == ("sub-001_ses-run1_task-sdi", "001", "1"), parsed
    assert parse_legacy_filename("/some/path/not_a_legacy_file.json") is None
    print(f"   OK: {parsed}")

    tmp_dir = tempfile.mkdtemp(prefix="migrate_legacy_")
    try:
        print("2. lookup_step2_initials(): finds the LAST 'R peaks inspected and corrected' row "
              "for a subject/run, ignoring 'claimed' and other Step values, and correctly "
              "prefers the MOST RECENT of several matching rows (mirrors a real re-visited "
              "session, e.g. sub-001 in the real log had two such rows)...")
        write_processing_log(tmp_dir, [
            ("sub-001", "1", "hlu", "claimed"),
            ("sub-001", "1", "hlu", "created QRS template"),
            ("sub-001", "1", "hlu", "R peaks inspected and corrected"),
            ("sub-001", "1", "xyz", "R peaks inspected and corrected"),  # a later, different-initials re-pass
        ])
        initials = lookup_step2_initials(tmp_dir, "001", "1")
        assert initials == "xyz", f"expected the LAST matching row's initials ('xyz'), got {initials!r}"
        print(f"   OK: resolved to {initials!r} (the most recent completion row)")

        print("3. lookup_step2_initials(): returns None when no matching row exists, and an "
              "--initials-for override takes priority when supplied...")
        assert lookup_step2_initials(tmp_dir, "999", "1") is None
        overridden = lookup_step2_initials(tmp_dir, "999", "1", overrides={("999", "1"): "zzz"})
        assert overridden == "zzz"
        print("   OK: None without a match, override respected when given")

        print("4. convert_one(): a normal legacy file (no bad_segments) converts cleanly -- "
              "correct initials (from the log), correct sfreq (from the real sidecar), sorted "
              "indices, no skip reason...")
        tsv_path, sfreq = make_fake_subject(tmp_dir, "001", "1")
        legacy_path = write_legacy_final_json(tmp_dir, "001", "1", [500, 100, 300], bad_segments=None)
        result = convert_one(legacy_path, tmp_dir)
        assert result["skip_reason"] is None, result["skip_reason"]
        assert result["initials"] == "xyz"
        assert result["sfreq"] == sfreq
        assert result["indices"] == [100, 300, 500], "indices should be sorted"
        assert result["bad_segments"] is None
        print(f"   OK: {result['initials']=}, {result['sfreq']=}, {result['indices']=}")

        print("5. convert_one(): a legacy file WITH non-empty bad_segments still converts (the "
              "caller decides whether to warn/drop it), reporting bad_segments so migrate() can "
              "warn loudly rather than silently dropping data...")
        legacy_path_bs = write_legacy_final_json(tmp_dir, "001", "1", [100, 200], bad_segments=[738769, 742035])
        result_bs = convert_one(legacy_path_bs, tmp_dir)
        assert result_bs["skip_reason"] is None
        assert result_bs["bad_segments"] == [738769, 742035]
        print(f"   OK: bad_segments={result_bs['bad_segments']} surfaced, not silently lost")

        print("6. convert_one(): no matching processing_log.csv row -> clear skip_reason, "
              "no crash, no guessing...")
        make_fake_subject(tmp_dir, "002", "1")
        legacy_path_2 = write_legacy_final_json(tmp_dir, "002", "1", [100, 200])
        result_2 = convert_one(legacy_path_2, tmp_dir)
        assert result_2["skip_reason"] is not None and "processing_log.csv" in result_2["skip_reason"]
        print(f"   OK: skipped cleanly -- {result_2['skip_reason']}")

        print("7. convert_one(): an --initials-for override resolves a subject/run the log "
              "can't, letting it convert instead of skipping...")
        result_2_override = convert_one(legacy_path_2, tmp_dir, overrides={("002", "1"): "manual"})
        assert result_2_override["skip_reason"] is None
        assert result_2_override["initials"] == "manual"
        print(f"   OK: resolved via override to {result_2_override['initials']!r}")

        print("8. migrate() dry-run (apply=False, the default): reports what it WOULD write, "
              "writes NOTHING...")
        tmp_dir_migrate = tempfile.mkdtemp(prefix="migrate_legacy_run_", dir=tmp_dir)
        make_fake_subject(tmp_dir_migrate, "001", "1")
        write_processing_log(tmp_dir_migrate, [("sub-001", "1", "hlu", "R peaks inspected and corrected")])
        write_legacy_final_json(tmp_dir_migrate, "001", "1", [100, 200, 300])
        results_dry = migrate(tmp_dir_migrate, tmp_dir_migrate, apply=False)
        expected_out = os.path.join(
            os.path.dirname(build_run_path(tmp_dir_migrate, "001", "1")),
            "sub-001_ses-run1_task-sdi_annotations_hlu.json",
        )
        assert not os.path.exists(expected_out), "dry-run must not write anything"
        assert len(results_dry) == 1 and results_dry[0]["skip_reason"] is None
        print(f"   OK: dry-run found 1 convertible file, wrote nothing")

        print("9. migrate() --apply: actually writes the converted file, in the SAME shape "
              "save_annotation_json() normally produces, and never touches the original "
              "*_ecg_final.json...")
        results_apply = migrate(tmp_dir_migrate, tmp_dir_migrate, apply=True)
        assert os.path.exists(expected_out), "expected the converted file to be written"
        with open(expected_out) as f:
            written = json.load(f)
        assert written["channels"]["ecg"] == {"mode": "point", "indices": [100, 200, 300]}
        assert written["initials"] == "hlu"
        original_legacy_path = os.path.join(
            os.path.dirname(build_run_path(tmp_dir_migrate, "001", "1")),
            "sub-001_ses-run1_task-sdi_ecg_final.json",
        )
        assert os.path.exists(original_legacy_path), "the original legacy file must survive untouched"
        print(f"   OK: wrote {os.path.basename(expected_out)} with channels.ecg = "
              f"{written['channels']['ecg']}, original legacy file untouched")

        test_upgrade(tmp_dir)
        print("\nALL MIGRATE_LEGACY_JSON TESTS PASSED.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
