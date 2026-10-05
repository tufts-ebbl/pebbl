#!/usr/bin/env python3
"""
Tests the 2026-09-26 safety fixes (HANDOFF.md section 55), found by the
old-annotation compatibility review (section 53):

  1. physioProcess's own files (*_physio.tsv.gz / *_physio.json) are never
     pushed back to Box, from a leftover local copy or after a session:
     a pre-reprocess leftover would otherwise overwrite Box's current file;
  2. a saved PPG review with no provenance (made before 2026-09-26, on the
     pre-fix PPG timing) is never seeded onto a real file's PPG: Step 2
     starts PPG fresh and keeps the old entry as ppg_points_legacy; Step 3
     refuses to reconcile it;
  3. an unreadable or damaged saved file stops with a plain message, not a
     traceback; a save over a damaged file writes the session to a
     *_RESCUED_* file instead of losing it; check_reprocess_alignment reports
     unreadable RA files (UNREADABLE_RA_FILE) instead of skipping them;
  4. a saved file whose sampling_rate differs from the data file, or whose
     marks fall outside it (or aren't whole numbers), stops plainly; an ECG
     template with peaks outside the file too;
  5. Option C (HLU): a drawn peak is the highest point inside the RA's box,
     climbing at most 15 ms only when the box cut off the top; the old second
     re-snap to the highest sample within +/-50 ms is gone, so a short R wave
     next to a taller MHD wave or artifact stays where the RA put it.

Run with: annotate_env\\Scripts\\python.exe test_safety_fixes.py
"""

import contextlib
import io
import json
import os
import shutil
import tempfile
import time
from unittest.mock import patch

import mne
import numpy as np

import check_reprocess_alignment as cra
import session_summary_gui
from annotation_io import load_saved_annotations, resolve_ecg_peaks, run_stage_b, save_annotation_json
from box_sync import find_changed_files, find_unpushed_local_files, push_changed_files_to_box
from channel_config import CHANNELS
from physio_io import generate_synthetic_demo, load_physio_tsv, read_sidecar
from reconcile import run_stage_c, save_reconciled_json

session_summary_gui.show_session_summary = session_summary_gui.headless_decision(finished=True)
QUIET = contextlib.redirect_stdout
STEM = "sub-001_ses-run1_task-sdi"


def write(path, text, encoding="utf-8"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding=encoding) as f:
        f.write(text)


def make_real_like_run(folder):
    """A synthetic run written like physioBatch.py's output, with a new-pipeline sidecar (so provenance exists)."""
    df, sfreq = generate_synthetic_demo(duration_sec=20)
    os.makedirs(folder, exist_ok=True)
    tsv = os.path.join(folder, f"{STEM}_physio.tsv.gz")
    df.to_csv(tsv, sep="\t", index=False, header=False, compression="gzip")
    with open(tsv.replace("_physio.tsv.gz", "_physio.json"), "w") as f:
        json.dump({"Columns": list(df.columns), "SamplingFrequency": sfreq,
                   "Provenance": {"NumberOfSamples": len(df), "PipelineCommit": "test", "CTTiming": {}}}, f)
    df, sfreq = load_physio_tsv(tsv)
    return tsv, df, sfreq, read_sidecar(tsv.replace("_physio.tsv.gz", "_physio.json"))


def expect_exit(fn, *needles):
    try:
        with QUIET(io.StringIO()):
            fn()
    except SystemExit as e:
        text = str(e)
        for needle in needles:
            assert needle in text, (needle, text)
        assert "Traceback" not in text
        return text
    raise AssertionError(f"expected a plain SystemExit containing {needles}")


def test_no_pipeline_push(tmp):
    print("1. physioProcess's own files are never pushed back to Box...")
    box, local = os.path.join(tmp, "box"), os.path.join(tmp, "local")
    rel_tsv = os.path.join("sub-001", "ses-run1", "beh", f"{STEM}_physio.tsv.gz")
    rel_json = rel_tsv.replace(".tsv.gz", ".json")
    rel_ann = os.path.join("sub-001", "ses-run1", "beh", f"{STEM}_annotations_hlu.json")
    for rel, text in ((rel_tsv, "RUN8"), (rel_json, "{}"), (rel_ann, "{}")):
        write(os.path.join(box, rel), text)
    for rel, text in ((rel_tsv, "OLD"), (rel_json, "{\"old\": 1}"), (rel_ann, "{\"channels\": {}}")):
        write(os.path.join(local, rel), text)
        future = time.time() + 3600
        os.utime(os.path.join(local, rel), (future, future))  # "newer" than Box, as a leftover could be
    assert find_unpushed_local_files(box, "001", local) == [rel_ann]
    assert find_changed_files(local, "001", {}) == [rel_ann]
    confirmed, failed = push_changed_files_to_box(local, box, [rel_tsv, rel_json, rel_ann])
    assert confirmed == [rel_ann] and failed == []
    assert open(os.path.join(box, rel_tsv)).read() == "RUN8" and open(os.path.join(box, rel_json)).read() == "{}"
    print("   OK: a leftover's old tsv/json are never listed, changed or pushed; RA work still is")


def test_ppg_without_provenance(tmp):
    print("2. A PPG review with no provenance is never seeded onto a real file (Step 2), nor reconciled (Step 3)...")
    folder = os.path.join(tmp, "ppg")
    tsv, df, sfreq, sidecar = make_real_like_run(folder)
    old = {"mode": "point", "indices": [123, 4567], "status": "complete"}
    with open(os.path.join(folder, f"{STEM}_annotations_hlu.json"), "w") as f:
        json.dump({"initials": "hlu", "sampling_rate": sfreq, "channels": {"ppg": old}}, f)
    out = io.StringIO()
    with patch.object(mne.io.RawArray, "plot", return_value=None), QUIET(out):
        r = run_stage_b(df, sfreq, {"ppg": CHANNELS["ppg"]}, "hlu", folder, STEM, STEM, sidecar=sidecar,
                        tsv_path=tsv)
    assert "saved before 2026-09-26" in out.getvalue(), out.getvalue()
    with open(r.saved_path) as f:
        channels = json.load(f)["channels"]
    seed = [int(i) for i in np.flatnonzero(df["ppg_peaks"].to_numpy())]
    assert channels["ppg"]["indices"] == seed and channels["ppg"]["provenance"]["signal_sha256"]
    assert channels["ppg_points_legacy"] == old, channels.keys()
    # bad_ppg's provenance is the ppg column's ("column": "ppg"), so the reprocess check finds nothing invalid.
    assert channels["bad_ppg"]["provenance"]["column"] == "ppg"
    assert channels["bad_ppg"]["provenance"]["signal_sha256"] == channels["ppg"]["provenance"]["signal_sha256"]
    assert cra.invalidated_channels(tsv, len(df), cra.saved_provenance([r.saved_path])) == []
    # A review saved by the current tool (with provenance) resumes as before.
    out = io.StringIO()
    with patch.object(mne.io.RawArray, "plot", return_value=None), QUIET(out):
        run_stage_b(df, sfreq, {"ppg": CHANNELS["ppg"]}, "hlu", folder, STEM, STEM, sidecar=sidecar, tsv_path=tsv)
    assert "saved before 2026-09-26" not in out.getvalue() and "No changes this session" in out.getvalue()
    # Synthetic data (no file, so no provenance) is unaffected.
    syn = os.path.join(tmp, "ppg_syn")
    os.makedirs(syn)
    with open(os.path.join(syn, "t_annotations_hlu.json"), "w") as f:
        json.dump({"sampling_rate": sfreq, "channels": {"ppg": dict(old)}}, f)
    with patch.object(mne.io.RawArray, "plot", return_value=None), QUIET(io.StringIO()) as out:
        run_stage_b(df, sfreq, {"ppg": CHANNELS["ppg"]}, "hlu", syn, "t", "t")
    assert "saved before 2026-09-26" not in out.getvalue()
    # Step 3 refuses a reviewer's provenance-less PPG on a real file.
    a = {"ppg": dict(old)}
    b = {"ppg": {"mode": "point", "indices": [123, 4567], "status": "complete"}}
    expect_exit(lambda: run_stage_c(df, sfreq, {"ppg": CHANNELS["ppg"]}, os.path.join(tmp, "rec"), STEM, STEM, a, b,
                                    "aa", "bb", reconciler="hl", sidecar=sidecar, tsv_path=tsv),
                "Reviewer aa's PPG review has no record", "redo PPG")
    print("   OK: Step 2 starts PPG fresh and keeps the old review as ppg_points_legacy; current-tool and "
          "synthetic reviews unaffected; Step 3 refuses")


def test_unreadable_files(tmp):
    print("3. Unreadable or damaged files: plain messages, rescued work, no false OK...")
    folder = os.path.join(tmp, "bad")
    path = os.path.join(folder, f"{STEM}_annotations_hlu.json")
    write(path, '{"channels": {"ecg": {"mode": "point", "indices": [1, 2')
    expect_exit(lambda: load_saved_annotations(folder, STEM, "hlu"), "can't be read", "Nothing has been changed")
    write(path, '{"channels": ["ecg"]}')
    expect_exit(lambda: load_saved_annotations(folder, STEM, "hlu"), "damaged 'channels' section")
    write(path, '{"channels": {"ecg": 5}}')
    expect_exit(lambda: load_saved_annotations(folder, STEM, "hlu"), "damaged 'channels' section")
    write(path, '["not", "an", "object"]')
    expect_exit(lambda: load_saved_annotations(folder, STEM, "hlu"), "isn't in the expected format")
    write(path, '{"channels": {"ecg": {"mode": "point", "indices": [1, 2]}}}', encoding="utf-8-sig")
    assert load_saved_annotations(folder, STEM, "hlu")[0]["ecg"]["indices"] == [1, 2], "a UTF-8 BOM is fine"

    # A save over a damaged file: the session is rescued, the damaged file left as it is.
    write(path, '{"channels": {"ecg": ')
    before = open(path, "rb").read()
    new = {"eda": {"mode": "segment", "bad_segments": [[10, 20]], "status": "complete"}}
    text = expect_exit(lambda: save_annotation_json(folder, STEM, "hlu", new, 1000, STEM), "can't be read", "RESCUED")
    rescued = [n for n in os.listdir(folder) if "_RESCUED_" in n]
    assert len(rescued) == 1 and json.load(open(os.path.join(folder, rescued[0])))["channels"] == new, text
    assert open(path, "rb").read() == before and os.listdir(os.path.join(folder, "backups"))
    rec_folder = os.path.join(tmp, "bad_rec")
    write(os.path.join(rec_folder, f"{STEM}_annotations_reconciled.json"), "{")
    expect_exit(lambda: save_reconciled_json(rec_folder, STEM, dict(new), 1000, STEM, "aa", "bb", reconciler="hl"),
                "reconciled file", "RESCUED")
    assert [n for n in os.listdir(rec_folder) if "_RESCUED_" in n]

    # An unreadable ECG template, or one with peaks outside the file.
    df, _ = generate_synthetic_demo(duration_sec=20)
    tpl = os.path.join(tmp, "tpl.json")
    write(tpl, "{oops")
    expect_exit(lambda: resolve_ecg_peaks(df, "template", tpl), "ECG template file", "can't be read")
    write(tpl, json.dumps({"ecg": {"corrected_peaks": [10, len(df) + 5]}}))
    expect_exit(lambda: resolve_ecg_peaks(df, "template", tpl), "outside this file")
    write(tpl, json.dumps({"ecg": {"corrected_peaks": [10, 2000]}}))
    assert resolve_ecg_peaks(df, "template", tpl)[0]["ecg_peaks"].sum() == 2

    # check_reprocess_alignment: an unreadable RA file makes the run not OK (it used to be skipped).
    write(path, "{broken")
    assert cra.unreadable_ra_files([path]) and cra.saved_indices([path]) is None
    with patch.object(cra, "find_runs_with_ra_work", return_value={("001", "1"): [path]}), \
         patch.object(cra, "compare_run", return_value={"status": "ALIGNED"}), QUIET(io.StringIO()) as out:
        code = cra.main(["--old-root", tmp, "--new-root", tmp])
    assert code == 1 and "UNREADABLE_RA_FILE" in out.getvalue() and "Do NOT copy" in out.getvalue(), out.getvalue()
    print("   OK: plain messages; a UTF-8 BOM is read; a save over a damaged file is rescued; templates checked; "
          "the reprocess check flags unreadable RA files")


def test_fit_checks(tmp):
    print("4. A saved file that doesn't fit the data file (rate, marks outside it) stops plainly...")
    folder = os.path.join(tmp, "fit")
    path = os.path.join(folder, f"{STEM}_annotations_hlu.json")

    def put(payload):
        write(path, json.dumps(payload))

    put({"sampling_rate": 500, "channels": {"ecg": {"mode": "point", "indices": [1]}}})
    expect_exit(lambda: load_saved_annotations(folder, STEM, "hlu", sfreq=1000, n_samples=100), "saved at 500 Hz")
    assert load_saved_annotations(folder, STEM, "hlu")[0], "without sfreq/n_samples, nothing is checked"
    put({"sampling_rate": 1000.0, "channels": {"ecg": {"mode": "point", "indices": [1, 100]}}})
    expect_exit(lambda: load_saved_annotations(folder, STEM, "hlu", sfreq=1000, n_samples=100), "1 ecg mark(s)",
                "outside this file")
    put({"sampling_rate": 1000, "channels": {"ecg": {"mode": "point", "indices": [1.5]}}})
    expect_exit(lambda: load_saved_annotations(folder, STEM, "hlu", sfreq=1000, n_samples=100), "whole sample")
    put({"sampling_rate": 1000, "channels": {"eda": {"mode": "segment", "bad_segments": [[90, 101]]}}})
    expect_exit(lambda: load_saved_annotations(folder, STEM, "hlu", sfreq=1000, n_samples=100), "eda")
    put({"sampling_rate": 1000, "channels": {"eda": {"mode": "segment", "bad_segments": [[90, 100]]},
                                             "rsp_points_legacy": {"mode": "point", "indices": [5000]}}})
    assert load_saved_annotations(folder, STEM, "hlu", sfreq=1000, n_samples=100), \
        "a segment may end at the file end; kept legacy entries aren't checked"
    # Step 2 applies the check with the current file's rate and length.
    df, sfreq = generate_synthetic_demo(duration_sec=20)
    put({"sampling_rate": 1000, "channels": {"ecg": {"mode": "point", "indices": [len(df) + 1]}}})
    expect_exit(lambda: run_stage_b(df, sfreq, {"ecg": CHANNELS["ecg"]}, "hlu", folder, STEM, STEM),
                "outside this file")
    print("   OK: rate mismatch, marks outside the file and non-integer marks stop plainly, in Step 2 too")


def test_option_c(tmp):
    print("5. Option C: a drawn peak is the box maximum, climbing at most 15 ms only if the box cut off the top...")
    from annotation_io import export_annotations, point_sample_from_annotation
    from physio_io import build_raw
    from reconcile import build_reconciliation_raw, export_reconciled
    climb = 15

    def wave(n=400, r_at=200, r_height=1.0, width=6, spike_at=None, spike_height=3.0):
        x = np.arange(n, dtype=float)
        sig = r_height * np.exp(-0.5 * ((x - r_at) / width) ** 2)
        if spike_at is not None:
            sig += spike_height * np.exp(-0.5 * ((x - spike_at) / 3) ** 2)
        return sig

    short_r = wave(spike_at=232)                       # a short R at 200, a taller MHD-like wave 32 ms later
    assert point_sample_from_annotation(190, 20, short_r, climb) == 200, "the box's top is kept"
    assert point_sample_from_annotation(190, 20, short_r) == 200
    legacy = int(np.argmax(short_r[150:250])) + 150    # what the old second re-snap (+/-50) would have picked
    assert legacy == 232, legacy
    assert point_sample_from_annotation(180, 14, short_r, climb) == 200, "box stops on the upslope: climbs"
    assert point_sample_from_annotation(205, 12, short_r, climb) == 200, "box starts on the downslope: climbs"
    rippled = wave()
    rippled[196] -= 0.2  # a one-sample dip on the upslope: a naive climb would stop at 195
    assert point_sample_from_annotation(186, 8, rippled, climb) == 200, "a ripple doesn't stop the climb"
    assert point_sample_from_annotation(170, 10, wave(), climb) == 179, "no top within 15 ms: the box point stays"
    flat = np.zeros(100)
    flat[40:60] = 1.0
    assert point_sample_from_annotation(30, 15, flat, climb) == 40, "a plateau across the edge: not rising, kept"
    assert point_sample_from_annotation(123, 0, short_r, climb) == 123, "an untouched seed keeps its sample"
    one = wave()
    assert point_sample_from_annotation(196, 1, one, climb) == 200, "a one-sample box on the slope climbs"

    # End to end, in Step 2 and Step 3: the short R is saved, never the taller neighbour.
    df, sfreq = generate_synthetic_demo(duration_sec=20)
    peaks = np.flatnonzero(df["ecg_peaks"].to_numpy())
    ecg = df["ecg"].to_numpy(dtype=float).copy()
    tops = [int(q) for q in peaks if 10 <= q < len(ecg) - 40 and ecg[q] >= ecg[q - 10:q + 11].max()]
    target = tops[5]
    ecg[target + 30] = ecg[target] * 3  # a taller artifact 30 ms after the R peak
    df = df.copy()
    df["ecg"] = ecg
    configs = {"ecg": CHANNELS["ecg"]}
    raw = build_raw(df, configs, sfreq)
    raw.set_annotations(mne.Annotations([(target - 10) / sfreq], [0.02], ["peak_ecg"]))
    with QUIET(io.StringIO()):
        out = export_annotations(raw, df, configs, sfreq)
    assert out["ecg"]["indices"] == [target], out["ecg"]["indices"]
    a = {"ecg": {"mode": "point", "indices": [int(peaks[1])]}}
    rraw, _ = build_reconciliation_raw(df, configs, sfreq, a, a, "aa", "bb", tolerance_samples=50)
    rraw.set_annotations(rraw.annotations + mne.Annotations([(target - 10) / sfreq], [0.02], ["peak_ecg_added"]))
    with QUIET(io.StringIO()):
        rec = export_reconciled(rraw, df, configs, sfreq, "aa", "bb")
    assert target in rec["ecg"]["indices"] and target + 30 not in rec["ecg"]["indices"], rec["ecg"]["indices"]
    print("   OK: the box maximum is kept (even beside a taller wave); a box cut on either slope climbs to the "
          "top; ripples, the 15-ms cap and plateaus behave; Steps 2 and 3 save the short R")


def test_retire_channels(tmp):
    print("6. retire_channels.py (HLU, 2026-10-02): set chosen channels aside so they can be reviewed afresh...")
    import retire_channels as rc
    folder = os.path.join(tmp, "retire")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{STEM}_annotations_hlu.json")
    prov = {"number_of_samples": 1000, "pipeline_commit": "old", "signal_sha256": "deadbeef",
            "signal_sha256_def": "physioprocess-column-sha256-v1"}
    payload = {"schema_version": 2, "initials": "hlu", "sampling_rate": 1000, "channels": {
        "ecg": {"mode": "point", "indices": [100, 900], "status": "complete", "provenance": prov},
        "sbp": {"mode": "segment", "bad_segments": [[10, 20]], "status": "complete", "provenance": prov},
        "dbp": {"mode": "segment", "bad_segments": [[10, 20]], "status": "complete", "provenance": prov},
        "ppg": {"mode": "point", "indices": [150], "status": "in_progress", "provenance": prov},
        "bad_ppg": {"mode": "segment", "bad_segments": [], "status": "in_progress", "provenance": {**prov, "column": "ppg"}}}}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    before = open(path).read()

    # Read-only by default.
    out = io.StringIO()
    with QUIET(out):
        assert rc.main(["--file", path, "--channels", "sbp, DBP,emg_cor"]) == 0
    assert open(path).read() == before and not os.path.exists(os.path.join(folder, "backups"))
    text = out.getvalue()
    assert "sbp -> sbp_segments_legacy  (would be set aside)" in text and "emg_cor: not in" in text, text
    assert "Read-only: nothing has been changed" in text

    # --apply: backed up first; entries moved in place with a "retired" record; others untouched.
    with QUIET(io.StringIO()):
        assert rc.main(["--file", path, "--channels", "sbp,dbp", "--reason", "run 10 changed sbp/dbp", "--apply"]) == 0
    backups = os.listdir(os.path.join(folder, "backups"))
    assert len(backups) == 1 and open(os.path.join(folder, "backups", backups[0])).read() == before
    with open(path) as f:
        after = json.load(f)
    ch = after["channels"]
    assert list(ch) == ["ecg", "sbp_segments_legacy", "dbp_segments_legacy", "ppg", "bad_ppg"], list(ch)
    assert ch["sbp_segments_legacy"]["bad_segments"] == [[10, 20]]
    assert ch["sbp_segments_legacy"]["retired"]["reason"] == "run 10 changed sbp/dbp"
    assert ch["sbp_segments_legacy"]["retired"]["from_channel"] == "sbp"
    assert ch["ecg"] == payload["channels"]["ecg"] and after["initials"] == "hlu"

    # Companions go with their parent: ppg takes bad_ppg along.
    plan, missing = rc.plan_retirement(ch, ["ppg"])
    assert plan == [("ppg", "ppg_points_legacy"), ("bad_ppg", "bad_ppg_segments_legacy")] and not missing

    # A second retirement of the same channel gets a numbered key; a legacy key can't be retired.
    ch2 = {**ch, "sbp": {"mode": "segment", "bad_segments": [], "status": "complete"}}
    assert rc.plan_retirement(ch2, ["sbp"])[0] == [("sbp", "sbp_segments_legacy_2")]
    try:
        rc.retire_channels(path, ["sbp_segments_legacy"])
        raise AssertionError("a legacy key was accepted")
    except SystemExit as e:
        assert "already set aside" in str(e)

    # check_reprocess_alignment ignores set-aside entries (they're never loaded).
    assert "sbp" not in cra.saved_provenance([path]) and "sbp_segments_legacy" not in cra.saved_provenance([path])
    assert set(cra.saved_provenance([path])) == {"ecg", "ppg", "bad_ppg"}
    assert "sbp_segments_legacy" not in cra.saved_channel_keys([path])

    # Step 2 ignores the retired entry, starts SBP fresh, and keeps the retired entry when it saves.
    df, sfreq = generate_synthetic_demo(duration_sec=20)
    df["sbp"] = 120.0
    step2 = os.path.join(tmp, "retire_step2")
    os.makedirs(step2, exist_ok=True)
    with open(os.path.join(step2, f"{STEM}_annotations_hlu.json"), "w") as f:
        json.dump({"schema_version": 2, "sampling_rate": sfreq, "channels": {
            "sbp_segments_legacy": {"mode": "segment", "bad_segments": [[5, 6]], "status": "complete",
                                    "provenance": prov, "retired": {"from_channel": "sbp"}}}}, f)
    seen = {}

    def plot(self, *args, **kwargs):
        seen["bad_sbp"] = sum(d == "bad_sbp" and dur > 0 for d, dur in zip(self.annotations.description,
                                                                           self.annotations.duration))

    with patch.object(mne.io.RawArray, "plot", plot), QUIET(io.StringIO()):
        r = run_stage_b(df, sfreq, {"sbp": CHANNELS["sbp"]}, "hlu", step2, STEM, "synthetic")
    assert seen["bad_sbp"] == 0, "the retired marks aren't loaded"
    with open(r.saved_path) as f:
        saved = json.load(f)["channels"]
    assert "sbp" in saved and saved["sbp_segments_legacy"]["bad_segments"] == [[5, 6]], sorted(saved)
    print("   OK: read-only by default; --apply backs up, moves entries in place with a reason; companions follow; "
          "numbered keys; legacy keys refused; ignored by check_reprocess_alignment; Step 2 starts fresh and keeps it")


def main():
    tmp = tempfile.mkdtemp()
    try:
        test_no_pipeline_push(tmp)
        test_ppg_without_provenance(tmp)
        test_unreadable_files(tmp)
        test_fit_checks(tmp)
        test_option_c(tmp)
        test_retire_channels(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL SAFETY FIX TESTS PASSED.")


if __name__ == "__main__":
    main()
