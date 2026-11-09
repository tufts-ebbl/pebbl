#!/usr/bin/env python3
"""
urglab certification practice (pebbl-private pebbl_urglab-practice-plan_20261101-2123.md,
approved 2026-11-08 with its addendum A1-A8). Built in stages; this suite grows with it.

  1. GOLDEN: sdi's practice is byte-for-byte what it was before urglab
     practice existed (seed 2024; SHA-256s recorded 2026-11-08 from main
     5fa347f, before any edit). A failure names the library versions the
     hashes were recorded with, since a different neurokit2/numpy/scipy/pandas
     can change them without any PEBBL change.
  2. The scorer reads both key shapes: sdi's (one participant, no "study")
     and urglab's (participants under "subjects", one key version per study).
     A wrong version or an unknown participant is reported, not scored.
  3. "Flip the ECG" is scored where the key says how the ECG was recorded:
     the wrong state fails ECG with a FIX line saying what to do; sdi keys
     have no flip line.
  4. Segment scoring's urglab additions: a machine stretch that must be kept
     fails when shortened; a neutral stretch may be marked or not; one
     channel can have a stricter real-event limit (COR frowns: 10%).
  5. The command-line scorer reads the participant from the file name, and
     the session's practice report names the participant.
  6. make --study-profile writes urglab's two participants and key (version
     2, seed 20261108). The signals are exactly the ones physiopipe computed
     its marks for (practice_urglab_machine_qc.json); make refuses another
     seed, a missing JSON, or signals that differ, and writes nothing then.
  7. The files have physiopipe's reference sidecar's shape and the study's
     own texts, and carry physiopipe's computed marks unchanged.
  8. Controls: each key r/R is the top of its beat; the breathing cases are
     on the belt; PEBBL hides exactly filter_edge; the display cut still
     shows the cases.
  9. A perfect review passes both participants.
 10. The untouched marks, saved through the real Step 2 code with urglab's
     settings, fail where the RA has work to do (and pass EDA, where the
     computer's marks are right).
 11. Each mistake the plan names fails; a missed single small pop, the whole
     ECG lead-off marked, and the drift after a belt step don't.
 12. A Step 1 + Step 2 practice session for 991 through physio_review.

Run with: annotate_env\\Scripts\\python.exe test_urglab_practice.py
"""

import contextlib
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
from unittest.mock import patch

import practice_certification as pc
import study_profile

# Recorded 2026-11-08 from pebbl-public main 5fa347f (practice_certification.py unchanged), make(seed=2024),
# twice, with the same result: SHA-256 of each unzipped TSV, each sidecar's bytes, and the decoded key
# (json.dumps(key, sort_keys=True)).
GOLDEN_SDI = {
    "sub-990_ses-run1_task-sdi_physio.tsv": "6f263a4fa5f4f847b667c1d34272c8aed6f2e31ab6e98b707db4fb40e6f0d149",
    "sub-990_ses-run1_task-sdi_physio.json": "9fe0365bde4c7899dbaa2a1e6a9cb27945c033f8355041c2ad498ec3423b4144",
    "sub-990_ses-run2_task-sdi_physio.tsv": "53e2ca32656b1ad41b9397d89a064f29a423e170e8f3628084678dff3e370755",
    "sub-990_ses-run2_task-sdi_physio.json": "8d57fab3a897d9853acefe7eb5d961afff11e99e72ca2327bcb42d122229a170",
    "key": "fc65ee486bc4e28cc83f2a49f1ccf8fe159aac08aaf8492e1d49c8162ce4a2bd",
}
GOLDEN_VERSIONS = {"neurokit2": "0.2.13", "numpy": "2.4.6", "scipy": "1.17.1", "pandas": "2.3.3"}


def sdi_hashes(out_dir):
    with contextlib.redirect_stdout(io.StringIO()):
        written, key_path = pc.make(out_dir, seed=2024)
    hashes = {}
    for tsv in written:
        with gzip.open(tsv, "rb") as f:
            hashes[os.path.basename(tsv).replace(".tsv.gz", ".tsv")] = hashlib.sha256(f.read()).hexdigest()
        sidecar = tsv.replace(".tsv.gz", ".json")
        with open(sidecar, "rb") as f:
            hashes[os.path.basename(sidecar)] = hashlib.sha256(f.read()).hexdigest()
    key = pc.load_key(key_path)
    hashes["key"] = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()
    return hashes


def library_versions():
    import neurokit2, numpy, pandas, scipy  # noqa: E401
    return {"neurokit2": neurokit2.__version__, "numpy": numpy.__version__, "scipy": scipy.__version__,
            "pandas": pandas.__version__}


N = 60000  # a 60-s run at 1000 Hz, for the scorer checks


def ecg_truth():
    return {"mode": "point", "true_peaks": [1000 * s for s in range(2, 58)], "traps": []}


def urglab_key(version=None):
    """A urglab-shaped key: 990 upright (rS), 991 upside down, each with one run."""
    run = {"n_samples": N, "channels": {
        "ecg": ecg_truth(),
        "emg_cor": {"mode": "segment", "artifacts": [{"type": "an electrode pop", "start": 10000, "end": 10500}],
                    "traps": [{"type": "a held frown: real, leave it", "start": 20000, "end": 30000}],
                    "must_keep": [{"type": "a clipped stretch", "start": 40000, "end": 42000}],
                    "neutral": [[50000, 55000]]}}}
    criteria = dict(pc.PASS_CRITERIA, trap_max_coverage_by_channel={"emg_cor": 0.10})
    return {"version": pc.KEY_VERSIONS["urglab"] if version is None else version, "study": "urglab",
            "criteria": criteria, "subjects": {"990": {"ecg_inverted": False, "runs": {"1": run}},
                                               "991": {"ecg_inverted": True, "runs": {"1": run}}}}


def review(flipped=False, cor=None):
    ecg = {"mode": "point", "indices": ecg_truth()["true_peaks"]}
    if flipped:
        ecg["inverted"] = True
    channels = {"ecg": ecg}
    if cor is not None:
        channels["emg_cor"] = {"mode": "segment", "bad_segments": cor}
    return {"initials": "zz", "channels": channels}


def main():
    tmp = tempfile.mkdtemp(prefix="urglab_practice_")
    try:
        print("1. GOLDEN: sdi's practice files and key are byte-for-byte unchanged (seed 2024)...")
        got = sdi_hashes(os.path.join(tmp, "sdi_practice"))
        if got != GOLDEN_SDI:
            changed = sorted(k for k in GOLDEN_SDI if got.get(k) != GOLDEN_SDI[k])
            raise AssertionError(f"sdi practice changed: {changed}. Recorded with {GOLDEN_VERSIONS}; this "
                                 f"environment has {library_versions()}. If the versions differ, that alone can "
                                 f"explain it; if they match, an edit changed sdi's practice.")
        plan = dict(pc.RUN_PLANS["1"], heart_rate=pc.RUN_PLANS["1"]["heart_rate"] + 1)
        with patch.dict(pc.RUN_PLANS, {"1": plan}):
            changed = sdi_hashes(os.path.join(tmp, "sdi_changed"))
        assert changed["sub-990_ses-run1_task-sdi_physio.tsv"] != GOLDEN_SDI["sub-990_ses-run1_task-sdi_physio.tsv"] \
            and changed["key"] != GOLDEN_SDI["key"], "negative control: a one-beat change must change the hashes"
        assert changed["sub-990_ses-run2_task-sdi_physio.tsv"] == GOLDEN_SDI["sub-990_ses-run2_task-sdi_physio.tsv"]
        print("   OK: both runs' TSVs and sidecars, and the decoded key (and a one-beat change to run 1 is caught)")

        print("2. The scorer reads sdi keys as before and urglab keys by participant...")
        sdi_key = {"version": pc.KEY_VERSION, "subject": "990", "runs": {"1": {"n_samples": N, "channels": {
            "ecg": ecg_truth()}}}}
        ok, results, lines = pc.score_run(sdi_key, review(), "1")
        assert ok and lines[0] == "Practice score: participant 990, run 1, initials zz", lines
        assert pc.key_study(sdi_key) == "sdi" and list(pc.key_subjects(sdi_key)) == ["990"]
        key = urglab_key()
        ok, results, lines = pc.score_run(key, review(flipped=True), "1", subject="991")
        assert ok and lines[0].startswith("Practice score: participant 991, run 1"), lines
        assert pc.run_key_for(key, "991", "1") is key["subjects"]["991"]["runs"]["1"]
        assert pc.run_key_for(key, "992", "1") is None and pc.run_key_for(key, "990", "2") is None
        ok, results, lines = pc.score_run(urglab_key(version=99), review(), "1", subject="990")
        assert not ok and results == {} and "version 99" in lines[0] and f"(version {pc.KEY_VERSIONS['urglab']})" \
            in lines[0], lines
        ok, results, lines = pc.score_run(key, review(), "1", subject="995")
        assert not ok and results == {} and "participant 995" in lines[0], lines
        ok, results, lines = pc.score_run(dict(key, study="nostudy"), review(), "1", subject="990")
        assert not ok and "version None" in lines[0], "a study PEBBL has no key version for is never scored"
        print("   OK")

        print("3. 'Flip the ECG' is scored for urglab, never for sdi...")
        ok, results, lines = pc.score_run(key, review(flipped=False), "1", subject="991")
        text = "\n".join(lines)
        assert not ok and results["ecg"] is False, text
        assert "FIX  Flip the ECG: this ECG is upside down" in text and "start the ECG over" in text, text
        ok, results, lines = pc.score_run(key, review(flipped=True), "1", subject="990")
        text = "\n".join(lines)
        assert not ok and "FIX  Flip the ECG: this ECG is upright, with rS beats" in text, text
        assert "mark the small r" in text, text
        ok, results, lines = pc.score_run(key, review(flipped=False), "1", subject="990")
        assert ok and "  OK   Flip the ECG: not ticked" in lines, lines
        ok, results, lines = pc.score_run(sdi_key, review(flipped=True), "1")
        assert ok and not any("Flip the ECG" in line for line in lines), "sdi: no flip line"
        print("   OK")

        print("4. Segment scoring: kept machine stretches, neutral stretches, COR's 10% frown limit...")
        good = [[10000, 10500], [40000, 42000]]

        def cor(segments):
            return pc.score_run(key, review(cor=segments), "1", channels=["emg_cor"], subject="990")

        ok, results, lines = cor(good)
        assert ok, lines
        ok, results, lines = cor([[10000, 10500], [40000, 41000]])
        text = "\n".join(lines)
        assert not ok and "FIX  the computer's mark on a clipped stretch at 40.0-42.0 s: only 50% left" in text, text
        ok, results, lines = cor(good + [[50000, 55000]])
        assert ok, "a neutral stretch may be marked"
        ok, results, lines = cor(good + [[20000, 21500]])  # 15% of the frown: over COR's 10%, under 25%
        text = "\n".join(lines)
        assert not ok and "FIX  a held frown: real, leave it at 20.0 s: 15% marked bad" in text, text
        ok, results, lines = cor(good + [[20000, 20900]])  # 9%
        assert ok, lines
        print("   OK")

        print("5. The command-line scorer reads the participant from the file name; the report names it...")
        assert pc.subject_from_filename("x/sub-991_ses-run1_task-urglab_annotations_zz.json") == "991"
        assert pc.subject_from_filename("notes.json") is None
        key_path = os.path.join(tmp, "key.json")
        with open(key_path, "w") as f:
            json.dump(key, f)
        saved = os.path.join(tmp, "sub-991_ses-run1_task-urglab_annotations_zz.json")
        with open(saved, "w") as f:
            json.dump(review(flipped=False), f)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            overall, _ = pc.score(key_path, saved, ["ecg"])
        assert not overall and "participant 991" in out.getvalue() and "upside down" in out.getvalue()
        with contextlib.redirect_stdout(io.StringIO()):
            assert pc.main(["score", "--key", key_path, "--annotations", saved, "--subject", "990",
                            "--channels", "ecg"]) == 0, "990's ECG is upright: unticked is right"

        import physio_review
        import practice_report_gui
        shown = []
        practice = os.path.join(tmp, "practice")
        os.makedirs(practice)
        with open(pc.key_path_for(practice), "w", encoding="utf-8") as f:
            f.write(pc.encode_key(key))
        with patch.object(practice_report_gui, "show_practice_report",
                          side_effect=lambda heading, lines, passed: shown.append((heading, passed)) or False), \
                contextlib.redirect_stdout(io.StringIO()):
            physio_review._practice_feedback(practice, "1", saved, None, 1000, ["ecg"], "zz", None, subject="991")
        assert shown == [("Practice participant 991, run 1: ecg", False)], shown
        print("   OK")

        check_generated(tmp)

        print("\nAll urglab practice checks passed.")
    finally:
        study_profile.activate(study_profile.BUILT_IN)
        study_profile.set_ecg_flipped(False)
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# The generated practice folder (practice_urglab.py)
# --------------------------------------------------------------------------

URGLAB_TOML = """schema_version = 1
name = "urglab"
task = "urglab"
runs = ["1"]
channels = ["ecg", "ppg", "rsp", "eda", "emg_cor"]
practice = true
ppg_requires_timing_marker = false
bad_ecg_from_machine_qc = true
ecg_inversion_option = true
keep_machine_stretches = ["emg_cor"]
raw_rows = ["rsp_raw"]
hidden_machine_reasons = { rsp = ["filter_edge"] }
"""
REFERENCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "urglab_reference_sidecar.json")
CERTIFIED = {"990": ("ecg", "bad_ecg", "ppg", "bad_ppg", "rsp", "eda", "emg_cor"),
             "991": ("ecg", "bad_ecg", "rsp", "emg_cor")}


def spans_of(items):
    return [[a["start"], a["end"]] for a in items]


def perfect_review(run_key, inverted):
    channels = {}
    for ch, truth in run_key["channels"].items():
        if truth["mode"] == "point":
            channels[ch] = {"mode": "point", "indices": list(truth["true_peaks"])}
        else:
            channels[ch] = {"mode": "segment",
                            "bad_segments": spans_of(truth["artifacts"]) + spans_of(truth.get("must_keep", []))}
    if inverted:
        channels["ecg"]["inverted"] = True
    return {"initials": "key", "channels": channels}


def check_generated(tmp):
    import numpy as np
    import mne
    import study_profile as sp
    import practice_urglab as pu
    import qrs_template
    import physio_review
    import practice_report_gui
    import processing_log
    import session_summary_gui
    from annotation_io import (combine_with_reference_channels, compute_scalings, raw_row_size_exclusions,
                               run_stage_b)
    from channel_config import RAW_ROW_CUT, select_channels
    from physio_io import hide_machine_reasons, load_physio_tsv, machine_qc_segments, read_sidecar, sidecar_path_for

    session_summary_gui.show_session_summary = session_summary_gui.headless_decision()
    toml = os.path.join(tmp, "urglab_pebbl_study.toml")
    with open(toml, "w", encoding="utf-8", newline="\n") as f:
        f.write(URGLAB_TOML)
    practice = os.path.join(tmp, "urglab_practice")

    print("6. make --study-profile <urglab's> writes sub-990 and sub-991 (run 1) and urglab's encoded key...")
    with contextlib.redirect_stdout(io.StringIO()):
        assert pc.main(["make", "--out-dir", practice, "--study-profile", toml,
                        "--key-copy", os.path.join(tmp, "staff", "urglab_key.json")]) == 0
    key = pc.load_key(pc.key_path_for(practice))
    assert key == pc.load_key(os.path.join(tmp, "staff", "urglab_key.json"))
    assert key["version"] == pc.KEY_VERSIONS["urglab"] == 2 and pc.key_study(key) == "urglab"
    assert key["seed"] == pu.SEED == 20261108, "version 2: one seed, because the marks are physiopipe's for it"
    computed = pu.load_machine_qc(pu.MACHINE_QC_FILE)
    for subject in pu.SUBJECTS:  # the signals are exactly the ones physiopipe computed its marks for
        df, _cases = pu.build_signals(subject)
        assert hashlib.sha256(pu.tsv_bytes(df)).hexdigest() == computed["subjects"][subject]["tsv_unzipped_sha256"], \
            (subject, "signals changed: physiopipe must recompute (practice_urglab.draft_for_physiopipe)")
    refused = os.path.join(tmp, "refused")
    real_build = pu.build_signals
    for patches, why in (
            ([], "always uses seed"),  # any other seed
            ([patch.object(pu, "MACHINE_QC_FILE", os.path.join(tmp, "nowhere.json"))], "is missing"),
            ([patch.object(pu, "build_signals", side_effect=lambda subject, seed=pu.SEED: (
                lambda df_cases: (df_cases[0].assign(ecg=df_cases[0]["ecg"] + 1e-6), df_cases[1]))(
                real_build(subject, seed)))], "aren't the ones physiopipe computed")):
        extra = ["--seed", "2024"] if why == "always uses seed" else []
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    pc.main(["make", "--out-dir", refused, "--study-profile", toml] + extra)
                raise AssertionError(f"make should refuse: {why}")
            except SystemExit as e:
                assert why in str(e), e
        assert not os.path.exists(pc.key_path_for(refused)), "nothing written when refused"
    assert {s: e["ecg_inverted"] for s, e in key["subjects"].items()} == {"990": False, "991": True}
    assert key["criteria"]["trap_max_coverage_by_channel"] == {"emg_cor": 0.10}
    assert key["descriptions_sha256"] == pu.description_hashes()
    other = os.path.join(tmp, "other.toml")
    with open(other, "w", encoding="utf-8", newline="\n") as f:
        f.write('schema_version = 1\nname = "x"\ntask = "elsewhere"\n')
    try:
        pc.main(["make", "--out-dir", os.path.join(tmp, "x"), "--study-profile", other])
        raise AssertionError("a study with no practice generator must be refused")
    except SystemExit as e:
        assert "no certification practice for the study with task 'elsewhere'" in str(e), e
    files = {}
    for subject in pu.SUBJECTS:
        tsv = os.path.join(practice, f"sub-{subject}", "ses-run1", "beh", f"sub-{subject}_ses-run1_task-urglab_physio.tsv.gz")
        df, sfreq = load_physio_tsv(tsv)
        sidecar = read_sidecar(sidecar_path_for(tsv))
        assert sfreq == 1000 and len(df) == pu.N_SAMPLES and list(df.columns) == pu.COLUMNS
        assert not df.isna().any().any(), "physiopipe writes no NaN"
        files[subject] = (tsv, df, sidecar)
    print("   OK")

    print("7. The files have the reference sidecar's shape (A6) and the study's own texts (A4)...")
    with open(REFERENCE, encoding="utf-8") as f:
        ref = json.load(f)
    with open(REFERENCE, "rb") as f:  # LF as physiopipe wrote it; a Windows checkout may have made it CRLF
        assert hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest().startswith("ced919006219"), "fixture"
    for subject, (tsv, df, side) in files.items():
        assert list(side) == list(ref), (subject, list(side))
        assert side["Columns"] == ref["Columns"] and side["SamplingFrequency"] == ref["SamplingFrequency"]
        assert side["StartTime"] == ref["StartTime"]
        for col in ref["Columns"]:
            assert set(side[col]) == set(ref[col]), col
            assert side[col]["Units"] == ref[col]["Units"] and side[col]["Description"] == ref[col]["Description"], col
        assert list(side["Events"]) == list(ref["Events"]) and side["Events"]["Codebook"] == ref["Events"]["Codebook"]
        assert set(side["Events"]["Rows"][0]) == set(ref["Events"]["Rows"][0])
        assert list(side["MachineQC"]) == list(ref["MachineQC"])
        assert side["MachineQC"]["CoordinateSpace"] == ref["MachineQC"]["CoordinateSpace"]
        for ch, entry in ref["MachineQC"]["Channels"].items():
            mine = side["MachineQC"]["Channels"][ch]
            assert list(mine) == list(entry) and mine["Rule"] == entry["Rule"], ch
            assert list(mine["SegmentsByReason"]) == list(entry["SegmentsByReason"]), (ch, list(mine["SegmentsByReason"]))
            spans = [s for v in mine["SegmentsByReason"].values() for s in v]
            for v in mine["SegmentsByReason"].values():
                assert all(isinstance(a, int) and isinstance(b, int) and 0 <= a < b <= len(df) for a, b in v), (ch, v)
                assert all(v[i][1] < v[i + 1][0] for i in range(len(v) - 1)), ("sorted, disjoint", ch, v)
            assert mine["BadSegments"] == pu._merge(spans), (ch, "BadSegments is the union")
        assert side["MachineQC"]["Channels"] == computed["subjects"][subject]["MachineQC"]["Channels"], \
            (subject, "the computer's marks are physiopipe's, unchanged")
        assert not side["MachineQC"]["Channels"]["rsp"]["SegmentsByReason"]["snr"]
        assert not side["MachineQC"]["Channels"]["rsp"]["SegmentsByReason"]["ecg_unconfirmed"]
        prov = side["Provenance"]
        assert prov["NumberOfSamples"] == len(df) and prov["Practice"] is True and prov["Junctions"] == []
        source = prov["SourceFiles"][0]
        assert source["SecondsFromFileStart"] == source["SecondsToFileEnd"] == pu.FILE_MARGIN_SEC
        assert side["MachineQC"]["Channels"]["rsp"]["SegmentsByReason"]["filter_edge"] == \
            [[0, 15000], [len(df) - 15000, len(df)]], "20 s from the file's ends, which lie 5 s outside the tsv"
    print("   OK: keys, columns, units, every reason list, Rules, Descriptions, spans; filter_edge 15 s")

    print("8. Controls: every key r/R is the top of its beat; the breathing cases are on the belt; PEBBL hides "
          "exactly filter_edge; the display cut still shows the cases...")
    for subject, (tsv, df, side) in files.items():
        rk = pc.run_key_for(key, subject, "1")
        upright = -df["ecg"].to_numpy() if key["subjects"][subject]["ecg_inverted"] else df["ecg"].to_numpy()
        noisy = spans_of(rk["channels"]["bad_ecg"]["artifacts"])
        for p in rk["channels"]["ecg"]["true_peaks"]:
            if not any(a <= p < b for a, b in noisy):
                assert upright[p] == upright[max(0, p - 15):p + 16].max(), (subject, p)
        belt = df["rsp_raw"].to_numpy()
        arts = {a["type"].split(" (")[0].split(":")[0]: (a["start"], a["end"]) for a in rk["channels"]["rsp"]["artifacts"]}
        if subject == "990":
            a, b = arts["no breathing"]
            # Only noise, after the AC coupling's small decay (under 0.15 V; breaths are about 1 V).
            assert belt[a + 8000:b - 1000].std() < 0.005 < 0.2 < belt[20000:a].std(), "the dead belt is only noise"
            assert np.ptp(belt[a:b - 1000]) < 0.15 * np.ptp(belt[16000:a]), "and nothing breath-sized in it"
            a, b = arts["a large slow drop at a task event"]
            assert belt[a - 500:a].mean() - belt[a:b + 500].min() > 2.0, "the slow drop: twice a breath's size"
        else:
            a, b = arts["the belt held at its limit, then a jump"]
            assert belt[a:b].min() <= -9.99 and belt[a:b].max() > 3.0, "at the limit, then past zero"
            a, b = arts["a 40-ms pop to the belt's limit"]
            assert belt[a:b].min() <= -9.99
            a, b = arts["a held breath"]
            assert np.ptp(belt[a + 3000:b]) < 0.5 * np.ptp(belt[a - 8000:a]), "no breaths while held"
        qc, _note = machine_qc_segments(side, len(df))
        shown = hide_machine_reasons(qc["rsp"], ["filter_edge"])
        assert shown["hidden_by_reason"] == {"filter_edge": [[0, 15000], [len(df) - 15000, len(df)]]}
        sp.activate(sp.load(path=toml))
        full = combine_with_reference_channels({"rsp": select_channels(["rsp"])["rsp"]}, df)
        scale = compute_scalings(df, full, exclude=raw_row_size_exclusions(full, side, len(df)))["seeg"]
        breathing = np.abs(belt[16000:24000] - belt.mean()).max()
        assert RAW_ROW_CUT * scale > 1.5 * breathing, (subject, scale, breathing)
    print("   OK")

    print("9. A perfect review passes every channel of both participants...")
    for subject in pu.SUBJECTS:
        rk = pc.run_key_for(key, subject, "1")
        ok, results, lines = pc.score_run(key, perfect_review(rk, key["subjects"][subject]["ecg_inverted"]), "1",
                                          subject=subject)
        assert ok and all(results.values()), "\n".join(lines)
    print("   OK")

    print("10. The untouched starting marks, saved through the real Step 2 code with urglab's settings, fail "
          "where the RA has work to do...")
    sp.activate(sp.load(path=toml))
    for subject in pu.SUBJECTS:
        tsv, df, side = files[subject]
        review = os.path.join(tmp, f"review_{subject}")
        stem = f"sub-{subject}_ses-run1_task-urglab"
        configs = {k: v for k, v in select_channels(None).items() if k in df.columns and k != "ppg"}
        with patch.object(mne.io.RawArray, "plot", return_value=None), contextlib.redirect_stdout(io.StringIO()):
            run_stage_b(df, 1000, configs, "zz", review, stem, "practice", sidecar=side, ecg_ppg_guide=False)
            result = run_stage_b(df, 1000, select_channels(["ppg"]), "zz", review, stem, "practice", sidecar=side)
        with open(result.saved_path) as f:
            untouched = json.load(f)
        by_reason = side["MachineQC"]["Channels"]["rsp"]["SegmentsByReason"]
        assert untouched["channels"]["rsp"]["bad_segments"] == pu._merge(
            [s for r, v in by_reason.items() if r != "filter_edge" for s in v]), "filter_edge is hidden, not saved"
        assert untouched["channels"]["emg_cor"]["bad_segments"] == side["MachineQC"]["Channels"]["emg_cor"]["BadSegments"]
        ok, results, lines = pc.score_run(key, untouched, "1", subject=subject)
        text = "\n".join(lines)
        assert not ok, text
        for ch in CERTIFIED[subject]:
            if ch in ("eda",):
                continue
            if subject == "991" and ch == "emg_cor":
                assert results[ch] is True and "OK   the computer's mark on a dead channel" in text, text
                continue
            assert results[ch] is False, (subject, ch, text)
        if subject == "990":
            assert results["eda"] is True, "the computer's EDA marks are right; real responses are left"
            dead_line = next(line for line in lines if "no breathing (the belt shows only fine noise)" in line)
            assert dead_line.startswith("  FIX") and dead_line.endswith("20% marked bad"), dead_line  # physiopipe's flat
            assert "FIX  a large slow drop at a task event" in text
            assert "OK   a sudden 4-V drop and return at a task event" in text, "the machine's swing is right"
            assert "MISS  a single small electrode pop" in text and "FIX  a train of electrode pops" in text
            assert "FIX  hand movement" in text and "sensor hit its limit" not in text, "dropped in version 2"
            assert "OK   electrode lift" in text and "OK   stuck sensor" in text, "physiopipe's floor and flat"
        else:
            assert "FIX  Flip the ECG: this ECG is upside down" in text, text
            assert "FIX  a held breath (no breathing)" in text and "OK   a 40-ms pop to the belt's limit" in text
            assert "marked bad where the signal is fine" in text and "FIX  " in text.split("RSP:")[1].split("\n\n")[0]
    print("   OK")

    print("11. Each mistake the plan names fails; a missed single small pop doesn't...")
    rk0, rk1 = pc.run_key_for(key, "990", "1"), pc.run_key_for(key, "991", "1")

    def score(subject, review, channels=None):
        return pc.score_run(key, review, "1", channels, subject)

    def segs(review, ch, spans):
        review["channels"][ch]["bad_segments"] = spans
        return review

    p0, p1 = perfect_review(rk0, False), perfect_review(rk1, True)
    assert not score("991", perfect_review(rk1, False), ["ecg"])[0], "991 reviewed unflipped"
    assert not score("990", perfect_review(rk0, True), ["ecg"])[0], "990 reviewed flipped"
    for offset in (42, 66):  # every mark on the S, or on the rebound after it
        moved = perfect_review(rk0, False)
        moved["channels"]["ecg"]["indices"] = [p + offset for p in rk0["channels"]["ecg"]["true_peaks"]]
        assert not score("990", moved, ["ecg"])[0], offset
    cor = rk0["channels"]["emg_cor"]
    frown = next(t for t in cor["traps"] if t["type"].startswith("a long held frown"))
    assert not score("990", segs(copy_review(p0), "emg_cor", p0["channels"]["emg_cor"]["bad_segments"]
                                 + [[frown["start"], frown["end"]]]), ["emg_cor"])[0], "a frown marked bad"
    keep = rk1["channels"]["emg_cor"]["must_keep"][0]  # 991's dead channel
    assert [keep["start"], keep["end"]] == [0, len(files["991"][1])] and "flat_record" in keep["type"]
    shortened = [[keep["start"], (keep["start"] + keep["end"]) // 2]]
    assert not score("991", segs(copy_review(p1), "emg_cor", shortened), ["emg_cor"])[0], "machine COR shortened"
    assert not cor.get("must_keep"), "990's COR has no computer marks in version 2"
    single = next(a for a in cor["artifacts"] if a.get("required") is False)
    without = [s for s in p0["channels"]["emg_cor"]["bad_segments"] if s != [single["start"], single["end"]]]
    ok, _, lines = score("990", segs(copy_review(p0), "emg_cor", without), ["emg_cor"])
    assert ok and any(line.startswith("  MISS  a single small electrode pop") for line in lines), lines
    rsp0, rsp1 = rk0["channels"]["rsp"], rk1["channels"]["rsp"]
    for subject, perfect, truth in (("990", p0, rsp0), ("991", p1, rsp1)):
        for trap in truth["traps"]:  # the sigh; shallow breathing
            marked = perfect["channels"]["rsp"]["bad_segments"] + [[trap["start"], trap["end"]]]
            assert not score(subject, segs(copy_review(perfect), "rsp", marked), ["rsp"])[0], trap["type"]
    rail = files["991"][2]["MachineQC"]["Channels"]["rsp"]["SegmentsByReason"]["rail"]
    kept_rail = pu._merge(p1["channels"]["rsp"]["bad_segments"] + rail)
    ok, _, lines = score("991", segs(copy_review(p1), "rsp", kept_rail), ["rsp"])
    assert not ok and any("marked bad where the signal is fine" in line and line.startswith("  FIX") for line in lines)
    machine_flat = files["990"][2]["MachineQC"]["Channels"]["rsp"]["SegmentsByReason"]["flat"]
    dead = next(a for a in rsp0["artifacts"] if a["type"].startswith("no breathing"))
    only_machine = [s for s in p0["channels"]["rsp"]["bad_segments"] if s != [dead["start"], dead["end"]]] + machine_flat
    assert not score("990", segs(copy_review(p0), "rsp", only_machine), ["rsp"])[0], "the partial flat alone"
    lead_off = p0["channels"]["bad_ecg"]["bad_segments"] + rk0["channels"]["bad_ecg"]["neutral"]
    assert score("990", segs(copy_review(p0), "bad_ecg", lead_off), ["bad_ecg"])[0], \
        "marking the whole lead-off, not only the computer's middle, is fine"
    drift = next(span for span in rsp0["neutral"] if span[0] > 20000 and span[1] < 160000)
    drifted = p0["channels"]["rsp"]["bad_segments"] + [drift]
    assert score("990", segs(copy_review(p0), "rsp", drifted), ["rsp"])[0], "the drift back after a step is neutral"
    print("   OK")

    print("12. A practice session for 991 through physio_review: Step 1, then Step 2, scored as participant 991...")
    reports = []
    snapshot_dir = tempfile.mkdtemp(prefix="urglab_snapshot_", dir=tmp)
    argv = ["physio_review.py", "--box-path", practice, "--subject", "991", "--initials", "xyz", "--no-local-copy",
            "--practice", "--study-profile", toml]
    with patch.object(mne.io.RawArray, "plot", return_value=None), \
         patch.object(qrs_template, "show_template_preview", return_value=True), \
         patch.object(processing_log, "LOCAL_SNAPSHOT_DIR", snapshot_dir), \
         patch.object(practice_report_gui, "show_practice_report",
                      side_effect=lambda heading, lines, passed: reports.append((heading, list(lines), passed)) or False), \
         patch("builtins.input", side_effect=AssertionError("no terminal prompt expected")), \
         contextlib.redirect_stdout(io.StringIO()):
        with patch.object(sys, "argv", argv + ["--stage", "1"]):
            physio_review.main()
        with patch.object(sys, "argv", argv + ["--stage", "2", "--run", "1", "--channels", "ecg"]):
            physio_review.main()
    assert len(reports) == 1, reports
    heading, lines, passed = reports[0]
    assert heading.startswith("Practice participant 991, run 1") and passed is False, heading
    assert "  FIX  Flip the ECG: this ECG is upside down (its main spikes point down). Tick 'Flip the ECG' and start " \
           "the ECG over" in lines, lines
    assert os.listdir(snapshot_dir) == [], "practice never touches the real log's local snapshot"
    print("   OK")


def copy_review(review):
    return json.loads(json.dumps(review))


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nFAILED: {e}")
        sys.exit(1)
