#!/usr/bin/env python3
"""
Tests practice_certification.py (key version 5, 2026-10-04) against answers
known in advance:
  1. make writes sub-990 runs 1 and 2 (loadable by the tool's own reader) and
     the encoded answer key in the practice folder; the key round-trips;
  2. a PERFECT review passes every channel in both runs;
  3. the UNTOUCHED starting marks, saved through the real Step 2 code (viewer
     mocked), fail and name what to fix (missed beats, extra marks, the
     machine's missed artifact and false flag);
  4. deleting the premature beat or marking a real skin conductance response
     fails; a PPG mark on either crest of a two-humped pulse passes, and two
     marks on one pulse count as an extra; beats inside the RA's own bad
     stretch are excused;
  5. the command-line scorer reads the run from the file name;
  6. the tool's interval check flags the premature beat (so RAs look at it,
     and must keep it).

Run with: annotate_env\\Scripts\\python.exe test_practice_certification.py
"""

import contextlib
import copy
import io
import json
import os
import shutil
import tempfile
from unittest.mock import patch

import mne

import practice_certification as pc
import session_summary_gui
from annotation_io import check_interval_regularity, run_stage_b
from channel_config import CHANNELS, select_channels
from physio_io import load_physio_tsv, read_sidecar, sidecar_path_for

session_summary_gui.show_session_summary = session_summary_gui.headless_decision()


def perfect(run_key):
    channels = {}
    for ch, truth in run_key["channels"].items():
        if truth["mode"] == "point":
            channels[ch] = {"mode": "point", "indices": list(truth["true_peaks"])}
        else:
            channels[ch] = {"mode": "segment", "bad_segments": [[a["start"], a["end"]] for a in truth["artifacts"]]}
    return {"initials": "key", "channels": channels}


def main():
    tmp = tempfile.mkdtemp(prefix="practice_cert_")
    try:
        practice = os.path.join(tmp, "practice")
        print("1. make: sub-990 runs 1 and 2 plus the encoded key in the practice folder...")
        with contextlib.redirect_stdout(io.StringIO()):
            written, key_path = pc.make(practice, seed=2024, plain_key_copy=os.path.join(tmp, "staff", "key.json"))
        assert key_path == os.path.join(practice, "practice_key.pebbl") and len(written) == 2
        with open(key_path, encoding="utf-8") as f:
            raw_key = f.read()
        assert raw_key.startswith(pc.KEY_HEADER) and '"true_peaks"' not in raw_key, "not readable at a glance"
        key = pc.load_key(key_path)
        assert key == pc.load_key(os.path.join(tmp, "staff", "key.json")), "the plain copy is the same key"
        assert key["version"] == pc.KEY_VERSION and sorted(key["runs"]) == ["1", "2"]
        for run, tsv in zip(pc.RUNS, written):
            assert f"ses-run{run}" in tsv
            df, sfreq = load_physio_tsv(tsv)
            assert sfreq == 1000 and len(df) == pc.DURATION_SEC * 1000 and list(df.columns) == pc.COLUMNS
            sidecar = read_sidecar(sidecar_path_for(tsv))
            assert sidecar["Provenance"]["Practice"] is True and sidecar["Provenance"]["NumberOfSamples"] == len(df)
            assert df["sbp"].isna().any(), "blank no-reading stretches"
        run1 = key["runs"]["1"]
        assert set(run1["channels"]) == {"ecg", "bad_ecg", "ppg", "bad_ppg", "rsp", "eda", "sbp", "dbp"}
        assert any(len(a) > 1 for a in run1["channels"]["ppg"]["accept"]), "some pulses have two crests"
        print(f"   OK: runs 1 and 2; ECG beats {[len(key['runs'][r]['channels']['ecg']['true_peaks']) for r in pc.RUNS]}")

        print("2. A perfect review passes every channel in both runs...")
        for run in pc.RUNS:
            ok, results, lines = pc.score_run(key, perfect(key["runs"][run]), run)
            assert ok and all(results.values()), "\n".join(lines)
        print("   OK")

        print("3. The untouched starting marks, saved through the real Step 2 code, fail and say what to fix...")
        tsv = written[0]
        df, sfreq = load_physio_tsv(tsv)
        sidecar = read_sidecar(sidecar_path_for(tsv))
        review = os.path.join(tmp, "review")
        everything_but_ppg = {k: v for k, v in select_channels(None).items() if k in df.columns}
        with patch.object(mne.io.RawArray, "plot", return_value=None), contextlib.redirect_stdout(io.StringIO()):
            run_stage_b(df, sfreq, everything_but_ppg, "zz", review, "sub-990_ses-run1_task-sdi", "practice",
                        sidecar=sidecar, ecg_ppg_guide=False)
            result = run_stage_b(df, sfreq, select_channels(["ppg"]), "zz", review, "sub-990_ses-run1_task-sdi",
                                 "practice", sidecar=sidecar)
        with open(result.saved_path) as f:
            untouched = json.load(f)
        for ch in ("rsp", "eda", "sbp", "dbp"):
            assert untouched["channels"][ch]["bad_segments"] == sidecar["MachineQC"]["Channels"][ch]["BadSegments"], ch
        ok, results, lines = pc.score_run(key, untouched, "1")
        text = "\n".join(lines)
        assert not ok and results["bad_ppg"] is True, (results, text)  # the machine's dropout mark is right
        for ch in ("ecg", "bad_ecg", "ppg", "rsp", "eda", "sbp", "dbp"):
            assert results[ch] is False, (ch, text)
        assert "extra mark(s) where there's no heartbeat" in text and "with no mark within 20 ms" in text, text
        for ch in ("rsp", "eda", "sbp", "dbp"):
            missed = run1["channels"][ch]["machine_qc"]["missed"]
            assert f"FIX  {missed}" in text, (ch, missed)
        assert "the machine's mark on the breath hold (flagged by mistake)" in text, text
        print(f"   OK: {sum(1 for v in results.values() if not v)} channels NOT YET; missed beats, extras, the "
              f"machine's missed artifacts and false flags all listed")

        print("4. Traps, either crest, two marks on one pulse, beats inside the RA's own bad stretch...")
        sub = perfect(run1)
        pvc = run1["channels"]["ecg"]["traps"][0]["sample"]
        sub["channels"]["ecg"]["indices"].remove(pvc)
        scr = run1["channels"]["eda"]["traps"][0]
        sub["channels"]["eda"]["bad_segments"].append([scr["start"], scr["end"]])
        ok, results, lines = pc.score_run(key, sub, "1", ["ecg", "eda", "ppg"])
        text = "\n".join(lines)
        assert results == {"ecg": False, "eda": False, "ppg": True}, results
        assert "premature (early) beat: real, keep its mark" in text and "skin conductance response" in text
        sub = perfect(run1)
        ppg = run1["channels"]["ppg"]
        i = next(k for k, a in enumerate(ppg["accept"]) if len(a) > 1)
        top = ppg["true_peaks"][i]
        other = next(c for c in ppg["accept"][i] if c != top)
        sub["channels"]["ppg"]["indices"] = [other if x == top else x for x in sub["channels"]["ppg"]["indices"]]
        assert pc.score_run(key, sub, "1", ["ppg"])[1] == {"ppg": True}, "the other hump counts"
        sub["channels"]["ppg"]["indices"].append(top)
        ok, results, lines = pc.score_run(key, sub, "1", ["ppg"])
        assert results == {"ppg": False} and "1 extra mark(s)" in "\n".join(lines), lines
        noise = run1["channels"]["bad_ecg"]["artifacts"][0]
        sub = perfect(run1)
        sub["channels"]["ecg"]["indices"] = [x for x in sub["channels"]["ecg"]["indices"]
                                             if not noise["start"] <= x < noise["end"]]
        ok, results, lines = pc.score_run(key, sub, "1", ["ecg", "bad_ecg"])
        assert ok and "inside your bad stretches (fine)" in "\n".join(lines), lines
        sub["channels"]["bad_ecg"]["bad_segments"] = []
        assert pc.score_run(key, sub, "1", ["ecg"])[1] == {"ecg": False}, "unmarked noise: the beats there count"
        print("   OK")

        print("5. The command-line scorer finds the run in the file name...")
        path = os.path.join(tmp, "sub-990_ses-run2_task-sdi_annotations_key.json")
        with open(path, "w") as f:
            json.dump(perfect(key["runs"]["2"]), f)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = pc.main(["score", "--key", key_path, "--annotations", path])
        assert code == 0 and "run 2" in out.getvalue() and "OVERALL: PASS" in out.getvalue(), out.getvalue()
        print("   OK")

        print("6. The tool's interval check flags the premature beat in a perfect ECG...")
        with contextlib.redirect_stdout(io.StringIO()):
            flagged = check_interval_regularity({"ecg": {"mode": "point", "indices": run1["channels"]["ecg"]["true_peaks"]}},
                                                {"ecg": CHANNELS["ecg"]}, sfreq)
        assert any(abs(onset - pvc / sfreq) < 2.0 for _ch, onset, *_ in flagged), flagged
        print(f"   OK: flagged near {pvc / sfreq:.1f} s")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL PRACTICE CERTIFICATION TESTS PASSED.")


if __name__ == "__main__":
    main()
