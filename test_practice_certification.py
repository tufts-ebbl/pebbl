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
import numpy as np

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
        assert not ok and results["bad_ppg"] is False, (results, text)
        assert "OK   finger-pulse dropout" in text, "the machine's dropout mark is right"
        assert "FIX  the weak, odd pulse after the premature beat: mark it bad_ppg" in text, \
            "the machine doesn't mark the weak pulse; the RA must"
        for ch in ("ecg", "bad_ecg", "ppg", "bad_ppg", "rsp", "eda", "sbp", "dbp"):
            assert results[ch] is False, (ch, text)
        assert "extra mark(s) where there's no heartbeat" in text and "with no mark within 20 ms" in text, text
        for ch in ("rsp", "eda", "sbp", "dbp"):
            missed = run1["channels"][ch]["machine_qc"]["missed"]
            assert f"FIX  {missed}" in text, (ch, missed)
        assert "the machine's mark on shallow breathing (flagged by mistake)" in text, text
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

        print("7. The ends of a run aren't scored (HLU, 2026-10-05: a correct mark on a visible beat at ~179.7 s "
              "was called extra, because the key leaves out the last 0.6 s); the premature beat's key sample is "
              "the top of the final signal...")
        n = run1["n_samples"]
        ecg_df, _ = load_physio_tsv(os.path.join(practice, "sub-990", "ses-run1", "beh",
                                                 "sub-990_ses-run1_task-sdi_physio.tsv.gz"))
        signal = ecg_df["ecg"].to_numpy()
        last_true = max(run1["channels"]["ecg"]["true_peaks"])
        tail = signal[last_true + 400:]  # a beat after the key's last one, if the run has one there
        sub = perfect(run1)
        sub["channels"]["ecg"]["indices"].append(n - 300)   # in the last second: never "extra"
        sub["channels"]["ecg"]["indices"].append(int(0.2 * sfreq))  # in the first half second: never "extra"
        assert pc.score_run(key, sub, "1", ["ecg"])[1] == {"ecg": True}, "edge marks aren't extra"
        sub = perfect(run1)
        sub["channels"]["ecg"]["indices"] = [i for i in sub["channels"]["ecg"]["indices"] if i < n - 1000]
        dropped = len(perfect(run1)["channels"]["ecg"]["indices"]) - len(sub["channels"]["ecg"]["indices"])
        ok, _res, lines = pc.score_run(key, sub, "1", ["ecg"])
        assert ok, f"beats in the last second aren't required ({dropped} left unmarked): {lines}"
        sub["channels"]["ecg"]["indices"].append(n - 5000)  # 5 s from the end is scored as usual
        assert not pc.score_run(key, sub, "1", ["ecg"])[0], "an extra mark away from the ends still fails"
        assert signal[pvc] == signal[max(0, pvc - 30):pvc + 31].max(), "the PVC's key sample is the final signal's top"
        print(f"   OK: marks in the first 0.5 s / last 1.0 s are never extra; beats there aren't required "
              f"({len(tail)} samples after the key's last beat); the PVC sample is the signal's top")

        print("8. RSP (HLU, 2026-10-05: mark bad wherever you can't see where each breath peaks): the flat held "
              "breath and real clipping (tops cut flat at the run's maximum) are artifacts; the sigh is one whole "
              "breath; the 'marked bad where the signal is fine' line is a total and says where...")
        rsp_key = run1["channels"]["rsp"]
        kinds = [a["type"] for a in rsp_key["artifacts"]]
        assert any(k.startswith("flat line (a held breath)") for k in kinds), kinds
        assert any(k.startswith("clipped breath tops") for k in kinds), kinds
        assert [t["type"].split(":")[0] for t in rsp_key["traps"]] == ["a sigh (one deep breath)", "shallow breathing"]
        rsp_signal = ecg_df["rsp"].to_numpy()
        clip = next(a for a in rsp_key["artifacts"] if a["type"].startswith("clipped"))
        at_max = np.flatnonzero(rsp_signal >= rsp_signal.max() - 1e-9)
        assert len(at_max) > 0.5 * sfreq and clip["start"] <= at_max.min() and at_max.max() < clip["end"], \
            "the flat tops sit at the run's maximum, inside the clipping artifact"
        sigh = rsp_key["traps"][0]
        assert 2.5 <= (sigh["end"] - sigh["start"]) / sfreq <= 8.0, "a whole breath"
        sub = perfect(run1)
        sub["channels"]["rsp"]["bad_segments"] += [[sigh["start"], sigh["end"]]]
        ok, _res, lines = pc.score_run(key, sub, "1", ["rsp"])
        fine_line = next(line for line in lines if "marked bad where the signal is fine" in line)
        assert not ok and "s in all marked bad" in fine_line, fine_line
        assert f"at {sigh['start'] / sfreq:.1f}-" in fine_line, fine_line
        print(f"   OK: {fine_line.strip()}")

        print("9. PPG (HLU, 2026-10-05): the weak pulse after the premature beat is a bad_ppg stretch (its own "
              "wave), not a pulse to keep; marking up to 2 s either side of the dropout is fine...")
        bad_ppg = run1["channels"]["bad_ppg"]
        drop, weak = bad_ppg["artifacts"]
        assert drop.get("leeway_sec") == 2.0 and "weak, odd pulse" in weak["type"]
        assert run1["channels"]["ppg"]["traps"] == []
        assert (weak["end"] - weak["start"]) / sfreq < 1.2, "the odd pulse itself, not the pause after it"
        sub = perfect(run1)
        sub["channels"]["bad_ppg"]["bad_segments"] = [[drop["start"] - 1900, drop["end"] + 1900],
                                                      [weak["start"], weak["end"]]]
        assert pc.score_run(key, sub, "1", ["bad_ppg"])[1] == {"bad_ppg": True}, "1.9 s each side of the dropout"
        sub["channels"]["bad_ppg"]["bad_segments"][0] = [drop["start"] - 4500, drop["end"] + 4500]
        assert pc.score_run(key, sub, "1", ["bad_ppg"])[1] == {"bad_ppg": False}, "4.5 s each side is too much"
        sub["channels"]["bad_ppg"]["bad_segments"] = [[drop["start"], drop["end"]]]
        ok, _res, lines = pc.score_run(key, sub, "1", ["bad_ppg"])
        assert not ok and any(l.startswith("  FIX  the weak, odd pulse") for l in lines), lines
        sub = perfect(run1)
        sub["channels"]["ppg"]["indices"].remove(run1["channels"]["ppg"]["true_peaks"][
            int(np.argmin(np.abs(np.asarray(run1["channels"]["ppg"]["true_peaks"]) - (weak["start"] + weak["end"]) / 2)))])
        assert pc.score_run(key, sub, "1", ["ppg", "bad_ppg"])[0], "no mark on the weak pulse inside its bad stretch"
        print("   OK: weak pulse is bad_ppg; 1.9 s around the dropout passes, 4.5 s doesn't; unmarked weak pulse fails; "
              "its peak needn't be marked")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL PRACTICE CERTIFICATION TESTS PASSED.")


if __name__ == "__main__":
    main()
