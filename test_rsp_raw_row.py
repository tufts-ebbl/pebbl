#!/usr/bin/env python3
"""
Tests the raw breathing row (HLU, 2026-11-08; pebbl-private pebbl_rsp-raw-row-plan_20261108-1108.md): in a study
whose profile lists raw_rows = ["rsp_raw"], RSP is shown as rsp_raw, the belt as recorded, in place of the filtered
rsp row; RAs judge and mark there, and the marks save under RSP as before. Also the profile's
hidden_machine_reasons (machine stretches the study's analysis applies itself, e.g. RSP's filter_edge: not shown),
RSP's plain reason names, and rsp_raw's fingerprint in saved reviews.

Data: fixtures/rsp_raw_planted_130-180s.npz, 50 s of physiopipe's synthetic breathing file (see fixtures/README.md):
a belt held at the -10 V limit (140.9-143.0 s, machine: rail + swing) and a single 3-V belt step with a slow drift
back (175.0 s, machine: swing).

  1. The profile settings are read and checked.
  2. The row is shown in RSP's place only when the profile lists it; a file without it stops an RSP review only.
  3. Its type, unit and data (the column, unfiltered).
  4. Its size leaves out the belt's limit and swings; its cut is 2.5 x that size, so the drift back after a step
     is drawn (a cut at the row's own edge, as for COR, would flatten it: the negative control).
  5. Hidden machine reasons and RSP's reason names.
  6. A whole Step 2 session: rows, banner, seeds, saved review, fingerprint, and the stop when rsp_raw changes.
  7. Step 3 shows the row the same way.

Run with: annotate_env\\Scripts\\python.exe test_rsp_raw_row.py
"""

import contextlib
import gzip
import io
import json
import os
import shutil
import tempfile
from unittest.mock import patch

import mne
import numpy as np
import pandas as pd

import annotation_io
import processing_log
import session_summary_gui
import study_profile
from annotation_io import (combine_with_reference_channels, export_annotations, raw_row_size_exclusions,
                           run_stage_b, session_checks_before_viewer)
from channel_config import CHANNELS, RAW_ROW_CUT, RAW_ROWS
from physio_io import build_raw, clamp_display_rows, compute_scalings, hide_machine_reasons, reason_summary
from reconcile import run_stage_c

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixtures", "rsp_raw_planted_130-180s.npz")
URGLAB_LIKE = study_profile.StudyProfile(name="urglab-like", task="planted", path="test", sha256="test",
                                         raw_rows=["rsp_raw"], hidden_machine_reasons={"rsp": ["filter_edge"]})


def load_fixture():
    z = np.load(FIXTURE)
    df = pd.DataFrame({"rsp": z["rsp"], "event": z["event"], "rsp_raw": z["rsp_raw"]})
    return df, int(z["sfreq"]), float(z["start_s"]), json.loads(str(z["by_reason"]))


def rows_at(df, sfreq, start_s, a_s, b_s):
    """Row numbers of the fixture for source-file seconds a_s..b_s."""
    return int((a_s - start_s) * sfreq), int((b_s - start_s) * sfreq)


def middle95_half(x):
    return (np.nanpercentile(x, 97.5) - np.nanpercentile(x, 2.5)) / 2


def write_profile(text):
    fd, path = tempfile.mkstemp(suffix=".toml")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write('schema_version = 1\nname = "t"\ntask = "t"\n' + text)
    return path


def refused(text):
    path = write_profile(text)
    try:
        study_profile.read_profile(path)
    except SystemExit as exc:
        return str(exc)
    finally:
        os.remove(path)
    return None


def main():
    df, sfreq, start_s, by_reason = load_fixture()
    n = len(df)

    print("1. The profile settings raw_rows and hidden_machine_reasons are read and checked...")
    path = write_profile('raw_rows = ["rsp_raw"]\nhidden_machine_reasons = { rsp = ["filter_edge"] }\n')
    profile = study_profile.read_profile(path)
    os.remove(path)
    assert profile.raw_rows == ["rsp_raw"] and profile.hidden_machine_reasons == {"rsp": ["filter_edge"]}
    assert study_profile.BUILT_IN.raw_rows == [] and study_profile.BUILT_IN.hidden_machine_reasons == {}
    assert "PEBBL doesn't know: eda_raw" in refused('raw_rows = ["eda_raw"]\n')
    assert "raw_rows must be a list" in refused('raw_rows = "rsp_raw"\n')
    assert "doesn't offer: rsp_raw" in refused('channels = ["ecg"]\nraw_rows = ["rsp_raw"]\n')
    assert "aren't bad-stretch channels: ecg" in refused('hidden_machine_reasons = { ecg = ["flat"] }\n')
    assert "must map channels to lists" in refused('hidden_machine_reasons = { rsp = [] }\n')
    print("   OK")

    print("2. The row is shown in RSP's place only when the profile lists it; a file without it stops RSP only...")
    rsp_only = {"rsp": CHANNELS["rsp"]}
    assert "rsp_raw" not in combine_with_reference_channels(rsp_only, df), "built-in settings: no raw row"
    study_profile.activate(URGLAB_LIKE)
    full = combine_with_reference_channels(rsp_only, df)
    assert list(full)[:2] == ["rsp_raw", "rsp"], list(full)
    raw = build_raw(df, full, sfreq)
    assert raw.ch_names == ["rsp_raw", "event"], raw.ch_names
    eda_session = combine_with_reference_channels({"eda": CHANNELS["eda"]}, df.assign(eda=1.0), ppg_guide=True)
    assert "rsp_raw" not in eda_session, "never in an EDA session, where RSP is only the guide"
    for broken, why in ((df.drop(columns="rsp_raw"), "no column"), (df.assign(rsp_raw=np.nan), "all NaN")):
        try:
            session_checks_before_viewer(rsp_only, None, False, df=broken)
            raise AssertionError(f"an RSP review should stop: {why}")
        except SystemExit as exc:
            assert "no rsp_raw column (the breathing belt as recorded)" in str(exc), str(exc)
        session_checks_before_viewer({"eda": CHANNELS["eda"]}, None, False, df=broken.assign(eda=1.0))
    print("   OK: rows", raw.ch_names)

    print("3. Its display type, unit and data: the column as recorded...")
    assert raw.get_channel_types(picks=[0]) == [RAW_ROWS["rsp_raw"]["mne_type"]]
    assert raw.get_channel_types(picks=[0])[0] not in {c["mne_type"] for c in CHANNELS.values()}
    assert mne.defaults.DEFAULTS["units"][RAW_ROWS["rsp_raw"]["mne_type"]] == "V"
    assert np.array_equal(raw.get_data(picks=[0])[0], df["rsp_raw"].to_numpy())
    print("   OK")

    print("4. Size without the belt's limit and swings; a cut at 2.5 x, which keeps the drift back after a step...")
    machine = {"rsp": {"BadSegments": sorted(s for spans in by_reason.values() for s in spans),
                       "SegmentsByReason": by_reason, "Rule": "planted"}}
    sidecar = {"Provenance": {"NumberOfSamples": n}, "MachineQC": {"Channels": machine}}
    exclude = raw_row_size_exclusions(full, sidecar, n)
    assert set(exclude) == {"rsp_raw"} and sorted(exclude["rsp_raw"]) == sorted(by_reason["rail"] + by_reason["swing"])
    kept = np.ones(n, dtype=bool)
    for s, e in exclude["rsp_raw"]:
        kept[s:e] = False
    scale = compute_scalings(df, full, exclude=exclude)["seeg"]
    assert np.isclose(scale, middle95_half(df["rsp_raw"].to_numpy()[kept]))
    whole = compute_scalings(df, full)["seeg"]
    assert np.isclose(whole, middle95_half(df["rsp_raw"].to_numpy())) and scale < whole, (scale, whole)
    assert np.isclose(compute_scalings(df, full, exclude={"rsp_raw": [[0, n]]})["seeg"], whole), "all left out"
    assert raw_row_size_exclusions(full, {"Provenance": {"NumberOfSamples": n}}, n) == {}, "no breakdown: all rows"

    hold = slice(*rows_at(df, sfreq, start_s, 141.2, 142.0))       # the belt held at its limit
    drift = slice(*rows_at(df, sfreq, start_s, 175.6, 178.5))      # the slow drift back after the step

    def at_cut(factor):
        r = build_raw(df, full, sfreq)
        with patch.dict(RAW_ROWS["rsp_raw"], {"clamp_display": factor}):
            clamp_display_rows(r, {**full, "rsp_raw": RAW_ROWS["rsp_raw"]}, {"seeg": scale})
        shown = r.get_data(picks=[0])[0]
        limit = factor * scale
        edge = np.isclose(np.abs(shown - df["rsp_raw"].mean()), limit, atol=1e-9)
        return shown, limit, edge

    shown, limit, edge = at_cut(RAW_ROW_CUT)
    assert np.abs(shown - df["rsp_raw"].mean()).max() <= limit + 1e-9
    assert edge[hold].mean() > 0.9, edge[hold].mean()          # the limit hold shows as a flat line at the cut
    assert edge[drift].mean() < 0.05, edge[drift].mean()       # the drift back is drawn, not cut
    _, _, edge_cor_style = at_cut(1.0)
    assert edge_cor_style[drift].mean() > 0.3, edge_cor_style[drift].mean()  # negative control: COR's cut flattens it
    print(f"   OK: size {scale:.3f} V (whole recording {whole:.3f} V); at the cut: limit hold "
          f"{edge[hold].mean():.0%}, drift back {edge[drift].mean():.0%} (with COR's cut {edge_cor_style[drift].mean():.0%})")

    print("5. Hidden machine reasons, and RSP's plain reason names...")
    entry = {"segments": [[0, 300], [500, 600]], "by_reason": {"filter_edge": [[0, 200]], "flat": [[100, 300]],
                                                                 "swing": [[500, 600]]}}
    shown_entry = hide_machine_reasons(entry, ["filter_edge"])
    assert shown_entry["segments"] == [[100, 300], [500, 600]], "a stretch another reason covers stays"
    assert shown_entry["hidden_by_reason"] == {"filter_edge": [[0, 200]]} and "filter_edge" not in shown_entry["by_reason"]
    assert hide_machine_reasons({**entry, "by_reason_invalid": True}, ["filter_edge"])["segments"] == entry["segments"]
    assert hide_machine_reasons({**entry, "segments": [[0, 700]]}, ["filter_edge"])["segments"] == [[0, 700]], \
        "reasons that don't add up to the stretches: everything shown"
    assert hide_machine_reasons(entry, [])["segments"] == entry["segments"]
    assert reason_summary({"flat": [[0, 1]], "flat_raw": [[2, 3]], "swing": [[4, 5]], "odd": [[6, 7]]}, "rsp") == \
        "no breathing movement (1), belt output dead or unplugged (1), sudden belt step bigger than a breath (1), odd (1)"
    assert reason_summary({"flat": [[0, 1]]}, "ecg") == "flat (1)", "another channel's 'flat' keeps its own name"
    print("   OK")

    print("6. A whole Step 2 session: rows, banner, seeds, saved review, fingerprint, and the stop on a changed row...")
    planted_edge = [[0, 2000]]  # a filter_edge stretch the study's analysis applies itself: not shown
    reasons = {**by_reason, "filter_edge": planted_edge}
    machine = {"rsp": {"BadSegments": [[0, 2000]] + machine["rsp"]["BadSegments"], "SegmentsByReason": reasons,
                       "Rule": "planted"}}
    columns = ["rsp", "event", "rsp_raw"]
    sidecar = {"Columns": columns, "Provenance": {"NumberOfSamples": n, "PipelineCommit": "planted"},
               "rsp": {"Units": "V", "Description": "Filtered breathing."},
               "rsp_raw": {"Units": "V", "Description": "The breathing belt as recorded (planted text)."},
               "MachineQC": {"Channels": machine}}
    tmp = tempfile.mkdtemp()
    tsv = os.path.join(tmp, "sub-999_ses-run1_task-planted_physio.tsv.gz")
    stem = "sub-999_ses-run1_task-planted"

    def write_tsv(frame):
        with gzip.open(tsv, "wt", newline="") as f:
            frame[columns].to_csv(f, sep="\t", header=False, index=False, lineterminator="\n")

    seen = {}

    def capture(self, *args, **kwargs):
        seen["rows"] = list(self.ch_names)
        seen["marks"] = [(o, d, desc, tuple(ch)) for o, d, desc, ch in zip(
            self.annotations.onset, self.annotations.duration, self.annotations.description, self.annotations.ch_names)]

    def session(frame):
        out = io.StringIO()
        with patch.object(mne.io.RawArray, "plot", capture), \
             patch.object(processing_log, "LOCAL_SNAPSHOT_DIR", tmp), \
             patch.object(session_summary_gui, "show_session_summary", session_summary_gui.headless_decision()), \
             contextlib.redirect_stdout(out):
            run_stage_b(frame.copy(), sfreq, dict(rsp_only), "aaa", tmp, stem, "planted", sidecar=sidecar, tsv_path=tsv)
        return out.getvalue()

    try:
        write_tsv(df)
        text = session(df)
        assert seen["rows"] == ["rsp_raw", "event"], seen["rows"]
        assert "RSP: Filtered breathing." in text, "the reviewed channel's Description is the one printed"
        assert "planted text" not in text, "the raw row's own Description isn't printed"
        assert RAW_ROWS["rsp_raw"]["text"] in text
        assert "RSP: not shown, the machine's first or last 20 s (1): the analysis leaves those stretches out" in text
        assert "belt at its limit (1), sudden belt step bigger than a breath (2)" in text, text[-2000:]
        with open(os.path.join(tmp, f"{stem}_annotations_aaa.json")) as f:
            saved = json.load(f)["channels"]["rsp"]
        want = sorted(by_reason["rail"] + by_reason["swing"])
        merged = []
        for s, e in want:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        assert saved["bad_segments"] == merged, (saved["bad_segments"], merged)
        assert saved["seed"]["not_shown"] == {"filter_edge": planted_edge}
        assert saved["provenance"]["raw_column"] == "rsp_raw"
        assert len(saved["provenance"]["raw_signal_sha256"]) == 64
        assert saved["value_range"]["max"] == float(df["rsp_raw"].max()), "the range of the row on screen"
        # A mark drawn while looking at rsp_raw, even tied to that row by Shift+click, saves under RSP.
        r = build_raw(df, full, sfreq)
        r.annotations.append(30.0, 1.0, "bad_rsp", ch_names=[["rsp_raw"]])
        assert export_annotations(r, df, full, sfreq)["rsp"]["bad_segments"] == [[30000, 31000]]
        # rsp_raw changed after the review was saved: reopening it stops.
        changed = df.copy()
        changed.loc[100, "rsp_raw"] += 0.5
        write_tsv(changed)
        try:
            session(changed)
            raise AssertionError("a changed rsp_raw should stop the reopened review")
        except SystemExit as exc:
            assert "rsp_raw column's values have changed" in str(exc), str(exc)
        write_tsv(df)
        print(f"   OK: {len(saved['bad_segments'])} machine stretch(es) shown, filter_edge held back; fingerprint "
              f"{saved['provenance']['raw_signal_sha256'][:12]}...")

        print("7. Step 3 (reconciliation) shows the row the same way...")
        seen.clear()
        reviews = {"rsp": {"mode": "segment", "bad_segments": merged}}
        out = io.StringIO()
        with patch.object(mne.io.RawArray, "plot", capture), \
             patch.object(processing_log, "LOCAL_SNAPSHOT_DIR", tmp), \
             patch.object(session_summary_gui, "show_session_summary", session_summary_gui.headless_decision()), \
             contextlib.redirect_stdout(out):
            run_stage_c(df.copy(), sfreq, dict(rsp_only), tmp, stem, "planted", reviews, reviews, "aaa", "bbb",
                        reconciler="hlu", sidecar=sidecar)
        assert seen["rows"] == ["rsp_raw", "event"], seen["rows"]
        assert RAW_ROWS["rsp_raw"]["text"] in out.getvalue()
        print("   OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        study_profile.activate(study_profile.BUILT_IN)

    print("\nALL RSP RAW ROW TESTS PASSED.")


if __name__ == "__main__":
    main()
