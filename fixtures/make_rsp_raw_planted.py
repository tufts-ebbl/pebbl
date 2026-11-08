"""
Makes fixtures/rsp_raw_planted_130-180s.npz (test_rsp_raw_row.py's data) from physiopipe's synthetic breathing
file, so it can be rebuilt if physiopipe's writer changes. See fixtures/README.md for its origin.

    annotate_env/Scripts/python.exe fixtures/make_rsp_raw_planted.py <..._task-urglab_physio.tsv.gz> <..._physio.json>
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

SOURCE_SHA256 = "914a1151532cbf493dde1f2272982b392589341b2567b2aefe9cb425b50f9605"   # the .tsv.gz
SIDECAR_SHA256 = "4169fe53b0369d10b0ff228753ab4826d90cf352499a8d05902ec7c23d7514f7"  # its .json
SFREQ = 1000
START_S, END_S = 130.0, 180.0  # tsv seconds
# The planted events inside the cut, in the source file's tsv seconds (physiopipe's fixture_plants.json).
EVENTS = (("belt held at the -10 V limit, then a jump past zero (machine: rail + swing)", 140.9, 143.0),
          ("single 3-V belt step, slow drift back with breathing on it (machine: swing)", 175.0, 175.1))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rsp_raw_planted_130-180s.npz")


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main(tsv_path, json_path):
    for path, want in ((tsv_path, SOURCE_SHA256), (json_path, SIDECAR_SHA256)):
        if sha256(path) != want:
            raise SystemExit(f"{path} isn't the synthetic breathing file this fixture is made from.")
    with open(json_path, encoding="utf-8") as f:
        sidecar = json.load(f)
    df = pd.read_csv(tsv_path, sep="\t", header=None, names=sidecar["Columns"])
    a, b = int(START_S * SFREQ), int(END_S * SFREQ)
    # The machine's RSP stretches, cut to the window and counted from its start.
    by_reason = {}
    for reason, spans in sidecar["MachineQC"]["Channels"]["rsp"]["SegmentsByReason"].items():
        by_reason[reason] = [[max(s, a) - a, min(e, b) - a] for s, e in spans if s < b and e > a]
    np.savez_compressed(OUT, rsp_raw=df["rsp_raw"].to_numpy(float)[a:b], rsp=df["rsp"].to_numpy(float)[a:b],
                        event=df["event"].to_numpy(int)[a:b], sfreq=SFREQ, start_s=START_S,
                        source_sha256=SOURCE_SHA256, by_reason=json.dumps(by_reason),
                        event_names=np.array([e[0] for e in EVENTS]), event_spans_s=np.array([e[1:] for e in EVENTS]))
    print(f"wrote {OUT} ({b - a} samples, {os.path.getsize(OUT)} bytes); machine reasons in the cut: {by_reason}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
