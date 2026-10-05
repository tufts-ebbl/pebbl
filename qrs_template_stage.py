#!/usr/bin/env python3
"""
Stage A: build an ECG QRS template and full-run corrected peaks, replacing
step1_qrs_template.ipynb -- no Jupyter required.

Works per SUBJECT, not per run, matching step1_qrs_template.ipynb: an RA
builds ONE template from a short window in ONE run, and that same template
is applied (via wavelet filter + cross-correlation) to every run that
exists for that subject -- no need to build a separate template per run.

This script handles argument parsing and data loading; the actual
interactive session and algorithm are in qrs_template.run_stage_a(), shared
with physio_review.py (the unified script that runs Stage A + Stage B back
to back). Standalone, this remains useful for troubleshooting Stage A in
isolation (e.g., retrying a messy template window) without also launching
the full Stage B review.

This script does not import from or modify physioProcess or any existing
physioCorrection notebook.

Usage:
    annotate_env\\Scripts\\python.exe qrs_template_stage.py --synthetic --initials hlu
    annotate_env\\Scripts\\python.exe qrs_template_stage.py --box-path <path> --subject 001 --initials hlu --template-start 20
    annotate_env\\Scripts\\python.exe qrs_template_stage.py --input-run1 <path1> --input-run2 <path2> --initials hlu
"""

import argparse
import os

from physio_io import (
    generate_synthetic_demo_two_runs,
    load_physio_tsv,
    read_sidecar,
    resolve_subject_run_paths,
    sidecar_path_for,
)
from physio_review import normalize_initials
from qrs_template import run_stage_a


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run1", type=str, default=None, help="Path to run 1's *_physio.tsv.gz.")
    parser.add_argument("--input-run2", type=str, default=None, help="Path to run 2's *_physio.tsv.gz.")
    parser.add_argument("--box-path", type=str, default=None,
                         help="Box derivatives folder. Used with --subject to find both runs' files. "
                              "Prompted for if omitted (unless --input-run1/2 given).")
    parser.add_argument("--subject", type=str, default=None,
                         help="Subject ID, e.g. '001' or 'sub-001'. Prompted for if omitted.")
    parser.add_argument("--synthetic", action="store_true",
                         help="Use fabricated demo data (split into two halves standing in for run 1/run 2) "
                              "instead of real files.")
    parser.add_argument("--duration", type=float, default=90.0,
                         help="Duration in seconds PER synthetic run (default: 90).")
    parser.add_argument("--initials", type=str, default=None,
                        help="RA initials: 2-4 letters, lower-cased (e.g. hlu). If omitted, you'll be prompted.")
    parser.add_argument("--out-dir", type=str, default=None,
                         help="Where to save outputs. Defaults to alongside each run's own input file "
                              "(real mode) or ./demo_output (synthetic mode).")
    parser.add_argument("--template-run", choices=["1", "2"], default="1",
                         help="Which run's window to build the template from (default: 1). Falls back to "
                              "whichever run is actually available if this one isn't found.")
    parser.add_argument("--template-start", type=float, default=0.0,
                         help="Start time (seconds) of the window used to build the template (default: 0). "
                              "If that window is too messy, close without correcting anything and re-run "
                              "with this shifted forward, e.g. 20, 40, ...")
    parser.add_argument("--template-window", type=float, default=20.0,
                         help="Length (seconds) of the template-building window (default: 20).")
    parser.add_argument("--no-ppg-guide", action="store_true",
                         help="Don't show the read-only PPG guide row under the ECG (it's off by default anyway).")
    parser.add_argument("--ecg-ppg-guide", action="store_true",
                         help="Also show the PPG guide row under the ECG (Steps 1-3; off by default since 2026-10-03, HLU): each run's pulse sits a different delay after its R peaks, so use it for counting beats only, never to place an R peak.")
    return parser.parse_args()


def load_run_data(args):
    """Returns (run_dfs, run_file_stems, run_out_dirs, sfreq, run_sidecars); run_sidecars is None for synthetic data."""
    run_dfs, run_file_stems, run_out_dirs, run_sidecars = {}, {}, {}, {}

    if args.synthetic:
        print(f"Generating {args.duration:.0f}s x 2 synthetic demo data (two halves standing in for run 1/run 2)...")
        run_dfs, sfreq = generate_synthetic_demo_two_runs(duration_sec_each=args.duration)
        demo_out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_output")
        for run in run_dfs:
            run_file_stems[run] = f"synthetic_demo_run{run}"
            run_out_dirs[run] = demo_out_dir
        return run_dfs, run_file_stems, run_out_dirs, sfreq, None

    run_paths = resolve_subject_run_paths(args.input_run1, args.input_run2, args.box_path, args.subject)
    sfreq = None
    for run, path in run_paths.items():
        if not path:
            continue
        print(f"Loading run {run}: {path}...")
        df, this_sfreq = load_physio_tsv(path)
        if sfreq is None:
            sfreq = this_sfreq
        run_dfs[run] = df
        run_file_stems[run] = os.path.basename(path).replace("_physio.tsv.gz", "")
        run_out_dirs[run] = args.out_dir or os.path.dirname(os.path.abspath(path))
        run_sidecars[run] = read_sidecar(sidecar_path_for(path))
    return run_dfs, run_file_stems, run_out_dirs, sfreq, run_sidecars


def main():
    args = parse_args()

    run_dfs, run_file_stems, run_out_dirs, sfreq, run_sidecars = load_run_data(args)

    # Lower-cased, 2-4 letters: the same rule as physio_review.py and the GUI (audit H6).
    initials = normalize_initials(args.initials or input("Enter your initials: "))

    template_run = args.template_run if args.template_run in run_dfs else next(iter(run_dfs))
    if template_run != args.template_run:
        print(f"Requested --template-run {args.template_run} isn't available; "
              f"using run {template_run} instead (found runs: {list(run_dfs.keys())}).")

    # Real files carry a sidecar; synthetic data has none (the PPG guide is
    # hidden on old-pipeline files, HLU's decision G2).
    run_stage_a(run_dfs, run_file_stems, run_out_dirs, sfreq, initials,
                template_run=template_run, template_start=args.template_start, template_window=args.template_window,
                ppg_guide=not args.no_ppg_guide, sidecars=run_sidecars, ecg_ppg_guide=args.ecg_ppg_guide)


if __name__ == "__main__":
    main()
