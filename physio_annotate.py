#!/usr/bin/env python3
"""
Multisignal annotation prototype.

Loads a multi-channel physio recording (either a real *_physio.tsv.gz from
physioProcess/physioBatch.py, or a synthetic demo file for testing) into a
single MNE Raw object covering every configured channel present (see
channel_config.py). Launches MNE's interactive annotation browser so an RA
can mark:
    - ECG/PPG: point annotations (peaks) -- pre-seeded from the
      automated peak-detection columns already in the file -- plus their
      bad stretches (labels bad_ecg / bad_ppg; bad_ppg starts from the
      machine's dropouts). PPG is reviewed in a session of its own
      (--channels ppg).
    - RSP, EDA, SBP, DBP, EMG (corrugator/zygomatic), finger temperature:
      segment annotations (bad/artifact spans); on new-pipeline files
      RSP/EDA/SBP/DBP start from the machine's own bad stretches.
    - Read-only guide rows (never marked or saved): PPG under SBP/DBP,
      EDA under RSP, RSP under EDA (--no-ppg-guide hides them); PPG under
      ECG only with --ecg-ppg-guide (opt-in since 2026-10-03, HLU).

On closing the browser window, the session-summary dialog lets the RA save,
discard, or go back; a save writes <file_stem>_annotations_<initials>.json,
so more than one RA's independent review of the same file can coexist
without overwriting. Troubleshooting entry point only: day-to-day use is
physio_review.py (logging, the Box copy lifecycle, and the tracker reminder).

This script does not import from or modify any existing physioProcess or
physioCorrection script -- it only reads the stable physio.tsv.gz/json
file format those scripts already produce.

Usage:
    annotate_env\\Scripts\\python.exe physio_annotate.py --synthetic
    annotate_env\\Scripts\\python.exe physio_annotate.py --input <path_to_physio.tsv.gz> --initials xyz
"""

import argparse
import os

from annotation_io import find_template_json, resolve_ecg_peaks, run_stage_b
from channel_config import CHANNELS, empty_channels, present_channels, select_channels
from physio_io import generate_synthetic_demo, load_physio_tsv, read_sidecar, resolve_real_input_path, sidecar_path_for
from physio_review import normalize_initials


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, default=None,
                         help="Path to a real *_physio.tsv.gz file. Alternative to --box-path/--subject/--run.")
    parser.add_argument("--box-path", type=str, default=None,
                         help="Box derivatives folder (e.g. ...\\physioProcessing\\derivatives). "
                              "Used with --subject/--run to build the file path. Prompted for if omitted.")
    parser.add_argument("--subject", type=str, default=None,
                         help="Subject ID, e.g. '001' or 'sub-001'. Prompted for if omitted.")
    parser.add_argument("--run", type=str, default=None,
                         help="Run number, '1' or '2'. Prompted for if omitted.")
    parser.add_argument("--synthetic", action="store_true",
                         help="Use fabricated demo data instead of a real file.")
    parser.add_argument("--duration", type=float, default=60.0,
                         help="Duration in seconds for synthetic demo data (default: 60).")
    parser.add_argument("--initials", type=str, default=None,
                         help="RA initials: 2-4 letters, lower-cased (e.g. hlu). If omitted, you'll be prompted.")
    parser.add_argument("--out-dir", type=str, default=None,
                         help="Where to save the annotations JSON. Defaults to alongside the input file "
                              "(real mode) or ./demo_output (synthetic mode).")
    parser.add_argument("--channels", type=str, default=None,
                         help="Comma-separated subset of channels to load, e.g. 'ecg,rsp'. "
                              f"Options: {','.join(CHANNELS.keys())}. Defaults to all of them except ppg, "
                              f"which is reviewed on its own (--channels ppg).")
    parser.add_argument("--ecg-source", choices=["auto", "template", "batch"], default="auto",
                         help="Where ECG seed peaks come from. 'auto' (default): use a QRS template "
                              "(qrs_template_stage.py output) if one exists for this file, else fall back "
                              "to the batch pipeline's peaks. 'template': require the template JSON to "
                              "exist. 'batch': always use the batch pipeline's peaks.")
    parser.add_argument("--no-ppg-guide", action="store_true",
                         help="Don't show the read-only guide rows (PPG under SBP/DBP, EDA under RSP, RSP under "
                              "EDA; also the PPG under ECG).")
    parser.add_argument("--ecg-ppg-guide", action="store_true",
                         help="Also show the PPG guide row under the ECG (Steps 1-3; off by default since 2026-10-03, HLU): each run's pulse sits a different delay after its R peaks, so use it for counting beats only, never to place an R peak.")
    parser.add_argument("--allow-old-ppg", action="store_true",
                         help="Lab staff only: allow PPG review on a file processed before the PPG timing fix.")
    parser.add_argument("--template-json", type=str, default=None,
                         help="Explicit path to a Stage A *_ecg_corrected_qrs_<initials>.json. Defaults to "
                              "auto-finding one in --out-dir; if more than one RA has built a template for "
                              "this file, your own --initials is preferred, otherwise you'll be asked to "
                              "pick one explicitly with this flag.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.synthetic:
        print(f"Generating {args.duration:.0f}s of synthetic demo data "
              f"(ECG/RSP/PPG/EDA simulated; EMG channels are flatline placeholders)...")
        df, sfreq = generate_synthetic_demo(duration_sec=args.duration)
        source_file = "<synthetic demo data>"
        file_stem = "synthetic_demo"
        out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_output")
        sidecar = None
        tsv_path = None
    else:
        input_path = resolve_real_input_path(args.input, args.box_path, args.subject, args.run)
        print(f"Loading {input_path}...")
        df, sfreq = load_physio_tsv(input_path)
        source_file = os.path.abspath(input_path)
        basename = os.path.basename(input_path)
        file_stem = basename.replace("_physio.tsv.gz", "")
        out_dir = args.out_dir or os.path.dirname(os.path.abspath(input_path))
        sidecar = read_sidecar(sidecar_path_for(input_path))
        tsv_path = input_path

    # Lower-cased, 2-4 letters: the same rule as physio_review.py and the GUI (audit H6).
    initials = normalize_initials(args.initials or input("Enter your initials: "))

    template_json_path = find_template_json(out_dir, file_stem, args.template_json, initials)
    df, ecg_source_label = resolve_ecg_peaks(df, args.ecg_source, template_json_path)
    print(f"ECG seed source: {ecg_source_label}")

    # Same rules as physio_review.py: PPG only on its own, left out of the default.
    active_channels = select_channels(args.channels.split(",") if args.channels else None)
    active_channels = present_channels(active_channels, df.columns, explicit=bool(args.channels),
                                       empty=empty_channels(df, active_channels))

    run_stage_b(df, sfreq, active_channels, initials, out_dir, file_stem, source_file, sidecar=sidecar,
                ppg_guide=not args.no_ppg_guide, tsv_path=tsv_path, allow_old_ppg=args.allow_old_ppg,
                seed_label=ecg_source_label, ecg_ppg_guide=args.ecg_ppg_guide)


if __name__ == "__main__":
    main()
