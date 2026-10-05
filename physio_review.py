#!/usr/bin/env python3
"""
Unified CLI: one command that ensures an ECG QRS template exists (Stage A),
then launches the full multi-signal review (Stage B) for the run you're
working on -- so an RA doesn't need to remember two separate scripts.

Behavior:
    - With no --stage given (the default): ensures YOUR OWN QRS template
      exists, building one via Stage A only if needed, then launches
      Stage B. Every RA builds and uses their own template, in full,
      regardless of who's reviewed this subject first -- one RA's template
      is never borrowed to seed another RA's review, even if it's the only
      one that exists (a deliberate reliability choice: sharing a seed
      would bias independent reviews toward agreement, or a shared blind
      spot, before anyone's even started). If YOUR OWN template already
      exists for this subject (any run), Stage A is SKIPPED -- e.g. after
      you've already built one while reviewing run 1, reviewing run 2
      won't make you build a second one. --ecg-source batch skips Stage A
      entirely and always uses the batch pipeline's peaks.
    - --stage 1: run ONLY Stage A (build/apply the template), then stop.
    - --stage 2: run ONLY Stage B, using your own existing template for
      ECG -- erroring clearly if you don't have one yet, never falling
      back to batch peaks silently and never borrowing another RA's
      template -- but NEVER building one interactively; that's what
      --stage 1 (or the combined flow above) is for.
    - --stage 3: reconcile two RAs' (or an RA's vs. a gold-standard
      reviewer's) already-saved annotation JSONs for the same run --
      requires --compare-initials A,B. See reconcile.py for the full design.

This is a thin orchestration layer over qrs_template.run_stage_a() and
annotation_io.run_stage_b() -- the same functions the standalone
qrs_template_stage.py and physio_annotate.py scripts use, so nothing here
is duplicated logic. Those two scripts remain available for troubleshooting
a single stage in isolation (e.g., retrying a messy template window without
also launching the full review).

This script does not import from or modify physioProcess or any existing
physioCorrection notebook.

Usage:
    annotate_env\\Scripts\\python.exe physio_review.py                                                          # no args -> pops up a simple form
    annotate_env\\Scripts\\python.exe physio_review.py --synthetic --initials hlu --run 1
    annotate_env\\Scripts\\python.exe physio_review.py --box-path <path> --local-path <local> --subject 001 --run 1 --initials hlu
    annotate_env\\Scripts\\python.exe physio_review.py --box-path <path> --local-path <local> --subject 001 --run 2 --initials hlu   # reuses run 1's template automatically if one already exists

Any real (non-synthetic) run also copies this subject's files from Box to
--local-path, works from there, pushes back only what's new/changed, and
removes the local copy once confirmed -- see box_sync.py. --local-path is
prompted for if omitted, same as --box-path; --no-local-copy skips all of
this and works directly against --box-path instead.
"""

import argparse
import re
import os
import sys
import textwrap
import traceback
from datetime import datetime

from annotation_io import SessionResult, find_template_json, load_saved_annotations, resolve_ecg_peaks, run_stage_b
from box_sync import (
    cleanup_local_copy,
    copy_subject_tree_to_local,
    find_changed_files,
    find_unpushed_local_files,
    push_changed_files_to_box,
)
from channel_config import CHANNELS, empty_channels, no_data_message, present_channels, select_channels
from physio_io import (
    compute_channel_value_ranges,
    format_channel_value_ranges,
    generate_synthetic_demo_two_runs,
    load_physio_tsv,
    read_sidecar,
    resolve_box_path_and_subject,
    resolve_subject_run_paths,
    sidecar_path_for,
)
from processing_log import append_processing_log_entry
from qrs_template import run_stage_a
from reconcile import run_stage_c
from session_log import SessionLog, footer_lines as session_log_footer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=(
        "Review and correct physiological signals. Run with no options for a simple form. "
        "Step 1 builds your ECG template, Step 2 reviews a channel, Step 3 reconciles two reviewers."))
    parser.add_argument("--input-run1", type=str, default=None, help="Path to run 1's *_physio.tsv.gz.")
    parser.add_argument("--input-run2", type=str, default=None, help="Path to run 2's *_physio.tsv.gz.")
    parser.add_argument("--box-path", type=str, default=None,
                         help="Box derivatives folder. Used with --subject to find both runs' files. "
                              "Prompted for if omitted (unless --input-run1/2 given).")
    parser.add_argument("--subject", type=str, default=None,
                         help="Subject ID, e.g. '001' or 'sub-001'. Prompted for if omitted.")
    parser.add_argument("--local-path", type=str, default=None,
                         help="Local folder to work from: this subject's files are copied here from Box, "
                              "worked on, pushed back to Box, confirmed, then removed automatically -- "
                              "matching the existing notebooks' own copy-locally-then-push workflow. "
                              "Prompted for if omitted (same as --box-path). Ignored with --synthetic, "
                              "--input-run1/2, or --no-local-copy.")
    parser.add_argument("--no-local-copy", action="store_true",
                         help="Skip the local-copy lifecycle entirely and work directly against "
                              "--box-path (e.g. for troubleshooting, or if --box-path already points "
                              "at a local folder you're managing yourself).")
    parser.add_argument("--keep-local", action="store_true",
                         help="Don't delete the local working copy after a successful push back to Box "
                              "(for troubleshooting). The copy is always preserved if any file's push "
                              "can't be confirmed, regardless of this flag.")
    parser.add_argument("--run", type=str, default=None,
                         help="Which run to review THIS session in Stage B. Prompted for if omitted and "
                              "more than one run is available; auto-selected if only one is.")
    parser.add_argument("--practice", action="store_true",
                         help="Certification practice (the form's 'Practice for certification' box): --box-path is "
                              "the practice folder. After each saved Step 2 review, PEBBL scores it against the "
                              "answer key there (practice_key.pebbl), shows the report, and can show your marks "
                              "next to the key. Not for Step 3.")
    parser.add_argument("--synthetic", action="store_true",
                         help="Use fabricated demo data (split into two halves standing in for run 1/run 2) "
                              "instead of real files.")
    parser.add_argument("--duration", type=float, default=90.0,
                         help="Duration in seconds PER synthetic run (default: 90).")
    parser.add_argument("--initials", type=str, default=None, help="RA initials. If omitted, you'll be prompted.")
    parser.add_argument("--out-dir", type=str, default=None,
                         help="Where to save outputs. Defaults to alongside each run's own input file "
                              "(real mode) or ./demo_output (synthetic mode).")
    parser.add_argument("--channels", type=str, default=None,
                         help="Comma-separated subset of channels for Stage B, e.g. 'ecg,eda'. "
                              f"Options: {','.join(CHANNELS.keys())}. Defaults to all of them except ppg, "
                              f"which is reviewed on its own (--channels ppg).")
    parser.add_argument("--stage", choices=["1", "2", "3"], default=None,
                         help="'1': run ONLY Stage A (build/apply the ECG QRS template), then stop -- no "
                              "Stage B. '2': run ONLY Stage B for --channels, using YOUR OWN existing "
                              "template for ECG -- erroring clearly if you don't have one yet, never "
                              "another RA's and never a silent fallback to batch peaks -- but NEVER "
                              "building one interactively. '3': reconcile two RAs' saved annotation JSONs "
                              "for --run (needs --compare-initials). Omit for the default combined "
                              "behavior: ensure your own template exists (building one if needed) then "
                              "review.")
    parser.add_argument("--compare-initials", type=str, default=None,
                         help="Comma-separated pair of initials to reconcile, e.g. 'hlu,xyz' -- required for "
                              "--stage 3. Order doesn't matter; both must already have a saved "
                              "_annotations_<initials>.json for --run.")
    parser.add_argument("--ecg-source", choices=["auto", "batch"], default="auto",
                         help="'auto' (default): use your own existing QRS template (per --initials) -- "
                              "with --stage 2, errors clearly if you don't have one yet; without --stage, "
                              "builds one now (Stage A) instead. Never borrows another RA's template, even "
                              "if theirs is the only one that exists -- each RA builds their own, full "
                              "stop. 'batch': skip the template requirement entirely and always use the "
                              "batch pipeline's own peak detection, regardless of --stage.")
    parser.add_argument("--no-ppg-guide", action="store_true",
                         help="Don't show the read-only guide rows (shown by default in Steps 2-3): PPG under "
                              "SBP/DBP, EDA under RSP, RSP under EDA. Also hides the PPG under ECG.")
    parser.add_argument("--ecg-ppg-guide", action="store_true",
                         help="Also show the PPG guide row under the ECG (Steps 1-3; off by default since 2026-10-03, HLU): each run's pulse sits a different delay after its R peaks, so use it for counting beats only, never to place an R peak.")
    parser.add_argument("--allow-old-ppg", action="store_true",
                         help="Lab staff only: allow PPG review on a file processed before the PPG timing fix.")
    parser.add_argument("--force-template", action="store_true",
                         help="Build a new QRS template (Stage A) even if one already exists for this subject.")
    parser.add_argument("--template-run", choices=["1", "2"], default="1",
                         help="Which run's window to build the template from, if Stage A runs (default: 1).")
    parser.add_argument("--template-start", type=float, default=0.0,
                         help="Start time (seconds) of the template-building window, if Stage A runs.")
    parser.add_argument("--template-window", type=float, default=20.0,
                         help="Length (seconds) of the template-building window, if Stage A runs (default: 20).")
    return parser.parse_args(argv)


def _quote_arg(value):
    """
    Always double-quotes a value for copy/paste safety across whichever
    shell an RA pastes into (cmd.exe, PowerShell, or Git Bash/MINGW64 --
    all preserve a double-quoted value's contents literally, including
    backslashes in a Windows path). Previously only quoted when a value
    had whitespace, on the assumption a bare backslash-containing path was
    otherwise safe -- true for cmd.exe/PowerShell, but NOT for bash: it
    treats an unquoted backslash before an ordinary letter as an escape
    sequence and silently drops it (\\U -> U), which is exactly what
    happened when an RA on Git Bash copy/pasted the echoed command and
    every backslash in --box-path/--local-path vanished (confirmed live --
    "No run 1 or run 2 physio.tsv.gz found" against a path that had lost
    all its path separators). See HANDOFF.md.
    """
    return '"' + value.replace('"', '\\"') + '"'


def _print_equivalent_command(argv):
    """
    Echoes the CLI command equivalent to what the GUI form just submitted,
    so an RA can copy/paste it (editing --run, --channels, etc. as needed)
    to repeat a similar session -- e.g. run 1 via the GUI, then run 2 by
    pasting this with --run changed -- without reopening the form each
    time. Uses sys.executable/sys.argv[0] (not a hardcoded path) so it
    matches exactly however this session was actually launched.
    """
    command = " ".join([_quote_arg(sys.executable), _quote_arg(sys.argv[0])] + [_quote_arg(a) for a in argv])
    print("\n" + "-" * 70)
    print("Equivalent command (copy/paste and edit --run/--channels/etc. to repeat this from the CLI):")
    print(f"  {command}")
    print("-" * 70 + "\n")


def parse_args_or_show_gui():
    """
    If invoked with literally no command-line arguments, shows a simple
    form (simple_gui.py) to collect box-path/subject/run/initials (or opt
    into synthetic demo data) instead of falling back to sequential console
    input() prompts. Any argument at all on the command line skips this and
    uses the normal CLI/prompt behavior -- this is an on-ramp for the
    simplest case, not a replacement for the CLI.
    """
    if len(sys.argv) > 1:
        return parse_args()

    from simple_gui import collect_inputs_via_gui
    values = collect_inputs_via_gui()
    if values is None:
        raise SystemExit("Canceled.")

    if values["synthetic"]:
        if values["stage"] == "3":
            argv = ["--synthetic", "--stage", "3",
                    "--compare-initials", values["compare_initials"], "--channels", values["channel"]]
            if values.get("initials"):
                argv += ["--initials", values["initials"]]
        else:
            argv = ["--synthetic", "--initials", values["initials"], "--stage", values["stage"]]
            if values["stage"] == "2":
                # The demo seeds ECG from the automatic detections: without this,
                # a first-time user's demo failed for lack of a Step 1 template
                # (audit finding H3).
                argv += ["--channels", values["channel"], "--ecg-source", "batch"]
        argv += _guide_flags(values)
    else:
        argv = ["--box-path", values["box_path"], "--subject", values["subject"], "--stage", values["stage"]]
        if values.get("no_local_copy"):
            argv += ["--no-local-copy"]
        elif values["local_path"]:
            argv += ["--local-path", values["local_path"]]
        if values.get("keep_local"):
            argv += ["--keep-local"]
        if values.get("practice"):
            argv += ["--practice"]

        if values["stage"] == "2":
            argv += ["--initials", values["initials"], "--run", values["run"], "--channels", values["channel"],
                     "--ecg-source", values["ecg_source"]]
        elif values["stage"] == "3":
            argv += ["--run", values["run"], "--channels", values["channel"],
                     "--compare-initials", values["compare_initials"]]
            if values.get("initials"):
                argv += ["--initials", values["initials"]]
        else:
            argv += ["--initials", values["initials"], "--template-start", str(values["template_start"]),
                     "--template-run", values["template_run"], "--template-window", str(values["template_window"])]
        argv += _guide_flags(values)

    _print_equivalent_command(argv)
    return parse_args(argv)


def _guide_flags(values):
    """
    The form's one guide box as CLI flags: with ECG (Step 1, or ECG chosen)
    ticked means --ecg-ppg-guide (the PPG under ECG is opt-in since
    2026-10-03, HLU); otherwise unticked means --no-ppg-guide.
    """
    on = values.get("ppg_guide")
    if values.get("stage") == "1" or values.get("channel") == "ecg":
        return ["--ecg-ppg-guide"] if on else []
    return ["--no-ppg-guide"] if on is False else []


def load_run_data(args):
    """
    Returns (run_dfs, run_file_stems, run_out_dirs, sfreq, run_sources) -- the
    first four the same shape as qrs_template_stage.py's loader. run_sources
    is {run: {"sidecar": dict, "tsv_path": str}} for real files and
    {run: None} for synthetic data.
    """
    run_dfs, run_file_stems, run_out_dirs, run_sources = {}, {}, {}, {}

    if args.synthetic:
        print(f"Generating {args.duration:.0f}s x 2 synthetic demo data (two halves standing in for run 1/run 2)...")
        run_dfs, sfreq = generate_synthetic_demo_two_runs(duration_sec_each=args.duration)
        demo_out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_output")
        for run in run_dfs:
            run_file_stems[run] = f"synthetic_demo_run{run}"
            run_out_dirs[run] = demo_out_dir
            run_sources[run] = None
        return run_dfs, run_file_stems, run_out_dirs, sfreq, run_sources

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
        run_sources[run] = {"sidecar": read_sidecar(sidecar_path_for(path)), "tsv_path": path}
    return run_dfs, run_file_stems, run_out_dirs, sfreq, run_sources


def _check_modes_match(outputs_a, outputs_b, active_channels, label_a, label_b):
    """
    Step 3 compares like with like: both reviews of a channel must be in the
    channel's CURRENT mode. RSP was a point (breath peak) channel until
    2026-09-25 and is now a segment (bad stretch) channel, so an older RSP
    review can't be reconciled against a newer one (sidecar plan Phase 1).
    """
    problems = []
    for ch_key, cfg in active_channels.items():
        modes = {label_a: outputs_a.get(ch_key, {}).get("mode"), label_b: outputs_b.get(ch_key, {}).get("mode")}
        wrong = [who for who, mode in modes.items() if mode != cfg["annotation_mode"]]
        if wrong:
            problems.append(f"{cfg['label']}: the review(s) by {', '.join(wrong)} marked "
                            f"{'/'.join(sorted({modes[w] + 's' for w in wrong if modes[w]}))}, but {cfg['label']} is "
                            f"now reviewed by marking "
                            f"{'bad stretches' if cfg['annotation_mode'] == 'segment' else 'peaks'}.")
    if problems:
        raise SystemExit("Step 3 can't compare reviews made in different ways:\n  " + "\n  ".join(problems) +
                         "\nThe reviewer(s) named need to review this channel again in Step 2 first. Other channels "
                         "can be reconciled now with --channels (or by picking them in the form).")


def _session_outcome(result):
    """
    (saved, finished, discarded, comment) from run_stage_b()/run_stage_c()'s
    SessionResult. Session comments now come from the session-summary
    dialog, not a terminal input() prompt: an RA who closed the terminal at
    that prompt stranded their saved work before the push back to Box (audit
    finding C2). A non-SessionResult (e.g. a test stub) counts as a plain save.
    """
    if isinstance(result, SessionResult):
        return result.saved, result.finished, result.discarded, result.comment
    return True, False, False, ""


INITIALS_PATTERN = re.compile(r"^[a-z]{2,4}$")


def normalize_initials(text):
    """
    Lower-cases and checks RA initials: 2-4 letters, nothing else (audit
    finding H6). Initials name the output files, and Windows/macOS paths
    ignore case, so "CC" and "cc" (both present in the lab's roster) would
    silently share one file while recording different initials inside it;
    free text also let a subject ID typed into the wrong box ("sub") become
    someone's initials.
    """
    initials = (text or "").strip().lower()
    if not INITIALS_PATTERN.match(initials):
        raise SystemExit(f"'{(text or '').strip()}' isn't valid initials: use 2-4 letters only (e.g. hlu).")
    return initials


def _has_saved_ecg(out_dir, file_stem, initials):
    saved = load_saved_annotations(out_dir, file_stem, initials)
    return bool(saved) and saved[0].get("ecg", {}).get("mode") == "point"


def _template_comment(results):
    """The optional comment typed in Step 1's template-preview dialog (same for every run)."""
    try:
        return next(iter(results.values())).get("comment", "") or ""
    except (StopIteration, AttributeError, TypeError):
        return ""


def _status_text(finished):
    return "finished" if finished else "not finished"


def _attach_session_log(info, targets, initials, step_text):
    """Starts writing this session to the RA's <file_stem>_annotations_<initials>.log in each target run folder."""
    log = info.get("session_log")
    if log is not None:
        subject = "synthetic demo data" if info.get("synthetic") else f"sub-{str(info.get('subject')).replace('sub-', '')}"
        log.attach(targets, initials, header_lines=[f"{subject}; {step_text}"])


def _run_session(args, real_box_path, info):
    """
    Everything that actually reviews/reconciles data, given fully-resolved
    args (box_path may point at a LOCAL copy by the time this runs -- see
    main() -- so every data read/write below goes through args.box_path,
    while every processing_log.csv entry uses real_box_path, the true Box
    location, since the log always lives there directly, never locally).

    `info` is a dict main() owns; this fills in what the session did (stage,
    run, channels, saved/finished/discarded) for the closing tracker reminder,
    which main() prints even if this function raises.
    """
    run_dfs, run_file_stems, run_out_dirs, sfreq, run_sources = load_run_data(args)

    if args.stage == "3":
        if args.practice:
            raise SystemExit("Step 3 isn't part of certification practice. Choose Step 1 or Step 2.")
        # Reconciliation compares two ALREADY-saved sessions, so it skips the
        # "claimed" log entry. The reconciler's OWN initials are still
        # required and recorded (audit finding M2: the reconciled file used
        # to record initials: null, so nobody could tell who reconciled).
        if not args.compare_initials or "," not in args.compare_initials:
            raise SystemExit("--stage 3 requires --compare-initials A,B (e.g. --compare-initials hlu,xyz).")
        label_a, label_b = (normalize_initials(s) for s in args.compare_initials.split(",", 1))
        if label_a == label_b:
            raise SystemExit(f"The two reviewers to compare must be different people (both were '{label_a}').")
        reconciler = normalize_initials(args.initials or input("Enter YOUR initials (the reconciler): "))

        available_runs = list(run_dfs.keys())
        target_run = args.run
        if target_run and target_run not in run_dfs:
            raise SystemExit(f"Run {target_run} isn't available (found: {available_runs}).")
        if not target_run:
            if len(available_runs) == 1:
                target_run = available_runs[0]
                print(f"Only run {target_run} is available; reconciling that one.")
            else:
                target_run = input(f"Which run do you want to reconcile? ({'/'.join(available_runs)}): ").strip()
                if target_run not in run_dfs:
                    raise SystemExit(f"Run {target_run} isn't available (found: {available_runs}).")

        target_df = run_dfs[target_run]
        target_file_stem = run_file_stems[target_run]
        target_out_dir = run_out_dirs[target_run]
        _attach_session_log(info, [(target_out_dir, target_file_stem)], reconciler,
                            f"Step 3 (reconcile {label_a} vs {label_b}), run {target_run}")

        # With this file's rate and length: an ill-fitting review stops plainly (2026-09-26 safety fix).
        saved_a = load_saved_annotations(target_out_dir, target_file_stem, label_a, sfreq=sfreq,
                                         n_samples=len(target_df))
        saved_b = load_saved_annotations(target_out_dir, target_file_stem, label_b, sfreq=sfreq,
                                         n_samples=len(target_df))
        if saved_a is None:
            raise SystemExit(f"No saved annotations found for '{label_a}' on run {target_run} "
                              f"(looked for {target_file_stem}_annotations_{label_a}.json in {target_out_dir}).")
        if saved_b is None:
            raise SystemExit(f"No saved annotations found for '{label_b}' on run {target_run} "
                              f"(looked for {target_file_stem}_annotations_{label_b}.json in {target_out_dir}).")
        outputs_a, _ = saved_a
        outputs_b, _ = saved_b

        shared_channels = set(outputs_a.keys()) & set(outputs_b.keys())
        if args.channels:
            requested = [c.strip() for c in args.channels.split(",")]
            select_channels(requested)  # unknown names; PPG only on its own
            # A channel with no data in this file is refused with that reason,
            # not as "not reviewed by both": its Step 2 is refused too, so
            # neither reviewer can have it (run-8 review, 2026-09-26).
            blank = empty_channels(target_df, requested)
            if blank:
                raise SystemExit(no_data_message(blank))
            not_shared = [c for c in requested if c not in shared_channels]
            if not_shared:
                raise SystemExit(f"Channel(s) {not_shared} weren't reviewed by both '{label_a}' and "
                                  f"'{label_b}' on run {target_run} -- nothing to reconcile there.")
            active_channels = {k: v for k, v in CHANNELS.items() if k in requested}
        else:
            if not shared_channels:
                raise SystemExit(f"'{label_a}' and '{label_b}' have no channels in common on run "
                                  f"{target_run} -- nothing to reconcile.")
            active_channels = {k: v for k, v in CHANNELS.items() if k in shared_channels}
            # Without --channels, leave out any channel one of the reviews made
            # in an older mode (e.g. point-mode RSP) instead of refusing the
            # whole session; naming it explicitly still gets the explanation
            # (checkpoint-1 review, S3-6).
            mismatched = [k for k, cfg in active_channels.items()
                          if not (outputs_a[k].get("mode") == outputs_b[k].get("mode") == cfg["annotation_mode"])]
            for k in mismatched:
                print(f"NOTE: skipping {CHANNELS[k]['label']}: the two reviews weren't both made the current way "
                      f"(run with --channels {k} to see why).")
                active_channels.pop(k)
            # And any channel whose column is empty in this file (e.g. SBP/DBP
            # on a run with no CareTaker vitals): nothing to reconcile. Named
            # explicitly, run_stage_c() refuses it with the reason.
            for k in empty_channels(target_df, list(active_channels)):
                print(f"NOTE: skipping {CHANNELS[k]['label']}: this file's {CHANNELS[k]['label']} data are empty, "
                      f"so there is nothing to reconcile.")
                active_channels.pop(k)
            if not active_channels:
                raise SystemExit("Nothing to reconcile: every shared channel was reviewed in an older way or has "
                                 "no data in this file.")
            # Then PPG, which is reconciled on its own (ppg-plan §4b) -- only
            # when something else is left (checkpoint-2 review: the order
            # mattered).
            alone = [k for k, cfg in active_channels.items() if cfg.get("review_alone")]
            if alone and len(active_channels) > len(alone):
                for k in alone:
                    print(f"NOTE: skipping {CHANNELS[k]['label']}: it is reconciled on its own "
                          f"(run Step 3 with --channels {k}).")
                    active_channels.pop(k)

        _check_modes_match(outputs_a, outputs_b, active_channels, label_a, label_b)
        source_file = f"<synthetic demo data, run {target_run}>" if args.synthetic else target_file_stem
        info.update(stage="3", run=target_run, channels=list(active_channels.keys()), file_stem=target_file_stem)
        _print_tracker_start(info)
        target_source = run_sources.get(target_run) or {}
        result = run_stage_c(target_df, sfreq, active_channels, target_out_dir, target_file_stem, source_file,
                             outputs_a, outputs_b, label_a, label_b, reconciler=reconciler,
                             sidecar=target_source.get("sidecar"), ppg_guide=not args.no_ppg_guide,
                             tsv_path=target_source.get("tsv_path"), allow_old_ppg=args.allow_old_ppg,
                             ecg_ppg_guide=args.ecg_ppg_guide)
        saved, finished, discarded, comment = _session_outcome(result)
        info.update(saved=saved, finished=finished, discarded=discarded)

        if real_box_path and saved:
            step_text = (f"reconciled {label_a} vs {label_b} ({', '.join(active_channels.keys())}) -- "
                         f"{_status_text(finished)}")
            agreement = getattr(result, "agreement_text", "")
            notes = f"{comment} | {agreement}" if comment and agreement else (comment or agreement)
            append_processing_log_entry(real_box_path, args.subject, target_run, reconciler, step_text, notes)
        return

    initials = normalize_initials(args.initials or input("Enter your initials: "))

    if real_box_path:
        append_processing_log_entry(real_box_path, args.subject, None, initials, "claimed",
                                    snapshot=not args.practice)

    available_runs = list(run_dfs.keys())

    if args.stage == "1":
        # Stage A only: build/apply the ECG template across every available
        # run, then stop -- deliberately no Stage B, no --run/--channels.
        _attach_session_log(info, [(run_out_dirs[r], run_file_stems[r]) for r in run_dfs], initials,
                            f"Step 1 (ECG template), run(s) {', '.join(run_dfs)}")
        template_run = args.template_run if args.template_run in run_dfs else next(iter(run_dfs))
        if template_run != args.template_run:
            print(f"Requested --template-run {args.template_run} isn't available; using run {template_run} "
                  f"instead (found runs: {available_runs}).")
        results = run_stage_a(run_dfs, run_file_stems, run_out_dirs, sfreq, initials,
                               template_run=template_run, template_start=args.template_start,
                               template_window=args.template_window, ppg_guide=not args.no_ppg_guide,
                               ecg_ppg_guide=args.ecg_ppg_guide,
                               sidecars={r: (src or {}).get("sidecar") for r, src in run_sources.items()})
        info.update(stage="1", channels=["ecg"])
        if real_box_path:
            notes = _template_comment(results)
            for run in results:
                append_processing_log_entry(real_box_path, args.subject, run, initials,
                                             "created QRS template", notes, snapshot=not args.practice)
        print("\nStep 1 complete.")
        return

    target_run = args.run
    if target_run and target_run not in run_dfs:
        raise SystemExit(f"Run {target_run} isn't available (found: {available_runs}).")
    if not target_run:
        if len(available_runs) == 1:
            target_run = available_runs[0]
            print(f"Only run {target_run} is available; reviewing that one.")
        else:
            target_run = input(f"Which run do you want to review this session? ({'/'.join(available_runs)}): ").strip()
            if target_run not in run_dfs:
                raise SystemExit(f"Run {target_run} isn't available (found: {available_runs}).")

    target_df = run_dfs[target_run]
    target_file_stem = run_file_stems[target_run]
    target_out_dir = run_out_dirs[target_run]
    ecg_source_label = None
    _attach_session_log(info, [(target_out_dir, target_file_stem)], initials, f"Step 2 (review), run {target_run}")

    # Unknown names are refused; PPG is reviewed only on its own, and the
    # all-channels default leaves it out (ppg-plan §4b).
    active_channels = select_channels(args.channels.split(",") if args.channels else None)
    # Channels the file lacks, or whose column is empty (e.g. SBP/DBP on a
    # run with no CareTaker vitals), are dropped from the default with a note
    # and refused by name.
    active_channels = present_channels(active_channels, target_df.columns, explicit=bool(args.channels),
                                       empty=empty_channels(target_df, active_channels))

    if "ecg" not in active_channels:
        # ECG isn't part of THIS session at all (e.g. --channels rsp) -- the
        # own-template requirement below is specifically about reviewing
        # ECG, so there's nothing to resolve or require here; target_df's
        # own (unresolved) batch ecg_peaks column is simply never read
        # since "ecg" won't be in active_channels either.
        pass
    elif args.ecg_source == "batch":
        ecg_source_label = "automatic detection (ecg_peaks column)"
        print("ECG seed source: automatic detection (no template; ECG peak source 'Batch only').")
    elif not args.force_template and _has_saved_ecg(target_out_dir, target_file_stem, initials):
        # Resuming YOUR OWN saved ECG review: seeding comes from that saved
        # file, not from a template, so none is required (audit finding H2 --
        # this blocked resuming migrated notebook-era work, which has no
        # template under the new naming).
        print("Resuming your saved ECG review for this run -- no template needed.")
    elif args.stage == "2":
        # Stage B only: use YOUR OWN existing template for ECG -- never
        # another RA's, even if theirs is the only one that exists, and
        # NEVER build one here (a standalone review session shouldn't
        # unexpectedly drop into interactive template-building; that's
        # what --stage 1 is for). Every RA does their own Stage A and
        # Stage B for ECG, regardless of who reviewed this subject first --
        # sharing a template would bias independent reviews toward
        # agreement (or a shared blind spot) before anyone's even started,
        # which undermines what Stage 3's reconciliation is meant to
        # measure. Errors clearly (via resolve_ecg_peaks()'s "template"
        # mode) rather than silently falling back to batch peaks if your
        # own template doesn't exist yet.
        template_json_path = find_template_json(target_out_dir, target_file_stem, initials=initials,
                                                  require_own_initials=True)
        target_df, ecg_source_label = resolve_ecg_peaks(target_df, "template", template_json_path)
        print(f"ECG seed source: {ecg_source_label}")
    else:
        # Combined flow: same "your own template only" rule as above, so
        # Stage A runs for every RA who hasn't built their own yet, even
        # when another RA already has one for this subject.
        template_json_path = find_template_json(target_out_dir, target_file_stem, initials=initials,
                                                  require_own_initials=True)
        if os.path.exists(template_json_path) and not args.force_template:
            print(f"Found your own existing ECG template for this subject -- skipping Step 1.\n"
                  f"  {template_json_path}")
        else:
            reason = "requested via --force-template" if args.force_template else "none found yet for your initials"
            print(f"Running Step 1 (build ECG template) first ({reason})...")
            _attach_session_log(info, [(run_out_dirs[r], run_file_stems[r]) for r in run_dfs], initials,
                                f"Step 1 (ECG template) before Step 2 on run {target_run}, run(s) {', '.join(run_dfs)}")
            template_run = args.template_run if args.template_run in run_dfs else next(iter(run_dfs))
            if template_run != args.template_run:
                print(f"Requested --template-run {args.template_run} isn't available; using run {template_run} "
                      f"instead (found runs: {available_runs}).")
            results = run_stage_a(run_dfs, run_file_stems, run_out_dirs, sfreq, initials,
                                   template_run=template_run, template_start=args.template_start,
                                   template_window=args.template_window, ppg_guide=not args.no_ppg_guide,
                               ecg_ppg_guide=args.ecg_ppg_guide,
                                   sidecars={r: (src or {}).get("sidecar") for r, src in run_sources.items()})
            if real_box_path:
                notes = _template_comment(results)
                for run in results:
                    append_processing_log_entry(real_box_path, args.subject, run, initials,
                                                 "created QRS template", notes)
            template_json_path = find_template_json(target_out_dir, target_file_stem, initials=initials,
                                                      require_own_initials=True)

        target_df, ecg_source_label = resolve_ecg_peaks(target_df, "template", template_json_path)
        print(f"ECG seed source: {ecg_source_label}")

    source_file = f"<synthetic demo data, run {target_run}>" if args.synthetic else target_file_stem
    print(f"\nStarting Step 2 (review) for run {target_run}...")
    info.update(stage="2", run=target_run, channels=list(active_channels.keys()), file_stem=target_file_stem)
    _print_tracker_start(info)
    target_source = run_sources.get(target_run) or {}
    result = run_stage_b(target_df, sfreq, active_channels, initials, target_out_dir, target_file_stem, source_file,
                         sidecar=target_source.get("sidecar"), ppg_guide=not args.no_ppg_guide,
                         tsv_path=target_source.get("tsv_path"), allow_old_ppg=args.allow_old_ppg,
                         ecg_ppg_guide=args.ecg_ppg_guide,
                         seed_label=ecg_source_label)
    saved, finished, discarded, notes = _session_outcome(result)
    info.update(saved=saved, finished=finished, discarded=discarded)
    if args.practice and saved and result.saved_path:
        _practice_feedback(real_box_path or args.box_path, target_run, result.saved_path, target_df, sfreq,
                           active_channels, initials, target_source.get("sidecar"))

    if real_box_path and saved:
        step_text = f"{', '.join(active_channels.keys())} reviewed -- {_status_text(finished)}"
        # Compact range/mean summary for whichever segment-mode channels were
        # actually reviewed this session, in its OWN Value_ranges column (the
        # user asked for it to be kept out of the RA's free-text Notes) -- a
        # durable, cross-session record of each channel's real values,
        # complementing the same info printed to the console in
        # run_stage_b() and stored numerically in the annotation JSON. See
        # channel_config.py's module docstring for why this is scoped to
        # segment-mode channels. "µ" is written as "u": Excel opens a UTF-8
        # CSV without a byte-order mark in the system code page, which shows
        # "µS" as "ÂµS".
        range_lines = format_channel_value_ranges(
            compute_channel_value_ranges(target_df, active_channels), active_channels
        )
        value_ranges = "; ".join(range_lines).replace("µ", "u")
        append_processing_log_entry(real_box_path, args.subject, target_run, initials, step_text, notes,
                                    value_ranges=value_ranges, snapshot=not args.practice)


def _practice_feedback(practice_root, run, saved_path, df, sfreq, active_channels, initials, sidecar):
    """
    Certification practice (HLU, 2026-10-04): score the review just saved
    against the answer key in the practice folder, print the report (so the
    session log keeps it), show it, and on request show the trainee's marks
    next to the answer key (view only). Never stops the session: a missing or
    unreadable key is reported and skipped.
    """
    import json
    import practice_certification as pc
    import practice_report_gui

    key_path = pc.key_path_for(practice_root)
    if not os.path.exists(key_path):
        print(f"\nPractice: no answer key at {key_path}, so this review can't be scored. Please tell the lab staff.")
        return
    try:
        key = pc.load_key(key_path)
        with open(saved_path) as f:
            submission = json.load(f)
    except Exception as e:  # noqa: BLE001 -- report, never crash the session
        print(f"\nPractice: couldn't read the answer key or your saved file ({type(e).__name__}: {e}). Please tell "
              f"the lab staff.")
        return
    channels = pc.channels_to_score(list(active_channels))
    passed, _results, lines = pc.score_run(key, submission, run, channels)
    print("\n" + "=" * 70)
    for line in lines:
        print(line)
    print("=" * 70 + "\n")
    heading = f"Practice participant 990, run {run}: {', '.join(active_channels)}"
    if practice_report_gui.show_practice_report(heading, lines, passed) and run in key.get("runs", {}):
        pc.show_comparison(df, sfreq, key["runs"][run], submission.get("channels", {}), list(active_channels),
                           initials, sidecar)


def _handle_leftover_local_copy(real_box_path, subject, local_root):
    """
    Refuses to start a copy-down over a leftover local copy that holds work
    never pushed back to Box (audit finding C2): copy_subject_tree_to_local()
    would otherwise silently overwrite it with Box's older version. Offers to
    push the leftover files first; otherwise stops without touching anything.
    """
    unpushed = find_unpushed_local_files(real_box_path, subject, local_root)
    if not unpushed:
        return
    print("\n" + "!" * 70)
    print(f"A local copy of this subject from an earlier session is still in {local_root},")
    print(f"and {len(unpushed)} file(s) in it were never copied back to Box:")
    for relpath in unpushed:
        print(f"  - {relpath}")
    print("!" * 70)
    answer = input("Copy these file(s) to Box now, then continue? [y/N]: ").strip().lower()
    if answer not in ("y", "yes"):
        raise SystemExit(f"Stopped without changing anything. The leftover local copy is still in {local_root}. "
                         f"Copy those file(s) to Box (or ask for help), then start again.")
    confirmed, failed = push_changed_files_to_box(local_root, real_box_path, unpushed)
    for relpath in confirmed:
        print(f"  confirmed on Box: {relpath}")
    if failed:
        raise SystemExit(f"Could not confirm {len(failed)} file(s) reached Box: {failed}. Nothing was deleted; "
                         f"the local copy is still in {local_root}. Copy them to Box yourself, then start again.")


def _push_back_to_box(args, local_root, real_box_path, before_snapshot):
    """Pushes new/changed files, confirms them, and cleans up -- the end of the local-copy lifecycle."""
    changed = find_changed_files(local_root, args.subject, before_snapshot)
    if not changed:
        print("No new or changed files this session -- nothing to push back to Box.")
        if not args.keep_local:
            cleanup_local_copy(local_root, args.subject)
        return

    print(f"\nPushing {len(changed)} new/changed file(s) back to Box:")
    confirmed, failed = push_changed_files_to_box(local_root, real_box_path, changed)
    for relpath in confirmed:
        print(f"  confirmed on Box: {relpath}")

    if failed:
        print("\n" + "!" * 70)
        print(f"WARNING: could not confirm {len(failed)} file(s) made it back to Box:")
        for relpath in failed:
            print(f"  - {relpath}")
        print(f"Your local working copy is PRESERVED at {local_root} -- please check these file(s) "
              f"manually and copy them to Box yourself. Nothing has been deleted.")
        print("!" * 70 + "\n")
        return

    if args.keep_local:
        print(f"--keep-local: local working copy preserved at {local_root}.")
    else:
        cleanup_local_copy(local_root, args.subject)
        print("Local working copy removed -- all changes confirmed on Box.")


def _tracker_names(info):
    who = (f"sub-{str(info['subject']).replace('sub-', '')}" if info.get("subject")
           else info.get("file_stem") or "this subject")
    run_text = f", run {info['run']}" if info.get("run") else ""
    columns = ", ".join(info.get("channels") or [])
    row = "your reconciler row" if info.get("stage") == "3" else "your row"
    return who, run_text, columns, row


def _tracker_start_text(info):
    """
    The opening reminder: set this session's channel(s) to "in progress" in
    the Google tracking spreadsheet, whose per-channel dropdowns are "in
    progress" / "finished" (HLU, 2026-09-27; it used to ask for dates).
    None for synthetic data.
    """
    if info.get("synthetic"):
        return None
    who, run_text, columns, row = _tracker_names(info)
    if info.get("stage") == "1":
        return (f"In the Google tracking spreadsheet, set the ecg column of your row for {who} to \"in progress\" "
                f"if it isn't already (Step 1 is part of ECG).")
    return (f"In the Google tracking spreadsheet, set the {columns} column(s) of {row} for {who}{run_text} to "
            f"\"in progress\" if it isn't already.")


def _tracker_reminder_text(info):
    """
    The closing reminder for the lab's Google tracking spreadsheet (NOT
    processing_log.csv): one row per (subject, run, role), one column per
    measure named like this tool's channel keys (ecg, rsp, ppg, eda, sbp,
    dbp, emg_cor, emg_zyg, finger_temperature), each a dropdown: "in
    progress" when an RA starts a channel, "finished" when they finish it
    (HLU, 2026-09-27; it used to ask for dates). "finished" is asked for only
    when the RA ticked "finished" in the session summary. None for
    synthetic demo data.
    """
    if info.get("synthetic"):
        return None
    who, run_text, columns, row = _tracker_names(info)
    where = f"the {columns} column(s) of {row} for {who}{run_text} in the Google tracking spreadsheet"

    if info.get("stage") == "1":
        return (f"Step 1 (the ECG template) is part of ECG: the ecg column of your row for {who} should say "
                f"\"in progress\". Set it to \"finished\" after you finish reviewing ECG in Step 2.")
    if info.get("discarded"):
        return (f"You discarded this session's changes. If you've started {columns}, {where} should still say "
                f"\"in progress\".")
    if not info.get("saved"):
        if info.get("ended_early"):
            return ("This session ended before anything was saved (see the messages above), so nothing changes "
                    "in the tracking spreadsheet.")
        if info.get("finished"):
            return f"Nothing changed this session. If you haven't already, set {where} to \"finished\"."
        return f"Nothing was saved this session. If you've started {columns}, {where} should say \"in progress\"."
    if info.get("finished"):
        return f"Set {where} to \"finished\"."
    return (f"You marked {columns} as not finished: {where} should say \"in progress\". Set it to \"finished\" "
            f"when you finish.")


def _print_tracker_start(info):
    text = _tracker_start_text(info)
    if text:
        print("\n" + "-" * 70)
        print("TRACKING SPREADSHEET: " + textwrap.fill(text, width=70, subsequent_indent="  "))
        print("-" * 70)


def _print_tracker_reminder(info):
    text = _tracker_reminder_text(info)
    if not text:
        return
    print("\n" + "=" * 70)
    print("TRACKING SPREADSHEET (the Google sheet, not processing_log.csv)")
    print(textwrap.fill(text, width=70))
    print("=" * 70)


def main():
    # Everything printed this session is also written to the RA's
    # <file_stem>_annotations_<initials>.log in the run folder (session_log.py;
    # HLU, 2026-10-03). Capture starts first, so the form's equivalent command
    # and any early message are kept too.
    session_log = SessionLog()
    session_log.start()
    try:
        _main(session_log)
    finally:
        session_log.close()  # restores the console; a no-op if already closed


def _main(session_log):
    args = parse_args_or_show_gui()

    # What this session did, filled in by _run_session(), for the closing
    # tracker reminder -- which is always the LAST thing printed, even when
    # the session ends in an error.
    info = {"synthetic": args.synthetic, "stage": args.stage, "subject": args.subject, "run": args.run,
            "channels": [], "saved": False, "finished": False, "discarded": False, "ended_early": False,
            "session_log": session_log}
    exit_code = 0
    pending_exit = None
    use_local_copy = False
    copied_down = False
    local_root = None
    real_box_path = None
    before_snapshot = {}

    # Everything below runs inside try so that however the session ends --
    # normally, Ctrl-C, an error, or a deliberate stop -- anything already
    # saved locally is still pushed back to Box and confirmed (audit finding
    # C2). Previously an interrupted session skipped the push entirely.
    try:
        # Resolved here (rather than left to load_run_data()'s internal call)
        # so the resolved values -- not just whatever was or wasn't passed on
        # the command line -- are available below for processing_log.csv
        # entries. Skipped for --synthetic (no real box/subject exists) and for
        # explicit --input-run1/--input-run2 (bypasses box-path/subject entirely,
        # same condition resolve_subject_run_paths() itself uses).
        if not args.synthetic and not (args.input_run1 or args.input_run2):
            args.box_path, args.subject = resolve_box_path_and_subject(args.box_path, args.subject)
        info["subject"] = args.subject

        real_box_path = args.box_path
        use_local_copy = (
            not args.synthetic
            and not (args.input_run1 or args.input_run2)
            and not args.no_local_copy
            and bool(real_box_path)
        )

        if use_local_copy:
            if not args.local_path:
                args.local_path = input(
                    "Local working folder (this subject's files are copied here from Box, worked on, "
                    "pushed back, confirmed, then removed automatically when the session finishes): "
                ).strip()
            if not args.local_path:
                raise SystemExit("A local working folder (--local-path) is required.")
            local_root = args.local_path
            _handle_leftover_local_copy(real_box_path, args.subject, local_root)
            print(f"Copying sub-{str(args.subject).replace('sub-', '')}'s files from {real_box_path} "
                  f"to {local_root} ...")
            before_snapshot = copy_subject_tree_to_local(real_box_path, args.subject, local_root)
            # Only a COMPLETE copy-down may be pushed back later: a partial one
            # would push files the snapshot never recorded.
            copied_down = True
            args.box_path = local_root  # reroute every data read/write below to the local copy

        _run_session(args, real_box_path, info)

    except SystemExit as exc:
        if exc.code not in (None, 0):
            info["ended_early"] = True
            if isinstance(exc.code, int):
                exit_code = exc.code
            else:
                print(f"\n{exc.code}")
                exit_code = 1
            pending_exit = exc
    except KeyboardInterrupt:
        info["ended_early"] = True
        print("\nStopped (Ctrl-C).")
        exit_code = 130
    except FileNotFoundError as exc:
        # e.g. no QRS template of your own yet -- a plain message, not a traceback.
        info["ended_early"] = True
        print(f"\n{exc}")
        exit_code = 1
    except Exception:  # noqa: BLE001 -- still push and remind; details are printed
        info["ended_early"] = True
        traceback.print_exc()
        exit_code = 1

    # The log is closed (with how the session ended) before the push, so the
    # complete file goes to Box with the annotation file.
    session_log.close(session_log_footer(info, exit_code))

    if use_local_copy and copied_down:
        _push_back_to_box(args, local_root, real_box_path, before_snapshot)

    _print_tracker_reminder(info)

    if exit_code:
        if pending_exit is not None:
            # The message was already printed above the reminder; exiting with
            # a numeric code keeps Python from printing it again after the
            # reminder, while str(exc) still carries the message for callers.
            pending_exit.code = exit_code
            raise pending_exit
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
