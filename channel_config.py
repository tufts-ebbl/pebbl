"""
Defines, per physiological channel, how it should be represented in MNE and
how RA annotations on that channel should be interpreted when exported.

annotation_mode:
    "point"   -> RA marks discrete events (e.g., R peaks, pulse peaks). Each
                 annotation is treated as a single index, taken at its onset
                 sample regardless of duration.
    "segment" -> RA marks bad/artifact regions (e.g., EMG motion artifact).
                 Each annotation is treated as an [onset, offset] span.
    "none"    -> read-only reference track (e.g. task event/phase codes).
                 Not something the RA marks or corrects -- it's already
                 computed upstream and just shown for context. Never
                 seeded with annotations, never appears in exported output,
                 and never selectable via --channels (see REFERENCE_CHANNELS
                 below, kept separate from CHANNELS for exactly that reason).

auto_peaks_col: name of the boolean peak column already present in an
    existing physio.tsv.gz (from physioBatch.py) that can be used to seed
    the channel with the automated detections before RA review. None means
    "start with no pre-existing annotations" (true for the EMG placeholders
    today, since there is no automated EMG detector yet).

machine_qc: True for the channels the new physioProcess sidecar's MachineQC
    block checks (rsp, eda, ppg, sbp, dbp). A fresh review of a segment
    channel starts from those bad segments (sidecar plan Phase 2); the flag
    only decides whether "this sidecar has no machine QC" is worth saying.

review_alone: True for a channel that must be reviewed in a session of its own
    (PPG); see select_channels().

check_interval_regularity: whether check_interval_regularity() (annotation_io.py)
    should flag unusually long/short gaps between this channel's exported
    peaks when Stage B closes -- a post-hoc, non-interactive stand-in for
    R-R interval QA (wishlist item 10c's live-updating track was judged too
    fragile to build; this covers the same "spot a missed or spurious peak"
    goal without touching mne-qt-browser's internals at all). Only makes
    sense for channels with a genuinely PERIODIC underlying rhythm (heart
    rate for ecg/ppg) -- deliberately omitted for eda, since SCR peaks are
    driven by stimuli/arousal and normally have irregular spacing; flagging
    that would just be noise. (rsp had it too until it became a segment
    channel on 2026-09-25.)

Display scale (compute_scalings() in physio_io.py) is computed from each
channel's OWN data automatically -- there's no per-channel override here.
A channel with degenerate data (all-NaN, or ~zero variance like today's
literal-zero EMG placeholder) automatically borrows another channel's
scale instead; real data (once real EMG/PPG replaces a placeholder) gets
its own properly auto-fit scale with no config change needed.

Every SEGMENT-mode channel's real min/mean/max is printed once at Stage B
startup, and logged (compactly) to processing_log.csv's Notes field for
whichever of them were actually reviewed this session -- see
compute_channel_value_ranges() (physio_io.py). This exists because MNE's
raw.plot() crosshair position is relative to a LOCAL, per-window
scrolling mean (remove_dc, on by default) -- confirmed there's no public
per-channel override, and disabling it globally was tested live and
confirmed broken for exactly the channels that'd need it (eda/sbp/dbp's
real baseline is many display-scale-units away from zero, pushing the
trace off-screen or producing a NaN crosshair reading; see HANDOFF.md).
Segment-mode is exactly the right dividing line here, not a separate
flag: it's every channel an RA reviews by flagging bad STRETCHES rather
than correcting individual points (rsp/eda/sbp/dbp/emg_cor/emg_zyg/
finger_temperature), where overall range/mean is useful QA context
regardless of whether the unit happens to be clinically "calibrated" --
point-mode ecg/ppg are read relative to their own waveform shape
either way, so this doesn't apply to them.
"""

import mne

CHANNELS = {
    "ecg": {
        # MNE's native "ecg" type assumes raw data in VOLTS (unit_scalings
        # 1e6, display "µV"). This project's ECG is calibrated in
        # MILLIVOLTS instead (traced to the raw Biopac "ECG - ECG100C"
        # channel, read via nk.read_acqknowledge() which honors that
        # hardware's own calibration; confirmed with the user) -- using
        # MNE's default as-is would inflate every displayed value 1000x.
        # See the DEFAULTS override below.
        "mne_type": "ecg",
        "annotation_mode": "point",
        "auto_peaks_col": "ecg_peaks",
        "point_label": "peak_ecg",
        "label": "ECG",
        "check_interval_regularity": True,
        "interval_check": "berntson",  # Berntson et al. (1990) criterion; see annotation_io.py
    },
    "rsp": {
        # MNE has no built-in defaults AT ALL for "resp" (confirmed: not a
        # key in mne.defaults.DEFAULTS["units"]) -- which meant, until the
        # override below, RSP silently got NO scale bar shown whatsoever,
        # regardless of any labeling question. This project's RSP comes
        # from a Biopac respiratory-effort belt (TSD221-MRI) with no
        # volume/pressure calibration step anywhere in the pipeline. The
        # new physioProcess sidecar labels it "V" (the belt transducer's
        # output voltage); the user chose that label (2026-09-25).
        #
        # segment-mode since 2026-09-25 (user's decision, sidecar plan
        # Phase 1): RAs mark bad stretches rather than correcting breath
        # peaks, and fresh reviews start from the pipeline's MachineQC bad
        # segments (Phase 2). Earlier point-mode RSP reviews are kept in the
        # saved file under "rsp_points_legacy" -- see save_annotation_json().
        "mne_type": "resp",
        "annotation_mode": "segment",
        "auto_peaks_col": None,
        "segment_label": "bad_rsp",
        "machine_qc": True,  # the new sidecar's MachineQC checks this channel
        "label": "RSP",
    },
    "ppg": {
        # "bio" is shared by nothing else in this project, so overriding
        # its DEFAULTS entry only affects PPG. MNE's native "bio" type
        # assumes raw VOLTS (unit_scalings 1e6, display "µV") -- but this
        # project's PPG doesn't come from Biopac at all (a separate
        # "CareTaker" device's pulse waveform), and that waveform is
        # inherently uncalibrated/arbitrary, not a voltage signal --
        # confirmed with the user. MNE's default would both mislabel it
        # AND inflate it 1,000,000x.
        "mne_type": "bio",
        "annotation_mode": "point",
        "auto_peaks_col": "ppg_peaks",
        "point_label": "peak_ppg",
        "machine_qc": True,  # the new sidecar's MachineQC checks this channel
        # Reviewed in its own session, never with other channels (ppg-plan
        # §4b): with ECG it would draw both channels' markers across both
        # rows, and the viewer's label picker resets to the alphabetically
        # first label, so a drag meant as one channel's mark can save as
        # another's. Left out of the "all channels" default.
        "review_alone": True,
        "label": "PPG",
        "check_interval_regularity": True,
        "interval_check": "berntson",  # Berntson et al. (1990) criterion; see annotation_io.py
    },
    "eda": {
        # "gsr" (galvanic skin response) is MNE's own native channel type
        # for this signal -- deliberately not "bio" (ppg/eda have very
        # different amplitude ranges and would be forced to share one
        # display scale) and not "misc" (which has no real physical unit
        # at all, hence the earlier "AU" label). See the DEFAULTS override
        # below for why this is really labeled "µS", not MNE's own
        # default "S" for this type.
        #
        # segment-mode, not point-mode: the user doesn't expect RAs to
        # correct individual SCR peaks (unlike ECG/RSP/PPG's R-peaks/
        # breaths/pulses, which genuinely need per-beat correction) --
        # only to flag bad/artifact stretches, same as EMG. Considered
        # keeping the automated SCR_Peaks detections visible as a
        # non-editable reference while still allowing segment marking, but
        # mne-qt-browser's AnnotRegion graphics item hardcodes
        # movable=True for every annotation region regardless of
        # description (confirmed by reading _graphic_items.py directly) --
        # there is no supported way to make some annotations on a channel
        # read-only while others stay editable, and this project avoids
        # patching mne_qt_browser's private internals to force it (same
        # reasoning as the merge-bug investigation elsewhere in this
        # project). So EDA now starts blank, exactly like EMG, rather than
        # showing peaks that would only ever be pseudo-read-only.
        "mne_type": "gsr",
        "annotation_mode": "segment",
        "auto_peaks_col": None,
        "segment_label": "bad_eda",
        "machine_qc": True,  # the new sidecar's MachineQC checks this channel
        "label": "EDA",
    },
    "emg_cor": {
        "mne_type": "emg",
        "annotation_mode": "segment",
        "auto_peaks_col": None,
        "segment_label": "bad_emg_cor",
        "label": "EMG_Corrugator",
    },
    "emg_zyg": {
        "mne_type": "emg",
        "annotation_mode": "segment",
        "auto_peaks_col": None,
        "segment_label": "bad_emg_zyg",
        "label": "EMG_Zygomatic",
    },
    "sbp": {
        # CareTaker's beat-to-beat systolic blood pressure -- already a
        # derived per-beat VALUE (physioProcess's processCT() reads it
        # straight from the device's own "Systolic (mmHg)" column), not a
        # raw waveform with a peak for an RA to nudge, so segment-mode
        # (flag bad/untrustworthy stretches) rather than point-mode. "misc"
        # is MNE's generic catch-all type -- deliberately not "bio" (PPG
        # already occupies it, labeled "AU"; sharing it would force SBP/DBP
        # to the same display unit as PPG, and MNE's unit override is keyed
        # by TYPE, not by individual channel) and nothing else in this
        # project uses "misc" yet, so no collision. See the DEFAULTS
        # override below for the "mmHg" label.
        "mne_type": "misc",
        "annotation_mode": "segment",
        "auto_peaks_col": None,
        "segment_label": "bad_sbp",
        "machine_qc": True,  # the new sidecar's MachineQC checks this channel
        "label": "SBP",
    },
    "dbp": {
        # Same reasoning as "sbp" above (CareTaker's diastolic pressure).
        "mne_type": "misc",
        "annotation_mode": "segment",
        "auto_peaks_col": None,
        "segment_label": "bad_dbp",
        "machine_qc": True,  # the new sidecar's MachineQC checks this channel
        "label": "DBP",
    },
    "finger_temperature": {
        # Not collected in the current study (no finger-temperature probe
        # in use); added ahead of a possible future in-person study that
        # would. Named specifically "finger_temperature" (not just
        # "temperature") to leave room for a different temperature source
        # later without a naming collision. MNE's "temperature" TYPE
        # already has correct built-in defaults (unit "C", unit_scalings
        # 1.0) -- confirmed via mne.defaults.DEFAULTS -- unlike
        # ecg/resp/bio/gsr/stim above, which all needed the override loop
        # below. If the eventual real device reports Fahrenheit instead,
        # add ("temperature", "°F") to that loop -- same "relabel only,
        # never rescale" approach as everything else here, since this
        # project never does unit MATH, only display-string correction.
        "mne_type": "temperature",
        "annotation_mode": "segment",
        "auto_peaks_col": None,
        "segment_label": "bad_finger_temperature",
        "label": "Finger Temperature",
    },
}

def select_channels(requested=None):
    """
    The CHANNELS subset for one review session. requested: a list of channel
    keys, or None for the default (every channel except those reviewed on
    their own, i.e. PPG). Raises SystemExit, in plain language, for an
    unknown channel or a "review_alone" channel combined with others. Shared
    by physio_review.py (Steps 2 and 3), physio_annotate.py and
    annotation_io.run_stage_b(), so no entry point can skip the rule.
    """
    if requested is None:
        return {k: v for k, v in CHANNELS.items() if not v.get("review_alone")}
    requested = [c.strip() for c in requested if c.strip()]
    unknown = [c for c in requested if c not in CHANNELS]
    if unknown:
        raise SystemExit(f"Unknown channel(s) {unknown}. Options: {list(CHANNELS.keys())}")
    check_review_alone(requested)
    return {k: v for k, v in CHANNELS.items() if k in requested}


def empty_channels(df, keys):
    """
    The keys among `keys` whose column is in the file but holds no values at
    all (every row NaN). E.g. PPG, SBP and DBP on the 7 runs with no CareTaker
    data: their columns are written but empty (physioProcess maintainer,
    2026-09-26). There is nothing to review on such a channel.
    """
    return [k for k in keys if k in df.columns and df[k].isna().all()]


def no_data_message(keys):
    """Why a session can't review these empty channel(s), in plain language."""
    labels = ", ".join(CHANNELS[k]["label"] if k in CHANNELS else k for k in keys)
    if set(keys) <= {"ppg"}:
        return ("This run has no usable PPG (the CareTaker pulse wasn't recorded, or doesn't cover the task), so "
                "there is nothing to review. Please tell the lab staff.")
    if set(keys) <= {"sbp", "dbp"}:
        return (f"This run has no usable {labels} (no CareTaker blood pressure was recorded, or none covers the "
                f"task), so there is nothing to review. Please tell the lab staff.")
    return f"This file's {labels} data are empty, so there is nothing to review there. Please tell the lab staff."


def present_channels(active_channels, df_columns, explicit, empty=()):
    """
    Narrows a session's channels to those the file actually has. A channel
    the RA asked for by name that isn't in the file, or whose column is empty
    (`empty`, from empty_channels(): e.g. SBP/DBP on a run with no CareTaker
    vitals), is refused; channels left out of an all-channels default are
    dropped with a note, so they are never logged or tracked as reviewed
    (checkpoint-2 review; run-8 decisions, 2026-09-26).
    """
    missing = [k for k in active_channels if k not in df_columns]
    if missing and explicit:
        labels = ", ".join(CHANNELS[k]["label"] for k in missing)
        raise SystemExit(f"This file has no {labels} data, so there's nothing to review there.")
    blank = [k for k in active_channels if k in empty and k in df_columns]
    if blank and explicit:
        raise SystemExit(no_data_message(blank))
    if missing:
        print(f"NOTE: this file has no {', '.join(CHANNELS[k]['label'] for k in missing)} data; "
              f"those channels are left out.")
    if blank:
        print(f"NOTE: this file's {', '.join(CHANNELS[k]['label'] for k in blank)} data are empty (nothing was "
              f"recorded); those channels are left out.")
    present = {k: v for k, v in active_channels.items() if k in df_columns and k not in blank}
    if not present:
        raise SystemExit("This file has none of the channels asked for.")
    return present


def check_review_alone(channel_keys):
    channel_keys = [k for k in channel_keys if k not in COMPANION_CHANNELS]  # companions ride with their parent
    alone = [k for k in channel_keys if CHANNELS.get(k, {}).get("review_alone")]
    if alone and len(channel_keys) > 1:
        labels = ", ".join(CHANNELS[k]["label"] for k in alone)
        raise SystemExit(f"{labels} is reviewed on its own: start a separate session with only {labels} "
                         f"(command line: --channels {','.join(alone)}).")


# Read-only reference tracks: shown alongside whatever CHANNELS the RA is
# reviewing (when present in the file), but never user-selectable via
# --channels/the GUI's channel picker, never seeded with annotations, and
# never included in Stage B's exported output -- there's nothing for an RA
# to correct here, it's already-computed context. Kept in a separate dict
# from CHANNELS specifically so the existing --channels validation/GUI
# dropdown logic doesn't need to know about them at all.
REFERENCE_CHANNELS = {
    "event": {
        "mne_type": "stim",
        "annotation_mode": "none",
        "label": "Event",
        # Phase codes from physioProcess's processEvents(): 0 = outside the
        # task (baseline padding before/after), 20/30/40/50/65 = the task's
        # own baseline/read/imagery/recovery/ratings phases. Only present in
        # files reprocessed after that upstream change (see HANDOFF.md §9);
        # older files simply won't have this column, and everything here
        # degrades gracefully to "channel not present, skip it" in that case.
    },
}

# Companion channels (HLU, 2026-09-27): reviewed automatically alongside
# their parent, with NO row of their own (MNE annotations span every row).
# "bad_ppg" holds the PPG's bad stretches in PPG sessions (Steps 2 and 3):
# a fresh one starts from the machine's PPG dropout flags (MachineQC.ppg,
# via machine_qc_source), RAs add or delete stretches with the "bad_ppg"
# label, and it is saved as its own segment entry next to the PPG peaks.
# Bad stretches never keep peaks: automated peaks inside a pre-marked
# stretch aren't loaded, and peaks inside any final stretch are dropped on
# save (annotation_io.drop_peaks_in_bad_stretches()). Its provenance is the
# ppg column's.
COMPANION_CHANNELS = {
    "bad_ppg": {
        "mne_type": "bio",
        "annotation_mode": "segment",
        "auto_peaks_col": None,
        "segment_label": "bad_ppg",
        "label": "PPG bad stretches",
        "companion_of": "ppg",
        "machine_qc": True,
        "machine_qc_source": "ppg",
        "banner": "mark stretches where there's no usable pulse",
    },
    # "bad_ecg" (HLU, 2026-09-27, after trying bad_ppg): the same for ECG, in
    # Steps 2 and 3 (never Step 1). physioProcess has no machine check for
    # ECG, so it starts blank. Mark bad only where R peaks can't be located:
    # correct peaks wherever they can be found.
    "bad_ecg": {
        "mne_type": "ecg",
        "annotation_mode": "segment",
        "auto_peaks_col": None,
        "segment_label": "bad_ecg",
        "label": "ECG bad stretches",
        "companion_of": "ecg",
        "machine_qc": False,
        "banner": "mark stretches only where R peaks can't be located (correct them wherever you can)",
    },
}


def with_companions(channel_configs):
    """channel_configs plus each companion of a reviewed parent, placed right after it (idempotent)."""
    combined = {}
    for key, cfg in channel_configs.items():
        combined[key] = cfg
        for comp_key, comp_cfg in COMPANION_CHANNELS.items():
            if comp_cfg["companion_of"] == key and cfg.get("annotation_mode") in ("point", "segment"):
                combined[comp_key] = comp_cfg
    return combined


def is_present(ch_key, cfg, names):
    """Whether a channel is in the session's rows (a companion counts through its parent's row)."""
    return ch_key in names or cfg.get("companion_of") in names

# Read-only "guide" rows: another channel shown right under the one being
# scored, for context only -- never annotated or exported, and shown exactly
# as stored (no blanking of machine-invalid stretches, "option 1"). Keyed by
# the column they plot; "guide_for" lists the scored channels a guide goes
# under (after the last of them in the session). A guide is added by
# annotation_io.combine_with_reference_channels() only when its own channel
# isn't being scored in that session, so it never collides with a review.
#   - PPG under ECG (sidecar plan Phase 1b; user's decisions 2026-09-25/26).
#     The pulse may be shifted by a beat or two relative to the ECG (a
#     per-run CareTaker-to-Biopac offset of up to ~2 s; physioProcess
#     maintainer, 2026-09-25), so RAs use it to notice a missing or extra
#     beat, never to place a peak.
#   - PPG under SBP/DBP, EDA under RSP, RSP under EDA (HLU, 2026-09-27).
#     SBP/DBP come from CareTaker's finger pulse (same device and clock as
#     the PPG). EDA and RSP are both Biopac (same clock). EDA helps one way
#     only: a rise 1-3 s after a big breath shows the breath was real, and
#     no rise means nothing. HLU decided against ECG under PPG: the
#     in-scanner ECG is messy and PPG is clean enough to judge on its own.
# A guide isn't shown when its column is missing or all-NaN, when the
# machine check marks more than 90% of that run's column as unusable (a
# dead RSP belt is stretched to fill its row and would look like fast,
# uneven breathing), or, for PPG, on files processed before the PPG timing
# fix; the console says which (annotation_io.guide_banner_lines()).
GUIDE_CHANNELS = {
    "ppg": {
        "mne_type": "bio",
        "annotation_mode": "none",
        "label": "PPG (guide)",
        "guide_for": ("ecg", "sbp", "dbp"),
    },
    "eda": {
        "mne_type": "gsr",
        "annotation_mode": "none",
        "label": "EDA (guide)",
        "guide_for": ("rsp",),
    },
    "rsp": {
        "mne_type": "resp",
        "annotation_mode": "none",
        "label": "RSP (guide)",
        "guide_for": ("eda",),
    },
}

GUIDE_TEXT = ("PPG (guide): the pulse, for rhythm only -- it may be shifted by a beat or two from the ECG. Use it to "
              "notice a missing or extra beat, never to decide where an R peak goes. Device dropouts look like low, "
              "smooth arcs with no clear pulse.")

# What each guide row is for, by (guide, scored channel), shown in the
# console when the guide is on screen. The SBP/DBP, RSP and EDA lines are
# HLU's wording (2026-09-27).
_PPG_UNDER_BP = ("PPG (guide): the finger pulse the blood pressure comes from. Where there's no usable pulse, the "
                 "pressure values aren't real.")
GUIDE_TEXTS = {
    ("ppg", "ecg"): GUIDE_TEXT,
    ("ppg", "sbp"): _PPG_UNDER_BP,
    ("ppg", "dbp"): _PPG_UNDER_BP,
    ("eda", "rsp"): ("EDA (guide): a rise 1-3 s after a big breath shows the breath was real. No rise means nothing. "
                     "Never mark RSP bad because of EDA."),
    ("rsp", "eda"): ("RSP (guide): a skin-conductance rise 1-3 s after a big breath is a real response to the breath, "
                     "not bad signal. Mark EDA bad only where the EDA itself is unusable."),
}
# Added to a guide's line when the scored channel's machine check has this
# reason, i.e. the machine pre-marks what the guide would show (HLU,
# 2026-09-27): BP inside PPG dropouts, from run 9's "ppg_dropout". Files
# without that reason (run 8 and earlier) keep the plain line.
GUIDE_PREMARK_REASONS = {("ppg", "sbp"): "ppg_dropout", ("ppg", "dbp"): "ppg_dropout"}
GUIDE_PREMARKED_TEXT = "The machine pre-marks those stretches; check the edges."

# An exception line shown under a guide's line when the GUIDE channel's own
# machine check has this reason (HLU, 2026-10-03): PPG's counter_loss (run
# 10), CareTaker pulse samples lost in transfer. The CT5 computes BP on board
# from its complete pulse signal and sends the readings separately from the
# pulse copy (CareTaker manual Rev 6, §7.7 and §8.10), and the readings carry
# on through every loss (physioProcess maintainer's check, 2026-10-03). So BP
# there is real even though the PPG guide can't confirm it. A device zero run
# inside such a stretch is still a dropout (pre-marked as ppg_dropout).
GUIDE_EXCEPTION_REASONS = {("ppg", "sbp"): "counter_loss", ("ppg", "dbp"): "counter_loss"}
GUIDE_COUNTER_LOSS_TEXT = ("PPG (guide): in {count} stretch(es) ({where}) CareTaker pulse samples were lost in "
                           "transfer, so the PPG guide can't be used to check BP there. The BP readings in those "
                           "stretches were still measured by the device; the loss affected only the pulse copy sent "
                           "to the app. Keep them, except where the machine marked a finger-pulse dropout.")
# At most this many stretches are listed by time; the rest are counted.
GUIDE_EXCEPTION_MAX_LISTED = 6

# How to add a PPG peak, printed in every PPG session (HLU, 2026-10-04,
# option A): a drawn peak is saved at the highest point inside the box
# (annotation_io._climb_if_cut, Option C), so a box over the whole top
# picks the higher crest without the RA judging it. The RA's peak is the
# anchor for physioProcess's upstroke search, which is the same under
# either crest, so an existing mark isn't moved between crests.
PPG_ADD_PEAK_TEXT = ("   PPG: to add a peak, drag the box over the whole top of the pulse (both humps, if it has two); "
                     "your mark lands on its highest point. Don't move an existing mark from one hump to the other.")

# Guide rows shown only on request (HLU, 2026-10-03, "option 2"): the PPG
# under ECG, in Steps 1-3. Each run's CareTaker timeline sits a run-specific
# constant offset from the Biopac's (physioProcess plan item 7a: R peak to
# placed reading 0.43 s in sub-001 r1 but 1.09 s in sub-085 r1), so placing
# an R peak "just before the pulse" can land in the wrong beat, most of all
# in noisy ECG, where the right mark is an ECG bad stretch. The rhythm is
# still sound, so an RA can ask for it (--ecg-ppg-guide, or ticking the
# form's guide box with ECG selected); the review records it (guide_shown).
# Under SBP/DBP the PPG stays on by default: same CareTaker clock as the BP.
# Values: the console line when the guide is off for that channel, and the
# line when it's on screen anyway for another scored channel (a CLI session
# scoring ECG with SBP/DBP).
GUIDE_OPT_IN = {
    ("ppg", "ecg"): {
        "off": ("PPG (guide): not shown under ECG by default -- the pulse's delay after the R peak differs from run "
                "to run, so it can mislead R-peak placement. To use it for counting beats only, tick the guide box "
                "or pass --ecg-ppg-guide."),
        "shown_for_others": ("PPG (guide): shown here for SBP/DBP only -- don't use it to place ECG R peaks (the "
                             "pulse's delay after the R peak differs from run to run)."),
    },
}

# Shown once whenever any guide row is on screen: the viewer shades a bad
# stretch across every row, the guide rows included.
GUIDE_CONTEXT_TEXT = "Guide rows are context only: mark only what you see on the scored row."

# Every channel type's display unit/scaling gets corrected here to match
# what this project's data ACTUALLY is, traced back to its raw hardware
# source and confirmed with the user for each one individually (never
# assumed) -- see the matching comment on each CHANNELS entry above for
# the specific reasoning per channel. In every case the fix is the same
# shape: the DATA is already in the unit named below, so unit_scalings is
# set to 1.0 (no multiplication at all) and only the LABEL changes --
# never the numbers a click-drag/crosshair/scale-bar reflects back.
#
# raw.plot() has no per-call parameter for overriding a channel type's
# display-unit string -- confirmed by reading mne/viz/raw.py's plot_raw()
# signature -- so the only way to correct this is to override MNE's own
# global DEFAULTS dict before any Raw gets plotted. This is done here, as
# a module-level side effect, since every module that touches channel
# display already imports channel_config -- and it edits mne.defaults.
# DEFAULTS, a PUBLIC, documented dict (unlike the private,
# underscore-prefixed mne_qt_browser internals this project has
# deliberately avoided touching elsewhere -- see HANDOFF.md's debugging
# history for why that line was drawn).
#
# ecg:  MNE's own default is raw Volts, 1e6x -> "µV". Ours is already mV.
# resp: MNE has NO default at all for "resp" -- RSP silently had no scale
#       bar whatsoever until this. Labeled "V" (the belt's output voltage),
#       matching the new physioProcess sidecar (user's decision, 2026-09-25).
# bio:  MNE's own default is raw Volts, 1e6x -> "µV" (shared by nothing
#       else in this project). Ours (PPG) is uncalibrated/arbitrary.
# gsr:  MNE's own default is raw Siemens, 1.0x -> "S". Ours is already µS.
# stim: MNE has NO default at all for "stim" either (same gap as "resp"
#       had) -- harmless for the scale bar itself (_add_scalebars()
#       explicitly excludes "stim" from ever getting one), but FATAL for
#       the crosshair/value-readout feature (toggle with "x"): hovering
#       over the "event" reference channel called
#       _get_channel_scaling()->unit_scalings["stim"] with no guard at
#       all, raising a live KeyError. Found by the user actually using
#       the crosshair feature right after it was documented -- exactly
#       the kind of thing that only surfaces from real use, not review.
# misc: MNE has NO default at all for "misc" either -- same gap as "resp"/
#       "stim". Shared by SBP and DBP (both genuinely mmHg, so no
#       per-channel conflict the way "bio" would have been).
for _mne_type, _unit in (("ecg", "mV"), ("resp", "V"), ("bio", "AU"), ("gsr", "µS"), ("stim", "AU"),
                          ("misc", "mmHg")):
    mne.defaults.DEFAULTS["units"][_mne_type] = _unit
    mne.defaults.DEFAULTS["si_units"][_mne_type] = _unit
    mne.defaults.DEFAULTS["scalings"][_mne_type] = 1.0
