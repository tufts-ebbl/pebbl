# Multisignal Annotation Prototype

A prototype for reviewing/correcting/rejecting **all** physio signals (ECG, RSP, PPG, EDA, EMG placeholders, and -- once collected -- SBP, DBP, and finger temperature) in one interactive viewer, built on MNE-Python's annotation system -- with no JupyterLab required.

**Two ways to run it, same underlying tool**: launch `physio_review.py`
with **no flags at all** for a simple point-and-click form (Box path,
subject, run, initials -- see "With no arguments at all" below), or run
it **entirely from the command line** with flags for scripting/automation
or if you just prefer it (see "With flags" below). Passing any flag at
all skips the form and uses the CLI path. To see every available flag and
what it does, run `annotate_env\Scripts\python.exe physio_review.py
--help` (this lists everything below, straight from the script itself,
so it's always current).

Two stages, plus a unified script that runs them together:

- **Stage A** (`qrs_template_stage.py`, optional) — builds an ECG QRS
  template from a short RA-corrected window in ONE run, and applies it via
  wavelet filter + cross-correlation to EVERY run that exists for that
  subject (no separate template needed per run, but every RA who reviews
  ECG builds their own -- see "Every RA builds their own ECG template"
  below). This replaces `step1_qrs_template.ipynb`, and is the approach
  needed for messy/in-scanner ECG. Explicitly passing `--ecg-source batch`
  skips it entirely and always uses the batch pipeline's own peak
  detection instead, for clean in-lab ECG that doesn't need it.
- **Stage B** (`physio_annotate.py`) — the full multi-signal review/rejection
  session (ECG, RSP, PPG, EDA, EMG placeholders, SBP, DBP, finger
  temperature) in one MNE window. Replaces `step2_physio_correction.ipynb`.
- **`physio_review.py` (recommended entry point)** — one command that runs
  both: it builds YOUR OWN QRS template only if you don't already have one
  for this subject (skipping Stage A automatically otherwise, e.g. when
  you already built one while reviewing run 1 and are now reviewing run 2
  yourself), then launches Stage B for the run you asked to review.
  `qrs_template_stage.py` and `physio_annotate.py` remain available
  standalone for troubleshooting a single stage in isolation.

This is intentionally self-contained: it does **not** import from or modify
anything in `physioProcess/` or the existing `physioCorrection` notebooks.
It only reads the `*_physio.tsv.gz` (+ `.json` sidecar) file format that
`physioBatch.py` already produces, and writes new, independently-named JSON
files reflecting RA decisions. The "second pass" (feeding those decisions
back through `processECG()`/`processRSA()`/etc.) is explicitly out of scope
for this prototype.

## Setup (one-time)

**Research assistants:** follow the PEBBL user manual
(`docs/PEBBL_User_Manual.pdf`, or the web version). In short (2026-10-04):
- install Python 3.11, Git and Box Drive;
- `git clone` the public pebbl repository into the home folder;
- run **Set up PEBBL.bat** (Windows) or `bash ~/pebbl/setup_pebbl.sh` (Mac)
  once. It makes `annotate_env` with Python 3.11 and installs
  `requirements.txt`;
- start with **Start PEBBL.bat** or **Start PEBBL.command**.

The start scripts run `pebbl_launcher.py`, which:
- pulls the latest version (`git pull --ff-only`, only in a clone of its own;
  never in this development folder);
- reinstalls the requirements when `requirements.txt` changed;
- opens the form, one session per process, and opens it again after each
  session (HLU, 2026-10-05).
  - It sets `PEBBL_LAUNCHER=1`. physio_review then exits 3 when the form is
    canceled (which closes PEBBL), and 4 when a session stops with its own
    message (which reopens the form, like a finished session's 0).
  - Ctrl-C (130), or any other failure such as an error before the form
    opens, closes PEBBL, so a broken update can't loop.
  - Each session finishes completely (copy back, cleanup, session log) before
    the next form appears.

The scripts set `PEBBL_NO_PAUSE=1` to skip their "press any key" pauses for
automated tests. `.gitattributes` keeps the `.bat` files CRLF and the Mac
scripts LF.

By hand, as before. This lives in its own virtual environment, separate from
`physio_correction_env`, so nothing about the existing RA workflow's
clean-environment guarantee is touched:

```
cd multisignal_annotation
python -m venv annotate_env
annotate_env\Scripts\activate
pip install -r requirements.txt
```

**The public repository.** RAs get PEBBL from a public `pebbl` GitHub
repository that holds only the tool. `export_public_pebbl.py` (lab staff)
copies the RA-facing files into a local clone of it. It is read-only unless
given `--apply`. It never copies this folder's internal notes, data or
history. The user manual's source is `docs/PEBBL_User_Manual.html`;
`docs/build_manual_pdf.py` rebuilds the PDF and the logo PNG with Edge or
Chrome. The screenshots in `docs/img/` were captured from synthetic data.
The export also writes `docs/index.html` (the manual as a standalone page,
with a link to the PDF) and `docs/.nojekyll`. GitHub Pages serves the public
repo's `docs` folder at https://tufts-ebbl.github.io/pebbl/, the manual's
public address (GitHub's own file view doesn't preview this PDF).

## The unified script: `physio_review.py`

### With no arguments at all: a simple form

```
annotate_env\Scripts\python.exe physio_review.py
```

Run it with **zero flags** and a small window pops up asking:
- **Step 1: Build ECG QRS template**, **Step 2: Review a channel**, or
  **Step 3: Reconcile two reviewers** (Step 2 is preselected, as the more
  common day-to-day action).
- **Data folder** (with a Browse... button): any folder holding the sub-XXX
  folders; for our lab, the Box derivatives folder (labeled "Box derivatives
  path" until 2026-10-05; `--box-path` and its alias `--data-path`). PEBBL
  finds each run's `sub-XXX_ses-runY_task-<any>_physio.tsv.gz`, preferring
  our `task-sdi` file whenever it exists (`physio_io.find_run_path`). Also a
  local working folder (with Browse..., see "Working from Box" below) and
  the subject.
- If Step 2 or Step 3: which run, and **which single channel** to
  review/reconcile (defaults to ECG) -- reviewing every available channel
  at once turned out to be too busy in practice, so the form asks for one
  at a time. Step 1 always works with ECG, so that field is locked/disabled
  automatically when Step 1 is chosen.
- If Step 1: a **template window start (s)** field (defaults to 0) -- the
  GUI equivalent of `--template-start`, for retrying a different 20-second
  window without dropping to the command line.
- Your initials (2-4 letters; stored lower-case, so "HLU" and "hlu" are the
  same person). In Step 3 these are the **reconciler's** initials, plus
  **two reviewers' initials to compare**, which must be different people.
  Your initials name your saved files, so the form **asks you to confirm
  them once** when they've never appeared in the Box `processing_log.csv`
  (expected on a new RA's first day; otherwise a sign of a typo), and
  always when they're "sub" (the start of a subject ID). No sends you back
  to fix them. This check applies only to real data, and only when the log
  can be read.
  There's also a checkbox to try any of the three steps with synthetic demo
  data instead, and a **Practice for certification** checkbox (subject 990
  from the `practice` folder next to the derivatives folder, scored after
  each Step 2 save; initials count as known if either log has them; see "RA
  certification practice" below).
- **Remembered between sessions:** the Box path, local working folder, and
  initials, stored in your own user settings on this computer (not in the
  repo) each time you start a real-data session.
- **Options**, with defaults that suit the average session:
  - **ECG peak source** (Step 2, ECG only): "Your own ECG template (built in
    Step 1)" (default; never another RA's) or "Automatic detection only
    (no template)". The GUI equivalent of `--ecg-source auto|batch`.
  - **Show the guide channel (read-only)** (on by default; offered for SBP,
    DBP, RSP and EDA): the read-only row under the channel you score (see
    "Guide rows" below). Unticking it is the GUI equivalent of
    `--no-ppg-guide`. **With ECG selected (or Step 1) the box reads "Show the
    PPG under the ECG (read-only; for counting beats only)" and starts
    unticked** (HLU, 2026-10-03). Ticking it is the GUI equivalent of
    `--ecg-ppg-guide`.
- **Show lab-staff options** (a checkbox): hides the options that can strand
  data or change template building unless you ask for them:
  - **Build template from run** and **template window length (s)** (Step 1
    only, default run "1" / 20s) -- the GUI equivalent of
    `--template-run`/`--template-window`.
  - **Skip local copy** -- work directly against the Box path instead of
    copying the subject down locally first (see "Working from Box" below);
    checking this makes the local working folder field optional and
    disables it. The GUI equivalent of `--no-local-copy`.
  - **Keep the local copy after pushing back to Box** -- for
    troubleshooting; the local copy is always kept regardless if a push
    can't be confirmed. The GUI equivalent of `--keep-local`.

Once you've entered/browsed to a box path, the Subject field's placeholder
suggests the next unclaimed subject (see `processing_log.csv` below) --
type over it if you're reviewing something else.

Fill it in, click Start, and it proceeds exactly as if you'd typed the
equivalent CLI flags (`--stage 1`, `--stage 2 --channels <channel>`, or
`--stage 3 --channels <channel> --compare-initials A,B`, see below). This
uses PyQt6, which is already installed (it's what runs the annotation
viewer itself), so no extra setup is needed.

Right after you click Start, that equivalent command is also printed to
the terminal, e.g.:
```
Equivalent command (copy/paste and edit --run/--channels/etc. to repeat this from the CLI):
  "annotate_env\Scripts\python.exe" "physio_review.py" "--box-path" "<path>" "--subject" "001" "--stage" "2" "--local-path" "<path>" "--initials" "hlu" "--run" "1" "--channels" "ecg" "--ecg-source" "auto"
```
Every value is quoted like this deliberately -- it makes the command safe
to paste into cmd.exe, PowerShell, or Git Bash alike (an unquoted Windows
path pastes fine into cmd.exe/PowerShell but silently loses its
backslashes in bash). Copy/paste it and just change `"1"` after `--run`
to `"2"` to repeat the same session for the other run without reopening
the form -- or edit `--channels` to review a different channel next, etc.

Passing **any** flag at all (even just `--initials hlu`) skips the form
entirely and falls back to the normal CLI behavior described below,
prompting on the console for whatever you didn't pass.

### With flags

```
annotate_env\Scripts\python.exe physio_review.py --synthetic --run 1 --initials hlu
```

```
annotate_env\Scripts\python.exe physio_review.py --box-path C:\Users\you\Box\DATA\Processed\physioProcessing\derivatives --local-path C:\Users\you\PhysioWorking --subject 001 --run 1 --initials hlu
```

Behavior, with no `--stage` given (the default, combined flow):
- Looks for **your own** QRS template for this subject (per `--initials`).
  If you already have one, Stage A is **skipped** and Stage B launches
  immediately -- e.g. reviewing run 2 right after building your own
  template while reviewing run 1 won't make you build a second one for
  yourself.
- If you don't have one yet (and `--ecg-source` isn't `batch`), Stage A
  runs first: you correct a short window, it builds the template and
  applies it to every available run, then Stage B launches for the run you
  asked for. **This happens even if another RA already built a template
  for this same subject** -- templates are never shared or borrowed
  between RAs; see "Every RA builds their own ECG template" below for why.
  Only applies when `ecg` is actually part of what you're reviewing this
  session (e.g. `--channels eda` alone needs no template at all).
- `--ecg-source batch` skips the template requirement entirely and always
  uses the batch pipeline's own peak detection (for clean in-lab studies).
- `--force-template` rebuilds even if you already have your own template.
- `--run` is prompted for if omitted and more than one run is available;
  auto-selected if only one is.
- All the Stage A (`--template-run`/`--template-start`/`--template-window`)
  and Stage B (`--channels`) flags described below work the same way here.

### `--stage`: running just one step explicitly

```
annotate_env\Scripts\python.exe physio_review.py --box-path <path> --local-path <local> --subject 001 --initials hlu --stage 1
annotate_env\Scripts\python.exe physio_review.py --box-path <path> --local-path <local> --subject 001 --run 1 --initials hlu --stage 2 --channels eda
```

- `--stage 1`: run **only** Stage A (build/apply the template), then stop
  -- no Stage B, no `--run` needed (it applies to every available run).
- `--stage 2`: run **only** Stage B for `--channels` (defaults to every
  configured channel present in the file except PPG, which is always
  reviewed on its own with `--channels ppg`). If `ecg` is among those
  channels, uses **your own** existing template -- erroring clearly if you
  don't have one, never another RA's and never a silent fallback to batch
  peaks -- but, unlike the default combined flow, **never builds one
  interactively**. A Step 2 session is meant to be a standalone review, not
  something that can unexpectedly drop you into template-building
  mid-session; if you hit that error, run `--stage 1` first (or drop
  `--stage` for the combined flow), or pass `--ecg-source batch` if this
  subject's ECG genuinely doesn't need scanner-grade correction.
- `--stage 3`: reconcile two already-saved RAs' (or an RA's vs. your own
  gold-standard) annotations for `--run` -- requires
  `--compare-initials A,B`. See "Stage 3: reconciling two reviewers" below.
- Omit `--stage` entirely for the original combined behavior described
  above. The GUI always passes the step you pick: Step 1, 2 or 3 runs as
  `--stage 1`, `--stage 2` or `--stage 3`.

Verified via `test_physio_review.py` (interactive `raw.plot()` mocked to a
no-op, so this exercises the entire real pipeline non-interactively):
a fresh subject correctly triggers Stage A once and skips it on a second
run of the same subject; `--force-template`, `--ecg-source batch`,
`--stage 1`, and `--stage 2` all override that decision correctly.

## Working from Box: the local-copy lifecycle

Matching the existing notebooks' own workflow, any real (non-synthetic)
session copies this subject's files from Box to a local folder you
specify, does all its actual work against that local copy, pushes back
whatever's new or changed when you're done, confirms it landed, and then
deletes the local copy automatically -- so subject data never lingers on
an RA's laptop. This is **on by default** for any `--box-path` session.

You'll be asked for (or can pass directly):

```
annotate_env\Scripts\python.exe physio_review.py --box-path <path> --local-path C:\Users\you\PhysioWorking --subject 001 --run 1 --initials hlu
```

- `--local-path`: a folder you reuse across sessions/subjects (like
  `--box-path`, prompted for if omitted). Each session copies
  `sub-XXX/` into it, works there, then removes just that `sub-XXX/`
  subfolder when done -- the `--local-path` folder itself is never deleted.
- physioProcess's own files (`*_physio.tsv.gz`, `*_physio.json`) are **never**
  pushed back, not even from a leftover local copy: after a reprocess, an old
  local copy would otherwise overwrite Box's current file (2026-09-26).
- Only NEW or CHANGED files get pushed back to Box (the annotations JSON,
  QRS template JSON/CSV, reconciled JSON, etc.) -- the untouched
  `physio.tsv.gz`/sidecar never gets re-copied, since nothing rewrites it.
- Each pushed-back file is confirmed by checking it landed on the Box
  side with a fresh modification time. This confirms the write to the Box
  Drive mount succeeded -- not that Box's cloud servers have finished
  syncing it, which isn't something plain file I/O can check.
- **If any file's push can't be confirmed, NOTHING is deleted.** You'll
  see a clear warning naming exactly which file(s) and where your local
  copy still is, so you can check and copy them to Box yourself.
- `--keep-local`: skip the cleanup step even after a fully successful
  push, for troubleshooting.
- `--no-local-copy`: skip this whole lifecycle and work directly against
  `--box-path` (e.g. if it already points at a local folder you manage
  yourself, or for scripting/testing).

`processing_log.csv` is unaffected by any of this -- it's always read and
written directly on Box (see below), never through the local copy.

## Stage A: building an ECG QRS template (optional)

**Run this via `physio_review.py --stage 1`** (examples below) -- not the
standalone `qrs_template_stage.py`, which shares the exact same underlying
logic (`run_stage_a()` in `qrs_template.py`) but skips `physio_review.py`'s
`processing_log.csv` logging and the local-copy-to-Box lifecycle entirely.
Keep `qrs_template_stage.py` for troubleshooting Stage A in isolation, not
day-to-day use.

Pass `--ecg-source batch` for clean in-lab ECG that doesn't need this at
all -- Stage B works fine without it. Otherwise use it for messy/
in-scanner data, where the wavelet + template cross-correlation approach
is the one that's actually held up. **`physio_review.py` requires a
template for ECG by default** (see "Every RA builds their own ECG
template" below) -- `--ecg-source batch` is how you deliberately opt out
of that requirement for a given subject.

**Works per SUBJECT, not per run**: you build the template once, from a
window in ONE run, and it's automatically applied to every run that exists
for that subject -- you don't need to (and shouldn't) build a separate
template for run 2. **It does NOT work per RA, though**: it's still
per-(subject, RA) -- see below.

### Every RA builds their own ECG template

Templates are never shared or borrowed between RAs, even when only one
exists and even though the file naming (`<file_stem>_ecg_corrected_qrs_
<initials>.json`) would technically allow it. If you review ECG for a
subject someone else has already built a template for, `physio_review.py`
still has you build your own -- it will never silently seed your review
from theirs.

This is a deliberate reliability choice, not just a safety net: seeding
every RA's review from the same automated starting point would bias
independent corrections toward agreement (or a shared blind spot) before
anyone's even started, which undermines what Stage 3's reconciliation is
meant to measure -- genuinely independent review, template-building
included, not just independent point-editing from a common seed. In
practice this means a missing template is never a silent fallback to
batch peaks: `--stage 2` errors clearly if your own template is missing,
and the combined flow builds one for you instead. `--ecg-source batch`
remains the explicit, deliberate way to skip the template requirement
altogether for a subject whose ECG doesn't need it.

```
annotate_env\Scripts\python.exe physio_review.py --synthetic --initials hlu --stage 1
```

For a real subject, either give both files directly:

```
annotate_env\Scripts\python.exe physio_review.py --input-run1 <path1> --input-run2 <path2> --initials hlu --stage 1
```

...or let it find both runs itself from a Box derivatives folder + subject
(prompted for whichever you omit, same as Stage B) -- working from the
local copy `--local-path` sets up (see "Working from Box" above):

```
annotate_env\Scripts\python.exe physio_review.py --box-path C:\Users\you\Box\DATA\Processed\physioProcessing\derivatives --local-path C:\Users\you\PhysioWorking --subject 001 --initials hlu --stage 1
```

If only one run exists yet (e.g. run 2 hasn't been collected), that's fine
-- it just builds and applies the template to whichever run(s) it finds.

This opens a 20-second window (starting at 0s by default, from run 1 unless
you pass `--template-run 2`) pre-marked with the batch pipeline's automated
peaks. Correct the peaks in that window -- add missed ones, delete false
ones -- then close the plot window.

**Closing the window doesn't commit anything by itself.** It builds the
QRS template from whatever peaks are present, then a **preview window**
pops up showing the individual corrected beats (thin gray lines) overlaid
with the resulting averaged template (bold black) -- the same kind of
beat-segment visualization `step1_qrs_template.ipynb` already used, but
shown *before* anything is applied or saved, with an optional **Comments**
box (recorded in `processing_log.csv`'s Notes) and two buttons:

- **Approve Template** -- proceeds to cross-correlate this template
  against every available run and save the results.
- **Try a Different Window** -- aborts cleanly, telling you to start Step 1
  again with a later **Template window start** (e.g. 20, then 40), the
  same idea as step1's "shift 20s" button.

(Since the window starts pre-seeded with the batch pipeline's automated
peaks, simply closing the correction window without editing anything is
*not* the same as rejecting it -- you'd still see a preview built from
those pre-seeded peaks and would need to click "Try a Different Window"
to reject it.)

On approving, it applies that SAME template to the wavelet-filtered full
signal of **every available run**, snapping to local maxima and applying
refractory-period cleanup for each -- saving one
`<file_stem>_ecg_corrected_qrs_<initials>.json` +
`<file_stem>_qrs_template_<initials>.csv` PER RUN, with JSON *content* in
the exact schema `step1_qrs_template.ipynb` already produces. Unlike
step1's notebook, **your initials are in the filename** -- so if a second
RA independently builds a template for the same subject, it's saved
alongside the first rather than overwriting it.

On a 90-second synthetic test run, this recovered 100% of the true peaks
with ~1% false positives. On the two-run transfer test (template built from
one 60s synthetic run, applied to a second), recall/precision stayed above
98% on both runs (see `test_qrs_template.py`).

**Heart rate range.** Detection holds at 60-180 bpm and when heart rate
changes during a run (`test_audit_fixes.py`). Until 2026-09-23 a fixed
600 ms minimum spacing between beats made anything above 100 bpm lose about
half its beats. On messy in-scanner ECG the current rule can instead leave
some extra markers to delete; see the constants block in `qrs_template.py`.

## Stage B: full multi-signal review

**Run this via `physio_review.py --stage 2`** (examples below) -- not the
standalone `physio_annotate.py`, which shares the exact same underlying
logic (`run_stage_b()` in `annotation_io.py`) but skips `processing_log.csv`
logging (it never calls `append_processing_log_entry()` at all) and the
local-copy-to-Box lifecycle entirely. Keep `physio_annotate.py` for
troubleshooting Stage B in isolation, not day-to-day use.

### Try it with synthetic demo data (no real files needed)

```
annotate_env\Scripts\python.exe physio_review.py --synthetic --run 1 --initials hlu --stage 2 --ecg-source batch
```

(`--ecg-source batch` here just skips the "build your own ECG template
first" requirement so this quick demo doesn't need a Stage A detour --
see "Every RA builds their own ECG template" above. Drop it once you're
trying this against real messy/in-scanner ECG. `--duration N` sets each
synthetic run's length in seconds: default 90 in `physio_review.py` and
`qrs_template_stage.py`, 60 in `physio_annotate.py`.)

(`--initials` takes whatever you type literally — it's not a keyword, just your initials for the output filename. For tests, use clearly fake initials such as `xyz`, so they never match a real RA's.)

This fabricates 60 seconds of plausible ECG/RSP/PPG/EDA (via neurokit2) plus
two EMG placeholder channels (corrugator, zygomatic) that are **literally
zero throughout** on purpose, so nothing about them looks like real muscle
activity worth
reviewing yet. It opens MNE's interactive multi-channel browser with:

- **ECG / PPG**: pre-marked with the automated peak detections
  (mirroring today's step1/step2 workflow, which also starts from automated
  peaks). Use annotation mode to add missed peaks or delete false ones.
  To add a peak, drag a small box over it: it's saved at the highest point
  inside the box, as the old notebook tool did. If the box cuts off the top
  of the wave, it moves up to that wave's top, at most 15 ms; it never jumps
  to a taller wave nearby, so a short R wave next to a taller MHD wave stays
  where you put it (Option C, 2026-09-26). PPG is reviewed in
  a session of its own (see "Reviewing PPG" below).
- **RSP, EDA, EMG (corrugator / zygomatic), SBP, DBP, and finger
  temperature**: drag to mark bad/artifact spans (segment-style, not
  point-style). RAs don't correct breath peaks, SCR peaks or beat-level BP
  values, only flag stretches that look untrustworthy. (RSP was a peak
  channel until 2026-09-25.) On files from the new physioProcess pipeline,
  **RSP, EDA, SBP and DBP start with the machine's own bad stretches
  already marked** (see "Machine-flagged bad stretches" below); EMG and
  finger temperature, which the machine doesn't check, start blank.
  (SBP/DBP/finger temperature only appear once `physioBatch.py` actually
  produces those columns -- see "Adding SBP/DBP/finger temperature" below.)
- **Guide rows** (read-only): PPG under SBP/DBP, EDA under RSP, RSP under
  EDA, when that channel isn't also being scored; PPG under ECG only on
  request (`--ecg-ppg-guide`): see "Guide rows" below.
- **Event** (only shown if the file has an `event` column -- see
  `physioProcess`'s `processEvents()`/`processPPG()`-adjacent fix): a
  **read-only reference track**, always shown alongside whatever you chose
  to review via `--channels`, even if you didn't ask for it. Values are
  task-phase codes (0 = outside the task; 20/30/40/50/65 = baseline/read/
  imagery/recovery/ratings) -- when it reads 0, the messiness in the other
  signals there is expected (pre/post-task padding), not something to
  correct. It's never seeded with annotations, never itself annotatable,
  and never appears in the saved output JSON -- there's nothing to correct
  on it, it's just context. Every time the code changes, you'll also see a
  **labeled vertical marker showing the actual numeric code** (e.g. `20`,
  `30`) spanning the full plot -- these are read-only MNE event markers,
  not annotations, so they can't be accidentally dragged, deleted, or
  merged; press `e` to toggle them off if they're in the way.

### Machine-flagged bad stretches (new-pipeline files)

Files from the new physioProcess pipeline carry the machine's own quality
check in their sidecar (`MachineQC`). A fresh RSP, EDA, SBP or DBP review
**starts with those bad stretches already marked**, exactly as the machine
gave them (like ECG and PPG starting from the automated peaks). Check each
one: delete any that are fine, and drag to add any it missed. The session
summary counts your changes against the machine's marks, and the saved file
records what the machine flagged (`seed`).

- When the machine flagged **nearly all of a run** (e.g. a disconnected
  sensor; sub-001's RSP), the banner says so and asks you to KEEP the flag
  unless you're sure: the viewer stretches every channel to fill its row,
  so a dead sensor's tiny noise can look like a real signal. The summary
  repeats the warning if you remove most of that flag. When the flags are
  only missing CareTaker readings (SBP/DBP on a run CareTaker barely
  covers), the banner instead says how much of the run CareTaker covers:
  keep those flags (the stretches are blank) and review the recorded part.
- A flagged stretch that stops within 5 s of the end of the file (the
  machine checks in whole windows) is noted; extend it if the end is bad too.
- **What the machine checks, per channel** (physioProcess maintainer,
  2026-09-26), so you know what it can miss:
  - **RSP:** a flat-signal test in whole 2-s windows, so the last 0-2 s of a
    file is never tested. A dead belt shows as the whole run flagged; on
    screen it can look like fast, uneven breathing (from run 8 the
    breathing filter keeps fast breathing at full size). From run 9 the
    test only flags a belt that isn't recording (std < 5e-5 V), so breath
    pauses and holds are no longer flagged. **Mark RSP bad wherever you
    can't see where each breath peaks** (HLU, 2026-10-05: "it's not possible
    to reliably place a peak that one can't see"). That covers a flat line,
    whether the belt stopped or a breath was held, and clipped tops cut flat
    at the recording's maximum. Breaths whose peaks show are not bad,
    however odd they look (sighs, very shallow breathing, pauses between
    breaths). This replaces the 2026-09-26 rule "mark only where the belt
    stopped; if you can see breaths, it's not bad", which left clipping and
    holds alone.
  - **EDA:** out of range (0.05-60 µS) or too steep (10 µS/s) on a smoothed
    copy, plus 0.5 s each side. It catches sustained faults only, and misses
    artifacts shorter than about 5 s: **mark brief EDA artifacts yourself.**
  - **SBP/DBP:** a missing reading (no CareTaker value within the 2-s merge
    window), out of range, or SBP less than 10 above DBP (marks both). Most
    flagged stretches are coverage gaps and run edges. **Short no-reading
    stretches (under 2 s) are gaps between CareTaker readings; the trace is
    blank there. Keep them.** They appear when two readings are more than 4 s
    apart, because each reading fills only 2 s each side; the banner counts
    them (HLU, 2026-10-02). From run 9 also
    **no usable finger pulse** (`ppg_dropout`): the pressure inside a PPG
    device dropout, where CareTaker has no pulse to measure it from, until
    the first reading after the pulse returns (HLU, 2026-09-27).
- **Why the machine flagged them:** on SBP/DBP in files from run 8 on, the
  banner and the session summary (even when you changed nothing) give the
  machine's reasons, e.g. "no CareTaker reading (12)"; the saved `seed`
  keeps them (`by_reason`).
- **A channel with no data at all** in a run (PPG, SBP and DBP on the 7 runs
  with no CareTaker data; the file has the column, but it's empty) isn't
  reviewed: the all-channels default leaves it out with a note, and naming
  it (`--channels sbp`, or SBP in the form) is refused with the reason.
- Older files (no machine check), or a machine check whose sample count
  doesn't match the data file, start blank, with a note.

### Guide rows

A guide row is another channel shown **read-only** right under the one you
score, **exactly as stored**: it's never marked or saved, and it only
appears when that channel isn't being scored in the same session. On by
default, **except the PPG under ECG, which is off unless asked for**
(below); untick "Show the guide channel (read-only)" or pass
`--no-ppg-guide` (the flag's name is historical: it turns off every guide
row).

| Scoring | Guide row | What it's for (the console line) |
|---|---|---|
| ECG (Steps 1-3) | PPG, **only with `--ecg-ppg-guide`** (or the ticked box) | the pulse, for counting beats only (below). Without it the console says: "PPG (guide): not shown under ECG by default -- the pulse's delay after the R peak differs from run to run, so it can mislead R-peak placement. To use it for counting beats only, tick the guide box or pass --ecg-ppg-guide." |
| SBP / DBP | PPG | "the finger pulse the blood pressure comes from. Where there's no usable pulse, the pressure values aren't real." On files whose machine check marks BP inside pulse dropouts (the `ppg_dropout` reason, "no usable finger pulse", from run 9), it adds: "The machine pre-marks those stretches; check the edges." On runs where CareTaker pulse samples were lost in transfer (PPG's `counter_loss` reason, from run 10), a second line gives the exception and where the stretches are: "PPG (guide): in 2 stretch(es) (13.8 s in all, at 100.0-106.9 s, 400.0-406.9 s) CareTaker pulse samples were lost in transfer, so the PPG guide can't be used to check BP there. The BP readings in those stretches were still measured by the device; the loss affected only the pulse copy sent to the app. Keep them, except where the machine marked a finger-pulse dropout." (HLU, 2026-10-03; the times are examples). |
| RSP | EDA | "a rise 1-3 s after a big breath shows the breath was real. No rise means nothing. Never mark RSP bad because of EDA." |
| EDA | RSP | "a skin-conductance rise 1-3 s after a big breath is a real response to the breath, not bad signal. Mark EDA bad only where the EDA itself is unusable." |

Whenever a guide row is shown, the console adds: "Guide rows are context
only: mark only what you see on the scored row." (The viewer shades a bad
stretch across every row, the guide row included.) There's no guide under
PPG: the PPG is clean enough to judge on its own, and the in-scanner ECG is
messy (HLU, 2026-09-27). SBP/DBP come from the same CareTaker device and
clock as the PPG; EDA and RSP are both recorded by the Biopac, on one clock.
Why BP is real where pulse samples were lost in transfer: the CareTaker wrist
unit (CT5) calculates BP on board from its own complete pulse signal and sends
the readings separately from the copy of the pulse that the tablet app saves
(CareTaker Platform User Manual Rev 6, §7.7 and §8.10). The readings carry on
through every loss at about one per beat (physioProcess maintainer's check,
2026-10-03). A device zero run inside such a stretch is still a real dropout,
and stays pre-marked.

A guide row is **not shown**, and the console says why, when its channel is
missing or empty in the file, when the machine check marks most (over 90%)
of that run's channel as unusable (a dead belt would otherwise be stretched
to look like fast breathing), or, for PPG, **on files processed before the
PPG timing fix** (their PPG is multi-humped and unreliable). Each saved
entry of a channel that can have a guide records whether it was on screen
(`guide_shown`).

#### The PPG guide under ECG

**Off by default in Steps 1-3 (HLU, 2026-10-03); shown only on request**
(`--ecg-ppg-guide`, or ticking the form's box with ECG selected).
- **Why it's off:** each run's CareTaker timeline (PPG and BP together) sits
  a run-specific constant offset from the Biopac's (physioProcess plan
  item 7a). From R peak to pulse it's about 0.43 s in sub-001 r1 but about
  1.09 s in sub-085 r1, more than a beat.
- **The danger:** placing an R peak "just before the pulse" can land in the
  wrong beat. That happens most in noisy ECG, where the right mark is an ECG
  bad stretch, and it would feed pulse timing into HRV and bias PAT.
- **What it's still good for:** the beat-to-beat rhythm is sound, so an RA
  who asks for it can use it to count beats, i.e. to notice a missing or
  extra beat, **never to decide where an R peak goes**. Device dropouts look
  like low, smooth arcs with no clear pulse.

The saved ECG entry records whether the guide was on screen (`guide_shown`).
On the command line, a session that scores ECG together with SBP/DBP still
shows the PPG row for SBP/DBP. The console then says "PPG (guide): shown
here for SBP/DBP only -- don't use it to place ECG R peaks (...)", and the
ECG entry records `guide_shown` true. In Step 1 the template is built from
the ECG alone, identical with or without the guide.

### Reviewing PPG

PPG peaks are reviewed in a **session of their own** (`--channels ppg`, or
PPG in the form), and only on files processed after the PPG timing fix; an
older file is refused with a plain message (lab staff: `--allow-old-ppg`).
A run with no usable PPG (no CareTaker pulse recorded, or none covering the
task: its ppg column is empty) is refused too; there's nothing to review.
A PPG review saved before 2026-09-26 (it has no `provenance`, and was made
on the old PPG timing) is never reopened onto a current file: PPG starts
fresh, the old review is kept as `ppg_points_legacy`, and Step 3 refuses it.
There is no Step 1 for PPG. Pulse onsets ("feet") are not marked here:
physioProcess computes them later from your corrected peaks.

- **Mark bad stretches with the `bad_ppg` label** (2026-09-27) wherever there
  is no usable pulse: a device dropout, or a stretch too messy to find the
  peaks. Pick `bad_ppg` in the annotation toolbar and drag, as for `bad_rsp`.
  The stretch is drawn across the rows; it has no row of its own.
  - **The machine's dropouts start already marked** (its widened stretches,
    0.6 s past each dropout). Delete a marked stretch if the pulse there is
    fine, or shrink it.
  - **Bad stretches never keep peaks.** Automated peaks inside a pre-marked
    stretch aren't loaded, which includes the ones physioProcess sometimes
    fills in by interpolation. Peaks inside a stretch you mark are left out
    when you save; the summary tells you how many. So you don't delete peaks
    in a bad stretch by hand. If you delete a stretch, add any real peaks
    back yourself.
  - **The gap check knows about bad stretches.** A long gap across a bad
    stretch isn't flagged. A long gap with no bad stretch still is: mark it
    bad if the signal there is unusable, or add the missed peak.
  - Saved as `bad_ppg` (bad stretches, `[start, end)`) next to `ppg`, and
    reconciled in Step 3 like any other bad-stretch channel.
- **To add a peak, drag the box over the whole top of the pulse** (both
  humps, if it has two, or the whole flat top). The tool saves your mark at
  the highest point inside the box, so you don't pick the spot yourself
  (HLU, 2026-10-04). Keep the box off the smaller wave after the notch and
  off the neighbouring pulses. The console repeats this in every PPG session.
- **Don't move an existing mark from one hump to the other.** CareTaker's
  pulse often has two systolic humps or a flat top. On some beats, and in
  some runs beat to beat, the machine's mark sits on the lower one.
  physioProcess finds the pulse's upstroke from your mark, and that's the
  same under either hump, so moving it gains nothing. Marks on a slope are
  flagged ("not on a crest of a pulse"): move those onto a crest.
- **Delete markers inside dropouts**: low, smooth arcs with no clear pulse.
  The pulse is smoothed, so where the CareTaker lost the signal you see slow,
  smooth curves, never a flat line. The machine's dropout stretches
  include 0.6 s on each side, where the pulse is often real: keep a marker
  there if it sits on a clear pulse.
- **Never add a peak where no pulse is visible**, even if the rhythm says
  one should be there.
- **To move a peak, delete it and then add it again** (see "Correcting peaks"
  below), rather than dragging over it.
- Until physioProcess adds its per-sample dropout mask (request R1), files
  say so at the start ("no per-sample PPG dropout mask yet"), and dropouts
  aren't flagged for you: look for the low arcs yourself.
- A long gap may be a real long beat, e.g. after a premature beat
  [UNVERIFIED: need PMID lookup for "premature beats can give weak or absent
  pulses"]; look before changing anything.

The session summary lists PPG markers not on a crest of a pulse (i.e. on a
slope; either crest of a double-crested or flat-topped pulse passes, since
2026-10-04), markers inside a machine-flagged dropout, and places where two markers seem to sit
on one pulse (in Step 3, that last one blocks saving until fixed). It shows
three times per line; when there are more, **the terminal lists them all**,
with the machine's dropout stretches (the viewer doesn't draw them).

### Correcting peaks: the click-drag mechanics that matter

MNE's annotation tool has a real quirk that's easy to trip over, so please
read this before doing a real review. Press **`a`** first to enter
annotation mode (a toolbar appears at the bottom showing the active
label, e.g. `peak_ecg`).

**To delete a peak:** click-and-drag a small box around it, click the
resulting selection to make sure it's the active one, then **right-click**
to remove it. (A bare right-click on a peak marker on its own usually
won't work -- a true point annotation is drawn with essentially zero
width, so there's nothing for a single right-click to land on. The
click-drag step is what gives you something clickable.)

**To mark a bad ECG or PPG stretch** (Steps 2 and 3): pick `bad_ecg` or
`bad_ppg` in the toolbar's label list and drag over it. Peaks inside are left
out when you save, and a gap across it isn't flagged. **Switch back to the
peak label** (`peak_ecg` / `peak_ppg`) before adding peaks, or a small drag
makes a tiny bad stretch instead. The viewer picks label colors itself, so
check the label name, not the colour. To page through a run, use
**Shift + →**, which moves one full window; plain → moves a quarter.

**To add a peak:** click-and-drag a small, precise box right on the beat
you want to mark, covering its top. On save it's placed at the highest point
inside your box; if the box cut off the top, it moves up to the top, at most
15 ms. It never jumps to a taller wave outside your box, so on a short R wave
next to a taller MHD wave or artifact, keep the box on the R wave.

**The one thing to avoid: don't click-drag to add a new peak on top of or
overlapping an existing one you want to keep.** MNE silently *merges*
overlapping same-label markers into one combined marker instead of
keeping both -- so if you're trying to correct a peak's position by
dragging near the original, you'll likely lose one of the two silently,
with no error. **If you want to move/replace a peak, delete the old one
first, then add the new one separately**, rather than dragging over top
of it.

As a safety net, the tool checks for this automatically: if it detects a
peak marker with a suspiciously large width (a sign two peaks got merged
into one), it prints a warning naming the channel and approximate time
when you close the window, so you can go back and check. Don't rely on
this catching everything, though -- following the "delete first, then
add" rule above is the reliable way to avoid it in the first place.

### Interval regularity check

When you close the window, ECG and PPG also get checked for unusually
irregular gaps between their final, corrected peaks. Any flagged spot is
listed in the session summary window and printed in the terminal, naming
the channel and approximate time.

- **ECG and PPG** use the artifact criterion of Berntson, Quigley, Jang, &
  Boysen (1990, *Psychophysiology*, PMID 2274622). It is built from this
  recording's own beat-to-beat differences:
  - criterion = (MED + MAD) / 2, where
  - MED = 3.32 × QD (QD = half the interquartile range of successive differences), and
  - MAD = (median beat − 2.9 × QD) / 3.

  A flagged long gap is reported as a possible missed peak if splitting it
  in half fits its neighbors. A flagged short gap is reported as a possible
  extra peak if adding it to its shorter neighbor fits. Anything else is
  reported as a "sudden change (check this beat)". A real premature
  (ectopic) beat usually looks like that, and should be kept.
- More than 5 flags on one channel are shown in the summary as one count
  line (all of them are still listed in the terminal).

For example:

```
NOTE: unusually irregular gap(s) between peaks detected. This can be
a real physiological irregularity (an ectopic beat, a natural pause)
or a missed/spurious peak worth a second look:
  - ecg: around 71.82s, gap is short (possible extra/spurious peak) (410ms vs. this channel's typical 985ms).
```

A long gap suggests a missed peak; a short one suggests an extra/spurious
one -- worth a quick look, but not necessarily wrong: a real ectopic beat
or a natural pause can look the same way. (EDA is never checked here --
it's segment-mode now, not point-mode, so there's no peak-to-peak
interval to check in the first place; see HANDOFF.md §25 for why EDA
moved to segment-mode.)

### Reading real values: units and the crosshair

Each channel's scale bar (and the numbers you see) are in real units,
traced back to the actual hardware/source and confirmed directly: **ECG in
mV, EDA in µS, RSP in V** (the respiration belt's output voltage; not a
calibrated volume), **PPG in AU** (CareTaker's uncalibrated pulse waveform,
not Biopac at all), and **SBP/DBP in mmHg**. On files from the new
physioProcess pipeline the labels come from the file's own sidecar (its
per-column `Units`), and the banner also prints each reviewed channel's
`Description` from the file. None of this changes the underlying data --
only what the label correctly says it means.

To read the **exact value at any point**, not just the scale bar's
overall range: press **`x`** to toggle crosshair mode, then hover over
any trace -- the exact value (in that channel's real unit) and time
appear live in the status bar at the bottom of the window. This works for
every channel, including the read-only `event` track.

Close the plot window when done. A **Session summary** window then shows
how many peaks/segments you added, removed or moved and any spots worth
checking (possible merged peaks; marks with a label that won't be saved; for
PPG, markers not on a crest of a pulse or inside a machine-flagged dropout;
unusual gaps between peaks), plus an optional
**Comments** box and an **"I've finished reviewing this for this run"**
checkbox. Choose:

- **Save** (the default) -- saves
  `demo_output/synthetic_demo_annotations_<initials>.json` (or the real
  file's `..._annotations_<initials>.json`). Each reviewed channel's status
  is stored as `"complete"` or `"in_progress"` from the checkbox. If
  nothing changed and the status is the same as last time, nothing is
  rewritten. (A peak moved by even a few milliseconds counts as a change;
  until 2026-09-26 moves of up to 50 ms were ignored and not saved.)
- **Discard this session's changes** -- asks you to confirm, then saves
  nothing. Your previous saved version (if any) is left exactly as it was.
- **Go back to the viewer** -- reopens the viewer with your current,
  unsaved marks still in place, e.g. to check a flagged spot. Closing the
  summary window with its X does the same.

Step 3 (reconciliation) ends with the same summary window.

The Google tracking spreadsheet has a dropdown per measure (e.g. `ecg`) in
your row: **"in progress"** when you start that channel, **"finished"** when
you finish it (2026-09-27; it used to ask for dates). At the start of every
real-data session the terminal reminds you to set the channel to "in
progress"; at the end, a **TRACKING SPREADSHEET** reminder says to set it to
"finished" if you ticked "finished", or to leave it "in progress". This is
separate from `processing_log.csv`, which the tool fills in itself.

**You don't have to finish a run in one sitting.** Save partway through
(leave "finished" unticked) to keep your current corrections. If you run the
*same* command again for the *same* run and initials, it picks up right
where you left off -- your deletions, added peaks, and marked bad segments
from last time are restored, instead of the automated peaks being
re-seeded from scratch. You'll see a `Resuming your previous session...`
message confirming what was restored. (This is keyed on the saved
`..._annotations_<initials>.json` file, so a different RA -- different
initials -- always starts fresh, and each RA's progress on a run is
independent.)

**"Finished" means the whole file.** Tick it only when you've checked the
whole run, including the stretches before and after the task (the viewer
always shows the whole file). A finished channel with nothing marked in a
stretch tells the analysis that stretch was reviewed and is clean.

### Showing only some channels

The default view loads every configured channel present in the file
(up to nine, once SBP/DBP/finger temperature are collected), which can
feel busy. Pass `--channels` with a comma-separated subset to declutter (channels the
file doesn't have, or whose data are empty, are left out of the default with
a note, and refused if named), e.g.:

```
annotate_env\Scripts\python.exe physio_review.py --synthetic --run 1 --initials hlu --stage 2 --channels ecg,eda --ecg-source batch
```

PPG can't be combined with other channels (`--channels ppg` on its own):
with ECG it would draw both channels' markers across both rows, and the
viewer's label picker silently resets to the alphabetically first label, so
a drag meant for one channel could be saved as the other's.

MNE's own browser toolbar may also support toggling channel visibility at
runtime (there's a channel-selection control in there), but that hasn't
been tested with this tool, so `--channels` is the reliable way to control
this for now.

### Using it on a real file

Either point straight at the file (`--input-run1` works standalone --
`--input-run2` is only needed if you also want Stage A/`--run 2` available
in the same invocation):

```
annotate_env\Scripts\python.exe physio_review.py --input-run1 <path_to>\sub-XXX_ses-runY_task-sdi_physio.tsv.gz --initials hlu --stage 2
```

...or let it build the path from a Box derivatives folder + subject + run
(matching the notebooks' `sub-XXX/ses-runY/beh/...` convention), working
from the local copy `--local-path` sets up (see "Working from Box" above):

```
annotate_env\Scripts\python.exe physio_review.py --box-path C:\Users\you\Box\DATA\Processed\physioProcessing\derivatives --local-path C:\Users\you\PhysioWorking --subject 001 --run 1 --initials hlu --stage 2
```

If you omit `--box-path`, `--subject`, and/or `--run` (and don't pass
`--input-run1`/`--input-run2`), it **prompts for whichever of those you
didn't provide** — same idea as `--initials` already prompting when
omitted. `--subject` accepts `5`, `005`, or `sub-005` interchangeably;
`--run` accepts `1` or `run1`. Same flags work identically for `--stage 1`.

If ECG is among the channels reviewed (the default) and you haven't built
your own template for this file yet, both commands above will error
clearly rather than silently using batch peaks -- see "Every RA builds
their own ECG template" above. Run `--stage 1` first, drop `--stage 2` for
the combined flow (which builds one for you automatically), or add
`--ecg-source batch` if this file's ECG doesn't need Stage A at all.

Saves `<same folder>\sub-XXX_ses-runY_task-sdi_physio_annotations_<initials>.json`
alongside the input file. Running it again with a different RA's initials on
the same file produces a second, separate JSON rather than overwriting the
first — so more than one RA can independently review the same recording.

### `processing_log.csv`: the shared tracking sheet

When you use `--box-path`/`--subject` (not `--synthetic`) with
`physio_review.py`, it writes to `<box_path>/processing_log.csv` — the same
shared tracking file `step1_qrs_template.ipynb`/`step2_physio_correction.ipynb`
already use, alongside the `sub-XXX` folders. **This is one of the concrete
things you lose by running the standalone `physio_annotate.py`/
`qrs_template_stage.py` instead of `physio_review.py`: neither of them
writes to `processing_log.csv` at all.** Three things happen automatically:

- **A `claimed` row** the moment your subject/box-path/initials are known,
  before any review happens — same as the notebooks' claim-on-submit
  behavior.
- **A `created QRS template` row per run** whenever Stage A actually builds
  a template (not when it's skipped because one already exists).
- **A row after Stage B** that was saved, listing whichever channels you
  actually reviewed that session plus the finished status, e.g.
  `ecg, eda reviewed -- finished` or `ecg reviewed -- not finished`.
  This is deliberately more general than the notebooks' fixed
  `R peaks inspected and corrected` text, since this tool can review more
  than just ECG in one sitting. Nothing is logged for a discarded session,
  or for one with no changes to save.

The optional comment for each row (except `claimed`) comes from the
**Comments** box in the template-preview window (Step 1) or the session
summary window (Steps 2 and 3), and is saved in the same `Notes` column the
notebooks use. It is no longer a terminal prompt: closing the terminal at
such a prompt used to stop the session before its files were copied back to
Box.

**If a session ends early** (Ctrl-C, an error, a stop you chose), whatever
was already saved is still copied back to Box and confirmed. If a leftover
local copy from an earlier session still holds files that never reached
Box, the next session lists them and asks before doing anything. Answer
`y` to copy them to Box first; anything else stops without touching them.

**Which subject to do next** comes from the Google tracking spreadsheet: RAs
are assigned files, and the tool no longer suggests a subject in the form or
at the command-line prompt (2026-09-27).

If `processing_log.csv` is open in Excel when a write is attempted, you'll
see a clear message telling you so — your actual annotation output is
already saved separately by that point and is unaffected either way.

Every write also refreshes a local, read-only mirror --
`processing_log_READONLY_SNAPSHOT.csv`, in the `physioCorrection/` folder
(one level up from `multisignal_annotation/`) -- matching exactly the
existing notebooks' own convention (same filename, same location), so
there's one shared, most-recent snapshot regardless of which tool wrote
last. The real log always lives on Box; this is just a convenience copy
that goes stale the moment anyone else processes data, hence the name.
An **installed** copy (a clone of the public repo, such as the shared lab
install in `C:\Users\Public\Downloads\pebbl`) writes the snapshot to the
user's own home folder instead, because the log holds participant IDs, RA
initials and comments, and the folder above a shared install is readable by
everyone (2026-10-05; `processing_log.default_snapshot_dir()`).

### Choosing the ECG seed source: `--ecg-source`

`physio_review.py`'s `--ecg-source` supports:

- `auto` (default): use **your own** QRS template (see "Every RA builds
  their own ECG template" above) if you have one. With `--stage 2`, errors
  clearly if you don't -- never a silent fallback to batch peaks, and
  never another RA's template even if it's the only one that exists.
  Without `--stage` (the combined flow), builds you one instead of
  erroring.
- `batch`: skip the template requirement entirely and always use the
  batch pipeline's own peaks, regardless of `--stage` or whether a
  template (yours or anyone else's) exists.

```
annotate_env\Scripts\python.exe physio_review.py --synthetic --run 1 --initials hlu --stage 2 --ecg-source batch
```

Standalone `physio_annotate.py` still has two things `physio_review.py`
deliberately doesn't: `--ecg-source template` alongside the old
lenient `auto` (use ANY single existing template regardless of whose
initials it's under, and only error if there are several with no match)
and `--template-json <path>` to force a specific one. Those exist there
for troubleshooting a single stage in isolation, not for day-to-day
review -- see "Stage B" above for why `physio_annotate.py` is the
exception, not the rule.

## Stage 3: reconciling two reviewers

Once two people have each saved their own `_annotations_<initials>.json`
for the same subject/run (two RAs, or an RA vs. your own gold-standard
pass), `--stage 3` compares them and gives you an interactive view to
reconcile the disagreements -- or, for training, lets an RA see exactly
where their annotations diverge from a gold standard, with the actual
signal as context.

```
annotate_env\Scripts\python.exe physio_review.py --box-path C:\Users\you\Box\DATA\Processed\physioProcessing\derivatives --local-path C:\Users\you\PhysioWorking --subject 001 --run 1 --stage 3 --compare-initials hlu,xyz --channels ecg
```

What happens: for each shared channel (channels only one side reviewed
are skipped -- nothing to compare there), peaks within about 50ms of each
other (scaled to this run's sampling rate, not a fixed sample count) are
treated as agreed; everything else is unique to one side. All
three categories are seeded into the same interactive viewer used for
Stage B, each with its own label (e.g. `peak_ecg_agree`,
`peak_ecg_only_hlu`, `peak_ecg_only_ask`) -- MNE colors each label
differently automatically, so agreement/disagreement is visually obvious
without any special UI. Bad segments are compared by overlap instead of
point distance. Two segments are "agreed" (shown as their combined span)
only if they overlap by at least half their combined span; smaller overlaps
are shown as two separate one-side marks. To add a mark neither reviewer
made, use the `..._added` label (e.g. `peak_ecg_added`, `bad_eda_added`),
which is always available.

Step 3 also asks for **your own initials as the reconciler**. They are
saved in the reconciled file (`reconciled_by`) and in `processing_log.csv`.

**Your job as reviewer**: delete any mark you don't believe, add anything
neither side caught, and leave agreed marks alone -- the exact same
click/drag/delete workflow as Stage B (see "Correcting peaks" above for
the mechanics). On close, whatever's left -- regardless of which of the
three categories it started in -- becomes the final peak/segment list. An
agreed point is saved at whichever reviewer's sample sits higher on the
signal, exactly (since 2026-09-26; previously the midpoint of the two picks,
re-snapped). Marks you draw follow the same rule as Step 2 (the highest point
in your box, climbing at most 15 ms if the box cut off the top); marks you
leave alone keep their sample.

Checks specific to Step 3:
- a warning when either reviewer's review isn't marked finished;
- for PPG, **two final markers on one pulse block saving** until one is
  removed (CareTaker's pulse has two humps, and two reviewers can mark the
  same beat on different humps; both marks would otherwise be saved);
- both reviews must have been made on the current file's signal (same
  sample count and content hash); otherwise Step 3 stops and says so;
- a shared channel whose data are empty in this file is skipped in the
  default (refused if named).

Saves `<same folder>\..._annotations_reconciled.json` -- no initials in
the name, since it's a consensus product of two reviewers rather than one
person's individual pass. The file **merges per channel**: reconciling
PPG today keeps the ECG reconciled last week (until 2026-09-26 each Step 3
save replaced the whole file). Each channel records who reconciled it and
from whom. A previous version is copied to `backups/` first. Step 3 always
starts again from the two reviewers' current files, so an earlier
reconciliation of the same channel isn't shown (the console says so);
saving replaces it.

The file also records **agreement statistics** between the two reviewers
under `"agreement"`, and a one-line summary goes into the log's Notes:
- **peak channels:** percent of peaks agreed, how many only one reviewer
  marked, and the median timing difference;
- **segment channels:** Cohen's kappa over samples, and the seconds each
  reviewer marked.

`--channels` defaults to whatever both sides reviewed in common, leaving
out PPG (reconciled on its own) and any channel one of the reviews made the
old way (e.g. RSP marked as breath peaks before 2026-09-25); request a
specific subset the same way as Stage B, but it'll error clearly if you ask
for a channel one side never reviewed or reviewed the old way.

For the training/gold-standard use case, just put your own initials as
one side: `--compare-initials <trainee>,<your_own_initials>`.

## RA certification practice (`practice_certification.py`)

A simulated practice participant, **sub-990**, with **runs 1 and 2** (180 s
each, every channel the real files have, built with neurokit2's
simulators), deliberate mistakes in the machine's marks, real-but-odd
"traps", and an **answer key** (the gold standard). Key version 7
(2026-10-05); older practice folders and keys can't be scored by this
version, so make a new folder.

1. **Lab staff make it once**, in a `practice` folder next to the Box
   `derivatives` folder (the form looks for it there):
   ```
   annotate_env\Scripts\python.exe practice_certification.py make --out-dir "C:\Users\<you>\Box\DATA\Processed\physioProcessing\practice"
   ```
   This writes `sub-990/ses-run1|2/beh/` (same layout as the real data) and
   the key, `practice_key.pebbl`, in the practice folder itself. The key is
   lightly encoded (zlib + base64 behind a header) so it isn't readable at a
   glance; it's not security. Without `--seed` a fresh random seed is used
   and stored in the key; `--key-copy <path>` also writes a plain-JSON copy
   (keep that outside the practice folder).
2. **RAs practice from the form:** tick **Practice for certification
   (practice participant 990)**. That locks the subject to 990, turns off
   Step 3, and points the session at `<derivatives>/../practice` (the form
   still remembers the derivatives path). Command line: `--practice` with
   `--box-path <practice folder> --subject 990`. RAs do **Step 1** (their
   own ECG template; not scored), then **Step 2** for each assigned run and
   channel. Practice sessions log to the practice folder's own
   `processing_log.csv`, never to the real log's local snapshot.
3. **Scoring is automatic.** After every saved Step 2 review, PEBBL scores
   it against the key, prints the report (so the session log keeps it), and
   shows it in a window. **Compare with the answer key** opens the viewer
   with the RA's marks next to the key's (`..._agree`, `..._only_key`,
   `..._only_<initials>`; view only, nothing saved). The RA resumes, fixes
   and saves again until PASS. Scoring is by **outcome** (where the final
   marks are), not by which seed marks were changed, so it works whatever
   the ECG seed source.
4. **Lab staff can rescore any saved file:**
   ```
   annotate_env\Scripts\python.exe practice_certification.py score --key <practice folder>\practice_key.pebbl --annotations <practice folder>\sub-990\ses-run2\beh\sub-990_ses-run2_task-sdi_annotations_<initials>.json [--channels ecg,eda]
   ```
   The run comes from the file name (or `--run`).

**What's planted** (per run; times differ between runs)

| Channel | To fix | Traps: real, keep or leave unmarked | Machine marks it starts with |
|---|---|---|---|
| ECG | scanner-like signal: tall T waves (taller in one stretch), spikes, 4 s of heavy noise to mark `bad_ecg`; the `ecg_peaks` seed also has 3 missed beats, 2 marks on T waves and 1 misplaced | a premature ventricular beat (wide, oddly shaped, large T wave pointing the other way) with its compensatory pause | — |
| PPG | 3 missed pulses, 2 marks on the dicrotic wave, 1 on the upslope, a device dropout (mark `bad_ppg`) with a spurious seed in it; the weak, odd pulse after the premature beat (mark `bad_ppg`; the machine doesn't) | 4 two-humped pulses seeded on the other hump (either hump passes) | the dropout (`zero_run`) |
| RSP | 2 stretches where the belt stopped (flat); a held breath (flat); clipped breath tops (deep breaths cut flat at the run's maximum) | a sigh (one whole breath, twice as deep), shallow breathing (eases in and out) | flags the held breath, the clipping and one dead stretch, misses the other, falsely flags the shallow breathing |
| EDA | electrode lift, motion spikes, stuck sensor (flat) | 5 real skin conductance responses | flags the lift and spikes, misses the flat stretch, falsely flags one SCR |
| SBP / DBP | leading blank, recalibration (held flat), implausible spike, a no-reading blank, a 1.2-s gap, BP inside the PPG dropout | — | flags all but the spike; falsely flags 6 s of clean BP |

The PPG looks like CareTaker's: an early and a main systolic hump, a
dicrotic wave, the pipeline's 0.5-8 Hz band-pass, and a dropout drawn the
way the device makes one (a run of zeros smoothed into a low arc). EMG and
finger temperature aren't in the practice file, as in the real data.

**Pass criteria** (`PASS_CRITERIA`, stored in the key)
- **Peaks (ECG, PPG):** every true beat has a mark within 20 ms (ECG) or
  30 ms (PPG), no extra marks; for a two-humped pulse either crest counts;
  beats inside a real artifact the RA marked bad are excused.
- **Bad stretches (`bad_ecg`, `bad_ppg`, RSP, EDA, SBP, DBP):**
  - every artifact at least 80% covered (including the one the machine missed);
  - no trap, and no machine false flag, more than 25% still covered;
  - at most 2 s marked outside the artifacts, with 1 s of leeway around each
    (2 s around the PPG dropout, `DROPOUT_LEEWAY_SEC`; HLU, 2026-10-05).

## The session log

Everything the tool prints during a session (the equivalent command, banners,
notes, warnings and any error with its details) is also written to a log
file **next to your annotation file, with the same
name ending `.log`**. For example, `sub-001_ses-run1_task-sdi_annotations_hlu.json`
has `sub-001_ses-run1_task-sdi_annotations_hlu.log` beside it.

- **One log per person per subject-run.** Each Step 1, 2 or 3 session you run
  on that run is added to the end, under a header (date, time, initials,
  step), and closed with a line saying how it ended: saved, finished or in
  progress, discarded, or ended early. An reconciler's Step 3 sessions go in
  the reconciler's own log.
- **Step 1** writes to the log of every run it builds templates for.
- **It's written as the session goes,** so a crash or a closed window still
  leaves the log. It's pushed to Box with your annotation file (or with the
  next push, if a session was cut short).
- **The save dialog is logged too:** what it showed (changes, warnings) and
  what you chose (Save, Discard or Go back; finished or not; your comment).
  These lines go only into the log, not the terminal.
- **The console shows exactly what it showed before.**

## The saved annotation file (output JSON shape)

```json
{
  "schema_version": 2,
  "initials": "hlu",
  "timestamp": "2026-09-26 16:03:00",
  "source_file": "...",
  "sampling_rate": 1000,
  "channels": {
    "ecg": {"mode": "point", "indices": [123, 456, ...], "snap_window_samples": 50, "status": "complete",
            "reviewed_span": [0, 742091], "guide_shown": true,
            "provenance": {"number_of_samples": 742091, "pipeline_commit": "715c747...", "signal_sha256": "9ac7..."},
            "edit_history": [{"saved_at": "2026-09-26 16:03:00", "initials": "hlu", "started_from": "QRS template (...)",
                              "added": [5180], "removed": [5010], "moved": [[9100, 9112]]}]},
    "rsp": {"mode": "segment", "bad_segments": [[0, 742000]], "status": "complete",
            "value_range": {"min": -4.0e-05, "mean": 1.1e-06, "max": 3.9e-05, "unit": "V"},
            "seed": {"source": "MachineQC", "rule": "...", "segments": [[0, 742000]], "pipeline_commit": "715c747..."},
            "reviewed_span": [0, 742091],
            "provenance": {...}, "edit_history": [...]},
    "rsp_points_legacy": {"mode": "point", "indices": [...], "status": "complete"},
    "eda": {"mode": "segment", "bad_segments": [[2000, 2500], ...], "status": "complete",
            "value_range": {"min": 0.21, "mean": 0.82, "max": 0.89, "unit": "µS"}}
  }
}
```

**If a saved file doesn't fit** (2026-09-26), the tool stops with a plain message
naming the file and changes nothing. That happens when the file is unreadable or
damaged, when it was saved at a different sampling rate, or when it has marks
outside the data file or not whole sample numbers. A save over a damaged file
writes the session to `..._RESCUED_<date>.json` instead, so no work is lost.
`check_reprocess_alignment.py` reports unreadable RA files as
`UNREADABLE_RA_FILE`; before, it skipped them.

Every reader relies only on `mode` / `indices` / `bad_segments` (sample
positions at `sampling_rate`; segments are `[start, end)`), which are
unchanged. The top-level `coordinate_space` states that convention in the
file itself: tsv row indices at `sampling_rate`, row 0 the file's first row,
point indices are rows, `bad_segments` are `[start, end)` (the same as the
sidecar's `MachineQC`). Version 2 (2026-09-26) adds, per channel:
- `status`: exactly `"complete"` (the RA or reconciler ticked "finished";
  the only FINAL value, and the only one physioProcess ingests) or
  `"in_progress"` (saved partway);
- `reviewed_span`: `[0, N)` in tsv rows, what `status` covers: **the whole
  file**, pads before and after the task included (HLU, 2026-09-26). On a
  `"complete"` channel, nothing marked in a stretch means "reviewed, clean",
  never "not reviewed". An entry without it (saved before 2026-09-26) covers
  the same whole file; it was added without a version change;
- `value_range` (segment channels): the channel's real min/mean/max;
- `seed` (segment channels started from the machine check): what the machine
  flagged, so RA-versus-machine agreement can be computed later, with the
  machine's reasons (`by_reason`) when the sidecar gives them (SBP/DBP from
  run 8: `no_coverage` / `out_of_range` / `ordering`);
- `provenance`: the sample count and a hash of the channel's values in the
  `.tsv.gz`, so a review is never applied to a changed signal. The hash is
  byte-exact and shared with physioProcess (see `provenance.py`): SHA-256
  over each row's raw field bytes plus `\n`, line endings removed, an empty
  field hashing as nothing. `signal_sha256_def` names that definition
  (`"physioprocess-column-sha256-v1"`; an entry without it is v1);
- `edit_history`: one entry per save, listing what that session added,
  removed and moved, and what it started from;
- `guide_shown` (ECG, SBP, DBP, RSP, EDA): true if the channel's guide row
  was on screen in any saved session (each `edit_history` entry also records
  it for that session);
- `<channel>_points_legacy`: an earlier review made in an older mode (e.g.
  RSP breath peaks), kept rather than lost.
- `bad_ecg` (with `ecg`, since 2026-09-27): the ECG's bad stretches, as a
  segment entry. It starts blank (no machine check for ECG). Mark bad only
  where R peaks can't be located; correct them wherever you can. `ecg` never
  has peaks inside them; its provenance is the ecg column's
  (`"column": "ecg"`). Not in Step 1.
- `bad_ppg` (with `ppg`, since 2026-09-27): the PPG's bad stretches, machine
  and RA combined, as a segment entry. `ppg` never has peaks inside them. Its
  `provenance` is the ppg column's (`"column": "ppg"`). A PPG edit history
  record may list `removed_in_bad_stretches` and
  `seeds_left_out_in_bad_stretches`.

A reconciled file (`..._annotations_reconciled.json`, the file physioProcess
reads) has the same per-channel `mode` / `indices` / `bad_segments` /
`status` / `reviewed_span` / `provenance`, plus per channel:
- `reconciled_by` and `reconciled_from` (also at the top level, for the
  latest session);
- `reviewer_origin`: the samples (or spans) that started as agreed, only
  reviewer A's, and only reviewer B's; a final mark in none of them was added
  by the reconciler;
- `edit_history`: one entry for the reconciliation session;
- for ECG, `guide_shown_by` ({reconciler, reviewer A, reviewer B}) and
  `guide_shown` (true if the guide was on for any of them).

Its `agreement` statistics are at the top level, keyed by channel. It
carries no `seed` or `value_range`.

## Adding SBP/DBP/finger temperature

`channel_config.py` already defines `sbp`, `dbp`, and `finger_temperature`
as segment-mode channels (like EMG -- RAs flag bad/untrustworthy stretches,
no peak-picking; see "Stage B" above for why), so the tool is ready for
them the moment a `physio.tsv.gz` file actually has those columns.
Nothing else changes: `build_raw()`/`compute_scalings()`/etc. already only
show whatever's actually present in a given file's columns (the same
mechanism that lets a study without PPG/EDA work fine today), so existing
files without these three columns are completely unaffected.

- **SBP/DBP**: CareTaker already measures these in the current study.
  `physioBatch.py` now saves them (a small addition to its column-building
  code -- described in `HANDOFF.md` §25, applied by the user themselves
  per this tool's standing rule of never modifying `physioProcess`
  directly), confirmed working end-to-end against real `sub-085` data:
  both runs' `physio.tsv.gz`/JSON sidecar include `sbp`/`dbp` with
  plausible real blood pressure values and no missing data.
- **Finger temperature**: not collected in the current study. Added ahead
  of a possible future in-person study that would. MNE's `"temperature"`
  channel type already has correct built-in defaults (Celsius) -- see
  `channel_config.py`'s comment for what to change if the eventual real
  device reports Fahrenheit instead.

## Migrating old `step2_physio_correction.ipynb` output (`migrate_legacy_json.py`)

If a subject was already corrected via the old notebook before this tool
existed, its `*_ecg_final.json` (written by the `systole` package's
`Editor.save()`) is a different, incompatible shape -- `{"ecg": {"valid",
"corrected_peaks", "bad_segments"}}`, one top-level key, no `"initials"`
field at all. `migrate_legacy_json.py` converts these into this tool's own
`*_annotations_<initials>.json` shape above, so old RA work isn't orphaned:

```
annotate_env\Scripts\python.exe migrate_legacy_json.py --box-path C:\Users\you\Box\DATA\Processed\physioProcessing\derivatives
annotate_env\Scripts\python.exe migrate_legacy_json.py --box-path C:\Users\you\Box\DATA\Processed\physioProcessing\derivatives --apply
```

**Read-only by default**: without `--apply`, it only prints what it would
write, for review -- nothing is written or touched, including the original
`*_ecg_final.json` files (never deleted or modified, even with `--apply`).

**Who corrected it?** The old format never recorded initials, so this
script looks it up from `processing_log.csv`'s own history: the LAST `"R
peaks inspected and corrected"` row (the notebook's fixed Step text) for
that subject/run -- the most recent one, in case a subject was revisited
across sessions by more than one person. If no matching row exists, that
file is skipped with a clear reason rather than guessed at; pass
`--initials-for <subject>:<run>:<initials>` (repeatable) to resolve it
manually.

**`bad_segments` is deliberately dropped, loudly.** Across this study's
entire real derivatives tree, exactly one `*_ecg_final.json` (sub-001,
run 1) was ever found to have a non-empty `bad_segments` -- confirmed via
a one-off PowerShell scan, not assumed. Building real ECG segment-marking
support into this tool (ECG is point-mode only -- EDA/EMG/SBP/DBP/finger
temperature are the segment-mode channels) wasn't judged worth it for a
single historical case. The script still reports any non-empty
`bad_segments` it finds (so nothing is silently lost) -- note it manually
(e.g. in `processing_log.csv`'s `Notes` column) if it matters for that
one subject.

### `--upgrade`: the migrated reviews in the current format (2026-09-26)

After run 8, the 13 migrated ECG reviews are rewritten in the current
format, so they resume like any other review and physioProcess can use them
once finished and reconciled:

```
annotate_env\Scripts\python.exe migrate_legacy_json.py --box-path <derivatives> --upgrade [--initials-for ...]
annotate_env\Scripts\python.exe migrate_legacy_json.py --box-path <derivatives> --upgrade --apply [--initials-for ...]
```

For each notebook `*_ecg_final.json`, the existing `*_annotations_<initials>.json`
(the same initials as the 2026-09-20 migration; pass the same `--initials-for`)
gets a new `ecg` entry:
- **The peaks, unchanged**, with `status: "in_progress"`. An RA reopens the
  review, checks it and ticks "finished".
- **`reviewed_span`, and `provenance` of the run-8 file.** The ECG timeline is
  identical to the file the peaks were made on.
- **`guide_shown: false`.** The notebook showed ECG only.
- **An `edit_history` record against the notebook QRS template** the step-2
  notebook started from, when that template's stored initials match the
  reviewer. Otherwise `added`/`removed`/`moved` are `null` (unknown).
- **`migrated_from`:** the notebook file and its SHA-256, any dropped
  `bad_segments`, the checks below, and every snapped peak's original position.

Two checks on the run-8 ECG:
- **Transfer check.** The notebook's Step 1 snapped template peaks to the
  ECG maximum, then its automatic correction filled in missed beats at
  interpolated positions, which were never snapped. So on an unchanged
  signal most template peaks, not all, are a local maximum within ±2
  samples: 91-98.5% on the real files. A changed signal would move nearly
  all of them. A run under 90% is flagged for a person to look at, and
  `--exclude SUBJECT:RUN` leaves a run out.
- **3-sample snap.** The notebook's editor stored edited peaks 1-2 samples
  before the R-wave maximum, while unedited peaks sat exactly on it. So a
  peak within 3 samples of a local maximum is moved onto it, with the
  original position kept, and the edit history is computed after that.
  Peaks further away, in messy stretches, stay exactly as the reviewer left
  them. A review is skipped, not overwritten, in three cases:
- it was already re-saved in the tool, or upgraded;
- its peaks differ from the notebook's;
- no migrated file exists under those initials.

The existing file is backed up first, and the notebook files are never
touched. Notebook QRS templates aren't converted: RAs redo Step 1 in the tool
(HLU's decision).

## Display scaling

MNE's built-in `scalings='auto'` assumes real physiological units (volts,
etc.), which our data isn't in, and caused real signals (RSP/PPG especially)
to visually overflow into neighboring channel rows. `compute_scalings()` in
`physio_io.py` instead computes a scale per channel type from the actual
loaded data, and is fully data-driven rather than hardcoded per channel:

- Normally, each channel type gets a robust (2.5th-97.5th percentile) scale
  computed from its own data, **NaN-safe** -- a real test file's PPG
  channel turned out to be ~91% NaN (valid for the first ~67s, then the
  sensor apparently failed for the rest of an ~12-minute run), which
  originally produced a `NaN` scale and would have broken the plot.
- If a channel's own data is *degenerate* (all-NaN, or ~zero variance like
  today's literal-zero EMG placeholder), it instead borrows the median
  scale of the file's other, non-degenerate channels, so it still renders
  at a sane row height instead of a NaN/zero scale breaking the plot or
  dwarfing everything else.
- Because this checks the actual data rather than a fixed per-channel flag,
  a channel that's a placeholder **today** (zero EMG) but carries real
  signal in a **future** file (real corrugator/zygomatic EMG, or a study
  that collects PPG) automatically gets its own properly auto-fit scale
  the moment real data replaces the placeholder -- no config change needed.

Checked in `test_smoke.py` (steps 3b/3c/3d: flatline-stays-flat, NaN-safety,
and real-data-gets-its-own-scale) and `test_real_data.py`/`test_column_subsets.py`.

## Files

- `channel_config.py` — per-channel MNE type + annotation mode (point vs.
  segment vs. `"none"`) + which existing auto-peak column (if any) seeds
  it. Also `REFERENCE_CHANNELS` (currently just `"event"`): read-only
  tracks always shown alongside whatever's selected via `--channels`, kept
  separate from the main `CHANNELS` dict so they're never user-selectable;
  `GUIDE_CHANNELS` (the read-only guide rows: PPG under ECG or SBP/DBP,
  EDA under RSP, RSP under EDA, with their console text in `GUIDE_TEXTS`;
  `GUIDE_OPT_IN` makes the PPG under ECG opt-in);
  and `select_channels()`, the one
  place the "PPG on its own" rule lives.
- `ppg_checks.py` — the PPG session-summary checks: markers not on a crest
  of a pulse, markers inside machine-flagged dropouts, and two markers on
  one pulse.
- `provenance.py` — each reviewed channel's provenance (sample count and the
  byte-exact content hash shared with physioProcess), the changed-signal
  check, and the new-pipeline test for PPG review.
- `test_sidecar_build.py`, `test_ppg_build.py`, `test_provenance.py` —
  known-answer tests for the 2026-09-26 build: the Step 3 merge and snap
  fixes, small moves saved, distinct start times, machine-flagged pre-fill on
  sub-001 (exactly its 12 SBP spans and 1 RSP span) and its guards, RSP's
  legacy key, the PPG guide (shown exactly as stored; the Step 1 template
  byte-identical with and without it), unit labels from the sidecar, the PPG
  gate and checks, provenance and edit history, the reprocess PPG check, and
  the practice PPG; the content hash reproduces physioProcess's published
  sub-001 test vectors.
- `physio_io.py` — reads a real `physio.tsv.gz`, or generates synthetic
  demo data; builds the `mne.RawArray`; computes display scalings
  (NaN-safe, data-driven); resolves a real file's path from
  `--box-path`/`--subject`/`--run`(s), prompting for whichever is missing.
- `annotation_io.py` — seeds automated peaks as initial annotations;
  exports the RA's final annotations to JSON (including local-max
  snapping for point channels); saves per-RA-initials JSON files;
  resolves which ECG peak source Stage B should use;
  `combine_with_reference_channels()`, which adds any available
  `REFERENCE_CHANNELS` to whatever's being reviewed; `run_stage_b()`, the
  shared Stage B session logic used by both `physio_annotate.py` and
  `physio_review.py`.
- `physio_annotate.py` — Stage B standalone CLI entry point.
- `qrs_template.py` — Stage A's algorithm (wavelet filter, template
  building, cross-correlation + refinement, output saving) AND
  `run_stage_a()`, the shared interactive-session logic used by both
  `qrs_template_stage.py` and `physio_review.py`.
- `qrs_template_stage.py` — Stage A standalone CLI entry point: works per
  SUBJECT (builds the template from one run's window, applies it to every
  available run), not per file.
- `template_preview_gui.py` — the PyQt6 + embedded-matplotlib dialog shown
  after building a candidate template, before it's applied/saved: overlaid
  raw beats + the averaged template, with Approve / Try a Different Window
  buttons.
- `physio_review.py` — the unified CLI (see above); orchestrates
  `run_stage_a()`/`run_stage_b()`/`run_stage_c()` without duplicating any
  of their logic; shows `simple_gui.py`'s form when launched with no
  arguments at all; owns the local-copy-lifecycle wiring (see "Working
  from Box" above) via `box_sync.py`.
- `simple_gui.py` — the minimal PyQt6 form shown by `physio_review.py` when
  it's run with zero arguments: Step 1/2/3 choice, Box path + local
  working folder (each with a Browse... button), subject, run + a
  single-channel picker (Steps 2/3; locked to ECG and disabled for Step
  1), initials (Steps 1/2) or two reviewers' initials to compare (Step
  3), or a synthetic-demo-data checkbox. Any CLI flag at all skips this
  and uses the normal prompt-driven CLI instead.
- `processing_log.py` — `append_processing_log_entry()` (writes directly
  to `<box_path>/processing_log.csv`, matching the existing notebooks'
  exact column shape, then mirrors it to a local read-only snapshot) and
  `find_next_available_subject()` (the autodetected-next-subject
  suggestion). See "`processing_log.csv`" above.
- `reconcile.py` — Stage 3's diffing (`match_points()`/`match_segments()`)
  and orchestration (`build_reconciliation_raw()`, `export_reconciled()`,
  `save_reconciled_json()`, `run_stage_c()`). See "Stage 3" above.
- `box_sync.py` — the local-copy lifecycle: `copy_subject_tree_to_local()`,
  `find_changed_files()`, `push_changed_files_to_box()`,
  `cleanup_local_copy()`. See "Working from Box" above.
- `migrate_legacy_json.py` — one-time converter from
  `step2_physio_correction.ipynb`'s old `*_ecg_final.json` output to this
  tool's `*_annotations_<initials>.json` shape, read-only unless `--apply`
  is passed. See "Migrating old step2_physio_correction.ipynb output" above.
- `template_preview_gui.py` — the PyQt6 + embedded-matplotlib dialog shown
  after building a candidate template, before it's applied/saved: overlaid
  raw beats + the averaged template, with Approve / Try a Different Window
  buttons.
- `test_smoke.py` — non-interactive test of Stage B's plumbing (data load →
  Raw → seed → simulated edits → export → JSON round-trip), the
  EMG-flatline display-scaling regression check, NaN-safety/real-data
  forward-compatibility checks for `compute_scalings()`, resume support,
  the backup-before-overwrite safety net, the merge-warning (fixed and
  adaptive thresholds), and `check_interval_regularity()`.
- `test_qrs_template.py` — non-interactive test of Stage A's algorithm
  (wavelet filter → template building → cross-correlation → refractory
  cleanup → JSON schema → `--ecg-source` resolution logic → template
  transfer from one run to another → multi-RA template disambiguation →
  the nonzero-`--template-start` fix), using ground truth peaks as a
  stand-in for RA correction.
- `test_path_resolution.py` — tests `resolve_real_input_path()` (Stage B,
  single run) and `resolve_subject_run_paths()` (Stage A, both runs):
  direct-path passthrough, path construction + subject/run normalization,
  interactive prompting for missing pieces, and missing-file errors.
- `test_physio_review.py` — tests the unified script's orchestration with
  `raw.plot()` mocked to a no-op, exercising the entire real pipeline
  non-interactively: Stage A run vs. skipped vs. forced vs. bypassed via
  `--ecg-source batch`, the template-window rejection prompt, all three
  `--stage` values, resume support, full `processing_log.csv` integration
  (including comments), `parse_args_or_show_gui()`'s GUI-to-CLI argv
  construction for every stage, and the full local-copy lifecycle
  (success + `--keep-local` + a simulated confirmation failure).
- `test_column_subsets.py` — confirms the pipeline works when a file has
  only a subset of the usual columns (e.g. no PPG/EDA, or ECG-only), an
  unrecognized extra column, or real (non-placeholder) EMG data.
- `test_real_data.py` — runs the same non-interactive checks against every
  real `test_input/**/*_physio.tsv.gz` file found locally (recursive, so a
  whole-folder Box copy like `test_input/sub-XXX/ses-runY/beh/...` works;
  skips cleanly if none are present -- this repo doesn't ship real subject
  data). This is what caught the PPG-NaN scaling bug above -- synthetic
  data never would have -- and later confirmed a `processPPG()` fix (PPG
  resampled/time-aligned via `merge_asof` instead of index-merged at its
  native ~31 Hz) by showing 0% NaN and a physiologically plausible peak
  rate on a freshly-reprocessed subject.
- `test_simple_gui.py` — tests `ReviewLauncherDialog`'s field
  validation/synthetic-toggle/Step-1-2-3-toggle (including the
  channel-locked-to-ECG-for-Step-1 and initials-vs-compare-initials-for-
  Step-3 behaviors)/submit logic directly (no `.exec()` call, so no
  blocking modal event loop) -- verifies the form's decisions, not what
  it looks like.
- `session_summary_gui.py` — the PyQt6 "Session summary" window shown when
  the Step 2/Step 3 viewer closes: change counts, spots to check, comment,
  finished checkbox, and Save / Discard / Go back. `count_changes()` is plain
  logic; tests replace the dialog with `headless_decision()`.
- `check_reprocess_alignment.py` — run after reprocessing through
  `physioBatch.py`, BEFORE copying the new files to Box. For every run with
  saved RA work, it checks that the new `.tsv.gz` still lines up sample for
  sample with the one RAs annotated:
  - the same length, and the ECG at zero lag in three windows;
  - no saved index past the new file's end;
  - separately, the PPG: identical, or at zero lag within +/- 10 s, since
    CareTaker timing can change while the ECG doesn't (`NEEDS_PPG_CHECK`
    when saved work includes PPG, including when the new file has no usable
    PPG);
  - SBP/DBP, when saved work includes them: they must be identical, since
    CareTaker's vitals have their own timing fit (`NEEDS_CARETAKER_CHECK`);
  - every saved review that records its provenance (since 2026-09-26): the
    new file's sample count and content hash for that channel must match, or
    the tool will refuse to reopen that review after the copy
    (`REVIEW_INVALIDATED`, naming the channel). Don't copy that run until it's
    resolved: keep the old file, or have the RA redo that channel.

  Saved annotations are sample positions, so a shifted file would silently
  misplace them. It is read-only and exits 1 if anything needs attention.
  It ignores set-aside (legacy) entries, which the tool never loads.
  Usage: `annotate_env\Scripts\python.exe check_reprocess_alignment.py
  --old-root <Box derivatives> --new-root <data\processed\bids> --report
  alignment_report.csv`. `--all` checks every run in both trees, not only
  runs with RA work; `--max-lag-sec` (default 2) is how far the ECG lag
  search looks each way.
  `test_check_reprocess_alignment.py` checks it against known answers:
  identical, a 250-sample shift, a shorter file, a missing file, an
  unrelated recording (must be UNCLEAR), and a noisy copy (must be ALIGNED).
- `retire_channels.py` (lab staff) — when a reprocess changes a channel an
  RA already reviewed, Step 2 refuses to reopen that review ("made on a
  different version of the file"). This command sets chosen channels aside in
  one saved review file so they can be reviewed afresh:
  - each entry moves to a legacy key (e.g. `sbp` -> `sbp_segments_legacy`)
    with a `retired` record (when, and why);
  - companions go with their parent (`ppg` takes `bad_ppg`, `ecg` takes
    `bad_ecg`);
  - the file is first copied to `backups/` next to it.

  It's read-only unless `--apply`. Run it after the Box copy and before the
  RA's next session, with no session open for that subject:
  `annotate_env\Scripts\python.exe retire_channels.py --file <...>_annotations_hlu.json --channels sbp,dbp --reason "run 10 changed sbp/dbp" --apply`.
  Tested in `test_safety_fixes.py`.
- `practice_certification.py` — makes the RA certification practice
  participant (sub-990, runs 1 and 2) and its encoded answer key, scores a
  saved review against it, and shows an RA's marks next to the key (see "RA
  certification practice" above).
- `practice_report_gui.py` — the practice score window shown after each
  saved practice review, with its "Compare with the answer key" button.
- `test_practice_certification.py` — checks the practice participant
  against known answers:
  - both runs load through the tool's own reader; the encoded key round-trips;
  - a perfect review passes every channel in both runs, and the untouched
    seed fails, naming the machine's misses and false flags;
  - deleting a trap, or marking real physiology, fails that channel; either
    crest of a two-humped pulse passes; beats inside a bad stretch are excused;
  - the command-line scorer finds the run from the file name;
  - the interval check flags the planted premature beat.
  The form's practice box is tested in `test_simple_gui.py` (item 16), and
  the scored practice session in `test_physio_review.py` (item 19).
- `test_audit_fixes.py` — regression tests for the 2026-09-23 audit fixes:
  template detection at 60–180 bpm and across heart-rate changes; click-drag
  peaks resolved correctly in Steps 1 and 3; an interrupted session still
  pushing to Box; the leftover-local-copy check; the session summary's
  discard/back/skip/status behavior; and the tracker reminder wording.
- `test_template_preview_gui.py` — tests `TemplatePreviewDialog`'s
  approve/reject logic and that it builds its plot without crashing on
  realistic, ragged-length, and empty beat lists (no `.exec()` call).
- `test_reference_channels.py` — tests the `REFERENCE_CHANNELS`/
  `combine_with_reference_channels()` mechanism: a no-op when `event` is
  absent, added automatically (even under a restricted `--channels`
  request) when present, rendered as a `stim`-type channel, never seeded
  with annotations, never in the exported output; plus
  `compute_event_transitions()`'s labeled-marker logic and its wiring
  into `run_stage_b()`.
- `test_processing_log.py` — tests `append_processing_log_entry()` (column
  shape, subject normalization, the local read-only snapshot mirror) and
  `find_next_available_subject()`.
- `test_reconcile.py` — tests `match_points()`/`match_segments()`,
  `build_reconciliation_raw()`, `export_reconciled()`, and a full
  `--stage 3` CLI end-to-end scenario with a deliberate RA disagreement.
- `test_box_sync.py` — tests the local-copy lifecycle's functions in
  isolation: mirroring, change detection, push-back with confirmation
  (including a simulated failure), and cleanup scoped to just one subject.
- `test_migrate_legacy_json.py` — tests the legacy-JSON converter: filename
  parsing, `processing_log.csv`-based initials lookup (including preferring
  the most recent of several matching rows, and an `--initials-for`
  override), a clean skip when no log row resolves, dry-run vs. `--apply`,
  and confirming the original `*_ecg_final.json` is never touched.

Run any test suite with `annotate_env\Scripts\python.exe <test_file>.py`.

## Known open questions (not addressed yet, by design)

- The "second pass" — re-running `processECG()`/`processRSA()`/`processHRV()`/
  a future `processEMG()` using these corrected JSONs instead of
  auto-detecting — is not built here. physioProcess plans it as its
  "Phase D": it reads the reconciled files, never re-detects, and computes
  PPG pulse feet (onsets) from the corrected peaks
  (`multisignal-annotation_ppg-plan_20260925-2327.md` §5).
- Stage A's refractory-period cleanup (`enforce_refractory_period()`) is a
  simpler stand-in for systole's `correct_peaks()` (no ectopic-beat
  interpolation) -- fine on the synthetic test (100% recall, ~99%
  precision), but not validated against real messy/in-scanner ECG yet
  (the one real file tried so far had a good, clean ECG channel).

## Display event markers below the signal in Stage B (DONE)

Reference track showing when the task was actually running vs. baseline/
rest, so an RA reviewing a signal knows not to worry about messiness
outside the task window (`event == 0`). Built as `REFERENCE_CHANNELS` in
`channel_config.py` + `combine_with_reference_channels()` in
`annotation_io.py` -- see the "Event" bullet above and HANDOFF.md §9 for
the full design (why a `stim`-type channel rather than background
`mne.Annotations`, and the upstream `physioProcess` patch that produces the
`event` column in the first place). Verified against real reprocessed data
(`test_input/sub-085`, both runs) in `test_real_data.py` and
`test_reference_channels.py`.

Later extended with **labeled event-code markers** (the actual numeric
value at each transition, via MNE's own read-only `events=` mechanism
rather than another annotation) -- see the "Event" bullet above and
HANDOFF.md §13.

## Wishlist

Recorded in detail (precedent from the existing notebooks, design options,
dependencies) in HANDOFF.md §10. Current status:

- **DONE** (§12): logging to `processing_log.csv` (claim entries, one row
  per Stage A run, a generalized `"<channels> reviewed"` row for Stage B),
  an open-ended comment prompt recorded in the same rows, suggesting the
  next unclaimed subject (console message + GUI placeholder text), and
  exposing `--template-start` in the GUI for Step 1.
- **DONE** (§13): labeled numeric event-code markers, described above.
- **DONE, in a different form than originally asked** (§14): rather than a
  live-updating R-R interval track inside the viewer (investigated and
  judged too fragile -- would need to hook `mne-qt-browser`'s undocumented
  internals), every point channel with a genuinely periodic rhythm
  (ecg/rsp/ppg, not eda) now gets an automatic check when you close Stage
  B: any gap between exported peaks that's a lot longer or shorter than
  that channel's own typical gap prints a warning naming the channel and
  approximate time -- the same by-hand R-R analysis already used a few
  times during development, now automatic. See "Interval regularity
  check" below and HANDOFF.md §14.
