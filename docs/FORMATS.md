# PEBBL file formats

What PEBBL reads, what it writes, and what it checks along the way. This is
the contract between PEBBL and the pipeline that prepares its input
(physioProcess, in the lab that built PEBBL), and between PEBBL and whatever
analyses its output. Each item names the function that reads or writes it,
so the code is the final word when this page and the code disagree.

Conventions used throughout:

- **Samples, not seconds.** Positions are rows of the recording, counted from
  0, at the recording's `SamplingFrequency`. A point is one row; a stretch is
  `[start, end)`, i.e. `start` is included and `end` is not.
- **Times** in saved files are local time with no time zone.
- **"Absent"** below means the key is missing; what happens when it is the
  wrong type is stated only where the code checks it (see "What PEBBL does not
  check" at the end).

## 1. Input: the recording, `*_physio.tsv.gz`

Read by `physio_io.load_physio_tsv`.

- A gzip-compressed, tab-separated text file with **no header row**. Row *i*
  is sample *i*. Line endings may be `\n` or `\r\n`.
- Column names come from the sidecar's `Columns` list, in order (section 2).
  A sidecar without `Columns` means `ecg, rsp, ppg, eda, ecg_peaks,
  rsp_peaks, ppg_peaks, eda_peaks` (`physio_io.REAL_PHYSIO_COLUMNS`). Either
  way, the number of names must equal the number of fields in the file's
  rows, or PEBBL stops: it never guesses which column is which.
- A recording needs its sidecar: without one, PEBBL stops with a message.
- Missing values are pandas' usual NA tokens (an empty field, `n/a`, `NaN`,
  `NA` and so on).
  - Signal columns keep them as NaN. A column that is entirely NaN counts as
    empty: it is left out of the default channel list with a note, and
    refused if asked for by name (`channel_config.present_channels`).
  - Any column whose name ends in `_peaks` is a 0/1 (or `True`/`False`) flag
    per sample. A missing value means "no peak".
- `ecg_peaks` and `ppg_peaks` seed the ECG and PPG reviews (the automated
  peaks the reviewer starts from). `rsp_peaks` and `eda_peaks` are read but
  not used. Other columns are used only if a channel in section 4 names them.
- `event`, if present, is a whole-number code per sample, shown read-only
  under the reviewed signal (section 4). A missing value there is a gap: no
  marker, and a marker where a code resumes.
- `rsp_raw`, if present, is the breathing belt as recorded (no breathing
  filter), in V. A study whose profile lists it (`raw_rows`, section 10) shows
  it in place of `rsp` in RSP reviews (section 4); otherwise it is read but
  not used.

## 2. Input: the sidecar, `*_physio.json`

The recording's path with `_physio.tsv.gz` replaced by `_physio.json`, read
as UTF-8 (with or without a byte-order mark) by `physio_io.read_sidecar`. It
must be a JSON object; a damaged or non-object sidecar stops PEBBL with a
plain message.

| Key | Read by | Expected | If absent |
|---|---|---|---|
| `Columns` | `load_physio_tsv`; `provenance.channel_provenance` | list of column names in file order | default columns (section 1); no signal hashes |
| `SamplingFrequency` | `load_physio_tsv` | samples per second, a positive number; runs loaded together must share it | PEBBL stops with a message (also for a non-number, or runs at different rates) |
| `<column>` (top level, keyed by column name) → `Units`, `Description`, `TechnicalDescription` | `physio_io.sidecar_unit_labels`, `sidecar_descriptions` | strings | default unit labels; no descriptions |
| `MachineQC.Channels.<channel>.BadSegments` | `physio_io.machine_qc_segments` | list of `[start, end)` whole-number pairs, `0 <= start < end <= N` | that channel starts with no pre-marked stretches, with a note |
| `MachineQC.Channels.<channel>.SegmentsByReason` | `machine_qc_segments` | `{reason: [[start, end], ...]}` | stretches are shown without reasons |
| `MachineQC.Channels.<channel>.Rule` | `machine_qc_segments` | text describing the rule | recorded as `""` |
| `Provenance.NumberOfSamples` | `machine_qc_segments` | the recording's row count | **all** of MachineQC is ignored, with a note (also if it doesn't match the row count) |
| `Provenance.PipelineCommit` | `channel_provenance` | text (the pipeline's version) | saved as `null` |
| `Provenance.CTTiming` | `provenance.new_pipeline_ppg` | anything: only its presence matters | the file counts as coming from before the PPG timing fix: PPG review is refused (unless `--allow-old-ppg`) and the PPG guide row is hidden |
| `Provenance.Practice` | `new_pipeline_ppg` | `true` | (marks the certification practice files) |

Notes:

- **Unit labels.** `Units` values understood (case-insensitive): `mV`, `V`,
  `microsiemens`, `arbitrary`, `mmHg`, `code`; `boolean` is skipped. They
  relabel the display only; PEBBL never rescales the data.
- **MachineQC** pre-marks bad stretches for every fresh segment channel
  that has an entry of its own in `MachineQC.Channels` (`emg_cor` included,
  when the pipeline checks it), PPG's stretches for its `bad_ppg`
  companion, and, with the profile setting `bad_ecg_from_machine_qc`, ECG's
  stretches for `bad_ecg` (`channel_config.machine_qc_source`). The
  `machine_qc` flag in `channel_config.CHANNELS` (RSP, PPG, EDA, SBP, DBP,
  `bad_ppg`) only marks the channels the machine is expected to check: a
  session whose fresh channels include none of them gets no "no machine
  check" note (`channel_config.uses_machine_qc`). A malformed entry makes that one
  channel start blank, with a note; it is never read as "nothing found".
  Stretches are used exactly as given (never widened or merged).
- **Never read:** `MachineQC.CoordinateSpace` and
  `MachineQC.Channels.<channel>.ProportionBad` (PEBBL computes the flagged
  fraction itself).

## 3. Where PEBBL looks for recordings

Built by `physio_io.find_run_path` from the data folder, subject and run:

```
<data folder>/sub-<SSS>/ses-run<R>/beh/sub-<SSS>_ses-run<R>_task-sdi_physio.tsv.gz
```

- The subject is zero-padded to three digits (`1` becomes `001`); runs are
  `1` and `2`.
- The task label (`sdi` above) is the study profile's `task` (section 10;
  `sdi` for synthetic demo data). If that file isn't there, PEBBL uses the
  one `sub-<SSS>_ses-run<R>_task-*_physio.tsv.gz` in that folder; with more
  than one it stops and lists them.
- **Outputs** go next to the recording (or to `--out-dir`), named from its
  stem, the file name minus `_physio.tsv.gz`: for example
  `sub-001_ses-run1_task-sdi_annotations_abc.json`.
- **Working from a shared folder** (`box_sync`): PEBBL copies `sub-<SSS>/` to
  the study's own folder inside the local working folder (`<local>/<task>/`,
  or `<local>/<task>-practice/` for certification practice; since
  2026-11-08), works there, then copies new or changed files back and
  confirms each one before deleting the local copy. It never copies the
  recording or sidecar (`*_physio.tsv.gz`, `*_physio.json`) back.
- **Initials** are 2-4 lowercase letters.

## 4. Channels

From `channel_config.py`. Point channels hold peak marks; segment channels
hold bad stretches.

| Key | Mode | Unit | Starts from | Notes |
|---|---|---|---|---|
| `ecg` | point | mV | `ecg_peaks`, or the reviewer's own QRS template | interval check |
| `rsp` | segment | V | MachineQC | was a point channel before 2026-09-25 |
| `ppg` | point | arbitrary | `ppg_peaks` | reviewed on its own; new-pipeline files only; interval check |
| `eda` | segment | µS | MachineQC | |
| `sbp`, `dbp` | segment | mmHg | MachineQC | |
| `emg_cor`, `emg_zyg` | segment | | MachineQC's own entry, if any (else blank) | no detector in PEBBL |
| `finger_temperature` | segment | | MachineQC's own entry, if any (else blank) | not collected in the original study |
| `bad_ecg` | segment | | blank (MachineQC's `ecg` entry with the profile setting `bad_ecg_from_machine_qc`) | companion of `ecg`: stretches where R peaks can't be located; no peaks are kept inside |
| `bad_ppg` | segment | | MachineQC's `ppg` entry | companion of `ppg`, same rule |
| `event` | (read-only) | code | | shown under the reviewed signal; never saved |
| `emg_cor_full` | (read-only) | `emg_cor`'s | | `emg_cor`'s column at full range, right under it whenever COR is reviewed; never saved (`channel_config.FULL_RANGE_CHANNELS`) |
| `rsp_raw` | (raw row) | V | | the `rsp_raw` column, shown **in place of** `rsp` when the study profile's `raw_rows` lists it; marks drawn on it are `bad_rsp` and save under `rsp`; its size leaves out RSP's machine stretches for `rail` and `swing`, and it is cut at 2.5 times that size (`channel_config.RAW_ROWS`) |

- **Guide rows** (shown, never saved): PPG under SBP and DBP, EDA under RSP,
  RSP under EDA; PPG under ECG only when asked for (`--ecg-ppg-guide`).
- **`event` codes.** PEBBL marks each change of value with its number; it
  attaches no meaning to the numbers. In the original study 0 meant outside
  the task, and 20, 30, 40, 50 and 65 were its phases.

## 5. Output: a reviewer's file, `<stem>_annotations_<initials>.json`

Written by `annotation_io.save_annotation_json` (schema version 2,
`annotation_io.ANNOTATION_SCHEMA_VERSION`); `json.dump` with `indent=2`, pure
ASCII (non-ASCII characters are escaped, e.g. `µS`).

**Top level**

| Key | Value |
|---|---|
| `schema_version` | `2` (files without it are version 1) |
| `coordinate_space` | the convention above, in words |
| `initials` | the reviewer |
| `timestamp` | `YYYY-MM-DD HH:MM:SS` |
| `source_file` | the recording's stem (`physio_review.py`), or its full path (`physio_annotate.py`) |
| `sampling_rate` | the sidecar's `SamplingFrequency` |
| `channels` | one entry per channel, below |
| `study_profile` | the study profile used (section 10): `{name, sha256}`, the profile's name and the SHA-256 of its file. Absent from synthetic demo reviews, and from real reviews saved before profiles were required (2026-11-01) without one |

**Per channel**

| Key | On | Value |
|---|---|---|
| `mode` | all | `"point"` or `"segment"` |
| `indices` | point | sorted sample numbers of the peaks. A peak the reviewer drew is placed at the highest sample inside the box they drew; only if the box cut off the top of the wave does it climb to that top, at most 15 ms (`annotation_io.point_sample_from_annotation`) |
| `snap_window_samples` | point | samples in 50 ms |
| `bad_segments` | segment | sorted `[start, end)` pairs (overlaps are not merged) |
| `status` | all reviewed this session, companions included | `"complete"` or `"in_progress"` |
| `reviewed_span` | all | `[0, N]`: the whole file |
| `value_range` | segment channels with data | `{min, mean, max, unit}` of the signal on screen (for RSP shown as `rsp_raw`, of `rsp_raw`) |
| `seed` | segment channels pre-filled from MachineQC | `{source: "MachineQC", rule, segments, pipeline_commit[, by_reason][, not_shown]}`: what the reviewer started from; `not_shown` (`{reason: [[start, end], ...]}`) holds the machine stretches the profile's `hidden_machine_reasons` kept off the screen, which `segments` and `by_reason` then leave out |
| `provenance` | when the recording has a sidecar | `{number_of_samples, pipeline_commit, signal_sha256, signal_sha256_def}`; companions add `column` (their parent's column); a channel shown as its raw row adds `raw_column` (e.g. `"rsp_raw"`) and `raw_signal_sha256` (that column's hash, same definition) |
| `guide_shown` | ECG, SBP, DBP, RSP, EDA | true if the guide row was on screen in any saved session |
| `inverted` | ECG, only when true | the ECG was reviewed flipped (section 10) |
| `edit_history` | all | one record per saved session, below |

- **`signal_sha256`** is the SHA-256 of the channel's column exactly as
  written: for every row, the field's raw bytes plus `\n` (an empty field is
  just `\n`; line endings don't matter; any change to a written value,
  including its formatting, does). `signal_sha256_def` names this definition,
  `"physioprocess-column-sha256-v1"`, shared with physioProcess
  (`provenance.column_sha256`). On reopening a review, a different sample
  count or hash stops the session, so marks are never applied to a changed
  signal. `raw_signal_sha256`, when saved, is checked the same way: the row the
  reviewer judged on can't change underneath their marks either.
- **`edit_history` records:** `saved_at`, `initials`, `started_from` (what
  the session began from: `"saved review"`, `"MachineQC"`, the ECG seed's
  source, a `"<column> column"`, or `"blank"`), and the changes: for point
  channels `added` and `removed` sample numbers and `moved` `[from, to]`
  pairs (matched one-to-one, closest first, within 50 ms); for segment
  channels `added` and `removed` stretches. Optional: `guide_shown`, and on a
  parent channel `removed_in_bad_stretches` and
  `seeds_left_out_in_bad_stretches`.
- **Legacy entries.** A review made in an older mode is kept under
  `<channel>_points_legacy` (or `_segments_legacy`, or `_unknown_legacy`;
  numbered `_2`, `_3` if taken). PEBBL never reads them back.
- **Lab tools** may add `retired: {at, reason, from_channel}`
  (`retire_channels.py`) or `migrated_from` (`migrate_legacy_json.py`).

**Saving over an earlier review.** The earlier file is first copied to
`backups/<stem>_annotations_<initials>_<YYYYMMDD_HHMMSS>.json`. Channels not
reviewed this session are kept unchanged. If the earlier file can't be read,
the session is written to `<stem>_annotations_<initials>_RESCUED_<YYYYMMDD_HHMMSS>.json`
next to it instead, and nothing is merged. Nothing is written when the
reviewer discards, or when nothing changed.

**Reopening a review** (`annotation_io.load_saved_annotations`) requires the
same `sampling_rate`, whole-number `indices` in `[0, N)`, and `bad_segments`
with `0 <= start < end <= N`; otherwise PEBBL stops with a message and
changes nothing.

## 6. Output: the reconciled file, `<stem>_annotations_reconciled.json`

Written by `reconcile.save_reconciled_json` after Step 3, where a third
person (the reconciler) settles two reviewers' disagreements. Same shape as
section 5, with these differences:

- Top level adds `reconciled_by` (the reconciler), `reconciled_from`
  (`[A, B]`, the reviewers' initials) and `agreement` (below); `initials` is
  the reconciler. `study_profile` is recorded as in section 5.
- Each channel adds `reconciled_by`, `reconciled_from` and, when any list is
  non-empty, `reviewer_origin`: `{"agreed": [...], "only_<A>": [...],
  "only_<B>": [...]}`, the marks as they stood before reconciling. A final
  mark in none of them was added by the reconciler.
- `edit_history` is a single record, `started_from` `"the two reviews (A, B)"`.
- Channels with a guide row add `guide_shown_by`:
  `{"adjudicator": ..., "<A>": ..., "<B>": ...}` (the reconciler's key keeps
  the role's old name).
- No `seed` or `value_range`.
- Backups and RESCUED files follow section 5's naming, with `reconciled` in
  place of the initials.

**`agreement.<channel>`** (`reconcile.agreement_statistics`) compares the
two reviewers' saved files, before the reconciler changes anything:

| Mode | Keys |
|---|---|
| point | `agreed`, `only_a`, `only_b`: marks matched one-to-one, closest first, within 50 ms. `percent_of_a_matched` = 100 × agreed / (agreed + only_a), and `percent_of_b_matched` likewise: the share of each reviewer's marks the other also placed. `percent_agreement` = 100 × agreed / (agreed + only_a + only_b), the share of all distinct marks that both placed (a Jaccard index, lower than either per-reviewer share). Each is `null` with no marks. `median_timing_difference_ms`: the median absolute difference within agreed pairs. |
| segment | `agreed`, `only_a`, `only_b`: stretches paired when their overlap is at least half their union. `seconds_marked_a`, `seconds_marked_b`. `cohen_kappa_seconds`: Cohen's kappa over 1-second bins, a bin bad for a reviewer when at least half its samples are. `cohen_kappa_samples`: the same over every sample; neighbouring samples aren't independent judgments, so prefer the per-second value. Each is `null` when undefined (e.g. neither marked anything). |

`percent_of_a_matched`, `percent_of_b_matched` and `cohen_kappa_seconds` were
added on 2026-10-31; reconciled files saved earlier don't have them.

`only_a` and `only_b` are literal key names; `reconciled_from` says which
reviewer is A.

## 7. The session log, `<stem>_annotations_<initials>.log`

Written by `session_log` for sessions run through `physio_review.py`, next to
the reviewer's file, as UTF-8, appended to. One file per reviewer per run
holds all their sessions on it (the reconciler's Step 3 sessions go in the
reconciler's log).

- **Header:** a line of `=`; `PEBBL session started YYYY-MM-DD HH:MM:SS -- <initials>`;
  the subject and step (e.g. `Step 2 (review), run 1`); a line of `=`.
- **Body:** everything the session printed, including the session summary
  shown in the dialog and the reviewer's choice.
- **Footer:** `Session ended <time>`; the step, run and channels; and
  `Outcome:` one of `saved, marked finished`, `saved, still in progress`,
  `discarded (nothing saved)`, `nothing saved`, `ended early (exit code N)`.

## 8. The shared tracking log, `processing_log.csv`

Written by `processing_log.append_processing_log_entry` in the data folder
(never in the local working copy), one row per event:

| Column | Value |
|---|---|
| `Subject` | `sub-<SSS>` |
| `Run` | the run (empty for `claimed`) |
| `Initials` | the reviewer (the reconciler for Step 3) |
| `Step` | `claimed`; `created QRS template`; `<channels> reviewed -- finished` or `-- not finished`; `reconciled <A> vs <B> (<channels>) -- finished` or `-- not finished` |
| `Notes` | the reviewer's comment (Step 3 adds the agreement text) |
| `Timestamp` | `M/D/YYYY H:MM`, local time |
| `Value_ranges` | for reviews: each channel's `LABEL: low-high unit (mean m)`, separated by `; ` |

- Rows are written only for real data (not synthetic demos), and the review
  and reconciliation rows only when something was saved.
- After each write PEBBL copies the log to
  `processing_log_READONLY_SNAPSHOT.csv`: in the user's home folder for a git
  clone of PEBBL, otherwise in the folder above the code
  (`processing_log.default_snapshot_dir`).

## 9. Other files

- **ECG template (Step 1):** `<stem>_ecg_corrected_qrs_<initials>.json`,
  `{"ecg": {"corrected_peaks": [...], "initials": [...], "ppg_guide_shown": ...}}`
  (plus `"inverted": true` when built on a flipped ECG),
  and `<stem>_qrs_template_<initials>.csv`, the averaged beat in one column
  `y` (`qrs_template.save_qrs_template_outputs`). Earlier versions are backed
  up to `backups/` first.
- **Certification practice** (`practice_certification.py make`): two
  synthetic runs for subject 990 in the layout of section 3, with sidecars in
  the format of section 2 (`Provenance.Practice: true`), and
  `practice_key.pebbl`, the answer key: one header line, then the key as
  zlib-compressed, base64-encoded JSON. With urglab's profile
  (`--study-profile`), `practice_urglab.py` writes subjects 990 and 991 (run
  1, `task-urglab`) in physiopipe's format, matching
  `fixtures/urglab_reference_sidecar.json`. Their key has `"study": "urglab"`,
  one entry per participant under `"subjects"` (with `ecg_inverted`), and
  `descriptions_sha256`.
- **Alignment report:** `check_reprocess_alignment.py --report <csv>` writes
  one row per checked run.
- **Install marker:** `annotate_env/requirements.installed.txt`, the
  requirements last installed by `pebbl_launcher.py`.

## 10. Study profiles, `pebbl_study.toml`

Read by `study_profile.load`. Every study's settings are a small TOML file
that lab staff put in that study's data folder, next to `processing_log.csv`
(or name with `physio_review.py --study-profile <file>`). **A review of real
data needs one** (HLU, 2026-11-01): a data folder without it stops the
session with a plain message, as does practice or `--input-run1` without
`--study-profile`. Only synthetic demo data use PEBBL's built-in settings,
those of the study it was first built for (task `sdi`). The minimal file:

```toml
schema_version = 1
name = "My study"   # shown at the start of a session; recorded in saved files
task = "mystudy"    # the recordings' task label (letters and digits)
```

Optional settings (the default, PEBBL's built-in value, in brackets):

| Setting | Effect |
|---|---|
| `runs` | the runs PEBBL looks for and offers, as text, e.g. `["1"]` (`["1", "2"]`) |
| `channels` | the channels offered, from section 4's keys (all) |
| `practice` | whether certification practice is offered (`true`) |
| `ppg_requires_timing_marker` | PPG review and the PPG guide need `Provenance.CTTiming` (`true`: the original study's pipeline history) |
| `bad_ecg_from_machine_qc` | `bad_ecg` starts from `MachineQC.Channels.ecg`'s stretches (`false`: starts blank) |
| `ecg_inversion_option` | the form offers "Flip the ECG (recorded upside down)", `--ecg-inverted` (`false`) |
| `ecg_seed_snap_ms` | how far Step 1 moves each template-found R peak to the highest point, 0-200 ms; 0 leaves it where the template alignment put it (`15`; 50 until 2026-10-31) |
| `keep_machine_stretches` | bad-stretch channels whose MachineQC stretches the study's analysis uses whatever the RA marks, e.g. `["emg_cor"]`: the opening text and the over-90% warning say to leave them and only add, and deleting or shortening one draws a check line before saving (none) |
| `raw_rows` | raw rows to show in place of their channel, e.g. `["rsp_raw"]` (section 4); a review of that channel on a file without the column (or with it all NaN) stops (none) |
| `hidden_machine_reasons` | MachineQC reasons, by channel, whose stretches the study's analysis applies from the sidecar itself, e.g. `{ rsp = ["filter_edge"] }`: they aren't pre-filled, so RAs can't remove them; the opening text says how many were left off, and the saved `seed` records them as `not_shown`. Applied only when the reasons' union is exactly the channel's `BadSegments`; otherwise every stretch is shown (none) |
| `[event_labels]` | names shown beside event codes in the viewer, e.g. `50 = "baseline"` (numbers only) |

**Flipping the ECG.** With `--ecg-inverted`, PEBBL negates the ECG in memory
for Steps 1-3, so its R peaks point up; the file, and so its signal hash, is
unchanged, and saved peaks are file samples as always. The Step 1 template
and the saved `ecg` entry record `"inverted": true`. A template made with
the opposite setting stops Step 2, with a message to rebuild it in Step 1.
A saved ECG review made with the opposite setting makes an ECG session ask
whether to start the ECG over (HLU, 2026-11-01): if the RA confirms, `ecg`
and `bad_ecg` start fresh, the earlier entries are kept as
`ecg_points_legacy` and `bad_ecg_segments_legacy` at save, and the other
channels are untouched; Cancel stops with nothing changed. Sessions on other
channels aren't affected. Step 3 warns when a reviewer's setting differs
from the reconciler's.

- PEBBL refuses a profile with an unknown setting, a wrong value, or text it
  can't read, with a plain message: a misspelled setting is never silently
  ignored.
- A recording whose task label isn't the profile's `task` stops the session
  before anything is saved.
- The session log's header names the profile, and saved review files record
  it (`study_profile`, sections 5 and 6).
- The form shows the study's name under the data folder, or why the folder
  can't be used (no profile, no `sub-*` folders, or a profile PEBBL refuses),
  and won't start a session on such a folder.
- `annotate_env/Scripts/python.exe study_profile.py <file>` checks a profile
  before it's copied into a data folder: it prints the settings PEBBL reads
  from it, or why it can't be used.
- Profiles hold settings only, never data. They are kept with each study's
  own records, not in this repository.

## What PEBBL does not check

Since 2026-10-31 PEBBL stops, rather than guessing, on a missing or damaged
sidecar, a `Columns` list that doesn't match the file, a sampling rate that
isn't a positive number, and runs recorded at different rates (sections 1
and 2). A known gap that remains:

- A header row in the `.tsv.gz` is read as data (physioProcess and
  physiopipe never write one).
