#!/usr/bin/env python3
"""
Lab staff only: set aside ("retire") chosen channels in one saved review
file, so those channels can be reviewed afresh.

When a reprocess changes a channel's signal, Step 2 and Step 3 refuse to
reopen a review made on the old signal ("made on a different version of the
file"), and check_reprocess_alignment.py reports it as REVIEW_INVALIDATED.
This command moves each chosen channel's entry to a legacy key (for example
"sbp" -> "sbp_segments_legacy", numbered if that name is taken). The tool
never loads legacy entries, so the channel starts fresh in the next session:
pre-marked from the machine check where there is one, otherwise from the
automated peaks or blank. Nothing is deleted. The old entry stays in the file
with a "retired" record (when, and why), and the whole file is first copied
to a backups/ folder next to it.

A channel's companions go with it: retiring ppg also retires bad_ppg, and
retiring ecg also retires bad_ecg, because their records are tied to the
parent signal.

Read-only unless --apply. Run it only when no review session is open for
that subject; an open session's local copy would be pushed back over it.
Works on an RA's *_annotations_<initials>.json or a reconciled
*_annotations_reconciled.json.

Usage:
    annotate_env\\Scripts\\python.exe retire_channels.py --file <path>\\sub-001_ses-run1_task-sdi_annotations_hlu.json --channels sbp,dbp
    annotate_env\\Scripts\\python.exe retire_channels.py --file <same path> --channels sbp,dbp --reason "run 10 changed sbp/dbp" --apply
"""

import argparse
import json
import os
import shutil
from datetime import datetime

from annotation_io import _is_legacy_key, legacy_channel_key, read_saved_json, saved_channel_entries
from channel_config import COMPANION_CHANNELS

DEFAULT_REASON = "set aside so the channel can be reviewed afresh"


def plan_retirement(channels, requested):
    """
    ([(channel, legacy key), ...], [requested channels not in the file]) for
    the requested channels present in `channels`, each followed by its
    companions (COMPANION_CHANNELS). Legacy keys never collide with existing
    keys or with each other.
    """
    wanted = []
    for ch_key in requested:
        if ch_key not in wanted:
            wanted.append(ch_key)
        for comp_key, comp_cfg in COMPANION_CHANNELS.items():
            if comp_cfg["companion_of"] == ch_key and comp_key not in wanted:
                wanted.append(comp_key)
    taken, plan, missing = set(channels), [], []
    for ch_key in wanted:
        if ch_key not in channels:
            if ch_key in requested:
                missing.append(ch_key)
            continue
        legacy_key = legacy_channel_key(ch_key, channels[ch_key].get("mode"), existing=taken)
        taken.add(legacy_key)
        plan.append((ch_key, legacy_key))
    return plan, missing


def retire_channels(path, requested, reason="", apply=False, now=None):
    """
    Plans (and with apply=True, carries out) the retirement. Returns
    (plan, missing, backup_path or None). Stops plainly on an unreadable file
    or when asked to retire a legacy key.
    """
    payload = read_saved_json(path)
    channels = saved_channel_entries(payload, path)
    legacy = [c for c in requested if _is_legacy_key(c)]
    if legacy:
        raise SystemExit(f"{', '.join(legacy)} {'is' if len(legacy) == 1 else 'are'} already set aside (a legacy "
                         f"entry). Nothing has been changed.")
    plan, missing = plan_retirement(channels, requested)
    if not apply or not plan:
        return plan, missing, None
    now = now or datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(os.path.dirname(os.path.abspath(path)), "backups")
    os.makedirs(backup_dir, exist_ok=True)
    backup = os.path.join(backup_dir, f"{os.path.splitext(os.path.basename(path))[0]}_{stamp}.json")
    shutil.copy2(path, backup)
    moved = dict(plan)
    record = {"at": now.strftime("%Y-%m-%d %H:%M:%S"), "reason": reason or DEFAULT_REASON}
    new_channels = {}
    for ch_key, entry in channels.items():  # same order; a retired entry keeps its place under its new key
        if ch_key in moved:
            new_channels[moved[ch_key]] = {**entry, "retired": {**record, "from_channel": ch_key}}
        else:
            new_channels[ch_key] = entry
    payload["channels"] = new_channels
    tmp = path + ".retire_tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    return plan, missing, backup


def main(argv=None):
    parser = argparse.ArgumentParser(description="Lab staff: set aside chosen channels in one saved review file so "
                                                 "they can be reviewed afresh (see the module docstring).")
    parser.add_argument("--file", required=True,
                        help="The saved review file: *_annotations_<initials>.json or *_annotations_reconciled.json.")
    parser.add_argument("--channels", required=True, help="Comma-separated channel keys, e.g. 'sbp,dbp'.")
    parser.add_argument("--reason", default="", help="Why, recorded in the file (e.g. 'run 10 changed sbp/dbp').")
    parser.add_argument("--apply", action="store_true", help="Actually change the file (backed up first). "
                                                             "Without this, only prints what would change.")
    args = parser.parse_args(argv)
    if not os.path.isfile(args.file):
        raise SystemExit(f"File not found: {args.file}. Nothing has been changed.")
    requested = [c.strip().lower() for c in args.channels.split(",") if c.strip()]
    plan, missing, backup = retire_channels(args.file, requested, reason=args.reason, apply=args.apply)
    name = os.path.basename(args.file)
    for ch_key in missing:
        print(f"{ch_key}: not in {name}; nothing to set aside.")
    if not plan:
        print("Nothing to set aside. Nothing has been changed.")
        return 0
    for ch_key, legacy_key in plan:
        print(f"{ch_key} -> {legacy_key}" + ("" if args.apply else "  (would be set aside)"))
    if not args.apply:
        print("Read-only: nothing has been changed. Re-run with --apply to set these aside.")
        return 0
    print(f"Done. The file was backed up to {backup} first.")
    print(f"Next: the RA reviews {', '.join(c for c, _ in plan)} for this run again in Step 2 (it starts fresh), "
          f"and updates the tracking sheet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
