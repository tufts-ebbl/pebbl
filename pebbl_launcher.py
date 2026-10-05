#!/usr/bin/env python3
"""
Start PEBBL for an RA: update, then open the form (HLU, 2026-10-04).

Run by "Start PEBBL.bat" (Windows) and "Start PEBBL.command" (Mac) with the
annotate_env Python. In order:

1. If this folder is its own git checkout (it has a .git folder, as a clone
   of the public pebbl repo does) and git is installed, pull the latest
   version (`git pull --ff-only`). A failed pull (no internet, git missing,
   a local edit in the way) is reported in one line and the current version
   starts anyway. A development checkout inside a larger repo has no .git
   here and is never pulled.
2. If requirements.txt differs from the copy saved at the last successful
   install (annotate_env/requirements.installed.txt), install it again and
   save the new copy. A failed install stops here: the form would fail
   anyway.
3. On a Mac, keep "Start PEBBL.command" executable (a pull can rewrite it
   without the executable bit).
4. Run physio_review.py with no arguments, which opens the form.

Only the standard library is used, so an update that changes the packages
can't break this script.
"""

import filecmp
import os
import shutil
import stat
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = os.path.join(HERE, "annotate_env")
REQUIREMENTS = os.path.join(HERE, "requirements.txt")
INSTALLED = os.path.join(ENV, "requirements.installed.txt")
MAC_LAUNCHER = os.path.join(HERE, "Start PEBBL.command")


def update_code():
    """Pull the latest PEBBL if this folder is its own git checkout. Returns a one-line status."""
    if not os.path.isdir(os.path.join(HERE, ".git")):
        return None
    if shutil.which("git") is None:
        return "Couldn't check for PEBBL updates (Git isn't installed); starting the version you have."
    try:
        result = subprocess.run(["git", "pull", "--ff-only", "--quiet"], cwd=HERE, capture_output=True, text=True,
                                timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        return f"Couldn't check for PEBBL updates ({type(e).__name__}); starting the version you have."
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        return ("Couldn't update PEBBL; starting the version you have. If this keeps happening, send the lab "
                "staff this message: " + (detail[-1] if detail else f"git exit code {result.returncode}"))
    return "PEBBL is up to date."


def requirements_changed():
    return not (os.path.exists(INSTALLED) and filecmp.cmp(REQUIREMENTS, INSTALLED, shallow=False))


def install_requirements():
    """Install requirements.txt into this environment; True on success (then the copy is saved)."""
    print("Installing updated PEBBL components (this can take a few minutes)...")
    code = subprocess.call([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-r",
                            REQUIREMENTS])
    if code == 0:
        shutil.copyfile(REQUIREMENTS, INSTALLED)
        return True
    return False


def keep_mac_launcher_executable():
    if sys.platform == "darwin" and os.path.exists(MAC_LAUNCHER):
        mode = os.stat(MAC_LAUNCHER).st_mode
        os.chmod(MAC_LAUNCHER, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def main():
    status = update_code()
    if status:
        print(status)
    if requirements_changed() and not install_requirements():
        print("PEBBL couldn't install its updated components. Take a photo of this window and send it to the "
              "lab staff.")
        return 1
    keep_mac_launcher_executable()
    return subprocess.call([sys.executable, os.path.join(HERE, "physio_review.py")], cwd=HERE)


if __name__ == "__main__":
    sys.exit(main())
