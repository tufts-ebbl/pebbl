#!/usr/bin/env python3
"""
Tests for pebbl_launcher.py, the "Start PEBBL" update-then-open step (HLU,
2026-10-04), with git and pip mocked: nothing is pulled or installed.

  1. A folder that isn't its own git checkout (a development copy) is never
     pulled.
  2. Git missing, a failed pull, a timeout: one plain line, then PEBBL starts
     anyway; a good pull says it's up to date. The pull is `git pull
     --ff-only` in PEBBL's folder.
  3. requirements.txt: reinstalled only when it differs from the copy saved
     at the last successful install; a failed install keeps it pending and
     stops before the form opens.
  4. The form opens with no arguments (physio_review.py).

Run: python test_pebbl_launcher.py
"""

import importlib.util
import os
import shutil
import subprocess
import tempfile
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))


def load_copy(folder):
    shutil.copy(os.path.join(HERE, "pebbl_launcher.py"), folder)
    spec = importlib.util.spec_from_file_location("pebbl_launcher_copy", os.path.join(folder, "pebbl_launcher.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    tmp = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmp, "requirements.txt"), "w") as f:
            f.write("six==1.17.0\n")
        os.makedirs(os.path.join(tmp, "annotate_env"))
        pl = load_copy(tmp)

        print("1. A development copy (no .git of its own) is never pulled...")
        with patch.object(pl.subprocess, "run") as run:
            assert pl.update_code() is None and not run.called
        print("   OK")

        print("2. Updating: git missing, failed pull, timeout, good pull...")
        os.makedirs(os.path.join(tmp, ".git"))
        with patch.object(pl.shutil, "which", return_value=None):
            assert "Git isn't installed" in pl.update_code()
        failed = subprocess.CompletedProcess(["git"], 1, "", "fatal: unable to access the repo: Could not resolve host\n")
        with patch.object(pl.shutil, "which", return_value="git"), \
             patch.object(pl.subprocess, "run", return_value=failed) as run:
            line = pl.update_code()
        assert line.startswith("Couldn't update PEBBL; starting the version you have.") and "Could not resolve" in line
        assert run.call_args[0][0] == ["git", "pull", "--ff-only", "--quiet"] and run.call_args[1]["cwd"] == pl.HERE
        with patch.object(pl.shutil, "which", return_value="git"), \
             patch.object(pl.subprocess, "run", side_effect=subprocess.TimeoutExpired("git", 120)):
            assert "TimeoutExpired" in pl.update_code()
        good = subprocess.CompletedProcess(["git"], 0, "", "")
        with patch.object(pl.shutil, "which", return_value="git"), patch.object(pl.subprocess, "run", return_value=good):
            assert pl.update_code() == "PEBBL is up to date."
        print("   OK")

        print("3. Components: reinstalled only when requirements.txt changed; a failure stops before the form...")
        assert pl.requirements_changed(), "never installed: install"
        shutil.copyfile(pl.REQUIREMENTS, pl.INSTALLED)
        assert not pl.requirements_changed()
        with open(pl.REQUIREMENTS, "a") as f:
            f.write("colorama==0.4.6\n")
        assert pl.requirements_changed()
        with patch.object(pl.subprocess, "call", return_value=0) as call, patch("builtins.print"):
            assert pl.install_requirements()
        assert call.call_args[0][0][1:4] == ["-m", "pip", "install"] and not pl.requirements_changed()
        with open(pl.REQUIREMENTS, "a") as f:
            f.write("idna==3.20\n")
        with patch.object(pl.subprocess, "call", return_value=1), patch("builtins.print"):
            assert not pl.install_requirements()
        assert pl.requirements_changed(), "a failed install stays pending"
        with patch.object(pl, "update_code", return_value=None), \
             patch.object(pl, "install_requirements", return_value=False), \
             patch.object(pl.subprocess, "call") as call, patch("builtins.print"):
            assert pl.main() == 1 and not call.called
        print("   OK")

        print("4. The form opens with no arguments...")
        shutil.copyfile(pl.REQUIREMENTS, pl.INSTALLED)
        with patch.object(pl, "update_code", return_value="PEBBL is up to date."), \
             patch.object(pl.subprocess, "call", side_effect=[pl.FORM_CANCELED_EXIT]) as call, patch("builtins.print"):
            assert pl.main() == 0
        args = call.call_args[0][0]
        assert args[1] == os.path.join(pl.HERE, "physio_review.py") and len(args) == 2, args
        assert call.call_args[1]["env"]["PEBBL_LAUNCHER"] == "1"
        print("   OK")

        print("5. Back to the form after each session (HLU, 2026-10-05): a finished session (0) or one that "
              "stopped with its own message (4) opens the form again; Cancel (3) closes PEBBL with 0; Ctrl-C or "
              "any other failure closes it with that code, so nothing loops...")
        with patch.object(pl.subprocess, "call", side_effect=[0, pl.SESSION_STOPPED_EXIT, 0, pl.FORM_CANCELED_EXIT]) \
                as call, patch("builtins.print") as printed:
            assert pl.run_sessions() == 0
        assert call.call_count == 4
        said = " ".join(str(c.args[0]) for c in printed.call_args_list)
        assert "Opening the form for your next session" in said and "stopped early" in said
        for code in (1, 130):
            with patch.object(pl.subprocess, "call", side_effect=[0, code]) as call, patch("builtins.print"):
                assert pl.run_sessions() == code and call.call_count == 2
        import ast
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "physio_review.py"), encoding="utf-8") as f:
            tree = ast.parse(f.read())
        consts = {t.id: n.value.value for n in tree.body if isinstance(n, ast.Assign)
                  for t in n.targets if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant)}
        assert consts["FORM_CANCELED_EXIT"] == pl.FORM_CANCELED_EXIT
        assert consts["SESSION_STOPPED_EXIT"] == pl.SESSION_STOPPED_EXIT
        assert consts["LAUNCHER_ENV"] == "PEBBL_LAUNCHER"
        print("   OK: loops on 0 and 4; Cancel ends with 0; 1 and 130 end with their code; codes match physio_review")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL PEBBL LAUNCHER TESTS PASSED.")


if __name__ == "__main__":
    main()
