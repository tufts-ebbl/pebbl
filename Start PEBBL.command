#!/bin/bash
# Double-click to start PEBBL on a Mac. It checks for updates, then opens the PEBBL form.
# The body is one function that bash reads in full before running it, so an update that
# rewrites this file while it runs can't garble it.
main() {
  cd "$(dirname "$0")" || return 1
  if [ ! -x annotate_env/bin/python ]; then
    echo "PEBBL isn't set up on this Mac yet. In Terminal, run:  bash ~/pebbl/setup_pebbl.sh"
    return 1
  fi
  annotate_env/bin/python pebbl_launcher.py
  echo
  echo "PEBBL has closed. You can close this window."
}
main "$@"
exit $?
