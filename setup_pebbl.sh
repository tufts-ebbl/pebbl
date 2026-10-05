#!/bin/bash
# One-time PEBBL setup on a Mac. In Terminal, run:  bash ~/pebbl/setup_pebbl.sh
# It makes PEBBL's own Python environment (annotate_env) and installs what PEBBL needs.
cd "$(dirname "$0")" || exit 1
fail() {
  echo
  echo "Setup didn't finish. Take a screenshot of this window and send it to the lab staff."
  exit 1
}
echo
echo "Setting up PEBBL on this Mac (one time only)."
echo "This downloads about 1 GB and takes 5-15 minutes. Please keep this window open."
echo
if ! command -v python3.11 >/dev/null 2>&1; then
  echo "Python 3.11 isn't installed. Install it from https://www.python.org/downloads/release/python-3119/"
  echo "as the PEBBL manual describes, then run this again."
  exit 1
fi
if [ ! -x annotate_env/bin/python ]; then
  echo "Creating PEBBL's own Python environment..."
  python3.11 -m venv annotate_env || fail
fi
annotate_env/bin/python -m pip install --disable-pip-version-check --upgrade pip || fail
annotate_env/bin/python -m pip install --disable-pip-version-check -r requirements.txt || fail
cp requirements.txt annotate_env/requirements.installed.txt
chmod +x "Start PEBBL.command"
echo
echo "PEBBL is set up. To start it, double-click \"Start PEBBL.command\" in your pebbl folder."
