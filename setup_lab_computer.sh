#!/usr/bin/env bash
# Set up PEBBL on a Windows lab computer, for every user (HLU, 2026-10-06).
#
# Lab staff: open Git Bash with "Run as administrator", then paste:
#
#   curl -fLO https://raw.githubusercontent.com/tufts-ebbl/pebbl/main/setup_lab_computer.sh && bash setup_lab_computer.sh
#
# It's safe to run again: it skips whatever is already done and updates the rest.
# Options:
#   --python-installer '\\server\share\python-3.11.4-amd64.exe'
#       Install Python from this copy instead of downloading 3.11.9 from
#       python.org. Use the full network path: an administrator window can't see
#       drive letters mapped in your normal session.
#   --dry-run   Show what it would do, without changing anything.
#   --check     Only check this computer and print the summary (no admin needed).
#
# What it does:
#   1. Installs Python 3.11 for all users (C:\Program Files\Python311) if it
#      isn't there, after checking that the installer is signed by the Python
#      Software Foundation. If Python 3.11 is installed just for the account
#      running this (which blocks an all-users install), it explains why and
#      ASKS before uninstalling that copy. It never touches any other Python
#      (other versions, other accounts' copies, Anaconda).
#   2. Lets every account's Git update the PEBBL folder (Git's safe.directory).
#   3. Downloads PEBBL into C:\Users\Public\Downloads\pebbl, or updates it.
#   4. Gives every user write access to that folder, so PEBBL can update itself
#      whoever starts it.
#   5. Makes PEBBL's Python environment from the all-users Python and installs
#      its packages. An environment made from one account's own Python is
#      replaced: other users couldn't run it.
#   6. Puts a PEBBL icon on every user's desktop.
#   7. Checks all of the above and prints a summary.
#
# Each RA then signs in to Box Drive, makes a PhysioWorking folder in their own
# home folder, and starts PEBBL from the desktop icon (see the user manual).

set -u

PEBBL_DIR="${PEBBL_DIR:-/c/Users/Public/Downloads/pebbl}"
PEBBL_REPO="https://github.com/tufts-ebbl/pebbl.git"
PYTHON_VERSION="3.11.9"   # the last 3.11 release with a Windows installer
PYTHON_EXE="${PEBBL_PYTHON:-/c/Program Files/Python311/python.exe}"
PYTHON_URL="https://www.python.org/ftp/python/${PYTHON_VERSION}/python-${PYTHON_VERSION}-amd64.exe"
SHORTCUT="${PEBBL_SHORTCUT:-/c/Users/Public/Desktop/PEBBL.lnk}"
USERS_SID="*S-1-5-32-545"   # the built-in Users group, whatever the Windows language

say() { printf '\n== %s\n' "$*"; }
note() { printf '   %s\n' "$*"; }
run() { if [ "$DRY_RUN" = 1 ]; then printf '   would run: %s\n' "$*"; else "$@"; fi; }
fail() { printf '\nSTOPPED: %s\nNothing after this step was done. Fix it, then run this script again.\n' "$*" >&2; exit 1; }

DRY_RUN=0
CHECK_ONLY=0
PY_INSTALLER="${PEBBL_PYTHON_INSTALLER:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --check) CHECK_ONLY=1 ;;
    --python-installer) shift; PY_INSTALLER="${1:-}" ;;
    *) fail "Unknown option: $1 (use --python-installer PATH, --dry-run or --check)" ;;
  esac
  shift
done
winpath() { cygpath -w "$1"; }   # C:\Users\...
mixpath() { cygpath -m "$1"; }   # C:/Users/... (the form Git's settings use)

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Small PowerShell helpers, run with -File so no quoting can go wrong.
cat > "$WORK/signature.ps1" <<'PS'
param([string]$Path)
$s = Get-AuthenticodeSignature -LiteralPath $Path
"$($s.Status)|$($s.SignerCertificate.Subject)"
PS
cat > "$WORK/install_python.ps1" <<'PS'
param([string]$Exe, [string]$Log)
$p = Start-Process -FilePath $Exe -Wait -PassThru -ArgumentList '/quiet', 'InstallAllUsers=1', 'PrependPath=1',
    'Include_launcher=1', 'InstallLauncherAllUsers=1', 'Include_test=0', ('/log "' + $Log + '"')
exit $p.ExitCode
PS
cat > "$WORK/per_user_python.ps1" <<'PS'
param([string]$Version)
$k = "HKCU:\Software\Python\PythonCore\$Version\InstallPath"
if (Test-Path -LiteralPath $k) { (Get-ItemProperty -LiteralPath $k).'(default)' }
PS
cat > "$WORK/per_user_python_uninstall.ps1" <<'PS'
# This account's own Python <Version> installs: the visible "Python 3.11.x (64-bit)" bundle entries in HKCU
# (they have a BundleVersion and a quiet uninstall command). Lists their names; with -Run, uninstalls them.
param([string]$Version, [switch]$Run)
$found = @(Get-ChildItem 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall' -ErrorAction SilentlyContinue |
    ForEach-Object { Get-ItemProperty -LiteralPath $_.PSPath } |
    Where-Object { $_.BundleVersion -and $_.QuietUninstallString -and $_.DisplayName -like "Python $Version.*" })
if (-not $Run) { $found | ForEach-Object { $_.DisplayName }; exit 0 }
$code = 0
foreach ($p in $found) {
    $proc = Start-Process -FilePath 'cmd.exe' -ArgumentList ('/c "' + $p.QuietUninstallString + '"') -Wait -PassThru -WindowStyle Hidden
    "$($p.DisplayName): uninstaller exit code $($proc.ExitCode)"
    if ($proc.ExitCode -ne 0 -and $proc.ExitCode -ne 3010) { $code = $proc.ExitCode }
}
exit $code
PS
cat > "$WORK/shortcut.ps1" <<'PS'
param([string]$Lnk, [string]$Target, [string]$Dir, [string]$Icon)
$s = (New-Object -ComObject WScript.Shell).CreateShortcut($Lnk)
$s.TargetPath = $Target
$s.WorkingDirectory = $Dir
if ($Icon -and (Test-Path -LiteralPath $Icon)) { $s.IconLocation = "$Icon,0" }
$s.Save()
PS
run_ps() { powershell -NoProfile -ExecutionPolicy Bypass -File "$(winpath "$WORK/$1")" "${@:2}"; }

# --------------------------------------------------------------------------
say "0. Checking this window"
if [ "$CHECK_ONLY" = 1 ]; then
  note "Checking only: nothing will be changed."
elif [ "$DRY_RUN" = 1 ]; then
  note "Dry run: nothing will be changed."
elif ! net session > /dev/null 2>&1; then
  fail "Git Bash isn't running as administrator. Close it, right-click Git Bash, choose 'Run as administrator', and run this again."
fi
command -v git > /dev/null || fail "Git isn't installed."
note "OK"
ENV="$PEBBL_DIR/annotate_env"
want_home="$(winpath "$(dirname "$PYTHON_EXE")")"

if [ "$CHECK_ONLY" = 0 ]; then

# --------------------------------------------------------------------------
say "1. Python ${PYTHON_VERSION%.*} for all users"
if [ -x "$PYTHON_EXE" ]; then
  note "Already installed: $(winpath "$PYTHON_EXE")"
else
  installer="$WORK/python-installer.exe"
  if [ -n "$PY_INSTALLER" ]; then
    src="$(cygpath -u "$PY_INSTALLER")"
    [ -f "$src" ] || fail "Can't find the Python installer at $PY_INSTALLER. Use its full network path (\\\\server\\share\\...), not a drive letter: an administrator window can't see drives mapped in your normal session."
  fi
  # A Python 3.11 installed just for this account blocks the all-users install: the installer tries to
  # change that copy instead, and fails (HLU's lab computer, 2026-10-06: code 67, log 0x643).
  if [ -n "${PEBBL_TEST_PER_USER+set}" ]; then
    per_user="$PEBBL_TEST_PER_USER"  # tests only
  else
    per_user="$(run_ps per_user_python.ps1 "${PYTHON_VERSION%.*}" | tr -d '\r')"
  fi
  if [ -n "$per_user" ]; then
    ver="${PYTHON_VERSION%.*}"
    by_hand="Settings > Apps > Installed apps > 'Python $ver.x (64-bit)' > Uninstall"
    if [ -n "${PEBBL_TEST_PER_USER_NAMES+set}" ]; then
      names="$PEBBL_TEST_PER_USER_NAMES"  # tests only
    else
      names="$(run_ps per_user_python_uninstall.ps1 "$ver" | tr -d '\r')"
    fi
    [ -n "$names" ] || fail "Python $ver is installed just for your account, in ${per_user}, and it blocks the all-users install. Uninstall it by hand ($by_hand), then run this script again."
    note "Python $ver is installed just for your account: $(echo "$names" | paste -sd ',' -), in ${per_user}."
    note "The all-users installer can't install next to it (it tries to change that copy instead, and fails),"
    note "so it has to go first. PEBBL doesn't use it, but anything installed into that copy goes with it."
    note "Nothing else is touched: other Python versions, other accounts' copies and Anaconda stay."
    printf '   Uninstall it now? [y/N] '
    read -r answer || answer=""
    answer="${answer//$'\r'/}"; answer="${answer// /}"  # tolerate a Windows line ending or stray spaces
    case "$answer" in
      y|Y|yes|Yes|YES) ;;
      *) fail "Left it in place. To go on, uninstall it ($by_hand), then run this script again." ;;
    esac
    run run_ps per_user_python_uninstall.ps1 "$ver" -Run \
      || fail "The uninstall didn't finish (exit codes above). Uninstall it by hand ($by_hand), then run this script again."
    if [ "$DRY_RUN" = 0 ]; then
      [ -z "$(run_ps per_user_python.ps1 "$ver" | tr -d '\r')" ] \
        || fail "It still looks installed. Uninstall it by hand ($by_hand), then run this script again."
      note "Uninstalled."
    fi
  fi
  if [ -n "$PY_INSTALLER" ]; then
    note "Copying the Python installer from $PY_INSTALLER..."
    run cp "$src" "$installer" || fail "Couldn't copy $PY_INSTALLER"
  else
    note "Downloading the Python ${PYTHON_VERSION} installer from python.org (about 26 MB)..."
    run curl -fL --retry 3 -o "$installer" "$PYTHON_URL" || fail "Couldn't download $PYTHON_URL"
  fi
  if [ "$DRY_RUN" = 0 ]; then
    sig="$(run_ps signature.ps1 "$(winpath "$installer")" | tr -d '\r')"
    case "$sig" in
      Valid\|*"Python Software Foundation"*) note "Installer signature: valid, Python Software Foundation." ;;
      *) fail "The downloaded installer isn't validly signed by the Python Software Foundation ($sig)." ;;
    esac
  fi
  note "Installing for all users (a few minutes; no windows appear)..."
  py_log="/tmp/pebbl_python_install.log"
  if [ "$DRY_RUN" = 0 ]; then
    run_ps install_python.ps1 "$(winpath "$installer")" "$(winpath "$py_log")"
    code=$?
    if [ "$code" != 0 ] && [ "$code" != 3010 ]; then
      case "$code" in
        67) meaning="a network location it needed couldn't be found" ;;
        1602) meaning="the installation was canceled" ;;
        1603) meaning="a fatal error during installation" ;;
        1618) meaning="another installation is already running (often Windows Update); wait for it, then run this again" ;;
        *) meaning="see its log" ;;
      esac
      printf '\n   The installer'"'"'s log is %s. Its last lines:\n' "$(winpath "$py_log")"
      tail -n 15 "$py_log" 2> /dev/null | tr -d '\r' | sed 's/^/     /'
      fail "The Python installer stopped with code $code ($meaning). If Python ${PYTHON_VERSION%.*} is installed for any single account on this computer, uninstall that copy first."
    fi
  else
    run "$installer" /quiet InstallAllUsers=1 PrependPath=1 Include_launcher=1 InstallLauncherAllUsers=1 Include_test=0
  fi
  if [ "$DRY_RUN" = 0 ] && [ ! -x "$PYTHON_EXE" ]; then
    fail "The installer finished, but there's no $(winpath "$PYTHON_EXE"). This usually means Python ${PYTHON_VERSION%.*} is already installed for one account on this computer. Uninstall that copy (Settings > Apps > 'Python ${PYTHON_VERSION%.*}...'), then run this script again."
  fi
fi
if [ "$DRY_RUN" = 0 ] || [ -x "$PYTHON_EXE" ]; then
  version="$("$PYTHON_EXE" -c 'import sys; print("%d.%d" % sys.version_info[:2])' | tr -d '\r')"
  [ "$version" = "${PYTHON_VERSION%.*}" ] || fail "$(winpath "$PYTHON_EXE") is Python $version, not ${PYTHON_VERSION%.*}."
  note "Python $version OK"
fi

# --------------------------------------------------------------------------
say "2. Letting every account's Git update $(winpath "$PEBBL_DIR")"
if git config --system --get-all safe.directory 2> /dev/null | tr -d '\r' | grep -qxF "$(mixpath "$PEBBL_DIR")"; then
  note "Already set."
else
  run git config --system --add safe.directory "$(mixpath "$PEBBL_DIR")" || fail "Couldn't change Git's system settings."
fi

# --------------------------------------------------------------------------
say "3. PEBBL in $(winpath "$PEBBL_DIR")"
if [ -d "$PEBBL_DIR/.git" ]; then
  origin="$(git -C "$PEBBL_DIR" remote get-url origin | tr -d '\r')"
  [ "$origin" = "$PEBBL_REPO" ] || fail "$(winpath "$PEBBL_DIR") is a copy of $origin, not $PEBBL_REPO."
  note "Already downloaded; updating..."
  run git -C "$PEBBL_DIR" pull --ff-only || fail "Couldn't update PEBBL (git pull)."
elif [ -e "$PEBBL_DIR" ] && [ -n "$(ls -A "$PEBBL_DIR" 2> /dev/null)" ]; then
  fail "$(winpath "$PEBBL_DIR") already exists and isn't a copy of PEBBL. Move it aside, then run this again."
else
  run git clone "$PEBBL_REPO" "$PEBBL_DIR" || fail "Couldn't download PEBBL (git clone)."
fi

# --------------------------------------------------------------------------
say "4. Write access for every user"
# //grant: Git Bash would otherwise read /grant as a path.
run icacls "$(winpath "$PEBBL_DIR")" //grant "${USERS_SID}:(OI)(CI)M" || fail "Couldn't set the folder's permissions."

# --------------------------------------------------------------------------
say "5. PEBBL's Python environment and packages"
make_env=1
if [ -f "$ENV/pyvenv.cfg" ]; then
  home="$(sed -n 's/^home *= *//p' "$ENV/pyvenv.cfg" | tr -d '\r')"
  if [ "$(echo "$home" | tr 'A-Z' 'a-z')" = "$(echo "$want_home" | tr 'A-Z' 'a-z')" ]; then
    note "Environment OK (made from $home)."
    make_env=0
  else
    note "The environment was made from $home, which other users can't run; making it again."
    run rm -rf "$ENV"
  fi
fi
if [ "$make_env" = 1 ]; then
  note "Making the environment (a minute)..."
  run "$PYTHON_EXE" -m venv "$(winpath "$ENV")" || fail "Couldn't make the environment."
fi
if [ "$make_env" = 0 ] && [ -f "$ENV/requirements.installed.txt" ] \
    && cmp -s "$PEBBL_DIR/requirements.txt" "$ENV/requirements.installed.txt"; then
  note "Packages already installed."
else
  note "Installing PEBBL's packages (about 1 GB; 5-15 minutes)..."
  run "$ENV/Scripts/python.exe" -m pip install --disable-pip-version-check --upgrade pip || fail "Couldn't update pip."
  run "$ENV/Scripts/python.exe" -m pip install --disable-pip-version-check -r "$(winpath "$PEBBL_DIR/requirements.txt")" \
    || fail "Couldn't install PEBBL's packages."
  run cp "$PEBBL_DIR/requirements.txt" "$ENV/requirements.installed.txt"
fi

# --------------------------------------------------------------------------
say "6. The PEBBL icon on every user's desktop"
run run_ps shortcut.ps1 "$(winpath "$SHORTCUT")" "$(winpath "$PEBBL_DIR/Start PEBBL.bat")" "$(winpath "$PEBBL_DIR")" \
  "$(winpath "$PEBBL_DIR/pebbl.ico")" || fail "Couldn't make the desktop shortcut."

fi  # end of the setup steps (skipped with --check)

# --------------------------------------------------------------------------
say "7. Checking"
problems=0
lower() { echo "$1" | tr 'A-Z' 'a-z'; }
check() {  # check "what" command...
  local what="$1"; shift
  if "$@" > /dev/null 2>&1; then printf '   OK       %s\n' "$what"; else printf '   PROBLEM  %s\n' "$what"; problems=$((problems + 1)); fi
}
has_python() { [ -x "$PYTHON_EXE" ]; }
is_pebbl_clone() { [ "$(git -C "$PEBBL_DIR" remote get-url origin 2> /dev/null | tr -d '\r')" = "$PEBBL_REPO" ]; }
git_may_update() { git config --system --get-all safe.directory 2> /dev/null | tr -d '\r' | grep -qxF "$(mixpath "$PEBBL_DIR")"; }
users_may_write() { icacls "$(winpath "$PEBBL_DIR")" 2> /dev/null | grep -qiF 'Users:(OI)(CI)(M)'; }
env_from_all_users_python() {
  [ -f "$ENV/pyvenv.cfg" ] && [ "$(lower "$(sed -n 's/^home *= *//p' "$ENV/pyvenv.cfg" | tr -d '\r')")" = "$(lower "$want_home")" ]
}
packages_installed() { cmp -s "$PEBBL_DIR/requirements.txt" "$ENV/requirements.installed.txt"; }
parts_load() { [ -x "$ENV/Scripts/python.exe" ] && "$ENV/Scripts/python.exe" -c "import mne, neurokit2, PyQt6"; }
has_icon() { [ -f "$SHORTCUT" ]; }

if [ "$DRY_RUN" = 1 ]; then
  note "(Dry run: these show the computer as it is now.)"
fi
check "Python ${PYTHON_VERSION%.*} for all users" has_python
check "PEBBL downloaded from $PEBBL_REPO" is_pebbl_clone
check "Git may update it from any account" git_may_update
check "Every user may write to it" users_may_write
check "Environment made from the all-users Python" env_from_all_users_python
check "Packages installed" packages_installed
check "PEBBL's parts load" parts_load
check "Desktop icon" has_icon

if [ "$problems" = 0 ]; then
  printf '\nPEBBL is set up on this computer for every user.\n'
  printf 'Test it: sign in as an RA (not an administrator), double-click PEBBL on the desktop, and look for\n'
  printf '"PEBBL is up to date" in the terminal before the form opens.\n'
else
  printf '\n%s check(s) above need attention.%s\n' "$problems" "$([ "$DRY_RUN" = 1 ] && echo ' (Expected in a dry run on a computer not set up yet.)')"
  [ "$DRY_RUN" = 1 ] || exit 1
fi
