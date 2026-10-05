@echo off
rem Double-click to start PEBBL on Windows. It checks for updates, then opens the PEBBL form.
rem The script is one parenthesized block, so cmd reads all of it before running anything:
rem an update that rewrites this file while it runs can't garble it.
(
  setlocal
  cd /d "%~dp0"
  if not exist "annotate_env\Scripts\python.exe" (
    echo PEBBL isn't set up on this computer yet. Double-click "Set up PEBBL.bat" first.
    echo.
    if not defined PEBBL_NO_PAUSE pause
    exit /b 1
  )
  "annotate_env\Scripts\python.exe" pebbl_launcher.py
  echo.
  echo PEBBL has closed. You can close this window.
  if not defined PEBBL_NO_PAUSE pause
  exit /b 0
)
