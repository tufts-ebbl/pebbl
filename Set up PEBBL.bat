@echo off
rem One-time PEBBL setup on Windows: double-click this file.
rem It makes PEBBL's own Python environment (annotate_env) and installs what PEBBL needs.
rem (PEBBL_NO_PAUSE=1 skips the "press any key" pauses; for automated tests only.)
setlocal
cd /d "%~dp0"
echo.
echo Setting up PEBBL on this computer (one time only).
echo This downloads about 1 GB and takes 5-15 minutes. Please keep this window open.
echo.
py -3.11 --version >nul 2>nul
if errorlevel 1 goto nopython
if not exist "annotate_env\Scripts\python.exe" (
  echo Creating PEBBL's own Python environment...
  py -3.11 -m venv annotate_env
  if errorlevel 1 goto failed
)
"annotate_env\Scripts\python.exe" -m pip install --disable-pip-version-check --upgrade pip
if errorlevel 1 goto failed
"annotate_env\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto failed
copy /y requirements.txt "annotate_env\requirements.installed.txt" >nul
echo.
echo PEBBL is set up. To start it, double-click "Start PEBBL.bat" in this folder.
echo.
if not defined PEBBL_NO_PAUSE pause
exit /b 0

:nopython
echo Python 3.11 isn't installed on this computer (or it was installed without the "py" launcher).
echo Install it from https://www.python.org/downloads/release/python-3119/ as the PEBBL manual describes, then double-click this file again.
echo.
if not defined PEBBL_NO_PAUSE pause
exit /b 1

:failed
echo.
echo Setup didn't finish. Take a photo of this window and send it to the lab staff.
echo.
if not defined PEBBL_NO_PAUSE pause
exit /b 1
