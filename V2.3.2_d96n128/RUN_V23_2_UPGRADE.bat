@echo off
setlocal
if not exist setup.py (
  echo ERROR: Run from the project root.
  exit /b 2
)
py -3.12 "%~dp0apply_v23_2_upgrade.py" --root "%CD%"
exit /b %ERRORLEVEL%
