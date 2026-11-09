@echo off
setlocal
if not exist setup.py (
  echo ERROR: Run this batch from the V2.2 project root.
  exit /b 2
)
py -3.12 "%~dp0restore_v22_overlay.py" --root "%CD%"
exit /b %ERRORLEVEL%
