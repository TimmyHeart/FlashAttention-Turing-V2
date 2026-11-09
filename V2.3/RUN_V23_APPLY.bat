@echo off
setlocal
if not exist setup.py (
  echo ERROR: Run from a project root that is V2 + V2.2 applied ^(contains setup.py^).
  exit /b 2
)
py -3.12 "%~dp0apply_v23_overlay.py" --root "%CD%"
exit /b %ERRORLEVEL%
