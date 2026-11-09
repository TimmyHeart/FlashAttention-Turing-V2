@echo off
setlocal
if not exist setup.py (
  echo ERROR: Run from the V2.3-applied FlashAttention project root.
  exit /b 2
)
py -3.12 "%~dp0apply_v23_dispatch_hotfix.py" --root "%CD%"
exit /b %ERRORLEVEL%
