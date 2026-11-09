@echo off
setlocal
if not exist setup.py (
  echo ERROR: Run this batch from a FlashAttention Turing V2 project root containing setup.py.
  exit /b 2
)
py -3.12 "%~dp0apply_v22_overlay.py" --root "%CD%"
exit /b %ERRORLEVEL%
