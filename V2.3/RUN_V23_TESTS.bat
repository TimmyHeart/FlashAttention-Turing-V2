@echo off
setlocal
if not exist setup.py (
  echo ERROR: Run from the project root.
  exit /b 2
)
set "PYTHONPATH=%CD%;%~dp0;%PYTHONPATH%"
echo === V2.3 tile correctness (kBlockM 128/64/32 + auto, ragged/causal/GQA/varlen/graph) ===
py -3.12 "%~dp0validation\test_v23_tiles.py"
if errorlevel 1 exit /b %ERRORLEVEL%
echo === V2 regression suite (public API, default auto tile) ===
py -3.12 "%~dp0validation\test_flashattn_all_head_dims_v1.py" --no-bench
if errorlevel 1 exit /b %ERRORLEVEL%
py -3.12 "%~dp0validation\test_flashattn_guardrails.py"
if errorlevel 1 exit /b %ERRORLEVEL%
echo ALL V2.3 TESTS PASSED.
exit /b 0
