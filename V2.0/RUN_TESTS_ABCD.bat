@echo off
setlocal
rem Run from the flash-attention-turing project root after BUILD_ABCD_FULL_SM75.bat succeeds.
if not exist setup.py (
  echo ERROR: Current directory is not the flash-attention-turing project root.
  exit /b 2
)
set "PYTHONPATH=%CD%;%PYTHONPATH%"
echo ============================================================
echo A+B+C+D integrated correctness run (72 cases + guardrails)
echo ============================================================
py -3.12 "%~dp0validation\test_flashattn_all_head_dims_v1.py" --no-bench
if errorlevel 1 exit /b %errorlevel%
echo.
echo ============================================================
echo Supplemental guardrail run (Q/K mismatch, B>1, non-contiguous)
echo ============================================================
py -3.12 test_flashattn_guardrails.py
if errorlevel 1 exit /b %errorlevel%
echo.
echo ALL INTEGRATED CORRECTNESS TESTS PASSED.
