@echo off
setlocal
if not exist setup.py (
  echo ERROR: Run from the V2.2 FlashAttention project root.
  exit /b 2
)
set "PYTHONPATH=%CD%;%~dp0;%PYTHONPATH%"
echo ============================================================
echo V2.2 Python dispatch mock + host-path correctness + frozen V2 regressions
echo ============================================================
py -3.12 "%~dp0validation\test_v22_python_mock.py"
if errorlevel 1 exit /b %ERRORLEVEL%
py -3.12 "%~dp0validation\test_v22_hostpath.py"
if errorlevel 1 exit /b %ERRORLEVEL%
py -3.12 "%~dp0validation\test_flashattn_all_head_dims_v1.py" --no-bench
if errorlevel 1 exit /b %ERRORLEVEL%
py -3.12 "%~dp0validation\test_flashattn_guardrails.py"
if errorlevel 1 exit /b %ERRORLEVEL%
echo ALL V2.2 + V2 REGRESSION TESTS PASSED.
exit /b 0
