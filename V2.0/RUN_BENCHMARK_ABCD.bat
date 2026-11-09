@echo off
setlocal
rem Run only after RUN_TESTS_ABCD.bat passes.
if not exist setup.py exit /b 2
set "PYTHONPATH=%CD%;%PYTHONPATH%"
py -3.12 tools\bench_v2.py --tag abcd_full --dims 32 64 96 128 --seqs 128 256 512 1024 --batches 1 2 4 --bwd > abcd_full_benchmark.txt 2>&1
exit /b %errorlevel%
