@echo off
setlocal EnableExtensions
if not exist setup.py (
  echo ERROR: Run from the project root.
  exit /b 2
)
set "PYTHONPATH=%CD%;%~dp0;%PYTHONPATH%"
if "%~1"=="" (set "TAG=full") else (set "TAG=%~1")
echo [1/2] B=1 H=4, D32/64/96/128, S=128/256/512, causal/non-causal (plus D96 M64 N128)
py -3.12 "%~dp0tools\bench_tiles_v23.py" --output "v23_tiles_b1h4_%TAG%.csv" > "v23_tiles_b1h4_%TAG%.log" 2>&1
if errorlevel 1 (echo ERROR: see v23_tiles_b1h4_%TAG%.log & exit /b 11)
echo [2/2] B=2 H=8, S=128/256/512/1024 (plus D96 M64 N128)
py -3.12 "%~dp0tools\bench_tiles_v23.py" --batch 2 --heads 8 --seqs 128 256 512 1024 --output "v23_tiles_b2h8_%TAG%.csv" > "v23_tiles_b2h8_%TAG%.log" 2>&1
if errorlevel 1 (echo ERROR: see v23_tiles_b2h8_%TAG%.log & exit /b 12)
echo Done. Send back the two .log files.
exit /b 0
