@echo off
setlocal EnableExtensions
if not exist setup.py (
  echo ERROR: Run from the V2.2 FlashAttention project root.
  exit /b 2
)
set "PYTHONPATH=%CD%;%~dp0;%PYTHONPATH%"
if "%~1"=="" (set "TAG=full") else (set "TAG=%~1")
(
  echo ==== V2.2 host-overhead benchmark environment ====
  echo Date: %DATE%  Time: %TIME%  Tag: %TAG%
  py -3.12 -c "import torch; import flash_attn_turing as ext; import flash_attention_interface as fi; print('PyTorch:',torch.__version__,'CUDA:',torch.version.cuda); print('GPU:',torch.cuda.get_device_name(),'capability:',torch.cuda.get_device_capability()); print('extension:',ext.__file__); print('interface:',fi.__file__); print('fwd_into:',hasattr(ext,'fwd_into'),'fastpath:',fi._FASTPATH)"
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free --format=csv
) > v22_benchmark_env_%TAG%.txt 2>&1
if errorlevel 1 (echo ERROR collecting environment. See v22_benchmark_env_%TAG%.txt & exit /b 3)
echo [1/2] Quick run: all dimensions, S=128/256, non-causal, graph N=1/4
py -3.12 "%~dp0tools\bench_host_overhead_v22.py" --quick --output "v22_host_overhead_quick_%TAG%.csv" > "v22_host_overhead_quick_%TAG%.log" 2>&1
if errorlevel 1 (echo ERROR: see v22_host_overhead_quick_%TAG%.log & exit /b 11)
echo [2/2] Full matched run: D32/64/96/128, S=128/256/512, causal/non-causal, graph N=1/4/16
py -3.12 "%~dp0tools\bench_host_overhead_v22.py" --dims 32 64 96 128 --seqs 128 256 512 --causal 0 1 --graph-ops 1 4 16 --output "v22_host_overhead_full_%TAG%.csv" > "v22_host_overhead_full_%TAG%.log" 2>&1
if errorlevel 1 (echo ERROR: see v22_host_overhead_full_%TAG%.log & exit /b 12)
echo Done: inspect v22_benchmark_env_%TAG%.txt, quick/full CSV and logs.
exit /b 0
