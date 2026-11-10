@echo off
setlocal
cd /d F:\ComfyUICU130\ComfyUI
if errorlevel 1 (
    echo [ERROR] Cannot enter F:\ComfyUICU130\ComfyUI
    pause
    exit /b 1
)

rem Keep CPU thread limits consistent with the previous stable launcher.
set "OMP_NUM_THREADS=4"
set "MKL_NUM_THREADS=4"
set "OPENBLAS_NUM_THREADS=4"
set "NUMEXPR_NUM_THREADS=4"
set "OMP_DYNAMIC=FALSE"
set "MKL_DYNAMIC=FALSE"

echo === ComfyUI CUDA Graph + FlashAttention Turing ===
echo Python: Python 3.12
echo PyTorch target: 2.11.0+cu130
echo GPU target: NVIDIA SM75 / Turing
echo Attention: Wan FlashAttention Turing CUDA Graph mode
echo ================================================
echo.

py -3.12 -c "import sys, torch, flash_attn_turing; print('Python:', sys.executable); print('PyTorch:', torch.__version__); print('Turing extension: import OK'); print('PyTorch intra-op threads:', torch.get_num_threads())"
if errorlevel 1 (
    echo [ERROR] Python, PyTorch or flash_attn_turing import check failed.
    pause
    exit /b 1
)
echo.

py -3.12 main.py ^
  --windows-standalone-build ^
  --enable-manager ^
  --use-turing-flash-attention ^
  --force-fp16 ^
  --fp16-unet ^
  --fp16-text-enc ^
  --fp16-intermediates

set "COMFYUI_EXITCODE=%ERRORLEVEL%"
if not "%COMFYUI_EXITCODE%"=="0" echo [ERROR] ComfyUI exited with code %COMFYUI_EXITCODE%.
pause
exit /b %COMFYUI_EXITCODE%
