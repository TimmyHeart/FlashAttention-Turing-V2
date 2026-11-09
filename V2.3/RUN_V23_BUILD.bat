@echo off
setlocal
rem Run this script from a FRESH CMD in the flash-attention-turing project root.
rem No parenthesized IF blocks are used, avoiding CMD parsing problems with (x86) paths.

if not exist setup.py goto :not_project

set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" goto :vcvars_missing

call "%VCVARS%" -vcvars_ver=14.29
set "VCVARS_RC=%ERRORLEVEL%"
if not "%VCVARS_RC%"=="0" goto :vcvars_failed

set "DISTUTILS_USE_SDK=1"
set "CUDA_HOME=E:\lung tung\Pythontest\cuda130-toolkit\nvidia\cu13"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set "TORCH_CUDA_ARCH_LIST=7.5"
set "MAX_JOBS=2"

rem Integrated experiment: native D32 + SEPV + FASTSM.
rem A fused-pad path remains in source as fallback; native D32 is selected here.
set "FLASH_TURING_NATIVE_D32=1"
set "FLASH_TURING_SEPV=1"
set "FLASH_TURING_FASTSM=1"
rem V2.3: also compile kBlockM=64 and kBlockM=32 forward kernels (use 1 for 64 only; build time grows).
set "FLASH_TURING_SMALLM=2"
set "FLASH_TURING_CAUSAL_REVERSE=1"

if exist build rmdir /s /q build
if errorlevel 1 goto :clean_failed

py -3.12 -m pip install --no-deps --no-build-isolation -v .
set "BUILD_RC=%ERRORLEVEL%"
if not "%BUILD_RC%"=="0" goto :build_failed

echo.
echo BUILD SUCCEEDED.
exit /b 0

:not_project
echo ERROR: Current directory is not the flash-attention-turing project root. setup.py was not found.
exit /b 2

:vcvars_missing
echo ERROR: Cannot find Visual Studio environment script:
echo "%VCVARS%"
exit /b 3

:vcvars_failed
echo ERROR: vcvars64.bat failed with exit code %VCVARS_RC%.
exit /b %VCVARS_RC%

:clean_failed
echo ERROR: Failed to remove the previous build directory.
exit /b 4

:build_failed
echo ERROR: pip build/install failed with exit code %BUILD_RC%.
exit /b %BUILD_RC%
