@echo off
rem Installs NVIDIA CUDA support into the portable runtime (internet required, ~1.5GB).
rem Without this, NVIDIA/AMD GPUs still work through DirectML.
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PY=runtime\python.exe
if not exist "%PY%" set PY=.venv\Scripts\python.exe

echo [1/3] Checking NVIDIA driver...
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
if errorlevel 1 (
    echo.
    echo NVIDIA driver not found. Install the latest driver from nvidia.com first, then run this again.
    pause
    exit /b 1
)

rem Keep the same onnxruntime version as the bundled DirectML build, so it matches numpy/onnx in the runtime
set ORTVER=
for /f "usebackq delims=" %%v in (`"%PY%" -c "import importlib.metadata as m; print(m.version('onnxruntime-directml'))" 2^>nul`) do set ORTVER=%%v
set PIN=
if defined ORTVER set PIN===%ORTVER%

echo [2/3] Removing DirectML build of onnxruntime...
"%PY%" -m pip uninstall -y onnxruntime-directml onnxruntime onnxruntime-gpu

echo [3/3] Installing onnxruntime-gpu%PIN% with CUDA/cuDNN libraries...
"%PY%" -m pip install --no-warn-script-location "onnxruntime-gpu[cuda,cudnn]%PIN%"
if errorlevel 1 if defined PIN (
    echo Version %ORTVER% not available, trying the latest version...
    "%PY%" -m pip install --no-warn-script-location "onnxruntime-gpu[cuda,cudnn]"
)
if errorlevel 1 (
    echo.
    echo Failed. Restoring DirectML version...
    "%PY%" -m pip uninstall -y onnxruntime-gpu
    "%PY%" -m pip install --no-warn-script-location "onnxruntime-directml%PIN%"
    pause
    exit /b 1
)
echo.
"%PY%" -m vup --devices
echo.
echo Done. If an "(CUDA)" device is listed above, leave the device on Auto or select it in the program.
echo If it is not listed, update the NVIDIA driver - CUDA 12 needs driver 528 or newer.
pause
