@echo off
rem Command line mode. Example:
rem   명령줄.bat input.mp4 --quality "일반 · 균형" --style 뽀샤시 --scale 2 --fps x2
rem   명령줄.bat --devices
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
if exist "runtime\python.exe" (
    "runtime\python.exe" -m vup %*
) else (
    ".venv\Scripts\python.exe" -m vup %*
)
