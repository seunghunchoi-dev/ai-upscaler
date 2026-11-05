@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
if exist "runtime\pythonw.exe" (
    start "" "runtime\pythonw.exe" -m vup.gui %*
) else (
    start "" ".venv\Scripts\pythonw.exe" -m vup.gui %*
)
