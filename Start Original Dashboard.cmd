@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" launch_dashboard.py main %*
) else (
  python launch_dashboard.py main %*
)
if errorlevel 1 pause
