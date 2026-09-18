@echo off
REM Hourly refresh of TASKS.md's live-status block, if not already running.
REM Registered as the scheduled task anulm_updater. See TASKS.md.
cd /d "%~dp0"
powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { $_.CommandLine -like '*update_tasks.py*' }) { exit 1 }"
if errorlevel 1 exit /b 0
"%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" -u update_tasks.py --loop 3600 >> update_tasks.log 2>> update_tasks.err
