@echo off
REM Keep the prompt sweep alive across session ends and crashes.
REM Registered as the scheduled task anulm_sweep, firing every 30 min. A
REM firing while it is running is a no-op; otherwise it resumes from what is
REM already in sweep/<segment>.jsonl. See TASKS.md.
cd /d "%~dp0"
if exist "sweep\summary.json" (
  echo %date% %time% already complete, nothing to do >> sweep_guard.log
  exit /b 0
)
powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { $_.CommandLine -like '*tools_sweep*' }) { exit 1 }"
if errorlevel 1 (
  echo %date% %time% already sweeping, nothing to do >> sweep_guard.log
  exit /b 0
)
echo %date% %time% launching detached >> sweep_guard.log
start "anulm-sweep" /min "%~dp0_sweep_run.cmd"
exit /b 0
