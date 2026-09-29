@echo off
REM Keep base-v2 pretraining alive across session ends, reboots and crashes.
REM Registered as the scheduled task anulm_v2, firing every 30 min. A firing
REM while any train.py is running is a no-op; a firing after a stop resumes
REM from ckpt_base_v2.pt.last, losing at most one --eval-every interval
REM (10,000 steps, ~47 min). Same pattern as run_tb_phase.cmd.
cd /d "%~dp0"

if not exist "data\v2\meta.json" (
  echo %date% %time% corpus not built yet, nothing to do >> v2_train_guard.log
  exit /b 0
)
if exist "v2_done.marker" exit /b 0
findstr /c:"| ckpt ckpt_base_v2.pt" v2_train_run.log >nul 2>&1 && (
  echo %date% %time% BASE V2 COMPLETE > v2_done.marker
  echo %date% %time% BASE V2 COMPLETE >> v2_train_guard.log
  exit /b 0
)

powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { $_.CommandLine -like '*train.py*' }) { exit 1 }"
if errorlevel 1 (
  echo %date% %time% already training, nothing to do >> v2_train_guard.log
  exit /b 0
)
echo %date% %time% launching detached >> v2_train_guard.log
start "anulm-v2" /min "%~dp0_v2_run.cmd"
exit /b 0
