@echo off
REM Keep a coder pretraining phase alive, and alert once when it completes.
REM   run_coder_phase.cmd 250000   train from the last save to step 250000
REM   run_coder_phase.cmd          no argument: run to the end (700k)
REM
REM Registered as the scheduled task nanosarvam_coder, firing every 30 min.
REM Each firing: alert if the phase just finished, then start training if it
REM is not already alive. The training itself is launched with `start`, in a
REM console of its own -- a task run with an interactive token otherwise
REM shares the interactive console, and a Ctrl+C there kills it
REM (STATUS_CONTROL_C_EXIT, -1073741510). That cost two restarts on
REM 2026-09-10/11. See TASKS.md.
cd /d "%~dp0"

if "%1"=="" goto :guard
if exist "phase_%1_done.marker" goto :guard
REM Two ways the trainer signals it is finished: "phase done at step N"
REM when --stop-at is hit below --steps, and "| ckpt ckpt_coder.pt" on the
REM final line when the whole run completes. Match either, or the alert
REM never fires on the last phase.
findstr /c:"phase done at step %1" /c:"| ckpt ckpt_coder.pt" coder_train_phase1.log >nul 2>&1 || goto :guard
echo %date% %time% PHASE %1 COMPLETE > "phase_%1_done.marker"
echo %date% %time% PHASE %1 COMPLETE >> coder_train_guard.log
copy "phase_%1_done.marker" "%USERPROFILE%\Desktop\nanosarvam_phase_%1_done.txt" >nul 2>&1
powershell -NoProfile -Command "1..6 | ForEach-Object { [console]::beep(880,400); Start-Sleep -Milliseconds 200 }"

:guard
powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { $_.CommandLine -like '*train.py*' }) { exit 1 }"
if errorlevel 1 (
  echo %date% %time% already training, nothing to do >> coder_train_guard.log
  exit /b 0
)
echo %date% %time% launching detached, target %1 >> coder_train_guard.log
start "nanosarvam-coder" /min "%~dp0_coder_run.cmd" %1
exit /b 0
