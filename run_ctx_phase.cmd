@echo off
REM Keep the long-context run alive across session ends, reboots and crashes.
REM   run_ctx_phase.cmd 10000   train from the last save to step 10000
REM Registered as the scheduled task anulm_ctx, firing every 30 min. A firing
REM while training is healthy is a no-op; a firing after a crash resumes from
REM ckpt_ctx2k.pt.last, losing at most one --eval-every interval.
cd /d "%~dp0"

if "%1"=="" goto :guard
if exist "ctx_%1_done.marker" goto :guard
findstr /c:"phase done at step %1" /c:"| ckpt ckpt_ctx2k.pt" ctx_train.log >nul 2>&1 || goto :guard
echo %date% %time% CTX %1 COMPLETE > "ctx_%1_done.marker"
echo %date% %time% CTX %1 COMPLETE >> ctx_train_guard.log
copy "ctx_%1_done.marker" "%USERPROFILE%\Desktop\anulm_ctx_%1_done.txt" >nul 2>&1
powershell -NoProfile -Command "1..6 | ForEach-Object { [console]::beep(880,400); Start-Sleep -Milliseconds 200 }"

:guard
powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { $_.CommandLine -like '*train.py*' }) { exit 1 }"
if errorlevel 1 (
  echo %date% %time% already training, nothing to do >> ctx_train_guard.log
  exit /b 0
)
echo %date% %time% launching detached, target %1 >> ctx_train_guard.log
start "anulm-ctx" /min "%~dp0_ctx_run.cmd" %1
exit /b 0
