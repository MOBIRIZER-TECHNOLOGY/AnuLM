@echo off
REM Instruction-tune the coder checkpoint, then pass@1 on both benchmarks.
REM
REM MUST be started by the Task Scheduler (task nanosarvam_sft), not from a
REM terminal and not via `start` from one. `start` gives a new console, which
REM defeats Ctrl+C, but the process is still a DESCENDANT of the launching
REM session and dies with it when that session's job object closes. Only a
REM task escapes that, because its parent is the scheduler service.
REM Losing this cost 2 h 20 min of idle GPU on 2026-09-12.
REM
REM The task fires every 30 minutes so that a killed run restarts itself.
REM Both guards below exist because of that: without the first, a firing
REM during a healthy run would start a SECOND fine-tune writing the same
REM checkpoint; without the second, it would re-tune forever after finishing.
cd /d "%~dp0"

if exist "sft_done.marker" exit /b 0

powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { $_.CommandLine -like '*finetune.py*' -or $_.CommandLine -like '*eval_code.py*' }) { exit 1 }"
if errorlevel 1 (
  echo %date% %time% sft already running, nothing to do >> coder_train_guard.log
  exit /b 0
)

echo %date% %time% sft start >> coder_train_guard.log
set EX_CAP=0
"C:\Program Files\Git\usr\bin\bash.exe" experiments/coder_sft.sh >> coder_sft.log 2>> coder_sft.err
echo %date% %time% sft exited >> coder_train_guard.log

REM coder_sft.sh ends by printing the two pass@1 tables; that line is the
REM only reliable "the whole probe finished" signal, since the tune alone
REM leaves no distinct marker.
findstr /c:"reference: CodeParrot" eval_humaneval_sft.log >nul 2>&1 || exit /b 0
findstr /c:"reference: CodeParrot" eval_mbpp_sft.log >nul 2>&1 || exit /b 0
echo %date% %time% SFT PROBE COMPLETE > "sft_done.marker"
echo %date% %time% SFT PROBE COMPLETE >> coder_train_guard.log
copy "sft_done.marker" "%USERPROFILE%\Desktop\nanosarvam_sft_probe_done.txt" >nul 2>&1
powershell -NoProfile -Command "1..6 | ForEach-Object { [console]::beep(880,400); Start-Sleep -Milliseconds 200 }"
