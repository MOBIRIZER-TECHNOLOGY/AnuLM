@echo off
REM The actual training invocation. Started by run_coder_phase.cmd through
REM `start`, which gives it its OWN console: a Ctrl+C delivered to the
REM console shared by the scheduled task and any interactive shell then
REM cannot reach it. Do not call this directly; call run_coder_phase.cmd.
cd /d "%~dp0"
echo %date% %time% train start, target %1 >> coder_train_guard.log
"C:\Program Files\Git\usr\bin\bash.exe" experiments/coder_train.sh %1 >> coder_train_phase1.log 2>> coder_train_phase1.err
echo %date% %time% train exited, target %1 >> coder_train_guard.log
