@echo off
REM The sweep, in its own console so a Ctrl+C in the interactive one cannot
REM reach it. Started by run_sweep.cmd; do not call directly.
cd /d "%~dp0"
echo %date% %time% sweep start >> sweep_guard.log
"%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" -u tools_sweep.py --device cuda --max-tokens 96 >> sweep.log 2>> sweep.err
echo %date% %time% sweep exited >> sweep_guard.log
