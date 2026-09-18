@echo off
REM The actual server process. Started by run_serve.cmd through `start` so it
REM has its own console. Do not call this directly; call run_serve.cmd.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
"%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" -u serve.py --ckpt ckpt_coder_sft.pt --host 127.0.0.1 --port 8000 >> serve.log 2>> serve.err
echo %date% %time% serve exited >> serve_guard.log
