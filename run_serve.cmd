@echo off
REM Keep the demo server alive: serve.py with the Python coder on
REM http://127.0.0.1:8000. Registered as the scheduled task anulm_serve,
REM firing every 30 minutes; a firing while the server is up is a no-op, a
REM firing after a crash or reboot brings it back. The guard matches the port,
REM so a second serve.py on another port (a test, a second checkpoint) is ignored. The server itself runs
REM in its own console via `start` (see TASKS.md for why), so nothing in an
REM interactive shell can Ctrl+C it.
REM
REM To stop serving for good:
REM   schtasks /change /tn anulm_serve /disable
REM   powershell "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | ? { $_.CommandLine -like '*serve.py*' } | % { Stop-Process -Id $_.ProcessId }"
cd /d "%~dp0"
powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { $_.CommandLine -like '*serve.py*' -and $_.CommandLine -like '*--port 8000*' }) { exit 1 }"
if errorlevel 1 exit /b 0
echo %date% %time% serve start >> serve_guard.log
start "anulm-serve" /min "%~dp0_serve_run.cmd"
exit /b 0
