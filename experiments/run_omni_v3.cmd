@echo off
rem Fired every 30 min by the scheduled task anulm_omni_v3. omni_v3.py itself
rem exits at once if a copy is already running or logs\omni_v3\done.marker
rem exists, so repeated firings are harmless. start gives it its own console,
rem out of reach of a Ctrl+C in the interactive one (TASKS.md).
rem Install: copy this and omni_v3.py to C:\workspace\AnuLM\logs\omni_v3\.
cd /d C:\workspace\AnuLM
if exist logs\omni_v3\done.marker exit /b 0
start "anulm-omni-v3" /min python -u logs\omni_v3\omni_v3.py --root %~dp0..
