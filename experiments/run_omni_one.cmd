@echo off
rem Start experiments/omni_one.py detached; output in logs\omni_one\.
rem Safe to run again at any time: finished steps are skipped, training resumes.
rem Add --publish to upload to Hugging Face when every release gate passes.
cd /d "%~dp0.."
if not exist logs\omni_one mkdir logs\omni_one
start "omni_one" /min cmd /c python -u experiments\omni_one.py %* ^>^> logs\omni_one\stdout.log 2^>^&1
