@echo off
REM One refresh of TASKS.md and the assistant's live-state memory file, then
REM exit. Registered as the scheduled task anulm_refresh, every 30 min.
REM
REM Deliberately has NO guard against a concurrent run_updater.cmd: that
REM wrapper's long-running `--loop` process can be an older build holding
REM stale code in memory (it survives `schtasks /end`), and the point of this
REM task is to refresh regardless. Both writers use os.replace, so the worst
REM case is last-writer-wins on identical content.
REM
REM Separate log files are NOT cosmetic: cmd's `>>` holds an exclusive handle,
REM so sharing update_tasks.log with the looping process makes every run here
REM die with "The process cannot access the file" (exit 1).
cd /d "%~dp0"
"%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" -u update_tasks.py >> refresh.log 2>> refresh.err
