@echo off
REM The actual training invocation for the long-context extension run
REM (docs/RESULTS.md section 27, TODO item "Longer context"). Started by
REM run_ctx_phase.cmd through `start`, which gives it its OWN console -- a
REM Ctrl+C in the console the scheduled task shares with any interactive
REM shell then cannot reach it. That trap cost two restarts on 2026-09-10/11;
REM TASKS.md has the story. Do not call this directly.
cd /d "%~dp0"
echo %date% %time% ctx train start, target %1 >> ctx_train_guard.log
"%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" -u train.py --preset 350m --grad-ckpt ^
  --moe-impl grouped --data data/ctx_mix.multi32k.bin --block-size 2048 ^
  --batch-size 2 --grad-accum 4 --steps %1 --lr 1e-4 --min-lr 1e-5 --warmup 200 --yarn ^
  --sliding-window 256 --max-window-layers 10 --seq-balance-alpha 1e-4 --router-z-alpha 1e-3 ^
  --cfg num_experts=24 num_experts_per_tok=4 num_shared_experts=0 moe_intermediate_size=192 ^
        bias_update_rate=3e-3 yarn_original_context=512 yarn_factor=4.0 ^
  --eval-every 250 --eval-windows 256 --out ckpt_ctx2k.pt --resume >> ctx_train.log 2>> ctx_train.err
echo %date% %time% ctx train exited, target %1 >> ctx_train_guard.log
