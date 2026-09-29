@echo off
REM The base-v2 training invocation, started by run_v2_phase.cmd through
REM `start` so it has its own console. Arguments match experiments/v2_train.sh.
REM Do not call this directly.
cd /d "%~dp0"
set RESUME=
if exist "ckpt_base_v2.pt.last" set RESUME=--resume
echo %date% %time% v2 train start %RESUME% >> v2_train_guard.log
"%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" -u train.py --preset 350m --device cuda ^
  --moe-impl grouped --data data/v2/meta.json --out ckpt_base_v2.pt ^
  --block-size 1024 --batch-size 4 --steps 2750000 --lr 6e-4 --min-lr 6e-5 --warmup 2000 ^
  --schedule wsd --decay-frac 0.2 --eval-every 10000 --eval-windows 400 --log-every 1000 ^
  --seed 2026 --seq-balance-alpha 1e-4 --router-z-alpha 1e-3 ^
  --cfg num_experts=24 num_experts_per_tok=4 num_shared_experts=0 moe_intermediate_size=192 ^
        sliding_window=256 max_window_layers=10 bias_update_rate=3e-3 ^
  %RESUME% >> v2_train_run.log 2>> v2_train_run.err
echo %date% %time% v2 train exited >> v2_train_guard.log
