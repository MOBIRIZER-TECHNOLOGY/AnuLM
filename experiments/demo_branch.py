"""
A demo-ready base without waiting for the main run: branch at 700k, decay, tune.

    python experiments/demo_branch.py

The main base-v2 run uses a warmup-stable-decay schedule, so most of what the
final decay buys can be had early by decaying a COPY of a mid-run checkpoint.

  1. wait for the main run's "eval @ 699999" (its resume point is saved first)
  2. pause it: disable the anulm_v2 task, stop that train.py
  3. copy its resume point to ckpt_demo_v2.pt.last and train the copy
     700k -> 800k with the learning rate decaying linearly to its floor
     (--steps 800000 --decay-frac 0.125); ~8.5 h
  4. instruction-tune the decayed base on make_chat.py's 23k pairs (chat)
  5. fit the retrieval reader (rc_pointer.py) on the decayed base and score it
  6. ALWAYS re-enable and restart anulm_v2, so the main run resumes from
     ckpt_base_v2.pt.last untouched (it finishes ~15 h later than planned)

Every step has a hard time cap and its own log under logs/demo_branch/.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs" / "demo_branch"
PY = sys.executable
MAIN_LOG = ROOT / "v2_train_run.log"
PAUSE_AT = "eval @ 699999"

TRAIN_ARGS = ["--preset", "350m", "--device", "cuda", "--moe-impl", "grouped", "--data", "data/v2/meta.json",
              "--block-size", "1024", "--batch-size", "4", "--lr", "6e-4", "--min-lr", "6e-5",
              "--warmup", "2000", "--schedule", "wsd", "--eval-every", "10000", "--eval-windows", "400",
              "--log-every", "1000", "--seed", "2026", "--seq-balance-alpha", "1e-4", "--router-z-alpha", "1e-3",
              "--cfg", "num_experts=24", "num_experts_per_tok=4", "num_shared_experts=0",
              "moe_intermediate_size=192", "sliding_window=256", "max_window_layers=10", "bias_update_rate=3e-3"]


def log(m):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOGS / "pipeline.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ps(cmd: str) -> str:
    return subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True).stdout


def train_py_running() -> bool:
    return "yes" in ps("if (Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
                       "Where-Object { $_.CommandLine -like '*train.py*' }) { 'yes' }")


def run(name: str, args: list[str], cap_h: float) -> int | str:
    lf = LOGS / f"{name}.log"
    log(f"--- {name} (cap {cap_h} h) -> {lf.name}")
    with open(lf, "w", encoding="utf-8") as f:
        p = subprocess.Popen([PY, "-u"] + args, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
        try:
            rc = p.wait(timeout=cap_h * 3600)
        except subprocess.TimeoutExpired:
            p.terminate()
            p.wait(120)
            rc = "cap reached"
    tail = [l for l in lf.read_text(encoding="utf-8", errors="replace").splitlines()
            if any(k in l for k in ("eval @", "done in", "held-out", "F1", "demo questions", "Traceback", "Error"))]
    for l in tail[-12:]:
        log("    " + l.strip()[:220])
    log(f"--- {name}: exit {rc}")
    return rc


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    log(f"waiting for '{PAUSE_AT}' in {MAIN_LOG.name}")
    while PAUSE_AT not in MAIN_LOG.read_text(encoding="utf-8", errors="replace"):
        time.sleep(60)
    try:
        # 2. pause the main run (its resume point was written before the eval line)
        log("pausing the main run: disable anulm_v2, stop its train.py")
        subprocess.run(["schtasks", "/change", "/tn", "anulm_v2", "/disable"], capture_output=True)
        ps("Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
           "Where-Object { $_.CommandLine -like '*train.py*' -and $_.CommandLine -like '*ckpt_base_v2*' } | "
           "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }")
        for _ in range(60):
            if not train_py_running():
                break
            time.sleep(5)
        time.sleep(20)
        # 3. decay a copy
        shutil.copy2(ROOT / "ckpt_base_v2.pt.last", ROOT / "ckpt_demo_v2.pt.last")
        shutil.copy2(ROOT / "ckpt_base_v2.pt", ROOT / "ckpt_demo_v2.pt")
        log("copied the 700k resume point to ckpt_demo_v2.pt.last")
        run("1_decay", ["train.py", *TRAIN_ARGS, "--out", "ckpt_demo_v2.pt", "--steps", "800000",
                        "--decay-frac", "0.125", "--resume"], 11.0)
        # 4. chat
        run("2_chat_sft", ["finetune.py", "--ckpt", "ckpt_demo_v2.pt", "--qa", "data/chat_train.jsonl",
                           "--heldout", "data/chat_heldout.jsonl", "--out", "ckpt_chat_v2.pt", "--epochs", "2",
                           "--batch-size", "4", "--lr", "5e-5", "--grad-ckpt", "--eval-every", "500",
                           "--log-every", "100"], 3.0)
        run("3_chat_samples", [str(ROOT / "experiments" / "chat_samples.py")], 0.5)
        # 5. retrieval reader
        run("4_reader", ["rc_pointer.py", "train", "--ckpt", "ckpt_demo_v2.pt", "--out", "ckpt_rc_pointer_v2.pt",
                         "--epochs", "1", "--batch-size", "8", "--lr", "5e-5", "--eval-every", "1000"], 2.0)
        run("5_reader_eval", ["rc_pointer.py", "eval", "--ckpt", "ckpt_rc_pointer_v2.pt", "--n", "300"], 1.0)
    finally:
        # 6. always hand the GPU back to the main run
        subprocess.run(["schtasks", "/change", "/tn", "anulm_v2", "/enable"], capture_output=True)
        subprocess.run(["schtasks", "/run", "/tn", "anulm_v2"], capture_output=True)
        log("=== main run re-enabled and restarted (anulm_v2); DEMO BRANCH DONE ===")


if __name__ == "__main__":
    main()
