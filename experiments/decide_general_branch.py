"""
Train the general AnuLM-Decide and test BANKING77 zero-shot, in a short pause.

    python experiments/decide_general_branch.py        # launched detached

Waits for the main base-v2 run's next "eval @" line (its resume point has just
been written), pauses it, trains decide.py's general version on the decayed
demo base (five label sets, BANKING77 and banking-like intents excluded),
evaluates BANKING77 zero-shot -- the setting in which Jev scored 80.1% --
and ALWAYS re-enables and restarts the main run.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs" / "decide_general"
PY = sys.executable
MAIN_LOG = ROOT / "v2_train_run.log"


def log(m):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOGS / "pipeline.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ps(cmd):
    return subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True).stdout


def evals():
    return MAIN_LOG.read_text(encoding="utf-8", errors="replace").count("eval @")


def run(name, args, cap_h):
    lf = LOGS / f"{name}.log"
    log(f"--- {name} (cap {cap_h} h)")
    with open(lf, "w", encoding="utf-8") as f:
        p = subprocess.Popen([PY, "-u"] + args, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT,
                             env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
        try:
            rc = p.wait(timeout=cap_h * 3600)
        except subprocess.TimeoutExpired:
            p.terminate()
            p.wait(120)
            rc = "cap reached"
    for l in lf.read_text(encoding="utf-8", errors="replace").splitlines():
        if any(k in l for k in ("eval @", "done in", "accuracy", "answers", "latency", "general training",
                                "Traceback", "Error")):
            log("    " + l.strip()[:220])
    log(f"--- {name}: exit {rc}")


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    n0 = evals()
    log(f"waiting for the main run's next save (eval #{n0 + 1})")
    while evals() <= n0:
        time.sleep(30)
    try:
        subprocess.run(["schtasks", "/change", "/tn", "anulm_v2", "/disable"], capture_output=True)
        ps("Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
           "Where-Object { $_.CommandLine -like '*train.py*' -and $_.CommandLine -like '*ckpt_base_v2*' } | "
           "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }")
        time.sleep(30)
        log("main run paused right after its save")
        run("1_train_general", ["decide.py", "train-general", "--ckpt", "ckpt_demo_v2.pt",
                                "--out", "ckpt_decide_general.pt", "--epochs", "1", "--batch-size", "8",
                                "--lr", "3e-5", "--eval-every", "1000"], 1.5)
        run("2_zero_shot_banking77", ["decide.py", "eval", "--ckpt", "ckpt_decide_general.pt"], 0.5)
        run("3_samples", [str(ROOT / "experiments" / "decide_samples.py"), "ckpt_decide_general.pt"], 0.3)
    finally:
        subprocess.run(["schtasks", "/change", "/tn", "anulm_v2", "/enable"], capture_output=True)
        subprocess.run(["schtasks", "/run", "/tn", "anulm_v2"], capture_output=True)
        log("=== main run re-enabled and restarted; DECIDE GENERAL DONE ===")


if __name__ == "__main__":
    main()
