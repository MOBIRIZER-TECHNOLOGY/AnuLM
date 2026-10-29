"""
Post-training on the final base (TASKS.md "After base v2", steps 2-6), unattended.

    python experiments/post_v2.py

  2. chat: full fine-tune of ckpt_base_v2.pt -> ckpt_chat_v3.pt, samples vs v2
  3. lookup reader -> ckpt_rc_pointer_v3.pt, SQuAD eval
  4. the LoRA comparison on ckpt_chat_v3.pt, BANKING77: runs A-F, then G-H
     (lr 1e-4 / 4e-4) on the best of A-F, then the full fine-tune reference
  5. the Hindi adapter (--task massive-hi) with the winning recipe; the
     winning banking run is the banking adapter
  6. Decide general (zero-shot) on the final base, scored on BANKING77

Nothing here touches the app or the demo checkpoints; step 8 (ship) is done
by hand after reading the results. Every step has a time cap and its own log
under logs/post_v2/; a failed step is logged and the rest still run.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs" / "post_v2"
PY = sys.executable
BASE = "ckpt_base_v2.pt"
CHAT = "ckpt_chat_v3.pt"

# run -> (extra decide.py args, adapter params in M, for the tie rule)
LORA_RUNS = {
    "A": (["--lora-targets", "qv"], 0.74),
    "B": (["--lora-targets", "attn"], 1.47),
    "C": (["--lora-targets", "dense"], 1.66),
    "D": (["--lora-targets", "all"], 28.3),
    "E": (["--lora-targets", "dense", "--lora-rank", "32"], 3.3),
    "F": (["--lora-targets", "dense", "--lora-rank", "64"], 6.6),
}
LORA_LR = "2e-4"


def log(m):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOGS / "pipeline.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


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
            if any(k in l for k in ("eval @", "done in", "held-out", "F1", "demo questions", "calibrated",
                                    "latency", "Traceback", "Error"))]
    for l in tail[-8:]:
        log("    " + l.strip()[:220])
    log(f"--- {name}: exit {rc}")
    return rc


def accuracy(name: str) -> float | None:
    """Calibrated held-out accuracy from a decide.py eval log."""
    lf = LOGS / f"{name}.log"
    if not lf.exists():
        return None
    m = re.search(r"calibrated \(T=[\d.]+.*?\): accuracy ([\d.]+)%", lf.read_text(encoding="utf-8", errors="replace"))
    return float(m.group(1)) if m else None


def lora(run_id: str, extra: list[str], lr: str = LORA_LR, task: str | None = None) -> float | None:
    out = f"ckpt_decide_lora_{run_id}.pt"
    t = ["--task", task] if task else []
    cap = 2.5 if "all" in extra else 1.0
    run(f"4_lora_{run_id}_train", ["decide.py", "train", "--ckpt", CHAT, "--out", out, "--mode", "lora",
                                   "--lr", lr, *extra, *t], cap)
    if not (ROOT / out).exists():
        return None
    run(f"4_lora_{run_id}_eval", ["decide.py", "eval", "--ckpt", out], 0.5)
    return accuracy(f"4_lora_{run_id}_eval")


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    log(f"=== POST V2 start; base {BASE} ===")

    # 2. chat
    run("2_chat_sft", ["finetune.py", "--ckpt", BASE, "--qa", "data/chat_train.jsonl",
                       "--heldout", "data/chat_heldout.jsonl", "--out", CHAT, "--epochs", "2",
                       "--batch-size", "4", "--lr", "5e-5", "--grad-ckpt", "--eval-every", "500",
                       "--log-every", "100"], 3.0)
    run("2_chat_samples", [str(ROOT / "experiments" / "chat_samples.py"), "ckpt_chat_v2.pt", CHAT], 0.5)

    # 3. reader
    run("3_reader", ["rc_pointer.py", "train", "--ckpt", BASE, "--out", "ckpt_rc_pointer_v3.pt",
                     "--epochs", "1", "--batch-size", "8", "--lr", "5e-5", "--eval-every", "1000"], 3.0)
    run("3_reader_eval", ["rc_pointer.py", "eval", "--ckpt", "ckpt_rc_pointer_v3.pt", "--n", "300"], 1.0)

    if not (ROOT / CHAT).exists():
        log(f"{CHAT} missing; skipping steps 4-5")
    else:
        # 4. LoRA comparison
        res = {}
        for rid, (extra, mparams) in LORA_RUNS.items():
            res[rid] = {"args": extra, "lr": LORA_LR, "params_M": mparams, "acc": lora(rid, extra)}
            log(f"    run {rid}: {res[rid]['acc']}%")
        scored = [(r, v) for r, v in res.items() if v["acc"] is not None]
        if scored:
            top = max(v["acc"] for _, v in scored)
            # within ~0.5 points of the best (one seed): take the smallest adapter
            best = min((rv for rv in scored if rv[1]["acc"] >= top - 0.5), key=lambda rv: rv[1]["params_M"])[0]
            log(f"    best of A-F: {best} ({res[best]['acc']}%)")
            for rid, lr in (("G", "1e-4"), ("H", "4e-4")):
                extra = res[best]["args"]
                res[rid] = {"args": extra, "lr": lr, "params_M": res[best]["params_M"], "acc": lora(rid, extra, lr)}
                log(f"    run {rid} ({best} at lr {lr}): {res[rid]['acc']}%")
            run("4_ref_full_train", ["decide.py", "train", "--ckpt", CHAT, "--out", "ckpt_decide_lora_ref.pt",
                                     "--mode", "full"], 1.5)
            run("4_ref_full_eval", ["decide.py", "eval", "--ckpt", "ckpt_decide_lora_ref.pt"], 0.5)
            res["ref"] = {"args": ["--mode", "full"], "lr": "3e-5", "params_M": 398, "acc": accuracy("4_ref_full_eval")}

            scored = [(r, v) for r, v in res.items() if r != "ref" and v["acc"] is not None]
            top = max(v["acc"] for _, v in scored)
            win = min((rv for rv in scored if rv[1]["acc"] >= top - 0.5), key=lambda rv: rv[1]["params_M"])[0]
            log(f"=== LoRA winner: {win} ({res[win]['acc']}%, {res[win]['params_M']}M, lr {res[win]['lr']}) ===")
            (LOGS / "lora_results.json").write_text(json.dumps({"runs": res, "winner": win}, indent=2))

            # 5. adapters with the winner
            shutil.copy2(ROOT / f"ckpt_decide_lora_{win}.pt", ROOT / "ckpt_decide_chat_lora_v3.pt")
            log("    banking adapter: ckpt_decide_chat_lora_v3.pt (copy of the winner)")
            out = "ckpt_decide_lora_hi_v3.pt"
            run("5_hi_train", ["decide.py", "train", "--ckpt", CHAT, "--out", out, "--mode", "lora",
                               "--lr", res[win]["lr"], *res[win]["args"], "--task", "massive-hi"], 1.5)
            run("5_hi_eval", ["decide.py", "eval", "--ckpt", out], 0.5)

    # 6. Decide general, zero-shot on BANKING77
    run("6_general_train", ["decide.py", "train-general", "--ckpt", BASE, "--out", "ckpt_decide_general_v3.pt"], 1.5)
    run("6_general_zero_shot", ["decide.py", "eval", "--ckpt", "ckpt_decide_general_v3.pt"], 0.5)

    log("=== POST V2 DONE ===")


if __name__ == "__main__":
    main()
