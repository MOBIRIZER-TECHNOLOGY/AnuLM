"""
Book example, not repository code.

Chapter 37, 37.7: what the November recipe did (ACCUM 8, LR 2.5e-5, chat 0.25).

    python book/code-examples/ch37_november_run.py [--save]

Reads only files the run wrote; trains and evaluates nothing:

- `ckpt_omni_curve.csv`: one row per evaluation, held-out loss per task
  (step 0 is the start);
- `logs/omni_one/6_train.log`: the training-batch losses, "batches seen" per
  task, the task sizes and the run's total time;
- `logs/omni_one/results.json`: the kept checkpoint's final measures
  (reading F1 was added from `eval_reading.json`, see 37.7);
- `logs/omni_one/eval_base_chat.json`: the chat model it started from;
- `logs/omni_one/export_results.json`: the exported bf16 copy;
- `experiments/omni_one.py` at the pin: `GATES`.

It prints the per-task ratios, the gates, and how many times the run read
its chat data, and writes ./part_viii_ix_omni_nov.json, or
book/measurements/ with --save.
"""

from __future__ import annotations

import ast
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PIN = "b42998d"
LOGS = ROOT / "logs" / "omni_one"
CURVE = ROOT / "ckpt_omni_curve.csv"
OUT = (ROOT / "book" / "measurements" if "--save" in sys.argv else Path.cwd()) / "part_viii_ix_omni_nov.json"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def close(a, b, tol, what):
    assert abs(a - b) <= tol, f"{what}: computed {a}, chapter says {b}"


def gates_at_pin() -> dict:
    src = subprocess.run(["git", "show", f"{PIN}:experiments/omni_one.py"], cwd=ROOT, capture_output=True,
                         text=True, encoding="utf-8", check=True).stdout
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "GATES":
            return ast.literal_eval(node.value)
    raise SystemExit("GATES not found")


def main() -> None:
    # 1. the curve: held-out loss per task at every evaluation
    with open(CURVE, encoding="utf-8") as f:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    names = [k for k in rows[0] if k not in ("step", "score")]
    first = {n: rows[0][n] for n in names}
    evals = []
    for r in rows[1:]:
        ratios = {n: r[n] / first[n] for n in names}
        recomputed = sum(ratios.values()) / len(ratios)
        close(recomputed, r["score"], 6e-5, f"score at step {int(r['step'])}")
        evals.append({"step": int(r["step"]), "score": r["score"], "losses": {n: r[n] for n in names},
                      "ratios": {n: round(v, 4) for n, v in ratios.items()}})
    best = min(e["score"] for e in evals)
    kept = next(e for e in evals if e["score"] == best)           # the trainer saves only on a strict improvement
    ties = [e["step"] for e in evals if e["score"] == best]
    last = evals[-1]
    print(f"1. {CURVE.name}: {len(evals)} evaluations after step 0; tasks {', '.join(names)}")
    print("   step    score    " + "  ".join(f"{n:>7}" for n in names))
    for e in evals:
        print(f"  {e['step']:6d}  {e['score']:.4f}    " + "  ".join(f"{e['ratios'][n]:7.3f}" for n in names))
    print(f"   kept: step {kept['step']} (score {best}; tied at steps {ties}, the first one is kept)")
    print(f"   chat ratio: {kept['ratios']['chat']:.3f} at the kept step, {last['ratios']['chat']:.3f} at step "
          f"{last['step']} (the October run ended at 1.651)")

    # 2. the log: training-batch losses, batches seen, task sizes, time
    log = (LOGS / "6_train.log").read_text(encoding="utf-8")
    done = re.findall(r"done in (\d+)s \| best score ([\d.]+)", log)
    assert len(done) == 1, "expected exactly one finished run in 6_train.log"
    seconds, logged_best = int(done[0][0]), float(done[0][1])
    close(logged_best, best, 1e-9, "best score in the log")
    sizes = {}
    for m in re.finditer(r"task (\w+)\s+\w+\s+(\d+) train\s+(\d+) held\s+batch (\d+)\s+weight ([\d.]+)", log):
        sizes[m.group(1)] = {"train": int(m.group(2)), "held": int(m.group(3)), "batch": int(m.group(4)),
                             "weight": float(m.group(5))}
    seen = ast.literal_eval(re.findall(r"eval @\s+%d .*?batches seen (\{.*?\})" % last["step"], log)[-1])
    train_losses = {}
    for m in re.finditer(r"^step\s+(\d+) \| (.*?) \| imbalance", log, re.M):
        train_losses[int(m.group(1))] = {k: float(v) for k, v in re.findall(r"(\w+) ([\d.]+)", m.group(2))}
    tail = [train_losses[s] for s in sorted(train_losses)[-3:]]         # steps 12,250-12,500
    train_tail = {n: round(sum(t[n] for t in tail) / len(tail), 3) for n in names}
    passes = {n: round(seen[n] * sizes[n]["batch"] / sizes[n]["train"], 2) for n in names}
    print(f"\n2. 6_train.log: done in {seconds} s = {seconds / 3600:.1f} h")
    print(f"   batches seen at step {last['step']}: {seen} (sum {sum(seen.values()):,})")
    for n in names:
        print(f"   {n:8s} {seen[n]:6d} batches x {sizes[n]['batch']} / {sizes[n]['train']:6d} examples = "
              f"{passes[n]:5.2f} passes; training-batch loss (last 3 logs) {train_tail[n]:.3f}, "
              f"held-out {last['losses'][n]:.3f}")

    # 3. the final measures and the gates
    res = json.loads((LOGS / "results.json").read_text(encoding="utf-8"))
    res.pop("caption_samples", None)
    reading = json.loads((LOGS / "eval_reading.json").read_text(encoding="utf-8"))
    for k in ("read_squad_en", "read_mlqa_hi"):
        assert res[k] == reading[k], k
    base = json.loads((LOGS / "eval_base_chat.json").read_text(encoding="utf-8"))
    export = json.loads((LOGS / "export_results.json").read_text(encoding="utf-8"))
    export.pop("caption_samples", None)
    hi = [res[k] for k in ("wer_hi_fleurs", "wer_hi_kathbath", "wer_hi_mucs")]
    res["wer_hi_mean"] = round(sum(hi) / 3, 1)
    gates = gates_at_pin()
    verdict = {k: res.get(k) is not None and (res[k] <= t if op == "<=" else res[k] >= t)
               for k, (op, t) in gates.items()}
    print(f"\n3. release gates (experiments/omni_one.py GATES @{PIN}) on logs/omni_one/results.json")
    for k, (op, t) in gates.items():
        print(f"   {k:20s} {res.get(k)!s:>8}  {op} {t:<6} {'pass' if verdict[k] else 'FAIL'}")
    print(f"   all pass: {all(verdict.values())}; chat started at {base['chat_heldout_loss']} ({base['ckpt']})")
    print(f"   exported bf16 copy: {export}")

    # the numbers the chapter prints
    close(kept["step"], 9375, 0, "kept step")
    close(best, 0.3643, 0, "kept score")
    close(last["ratios"]["chat"], 1.094, 5e-4, "final chat ratio")
    close(kept["ratios"]["chat"], 1.087, 5e-4, "kept chat ratio")
    close(seconds, 87141, 0, "training seconds")
    close(passes["chat"], 7.39, 5e-3, "passes over chat")
    close(train_tail["vqa"], 1.925, 5e-4, "VQA training-batch loss")
    close(last["losses"]["vqa"], 3.599, 5e-4, "VQA held-out loss")
    close(res["chat_heldout_loss"], 3.4754, 0, "chat held-out loss")
    close(base["chat_heldout_loss"], 3.146, 0, "starting chat loss")
    close(res["wer_hi_mean"], 32.0, 0, "Hindi mean WER")
    assert [k for k, v in verdict.items() if not v] == ["chat_heldout_loss"]

    OUT.write_text(json.dumps({
        "source": "ckpt_omni_curve.csv; logs/omni_one/6_train.log, results.json, eval_reading.json, "
                  "eval_base_chat.json, export_results.json (numbers only)",
        "where": f"the November run (2026-11-03 08:32 to 2026-11-04 08:47); GATES from experiments/omni_one.py @{PIN}",
        "first": first, "evals": evals, "kept_step": kept["step"], "tied_steps": ties, "seconds": seconds,
        "batches_seen": seen, "tasks": sizes, "passes": passes, "train_loss_tail": train_tail,
        "results": res, "export_results": export, "base_chat": base,
        "gates": {k: [op, t, verdict[k]] for k, (op, t) in gates.items()}}, indent=1) + "\n", encoding="utf-8")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
