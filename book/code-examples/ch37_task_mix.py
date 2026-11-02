"""
Book example, not repository code.

Chapter 37: one model, many tasks (`omni_train.py`).

    python book/code-examples/ch37_task_mix.py [--save]

`omni_train.py` has uncommitted edits in the working tree, so this script loads
it, and `experiments/omni_one.py`, from the pinned commit with `git show`, and
runs that code unchanged. It checks:

1. how `Task` parses a task spec, Windows drive letters included;
2. that `text_batches` gives the same loss through `embeds=` as plain ids;
3. that `train`'s task draw, replayed from its seed, reproduces the
   "batches seen" counts the real run logged, resume included;
4. that every logged score is the mean of the tasks' held-out losses divided
   by their step-0 values, and what that mean hid.

Item 4 reads `logs/omni_one/6_train.log`, the training log of the joint run
(numbers only) and writes ./part_viii_ix_omni_log.json, or
book/measurements/ with --save.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm import model as M                                    # noqa: E402  repository code, unchanged

PIN = "773b6a8"
LOG = ROOT / "logs" / "omni_one" / "6_train.log"
OUT = (ROOT / "book" / "measurements" if "--save" in sys.argv else Path.cwd()) / "part_viii_ix_omni_log.json"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def at_pin(path: str) -> str:
    return subprocess.run(["git", "show", f"{PIN}:{path}"], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout


def pinned_module(path: str, name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__file__ = str(ROOT / path)
    exec(compile(at_pin(path), str(ROOT / path), "exec"), mod.__dict__)
    return mod


OT = pinned_module("omni_train.py", "omni_train_at_pin")


def tiny(**kw) -> M.AnuLMConfig:
    base = dict(vocab_size=64, block_size=32, n_layer=3, hidden_size=32, n_head=4, head_dim=8,
                n_kv_head=2, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16,
                intermediate_size=64, n_group=1, topk_group=1)
    base.update(kw)
    return M.AnuLMConfig(**base)


def specs() -> None:
    print("1. task specs")
    for s in ("chat=text:data/chat_train.jsonl:0.07:8", "asr_hi=asr:C:\\data\\hindi:0.32",
              "vqa=image:data/vqa.jsonl", "cap=image:D:/imgs/train.jsonl::16"):
        t = OT.Task(s)
        print(f"   {s:40s} -> name {t.name}, kind {t.kind}, path {t.path}, weight {t.weight}, batch {t.batch}")


def text_path() -> None:
    torch.manual_seed(0)
    m = M.AnuLM(tiny()).eval()
    rows = [{"p": [3, 4, 5], "a": [9, 10, 1]}, {"p": [6, 7], "a": [11, 1]}]
    eb, tb = next(OT.text_batches(rows, m, "cpu", 2, shuffle=False))
    ids = torch.zeros(tb.shape, dtype=torch.long)
    for k, r in enumerate(rows):
        ids[k, : len(r["p"]) + len(r["a"])] = torch.tensor(r["p"] + r["a"])
    with torch.no_grad():
        _, a = m(None, tb, embeds=eb)
        _, b = m(ids, tb)
    print(f"\n2. text_batches: embeds {tuple(eb.shape)}, targets {tb.tolist()}")
    print(f"   loss through embeds= {a.item():.6f}, through ids {b.item():.6f}")
    assert torch.allclose(a, b, atol=1e-6)


OO = pinned_module("experiments/omni_one.py", "omni_one_at_pin")     # definitions only; main() not run


def pinned_tasks() -> list[tuple]:
    return list(OO.TASKS)


def parse_log() -> dict:
    text = LOG.read_text(encoding="utf-8", errors="replace")
    first = dict((k, float(v)) for k, v in re.findall(r"(\w+) ([\d.]+)",
                 re.search(r"held-out at step 0: (.*)", text).group(1)))
    evals = []
    for m in re.finditer(r"eval @\s+(\d+) \| score ([\d.]+)[^|]*\| (.*?) \| batches seen (\{.*?\})", text):
        losses = dict((k, float(v)) for k, v in re.findall(r"(\w+) ([\d.]+)", m.group(3)))
        evals.append({"step": int(m.group(1)), "score": float(m.group(2)), "losses": losses,
                      "seen": ast.literal_eval(m.group(4))})
    resumed = [int(s) for s in re.findall(r"resumed from \S+ at step (\d+)", text)]
    return {"first": first, "evals": evals, "resumed_at": resumed}


def replay_draw(names, w, resumed_at, last_step) -> dict:
    """train()'s draw: rng = default_rng(args.seed + start), one task per step; seed 0."""
    seen = {n: 0 for n in names}
    starts = [1] + [r + 1 for r in resumed_at] + [last_step + 1]
    for a, b in zip(starts[:-1], starts[1:]):
        rng = np.random.default_rng(0 + a)
        for _ in range(a, b):
            seen[names[rng.choice(len(names), p=w)]] += 1
    return seen


def run_log() -> None:
    tasks = pinned_tasks()
    names = [t[0] for t in tasks]
    w = np.array([t[3] for t in tasks], dtype=float)
    w /= w.sum()
    print(f"\n3. experiments/omni_one.py TASKS at @{PIN}: " + ", ".join(f"{n} {t[1]} {t[3]} x{t[4]}" for n, t in zip(names, tasks)))
    if not LOG.exists():
        print("   (logs/omni_one/6_train.log not present: skipped)")
        return
    r = parse_log()
    last = r["evals"][-1]
    got = replay_draw(names, w, r["resumed_at"], last["step"])
    print(f"   replayed draw to step {last['step']} (resumed at {r['resumed_at']}): {got}")
    print(f"   the log's batches seen:                         {last['seen']}")
    assert got == last["seen"]
    print("\n4. score = mean over tasks of held-out loss / step-0 loss")
    print("   step    score  recomputed   " + "  ".join(f"{n:>7s}" for n in names))
    print("      0   1.0000      1.0000   " + "  ".join(f"{1.0:7.3f}" for _ in names))
    rows = []
    for e in r["evals"]:
        ratios = {n: e["losses"][n] / r["first"][n] for n in names}
        score = float(np.mean(list(ratios.values())))
        assert abs(score - e["score"]) < 5e-5, (e["step"], score, e["score"])
        print(f"   {e['step']:6d}   {e['score']:.4f}      {score:.4f}   " +
              "  ".join(f"{ratios[n]:7.3f}" for n in names))
        rows.append({"step": e["step"], "score": e["score"], "losses": e["losses"],
                     "ratios": {n: round(v, 4) for n, v in ratios.items()}})
    c0, c1 = r["first"]["chat"], last["losses"]["chat"]
    print(f"   chat: {c0} at step 0 -> {c1} at step {last['step']} (x{c1 / c0:.3f}) while the score fell to "
          f"{last['score']}")
    OUT.write_text(json.dumps({"source": "logs/omni_one/6_train.log (numbers only)", "first": r["first"],
                               "evals": rows, "resumed_at": r["resumed_at"], "tasks": tasks,
                               "batches_seen_replayed": got}, indent=1) + "\n", encoding="utf-8")
    print(f"   -> {OUT}")


def release_gates() -> None:
    f = ROOT / "logs" / "omni_one" / "results.json"
    if not f.exists():
        return
    res = {k: v for k, v in json.loads(f.read_text(encoding="utf-8")).items() if not isinstance(v, list)}
    ok = OO.gates(res)                               # the pinned gate rule, unchanged
    print("\n5. release gates (experiments/omni_one.py GATES at the pin) on logs/omni_one/results.json")
    for k, (op, thr) in OO.GATES.items():
        print(f"   {k:20s} {res.get(k)!s:>7}  {op} {thr:<5}  {'pass' if ok[k] else 'FAIL'}")
    print(f"   all pass: {all(ok.values())}")
    m = json.loads(OUT.read_text(encoding="utf-8"))
    m["release_gates"] = {k: {"value": res.get(k), "gate": list(OO.GATES[k]), "pass": ok[k]} for k in OO.GATES}
    OUT.write_text(json.dumps(m, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    specs()
    text_path()
    run_log()
    release_gates()
    print("\nall checks passed")
