"""
Book example, not repository code.

Chapter 22: the released run, read from its own log.

    python book/code-examples/ch22_read_run_log.py

`v2_train_run.log` (repository root, written by `train.py` through
`experiments/v2_train.sh` / `_v2_run.cmd`) holds every line the base-v2 run
printed: one training line per 1,000 steps and one evaluation per 10,000. A
resumed run repeats the steps it lost, so for each step the last line wins.
With `--save` the results go to book/measurements/part_v_a_run.json; without it, to
./part_v_a_run.json in the current directory.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "v2_train_run.log"
NAME = "part_v_a_run.json"
OUT = (ROOT / "book" / "measurements" / NAME) if "--save" in sys.argv else Path(NAME)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STEP = re.compile(r"^step\s+(\d+) \| loss ([\d.]+) \| aux ([\d.]+) \| lr ([\d.e+-]+) \| imbalance\s+([\d.]+)x "
                  r"\(median ([\d.]+)x\) \|\s+(\d+) tok/s \|\s+(\d+) ms/step \| epoch ([\d.]+)")
EVAL = re.compile(r"^\s+eval @\s*(\d+) \| val loss ([\d.]+) \+/- ([\d.]+) \| bits/byte ([\d.]+)")


def main():
    train, evals, resumes, invocations, planned = {}, {}, 0, 0, []
    for line in LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        if m := STEP.match(line):
            s = int(m[1])
            train[s] = dict(loss=float(m[2]), aux=float(m[3]), lr=float(m[4]), imb=float(m[5]),
                            imb_median=float(m[6]), tok_s=int(m[7]), ms=int(m[8]), epoch=float(m[9]))
        elif m := EVAL.match(line):
            evals[int(m[1])] = dict(val=float(m[2]), sem=float(m[3]), bpb=float(m[4]))
        elif line.startswith("resuming from"):
            resumes += 1
        elif line.startswith("memory-mapped corpus"):
            invocations += 1
        elif line.startswith("sampler:"):
            planned.append(line.split("; ")[1])
    steps = sorted(train)
    first, last = train[steps[0]], train[steps[-1]]
    ev = sorted(evals)
    print(f"{len(steps)} training lines (steps {steps[0]:,} to {steps[-1]:,}), {len(ev)} evaluations, "
          f"{resumes} resumes, {invocations} invocations")
    print(f"sampler lines: first {planned.count(planned[0])} say '{planned[0]}', the last "
          f"{len(planned) - planned.count(planned[0])} say '{planned[-1]}'")
    print(f"step 0: loss {first['loss']:.4f} (ln 32768 = {math.log(32768):.4f}), aux {first['aux']:.4f}, "
          f"imbalance {first['imb']:.2f}x")
    tail = [train[s]["loss"] for s in steps[-20:]]
    print(f"last 20 logged steps: train loss {min(tail):.4f} to {max(tail):.4f}, "
          f"stdev {statistics.stdev(tail):.3f}")
    stable_end = max(s for s in ev if s < 1_590_000)
    for s in (ev[0], 99_999, stable_end, ev[-1]):
        e = evals[s]
        print(f"eval @ {s:>9,}: val {e['val']:.4f} +/- {e['sem']:.4f}, bits/byte {e['bpb']:.3f}")
    drop = evals[stable_end]["val"] - evals[ev[-1]]["val"]
    print(f"the decay (steps 1,590,000 to 1,900,000, 16% of the run) lowered val by {drop:.4f} from its last flat eval")
    flat = [s for s in ev if s < 1_590_000]
    best = min(flat, key=lambda s: evals[s]["val"])
    late = [evals[s]["val"] for s in flat if s >= 1_000_000]
    last20 = [evals[s]["val"] for s in flat[-20:]]
    m, sd = statistics.mean(late), statistics.stdev(late)
    print(f"flat phase: best {evals[best]['val']:.4f} at {best:,}; evals 1.0M-1.58M mean {m:.4f} (sd {sd:.4f}, "
          f"n={len(late)}); {evals[stable_end]['val']:.4f} ranks {sorted(last20, reverse=True).index(evals[stable_end]['val']) + 1} "
          f"of the last 20 flat evals from the top (highest {max(last20):.4f})")
    print(f"decay drop measured from the flat mean {m - evals[ev[-1]]['val']:.4f}, from the flat best "
          f"{evals[best]['val'] - evals[ev[-1]]['val']:.4f}")
    toks = [train[s]["tok_s"] for s in steps if s >= 10_000]
    ms = [train[s]["ms"] for s in steps if s >= 10_000]
    print(f"throughput: median {statistics.median(toks):,.0f} tok/s (cumulative per invocation), "
          f"median {statistics.median(ms)} ms/step")
    tokens = (steps[-1] + 1) * 4 * 1024
    print(f"tokens trained: {steps[-1] + 1:,} steps x 4 x 1,024 = {tokens:,}")
    print(f"bits/byte check: {evals[ev[-1]]['val']:.4f} / ln 2 / 6.397 = "
          f"{evals[ev[-1]]['val'] / math.log(2) / 6.397:.4f}")
    # The schedule: train.lr_at with the arguments of experiments/v2_train.sh, against every logged lr.
    sys.path.insert(0, str(ROOT))
    from anulm.training import train as T
    from types import SimpleNamespace
    a = SimpleNamespace(lr=6e-4, min_lr=6e-5, warmup=2000, steps=1_900_000, schedule="wsd", decay_frac=0.1631579)
    worst = max(abs(T.lr_at(s, a) - train[s]["lr"]) / T.lr_at(s, a) for s in steps)
    print(f"lr_at: decay starts after step {int(a.steps * (1 - a.decay_frac)):,}; lr at step 0 {T.lr_at(0, a):.2e}, "
          f"at 1,899,999 {T.lr_at(1_899_999, a):.2e}; every logged lr matches to {100 * worst:.2f}% (3 digits logged)")
    OUT.write_text(json.dumps({"train": {str(s): train[s] for s in steps},
                               "eval": {str(s): evals[s] for s in ev},
                               "resumes": resumes, "invocations": invocations, "stable_end": stable_end,
                               "decay_drop": drop,
                               "tokens": tokens, "tail_loss_min": min(tail), "tail_loss_max": max(tail)},
                              indent=1), encoding="utf-8")
    meta = json.loads((ROOT / "data" / "v2" / "meta.json").read_text())
    vb = sum(s["val_tokens"] * s["bytes_per_token"] for s in meta["sources"].values()) / meta["val_tokens"]
    print(f"bits/byte with the held-out split's own {vb:.3f} bytes/token: "
          f"{evals[ev[-1]]['val'] / math.log(2) / vb:.4f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
