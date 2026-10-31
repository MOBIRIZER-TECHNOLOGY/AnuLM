"""
Chapter 6 lab: give Chapter 4's skeleton attention and measure what it buys.

    python lab.py                      # your course/build/anulm.py
    python lab.py --impl solution      # the reference solution
    python lab.py --data my_text.txt   # your own text

CPU, a few minutes. Writes curve.csv and results.json for the figures. Four
measurements:

  1. learning: the same sizes, steps and data as Chapter 4's skeleton, now with
     attention: does it get past the bigram ceiling (2.48)?
  2. the KV cache: time to generate with and without it, and whether the
     tokens are identical;
  3. cache memory: bytes per token for multi-head, grouped and multi-query
     attention at the 400M model's sizes;
  4. what one head looks at: its attention weights over a line of the
     validation text, after training.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
COURSE = HERE.parent
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lab1 = load(COURSE / "ch01-smallest-model" / "lab.py")


@torch.no_grad()
def head_weights(model, M, idx, layer: int, head: int):
    """Attention weights (S, S) of one head, recomputed from the model's own modules."""
    x = model.embed_tokens(idx)
    S = idx.shape[1]
    cos, sin = model.rotary(S, x.device, x.dtype)
    for i, blk in enumerate(model.layers):
        if i == layer:
            a = blk.attn
            h = blk.input_layernorm(x)
            qkv = a.query_key_value(h).view(1, S, a.n_head + 2 * a.n_kv_head, a.head_dim)
            q, k, _ = qkv.split([a.n_head, a.n_kv_head, a.n_kv_head], dim=-2)
            q, k = q.transpose(1, 2), k.transpose(1, 2)
            if a.use_qk_norm:
                q, k = a.query_layernorm(q), a.key_layernorm(k)
            q, k = M.apply_rope(q, k, cos, sin)
            k = k.repeat_interleave(a.n_rep, dim=1)
            att = (q[0, head] @ k[0, head].T) * a.scale
            att = att.masked_fill(torch.ones(S, S, dtype=torch.bool).triu(1), float("-inf"))
            return F.softmax(att, dim=-1)
        x = blk(x, cos, sin)
    raise IndexError(layer)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=["build", "solution"], default="build")
    p.add_argument("--data", default=str(COURSE.parent / "data" / "input.txt"))
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--seed", type=int, default=1337)
    a = p.parse_args()
    M = load(COURSE / "build" / ("solutions/ch06.py" if a.impl == "solution" else "anulm.py"))
    torch.manual_seed(a.seed)
    g = torch.Generator().manual_seed(a.seed)
    train, val = lab1.load_bytes(a.data)
    _, bigram_loss = lab1.bigram(train, val)
    ctx = 128
    try:
        model = M.AttnLM(vocab_size=259, hidden_size=128, n_layer=4, n_head=4, n_kv_head=2, head_dim=32,
                         intermediate_size=256, block_size=256)
    except NotImplementedError as e:
        sys.exit(f"course/build/anulm.py is not finished yet ({e}).\n"
                 f"Fill it in until `python course/checks/ch06.py` passes, or run with --impl solution.")
    n_params = sum(t.numel() for t in model.parameters())

    @torch.no_grad()
    def evaluate():
        model.eval()
        eg = torch.Generator().manual_seed(0)
        loss = sum(model(*lab1.batch(val, ctx, 32, eg))[1].item() for _ in range(20)) / 20
        model.train()
        return loss

    # 1. learning
    print(f"1. the skeleton plus attention: {n_params:,} parameters, {a.steps} steps on CPU")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    rows, t0 = [(0, evaluate())], time.time()
    for step in range(1, a.steps + 1):
        for group in opt.param_groups:
            group["lr"] = 1e-3 * min(1.0, step / 100) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        _, loss = model(*lab1.batch(train, ctx, 32, g))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            rows.append((step, evaluate()))
            print(f"   step {step:5d}  validation loss {rows[-1][1]:.3f}  ({time.time() - t0:.0f}s)")
    train_s = time.time() - t0
    ch4 = COURSE / "ch04-skeleton" / "results.json"
    skeleton = json.loads(ch4.read_text(encoding="utf-8"))["final_loss"] if ch4.exists() else None
    print(f"   with attention {rows[-1][1]:.3f}   without (Chapter 4) {skeleton}   bigram table {bigram_loss:.3f}")

    # 2. the KV cache
    model.eval()
    prompt = torch.tensor([list(b"ROMEO:\n")])
    timings, outs = {}, {}
    for use_cache in (False, True):
        t = time.time()
        outs[use_cache] = model.generate(prompt, 240, temperature=0.8, use_cache=use_cache,
                                         generator=torch.Generator().manual_seed(7))
        timings[use_cache] = time.time() - t
    same = torch.equal(outs[False], outs[True])
    sample = bytes(outs[True][0].tolist()).decode("utf-8", errors="replace")
    print(f"2. 240 tokens: {timings[False]:.2f}s without the cache, {timings[True]:.2f}s with it "
          f"({timings[False] / timings[True]:.1f}x); identical tokens: {same}")
    print("   " + sample.replace("\n", "\n   "))

    # 3. cache memory at the 400M model's sizes (20 layers, 16 query heads of 64, bf16)
    layers, heads, dh, bytes_ = 20, 16, 64, 2
    mem = {name: 2 * kv * dh * bytes_ * layers for name, kv in (("multi-head (16 kv heads)", 16),
                                                               ("grouped, as AnuLM (4 kv heads)", 4),
                                                               ("multi-query (1 kv head)", 1))}
    for name, b in mem.items():
        print(f"3. {name}: {b / 1024:.0f} KiB per token, {b * 1024 / 2 ** 20:.0f} MiB for a 1,024-token context")

    # 4. what one head looks at
    line = val[val.tolist().index(10) + 1:][:48].unsqueeze(0)       # 48 bytes from the start of a line
    best, pick = -1.0, (0, 0)
    for layer in range(len(model.layers)):
        for head in range(4):
            w = head_weights(model, M, line, layer, head)
            prev = torch.diagonal(w, -1).mean().item()             # weight on the byte just before
            if prev > best:
                best, pick = prev, (layer, head)
    w = head_weights(model, M, line, *pick)
    text = bytes(line[0].tolist()).decode("latin-1")
    print(f"4. layer {pick[0]} head {pick[1]} puts {best:.0%} of its attention on the previous byte, on average")

    with open(HERE / "curve.csv", "w", newline="") as f:
        csv.writer(f).writerows([("step", "val_loss")] + [(s, round(v, 4)) for s, v in rows])
    (HERE / "results.json").write_text(json.dumps({
        "data": Path(a.data).name, "params": n_params, "steps": a.steps, "seconds": round(train_s),
        "final_loss": round(rows[-1][1], 4), "skeleton_loss": skeleton, "bigram_loss": round(bigram_loss, 4),
        "gen_tokens": 240, "gen_seconds_uncached": round(timings[False], 3),
        "gen_seconds_cached": round(timings[True], 3), "cached_equals_uncached": same, "sample": sample,
        "cache_bytes_per_token": mem, "head": {"layer": pick[0], "head": pick[1], "prev_weight": round(best, 4),
                                               "text": text, "weights": [[round(v, 4) for v in r] for r in w.tolist()]},
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print("saved curve.csv, results.json")


if __name__ == "__main__":
    main()
