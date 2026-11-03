"""
Book example, not repository code.

Chapter 27: the KV cache on the released model, `ckpt_base_v2.pt`.

    python book/code-examples/ch27_measure_cache.py            # writes ./part_vi_ch27_cache.json
    python book/code-examples/ch27_measure_cache.py --save     # the book's copy, book/measurements/part_vi_ch27_cache.json

Runs `model.py` unchanged and measures four things:

1. what the cache holds under bf16 autocast, the precision `serve.py`,
   `sample.py` and `api.py` run in: dtypes and bytes per token;
2. how far cached logits are from full-forward logits, in fp32 and in bf16, on
   windows of the held-out split `data/v2/val.u16`;
3. whether greedy decoding with and without the cache picks the same tokens;
4. the time of one decode step against one uncached forward at several
   context lengths, and of the rebuild step once the context is full.

Only token ids and statistics are used; no held-out text is printed or saved.
Results go to ./part_vi_ch27_cache.json, or with --save to book/measurements/part_vi_ch27_cache.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm import model as M  # noqa: E402  the repository's model.py, unchanged

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# The book's copy is written only with --save; otherwise the result goes to the current directory,
# so a reader or reviewer re-running the script never overwrites the numbers the chapter quotes.
SAVE = "--save" in sys.argv[1:]
OUT = (ROOT / "book" / "measurements" if SAVE else Path.cwd()) / "part_vi_ch27_cache.json"
SEED, EOS = 0, 32767


def load():
    ck = torch.load(ROOT / "ckpt_base_v2.pt", map_location="cpu", weights_only=False)
    m = M.AnuLM(ck["cfg"])
    m.load_state_dict(ck["model"], strict=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return m.to(dev).eval(), ck["cfg"], dev


def windows(n, length):
    val = np.memmap(ROOT / "data" / "v2" / "val.u16", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, len(val) - length - 1, size=n)
    return torch.from_numpy(np.stack([val[s: s + length] for s in starts]).astype(np.int64))


def set_impl(m, impl):
    """The dispatch path is not part of the weights. serve.py uses "grouped" (bf16 by contract) on CUDA;
    an fp32 comparison needs "sparse", or the experts would still run in bf16."""
    for layer in m.layers:
        if layer.is_moe:
            layer.mlp.impl = impl


def amp(dev, on):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=on and dev == "cuda")


@torch.no_grad()
def cache_contents(m, dev):
    x = windows(1, 100).to(dev)
    cache = m.new_cache()
    with amp(dev, True):
        m(x, cache=cache)
    k, v = cache[0]["k"], cache[0]["v"]
    per_tok = sum(c["k"][0, :, 0].numel() * c["k"].element_size() + c["v"][0, :, 0].numel() * c["v"].element_size()
                  for c in cache)
    return {"k_shape": list(k.shape), "k_dtype": str(k.dtype), "v_dtype": str(v.dtype), "bytes_per_token": per_tok,
            "bytes_per_1024": per_tok * 1024}


@torch.no_grad()
def cached_vs_full(m, dev, bf16, n=8, prompt=512, steps=16):
    """Prefill `prompt` tokens, then feed the next `steps` true tokens one at a time through the cache;
    compare each step's logits with a full forward over the same tokens."""
    set_impl(m, "grouped" if bf16 and dev == "cuda" else "sparse")
    x = windows(n, prompt + steps).to(dev)
    worst, agree, total = 0.0, 0, 0
    for row in x:
        row = row[None]
        with amp(dev, bf16):
            full, _ = m(row, row)                     # (1, S, V): every position
            cache = m.new_cache()
            got = [m(row[:, :prompt], cache=cache, start=0)[0][:, -1]]
            for t in range(prompt, prompt + steps - 1):
                got.append(m(row[:, t:t + 1], cache=cache, start=t)[0][:, -1])
        got = torch.cat(got).float()                  # (steps, V)
        ref = full[0, prompt - 1: prompt + steps - 1].float()
        worst = max(worst, (got - ref).abs().max().item())
        agree += int((got.argmax(-1) == ref.argmax(-1)).sum())
        total += steps
    return {"max_abs_logit_diff": worst, "argmax_agree": agree, "positions": total}


@torch.no_grad()
def greedy_agreement(m, dev, n=8, prompt=256, new=64):
    """64 greedy tokens after each of n 256-token held-out prompts, three ways.

    `top_k=1` is not exactly greedy: generate keeps every logit equal to the k-th
    and draws with torch.multinomial, so a bf16 tie at the top is a random choice.
    So: (1) a control, the uncached path twice with different seeds; (2) cached vs
    uncached, top_k=1, with the same seed before each call; (3) cached vs uncached
    with true argmax, torch.multinomial replaced by an argmax for the call."""
    x = windows(n, prompt).to(dev)

    def run(row, cache, seed):
        torch.manual_seed(seed)
        with amp(dev, True):
            return m.generate(row[None].clone(), new, top_k=1, use_cache=cache)

    def first_div(a, b):
        return None if torch.equal(a, b) else int((a != b).nonzero()[0, 1]) - prompt

    out = {"prompts": n, "prompt_tokens": prompt, "new_tokens": new}
    out["uncached_vs_uncached_other_seed"] = [first_div(run(r, False, 1), run(r, False, 2)) for r in x]
    out["cached_vs_uncached_same_seed"] = [first_div(run(r, False, 1), run(r, True, 1)) for r in x]
    orig = torch.multinomial
    torch.multinomial = lambda p, num_samples: p.argmax(-1, keepdim=True)
    try:
        out["cached_vs_uncached_argmax"] = [first_div(run(r, False, 1), run(r, True, 1)) for r in x]
    finally:
        torch.multinomial = orig
    return out


def timed(fn, reps=25):
    out = []
    for _ in range(reps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        out.append(time.perf_counter() - t0)
    return 1000 * float(np.median(out))


def step_times(m, dev, bs, sweeps=3):
    """The sweep below, run `sweeps` times; per context, the lowest of the sweeps' medians.
    The machine is shared by other jobs, which at batch 1 (launch-bound) can double a
    sweep's time for a whole context; the lowest sweep is the least disturbed one. Every
    sweep's value is kept in the output."""
    runs = [one_sweep(m, dev, bs) for _ in range(sweeps)]
    rows = [{"context": r[0]["context"],
             "cached_ms": min(x["cached_ms"] for x in r),
             "uncached_ms": min(x["uncached_ms"] for x in r),
             "sweeps_cached": [x["cached_ms"] for x in r], "sweeps_uncached": [x["uncached_ms"] for x in r]}
            for r in zip(*[run[0] for run in runs])]
    return rows, min(run[1] for run in runs)


@torch.no_grad()
def one_sweep(m, dev, bs):
    x = windows(1, bs).to(dev)
    rows = []
    for t in (64, 128, 256, 512, 768, bs - 1):
        with amp(dev, True):
            cache = m.new_cache()
            m(x[:, :t], cache=cache)                              # prefill t tokens once

            def decode():                                         # one cached step at position t
                for c in cache:                                   # drop the entry the last step appended
                    if c["k"].shape[2] > t:
                        c["k"], c["v"] = c["k"][:, :, :t], c["v"][:, :, :t]
                m(x[:, t:t + 1], cache=cache, start=t)
            ms_dec = timed(decode)
            ms_full = timed(lambda: m(x[:, :t + 1]))              # the uncached path: all t + 1 tokens again
        rows.append({"context": t + 1, "cached_ms": round(ms_dec, 2), "uncached_ms": round(ms_full, 2)})
    with amp(dev, True):
        ms_rebuild = timed(lambda: m(x[:, :bs], cache=m.new_cache(), start=0))
    return rows, round(ms_rebuild, 2)


def main():
    torch.manual_seed(SEED)
    m, cfg, dev = load()
    set_impl(m, "grouped" if dev == "cuda" else "sparse")
    print(f"device {dev}; block_size {cfg.block_size}")
    res = {"checkpoint": "ckpt_base_v2.pt", "data": "data/v2/val.u16", "seed": SEED, "device": dev,
           "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None, "torch": torch.__version__}
    res["cache_bf16_autocast"] = cache_contents(m, dev)
    print("cache under bf16 autocast:", res["cache_bf16_autocast"])
    res["fp32_precision"] = "fp32, no autocast, moe_impl sparse (the checkpoint's grouped path would run experts in bf16)"
    res["bf16_precision"] = "bf16 autocast, moe_impl grouped on CUDA, as serve.py"
    res["cached_vs_full_fp32"] = cached_vs_full(m, dev, bf16=False)
    print("cached vs full, fp32:", res["cached_vs_full_fp32"])
    res["cached_vs_full_bf16"] = cached_vs_full(m, dev, bf16=True)
    print("cached vs full, bf16:", res["cached_vs_full_bf16"])
    set_impl(m, "grouped" if dev == "cuda" else "sparse")
    res["greedy_bf16"] = greedy_agreement(m, dev)
    print("greedy, cached vs uncached, bf16:", res["greedy_bf16"])
    rows, rebuild = step_times(m, dev, cfg.block_size)
    res["step_ms_bf16"], res["rebuild_ms_bf16"] = rows, rebuild
    for r in rows:
        print(f"context {r['context']:5d}: cached step {r['cached_ms']:6.2f} ms, uncached forward {r['uncached_ms']:6.2f} ms")
    print(f"rebuild at block_size ({cfg.block_size} tokens): {rebuild:.2f} ms")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
