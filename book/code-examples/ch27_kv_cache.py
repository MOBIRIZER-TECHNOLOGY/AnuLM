"""
Book example, not repository code.

Chapter 27: the KV cache in `AnuLM.forward` and `AnuLM.generate`, checked
against the uncached path, on the test model `tiny()` (CPU, under a second).

    python book/code-examples/ch27_kv_cache.py

It runs `model.py` unchanged. It prints the cache growing through a prefill
and two decode steps, the offset mask for a chunk against a non-empty cache
(and what goes wrong with `is_causal` there), the calls `generate` makes before
and after the context fills, why the cache cannot simply drop its oldest entry,
and the bytes a cached token costs. Every number in Chapter 27 that is not
from the released model comes from this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm import model as M  # noqa: E402  the repository's model.py, unchanged

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def tiny(**kw) -> M.AnuLMConfig:
    """test_model.py's tiny(), copied so this script needs no test harness."""
    base = dict(vocab_size=64, block_size=32, n_layer=3, hidden_size=32, n_head=4,
                head_dim=8, n_kv_head=2, num_experts=8, num_experts_per_tok=2,
                moe_intermediate_size=16, intermediate_size=64, n_group=1, topk_group=1)
    base.update(kw)
    return M.AnuLMConfig(**base)


def shape(t):
    return tuple(t.shape)


@torch.no_grad()
def prefill_then_decode():
    torch.manual_seed(0)
    m = M.AnuLM(tiny()).eval()
    x = torch.randint(0, 64, (2, 7))                  # (B, S) = (2, 7)
    cache = m.new_cache()                             # one empty dict per layer
    logits, _ = m(x[:, :5], cache=cache, start=0)     # prefill: 5 tokens, one call
    print("prefill start=0 S=5  k", shape(cache[0]["k"]), " logits", shape(logits))
    for t in (5, 6):                                  # decode: one token per call
        logits, _ = m(x[:, t:t + 1], cache=cache, start=t)
        print(f"decode  start={t} S=1  k", shape(cache[0]["k"]), " logits", shape(logits))
    full, _ = m(x, x)                                 # no cache: all positions at once
    err = (logits[:, -1] - full[:, -1]).abs().max().item()
    print(f"last position, cached vs full forward: max |difference| {err:.1e}")
    assert err < 1e-4


@torch.no_grad()
def offset_mask():
    right = M._attn_mask(3, 5, 2, None, "cpu")        # queries at 2, 3, 4; keys 0..4
    top_left = torch.ones(3, 5, dtype=torch.bool).tril()   # is_causal for 3 x 5
    for name, mk in (("_attn_mask", right), ("is_causal", top_left)):
        rows = ["".join("x" if v else "." for v in r) for r in mk.tolist()]
        print(f"{name:11s}", "  ".join(rows))
    torch.manual_seed(0)
    m = M.AnuLM(tiny()).eval()
    x = torch.randint(0, 64, (1, 5))
    full, _ = m(x, x)
    errs = {}
    for name, fn in (("_attn_mask", M._attn_mask), ("is_causal", lambda *a: None)):
        orig, M._attn_mask = M._attn_mask, fn         # None makes _sdpa use is_causal
        try:
            cache = m.new_cache()
            m(x[:, :2], cache=cache, start=0)         # two tokens cached
            got, _ = m(x[:, 2:], cache=cache, start=2)    # then a chunk of three
        finally:
            M._attn_mask = orig
        errs[name] = (got[:, -1] - full[:, -1]).abs().max().item()
        print(f"chunk of 3 after 2 cached, {name:10s}: max |diff| {errs[name]:.1e}")
    assert errs["_attn_mask"] < 1e-4 < errs["is_causal"]


@torch.no_grad()
def the_slide():
    torch.manual_seed(0)
    m = M.AnuLM(tiny(block_size=8)).eval()
    calls = []

    def record(mod, args, kw):                        # runs before every forward
        calls.append((kw.get("start", 0), args[0].shape[1]))

    hook = m.register_forward_pre_hook(record, with_kwargs=True)
    idx = torch.randint(0, 64, (1, 6))
    a = m.generate(idx.clone(), 5, top_k=1, use_cache=True)
    hook.remove()
    b = m.generate(idx.clone(), 5, top_k=1, use_cache=False)
    print("cached calls (start, S):", calls)
    print("cached == uncached:", torch.equal(a, b))
    assert torch.equal(a, b) and calls == [(0, 6), (6, 1), (7, 1), (0, 8), (0, 8)]
    return m, a


@torch.no_grad()
def why_not_drop_the_oldest(m, seq):
    """Fill the cache with 8 tokens, drop the oldest entry, decode the next token at
    position 8. RoPE is relative, so keys at 1..7 and a query at 8 have the same
    distances as keys at 0..6 and a query at 7: any difference from the rebuild
    comes from what the cached entries were computed from."""
    bs = m.cfg.block_size
    ctx, nxt = seq[:, -bs - 1:-1], seq[:, -1:]        # 8 tokens, then the next one
    cache = m.new_cache()
    m(ctx, cache=cache, start=0)
    for c in cache:                                   # the shortcut: forget the oldest
        c["k"], c["v"] = c["k"][:, :, 1:], c["v"][:, :, 1:]
    m.cfg.block_size = bs + 1                         # let position 8 past the check
    try:
        dropped, _ = m(nxt, cache=cache, start=bs)
    finally:
        m.cfg.block_size = bs
    ref, _ = m(seq[:, -bs:])                          # what generate computes instead
    err = (dropped[:, -1] - ref[:, -1]).abs().max().item()
    print(f"drop-the-oldest cache vs the rebuild: max |logit difference| {err:.1e}")
    assert err > 1e-3


@torch.no_grad()
def drop_oldest_generate(m, idx, n):
    """Greedy decoding with the shortcut: keep the cache at block_size - 1 entries
    before each step, positions growing past block_size (rotary tables grow)."""
    bs, cache = m.cfg.block_size, m.new_cache()
    m.cfg.block_size = 10 ** 4                        # let positions pass the check
    try:
        logits, _ = m(idx, cache=cache, start=0)
        pos = idx.shape[1]
        for i in range(n):
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            idx = torch.cat((idx, nxt), 1)
            if i == n - 1:
                break
            for c in cache:                           # GQA: (B, H, T, D); MLA latent: (B, T, D)
                for key in c:
                    c[key] = c[key][:, :, -(bs - 1):] if c[key].dim() == 4 else c[key][:, -(bs - 1):]
            logits, _ = m(nxt, cache=cache, start=pos)
            pos += 1
    finally:
        m.cfg.block_size = bs
    return idx


@torch.no_grad()
def does_the_test_catch_it(seeds=20):
    """cached_generation_matches_uncached's setting (tiny, block_size 24, prompt 6,
    30 tokens), with the drop-the-oldest shortcut in place of generate's cache."""
    caught = {"gqa": 0, "mla": 0}
    for seed in range(seeds):
        for attn in ("gqa", "mla"):
            torch.manual_seed(seed)
            m = M.AnuLM(tiny(attn=attn, block_size=24)).eval()
            s = torch.randint(0, 64, (2, 6))
            a = m.generate(s.clone(), 30, temperature=1.0, top_k=1, use_cache=False)
            b = drop_oldest_generate(m, s.clone(), 30)
            caught[attn] += not torch.equal(a, b)
            if seed == 0:
                print(f"seed 0, {attn}: shortcut gives {'different' if not torch.equal(a, b) else 'the same'} tokens")
    print(f"over {seeds} seeds the test's comparison catches the shortcut: "
          f"GQA {caught['gqa']}/{seeds}, MLA {caught['mla']}/{seeds}")
    return caught


@torch.no_grad()
def bytes_per_token():
    torch.manual_seed(0)
    m = M.AnuLM(tiny()).eval()
    for name, ctx in (("fp32", torch.autocast("cpu", enabled=False)),
                      ("bf16 autocast", torch.autocast("cpu", dtype=torch.bfloat16))):
        cache = m.new_cache()
        with ctx:
            m(torch.randint(0, 64, (1, 4)), cache=cache)
        k, v = cache[0]["k"], cache[0]["v"]
        print(f"{name:14s} keys {k.dtype}, values {v.dtype}")
    n = 20 * 4 * 64                                   # L * Hkv * Dh, released model
    for name, kb, vb in (("all bf16", 2, 2), ("fp32 K, bf16 V", 4, 2), ("all fp32", 4, 4)):
        per = n * (kb + vb)                           # keys plus values, in bytes
        print(f"released, {name:14s}: {per:6,} bytes/token, "
              f"{per * 1024 / 2**20:.0f} MiB per 1,024 tokens")


def positions_processed(prompt, new, bs, cached):
    """Token positions run, summed over generate's calls (its lines 1144-1151)."""
    total, length, start = 0, prompt, None
    for _ in range(new):
        if not cached:
            total += min(length, bs)
        elif start is None or start + 1 > bs:         # (re)build: last bs tokens
            total, start = total + min(length, bs), min(length, bs)
        else:
            total, start = total + 1, start + 1
        length += 1
    return total


if __name__ == "__main__":
    print("== prefill, then decode")
    prefill_then_decode()
    print("\n== the offset mask")
    offset_mask()
    print("\n== the slide at block_size = 8 (prompt 6, 5 new tokens)")
    m, seq = the_slide()
    why_not_drop_the_oldest(m, seq)
    does_the_test_catch_it()
    print("\n== what a cached token costs")
    bytes_per_token()
    print("\n== token positions run, released block_size 1,024, prompt 100")
    assert positions_processed(6, 5, 8, True) == 6 + 1 + 1 + 8 + 8
    for new in (100, 900, 1900):
        u = positions_processed(100, new, 1024, False)
        c = positions_processed(100, new, 1024, True)
        print(f"{new:5d} new: uncached {u:9,}  cached {c:9,}  ratio {u / c:5.1f}")
    ex = positions_processed(1000, 24, 1024, True), positions_processed(1000, 24, 1024, False)
    print(f"exercise 1, prompt 1,000 and 24 new: cached {ex[0]:,}, uncached {ex[1]:,}")
    assert ex == (1023, 24276)
