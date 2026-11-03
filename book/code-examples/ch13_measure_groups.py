"""
Book example, not repository code.

Chapter 13: do query heads that share a key/value head attend alike?

    python book/code-examples/ch13_measure_groups.py     # GPU if free, else CPU (slow); add --save to write book/measurements/

Runs `ckpt_base_v2.pt` through `model.py` unchanged over 16 windows of 512
tokens from the base run's held-out split (`data/v2/val.u16`), seed 0. It wraps
`model._sdpa` as `measure_attention.py` does: the wrapper calls the original
and recomputes the attention weights beside it. For every layer and every pair
of the 16 query heads it records the total-variation distance between their
weight rows, ½ Σ_t |w_i(t) − w_j(t)|, averaged over queries at positions
64–511. Pairs in the same group (same key/value head, h // 4) and pairs in
different groups are averaged separately.

Precision: fp32 with no autocast, except the experts, which the released
configuration runs in bf16 on its grouped path (`model.py` 756, 784).

Only statistics are saved; no held-out text is printed.
With --save, results go to book/measurements/part_iv_a_groups.json.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm import model as M  # noqa: E402  the repository's model.py, unchanged

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# Writes to book/measurements/ only with --save; otherwise to the current directory.
OUT = (ROOT / "book" / "measurements" if "--save" in sys.argv else Path.cwd()) / "part_iv_a_groups.json"
WINDOWS, SEQ, BATCH, SEED, LO = 16, 512, 4, 0, 64


def load():
    ck = torch.load(ROOT / "ckpt_base_v2.pt", map_location="cpu", weights_only=False)
    m = M.AnuLM(ck["cfg"])
    m.load_state_dict(ck["model"], strict=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return m.to(dev).eval(), ck["cfg"], dev


def main() -> None:
    m, cfg, dev = load()
    L, H, rep = cfg.n_layer, cfg.n_head, cfg.n_head // cfg.n_kv_head
    pairs = list(itertools.combinations(range(H), 2))
    tv = torch.zeros(L, len(pairs), dtype=torch.float64)
    calls = {"n": 0}
    orig = M._sdpa

    def wrapped(q, k, v, mask, S, scale, drop):
        y = orig(q, k, v, mask, S, scale, drop)
        layer = calls["n"] % L
        calls["n"] += 1
        T = k.shape[-2]
        s = (q.float() @ k.float().transpose(-2, -1)) * scale                 # (B, H, S, T)
        if mask is None:
            mask = torch.ones(S, T, dtype=torch.bool, device=q.device).tril(T - S)
        w = s.masked_fill(~mask, float("-inf")).softmax(-1)
        err = ((w @ v.float()) - y.float()).abs().max().item()
        assert err < 1e-3, f"layer {layer}: recomputed weights differ from _sdpa by {err:.1e}"
        rows = w[:, :, LO:]                                                    # queries 64..511
        for n, (i, j) in enumerate(pairs):
            tv[layer, n] += 0.5 * (rows[:, i] - rows[:, j]).abs().sum(-1).mean().double().cpu()
        return y

    val = np.memmap(ROOT / "data" / "v2" / "val.u16", dtype=np.uint16, mode="r")
    starts = np.random.default_rng(SEED).integers(0, len(val) - SEQ - 1, size=WINDOWS)
    M._sdpa = wrapped
    try:
        with torch.no_grad():
            for b in range(0, WINDOWS, BATCH):
                rows = np.stack([val[s: s + SEQ] for s in starts[b: b + BATCH]]).astype(np.int64)
                m(torch.from_numpy(rows).to(dev))
    finally:
        M._sdpa = orig
    batches = WINDOWS // BATCH
    assert calls["n"] == L * batches, "one _sdpa call per layer per batch (NATIVE_GQA must be off)"
    tv /= batches
    same = torch.tensor([i // rep == j // rep for i, j in pairs])
    res = {"checkpoint": "ckpt_base_v2.pt", "data": "data/v2/val.u16", "windows": WINDOWS, "seq": SEQ,
           "seed": SEED, "queries_from_position": LO, "precision": "fp32, no autocast, except the experts: bf16 on the released grouped path (model.py 756, 784)", "device": dev,
           "pairs_within": int(same.sum()), "pairs_across": int((~same).sum()),
           "within": tv[:, same].mean(1).tolist(), "across": tv[:, ~same].mean(1).tolist()}
    flat = [(tv[l, n].item(), l, i, j) for l in range(L) for n, (i, j) in enumerate(pairs)]
    res["closest_pairs"] = [{"tv": t, "layer": l, "heads": [i, j], "same_group": i // rep == j // rep}
                            for t, l, i, j in sorted(flat)[:10]]
    w_all, a_all = tv[:, same].mean().item(), tv[:, ~same].mean().item()
    res["within_all"], res["across_all"] = w_all, a_all
    closer = sum(res["within"][l] < res["across"][l] for l in range(L))
    res["layers_within_closer"] = closer
    print(f"device {dev}; {WINDOWS} windows of {SEQ}, seed {SEED}; {res['pairs_within']} pairs within groups, "
          f"{res['pairs_across']} across")
    for l in range(L):
        print(f"  layer {l:2d}: within {res['within'][l]:.3f}  across {res['across'][l]:.3f}")
    print(f"  all layers: within {w_all:.3f}, across {a_all:.3f}; within closer in {closer} of {L} layers")
    print("  closest pairs:", ", ".join(f"L{p['layer']} h{p['heads'][0]}–h{p['heads'][1]} {p['tv']:.3f}"
                                         f"{' (same group)' if p['same_group'] else ''}"
                                         for p in res["closest_pairs"][:5]))
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
