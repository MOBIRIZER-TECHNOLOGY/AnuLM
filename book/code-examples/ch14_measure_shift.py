"""
Book example, not repository code.

Chapter 14: is the released model's attention relative in practice?

    python book/code-examples/ch14_measure_shift.py     # GPU if free, else CPU (slow); add --save to write book/measurements/

RoPE makes a query–key score depend only on the gap between their positions,
so moving every position of a window by the same offset should change
nothing. This script runs `ckpt_base_v2.pt` through `model.py` unchanged over
16 windows of 512 tokens from the held-out split (`data/v2/val.u16`, seed 0),
with the rotary tables asked for positions offset+0 … offset+511 instead of
0 … 511. The tables come from the repository's own `Rotary` (its `_grow` builds
them in fp32); only the `start` it is called with is shifted. For each offset
it records, against offset 0:

- the loss;
- the change in layer 0's attention output (before any expert routing), as
  the largest absolute change and relative to the output's largest value;
- the change in the logits: the largest, the median over positions of each
  position's largest change, and the share of positions whose top-1 token
  differs;
- how many (token, MoE layer) pairs were routed to a different set of
  experts.

Offset 0 is run twice; the second run is a determinism control. It does all
this twice: with the experts on the released configuration's own path,
`moe_impl = "grouped"`, which runs them in bf16 (`model.py` 756, 784), and on
the `"sparse"` path, which keeps the whole forward pass in fp32. The router
itself runs in fp32 on both paths (580–586). A router's choice is discrete,
so a tiny change upstream can switch an expert.

Only statistics are saved; no held-out text is printed.
With --save, results go to book/measurements/part_iv_a_shift.json.
"""

from __future__ import annotations

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
OUT = (ROOT / "book" / "measurements" if "--save" in sys.argv else Path.cwd()) / "part_iv_a_shift.json"
WINDOWS, SEQ, BATCH, SEED, EOS = 16, 512, 4, 0, 32767
RUNS = (("0", 0), ("0, again", 0), ("1024", 1_024), ("65536", 65_536), ("1048576", 1_048_576))


def forward_all(m, val, starts, dev, att, routes):
    """Loss, logits, layer-0 attention output and expert choices over every window."""
    tot, n, outs, atts = 0.0, 0, [], []
    routes.clear()
    with torch.no_grad():
        for b in range(0, WINDOWS, BATCH):
            rows = np.stack([val[s: s + SEQ + 1] for s in starts[b: b + BATCH]]).astype(np.int64)
            x = torch.from_numpy(rows[:, :-1]).to(dev)
            y = torch.from_numpy(rows[:, 1:]).to(dev).masked_fill(x == EOS, -1)
            logits, loss = m(x, y)
            k = int((y != -1).sum())
            tot, n = tot + loss.item() * k, n + k
            outs.append(logits.float().cpu())
            atts.append(att["y"])
    per_layer = {}
    for layer, idx in routes:                                              # one entry per layer per batch
        per_layer.setdefault(layer, []).append(idx)
    chosen = torch.stack([torch.cat(per_layer[l]) for l in sorted(per_layer)])   # (MoE layers, N, A), sorted ids
    return tot / n, n, torch.cat(outs), torch.cat(atts), chosen


def main() -> None:
    ck = torch.load(ROOT / "ckpt_base_v2.pt", map_location="cpu", weights_only=False)
    m = M.AnuLM(ck["cfg"])
    m.load_state_dict(ck["model"], strict=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = m.to(dev).eval()
    val = np.memmap(ROOT / "data" / "v2" / "val.u16", dtype=np.uint16, mode="r")
    starts = np.random.default_rng(SEED).integers(0, len(val) - SEQ - 1, size=WINDOWS)
    real = m.rotary.forward
    att = {}
    m.layers[0].attn.register_forward_hook(lambda mod, inp, out: att.__setitem__("y", out.float().cpu()))
    routes = []
    for i, layer in enumerate(m.layers):
        if layer.is_moe:                     # Router.forward returns (topk_idx, topk_w, aux); keep the ids
            layer.mlp.router.register_forward_hook(
                lambda mod, inp, out, i=i: routes.append((i, out[0].sort(-1).values.cpu())))
    res = {"checkpoint": "ckpt_base_v2.pt", "data": "data/v2/val.u16", "windows": WINDOWS, "seq": SEQ,
           "seed": SEED, "device": dev,
           "precision": "no autocast; sparse run: fp32 throughout; grouped run: fp32 except the experts, "
                        "bf16 (model.py 756, 784); the router is fp32 on both"}
    for impl in ("sparse", "grouped"):
        for layer in m.layers:
            if layer.is_moe:
                layer.mlp.impl = impl
        print(f"experts on the {impl!r} path")
        res[impl], base, base_att, base_chosen = {}, None, None, None
        for label, off in RUNS:
            m.rotary.forward = lambda S, device, dtype, start=0, off=off: real(S, device, dtype, start=start + off)
            try:
                loss, n, out, a, chosen = forward_all(m, val, starts, dev, att, routes)
            finally:
                m.rotary.forward = real
            if base is None:
                base, base_att, base_chosen = out, a, chosen
            per_pos = (out - base).abs().amax(-1)                              # (windows, S)
            r = {"loss": loss,
                 "layer0_attn_max_change": (a - base_att).abs().max().item(),
                 "layer0_attn_rel_change": ((a - base_att).abs().max() / base_att.abs().max()).item(),
                 "logit_max_change": per_pos.max().item(),
                 "logit_median_position_change": per_pos.median().item(),
                 "top1_changed_share": (out.argmax(-1) != base.argmax(-1)).float().mean().item(),
                 "routes_changed": int((chosen != base_chosen).any(-1).sum()),
                 "routes_total": int(chosen.shape[0] * chosen.shape[1])}
            res[impl][label] = r
            print(f"  offset {label:>9s}: routes changed {r['routes_changed']} of {r['routes_total']:,}; loss {r['loss']:.4f}; layer-0 attention change "
                  f"{r['layer0_attn_rel_change']:.1e} of its largest value; logits: largest change "
                  f"{r['logit_max_change']:.1e}, median position {r['logit_median_position_change']:.1e}; "
                  f"top-1 changed at {r['top1_changed_share']:.2%} of positions")
    res["graded"] = n
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
