"""
Book example, not repository code.

Chapter 26 measurement: how many expert choices would the released model's
routers change if the gate ran under bf16 autocast, as every GPU run did before
the fix recorded in Router.forward?

    python book/code-examples/ch26_measure_router_bf16.py      # needs the GPU, under a minute

It loads ckpt_base_v2.pt, runs 32 strided windows of 1,024 tokens from the
model's own held-out split (data/v2/val.u16) under bf16 autocast, as train.py
and the serving scripts do, and captures what each of the 19 routers receives.
For every token it then compares two expert sets:

  fp32   Router.forward as committed: autocast off for the whole gate;
  bf16   the gate before the fix: the same Router._gate, inputs cast to fp32,
         but called with autocast on, so F.linear is re-cast to bf16.

Both read the same weights, bias and inputs; only the precision of the
score computation differs. Only counts are kept; no text is decoded.
With --save the results go to book/measurements/part_v_b_router_bf16.json;
without it, to the current directory.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm.training import train as T  # noqa: E402  the repository's train.py, unchanged
from anulm.model import AnuLM, Router, load_checkpoint  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# Results go to book/measurements/ only with --save, so a reader's rerun never overwrites them.
SAVE = "--save" in sys.argv[1:]
OUT = (ROOT / "book" / "measurements" if SAVE else Path.cwd()) / "part_v_b_router_bf16.json"
DEV, BLOCK, N_WIN = "cuda", 1024, 32


def main():
    t0 = time.time()
    ck = load_checkpoint(str(ROOT / "ckpt_base_v2.pt"), "cpu")
    m = AnuLM(replace(ck["cfg"], moe_impl="grouped")).to(DEV).eval()
    m.load_state_dict(ck["model"])
    _, va, _ = T.load_memmap(str(ROOT / "data" / "v2" / "meta.json"))
    routers = [(i, l.mlp.router) for i, l in enumerate(m.layers) if isinstance(getattr(l.mlp, "router", None), Router)]
    seen: dict[int, list[torch.Tensor]] = {i: [] for i, _ in routers}
    hooks = [r.register_forward_pre_hook(lambda mod, a, i=i: seen[i].append(a[0].detach()))
             for i, r in routers]

    counts = {i: [0, 0, 0] for i, _ in routers}             # tokens, sets differ, top-1 differs
    with torch.no_grad():
        for ix in T.strided_windows(va, 4, BLOCK, N_WIN):
            x, y = T.make_batch(va, ix, BLOCK, DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                m(x, y)
            for i, r in routers:
                h = seen[i].pop()
                idx32, w32, _ = r(h)                                         # as committed
                seen[i].clear()                                              # r(h) fired the hook too
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    idx16, w16, _ = r._gate(h.float(), None)                 # the gate before the fix
                s32, s16 = idx32.sort(-1)[0], idx16.sort(-1)[0]
                counts[i][0] += h.shape[0]
                counts[i][1] += int((s32 != s16).any(-1).sum())
                counts[i][2] += int((idx32[:, 0] != idx16[:, 0]).sum())
    for h in hooks:
        h.remove()

    per_layer = {i: {"tokens": c[0], "set_differs": c[1] / c[0], "first_choice_differs": c[2] / c[0]}
                 for i, c in counts.items()}
    tot = sum(c[0] for c in counts.values())
    allset = sum(c[1] for c in counts.values()) / tot
    print(f"{N_WIN} windows x {BLOCK} tokens, {len(routers)} routers, top-{routers[0][1].top_k} of "
          f"{routers[0][1].num_experts}")
    for i, d in per_layer.items():
        print(f"  layer {i:2d}: expert set differs for {100 * d['set_differs']:5.2f}% of tokens "
              f"(first choice {100 * d['first_choice_differs']:5.2f}%)")
    print(f"all layers: {100 * allset:.2f}% of (token, layer) expert sets differ "
          f"({sum(c[1] for c in counts.values()):,} of {tot:,})")
    sets = [d["set_differs"] for d in per_layer.values()]
    R = {"checkpoint": "ckpt_base_v2.pt", "windows": N_WIN, "block": BLOCK, "routers": len(routers),
         "per_layer": per_layer, "all_layers_set_differs": allset, "token_layer_pairs": tot,
         "min_layer": min(sets), "max_layer": max(sets), "seconds": time.time() - t0}
    OUT.write_text(json.dumps(R, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({R['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
