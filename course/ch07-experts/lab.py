"""
Chapter 7 lab: experts against a dense MLP at the same compute, and what the
balancer is for.

    python lab.py                      # your course/build/anulm.py
    python lab.py --impl solution      # the reference solution
    python lab.py --data my_text.txt   # your own text

CPU, about eight minutes. Three runs with the same attention, data and steps
(Chapter 6's sizes), and the same ACTIVE MLP width per token (256):

  dense       one SwiGLU of width 256 in every layer
  balanced    layers 1-3: 8 experts of width 64, top 3, plus a shared expert of 64
  unbalanced  the same, with the balancing bias switched off (rate 0)

Then, on the balanced model's first expert layer: how load is spread over the
experts, and how many tokens change experts if the router runs in bfloat16.
Writes results.json for the figures.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch

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
SIZES = dict(vocab_size=259, block_size=256, n_layer=4, hidden_size=128, n_head=4, head_dim=32, n_kv_head=2,
             rope_theta=10000.0)
RUNS = {"dense": dict(first_k_dense_replace=4, intermediate_size=256),
        "balanced": dict(intermediate_size=256, num_experts=8, num_experts_per_tok=3, num_shared_experts=1,
                         moe_intermediate_size=64, bias_update_rate=1e-3),
        "unbalanced": dict(intermediate_size=256, num_experts=8, num_experts_per_tok=3, num_shared_experts=1,
                           moe_intermediate_size=64, bias_update_rate=0.0)}


def train(M, name, kw, train_data, val_data, steps, seed):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = M.MoELM(**SIZES, **kw)
    total, active = model.num_params()
    ctx = 128

    @torch.no_grad()
    def evaluate():
        model.eval()
        eg = torch.Generator().manual_seed(0)
        loss = sum(model(*lab1.batch(val_data, ctx, 32, eg))[1].item() for _ in range(20)) / 20
        model.train()
        return loss

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    curve, imb, t0 = [(0, evaluate())], [], time.time()
    for step in range(1, steps + 1):
        for group in opt.param_groups:
            group["lr"] = 1e-3 * min(1.0, step / 100) * 0.5 * (1 + math.cos(math.pi * step / steps))
        _, loss = model(*lab1.batch(train_data, ctx, 32, g))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ratio = model.update_expert_biases()
        if step % 25 == 0:
            imb.append((step, round(ratio, 3)))
        if step % 100 == 0:
            curve.append((step, evaluate()))
            print(f"   {name:10s} step {step:5d}  validation {curve[-1][1]:.3f}  worst-layer imbalance "
                  f"{ratio:.2f}x  ({time.time() - t0:.0f}s)")
    return model, {"total": total, "active": active, "final": round(curve[-1][1], 4),
                   "seconds": round(time.time() - t0), "curve": [(s, round(v, 4)) for s, v in curve],
                   "imbalance": imb}


@torch.no_grad()
def layer_routing(model, idx):
    """The first expert layer's router: its inputs, for the measurements below."""
    x = model.embed_tokens(idx)
    cos, sin = model.rotary(idx.shape[1], x.device, x.dtype)
    for layer in model.layers:
        if type(layer.mlp).__name__ == "MoE":
            h = layer.input_layernorm(x)
            x = x + layer.attn(h, cos, sin, window=layer.window)
            return layer.mlp.router, layer.post_attention_layernorm(x).view(-1, x.shape[-1])
        x = layer(x, cos, sin)
    raise RuntimeError("no expert layer")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=["build", "solution"], default="build")
    p.add_argument("--data", default=str(COURSE.parent / "data" / "input.txt"))
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--seed", type=int, default=1337)
    a = p.parse_args()
    M = load(COURSE / "build" / ("solutions/ch07.py" if a.impl == "solution" else "anulm.py"))
    try:
        M.MoELM(**SIZES)
    except NotImplementedError as e:
        sys.exit(f"course/build/anulm.py is not finished yet ({e}).\n"
                 f"Fill it in until `python course/checks/ch07.py` passes, or run with --impl solution.")
    train_data, val_data = lab1.load_bytes(a.data)
    out, models = {"steps": a.steps, "data": Path(a.data).name}, {}
    for name, kw in RUNS.items():
        print(f"{name}: training {a.steps} steps")
        models[name], out[name] = train(M, name, kw, train_data, val_data, a.steps, a.seed)
        print(f"   {name}: {out[name]['total']:,} parameters, {out[name]['active']:,} active, "
              f"validation {out[name]['final']:.3f} in {out[name]['seconds']} s")

    idx = val_data[:4 * 128].view(4, 128)
    for name in ("balanced", "unbalanced"):
        r, h = layer_routing(models[name].eval(), idx)
        chosen, _, _ = r(h)
        share = torch.bincount(chosen.flatten(), minlength=r.num_experts).float()
        out[name]["expert_share"] = [round(v, 4) for v in (share / share.sum()).tolist()]
    r, h = layer_routing(models["balanced"].eval(), idx)
    want, _, _ = r(h)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=True):
        logits = torch.nn.functional.linear(h, r.weight)            # what an unguarded router computes
    scores = torch.sigmoid(logits.float()) + r.expert_bias
    got = scores.topk(r.top_k, dim=-1)[1]
    flipped = (got.sort(-1)[0] != want.sort(-1)[0]).any(-1).float().mean().item()
    out["bf16_flip_share"] = round(flipped, 4)
    print(f"bf16 routing: {flipped:.1%} of {h.shape[0]} tokens would choose a different set of experts")
    print("expert shares, first expert layer:")
    for name in ("balanced", "unbalanced"):
        print(f"   {name:10s} " + " ".join(f"{v:.0%}" for v in out[name]["expert_share"]))
    (HERE / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("saved results.json")


if __name__ == "__main__":
    main()
