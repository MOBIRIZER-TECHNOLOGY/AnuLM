"""Grow a trained AnuLM into a larger one that computes the same function.

    python -m anulm.training.grow ckpt_base_v2.pt ckpt_base_1b.pt.last --sampler-from ckpt_base_v2.pt.last
    python -m anulm.training.grow ckpt_base_v2.pt ckpt_base_1b.pt.last --experts-width 512 \
        --insert-after 11 14 17 19

Two kinds of growth, both exact at step 0 (docs/PLAN_1B.md):

  wider experts   every routed expert's hidden width I -> I' (192 -> 512 for the
                  1B plan). The new gate/up rows get a fresh random init; the
                  matching new columns of down_proj are zero, so each expert's
                  output is unchanged until training moves them. Routing is
                  untouched: same experts, same router, same expert biases.
  deeper          new decoder layers inserted after the listed old layers. A new
                  layer starts as the identity: its attention output projection
                  (`dense`) and every expert's down_proj are zero, so the
                  residual stream passes through unchanged.

A zero output projection still receives a gradient (its input is not zero),
and once it moves, the rows feeding it do too, so nothing is stuck at zero.

The output is written as a resume point (`<name>.pt.last`, step -1, no
optimizer state): `train.py --resume --out <name>.pt` then starts at step 0 with
these weights and a fresh optimizer. `train.py` needs the grown shape on its
command line; `--print-cfg` prints the `--cfg` overrides that rebuild it.

Insert points: layers [0, max_window_layers) use sliding-window attention, by
index. New layers must go at or after that point, or the old layers behind them
would change attention type; the tool refuses otherwise.
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch

from anulm.model import AnuLM, AnuLMConfig, safe_load


def layer_plan(n_old: int, insert_after: list[int]) -> list[int | None]:
    """For each new layer, the old layer it copies, or None for an inserted one."""
    if any(i < 0 or i >= n_old for i in insert_after):
        raise ValueError(f"--insert-after must name old layers 0..{n_old - 1}")
    plan: list[int | None] = []
    for i in range(n_old):
        plan.append(i)
        plan.extend([None] * insert_after.count(i))
    return plan


def grown_config(cfg: AnuLMConfig, experts_width: int, insert_after: list[int]) -> AnuLMConfig:
    if experts_width < cfg.moe_intermediate_size:
        raise ValueError("experts can only grow wider")
    plan = layer_plan(cfg.n_layer, insert_after)
    if cfg.sliding_window and cfg.max_window_layers:
        first_new = plan.index(None) if None in plan else len(plan)
        if first_new < cfg.max_window_layers:
            raise ValueError(f"a new layer at index {first_new} would shift windowed layers; "
                             f"insert after layer {cfg.max_window_layers - 1} or later")
    if cfg.first_k_dense_replace and None in plan and plan.index(None) < cfg.first_k_dense_replace:
        raise ValueError("new layers must come after the dense layers")
    return dataclasses.replace(cfg, n_layer=len(plan), moe_intermediate_size=experts_width)


@torch.no_grad()
def grow(old_sd: dict, old_cfg: AnuLMConfig, new_cfg: AnuLMConfig, insert_after: list[int],
         seed: int = 0) -> AnuLM:
    torch.manual_seed(seed)
    model = AnuLM(new_cfg)                       # fresh init for everything new
    new_sd = model.state_dict()
    plan = layer_plan(old_cfg.n_layer, insert_after)

    def put(key: str, value: torch.Tensor, rows: int | None = None, cols: int | None = None):
        dst = new_sd[key]
        if rows is not None:
            dst[:rows] = value
        elif cols is not None:
            dst[:, :cols] = value
        else:
            if dst.shape != value.shape:
                raise ValueError(f"{key}: {tuple(value.shape)} -> {tuple(dst.shape)}")
            dst.copy_(value)

    # Everything outside the layers: embeddings, final norm, head, rotary buffers.
    for k, v in old_sd.items():
        if not k.startswith("layers."):
            put(k, v)

    for new_i, old_i in enumerate(plan):
        pre_new = f"layers.{new_i}."
        if old_i is None:
            # Identity at start: zero the residual-stream writers.
            for k in new_sd:
                if k.startswith(pre_new) and k.endswith(("attn.dense.weight", "attn.o_proj.weight",
                                                         "down_proj.weight")):
                    new_sd[k].zero_()
            continue
        pre_old = f"layers.{old_i}."
        for k, v in old_sd.items():
            if not k.startswith(pre_old):
                continue
            nk = pre_new + k[len(pre_old):]
            if (".experts." in nk or ".shared_expert." in nk) and new_sd[nk].shape != v.shape:
                # Routed experts, and a shared expert (width = experts width x count) if any.
                if nk.endswith(("gate_proj.weight", "up_proj.weight")):
                    put(nk, v, rows=v.shape[0])          # new rows keep their fresh init
                    continue
                if nk.endswith("down_proj.weight"):
                    new_sd[nk].zero_()
                    put(nk, v, cols=v.shape[1])          # new columns stay zero
                    continue
            if nk.endswith("router.load_counts"):
                new_sd[nk].zero_()
                continue
            put(nk, v)

    model.load_state_dict(new_sd)
    return model


def cfg_overrides(old: AnuLMConfig, new: AnuLMConfig, preset: str = "350m") -> list[str]:
    """`--cfg KEY=VALUE` items that rebuild `new` from train.py's `--preset` (base v2
    was trained with 350m). The fields train.py sets from the data and its own
    flags are left out."""
    from anulm.training.train import PRESETS
    base = PRESETS[preset]()
    out = []
    for f in dataclasses.fields(AnuLMConfig):
        v = getattr(new, f.name)
        if v != getattr(base, f.name) and f.name not in ("vocab_size", "tokenizer_path", "block_size",
                                                         "yarn_original_context", "moe_impl"):
            out.append(f"{f.name}={v}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="trained checkpoint (.pt with model + cfg)")
    ap.add_argument("out", help="resume point to write, e.g. ckpt_base_1b.pt.last")
    ap.add_argument("--experts-width", type=int, default=512)
    ap.add_argument("--insert-after", type=int, nargs="*", default=[11, 14, 17, 19])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--print-cfg", action="store_true", help="only print train.py --cfg overrides")
    ap.add_argument("--preset", default="350m", choices=("tiny-gqa", "tiny-mla", "350m", "1b"),
                    help="the train.py --preset the overrides apply to")
    ap.add_argument("--sampler-from", help="a .last resume point whose data position (and eval spec) "
                    "to carry over, so training continues on the windows that run never read")
    a = ap.parse_args()

    ck = safe_load(a.src, map_location="cpu")
    old_cfg = ck["cfg"]
    new_cfg = grown_config(old_cfg, a.experts_width, a.insert_after)
    if a.print_cfg:
        print(" ".join(cfg_overrides(old_cfg, new_cfg, a.preset)))
        return
    model = grow(ck["model"], old_cfg, new_cfg, a.insert_after, a.seed)
    total, active = model.num_params()
    out = Path(a.out)
    if not out.name.endswith(".last"):
        raise SystemExit("write a resume point: the name must end in .last (train.py --resume)")
    extra = {}
    if a.sampler_from:
        last = torch.load(a.sampler_from, map_location="cpu", weights_only=False, mmap=True)
        extra = {"sampler": last["sampler"], "eval_spec": last.get("eval_spec")}
        done, total_w = last["sampler"]["pos"], len(last["sampler"]["order"])
        print(f"data position from {a.sampler_from}: epoch {last['sampler']['epoch']}, "
              f"window {done:,} of {total_w:,} ({total_w - done:,} not yet read)")
    torch.save({"model": model.state_dict(), "cfg": new_cfg, "step": -1, **extra,
                "best_val": float("inf"),
                "grown_from": {"ckpt": str(a.src), "step": ck.get("step"), "val_loss": ck.get("val_loss"),
                               "experts_width": a.experts_width, "insert_after": a.insert_after,
                               "seed": a.seed}}, out)
    print(f"{a.src} ({old_cfg.n_layer} layers, experts {old_cfg.moe_intermediate_size}) -> {out} "
          f"({new_cfg.n_layer} layers, experts {new_cfg.moe_intermediate_size}): "
          f"{total / 1e6:.0f}M total, {active / 1e6:.0f}M active")


if __name__ == "__main__":
    main()
