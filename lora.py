"""LoRA for AnuLM: fine-tune a released checkpoint instead of training one.

    python finetune.py --ckpt release/AnuLM-Base-400M --qa my_pairs.jsonl \
                       --lora --out ckpt_mine.pt
    python sample.py  --ckpt ckpt_mine.pt --prompt "..."

Low-rank adaptation (Hu et al., 2021): freeze W, learn a rank-r correction
`W + (alpha/r) * B @ A` with A of shape (r, in) and B of (out, r). B starts at
zero, so the adapted model begins exactly as the base model did and cannot be
worse at step 0. Only A and B receive gradients.

**Why this is worth having at 398M**, where the usual "the model does not fit"
argument does not apply. Full fine-tuning here costs 1.6 GB of weights plus
1.6 GB of gradients plus 3.2 GB of AdamW moments -- 6.4 GB before activations,
which is why `finetune.py` needs `--grad-ckpt` on an 8 GB card. LoRA trains
~1% of that: gradients and moments exist only for the adapters, so the same
job fits in about 2.5 GB, and what you keep afterwards is a 4-9 MB file rather
than another 1.6 GB checkpoint. Ten fine-tunes of the same base cost ten
adapters, not sixteen gigabytes.

**What it does not do.** It is not QLoRA: the base weights stay in float32
(see `docs/DEVELOPING.md` on why this model must not be run in bfloat16).
Quantising the base to 4 bits needs bitsandbytes and would save ~1.2 GB here
-- worth it at 7B, mostly not worth the dependency at 398M.

**Which layers.** By default the attention projections, which is where LoRA
is normally applied and which is 52M of this model's 398M. The 269M in the
experts are deliberately left alone: adapting 24 experts x 19 layers means
1,368 adapters, most of which see a fraction of the tokens, and the router
decides who learns what. `--lora-targets` overrides this if you want to try.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

DEFAULT_TARGETS = ("attn.query_key_value", "attn.dense")


class LoRALinear(nn.Module):
    """A frozen Linear plus a trainable rank-r correction."""

    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0,
                 dropout: float = 0.0):
        super().__init__()
        assert r > 0, "rank must be positive"
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha = r, alpha
        self.scale = alpha / r
        self.lora_a = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r))
        # A ~ Kaiming, B = 0: the product is zero, so the wrapped layer starts
        # as the one it replaced, to float noise -- the extra add is not
        # always the same instruction sequence, so expect ~1e-8 on the logits
        # rather than bit-equality.
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        h = self.drop(x) @ self.lora_a.t().to(x.dtype)
        return out + (h @ self.lora_b.t().to(x.dtype)) * self.scale

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        return self.base.weight + (self.lora_b @ self.lora_a) * self.scale

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, scale={self.scale:g}"


def apply_lora(model: nn.Module, r: int = 16, alpha: float = 32.0,
               dropout: float = 0.0, targets=DEFAULT_TARGETS) -> int:
    """Freeze everything, then wrap every Linear whose name ends in one of
    `targets`. Returns the number of adapters added."""
    for p in model.parameters():
        p.requires_grad_(False)
    hits = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and any(full.endswith(t) for t in targets):
                setattr(module, child_name, LoRALinear(child, r, alpha, dropout))
                hits += 1
    if not hits:
        raise ValueError(f"no Linear matched {targets}; nothing would train")
    return hits


def lora_parameters(model: nn.Module):
    return [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]


def lora_state_dict(model: nn.Module) -> dict:
    """Just the adapters -- a few MB instead of 1.6 GB."""
    return {n: p.detach().cpu() for n, p in model.state_dict().items() if "lora_" in n}


def router_state_dict(model: nn.Module) -> dict:
    """The aux-loss-free balancer's per-expert bias.

    It is a buffer, not a parameter, so it has no gradient and LoRA does not
    freeze it -- `update_expert_biases()` still nudges it after every step.
    That means the adapters alone do not reproduce the model that was
    trained: merging them into a pristine base leaves the router where the
    base left it, and the experts a token reaches change. It is 24 floats per
    layer, so the adapter carries it and merging restores it.
    """
    return {n: b.detach().cpu() for n, b in model.named_buffers()
            if n.endswith("expert_bias")}


def load_lora(model: nn.Module, state: dict, strict: bool = True) -> None:
    missing = [n for n in state if n not in dict(model.named_parameters())]
    if missing and strict:
        raise KeyError(f"adapter has {len(missing)} keys the model does not: {missing[:3]}")
    model.load_state_dict(state, strict=False)


@torch.no_grad()
def merge_lora(model: nn.Module) -> int:
    """Fold every adapter into its base weight and put the plain Linear back.

    Do this before exporting: a merged model is an ordinary AnuLM checkpoint
    that `serve.py`, `export_hf.py` and the transformers wrapper all load with
    no knowledge of LoRA, and costs nothing at inference.
    """
    merged = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                base = child.base
                base.weight.copy_(child.merged_weight())
                setattr(module, child_name, base)
                merged += 1
    return merged


def summarise(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (f"LoRA: {train/1e6:.2f}M trainable of {total/1e6:.1f}M "
            f"({100*train/total:.2f}%), {train*4/1e6:.1f} MB of gradients and "
            f"{train*8/1e6:.1f} MB of AdamW moments")


# ---------------------------------------------------------------- CLI

def _merge_cli(args) -> int:
    """adapter + base -> an ordinary checkpoint anything here can load."""
    import torch
    from model import AnuLM, load_checkpoint

    ad = torch.load(args.adapter, map_location="cpu", weights_only=False)
    assert "lora" in ad, f"{args.adapter} has no adapters; is it a full checkpoint?"
    base_path = args.base or ad.get("base_ckpt")
    assert base_path, "the adapter does not record its base; pass --base"
    print(f"base    {base_path}")
    print(f"adapter {args.adapter}  ({len(ad['lora'])} tensors, "
          f"r={ad.get('lora_cfg', {}).get('r')})")

    base = load_checkpoint(base_path, "cpu")
    model = AnuLM(base["cfg"])
    model.load_state_dict(base["model"])
    cfg = ad.get("lora_cfg") or {}
    apply_lora(model, cfg.get("r", 16), cfg.get("alpha", 32.0),
               targets=tuple(cfg.get("targets") or DEFAULT_TARGETS))
    load_lora(model, ad["lora"])
    if ad.get("router"):
        model.load_state_dict(ad["router"], strict=False)
        print(f"restored {len(ad['router'])} router biases")
    n = merge_lora(model)

    out = {"model": model.state_dict(), "cfg": base["cfg"],
           "step": ad.get("step"), "val_loss": ad.get("val_loss"),
           "base_ckpt": base_path}
    for k in ("qa_template", "qa_templates"):
        if ad.get(k):
            out[k] = ad[k]
    torch.save(out, args.out)
    print(f"folded {n} adapters -> {args.out}")
    return 0


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge", help="fold an adapter into its base checkpoint")
    m.add_argument("adapter")
    m.add_argument("out")
    m.add_argument("--base", help="override the base recorded in the adapter")
    m.set_defaults(fn=_merge_cli)
    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    import sys
    sys.exit(main())
