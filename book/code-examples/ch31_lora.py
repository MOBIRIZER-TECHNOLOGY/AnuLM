"""
Book example, not repository code.

Chapter 31: LoRA written out in a few lines, checked against lora.py, then the
mixture-of-experts trap that lora.py's router_state_dict exists for.

    python book/code-examples/ch31_lora.py

1. `lora_forward` and `lora_merge` by hand equal LoRALinear's forward and
   merged_weight on the same factors.
2. The adapter's size on the released configuration, by formula and by
   apply_lora on a real AnuLM.
3. On the tiny test model: step 0 equals the base; ten fine-tuning steps the
   way finetune.py takes them (optimizer step, then update_expert_biases);
   merging with the saved router biases reproduces the trained model, and
   merging without them changes which experts tokens reach.

The tiny model and its random data are made up for teaching.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from anulm import model as M  # noqa: E402
from anulm.training.lora import (LoRALinear, apply_lora, load_lora, lora_parameters, lora_state_dict,  # noqa: E402
                  merge_lora, router_state_dict)
from trace_forward import released  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def lora_forward(x, W, A, B, alpha, r):
    """y = x W^T + (alpha / r) * x A^T B^T: the frozen path plus a rank-r correction."""
    return x @ W.T + (x @ A.T @ B.T) * (alpha / r)


def lora_merge(W, A, B, alpha, r):
    """Fold the correction into the weight: W + (alpha / r) * B A, same shape as W."""
    return W + (alpha / r) * (B @ A)


def expert_sets(model, x):
    """The top-k expert ids each token reaches in every MoE layer, as one tensor."""
    out = []
    hooks = [m.register_forward_hook(lambda mod, i, o: out.append(o[0]))
             for m in model.modules() if isinstance(m, M.Router)]
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return torch.stack([o.sort(-1).values for o in out])          # (layers, N, k)


def main() -> None:
    # 1. by hand against LoRALinear
    torch.manual_seed(0)
    base = nn.Linear(32, 48, bias=False)
    lin = LoRALinear(base, r=4, alpha=8.0)
    nn.init.normal_(lin.lora_b, std=0.1)                       # pretend training moved B
    x = torch.randn(5, 32)
    mine = lora_forward(x, base.weight, lin.lora_a, lin.lora_b, 8.0, 4)
    print(f"by hand vs LoRALinear.forward:      largest difference {(mine - lin(x)).abs().max():.1e}")
    Wm = lora_merge(base.weight, lin.lora_a, lin.lora_b, 8.0, 4)
    print(f"by hand vs LoRALinear.merged_weight: largest difference {(Wm - lin.merged_weight()).abs().max():.1e}")
    print(f"merged weight used alone vs adapted: largest difference {(x @ Wm.T - lin(x)).abs().max():.1e}")

    # 2. the adapter on the released configuration
    cfg = released()
    D, H, Hkv, Dh, n_layer = cfg.hidden_size, cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.n_layer
    r = 16
    qkv = r * (D + (H + 2 * Hkv) * Dh)                         # A: r x D, B: (H+2Hkv)Dh x r
    dense = r * (H * Dh + D)
    formula = n_layer * (qkv + dense)
    big = M.AnuLM(cfg)
    total_before = sum(p.numel() for p in big.parameters())
    n = apply_lora(big, r=r, alpha=32.0)
    train = sum(p.numel() for p in lora_parameters(big))
    biases = sum(b.numel() for b in router_state_dict(big).values())
    assert train == formula
    print(f"released config, r = {r}: {n} adapters, {train:,} trainable "
          f"({100 * train / total_before:.2f}% of {total_before:,}); formula {formula:,}")
    print(f"adapter file: {train * 4 / 1e6:.2f} MB of fp32 factors + {biases} router biases")
    del big

    # 3. the MoE trap, on tiny()
    from tests.test_model import tiny
    cfg = tiny()
    torch.manual_seed(0)
    base_model = M.AnuLM(cfg).eval()
    base_sd = copy.deepcopy(base_model.state_dict())
    xb = torch.randint(0, cfg.vocab_size, (4, 16))
    with torch.no_grad():
        want, _ = base_model(xb)
    model = M.AnuLM(cfg)
    model.load_state_dict(base_sd)
    apply_lora(model, r=4, alpha=8.0)
    model.eval()
    with torch.no_grad():
        got, _ = model(xb)
    print(f"step 0, adapted vs base logits: largest difference {(got - want).abs().max():.1e}")
    opt = torch.optim.AdamW(lora_parameters(model), lr=1e-2)
    model.train()
    for _ in range(10):                                         # as finetune.py: step, then the balancer
        _, loss = model(xb, xb.roll(-1, 1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.update_expert_biases()
    model.eval()
    with torch.no_grad():
        trained, _ = model(xb)
    ad, router = lora_state_dict(model), router_state_dict(model)
    drift = max((router[k] - base_sd[k]).abs().max().item() for k in router)
    print(f"after 10 steps: router biases moved by up to {drift:.3f} (frozen weights, but a buffer)")

    def rebuild(with_router: bool):
        m = M.AnuLM(cfg)
        m.load_state_dict(base_sd)
        apply_lora(m, r=4, alpha=8.0)
        load_lora(m, ad)
        if with_router:
            m.load_state_dict(router, strict=False)
        merge_lora(m)
        return m.eval()

    for flag in (True, False):
        m = rebuild(flag)
        with torch.no_grad():
            out, _ = m(xb)
        moved = (expert_sets(m, xb) != expert_sets(model, xb)).any(-1).float().mean().item()
        print(f"merged {'with' if flag else 'without'} router biases: logits differ by up to "
              f"{(out - trained).abs().max():.1e}; tokens whose expert set changed: {100 * moved:.0f}%")


if __name__ == "__main__":
    main()
