"""
Chapter 10 checks: LoRA and QLoRA on the model you built.

    python ch10.py                  # your course/build/post.py
    python ch10.py --impl solution  # the reference solution
    python ch10.py --self-test      # proves each check catches the bug it is for

Adapter bugs hide well: an adapter that doesn't start at zero still trains, a
merge that forgets the scale still loads, and adapters saved without the
router's state still give an almost-right model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import main, solution, with_replaced  # noqa: E402
import lora as ref_lora  # noqa: E402  the repository's lora.py

CHECKS = []
KW = dict(vocab_size=64, block_size=32, n_layer=3, hidden_size=32, n_head=4, head_dim=8, n_kv_head=2,
          num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16, intermediate_size=64,
          rope_theta=10000.0, bias_update_rate=0.05)
B = solution(8)                        # the model and train_step (Chapter 8's reference)


def check(fn):
    CHECKS.append(fn)
    return fn


def base_model():
    torch.manual_seed(0)
    return B.MoELM(**KW)


@check
def lora_linear_matches_lora_py(M):
    torch.manual_seed(0)
    lin = nn.Linear(16, 24, bias=False)
    theirs = ref_lora.LoRALinear(lin, 4, 8.0)
    with torch.no_grad():
        theirs.lora_b.normal_()                               # non-zero, so the adapter matters
    mine = M.LoRALinear(nn.Linear(16, 24, bias=False), 4, 8.0)
    mine.load_state_dict(theirs.state_dict())
    x = torch.randn(3, 16)
    torch.testing.assert_close(mine(x), theirs(x), msg="LoRALinear output differs from lora.py's")
    torch.testing.assert_close(mine.merged_weight(), theirs.merged_weight(), msg="merged_weight differs")


@check
def a_new_adapter_is_the_base_model(M):
    """B starts at zero, so wrapping changes nothing: fine-tuning starts from the base."""
    m = base_model().eval()
    x = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        before = m(x)[0]
        M.apply_lora(m, 4, 8.0)
        after = m(x)[0]
    torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-6, msg="the adapted model is not the base model at step 0")


@check
def only_the_adapters_train(M):
    m = base_model()
    hits = M.apply_lora(m, 4, 8.0)
    assert hits == 2 * KW["n_layer"], f"{hits} layers wrapped; the attention projections are {2 * KW['n_layer']}"
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable), f"non-adapter parameters train: {trainable[:3]}"


@check
def merging_is_exact(M):
    """After training, folding the adapters into the weights must give the same model, as plain Linears."""
    m = base_model().train()
    M.apply_lora(m, 4, 8.0)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    x = torch.randint(0, 64, (4, 16))
    for _ in range(3):
        B.train_step(m, opt, [(x, x)], 1e-2)
    m.eval()
    with torch.no_grad():
        before = m(x)[0]
        n = M.merge_lora(m)
        after = m(x)[0]
    assert n == 2 * KW["n_layer"], f"merged {n} adapters"
    assert not any(isinstance(mod, M.LoRALinear) for mod in m.modules()), "adapters left in place after merging"
    torch.testing.assert_close(after, before, rtol=1e-4, atol=1e-5, msg="the merged model differs from the adapted one")


@check
def the_adapter_carries_the_router(M):
    """In an MoE model the balancer moves the experts' biases during LoRA training. Saved without
    them, the adapters rebuild a slightly different model from a fresh base."""
    m = base_model().train()
    M.apply_lora(m, 4, 8.0)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    x = torch.randint(0, 64, (8, 16))
    for _ in range(5):
        B.train_step(m, opt, [(x, x)], 1e-2)
    state = M.adapter_state(m)
    fresh = base_model()
    M.apply_lora(fresh, 4, 8.0)
    fresh.load_state_dict(state, strict=False)
    m.eval(), fresh.eval()
    with torch.no_grad():
        torch.testing.assert_close(fresh(x)[0], m(x)[0], rtol=1e-5, atol=1e-6,
                                   msg="adapters applied to a fresh base do not rebuild the fine-tuned model")


@check
def nf4_round_trip_stays_within_half_a_level(M):
    """Every weight comes back within half the gap between its two nearest NF4 levels, times its
    block's scale; each block has its own scale, so small blocks keep their detail; and two 4-bit
    codes share each byte."""
    torch.manual_seed(0)
    w = torch.randn(48, 64) * torch.linspace(0.1, 3, 48)[:, None]       # blocks of very different sizes
    packed, absmax = M.nf4_quantize(w)
    assert packed.dtype == torch.uint8 and packed.numel() == w.numel() // 2, "not two codes per byte"
    torch.testing.assert_close(absmax.float(), w.reshape(-1, 64).abs().amax(1),
                               msg="each block of 64 needs its own scale: its largest absolute weight")
    back = M.nf4_dequantize(packed, absmax, w.shape)
    gap = (M.NF4[1:] - M.NF4[:-1]).max().item() / 2
    bound = (absmax[:, None] * gap).expand(-1, 64).reshape(w.shape)
    assert torch.all((back - w).abs() <= bound + 1e-6), "a weight came back further than half a level away"


@check
def qlora_starts_as_the_4bit_base(M):
    """With quantize=True and zero adapters, the model equals its weights rounded to NF4."""
    m = base_model().eval()
    x = torch.randint(0, 64, (2, 16))
    ref = base_model().eval()
    with torch.no_grad():
        for name, mod in ref.named_modules():
            if isinstance(mod, nn.Linear) and name.endswith(("attn.query_key_value", "attn.dense")):
                mod.weight.copy_(M.nf4_dequantize(*M.nf4_quantize(mod.weight), mod.weight.shape))
        M.apply_lora(m, 4, 8.0, quantize=True)
        torch.testing.assert_close(m(x)[0], ref(x)[0], rtol=1e-5, atol=1e-6, msg="QLoRA at step 0 is not the 4-bit base")
    frozen = [n for n, p in m.named_parameters() if "attn.query_key_value" in n and "lora_" not in n]
    assert not frozen, f"the quantized layers still hold full-precision parameters: {frozen[:2]}"


# --- broken versions for --self-test ----------------------------------------------------

NAMES = ("LoRALinear", "apply_lora", "merge_lora", "adapter_state", "nf4_quantize", "nf4_dequantize",
         "NF4Linear", "NF4", "BLOCK")


def MUTANTS(sol):
    class RandomB(sol.LoRALinear):
        def __init__(self, base, r=16, alpha=32.0):
            super().__init__(base, r, alpha)
            nn.init.normal_(self.lora_b, std=0.02)

    class ScaleInverted(sol.LoRALinear):
        def __init__(self, base, r=16, alpha=32.0):
            super().__init__(base, r, alpha)
            self.scale = r / alpha

    def apply_without_freezing(model, r=16, alpha=32.0, targets=sol.DEFAULT_TARGETS, quantize=False):
        params = list(model.parameters())
        n = sol.apply_lora(model, r, alpha, targets, quantize)
        for p in params:
            p.requires_grad_(True)
        return n

    def merge_without_scale(model):
        merged = 0
        for name, module in list(model.named_modules()):
            for cn, child in list(module.named_children()):
                if isinstance(child, sol.LoRALinear):
                    with torch.no_grad():
                        child.base.weight += child.lora_b @ child.lora_a
                    setattr(module, cn, child.base)
                    merged += 1
        return merged

    def adapters_only(model):
        return {n: t.detach().clone() for n, t in model.state_dict().items() if "lora_" in n}

    def per_tensor_scale(w):
        flat = w.detach().float().reshape(-1, sol.BLOCK)
        absmax = flat.abs().max().clamp_min(1e-12).expand(flat.shape[0]).clone()
        codes = torch.bucketize(flat / absmax[:, None], (sol.NF4[1:] + sol.NF4[:-1]) / 2).to(torch.uint8).reshape(-1)
        return (codes[0::2] << 4) | codes[1::2], absmax

    def floor_level(w):
        flat = w.detach().float().reshape(-1, sol.BLOCK)
        absmax = flat.abs().max(dim=1, keepdim=True).values.clamp_min(1e-12)
        codes = (torch.bucketize(flat / absmax, sol.NF4, right=True) - 1).clamp(0, 15).to(torch.uint8).reshape(-1)
        return (codes[0::2] << 4) | codes[1::2], absmax.squeeze(1)

    def swapped_unpack(packed, absmax, shape):
        codes = torch.stack([packed & 15, packed >> 4], dim=1).reshape(-1).long()
        return (sol.NF4[codes].reshape(-1, sol.BLOCK) * absmax[:, None]).reshape(shape)

    def quantize_ignored(model, r=16, alpha=32.0, targets=sol.DEFAULT_TARGETS, quantize=False):
        return sol.apply_lora(model, r, alpha, targets, False)

    bugs = [("lora_b initialised randomly", dict(LoRALinear=RandomB)),
            ("scale r / alpha instead of alpha / r", dict(LoRALinear=ScaleInverted)),
            ("base weights left trainable", dict(apply_lora=apply_without_freezing)),
            ("merge forgetting the scale", dict(merge_lora=merge_without_scale)),
            ("adapters saved without the router biases", dict(adapter_state=adapters_only)),
            ("one NF4 scale for the whole tensor", dict(nf4_quantize=per_tensor_scale)),
            ("rounding down to a level instead of nearest", dict(nf4_quantize=floor_level)),
            ("codes unpacked in the wrong order", dict(nf4_dequantize=swapped_unpack)),
            ("quantize=True ignored", dict(apply_lora=quantize_ignored))]
    out = []
    for d, r in bugs:
        ns = with_replaced(sol, NAMES, **r)
        if "LoRALinear" in r:                 # apply_lora must build the broken class
            cls = r["LoRALinear"]

            def apply_with(model, rr=16, alpha=32.0, targets=sol.DEFAULT_TARGETS, quantize=False, _c=cls):
                for p in model.parameters():
                    p.requires_grad_(False)
                n = 0
                for name, module in list(model.named_modules()):
                    for cn, child in list(module.named_children()):
                        full = f"{name}.{cn}" if name else cn
                        if isinstance(child, nn.Linear) and any(full.endswith(t) for t in targets):
                            setattr(module, cn, _c(sol.NF4Linear(child) if quantize else child, rr, alpha))
                            n += 1
                return n
            ns.apply_lora = apply_with
        out.append((d, ns))
    return out


if __name__ == "__main__":
    main(10, CHECKS, MUTANTS, __doc__, learner="post.py")
