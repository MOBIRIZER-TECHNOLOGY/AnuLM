"""
Book example, not repository code.

Chapter 23: precision and memory, on the CPU.

1. The three floating-point formats, from torch.finfo, and what rounding to
   each does to a few numbers.
2. Which dtype each part of model.py produces under bf16 autocast, read with
   forward hooks on a small model.
3. The memory one training step needs, by arithmetic on the released
   configuration.
4. What gradient checkpointing and the chunked loss save, counted with
   torch.autograd.graph.saved_tensors_hooks on a small model: every tensor
   autograd keeps for the backward pass is added up. Loss and gradients are
   checked equal to the plain path.

    python book/code-examples/ch23_precision.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm import model as M  # noqa: E402  the repository's model.py, unchanged

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def formats():
    print(f"{'format':9s} {'bits':>4s} {'exponent':>8s} {'mantissa':>8s} {'eps':>10s} {'max':>10s} {'min normal':>11s}")
    for dt, e, m in ((torch.float32, 8, 23), (torch.bfloat16, 8, 7), (torch.float16, 5, 10)):
        f = torch.finfo(dt)
        assert f.bits == 1 + e + m and f.eps == 2.0 ** -m
        print(f"{str(dt)[6:]:9s} {f.bits:4d} {e:8d} {m:8d} {f.eps:10.3g} {f.max:10.3g} {f.tiny:11.3g}")
    for v in (1 + 2 ** -8, 257.0, 3.3523, 1e-8, 70000.0):
        t = torch.tensor(v)
        print(f"  {v:<12g} -> bf16 {t.bfloat16().item():<12g} fp16 {t.half().item():g}")


def autocast_dtypes():
    torch.manual_seed(0)
    cfg = M.AnuLMConfig(vocab_size=64, block_size=32, n_layer=3, hidden_size=32, n_head=4, head_dim=8, n_kv_head=2,
                        num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16, intermediate_size=64,
                        n_group=1, topk_group=1)
    m = M.AnuLM(cfg).train()
    seen = {}
    names = {"layers.0": "embedding (layer 0 input)", "layers.1.input_layernorm": "RMSNorm",
             "layers.1.attn.query_key_value": "attention projection",
             "layers.1.mlp.experts.0": "one expert", "lm_head": "lm_head (logits)"}
    for n, mod in m.named_modules():
        if n == "layers.0":
            mod.register_forward_pre_hook(lambda _m, i: seen.update({"layers.0": i[0].dtype}))
        elif n in names:
            mod.register_forward_hook(lambda _m, _i, o, n=n: seen.update({n: (o[1] if isinstance(o, tuple) else o).dtype}))
    gate = spy_on_gate(m.layers[1].mlp.router)
    x = torch.randint(0, 64, (2, 16))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        _, loss = m(x, x)
        norm_bf16 = m.layers[1].input_layernorm(torch.randn(2, 3, 32).bfloat16()).dtype
        m.layers[1].mlp.router._gate(torch.randn(5, 32), None)          # the gate called under autocast
    for n, label in names.items():
        print(f"  {label:22s} {str(seen[n])[6:]}")
    print(f"  {'RMSNorm, bf16 input':22s} {str(norm_bf16)[6:]}")
    print(f"  {'router logits':22s} {str(gate[0])[6:]} (Router.forward: autocast off)")
    print(f"  {'  the gate unguarded':22s} {str(gate[1])[6:]} (the same _gate under autocast, as before the fix)")
    print(f"  {'loss':22s} {str(loss.dtype)[6:]}")
    print(f"  {'weights':22s} {str(m.lm_head.weight.dtype)[6:]} (autocast casts copies, never the parameters)")


def spy_on_gate(router):
    """Record the dtype of the F.linear output inside router._gate (the logits), each call."""
    import torch.nn.functional as Fn
    seen, gate, linear = [], router._gate, Fn.linear

    def spy(*a, **k):
        def lin(*la, **lk):
            out = linear(*la, **lk)
            seen.append(out.dtype)
            return out
        Fn.linear = lin                                          # model.py calls F.linear: the same module
        try:
            return gate(*a, **k)
        finally:
            Fn.linear = linear
    router._gate = spy
    return seen


def step_memory():
    cfg = M.AnuLMConfig.nano_350m(vocab_size=32768, block_size=1024, sliding_window=256, max_window_layers=10,
                                  num_experts=24, num_experts_per_tok=4, num_shared_experts=0,
                                  moe_intermediate_size=192)
    with torch.device("meta"):                                    # shapes only: no memory is allocated
        n = sum(p.numel() for p in M.AnuLM(cfg).parameters())
    gib = 2 ** 30
    print(f"  parameters {n:,}: weights fp32 {4 * n / gib:.2f} GiB, gradients {4 * n / gib:.2f}, "
          f"AdamW moments {8 * n / gib:.2f}: {16 * n / gib:.2f} GiB in all")
    print(f"  always resident between steps (weights + moments; zero_grad(set_to_none=True) frees the "
          f"gradients): {12 * n / gib:.2f} GiB")
    logits = 4 * 1024 * cfg.vocab_size
    print(f"  logits at 4 x 1,024 x {cfg.vocab_size:,}: {logits:,} numbers = {2 * logits / gib:.2f} GiB in bf16 "
          f"+ {4 * logits / gib:.2f} GiB as the fp32 copy the loss takes = {6 * logits / gib:.2f} GiB")
    print(f"  exercise 1: 1e9 parameters x 16 bytes = {16e9 / gib:.1f} GiB")


def saved_bytes(m, x, y):
    """Bytes autograd keeps for backward during one forward (each storage counted once)."""
    seen, params = {}, {p.untyped_storage().data_ptr() for p in m.parameters()}

    def pack(t):
        ptr = t.untyped_storage().data_ptr()
        if ptr not in params:                                     # weights exist anyway: not extra memory
            seen[ptr] = t.untyped_storage().nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        _, loss = m(x, y)
    return sum(seen.values()), loss


def checkpoint_savings():
    torch.manual_seed(0)
    cfg = M.AnuLMConfig(vocab_size=8192, block_size=256, n_layer=4, hidden_size=128, n_head=4, head_dim=32,
                        n_kv_head=2, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=64,
                        intermediate_size=256, n_group=1, topk_group=1)
    base = M.AnuLM(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (4, 256))
    y = torch.randint(0, cfg.vocab_size, (4, 256))
    ref_grads, ref_loss = None, None
    for name, ckpt, chunk in (("plain", False, 0), ("gradient checkpointing", True, 0),
                              ("chunked loss (256 rows)", False, 256), ("both", True, 256)):
        m = M.AnuLM(cfg).train()
        m.load_state_dict(base.state_dict())
        m.enable_gradient_checkpointing(ckpt)
        m.enable_chunked_loss(chunk)
        nbytes, loss = saved_bytes(m, x, y)
        loss.backward()
        grads = [p.grad for p in m.parameters()]
        if ref_grads is None:
            ref_grads, ref_loss = grads, loss.item()
        err = max((a - b).abs().max().item() for a, b in zip(grads, ref_grads))
        assert abs(loss.item() - ref_loss) < 1e-5 and err < 1e-5
        print(f"  {name:26s} saved for backward {nbytes / 2**20:7.1f} MiB | loss {loss.item():.6f} | "
              f"max gradient difference {err:.1e}")


def main():
    print("1. formats (torch.finfo)")
    formats()
    print("\n2. dtypes under torch.autocast(device_type='cpu', dtype=torch.bfloat16)")
    autocast_dtypes()
    print("\n3. one training step of the released model, by arithmetic")
    step_memory()
    print("\n4. what the backward pass keeps, B = 4, S = 256, V = 8,192, 4 layers")
    checkpoint_savings()


if __name__ == "__main__":
    main()
