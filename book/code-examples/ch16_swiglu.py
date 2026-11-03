"""Book example, not repository code.

Chapter 16: the SwiGLU feed-forward network, by hand and in `model.py`.

    python book/code-examples/ch16_swiglu.py

1. One token through a SwiGLU two numbers wide with three inner channels,
   step by step (illustrative numbers), then the same weights loaded into
   `model.SwiGLU` to check it gives the same answer.
2. SiLU's shape: where it is negative and its smallest value.
3. A SwiGLU works on each token alone: shuffling the tokens shuffles the output.
4. The arithmetic for the released model: parameters and per-token work of
   the dense layer 0 against one MoE layer, from `model.py`'s own modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from anulm import model as M  # noqa: E402  the repository's model.py, unchanged
from trace_forward import released  # noqa: E402  the released configuration

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
torch.set_printoptions(precision=3, sci_mode=False)


def by_hand():
    x = torch.tensor([1., 1.])                       # bit's token vector, Chapter 11's toy
    W_gate = torch.tensor([[1., -1., 0.],            # (D, I) = (2, 3)
                           [0., 1., -2.]])
    W_up = torch.tensor([[1., 0., 1.],
                         [1., 2., 0.]])
    W_down = torch.tensor([[1., 0.],                 # (I, D) = (3, 2)
                           [0., 1.],
                           [1., 1.]])
    g = x @ W_gate                                   # gate inputs, one per channel
    a = g * torch.sigmoid(g)                         # SiLU(g) = g * sigmoid(g)
    up = x @ W_up                                    # what each channel would write
    h = a * up                                       # the gate scales it
    out = h @ W_down                                 # back to width D
    for name, t in (("g", g), ("SiLU(g)", a), ("up", up), ("h", h), ("out", out)):
        print(f"{name:8s} {t}")

    ffn = M.SwiGLU(2, 3)                             # the repository's module
    with torch.no_grad():                            # nn.Linear stores W transposed: (out, in)
        ffn.gate_proj.weight.copy_(W_gate.T)
        ffn.up_proj.weight.copy_(W_up.T)
        ffn.down_proj.weight.copy_(W_down.T)
        got = ffn(x)
    assert torch.allclose(got, out, atol=1e-6), (got, out)
    print("model.SwiGLU gives", got, "(matches)")
    assert torch.allclose(F.silu(g), a), "F.silu is g * sigmoid(g)"


def silu_shape():
    x = torch.linspace(-6, 0, 600_001, dtype=torch.float64)
    y = F.silu(x)
    i = y.argmin()
    print(f"SiLU is negative for every x < 0; smallest value {y[i]:.4f} at x = {x[i]:.4f}; "
          f"SiLU(-4) = {F.silu(torch.tensor(-4.)):.4f}, SiLU(4) = {F.silu(torch.tensor(4.)):.4f}")


def per_token():
    torch.manual_seed(0)
    ffn = M.SwiGLU(32, 64).eval()
    x = torch.randn(2, 5, 32)                        # (B, S, D)
    perm = torch.randperm(5)
    with torch.no_grad():
        assert torch.allclose(ffn(x)[:, perm], ffn(x[:, perm]), atol=1e-6)
    print("shuffling the 5 positions shuffles the output the same way: each token is processed alone")


def arithmetic():
    cfg = released()
    D, I, Ie, E, A = (cfg.hidden_size, cfg.intermediate_size, cfg.moe_intermediate_size,
                      cfg.num_experts, cfg.num_experts_per_tok)
    m = M.AnuLM(cfg)
    kinds = ["MoE" if b.is_moe else "dense SwiGLU" for b in m.layers]
    print(f"layer 0: {kinds[0]}; layers 1-{len(kinds) - 1}: {set(kinds[1:])}")
    dense = sum(p.numel() for p in m.layers[0].mlp.parameters())
    expert = sum(p.numel() for p in m.layers[1].mlp.experts[0].parameters())
    router = m.layers[1].mlp.router.weight.numel()
    assert dense == 3 * D * I and expert == 3 * D * Ie
    swiglus = sum(isinstance(mod, M.SwiGLU) for mod in m.modules())
    print(f"SwiGLU modules in the released model: {swiglus} (1 dense + {len(kinds) - 1} x {E} experts)")
    print(f"dense layer 0: 3*D*I = 3*{D}*{I} = {dense:,} parameters, all used by every token")
    print(f"one expert: 3*D*I_e = 3*{D}*{Ie} = {expert:,}; one MoE layer holds {E * expert + router:,}")
    active = A * expert + router
    print(f"per token, multiply-adds: layer 0 {dense:,}; an MoE layer {A}*{expert:,} + router {router:,} "
          f"= {active:,}; ratio {dense / active:.2f}")
    print(f"a two-matrix FFN at 4D: 2*D*4D = {2 * D * 4 * D:,}; SwiGLU width with the same count, "
          f"(2/3)*4D = {2 / 3 * 4 * D:.2f}; the configured I is {I} = {I / (2 / 3 * 4 * D):.5f} times that")
    print(f"(2/3)*4D rounded up to a multiple of 256: {-(-int(2 / 3 * 4 * D) // 256) * 256}")
    print(f"down_proj init std: 0.02/sqrt(2*{cfg.n_layer}) = {0.02 / (2 * cfg.n_layer) ** 0.5:.5f}; "
          f"measured {m.layers[0].mlp.down_proj.weight.std():.5f}; gate_proj {m.layers[0].mlp.gate_proj.weight.std():.5f}")


def mistakes():
    torch.manual_seed(0)
    a = M.SwiGLU(8, 8).eval()                        # D = I, the case where mistakes load silently
    b = M.SwiGLU(8, 8).eval()
    x = torch.randn(4, 8)
    with torch.no_grad():
        b.load_state_dict(a.state_dict())
        b.gate_proj.weight.copy_(a.up_proj.weight)   # gate and up swapped
        b.up_proj.weight.copy_(a.gate_proj.weight)
        swapped = (a(x) - b(x)).abs().max().item()
        b.load_state_dict(a.state_dict())
        b.gate_proj.weight.copy_(a.gate_proj.weight.T)   # a forgotten transpose: no error when D = I
        transposed = (a(x) - b(x)).abs().max().item()
    scale = a(x).abs().max().item()
    print(f"output scale {scale:.4f}; gate/up swapped: max difference {swapped:.4f}; "
          f"gate transposed: {transposed:.4f}; neither raised an error")


def exercises():
    x = torch.tensor([1., 1.])
    W_gate = torch.tensor([[1., -1., 0.], [0., 1., 0.]])          # exercise 1: second row changed
    W_up = torch.tensor([[1., 0., 1.], [1., 2., 0.]])
    W_down = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    g = x @ W_gate
    h = F.silu(g) * (x @ W_up)
    print(f"exercise 1: g {g}, h {h}, out {h @ W_down}")
    print(f"exercise 2: 3*1024*4096 = {3 * 1024 * 4096:,}")
    D, I, Ie, A, E = 384, 768, 96, 2, 16
    print(f"exercise 3: layer 0 {3 * D * I:,}; MoE {A}*{3 * D * Ie:,} + shared {3 * D * Ie:,} = "
          f"{A * 3 * D * Ie + 3 * D * Ie:,}, router {E * D:,}")
    cfg = M.AnuLMConfig.nano_30b()
    assert sum(p.numel() for p in M.SwiGLU(cfg.hidden_size, cfg.intermediate_size).parameters()) == 3 * D * I


if __name__ == "__main__":
    by_hand()
    silu_shape()
    per_token()
    arithmetic()
    mistakes()
    exercises()
