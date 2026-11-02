"""
Book example, not repository code.

Chapter 6: gradients and backpropagation, checked against model.py.

- a finite difference against autograd, on one weight of a dense tiny() model;
- the chain rule written out by hand through model.py's SwiGLU, against autograd;
- one plain gradient step by hand, and the loss it lowers;
- causality in the gradients: how much each input position moves one output.

    python book/code-examples/ch06_backprop.py [--save]    # a few seconds on a CPU; --save writes book/measurements/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm.model import AnuLM, SwiGLU          # noqa: E402  unchanged
from tests.test_model import tiny              # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def finite_difference(seed: int = 0) -> None:
    torch.manual_seed(seed)
    m = AnuLM(tiny(first_k_dense_replace=3)).double().eval()   # dense layers only: no discrete routing
    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))
    loss_of = lambda: F.cross_entropy(m(x, y)[0].double().flatten(0, 1), y.flatten())   # the loss, in fp64
    w = m.layers[1].mlp.down_proj.weight
    loss_of().backward()
    i, j = divmod(w.grad.abs().argmax().item(), w.shape[1])   # the entry with the largest gradient
    auto = w.grad[i, j].item()
    eps = 1e-3
    with torch.no_grad():
        w[i, j] += eps
        up = loss_of().item()
        w[i, j] -= 2 * eps
        down = loss_of().item()
        w[i, j] += eps
    numeric = (up - down) / (2 * eps)                          # central difference
    print(f"d loss / d w[{i}, {j}]: autograd {auto:.6f}   (L(w+e) - L(w-e)) / 2e {numeric:.6f}")
    assert abs(auto - numeric) < 1e-3 * abs(auto)


def swiglu_by_hand(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    f = SwiGLU(8, 16)                                    # D = 8, I = 16
    x = torch.randn(5, 8, requires_grad=True)            # 5 tokens
    G = torch.randn(5, 8)                                # dL/dy, arriving from above
    y = f(x)
    (y * G).sum().backward()                             # a loss whose gradient at y is G
    Wg, Wu, Wd = f.gate_proj.weight, f.up_proj.weight, f.down_proj.weight
    with torch.no_grad():
        a, b = x @ Wg.T, x @ Wu.T                        # (5, 16) each
        sig = torch.sigmoid(a)
        s = a * sig                                      # silu(a)
        h = s * b                                        # (5, 16)
        dh = G @ Wd                                      # back through down_proj: (5, 8) @ (8, 16)
        dWd = G.T @ h                                    # (8, 5) @ (5, 16) = (8, 16), down_proj's shape
        ds, db = dh * b, dh * s                          # product rule
        da = ds * sig * (1 + a * (1 - sig))              # silu'(a) = sig + a * sig * (1 - sig)
        dWg, dWu = da.T @ x, db.T @ x                    # (16, 8) each
        dx = da @ Wg + db @ Wu                           # x feeds both branches: the gradients add
    err = {"down_proj": (dWd - Wd.grad).abs().max().item(), "gate_proj": (dWg - Wg.grad).abs().max().item(),
           "up_proj": (dWu - Wu.grad).abs().max().item(), "x": (dx - x.grad).abs().max().item()}
    print("SwiGLU by hand vs autograd, largest difference: " + ", ".join(f"{k} {v:.1e}" for k, v in err.items()))
    assert max(err.values()) < 1e-5
    return err


def one_step(lr: float = 0.5, seed: int = 0) -> None:
    torch.manual_seed(seed)
    m = AnuLM(tiny()).train()
    x = torch.randint(0, 64, (4, 32))
    y = torch.roll(x, -1, dims=1)                         # any fixed targets will do
    _, before = m(x, y)
    before.backward()
    with torch.no_grad():
        for p in m.parameters():
            p -= lr * p.grad                             # w <- w - lr * dL/dw
    _, after = m(x, y)
    print(f"one step of plain gradient descent, lr {lr}: loss {before.item():.4f} -> {after.item():.4f}")
    assert after.item() < before.item()


def causal_gradients(seed: int = 0) -> list[float]:
    """|d logits[position 5] / d embedding[position j]| for each j, as the causality test measures it."""
    torch.manual_seed(seed)
    m = AnuLM(tiny()).eval()
    x = torch.randint(0, 64, (1, 12))
    emb = m.embed_tokens(x).detach().requires_grad_(True)
    h = emb
    cos, sin = m.rotary(12, emb.device, emb.dtype)
    for layer in m.layers:
        h, _ = layer(h, cos, sin)
    logits = m.lm_head(m.norm(h))
    logits[0, 5].sum().backward()
    g = emb.grad[0].abs().amax(-1).tolist()             # one number per input position
    print("largest |gradient| at each input position, output position 5:")
    print("  " + "  ".join(f"{j}:{v:.1e}" for j, v in enumerate(g)))
    assert all(v == 0.0 for v in g[6:]) and all(v > 0 for v in g[:6])
    return g


if __name__ == "__main__":
    finite_difference()
    swiglu_by_hand()
    one_step()
    g = causal_gradients()
    out = ROOT / "book" / "measurements" / "part_i_ii_causal_grad.json" if "--save" in sys.argv else Path.cwd() / "part_i_ii_causal_grad.json"
    out.write_text(json.dumps({"model": "tiny(), seed 0, eval", "output_position": 5, "max_abs_grad": g}, indent=1))
    print(f"wrote {out}  (book/measurements/ only with --save)")
