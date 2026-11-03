"""
Book example, not repository code.

Chapter 7: AdamW, schedules and clipping, checked against train.py and muon.py.

- AdamW written out by hand, against the optimizer train.py's make_optimizer builds;
- bias correction with a constant gradient;
- which tiny() parameters get weight decay;
- train.py's lr_at at the released run's settings, cosine against WSD;
- gradient clipping by hand against torch.nn.utils.clip_grad_norm_;
- Muon's Newton-Schulz step: what it does to a matrix's singular values.

    python book/code-examples/ch07_optim.py     # a few seconds on a CPU
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm.model import AnuLM                                      # noqa: E402  unchanged
from anulm.training.train import lr_at, make_optimizer                      # noqa: E402  unchanged
from anulm.muon import zeropower_via_newtonschulz5                 # noqa: E402  unchanged
from tests.test_model import tiny                                  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# The released run's command line (_v2_run.cmd at the pin); weight decay and clipping are train.py's defaults.
RELEASED = SimpleNamespace(lr=6e-4, min_lr=6e-5, warmup=2000, steps=1_900_000, schedule="wsd",
                           decay_frac=0.1631579, rewarmup=0, weight_decay=0.1, grad_clip=1.0)


def adamw_by_hand(p, g, state, lr, wd, b1=0.9, b2=0.95, eps=1e-8):
    """One AdamW step on tensor p with gradient g. state holds m, v and the step count t."""
    state["t"] += 1
    t = state["t"]
    state["m"].mul_(b1).add_(g, alpha=1 - b1)                  # first moment: a running mean of g
    state["v"].mul_(b2).addcmul_(g, g, value=1 - b2)           # second moment: a running mean of g^2
    m_hat = state["m"] / (1 - b1 ** t)                         # bias correction: both start at 0
    v_hat = state["v"] / (1 - b2 ** t)
    p.mul_(1 - lr * wd)                                        # decoupled weight decay
    p.add_(-lr * m_hat / (v_hat.sqrt() + eps))


def check_adamw(steps: int = 5) -> None:
    torch.manual_seed(0)
    m = AnuLM(tiny())
    mine = {n: p.detach().clone() for n, p in m.named_parameters()}
    args = SimpleNamespace(weight_decay=0.1, device="cpu", optimizer="adamw", lr=3e-4)
    with contextlib.redirect_stdout(io.StringIO()):
        opt = make_optimizer(m, args)                          # train.py's two groups and betas
    state = {n: {"t": 0, "m": torch.zeros_like(p), "v": torch.zeros_like(p)} for n, p in mine.items()}
    for _ in range(steps):
        for n, p in m.named_parameters():
            p.grad = torch.randn_like(p)
        opt.step()
        for n, p in m.named_parameters():
            wd = args.weight_decay if p.dim() >= 2 else 0.0    # make_optimizer's rule
            adamw_by_hand(mine[n], p.grad, state[n], args.lr, wd)
    err = max((mine[n] - p.detach()).abs().max().item() for n, p in m.named_parameters())
    print(f"AdamW by hand vs make_optimizer's AdamW after {steps} steps: largest difference {err:.1e}")
    assert err < 1e-6


def bias_correction() -> None:
    m, out = 0.0, []
    for t in range(1, 6):
        m = 0.9 * m + 0.1 * 1.0                                # a constant gradient of 1
        out.append((t, m, m / (1 - 0.9 ** t)))
    print("constant gradient 1:  " + "  ".join(f"t={t}: m {a:.3f}, m_hat {b:.3f}" for t, a, b in out))


def decay_groups() -> None:
    m = AnuLM(tiny())
    decay = [n for n, p in m.named_parameters() if p.dim() >= 2]
    no_decay = [n for n, p in m.named_parameters() if p.dim() < 2]
    buffers = [n for n, _ in m.named_buffers() if "expert_bias" in n]
    print(f"tiny(): {len(decay)} tensors decayed ({sum(p.numel() for n, p in m.named_parameters() if p.dim() >= 2):,} "
          f"numbers), {len(no_decay)} not ({sum(p.numel() for n, p in m.named_parameters() if p.dim() < 2):,}); "
          f"e.g. {no_decay[0]}, {no_decay[-1]}")
    print(f"  expert_bias is a buffer, in no group: {buffers}")


def schedules() -> dict:
    cos = SimpleNamespace(**{**vars(RELEASED), "schedule": "cosine"})
    pts = [0, 999, 1999, 2000, 500_000, 1_000_000, 1_589_999, 1_590_000, 1_700_000, 1_899_999]
    rows = {s: (lr_at(s, RELEASED), lr_at(s, cos)) for s in pts}
    print("step        WSD (released)   cosine (same lr, min_lr, warmup)")
    for s, (a, b) in rows.items():
        print(f"{s:>9,d}   {a:.3e}        {b:.3e}")
    decay_start = int(RELEASED.steps * (1 - RELEASED.decay_frac))
    print(f"WSD decay starts at step {decay_start:,}")
    return {"decay_start": decay_start}


def clip_by_hand(seed: int = 0) -> None:
    torch.manual_seed(seed)
    m = AnuLM(tiny())
    _, loss = m(*(2 * [torch.randint(0, 64, (4, 32))]))
    loss.backward()
    grads = [p.grad.clone() for p in m.parameters()]
    total = math.sqrt(sum(float(g.pow(2).sum()) for g in grads))     # one norm over every parameter
    max_norm = 0.05
    scale = min(1.0, max_norm / (total + 1e-6))
    reported = torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm)
    err = max((g * scale - p.grad).abs().max().item() for g, p in zip(grads, m.parameters()))
    print(f"total gradient norm {total:.4f} (clip_grad_norm_ returned {reported.item():.4f}); "
          f"clip at {max_norm}: scale {scale:.4f}; largest difference {err:.1e}")
    assert abs(total - reported.item()) < 1e-4 and err < 1e-7


def newton_schulz(seed: int = 0) -> None:
    torch.manual_seed(seed)
    G = torch.randn(64, 24) @ torch.diag(torch.logspace(-2, 0, 24))   # singular values over two decades
    X = zeropower_via_newtonschulz5(G).float()
    s_in, s_out = torch.linalg.svdvals(G), torch.linalg.svdvals(X)
    print(f"Newton-Schulz on a (64, 24) update: singular values {s_in.min():.3f}..{s_in.max():.3f} -> "
          f"{s_out.min():.3f}..{s_out.max():.3f}")


def memory() -> None:
    """Training-state bytes for the released model, from muon.py's own split of its parameters."""
    from anulm.muon import build_optimizers                          # unchanged
    sys.path.insert(0, str(ROOT / "book" / "code-examples"))
    from trace_forward import released                         # Chapter 20's released configuration
    with torch.device("meta"):                                  # shapes only: no memory, no init cost
        m = AnuLM(released())
    n = sum(p.numel() for p in m.parameters())
    opts, desc = build_optimizers(m)                            # Muon on hidden 2-D weights, AdamW on the rest
    on_muon = sum(p.numel() for g in opts[0].param_groups for p in g["params"])
    print(f"released model, {n:,} parameters; {desc}")
    print(f"AdamW on everything: 16 bytes/param -> {16 * n / 1e9:.2f} GB")
    print(f"--optimizer muon: 12 x {on_muon:,} + 16 x {n - on_muon:,} -> {(12 * on_muon + 16 * (n - on_muon)) / 1e9:.2f} GB"
          f"  (12 bytes on every parameter would be {12 * n / 1e9:.2f} GB)")


if __name__ == "__main__":
    check_adamw()
    bias_correction()
    decay_groups()
    schedules()
    clip_by_hand()
    newton_schulz()
    memory()
    r = json.loads((ROOT / "book" / "measurements" / "part_i_ii_released.json").read_text())
    norms = r["grad_norm_batches_4x1024_bf16"]
    print("released model, gradient norm per 4 x 1,024 batch: " + ", ".join(f"{v:.2f}" for v in norms)
          + f"; clip 1.0 scales by {1 / max(norms):.2f} to {1 / min(norms):.2f}")
