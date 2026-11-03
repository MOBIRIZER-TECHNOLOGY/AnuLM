"""
Book example, not repository code.

Chapter 3: the loss AnuLM.forward returns, recomputed by hand as the average
surprise -log p(next token); why an untrained model starts at ln V (and a
little above it, by D * 0.02**2 / 2); and the bits-per-byte conversion train.py
prints.

    python book/code-examples/ch03_one_job.py     # a few seconds on a CPU

It imports model.py, train.py and test_model.py unchanged.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "book" / "code-examples"))
from anulm.model import AnuLM, AnuLMConfig     # noqa: E402  unchanged
from tests.test_model import tiny              # noqa: E402
from trace_forward import released       # noqa: E402  Chapter 20's released configuration

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def surprise_by_hand(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean of -log p(target) over the positions whose target is not -1."""
    logp = F.log_softmax(logits.float(), dim=-1)                  # (B, S, V): log-probabilities
    keep = targets != -1                                          # (B, S): graded positions
    s = -logp[keep].gather(1, targets[keep][:, None])[:, 0]       # (n,): one surprise per graded token
    return s.mean()


def check_loss() -> None:
    torch.manual_seed(0)
    cfg = tiny()
    model = AnuLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    y[:, :3] = -1                                                 # a prompt: not graded
    with torch.no_grad():
        logits, loss = model(x, y)
    mine = surprise_by_hand(logits, y)
    print(f"model loss {loss.item():.6f}   by hand {mine.item():.6f}   graded {(y != -1).sum().item()} of {y.numel()}")
    assert abs(loss.item() - mine.item()) < 1e-5


def initial_loss(cfg: AnuLMConfig, B: int = 16, S: int = 128, seed: int = 0) -> tuple[float, float]:
    """Measured loss of a fresh model on random targets, and the prediction ln V + D * std**2 / 2."""
    torch.manual_seed(seed)
    model = AnuLM(cfg).eval()
    S = min(S, cfg.block_size)
    x = torch.randint(0, cfg.vocab_size, (B, S))
    y = torch.randint(0, cfg.vocab_size, (B, S))
    with torch.no_grad():
        _, loss = model(x, y)
    var = cfg.hidden_size * cfg.initializer_range ** 2           # each logit's variance at init
    return loss.item(), math.log(cfg.vocab_size) + var / 2


def bits_per_byte(nats_per_token: float, bytes_per_token: float) -> float:
    return nats_per_token / math.log(2) / bytes_per_token         # train.py line 740


if __name__ == "__main__":
    check_loss()
    print("\nfresh model: loss on random targets against ln V and ln V + D*0.02^2/2")
    for name, cfg in [("tiny()", tiny()), ("nano_30b", AnuLMConfig.nano_30b()), ("released 400M", released()),
                      ("tiny(), init std 1.0", tiny(initializer_range=1.0))]:
        got, pred = initial_loss(cfg)                       # 16 sequences of up to 128 tokens: a small sample
        print(f"  {name:22s} V={cfg.vocab_size:6d} D={cfg.hidden_size:5d}  loss {got:7.4f}  ln V {math.log(cfg.vocab_size):7.4f}"
              f"  predicted {pred:7.4f}")
        if "1.0" not in name:
            assert abs(got - pred) < 0.05, "the small-variance prediction should hold at std 0.02"
    meta = json.loads((ROOT / "data" / "v2" / "meta.json").read_text())
    print(f"\nbytes per token in data/v2 (meta.json): {meta['bytes_per_token']}")
    for label, nats in [("RESULTS.md §39 final val", 3.3523), ("Chapter 20 sample", 3.3790)]:
        print(f"  {label:26s} {nats:.4f} nats/token = {nats / math.log(2):.3f} bits/token = "
              f"{bits_per_byte(nats, meta['bytes_per_token']):.4f} bits/byte")
