"""
Book example, not repository code.

Chapter 10: the embedding lookup as a one-hot matrix multiply, the two token
tables of AnuLM, what their initialisation does to the first loss, and the
merge-tree reparameterisation, all checked against model.py unchanged.

    python book/code-examples/ch10_embeddings.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm import model as M  # noqa: E402  the repository's model.py, unchanged

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore", message="Sparse invariant checks")


def tiny(**kw) -> M.AnuLMConfig:
    """test_model.py's tiny(), copied so this script needs no test harness."""
    base = dict(vocab_size=64, block_size=32, n_layer=3, hidden_size=32, n_head=4, head_dim=8, n_kv_head=2,
                num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16, intermediate_size=64,
                n_group=1, topk_group=1)
    base.update(kw)
    return M.AnuLMConfig(**base)


def lookup_is_a_matmul() -> None:
    torch.manual_seed(0)
    m = M.AnuLM(tiny())
    W = m.embed_tokens.weight                                  # (V, D) = (64, 32)
    idx = torch.tensor([[5, 9, 5]])                            # (B, S) = (1, 3)
    x = F.embedding(idx, W)                                    # (1, 3, 32)
    one_hot = F.one_hot(idx, num_classes=W.shape[0]).float()   # (1, 3, 64)
    assert torch.equal(x, one_hot @ W)
    x.sum().backward()
    rows = W.grad.abs().sum(-1).nonzero().flatten().tolist()
    print(f"F.embedding == one_hot @ W: {tuple(idx.shape)} ids -> {tuple(x.shape)} vectors; "
          f"rows with a gradient: {rows}, row 5 grad = {W.grad[5, 0].item():.0f} (used twice)")


def two_tables() -> None:
    for tie in (False, True):
        m = M.AnuLM(tiny(tie_word_embeddings=tie))
        n = sum(p.numel() for p in m.parameters())
        print(f"tie_word_embeddings={tie!s:5}: same tensor? {m.lm_head.weight is m.embed_tokens.weight!s:5} "
              f"parameters {n:,}")
    V, D = 32768, 1024
    print(f"released model: each table {V} x {D} = {V * D:,} parameters, both {2 * V * D:,}, "
          f"{2 * V * D / 397_659_648:.1%} of its 397,659,648 (Chapter 20)")


def first_loss() -> None:
    torch.manual_seed(0)
    cfg = tiny()
    m = M.AnuLM(cfg).eval()
    print(f"embed std {m.embed_tokens.weight.std():.4f}, lm_head std {m.lm_head.weight.std():.4f}, "
          f"init range {cfg.initializer_range}")
    x = torch.randint(0, cfg.vocab_size, (4, 32))
    with torch.no_grad():
        logits, loss = m(x, x)
    print(f"tiny(): logit std {logits.std():.3f} (0.02 x sqrt(32) = {0.02 * math.sqrt(32):.3f}), "
          f"loss {loss:.3f}, ln V = {math.log(cfg.vocab_size):.3f}")
    # The released head on its own: a unit-RMS vector (what the final RMSNorm emits) times W.
    V, D = 32768, 1024
    g = torch.Generator().manual_seed(0)
    W = torch.randn(V, D, generator=g) * 0.02
    h = torch.randn(256, D, generator=g)
    h = h / h.pow(2).mean(-1, keepdim=True).sqrt()             # RMS exactly 1
    z = h @ W.T                                                # (256, V)
    y = torch.randint(0, V, (256,), generator=g)
    print(f"released size: logit std {z.std():.3f} (0.02 x sqrt(1024) = {0.02 * math.sqrt(D):.2f}), "
          f"loss {F.cross_entropy(z, y):.3f}, ln V = {math.log(V):.3f}, "
          f"ln V + std^2/2 = {math.log(V) + (0.02 * math.sqrt(D)) ** 2 / 2:.3f}")
    for name, s in (("tiny()", 0.02 * math.sqrt(32)), ("released", 0.02 * math.sqrt(D)),
                    ("released at initializer_range 0.05", 0.05 * math.sqrt(D))):
        print(f"predicted first-loss bump, {name}: sigma {s:.3f}, sigma^2/2 = {s * s / 2:.3f}")


def merge_tree() -> None:
    merges = [[97, 98], [256, 99], [257, 257], [100, 101]]      # ab, abc, abcabc, de
    P = M.merge_tree_matrix(merges, 261)
    for t in (97, 256, 257, 258, 259):
        row = P[t].coalesce()
        terms = " + ".join(f"{v:g}*W[{k}]" for k, v in zip(row.indices()[0].tolist(), row.values().tolist()))
        print(f"E[{t}] = {terms}")
    print(f"P: {tuple(P.shape)}, {P._nnz()} nonzeros")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tok.json")
        json.dump({"type": "anulm-bpe", "version": 1, "merges": merges}, open(path, "w"))
        torch.manual_seed(0)
        m = M.AnuLM(tiny(vocab_size=261, tokenizer_path=path, merge_tree="both"))
        x = torch.randint(0, 97, (2, 12))
        x[0, 3] = 258
        m.eval()
        with torch.no_grad():
            before, _ = m(x, x)
            m.bake_merge_tree()
            after, _ = m(x, x)
        print(f"bake_merge_tree: largest change in logits {(before - after).abs().max():.2e}, "
              f"merge_tree now {m.cfg.merge_tree!r}")
    W = torch.nn.Parameter(torch.zeros(3, 2))
    v = W._version
    W.data.add_(0.5)                                           # how the test fakes an optimizer step
    print(f"W.data.add_ changes the weights but W._version moves by {W._version - v}")


def main() -> None:
    lookup_is_a_matmul()
    two_tables()
    first_loss()
    merge_tree()
    print(f"exercise 1: one-hot (8, 1024, 32768) = {8 * 1024 * 32768:,} entries, "
          f"{8 * 1024 * 32768 * 4 / 2**30:.0f} GiB in fp32")
    print(f"exercise 2: tied released model saves {32768 * 1024:,} parameters, "
          f"{32768 * 1024 * 4 / 2**20:.0f} MiB in fp32")


if __name__ == "__main__":
    main()
