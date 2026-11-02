"""
Book example, not repository code.

Chapter 2: the ladder from counts to context, on the lab book's Shakespeare
(data/input.txt, the first 90% of its bytes to train, the last 10% to score).

1. a unigram count model, which ignores order (the "bag of words" idea);
2. a bigram count model, the lab book's Chapter 1 table;
3. the repository's own bigram teacher (bigram_teacher.py), used as a model;
4. AnuLM with zero blocks (model.py unchanged): an embedding, a norm and the
   output head, trained with train.py's own optimizer and batches. It can only
   learn a bigram, and it lands on the count table's loss;
5. the check that explains why: with no blocks, the last logits ignore every
   earlier token; with attention blocks they do not.

    python book/code-examples/ch02_counts_to_attention.py [--save]   # seconds on a CPU; --save writes book/measurements/
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm.model import AnuLM, AnuLMConfig                     # noqa: E402  unchanged
from anulm.training.train import make_batch, make_optimizer             # noqa: E402  unchanged
from anulm.training.bigram_teacher import build_bigram_teacher          # noqa: E402  unchanged
from tests.test_model import tiny                              # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load() -> tuple[torch.Tensor, torch.Tensor]:
    """The lab book's split (course/ch01-smallest-model/lab.py, load_bytes)."""
    data = torch.tensor(list((ROOT / "data" / "input.txt").read_bytes()), dtype=torch.long)
    cut = int(0.9 * len(data))
    return data[:cut], data[cut:]


def unigram_loss(train, val, V=256):
    counts = torch.ones(V).index_add_(0, train, torch.ones(len(train)))   # add-one smoothing
    p = counts / counts.sum()                                               # order ignored
    return -p[val[1:]].log().mean().item()


def bigram_loss(train, val, V=256, alpha=1.0):
    counts = torch.full((V, V), alpha)                                      # every pair starts at alpha (1: add-one)
    counts.index_put_((train[:-1], train[1:]), torch.ones(len(train) - 1), accumulate=True)
    p = counts / counts.sum(1, keepdim=True)                                # row = previous byte
    return -p[val[:-1], val[1:]].log().mean().item()


def teacher_loss(train, val, V=256):
    log_uni, idx, res = build_bigram_teacher(train, V, k=V, device="cpu")
    z = log_uni.expand(V, -1).scatter_add(-1, idx, res)                     # one row per previous byte
    logp = F.log_softmax(z, -1)                                             # as teacher_kl normalises it
    return -logp[val[:-1], val[1:]].mean().item()


def zero_block_model(train, val, steps=1500, seed=0):
    torch.manual_seed(seed)
    cfg = AnuLMConfig(vocab_size=259, n_layer=0, hidden_size=64, n_head=4, block_size=len(val))
    model = AnuLM(cfg)
    args = SimpleNamespace(weight_decay=0.0, device="cpu", optimizer="adamw", lr=1e-2)
    opt = make_optimizer(model, args)                                       # train.py's AdamW
    g = torch.Generator().manual_seed(seed)
    model.train()
    for step in range(steps):
        ix = torch.randint(len(train) - 129, (32,), generator=g)
        x, y = make_batch(train, ix, 128, "cpu")                            # y = x shifted by one
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():                                                   # every pair of the split
        _, val_loss = model(val[:-1][None], val[1:][None])
    return model, val_loss.item(), sum(p.numel() for p in model.parameters())


def last_logits_move(model, vocab, seed=0) -> float:
    """Change every token but the last; how far do the last position's logits move?"""
    torch.manual_seed(seed)
    a = torch.randint(0, vocab, (1, 16))
    b = torch.randint(0, vocab, (1, 16))
    b[0, -1] = a[0, -1]
    with torch.no_grad():
        return (model(a)[0] - model(b)[0]).abs().max().item()


if __name__ == "__main__":
    train, val = load()
    print(f"data/input.txt: {len(train):,} training bytes, {len(val):,} held-out bytes")
    uniform = math.log(256)
    uni, bi, teach = unigram_loss(train, val), bigram_loss(train, val), teacher_loss(train, val)
    model, zero, n_params = zero_block_model(train, val)
    course = json.loads((ROOT / "course" / "ch01-smallest-model" / "results.json").read_text())
    rows = [("uniform guess over 256 bytes", uniform),
            ("unigram counts (order ignored)", uni),
            ("bigram counts (one byte back)", bi),
            ("bigram_teacher.py's table", teach),
            (f"AnuLM, zero blocks ({n_params:,} params)", zero),
            ("lab book's tiny transformer (recorded)", course["tiny_val_loss"])]
    print("\nheld-out loss, nats per byte")
    for name, v in rows:
        print(f"  {name:42s} {v:6.4f}")
    assert abs(bi - course["bigram_val_loss"]) < 1e-3, "the count table should reproduce the lab book's 2.4931"
    assert abs(zero - bi) < 0.05, "a model with no blocks can only learn a bigram"
    zero_move = last_logits_move(model, 259)
    torch.manual_seed(0)
    deep_move = last_logits_move(AnuLM(tiny()).eval(), 64)
    print(f"\nlast-position logits move when earlier tokens change:")
    print(f"  zero blocks     {zero_move:.2e}")
    print(f"  tiny(), 3 blocks {deep_move:.2e}")
    assert zero_move == 0.0 and deep_move > 1e-4
    out = {"uniform": uniform, "unigram": uni, "bigram_counts": bi, "bigram_teacher": teach,
           "zero_block_anulm": zero, "zero_block_params": n_params,
           "course_tiny_transformer": course["tiny_val_loss"], "course_bigram": course["bigram_val_loss"],
           "zero_block_move": zero_move, "tiny_move": deep_move,
           "data": "data/input.txt", "split": "first 90% train, last 10% held out", "steps": 1500, "seed": 0}
    smooth = {a: bigram_loss(train, val, alpha=a) for a in (0.5, 0.1)}
    print("bigram counts with less smoothing: " + ", ".join(f"add-{a}: {v:.4f}" for a, v in smooth.items()))
    out["bigram_add_0.5"], out["bigram_add_0.1"] = smooth[0.5], smooth[0.1]
    path = (ROOT / "book" / "measurements" if "--save" in sys.argv else Path.cwd()) / "part_i_ii_ladder.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {path}  (book/measurements/ only with --save)")
