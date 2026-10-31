"""
Chapter 1 lab: the smallest language model there is, then a tiny transformer.

    python lab.py                          # Shakespeare, data/input.txt (public domain)
    python lab.py --data my_notes.txt      # any text file on your own disk, any language

Runs on a laptop CPU in a few minutes. Writes, next to this file:
    tiny_lm.pt       the trained transformer (Chapter 7 fine-tunes it)
    curve.csv        validation loss as it trains
    results.json     the numbers the chapter's figures are drawn from

Everything is bytes: the model reads and writes one byte (0-255) at a time,
so there is no tokenizer to build yet (that is Chapter 3) and Hindi, English
or Python all work unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
LN2 = math.log(2)

# Generated text can be any bytes, including Hindi; Windows consoles default to a
# code page that cannot print it, so say UTF-8 and replace what still fails.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def load_bytes(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The file as a sequence of byte values, split 90% train / 10% validation."""
    data = torch.tensor(list(Path(path).read_bytes()), dtype=torch.long)
    cut = int(0.9 * len(data))
    return data[:cut], data[cut:]


# --- Part A: a model that only counts ------------------------------------

def bigram(train: torch.Tensor, val: torch.Tensor) -> tuple[torch.Tensor, float]:
    """P(next byte | this byte), read straight off a 256 x 256 table of counts.
    Every cell starts at 1 so no pair is ever impossible (add-one smoothing)."""
    counts = torch.ones(256, 256)
    counts.index_put_((train[:-1], train[1:]), torch.ones(len(train) - 1), accumulate=True)
    probs = counts / counts.sum(1, keepdim=True)
    loss = -probs[val[:-1], val[1:]].log().mean().item()   # average surprise on unseen text
    return probs, loss


def sample_bigram(probs: torch.Tensor, start: bytes, n: int, g: torch.Generator) -> str:
    out = list(start)
    for _ in range(n):
        out.append(torch.multinomial(probs[out[-1]], 1, generator=g).item())
    return bytes(out).decode("utf-8", errors="replace")


# --- Part B: a tiny transformer -------------------------------------------

class Block(nn.Module):
    """One transformer layer: look back at earlier bytes (attention), then think
    about what was found (a small MLP). Each adds to the running representation."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = (t.view(B, T, self.heads, D // self.heads).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # never look ahead
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self, d: int = 128, layers: int = 4, heads: int = 4, ctx: int = 128):
        super().__init__()
        self.cfg = dict(d=d, layers=layers, heads=heads, ctx=ctx)
        self.tok = nn.Embedding(256, d)            # one vector per byte value
        self.pos = nn.Embedding(ctx, d)            # one vector per position
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 256)              # a score for every possible next byte

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln(x))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, 256), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, start: bytes, n: int, g: torch.Generator, temperature: float = 0.8) -> str:
        idx = torch.tensor([list(start)])
        for _ in range(n):
            logits, _ = self(idx[:, -self.cfg["ctx"]:])
            p = F.softmax(logits[0, -1] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(p, 1, generator=g)[None]], dim=1)
        return bytes(idx[0].tolist()).decode("utf-8", errors="replace")


def batch(data: torch.Tensor, ctx: int, size: int, g: torch.Generator):
    """`size` random windows of `ctx` bytes; the target is each window shifted by one."""
    i = torch.randint(len(data) - ctx - 1, (size,), generator=g)
    x = torch.stack([data[j: j + ctx] for j in i])
    y = torch.stack([data[j + 1: j + ctx + 1] for j in i])
    return x, y


@torch.no_grad()
def evaluate(model: TinyLM, data: torch.Tensor, ctx: int, g: torch.Generator, batches: int = 20) -> float:
    model.eval()
    loss = sum(model(*batch(data, ctx, 32, g))[1].item() for _ in range(batches)) / batches
    model.train()
    return loss


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(HERE.parent.parent / "data" / "input.txt"))
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--prompt", default="ROMEO:\n")
    p.add_argument("--seed", type=int, default=1337)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    g = torch.Generator().manual_seed(a.seed)

    train, val = load_bytes(a.data)
    print(f"{a.data}: {len(train):,} training bytes, {len(val):,} validation bytes")

    probs, bigram_loss = bigram(train, val)
    print(f"\nA. bigram (a table of counts): validation loss {bigram_loss:.3f} nats/byte "
          f"= {bigram_loss / LN2:.2f} bits/byte")
    bigram_text = sample_bigram(probs, a.prompt.encode(), 200, g)
    print(bigram_text)

    model = TinyLM()
    ctx = model.cfg["ctx"]
    n_params = sum(t.numel() for t in model.parameters())
    print(f"\nB. tiny transformer: {n_params:,} parameters, training {a.steps} steps on CPU")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    rows, t0 = [(0, evaluate(model, val, ctx, g))], time.time()
    for step in range(1, a.steps + 1):
        lr = 1e-3 * min(1.0, step / 100) * 0.5 * (1 + math.cos(math.pi * step / a.steps))  # warm up, then cool down
        for group in opt.param_groups:
            group["lr"] = lr
        _, loss = model(*batch(train, ctx, 32, g))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0:
            rows.append((step, evaluate(model, val, ctx, g)))
            print(f"  step {step:5d}  validation loss {rows[-1][1]:.3f}  ({time.time() - t0:.0f}s)")
    tiny_text = model.generate(a.prompt.encode(), 300, g)
    print(f"\n{tiny_text}")

    # The next-byte distribution after a familiar phrase, for Figure 1.1.
    model.eval()
    phrase = b"To be or not to b"
    with torch.no_grad():
        nxt = F.softmax(model(torch.tensor([list(phrase)]))[0][0, -1], -1)
    top = nxt.topk(6)

    torch.save({"cfg": model.cfg, "model": model.state_dict(), "data": a.data}, HERE / "tiny_lm.pt")
    with open(HERE / "curve.csv", "w", newline="") as f:
        csv.writer(f).writerows([("step", "val_loss")] + [(s, round(v, 4)) for s, v in rows])
    letters = b"etaoinshr \n"
    sub = probs[list(letters)][:, list(letters)]
    (HERE / "results.json").write_text(json.dumps({
        "data": Path(a.data).name, "train_bytes": len(train), "params": n_params, "steps": a.steps,
        "seconds": round(time.time() - t0), "bigram_val_loss": round(bigram_loss, 4),
        "tiny_val_loss": round(rows[-1][1], 4), "uniform_loss": round(math.log(256), 4),
        "bigram_sample": bigram_text, "tiny_sample": tiny_text,
        "next_after": phrase.decode(), "next_top": [[bytes([int(i)]).decode("latin-1"), round(float(v), 4)]
                                                   for v, i in zip(top.values, top.indices)],
        "bigram_letters": letters.decode(), "bigram_sub": [[round(float(x), 4) for x in r] for r in sub],
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved tiny_lm.pt, curve.csv, results.json  (bigram {bigram_loss:.3f} -> transformer "
          f"{rows[-1][1]:.3f} nats/byte; a uniform guess is {math.log(256):.3f})")


if __name__ == "__main__":
    main()
