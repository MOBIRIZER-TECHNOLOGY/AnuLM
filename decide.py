"""
AnuLM-Decide: typed, calibrated decisions instead of text (a "System One" model).

    python decide.py train --ckpt ckpt_demo_v2.pt --out ckpt_decide.pt
    python decide.py eval  --ckpt ckpt_decide.pt
    python decide.py ask   "I still haven't received my card"

The idea, from TypeSafe's Jev (docs/RESULTS.md section 34): software often
needs a decision -- which queue, which intent, is this phishing -- not prose.
So the model never writes text. It is given a message and a declared list of
options, reads both in ONE forward pass, and returns a probability for every
option; the answer can only be one of the options, and the confidence is
calibrated so that "0.9" means right about 90% of the time.

How, on this project's own from-scratch backbone:

  * the prompt is the message, then every option's name: "Message: ... Options:
    - card arrival - card linking - ...". The backbone is causal, so the
    hidden state at the LAST token of each option's name has read the message
    and that option; a linear head scores it. Softmax over the options gives
    the distribution. One pass, whatever the number of options (77 here).
  * options are shuffled on every training example, so the head learns to
    read option names, not positions -- the same model takes any option list.
  * trained with cross-entropy (a proper scoring rule), then calibrated with a
    single temperature fitted on held-out data (Guo et al. 2017). Reported:
    accuracy, expected calibration error (ECE), Brier score, and accuracy when
    it answers only above a confidence threshold, handing the rest to a human.

First use case: BANKING77 (CC BY 4.0), 77 customer-support intents, 10,003
training / 3,080 test queries, from PolyAI's repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import AnuLM, fit_memory, load_checkpoint

HERE = Path(__file__).parent
DATA = HERE / "data" / "banking77"


# --------------------------------------------------------------------- data
def load_split(name: str) -> list[tuple[str, str]]:
    return [(r["text"], r["category"]) for r in csv.DictReader(open(DATA / f"{name}.csv", encoding="utf-8"))]


def labels() -> list[str]:
    return json.load(open(DATA / "categories.json"))


def readable(label: str) -> str:
    return label.replace("_", " ")


def encode(tok, text: str, options: list[str], max_len: int = 1024):
    """-> (ids, positions) where positions[k] is the index of the last token of
    option k's name. Pieces are encoded separately so the positions are exact."""
    ids = [tok.eos_id] + tok.encode(f"Message: {text.strip()}\nWhich of these does it ask for?\nOptions:")
    pos = []
    for o in options:
        ids += tok.encode(f"\n- {readable(o)}")
        pos.append(len(ids) - 1)
    assert len(ids) <= max_len, f"{len(ids)} tokens; too many options for the context"
    return ids, pos


# --------------------------------------------------------------------- model
class Decider(nn.Module):
    def __init__(self, model: AnuLM):
        super().__init__()
        self.model = model
        self.head = nn.Linear(model.cfg.hidden_size, 1)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("temperature", torch.ones(()))

    def hidden(self, idx):
        m = self.model
        x = F.embedding(idx, m._table("in"))
        cos, sin = m.rotary(idx.shape[1], x.device, x.dtype)
        for layer in m.layers:
            if m.grad_ckpt and self.training:
                x, _ = torch.utils.checkpoint.checkpoint(layer, x, cos, sin, use_reentrant=False)
            else:
                x, _ = layer(x, cos, sin)
        return m.norm(x)

    def forward(self, idx, pos):
        """idx (B, S); pos (B, K) option positions -> raw logits (B, K)."""
        h = self.hidden(idx)
        g = h.gather(1, pos[..., None].expand(-1, -1, h.shape[-1]))
        return self.head(g).squeeze(-1).float()


def batchify(examples, tok, pad):
    L = max(len(e[0]) for e in examples)
    x = torch.full((len(examples), L), pad, dtype=torch.long)
    for b, e in enumerate(examples):
        x[b, :len(e[0])] = torch.tensor(e[0])
    pos = torch.tensor([e[1] for e in examples])
    y = torch.tensor([e[2] for e in examples])
    return x, pos, y


def make_examples(rows, tok, labs, rng=None):
    out = []
    for text, cat in rows:
        opts = labs[:]
        if rng:
            rng.shuffle(opts)
        ids, pos = encode(tok, text, opts)
        out.append((ids, pos, opts.index(cat), opts))
    return out


def load(ckpt: str, device: str):
    from hf_tok import load_tokenizer
    ck = load_checkpoint(ckpt, "cpu")
    cfg = replace(ck["cfg"], moe_impl="grouped" if device.startswith("cuda") else "sparse")
    m = AnuLM(cfg)
    m.load_state_dict(ck["model"])
    d = Decider(m)
    if "decide_head" in ck:
        d.head.load_state_dict(ck["decide_head"])
        d.temperature.fill_(ck.get("temperature", 1.0))
    return d.to(device), load_tokenizer(cfg.tokenizer_path), ck


# --------------------------------------------------------------------- calibration and metrics
def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    """The single T minimising held-out NLL of softmax(logits / T)."""
    best = (float("inf"), 1.0)
    for t in [x / 100 for x in range(30, 501, 5)]:
        nll = F.cross_entropy(logits / t, y).item()
        best = min(best, (nll, t))
    return best[1]


def metrics(probs: torch.Tensor, y: torch.Tensor, bins: int = 15) -> dict:
    conf, pred = probs.max(1)
    correct = (pred == y).float()
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean().item() * abs(conf[m].mean().item() - correct[m].mean().item())
    onehot = F.one_hot(y, probs.shape[1]).float()
    brier = ((probs - onehot) ** 2).sum(1).mean().item()
    out = {"accuracy": correct.mean().item(), "ece": ece, "brier": brier,
           "mean_confidence": conf.mean().item(), "n": len(y)}
    for tau in (0.5, 0.7, 0.9):
        keep = conf >= tau
        out[f"answer_if_conf>={tau}"] = {"coverage": keep.float().mean().item(),
                                         "accuracy": correct[keep].mean().item() if keep.any() else float("nan")}
    return out


@torch.no_grad()
def predict_logits(d: Decider, examples, tok, device, bs=16):
    d.eval()
    out = []
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda"))
    for i in range(0, len(examples), bs):
        x, pos, _ = batchify(examples[i:i + bs], tok, tok.eos_id)
        with ac:
            out.append(d(x.to(device), pos.to(device)).cpu())
    return torch.cat(out)


# --------------------------------------------------------------------- train / eval / ask
def train(a):
    torch.manual_seed(0)
    rng = random.Random(0)
    d, tok, ck = load(a.ckpt, a.device)
    fit_memory(d.model)
    if a.grad_ckpt:
        d.model.enable_gradient_checkpointing()
    labs = labels()
    rows = load_split("train")
    rng.shuffle(rows)
    if a.limit:
        rows = rows[:a.limit]
    n_val = min(1000, len(rows) // 10)
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    val = make_examples(val_rows, tok, labs)                       # canonical order, as at test
    opt = torch.optim.AdamW([{"params": d.model.parameters(), "lr": a.lr},
                             {"params": d.head.parameters(), "lr": 1e-3}], betas=(0.9, 0.95),
                            weight_decay=0.0, fused=a.device.startswith("cuda"))
    steps = int(len(train_rows) * a.epochs) // a.batch_size
    base = [g["lr"] for g in opt.param_groups]
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.device.startswith("cuda"))
    print(f"train {len(train_rows)} / val {len(val_rows)} queries, {len(labs)} intents, {steps} steps", flush=True)
    t0, best, order = time.time(), -1.0, []
    d.train()
    for step in range(steps):
        if len(order) < a.batch_size:
            order = list(range(len(train_rows)))
            rng.shuffle(order)
        batch = make_examples([train_rows[order.pop()] for _ in range(a.batch_size)], tok, labs, rng)
        x, pos, y = batchify(batch, tok, tok.eos_id)
        scale = min(1.0, (step + 1) / 100) * max(0.0, 1 - step / steps)
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * scale
        with ac:
            logits = d(x.to(a.device), pos.to(a.device))
        loss = F.cross_entropy(logits, y.to(a.device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(d.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        d.model.update_expert_biases()
        if step % 100 == 0:
            print(f"step {step:5d}/{steps} | loss {loss.item():.4f} | {time.time() - t0:.0f}s", flush=True)
        if (step + 1) % a.eval_every == 0 or step == steps - 1:
            lv = predict_logits(d, val, tok, a.device)
            yv = torch.tensor([e[2] for e in val])
            acc = (lv.argmax(1) == yv).float().mean().item()
            flag = ""
            if acc > best:
                best = acc
                t = fit_temperature(lv, yv)
                torch.save({"model": d.model.state_dict(), "decide_head": d.head.state_dict(), "temperature": t,
                            "cfg": d.model.cfg, "step": step, "val_accuracy": acc, "labels": labs,
                            "base_ckpt": a.ckpt}, a.out)
                flag = f"  <- saved (temperature {t:.2f})"
            print(f"  eval @ {step:5d} | val accuracy {100 * acc:.2f}%{flag}", flush=True)
            d.train()
    print(f"done in {time.time() - t0:.0f}s | best val accuracy {100 * best:.2f}% | {a.out}")


def evaluate(a):
    d, tok, ck = load(a.ckpt, a.device)
    labs = ck.get("labels") or labels()
    test = make_examples(load_split("test")[:a.limit or None], tok, labs)
    y = torch.tensor([e[2] for e in test])
    logits = predict_logits(d, test, tok, a.device)
    t = float(d.temperature)
    for name, temp in (("raw (T=1)", 1.0), (f"calibrated (T={t:.2f}, fitted on val)", t)):
        m = metrics(F.softmax(logits / temp, 1), y)
        print(f"\n{name}: accuracy {100 * m['accuracy']:.2f}% on {m['n']} test queries | "
              f"ECE {100 * m['ece']:.2f}% | Brier {m['brier']:.4f} | mean confidence {100 * m['mean_confidence']:.1f}%")
        for k, v in m.items():
            if k.startswith("answer_if"):
                print(f"  {k}: answers {100 * v['coverage']:.1f}% of queries at {100 * v['accuracy']:.2f}% accuracy, "
                      f"hands {100 * (1 - v['coverage']):.1f}% to a human")
    # latency: one query, one forward pass
    ex = test[:50]
    with torch.no_grad():
        for e in ex[:3]:
            predict_logits(d, [e], tok, a.device, bs=1)
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.time()
        for e in ex:
            predict_logits(d, [e], tok, a.device, bs=1)
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()
    print(f"\nlatency: {1000 * (time.time() - t0) / len(ex):.1f} ms per query on {a.device} (77 options, one pass)")


class Decide:
    """The typed interface: decide(message, options) -> choice, probabilities, confidence, action."""

    def __init__(self, ckpt: str = "ckpt_decide.pt", device: str = "cuda", threshold: float = 0.7):
        self.d, self.tok, ck = load(ckpt, device)
        self.d.eval()
        self.device, self.threshold = device, threshold
        self.labels = ck.get("labels") or labels()

    @torch.no_grad()
    def __call__(self, message: str, options: list[str] | None = None, top: int = 3) -> dict:
        options = options or self.labels
        ids, pos = encode(self.tok, message, options)
        logits = predict_logits(self.d, [(ids, pos, 0, options)], self.tok, self.device, bs=1)[0]
        p = F.softmax(logits / float(self.d.temperature), 0)
        k = p.argsort(descending=True)[:top]
        conf = float(p[k[0]])
        return {"choice": options[int(k[0])],
                "probabilities": {options[int(i)]: round(float(p[i]), 4) for i in k},
                "confidence": round(conf, 4),
                "action": "auto-route" if conf >= self.threshold else "send to human"}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--ckpt", default="ckpt_demo_v2.pt")
    t.add_argument("--out", default="ckpt_decide.pt")
    t.add_argument("--epochs", type=float, default=3.0)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--lr", type=float, default=3e-5)
    t.add_argument("--eval-every", type=int, default=500)
    t.add_argument("--grad-ckpt", action="store_true")
    t.add_argument("--limit", type=int, default=0, help="use only this many training queries (smoke tests)")
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    e = sub.add_parser("eval")
    e.add_argument("--ckpt", default="ckpt_decide.pt")
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    q = sub.add_parser("ask")
    q.add_argument("message")
    q.add_argument("--ckpt", default="ckpt_decide.pt")
    q.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.cmd == "train":
        train(a)
    elif a.cmd == "eval":
        evaluate(a)
    else:
        print(json.dumps(Decide(a.ckpt, a.device)(a.message), indent=2))


if __name__ == "__main__":
    main()
