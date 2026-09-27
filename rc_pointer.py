"""
Extractive reader for rag.py: point at the answer instead of generating it.

    python rc_pointer.py train --ckpt ckpt_ctx2k.pt --out ckpt_rc_pointer.pt
    python rc_pointer.py eval  --ckpt ckpt_rc_pointer.pt
    python rc_pointer.py ask   "What is the capital of India?"

Why, from the first attempt (docs/RESULTS.md section 33): a generative reader
fine-tuned on the same 59k examples learned the format and to abstain, but
not to read -- F1 4.9 English / 1.5 Hindi, 0 of 20 demo questions. Two
things were wrong for a model this size:

  1. The question came AFTER the passages. In a left-to-right model every
     passage token was encoded without knowing what to look for. Here the
     question comes first, so each passage token's state already depends on it.
  2. The signal was weak: 1% of positions carried loss, and the answer had
     to be generated token by token. Here a 2-way linear head on the backbone
     scores every position as the answer's start and end -- a dense, exact
     target -- and the model can only ever return text that is in the passages.

Unanswerable examples (make_rc.py's 15%) point both ends at position 0, a
reserved EOS placed before the question: "no answer here".
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from bpe import BPE
from model import AnuLM, load_checkpoint

HERE = Path(__file__).parent
MAX_LEN = 640
MAX_ANS = 30
DEV = re.compile(r"[ऀ-ॿ]")
NOT_FOUND = ("not found in the passages", "अनुच्छेदों में उत्तर नहीं मिला")


class Pointer(nn.Module):
    def __init__(self, model: AnuLM):
        super().__init__()
        self.model = model
        self.head = nn.Linear(model.cfg.hidden_size, 2)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, idx):
        m = self.model
        x = F.embedding(idx, m._table("in"))
        cos, sin = m.rotary(idx.shape[1], x.device, x.dtype)
        for layer in m.layers:
            if m.grad_ckpt and self.training:
                x, _ = torch.utils.checkpoint.checkpoint(layer, x, cos, sin, use_reentrant=False)
            else:
                x, _ = layer(x, cos, sin)
        s, e = self.head(m.norm(x)).float().unbind(-1)
        return s, e                                             # (B, S) each


def split_row(row):
    """make_rc.py stores '[1] p1\\n\\n[2] p2 ...\\n\\n<qword>: question'."""
    qword = "प्रश्न" if row["lang"] == "rc-hi" else "Question"
    body, _, q = row["question"].rpartition(f"\n\n{qword}: ")
    return qword, q, body


def encode(tok, qword, q, body, answer=None):
    """-> (ids, start, end). Question first, then passages. With `answer`,
    the passages are encoded in three pieces around its first occurrence so
    the span's token positions are exact; (0, 0) if absent / unanswerable."""
    head = [tok.eos_id] + tok.encode(f"{qword}: {q}\n\n")
    if answer and answer not in NOT_FOUND:
        i = body.find(answer)
        if i >= 0:
            a0 = i - 1 if i > 0 and body[i - 1] == " " else i      # BPE keeps the space with the word
            pre, span, post = tok.encode(body[:a0]), tok.encode(body[a0:i + len(answer)]), \
                tok.encode(body[i + len(answer):])
            ids = head + pre + span + post
            st = len(head) + len(pre)
            en = st + len(span) - 1
            if en < MAX_LEN and span:
                return ids[:MAX_LEN], st, en
            return None
        return None
    return (head + tok.encode(body))[:MAX_LEN], 0, 0


def load_rows(path, tok, n=None):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")][:n]
    out = []
    for r in rows:
        qword, q, body = split_row(r)
        enc = encode(tok, qword, q, body, r["answer"])
        if enc:
            out.append((*enc, r))
    return out


def best_span(s, e, n):
    """argmax s[i] + e[j] over 1 <= i <= j < i + MAX_ANS, against the no-answer
    score s[0] + e[0]. Returns (i, j) or None."""
    s, e = s[:n], e[:n]
    na = float(s[0] + e[0])
    S = s[1:, None] + e[None, 1:]                                   # (n-1, n-1)
    i = torch.arange(n - 1, device=s.device)
    ok = (i[None, :] >= i[:, None]) & (i[None, :] < i[:, None] + MAX_ANS)
    S = S.masked_fill(~ok, float("-inf"))
    k = int(S.argmax())
    st, en = divmod(k, n - 1)
    if float(S.view(-1)[k]) <= na:
        return None
    return st + 1, en + 1


def load(ckpt, device):
    ck = load_checkpoint(ckpt, "cpu")
    cfg = replace(ck["cfg"], moe_impl="grouped" if device.startswith("cuda") else "sparse")
    m = AnuLM(cfg)
    if "pointer_head" in ck:
        m.load_state_dict(ck["model"])
        p = Pointer(m)
        p.head.load_state_dict(ck["pointer_head"])
    else:
        m.load_state_dict(ck["model"])
        p = Pointer(m)
    return p.to(device), BPE.load(cfg.tokenizer_path), ck


def train(a):
    torch.manual_seed(0)
    p, tok, ck = load(a.ckpt, "cuda")
    p.model.enable_gradient_checkpointing(a.grad_ckpt)
    data = load_rows(HERE / "data" / "rc_train.jsonl", tok)
    held = load_rows(HERE / "data" / "rc_heldout.jsonl", tok)
    print(f"train {len(data)} examples, held-out {len(held)}; "
          f"{sum(1 for d in data if d[1] == 0) / len(data):.0%} unanswerable", flush=True)
    opt = torch.optim.AdamW([{"params": p.model.parameters(), "lr": a.lr},
                             {"params": p.head.parameters(), "lr": 1e-3}],
                            betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    steps = int(len(data) * a.epochs) // a.batch_size
    base = [g["lr"] for g in opt.param_groups]
    ac = torch.autocast("cuda", dtype=torch.bfloat16)

    def batch(items):
        L = max(len(d[0]) for d in items)
        x = torch.full((len(items), L), tok.eos_id, dtype=torch.long)
        mask = torch.zeros(len(items), L, dtype=torch.bool)
        for b, d in enumerate(items):
            x[b, :len(d[0])] = torch.tensor(d[0])
            mask[b, :len(d[0])] = True
        st = torch.tensor([d[1] for d in items])
        en = torch.tensor([d[2] for d in items])
        return x.cuda(), mask.cuda(), st.cuda(), en.cuda()

    def loss_of(items):
        x, mask, st, en = batch(items)
        with ac:
            s, e = p(x)
        s, e = s.masked_fill(~mask, -1e4), e.masked_fill(~mask, -1e4)
        return (F.cross_entropy(s, st) + F.cross_entropy(e, en)) / 2

    @torch.no_grad()
    def held_loss():
        p.eval()
        v = sum(loss_of(held[i:i + 16]).item() * len(held[i:i + 16]) for i in range(0, len(held), 16))
        p.train()
        return v / len(held)

    rng = random.Random(0)
    order = []
    t0, best = time.time(), float("inf")
    p.train()
    for step in range(steps):
        if len(order) < a.batch_size:
            order = list(range(len(data)))
            rng.shuffle(order)
        items = [data[order.pop()] for _ in range(a.batch_size)]
        scale = min(1.0, (step + 1) / 200) * max(0.0, 1 - step / steps)
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * scale
        loss = loss_of(items)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(p.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        p.model.update_expert_biases()
        if step % 250 == 0:
            print(f"step {step:5d}/{steps} | loss {loss.item():.3f} | {time.time() - t0:.0f}s", flush=True)
        if (step + 1) % a.eval_every == 0 or step == steps - 1:
            v = held_loss()
            flag = ""
            if v < best:
                best = v
                torch.save({"model": p.model.state_dict(), "pointer_head": p.head.state_dict(),
                            "cfg": p.model.cfg, "step": step, "val_loss": v, "base_ckpt": a.ckpt}, a.out)
                flag = "  <- saved"
            print(f"  eval @ {step:5d} | held-out span loss {v:.4f}{flag}", flush=True)
    print(f"done in {time.time() - t0:.0f}s | best held-out span loss {best:.4f} | {a.out}")


class PointerQA:
    def __init__(self, ckpt="ckpt_rc_pointer.pt", device="cuda", retriever=None):
        self.p, self.tok, _ = load(ckpt, device)
        self.p.eval()
        self.device = device
        self.r = retriever

    @torch.no_grad()
    def extract(self, qword, q, body):
        ids, _, _ = encode(self.tok, qword, q, body)
        x = torch.tensor([ids], device=self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            s, e = self.p(x)
        span = best_span(s[0], e[0], len(ids))
        return self.tok.decode(ids[span[0]:span[1] + 1]).strip() if span else ""

    def ask(self, question, k=3):
        from rag import Retriever
        if self.r is None:
            self.r = Retriever()
        t0 = time.time()
        hits = self.r.search(question, k)
        body = "\n\n".join(f"[{i + 1}] {' '.join(h['text'].split()[:120])}" for i, h in enumerate(hits))
        qword = "प्रश्न" if DEV.search(question) else "Question"
        t1 = time.time()
        ans = self.extract(qword, question, body)
        return {"answer": ans, "sources": [h["title"] for h in hits],
                "retrieve_s": round(t1 - t0, 3), "read_s": round(time.time() - t1, 3)}


def evaluate(a):
    from rag import em_f1
    qa = PointerQA(a.ckpt)
    for name in ("squad_en", "mlqa_hi", "xquad_hi", "indicqa_hi"):
        rows = [json.loads(l) for l in open(HERE / "data" / f"rc_eval_{name}.jsonl", encoding="utf-8")][:a.n]
        ems = f1s = empty = 0.0
        for r in rows:
            pred = qa.extract(*split_row(r))
            e, f = em_f1(pred, r["answers"])
            ems, f1s, empty = ems + e, f1s + f, empty + (pred == "")
        k = len(rows)
        print(f"  {name:12s} EM {100 * ems / k:5.1f}  F1 {100 * f1s / k:5.1f}  "
              f"no-answer {100 * empty / k:4.1f}%  (n={k})", flush=True)
    items = [json.loads(l) for l in open(HERE / "data" / "rag_demo_questions.jsonl", encoding="utf-8")]
    right = 0
    for it in items:
        out = qa.ask(it["q"])
        ok = bool(out["answer"]) and any(x in out["answer"] for x in it["a"])
        right += ok
        print(f"  {'OK ' if ok else '-- '} {it['q']}  ->  {out['answer'] or '(no answer)'}   "
              f"[{', '.join(out['sources'][:2])}]")
    print(f"  demo questions: {right}/{len(items)} answered correctly end to end")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--ckpt", default="ckpt_ctx2k.pt")
    t.add_argument("--out", default="ckpt_rc_pointer.pt")
    t.add_argument("--epochs", type=float, default=1.0)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--eval-every", type=int, default=1000)
    t.add_argument("--grad-ckpt", action="store_true")
    e = sub.add_parser("eval")
    e.add_argument("--ckpt", default="ckpt_rc_pointer.pt")
    e.add_argument("--n", type=int, default=300)
    q = sub.add_parser("ask")
    q.add_argument("question")
    q.add_argument("--ckpt", default="ckpt_rc_pointer.pt")
    a = ap.parse_args()
    if a.cmd == "train":
        train(a)
    elif a.cmd == "eval":
        evaluate(a)
    else:
        out = PointerQA(a.ckpt).ask(a.question)
        print(f"{out['answer'] or '(no answer in the passages)'}\n  sources: {out['sources']}")


if __name__ == "__main__":
    main()
