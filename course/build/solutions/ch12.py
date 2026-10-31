"""
Reference solution for Chapter 12: decisions (Decide) and lookup (BM25).

encode_options, readable, fit_temperature and calibration match decide.py's
encode, readable, fit_temperature and metrics; words and BM25 match rag.py's
tokeniser and scoring (without its hashing into buckets, which only matters at
millions of words). course/checks/ch12.py proves both.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Decide: one forward pass, a probability for every option -----------------------------

def readable(label: str) -> str:
    """'card_arrival' -> 'card arrival'; 'EducationalInstitution' -> 'educational institution'."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", label).replace("_", " ").lower()


def encode_options(encode, bos: int, text: str, options):
    """(ids, positions): the message, then every option's name on its own line. positions[k] is the
    index of the LAST id of option k's name, where the causal model has read the message and
    that option. Pieces are encoded separately, so the positions are exact."""
    ids = [bos] + encode(f"Message: {text.strip()}\nWhich of these does it ask for?\nOptions:")
    pos = []
    for o in options:
        ids += encode(f"\n- {readable(o)}")
        pos.append(len(ids) - 1)
    return ids, pos


def final_hidden(model, idx):
    """The model's last hidden states (after its final norm), shape (B, S, hidden)."""
    seen = {}
    hook = model.norm.register_forward_hook(lambda mod, inp, out: seen.__setitem__("h", out))
    try:
        model(idx)
    finally:
        hook.remove()
    return seen["h"]


class Decider(nn.Module):
    """A language model with a one-number head read at each option's last position."""

    def __init__(self, model, hidden_size: int):
        super().__init__()
        self.model = model
        self.head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("temperature", torch.ones(()))

    def forward(self, idx, pos, mask=None):
        """idx (B, S), pos (B, K), mask (B, K) True for real options -> logits (B, K)."""
        h = final_hidden(self.model, idx)
        g = h.gather(1, pos[..., None].expand(-1, -1, h.shape[-1]))
        logits = self.head(g).squeeze(-1).float()
        return logits if mask is None else logits.masked_fill(~mask, float("-inf"))


def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    """The single T in 0.30, 0.35, ..., 5.00 minimising held-out NLL of softmax(logits / T)."""
    best = (float("inf"), 1.0)
    for t in [x / 100 for x in range(30, 501, 5)]:
        best = min(best, (F.cross_entropy(logits / t, y).item(), t))
    return best[1]


def calibration(probs: torch.Tensor, y: torch.Tensor, bins: int = 15) -> dict:
    """Accuracy; expected calibration error (bins of confidence, weighted |confidence - accuracy|);
    Brier score; and, for thresholds 0.5, 0.7, 0.9, how much is answered and how accurately."""
    conf, pred = probs.max(1)
    correct = (pred == y).float()
    ece = 0.0
    for b in range(bins):
        m = (conf > b / bins) & (conf <= (b + 1) / bins)
        if m.any():
            ece += m.float().mean().item() * abs(conf[m].mean().item() - correct[m].mean().item())
    brier = ((probs - F.one_hot(y, probs.shape[1]).float()) ** 2).sum(1).mean().item()
    out = {"accuracy": correct.mean().item(), "ece": ece, "brier": brier,
           "mean_confidence": conf.mean().item(), "n": len(y)}
    for tau in (0.5, 0.7, 0.9):
        keep = conf >= tau
        out[f"answer_if_conf>={tau}"] = {"coverage": keep.float().mean().item(),
                                         "accuracy": correct[keep].mean().item() if keep.any() else float("nan")}
    return out


# --- lookup: BM25 ---------------------------------------------------------------------------

WORD = re.compile(r"[0-9A-Za-zÀ-ɏऀ-ॿ]+")      # rag.py's: Devanagari vowel signs are word characters


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


class BM25:
    """Okapi BM25 over a list of passages, with rag.py's constants and idf."""

    def __init__(self, docs, k1: float = 1.2, b: float = 0.75, tokenize=None):
        self.k1, self.b, self.tokenize = k1, b, tokenize or words
        self.tf = [Counter(self.tokenize(d)) for d in docs]
        self.lens = torch.tensor([sum(c.values()) for c in self.tf], dtype=torch.float64)
        self.avgdl = self.lens.mean().item()
        n = len(docs)
        df = Counter(w for c in self.tf for w in c)
        self.idf = {w: math.log1p((n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def scores(self, query: str) -> torch.Tensor:
        """One score per passage. Each distinct query word counts once."""
        s = torch.zeros(len(self.tf), dtype=torch.float64)
        norm = self.k1 * (1 - self.b + self.b * self.lens / self.avgdl)
        for w in set(self.tokenize(query)):
            if w not in self.idf:
                continue
            tf = torch.tensor([c.get(w, 0) for c in self.tf], dtype=torch.float64)
            s += self.idf[w] * tf * (self.k1 + 1) / (tf + norm)
        return s

    def search(self, query: str, k: int = 3) -> list[int]:
        s = self.scores(query)
        top = torch.argsort(-s, stable=True)[:k]
        return [int(i) for i in top if s[i] > 0]
