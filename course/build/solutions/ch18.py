"""
Reference solution for Chapter 18: chat.

conversation_ids extends Chapter 9's pair_ids to several turns (one turn gives
finetune.py's ids exactly); pack_conversations is finetune.pack for them;
split_heldout is make_chat.py's split, with one fix: copies of the same question
stay on one side. course/checks/ch18.py proves them.
"""

from __future__ import annotations

import re

import torch

CHAT = "User: {q}\nAssistant:"


def conversation_ids(encode, eos: int, turns, template: str = CHAT):
    """turns: [(user text, assistant reply), ...]. Each turn is the template with the user's text,
    then " " + the reply and EOS. Returns (ids, labels): labels are -1 on everything the user wrote
    and the template, and the id itself on every reply id and its EOS, in every turn."""
    ids, labels = [], []
    for user, reply in turns:
        p = encode(template.format(q=user))
        a = encode(" " + reply) + [eos]
        ids += p + a
        labels += [-1] * len(p) + a
    return ids, labels


def pack_conversations(convs, block_size: int, eos: int):
    """finetune.pack for (ids, labels) conversations: rows of block_size + 1, a conversation never split
    across rows (one longer than block_size is dropped), rows padded with eos and -1. Returns
    (rows[:, :-1], labels[:, 1:]), so position t is graded on id t + 1 when that id is a reply's."""
    L = block_size + 1
    rows, tgts, cur, lab = [], [], [], []

    def flush():
        rows.append(cur + [eos] * (L - len(cur)))
        tgts.append(lab + [-1] * (L - len(lab)))

    for ids, labels in convs:
        if len(ids) > block_size:
            continue
        if len(cur) + len(ids) > L:
            flush()
            cur, lab = [], []
        cur, lab = cur + list(ids), lab + list(labels)
    if cur:
        flush()
    if not rows:
        return torch.zeros(0, block_size, dtype=torch.long), torch.zeros(0, block_size, dtype=torch.long)
    x, y = torch.tensor(rows), torch.tensor(tgts)
    return x[:, :-1].contiguous(), y[:, 1:].contiguous()


def normal(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def split_heldout(rows, n: int, rng, max_answer: int = 1500):
    """make_chat.py's split: drop rows with an empty question or answer, or an answer longer than
    max_answer; shuffle; hold out n rows, never one from the "identity" source. Unlike make_chat.py,
    every copy of a held-out question (compared case- and space-insensitively) is held out with it, so
    the held-out set never asks what training already answered. Returns (train, held)."""
    kept = [r for r in rows if r["question"] and r["answer"] and len(r["answer"]) <= max_answer]
    rng.shuffle(kept)
    held_q = []
    seen = set()
    for r in kept:
        if len(held_q) == n:
            break
        q = normal(r["question"])
        if r.get("source") != "identity" and q not in seen:
            seen.add(q)
            held_q.append(q)
    held = [r for r in kept if normal(r["question"]) in seen]
    train = [r for r in kept if normal(r["question"]) not in seen]
    return train, held
