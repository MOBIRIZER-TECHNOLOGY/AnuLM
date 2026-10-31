"""
Reference solution for Chapter 9: supervised fine-tuning.

pair_ids and pack match finetune.py's _pair_ids and pack exactly
(course/checks/ch09.py proves it): the template, the space before the answer,
the end-of-text id, the prompt masked out of the loss, and examples that never
straddle two rows.
"""

from __future__ import annotations

import torch

PROMPT_EN = "Question: {q}\nAnswer:"          # make_qa.py's English template
BYTE_EOS = 257                                 # byte models: 256 BOS, 257 EOS, 258 PAD


def pair_ids(encode, eos: int, question: str, answer: str, template: str = PROMPT_EN):
    """(prompt ids, answer ids + EOS). The answer is encoded with its leading space, the way the
    text would be split if prompt and answer were written out together."""
    return encode(template.format(q=question)), encode(" " + answer) + [eos]


def pack(pairs, block_size: int, eos: int):
    """Pack (prompt, answer) pairs into rows of block_size: x the ids, y the targets, with -1 on
    every position that is not predicting an answer token. A pair never straddles two rows; one
    longer than a row is dropped, never cut. Rows are padded with eos (and -1 targets)."""
    L = block_size + 1
    rows, tgts, ids, lab = [], [], [], []

    def flush():
        rows.append(ids + [eos] * (L - len(ids)))
        tgts.append(lab + [-1] * (L - len(lab)))

    for p, a in pairs:
        ex = p + a
        if len(ex) > block_size:
            continue
        if len(ids) + len(ex) > L:
            flush()
            ids, lab = [], []
        ids = ids + ex
        lab = lab + [-1] * len(p) + list(a)
    if ids:
        flush()
    if not rows:
        return torch.zeros(0, block_size, dtype=torch.long), torch.zeros(0, block_size, dtype=torch.long)
    x, y = torch.tensor(rows), torch.tensor(tgts)
    return x[:, :-1].contiguous(), y[:, 1:].contiguous()       # position t predicts id t + 1
