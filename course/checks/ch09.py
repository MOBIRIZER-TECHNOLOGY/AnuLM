"""
Chapter 9 checks: formatting and packing examples for fine-tuning.

    python ch09.py                  # your course/build/post.py
    python ch09.py --impl solution  # the reference solution
    python ch09.py --self-test      # proves each check catches the bug it is for

Fine-tuning bugs teach the model the wrong thing politely: a prompt left in the
loss, an answer cut across two rows, a missing end token. It still learns,
just not what you'll ask of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import main, with_replaced  # noqa: E402
from anulm.training import finetune  # noqa: E402
from anulm.bpe import BPE  # noqa: E402
from anulm.data.make_qa import PROMPT_EN  # noqa: E402

CHECKS = []
TOK = BPE.train(("Question: what is the capital of India?\nAnswer: New Delhi is the capital. "
                 "Question: who wrote it?\nAnswer: a poet wrote it. ") * 30, 380, verbose=False)
ITEMS = [{"question": "what is the capital of India?", "answer": "New Delhi.", "lang": "en"},
         {"question": "who wrote the Ramayana?", "answer": "Valmiki wrote it, in Sanskrit.", "lang": "en"},
         {"question": "a long one " * 6 + "?", "answer": "and a long answer " * 4, "lang": "en"},
         {"question": "2+2?", "answer": "4", "lang": "en"}]


def check(fn):
    CHECKS.append(fn)
    return fn


def pairs(M):
    return [M.pair_ids(TOK.encode, TOK.eos_id, it["question"], it["answer"], PROMPT_EN) for it in ITEMS]


@check
def pair_ids_match_finetune(M):
    """Including the space before the answer: without it, the answer's first token is one the
    model never sees when the prompt and answer are written together."""
    for it, got in zip(ITEMS, pairs(M)):
        want = finetune._pair_ids(TOK, it)
        assert (list(got[0]), list(got[1])) == (list(want[0]), list(want[1])), \
            f"{it['question'][:30]!r}: ids differ from finetune._pair_ids"


@check
def the_answer_ends_with_eos(M):
    """The end token is what teaches the model to stop answering."""
    for p, a in pairs(M):
        assert a[-1] == TOK.eos_id, "the answer does not end with EOS"


@check
def pack_matches_finetune(M):
    ps = [finetune._pair_ids(TOK, it) for it in ITEMS]
    for block in (24, 40, 64):
        x, y = M.pack(ps, block, TOK.eos_id)
        wx, wy = finetune.pack(ps, block, TOK.eos_id)
        assert x.shape == wx.shape and torch.equal(x.long(), wx.long()) and torch.equal(y.long(), wy.long()), \
            f"pack(block_size={block}) differs from finetune.pack"


@check
def loss_lands_on_answers_only(M):
    """Every answer token is a target exactly once, and nothing else is. A prompt left in the loss
    trains the model to write questions too."""
    ps = [finetune._pair_ids(TOK, it) for it in ITEMS]
    x, y = M.pack(ps, 64, TOK.eos_id)
    kept = [(p, a) for p, a in ps if len(p) + len(a) <= 64]
    labelled = [int(v) for row in y for v in row if v != -1]
    assert labelled == [t for _, a in kept for t in a], "targets are not exactly the answers' tokens, in order"


@check
def targets_are_the_next_id(M):
    """Wherever a target is set, it is the id one position later: x[t + 1] == y[t]."""
    ps = [finetune._pair_ids(TOK, it) for it in ITEMS]
    x, y = M.pack(ps, 40, TOK.eos_id)
    for r in range(x.shape[0]):
        for t in range(x.shape[1] - 1):
            if y[r, t] != -1:
                assert int(y[r, t]) == int(x[r, t + 1]), "a target is not the next id (labels shifted wrongly)"


@check
def examples_are_never_cut(M):
    """A pair longer than a row is dropped, not truncated, and no pair straddles two rows."""
    ps = [finetune._pair_ids(TOK, it) for it in ITEMS]
    x, y = M.pack(ps, 8, TOK.eos_id)
    assert x.shape[0] == 0 or all(len(p) + len(a) <= 8 for p, a in ps), "an example longer than a row was cut"
    x, y = M.pack(ps, 40, TOK.eos_id)
    for row in y:
        ends = [i for i, v in enumerate(row.tolist()) if v == TOK.eos_id]
        firsts = [i for i in range(len(row)) if row[i] != -1 and (i == 0 or row[i - 1] == -1)]
        assert len(ends) == len(firsts), "a row holds part of an answer: an example straddles two rows"


# --- broken versions for --self-test ----------------------------------------------------

NAMES = ("pair_ids", "pack")


def MUTANTS(sol):
    def no_space(encode, eos, q, a, template=sol.PROMPT_EN):
        return encode(template.format(q=q)), encode(a) + [eos]

    def no_eos(encode, eos, q, a, template=sol.PROMPT_EN):
        return encode(template.format(q=q)), encode(" " + a)

    def prompt_in_loss(pairs, block_size, eos):
        return sol.pack([([], p + a) for p, a in pairs], block_size, eos)

    def unshifted(pairs, block_size, eos):
        x, _ = sol.pack(pairs, block_size, eos)
        L = block_size + 1
        tg, lab = [], []
        for p, a in pairs:
            ex = p + a
            if len(ex) > block_size:
                continue
            if len(lab) + len(ex) > L:
                tg.append(lab + [-1] * (L - len(lab)))
                lab = []
            lab = lab + [-1] * len(p) + list(a)
        if lab:
            tg.append(lab + [-1] * (L - len(lab)))
        return x, torch.tensor(tg)[:, :-1] if tg else x

    def straddles(pairs, block_size, eos):
        flat_x, flat_y = [], []
        for p, a in pairs:
            flat_x += p + a
            flat_y += [-1] * len(p) + list(a)
        L = block_size + 1
        rows = [flat_x[i:i + L] for i in range(0, len(flat_x), L)]
        labs = [flat_y[i:i + L] for i in range(0, len(flat_y), L)]
        rows = [r + [eos] * (L - len(r)) for r in rows]
        labs = [l + [-1] * (L - len(l)) for l in labs]
        x, y = torch.tensor(rows), torch.tensor(labs)
        return x[:, :-1], y[:, 1:]

    bugs = [("answer encoded without its leading space", dict(pair_ids=no_space)),
            ("answer without EOS", dict(pair_ids=no_eos)),
            ("prompt left in the loss", dict(pack=prompt_in_loss)),
            ("labels not shifted by one", dict(pack=unshifted)),
            ("examples cut across rows", dict(pack=straddles))]
    return [(d, with_replaced(sol, NAMES, **r)) for d, r in bugs]


if __name__ == "__main__":
    main(9, CHECKS, MUTANTS, __doc__, learner="post.py")
