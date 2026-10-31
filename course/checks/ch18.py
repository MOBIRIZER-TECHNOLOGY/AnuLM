"""
Chapter 18 checks: chat.

    python ch18.py                  # your course/build/further.py
    python ch18.py --impl solution  # the reference solution
    python ch18.py --self-test      # proves each check catches the bug it is for

A conversation has more places to go wrong than a question: the user's turns
graded, only the last reply graded, a reply with no end, a conversation cut in
two, and a held-out set that asks what training already answered.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import main, with_replaced  # noqa: E402
import finetune  # noqa: E402
from bpe import BPE  # noqa: E402
from make_qa import PROMPT_EN  # noqa: E402

CHECKS = []
TOK = BPE.train(("User: what is the capital of India?\nAssistant: New Delhi is the capital. "
                 "Question: who wrote it?\nAnswer: a poet wrote it. ") * 30, 380, verbose=False)
enc = lambda s: list(s.encode("utf-8"))
EOS = 257
TALK = [("what is 23 plus 45?", "68"), ("and plus 17?", "85"), ("thanks", "You are welcome.")]


def check(fn):
    CHECKS.append(fn)
    return fn


@check
def one_turn_is_finetune_py(M):
    for q, a in (("what is the capital of India?", "New Delhi."), ("who wrote it?", "A poet.")):
        ids, labels = M.conversation_ids(TOK.encode, TOK.eos_id, [(q, a)], PROMPT_EN)
        p, ans = finetune._pair_ids(TOK, {"question": q, "answer": a, "lang": "en"})
        assert ids == p + ans, "one turn's ids differ from finetune._pair_ids"
        assert labels == [-1] * len(p) + ans, "one turn's labels differ: the prompt -1, the answer itself"


@check
def every_reply_is_graded_and_nothing_else(M):
    ids, labels = M.conversation_ids(enc, EOS, TALK)
    want = [t for _, r in TALK for t in enc(" " + r) + [EOS]]
    assert [l for l in labels if l != -1] == want, "graded ids are not exactly every reply and its EOS"
    assert len(ids) == len(labels), "ids and labels have different lengths"
    assert [i for i, l in zip(ids, labels) if l != -1] == want, "a label is not the id at its position"


@check
def single_turns_pack_like_finetune_py(M):
    items = [{"question": q, "answer": a, "lang": "en"} for q, a in
             (("what is the capital of India?", "New Delhi."), ("2+2?", "4"), ("a long one " * 6 + "?", "yes " * 9))]
    pairs = [finetune._pair_ids(TOK, it) for it in items]
    convs = [(p + a, [-1] * len(p) + a) for p, a in pairs]
    for block in (24, 40, 64):
        x, y = M.pack_conversations(convs, block, TOK.eos_id)
        wx, wy = finetune.pack(pairs, block, TOK.eos_id)
        assert x.shape == wx.shape and torch.equal(x.long(), wx.long()) and torch.equal(y.long(), wy.long()), \
            f"block {block}: differs from finetune.pack"


@check
def a_conversation_is_never_cut(M):
    convs = [M.conversation_ids(enc, EOS, TALK[:k]) for k in (1, 2, 3, 2, 1)]
    x, y = M.pack_conversations(convs, 120, EOS)
    for row in y:
        replies = sum(1 for t in row.tolist() if t == EOS)
        starts = sum(1 for i, t in enumerate(row.tolist()) if t != -1 and (i == 0 or row[i - 1] == -1))
        assert replies == starts, "a row holds part of a reply: a conversation was cut between rows"
    assert sum(1 for t in y.reshape(-1).tolist() if t == EOS) == sum(len(c[1]) and c[1].count(EOS) for c in convs if len(c[0]) <= 120), \
        "replies went missing in packing"


def rows():
    out = [{"question": f"question {i}", "answer": f"answer {i}", "source": "dolly"} for i in range(200)]
    out += [{"question": f"  QUESTION   {i}", "answer": "a copy", "source": "oasst2"} for i in range(0, 200, 4)]
    out += [{"question": f"Who are you, version {i}?", "answer": "I am AnuLM.", "source": "identity"}
            for i in range(150)] * 2                    # many, so a split that allows them would pick some
    out += [{"question": "too long", "answer": "x" * 2000, "source": "dolly"},
            {"question": "", "answer": "no question", "source": "dolly"}]
    return out


@check
def the_held_out_set_asks_nothing_training_answered(M):
    train, held = M.split_heldout(rows(), 30, random.Random(0))
    norm = lambda q: " ".join(q.lower().split())
    tq = {norm(r["question"]) for r in train}
    assert len({norm(r["question"]) for r in held}) == 30, "not 30 distinct held-out questions"
    assert not any(norm(r["question"]) in tq for r in held), "a held-out question (or a copy of it) is in training"
    assert not any(r["question"].startswith("Who are you") for r in held), "identity rows were held out"
    assert not any(len(r["answer"]) > 1500 or not r["question"] for r in train + held), "bad rows were kept"


# --- broken versions for --self-test ----------------------------------------------------

NAMES = ("conversation_ids", "pack_conversations", "split_heldout", "CHAT")


def MUTANTS(sol):
    def users_graded(encode, eos, turns, template=sol.CHAT):
        ids, labels = sol.conversation_ids(encode, eos, turns, template)
        return ids, list(ids)

    def last_reply_only(encode, eos, turns, template=sol.CHAT):
        ids, labels = [], []
        for k, (u, r) in enumerate(turns):
            p, a = encode(template.format(q=u)), encode(" " + r) + [eos]
            ids += p + a
            labels += [-1] * len(p) + (a if k == len(turns) - 1 else [-1] * len(a))
        return ids, labels

    def no_eos_between(encode, eos, turns, template=sol.CHAT):
        ids, labels = [], []
        for k, (u, r) in enumerate(turns):
            p = encode(template.format(q=u))
            a = encode(" " + r) + ([eos] if k == len(turns) - 1 else [])
            ids += p + a
            labels += [-1] * len(p) + a
        return ids, labels

    def straddling(convs, block_size, eos):
        flat_x = [t for ids, _ in convs for t in ids]
        flat_y = [t for _, lab in convs for t in lab]
        L = block_size + 1
        xs = [flat_x[i:i + L] for i in range(0, len(flat_x), L)]
        ys = [flat_y[i:i + L] for i in range(0, len(flat_y), L)]
        x = torch.tensor([r + [eos] * (L - len(r)) for r in xs])
        y = torch.tensor([r + [-1] * (L - len(r)) for r in ys])
        return x[:, :-1], y[:, 1:]

    def make_chat_split(rows, n, rng, max_answer=1500):
        kept = [r for r in rows if r["question"] and r["answer"] and len(r["answer"]) <= max_answer]
        rng.shuffle(kept)
        held = [r for r in kept if r["source"] != "identity"][:n]
        ids = {id(r) for r in held}
        return [r for r in kept if id(r) not in ids], held

    def identity_too(rows, n, rng, max_answer=1500):
        return sol.split_heldout([dict(r, source="dolly") for r in rows], n, rng, max_answer)

    def no_filter(rows, n, rng, max_answer=1500):
        return sol.split_heldout(rows, n, rng, 10 ** 9)

    bugs = [("the user's turns graded", dict(conversation_ids=users_graded)),
            ("only the last reply graded", dict(conversation_ids=last_reply_only)),
            ("no EOS after each reply", dict(conversation_ids=no_eos_between)),
            ("conversations cut across rows", dict(pack_conversations=straddling)),
            ("make_chat.py's split: copies leak", dict(split_heldout=make_chat_split)),
            ("identity rows held out", dict(split_heldout=identity_too)),
            ("long answers kept", dict(split_heldout=no_filter))]
    return [(d, with_replaced(sol, NAMES, **r)) for d, r in bugs]


if __name__ == "__main__":
    main(18, CHECKS, MUTANTS, __doc__, learner="further.py")
