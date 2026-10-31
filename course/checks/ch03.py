"""
Chapter 3 checks: the tokenizer.

    python ch03.py                  # your course/build/data.py
    python ch03.py --impl solution  # the reference solution
    python ch03.py --self-test      # proves each check catches the bug it is for

A tokenizer bug is permanent: every token the model ever sees passes through it,
and a checkpoint is only usable with the exact tokenizer it was trained on.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, main, with_replaced  # noqa: E402
import bpe  # noqa: E402

CHECKS = []
TEXT = ((ROOT / "data" / "input.txt").read_text(encoding="utf-8")[:30000]
        + (ROOT / "data" / "bench_hindi.txt").read_text(encoding="utf-8")[:30000])
TRICKY = "नमस्ते दुनिया!  Hello,   world.\n\n\tதமிழ் 🙂 end  \n"


def check(fn):
    CHECKS.append(fn)
    return fn


_ref = {}


def reference(vocab=400):
    if vocab not in _ref:
        _ref[vocab] = bpe.BPE.train(TEXT, vocab, verbose=False)
    return _ref[vocab]


@check
def merges_match_bpe_py(M):
    """Same text, same vocabulary size: the same merges in the same order as bpe.py."""
    mine = [tuple(p) for p in M.train_bpe(TEXT, 400)]
    want = [tuple(p) for p in reference().merges]
    assert len(mine) == len(want), f"{len(mine)} merges, bpe.py learns {len(want)} (one id is kept for EOS)"
    first = next((i for i, (a, b) in enumerate(zip(mine, want)) if a != b), None)
    assert first is None, f"merge {first} is {mine[first]}, bpe.py's is {want[first]}"


@check
def ties_go_to_the_smallest_pair(M):
    """'ab' and 'cd' appear equally often: the first merge must be (97, 98), as bpe.py's heap picks."""
    merges = M.train_bpe("ab cd " * 10, 258)
    want = bpe.BPE.train("ab cd " * 10, 258, verbose=False).merges
    assert [tuple(p) for p in merges] == [tuple(p) for p in want], f"{merges} != bpe.py's {want}"


@check
def merges_stop_when_nothing_repeats(M):
    """A pair seen once is not worth an id. bpe.py stops when the best count is below 2."""
    text = "abcdefgh"
    mine = M.train_bpe(text, 300)
    want = bpe.BPE.train(text, 300, verbose=False).merges
    assert len(mine) == len(want), f"{len(mine)} merges from text with no repeated pair; bpe.py learns {len(want)}"


@check
def encode_matches_bpe_py(M):
    tok = reference()
    merges = [tuple(p) for p in tok.merges]
    for s in (TRICKY, TEXT[1000:3000], TEXT[-3000:]):
        assert M.encode(s, merges) == tok.encode(s), "ids differ from bpe.py's encode"


@check
def decode_inverts_encode_exactly(M):
    """decode(encode(x)) == x for any text: Hindi, Tamil (never trained on), emoji, runs of spaces."""
    merges = [tuple(p) for p in reference().merges]
    assert M.decode(M.encode(TRICKY, merges), merges) == TRICKY, "decode(encode(x)) is not x"


@check
def merges_never_cross_chunks(M):
    """Chunks are a word plus the spaces before it; no merge may join the end of one word to the
    start of the next ('a b' must not become one token)."""
    merges = [tuple(p) for p in M.train_bpe("a ba ba ba ba b" * 5, 270)]
    v = {i: bytes([i]) for i in range(256)}
    for i, (a, b) in enumerate(merges):
        v[256 + i] = v[a] + v[b]
        assert not v[256 + i].strip(b" ").count(b" "), f"merge {v[256 + i]!r} spans two words"


@check
def documents_end_with_eos(M):
    """One end-of-document id after each document, with id 256 + len(merges); none inside."""
    tok = reference()
    merges = [tuple(p) for p in tok.merges]
    text = "first doc\nwith lines\n\n\nsecond\n\n\n  \n\n\nthird"
    assert M.encode_documents(text, merges) == tok.encode_documents(text), "differs from bpe.py's encode_documents"


# --- broken versions for --self-test ----------------------------------------------------

NAMES = ("train_bpe", "encode", "decode", "encode_documents")


def MUTANTS(sol):
    import re
    from collections import Counter

    def trainer(**bug):
        def train(text, vocab_size):
            pattern = re.compile(r"\S+|\s+") if bug.get("bare_words") else sol.CHUNK_RE
            if bug.get("whole_text"):
                words = [(list(text.encode("utf-8")), 1)]
            else:
                chunks = Counter(c.encode("utf-8") for c in pattern.findall(text))
                words = [(list(c), f) for c, f in chunks.items() if 2 <= len(c) <= sol.MAX_TRAIN_CHUNK_BYTES]
            merges = []
            limit = vocab_size if bug.get("no_eos_slot") else vocab_size - 1
            while 256 + len(merges) < limit:
                counts = Counter()
                for seq, f in words:
                    for p in zip(seq, seq[1:]):
                        counts[p] += f
                if not counts:
                    break
                best = max(counts.values())
                if best < 2 and not bug.get("no_stop"):
                    break
                pair = max(p for p, c in counts.items() if c == best) if bug.get("largest_tie") else \
                    min(p for p, c in counts.items() if c == best)
                merges.append(pair)
                words = [(sol._merge(seq, pair, 256 + len(merges) - 1), f) for seq, f in words]
            return merges
        return train

    def greedy_encode(text, merges):
        ranks = {tuple(p): 256 + i for i, p in enumerate(merges)}
        ids = []
        for c in sol.CHUNK_RE.findall(text):
            seq = list(c.encode("utf-8"))
            changed = True
            while changed:                               # first mergeable pair from the left, not the earliest learned
                changed = False
                for j in range(len(seq) - 1):
                    r = ranks.get((seq[j], seq[j + 1]))
                    if r is not None:
                        seq = seq[:j] + [r] + seq[j + 2:]
                        changed = True
                        break
            ids.extend(seq)
        return ids

    def lossy_decode(ids, merges):
        v = sol.vocab(merges)
        return b"".join(v.get(int(i), b"") for i in ids).decode("utf-8", errors="ignore").strip()

    def eos_per_line(text, merges):
        eos = 256 + len(merges)
        ids = []
        for line in text.split("\n"):
            if line.strip():
                ids += sol.encode(line, merges) + [eos]
        return ids

    bugs = [("ties broken by the largest pair", dict(train_bpe=trainer(largest_tie=True))),
            ("words without their leading spaces", dict(train_bpe=trainer(bare_words=True))),
            ("merges learned across the whole text", dict(train_bpe=trainer(whole_text=True))),
            ("no id kept for end-of-document", dict(train_bpe=trainer(no_eos_slot=True))),
            ("merging pairs seen only once", dict(train_bpe=trainer(no_stop=True))),
            ("encode merging left to right, not by rank", dict(encode=greedy_encode)),
            ("decode dropping bad bytes and spaces", dict(decode=lossy_decode)),
            ("end-of-document after every line", dict(encode_documents=eos_per_line))]
    return [(d, with_replaced(sol, NAMES, **r)) for d, r in bugs]


if __name__ == "__main__":
    main(3, CHECKS, MUTANTS, __doc__, learner="data.py")
