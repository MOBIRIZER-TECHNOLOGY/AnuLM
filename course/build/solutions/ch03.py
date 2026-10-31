"""
Reference solution for Chapter 3: a byte-level BPE tokenizer.

train_bpe learns the same merges, in the same order, as bpe.py's BPE.train;
encode, decode and encode_documents give the same ids as BPE.encode,
BPE.decode and BPE.encode_documents. bpe.py is faster (incremental counts, a
heap); this is the plain loop, which is easier to read and gives the same answer.
"""

from __future__ import annotations

import re
from collections import Counter

CHUNK_RE = re.compile(r"\s*\S+|\s+")     # a word with the spaces before it, or a run of spaces
MAX_TRAIN_CHUNK_BYTES = 48                # longer chunks (URLs, junk) are not learned from
DOC_SEP = "\n\n\n"


def _merge(seq, pair, new_id):
    out, j = [], 0
    while j < len(seq):
        if j < len(seq) - 1 and seq[j] == pair[0] and seq[j + 1] == pair[1]:
            out.append(new_id)
            j += 2
        else:
            out.append(seq[j])
            j += 1
    return out


def train_bpe(text: str, vocab_size: int) -> list[tuple[int, int]]:
    """Merges, in the order learned. Ids 0-255 are bytes, merge i becomes id 256 + i, and one id
    is kept free at the top for end-of-document."""
    chunks = Counter(c.encode("utf-8") for c in CHUNK_RE.findall(text))
    words = [(list(c), f) for c, f in chunks.items() if 2 <= len(c) <= MAX_TRAIN_CHUNK_BYTES]
    merges = []
    while 256 + len(merges) < vocab_size - 1:
        counts = Counter()
        for seq, f in words:                                   # pairs never cross a chunk boundary
            for p in zip(seq, seq[1:]):
                counts[p] += f
        if not counts:
            break
        best = max(counts.values())
        if best < 2:
            break                                              # nothing left worth merging
        pair = min(p for p, c in counts.items() if c == best)  # ties: the smallest pair, as bpe.py's heap
        new_id = 256 + len(merges)
        merges.append(pair)
        words = [(_merge(seq, pair, new_id), f) for seq, f in words]
    return merges


def encode(text: str, merges) -> list[int]:
    """Each chunk: start from its bytes, then repeatedly apply the earliest-learned merge present."""
    ranks = {tuple(p): 256 + i for i, p in enumerate(merges)}
    ids = []
    for c in CHUNK_RE.findall(text):
        seq = list(c.encode("utf-8"))
        while len(seq) >= 2:
            rank, pair = min((ranks.get(p, 1 << 30), p) for p in zip(seq, seq[1:]))
            if rank == 1 << 30:
                break
            seq = _merge(seq, pair, rank)
        ids.extend(seq)
    return ids


def vocab(merges) -> dict[int, bytes]:
    v = {i: bytes([i]) for i in range(256)}
    for i, (a, b) in enumerate(merges):
        v[256 + i] = v[a] + v[b]
    return v


def decode(ids, merges) -> str:
    """Bytes back to text; ids outside the vocabulary (end-of-document) decode to nothing."""
    v = vocab(merges)
    return b"".join(v.get(int(i), b"") for i in ids).decode("utf-8", errors="replace")


def encode_documents(text: str, merges) -> list[int]:
    """encode, plus the end-of-document id (256 + len(merges)) after every document."""
    eos = 256 + len(merges)
    ids = []
    for doc in text.split(DOC_SEP):
        if doc.strip():
            ids += encode(doc, merges) + [eos]
    return ids
