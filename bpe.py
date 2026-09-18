"""
Byte-level BPE, from scratch, stdlib only.

Why this exists: the byte tokenizer costs 2.48 tokens per Devanagari character
(docs/RESULTS.md §2). Every one of those is context, KV cache and latency. This
is the nano version of Sarvam's actual moat — their 262,144-token, 22-language
tokenizer — and it is the single largest efficiency lever in this project.

    python bpe.py train  --data data/hindi.txt --vocab 16384 --out data/hi16k.json
    python bpe.py encode --tokenizer data/hi16k.json --data data/hindi_big.txt \
                         --out data/hindi_big.hi16k.bin
    python bpe.py stats  --tokenizer data/hi16k.json --data data/hindi.txt

Design, briefly:
  * Base alphabet is the 256 bytes, so ANY text encodes — unseen scripts just
    fall back toward bytes. Merges get ids 256, 257, ...
  * Pre-tokenization: text is split into `\\s*\\S+` chunks (leading whitespace
    attached to the word, GPT-2 style) plus pure-whitespace runs, so
    decode(encode(x)) == x exactly. Merges never cross chunk boundaries.
  * Training is the classic pair-merge loop, made tractable in pure Python by
    (a) operating on unique chunks weighted by frequency, not the raw corpus,
    (b) incremental pair-count updates via a pair -> chunk index, and
    (c) a lazy-deletion max-heap instead of a full scan per merge.
  * Encoding memoises per unique chunk — Zipf does the rest.
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import re
import time
from array import array
from collections import Counter, defaultdict
from pathlib import Path

CHUNK_RE = re.compile(r"\s*\S+|\s+")
# Document separator in corpus .txt files. fetch_hindi.py writes one article
# per document; `encode` turns each boundary into EOS so the model learns
# where documents end and `generate` has something to stop on. Three
# newlines is unambiguous: strip_wikitext collapses any run of 3+ inside an
# article down to 2. Older corpora without it just become one long document.
DOC_SEP = "\n\n\n"
MAX_TRAIN_CHUNK_BYTES = 48        # ignore absurd chunks (URLs, minified junk)
                                  # when *counting merges*; encoding still
                                  # handles any length.


class BPE:
    def __init__(self, merges: list[tuple[int, int]]):
        self.merges = merges
        # rank = priority at encode time = order the merge was learned
        self.ranks = {tuple(p): 256 + i for i, p in enumerate(merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        # EOS sits above the merges, so bytes and merges keep their ids and a
        # tokenizer file needs no new field. `train` reserves the slot so a
        # `--vocab 16384` tokenizer is 16,384 ids *including* EOS.
        self.eos_id = 256 + len(merges)
        self._cache: dict[bytes, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return self.eos_id + 1

    # ---------------- training ----------------

    @classmethod
    def train(cls, text: str, vocab_size: int, verbose: bool = True) -> "BPE":
        assert vocab_size > 256
        t0 = time.time()
        chunks = Counter(
            c.encode("utf-8") for c in CHUNK_RE.findall(text)
        )
        seqs, freqs = [], []
        for chunk, f in chunks.items():
            if len(chunk) <= MAX_TRAIN_CHUNK_BYTES and len(chunk) >= 2:
                seqs.append(list(chunk))
                freqs.append(f)
        if verbose:
            print(f"  {len(chunks)} unique chunks, {len(seqs)} used for training "
                  f"({time.time()-t0:.1f}s to count)")

        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        pair_where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for i, (seq, f) in enumerate(zip(seqs, freqs)):
            for p in zip(seq, seq[1:]):
                pair_counts[p] += f
                pair_where[p].add(i)

        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)
        merges: list[tuple[int, int]] = []

        while 256 + len(merges) < vocab_size - 1 and heap:      # -1: EOS slot
            neg, pair = heapq.heappop(heap)
            # lazy deletion: stale entries no longer match the live count
            if -neg != pair_counts.get(pair, 0):
                continue
            if -neg < 2:
                break                      # nothing left worth merging
            new_id = 256 + len(merges)
            merges.append(pair)
            a, b = pair

            touched: set[tuple[int, int]] = set()
            for i in list(pair_where[pair]):
                seq, f = seqs[i], freqs[i]
                # subtract this chunk's current pairs
                for p in zip(seq, seq[1:]):
                    pair_counts[p] -= f
                    pair_where[p].discard(i)
                    touched.add(p)
                # apply the merge
                out, j = [], 0
                while j < len(seq):
                    if j < len(seq) - 1 and seq[j] == a and seq[j + 1] == b:
                        out.append(new_id)
                        j += 2
                    else:
                        out.append(seq[j])
                        j += 1
                seqs[i] = out
                # add back the new pairs
                for p in zip(out, out[1:]):
                    pair_counts[p] += f
                    pair_where[p].add(i)
                    touched.add(p)
            for p in touched:
                c = pair_counts.get(p, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, p))
            if verbose and len(merges) % 2000 == 0:
                tok = cls(merges).vocab[new_id]
                print(f"  merge {len(merges):6d}/{vocab_size-257}  "
                      f"count {-neg:7d}  {tok!r}  ({time.time()-t0:.0f}s)")

        if verbose:
            print(f"  trained {len(merges)} merges in {time.time()-t0:.0f}s")
        return cls(merges)

    # ---------------- encode / decode ----------------

    def _encode_chunk(self, chunk: bytes) -> list[int]:
        hit = self._cache.get(chunk)
        if hit is not None:
            return hit
        seq = list(chunk)
        while len(seq) >= 2:
            best_rank, best_pair = min(
                (self.ranks.get(p, 1 << 30), p) for p in zip(seq, seq[1:])
            )
            if best_rank == 1 << 30:
                break
            a, b = best_pair
            out, j = [], 0
            while j < len(seq):
                if j < len(seq) - 1 and seq[j] == a and seq[j + 1] == b:
                    out.append(best_rank)
                    j += 2
                else:
                    out.append(seq[j])
                    j += 1
            seq = out
        if len(chunk) <= 64:               # don't let one giant blob evict RAM
            self._cache[chunk] = seq
        return seq

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for c in CHUNK_RE.findall(text):
            ids.extend(self._encode_chunk(c.encode("utf-8")))
        return ids

    def encode_documents(self, text: str) -> list[int]:
        """`encode`, plus EOS after every document. Documents are DOC_SEP-separated;
        a corpus without DOC_SEP is one document and gets a single EOS at the end."""
        ids: list[int] = []
        for doc in text.split(DOC_SEP):
            if doc.strip():
                ids.extend(self.encode(doc))
                ids.append(self.eos_id)
        return ids

    def decode(self, ids) -> str:
        # EOS (and anything else outside the vocab) decodes to nothing.
        return b"".join(self.vocab.get(int(i), b"") for i in ids).decode("utf-8", errors="replace")

    # ---------------- persistence ----------------

    # The format id the project wrote before it was renamed on 2026-09-18.
    # Every tokenizer trained until then carries it, including the four
    # uploaded to the Hub beside the released weights, so load() keeps
    # accepting it: the bytes after the type field are identical either way.
    TYPE = "anulm-bpe"
    TYPES = (TYPE, "nanosarvam-bpe")

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps({
            "type": self.TYPE, "version": 1,
            "merges": [list(p) for p in self.merges],
        }), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPE":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        assert d.get("type") in cls.TYPES, (
            f"{path}: not an AnuLM tokenizer (type {d.get('type')!r}, "
            f"expected one of {cls.TYPES})")
        return cls([tuple(p) for p in d["merges"]])


# ---------------- CLI ----------------

def cmd_train(args):
    text = Path(args.data).read_text(encoding="utf-8")
    print(f"training vocab {args.vocab} on {len(text.encode('utf-8'))/1e6:.1f} MB")
    tok = BPE.train(text, args.vocab)
    tok.save(args.out)
    print(f"saved {args.out}")
    _stats(tok, text[: 2_000_000])


def iter_documents(path, chunk_chars: int = 1 << 24):
    """DOC_SEP-separated documents from a file of any size, one at a time,
    without reading the whole file: the corpora this feeds are now measured
    in tens of GB. A file without DOC_SEP is one document."""
    buf = ""
    with open(path, encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(chunk_chars)
            if not chunk:
                break
            buf += chunk
            parts = buf.split(DOC_SEP)
            buf = parts.pop()
            for d in parts:
                if d.strip():
                    yield d
    if buf.strip():
        yield buf


_worker_tok: BPE | None = None


def _encode_init(tokenizer_path):
    global _worker_tok
    _worker_tok = BPE.load(tokenizer_path)


def _encode_batch(docs):
    """Worker: ids for a batch of documents, EOS after each, packed as int16."""
    ids = array("h")
    for d in docs:
        ids.extend(_worker_tok.encode(d))
        ids.append(_worker_tok.eos_id)
    return ids.tobytes(), sum(len(d.encode("utf-8")) for d in docs), len(docs)


def cmd_encode(args):
    """Stream the corpus through a pool of encoders into one int16 array.

    Encoding is pure Python and the memo cache makes it fast, but a 3B-token
    corpus as a Python list would be ~80 GB of ints; the array is 2 bytes a
    token. Workers each load the tokenizer once and encode 4 MB batches of
    whole documents, so document order and EOS placement are exactly what
    `encode_documents` produces on the same text."""
    import torch
    from multiprocessing import Pool
    tok = BPE.load(args.tokenizer)
    assert tok.vocab_size <= 32768, "int16 storage assumes vocab <= 32768"
    t0 = time.time()

    def batches():
        b, size = [], 0
        for d in iter_documents(args.data):
            b.append(d)
            size += len(d)
            if size >= 4_000_000:
                yield b
                b, size = [], 0
        if b:
            yield b

    out = array("h")
    n_bytes = n_docs = 0
    workers = max(1, min(args.workers, (os.cpu_count() or 2) - 2))
    with Pool(workers, initializer=_encode_init, initargs=(args.tokenizer,)) as pool:
        for i, (blob, nb, nd) in enumerate(pool.imap(_encode_batch, batches())):
            out.frombytes(blob)
            n_bytes += nb
            n_docs += nd
            if i % 25 == 0:
                print(f"\r  {n_bytes/1e6:.0f} MB -> {len(out)/1e6:.1f}M tokens  "
                      f"({workers} workers, {time.time()-t0:.0f}s)", end="", flush=True)
    assert len(out), "no documents encoded"
    ids = torch.frombuffer(out, dtype=torch.int16).clone()
    torch.save({
        "ids": ids,
        "vocab_size": tok.vocab_size,
        "eos_id": tok.eos_id,
        "tokenizer": str(args.tokenizer),
        "n_bytes": n_bytes,
        "n_docs": n_docs,
        "bytes_per_token": n_bytes / len(ids),
    }, args.out)
    print(f"\n{n_bytes/1e6:.1f} MB -> {len(ids)/1e6:.1f}M tokens in {n_docs} documents "
          f"({n_bytes/len(ids):.2f} bytes/token) in {time.time()-t0:.0f}s -> {args.out}")


def _stats(tok, text):
    ids = tok.encode(text)
    raw = text.encode("utf-8")
    words = len([c for c in CHUNK_RE.findall(text) if c.strip()])
    print(f"  bytes/token : {len(raw)/len(ids):5.2f}   (byte tokenizer: 1.00)")
    print(f"  tokens/word : {len(ids)/max(words,1):5.2f}   "
          f"(Sarvam reports 1.4-2.1 on Indic; bytes were ~15-20)")
    rt = tok.decode(ids)
    print(f"  exact roundtrip: {rt == text}")


def cmd_stats(args):
    tok = BPE.load(args.tokenizer)
    text = Path(args.data).read_text(encoding="utf-8")[: 5_000_000]
    print(f"{args.tokenizer}: vocab {tok.vocab_size}")
    _stats(tok, text)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--vocab", type=int, default=16384)
    t.add_argument("--out", required=True)
    e = sub.add_parser("encode")
    e.add_argument("--tokenizer", required=True)
    e.add_argument("--data", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--workers", type=int, default=16)
    s = sub.add_parser("stats")
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--data", required=True)
    args = p.parse_args()
    {"train": cmd_train, "encode": cmd_encode, "stats": cmd_stats}[args.cmd](args)
