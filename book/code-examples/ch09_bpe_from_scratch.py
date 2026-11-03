"""
Book example, not repository code.

Chapter 9: byte-level BPE the slow, obvious way, checked merge for merge and
token for token against bpe.py; then fertility, and the released tokenizer.json
checked against bpe.py on text committed at the pin.

    python book/code-examples/ch09_bpe_from_scratch.py

It imports bpe.py unchanged. The naive trainer recounts every pair after every
merge; the naive encoder applies the merges one by one in the order they were
learned. Both must agree exactly with bpe.py, whose job is to do the same thing
fast.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm import bpe  # noqa: E402  the repository's bpe.py, unchanged
from anulm.bpe import BPE, CHUNK_RE  # noqa: E402
from anulm.training.train import FALLBACK  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PIN = "084cbdd"


def merge(seq: list[int], a: int, b: int, new: int) -> list[int]:
    """Replace every non-overlapping (a, b), left to right, by `new`."""
    out, j = [], 0
    while j < len(seq):
        if j < len(seq) - 1 and seq[j] == a and seq[j + 1] == b:
            out.append(new)
            j += 2
        else:
            out.append(seq[j])
            j += 1
    return out


def naive_train(text: str, vocab_size: int, log: list | None = None) -> list[tuple[int, int]]:
    """Recount every pair of every chunk after every merge."""
    chunks = Counter(c.encode("utf-8") for c in CHUNK_RE.findall(text))
    words = [(list(c), f) for c, f in chunks.items()
             if 2 <= len(c) <= bpe.MAX_TRAIN_CHUNK_BYTES]
    merges = []
    while 256 + len(merges) < vocab_size - 1:              # one id kept for EOS
        counts = Counter()
        for seq, f in words:
            for p in zip(seq, seq[1:]):
                counts[p] += f
        if not counts:
            break
        best = min(counts, key=lambda p: (-counts[p], p))  # most frequent; ties: smallest pair
        if counts[best] < 2:
            break
        merges.append(best)
        if log is not None:
            log.append(counts[best])
        words = [(merge(seq, *best, 255 + len(merges)), f) for seq, f in words]
    return merges


def naive_encode(text: str, merges: list[tuple[int, int]]) -> list[int]:
    """Every merge, in the order learned, over each chunk."""
    ids = []
    for c in CHUNK_RE.findall(text):
        seq = list(c.encode("utf-8"))
        for i, (a, b) in enumerate(merges):
            seq = merge(seq, a, b, 256 + i)
        ids += seq
    return ids


def show(tok: BPE, ids: list[int]) -> list[str]:
    return [tok.vocab[i].decode("utf-8", errors="replace") for i in ids]


def at_pin(path: str) -> str:
    return subprocess.run(["git", "show", f"{PIN}:{path}"], cwd=ROOT, capture_output=True,
                          encoding="utf-8", check=True).stdout


def main() -> None:
    # 1. By hand: a ten-word corpus, five merges
    toy = "low lower lowest low slow slower newest low new lower"
    print("chunks:", CHUNK_RE.findall(toy)[:4], "...")
    tok = BPE.train(toy, 256 + 5 + 1, verbose=False)
    counts = []
    assert naive_train(toy, 262, counts) == tok.merges
    for i, (a, b) in enumerate(tok.merges):
        print(f"merge {i}: id {256 + i} = {tok.vocab[a]!r} + {tok.vocab[b]!r} -> {tok.vocab[256 + i]!r}"
              f"  (count {counts[i]})")
    print("encode(' slowest') ->", tok.encode(" slowest"), show(tok, tok.encode(" slowest")))
    print(f"vocab_size {tok.vocab_size}, eos_id {tok.eos_id}")

    # 2. The fast trainer against the naive one, on real text at the pin
    corpus = at_pin("docs/RESULTS.md")
    for n in (257 + 300, 257 + 1000):
        t0 = time.perf_counter()
        fast = BPE.train(corpus, n, verbose=False)
        t1 = time.perf_counter()
        slow = naive_train(corpus, n)
        t2 = time.perf_counter()
        assert fast.merges == slow
        print(f"{n - 257} merges on {len(corpus.encode()):,} bytes: same merges; "
              f"bpe.py {t1 - t0:.1f} s, naive {t2 - t1:.1f} s")

    # How often _encode_chunk's cache can answer: a chunk of <= 64 bytes seen before in the same text
    seen, hits, chunks = set(), 0, CHUNK_RE.findall(corpus)
    for c in chunks:
        k = c.encode("utf-8")
        hits += k in seen
        if len(k) <= 64:
            seen.add(k)
    print(f"cache: {hits / len(chunks):.0%} of {len(chunks):,} chunks in RESULTS.md repeat an earlier cachable chunk")

    # 3. The fast encoder against the naive one
    text = at_pin("README.md")[:20000]
    ids = fast.encode(text)
    assert ids == naive_encode(text, fast.merges)
    assert fast.decode(ids) == text
    print(f"encode: first 20,000 characters of README.md -> {len(ids):,} ids for {len(text.encode()):,} bytes, "
          f"same as the naive encoder; "
          f"decode(encode(x)) == x")

    # 4. The released tokenizer: bpe.py's multi32k against tokenizer.json
    multi = BPE.load(ROOT / "release" / "AnuLM-Base-v2-400M" / "tokenizer.multi32k.json")
    print(f"\nmulti32k: {len(multi.merges):,} merges, vocab_size {multi.vocab_size:,}, eos_id {multi.eos_id}")
    for s in ("The dog bit the man.", " भारत की राजधानी", "def f(x):\n    return x"):
        print(f"{s!r:26} -> {show(multi, multi.encode(s))}")
    # The first two of the three bytes of 'भ' (E0 A4 AD): byte ids are the bytes themselves.
    half = list("भ".encode("utf-8")[:2])
    assert multi.decode(half) == "�" and multi.decode_complete(half) == ""
    assert multi.decode_complete(half + [0xAD]) == "भ"
    assert multi.decode_complete(list("�".encode("utf-8"))) == "�"
    print(f"two of the three bytes of 'भ': decode -> {ascii(multi.decode(half))} (U+FFFD), "
          f"decode_complete -> {multi.decode_complete(half)!r}; "
          f"all three -> {multi.decode_complete(half + [0xAD])!r}")
    from tokenizers import Tokenizer
    fast_tok = Tokenizer.from_file(str(ROOT / "release" / "AnuLM-Base-v2-400M" / "tokenizer.json"))
    texts = [at_pin(p) for p in ("docs/RESULTS.md", "README.md", "anulm/model.py", "anulm/bpe.py", "docs/MODEL_CARD_BASE_V2.md")]
    n_tok = 0
    for t in texts:
        ours, theirs = multi.encode(t), fast_tok.encode(t, add_special_tokens=False).ids
        assert ours == theirs
        n_tok += len(ours)
    print(f"tokenizer.json == bpe.py on {len(texts)} files at the pin, "
          f"{sum(len(t.encode()) for t in texts):,} bytes, {n_tok:,} tokens")

    # 5. Fertility on those files, as bpe._stats measures it
    for name, t in zip(("RESULTS.md", "README.md", "model.py"), texts):
        ids = multi.encode(t)
        words = len([c for c in CHUNK_RE.findall(t) if c.strip()])
        print(f"{name:11} bytes/token {len(t.encode()) / len(ids):5.2f}   tokens/word {len(ids) / words:5.2f}")

    # 6. Exercise answers
    six = BPE.train(toy, 256 + 6 + 1, verbose=False)
    print(f"\nexercise 1: sixth merge {six.merges[5]} = {six.vocab[six.merges[5][0]] + six.vocab[six.merges[5][1]]!r}")
    print(f"exercise 3: 16,127 merges -> eos_id {BPE([(0, 0)] * 16127).eos_id}, "
          f"vocab_size {BPE([(0, 0)] * 16127).vocab_size}")
    print(f"exercise 4: {CHUNK_RE.findall('a  b' + chr(10))}")


if __name__ == "__main__":
    main()
