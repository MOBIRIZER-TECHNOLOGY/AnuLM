"""
Reference solution for Chapter 2: turning text on your disk into a training corpus.

split_documents, script_share, filter_doc and interleaved_split match the
repository's bpe.iter_documents, fetch_hindi.devanagari_ratio,
filter_prose.filter_doc and train.interleaved_split. word_ngrams agrees with
convert_parquet.ngrams on English and, unlike it, also sees Devanagari.
course/checks/ch02.py proves both.
"""

from __future__ import annotations

import hashlib
import re

DOC_SEP = "\n\n\n"                          # three newlines end a document (bpe.py)
SENTENCE_END = re.compile(r"[।.!?]['\")\]]*\s*$")
# The repository's word pattern, [a-z0-9_]+, plus the Devanagari block: the same words
# on English text, and Hindi words instead of nothing.
WORDS = re.compile(r"[a-z0-9_ऀ-ॿ]+")


def split_documents(text: str) -> list[str]:
    """A corpus file is documents separated by DOC_SEP; blank ones are dropped."""
    return [d for d in text.split(DOC_SEP) if d.strip()]


def exact_dedup(docs: list[str]) -> tuple[list[str], int]:
    """Keep the first copy of every document, in order. Returns (kept, how many removed)."""
    seen, kept = set(), []
    for d in docs:
        h = hashlib.sha1(d.encode("utf-8")).digest()
        if h not in seen:
            seen.add(h)
            kept.append(d)
    return kept, len(docs) - len(kept)


def script_share(text: str, lo: str = "ऀ", hi: str = "ॿ") -> float:
    """The share of characters inside one Unicode block (Devanagari by default)."""
    return sum(1 for ch in text if lo <= ch <= hi) / len(text) if text else 0.0


def filter_doc(doc: str, min_line_chars: int = 40, min_sentence_frac: float = 0.4, min_chars: int = 600):
    """Drop short lines that don't end a sentence (headings, list items, table cells); then keep the
    document only if enough of its lines are sentences and enough text remains. The title, the
    first paragraph, is kept as it is. Returns the cleaned document or None."""
    title, _, body = doc.partition("\n\n")
    kept = []
    for line in body.split("\n"):
        s = line.strip()
        if not s:
            kept.append("")
            continue
        if not SENTENCE_END.search(s) and len(s) < min_line_chars:
            continue
        kept.append(s)
    lines = [l for l in kept if l]
    if not lines:
        return None
    frac = sum(1 for l in lines if SENTENCE_END.search(l)) / len(lines)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    if frac < min_sentence_frac or len(text) < min_chars:
        return None
    return title.strip() + "\n\n" + text


def word_ngrams(text: str, n: int = 13) -> set[tuple[str, ...]]:
    """Every run of n consecutive words, lower-cased."""
    w = WORDS.findall(text.lower())
    return {tuple(w[i:i + n]) for i in range(max(len(w) - n + 1, 0))}


def contaminated(train_docs: list[str], bench_items: list[str], n: int = 13) -> list[int]:
    """Indices of benchmark items that share at least one n-word window with the training text."""
    seen = set()
    for d in train_docs:
        seen |= word_ngrams(d, n)
    return [i for i, item in enumerate(bench_items) if word_ngrams(item, n) & seen]


def interleaved_split(data, chunk: int, every: int = 10):
    """Hold out every `every`-th chunk (starting with chunk every // 2) instead of the tail, so the
    validation set has the whole corpus's mix. Works on strings, lists and tensors."""
    chunks = [data[i:i + chunk] for i in range(0, len(data), chunk)]
    val = [c for i, c in enumerate(chunks) if i % every == every // 2]
    train = [c for i, c in enumerate(chunks) if i % every != every // 2]
    join = (lambda cs: "".join(cs)) if isinstance(data, str) else \
           (lambda cs: sum(cs, [])) if isinstance(data, list) else \
           (lambda cs: __import__("torch").cat(cs))
    return join(train), join(val)
