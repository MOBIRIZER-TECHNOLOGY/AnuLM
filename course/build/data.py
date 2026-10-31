"""
Your data pipeline (Chapter 2) and tokenizer (Chapter 3).

    python ../checks/ch02.py                   # check what you have built
    python ../checks/ch02.py --impl solution   # see the checks pass on the reference

Replace each `raise NotImplementedError`. The reference is in solutions/; the
repository's own versions are named in each docstring.
"""

from __future__ import annotations

import hashlib  # noqa: F401  (exact_dedup)
import re

DOC_SEP = "\n\n\n"                                   # three newlines end a document (bpe.py)
SENTENCE_END = re.compile(r"[।.!?]['\")\]]*\s*$")   # a line that ends a sentence (. ! ? or ।)


# --- Chapter 2: your own data ------------------------------------------------------------

def split_documents(text: str) -> list[str]:
    """Split on DOC_SEP and drop documents that are only whitespace (bpe.iter_documents)."""
    raise NotImplementedError("Chapter 2, section 2.2")


def exact_dedup(docs: list[str]) -> tuple[list[str], int]:
    """Keep the FIRST copy of each document, in the original order, comparing sha1 hashes of the
    UTF-8 text. Return (kept documents, number removed)."""
    raise NotImplementedError("Chapter 2, section 2.3")


def script_share(text: str, lo: str = "ऀ", hi: str = "ॿ") -> float:
    """The fraction of characters with lo <= ch <= hi (Devanagari by default); 0.0 for ''
    (fetch_hindi.devanagari_ratio)."""
    raise NotImplementedError("Chapter 2, section 2.3")


def filter_doc(doc: str, min_line_chars: int = 40, min_sentence_frac: float = 0.4, min_chars: int = 600):
    """filter_prose.filter_doc: the title is everything before the first blank line and is kept
    as is. In the body, drop each non-empty line that does not end a sentence (SENTENCE_END) and
    is shorter than min_line_chars; keep empty lines. Return None if no non-empty lines remain,
    if fewer than min_sentence_frac of them end a sentence, or if the remaining text (runs of 3+
    newlines collapsed to 2, stripped) is shorter than min_chars. Otherwise title + "\\n\\n" + text.
    """
    raise NotImplementedError("Chapter 2, section 2.3")


def word_ngrams(text: str, n: int = 13) -> set:
    """Every run of n consecutive words of text.lower(), as tuples. A word is a run of
    [a-z0-9_] or Devanagari (U+0900-U+097F) characters."""
    raise NotImplementedError("Chapter 2, section 2.5")


def contaminated(train_docs: list[str], bench_items: list[str], n: int = 13) -> list[int]:
    """Indices (in order) of benchmark items sharing at least one n-word window with any training
    document."""
    raise NotImplementedError("Chapter 2, section 2.5")


def interleaved_split(data, chunk: int, every: int = 10):
    """Cut data (a str, list or tensor) into chunks of `chunk`; chunk i goes to validation when
    i % every == every // 2, to training otherwise. Return (train, val), each joined back into the
    input's type (train.interleaved_split)."""
    raise NotImplementedError("Chapter 2, section 2.4")


# --- Chapter 3: the tokenizer ------------------------------------------------------------

CHUNK_RE = re.compile(r"\s*\S+|\s+")     # a word with the spaces before it, or a run of spaces
MAX_TRAIN_CHUNK_BYTES = 48                # chunks longer than this are not learned from


def train_bpe(text: str, vocab_size: int) -> list[tuple[int, int]]:
    """Byte-level BPE, the plain loop (bpe.BPE.train gives the same merges, faster):

      1. split text with CHUNK_RE; count each chunk's UTF-8 bytes (collections.Counter);
         learn only from chunks of 2 to MAX_TRAIN_CHUNK_BYTES bytes, weighted by their count;
      2. while 256 + len(merges) < vocab_size - 1   (one id stays free for end-of-document):
           count every adjacent pair inside every chunk (weighted); stop if none, or if the best
           count is below 2; take the most frequent pair, breaking ties by the SMALLEST pair;
           give it id 256 + len(merges) and replace it everywhere.
    Return the merges as a list of (a, b) tuples, in the order learned.
    """
    raise NotImplementedError("Chapter 3, section 3.2")


def encode(text: str, merges) -> list[int]:
    """For each CHUNK_RE chunk: start from its bytes and repeatedly apply the merge with the
    lowest id (the earliest learned) among the adjacent pairs present, until none applies."""
    raise NotImplementedError("Chapter 3, section 3.3")


def decode(ids, merges) -> str:
    """Join each id's bytes (bytes 0-255, merge i = the bytes of its two parts) and decode
    UTF-8 with errors="replace". Ids with no bytes (end-of-document) decode to nothing."""
    raise NotImplementedError("Chapter 3, section 3.3")


def encode_documents(text: str, merges) -> list[int]:
    """encode each non-blank DOC_SEP document and put the end-of-document id,
    256 + len(merges), after each one."""
    raise NotImplementedError("Chapter 3, section 3.4")
