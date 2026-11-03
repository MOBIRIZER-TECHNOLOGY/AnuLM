"""
Book example, not repository code.

Chapter 8: text as bytes. UTF-8 written out by hand and checked against
Python's encoder; train.load_bytes re-implemented in a few lines and checked
against the repository's own; what a cut through a multi-byte character
looks like; and the bits-per-byte arithmetic train.py prints.

    python book/code-examples/ch08_bytes.py
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm.bpe import DOC_SEP  # noqa: E402  the repository's code, unchanged
from anulm.training.train import BYTE_EOS, FALLBACK, load_bytes  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def utf8_encode(cp: int) -> bytes:
    """One code point to UTF-8, by the bit patterns of the standard."""
    if cp < 0x80:                                   # 0xxxxxxx
        return bytes([cp])
    if cp < 0x800:                                  # 110xxxxx 10xxxxxx
        return bytes([0xC0 | cp >> 6, 0x80 | cp & 0x3F])
    if cp < 0x10000:                                # 1110xxxx 10xxxxxx 10xxxxxx
        return bytes([0xE0 | cp >> 12, 0x80 | cp >> 6 & 0x3F, 0x80 | cp & 0x3F])
    return bytes([0xF0 | cp >> 18, 0x80 | cp >> 12 & 0x3F,      # 11110xxx + three
                  0x80 | cp >> 6 & 0x3F, 0x80 | cp & 0x3F])


def my_load_bytes(raw: bytes) -> list[int]:
    """train.load_bytes, for a file that exists, as a plain list of ids."""
    raw = raw.replace(b"\r\n", b"\n")
    sep = DOC_SEP.encode("utf-8")
    if sep not in raw:
        return list(raw)                            # one document, no EOS
    ids = []
    for doc in raw.split(sep):
        if doc.strip():
            ids += list(doc) + [BYTE_EOS]
    return ids


def main() -> None:
    # 1. UTF-8 by hand, checked on every character of the fallback corpus
    for ch in "aéभ€😀":
        mine = utf8_encode(ord(ch))
        bits = " ".join(f"{b:08b}" for b in mine)
        print(f"U+{ord(ch):04X} {ch!r:5} -> {mine.hex(' ')}  {bits}")
        assert mine == ch.encode("utf-8")
    assert all(utf8_encode(ord(c)) == c.encode("utf-8") for c in FALLBACK)
    every = [cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]   # surrogates are not characters
    assert all(utf8_encode(cp) == chr(cp).encode("utf-8") for cp in every)
    print(f"utf8_encode matches str.encode on all {len(every):,} code points from U+0000 to U+10FFFF")
    dev = {chr(cp).encode("utf-8")[:2].hex(" ") for cp in range(0x0900, 0x0980)}
    print(f"every Devanagari code point (U+0900-U+097F) starts with one of: {sorted(dev)}")

    # 2. Bytes per character, line by line
    print("\nline of FALLBACK                     chars  bytes  bytes/char  continuation bytes")
    for line in FALLBACK.strip().split("\n"):
        b = line.encode("utf-8")
        cont = sum(1 for x in b if x >> 6 == 0b10)  # 10xxxxxx: never starts a character
        print(f"{line[:34]:36} {len(line):5}  {len(b):5}  {len(b) / len(line):10.2f}  {cont / len(b):17.0%}")

    # 3. load_bytes against the re-implementation, on a CRLF file with two documents
    docs = ["शीर्षक\n\nपहला दस्तावेज़।", "Title\n\nThe dog bit."]
    raw = (DOC_SEP.join(docs) + DOC_SEP).replace("\n", "\r\n").encode("utf-8")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "corpus.txt")
        Path(p).write_bytes(raw)
        ids = load_bytes(p)
        Path(p).write_bytes("no separator here".encode("utf-8"))
        plain = load_bytes(p)
    assert ids.tolist() == my_load_bytes(raw)
    n_cr, n_sep = raw.count(b"\r"), raw.replace(b"\r\n", b"\n").count(DOC_SEP.encode())
    print(f"\nthe file holds {n_cr} CR bytes and {n_sep} separators of {len(DOC_SEP)} bytes each")
    eos_at = (ids == BYTE_EOS).nonzero().flatten().tolist()
    print(f"\nload_bytes: {len(raw)} bytes on disk -> {len(ids)} ids, dtype {ids.dtype}, "
          f"EOS ({BYTE_EOS}) at {eos_at}")
    print(f"without DOC_SEP: dtype {plain.dtype}, {len(plain)} ids, EOS count {int((plain == BYTE_EOS).sum())}")
    print("my_load_bytes agrees id for id")
    wide = torch.tensor([BYTE_EOS], dtype=torch.int16)
    print(f"EOS cast to uint8: {wide.to(torch.uint8).item()} (silent); ", end="")
    try:
        torch.tensor([BYTE_EOS], dtype=torch.uint8)
    except RuntimeError as e:
        print(f"built as uint8: RuntimeError ({e})")

    # 4. A window that starts or ends inside a character
    b = "भारत".encode("utf-8")
    print(f"\n'भारत' is {len(b)} bytes: {b.hex(' ')}")
    for cut in (1, 4, 11):
        print(f"first {cut:2d} bytes decode to {b[:cut].decode('utf-8', errors='replace')!r}")
    out = [*b[:7], 257]                             # generated ids: 2 characters, a cut one, EOS
    try:
        bytes(out)
    except ValueError as e:
        print(f"bytes(ids) with EOS in it: ValueError ({e})")
    print("sample.py's rule, drop ids >= 256 then decode:",
          repr(bytes(i for i in out if i < 256).decode("utf-8", errors="replace")))

    # 5. bits per byte, as train.py line 740 computes it, and per character
    val_nats = 0.6734                               # docs/RESULTS.md §2, GQA 2000, nats per byte
    bpb = val_nats / math.log(2) / 1.0              # bytes_per_token is 1.0 for bytes
    print(f"\nval loss {val_nats} nats/byte -> {bpb:.3f} bits/byte -> {bpb * 2.48:.2f} bits/char at 2.48 bytes/char")
    print(f"English (§1, tinyshakespeare): 2.192 bits/byte x 1.00 bytes/char = {2.192 * 1.00:.2f} bits/char")
    print(f"a 1,024-byte window holds about {1024 / 2.48:.0f} Hindi characters, or 1,024 English ones")

    # 6. Exercise answers
    print(f"\nexercise 1: U+0915 -> {utf8_encode(0x0915).hex(' ')}")
    print(f"exercise 3: 1.50 bits/byte = {1.50 * math.log(2):.3f} nats/byte")


if __name__ == "__main__":
    main()
