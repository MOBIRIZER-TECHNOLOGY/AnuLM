"""Convert an AnuLM tokenizer (bpe.py's JSON) into a `tokenizers` tokenizer.json.

    python tools_hf_tokenizer.py data/code32k.json release/AnuLM-Coder-400M

Why: `bpe.py` uses its own format, so loading a released checkpoint means
cloning this repository. A tokenizer.json makes `AutoTokenizer.from_pretrained`
work with no custom code at all, which is the last piece of the export that
needed this project's source (TODO item 10).

The conversion must be *exact*, not merely close: a tokenizer that disagrees
on one merge produces plausible text and quietly different numbers. So this
script verifies token-for-token equality on real text in all three scripts
before it writes anything, and refuses to write a file that disagrees.

What has to line up:

  * **Pre-tokenization.** `bpe.py` splits on `\\s*\\S+|\\s+` -- a word with its
    leading whitespace, or a run of whitespace -- and merges never cross a
    chunk boundary. The same regex goes in as a `Split` pre-tokenizer.
  * **The byte alphabet.** Ours is the 256 raw bytes. `tokenizers` wants
    printable strings, so bytes are mapped through GPT-2's bytes-to-unicode
    table; `ByteLevel` maps them back on decode. The mapping is a bijection,
    so no information moves.
  * **Merge order.** Our rank is the order a merge was learned, which is
    exactly the priority `tokenizers` gives a merges list.
  * **EOS.** Ours sits one past the last merge, as `<|endoftext|>`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from bpe import BPE, CHUNK_RE

# The sentence that has to survive the round trip, in every script the project
# trains on, plus the shapes that break naive tokenizers: repeated whitespace,
# a tab, a trailing newline, punctuation runs, digits, and a word split across
# scripts.
PROBES = [
    "भारत की राजधानी नई दिल्ली है। यह एक बड़ा शहर है।",
    "हिन्दी साहित्य में प्रेमचंद का स्थान बहुत ऊँचा है।",
    "The quick brown fox jumps over the lazy dog.",
    "def is_prime(n):\n    if n < 2:\n        return False\n    return all(n % i for i in range(2, n))\n",
    "Question: What does len() return?\nAnswer: The number of items.",
    "English: Where is the hospital?\nHindi: अस्पताल कहाँ है?",
    "  leading and   internal   spaces\tand a tab\n\n\nand three newlines",
    "punctuation!!! ... ???  numbers 1234567890 and 3.14159",
    "mixed देवनागरी and ASCII in one line, plus emoji-free bytes: \x00\x01\xff",
    "",
    " ",
    "\n",
    "a",
]


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte -> printable-character map, as `tokenizers` uses."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) \
        + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


B2U = bytes_to_unicode()

TOKENIZER_CONFIG = {
    "tokenizer_class": "PreTrainedTokenizerFast",
    "model_max_length": 512,          # every checkpoint here is trained at 512
    "eos_token": "<|endoftext|>",
    "bos_token": None,
    "unk_token": None,                # a byte-level alphabet cannot miss
    "pad_token": "<|endoftext|>",
    "clean_up_tokenization_spaces": False,
}


def as_token(raw: bytes) -> str:
    return "".join(B2U[b] for b in raw)


def build(tok: BPE) -> dict:
    """The tokenizer.json structure, filled from our merges."""
    vocab = {as_token(tok.vocab[i]): i for i in range(256 + len(tok.merges))}
    assert len(vocab) == 256 + len(tok.merges), "byte->unicode map is not injective"
    merges = [[as_token(tok.vocab[a]), as_token(tok.vocab[b])] for a, b in tok.merges]
    eos = "<|endoftext|>"
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [{
            "id": tok.eos_id, "content": eos, "single_word": False, "lstrip": False,
            "rstrip": False, "normalized": False, "special": True,
        }],
        "normalizer": None,
        # Two stages, in this order: cut the text into bpe.py's chunks, then map
        # each chunk's bytes into the printable alphabet WITHOUT splitting again
        # (use_regex False -- ByteLevel's own GPT-2 regex would re-cut the text
        # differently and is what makes naive conversions disagree).
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split",
                 "pattern": {"Regex": r"\s*\S+|\s+"},
                 "behavior": "Isolated",
                 "invert": False},
                {"type": "ByteLevel", "add_prefix_space": False,
                 "trim_offsets": True, "use_regex": False},
            ],
        },
        "post_processor": {"type": "ByteLevel", "add_prefix_space": False,
                           "trim_offsets": True, "use_regex": False},
        "decoder": {"type": "ByteLevel", "add_prefix_space": False,
                    "trim_offsets": True, "use_regex": False},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": merges,
        },
    }


def real_text() -> list[str]:
    """Text that is always in a clone, so the check reproduces anywhere.

    Thirteen hand-written probes cannot exercise a 32,768-merge vocabulary.
    These can: the Hindi golden set is real Devanagari prose, this
    repository's own sources are real Python, and its documents are real
    English with tables and code fences in them. Any corpus left in data/
    joins in when there is one.
    """
    out: list[str] = []
    here = Path(__file__).parent
    g = here / "golden" / "golden_hindi.jsonl"
    if g.exists():
        for line in g.read_text(encoding="utf-8").splitlines()[:400]:
            item = json.loads(line)
            out.append(item.get("prefix", "") + " " + item.get("answer", ""))
    for name in ("model.py", "train.py", "bpe.py", "finetune.py"):
        f = here / name
        if f.exists():
            body = f.read_text(encoding="utf-8")
            out += [body[i:i + 3000] for i in range(0, len(body), 3000)]
    for name in ("README.md", "docs/RESULTS.md"):
        f = here / name
        if f.exists():
            body = f.read_text(encoding="utf-8")
            out += [body[i:i + 3000] for i in range(0, len(body), 3000)]
    for name in ("data/input.txt", "data/hindi.txt", "data/mixed.txt", "data/coder.txt"):
        f = here / name
        if f.exists():
            body = f.read_text(encoding="utf-8", errors="replace")[:300_000]
            out += [body[i:i + 3000] for i in range(0, len(body), 3000)]
    return out


def verify(tok: BPE, path: Path, extra: list[str]) -> None:
    """Refuse to ship a tokenizer that disagrees with bpe.py on any probe."""
    from tokenizers import Tokenizer
    fast = Tokenizer.from_file(str(path))
    bad = 0
    for text in PROBES + extra:
        ours = tok.encode(text)
        theirs = fast.encode(text, add_special_tokens=False).ids
        if ours != theirs:
            bad += 1
            print(f"  MISMATCH on {text[:44]!r}")
            print(f"    bpe.py     {ours[:16]}")
            print(f"    tokenizers {theirs[:16]}")
            for i, (a, b) in enumerate(zip(ours, theirs)):
                if a != b:
                    print(f"    first difference at {i}: {a} ({tok.vocab.get(a)!r}) "
                          f"vs {b} ({tok.vocab.get(b)!r})")
                    break
        elif tok.decode(ours) != text:
            # bpe.py is lossless by construction; check we did not break that
            bad += 1
            print(f"  ROUND TRIP FAILED on {text[:44]!r}")
    if bad:
        path.unlink(missing_ok=True)
        raise SystemExit(f"\n{bad} mismatch(es): the converted tokenizer is NOT "
                         f"equivalent, so it was not written.")
    print(f"  verified token-for-token on {len(PROBES) + len(extra)} probes")


def main_for(src: Path, outdir: Path) -> None:
    """Convert `src` into `outdir`, verify, and raise rather than ship a
    tokenizer that disagrees. This is what export_hf.py calls."""
    tok = BPE.load(src)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "tokenizer.json"
    out.write_text(json.dumps(build(tok), ensure_ascii=False), encoding="utf-8")
    verify(tok, out, real_text())
    (outdir / "tokenizer_config.json").write_text(json.dumps(TOKENIZER_CONFIG, indent=2),
                                                  encoding="utf-8")


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    src, outdir = Path(sys.argv[1]), Path(sys.argv[2])
    tok = BPE.load(src)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "tokenizer.json"

    print(f"{src} -> {out}")
    print(f"  {len(tok.merges)} merges, vocab {tok.vocab_size}, eos {tok.eos_id}")
    out.write_text(json.dumps(build(tok), ensure_ascii=False), encoding="utf-8")

    verify(tok, out, real_text())

    (outdir / "tokenizer_config.json").write_text(json.dumps(TOKENIZER_CONFIG, indent=2),
                                                  encoding="utf-8")
    print(f"  wrote {outdir / 'tokenizer_config.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
