"""
Chapter 3 lab: train your tokenizer on your own text and measure what it buys.

    python lab.py                      # your course/build/data.py
    python lab.py --impl solution      # the reference solution

About a minute on a CPU. Trains a 1,024-id tokenizer with your plain loop (and
the same with bpe.py, to time both) on Hindi Wikipedia and Shakespeare, then
measures, on held-out text in each language, how many bytes each token carries
and how much text fits in a 1,024-token window: for raw bytes, your tokenizer,
and the 32,768-id tokenizer the 400M AnuLM model uses (data/multi32k.json).
Writes results.json for the figures.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
COURSE = HERE.parent
ROOT = COURSE.parent
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=["build", "solution"], default="build")
    p.add_argument("--vocab", type=int, default=1024)
    a = p.parse_args()
    D = load(COURSE / "build" / ("solutions/ch03.py" if a.impl == "solution" else "data.py"))
    from anulm import bpe
    en = (ROOT / "data" / "input.txt").read_text(encoding="utf-8")
    hi = (ROOT / "data" / "bench_hindi.txt").read_text(encoding="utf-8")
    train_text = en[:120_000] + hi[:120_000]
    held = {"English": en[-40_000:], "Hindi": hi[-40_000:]}

    t = time.time()
    try:
        merges = [tuple(m) for m in D.train_bpe(train_text, a.vocab)]
    except NotImplementedError as e:
        sys.exit(f"course/build/data.py is not finished yet ({e}). Run course/checks/ch03.py, or use --impl solution.")
    t_plain = time.time() - t
    t = time.time()
    ref = bpe.BPE.train(train_text, a.vocab, verbose=False)
    t_fast = time.time() - t
    same = merges == [tuple(m) for m in ref.merges]
    print(f"trained {len(merges)} merges: your loop {t_plain:.1f}s, bpe.py {t_fast:.1f}s, identical merges: {same}")

    v = {i: bytes([i]) for i in range(256)}
    for i, (x, y) in enumerate(merges):
        v[256 + i] = v[x] + v[y]
    first = [v[256 + i].decode("utf-8", errors="replace") for i in range(12)]
    longest = sorted((v[i] for i in range(256, 256 + len(merges))), key=len, reverse=True)
    hindi_long = [t.decode("utf-8", errors="replace") for t in longest
                  if any(0x0900 <= ord(c) <= 0x097F for c in t.decode("utf-8", errors="ignore"))][:6]
    english_long = [t.decode("utf-8", errors="replace") for t in longest if t.isascii()][:6]
    print("first merges:", [f.replace(" ", "␣") for f in first])
    print("longest tokens:", english_long, hindi_long)

    # how real words are split by these merges, for Figure 3.1
    out_examples = {}
    for word in (" the", " Romeo", " भारत"):
        ids = D.encode(word, merges)
        out_examples[word] = {"bytes": len(word.encode("utf-8")),
                              "tokens": [v[i].decode("utf-8", errors="replace") for i in ids]}
    big = bpe.BPE.load(ROOT / "data" / "multi32k.json")
    out = {"vocab": a.vocab, "merges": len(merges), "seconds_plain": round(t_plain, 1), "seconds_bpe_py": round(t_fast, 1),
           "identical": same, "first": first, "longest_en": english_long, "longest_hi": hindi_long, "fertility": {},
           "examples": out_examples}
    for lang, text in held.items():
        nbytes, nchars = len(text.encode("utf-8")), len(text)
        row = {}
        for name, n_tokens in (("bytes", nbytes), (f"yours ({a.vocab})", len(D.encode(text, merges))),
                               ("AnuLM 400M (32,768)", len(big.encode(text)))):
            row[name] = {"bytes_per_token": round(nbytes / n_tokens, 3),
                         "chars_in_1024": round(1024 * nchars / n_tokens)}
        out["fertility"][lang] = row
        print(f"{lang}: " + "   ".join(f"{k}: {r['bytes_per_token']:.2f} bytes/token, {r['chars_in_1024']:,} chars "
                                       f"per 1,024 tokens" for k, r in row.items()))
    (HERE / "results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("saved results.json")


if __name__ == "__main__":
    main()
