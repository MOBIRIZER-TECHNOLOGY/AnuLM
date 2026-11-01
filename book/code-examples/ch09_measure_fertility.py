"""
Book example, not repository code.

Chapter 9: what the released tokenizer (multi32k) does on held-out text.
Draws whole documents from the base run's held-out split, data/v2/val.u16,
decodes them with bpe.py, and records statistics only: no text is printed or
saved. Also re-encodes each decoded document with bpe.py and compares the ids
with the stored ones, which were written by the Rust `tokenizers` build
(experiments/build_v2.py).

    python book/code-examples/ch09_measure_fertility.py   # -> ./part_iii_fertility.json
    python book/code-examples/ch09_measure_fertility.py --save   # -> book/measurements/part_iii_fertility.json
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm.tokenization.bpe import BPE, CHUNK_RE  # noqa: E402  the repository's bpe.py, unchanged

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# --save writes the book's copy; otherwise the result goes to the current directory.
OUT = (ROOT / "book" / "measurements" if "--save" in sys.argv else Path.cwd()) / "part_iii_fertility.json"
N_DOCS, SEED, MAX_TOKENS = 3000, 0, 8192


def script_of(text: str) -> str:
    dev = sum(1 for c in text if "ऀ" <= c <= "ॿ")
    lat = sum(1 for c in text if c.isascii() and c.isalpha())
    return "devanagari" if dev > lat else "latin"


def vocab_census(tok: BPE) -> dict:
    """What the 32,511 merged tokens are, by their bytes alone."""
    kinds = Counter()
    lengths, dev_lengths = Counter(), Counter()
    for i in range(256, tok.eos_id):
        b = tok.vocab[i]
        lengths[min(len(b), 16)] += 1
        try:
            s = b.decode("utf-8")
        except UnicodeDecodeError:
            kinds["not valid UTF-8 on its own"] += 1
            continue
        if any("ऀ" <= c <= "ॿ" for c in s):
            kinds["Devanagari"] += 1
            dev_lengths[min(len(b), 16)] += 1
            core = s.lstrip(" ")
            if all("ऀ" <= c <= "ॿ" for c in core):
                kinds["Devanagari only, after leading spaces"] += 1
        elif s.isascii():
            kinds["ASCII"] += 1
        else:
            kinds["other valid UTF-8"] += 1
    return {"kinds": dict(kinds), "byte_length": {str(k): v for k, v in sorted(lengths.items())},
            "devanagari_byte_length": {str(k): v for k, v in sorted(dev_lengths.items())}}


def main() -> None:
    tok = BPE.load(ROOT / "data" / "multi32k.json")
    meta = json.loads((ROOT / "data" / "v2" / "meta.json").read_text(encoding="utf-8"))
    assert meta["tokenizer"] == "data/multi32k.json" and meta["eos_id"] == tok.eos_id
    val = np.memmap(ROOT / "data" / "v2" / "val.u16", dtype=np.uint16, mode="r")
    print(f"val.u16: {len(val):,} tokens; tokenizer {meta['tokenizer']}, eos {tok.eos_id}")

    # Which documents this sampler can reach: a draw at offset p takes the document after the
    # first EOS in val[p : p + 2 * MAX_TOKENS], and needs that document's own EOS inside the window.
    eos_all = np.flatnonzero(np.asarray(val) == tok.eos_id)
    starts = np.r_[0, eos_all[:-1] + 1]
    lens = eos_all - starts
    n_offsets = np.maximum(0, eos_all[:-1] - np.maximum(starts[:-1], eos_all[1:] - 2 * MAX_TOKENS + 1) + 1)
    never = n_offsets == 0
    bias = {"documents": int(len(eos_all)), "never_drawable_docs": int(never.sum()),
            "never_drawable_token_share": round(float(lens[1:][never].sum() / len(val)), 4),
            "docs_over_4096_token_share": round(float(lens[lens > 4096].sum() / len(val)), 4)}
    print(f"sampler reach: {bias['never_drawable_docs']} of {bias['documents']:,} documents can never be drawn "
          f"({100 * bias['never_drawable_token_share']:.2f}% of held-out tokens); documents over 4,096 tokens "
          f"hold {100 * bias['docs_over_4096_token_share']:.1f}% of tokens and are drawn less often")

    rng = random.Random(SEED)
    seen, docs = set(), []
    while len(docs) < N_DOCS:
        p = rng.randrange(len(val) - MAX_TOKENS * 2)
        window = np.asarray(val[p:p + MAX_TOKENS * 2])
        eos = np.flatnonzero(window == tok.eos_id)
        if len(eos) < 2 or p + eos[0] in seen:
            continue                                  # need one whole document after the first EOS
        seen.add(p + eos[0])
        docs.append(window[eos[0] + 1:eos[1]].astype(np.int64).tolist())

    t0 = time.time()
    stats = {k: Counter() for k in ("devanagari", "latin", "all")}
    used = Counter()
    exact = bad_utf8 = 0
    for ids in docs:
        raw = b"".join(tok.vocab[i] for i in ids)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            bad_utf8 += 1
            continue
        exact += tok.encode(text) == ids
        used.update(ids)
        for k in (script_of(text), "all"):
            s = stats[k]
            s["docs"] += 1
            s["tokens"] += len(ids)
            s["bytes"] += len(raw)
            s["chars"] += len(text)
            s["words"] += sum(1 for c in CHUNK_RE.findall(text) if c.strip())
            s["single_byte_tokens"] += sum(1 for i in ids if i < 256)
    print(f"{len(docs)} documents, seed {SEED}, re-encoded in {time.time() - t0:.0f} s")

    res = {"source": "data/v2/val.u16 (held-out split of the base-v2 run)", "tokenizer": "data/multi32k.json",
           "n_docs": len(docs), "seed": SEED, "invalid_utf8_docs": bad_utf8,
           "bpe_py_reencode_exact": exact, "distinct_ids_used": len(used),
           "meta_bytes_per_token": {k: v["bytes_per_token"] for k, v in meta["sources"].items()},
           "meta_bytes_per_token_all": meta["bytes_per_token"], "sampler_reach": bias, "by_script": {}}
    for k, s in stats.items():
        r = {"docs": s["docs"], "tokens": s["tokens"], "bytes_per_token": round(s["bytes"] / s["tokens"], 3),
             "chars_per_token": round(s["chars"] / s["tokens"], 3), "tokens_per_word": round(s["tokens"] / s["words"], 3),
             "bytes_per_char": round(s["bytes"] / s["chars"], 3),
             "single_byte_token_share": round(s["single_byte_tokens"] / s["tokens"], 5)}
        res["by_script"][k] = r
        print(f"{k:11} docs {r['docs']:5}  tokens {r['tokens']:9,}  bytes/token {r['bytes_per_token']:5.2f}  "
              f"chars/token {r['chars_per_token']:5.2f}  tokens/word {r['tokens_per_word']:4.2f}  "
              f"bytes/char {r['bytes_per_char']:4.2f}  single-byte tokens {100 * r['single_byte_token_share']:.2f}%")
    dev, lat = res["by_script"]["devanagari"], res["by_script"]["latin"]
    res["window_chars_1024"] = {"bytes_devanagari": round(1024 / dev["bytes_per_char"]),
                                "bytes_latin": round(1024 / lat["bytes_per_char"]),
                                "multi32k_devanagari": round(1024 * dev["chars_per_token"]),
                                "multi32k_latin": round(1024 * lat["chars_per_token"])}
    print("characters in a 1,024-token window:", res["window_chars_1024"])
    res["vocab"] = vocab_census(tok)
    print(f"bpe.py re-encodes {exact} of {len(docs) - bad_utf8} documents to exactly the stored ids; "
          f"{bad_utf8} not valid UTF-8")
    print(f"distinct ids used: {len(used):,} of {tok.vocab_size:,}")
    print("vocabulary census:", res["vocab"]["kinds"])
    print("merged-token byte lengths (16 = 16 or more):", res["vocab"]["byte_length"])
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
