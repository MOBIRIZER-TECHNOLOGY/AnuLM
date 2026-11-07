"""Build the clean pretraining corpus for the from-scratch 1B model (data/v3).

    python -I experiments/build_v3.py --v2 data/v2 --code data/code_clean --out data/v3

What goes in (docs/PLAN_1B.md, "Data"):

    english     fineweb-edu sample/10BT              ODC-BY         6.00B tokens  } copied from
    hindi_web   fineweb-2 hin_Deva                    ODC-BY         4.50B         } data/v2, the
    hindi_wiki  Wikipedia 20231101.hi                 CC BY-SA 4.0   0.07B         } same tokens
    python      codeparrot/github-code-clean, MIT /   permissive     ~0.84B        new
                Apache-2.0 / BSD / ISC / CC0 / Unlicense files only

What does not, though base v2 had it: codeparrot-clean (GPL and AGPL files
mixed in), OpenCodeInstruct (model-generated), opc-sft-stage1 (ChatGPT-derived).

The first three sources are copied as tokens, not re-encoded. experiments/build_v2.py
wrote its sources one after another, so they are a prefix of data/v2/train.u16 and
val.u16; the prefix lengths come from data/v2/meta.json, and this script checks that
each one ends on an end-of-document token and that the next token starts the Python.

The code is encoded the way build_v2.py encoded everything: the Rust build of the
multi32k tokenizer, an end-of-document id after each file, and the same held-out
rule (a CRC of the first 200 characters, lowest 1% to val). Files sharing a 13-word
window with any MBPP or HumanEval prompt, solution or test are dropped first,
so pass@1 on those benchmarks stays a measurement rather than recall.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EOS = 32767
CLEAN = ("english", "hindi_web", "hindi_wiki")


def copy_prefix(src: Path, dst: Path, n_tokens: int, chunk: int = 1 << 28) -> None:
    """Copy the first n_tokens uint16 tokens of src to dst, in chunks."""
    left = n_tokens * 2
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while left:
            b = fi.read(min(chunk, left))
            if not b:
                raise SystemExit(f"{src} is shorter than {n_tokens:,} tokens")
            fo.write(b)
            left -= len(b)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v2", default="data/v2")
    ap.add_argument("--code", default="data/code_clean")
    ap.add_argument("--out", default="data/v3")
    ap.add_argument("--bench-root", default="data/raw", help="where MBPP and HumanEval parquet live")
    ap.add_argument("--tokenizer", default=str(ROOT / "release" / "AnuLM-Base-v2-400M" / "tokenizer.json"))
    ap.add_argument("--val-pct", type=float, default=1.0)
    a = ap.parse_args()
    v2, code, out = Path(a.v2), Path(a.code), Path(a.out)
    meta2 = json.loads((v2 / "meta.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. The clean prefix, checked at its boundary.
    n_tr = sum(meta2["sources"][s]["train_tokens"] for s in CLEAN)
    n_va = sum(meta2["sources"][s]["val_tokens"] for s in CLEAN)
    for name, n in (("train", n_tr), ("val", n_va)):
        mm = np.memmap(v2 / f"{name}.u16", dtype=np.uint16, mode="r")
        if mm[n - 1] != EOS:
            raise SystemExit(f"{name}.u16: token {n - 1:,} is not end-of-document; the prefix is not clean")
        del mm
    print(f"copying the clean prefix: {n_tr:,} train + {n_va:,} val tokens", flush=True)
    copy_prefix(v2 / "train.u16", out / "train.u16", n_tr)
    copy_prefix(v2 / "val.u16", out / "val.u16", n_va)
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    # 2. The permissive code, decontaminated and encoded.
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    from anulm.data.convert_parquet import benchmark_ngrams, ngrams
    bench = benchmark_ngrams(Path(a.bench_root))
    tk = Tokenizer.from_file(a.tokenizer)
    stats = {"train_tokens": 0, "val_tokens": 0, "docs": 0, "bytes": 0, "dropped_benchmark_overlap": 0,
             "licenses": {}}
    with open(out / "train.u16", "ab") as ftr, open(out / "val.u16", "ab") as fva:
        for f in sorted(code.glob("part-*.parquet")):
            t = pq.read_table(f, columns=["text", "license"]).to_pylist()
            docs = []
            for r in t:
                if ngrams(r["text"]) & bench:
                    stats["dropped_benchmark_overlap"] += 1
                    continue
                docs.append(r["text"])
                stats["licenses"][r["license"]] = stats["licenses"].get(r["license"], 0) + 1
            for d, enc in zip(docs, tk.encode_batch(docs, add_special_tokens=False)):
                ids = np.array(enc.ids + [EOS], dtype=np.uint16)
                stats["docs"] += 1
                stats["bytes"] += len(d.encode("utf-8"))
                if zlib.crc32(d[:200].encode("utf-8")) % 10000 < a.val_pct * 100:
                    fva.write(ids.tobytes())
                    stats["val_tokens"] += len(ids)
                else:
                    ftr.write(ids.tobytes())
                    stats["train_tokens"] += len(ids)
            print(f"  {f.name}: {stats['docs']:,} files, {stats['train_tokens'] / 1e9:.3f}B tokens, "
                  f"{stats['dropped_benchmark_overlap']} dropped for benchmark overlap, "
                  f"{time.time() - t0:.0f}s", flush=True)

    sources = {s: meta2["sources"][s] for s in CLEAN}
    sources["python"] = {"train_tokens": stats["train_tokens"], "val_tokens": stats["val_tokens"],
                         "docs": stats["docs"],
                         "bytes_per_token": round(stats["bytes"] / max(stats["train_tokens"] + stats["val_tokens"], 1), 3),
                         "source": "codeparrot/github-code-clean, permissive licences only",
                         "licenses": stats["licenses"],
                         "dropped_benchmark_overlap": stats["dropped_benchmark_overlap"]}
    tot_tr = sum(s["train_tokens"] for s in sources.values())
    tot_va = sum(s["val_tokens"] for s in sources.values())
    total_bytes = sum(s["bytes_per_token"] * (s["train_tokens"] + s["val_tokens"]) for s in sources.values())
    meta = {**{k: meta2[k] for k in ("format", "vocab_size", "eos_id", "tokenizer")},
            "train": "train.u16", "val": "val.u16", "train_tokens": tot_tr, "val_tokens": tot_va,
            "bytes_per_token": round(total_bytes / max(tot_tr + tot_va, 1), 3), "sources": sources,
            "built_from": {"prefix_of": str(v2), "code": str(code)}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    for name, n in (("train", tot_tr), ("val", tot_va)):        # the files must say what meta says
        if (out / f"{name}.u16").stat().st_size != 2 * n:
            raise SystemExit(f"{name}.u16 size does not match meta.json")
    print(f"built {tot_tr / 1e9:.2f}B train + {tot_va / 1e6:.0f}M val tokens in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
