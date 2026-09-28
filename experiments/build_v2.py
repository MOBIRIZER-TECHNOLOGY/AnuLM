"""
Build the base-v2 pretraining corpus: ~11B tokens of Hindi, English and Python
through the project's own multi32k tokenizer, written as raw uint16 files that
train.py memory-maps instead of loading.

    python experiments/build_v2.py                    # -> data/v2/{train,val}.u16, meta.json
    python experiments/build_v2.py --scale 0.001      # a ~11M-token smoke build

Sources (all permissive; docs/DATASETS.md has the links):

    english  fineweb-edu sample/10BT, 14 shards      ODC-BY        6.0B tokens
    hindi    fineweb-2 hin_Deva, shards 000-003      ODC-BY        4.5B
    hindi    Wikipedia 20231101.hi                   CC BY-SA      all (~0.08B)
    python   gh_python.clean + OpenCodeInstruct + OPC text on disk
                                                     mixed / CC BY / MIT   all (~0.45B)

jinaai/code_exercises is deliberately left out: its CC BY-NC-SA terms would
make the whole base non-commercial (docs/PUBLISHING.md section 1).

Encoding uses the Rust `tokenizers` build of multi32k
(release/AnuLM-Base-400M/tokenizer.json). It was checked identical to
bpe.py on 400 fineweb-2 Hindi and 400 fineweb-edu documents, at 20-36 MB/s
against bpe.py's 3.4-4 MB/s.

Held-out split: each document goes to val.u16 if a hash of its first 200
characters falls in the lowest 1%, so validation samples every source in
proportion -- not the tail of an ordered file (docs/RESULTS.md section 32).
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
TOKENIZER = ROOT / "release" / "AnuLM-Base-400M" / "tokenizer.json"
EOS = 32767


def parquet_docs(files, column="text", min_chars=200):
    import pyarrow.parquet as pq
    for f in files:
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=2000, columns=[column]):
            for d in b.column(column).to_pylist():
                if d and len(d) >= min_chars:
                    yield d


def text_docs(files, sep="\n\n\n", min_chars=200):
    for f in files:
        buf = ""
        with open(f, encoding="utf-8", errors="replace") as fh:
            while True:
                chunk = fh.read(1 << 24)
                if not chunk:
                    break
                buf += chunk
                parts = buf.split(sep)
                buf = parts.pop()
                for d in parts:
                    if len(d) >= min_chars:
                        yield d
        if len(buf) >= min_chars:
            yield buf


SOURCES = {
    "english": (6.0e9, lambda: parquet_docs(sorted((RAW / "HuggingFaceFW__fineweb-edu/sample/10BT").glob("*.parquet")))),
    "hindi_web": (4.5e9, lambda: parquet_docs(sorted((RAW / "HuggingFaceFW__fineweb-2/data/hin_Deva/train").glob("*.parquet")))),
    "hindi_wiki": (1e12, lambda: parquet_docs(sorted((ROOT / "data/rag/wiki/20231101.hi").glob("*.parquet")))),
    "python": (1e12, lambda: text_docs([ROOT / "data" / f for f in
                                        ("gh_python.clean.txt", "tb_opencode.txt", "tb_opc.txt")
                                        if (ROOT / "data" / f).exists()])),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "v2"))
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every budget (smoke builds)")
    ap.add_argument("--val-pct", type=float, default=1.0)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(TOKENIZER))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ftr, fva = open(out / "train.u16", "wb"), open(out / "val.u16", "wb")
    stats, t0 = {}, time.time()
    for name, (budget, docs) in SOURCES.items():
        budget *= a.scale
        n_tr = n_va = n_docs = n_bytes = 0
        batch = []

        def flush(batch):
            nonlocal n_tr, n_va, n_docs, n_bytes
            for d, enc in zip(batch, tk.encode_batch(batch, add_special_tokens=False)):
                ids = np.array(enc.ids + [EOS], dtype=np.uint16)
                n_docs += 1
                n_bytes += len(d.encode("utf-8"))
                if zlib.crc32(d[:200].encode("utf-8")) % 10000 < a.val_pct * 100:
                    fva.write(ids.tobytes())
                    n_va += len(ids)
                else:
                    ftr.write(ids.tobytes())
                    n_tr += len(ids)

        for d in docs():
            batch.append(d)
            if len(batch) == 1000:
                flush(batch)
                batch = []
                if n_tr >= budget:
                    break
                if n_docs % 200000 == 0:
                    print(f"  {name}: {n_tr / 1e9:.2f}B tokens, {n_docs:,} docs, "
                          f"{time.time() - t0:.0f}s", flush=True)
        if batch and n_tr < budget:
            flush(batch)
        stats[name] = {"train_tokens": n_tr, "val_tokens": n_va, "docs": n_docs,
                       "bytes_per_token": round(n_bytes / max(n_tr + n_va, 1), 3)}
        print(f"{name}: {n_tr / 1e9:.3f}B train + {n_va / 1e6:.1f}M val tokens from {n_docs:,} docs "
              f"({time.time() - t0:.0f}s)", flush=True)
    ftr.close(), fva.close()
    tot_tr = sum(s["train_tokens"] for s in stats.values())
    tot_va = sum(s["val_tokens"] for s in stats.values())
    total_bytes = sum(s["bytes_per_token"] * (s["train_tokens"] + s["val_tokens"]) for s in stats.values())
    meta = {"format": "uint16", "vocab_size": 32768, "eos_id": EOS,
            "tokenizer": "data/multi32k.json", "train": "train.u16", "val": "val.u16",
            "train_tokens": tot_tr, "val_tokens": tot_va,
            "bytes_per_token": round(total_bytes / max(tot_tr + tot_va, 1), 3), "sources": stats}
    json.dump(meta, open(out / "meta.json", "w"), indent=2)
    print(f"\nbuilt {tot_tr / 1e9:.2f}B train + {tot_va / 1e6:.0f}M val tokens in "
          f"{time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
