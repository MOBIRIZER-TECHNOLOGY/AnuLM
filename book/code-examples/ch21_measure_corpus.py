"""
Book example, not repository code.

Chapter 21: statistics of the base-v2 corpus, for the page.

    python book/code-examples/ch21_measure_corpus.py

Two sources, both read-only:

- `data/v2/meta.json`, written by `experiments/build_v2.py`: tokens per source,
  the hash split's share of each source, and what a tail split of the same
  ordered file would have held out instead.
- `data/v2/val.u16`, the held-out split, memory-mapped through `train.load_memmap`
  unchanged. Only counts are taken from it (end-of-document ids, window
  boundaries). No held-out text is decoded, printed or saved.

With `--save` the results go to book/measurements/part_v_a_corpus.json; without it, to
./part_v_a_corpus.json in the current directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from anulm.training import train as T  # noqa: E402  the repository's train.py, unchanged

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
NAME = "part_v_a_corpus.json"
OUT = (ROOT / "book" / "measurements" / NAME) if "--save" in sys.argv else Path(NAME)
S, N_WIN = 1024, 400                         # the released run's block size and --eval-windows


def main():
    tr, va, meta = T.load_memmap(str(ROOT / "data" / "v2" / "meta.json"))
    eos = meta["eos_id"]
    assert len(tr) == meta["train_tokens"] and len(va) == meta["val_tokens"]
    src = meta["sources"]
    n_tr = meta["train_tokens"]
    res = {"train_tokens": n_tr, "val_tokens": meta["val_tokens"], "bytes_per_token": meta["bytes_per_token"],
           "sources": {}}
    print(f"train {n_tr:,} tokens, val {meta['val_tokens']:,} tokens "
          f"({100 * meta['val_tokens'] / (n_tr + meta['val_tokens']):.2f}% held out)")
    for name, s in src.items():
        share = s["train_tokens"] / n_tr
        held = s["val_tokens"] / (s["train_tokens"] + s["val_tokens"])
        vshare = s["val_tokens"] / meta["val_tokens"]
        res["sources"][name] = {"train_share": share, "val_share": vshare, "held_out": held,
                                "bytes_per_token": s["bytes_per_token"], "docs": s["docs"]}
        print(f"  {name:10s} train {s['train_tokens']:>14,} ({100 * share:5.2f}%)  val share {100 * vshare:5.2f}%"
              f"  held out {100 * held:.2f}%  {s['bytes_per_token']:.3f} bytes/token  {s['docs']:,} docs")
    import build_v2                                          # the budgets, read from the script itself
    for name, (budget, _) in build_v2.SOURCES.items():
        if budget < 1e12:
            over = src[name]["train_tokens"] - budget
            res["sources"][name]["over_budget"] = over
            print(f"  {name}: budget {budget:,.0f}, written {src[name]['train_tokens']:,} (+{over:,.0f}: the budget "
                  f"is checked after each batch of 1,000 documents)")
        else:
            print(f"  {name}: budget 1e12, i.e. every document on disk")
    val_bpt = sum(s["val_tokens"] * s["bytes_per_token"] for s in src.values()) / meta["val_tokens"]
    res["val_bytes_per_token"] = val_bpt
    print(f"bytes per token: corpus-wide {meta['bytes_per_token']:.3f} (meta.json), held-out split {val_bpt:.3f} "
          f"(per-source values weighted by held-out tokens)")

    size = (ROOT / "data" / "v2" / meta["train"]).stat().st_size
    res["train_file_bytes"] = size
    print(f"train.u16 is {size:,} bytes = {size / n_tr:.0f} bytes per token")

    # A tail split of the same file: build_v2 writes the sources in this order, one after another.
    tail = n_tr - int(0.9 * n_tr)
    left, comp = tail, {}
    for name in reversed(list(src)):
        take = min(left, src[name]["train_tokens"])
        comp[name] = take / tail
        left -= take
    res["tail_split"] = comp
    print("a 10% tail split of train.u16 would hold out: "
          + ", ".join(f"{k} {100 * v:.1f}%" for k, v in comp.items()))

    # Windows per epoch at the released run's block size, and how far 1.9M steps of 4 windows got.
    sampler = T.WindowSampler(n_tr, S, 4, seed=2026)
    w = len(sampler.order)
    res["windows_per_epoch"] = w
    res["epochs_at_1p9M_steps"] = 1_900_000 * 4 / w
    print(f"windows of {S} per epoch: {w:,}; 1,900,000 steps x 4 windows = {1_900_000 * 4 / w:.3f} epochs")

    # Documents in the held-out split (counts only).
    ends = np.flatnonzero(np.asarray(va) == eos)
    lengths = np.diff(np.concatenate([[-1], ends]))
    res["val_docs"] = int(len(ends))
    res["val_doc_tokens_mean"] = float(lengths.mean())
    res["val_doc_tokens_median"] = float(np.median(lengths))
    res["val_docs_longer_than_S"] = float((lengths > S).mean())
    print(f"held-out documents: {len(ends):,}; tokens per document mean {lengths.mean():.0f}, "
          f"median {np.median(lengths):.0f}; longer than {S}: {100 * (lengths > S).mean():.1f}%")

    # The released run's own validation windows: strided_windows, unchanged.
    starts = np.concatenate([b.numpy() for b in T.strided_windows(va, 4, S, N_WIN)])
    crosses, mixed = 0, 0
    for st in starts:
        x = np.asarray(va[st: st + S])                  # the input half of the window
        pos = np.flatnonzero(x == eos)
        if len(pos):
            crosses += 1
            mixed += S - 1 - pos[0]                     # positions after the first EOS see an earlier document
    res["windows_checked"] = int(len(starts))
    res["windows_with_a_boundary"] = crosses / len(starts)
    res["positions_seeing_an_earlier_document"] = mixed / (len(starts) * S)
    print(f"{len(starts)} strided windows of {S}: {100 * crosses / len(starts):.1f}% contain an end of document; "
          f"{100 * mixed / (len(starts) * S):.1f}% of positions can attend to an earlier document")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
