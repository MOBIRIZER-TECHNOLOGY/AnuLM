"""
Book example, not repository code.

Chapter 21: the data path end to end, on a corpus small enough to read. It runs
the repository's own code unchanged:

1. `experiments/build_v2.py` builds train.u16 / val.u16 / meta.json from three
   made-up sources (its SOURCES table is swapped for them; nothing else changes);
2. `train.load_memmap` opens the result and `train.make_batch` cuts (x, y);
3. three ways of holding data out are compared on the same ordered file;
4. `train.WindowSampler` is checked for coverage against sampling with
   replacement.

    python book/code-examples/ch21_windows.py
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import build_v2  # noqa: E402  the repository's corpus builder, unchanged
from anulm.training import train as T  # noqa: E402  the repository's train.py, unchanged

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rng = random.Random(0)
WORDS = {"english": "the river city school market evening train letter garden".split(),
         "hindi": "नदी शहर विद्यालय बाज़ार शाम रेल पत्र बगीचा".split(),
         "python": "def return import for in if else print range len".split()}


def docs(kind, n):
    """n made-up documents of one source, 40-120 words each."""
    return [" ".join(rng.choice(WORDS[kind]) for _ in range(rng.randint(40, 120))) for _ in range(n)]


def build(out: Path):
    """Run build_v2.main() unchanged on three small sources, 10% held out by hash."""
    build_v2.SOURCES = {k: (1e12, (lambda k=k: iter(docs(k, 600)))) for k in WORDS}
    sys.argv = ["build_v2.py", "--out", str(out), "--val-pct", "10"]
    build_v2.main()
    return json.loads((out / "meta.json").read_text())


def composition(ranges, lo, hi):
    """Share of each source in token range [lo, hi), given each source's [start, end)."""
    return {k: max(0, min(hi, b) - max(lo, a)) / (hi - lo) for k, (a, b) in ranges.items()}


def main():
    with tempfile.TemporaryDirectory() as d:
        meta = build(Path(d))
        tr, va, _ = T.load_memmap(str(Path(d) / "meta.json"))
        print(f"\nmeta.json: train {meta['train_tokens']:,} / val {meta['val_tokens']:,} tokens, "
              f"EOS id {meta['eos_id']}; "
              f"{100 * meta['val_tokens'] / (meta['train_tokens'] + meta['val_tokens']):.1f}% held out")
        assert len(tr) == meta["train_tokens"] and int((np.asarray(tr) == meta["eos_id"]).sum()) > 0

        # --- (x, y) from the memory-mapped file: y is x shifted by one ---------------
        ix = torch.tensor([0, 500, 2000])
        x, y = T.make_batch(tr, ix, 16, "cpu")
        assert torch.equal(x[:, 1:], y[:, :-1]) and x.dtype == torch.long
        same = T.make_batch(torch.from_numpy(np.asarray(tr).astype(np.int64)), ix, 16, "cpu")
        assert torch.equal(x, same[0]) and torch.equal(y, same[1])
        print(f"make_batch: x {tuple(x.shape)}, y {tuple(y.shape)}; y[:, :-1] == x[:, 1:]; "
              f"memmap and in-memory batches identical")

        # --- three held-out splits of the same ordered training file ----------------
        n, edges, at = len(tr), {}, 0
        for k, s in meta["sources"].items():
            edges[k] = (at, at + s["train_tokens"])
            at += s["train_tokens"]
        tail = composition(edges, int(0.9 * n), n)
        lab = torch.zeros(n, dtype=torch.int16)
        for i, (a, b) in enumerate(edges.values()):
            lab[a:b] = i
        _, val_lab = T.interleaved_split(lab, chunk=256)          # same split, applied to source labels
        inter = {k: float((val_lab == i).float().mean()) for i, k in enumerate(edges)}
        hashed = {k: s["val_tokens"] / meta["val_tokens"] for k, s in meta["sources"].items()}
        whole = {k: (b - a) / n for k, (a, b) in edges.items()}
        print(f"\n{'held out':22s}" + "".join(f"{k:>9s}" for k in edges))
        for name, c in (("the corpus itself", whole), ("tail 10% (default)", tail),
                        ("interleaved_split", inter), ("hash of each document", hashed)):
            print(f"{name:22s}" + "".join(f"{100 * c[k]:8.1f}%" for k in edges))
        assert tail["english"] == 0.0 and all(v > 0.15 for v in inter.values()) and all(v > 0.15 for v in hashed.values())
        del tr, va                                              # release the files before the folder goes

    # --- windows without replacement, against windows with it --------------------
    n, S, B = 100_000, 64, 8
    s = T.WindowSampler(n, S, B, seed=3)
    w0, off0 = len(s.order), int(s.order.min())
    seen = torch.cat([s.next() for _ in range(w0 // B)])
    assert len(set(seen.tolist())) == len(seen) and s.epoch == 0
    s.next()                                                    # the remainder starts epoch 1
    print(f"\nWindowSampler: {w0} windows of {S} per epoch; epoch 0 offset {off0}, "
          f"epoch 1 offset {int(s.order.min())}; {len(seen)} windows in epoch 0, none repeated")
    g = torch.Generator().manual_seed(3)
    starts = torch.randint(n - S - 1, (w0,), generator=g)      # train.py --sample-with-replacement
    covered = torch.zeros(n, dtype=torch.bool)
    for a in starts.tolist():
        covered[a:a + S] = True
    print(f"with replacement, the same {w0} draws: {100 * (1 - covered.float().mean()):.1f}% of tokens never seen "
          f"(e^-1 = {100 * np.exp(-1):.1f}%); without: {100 * (1 - len(seen) * S / n):.1f}%")
    unseen, skipped = n - len(seen) * S, (w0 - len(seen)) * S
    print(f"  the {unseen} unseen tokens without replacement: {skipped} in the one skipped window "
          f"({w0} windows, {w0 // B} batches of {B}), {off0} before the offset, {unseen - skipped - off0} after the grid")

    # --- which integer dtypes the model's two id consumers accept (why make_batch widens to int64)
    emb, logits = torch.randn(10, 4), torch.randn(3, 10)
    for dt in (torch.int64, torch.int32, torch.uint16):
        ids = torch.tensor([1, 2, 3]).to(dt)
        res = []
        for f in (lambda: torch.nn.functional.embedding(ids, emb), lambda: torch.nn.functional.cross_entropy(logits, ids)):
            try:
                f()
                res.append("ok")
            except Exception as e:
                res.append(type(e).__name__)
        print(f"ids as {str(dt)[6:]:6s}: F.embedding {res[0]:12s} F.cross_entropy {res[1]}")

    # --- the exercises' answers ----------------------------------------------------
    w = [len(T.WindowSampler(1000, 64, 1, seed=k).order) for k in range(200)]
    print(f"\nexercise 2: N=1000, S=64 -> windows per epoch between {min(w)} and {max(w)} over 200 seeds; "
          f"formula {(999 - 63) // 64}..{999 // 64}")
    tr_, va_ = T.interleaved_split(torch.zeros(10_000_000, dtype=torch.int16))
    print(f"exercise 3: interleaved_split of 10M ids holds out {len(va_):,} ({100 * len(va_) / 1e7:.2f}%), "
          f"from {-(-10_000_000 // 262144)} chunks")


if __name__ == "__main__":
    main()
