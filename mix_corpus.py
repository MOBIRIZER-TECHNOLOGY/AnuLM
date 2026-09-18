"""
Interleave several DOC_SEP corpora into one training file, streaming.

    python mix_corpus.py --source python data/python_big.txt 9000 \
                         --source english data/english_edu.txt 3500 \
                         --source hindi data/mixed.txt 1300 \
                         --out data/coder.txt --sample-out data/coder_toksample.txt

Each --source is NAME PATH MB: the file and how many MB of it to use. Documents
are drawn from the sources at random in proportion to their remaining budget,
so the mix is uniform along the file rather than one language followed by
another. Every 50th document of each source goes to data/bench_<name>.txt
instead (held out, never trained on), unless that file already exists. The
tokenizer sample takes every k-th document up to --sample-mb per source.

Nothing is held in memory beyond one document, so a 15 GB corpus is fine.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from bpe import DOC_SEP, iter_documents


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", nargs=3, action="append", metavar=("NAME", "PATH", "MB"), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sample-out", default=None)
    p.add_argument("--sample-mb", type=float, nargs="*", default=None,
                   help="per source, same order as --source; default 5%% of each budget")
    p.add_argument("--heldout-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()
    rng = random.Random(args.seed)

    srcs = []
    for i, (name, path, mb) in enumerate(args.source):
        budget = float(mb) * 1024 * 1024
        sample = (args.sample_mb[i] if args.sample_mb else float(mb) * 0.05) * 1024 * 1024
        bench = Path(f"data/bench_{name}.txt")
        srcs.append(dict(name=name, it=iter_documents(path), left=budget, used=0.0, docs=0,
                         sample_left=sample, sample_every=max(1, int(budget / max(sample, 1))),
                         bench=None if bench.exists() else bench.open("w", encoding="utf-8", newline="\n"),
                         bench_n=0))
    out = Path(args.out).open("w", encoding="utf-8", newline="\n")
    sample = Path(args.sample_out).open("w", encoding="utf-8", newline="\n") if args.sample_out else None
    t0 = time.time()
    total = 0
    while True:
        live = [s for s in srcs if s["left"] > 0 and s["it"] is not None]
        if not live:
            break
        s = rng.choices(live, weights=[x["left"] for x in live])[0]
        try:
            doc = next(s["it"])
        except StopIteration:
            s["it"] = None
            continue
        n = len(doc.encode("utf-8")) + len(DOC_SEP)
        s["docs"] += 1
        if s["bench"] is not None and s["docs"] % args.heldout_every == 0:
            s["bench"].write(doc + DOC_SEP)
            s["bench_n"] += 1
            continue
        out.write(doc + DOC_SEP)
        s["left"] -= n
        s["used"] += n
        total += n
        if sample is not None and s["sample_left"] > 0 and s["docs"] % s["sample_every"] == 0:
            sample.write(doc + DOC_SEP)
            s["sample_left"] -= n
        if s["docs"] % 20000 == 0:
            print(f"\r  {total/1e6:.0f} MB written  ({time.time()-t0:.0f}s)", end="", flush=True)
    out.close()
    if sample is not None:
        sample.close()
    print(f"\r{args.out}: {total/1e6:.0f} MB in {time.time()-t0:.0f}s")
    for s in srcs:
        if s["bench"] is not None:
            s["bench"].close()
        short = "" if s["it"] is not None or s["left"] <= 0 else f"  (source ran out {s['left']/1e6:.0f} MB short)"
        print(f"  {s['name']:8s} {s['used']/1e6:7.0f} MB  {s['docs']:8d} docs"
              f"{'  bench ' + str(s['bench_n']) + ' docs' if s['bench_n'] else ''}{short}")


if __name__ == "__main__":
    main()
