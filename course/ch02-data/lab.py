"""
Chapter 2 lab: run your pipeline on real text, and catch what it is for.

    python lab.py                      # your course/build/data.py
    python lab.py --impl solution      # the reference solution

Seconds on a CPU. The corpus: 437 Hindi Wikipedia articles (data/bench_hindi.txt,
CC BY-SA 4.0) and Shakespeare's speeches (data/input.txt, public domain). Into it
the lab plants what real corpora contain by accident: duplicated documents, and
a handful of "benchmark" passages we later want to score the model on. Then:

  1. dedup: are the planted copies found?
  2. prose filter: how much of the Hindi survives, and what gets cut?
  3. the split: Hindi then English, as a corpus built source by source; what is
     in the validation set if you take the tail, and if you interleave?
  4. contamination: the planted benchmark passages, found by your 13-word
     windows, and by the repository's (ASCII-only) ones.
Writes results.json for the figures.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
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
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    D = load(COURSE / "build" / ("solutions/ch02.py" if a.impl == "solution" else "data.py"))
    import convert_parquet
    rnd = random.Random(a.seed)
    try:
        hindi = D.split_documents((ROOT / "data" / "bench_hindi.txt").read_text(encoding="utf-8"))
    except NotImplementedError as e:
        sys.exit(f"course/build/data.py is not finished yet ({e}). Run course/checks/ch02.py, or use --impl solution.")
    speeches = [s.strip() for s in (ROOT / "data" / "input.txt").read_text(encoding="utf-8").split("\n\n") if s.strip()]
    english = [s for s in speeches if len(s.split()) >= 40][:400]           # longer speeches as documents
    out = {"hindi_docs": len(hindi), "english_docs": len(english)}

    # benchmark passages: 15 Hindi and 15 English 20-word passages from documents held back
    held_h, hindi = hindi[-30:], hindi[:-30]
    held_e, english = english[-30:], english[:-30]
    def passage(doc):
        w = doc.split()
        start = rnd.randrange(0, max(1, len(w) - 20))
        return " ".join(w[start:start + 20])
    bench = [passage(d) for d in held_h[:15]] + [passage(d) for d in held_e[:15]]
    planted = sorted(rnd.sample(range(15), 5)) + sorted(rnd.sample(range(15, 30), 5))
    corpus = hindi + english
    for i in planted:                                                       # hide each one in a document
        k = rnd.randrange(len(corpus))
        corpus[k] = corpus[k] + "\n" + bench[i] + "\n"
    dups = rnd.sample(range(len(corpus)), 25)
    corpus = corpus + [corpus[i] for i in dups] + [corpus[i] for i in dups[:10]]   # 35 copies of 25 documents
    rnd.shuffle(corpus)

    # 1. dedup
    kept, removed = D.exact_dedup(corpus)
    out["dedup"] = {"in": len(corpus), "kept": len(kept), "removed": removed, "planted": 35}
    print(f"1. dedup: {len(corpus)} documents in, {removed} exact copies removed (35 were planted)")

    # 2. prose filter on the Hindi articles
    h_kept = [D.filter_doc(d) for d in kept if D.script_share(d) > 0.5]
    n_in = sum(1 for d in kept if D.script_share(d) > 0.5)
    survivors = [d for d in h_kept if d is not None]
    chars_in = sum(len(d) for d in kept if D.script_share(d) > 0.5)
    chars_out = sum(len(d) for d in survivors)
    out["filter"] = {"docs_in": n_in, "docs_kept": len(survivors), "chars_in": chars_in, "chars_kept": chars_out}
    print(f"2. prose filter: {len(survivors)} of {n_in} Hindi articles kept, {chars_out / chars_in:.0%} of their text")

    # 3. the split: a corpus built source by source (all Hindi, then all English)
    ordered = "\n\n\n".join([d for d in kept if D.script_share(d) > 0.5] + [d for d in kept if D.script_share(d) <= 0.5])
    cut = len(ordered) - len(ordered) // 10
    tail = ordered[cut:]
    _, inter = D.interleaved_split(ordered, 4096, 10)
    whole = D.script_share(ordered)
    out["split"] = {"corpus": round(whole, 4), "tail": round(D.script_share(tail), 4),
                    "interleaved": round(D.script_share(inter), 4)}
    print(f"3. Devanagari share: whole corpus {whole:.0%}, last 10% {D.script_share(tail):.0%}, "
          f"interleaved 10% {D.script_share(inter):.0%}")

    # 4. contamination
    mine = D.contaminated(kept, bench, 13)
    seen = set()
    for d in kept:
        seen |= convert_parquet.ngrams(d)
    theirs = [i for i, b in enumerate(bench) if convert_parquet.ngrams(b) & seen]
    def tally(found):
        return {"hindi": sum(1 for i in found if i < 15), "english": sum(1 for i in found if i >= 15),
                "false": sum(1 for i in found if i not in planted)}
    out["contamination"] = {"planted": {"hindi": 5, "english": 5}, "yours": tally(mine), "repository": tally(theirs)}
    print(f"4. planted benchmark passages found (5 Hindi, 5 English planted):")
    print(f"   your 13-word windows:        {tally(mine)}")
    print(f"   the repository's ([a-z0-9_]): {tally(theirs)}")
    (HERE / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("saved results.json")


if __name__ == "__main__":
    main()
