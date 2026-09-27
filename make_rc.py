"""
Reading-comprehension data for retrieval-augmented answering (rag.py).

    python make_rc.py                     # -> data/rc_train.jsonl, data/rc_heldout.jsonl, data/rc_eval_*.jsonl

Each example is three passages and a question; the answer is a short span.
What makes it match how the model will actually be used:

  * The passages are the right one plus two HARD NEGATIVES -- what rag.py's
    own BM25 returns for that question, minus anything containing the answer.
    Those are exactly the near-misses the model will be handed at inference.
  * In 15% of examples the right passage is left out and the answer is
    "not found in the passages" (Hindi: "अनुच्छेदों में उत्तर नहीं मिला"). A reader that has
    only ever seen answerable questions invents an answer from whatever it is
    given; this teaches it to say so instead.
  * Passage order is shuffled, so the right one is not always first.

Sources: SQuAD v1.1 train (English, CC BY-SA 4.0) and MLQA translate-train
Hindi (SQuAD machine-translated, CC BY-SA 3.0), 30k each. Held-out
evaluation, never trained on: SQuAD dev (English), MLQA hi.hi test (native
Hindi questions on Hindi Wikipedia), XQuAD hi, IndicQA hi.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
RC = HERE / "data" / "rc"
NOT_FOUND = {"en": "not found in the passages", "hi": "अनुच्छेदों में उत्तर नहीं मिला"}
PASSAGE_WORDS = 120
N_TRAIN = 30000
NEG_FRAC = 0.15

_R = None


def _init():
    global _R
    from rag import Retriever
    _R = Retriever()


def window(context: str, answer: str, start: int, words: int = PASSAGE_WORDS) -> str:
    """At most `words` words of `context`, always including the answer span."""
    w = context.split()
    if len(w) <= words:
        return context
    pos = len(context[:max(start, 0)].split())
    lo = max(0, min(pos - words // 2, len(w) - words))
    return " ".join(w[lo:lo + words])


def _negatives(item):
    q, answers = item["question"], item["answers"]
    hits = _R.search(q, 8)
    negs = [h["text"] for h in hits if not any(a.lower() in h["text"].lower() for a in answers)]
    return negs[:2]


def build_example(rng, it, negs, lang, allow_unanswerable=True):
    passages = [" ".join(n.split()[:PASSAGE_WORDS]) for n in negs]
    unanswerable = allow_unanswerable and rng.random() < NEG_FRAC and len(passages) >= 2
    if not unanswerable:
        passages.append(window(it["context"], it["answers"][0], it["start"]))
    rng.shuffle(passages)
    qword = "Question" if lang == "en" else "प्रश्न"
    body = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    return {"question": f"{body}\n\n{qword}: {it['question']}",
            "answer": NOT_FOUND[lang] if unanswerable else it["answers"][0],
            "answers": [NOT_FOUND[lang]] if unanswerable else it["answers"],
            "lang": "rc-en" if lang == "en" else "rc-hi", "unanswerable": unanswerable}


def load(path, n=None, seed=0):
    import pandas as pd
    df = pd.read_parquet(path)
    if n and len(df) > n:
        df = df.sample(n=n, random_state=seed)
    out = []
    for c, q, a in zip(df.context, df.question, df.answers):
        texts = [t for t in a["text"] if t]
        if texts:
            out.append({"context": c, "question": q, "answers": texts, "start": int(a["answer_start"][0])})
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rng = random.Random(0)
    jobs = {
        "train_en": (RC / "rajpurkar__squad/plain_text/train-00000-of-00001.parquet", N_TRAIN, "en", True),
        "train_hi": (RC / "facebook__mlqa/mlqa-translate-train.hi/train/0000.parquet", N_TRAIN, "hi", True),
        "eval_squad_en": (RC / "rajpurkar__squad/plain_text/validation-00000-of-00001.parquet", 500, "en", False),
        "eval_mlqa_hi": (RC / "facebook__mlqa/mlqa.hi.hi/test/0000.parquet", 500, "hi", False),
        "eval_xquad_hi": (RC / "google__xquad/xquad.hi/validation-00000-of-00001.parquet", 500, "hi", False),
        "eval_indicqa_hi": (RC / "ai4bharat__IndicQA/indicqa.hi/test/0000.parquet", 500, "hi", False),
    }
    with mp.Pool(6, initializer=_init) as pool:
        out = {}
        for name, (path, n, lang, unans) in jobs.items():
            items = load(path, n)
            negs = pool.map(_negatives, items, chunksize=200)
            out[name] = [build_example(rng, it, ng, lang, unans) for it, ng in zip(items, negs)]
            print(f"{name}: {len(out[name])} examples", flush=True)
    train = out["train_en"] + out["train_hi"]
    rng.shuffle(train)
    held, train = train[:600], train[600:]
    for name, rows in [("rc_train", train), ("rc_heldout", held)] + \
            [(f"rc_{k}", v) for k, v in out.items() if k.startswith("eval")]:
        with open(HERE / "data" / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train {len(train)}, held-out {len(held)}, "
          f"unanswerable {sum(r['unanswerable'] for r in train) / len(train):.0%}")
    ex = train[0]
    print("\nexample:\n" + ex["question"][:600] + "\n  -> " + ex["answer"])


if __name__ == "__main__":
    main()
