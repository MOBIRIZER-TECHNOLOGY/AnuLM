"""
Book example, not repository code.

Chapter 33: retrieval and reading, small enough to check by hand, run through
rag.py and rc_pointer.py unchanged.

    python book/code-examples/ch33_retrieval.py

1. Four made-up articles go through rag.build (into a temporary folder, from
   tiny parquet files) and rag.Retriever.bm25; `bm25_by_hand` recomputes every
   score from the textbook formula with rag's k1 and b.
2. Reciprocal-rank fusion: `rrf_by_hand` against Retriever.search(mode="hybrid"),
   with a stand-in dense retriever that returns a fixed ranking.
3. The generative reader's prompt, from RagQA.prompt.
4. The pointer reader's span search: rc_pointer.best_span against a brute-force
   loop, with and without an answer.
5. The size of the real index on disk, from data/rag/meta.json, if present.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import types
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm.tasks import rag as rag  # noqa: E402  the repository's rag.py, unchanged
from anulm.tasks import rc_pointer as rc_pointer  # noqa: E402
from anulm.data.make_qa import PROMPTS  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ARTICLES = [("Delhi", "Delhi is the capital of India. New Delhi hosts the government of India."),
            ("Vaishali", "Vaishali was an ancient city. It was a capital of the Vajji league in ancient India."),
            ("Tokyo", "Tokyo is the capital of Japan and its largest city."),
            ("Ganga", "The Ganga is a river of India. It flows past Varanasi and Patna.")]


def bm25_by_hand(query, docs, k1=rag.K1, b=rag.B):
    """docs: list of word lists (title + text). Score = sum over query words of
    idf(w) * tf (k1 + 1) / (tf + k1 (1 - b + b |d| / avgdl)), idf = ln(1 + (N - df + .5) / (df + .5))."""
    N, avgdl = len(docs), sum(map(len, docs)) / len(docs)
    df = Counter(w for d in docs for w in set(d))
    scores = []
    for d in docs:
        tf, s = Counter(d), 0.0
        for w in set(rag.words(query)):
            if tf[w]:
                idf = math.log1p((N - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avgdl))
        scores.append(s)
    return scores


def rrf_by_hand(rankings, k=60):
    """Each list ranks chunk ids best first; an id at 0-based position `rank` scores 1 / (k + rank)."""
    fused = Counter()
    for ranking in rankings:
        for rank, i in enumerate(ranking):
            fused[i] += 1 / (k + rank)
    return fused.most_common()


def main() -> None:
    import pandas as pd
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        wiki = tmp / "wiki" / "20231101.{lang}"
        Path(str(wiki).format(lang="simple")).mkdir(parents=True)
        pd.DataFrame({"title": [t for t, _ in ARTICLES], "text": [x for _, x in ARTICLES]}).to_parquet(
            Path(str(wiki).format(lang="simple")) / "part.parquet")
        rag.RAG, rag.WIKI = tmp, wiki
        rag.build(langs=("simple",), workers=1)
        r = rag.Retriever(tmp)

        # 1. BM25 against the hand formula
        q = "What is the capital of India?"
        hits = r.bm25(q, k=4)
        docs = [rag.words(t + " " + x) for t, x in ARTICLES]
        mine = bm25_by_hand(q, docs)
        print(f"query {q!r}: rag.bm25 ranks {[h['title'] for h in hits]}")
        for h in hits:
            i = h["_i"]
            print(f"  {h['title']:9s} rag {h['score']:.4f}   by hand {mine[i]:.4f}")
            assert abs(h["score"] - mine[i]) < 5e-3            # weights are stored as float16

        # 2. fusion against Retriever.search(mode="hybrid")
        dense_rank = [2, 0, 3, 1]                              # a stand-in dense ranking: Tokyo first
        r.dense = types.SimpleNamespace(search=lambda query, k: (dense_rank[:k], [0.0] * min(k, 4)))
        fused = r.search(q, k=4, mode="hybrid")
        bm25_rank = [h["_i"] for h in r.bm25(q, 100)]
        hand = rrf_by_hand([dense_rank, bm25_rank])
        print(f"bm25 ranking {bm25_rank}, dense ranking {dense_rank}")
        print("fused:", [(ARTICLES[i][0], round(s, 5)) for i, s in hand])
        assert [h["title"] for h in fused] == [ARTICLES[i][0] for i, _ in hand]
        assert all(abs(h["score"] - s) < 1e-12 for h, (_, s) in zip(fused, hand))
        print("Retriever.search(mode='hybrid') gives the same order and scores: True")
        del r                                                  # release the memory-mapped index files

    # 3. the generative reader's prompt
    qa = types.SimpleNamespace(templates=PROMPTS)
    prompt, lang = rag.RagQA.prompt(qa, q, [x for _, x in ARTICLES[:2]])
    print(f"reader prompt ({lang}), {len(prompt)} characters:")
    print("  " + prompt.replace("\n", "\n  "))

    # 4. the pointer's span search
    torch.manual_seed(0)
    n = 12
    s, e = torch.randn(n), torch.randn(n)
    s[4], e[6] = 4.0, 4.0                                      # a clear answer: tokens 4..6
    best = max(((i, j) for i in range(1, n) for j in range(i, min(n, i + rc_pointer.MAX_ANS))),
               key=lambda ij: float(s[ij[0]] + e[ij[1]]))
    print(f"best_span {rc_pointer.best_span(s, e, n)}, brute force {best}, "
          f"no-answer score {float(s[0] + e[0]):.3f}")
    s[0], e[0] = 9.0, 9.0                                      # position 0 now beats every span
    print(f"with position 0 at 9 + 9: best_span returns {rc_pointer.best_span(s, e, n)}")

    # 5. the real index
    meta = ROOT / "data" / "rag" / "meta.json"
    if meta.exists():
        m = json.loads(meta.read_text())
        print(f"data/rag/meta.json: {m['chunks']:,} chunks, {m['postings']:,} postings, "
              f"avgdl {m['avgdl']:.1f} hashed words, k1 {m['k1']}, b {m['b']}, {m['buckets']:,} buckets")


if __name__ == "__main__":
    main()
