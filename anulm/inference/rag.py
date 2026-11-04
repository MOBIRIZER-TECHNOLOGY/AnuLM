"""
Retrieval for AnuLM: let a 400M model look facts up instead of remembering them.

    python -m anulm rag build                          # ~405k Wikipedia articles -> data/rag/
    python -m anulm rag build-dense                    # + dense_e5s.npy, the index the demo reads
    python -m anulm rag search "What is the capital of India?"
    python -m anulm rag eval                           # does top-k contain the answer?

A model this size stores few facts (the base read ~230M tokens) but can learn
to *read*: given the right paragraph, find the answer in it. This file is the
finding. It is BM25 over paragraphs of Simple English Wikipedia (242k
articles) and Hindi Wikipedia (163k), both from `wikimedia/wikipedia`
20231101, CC BY-SA 4.0, written from scratch in numpy because the usual
packages import scipy, which Smart App Control blocks on the training box.

  * Articles are cut into ~100-word chunks on paragraph boundaries, each
    remembering its title (the title is also indexed with the chunk: "Delhi"
    is often not in a paragraph about Delhi).
  * Words are hashed to 2^24 buckets with crc32 instead of a vocabulary, so
    12 worker processes tokenise in parallel with nothing to merge. At ~2M
    distinct words the collision rate is ~6%, all between rare words.
  * The BM25 weight of every (word, chunk) posting is computed at build
    time, so a query is one weighted bincount per query word.
  * Devanagari vowel signs are word characters here -- Python's \\w is not,
    and would split every Hindi word at its matras.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]
RAG = HERE / "data" / "rag"
WIKI = RAG / "wiki" / "20231101.{lang}"
BUCKETS = 1 << 24
K1, B = 1.2, 0.75
CHUNK_WORDS = 100

WORD = re.compile(r"[0-9A-Za-zÀ-ɏऀ-ॿ]+")
SKIP_SECTIONS = ("References", "Related pages", "Other websites", "संदर्भ", "बाहरी कड़ियाँ",
                 "इन्हें भी देखें", "सन्दर्भ")


def words(text: str) -> list[str]:
    """Lower-cased words of `text`: runs of Latin letters, digits and
    Devanagari (matras included), as the index tokenises them."""
    return WORD.findall(text.lower())


def term_ids(text: str) -> np.ndarray:
    """The crc32 hash bucket of each word in `text`, int64, in order and with
    repeats."""
    return np.array([zlib.crc32(w.encode()) & (BUCKETS - 1) for w in words(text)], dtype=np.int64)


def chunk_article(title: str, text: str) -> list[str]:
    """Paragraph-aligned chunks of roughly CHUNK_WORDS words; reference-type
    sections at the end are dropped."""
    out, cur, n = [], [], 0
    for para in text.split("\n"):
        p = para.strip()
        if not p:
            continue
        if p in SKIP_SECTIONS:
            break
        w = len(p.split())
        if cur and n + w > CHUNK_WORDS * 1.5:
            out.append(" ".join(cur))
            cur, n = [], 0
        cur.append(p)
        n += w
        if n >= CHUNK_WORDS:
            out.append(" ".join(cur))
            cur, n = [], 0
    if cur and (n >= 12 or not out):
        out.append(" ".join(cur))
    return out


def _tokenise(batch):
    """Worker: [(title, chunk)] -> (local chunk index, term, tf) arrays + lengths."""
    rows, terms, tfs, lens = [], [], [], []
    for i, (title, chunk) in enumerate(batch):
        ids = term_ids(title + " " + chunk)
        lens.append(len(ids))
        if len(ids) == 0:
            continue
        u, c = np.unique(ids, return_counts=True)
        rows.append(np.full(len(u), i, dtype=np.int32))
        terms.append(u)
        tfs.append(c.astype(np.int32))
    cat = (lambda a, dt: np.concatenate(a) if a else np.zeros(0, dt))
    return cat(rows, np.int32), cat(terms, np.int64), cat(tfs, np.int32), np.array(lens, np.int32)


def build(langs=("simple", "hi"), workers: int = 12, out: Path = RAG, limit: int | None = None) -> None:
    """Build the BM25 index under `out` (data/rag/) from the Wikipedia parquet
    files of `langs` (data/rag/wiki/20231101.{lang}/, which `python -m anulm
    omni_data fetch read` downloads): chunk every article, tokenise the
    chunks in `workers` processes, and save the CSR postings with
    precomputed BM25 weights, the per-article title index, and chunks.jsonl.
    `limit` keeps the first `limit` articles of each language, for a quick
    trial build. The dense index is separate (`build_dense`)."""
    import multiprocessing as mp

    import pandas as pd
    RAG = Path(out)                       # every write below goes to `out`
    RAG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    chunks: list[tuple] = []
    titles: list[str] = []
    for lang in langs:
        files = sorted(Path(str(WIKI).format(lang=lang)).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no {WIKI.parent / ('20231101.' + lang)}/*.parquet: "
                                    "run `python -m anulm omni_data fetch read` first")
        seen = 0
        for f in files:
            if limit is not None and seen >= limit:
                break
            df = pd.read_parquet(f, columns=["title", "text"])
            if limit is not None:
                df = df.head(limit - seen)
            seen += len(df)
            for title, text in zip(df.title, df.text):
                art = len(titles)
                titles.append(title)
                for pos, c in enumerate(chunk_article(title, text)):
                    chunks.append((lang, title, c, pos, art))
        print(f"{lang}: {len(chunks):,} chunks so far ({time.time() - t0:.0f}s)", flush=True)
    pairs = [(t, c) for _, t, c, _, _ in chunks]
    size = 20000
    batches = [pairs[i:i + size] for i in range(0, len(pairs), size)]
    with mp.Pool(workers) as pool:
        parts = pool.map(_tokenise, batches)
    rows = np.concatenate([p[0] + i * size for i, p in enumerate(parts)])
    terms = np.concatenate([p[1] for p in parts])
    tfs = np.concatenate([p[2] for p in parts]).astype(np.float32)
    lens = np.concatenate([p[3] for p in parts]).astype(np.float32)
    print(f"tokenised: {len(rows):,} postings ({time.time() - t0:.0f}s)", flush=True)

    N, avgdl = len(pairs), float(lens.mean())
    df = np.bincount(terms, minlength=BUCKETS)
    idf = np.log1p((N - df + 0.5) / (df + 0.5)).astype(np.float32)
    w = idf[terms] * tfs * (K1 + 1) / (tfs + K1 * (1 - B + B * lens[rows] / avgdl))
    order = np.argsort(terms, kind="stable")
    indptr = np.zeros(BUCKETS + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(np.bincount(terms, minlength=BUCKETS))
    np.save(RAG / "indptr.npy", indptr)
    np.save(RAG / "docs.npy", rows[order].astype(np.int32))
    np.save(RAG / "weights.npy", w[order].astype(np.float16))
    np.save(RAG / "idf.npy", idf)
    # Title field, per article: which articles' titles contain each word.
    t_rows, t_terms, t_len = [], [], []
    for a, title in enumerate(titles):
        u = np.unique(term_ids(title))
        t_rows.append(np.full(len(u), a, dtype=np.int32))
        t_terms.append(u)
        t_len.append(max(len(u), 1))
    t_rows, t_terms = np.concatenate(t_rows), np.concatenate(t_terms)
    t_order = np.argsort(t_terms, kind="stable")
    t_indptr = np.zeros(BUCKETS + 1, dtype=np.int64)
    t_indptr[1:] = np.cumsum(np.bincount(t_terms, minlength=BUCKETS))
    np.save(RAG / "t_indptr.npy", t_indptr)
    np.save(RAG / "t_docs.npy", t_rows[t_order])
    np.save(RAG / "t_len.npy", np.array(t_len, dtype=np.float32))
    np.save(RAG / "chunk_art.npy", np.array([c[4] for c in chunks], dtype=np.int32))
    np.save(RAG / "chunk_pos.npy", np.array([c[3] for c in chunks], dtype=np.int32))
    with open(RAG / "chunks.jsonl", "w", encoding="utf-8") as f:
        for lang, title, c, pos, _ in chunks:
            f.write(json.dumps({"lang": lang, "title": title, "text": c}, ensure_ascii=False) + "\n")
    json.dump({"chunks": N, "postings": int(len(rows)), "avgdl": avgdl, "langs": list(langs),
               "k1": K1, "b": B, "buckets": BUCKETS}, open(RAG / "meta.json", "w"))
    print(f"built {N:,} chunks, {len(rows):,} postings in {time.time() - t0:.0f}s -> {RAG}")


E5 = RAG / "e5-small"          # intfloat/multilingual-e5-small, MIT, 118M params


class Dense:
    """Dense retrieval: multilingual-e5-small embeddings of every chunk.

    A borrowed component, like Whisper and SigLIP elsewhere in the project --
    it finds passages by meaning rather than shared words, which is where
    BM25 fails ("capital of India" ranked an ancient capital, Vaishali, first)
    and it matches across scripts: a Hindi question scores an English
    passage that answers it above a Hindi one that does not.
    """

    def __init__(self, device: str | None = None, index: Path = RAG / "dense_e5s.npy", e5: Path = E5):
        import torch
        from transformers import AutoModel, AutoTokenizer
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # fp16 on the GPU; fp32 on the CPU, where half-precision matmuls are slow.
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.torch, self.device, self.index = torch, device, Path(index)
        self.tok = AutoTokenizer.from_pretrained(str(e5))
        self.m = AutoModel.from_pretrained(str(e5)).to(device, self.dtype).eval()
        self.mat = None

    def embed(self, texts: list[str], max_len: int = 256):
        """Mean-pooled, L2-normalised e5 embeddings, (len(texts), hidden) in
        `self.dtype`. Callers add e5's "query: " / "passage: " prefixes."""
        torch = self.torch
        b = self.tok(texts, padding=True, truncation=True, max_length=max_len,
                     return_tensors="pt").to(self.device)
        with torch.no_grad():
            h = self.m(**b).last_hidden_state
        mask = b["attention_mask"][..., None]
        v = (h * mask).sum(1) / mask.sum(1)
        return torch.nn.functional.normalize(v.float(), dim=-1).to(self.dtype)

    def build(self, chunks: list[dict], batch: int = 512) -> None:
        """Embed every chunk as "passage: title. text" and save the (N, hidden)
        fp16 matrix to `self.index` (data/rag/dense_e5s.npy), rows in `chunks`
        order."""
        torch = self.torch
        t0 = time.time()
        self.index.parent.mkdir(parents=True, exist_ok=True)
        order = sorted(range(len(chunks)), key=lambda i: len(chunks[i]["text"]))   # less padding
        out = torch.empty(len(chunks), self.m.config.hidden_size, dtype=torch.float16)
        for s in range(0, len(order), batch):
            ix = order[s:s + batch]
            out[ix] = self.embed([f"passage: {chunks[i]['title']}. {chunks[i]['text']}" for i in ix]).cpu().half()
            if s % (batch * 200) == 0:
                print(f"  dense: {s:,}/{len(chunks):,} ({time.time() - t0:.0f}s)", flush=True)
        np.save(self.index, out.numpy())
        print(f"dense index: {tuple(out.shape)} in {time.time() - t0:.0f}s -> {self.index}")

    def search(self, query: str, k: int):
        """-> (chunk indices, cosine scores) of the `k` nearest chunks. Loads the
        saved matrix onto the device on first use."""
        torch = self.torch
        if self.mat is None:
            self.mat = torch.from_numpy(np.load(self.index)).to(self.device, self.dtype)
        s = (self.mat @ self.embed([f"query: {query}"])[0]).float()
        v, i = s.topk(k)
        return i.tolist(), v.tolist()


E5_HUB = "intfloat/multilingual-e5-small"
E5_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "sentencepiece.bpe.model"]


def build_dense(root: Path = RAG, out: Path | None = None, limit: int | None = None,
                device: str | None = None, batch: int | None = None) -> Path:
    """The dense index the demo's lookup reads: every line of
    `root`/chunks.jsonl (written by `build`) embedded by multilingual-e5-small
    as "passage: title. text", saved as an (N, 384) float16 matrix to `out`
    (default `root`/dense_e5s.npy), row i for chunk i. The encoder is fetched
    from the Hub into data/rag/e5-small/ (~470 MB) the first time.

    All 932,648 chunks: 716 MB, ~5 minutes on one GPU; on a CPU ~45 s per
    1,000 chunks (measured on the training box), so most of a day. `limit`
    embeds only the first `limit` chunks -- a trial run; such a file does not
    match the full chunks.jsonl and the Retriever refuses it. Its rows equal
    the shipped index's first rows (cosine 1.000 on the first 1,000)."""
    import torch
    root = Path(root)
    out = Path(out) if out else root / "dense_e5s.npy"
    if not (E5 / "model.safetensors").exists():
        from huggingface_hub import hf_hub_download
        for f in E5_FILES:
            hf_hub_download(E5_HUB, f, local_dir=str(E5))
    with open(root / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [json.loads(l) for l in itertools.islice(f, limit)]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"embedding {len(chunks):,} chunks of {root / 'chunks.jsonl'} on {device}", flush=True)
    Dense(device, out).build(chunks, batch or (512 if device.startswith("cuda") else 64))
    return out


_QUALIFIER = re.compile(r"\s*\(([^)]*)\)\s*$")
_SIDE_PAGE = re.compile(r"^lists? of |\((disambiguation|बहुविकल्पी)\)$|की सूची$", re.I)
_LETTER = "0-9A-Za-zÀ-ɏऀ-ॿ"
# e5's cosine scores for the top hits sit within a few hundredths of each
# other (0.870 for "Hamlet (1964 movie)", 0.866 for "Hamlet"), so 0.01 would
# reorder near-ties without lifting an off-topic page. Measured 2026-11-04 and
# left OFF, like TITLE_W below: retrieval recall did not move (MLQA hi R@1/3
# 26.8/40.2% -> 26.2/41.0%, SQuAD 11.2/19.2% -> 11.5/18.8%, 400 each), and
# end to end with the Omni reader it scored 22/42 demo + held-out questions
# against 25/42 without it (the reader, not the ranking, decides those).
TITLE_BONUS = 0.0


def title_match(title: str, question: str) -> int:
    """How an article title relates to the question, for `rerank`:

    +1  the title, minus a trailing "(qualifier)", is a phrase of the
        question -- in Latin script one the question capitalises, i.e. a
        name: "Hamlet" for "Who wrote Hamlet?", but not "Planet" for "the
        largest planet" -- and any qualifier is in the question too.
    -1  a side page: a list ("List of planets", "... की सूची"), a
        disambiguation page, or a page whose qualifier the question does not
        mention ("Hamlet (1964 movie)" for "Who wrote Hamlet?").
     0  anything else."""
    if _SIDE_PAGE.search(title.strip()):
        return -1
    m = _QUALIFIER.search(title)
    base, qual = (title[:m.start()], m.group(1)) if m else (title, "")
    bw = words(base)
    if not bw:
        return 0
    phrase = f"(?<![{_LETTER}])" + f"[^{_LETTER}]+".join(map(re.escape, bw)) + f"(?![{_LETTER}])"
    span = re.search(phrase, question, re.I)
    if not span or (span.group(0)[0].isascii() and not span.group(0)[0].isupper()):
        return 0
    if qual and not set(words(qual)) <= set(words(question)):
        return -1
    return 1


def rerank(question: str, hits: list[dict], bonus: float = TITLE_BONUS) -> list[dict]:
    """`hits` re-sorted by score + bonus x title_match(title, question), ties
    kept in their order. Dense retrieval ranks by meaning, so "Who wrote
    Hamlet?" put the 1964 film first; the article the question names is
    where an encyclopedia answers it."""
    key = [h["score"] + bonus * title_match(h["title"], question) for h in hits]
    return [hits[i] for i in sorted(range(len(hits)), key=lambda i: -key[i])]


NOT_FOUND = ("not found in the passages", "अनुच्छेदों में उत्तर नहीं मिला")     # make_rc.py's


def echoes(answer: str, question: str) -> bool:
    """True when `answer` says nothing the question did not: every word of it
    is a question word ("Hamlet" for "Who wrote Hamlet?", "The Taj Mahal" for
    "Who built the Taj Mahal?"). Such an answer is never right."""
    a = words(answer)
    return not a or set(a) <= set(words(question))


def cite(answer: str, hits: list[dict]) -> str:
    """The title of the first passage that contains `answer` (case-insensitive,
    words in order), else of the first passage: the page the answer came
    from, not just the top hit."""
    a = " ".join(words(answer))
    for h in hits:
        if a and f" {a} " in f" {' '.join(words(h['text']))} ":
            return h["title"]
    return hits[0]["title"] if hits else ""


class Retriever:
    """Search over the chunks `build` wrote: BM25, or dense e5 retrieval when
    dense_e5s.npy exists (the default mode is then "dense"), or a fusion of
    the two. Hits are the chunk dicts (lang, title, text) plus a score.
    `device` is where the dense e5 encoder runs; None picks CUDA when it is
    available. Callers pass their own model's device, so `--device cpu` keeps
    retrieval off a GPU that another job owns."""
    def __init__(self, root: Path = RAG, device: str | None = None):
        self.device, self.root = device, Path(root)
        self.indptr = np.load(root / "indptr.npy")
        self.docs = np.load(root / "docs.npy", mmap_mode="r")
        self.weights = np.load(root / "weights.npy", mmap_mode="r")
        with open(root / "chunks.jsonl", encoding="utf-8") as f:
            self.chunks = [json.loads(l) for l in f]
        self.n = len(self.chunks)
        self.idf = np.load(root / "idf.npy")
        self.t_indptr = np.load(root / "t_indptr.npy")
        self.t_docs = np.load(root / "t_docs.npy")
        self.t_len = np.load(root / "t_len.npy")
        self.art = np.load(root / "chunk_art.npy")
        self.lead = (np.load(root / "chunk_pos.npy") == 0).astype(np.float32)
        # Dense when its index exists: on the 20 demo questions it put the answer in
        # the top 3 every time (BM25: 15/20) and took our own reader from 0/20 to 5/20.
        # Hybrid fusion scored worse end to end (2/20): BM25's near-misses confuse the reader.
        self.mode = "dense" if (root / "dense_e5s.npy").exists() else "bm25"
        if self.mode == "dense":
            rows = np.load(root / "dense_e5s.npy", mmap_mode="r").shape[0]
            if rows != self.n:      # a --limit trial, or chunks.jsonl rebuilt since: rows would point at the wrong text
                raise ValueError(f"{root / 'dense_e5s.npy'} has {rows:,} rows but chunks.jsonl has {self.n:,}: "
                                 "rebuild it with `python -m anulm rag build-dense`")
        self.dense = None
        langs = np.array([c["lang"] for c in self.chunks])
        self._lm = {l: langs == l for l in set(langs.tolist())}

    def search(self, query: str, k: int = 3, lang: str | None = None,
               title_w: float = 0.0, lead_b: float = 0.0, mode: str | None = None) -> list[dict]:
        """mode "bm25", "dense", or "hybrid": reciprocal-rank fusion of both
        top-100 lists, sum of 1 / (60 + rank). Without `mode`, the retriever's
        own: "dense" once dense_e5s.npy exists, else "bm25". title_w and lead_b
        apply to bm25 only; dense hits pass through `rerank` from the top 20
        (a no-op while TITLE_BONUS is 0)."""
        mode = mode or self.mode
        if mode == "bm25":
            return self.bm25(query, k, lang, title_w, lead_b)
        if self.dense is None:
            e5 = self.root / "e5-small"
            self.dense = Dense(self.device, self.root / "dense_e5s.npy", e5 if e5.exists() else E5)
        di, dv = self.dense.search(query, 100 if mode == "hybrid" else max(k, 20))
        if mode == "dense":
            return rerank(query, [{**self.chunks[i], "score": v, "_i": i} for i, v in zip(di, dv)])[:k]
        fused: dict[int, float] = {}
        for rank, i in enumerate(di):
            fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank)
        for rank, h in enumerate(self.bm25(query, 100, lang, title_w, lead_b)):
            fused[h["_i"]] = fused.get(h["_i"], 0.0) + 1.0 / (60 + rank)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [{**self.chunks[i], "score": fused[i]} for i in top]

    def bm25(self, query: str, k: int = 3, lang: str | None = None,
             title_w: float = 0.0, lead_b: float = 0.0) -> list[dict]:
        """BM25 over chunk text, plus `title_w` x the idf-weighted share of the
        query's words found in the article title, plus `lead_b` for an
        article's first chunk (where an encyclopedia states its main facts)."""
        scores = np.zeros(self.n, dtype=np.float32)
        qt = set(term_ids(query).tolist())
        for t in qt:
            a, b = self.indptr[t], self.indptr[t + 1]
            if b > a:
                scores += np.bincount(self.docs[a:b], weights=self.weights[a:b].astype(np.float32),
                                      minlength=self.n).astype(np.float32)
        if title_w:
            ts = np.zeros(len(self.t_len), dtype=np.float32)
            for t in qt:
                a, b = self.t_indptr[t], self.t_indptr[t + 1]
                if b > a:
                    ts[self.t_docs[a:b]] += self.idf[t]
            scores += title_w * (ts / np.sqrt(self.t_len))[self.art]
        if lead_b:
            scores += lead_b * self.lead * (scores > 0)
        if lang in self._lm:
            scores[~self._lm[lang]] = 0
        top = np.argpartition(-scores, min(k, self.n - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [{**self.chunks[i], "score": float(scores[i]), "_i": int(i)} for i in top if scores[i] > 0]


# From `rag.py eval` and 20 demo-style questions: plain BM25 won on all three
# (title_w 2 cost 4-6 points of R@1; a lead-paragraph bonus cost more).
TITLE_W, LEAD_B = 0.0, 0.0
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def torch_no_grad():
    """torch.no_grad(), with torch imported only when it is called, so this
    module imports without torch."""
    import torch
    return torch.no_grad()


class RagQA:
    """Retrieve 3 passages, hand them to a reader checkpoint tuned by
    finetune.py on make_rc.py's data, decode the short answer greedily."""

    def __init__(self, ckpt: str, device: str = "cuda", retriever: Retriever | None = None):
        from anulm.bpe import BPE
        from dataclasses import replace
        from anulm.model import AnuLM, load_checkpoint
        ck = load_checkpoint(ckpt, "cpu")
        cfg = replace(ck["cfg"], moe_impl="grouped" if device.startswith("cuda") else "sparse")
        self.model = AnuLM(cfg).to(device).eval()
        self.model.load_state_dict(ck["model"])
        self.tok = BPE.load(cfg.tokenizer_path)
        from anulm.data.make_qa import PROMPTS
        # A checkpoint tuned before the rc-* templates existed still gets them,
        # so an untuned model can be scored as the control.
        self.templates = {**PROMPTS, **(ck.get("qa_templates") or {})}
        self.device = device
        self.r = retriever or Retriever(device=device)

    def prompt(self, question: str, passages: list[str]) -> tuple[str, str]:
        """-> (prompt, lang). Numbered passages, each cut to 120 words, then the
        question, in the checkpoint's rc-hi or rc-en template; Hindi when the
        question contains Devanagari."""
        lang = "hi" if DEVANAGARI.search(question) else "en"
        qword = "प्रश्न" if lang == "hi" else "Question"
        body = "\n\n".join(f"[{i + 1}] {' '.join(p.split()[:120])}" for i, p in enumerate(passages))
        return self.templates[f"rc-{lang}"].format(q=f"{body}\n\n{qword}: {question}"), lang

    @torch_no_grad()
    def read(self, prompt: str, max_tokens: int = 32) -> str:
        """Greedy answer to `prompt`, at most `max_tokens` tokens, first line only.
        The prompt is cut from the left to leave room for the answer."""
        import torch
        ids = self.tok.encode(prompt)[-(self.model.cfg.block_size - max_tokens):]
        x = torch.tensor([ids], device=self.device)
        ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda"))
        with ac:
            out = self.model.generate(x, max_tokens, temperature=1.0, top_k=1, eos_id=self.tok.eos_id)
        new = [t for t in out[0, len(ids):].tolist() if t != self.tok.eos_id]
        return self.tok.decode(new).strip().split("\n")[0]

    def ask(self, question: str, k: int = 3) -> dict:
        """Retrieve `k` passages and read the answer from them -> {answer, lang,
        sources (titles), retrieve_s, read_s}."""
        t0 = time.time()
        hits = self.r.search(question, k)
        prompt, lang = self.prompt(question, [h["text"] for h in hits])
        t1 = time.time()
        ans = self.read(prompt)
        return {"answer": ans, "lang": lang, "sources": [h["title"] for h in hits],
                "retrieve_s": round(t1 - t0, 3), "read_s": round(time.time() - t1, 3)}


_PUNCT = re.compile(r"[\"'()\[\],;:।.!?\-–—]")


def _norm(s: str) -> list[str]:
    s = _PUNCT.sub(" ", s.lower())
    return [w for w in s.split() if w not in ("a", "an", "the")]


def em_f1(pred: str, golds: list[str]) -> tuple[float, float]:
    """SQuAD-style exact match and token F1 of `pred` against the best of
    `golds`, after lower-casing, stripping punctuation (the danda included)
    and dropping English articles."""
    from collections import Counter
    p = _norm(pred)
    best_em = best_f1 = 0.0
    for g in golds:
        gt = _norm(g)
        best_em = max(best_em, float(p == gt))
        common = sum((Counter(p) & Counter(gt)).values())
        if common:
            pr, rc = common / len(p), common / len(gt)
            best_f1 = max(best_f1, 2 * pr * rc / (pr + rc))
    return best_em, best_f1


def eval_reader(ckpt: str, n: int = 300, device: str = "cuda") -> None:
    """Reader alone, on make_rc.py's held-out sets (right passage + 2 hard
    negatives); then the whole pipeline on data/rag_demo_questions.jsonl."""
    qa = RagQA(ckpt, device)
    for name in ("squad_en", "mlqa_hi", "xquad_hi", "indicqa_hi"):
        path = HERE / "data" / f"rc_eval_{name}.jsonl"
        rows = [json.loads(l) for l in open(path, encoding="utf-8")][:n]
        ems = f1s = abstain = 0.0
        for r in rows:
            pred = qa.read(qa.templates[r["lang"]].format(q=r["question"]))
            e, f = em_f1(pred, r["answers"])
            ems, f1s = ems + e, f1s + f
            abstain += pred.strip() in ("not found in the passages", "अनुच्छेदों में उत्तर नहीं मिला")
        k = len(rows)
        print(f"  {name:12s} EM {100 * ems / k:5.1f}  F1 {100 * f1s / k:5.1f}  "
              f"says not-found {100 * abstain / k:4.1f}%  (n={k})", flush=True)
    items = [json.loads(l) for l in open(HERE / "data" / "rag_demo_questions.jsonl", encoding="utf-8")]
    right = 0
    for it in items:
        out = qa.ask(it["q"])
        ok = any(a in out["answer"] for a in it["a"])
        right += ok
        print(f"  {'OK ' if ok else '-- '} {it['q']}  ->  {out['answer']}   [{', '.join(out['sources'][:2])}]")
    print(f"  demo questions: {right}/{len(items)} answered correctly end to end")


def eval_sets(n: int):
    """(name, [(question, [answers])]) for retrieval recall. MLQA hi.hi: real
    Hindi questions on Hindi Wikipedia paragraphs. SQuAD dev: English questions
    on (full) English Wikipedia, so an answer missing from Simple Wikipedia is
    a coverage miss, not a ranking one -- read it as a lower bound."""
    import pandas as pd
    rc = HERE / "data" / "rc"
    hi = pd.read_parquet(rc / "facebook__mlqa" / "mlqa.hi.hi" / "validation" / "0000.parquet")
    en = pd.read_parquet(rc / "rajpurkar__squad" / "plain_text" / "validation-00000-of-00001.parquet")
    en = en.sample(n=min(n, len(en)), random_state=0)
    out = []
    for name, df in (("hindi MLQA", hi.head(n)), ("english SQuAD", en)):
        out.append((name, [(q, list(a["text"])) for q, a in zip(df.question, df.answers)]))
    return out


def evaluate(n: int, grid=((0, 0), (1, 0), (2, 0), (4, 0), (2, 2), (4, 4), (2, 5))) -> None:
    """Retrieval recall: R@1/3/10 (the answer string appears in the top-k
    chunks) on `n` MLQA Hindi and SQuAD questions, for each (title_w, lead_b)
    in `grid`."""
    r = Retriever()
    for name, items in eval_sets(n):
        print(f"\n{name}: {len(items)} questions, answer string in top-k chunks")
        for tw, lb in grid:
            hit = {1: 0, 3: 0, 10: 0}
            for q, answers in items:
                hits = r.search(q, 10, title_w=tw, lead_b=lb)
                for k in hit:
                    text = " ".join(h["text"] for h in hits[:k]).lower()
                    hit[k] += any(a.lower() in text for a in answers)
            print(f"  title_w {tw:3.0f}  lead_b {lb:3.0f}   " +
                  "   ".join(f"R@{k} {100 * v / len(items):5.1f}%" for k, v in hit.items()), flush=True)


def main() -> None:
    """Command-line entry point: build, build-dense, search, eval, eval-reader and ask."""
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    bd = sub.add_parser("build", help="BM25 index + chunks.jsonl from the Wikipedia parquet files")
    bd.add_argument("--out", type=Path, default=RAG)
    bd.add_argument("--limit", type=int, help="first N articles per language (a trial build)")
    dn = sub.add_parser("build-dense", help="dense_e5s.npy, the index the demo's lookup needs")
    dn.add_argument("--root", type=Path, default=RAG, help="directory holding build's chunks.jsonl")
    dn.add_argument("--out", type=Path, help="default: ROOT/dense_e5s.npy")
    dn.add_argument("--limit", type=int, help="first N chunks only (a trial run)")
    dn.add_argument("--device", help="cuda or cpu (default: cuda when available)")
    dn.add_argument("--batch", type=int, help="default 512 on a GPU, 64 on a CPU")
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--root", type=Path, default=RAG)
    s.add_argument("--device", help="where the dense encoder runs (default: cuda when available)")
    s.add_argument("-k", type=int, default=3)
    s.add_argument("--title-w", type=float, default=TITLE_W)
    s.add_argument("--lead-b", type=float, default=LEAD_B)
    e = sub.add_parser("eval")
    e.add_argument("--n", type=int, default=400)
    rd = sub.add_parser("eval-reader")
    rd.add_argument("--ckpt", required=True)
    rd.add_argument("--n", type=int, default=300)
    ak = sub.add_parser("ask")
    ak.add_argument("question")
    ak.add_argument("--ckpt", default="ckpt_rag_reader.pt")
    a = p.parse_args()
    if a.cmd == "build":
        build(out=a.out, limit=a.limit)
    elif a.cmd == "build-dense":
        build_dense(a.root, a.out, a.limit, a.device, a.batch)
    elif a.cmd == "eval":
        evaluate(a.n)
    elif a.cmd == "eval-reader":
        eval_reader(a.ckpt, a.n)
    elif a.cmd == "ask":
        out = RagQA(a.ckpt).ask(a.question)
        print(f"{out['answer']}\n  sources: {out['sources']}  "
              f"(retrieve {out['retrieve_s']}s, read {out['read_s']}s)")
    else:
        r = Retriever(a.root, a.device)
        t0 = time.time()
        hits = r.search(a.query, a.k, title_w=a.title_w, lead_b=a.lead_b)
        print(f"{(time.time() - t0) * 1000:.0f} ms")
        for h in hits:
            print(f"\n[{h['score']:.3f}] {h['title']} ({h['lang']})\n  {h['text'][:400]}")


if __name__ == "__main__":
    main()
