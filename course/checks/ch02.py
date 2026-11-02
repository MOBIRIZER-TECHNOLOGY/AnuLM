"""
Chapter 2 checks: the data pipeline.

    python ch02.py                  # your course/build/data.py
    python ch02.py --impl solution  # the reference solution
    python ch02.py --self-test      # proves each check catches the bug it is for

Data bugs are the quietest of all: nothing crashes, the model trains, and it
learns from duplicates, from list debris, or from the very benchmark it is
later scored on.
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import main, with_replaced  # noqa: E402
from anulm import bpe  # noqa: E402
from anulm.data import convert_parquet  # noqa: E402
from anulm.data import fetch_hindi  # noqa: E402
from anulm.data import filter_prose  # noqa: E402
from anulm.training import train as ref_train  # noqa: E402

CHECKS = []
HINDI = ("भारत दक्षिण एशिया में स्थित एक विशाल देश है जिसकी राजधानी नई दिल्ली है और जहाँ अनेक "
         "भाषाएँ बोली जाती हैं तथा अनेक धर्मों के लोग एक साथ रहते हैं।")
ENGLISH = ("the quick brown fox jumps over the lazy dog while the cat sleeps on the warm "
           "windowsill and the rain falls softly on the old stone roof of the house")


def check(fn):
    CHECKS.append(fn)
    return fn


@check
def split_documents_matches_bpe(M):
    text = "first doc\nline\n\n\nsecond doc\n\n\n   \n\n\nthird"
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.txt"
        p.write_text(text, encoding="utf-8")
        want = list(bpe.iter_documents(str(p)))
    assert M.split_documents(text) == want, f"{M.split_documents(text)!r} != bpe.iter_documents' {want!r}"


@check
def dedup_keeps_the_first_copy_in_order(M):
    docs = ["a", "b", "a", "c", "b", "a"]
    kept, removed = M.exact_dedup(docs)
    assert kept == ["a", "b", "c"] and removed == 3, f"got {kept}, {removed} removed; want ['a','b','c'], 3"


@check
def script_share_matches_fetch_hindi(M):
    for s in ("", "abc", HINDI, "मैं office जा रहा हूँ", "Hello भारत!", "café naïve — தமிழ் भारत"):   # other non-ASCII too
        assert abs(M.script_share(s) - fetch_hindi.devanagari_ratio(s)) < 1e-12, f"script_share({s!r}) differs"


DOCS = [
    "Title\n\n" + "This is a sentence that ends properly. " * 20 + "\nshort list item\nAnother line.\n",
    "Station list\n\nDelhi\nAgra\nKanpur\nLucknow\nVaranasi\nPatna\n",
    "Short\n\nOne sentence only.",
    "शीर्षक\n\n" + "यह एक पूरा वाक्य है। " * 40 + "\nसूची\nएक और वाक्य।",
    "Mixed\n\n" + ("A long line without any ending mark that runs on for more than forty characters\n" * 8)
    + "Then a sentence. " * 30,
]


@check
def filter_doc_matches_filter_prose(M):
    """The prose filter: the title stays, headings and list debris go, list-shaped documents go."""
    for i, d in enumerate(DOCS):
        for kw in (dict(), dict(min_line_chars=20, min_sentence_frac=0.6, min_chars=200)):
            want = filter_prose.filter_doc(d, kw.get("min_line_chars", 40), kw.get("min_sentence_frac", 0.4),
                                           kw.get("min_chars", 600))
            got = M.filter_doc(d, **kw)
            assert got == want, f"document {i} with {kw or 'defaults'}: differs from filter_prose.filter_doc"


@check
def ngrams_match_the_repo_on_english(M):
    """On English, your n-grams and convert_parquet.ngrams agree."""
    text = ("The Quick brown fox, it's said, jumps over the lazy dog; while the cat sleeps on the warm "
            "windowsill. And the rain (soft, grey) falls on the old stone roof of the house!")
    mine = {hash(g) for g in M.word_ngrams(text, 13)}
    assert mine == convert_parquet.ngrams(text),         "13-grams differ from convert_parquet.ngrams on English with punctuation and capitals"


@check
def ngrams_see_hindi(M):
    """The repository's 13-gram pattern, [a-z0-9_]+, finds no words in Hindi at all. Yours must."""
    words = len(HINDI.split())
    got = M.word_ngrams(HINDI, 13)
    assert len(got) >= words - 12 - 1 and len(got) > 0, \
        f"{len(got)} 13-grams in a {words}-word Hindi sentence: Devanagari words are being dropped"


@check
def contamination_finds_planted_copies_in_both_scripts(M):
    """Plant one English and one Hindi benchmark item inside training documents: both must be found,
    and an item that was never planted must not be."""
    other = "a completely different sentence about mountains rivers forests and the many birds that live in them every year"
    train = ["some text before. " + ENGLISH + " and some after.", "पहले कुछ। " + HINDI + " बाद में कुछ।", "unrelated"]
    found = M.contaminated(train, [ENGLISH, HINDI, other], 13)
    assert found == [0, 1], f"found {found}, expected [0, 1] (English and Hindi planted, the third clean)"


@check
def interleaved_split_matches_train(M):
    data = torch.arange(1000)
    for chunk, every in ((37, 10), (100, 4)):
        a, b = M.interleaved_split(data, chunk, every), ref_train.interleaved_split(data, chunk, every)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), f"differs from train.interleaved_split ({chunk}, {every})"


@check
def validation_is_not_the_tail(M):
    """A corpus built source by source (all Hindi, then all English) has a tail that is one source.
    The validation split must carry the whole mix."""
    corpus = HINDI * 50 + ENGLISH * 50
    _, val = M.interleaved_split(corpus, 200, 10)
    share = fetch_hindi.devanagari_ratio(val)
    assert 0.2 < share < 0.8, f"validation is {share:.0%} Devanagari: it is not the corpus's mix"


# --- broken versions for --self-test -------------------------------------------------------

NAMES = ("split_documents", "exact_dedup", "script_share", "filter_doc", "word_ngrams", "contaminated",
         "interleaved_split")


def MUTANTS(sol):
    import re

    def keeps_blank(text):
        return text.split(sol.DOC_SEP)

    def keeps_last(docs):
        seen, out = {}, []
        for d in docs:
            seen[hashlib.sha1(d.encode()).digest()] = d
        kept = list(seen.values())
        return kept[::-1] if len(kept) > 1 else kept, len(docs) - len(kept)

    def non_ascii_share(text, lo="ऀ", hi="ॿ"):
        return sum(1 for ch in text if ord(ch) > 127) / len(text) if text else 0.0

    def drops_title(doc, min_line_chars=40, min_sentence_frac=0.4, min_chars=600):
        out = sol.filter_doc(doc, min_line_chars, min_sentence_frac, min_chars)
        return None if out is None else out.partition("\n\n")[2]

    def keeps_debris(doc, min_line_chars=40, min_sentence_frac=0.4, min_chars=600):
        return sol.filter_doc(doc, 0, min_sentence_frac, min_chars)

    def ascii_ngrams(text, n=13):
        w = re.findall(r"[a-z0-9_]+", text.lower())
        return {tuple(w[i:i + n]) for i in range(max(len(w) - n + 1, 0))}

    def whitespace_ngrams(text, n=13):
        w = text.lower().split()
        return {tuple(w[i:i + n]) for i in range(max(len(w) - n + 1, 0))}

    def contaminated_with(ngrams):
        def f(train_docs, bench_items, n=13):
            seen = set()
            for d in train_docs:
                seen |= ngrams(d, n)
            return [i for i, item in enumerate(bench_items) if ngrams(item, n) & seen]
        return f

    def tail_split(data, chunk, every=10):
        cut = len(data) - len(data) // every
        return data[:cut], data[cut:]

    def first_chunk_split(data, chunk, every=10):
        chunks = [data[i:i + chunk] for i in range(0, len(data), chunk)]
        val = [c for i, c in enumerate(chunks) if i % every == 0]
        train = [c for i, c in enumerate(chunks) if i % every != 0]
        j = (lambda cs: "".join(cs)) if isinstance(data, str) else (lambda cs: torch.cat(cs))
        return j(train), j(val)

    bugs = [("blank documents kept", dict(split_documents=keeps_blank)),
            ("dedup keeping the last copy, reordered", dict(exact_dedup=keeps_last)),
            ("script share counting any non-ASCII", dict(script_share=non_ascii_share)),
            ("prose filter dropping the title", dict(filter_doc=drops_title)),
            ("prose filter keeping list debris", dict(filter_doc=keeps_debris)),
            ("ASCII-only words (the repository's pattern)", dict(word_ngrams=ascii_ngrams,
                                                                 contaminated=contaminated_with(ascii_ngrams))),
            ("words split on whitespace only", dict(word_ngrams=whitespace_ngrams)),
            ("validation from the tail", dict(interleaved_split=tail_split)),
            ("validation chunks offset by every // 2", dict(interleaved_split=first_chunk_split))]
    return [(d, with_replaced(sol, NAMES, **r)) for d, r in bugs]


if __name__ == "__main__":
    main(2, CHECKS, MUTANTS, __doc__, learner="data.py")
