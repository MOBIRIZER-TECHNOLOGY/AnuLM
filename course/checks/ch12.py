"""
Chapter 12 checks: decisions (Decide) and lookup (BM25).

    python ch12.py                  # your course/build/post.py
    python ch12.py --impl solution  # the reference solution
    python ch12.py --self-test      # proves each check catches the bug it is for

A decision model that reads the wrong position still trains; a calibration
number computed wrongly still looks like a number; a search engine that splits
Hindi words in half still returns passages. These checks hold each piece to
decide.py and rag.py.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import main, solution, with_replaced  # noqa: E402
from anulm.tasks import decide  # noqa: E402
from anulm.inference import rag  # noqa: E402
from anulm.bpe import BPE  # noqa: E402

CHECKS = []
B = solution(8)
TOK = BPE.train(("Message: my card has not arrived yet\nWhich of these does it ask for?\nOptions:\n- card arrival"
                 "\n- lost or stolen card\n- exchange rate ") * 20, 400, verbose=False)
KW = dict(vocab_size=64, block_size=48, n_layer=2, hidden_size=32, n_head=4, head_dim=8, n_kv_head=2,
          num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16, intermediate_size=64,
          rope_theta=10000.0)
DOCS = ["भारत की राजधानी नई दिल्ली है। दिल्ली एक बड़ा शहर है।",
        "मुंबई महाराष्ट्र की राजधानी है और भारत का सबसे बड़ा शहर है।",
        "The capital of India is New Delhi, a city of many millions.",
        "Romeo and Juliet is a play. Romeo loves Juliet; Juliet loves Romeo.",
        "India India India India India India India India India India India India India India India",
        "A short one."]


def check(fn):
    CHECKS.append(fn)
    return fn


@check
def options_are_encoded_like_decide(M):
    """Same ids and same positions as decide.encode: the score is read at each option's last token."""
    for text, opts in (("  my card has not arrived yet ", ["card_arrival", "lost_or_stolen_card", "exchange_rate"]),
                       ("Which school is this?", ["EducationalInstitution", "card_arrival"])):
        want = decide.encode(TOK, text, opts)
        got = M.encode_options(TOK.encode, TOK.eos_id, text, opts)
        assert list(got[0]) == list(want[0]), f"ids differ from decide.encode for {opts}"
        assert list(got[1]) == list(want[1]), f"positions {got[1]}, decide.encode's {want[1]}"
    assert M.readable("EducationalInstitution") == "educational institution", "readable() splits names wrongly"


@check
def each_score_is_read_at_its_option(M):
    """logits[b, k] is the head applied to the hidden state at pos[b, k], and nothing else."""
    torch.manual_seed(0)
    m = B.MoELM(**KW).eval()
    d = M.Decider(m, KW["hidden_size"]).eval()
    with torch.no_grad():
        d.head.weight.normal_()
        d.head.bias.fill_(0.3)
        idx = torch.randint(1, 64, (2, 20))
        pos = torch.tensor([[3, 9, 19], [5, 6, 11]])
        h = []
        hook = m.norm.register_forward_hook(lambda mod, i, o: h.append(o))
        m(idx)
        hook.remove()
        want = (h[0].gather(1, pos[..., None].expand(-1, -1, 32)) @ d.head.weight.T).squeeze(-1) + 0.3
        got = d(idx, pos)
    torch.testing.assert_close(got, want.float(), rtol=1e-4, atol=1e-5, msg="scores not read at the options' positions")
    assert d.temperature.item() == 1.0 and "temperature" in dict(d.named_buffers()), "no temperature buffer of 1"


@check
def missing_options_get_no_probability(M):
    """In a batch with fewer options in some rows, the padded slots are -inf: probability 0."""
    torch.manual_seed(0)
    d = M.Decider(B.MoELM(**KW).eval(), KW["hidden_size"]).eval()
    idx = torch.randint(1, 64, (2, 12))
    pos = torch.tensor([[2, 7, 11], [4, 9, 0]])
    mask = torch.tensor([[True, True, True], [True, True, False]])
    with torch.no_grad():
        p = F.softmax(d(idx, pos, mask), -1)
    assert p[1, 2].item() == 0.0 and abs(p[1].sum().item() - 1) < 1e-6, "a padded option still has probability"


def overconfident(seed=0, n=600, k=5):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(k, (n,), generator=g)
    logits = torch.randn(n, k, generator=g)
    logits[torch.arange(n), y] += 1.5
    return logits * 3.0, y                                    # right 3 times in 4, far too sure


@check
def temperature_matches_decide(M):
    logits, y = overconfident()
    t = M.fit_temperature(logits, y)
    assert t == decide.fit_temperature(logits, y), f"T = {t}, decide.py finds {decide.fit_temperature(logits, y)}"
    assert t > 1.5, f"T = {t} for an overconfident model; it should soften (T > 1)"


@check
def calibration_matches_decide(M):
    """Accuracy, ECE, Brier and the coverage table, against decide.metrics, before and after T."""
    logits, y = overconfident(1)
    for t in (1.0, 2.5):
        p = F.softmax(logits / t, -1)
        got, want = M.calibration(p, y), decide.metrics(p, y)
        for key in ("accuracy", "ece", "brier", "mean_confidence"):
            assert abs(got[key] - want[key]) < 1e-6, f"{key} {got[key]:.4f}, decide.metrics {want[key]:.4f} (T = {t})"
        for tau in (0.5, 0.7, 0.9):
            a, b = got[f"answer_if_conf>={tau}"], want[f"answer_if_conf>={tau}"]
            assert abs(a["coverage"] - b["coverage"]) < 1e-6, f"coverage at {tau} differs from decide.metrics"


@check
def a_calibrated_model_has_small_ece(M):
    """When confidence really is the chance of being right, ECE is near 0."""
    g = torch.Generator().manual_seed(2)
    n, k = 20000, 4
    p = F.softmax(torch.randn(n, k, generator=g) * 2, -1)
    y = torch.multinomial(p, 1, generator=g).squeeze(1)       # labels drawn FROM the probabilities
    assert M.calibration(p, y)["ece"] < 0.02, "ECE is large for a model that is calibrated by construction"


@check
def hindi_words_stay_whole(M):
    """Devanagari vowel signs are part of a word. Python's \\w is not, and splits every Hindi word."""
    assert M.words("भारत की राजधानी नई दिल्ली है") == ["भारत", "की", "राजधानी", "नई", "दिल्ली", "है"], \
        f"Hindi split wrongly: {M.words('भारत की राजधानी')}"
    assert M.words("Romeo, JULIET's 2 cafés") == rag.words("Romeo, JULIET's 2 cafés"), "English or Latin words differ from rag.words"


def rag_scores(docs, query):
    """rag.py's own index-building and search arithmetic, on a small set of passages."""
    rows, terms, tfs, lens = rag._tokenise([("", d) for d in docs])
    tfs, lens = tfs.astype(np.float32), lens.astype(np.float32)
    df = np.bincount(terms, minlength=rag.BUCKETS)
    idf = np.log1p((len(docs) - df + 0.5) / (df + 0.5))
    w = idf[terms] * tfs * (rag.K1 + 1) / (tfs + rag.K1 * (1 - rag.B + rag.B * lens[rows] / lens.mean()))
    s = np.zeros(len(docs))
    for t in set(rag.term_ids(query).tolist()):
        s += np.bincount(rows[terms == t], weights=w[terms == t], minlength=len(docs))
    return s


@check
def bm25_scores_match_rag(M):
    index = M.BM25(DOCS)
    for q in ("भारत की राजधानी", "capital of India", "Romeo Romeo Juliet", "India", "nothing here"):
        got = index.scores(q).double().numpy()
        want = rag_scores(DOCS, q)
        assert np.allclose(got, want, rtol=1e-4, atol=1e-5), f"{q!r}: {np.round(got, 3)} vs rag.py {np.round(want, 3)}"


@check
def search_returns_the_best_first(M):
    index = M.BM25(DOCS)
    assert index.search("राजधानी दिल्ली", 2)[0] == 0, "the passage about Delhi is not first"
    assert index.search("Romeo", 5) == [3], "search returns passages with no matching word"


# --- broken versions for --self-test ----------------------------------------------------

NAMES = ("readable", "encode_options", "final_hidden", "Decider", "fit_temperature", "calibration",
         "WORD", "words", "BM25")


def MUTANTS(sol):
    def first_token_positions(encode, bos, text, options):
        ids = [bos] + encode(f"Message: {text.strip()}\nWhich of these does it ask for?\nOptions:")
        pos = []
        for o in options:
            pos.append(len(ids))
            ids += encode(f"\n- {sol.readable(o)}")
        return ids, pos

    class LastTokenDecider(sol.Decider):
        def forward(self, idx, pos, mask=None):
            return super().forward(idx, torch.full_like(pos, idx.shape[1] - 1), mask)

    class NoMask(sol.Decider):
        def forward(self, idx, pos, mask=None):
            return super().forward(idx, pos, None)

    def nll_on_probs(logits, y):
        best = (float("inf"), 1.0)
        for t in [x / 100 for x in range(30, 501, 5)]:
            best = min(best, (F.nll_loss(F.softmax(logits / t, -1), y).item(), t))
        return best[1]

    def ece_unweighted(probs, y, bins=15):
        out = sol.calibration(probs, y, bins)
        conf, pred = probs.max(1)
        correct = (pred == y).float()
        gaps = [abs(conf[m].mean().item() - correct[m].mean().item()) for b in range(bins)
                for m in [(conf > b / bins) & (conf <= (b + 1) / bins)] if m.any()]
        out["ece"] = sum(gaps) / len(gaps)
        return out

    def brier_top_only(probs, y, bins=15):
        out = sol.calibration(probs, y, bins)
        conf, pred = probs.max(1)
        out["brier"] = ((conf - (pred == y).float()) ** 2).mean().item()
        return out

    W = re.compile(r"\w+")

    def w_words(text):
        return W.findall(text.lower())

    def no_lower(text):
        return sol.WORD.findall(text)

    class PlainIdf(sol.BM25):
        def __init__(self, docs, k1=1.2, b=0.75, tokenize=None):
            super().__init__(docs, k1, b, tokenize)
            n = len(self.tf)
            for w in self.idf:
                d = sum(1 for c in self.tf if w in c)
                self.idf[w] = math.log((n - d + 0.5) / (d + 0.5))

    class NoLengthNorm(sol.BM25):
        def __init__(self, docs, k1=1.2, b=0.75, tokenize=None):
            super().__init__(docs, k1, 0.0, tokenize)

    class RepeatsCount(sol.BM25):
        def scores(self, query):
            s = torch.zeros(len(self.tf), dtype=torch.float64)
            for w in self.tokenize(query):
                s += sol.BM25.scores(self, w)
            return s

    class RawTf(sol.BM25):
        def scores(self, query):
            s = torch.zeros(len(self.tf), dtype=torch.float64)
            for w in set(self.tokenize(query)):
                if w in self.idf:
                    s += self.idf[w] * torch.tensor([c.get(w, 0) for c in self.tf], dtype=torch.float64)
            return s

    class ZerosToo(sol.BM25):
        def search(self, query, k=3):
            return [int(i) for i in torch.argsort(-self.scores(query), stable=True)[:k]]

    bugs = [("score read at each option's first token", dict(encode_options=first_token_positions)),
            ("every option scored at the sequence's end", dict(Decider=LastTokenDecider)),
            ("padded options not masked", dict(Decider=NoMask)),
            ("temperature fitted on probabilities", dict(fit_temperature=nll_on_probs)),
            ("ECE bins not weighted by size", dict(calibration=ece_unweighted)),
            ("Brier score on the top option only", dict(calibration=brier_top_only)),
            ("\\w+ splitting Hindi at its vowel signs", dict(words=w_words)),
            ("words not lower-cased", dict(words=no_lower)),
            ("idf without log1p (negative for common words)", dict(BM25=PlainIdf)),
            ("no length normalisation", dict(BM25=NoLengthNorm)),
            ("a repeated query word counted twice", dict(BM25=RepeatsCount)),
            ("raw term frequency, no saturation", dict(BM25=RawTf)),
            ("passages with score 0 returned", dict(BM25=ZerosToo))]
    return [(d, with_replaced(sol, NAMES, **r)) for d, r in bugs]


if __name__ == "__main__":
    main(12, CHECKS, MUTANTS, __doc__, learner="post.py")
