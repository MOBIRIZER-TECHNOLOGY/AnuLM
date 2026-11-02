"""
Book example, not repository code.

Chapter 28: how `AnuLM.generate` turns logits into a token, watched from the
outside, and top-p sampling as an EXTENSION patch with its tests.

    python book/code-examples/ch28_sampling.py

It runs `model.py` unchanged. A subclass whose forward returns chosen logits
lets the repository's own `generate` run its sampling code; a spy on
`torch.multinomial` records the probabilities it samples from. The book's
re-statement of each step is checked against those probabilities.

Top-p is NOT IN THE REPO. `top_p_filter` below is the book's, and the patch is
applied to the text of the pinned `generate` and executed, so the test runs
the repository's loop with exactly three lines added.
"""

from __future__ import annotations

import inspect
import math
import sys
import textwrap
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm import model as M  # noqa: E402  the repository's model.py, unchanged

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def tiny(**kw) -> M.AnuLMConfig:
    """test_model.py's tiny(), copied so this script needs no test harness."""
    base = dict(vocab_size=64, block_size=32, n_layer=3, hidden_size=32, n_head=4,
                head_dim=8, n_kv_head=2, num_experts=8, num_experts_per_tok=2,
                moe_intermediate_size=16, intermediate_size=64, n_group=1, topk_group=1)
    base.update(kw)
    return M.AnuLMConfig(**base)


class FixedLogits(M.AnuLM):
    """An AnuLM whose forward returns the same logits for every row."""

    def __init__(self, z):
        super().__init__(tiny(vocab_size=z.numel()))
        self.z = z

    def forward(self, idx, targets=None, cache=None, start=0, embeds=None):
        B = (idx if idx is not None else embeds).shape[0]
        return self.z.expand(B, 1, -1).clone(), None                # (B, 1, V)


def probs_from_generate(z, prompt, cls=FixedLogits, **kw):
    """Run the repository's generate for one token; return what it sampled from."""
    seen, orig = [], torch.multinomial

    def spy(p, num_samples):                         # generate calls this, line 1105
        seen.append(p.clone())
        return orig(p, num_samples)

    torch.multinomial = spy
    try:
        cls(z).generate(torch.tensor([prompt]), 1, **kw)
    finally:
        torch.multinomial = orig
    return seen[0][0]


def book_probs(z, seen_ids, temperature=1.0, top_k=None, penalty=1.0):
    """generate's lines 1096-1104, one row at a time, as Chapter 28 states them."""
    z = z.clone().float()
    for i in set(seen_ids) if penalty != 1.0 else ():
        z[i] = z[i] / penalty if z[i] > 0 else z[i] * penalty   # generate's sign rule
    z = z / max(temperature, 1e-5)
    if top_k is not None:
        kth = z.sort(descending=True).values[min(top_k, len(z)) - 1]
        z[z < kth] = -math.inf                       # ties with the k-th survive
    return z.softmax(-1)


def show(name, p):
    print(f"{name:34s}", "  ".join(f"{v:.3f}" for v in p.tolist()))


def by_hand():
    z = torch.tensor([2.0, 1.0, 0.5, -1.0])          # illustrative logits, V = 4
    prompt = [0, 3]                                  # tokens 0 and 3 already seen
    cases = [("T = 1", {}), ("T = 0.5", dict(temperature=0.5)),
             ("T = 2", dict(temperature=2.0)), ("T = 0 (floored at 1e-5)", dict(temperature=0.0)),
             ("top_k = 2", dict(top_k=2)), ("repetition_penalty = 1.3", dict(penalty=1.3))]
    for name, kw in cases:
        gen_kw = {("repetition_penalty" if k == "penalty" else k): v for k, v in kw.items()}
        p = probs_from_generate(z, prompt, **gen_kw)
        assert torch.allclose(p, book_probs(z, prompt, **kw), atol=1e-6), name
        show(name, p)
    tie = torch.tensor([1.0, 1.0, 1.0, 0.0])
    p = probs_from_generate(tie, [0], top_k=2)
    show("ties, top_k = 2", p)
    assert int((p > 0).sum()) == 3
    pen = book_probs(z, prompt, penalty=1.3).log() - book_probs(z, prompt).log()
    print("logit change from the penalty     ", "  ".join(f"{v:+.3f}" for v in (pen - pen[1]).tolist()))


# ------------------------------------------------------------- EXTENSION: top-p
def top_p_filter(logits, p):
    """EXTENSION, not in the repository. Keep the smallest set of top tokens whose
    probability reaches p, plus any token tied with the last one kept (top-k's
    rule); the top token always survives. The rest become -inf. logits: (B, V)."""
    if p >= 1.0:
        return logits
    sorted_z = logits.sort(dim=-1, descending=True).values
    probs = sorted_z.softmax(-1)
    keep = probs.cumsum(-1) - probs < p              # mass ranked above is short of p
    keep[..., 0] = True                              # p <= 0 still keeps the top token
    cut = sorted_z.masked_fill(~keep, math.inf).min(-1, keepdim=True).values
    return logits.masked_fill(logits < cut, -math.inf)   # ties with the cut survive


PATCH = (("repetition_penalty: float = 1.0, on_token=None):",
          "repetition_penalty: float = 1.0, on_token=None, top_p: float | None = None):"),
         ("        probs = F.softmax(logits, dim=-1)\n",       # as indented after dedent
          "        if top_p is not None:\n"
          "            logits = top_p_filter(logits, top_p)\n"
          "        probs = F.softmax(logits, dim=-1)\n"))


def patched_generate():
    """The pinned generate's own source with PATCH applied, compiled in model.py's
    namespace."""
    src = textwrap.dedent(inspect.getsource(M.AnuLM.generate))
    for old, new in PATCH:
        assert src.count(old) == 1, f"patch does not apply: {old!r}"
        src = src.replace(old, new)
    ns = dict(vars(M), top_p_filter=top_p_filter)
    exec(src, ns)
    return ns["generate"]


class TopPAnuLM(M.AnuLM):
    generate = patched_generate()


class FixedLogitsTopP(FixedLogits):
    generate = patched_generate()


# the two classic bugs, as probability functions
def right(z, p):
    return top_p_filter(z[None], p)[0].softmax(-1)


def off_by_one(z, p):                                # drops the token that crosses p
    s, order = z.sort(descending=True)
    keep = s.softmax(-1).cumsum(-1) <= p
    s = s.masked_fill(~keep, -math.inf)
    return torch.empty_like(z).scatter(0, order, s).softmax(-1)


def not_renormalised(z, p):                          # zeroes the tail after softmax
    probs = z.softmax(-1)
    s, order = probs.sort(descending=True)
    s = s.masked_fill(s.cumsum(-1) - s >= p, 0.0)
    return torch.empty_like(probs).scatter(0, order, s)


def failures(fn):
    """The EXTENSION's test: which of its checks fn fails. It calls the filter
    function directly; only test_patch runs the patched generate."""
    z = torch.tensor([0.5, 0.3, 0.15, 0.05]).log()   # probabilities 0.5, 0.3, 0.15, 0.05
    tie = torch.tensor([0.4, 0.3, 0.3]).log()        # a tie at the boundary
    out = []
    for zz, p, keep in ((z, 0.3, {0}), (z, 0.7, {0, 1}), (z, 0.85, {0, 1, 2}),
                        (z, 0.0, {0}), (tie, 0.6, {0, 1, 2})):
        got = fn(zz, p)
        if set((got > 0).nonzero().flatten().tolist()) != keep:
            out.append(f"p={p}: kept {sorted((got > 0).nonzero().flatten().tolist())}")
        if abs(got.sum().item() - 1) > 1e-6:
            out.append(f"p={p}: sums to {got.sum().item():.3f}")
    if not torch.equal(fn(z, 1.0), z.softmax(-1)):
        out.append("p=1 changes the distribution")
    return out


def test_patch():
    z = torch.tensor([0.5, 0.3, 0.15, 0.05]).log()
    p = probs_from_generate(z, [0], cls=FixedLogitsTopP, top_p=0.7)
    show("patched generate, top_p = 0.7", p)
    assert torch.allclose(p, torch.tensor([0.625, 0.375, 0.0, 0.0]), atol=1e-6)
    torch.manual_seed(0)
    m = M.AnuLM(tiny()).eval()
    m2 = TopPAnuLM(tiny()).eval()
    m2.load_state_dict(m.state_dict())
    start = torch.randint(0, 64, (2, 4))
    torch.manual_seed(1)
    a = m.generate(start.clone(), 12)
    torch.manual_seed(1)
    b = m2.generate(start.clone(), 12)               # top_p=None: must change nothing
    print("patched generate with top_p=None reproduces the original:", torch.equal(a, b))
    assert torch.equal(a, b)
    for name, fn in (("right", right), ("off_by_one", off_by_one),
                     ("not_renormalised", not_renormalised)):
        f = failures(fn)
        print(f"{name:17s}", "passes" if not f else "FAILS: " + "; ".join(f))
    assert not failures(right) and failures(off_by_one) and failures(not_renormalised)


def ngram_block():
    """decode_from_embeds on fixed logits: greedy would repeat token 0 for ever."""
    m = FixedLogits(torch.tensor([3.0, 2.0, 1.0, 0.0]))
    prefix = torch.zeros(2, m.cfg.hidden_size)       # (T, D): stands in for audio or image
    for n in (0, 2, 3):
        out = m.decode_from_embeds(prefix, 8, stop=lambda i: False, no_repeat_ngram=n)
        print(f"no_repeat_ngram = {n}:", out)


def exercises():
    p1 = book_probs(torch.tensor([3.0, 3.0, 1.0]), [], temperature=1e-3)
    assert torch.allclose(p1, probs_from_generate(torch.tensor([3.0, 3.0, 1.0]), [0], temperature=1e-3))
    z = torch.tensor([4.0, -4.0, 0.5])
    p2 = book_probs(z, [0, 1], penalty=2.0).log() - book_probs(torch.tensor([2.0, -8.0, 0.5]), []).log()
    print("exercise 1:", [round(v, 3) for v in p1.tolist()],
          "  exercise 2: penalty 2 equals logits [2, -8]:", bool(p2.abs().max() < 1e-5))


if __name__ == "__main__":
    print("== generate's distribution, logits [2.0, 1.0, 0.5, -1.0], tokens 0 and 3 seen")
    by_hand()
    print("\n== decode_from_embeds, fixed logits [3, 2, 1, 0]")
    ngram_block()
    exercises()
    print("\n== EXTENSION: top-p")
    test_patch()
