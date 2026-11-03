"""
Book example, not repository code.

Chapter 41, checkpoint 3: top-p (nucleus) sampling, the one sampling rule
`AnuLM.generate` does not have, written as an extension and held to three tests.

    python book/code-examples/ch41_top_p.py

Top-p keeps the smallest set of most likely tokens whose probabilities add up
to at least p, renormalises them, and samples from that set. The script:

1. defines `top_p`, the extension, and `generate_top_p`, a copy of
   `AnuLM.generate`'s uncached path with `top_p` added after the softmax;
2. defines two known-wrong filters, each with one of the classic bugs: the
   threshold off by one, and the renormalisation forgotten;
3. runs three tests against the right filter and against both wrong ones.
   Each wrong filter must fail at least one test, or the tests are too weak;
4. checks `generate_top_p` against the repository's own `generate` at the two
   limits where they must agree: p = 1 (no filtering) and p close to 0 (greedy).

It imports `anulm/model.py` unchanged.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from anulm import model as M  # noqa: E402  the repository's anulm/model.py, unchanged
from trace_forward import tiny  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------- the extension
def top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """(B, V) probabilities -> (B, V) probabilities of the nucleus: the smallest set of
    most likely tokens whose mass reaches p, renormalised; zero everywhere else."""
    if p >= 1.0:
        return probs
    sorted_probs, order = torch.sort(probs, dim=-1, descending=True)        # (B, V) each
    mass_before = sorted_probs.cumsum(dim=-1) - sorted_probs   # mass ranked above each token
    drop = mass_before >= p                    # the set already reached p without this token
    drop[:, 0] = False                         # always keep the most likely token
    sorted_probs = sorted_probs.masked_fill(drop, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)     # renormalise
    return torch.empty_like(probs).scatter_(-1, order, sorted_probs)       # vocabulary order


@torch.no_grad()
def generate_top_p(model, idx, max_new_tokens, temperature=1.0, top_k=None, p=1.0):
    """AnuLM.generate's uncached path (no cache, no text_vocab mask, no penalty, no EOS), plus top_p."""
    bs = model.cfg.block_size
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -bs:])
        logits = logits[:, -1, :].float()
        if temperature <= 0:
            nxt = logits.argmax(dim=-1, keepdim=True)                        # (B, 1), no draw
        else:
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            probs = top_p(probs, p)                                          # the one new line
            nxt = torch.multinomial(probs, num_samples=1)                    # (B, 1)
        idx = torch.cat((idx, nxt), dim=1)
    return idx


# --------------------------------------------------------------------------- two known-wrong versions
def wrong_off_by_one(probs, p):
    """Drops the token that crosses the threshold, so the kept mass falls short of p."""
    sorted_probs, order = torch.sort(probs, dim=-1, descending=True)
    sorted_probs = sorted_probs.masked_fill(sorted_probs.cumsum(dim=-1) > p, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    return torch.empty_like(probs).scatter_(-1, order, sorted_probs)


def wrong_no_renorm(probs, p):
    """The right set, but the tail is zeroed and the rest is not divided by the kept mass."""
    keep = top_p(probs, p) > 0
    return probs * keep


# --------------------------------------------------------------------------- the tests
TOY = torch.tensor([[0.50, 0.25, 0.15, 0.10]])          # four tokens, already in order


def test_keeps_the_smallest_set_reaching_p(f):
    """p = 0.6 needs tokens 0 and 1 (0.50 < 0.6 <= 0.75); p = 0.8 needs a third."""
    for p, want in ((0.4, [0]), (0.6, [0, 1]), (0.8, [0, 1, 2]), (0.95, [0, 1, 2, 3])):
        kept = (f(TOY, p)[0] > 0).nonzero().flatten().tolist()
        assert kept == want, f"p={p}: kept {kept}, want {want}"


def test_renormalises_within_the_nucleus(f):
    """The kept probabilities sum to 1 and keep their ratios to one another."""
    q = f(TOY, 0.6)[0]
    assert abs(q.sum().item() - 1.0) < 1e-6, f"the nucleus sums to {q.sum():.4f}"
    assert abs(q[0].item() - 0.50 / 0.75) < 1e-6 and abs(q[1].item() - 0.25 / 0.75) < 1e-6, \
        f"ratios changed: {q.tolist()}"


def nucleus_by_loop(probs: list[float], p: float) -> dict[int, float]:
    """The oracle, written independently of top_p: walk the tokens from most to least
    likely, adding each one until the kept mass reaches p, then renormalise."""
    kept, mass = {}, 0.0
    for i in sorted(range(len(probs)), key=lambda i: -probs[i]):
        if mass >= p:
            break
        kept[i] = probs[i]
        mass += probs[i]
    return {i: q / mass for i, q in kept.items()}


def test_samples_match_the_nucleus(f):
    """20,000 draws from each of three random 50-token distributions land where the oracle says."""
    for seed in (0, 1, 2):
        g = torch.Generator().manual_seed(seed)
        probs = F.softmax(torch.randn(1, 50, generator=g) * 2, dim=-1)
        oracle = nucleus_by_loop(probs[0].tolist(), 0.9)
        want = torch.zeros(50)
        want[list(oracle)] = torch.tensor(list(oracle.values()))
        draws = torch.multinomial(f(probs, 0.9).expand(4, -1), 5000, replacement=True, generator=g).flatten()
        got = torch.bincount(draws, minlength=50).float() / draws.numel()
        assert (got[want == 0] == 0).all(), f"seed {seed}: a token outside the nucleus was sampled"
        assert (got[want >= 0.002] > 0).all(), f"seed {seed}: a nucleus token was never sampled"
        assert (got - want).abs().max() < 0.015, f"seed {seed}: frequencies off by {(got - want).abs().max():.3f}"


TESTS = (test_keeps_the_smallest_set_reaching_p, test_renormalises_within_the_nucleus,
         test_samples_match_the_nucleus)


def run_tests(f) -> list[str]:
    failed = []
    for t in TESTS:
        try:
            t(f)
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed.append(t.__name__)
            print(f"  FAIL  {t.__name__}: {e}")
    return failed


def agrees_with_generate() -> None:
    """At p = 1 and at p -> 0 the extension must reproduce the repository's generate; just
    below 1, where the filter really runs, it must leave a model's distribution unchanged."""
    torch.manual_seed(0)
    m = M.AnuLM(tiny()).eval()
    start = torch.randint(0, 64, (2, 4))
    torch.manual_seed(1)
    ref = m.generate(start.clone(), 12, temperature=1.0, use_cache=False)
    torch.manual_seed(1)
    ours = generate_top_p(m, start.clone(), 12, temperature=1.0, p=1.0)
    assert torch.equal(ref, ours), "p = 1 must leave sampling exactly as generate does it"
    greedy = m.generate(start.clone(), 12, temperature=1.0, top_k=1, use_cache=False)
    tiny_p = generate_top_p(m, start.clone(), 12, temperature=1.0, p=1e-9)
    assert torch.equal(greedy, tiny_p), "p -> 0 must keep one token: greedy decoding"
    with torch.no_grad():
        probs = F.softmax(m(start)[0][:, -1, :].float(), dim=-1)            # (2, 64)
    near = top_p(probs, 1 - 1e-6)
    err = (near - probs).abs().max().item()
    assert err < 1e-6, f"p = 1 - 1e-6 moved the distribution by {err:.1e}"
    print(f"  generate_top_p matches AnuLM.generate at p = 1 and at p = 1e-9 "
          f"({tuple(ours.shape)} ids each); at p = 1 - 1e-6 top_p changes tiny()'s "
          f"probabilities by at most {err:.1e}")


if __name__ == "__main__":
    print("nucleus of the toy [0.50, 0.25, 0.15, 0.10] at p = 0.6:",
          [round(v, 4) for v in top_p(TOY, 0.6)[0].tolist()])
    ok = True
    for name, f in (("top_p", top_p), ("wrong_off_by_one", wrong_off_by_one),
                    ("wrong_no_renorm", wrong_no_renorm)):
        print(f"\n{name}:")
        failed = run_tests(f)
        ok &= (not failed) if f is top_p else bool(failed)
    dropped = []                              # what the off-by-one version loses in the sampling test
    for seed in (0, 1, 2):
        g = torch.Generator().manual_seed(seed)
        probs = F.softmax(torch.randn(1, 50, generator=g) * 2, dim=-1)
        oracle = nucleus_by_loop(probs[0].tolist(), 0.9)
        w = wrong_off_by_one(probs, 0.9)[0]
        dropped += [round(q, 4) for i, q in oracle.items() if w[i] == 0]
    print(f"off by one drops, per seed, a nucleus token of probability {dropped}")
    for p in (0.75, 0.7):                     # Exercise 3
        kept = {n: (f(TOY, p)[0] > 0).nonzero().flatten().tolist()
                for n, f in (("top_p", top_p), ("wrong_off_by_one", wrong_off_by_one))}
        print(f"exercise 3, p = {p}: {kept}")
    print("\nagainst the repository:")
    try:
        agrees_with_generate()
    except AssertionError:
        traceback.print_exc()
        ok = False
    print("\nthe right filter passes all three; each wrong one fails at least one" if ok else "\nCHECK FAILED")
    sys.exit(0 if ok else 1)
