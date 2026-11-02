"""
Chapter 16 checks: continued pre-training and longer context.

    python ch16.py                  # your course/build/further.py
    python ch16.py --impl solution  # the reference solution
    python ch16.py --self-test      # proves each check catches the bug it is for

Continuing a run fails quietly: a learning rate that jumps on resume, a held-out
set that went into the mix, a corpus that is one language and then another, a
longer window the model reads with the wrong temperature.
"""

from __future__ import annotations

import random
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import main, solution, with_replaced  # noqa: E402
from anulm import model as ref  # noqa: E402
from anulm.training import train  # noqa: E402

CHECKS = []
B = solution(8)
KW = dict(vocab_size=64, block_size=32, n_layer=2, hidden_size=32, n_head=4, head_dim=8, n_kv_head=2,
          num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16, intermediate_size=64,
          rope_theta=10000.0)


def check(fn):
    CHECKS.append(fn)
    return fn


@check
def a_resumed_run_ramps_back_up_like_train_py(M):
    """Extending a finished cosine run with --rewarmup 50 from step 1,000."""
    args = types.SimpleNamespace(warmup=100, lr=1e-3, min_lr=1e-4, steps=2000, schedule="cosine",
                                 decay_frac=0.2, rewarmup=50)
    for step in (999, 1000, 1001, 1025, 1049, 1050, 1500):
        want = train.lr_at(step, args, start_step=1000)
        got = M.rewarm(B.lr_at(step, 2000, 1e-3, 1e-4, 100, "cosine", 0.2), step, 1000, 50)
        assert abs(got - want) < 1e-12, f"step {step}: lr {got:.3e}, train.py {want:.3e}"


def corpora(n=400):
    rnd = random.Random(0)
    return {"hindi": [f"h{i} " + "x" * rnd.randint(20, 60) for i in range(n)],
            "english": [f"e{i} " + "y" * rnd.randint(20, 60) for i in range(n)],
            "python": [f"p{i} " + "z" * rnd.randint(20, 60) for i in range(n)]}


@check
def held_out_documents_never_enter_the_mix(M):
    src = corpora()
    mixed, held = M.mix_documents(src, {"hindi": 6000, "english": 6000, "python": 3000}, 50)
    for name, docs in held.items():
        assert docs, f"nothing held out of {name}"
        assert not set(docs) & set(mixed), f"held-out {name} documents are in the mix: it is not held out"
        want = [d for k, d in enumerate(src[name], 1) if k % 50 == 0][:len(docs)]
        assert docs == want, f"{name}: held out {docs[:2]}..., want every 50th document"


@check
def the_mix_follows_the_budgets(M):
    mixed, _ = M.mix_documents(corpora(), {"hindi": 6000, "english": 6000, "python": 3000}, 50)
    chars = {k: sum(len(d) for d in mixed if d[0] == k[0]) for k in ("hindi", "english", "python")}
    total = sum(chars.values())
    for k, share in (("hindi", 0.4), ("english", 0.4), ("python", 0.2)):
        assert abs(chars[k] / total - share) < 0.04, f"{k} is {chars[k] / total:.0%} of the mix, budget {share:.0%}"


@check
def the_mix_is_even_along_the_file(M):
    """Each half of the mixed corpus holds every source in about its share, so a run stopped halfway
    has seen all of them."""
    mixed, _ = M.mix_documents(corpora(), {"hindi": 6000, "english": 6000, "python": 3000}, 50)
    half = len(mixed) // 2
    for part in (mixed[:half], mixed[half:]):
        share = sum(d[0] == "p" for d in part) / len(part)
        assert 0.1 < share < 0.35, f"python is {share:.0%} of one half of the mix: the sources come one after another"


def ref_by_position(m, data, length, batches, bins, seed):
    g = torch.Generator().manual_seed(seed)
    edges = [length * b // bins for b in range(bins + 1)]
    totals = [0.0] * bins
    with torch.no_grad():
        for _ in range(batches):
            i = torch.randint(len(data) - length - 1, (1,), generator=g).item()
            x, y = data[i: i + length].long()[None], data[i + 1: i + 1 + length].long()[None]
            tok = F.cross_entropy(m(x)[0][0].float(), y[0], reduction="none")
            for b in range(bins):
                totals[b] += tok[edges[b]: edges[b + 1]].mean().item()
    return [t / batches for t in totals]


@check
def loss_is_measured_by_position_like_eval_context(M):
    torch.manual_seed(0)
    m = B.MoELM(**KW).eval()
    data = torch.randint(0, 64, (3000,))
    got = M.loss_by_position(m, data, 30, 6, 3, seed=4)
    want = ref_by_position(m, data, 30, 6, 3, 4)
    assert len(got) == 3 and all(abs(a - b) < 1e-5 for a, b in zip(got, want)), \
        f"by position {[round(v, 4) for v in got]}, eval_context's way {[round(v, 4) for v in want]}"


@check
def your_extended_model_is_model_py_with_yarn(M):
    """Your model, extended from 16 to 64 positions, gives exactly what model.py's AnuLM configured
    with YaRN gives, at positions it never trained on."""
    torch.manual_seed(0)
    mine = B.MoELM(**KW).eval()
    M.extend_context(mine, 64, 4.0, 16, 10000.0)
    theirs = ref.AnuLM(ref.AnuLMConfig(**dict(KW, block_size=64), yarn=True, yarn_factor=4.0,
                                       yarn_original_context=16)).eval()
    theirs.load_state_dict(mine.state_dict(), strict=True)
    x = torch.randint(0, 64, (2, 60))
    with torch.no_grad():                              # with targets: model.py then returns every position
        torch.testing.assert_close(mine(x, x)[0], theirs(x, x)[0], rtol=1e-4, atol=1e-5,
                                   msg="your extended model differs from model.py's YaRN")


# --- broken versions for --self-test ----------------------------------------------------

NAMES = ("rewarm", "mix_documents", "loss_by_position", "extend_context")


def MUTANTS(sol):
    def jump(lr, step, start_step, rewarmup):
        return lr

    def from_zero(lr, step, start_step, rewarmup):
        if rewarmup and start_step and step < start_step + rewarmup:
            return lr * (step - start_step) / rewarmup
        return lr

    def leaky(sources, budgets, heldout_every=50, seed=1337):
        mixed, held = sol.mix_documents(sources, budgets, heldout_every, seed)
        return mixed + [d for docs in held.values() for d in docs], held

    def one_after_another(sources, budgets, heldout_every=50, seed=1337):
        mixed, held = [], {}
        for n, docs in sources.items():
            m, h = sol.mix_documents({n: docs}, {n: budgets[n]}, heldout_every, seed)
            mixed += m
            held[n] = h[n]
        return mixed, held

    def uniform(sources, budgets, heldout_every=50, seed=1337):
        rng = random.Random(seed)
        its = {n: iter(d) for n, d in sources.items()}
        left, count = dict(budgets), {n: 0 for n in sources}
        mixed, held = [], {n: [] for n in sources}
        while any(left[n] > 0 for n in sources):
            n = rng.choice([k for k in sources if left[k] > 0])
            doc = next(its[n], None)
            if doc is None:
                left[n] = 0
                continue
            count[n] += 1
            if count[n] % heldout_every == 0:
                held[n].append(doc)
                continue
            mixed.append(doc)
            left[n] -= len(doc)
        return mixed, held

    def one_average(model, data, length, batches, bins=4, seed=0):
        whole = sol.loss_by_position(model, data, length, batches, 1, seed)[0]
        return [whole] * bins

    def no_temperature(model, block_size, factor, original_context, rope_theta):
        scales = [l.attn.scale for l in model.layers]
        sol.extend_context(model, block_size, factor, original_context, rope_theta)
        for l, s in zip(model.layers, scales):
            l.attn.scale = s

    def naive(model, block_size, factor, original_context, rope_theta):
        model.block_size = block_size

    def wrong_original(model, block_size, factor, original_context, rope_theta):
        sol.extend_context(model, block_size, factor, block_size, rope_theta)

    bugs = [("the learning rate jumps on resume", dict(rewarm=jump)),
            ("rewarmup starting from zero", dict(rewarm=from_zero)),
            ("held-out documents also mixed in", dict(mix_documents=leaky)),
            ("sources one after another", dict(mix_documents=one_after_another)),
            ("sources drawn uniformly, not by budget", dict(mix_documents=uniform)),
            ("one average copied to every position", dict(loss_by_position=one_average)),
            ("YaRN without its attention temperature", dict(extend_context=no_temperature)),
            ("a longer window with plain RoPE", dict(extend_context=naive)),
            ("the wrong original context", dict(extend_context=wrong_original))]
    return [(d, with_replaced(sol, NAMES, **r)) for d, r in bugs]


if __name__ == "__main__":
    main(16, CHECKS, MUTANTS, __doc__, learner="further.py")
