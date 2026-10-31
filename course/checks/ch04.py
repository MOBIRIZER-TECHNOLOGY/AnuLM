"""
Chapter 4 checks: is your skeleton right, including in the ways training would not show?

    python ch04.py                  # checks course/build/anulm.py (yours)
    python ch04.py --impl solution  # the same checks on the reference solution
    python ch04.py --self-test      # proves each check catches the bug it is for

Two kinds of check. *Equivalence*: copy the same weights into your module
and model.py's, and the outputs must match. *Silent failures*: properties a
correct implementation has and a subtly wrong one lacks, even though both
would train and their loss would go down.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import traceback
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import model as ref  # noqa: E402  the repository's model.py, the reference

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def load(path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def norm_input(dtype=torch.float32):
    """Ordinary rows, plus one row of tiny values where eps decides the answer."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, 7, 256, generator=g) * 3
    x[0, 0] = torch.randn(256, generator=g) * 1e-4
    return x.to(dtype)


# --- RMSNorm -------------------------------------------------------------------

@check
def rmsnorm_matches_model(M):
    """Same input, same gain: your RMSNorm and model.py's must agree."""
    mine, theirs = M.RMSNorm(256), ref.RMSNorm(256)
    with torch.no_grad():
        theirs.weight.copy_(torch.randn(256, generator=torch.Generator().manual_seed(1)))
        mine.weight.copy_(theirs.weight)
    x = norm_input()
    torch.testing.assert_close(mine(x), theirs(x), rtol=1e-5, atol=1e-6,
                               msg="RMSNorm differs from model.py. Check: mean over the last dimension "
                                   "of x**2, eps inside the square root, then times weight.")


@check
def rmsnorm_gain_starts_at_one(M):
    """A gain that starts at zero (or random) gives a dead or noisy model that still trains, badly."""
    n = M.RMSNorm(64)
    assert isinstance(n.weight, nn.Parameter), "weight must be an nn.Parameter so it learns"
    assert tuple(n.weight.shape) == (64,), f"weight shape {tuple(n.weight.shape)}, expected (64,)"
    assert torch.all(n.weight == 1), "weight must start at ones: the norm begins as a pure rescale"


@check
def rmsnorm_is_scale_invariant(M):
    """RMSNorm(c * x) == RMSNorm(x) for any c > 0. A missing square root or a sum instead of a
    mean still trains, because the gain can partly compensate, but it breaks this."""
    n = M.RMSNorm(256)
    x = torch.randn(3, 256)
    torch.testing.assert_close(n(7.5 * x), n(x), rtol=1e-4, atol=1e-5,
                               msg="Scaling the input changed the output: is it mean(x**2), under a sqrt?")


@check
def rmsnorm_statistics_are_float32(M):
    """With bfloat16 activations (as in mixed-precision training), the mean of squares must be
    taken in float32. In bfloat16 it is quietly less accurate; the loss still goes down."""
    mine, theirs = M.RMSNorm(256), ref.RMSNorm(256)
    x = norm_input(torch.bfloat16)
    out, want = mine(x), theirs(x)
    assert out.dtype == want.dtype, f"output dtype {out.dtype}, model.py gives {want.dtype}"
    assert torch.equal(out, want), ("bfloat16 input gives different numbers from model.py: compute the "
                                    "statistics on x.float(), then cast back with .to(x.dtype)")


# --- SwiGLU --------------------------------------------------------------------

@check
def swiglu_matches_model(M):
    """Same weights in, same output out. If this fails while the shapes are right, the usual
    cause is silu on the wrong branch (up_proj instead of gate_proj)."""
    torch.manual_seed(0)
    theirs, mine = ref.SwiGLU(64, 160), M.SwiGLU(64, 160)
    mine.load_state_dict(theirs.state_dict())
    x = torch.randn(2, 5, 64)
    torch.testing.assert_close(mine(x), theirs(x), msg="SwiGLU differs from model.py with the same weights")


@check
def swiglu_has_no_biases_and_3DI_parameters(M):
    """model.py's MLPs have no biases. Extra biases still train, and then no checkpoint loads."""
    m = M.SwiGLU(64, 160)
    n = sum(p.numel() for p in m.parameters())
    assert n == 3 * 64 * 160, f"{n} parameters, expected 3 * 64 * 160 = {3 * 64 * 160} (bias=False?)"
    assert set(dict(m.named_parameters())) == {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}, \
        f"parameter names {sorted(dict(m.named_parameters()))}: use gate_proj, up_proj, down_proj"


# --- the block -------------------------------------------------------------------

@check
def block_is_a_pre_norm_residual(M):
    """x + mlp(norm(x)): normalise the branch's input, add its output back. A post-norm
    (norm(x + mlp(x))) or a missing residual both train at this depth, and both are a
    different model from model.py's."""
    torch.manual_seed(0)
    b = M.MLPBlock(64, 160)
    x = torch.randn(2, 5, 64)
    want = x + b.mlp(b.post_attention_layernorm(x))
    torch.testing.assert_close(b(x), want, msg="block output is not x + mlp(post_attention_layernorm(x))")
    with torch.no_grad():
        b.mlp.down_proj.weight.zero_()
    torch.testing.assert_close(b(x), x, msg="with the MLP silenced, the block must pass x through unchanged")


# --- the whole skeleton ------------------------------------------------------------

def skeleton(M, **kw):
    torch.manual_seed(0)
    cfg = dict(vocab_size=259, hidden_size=128, n_layer=4, intermediate_size=256)
    cfg.update(kw)
    return M.SkeletonLM(**cfg)


@check
def initialisation_matches_model(M):
    """Weights N(0, 0.02); the residual-writing down_proj N(0, 0.02 / sqrt(2 * n_layer)).
    PyTorch's default Embedding init is N(0, 1), 50x larger, and still trains."""
    m = skeleton(M, n_layer=8)
    std = lambda t: t.detach().float().std().item()
    for name, p in m.named_parameters():
        if name.endswith("down_proj.weight"):
            want = 0.02 / math.sqrt(2 * 8)
        elif p.dim() == 2:
            want = 0.02
        else:
            assert torch.all(p == 1), f"{name} should start at ones"
            continue
        assert abs(std(p) - want) < 0.15 * want, f"{name}: std {std(p):.4f}, expected {want:.4f}"


@check
def initial_loss_is_ln_vocab(M):
    """Before training, the model should be equally unsure of every token: loss = ln(vocab).
    Far above it means the logits start large and confident, a sign of a wrong init."""
    m = skeleton(M).eval()
    x = torch.randint(0, 259, (4, 64), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        _, loss = m(x, x)
    assert abs(loss.item() - math.log(259)) < 0.35, \
        f"initial loss {loss.item():.3f}, expected about ln(259) = {math.log(259):.3f}"


@check
def all_parameters_receive_gradients(M):
    """A parameter that is created but never used (a forgotten norm, an unused layer) still
    'trains' happily; it just never learns. Every parameter must get a gradient."""
    m = skeleton(M).train()
    x = torch.randint(0, 259, (4, 32))
    m(x, x)[1].backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reaches: {missing[:4]}"


@check
def targets_of_minus_one_are_ignored(M):
    """Padding and prompts are marked -1 and must not count towards the loss."""
    m = skeleton(M).eval()
    x = torch.randint(0, 259, (2, 16))
    t = x.clone()
    t[:, :8] = -1
    with torch.no_grad():
        logits, loss = m(x, t)
    want = F.cross_entropy(logits[:, 8:].reshape(-1, 259), t[:, 8:].reshape(-1))
    torch.testing.assert_close(loss, want, msg="positions with target -1 are being counted")


@check
def parameter_count_is_what_the_formula_says(M):
    """untied: 2·V·D (embeddings and head) + L·(D + 3·D·I) (each block's norm and MLP) + D."""
    V, D, L, I = 259, 128, 4, 256
    n = sum(p.numel() for p in skeleton(M).parameters())
    want = 2 * V * D + L * (D + 3 * D * I) + D
    assert n == want, f"{n:,} parameters, the formula gives {want:,}"
    tied = sum(p.numel() for p in skeleton(M, tie_word_embeddings=True).parameters())
    assert tied == want - V * D, f"tied: {tied:,}, expected {want - V * D:,} (head and embeddings share one matrix)"


@check
def same_weights_same_logits_as_the_reference(M):
    """The whole skeleton, end to end: load your weights into the reference and compare."""
    sol = load(Path(__file__).resolve().parents[1] / "build" / "solutions" / "ch04.py")
    mine, theirs = skeleton(M).eval(), skeleton(sol).eval()
    theirs.load_state_dict(mine.state_dict())
    x = torch.randint(0, 259, (2, 24))
    torch.testing.assert_close(mine(x)[0], theirs(x)[0], msg="same weights, different logits")


# --- running ---------------------------------------------------------------------------

def run(M, quiet=False) -> list[str]:
    failed = []
    for fn in CHECKS:
        try:
            fn(M)
            if not quiet:
                print(f"  pass  {fn.__name__}")
        except NotImplementedError as e:
            failed.append(fn.__name__)
            if not quiet:
                print(f"  todo  {fn.__name__}  ({e})")
        except Exception as e:  # noqa: BLE001 -- report every failure, keep going
            failed.append(fn.__name__)
            if not quiet:
                msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
                print(f"  FAIL  {fn.__name__}\n        {msg}")
    return failed


def mutants(sol):
    """Deliberately broken versions of the reference, one bug each."""
    class StatsInBf16(sol.RMSNorm):
        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

    class EpsOutsideSqrt(sol.RMSNorm):
        def forward(self, x):
            d = x.dtype
            x = x.float()
            return (x / (x.pow(2).mean(-1, keepdim=True).sqrt() + self.eps)).to(d) * self.weight

    class NoSqrt(sol.RMSNorm):
        def forward(self, x):
            d = x.dtype
            x = x.float()
            return (x / (x.pow(2).mean(-1, keepdim=True) + self.eps)).to(d) * self.weight

    class GainZero(sol.RMSNorm):
        def __init__(self, dim, eps=1e-6):
            super().__init__(dim, eps)
            nn.init.zeros_(self.weight)

    class SiluOnUp(sol.SwiGLU):
        def forward(self, x):
            return self.down_proj(self.gate_proj(x) * F.silu(self.up_proj(x)))

    class WithBias(sol.SwiGLU):
        def __init__(self, h, i):
            super().__init__(h, i)
            self.up_proj = nn.Linear(h, i, bias=True)

    class PostNorm(sol.MLPBlock):
        def forward(self, x):
            return self.post_attention_layernorm(x + self.mlp(x))

    class NoResidual(sol.MLPBlock):
        def forward(self, x):
            return self.mlp(self.post_attention_layernorm(x))

    class DefaultInit(sol.SkeletonLM):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            for m in self.modules():
                if isinstance(m, nn.Embedding):
                    m.reset_parameters()                # PyTorch's N(0, 1)

    class UnscaledResidualInit(sol.SkeletonLM):
        def __init__(self, *a, n_layer=4, **k):
            super().__init__(*a, n_layer=n_layer, **k)
            for name, p in self.named_parameters():
                if name.endswith("down_proj.weight"):
                    nn.init.normal_(p, std=0.02)

    class CountsPadding(sol.SkeletonLM):
        def forward(self, idx, targets=None):
            logits, _ = super().forward(idx)
            if targets is None:
                return logits, None
            return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.clamp_min(0).reshape(-1))

    class UnusedLayer(sol.SkeletonLM):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.extra = nn.Linear(4, 4)

    out = []
    for name, cls, slot in [("RMSNorm statistics in bfloat16", StatsInBf16, "RMSNorm"),
                            ("RMSNorm eps outside the sqrt", EpsOutsideSqrt, "RMSNorm"),
                            ("RMSNorm without the square root", NoSqrt, "RMSNorm"),
                            ("RMSNorm gain starting at zero", GainZero, "RMSNorm"),
                            ("SwiGLU with silu on the up branch", SiluOnUp, "SwiGLU"),
                            ("SwiGLU with a bias", WithBias, "SwiGLU"),
                            ("block normalising after the residual", PostNorm, "MLPBlock"),
                            ("block without the residual", NoResidual, "MLPBlock"),
                            ("embeddings left at PyTorch's N(0, 1)", DefaultInit, "SkeletonLM"),
                            ("down_proj not scaled down at init", UnscaledResidualInit, "SkeletonLM"),
                            ("loss counting -1 positions", CountsPadding, "SkeletonLM"),
                            ("a layer that never gets a gradient", UnusedLayer, "SkeletonLM")]:
        ns = types.SimpleNamespace(**{k: getattr(sol, k) for k in ("RMSNorm", "SwiGLU", "MLPBlock", "SkeletonLM")})
        setattr(ns, slot, cls)
        out.append((name, ns))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=["build", "solution"], default="build")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    here = Path(__file__).resolve().parents[1] / "build"
    sol = load(here / "solutions" / "ch04.py")
    if a.self_test:
        print("reference solution must pass everything:")
        bad = run(sol, quiet=True)
        print("  ok" if not bad else f"  FAILED: {bad}")
        print("each broken version must be caught by at least one check:")
        missed = 0
        for name, ns in mutants(sol):
            caught = run(ns, quiet=True)
            missed += not caught
            print(f"  {'caught' if caught else 'MISSED'}  {name:42s} by {', '.join(caught) or '-'}")
        sys.exit(1 if bad or missed else 0)
    M = sol if a.impl == "solution" else load(here / "anulm.py")
    print(f"Chapter 4 checks on {'the reference solution' if a.impl == 'solution' else 'course/build/anulm.py'}:")
    failed = run(M)
    total = len(CHECKS)
    print(f"\n{total - len(failed)} of {total} pass")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
