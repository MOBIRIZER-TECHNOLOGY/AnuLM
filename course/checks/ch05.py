"""
Chapter 5 checks: rotary positions and YaRN.

    python ch05.py                  # your course/build/anulm.py
    python ch05.py --impl solution  # the reference solution
    python ch05.py --self-test      # proves each check catches the bug it is for

Positions are where silent bugs live: a model with broken rotary embeddings
still trains, it just loses track of order and distance, and nothing in the
loss curve says so.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))      # however this file is launched
from _common import main, with_replaced  # noqa: E402
from anulm import model as ref  # the repository's model.py

CHECKS = []
CPU = torch.device("cpu")


def check(fn):
    CHECKS.append(fn)
    return fn


def theirs(dim, **kw):
    return ref.Rotary(ref.AnuLMConfig(**kw), dim)


# --- the rotation itself -----------------------------------------------------------

@check
def rotate_half_matches_model(M):
    """(x1, x2) -> (-x2, x1). The opposite sign also trains (every vector turns the other
    way) but no checkpoint from model.py would work with it."""
    x = torch.randn(3, 2, 5, 16)
    torch.testing.assert_close(M.rotate_half(x), ref.rotate_half(x), msg="rotate_half differs from model.py")


@check
def apply_rope_matches_model(M):
    """Same q, k and angles in, same rotated q, k out."""
    q, k = torch.randn(2, 3, 9, 16), torch.randn(2, 3, 9, 16)
    cos, sin = theirs(16)(9, CPU, torch.float32)
    for mine, want in zip(M.apply_rope(q, k, cos, sin), ref.apply_rope(q, k, cos, sin)):
        torch.testing.assert_close(mine, want, msg="apply_rope differs from model.py (is k rotated too?)")


@check
def rope_preserves_norms(M):
    """A rotation never changes a vector's length. If yours does, the cos/sin table's layout
    does not match rotate_half's pairing (interleaved table, split-halves rotation)."""
    cos, sin = M.Rotary(16)(16, CPU, torch.float32)
    q = torch.randn(2, 3, 16, 16)
    qr, _ = M.apply_rope(q, q, cos, sin)
    err = (qr.norm(dim=-1) - q.norm(dim=-1)).abs().max().item()
    assert err < 1e-4, f"rotation changed vector lengths by up to {err:.2e}"


@check
def rope_is_relative(M):
    """The point of RoPE: after rotation, q·k depends only on the distance i - j, never on
    where in the text the pair sits."""
    torch.manual_seed(0)
    cos, sin = M.Rotary(16, rope_theta=10000.0)(48, CPU, torch.float32)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)
    qq, kk = M.apply_rope(q.expand(1, 1, 48, 16).clone(), k.expand(1, 1, 48, 16).clone(), cos, sin)
    for gap in (1, 5, 11):
        vals = [(qq[0, 0, i] @ kk[0, 0, i - gap]).item() for i in range(gap, 40, 6)]
        assert max(vals) - min(vals) < 1e-4, \
            f"distance {gap}: q·k changes with absolute position (spread {max(vals) - min(vals):.2e})"


# --- the tables ------------------------------------------------------------------------

@check
def inv_freq_matches_model(M):
    """One frequency per pair of dimensions: 1 / theta^(2i/dim), fastest first."""
    for dim, theta in ((16, 1e4), (64, 1e6), (24, 1e4)):
        torch.testing.assert_close(M.Rotary(dim, rope_theta=theta).inv_freq, theirs(dim, rope_theta=theta).inv_freq,
                                   msg=f"inv_freq differs for dim={dim}, theta={theta:g}")


@check
def position_zero_is_not_rotated(M):
    """Position 0 has angle 0: cos 1, sin 0. Positions counted from 1 shift every angle and
    still train."""
    cos, sin = M.Rotary(16)(4, CPU, torch.float32)
    assert torch.allclose(cos[0], torch.ones(16)) and torch.allclose(sin[0], torch.zeros(16)), \
        "position 0 is rotated: are positions counted from 0?"


@check
def tables_match_model(M):
    for dim, theta in ((16, 1e4), (64, 1e6)):
        mine, want = M.Rotary(dim, rope_theta=theta)(300, CPU, torch.float32), theirs(dim, rope_theta=theta)(
            300, CPU, torch.float32)
        for a, b in zip(mine, want):
            torch.testing.assert_close(a, b, msg=f"cos/sin tables differ for dim={dim}")


@check
def angles_are_computed_in_float32(M):
    """Asked for bfloat16 tables, compute the angles in float32 and cast only cos and sin.
    Angles computed in bfloat16 drift by most of a radian within a few hundred positions,
    and the model still trains."""
    mine = M.Rotary(64, rope_theta=1e6)(1024, CPU, torch.bfloat16)
    want = theirs(64, rope_theta=1e6)(1024, CPU, torch.bfloat16)
    for a, b in zip(mine, want):
        assert a.dtype == torch.bfloat16, f"tables are {a.dtype}, asked for bfloat16"
        assert torch.equal(a, b), "bfloat16 tables differ from model.py's: compute angles in float32"


@check
def cache_tracks_dtype_and_positions(M):
    """The cache must notice a new dtype, and `start` must index absolute positions
    (it is how decoding with a KV cache asks for position 100 alone, in Chapter 6)."""
    rot = M.Rotary(8)
    c32, _ = rot(8, CPU, torch.float32)
    assert c32.dtype == torch.float32
    c16, _ = rot(8, CPU, torch.bfloat16)
    assert c16.dtype == torch.bfloat16, "a cached float32 table was handed back for bfloat16"
    full, _ = rot(16, CPU, torch.float32)
    tail, _ = rot(4, CPU, torch.float32, start=12)
    assert torch.equal(tail, full[12:16]), "start= does not index absolute positions"


# --- YaRN -----------------------------------------------------------------------------

YARN = dict(rope_theta=10000.0, yarn=True, yarn_factor=8.0, yarn_original_context=32)


@check
def yarn_mscale_matches_model(M):
    for scale, ms in ((1.0, 1.0), (4.0, 1.0), (40.0, 1.0), (40.0, 0.707)):
        assert abs(M.yarn_mscale(scale, ms) - ref.yarn_mscale(scale, ms)) < 1e-12, \
            f"yarn_mscale({scale}, {ms}) = {M.yarn_mscale(scale, ms)}, model.py gives {ref.yarn_mscale(scale, ms)}"


@check
def yarn_correction_range_matches_model(M):
    for args in ((32, 1, 16, 10000.0, 32), (32, 1, 64, 1e6, 256), (32, 1, 24, 10000.0, 4096)):
        assert M.yarn_correction_range(*args) == ref._yarn_correction_range(*args), \
            f"correction range {M.yarn_correction_range(*args)} != {ref._yarn_correction_range(*args)} for {args}"


@check
def yarn_inv_freq_matches_model(M):
    for dim, kw in ((16, YARN), (24, dict(YARN, yarn_factor=40.0, yarn_original_context=4096))):
        torch.testing.assert_close(M.Rotary(dim, **kw).inv_freq, theirs(dim, **kw).inv_freq,
                                   msg=f"YaRN inv_freq differs for dim={dim}")


@check
def yarn_keeps_fast_pairs_and_stretches_slow_ones(M):
    """Pairs that turn many times within the trained context keep their frequency; pairs
    that never complete a turn are divided by the factor. Reversed, it still runs."""
    plain, yarned = M.Rotary(16, rope_theta=10000.0).inv_freq, M.Rotary(16, **YARN).inv_freq
    ratio = plain / yarned
    assert abs(ratio[0].item() - 1.0) < 1e-4, f"the fastest pair was changed ({ratio[0]:.3f}x)"
    assert abs(ratio[-1].item() - 8.0) < 1e-3, f"the slowest pair was divided by {ratio[-1]:.3f}, expected 8"
    assert torch.all(ratio[1:] >= ratio[:-1] - 1e-5), "the ramp between them is not monotonic"


@check
def yarn_temperature_is_set(M):
    """YaRN has two halves: the frequency blend and an attention temperature. Dropping the
    second still runs and still extends, and you credit results to the wrong mechanism.
    model.py's grouped-query attention once did exactly that (see the chapter)."""
    for kw in (YARN, dict(YARN, yarn_factor=40.0)):
        mine, want = M.Rotary(16, **kw), theirs(16, **kw)
        assert abs(mine.attn_mscale - want.attn_mscale) < 1e-12 and mine.attn_mscale > 1, \
            f"attn_mscale {mine.attn_mscale}, expected {want.attn_mscale}"
        assert abs(mine.table_mscale - want.table_mscale) < 1e-12, "table_mscale differs from model.py"
    plain = M.Rotary(16)
    assert plain.attn_mscale == 1.0 and plain.table_mscale == 1.0, "without YaRN both temperatures must be 1"


# --- broken versions for --self-test -------------------------------------------------

NAMES = ("rotate_half", "apply_rope", "yarn_mscale", "yarn_correction_range", "Rotary")


def MUTANTS(sol):
    def flipped_rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((x2, -x1), dim=-1)

    def rope_on_q_only(q, k, cos, sin):
        return sol.apply_rope(q, k, cos, sin)[0], k

    def log10_mscale(scale, mscale=1.0):
        return 1.0 if scale <= 1 else 0.1 * mscale * math.log10(scale) + 1.0

    class Interleaved(sol.Rotary):
        def forward(self, seq_len, device, dtype, start=0):
            t = torch.arange(start, start + seq_len, dtype=torch.float32)
            emb = torch.outer(t, self.inv_freq).repeat_interleave(2, dim=-1)
            return emb.cos().to(dtype), emb.sin().to(dtype)

    class NoFactorTwo(sol.Rotary):
        def __init__(self, dim, rope_theta=10000.0, **kw):
            super().__init__(dim, rope_theta=rope_theta, **kw)
            self.inv_freq = 1.0 / rope_theta ** (torch.arange(dim // 2, dtype=torch.float32) / dim)

    class CountsFromOne(sol.Rotary):
        def forward(self, seq_len, device, dtype, start=0):
            t = torch.arange(start + 1, start + seq_len + 1, dtype=torch.float32)
            emb = torch.cat([torch.outer(t, self.inv_freq)] * 2, dim=-1)
            return emb.cos().to(dtype), emb.sin().to(dtype)

    class AnglesInDtype(sol.Rotary):
        def forward(self, seq_len, device, dtype, start=0):
            t = torch.arange(start, start + seq_len, dtype=torch.float32).to(dtype)
            emb = torch.cat([torch.outer(t, self.inv_freq.to(dtype))] * 2, dim=-1)
            return emb.cos(), emb.sin()

    class IgnoresStart(sol.Rotary):
        def forward(self, seq_len, device, dtype, start=0):
            return super().forward(seq_len, device, dtype, 0)

    class StaleDtype(sol.Rotary):
        def forward(self, seq_len, device, dtype, start=0):
            if self._cached_len >= start + seq_len:
                return self.cos_cached[start:start + seq_len], self.sin_cached[start:start + seq_len]
            return super().forward(seq_len, device, dtype, start)

    class ReversedRamp(sol.Rotary):
        def __init__(self, dim, rope_theta=10000.0, yarn=False, yarn_factor=8.0, **kw):
            super().__init__(dim, rope_theta=rope_theta, yarn=yarn, yarn_factor=yarn_factor, **kw)
            if yarn:
                plain = 1.0 / rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
                self.inv_freq = plain * plain / sol.Rotary(dim, rope_theta=rope_theta, yarn=True,
                                                           yarn_factor=yarn_factor, **kw).inv_freq / yarn_factor

    class NoTemperature(sol.Rotary):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.attn_mscale = 1.0

    out = []
    for desc, repl in [("rotate_half with the sign flipped", dict(rotate_half=flipped_rotate_half)),
                       ("RoPE applied to q but not k", dict(apply_rope=rope_on_q_only)),
                       ("interleaved table, split-halves rotation", dict(Rotary=Interleaved)),
                       ("frequencies without the factor 2", dict(Rotary=NoFactorTwo)),
                       ("positions counted from 1", dict(Rotary=CountsFromOne)),
                       ("angles computed in the table's dtype", dict(Rotary=AnglesInDtype)),
                       ("start= ignored", dict(Rotary=IgnoresStart)),
                       ("cache keyed on length only", dict(Rotary=StaleDtype)),
                       ("YaRN stretching the fast pairs instead", dict(Rotary=ReversedRamp)),
                       ("YaRN temperature dropped", dict(Rotary=NoTemperature)),
                       ("yarn_mscale with log10", dict(yarn_mscale=log10_mscale))]:
        out.append((desc, with_replaced(sol, NAMES, **repl)))
    return out


if __name__ == "__main__":
    main(5, CHECKS, MUTANTS, __doc__)
