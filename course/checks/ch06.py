"""
Chapter 6 checks: attention, the causal mask, the KV cache, sliding windows.

    python ch06.py                  # your course/build/anulm.py
    python ch06.py --impl solution  # the reference solution
    python ch06.py --self-test      # proves each check catches the bug it is for

The most dangerous bug in this book lives here: a mask that lets a position
see the future. The model trains beautifully, because it is reading the answers,
and every number it reports is meaningless.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import main, with_replaced  # noqa: E402
import model as ref  # noqa: E402  the repository's model.py

CHECKS = []
CPU = torch.device("cpu")
D, H, HKV, DH = 64, 4, 2, 16


def check(fn):
    CHECKS.append(fn)
    return fn


def cfg(**kw):
    base = dict(hidden_size=D, n_head=H, n_kv_head=HKV, head_dim=DH, intermediate_size=96, use_qk_norm=True)
    base.update(kw)
    return ref.AnuLMConfig(**base)


def pair(M, n_kv=HKV, qk_norm=True, yarn=False):
    """model.py's attention and yours, with the same (random, non-trivial) weights."""
    torch.manual_seed(0)
    c = cfg(n_kv_head=n_kv, use_qk_norm=qk_norm, yarn=yarn, yarn_factor=4.0, yarn_original_context=64)
    rot = ref.Rotary(c, DH)
    theirs = ref.GQAttention(c, rot)
    if qk_norm:                         # gains away from 1, so their position in the pipeline matters
        with torch.no_grad():
            theirs.query_layernorm.weight.uniform_(0.5, 1.5)
            theirs.key_layernorm.weight.uniform_(0.5, 1.5)
    mine = M.GQAttention(D, H, n_kv, DH, use_qk_norm=qk_norm, attn_mscale=rot.attn_mscale)
    mine.load_state_dict(theirs.state_dict())
    return theirs.eval(), mine.eval(), rot


def tables(n, start=0):
    return ref.Rotary(cfg(), DH)(n, CPU, torch.float32, start=start)


# --- the mask -----------------------------------------------------------------------

@check
def attn_mask_matches_model(M):
    """Including when to return None (so the fast built-in causal path is used)."""
    for S, T, start, window in ((5, 5, 0, None), (1, 9, 8, None), (3, 8, 5, None), (6, 6, 0, 3), (2, 10, 8, 4)):
        mine, want = M.attn_mask(S, T, start, window, CPU), ref._attn_mask(S, T, start, window, CPU)
        if want is None:
            assert mine is None, f"attn_mask{(S, T, start, window)} should be None (is_causal suffices)"
        else:
            assert mine is not None and torch.equal(mine, want), f"attn_mask{(S, T, start, window)} differs"


# --- attention ------------------------------------------------------------------------

@check
def gqa_matches_model(M):
    """Same weights, same output: with grouped (2 kv heads), multi-head (4) and multi-query (1)
    attention, and with QK-norm gains that are not all 1."""
    x = torch.randn(2, 9, D)
    cos, sin = tables(9)
    for n_kv in (2, 4, 1):
        theirs, mine, _ = pair(M, n_kv=n_kv)
        with torch.no_grad():
            torch.testing.assert_close(mine(x, cos, sin), theirs(x, cos, sin), rtol=1e-4, atol=1e-5,
                                       msg=f"attention differs from model.py with {n_kv} kv heads")


@check
def gqa_scale_carries_yarn_temperature(M):
    """head_dim ** -0.5, times YaRN's mscale squared when YaRN is on (Chapter 5)."""
    for yarn in (False, True):
        theirs, mine, _ = pair(M, yarn=yarn)
        assert abs(mine.scale - theirs.scale) < 1e-12, f"scale {mine.scale}, model.py {theirs.scale} (yarn={yarn})"


@check
def causal_mask_does_not_leak_future(M):
    """Rewrite positions 6 onward: outputs at 0-5 must not move. If they do, the model
    trains on answers it can see."""
    _, mine, _ = pair(M)
    x = torch.randn(2, 12, D)
    cos, sin = tables(12)
    x2 = x.clone()
    x2[:, 6:] = torch.randn(2, 6, D)
    with torch.no_grad():
        drift = (mine(x, cos, sin)[:, :6] - mine(x2, cos, sin)[:, :6]).abs().max().item()
    assert drift < 1e-6, f"outputs at positions 0-5 moved by {drift:.2e} when later positions changed"


@check
def causality_holds_under_gradients_too(M):
    """A mask can hide the future in the forward pass and still leak it through gradients
    (a mask added after the softmax, say). Position 5's output must have no gradient into 6+."""
    _, mine, _ = pair(M)
    x = torch.randn(1, 12, D, requires_grad=True)
    cos, sin = tables(12)
    mine(x, cos, sin)[0, 5].sum().backward()
    assert x.grad[0, 6:].abs().max().item() == 0.0, "position 5 has gradient into the future"
    assert x.grad[0, :6].abs().max().item() > 0.0, "position 5 has no gradient into its own past"


@check
def cache_stores_the_shared_kv_heads(M):
    """The cache holds n_kv_head heads, not their copies: that ratio is GQA's whole memory
    saving at inference (16:1 in the real 30B). Storing copies still gives correct output."""
    _, mine, _ = pair(M)
    cache = {}
    with torch.no_grad():
        mine(torch.randn(1, 9, D), *tables(9), cache=cache)
    assert tuple(cache["k"].shape) == (1, HKV, 9, DH), f"cache['k'] is {tuple(cache['k'].shape)}, want (1, {HKV}, 9, {DH})"
    assert tuple(cache["v"].shape) == (1, HKV, 9, DH), f"cache['v'] is {tuple(cache['v'].shape)}"


@check
def incremental_matches_full(M):
    """Feeding a sequence in pieces through the cache must give exactly what feeding it whole
    gives: a 7-token prefill, one token, then chunks of 7 and 5. The chunks are the trap:
    PyTorch's is_causal aligns its triangle to the top-left, which is wrong once the cache
    already holds earlier tokens."""
    _, mine, _ = pair(M)
    x = torch.randn(2, 20, D)
    with torch.no_grad():
        full = mine(x, *tables(20))
        cache = {}
        for lo, hi in ((0, 7), (7, 8), (8, 15), (15, 20)):
            got = mine(x[:, lo:hi], *tables(hi - lo, start=lo), cache=cache, start=lo)
            torch.testing.assert_close(got, full[:, lo:hi], rtol=1e-4, atol=1e-5,
                                       msg=f"positions {lo}-{hi - 1} through the cache differ from the full pass")


@check
def sliding_window_limits_reach(M):
    """With window W, a token exactly W back must have no effect; W - 1 back must."""
    _, mine, _ = pair(M)
    W = 4
    x = torch.randn(1, 12, D)
    cos, sin = tables(12)
    with torch.no_grad():
        base = mine(x, cos, sin, window=W)[0, 11]
        far, near = x.clone(), x.clone()
        far[0, 11 - W] += 3.0
        near[0, 11 - W + 1] += 3.0
        d_far = (mine(far, cos, sin, window=W)[0, 11] - base).abs().max().item()
        d_near = (mine(near, cos, sin, window=W)[0, 11] - base).abs().max().item()
    assert d_far == 0.0, f"a token {W} back (outside the window) moved the output by {d_far:.2e}"
    assert d_near > 1e-6, f"a token {W - 1} back (inside the window) had no effect"


@check
def sliding_window_agrees_with_cache(M):
    _, mine, _ = pair(M)
    x = torch.randn(1, 14, D)
    with torch.no_grad():
        full = mine(x, *tables(14), window=5)
        cache = {}
        for lo, hi in ((0, 6), (6, 7), (7, 14)):
            got = mine(x[:, lo:hi], *tables(hi - lo, start=lo), cache=cache, start=lo, window=5)
            torch.testing.assert_close(got, full[:, lo:hi], rtol=1e-4, atol=1e-5,
                                       msg=f"windowed positions {lo}-{hi - 1} through the cache differ")


# --- the block and the model ----------------------------------------------------------

@check
def block_matches_model_layer_zero(M):
    """Your Block against model.py's first layer (attention plus a dense SwiGLU)."""
    torch.manual_seed(0)
    c = cfg()
    rot = ref.Rotary(c, DH)
    theirs = ref.Block(c, 0, rot).eval()
    mine = M.Block(D, H, HKV, DH, c.intermediate_size).eval()
    mine.load_state_dict(theirs.state_dict())
    x = torch.randn(2, 9, D)
    cos, sin = tables(9)
    with torch.no_grad():
        torch.testing.assert_close(mine(x, cos, sin), theirs(x, cos, sin)[0], rtol=1e-4, atol=1e-5,
                                   msg="Block differs from model.py's layer 0")


def lm(M, **kw):
    torch.manual_seed(0)
    base = dict(vocab_size=259, hidden_size=64, n_layer=4, n_head=4, n_kv_head=2, head_dim=16,
                intermediate_size=128, block_size=64)
    base.update(kw)
    return M.AttnLM(**base)


@check
def model_initial_loss_and_gradients(M):
    """ln(vocab) before training, and a gradient for every parameter."""
    m = lm(M)
    x = torch.randint(0, 259, (4, 32))
    m.eval()
    with torch.no_grad():
        loss = m(x, x)[1].item()
    assert abs(loss - math.log(259)) < 0.35, f"initial loss {loss:.3f}, expected about {math.log(259):.3f}"
    m.train()
    m(x, x)[1].backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reaches {missing[:4]}"


@check
def model_scales_both_residual_projections(M):
    """Each block writes into the residual twice (attention's dense, the MLP's down_proj);
    both start at 0.02 / sqrt(2 * n_layer)."""
    m = lm(M, n_layer=8)
    want = 0.02 / math.sqrt(16)
    for name, p in m.named_parameters():
        if name.endswith(("dense.weight", "down_proj.weight")):
            assert abs(p.std().item() - want) < 0.15 * want, f"{name}: std {p.std().item():.4f}, want {want:.4f}"


@check
def model_refuses_more_than_block_size(M):
    """Rotary tables and caches are sized for block_size; past it, fail loudly, not quietly."""
    m = lm(M).eval()
    with torch.no_grad():
        m(torch.randint(0, 259, (1, 64)))
    try:
        with torch.no_grad():
            m(torch.randint(0, 259, (1, 65)))
    except AssertionError:
        return
    raise AssertionError("a sequence longer than block_size was silently accepted")


@check
def model_cached_generation_matches_uncached(M):
    """With the same random draws, generating with the KV cache must give exactly the tokens
    generating without it gives. Positions passed wrongly to the rotary tables while decoding
    is the usual reason it does not."""
    m = lm(M).eval()
    prompt = torch.randint(0, 259, (2, 8))
    a = m.generate(prompt, 20, use_cache=True, generator=torch.Generator().manual_seed(1))
    b = m.generate(prompt, 20, use_cache=False, generator=torch.Generator().manual_seed(1))
    assert torch.equal(a, b), "the cached and uncached paths sampled different tokens"


@check
def model_matches_the_reference(M):
    sol = _sol()
    mine, theirs = lm(M).eval(), lm(sol).eval()
    theirs.load_state_dict(mine.state_dict())
    x = torch.randint(0, 259, (2, 24))
    with torch.no_grad():
        torch.testing.assert_close(mine(x)[0], theirs(x)[0], rtol=1e-4, atol=1e-5, msg="same weights, different logits")


def _sol():
    from _common import solution
    return solution(6)


# --- broken versions for --self-test ---------------------------------------------------

NAMES = ("attn_mask", "GQAttention", "Block", "AttnLM")


def MUTANTS(sol):
    def buggy_attention(**bug):
        class Buggy(sol.GQAttention):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                if bug.get("no_temperature"):
                    self.scale = self.head_dim ** -0.5
                if bug.get("no_sqrt"):
                    self.scale = self.head_dim ** -1.0

            def forward(self, x, cos, sin, cache=None, start=0, window=None):
                B, S, _ = x.shape
                qkv = self.query_key_value(x).view(B, S, self.n_head + 2 * self.n_kv_head, self.head_dim)
                q, k, v = qkv.split([self.n_head, self.n_kv_head, self.n_kv_head], dim=-2)
                q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                if bug.get("norm_after_rope"):
                    q, k = sol.apply_rope(q, k, cos, sin)
                    q, k = self.query_layernorm(q), self.key_layernorm(k)
                else:
                    q, k = self.query_layernorm(q), self.key_layernorm(k)
                    q, k = sol.apply_rope(q, k, cos, sin)
                rep = (lambda t: t.repeat(1, self.n_rep, 1, 1)) if bug.get("tile") else \
                      (lambda t: t.repeat_interleave(self.n_rep, dim=1))
                if cache is not None:
                    if bug.get("cache_copies"):
                        k, v = rep(k), rep(v)
                    if "k" in cache:
                        k = torch.cat([cache["k"], k], dim=2)
                        v = torch.cat([cache["v"], v], dim=2)
                    cache["k"], cache["v"] = k, v
                T = k.shape[2]
                if bug.get("no_mask"):
                    mask, causal = None, False
                elif bug.get("top_left"):
                    mask, causal = (None, S > 1) if window is None else (sol.attn_mask(S, T, start, window, x.device), False)
                elif bug.get("window_off_by_one") and window is not None:
                    qp = torch.arange(start, start + S)[:, None]
                    kp = torch.arange(T)[None, :]
                    mask, causal = (kp <= qp) & (kp >= qp - window), False
                else:
                    mask = sol.attn_mask(S, T, start, window, x.device)
                    causal = mask is None and S > 1
                if not (cache is not None and bug.get("cache_copies")):
                    k, v = rep(k), rep(v)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal, scale=self.scale)
                return self.dense(y.transpose(1, 2).reshape(B, S, -1))
        return Buggy

    def off_by_one_mask(S, T, start, window, device):
        m = sol.attn_mask(S, T, start, window, device)
        if m is not None and window is not None:
            qp = torch.arange(start, start + S)[:, None]
            kp = torch.arange(T)[None, :]
            m = (kp <= qp) & (kp >= qp - window)
        return m

    class NoAttnResidual(sol.Block):
        def forward(self, x, cos, sin, cache=None, start=0):
            x = self.attn(self.input_layernorm(x), cos, sin, cache=cache, start=start, window=self.window)
            return x + self.mlp(self.post_attention_layernorm(x))

    class DenseUnscaled(sol.AttnLM):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            for name, p in self.named_parameters():
                if name.endswith("dense.weight"):
                    nn.init.normal_(p, std=0.02)

    class NoGuard(sol.AttnLM):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.block_size = 10 ** 9

    class RotaryFromZero(sol.AttnLM):
        def forward(self, idx, targets=None, cache=None, start=0):
            B, S = idx.shape
            x = self.embed_tokens(idx)
            cos, sin = self.rotary(S, x.device, x.dtype, start=0)      # forgets the offset
            for i, layer in enumerate(self.layers):
                x = layer(x, cos, sin, cache=None if cache is None else cache[i], start=start)
            logits = self.lm_head(self.norm(x))
            return logits, (None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)),
                                                                           targets.reshape(-1)))

    bugs = [("attention with no mask at all", dict(GQAttention=buggy_attention(no_mask=True))),
            ("is_causal used against a non-empty cache", dict(GQAttention=buggy_attention(top_left=True))),
            ("window one position too wide (in attention)", dict(GQAttention=buggy_attention(window_off_by_one=True))),
            ("window one position too wide (in attn_mask)", dict(attn_mask=off_by_one_mask)),
            ("kv heads tiled instead of interleaved", dict(GQAttention=buggy_attention(tile=True))),
            ("cache storing the repeated copies", dict(GQAttention=buggy_attention(cache_copies=True))),
            ("QK-norm applied after RoPE", dict(GQAttention=buggy_attention(norm_after_rope=True))),
            ("scale 1/head_dim instead of 1/sqrt", dict(GQAttention=buggy_attention(no_sqrt=True))),
            ("YaRN temperature dropped from the scale", dict(GQAttention=buggy_attention(no_temperature=True))),
            ("block without the attention residual", dict(Block=NoAttnResidual)),
            ("attention's dense projection not scaled", dict(AttnLM=DenseUnscaled)),
            ("no block_size guard", dict(AttnLM=NoGuard)),
            ("rotary positions restart at 0 when decoding", dict(AttnLM=RotaryFromZero))]
    return [(d, with_replaced(sol, NAMES, **r)) for d, r in bugs]


if __name__ == "__main__":
    main(6, CHECKS, MUTANTS, __doc__)
