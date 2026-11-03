"""
Tests for AnuLM. Plain asserts, no pytest needed:

    python -m tests.test_model

The same tests also run under a standard runner -- `python -m unittest
tests.test_model` or pytest -- through the TestCase generated at the bottom.
A test that cannot run here (a missing optional package, no grouped GEMM)
raises unittest.SkipTest and is reported as SKIP, never as PASS.

These target the failure modes that TRAINING WOULD NOT CATCH. A leaky causal
mask still converges (it just cheats). Broken RoPE still converges (it just
loses long-range structure). A MoE that silently drops experts still converges.
Loss going down proves almost nothing about correctness.
"""

from __future__ import annotations

import contextlib
import math
import sys
import traceback
import unittest

import torch
import torch.nn.functional as F

from anulm.model import (GQAttention, MLAttention, MoE, AnuLM, AnuLMConfig,
                   Rotary, Router, apply_rope, yarn_mscale)

TESTS = []


def test(fn):
    """Register fn as a test. Not a test itself: pytest collects any
    module-level callable named test*, hence __test__ below."""
    TESTS.append(fn)
    return fn


test.__test__ = False


def tiny(**kw) -> AnuLMConfig:
    base = dict(vocab_size=64, block_size=32, n_layer=3, hidden_size=32, n_head=4,
                head_dim=8, n_kv_head=2, num_experts=8, num_experts_per_tok=2,
                moe_intermediate_size=16, intermediate_size=64, n_group=1, topk_group=1)
    base.update(kw)
    return AnuLMConfig(**base)


# =============================================================================
# CAUSALITY -- the one that matters most
# =============================================================================

@test
def causal_mask_does_not_leak_future():
    """Changing token t+1.. must not change logits at positions <= t.

    If this fails the model is training on answers it can see, and every loss
    number reported so far is meaningless.
    """
    for attn in ("gqa", "mla"):
        torch.manual_seed(0)
        m = AnuLM(tiny(attn=attn)).eval()
        x = torch.randint(0, 64, (2, 16))
        with torch.no_grad():
            a, _ = m(x, x)
        x2 = x.clone()
        x2[:, 8:] = torch.randint(0, 64, (2, 8))     # rewrite the future
        with torch.no_grad():
            b, _ = m(x2, x2)
        drift = (a[:, :8] - b[:, :8]).abs().max().item()
        assert drift < 1e-5, f"{attn}: past logits moved by {drift:.2e} when the future changed"


@test
def causality_holds_under_gradients_too():
    """Gradient of loss at position t w.r.t. embedding at t+1 must be zero."""
    torch.manual_seed(0)
    m = AnuLM(tiny(attn="mla"))
    x = torch.randint(0, 64, (1, 12))
    emb = m.embed_tokens(x).detach().requires_grad_(True)
    h = emb
    cos, sin = m.rotary(12, emb.device, emb.dtype)
    for layer in m.layers:
        h, _ = layer(h, cos, sin)
    logits = m.lm_head(m.norm(h))
    logits[0, 5].sum().backward()
    future_grad = emb.grad[0, 6:].abs().max().item()
    past_grad = emb.grad[0, :6].abs().max().item()
    assert future_grad == 0.0, f"position 5 has gradient {future_grad:.2e} into the future"
    assert past_grad > 0.0, "position 5 has no gradient into its own past -- something is severed"


# =============================================================================
# ROPE
# =============================================================================

@test
def rope_is_relative():
    """q.k after RoPE must depend only on (i - j), not on absolute i and j."""
    torch.manual_seed(0)
    cfg = tiny()
    rot = Rotary(cfg, dim=8)
    cos, sin = rot(32, torch.device("cpu"), torch.float32)
    q = torch.randn(1, 1, 32, 8)
    k = torch.randn(1, 1, 32, 8)
    # The SAME q and k content rotated to every position; q.k must then depend
    # only on the gap between the positions it is read at.
    qq, kk = apply_rope(q[:, :, :1].expand(1, 1, 32, 8).clone(),
                        k[:, :, :1].expand(1, 1, 32, 8).clone(), cos, sin)
    for gap in (1, 5, 11):
        vals = [(qq[0, 0, i] @ kk[0, 0, i - gap]).item() for i in range(gap, 25, 6)]
        spread = max(vals) - min(vals)
        assert spread < 1e-4, f"gap {gap}: dot product varies by {spread:.2e} with absolute position"


@test
def rope_preserves_norms():
    torch.manual_seed(0)
    rot = Rotary(tiny(), dim=8)
    cos, sin = rot(16, torch.device("cpu"), torch.float32)
    q = torch.randn(2, 3, 16, 8)
    qr, _ = apply_rope(q, q, cos, sin)
    err = (qr.norm(dim=-1) - q.norm(dim=-1)).abs().max().item()
    assert err < 1e-5, f"rotation changed vector norms by {err:.2e}"


@test
def yarn_blends_the_right_bands():
    """Fast dims keep their frequency, slow dims are divided by `factor`."""
    cfg = tiny(yarn=True, yarn_factor=8.0, yarn_original_context=32, rope_theta=10000.0)
    dim = 16
    plain = Rotary._build_inv_freq(tiny(yarn=False, rope_theta=10000.0), dim)
    yarned = Rotary._build_inv_freq(cfg, dim)
    ratio = (plain / yarned)
    assert abs(ratio[0].item() - 1.0) < 1e-4, f"fastest dim was scaled ({ratio[0]:.3f}), should be untouched"
    assert abs(ratio[-1].item() - 8.0) < 1e-3, f"slowest dim scaled by {ratio[-1]:.3f}, expected 8.0"
    assert torch.all(ratio[1:] >= ratio[:-1] - 1e-5), "the ramp is not monotonic"


@test
def yarn_temperature_reaches_attention():
    """BOTH attention paths must carry the mscale**2 logit temperature.

    YaRN is two halves: the band blend in the rotary tables and the softmax
    temperature. Applying only the first is not YaRN, and it fails silently --
    the model still runs, still extends, and reports numbers you would then
    attribute to the wrong mechanism. GQA dropped this half until it was
    caught by reading, not by any test, which is why this covers both.
    """
    mla = tiny(attn="mla", yarn=True, yarn_factor=40.0, qk_nope_head_dim=8,
               qk_rope_head_dim=8, v_head_dim=8, kv_lora_rank=16)
    a = MLAttention(mla, Rotary(mla, dim=8))
    expected = (16 ** -0.5) * (yarn_mscale(40.0, 1.0) ** 2)
    assert abs(a.scale - expected) < 1e-9, f"mla scale {a.scale} != {expected}"
    assert a.scale > 16 ** -0.5, "mla: YaRN temperature term was dropped"

    gqa = tiny(attn="gqa", yarn=True, yarn_factor=40.0, head_dim=8)
    g = GQAttention(gqa, Rotary(gqa, dim=8))
    expected = (8 ** -0.5) * (yarn_mscale(40.0, 1.0) ** 2)
    assert abs(g.scale - expected) < 1e-9, f"gqa scale {g.scale} != {expected}"
    assert g.scale > 8 ** -0.5, "gqa: YaRN temperature term was dropped"

    # ... and must NOT carry it when YaRN is off.
    for cfg, cls, dim in ((tiny(attn="mla", qk_nope_head_dim=8, qk_rope_head_dim=8,
                                v_head_dim=8, kv_lora_rank=16), MLAttention, 16),
                          (tiny(attn="gqa", head_dim=8), GQAttention, 8)):
        m = cls(cfg, Rotary(cfg, dim=8))
        assert abs(m.scale - dim ** -0.5) < 1e-9, (
            f"{cfg.attn}: scale {m.scale} != plain {dim ** -0.5} with yarn off")


# =============================================================================
# MoE
# =============================================================================

@test
def moe_matches_naive_per_token_loop():
    torch.manual_seed(0)
    cfg = tiny(num_experts=8, num_experts_per_tok=3, n_group=1, topk_group=1)
    moe = MoE(cfg).eval()
    x = torch.randn(2, 5, cfg.hidden_size)
    with torch.no_grad():
        got, _ = moe(x)
        flat = x.view(-1, cfg.hidden_size)
        idx, w, _ = moe.router(flat)
        ref = torch.zeros_like(flat)
        for n in range(flat.shape[0]):
            for s in range(cfg.num_experts_per_tok):
                ref[n] += w[n, s] * moe.experts[idx[n, s]](flat[n])
        ref = ref.view_as(x) + moe.shared_expert(x)
    err = (got - ref).abs().max().item()
    assert err < 1e-5, f"MoE output differs from the naive loop by {err:.2e}"


@test
def moe_with_all_experts_selected_is_dense():
    """top_k == num_experts must reduce to a plain weighted sum over all experts."""
    torch.manual_seed(0)
    cfg = tiny(num_experts=4, num_experts_per_tok=4, n_group=1, topk_group=1,
               num_shared_experts=0, routed_scaling_factor=1.0)
    moe = MoE(cfg).eval()
    x = torch.randn(1, 4, cfg.hidden_size)
    with torch.no_grad():
        got, _ = moe(x)
        flat = x.view(-1, cfg.hidden_size)
        logits = F.linear(flat.float(), moe.router.weight.float())
        s = torch.sigmoid(logits)
        w = s / s.sum(-1, keepdim=True)
        ref = sum(w[:, e: e + 1] * moe.experts[e](flat) for e in range(4)).view_as(x)
    err = (got - ref).abs().max().item()
    assert err < 1e-5, f"dense-equivalent case differs by {err:.2e}"


@test
def router_respects_group_limits():
    torch.manual_seed(0)
    cfg = tiny(num_experts=16, num_experts_per_tok=2, n_group=4, topk_group=2)
    r = Router(cfg).eval()
    x = torch.randn(64, cfg.hidden_size)
    idx, w, _ = r(x)
    per_group = 16 // 4
    for n in range(64):
        groups = {(e // per_group) for e in idx[n].tolist()}
        assert len(groups) <= 2, f"token {n} routed into {len(groups)} groups, limit is 2"
    assert (w.sum(-1) - cfg.routed_scaling_factor).abs().max() < 1e-5, \
        "combine weights do not sum to routed_scaling_factor"


@test
def router_weights_come_from_unbiased_scores():
    """The balancing bias must steer selection without changing magnitudes."""
    torch.manual_seed(0)
    cfg = tiny(num_experts=8, num_experts_per_tok=2)
    r = Router(cfg).eval()
    x = torch.randn(32, cfg.hidden_size)
    idx0, w0, _ = r(x)
    r.expert_bias[3] += 5.0            # make expert 3 irresistible
    idx1, w1, _ = r(x)
    assert not torch.equal(idx0, idx1), "a huge bias did not change routing at all"
    assert (idx1 == 3).any(), "biased expert was never selected"
    # For tokens whose selection is unchanged, weights must be identical.
    same = (idx0 == idx1).all(-1)
    if same.any():
        err = (w0[same] - w1[same]).abs().max().item()
        assert err < 1e-6, f"bias leaked into combine weights by {err:.2e}"


@test
def rmsnorm_and_swiglu_match_their_formulas():
    """The two building blocks against their definitions, with weights a
    trained checkpoint would have (not the init's all-ones gain).

    RMSNorm: x / sqrt(mean(x^2) + eps) * g -- eps INSIDE the root, which only
    shows on small activations, and keeps an all-zero row at zero instead of
    0 * inf = NaN. SwiGLU: down(silu(gate(x)) * up(x)). Swapping which
    projection goes through silu trains just as well from scratch, so no
    loss curve sees it, but every imported or exported checkpoint then runs
    a different function.
    """
    from anulm.model import RMSNorm, SwiGLU
    torch.manual_seed(0)
    norm = RMSNorm(16, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(16))
    x = torch.randn(4, 16) * 1e-3                   # mean(x^2) ~ eps: eps matters
    x[0] = 0.0
    with torch.no_grad():
        got = norm(x)
    want = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * norm.weight
    assert torch.isfinite(got).all(), "RMSNorm of an all-zero row is not finite"
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-7), \
        f"RMSNorm differs from its formula by {(got - want).abs().max().item():.2e}"

    mlp = SwiGLU(16, 24)
    x = torch.randn(5, 16)
    with torch.no_grad():
        got = mlp(x)
        want = F.linear(F.silu(F.linear(x, mlp.gate_proj.weight)) * F.linear(x, mlp.up_proj.weight),
                        mlp.down_proj.weight)
    assert torch.allclose(got, want, atol=1e-6), "SwiGLU is not down(silu(gate(x)) * up(x))"


@test
def tied_embeddings_share_one_table():
    """tie_word_embeddings=True must make the LM head the embedding table --
    one tensor, so an update to either is an update to both -- and False must
    keep two."""
    torch.manual_seed(0)
    tied = AnuLM(tiny(tie_word_embeddings=True))
    assert tied.lm_head.weight is tied.embed_tokens.weight, "tie_word_embeddings=True left two tables"
    n_tied = sum(p.numel() for p in tied.parameters())
    untied = AnuLM(tiny(tie_word_embeddings=False))
    assert untied.lm_head.weight is not untied.embed_tokens.weight
    n_untied = sum(p.numel() for p in untied.parameters())
    assert n_untied - n_tied == 64 * 32, "tying must save exactly one V x D table"


@test
def router_choices_survive_autocast():
    """Under bf16 autocast the router must pick exactly the experts it picks
    in fp32 for the same input, with the same combine weights.

    Router.forward turns autocast off and runs the whole gate in fp32 on
    purpose: autocast re-casts F.linear to bf16 whatever dtype it is handed,
    and bf16 rounding of the logits flips near-tied top-k choices (3.2% of
    expert sets on a real checkpoint). What it guarantees is agreement for
    the same input tensor -- it cannot undo rounding already done upstream, so
    a bf16 activation is compared with its own value upcast to fp32, not with
    the fp32 activation it was rounded from.
    """
    torch.manual_seed(0)
    cfg = tiny(num_experts=8, num_experts_per_tok=2)
    r = Router(cfg).eval()
    x = torch.randn(2048, cfg.hidden_size)
    for inp in (x, x.bfloat16()):
        with torch.no_grad():
            idx_ref, w_ref, _ = r(inp.float())
            with torch.autocast("cpu", dtype=torch.bfloat16):
                idx_ac, w_ac, _ = r(inp)
        flips = (idx_ref != idx_ac).any(-1).sum().item()
        assert flips == 0, f"{inp.dtype}: {flips} of {len(x)} tokens changed experts under autocast"
        # Weights come back in the input's dtype: exact for fp32, one rounding for bf16.
        assert torch.equal(w_ref.to(w_ac.dtype), w_ac), \
            f"{inp.dtype}: combine weights moved under autocast"


@test
def bias_update_moves_in_the_right_direction():
    """The exact contract: overloaded experts get pushed down, starved ones up.

    This is what the rule guarantees on every single step. It does NOT guarantee
    monotone improvement -- see bias_update_recovers_from_collapse below.
    """
    torch.manual_seed(0)
    cfg = tiny(num_experts=8, num_experts_per_tok=2, bias_update_rate=0.05)
    m = AnuLM(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (8, 32))
    r = next(mod for mod in m.modules() if isinstance(mod, Router))
    for step in range(8):
        _, loss = m(x, x)
        loss.backward()
        m.zero_grad(set_to_none=True)
        load = r.load_counts.clone()
        mean = load.mean()
        before = r.expert_bias.clone()
        m.update_expert_biases()
        delta = r.expert_bias - before
        over = load > mean
        under = load < mean
        assert not over.any() or (delta[over] < 0).all(), f"step {step}: overloaded expert bias rose"
        assert not under.any() or (delta[under] > 0).all(), f"step {step}: starved expert bias fell"
        assert torch.allclose(delta.abs()[load != mean],
                              torch.full_like(delta[load != mean], cfg.bias_update_rate)), \
            "step size is not bias_update_rate"


@test
def bias_update_recovers_from_collapse():
    """From a deliberately collapsed router, the balancer must pull load back.

    Compares windowed means, because the sign-based rule oscillates around the
    fixed point rather than settling on it: sign() applies a full-size step
    however small the error. With rate=0.05 on a frozen batch it visibly
    ping-pongs (1.19x <-> 1.73x, period 2). That is inherent to the DeepSeek-V3
    rule, not a defect here -- in real training the rate is 50x smaller and
    batch variation averages the kicks out.

    Budget matters, and it is arithmetic, not luck. An overloaded expert's bias
    falls by `rate` while a starved one rises by `rate`, so a skew of S closes
    at 2*rate per step and needs ~S/(2*rate) steps before routing can change at
    all. Measured: skew 3.0 at rate 0.01 shows ZERO movement in 120 steps
    (needs ~150 just to reach parity); 200 steps at skew 1.0 lands at 1.10x.
    """
    torch.manual_seed(0)
    cfg = tiny(num_experts=8, num_experts_per_tok=2, bias_update_rate=0.01)
    m = AnuLM(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (8, 32))
    r = next(mod for mod in m.modules() if isinstance(mod, Router))
    with torch.no_grad():               # force everything onto two experts
        r.expert_bias[0] = 1.0
        r.expert_bias[1] = 0.85

    history = []
    for _ in range(200):
        _, loss = m(x, x)
        loss.backward()
        m.zero_grad(set_to_none=True)
        history.append(m.update_expert_biases())

    early = sum(history[:10]) / 10
    late = sum(history[-10:]) / 10
    assert early > 2.0, f"setup failed: router was not actually collapsed ({early:.2f}x)"
    assert late < early * 0.6, f"balancer failed to recover: {early:.2f}x -> {late:.2f}x"


@test
def expert_bias_is_not_trainable():
    m = AnuLM(tiny())
    for name, p in m.named_parameters():
        assert "expert_bias" not in name, f"{name} is a Parameter; it would be hit by AdamW"
    r = next(mod for mod in m.modules() if isinstance(mod, Router))
    assert "expert_bias" in dict(r.named_buffers()), "expert_bias is not a buffer"
    assert "expert_bias" in m.state_dict().keys() or any(
        k.endswith("expert_bias") for k in m.state_dict()), "expert_bias missing from state_dict"


# =============================================================================
# MODEL PLUMBING
# =============================================================================

@test
def all_parameters_receive_gradients():
    """With enough tokens every expert should be hit at least once."""
    for attn in ("gqa", "mla"):
        torch.manual_seed(0)
        m = AnuLM(tiny(attn=attn)).train()
        x = torch.randint(0, 64, (16, 32))
        _, loss = m(x, x)
        loss.backward()
        missing = [n for n, p in m.named_parameters() if p.grad is None]
        assert not missing, f"{attn}: no gradient for {missing[:4]}"


@test
def initial_loss_is_ln_vocab():
    """Init loss sits at ln(V), and the init scales are the configured ones.

    The loss alone is a weak check: the final RMSNorm divides out the
    embedding scale, so embeddings at std 1.0 (50x too large) still start
    within 0.03 of ln(V). Measured on this config the real init is within
    0.028 (gqa) and 0.009 (mla); 0.1 leaves room for seeds. The scales the
    loss cannot see are asserted directly.
    """
    for attn in ("gqa", "mla"):
        torch.manual_seed(0)
        cfg = tiny(attn=attn)
        m = AnuLM(cfg).eval()
        x = torch.randint(0, cfg.vocab_size, (4, 32))
        with torch.no_grad():
            _, loss = m(x, x)
        expected = math.log(cfg.vocab_size)
        assert abs(loss.item() - expected) < 0.1, \
            f"{attn}: init loss {loss.item():.3f}, expected ~{expected:.3f} -- init is off"
        std = cfg.initializer_range
        resid = std / math.sqrt(2 * cfg.n_layer)
        for name, p in m.named_parameters():
            if p.dim() < 2 or ".router." in name:   # norms, and the kaiming router
                continue
            want = resid if name.endswith(("o_proj.weight", "dense.weight",
                                           "down_proj.weight")) else std
            got = p.std().item()
            assert abs(got - want) < 0.15 * want, \
                f"{attn}: {name} initialised at std {got:.4f}, expected {want:.4f}"


@test
def checkpoint_roundtrip_is_exact():
    import io
    torch.manual_seed(0)
    cfg = tiny(attn="mla")
    m = AnuLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        before, _ = m(x, x)
    buf = io.BytesIO()
    torch.save({"model": m.state_dict(), "cfg": cfg}, buf)
    buf.seek(0)
    ck = torch.load(buf, weights_only=False)
    m2 = AnuLM(ck["cfg"]).eval()
    m2.load_state_dict(ck["model"])
    with torch.no_grad():
        after, _ = m2(x, x)
    err = (before - after).abs().max().item()
    assert err == 0.0, f"reloaded model differs by {err:.2e}"


@test
def block_size_boundary():
    cfg = tiny(block_size=32)
    m = AnuLM(cfg).eval()
    with torch.no_grad():
        m(torch.randint(0, 64, (1, 32)))          # exactly at the limit: fine
    try:
        with torch.no_grad():
            m(torch.randint(0, 64, (1, 33)))
    except ValueError:                             # a real error: python -O strips asserts
        return
    raise AssertionError("sequence longer than block_size was silently accepted")


@test
def generation_is_deterministic_at_top_k_1():
    torch.manual_seed(0)
    cfg = tiny()
    m = AnuLM(cfg).eval()
    start = torch.randint(0, cfg.vocab_size, (1, 4))
    a = m.generate(start.clone(), 12, temperature=1.0, top_k=1)
    b = m.generate(start.clone(), 12, temperature=1.0, top_k=1)
    assert torch.equal(a, b), "greedy decoding is not deterministic"
    assert a.shape == (1, 16), f"generate returned {tuple(a.shape)}, expected (1, 16)"


@test
def param_count_matches_analysis():
    cfg = AnuLMConfig.nano_30b()
    m = AnuLM(cfg)
    total, active = m.num_params()
    assert abs(total - 17_440_384) < 1000, f"total {total} != 17.44M"
    assert abs(active - 6_602_368) < 1000, f"active {active} != 6.60M"
    cfg = AnuLMConfig.nano_105b()
    total, active = AnuLM(cfg).num_params()
    assert abs(total - 47_389_728) < 1000, f"total {total} != 47.39M"
    assert abs(active - 13_327_392) < 1000, f"active {active} != 13.33M"


@test
def config_rejects_unreachable_topk():
    try:
        tiny(num_experts=16, n_group=4, topk_group=1, num_experts_per_tok=8)
    except ValueError:
        return
    raise AssertionError("config accepted a top_k the groups cannot supply")


@test
def mla_shares_one_rope_key_across_heads():
    """k_pe is projected once and broadcast; every head must see the same one."""
    torch.manual_seed(0)
    cfg = tiny(attn="mla", n_head=4, qk_nope_head_dim=8, qk_rope_head_dim=8,
               v_head_dim=8, kv_lora_rank=16)
    rot = Rotary(cfg, dim=8)
    a = MLAttention(cfg, rot).eval()
    x = torch.randn(1, 6, cfg.hidden_size)
    compressed = a.kv_a_proj_with_mqa(x)
    _, k_pe = compressed.split([a.kv_lora_rank, a.qk_rope], dim=-1)
    assert k_pe.shape == (1, 6, 8), f"k_pe has shape {tuple(k_pe.shape)}, expected one head's worth"
    kv = a.kv_b_proj(a.kv_a_layernorm(compressed.split([a.kv_lora_rank, a.qk_rope], -1)[0]))
    assert kv.shape[-1] == cfg.n_head * (8 + 8), "kv_b_proj output width is wrong"


@test
def gradient_accumulation_equals_large_batch():
    """Two half-batches accumulated must equal one full batch."""
    torch.manual_seed(0)
    cfg = tiny()
    m = AnuLM(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (8, 16))

    _, loss = m(x, x)
    loss.backward()
    full = [p.grad.clone() for p in m.parameters()]
    m.zero_grad(set_to_none=True)

    for half in (x[:4], x[4:]):
        _, l = m(half, half)
        (l / 2).backward()
    split = [p.grad.clone() for p in m.parameters()]

    err = max((a - b).abs().max().item() for a, b in zip(full, split))
    assert err < 2e-5, f"grad accumulation differs from one big batch by {err:.2e}"


# =============================================================================
# torch.compile
# =============================================================================

@test
def dense_and_sparse_moe_agree():
    """The two dispatch implementations must be numerically interchangeable."""
    for attn in ("gqa", "mla"):
        torch.manual_seed(0)
        a = AnuLM(tiny(attn=attn, moe_impl="sparse")).eval()
        torch.manual_seed(0)
        b = AnuLM(tiny(attn=attn, moe_impl="dense")).eval()
        b.load_state_dict(a.state_dict())
        x = torch.randint(0, 64, (2, 16))
        with torch.no_grad():
            ya, _ = a(x, x)
            yb, _ = b(x, x)
        err = (ya - yb).abs().max().item()
        assert err < 1e-5, f"{attn}: sparse and dense MoE differ by {err:.2e}"


@test
def dense_moe_compiles_without_graph_breaks():
    """moe_impl='dense' exists precisely so Dynamo can capture one graph."""
    import torch._dynamo as dynamo
    torch.manual_seed(0)
    m = AnuLM(tiny(moe_impl="dense")).eval()
    x = torch.randint(0, 64, (2, 16))
    dynamo.reset()
    exp = dynamo.explain(lambda t: m(t, t))(x)
    dynamo.reset()
    assert exp.graph_break_count == 0, \
        f"dense MoE broke the graph {exp.graph_break_count} times"
    assert exp.graph_count == 1, f"expected 1 graph, got {exp.graph_count}"


@test
def sparse_moe_breaks_the_graph_as_documented():
    """Guards the claim in the config comment. If a future torch handles
    data-dependent `nonzero` without breaking, this fails and the docs (and the
    reason moe_impl exists) need revisiting."""
    import torch._dynamo as dynamo
    torch.manual_seed(0)
    m = AnuLM(tiny(moe_impl="sparse")).eval()
    x = torch.randint(0, 64, (2, 16))
    dynamo.reset()
    exp = dynamo.explain(lambda t: m(t, t))(x)
    dynamo.reset()
    assert exp.graph_break_count > 0, \
        "sparse MoE no longer breaks the graph -- update the moe_impl docs"


@test
def compiled_forward_matches_eager():
    """aot_eager exercises Dynamo + AOTAutograd without needing a C++ compiler,
    so this runs on machines with no MSVC/gcc where Inductor cannot."""
    import torch._dynamo as dynamo
    for impl in ("sparse", "dense"):
        torch.manual_seed(0)
        m = AnuLM(tiny(moe_impl=impl)).eval()
        x = torch.randint(0, 64, (2, 16))
        with torch.no_grad():
            ref, _ = m(x, x)
        dynamo.reset()
        cm = torch.compile(m, backend="aot_eager")
        with torch.no_grad():
            got, _ = cm(x, x)
        dynamo.reset()
        err = (ref - got).abs().max().item()
        assert err == 0.0, f"{impl}: compiled output differs from eager by {err:.2e}"


@test
def compiled_backward_produces_gradients():
    import torch._dynamo as dynamo
    torch.manual_seed(0)
    m = AnuLM(tiny(moe_impl="dense")).train()
    x = torch.randint(0, 64, (8, 16))
    dynamo.reset()
    cm = torch.compile(m, backend="aot_eager")
    _, loss = cm(x, x)
    loss.backward()
    dynamo.reset()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, f"no gradient through the compiled graph for {missing[:4]}"


# =============================================================================
# KV CACHE / INCREMENTAL DECODING
# =============================================================================

@test
def incremental_forward_matches_full_forward():
    """Feeding a sequence in pieces through the cache must give the same last
    logits as feeding it whole -- for both attention variants, and for a
    multi-token chunk against a non-empty cache (the case `is_causal` gets wrong)."""
    for attn in ("gqa", "mla"):
        torch.manual_seed(0)
        m = AnuLM(tiny(attn=attn)).eval()
        x = torch.randint(0, 64, (2, 20))
        with torch.no_grad():
            full, _ = m(x, x)                         # (B, 20, V), all positions
            cache = m.new_cache()
            outs = []
            for lo, hi in ((0, 7), (7, 8), (8, 15), (15, 20)):   # prefill, 1, chunk, chunk
                lg, _ = m(x[:, lo:hi], cache=cache, start=lo)
                outs.append(lg[:, -1, :])
        for (lo, hi), got in zip(((0, 7), (7, 8), (8, 15), (15, 20)), outs):
            err = (got - full[:, hi - 1, :]).abs().max().item()
            assert err < 1e-4, f"{attn}: cached logits at position {hi-1} differ by {err:.2e}"


@test
def cache_holds_what_it_should():
    """GQA caches Hkv heads (not the H broadcast copies); MLA caches the latent
    plus one rope key -- the whole reason MLA exists."""
    torch.manual_seed(0)
    g = AnuLM(tiny(attn="gqa", n_head=4, n_kv_head=2, head_dim=8)).eval()
    cache = g.new_cache()
    with torch.no_grad():
        g(torch.randint(0, 64, (1, 9)), cache=cache)
    assert cache[0]["k"].shape == (1, 2, 9, 8), f"gqa cached {tuple(cache[0]['k'].shape)}, want (1, 2, 9, 8)"
    cfg = tiny(attn="mla", n_head=4, qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=8, kv_lora_rank=16)
    m = AnuLM(cfg).eval()
    cache = m.new_cache()
    with torch.no_grad():
        m(torch.randint(0, 64, (1, 9)), cache=cache)
    assert cache[0]["c"].shape == (1, 9, 16), f"mla cached {tuple(cache[0]['c'].shape)}, want (1, 9, 16)"
    assert cache[0]["k_pe"].shape == (1, 1, 9, 8), "mla must cache ONE rope key, shared by all heads"
    per_token_mla = 16 + 8
    per_token_gqa_equiv = 2 * 4 * (8 + 8)        # k and v for 4 heads of nope+rope
    assert per_token_mla < per_token_gqa_equiv


@test
def cached_generation_matches_uncached():
    """Greedy decoding with and without the cache must agree token-for-token,
    including after the context fills block_size and the cache is rebuilt."""
    for attn in ("gqa", "mla"):
        torch.manual_seed(0)
        cfg = tiny(attn=attn, block_size=24)
        m = AnuLM(cfg).eval()
        start = torch.randint(0, 64, (2, 6))
        a = m.generate(start.clone(), 30, temperature=1.0, top_k=1, use_cache=False)
        b = m.generate(start.clone(), 30, temperature=1.0, top_k=1, use_cache=True)
        assert a.shape == (2, 36)
        assert torch.equal(a, b), f"{attn}: cached and uncached decoding diverge at " \
            f"position {(a != b).nonzero()[0, 1].item() if (a != b).any() else '?'}"


@test
def generation_stops_at_eos():
    torch.manual_seed(0)
    cfg = tiny(vocab_size=64)
    m = AnuLM(cfg).eval()
    with torch.no_grad():
        m.lm_head.weight[63] += 5.0           # make id 63 overwhelmingly likely
    start = torch.randint(0, 63, (2, 4))
    out = m.generate(start, 50, temperature=1.0, top_k=1, eos_id=63)
    assert out.shape[1] < 54, "generate did not stop at eos"
    assert (out[:, 4:] == 63).any(1).all(), "eos never produced"
    # Without eos_id the same call runs to max_new_tokens.
    out2 = m.generate(start.clone(), 12, temperature=1.0, top_k=1)
    assert out2.shape == (2, 16)


@test
def rotary_cache_tracks_dtype_and_positions():
    cfg = tiny()
    rot = Rotary(cfg, dim=8)
    c32, _ = rot(8, torch.device("cpu"), torch.float32)
    assert c32.dtype == torch.float32
    c16, _ = rot(8, torch.device("cpu"), torch.bfloat16)
    assert c16.dtype == torch.bfloat16, "rotary handed back a stale-dtype table"
    full, _ = rot(16, torch.device("cpu"), torch.float32)
    tail, _ = rot(4, torch.device("cpu"), torch.float32, start=12)
    assert torch.equal(tail, full[12:16]), "start offset does not index absolute positions"


# =============================================================================
# SLIDING WINDOW
# =============================================================================

@test
def sliding_window_limits_reach():
    """On a windowed layer, a token more than `window` back must not influence
    the output; one inside the window must. Checked on the attention module
    directly (the residual stream would smear the effect across layers)."""
    torch.manual_seed(0)
    W = 4
    for attn in ("gqa", "mla"):
        cfg = tiny(attn=attn, sliding_window=W, max_window_layers=2)
        m = AnuLM(cfg).eval()
        a0, a2 = m.layers[0].attn, m.layers[2].attn      # windowed, and not
        assert m.layers[0].window == W and m.layers[2].window is None
        x = torch.randn(1, 12, cfg.hidden_size)
        cos, sin = m.rotary(12, x.device, x.dtype)
        with torch.no_grad():
            base = a0(x, cos, sin, window=W)[0, 11]
            far = x.clone(); far[0, 11 - W] += 3.0      # exactly W back: outside
            near = x.clone(); near[0, 11 - W + 1] += 3.0  # W-1 back: inside
            d_far = (a0(far, cos, sin, window=W)[0, 11] - base).abs().max().item()
            d_near = (a0(near, cos, sin, window=W)[0, 11] - base).abs().max().item()
            d_full = (a2(far, cos, sin)[0, 11] - a2(x, cos, sin)[0, 11]).abs().max().item()
        assert d_far == 0.0, f"{attn}: token outside the window moved the output by {d_far:.2e}"
        assert d_near > 1e-6, f"{attn}: token inside the window had no effect"
        assert d_full > 1e-6, f"{attn}: the non-windowed layer lost its reach"


@test
def sliding_window_agrees_with_cache():
    """The window must be exactly `window` keys wide on the cached path too.

    Generation alone cannot see an off-by-one window: cached and uncached
    decoding share _attn_mask, so a mask one key too wide (or narrow) agrees
    with itself. So the mask is checked against its definition -- query p sees
    keys p-W+1..p -- for a multi-token step against a cache, and a cached key
    exactly W back is perturbed (must not matter) next to one W-1 back (must).
    """
    from anulm.model import _attn_mask
    W = 5
    for S, start in ((1, 9), (3, 8), (12, 0)):
        T = start + S
        got = _attn_mask(S, T, start, W, "cpu")
        want = torch.tensor([[p - W < k <= p for k in range(T)]
                             for p in range(start, start + S)])
        assert torch.equal(got, want), f"S={S} start={start}: window mask is not {W} keys wide"

    torch.manual_seed(0)
    for attn in ("gqa", "mla"):
        cfg = tiny(attn=attn, sliding_window=W, max_window_layers=3, block_size=32)
        m = AnuLM(cfg).eval()
        a = m.layers[0].attn
        x = torch.randn(1, 11, cfg.hidden_size)
        cos, sin = m.rotary(11, x.device, x.dtype)

        def last(inp):                 # prefill 8 into the cache, then step 3
            cache = {}
            a(inp[:, :8], cos[:8], sin[:8], cache=cache, window=W)
            return a(inp[:, 8:], cos[8:], sin[8:], cache=cache, start=8, window=W)[0, 0]

        with torch.no_grad():
            base = last(x)
            far = x.clone(); far[0, 8 - W] += 3.0        # query 8, exactly W back
            near = x.clone(); near[0, 8 - W + 1] += 3.0  # W-1 back
            d_far = (last(far) - base).abs().max().item()
            d_near = (last(near) - base).abs().max().item()
        assert d_far == 0.0, f"{attn}: a cached key {W} back moved the output by {d_far:.2e}"
        assert d_near > 1e-6, f"{attn}: a cached key {W - 1} back had no effect"

        start_ids = torch.randint(0, 64, (1, 3))
        g1 = m.generate(start_ids.clone(), 20, top_k=1, use_cache=False)
        g2 = m.generate(start_ids.clone(), 20, top_k=1, use_cache=True)
        assert torch.equal(g1, g2), f"{attn}: windowed decoding diverges between cached and uncached"


# =============================================================================
# GROUPED-GEMM MoE + ROUTER REGULARISERS
# =============================================================================

@test
def grouped_moe_matches_sparse_under_bf16():
    """Same weights, same routing, one grouped GEMM per projection. Both run in
    bf16 (the grouped kernel's contract), so agreement is to bf16 precision."""
    import torch.nn.functional as F
    if not hasattr(F, "grouped_mm"):
        raise unittest.SkipTest("this torch has no F.grouped_mm")
    for attn in ("gqa", "mla"):
        torch.manual_seed(0)
        a = AnuLM(tiny(attn=attn, moe_impl="sparse")).eval()
        b = AnuLM(tiny(attn=attn, moe_impl="grouped")).eval()
        b.load_state_dict(a.state_dict())
        x = torch.randint(0, 64, (2, 16))
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            ya, _ = a(x, x)
            yb, _ = b(x, x)
        err = (ya.float() - yb.float()).abs().max().item()
        scale = ya.float().abs().max().item()
        assert err < 0.05 * scale, f"{attn}: grouped MoE differs from sparse by {err:.3e} (scale {scale:.2e})"


@test
def grouped_moe_trains():
    """Backward through grouped_mm with real (index-gathered) gradients: every
    expert that received a token gets a gradient, and none is NaN."""
    import torch.nn.functional as F
    if not hasattr(F, "grouped_mm"):
        raise unittest.SkipTest("this torch has no F.grouped_mm")
    torch.manual_seed(0)
    m = AnuLM(tiny(moe_impl="grouped")).train()
    x = torch.randint(0, 64, (16, 32))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, loss = m(x, x)
    loss.backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, f"no gradient for {missing[:4]}"
    assert all(torch.isfinite(p.grad).all() for p in m.parameters()), "non-finite gradient"


@test
def grouped_moe_compiles_without_graph_breaks():
    """The reason it exists: no `nonzero`, so Dynamo sees static shapes."""
    import torch.nn.functional as F
    import torch._dynamo as dynamo
    if not hasattr(F, "grouped_mm"):
        raise unittest.SkipTest("this torch has no F.grouped_mm")
    torch.manual_seed(0)
    m = AnuLM(tiny(moe_impl="grouped")).eval()
    x = torch.randint(0, 64, (2, 16))
    dynamo.reset()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        exp = dynamo.explain(lambda t: m(t, t))(x)
    dynamo.reset()
    assert exp.graph_break_count == 0, f"grouped MoE broke the graph {exp.graph_break_count} times"


@test
def router_regularisers_are_off_by_default_and_work_when_on():
    x = torch.randint(0, 64, (4, 16))
    torch.manual_seed(0)                  # identical weights for both models
    m0 = AnuLM(tiny()).train()
    _, l0 = m0(x, x)
    assert m0.aux_loss is None, "aux loss present with both alphas at 0"

    torch.manual_seed(0)
    m1 = AnuLM(tiny(seq_balance_alpha=1e-2, router_z_alpha=1e-3)).train()
    _, l1 = m1(x, x)
    assert m1.aux_loss is not None and torch.isfinite(m1.aux_loss)
    assert m1.aux_loss.item() > 0
    assert abs((l1 - l0).item() - m1.aux_loss.item()) < 1e-5, "loss != lm loss + aux"
    l1.backward()
    r = next(mod for mod in m1.modules() if isinstance(mod, Router))
    assert r.weight.grad is not None and r.weight.grad.abs().sum() > 0, "regulariser has no gradient path to the router"
    # Both regularisers are train-time only.
    m1.eval()
    with torch.no_grad():
        m1(x, x)
    assert m1.aux_loss is None


@test
def seq_balance_loss_matches_the_formula():
    """Check the implementation against DeepSeek-V3's definition computed by
    hand: with every score = sigmoid(0) = 0.5, P_i = 1/E exactly, and f comes
    from whatever topk chose; the loss must be mean over sequences of
    sum_i f_i * P_i (which is 1.0 only when selection is also uniform)."""
    torch.manual_seed(0)
    cfg = tiny(num_experts=4, num_experts_per_tok=2, seq_balance_alpha=1.0)
    r = Router(cfg).train()
    with torch.no_grad():
        r.weight.zero_()                    # every score = sigmoid(0) = 0.5 -> P_i = 1/E
    N, T = 8, 4
    x = torch.randn(N * T, cfg.hidden_size)
    idx, w, aux = r(x, seq_len=T)
    onehot = torch.nn.functional.one_hot(idx, 4).sum(1).float().view(N, T, 4).mean(1) * (4 / 2)
    P = torch.full((N, 4), 0.25)
    expected = (onehot * P).sum(-1).mean().item()
    assert abs(aux.item() - expected) < 1e-6, f"seq-balance loss {aux.item():.6f} != {expected:.6f}"


# =============================================================================
# DATA PIPELINE -- sampler, documents / EOS, loaders, prose filter
# =============================================================================

@test
def window_sampler_covers_every_window_once_per_epoch():
    from anulm.training.train import WindowSampler
    n, bs, B = 10_000, 64, 8
    s = WindowSampler(n, bs, B, seed=3)
    offsets, carried = [], []
    for epoch in range(3):
        offset = int(s.order.min())
        offsets.append(offset)
        expected = (n - 1 - offset) // bs
        starts = s.order.tolist()
        assert len(starts) == expected, f"{len(starts)} windows, expected {expected}"
        assert sorted(starts) == [offset + bs * i for i in range(expected)], "not a permutation of the offset grid"
        assert max(starts) + bs + 1 <= n, "last window runs off the end of the data"
        seen = list(carried)                         # the batch the rollover call already took
        while s.pos + B <= len(s.order):             # every remaining full batch of this epoch
            seen.extend(s.next().tolist())
        assert s.epoch == epoch, "epoch rolled over early"
        assert len(set(seen)) == len(seen) and set(seen) <= set(starts), "a window was repeated or invented"
        assert len(seen) >= expected - B, "an epoch left more than one batch of windows unvisited"
        carried = s.next().tolist()                  # the partial remainder triggers the next epoch
        assert s.epoch == epoch + 1
    assert len(set(offsets)) > 1, "epoch offsets never move, so window boundaries never move"


@test
def window_sampler_resumes_exactly():
    from anulm.training.train import WindowSampler
    a = WindowSampler(50_000, 128, 4, seed=11)
    for _ in range(137):
        a.next()
    state = a.state()
    b = WindowSampler(50_000, 128, 4, seed=999)     # different seed: state must win
    b.load_state(state)
    for _ in range(300):                             # crosses an epoch boundary
        assert torch.equal(a.next(), b.next()), "resumed sampler diverged from the original"
    assert a.epoch == b.epoch and a.pos == b.pos


@test
def bpe_documents_get_eos_and_the_vocab_is_exact():
    from anulm.bpe import BPE, DOC_SEP
    from anulm.training.train import FALLBACK
    tok = BPE.train(FALLBACK * 20, 300, verbose=False)
    # EOS is the last id; `--vocab N` means N ids INCLUDING it (fewer only if
    # the text runs out of pairs worth merging before the cap).
    assert tok.vocab_size == 256 + len(tok.merges) + 1 <= 300, \
        f"vocab {tok.vocab_size}, merges {len(tok.merges)}"
    assert tok.eos_id == tok.vocab_size - 1 and tok.eos_id not in tok.vocab
    docs = ["पहला लेख\n\nयह पहला है।", "दूसरा लेख\n\nयह दूसरा है।", "तीसरा"]
    ids = tok.encode_documents(DOC_SEP.join(docs) + DOC_SEP)
    assert ids.count(tok.eos_id) == 3, "one EOS per document"
    assert tok.decode(ids) == "".join(docs), "decode must drop EOS and round-trip the text"
    assert tok.eos_id not in tok.encode(docs[0]), "plain encode must never emit EOS"
    # a document-less corpus is one document
    assert tok.encode_documents("कुछ पाठ").count(tok.eos_id) == 1


@test
def the_demo_page_offers_the_right_modes_for_each_checkpoint():
    """app.py's dropdown reshapes the page around whichever checkpoint is
    loaded. The reshaping is pure functions of the checkpoint's own info dict,
    so it can be checked without weights, a browser or gradio running -- which
    is the point of keeping them out of the UI callbacks."""
    try:
        import gradio  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("gradio not installed")
    from anulm.apps import app

    # The four kinds serve.py distinguishes, as /info reports them.
    infos = {
        "coder": {"coder": True, "translate": False, "qa_template": "Question: {q}"},
        "translate": {"coder": False, "translate": True, "qa_template": "English: {q}"},
        "qa": {"coder": False, "translate": False, "qa_template": "प्रश्न: {q}"},
        "base": {"coder": False, "translate": False, "qa_template": None},
    }
    for want, info in infos.items():
        assert app.kind_of(info) == want, (want, app.kind_of(info))

    assert list(app.labels_for("coder").values()) == ["write a function", "continue the code"]
    assert list(app.labels_for("translate").values()) == ["translate", "continue the text"]
    assert list(app.labels_for("base").values()) == ["continue the text"],         "a base model must not be offered a mode it has no template for"
    for kind in ("coder", "translate", "qa", "base"):
        ex = app.examples_for(kind)
        assert ex and all(len(row) == 2 for row in ex), kind
        labels = set(app.labels_for(kind).values())
        assert {row[1] for row in ex} <= labels, f"{kind}: an example names a missing mode"
        assert 0.1 <= app.default_temp(kind) <= 1.5

    # Every entry in the picker is a Hub repo id, and the loader holds one model.
    assert all("/" in repo for repo in app.MODELS.values()), app.MODELS
    loader = app.Loader("cpu")
    assert loader.engine is None and loader.source is None

    # The header must name the model, not the directory it was downloaded to.
    # snapshot_download returns .../snapshots/<commit sha>, and Engine takes a
    # checkpoint's name from its path, so the page showed a 40-character hash
    # for every model loaded from the Hub until Loader overrode it.
    import json, tempfile, dataclasses
    from pathlib import Path as P
    try:
        from safetensors.torch import save_file
    except ImportError:
        raise unittest.SkipTest("safetensors not installed; the Hub-name half of this test did not run")
    # 259 = the byte vocabulary (256 bytes + BOS/EOS/PAD). Engine warms itself
    # up by generating from a Devanagari prompt encoded as raw UTF-8 bytes, so
    # a narrower vocab indexes off the end of the embedding.
    cfg = tiny(tokenizer_path=None, vocab_size=259)
    torch.manual_seed(0)
    m = AnuLM(cfg).eval()
    with tempfile.TemporaryDirectory() as d:
        folder = P(d) / "snapshots" / "0123456789abcdef0123456789abcdef01234567"
        folder.mkdir(parents=True)
        save_file({k: v.contiguous() for k, v in m.state_dict().items()},
                  str(folder / "model.safetensors"))
        (folder / "config.json").write_text(json.dumps(
            {"model_type": "anulm", "config": dataclasses.asdict(cfg), "step": 1}),
            encoding="utf-8")
        engine = loader.load(str(folder))
        assert engine.info["ckpt"] == str(folder), engine.info["ckpt"]
        assert app.header_for(engine).startswith(f"**{folder}**")
        # and the loader holds exactly this one
        assert loader.source == str(folder) and loader.engine is engine


@test
def an_exported_folder_loads_everywhere_a_pt_does():
    """docs/TUTORIAL.md section 11 and every model card say the same thing:
    "Every script accepts an exported folder wherever it accepts a .pt".

    That was not true. Five evaluation scripts called torch.load directly, so
    anyone who downloaded a checkpoint from the Hub and ran eval_translate.py,
    eval_qa.py, eval_golden.py, eval_bench.py or eval_context.py on it hit a
    crash on the first line that touched the file. This checks the promise
    rather than the prose: the folder path loads, and no script that takes a
    checkpoint reads it with torch.load any more.
    """
    import json, re, tempfile
    from pathlib import Path as P
    try:
        from safetensors.torch import save_file
    except ImportError:
        raise unittest.SkipTest("safetensors not installed")
    import dataclasses
    from anulm.model import load_checkpoint

    cfg = tiny()
    torch.manual_seed(0)
    m = AnuLM(cfg).eval()
    with tempfile.TemporaryDirectory() as d:
        folder = P(d) / "Export-Test"
        folder.mkdir()
        sd = {k: v.contiguous() for k, v in m.state_dict().items()}
        save_file(sd, str(folder / "model.safetensors"))
        (folder / "config.json").write_text(json.dumps({
            "model_type": "anulm", "config": dataclasses.asdict(cfg),
            "step": 7, "val_loss": 1.25,
        }), encoding="utf-8")

        ck = load_checkpoint(str(folder), "cpu")
        assert ck["step"] == 7 and abs(ck["val_loss"] - 1.25) < 1e-9
        assert ck["cfg"].n_layer == cfg.n_layer and ck["cfg"].num_experts == cfg.num_experts
        again = AnuLM(ck["cfg"]).eval()
        again.load_state_dict(ck["model"])
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        with torch.no_grad():
            a, _ = m(x)
            b, _ = again(x)
        assert torch.equal(a, b), "a folder round trip changed the model"

    # The promise, enforced on the scripts themselves.
    root = P(__file__).resolve().parents[1]
    for name in ("eval_bench.py", "eval_context.py", "eval_golden.py", "eval_qa.py",
                 "eval_translate.py", "eval_code.py", "sample.py", "sample_many.py",
                 "ask.py", "serve.py", "finetune.py"):
        src = next(root.glob(f"anulm/**/{name}")).read_text(encoding="utf-8")
        assert "load_checkpoint" in src, f"{name} does not use load_checkpoint"
        bad = re.search(r"torch\.load\(\s*(args\.)?ckpt", src)
        assert not bad, f"{name} still reads a checkpoint with torch.load: {bad.group(0)}"


@test
def eval_windows_are_frozen_and_strided_windows_cover_the_split():
    """Two contracts about how val loss is measured.

    The first is a regression guard with teeth: every val loss in
    docs/RESULTS.md was read through `eval_windows`, so if it ever returns
    different indices, every number in that file silently stops meaning what
    it says. These are the indices it has always returned.

    The second is what `--eval-windows` promises instead: the same N windows
    whatever the batch size, spread over the whole split, no duplicates.
    """
    import argparse
    from anulm.training.train import eval_windows, strided_windows, evaluate

    data = torch.arange(100_000)
    w = eval_windows(data, 8, 512, 20)
    assert len(w) == 20 and all(len(b) == 8 for b in w)
    assert w[0][:4].tolist() == [90040, 77448, 53808, 91071], w[0][:4].tolist()
    assert w[-1][:4].tolist() == [9859, 87228, 2271, 62460], w[-1][:4].tolist()

    flat = lambda bs: [int(i) for b in bs for i in b]
    a, b = strided_windows(data, 8, 512, 100), strided_windows(data, 16, 512, 100)
    assert flat(a) == flat(b), "the window set must not depend on batch size"
    assert len(set(flat(a))) == 100, "strided windows must not repeat"
    assert all(len(x) == 8 for x in a[:-1]) and len(flat(a)) == 100
    span = len(data) - 512 - 1
    assert min(flat(a)) < span * 0.02 and max(flat(a)) > span * 0.98, "must cover the split"
    assert flat(strided_windows(data, 8, 512, 100)) == flat(a), "must be deterministic"
    # asking for more windows than exist is clamped, not an error
    assert len(flat(strided_windows(torch.arange(600), 8, 512, 1000))) <= 600

    # evaluate() reports the mean and the standard error of that mean.
    cfg = tiny()
    torch.manual_seed(0)
    m = AnuLM(cfg).eval()
    d = torch.randint(0, cfg.vocab_size, (4000,))
    args = argparse.Namespace(batch_size=4, block_size=cfg.block_size, device="cpu",
                              autocast=contextlib.nullcontext(), eval_iters=6, eval_windows=0)
    mean, sem = evaluate(m, d, args)
    assert math.isfinite(mean) and 0.0 < sem < 1.0, (mean, sem)
    args.eval_windows = 24
    mean2, sem2 = evaluate(m, d, args)
    assert math.isfinite(mean2) and math.isfinite(sem2)
    # one batch cannot have a spread, and must say so rather than claim zero
    args.eval_windows, args.batch_size = 4, 4
    _, sem1 = evaluate(m, d, args)
    assert math.isnan(sem1), sem1


@test
def lora_starts_as_the_base_model_and_merges_back_exactly():
    """Three promises LoRA has to keep, and one this model makes it break.

    B is initialised to zero, so an adapted model is numerically identical to
    the one it wrapped until the first step -- a fine-tune cannot start worse
    than the checkpoint it began from. Only the adapters get gradients.
    Merging folds B@A into the base weight and gives back an ordinary AnuLM
    that serve.py and export_hf.py load without knowing LoRA exists.

    The fourth: the aux-loss-free balancer's `expert_bias` is a buffer, so it
    has no gradient and LoRA does not freeze it -- it keeps moving every step.
    Merging adapters into a pristine base therefore restores the *base's*
    routing, not the routing that was trained, and the model changes which
    experts a token reaches. The adapter carries the bias for that reason.
    """
    from anulm.lora import (apply_lora, lora_parameters, lora_state_dict, load_lora,
                      merge_lora, router_state_dict)

    # One balancer step at the default 1e-3 re-routes nothing here; 0.1 stands
    # in for the drift of a whole fine-tune, so the bias path is exercised.
    cfg = tiny(bias_update_rate=0.1)
    torch.manual_seed(0)
    base = AnuLM(cfg).eval()
    base_sd = {k: v.clone() for k, v in base.state_dict().items()}
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        want, _ = base(x)

    model = AnuLM(cfg).eval()
    model.load_state_dict(base_sd)
    n = apply_lora(model, r=4, alpha=8.0)
    assert n > 0
    with torch.no_grad():
        got, _ = model(x)
    # Not bit-identical: the wrapped layer adds a zero tensor, and that extra
    # add is not always the same instruction sequence as the fused one it
    # replaced. The claim is that step 0 is the base model to float noise.
    d0 = (want - got).abs().max().item()
    assert d0 < 1e-6, f"B must start at zero: step 0 moved the logits by {d0}"

    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in name for name in trainable), \
        f"something other than the adapters is trainable: {sorted(trainable)[:3]}"

    # One real step, so the adapters stop being zero -- and the balancer's
    # update after it, as the training loop does, so the router bias drifts.
    opt = torch.optim.AdamW(lora_parameters(model), lr=1e-2)
    model.train()
    _, loss = model(x, x)
    loss.backward()
    opt.step()
    model.update_expert_biases()
    model.eval()
    drifted = [n for n, b in model.named_buffers()
               if n.endswith("expert_bias") and not torch.equal(b, base_sd[n])]
    assert drifted, "update_expert_biases() did not move any router bias"
    with torch.no_grad():
        adapted, _ = model(x, x)          # every position, so a re-route shows
    assert not torch.equal(adapted[:, -1:], want), "after a step the adapters must do something"

    # Save the adapter, rebuild from a pristine base, and land in the same place.
    ad, router = lora_state_dict(model), router_state_dict(model)
    assert all("lora_" in k for k in ad) and ad
    assert router and all(k.endswith("expert_bias") for k in router)
    size = sum(v.numel() for v in ad.values())
    assert size < sum(p.numel() for p in base.parameters()) * 0.1, \
        "an adapter that big is not worth the machinery"

    rebuilt = AnuLM(cfg).eval()
    rebuilt.load_state_dict(base_sd)
    apply_lora(rebuilt, r=4, alpha=8.0)
    load_lora(rebuilt, ad)
    rebuilt.load_state_dict(router, strict=False)
    merge_lora(rebuilt)
    assert not any("lora" in n for n, _ in rebuilt.named_parameters()), \
        "a merged model must be a plain AnuLM again"
    for n in drifted:
        assert torch.equal(rebuilt.state_dict()[n], model.state_dict()[n]), \
            f"{n}: the merged model did not get the trained router bias back"
    with torch.no_grad():
        merged_out, _ = rebuilt(x, x)
    d = (adapted - merged_out).abs().max().item()
    assert d < 1e-4, f"merging changed the model by {d}"

    # And without the router bias it does NOT come back the same, which is
    # why the adapter carries it: the base's bias routes some token elsewhere.
    naive = AnuLM(cfg).eval()
    naive.load_state_dict(base_sd)
    apply_lora(naive, r=4, alpha=8.0)
    load_lora(naive, ad)
    merge_lora(naive)
    with torch.no_grad():
        naive_out, _ = naive(x, x)
    d_naive = (adapted - naive_out).abs().max().item()
    assert d_naive > 1e-3, \
        f"dropping the trained router bias changed nothing ({d_naive:.1e}): no drift exercised"


@test
def lora_scale_is_alpha_over_r():
    """A trained adapter adds (alpha / r) * B @ A -- in the forward pass and
    in the merged weight alike. The usual LoRA test cannot see the scale: B
    starts at zero, and if forward and merge share one wrong scale they still
    agree with each other. So B is set by hand and checked against the formula."""
    from anulm.lora import LoRALinear
    torch.manual_seed(0)
    base = torch.nn.Linear(12, 10, bias=False)
    lin = LoRALinear(base, r=4, alpha=8.0)
    with torch.no_grad():
        lin.lora_b.copy_(torch.randn(10, 4))
    x = torch.randn(3, 12)
    delta = (8.0 / 4) * lin.lora_b @ lin.lora_a
    with torch.no_grad():
        got = lin(x)
    want = x @ (base.weight + delta).t()
    assert torch.allclose(got, want, atol=1e-5), "LoRA forward is not W + (alpha/r) B A"
    assert torch.allclose(lin.merged_weight(), base.weight + delta, atol=1e-6), \
        "merged weight is not W + (alpha/r) B A"


@test
def an_exported_qa_checkpoint_still_offers_its_question_mode():
    """The released question answerer lost its headline feature when loaded
    the way everyone loads it.

    finetune.py writes both `qa_template` (a single Hindi string, legacy) and
    `qa_templates` (one per language) into a .pt. export_hf.py writes only the
    plural. serve.py's Engine read only the singular, so a checkpoint loaded
    from an exported folder reported qa_template None, /info described a plain
    base model, and both web pages hid "answer a question" -- on the model
    whose entire purpose is answering questions. Asking it directly worked the
    whole time, which is why nothing caught it.
    """
    try:
        import gradio  # noqa: F401  -- kind_of lives in app.py
    except ImportError:
        raise unittest.SkipTest("gradio not installed")
    try:
        from safetensors.torch import save_file
    except ImportError:
        raise unittest.SkipTest("safetensors not installed")
    import dataclasses, json, tempfile
    from pathlib import Path as P
    from anulm.apps import app
    from anulm.inference.serve import Engine

    cfg = tiny(vocab_size=259)
    torch.manual_seed(0)
    m = AnuLM(cfg).eval()
    templates = {"hi": "प्रश्न: {q}\nउत्तर:", "en": "Question: {q}\nAnswer:"}
    with tempfile.TemporaryDirectory() as d:
        folder = P(d) / "Export-QA"
        folder.mkdir()
        save_file({k: v.contiguous() for k, v in m.state_dict().items()},
                  str(folder / "model.safetensors"))
        # exactly what export_hf.py writes: the plural, never the singular
        (folder / "config.json").write_text(json.dumps({
            "model_type": "anulm", "config": dataclasses.asdict(cfg),
            "step": 3, "qa_templates": templates,
        }, ensure_ascii=False), encoding="utf-8")

        engine = Engine(str(folder), "cpu")
        assert engine.info["qa_template"], \
            "an export carrying qa_templates must still report a qa_template"
        assert app.kind_of(engine.info) == "qa", app.kind_of(engine.info)
        assert "answer a question" in app.labels_for("qa").values()
        # and the mode must actually run rather than silently fall back
        r = engine.generate("test?", 4, 0.8, 50, 0, mode="question")
        assert r["mode"] == "question", r["mode"]

        # A checkpoint with no templates at all is still a base model.
        (folder / "config.json").write_text(json.dumps({
            "model_type": "anulm", "config": dataclasses.asdict(cfg), "step": 3,
        }), encoding="utf-8")
        plain = Engine(str(folder), "cpu")
        assert not plain.info["qa_template"] and app.kind_of(plain.info) == "base"


@test
def an_export_keeps_what_makes_a_checkpoint_more_than_a_language_model():
    """export_hf.py used to write `model` and `cfg` and nothing else, so a
    speech model lost its projector, a Decide adapter its LoRA weights, head
    and labels, and the folder still loaded -- as a plain language model. The
    extras must come back under the keys the .pt had, and an adapter exported
    without its backbone must find that backbone beside it."""
    try:
        from safetensors.torch import save_file  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("safetensors not installed")
    import sys as _sys, tempfile
    from pathlib import Path as P
    from anulm.hf import export_hf
    from anulm.model import load_checkpoint

    torch.manual_seed(0)
    cfg = tiny()
    backbone = {k: v.clone() for k, v in AnuLM(cfg).state_dict().items()}
    with tempfile.TemporaryDirectory() as d:
        d = P(d)
        torch.save({"model": backbone, "cfg": cfg, "step": 7}, d / "chat.pt")
        adapter = {"model": backbone, "cfg": cfg, "step": 9, "mode": "lora", "lora_rank": 4,
                   "labels": ["a", "b"], "temperature": 2.1,
                   "lora": {"w.A": torch.randn(4, 32), "w.B": torch.randn(32, 4)},
                   "decide_head": {"weight": torch.randn(1, 32)}}
        torch.save(adapter, d / "adapter.pt")
        argv = _sys.argv
        try:
            _sys.argv = ["export_hf.py", str(d / "chat.pt"), str(d / "Chat"), "--license", "mit"]
            export_hf.main()
            _sys.argv = ["export_hf.py", str(d / "adapter.pt"), str(d / "Adapter"), "--kind", "decide",
                         "--license", "mit", "--adapter-of", str(d / "chat.pt"), "--base-model", "someone/Chat"]
            export_hf.main()
        finally:
            _sys.argv = argv
        assert not (d / "Adapter" / "model.safetensors").exists(), "an adapter export must not copy its backbone"
        ck = load_checkpoint(d / "Adapter")
        for k in ("mode", "lora_rank", "labels", "temperature", "step"):
            assert ck[k] == adapter[k], (k, ck[k], adapter[k])
        for g in ("lora", "decide_head"):
            assert all(torch.equal(ck[g][n], v) for n, v in adapter[g].items()), g
        # the backbone, from the folder named after --base-model beside the adapter
        assert ck["cfg"].hidden_size == cfg.hidden_size
        assert all(torch.allclose(ck["model"][k].float(), v, rtol=1e-2, atol=1e-2) for k, v in backbone.items())
        assert "model" not in load_checkpoint(d / "Adapter", base=False)


@test
def eval_code_sandbox_contains_generated_code():
    """eval_code.py runs a language model's unreviewed output. These are the
    containment promises its docstring makes, and the ones that broke before:
    a program cannot leave anything behind for the next one, cannot outlive
    its timeout, and cannot leave a child running after it is killed."""
    import os, tempfile, time
    from anulm.eval.eval_code import POSIX, run_program

    with tempfile.TemporaryDirectory() as wd:
        assert run_program("print('ok')", wd, timeout=20) is True
        assert run_program("raise SystemExit(1)", wd, timeout=20) is False
        assert run_program("assert 1 == 2", wd, timeout=20) is False

        # stdin is closed, not inherited: reading it ends rather than hangs.
        assert run_program("import sys; assert sys.stdin.read() == ''", wd, timeout=20) is True

        # A runaway program is killed, and does not take the run with it.
        t0 = time.time()
        assert run_program("while True: pass", wd, timeout=3) is False
        assert time.time() - t0 < 25, "the timeout did not fire"

        # Each program gets its own directory. Without that, a generated file
        # called random.py shadows the standard library for everything after
        # it -- a silent, one-directional corruption of the benchmark.
        run_program("open('random.py','w').write('raise RuntimeError()')", wd, timeout=20)
        assert run_program("import random; random.randint(1, 2)", wd, timeout=20) is True
        assert os.listdir(wd) == [], f"scratch left behind: {os.listdir(wd)}"

    # A child started by the program must not outlive the kill. The marker is
    # written outside the scratch directory, so its absence is the evidence --
    # but only if the grandchild was started at all, so the program also
    # writes `spawned.txt` once Popen has returned. Without that, a program
    # that failed before Popen (a bad path, a sandbox refusing to spawn) made
    # this check pass without testing anything.
    with tempfile.TemporaryDirectory() as outside:
        marker = os.path.join(outside, "survived.txt").replace("\\", "\\\\")
        spawned = os.path.join(outside, "spawned.txt").replace("\\", "\\\\")
        src = ("import subprocess, sys, time\n"
               f"subprocess.Popen([sys.executable, '-c', \"import time; time.sleep(3);"
               f" open(r'{marker}','w').write('x')\"])\n"
               f"open(r'{spawned}','w').write('x')\n"
               "time.sleep(60)\n")
        with tempfile.TemporaryDirectory() as wd:
            assert run_program(src, wd, timeout=2) is False
        time.sleep(5)
        assert os.path.exists(os.path.join(outside, "spawned.txt")), \
            "the grandchild was never started, so the kill was not tested"
        assert not os.path.exists(os.path.join(outside, "survived.txt")), \
            "a grandchild outlived the timeout kill"

    if POSIX:
        # setrlimit exists here, so the memory cap is real. On Windows it is
        # not, which the module docstring says plainly.
        with tempfile.TemporaryDirectory() as wd:
            hog = f"x = bytearray({1024 * 1024 * 1024} * 8)"      # 8 GB
            assert run_program(hog, wd, timeout=30) is False, \
                "RLIMIT_AS did not stop an 8 GB allocation"


@test
def transformers_wrapper_matches_the_native_model():
    """modeling_anulm.py must produce this repository's numbers, not its own.

    The wrapper exists so `AutoModelForCausalLM.from_pretrained(...,
    trust_remote_code=True)` works without a clone. It builds the real modules
    from model.py, so the danger is not the maths but the plumbing: a
    transformers version that materialises the model on the meta device leaves
    the rotary tables -- registered non-persistently, so absent from the
    checkpoint -- as uninitialised memory. Every weight then loads, nothing
    warns, and the positions are noise. That happened; this is the guard.
    """
    try:
        import transformers  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("transformers not installed")
    import tempfile
    from anulm.hf.configuration_anulm import AnuLMConfig as HFConfig
    from anulm.hf.modeling_anulm import AnuLMForCausalLM

    cfg = tiny()
    torch.manual_seed(0)
    native = AnuLM(cfg).eval()

    hf = AnuLMForCausalLM(HFConfig.from_native(cfg)).eval()
    missing, unexpected = hf.load_state_dict(native.state_dict(), strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    assert torch.equal(native.rotary.inv_freq, hf.rotary.inv_freq), "rotary tables differ"

    x = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        nat, _ = native(x)                      # last position only
        out = hf(input_ids=x)
    assert out.logits.shape == (2, 16, cfg.vocab_size), out.logits.shape
    # Not bit-identical on purpose: the native path multiplies one row through
    # lm_head, this one multiplies all of them, and the two GEMMs reduce in a
    # different order. Same maths, same weights, different rounding.
    d = (nat[:, -1] - out.logits[:, -1]).abs().max().item()
    assert d < 1e-4, f"logits differ by {d}"

    # What actually has to hold: the same tokens come out, cached or not.
    with torch.no_grad():
        want = native.generate(x[:1], 12, temperature=1.0, top_k=1)[0].tolist()
        for use_cache in (True, False):
            got = hf.generate(input_ids=x[:1], max_new_tokens=12, do_sample=False,
                              use_cache=use_cache, pad_token_id=0)[0].tolist()
            assert got == want, f"use_cache={use_cache}: {got} != {want}"

    # A round trip through disk is what a user actually does.
    with tempfile.TemporaryDirectory() as d_:
        hf.save_pretrained(d_)
        again = AnuLMForCausalLM.from_pretrained(d_, dtype=torch.float32).eval()
        assert torch.equal(again.rotary.inv_freq, native.rotary.inv_freq), \
            "rotary tables were not rebuilt after from_pretrained"
        with torch.no_grad():
            back = again(input_ids=x).logits
        # Not bit-identical, and that is not the round trip's fault: the
        # reloaded model rebuilds its rotary tables from scratch while the
        # original had already cached them from the forward above, and the
        # two agree to about 5e-08. The weights themselves are equal.
        for a, b in zip(hf.state_dict().values(), again.state_dict().values()):
            assert torch.equal(a, b), "save/load changed a weight"
        d2 = (back - out.logits).abs().max().item()
        assert d2 < 1e-6, f"save/load round trip moved the logits by {d2}"
        assert torch.equal(back.argmax(-1), out.logits.argmax(-1)),             "save/load round trip changed the predicted tokens"


@test
def hf_tokenizer_conversion_is_exact():
    """tokenizer.json must agree with bpe.py on every token, or the numbers in
    docs/RESULTS.md do not describe the model people download."""
    try:
        from tokenizers import Tokenizer
    except ImportError:
        raise unittest.SkipTest("tokenizers not installed")
    import tempfile
    from pathlib import Path
    from anulm.hf import hf_tokenizer as conv
    from anulm.bpe import BPE
    from anulm.training.train import FALLBACK

    tok = BPE.train(FALLBACK * 8, 400, verbose=False)
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "tokenizer.json"
        out.write_text(__import__("json").dumps(conv.build(tok), ensure_ascii=False),
                       encoding="utf-8")
        fast = Tokenizer.from_file(str(out))
        probes = conv.PROBES + [FALLBACK[:2000], FALLBACK[3000:5000]]
        for text in probes:
            ours, theirs = tok.encode(text), fast.encode(text, add_special_tokens=False).ids
            assert ours == theirs, f"{text[:40]!r}: {ours[:12]} != {theirs[:12]}"
            assert tok.decode(ours) == text, f"round trip failed on {text[:40]!r}"


@test
def bpe_files_load_under_both_format_ids():
    """The project was renamed on 2026-09-18 and the tokenizer format id with
    it. Every tokenizer trained before then -- including the four beside the
    released weights on the Hub -- says "nanosarvam-bpe", so load() must keep
    taking it, and the committed tokenizers under data/ must keep loading."""
    import glob, json, os, tempfile
    from anulm.bpe import BPE
    assert BPE.TYPE == "anulm-bpe" and "nanosarvam-bpe" in BPE.TYPES
    tok = BPE.train("दो शब्द, two words, def f(): " * 40, 300, verbose=False)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.json")
        tok.save(p)
        assert json.load(open(p, encoding="utf-8"))["type"] == "anulm-bpe",             "save must write the current id"
        for legacy in BPE.TYPES:
            blob = json.load(open(p, encoding="utf-8"))
            blob["type"] = legacy
            json.dump(blob, open(p, "w", encoding="utf-8"))
            assert BPE.load(p).merges == tok.merges, f"{legacy} must still load"
        blob["type"] = "gpt2"                      # anything else is refused
        json.dump(blob, open(p, "w", encoding="utf-8"))
        # ValueError, not an assert: `python -O` strips asserts, which let the
        # foreign file through. And the failure is an AssertionError raised
        # outside the try -- the SystemExit this used to raise is not an
        # Exception, so it ended the whole run instead of failing one test.
        try:
            BPE.load(p)
        except ValueError:
            pass
        else:
            raise AssertionError("a foreign tokenizer file must not load")
    # The COMMITTED tokenizers, not every json that happens to be in data/.
    # data/ is gitignored except the tokenizers, so git's tracked list is exactly
    # that set; globbing picked up untracked working files (contamination
    # reports, say) and failed on them. Filtering to "files that look like a
    # tokenizer" would be worse: a corrupted committed tokenizer could stop
    # looking like one, be skipped, and let this test pass over the breakage it
    # exists to catch. Without git, fall back to the glob.
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        import subprocess
        tracked = subprocess.run(["git", "ls-files", "--", "data/*.json"], cwd=here,
                                 capture_output=True, text=True, check=True).stdout.split()
        shipped = sorted(os.path.join(here, t) for t in tracked)
    except (OSError, subprocess.CalledProcessError):
        shipped = sorted(glob.glob(os.path.join(here, "data", "*.json")))
    assert shipped, "the committed tokenizers under data/ went missing"
    for f in shipped:
        assert BPE.load(f).merges, f"{f} failed to load"


@test
def byte_loader_inserts_eos_at_document_boundaries():
    import os, tempfile
    from anulm.bpe import DOC_SEP
    from anulm.training.train import BYTE_EOS, load_bytes
    docs = ["शीर्षक\n\nपहला दस्तावेज़।", "दूसरा\n\nदूसरा दस्तावेज़।"]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.txt")
        # CRLF on purpose: what a Windows text-mode write produces
        raw = (DOC_SEP.join(docs) + DOC_SEP).replace("\n", "\r\n").encode("utf-8")
        open(p, "wb").write(raw)
        data = load_bytes(p)
        assert data.dtype == torch.int16, "EOS does not fit uint8; loader must widen"
        assert int((data == BYTE_EOS).sum()) == 2, "one EOS per document"
        assert b"\r" not in bytes(data[data < 256].to(torch.uint8).tolist()), "CRLF was not normalised"
        body = bytes(data[data < 256].to(torch.uint8).tolist()).decode("utf-8")
        assert body == "".join(docs).replace("\n", "\n"), "bytes between the EOS are not the documents"
        plain = os.path.join(d, "plain.txt")
        open(plain, "wb").write("बिना सीमा".encode("utf-8"))
        assert load_bytes(plain).dtype == torch.uint8, "a corpus without DOC_SEP must stay uint8"


@test
def prose_filter_drops_lists_and_keeps_prose():
    from anulm.data.filter_prose import filter_doc
    prose = "यह एक लम्बा वाक्य है जो पूरी तरह से गद्य में लिखा गया है और दंड पर समाप्त होता है।"
    doc = "शीर्षक\n\n" + prose + "\nसड़क मार्ग\nरेलवे स्टेशन - दिल्ली\n" + prose + "\n" + prose
    out = filter_doc(doc, min_line_chars=40, min_sentence_frac=0.4, min_chars=50)
    assert out is not None and out.startswith("शीर्षक\n\n"), "title must survive as the first line"
    assert "सड़क मार्ग" not in out and "रेलवे स्टेशन" not in out, "short sentence-less lines must go"
    assert out.count(prose) == 3, "prose lines must all survive"
    listy = "शीर्षक\n\n" + "\n".join(f"पुरस्कार - {2000 + i}" for i in range(30))
    assert filter_doc(listy, 40, 0.4, 50) is None, "a list-shaped article must be dropped"


@test
def golden_items_are_well_formed_and_the_scorer_runs():
    """The golden set's contract: the prefix ends on the word before the blank
    with exactly one space between, the gold word is in the prefix for
    in_context items and absent for novel ones, the choices hold the gold
    word once at answer_idx and no distractor appears in the prefix. Then
    the scorer must run end to end on a byte checkpoint."""
    import os, random, tempfile
    from anulm.data.make_golden import build, DOC_SEP
    from anulm.eval.eval_golden import score
    rng = random.Random(0)
    # A few hundred distinct words, not a few dozen: with a small vocabulary
    # every word recurs inside any 60-word window, and a builder that looked
    # for the gold word anywhere earlier in the document instead of inside
    # the prefix it actually emits would pass by luck. It did, once.
    syl = "का ति रु मे लो सा नी पू धा बे गि हो".split()
    vocab = [a + b + c for a in syl for b in syl for c in syl][:400]
    docs = []
    for d in range(30):
        words = [rng.choice(vocab) for _ in range(120)]
        # Twelve-word sentences ending on a danda, so every line is prose.
        lines = [" ".join(words[i: i + 12]) + "।" for i in range(0, 120, 12)]
        docs.append(f"शीर्षक {d}\n\n" + "\n".join(lines))
    items = build(DOC_SEP.join(docs), "synthetic", per_type=4, context_words=60,
                  min_words=80, rng=rng)
    assert items, "the builder found no items in a prose corpus"
    for it in items:
        assert it["raw_next"].split()[0].rstrip("।") == it["answer"], "the blank is not the next word"
        assert not it["prefix"].endswith((" ", "\n")), "prefix must end on the previous word"
        prev = {w.rstrip("।") for w in it["prefix"].split()}
        assert (it["answer"] in prev) == (it["type"] == "in_context"), it["type"]
        assert it["choices"].count(it["answer"]) == 1 and len(it["choices"]) == 4
        assert it["choices"][it["answer_idx"]] == it["answer"]
        assert all(c not in prev for c in it["choices"] if c != it["answer"]), \
            "a distractor must not already appear in the prefix"

    cfg = tiny(vocab_size=259, block_size=64)
    m = AnuLM(cfg)
    with tempfile.TemporaryDirectory() as d:
        ck = os.path.join(d, "t.pt")
        torch.save({"model": m.state_dict(), "cfg": cfg, "step": 0, "val_loss": 0.0}, ck)
        r = score(ck, items[:3], "cpu", max_gen=6)
    assert r["cells"]["all"]["n"] == 3
    assert 0.0 <= r["cells"]["all"]["mc"] <= 1.0 and 0.0 <= r["cells"]["all"]["em"] <= 1.0


@test
def qa_packing_masks_prompts_and_keeps_examples_whole():
    """finetune.py's contract: the loss lands on answer tokens (plus EOS) and
    nowhere else, an example never straddles two rows, and prompt + answer
    tokenise exactly as the joined string does, so what the model is trained
    on is what serve.py will feed it."""
    from anulm.bpe import BPE
    from anulm.training.finetune import encode_pairs, pack
    from anulm.data.make_qa import PROMPT
    text = ("काशी नगरी वर्तमान वाराणसी शहर में स्थित पौराणिक नगरी है। योग एक प्राचीन "
            "भारतीय आध्यात्मिक प्रक्रिया है। प्रश्न: उत्तर: क्या है? के बारे में बताइए। ") * 20
    tok = BPE.train(text, 400, verbose=False)
    items = [{"question": "काशी क्या है?", "answer": "काशी नगरी वाराणसी में स्थित पौराणिक नगरी है।"},
             {"question": "योग के बारे में बताइए।", "answer": "योग एक प्राचीन भारतीय प्रक्रिया है।"}]
    pairs = encode_pairs(tok, items)
    for it, (p, a) in zip(items, pairs):
        joined = tok.encode(PROMPT.format(q=it["question"]) + " " + it["answer"]) + [tok.eos_id]
        assert p + a == joined, "prompt + answer must tokenise like the joined string"
        assert a[-1] == tok.eos_id
    x, y = pack(pairs, block_size=48, eos=tok.eos_id)
    assert x.shape == y.shape and x.shape[1] == 48
    n_ans = sum(len(a) for _, a in pairs)
    assert int((y != -1).sum()) == n_ans, "every answer token labelled once, nothing else"
    # Each row's labelled positions reproduce whole answers, in order, never split.
    labelled = [int(v) for row in y for v in row if v != -1]
    assert labelled == [t for _, a in pairs for t in a]
    for row_x, row_y in zip(x, y):
        marks = (row_y != -1).tolist()
        # Within a row, a labelled stretch must end on EOS (an answer is whole).
        for i in range(len(marks)):
            if marks[i] and (i + 1 == len(marks) or not marks[i + 1]):
                assert int(row_x[i + 1]) == tok.eos_id if i + 1 < len(marks) else True
    # A pair longer than the block is dropped rather than truncated mid-answer.
    x2, _ = pack(pairs, block_size=8, eos=tok.eos_id)
    assert x2.shape[0] == 0


@test
def chunked_loss_matches_the_full_loss_and_its_gradients():
    """enable_chunked_loss must be a memory change and nothing else: the same
    loss and the same gradient on every parameter, including with ignored
    (-1) targets and a last chunk shorter than the others."""
    torch.manual_seed(0)
    cfg = tiny()
    a, b = AnuLM(cfg).train(), AnuLM(cfg).train()
    b.load_state_dict(a.state_dict())
    b.enable_chunked_loss(7)                        # 2*16 = 32 rows -> 5 chunks, last of 4
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = x.clone()
    y[0, :5] = -1
    la = a(x, y)[1]
    logits, lb = b(x, y)
    assert logits is None, "the chunked path must not build the full logits"
    assert abs(la.item() - lb.item()) < 1e-5, (la.item(), lb.item())
    la.backward()
    lb.backward()
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        if pa.grad is None:
            assert pb.grad is None, n
            continue
        assert torch.allclose(pa.grad, pb.grad, atol=1e-5), f"gradient differs at {n}"
    b.eval()
    with torch.no_grad():
        assert b(x, y)[0] is not None, "eval mode keeps returning logits"


@test
def cached_embeds_decoding_matches_prefix_replay():
    """decode_from_embeds with the KV cache must emit exactly what re-running
    the whole prefix every step emits -- for a prompt made of embeddings, as
    the speech and vision paths build it, with and without n-gram blocking and
    the repetition penalty."""
    for attn in ("gqa", "mla"):
        torch.manual_seed(1)
        m = AnuLM(tiny(attn=attn, block_size=64)).eval()
        prefix = torch.randn(9, m.cfg.hidden_size) * 0.5
        for rp in (1.5, 1.0):
            for n in (0, 3):
                a = m.decode_from_embeds(prefix, 20, stop=lambda t: False, no_repeat_ngram=n,
                                         repetition_penalty=rp)
                b = m.decode_from_embeds(prefix, 20, stop=lambda t: False, no_repeat_ngram=n,
                                         use_cache=False, repetition_penalty=rp)
                assert a == b, f"{attn} n={n} rp={rp}: cached {a} vs replay {b}"
                assert len(a) == 20
        assert a == m.decode_from_embeds(prefix, 20, stop=lambda t: False, no_repeat_ngram=3), \
            "repetition_penalty=1.0 must be plain greedy"
        stop_at = a[4]
        c = m.decode_from_embeds(prefix, 20, stop=lambda t: t == stop_at)
        assert stop_at not in c and c == a[:a.index(stop_at)], "stop must end decoding"


@test
def native_gqa_matches_repeat_interleave():
    """NATIVE_GQA hands the Hkv heads to SDPA's enable_gqa instead of copying
    them; outputs must agree on the full-sequence, cached and windowed paths.

    With several KV heads, so the pairing is tested: query head h belongs to
    KV head h // n_rep (repeat_interleave). Tiling the heads (repeat) instead
    pairs h with h % n_kv -- invisible with one KV head, wrong with two or more.
    """
    from anulm import model as M
    saved = M.NATIVE_GQA
    try:
        for n_head, n_kv in ((4, 2), (6, 2), (6, 3), (4, 1)):
            torch.manual_seed(2)
            m = AnuLM(tiny(n_head=n_head, n_kv_head=n_kv, head_dim=8,
                           sliding_window=4, max_window_layers=1)).eval()
            x = torch.randint(0, 64, (2, 12))
            outs = []
            for native in (False, True):
                M.NATIVE_GQA = native
                with torch.no_grad():
                    full, _ = m(x, x)
                    cache = m.new_cache()
                    m(x[:, :8], cache=cache)
                    step, _ = m(x[:, 8:], cache=cache, start=8)
                outs.append((full, step))
            tag = f"{n_head}q/{n_kv}kv"
            assert torch.allclose(outs[0][0], outs[1][0], atol=1e-5), f"{tag}: full-sequence logits differ"
            assert torch.allclose(outs[0][1], outs[1][1], atol=1e-5), f"{tag}: cached-step logits differ"
    finally:
        M.NATIVE_GQA = saved


@test
def wsd_schedule_is_flat_then_decays_to_min_lr():
    from types import SimpleNamespace
    from anulm.training.train import lr_at
    a = SimpleNamespace(lr=1e-3, min_lr=1e-4, warmup=10, steps=100, schedule="wsd",
                        decay_frac=0.2, rewarmup=0)
    lrs = [lr_at(s, a) for s in range(100)]
    assert lrs[0] < lrs[9] == 1e-3, "warmup ramps to the peak"
    assert all(v == 1e-3 for v in lrs[10:80]), "flat until the last 20%"
    assert all(x > y for x, y in zip(lrs[80:], lrs[81:])), "strictly decreasing in the tail"
    assert abs(lr_at(100, a) - 1e-4) < 1e-12, "ends at min_lr"


@test
def merge_tree_composes_trains_and_bakes_losslessly():
    """merge_tree="both": row t of each table is W[t] + (E[a] + E[b]) / 2 over
    the BPE merge (a, b); gradients reach both tables; and bake_merge_tree()
    leaves an ordinary model with identical outputs, tied or not."""
    import json, os, tempfile
    from anulm.model import merge_tree_matrix
    merges = [[97, 98], [256, 99], [257, 257], [100, 101]]      # ab, abc, abcabc, de
    P = merge_tree_matrix(merges, 261)
    W = torch.randn(261, 4)
    E = torch.sparse.mm(P, W)
    assert torch.allclose(E[97], W[97])
    assert torch.allclose(E[256], W[256] + (W[97] + W[98]) / 2)
    assert torch.allclose(E[257], W[257] + (E[256] + W[99]) / 2)
    assert torch.allclose(E[258], W[258] + E[257]), "a self-pair counts both halves"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tok.json")
        json.dump({"type": "anulm-bpe", "version": 1, "merges": merges}, open(path, "w"))
        for tie in (False, True):
            torch.manual_seed(0)
            m = AnuLM(tiny(vocab_size=261, tokenizer_path=path, merge_tree="both",
                           tie_word_embeddings=tie)).train()
            x = torch.randint(0, 97, (2, 12))
            x[0, 3] = 258                       # abcabc: 97 reaches it only through the tree
            _, loss = m(x, x)
            loss.backward()
            assert m.embed_tokens.weight.grad[97].abs().sum() > 0, "an ancestor must get the gradient"
            m.eval()
            with torch.no_grad():
                before, _ = m(x, x)
                m.bake_merge_tree()
                after, _ = m(x, x)
            assert m.cfg.merge_tree == "" and m.merge_P is None
            assert torch.allclose(before, after, atol=1e-5), f"tie={tie}: baking changed outputs"


@test
def merge_tree_eval_cache_sees_optimizer_updates():
    """The eval-mode cache of composed tables must never outlive a training
    step. Optimizers (fused AdamW) update weights without bumping their
    version counter, which is how a version-keyed cache once scored every
    eval of a 10,000-step run against step 999's embeddings."""
    import json, os, tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tok.json")
        json.dump({"type": "anulm-bpe", "version": 1, "merges": [[97, 98], [256, 99]]}, open(path, "w"))
        torch.manual_seed(0)
        m = AnuLM(tiny(vocab_size=259, tokenizer_path=path, merge_tree="both",
                       tie_word_embeddings=True))
        x = torch.randint(0, 259, (2, 8))
        m.eval()
        with torch.no_grad():
            a, _ = m(x, x)
        m.train()
        with torch.no_grad():
            m.embed_tokens.weight.data.add_(0.5)      # an update invisible to _version
        m.eval()
        with torch.no_grad():
            b, _ = m(x, x)
        assert not torch.allclose(a, b), "eval after a training step reused stale tables"


@test
def decide_scores_each_option_at_its_last_token_and_calibrates():
    """decide.py's contract: option k's score is read at the last token of
    option k's name (so it has seen the message and that option), a forward
    pass gives one logit per option, ECE is 0 for a perfectly calibrated set,
    and temperature scaling never changes the argmax."""
    from anulm.tasks import decide as D
    from anulm.bpe import BPE
    tok = BPE.train("Message: which of these does it ask for options card arrival "
                    "lost card refund exchange rate\n- " * 30, 300, verbose=False)
    opts = ["card_arrival", "lost_card", "refund"]
    ids, pos = D.encode(tok, "where is my card", opts)
    for k, o in enumerate(opts):
        assert tok.decode(ids[:pos[k] + 1]).endswith(D.readable(o)), o
    torch.manual_seed(0)
    d = D.Decider(AnuLM(tiny(vocab_size=300, block_size=128))).eval()
    x, p, y = D.batchify([(ids, pos, 0, opts), (ids, pos, 2, opts)], tok, tok.eos_id)
    with torch.no_grad():
        logits = d(x, p)
    assert logits.shape == (2, 3)
    probs = torch.tensor([[0.8, 0.1, 0.1]] * 5)
    y5 = torch.tensor([0, 0, 0, 0, 1])                     # 80% confident, 80% right
    assert D.metrics(probs, y5)["ece"] < 1e-6
    t = D.fit_temperature(logits.repeat(4, 1), torch.tensor([0, 2] * 4))
    assert (F.softmax(logits / t, 1).argmax(1) == logits.argmax(1)).all()


@test
def batched_muon_matches_per_matrix_muon():
    """Stacking same-shaped updates into one batched Newton-Schulz must give
    the same step as orthogonalising each matrix alone -- it is a speed change
    (2.3x on the MoE proxy, where experts share shapes), not a new optimizer."""
    from anulm.muon import Muon
    torch.manual_seed(0)
    shapes = [(64, 24)] * 5 + [(24, 64), (32, 32)]
    a = [torch.nn.Parameter(torch.randn(*s)) for s in shapes]
    b = [torch.nn.Parameter(p.detach().clone()) for p in a]
    for _ in range(3):
        for p, q in zip(a, b):
            g = torch.randn_like(p)
            p.grad, q.grad = g.clone(), g.clone()
        Muon(a, batched=True).step()
        Muon(b, batched=False).step()
    for p, q in zip(a, b):
        assert torch.allclose(p, q, atol=1e-6), (p - q).abs().max()


@test
def memmap_batches_match_in_memory_batches():
    """A build_v2 corpus (meta.json + raw uint16 files) must give train.py the
    same (x, y) windows as the same ids held in memory as a tensor."""
    import json, os, tempfile
    import numpy as np
    from anulm.training.train import load_memmap, make_batch
    ids = np.random.RandomState(0).randint(0, 32768, 5000).astype(np.uint16)
    with tempfile.TemporaryDirectory() as d:
        ids.tofile(os.path.join(d, "train.u16"))
        ids[:700].tofile(os.path.join(d, "val.u16"))
        json.dump({"train": "train.u16", "val": "val.u16", "vocab_size": 32768,
                   "tokenizer": "x", "bytes_per_token": 4.0}, open(os.path.join(d, "meta.json"), "w"))
        tr, va, meta = load_memmap(os.path.join(d, "meta.json"))
        assert len(tr) == 5000 and len(va) == 700
        ix = torch.tensor([0, 17, 4000])
        xm, ym = make_batch(tr, ix, 64, "cpu")
        xt, yt = make_batch(torch.from_numpy(ids.astype(np.int64)), ix, 64, "cpu")
        assert torch.equal(xm, xt) and torch.equal(ym, yt) and xm.dtype == torch.long
        del tr, va                                # release the files before the dir goes


@test
def interleaved_split_samples_the_whole_corpus():
    """An ordered corpus (source A then source B) split at the tail holds out
    only B; the interleaved split must hold out both, and lose nothing."""
    from anulm.training.train import interleaved_split
    data = torch.cat([torch.zeros(8_000_000, dtype=torch.int16),
                      torch.ones(8_000_000, dtype=torch.int16)])
    train, val = interleaved_split(data)
    assert len(train) + len(val) == len(data)
    assert 0.05 < len(val) / len(data) < 0.15
    frac_b = float((val == 1).float().mean())
    assert 0.3 < frac_b < 0.7, f"held-out split is {frac_b:.0%} source B"
    tail = data[int(0.9 * len(data)):]
    assert float((tail == 1).float().mean()) == 1.0, "the old tail split is all B"


# =============================================================================
# MULTIMODAL AND PIPELINE HARDENING
# =============================================================================

@test
def whisper_audio_chunks_cover_long_recordings():
    """Every sample past Whisper's window must reach a later chunk, while a
    clip that already fits keeps the original single-chunk path."""
    import numpy as np
    from anulm.multimodal.audio_codec import ASR_RATE, audio_chunks
    window = 30 * ASR_RATE
    audio = np.arange(2 * window + 17, dtype=np.float32)
    chunks = audio_chunks(audio)
    assert [len(c) for c in chunks] == [window, window, 17]
    assert np.array_equal(np.concatenate(chunks), audio)
    assert len(audio_chunks(audio[:window])) == 1
    assert len(audio_chunks(audio[:0])) == 1


@test
def continuous_asr_decodes_long_chunks_in_order():
    """Continuous ASR must decode each long-recording chunk independently and
    join the texts in audio order, not return only Whisper's first 30 seconds."""
    from types import SimpleNamespace
    import numpy as np
    from anulm.multimodal.audio_codec import ASR_RATE
    from anulm.multimodal.speech_encoder import transcribe

    class Encoder:
        def __init__(self):
            self.sizes = []

        def encode(self, audio):
            self.sizes.append(len(audio))
            return np.zeros((4, 2), dtype=np.float32)

    class Projector:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def parameters(self):
            return iter([self.weight])

        def __call__(self, states):
            return states

    class Model:
        def __init__(self):
            self.calls = 0
            self.cfg = SimpleNamespace(block_size=1024)      # transcribe sizes its budget from it

        def embed_tokens(self, ids):
            return torch.zeros((len(ids), 2), dtype=torch.float32)

        def decode_from_embeds(self, prefix, *args, **kwargs):
            self.calls += 1
            return torch.tensor([self.calls])

    class Tokenizer:
        eos_id = 99

        def __init__(self, texts):
            self.texts = texts

        def decode(self, ids):
            return self.texts[int(ids[0]) - 1]

    def run(audio, texts):
        model = Model()
        sm = SimpleNamespace(
            proj=Projector(), model=model,
            vocab=SimpleNamespace(speech_bos=1, text_bos=2, text=100))
        enc = Encoder()
        text = transcribe(sm, enc, Tokenizer(texts), audio, device="cpu")
        return text, enc.sizes

    text, sizes = run(np.zeros(65 * ASR_RATE, dtype=np.float32), ["one", "two", "three"])
    assert text == "one two three"
    assert sizes == [30 * ASR_RATE, 30 * ASR_RATE, 5 * ASR_RATE]
    text, sizes = run(np.zeros(10, dtype=np.float32), ["short\nignored"])
    assert text == "short" and len(sizes) == 1


@test
def image_feature_cache_reuses_invalidates_and_recovers():
    """The frozen-tower cache is reusable only while its sidecar and bytes
    match. Legacy, partial and interrupted temporary files must cause a clean
    rebuild rather than stale reads or an append to a partial file."""
    import tempfile
    from pathlib import Path
    from anulm.multimodal.vision_encoder import build_cache

    class Tower:
        name = "fake-tower"

        def __init__(self):
            self.calls = []

        def encode(self, path):
            raw = Path(path).read_bytes()
            self.calls.append(raw)
            return torch.full((2, 2), float(sum(raw)))

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        first, second = root / "first.jpg", root / "second.jpg"
        first.write_bytes(b"first")
        second.write_bytes(b"second image")
        rows = [{"image": str(first)}, {"image": str(second)}, {"image": str(first)}]
        path = root / "features.npy"
        sidecar = Path(str(path) + ".json")
        tower = Tower()

        mm, index = build_cache(rows, tower, path, dtype="float32")
        assert mm.shape == (2, 2, 2) and len(index) == 2
        assert len(tower.calls) == 3                 # one probe, then two distinct images
        del mm

        before = len(tower.calls)
        mm, _ = build_cache(rows, tower, path, dtype="float32")
        assert len(tower.calls) == before + 1         # the shape probe only; no re-encoding
        del mm

        first.write_bytes(b"first image changed")
        before = len(tower.calls)
        mm, _ = build_cache(rows, tower, path, dtype="float32")
        assert len(tower.calls) == before + 3, "an edited image must invalidate the cache"
        del mm

        sidecar.unlink()                              # a legacy cache has no fingerprint
        before = len(tower.calls)
        mm, _ = build_cache(rows, tower, path, dtype="float32")
        assert len(tower.calls) == before + 3 and sidecar.exists()
        del mm

        path.write_bytes(b"partial")                 # matching metadata cannot validate short bytes
        Path(str(path) + ".tmp").write_bytes(b"interrupted")
        Path(str(sidecar) + ".tmp").write_text("stale", encoding="utf-8")
        before = len(tower.calls)
        mm, _ = build_cache(rows, tower, path, dtype="float32")
        assert len(tower.calls) == before + 3 and mm.shape == (2, 2, 2)
        assert not Path(str(path) + ".tmp").exists()
        assert not Path(str(sidecar) + ".tmp").exists()
        del mm


@test
def sft_packed_cache_invalidates_same_count_content_edits():
    """A packed cache keyed by row count silently reuses the wrong labels after
    an edit. Source content must decide reuse even when the JSONL count is equal."""
    import json, tempfile
    from pathlib import Path
    from anulm.training import finetune as F

    original = F.encode_pairs_parallel
    calls = []

    def encode(tokenizer_path, items, workers):
        calls.append([dict(r) for r in items])
        return [([1], [ord(r["answer"]), 99]) for r in items]

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        source, tokenizer = root / "train.jsonl", root / "tok.json"
        tokenizer.write_bytes(b"tokenizer")
        source.write_text(json.dumps({"question": "q", "answer": "a"}) + "\n", encoding="utf-8")
        cache = Path(f"{source}.tok.b8.pt")
        F.encode_pairs_parallel = encode
        try:
            x1, y1 = F.load_packed(str(source), str(tokenizer), 8, 99, 1)
            first = torch.load(cache, weights_only=False)
            x2, y2 = F.load_packed(str(source), str(tokenizer), 8, 99, 1)
            assert len(calls) == 1 and torch.equal(x1, x2) and torch.equal(y1, y2)

            source.write_text(json.dumps({"question": "q", "answer": "b"}) + "\n", encoding="utf-8")
            x3, y3 = F.load_packed(str(source), str(tokenizer), 8, 99, 1)
            second = torch.load(cache, weights_only=False)
            assert len(calls) == 2 and not torch.equal(y1, y3)
            assert first["n_items"] == second["n_items"] == 1
            assert first["cache_key"] != second["cache_key"]
        finally:
            F.encode_pairs_parallel = original


@test
def omni_api_availability_and_missing_reader_fallback():
    """One Omni checkpoint provides the chat/answer/vision/ASR APIs without the
    legacy component files, and a missing reader falls back to labelled memory."""
    try:
        from anulm.inference import api
    except ImportError as e:
        raise unittest.SkipTest(f"API dependency unavailable: {e}")
    import tempfile
    from pathlib import Path

    old = api.A.USE_OMNI, api.A.HERE, api.HERE, api.models, api.Generation, api.chat_prompt
    try:
        with tempfile.TemporaryDirectory() as d:
            api.A.HERE = api.HERE = Path(d)
            api.A.USE_OMNI = False
            assert api.A.Models("cpu", "").reader() is None

            api.A.USE_OMNI = True
            ids = {m["id"] for m in api.list_models()["data"]}
            assert {"anulm-chat", "anulm-answer", "anulm-vision", "anulm-asr"} <= ids
            assert not api.available("anulm-decide-india-banking")

            class Models:
                def reader(self):
                    return None

            class Generation:
                calls = 0

                def __init__(self, *args):
                    Generation.calls += 1

                def run(self):
                    return "memory answer"

            api.models = Models()
            api.Generation = Generation
            api.chat_prompt = lambda q: q
            out = api.answer_question(api.AnswerRequest(question="Where?", fallback=True))
            assert out["answer"] == "memory answer" and not out["found"]
            assert out["from"] == "memory (anulm-chat; unreliable)"
            out = api.answer_question(api.AnswerRequest(question="Where?", fallback=False))
            assert out["answer"] is None and Generation.calls == 1
    finally:
        (api.A.USE_OMNI, api.A.HERE, api.HERE,
         api.models, api.Generation, api.chat_prompt) = old


@test
def legacy_half_keeps_routers_fp32_and_projectors_runnable():
    """The legacy GPU memory path: routers stay fp32 (bf16 cannot hold their
    biases), everything else is bf16, and a projector still runs outside
    autocast. An earlier version kept every small tensor in fp32, which left
    nn.Linear biases fp32 beside bf16 weights and broke transcribe/caption."""
    try:
        from anulm.apps.app_omni import Models
    except ImportError as e:
        raise unittest.SkipTest(f"app dependency unavailable: {e}")
    from anulm.model import AnuLMConfig, Router
    cfg = AnuLMConfig.nano_30b()
    module = torch.nn.Module()
    module.router = Router(cfg)
    module.proj = torch.nn.Linear(8, 8)
    with torch.no_grad():
        module.router.expert_bias.add_(500.0 + torch.arange(cfg.num_experts) * 0.01)
    out = Models("cuda", "")._half(module)
    assert out is module
    assert module.router.weight.dtype == torch.float32
    assert module.router.expert_bias.dtype == torch.float32
    assert module.router.expert_bias.unique().numel() == cfg.num_experts, "router biases were rounded"
    assert module.proj.weight.dtype == module.proj.bias.dtype == torch.bfloat16
    module.proj(torch.randn(2, 8, dtype=torch.bfloat16))          # no dtype mismatch outside autocast
    assert all(t.device.type == "cpu" for t in module.parameters())


@test
def downloader_restarts_when_server_ignores_range():
    """A 200 response to a resume request is a full file, not a continuation:
    the stale .part bytes must be truncated rather than prepended to it."""
    import tempfile
    from pathlib import Path
    from anulm.multimodal import omni_data as D

    payload = b"fresh complete payload"
    globals_ = D.download.__globals__
    old_root, old_urlopen = globals_["ROOT"], globals_["urllib"].request.urlopen

    class Response:
        status = 200
        headers = {"Content-Length": str(len(payload))}

        def __init__(self):
            self.done = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size):
            if self.done:
                return b""
            self.done = True
            return payload

    requests = []

    def urlopen(req, timeout):
        requests.append(req)
        return Response()

    try:
        with tempfile.TemporaryDirectory() as d:
            globals_["ROOT"] = Path(d)
            globals_["urllib"].request.urlopen = urlopen
            dest = globals_["ROOT"] / "file.bin"
            part = Path(str(dest) + ".part")
            part.write_bytes(b"stale")
            out = D.download("https://example.invalid/file", dest, size=len(payload))
            assert requests[0].get_header("Range") == "bytes=5-"
            assert out == dest and dest.read_bytes() == payload
            assert not part.exists()
    finally:
        globals_["ROOT"], globals_["urllib"].request.urlopen = old_root, old_urlopen


@test
def training_fingerprint_preserves_resume_and_invalidates_downstream():
    """An unchanged recipe keeps interrupted training state; a changed one
    archives generated files and clears every downstream completion marker."""
    import tempfile
    from pathlib import Path
    import experiments.omni_one as O

    assert {"anulm/multimodal/omni_train.py", "anulm/multimodal/speech_encoder.py",
            "anulm/multimodal/vision_encoder.py", "anulm/multimodal/audio_codec.py",
            "anulm/multimodal/speech_vocab.py"} <= set(O.RECIPE_CODE)
    old = O.ROOT, O.LOGS, O.training_fingerprint
    try:
        with tempfile.TemporaryDirectory() as d:
            O.ROOT = Path(d)
            O.LOGS = O.ROOT / "logs" / "omni_one"
            O.LOGS.mkdir(parents=True)
            O.training_fingerprint = lambda args: "new-fingerprint"
            out, resume = O.ROOT / O.OUT, O.ROOT / (O.OUT + ".last")
            curve = O.ROOT / f"{Path(O.OUT).stem}_curve.csv"
            results, report = O.LOGS / "results.json", O.LOGS / "REPORT.md"
            out.write_bytes(b"old")
            resume.write_bytes(b"resume")
            curve.write_text("old curve", encoding="utf-8")
            results.write_text("old results", encoding="utf-8")
            report.write_text("old report", encoding="utf-8")
            collision = O.LOGS / "archive" / "old-fingerprint"
            collision.mkdir(parents=True)
            (collision / "keep.txt").write_text("keep", encoding="utf-8")
            O.save_state({"training_fingerprint": "old-fingerprint", "5_base": {"ok": 1},
                          "6_train": {"ok": 1}, "7_eval": {"ok": 1}})

            O.prepare_training([])
            state = O.state()
            archive = O.LOGS / "archive" / "old-fingerprint-2"
            assert state["training_fingerprint"] == "new-fingerprint"
            assert "6_train" not in state and "7_eval" not in state
            assert state["5_base"] == {"ok": 1}
            assert not out.exists() and not resume.exists()
            assert not curve.exists() and not results.exists() and not report.exists()
            assert (archive / out.name).read_bytes() == b"old"
            assert (archive / resume.name).read_bytes() == b"resume"
            assert (archive / curve.name).read_text(encoding="utf-8") == "old curve"
            assert (archive / results.name).read_text(encoding="utf-8") == "old results"
            assert (archive / report.name).read_text(encoding="utf-8") == "old report"
            assert (collision / "keep.txt").read_text(encoding="utf-8") == "keep"

            out.write_bytes(b"new")
            resume.write_bytes(b"new resume")
            results.write_text("new results", encoding="utf-8")
            before = O.state()
            archived = sorted(archive.iterdir())
            O.prepare_training([])
            assert out.read_bytes() == b"new" and resume.read_bytes() == b"new resume"
            assert results.read_text(encoding="utf-8") == "new results"
            assert O.state() == before and sorted(archive.iterdir()) == archived
    finally:
        O.ROOT, O.LOGS, O.training_fingerprint = old


@test
def experiment_scripts_resolve_root_to_the_repository():
    """Every experiments/*.py that sets ROOT from its own path lands on the
    repository root. A one-level slip sent omni_one's chdir, data paths and
    subprocesses to the folder above, where they found nothing."""
    import ast
    from pathlib import Path as P
    repo = P(__file__).resolve().parents[1]

    # Walk the expression instead of eval()-ing source text: only the handful
    # of pathlib steps these scripts use is understood, and anything else is
    # a failure that names the expression rather than code that gets run.
    def walk(node, f):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Path"
                and len(node.args) == 1 and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "__file__" and not node.keywords):
            return P(f)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "resolve" and not node.args and not node.keywords):
            return walk(node.func.value, f).resolve()
        if isinstance(node, ast.Attribute) and node.attr == "parent":
            return walk(node.value, f).parent
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute)
                and node.value.attr == "parents" and isinstance(node.slice, ast.Constant)
                and type(node.slice.value) is int):
            return walk(node.value.value, f).parents[node.slice.value]
        raise AssertionError(f"{f.name}: ROOT uses {ast.unparse(node)!r}, which this test cannot follow")

    checked = 0
    for f in sorted((repo / "experiments").glob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "ROOT"
                    and ast.unparse(node.value).startswith("Path(__file__)")):
                root = walk(node.value, f)
                assert root.resolve() == repo, f"{f.name}: ROOT is {root}"
                checked += 1
    assert checked >= 5, f"only {checked} experiment scripts set ROOT; the pattern changed?"


# =============================================================================
# FROM-SCRATCH PIPELINE
# =============================================================================


@test
def from_scratch_pipeline_runs_and_skips_completed_stages():
    """The real text-only pipeline trains end to end, then verifies and skips
    tokenizer, encoding, pretraining, vocabulary growth, and omni training."""
    import json
    import os
    import subprocess
    import tempfile
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as d:
        work = Path(d)
        corpus = work / "corpus.txt"
        tasks = work / "tasks.jsonl"
        paragraph = ("AnuLM trains from scratch on a small text corpus. "
                     "This document supplies enough windows for pretraining and validation.")
        corpus.write_text((paragraph + "\n\n\n") * 12, encoding="utf-8", newline="\n")
        rows = [{"question": "Hi?", "answer": "Hello.", "lang": "en"},
                {"question": "Test?", "answer": "Pass.", "lang": "en"},
                {"question": "Done?", "answer": "Yes.", "lang": "en"}]
        tasks.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8", newline="\n")
        cmd = [sys.executable, "-u", "-m", "anulm.training.train_from_scratch",
               "--corpus", str(corpus), "--work-dir", str(work / "run"),
               "--workers", "1", "--vocab", "280", "--preset", "30b", "--device", "cpu",
               "--block-size", "64", "--batch-size", "1", "--pretrain-steps", "1",
               "--warmup", "1", "--eval-every", "1", "--eval-iters", "1", "--log-every", "1",
               "--task", f"chat=text:{tasks}:1:1", "--omni-steps", "1",
               "--omni-warmup", "1", "--omni-eval-every", "1", "--omni-eval-n", "1"]
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

        def invoke():
            result = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=120)
            assert result.returncode == 0, result.stdout + "\n" + result.stderr
            return result.stdout

        invoke()
        outputs = ("bpe.json", "corpus.bin", "pretrained.pt", "multimodal_init.pt", "omni.pt")
        assert all((work / "run" / name).is_file() for name in outputs)
        second = invoke()
        for stage in ("tokenizer", "encode", "pretrain", "grow", "omni"):
            assert f"skip {stage}: verified" in second, second
        assert "\n$ " not in second, second


def main():
    passed, skipped, failed = [], [], []
    for fn in TESTS:
        try:
            fn()
            passed.append(fn.__name__)
            print(f"  PASS  {fn.__name__}")
        except unittest.SkipTest as e:        # before Exception: SkipTest is one
            skipped.append((fn.__name__, e))
            print(f"  SKIP  {fn.__name__}  ({e})")
        except Exception as e:
            failed.append((fn.__name__, e))
            print(f"  FAIL  {fn.__name__}")
            print("        " + "\n        ".join(
                traceback.format_exception_only(type(e), e)).rstrip())
    # With nothing skipped this is the line the docs and the course quote
    # ("80 passed, 0 failed, 80 total"), so it keeps that exact shape.
    if skipped:
        print(f"\n{len(passed)} passed, {len(skipped)} skipped, {len(failed)} failed, {len(TESTS)} total")
    else:
        print(f"\n{len(passed)} passed, {len(failed)} failed, {len(TESTS)} total")
    return 1 if failed else 0


class ModelTests(unittest.TestCase):
    """Every @test above as a unittest method, so `python -m unittest
    tests.test_model`, `unittest discover` and pytest run the same 80 tests
    main() does. Methods are generated rather than written out so the two
    lists cannot drift apart."""


def _as_method(fn):
    def method(self):
        fn()
    method.__name__ = method.__qualname__ = f"test_{fn.__name__}"
    method.__doc__ = fn.__doc__
    return method


for _fn in TESTS:
    assert not hasattr(ModelTests, f"test_{_fn.__name__}"), f"two tests are called {_fn.__name__}"
    setattr(ModelTests, f"test_{_fn.__name__}", _as_method(_fn))
del _fn


if __name__ == "__main__":
    sys.exit(main())
