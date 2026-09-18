"""
AnuLM -- a small, trainable model with Sarvam 30B / 105B's architecture.

Distilled from the annotated reference in sarvam/. Every non-obvious choice
here traces to a specific line of Sarvam's released modelling code; the mapping
is in README.md and in the comments below.

What is faithful:
  * sparse MoE with a shared expert and a dense first layer
  * sigmoid router, aux-loss-free bias balancing, group-limited top-k
  * BOTH attention variants -- GQA (30B) and MLA (105B) -- switchable
  * QK-norm, pre-norm blocks, RMSNorm, SwiGLU, untied embeddings
  * YaRN context extension with the band blend and the logit-temperature term

What is deliberately different:
  * it can actually train (the released 105B file cannot -- see README)
  * RoPE uses the split-halves convention throughout, not the 105B
    checkpoint's interleaved layout, since we train from scratch
  * no expert parallelism, no vLLM branch, no FlashAttention subclass:
    SDPA everywhere, which is what a single-GPU run wants anyway
"""

from __future__ import annotations

import math
import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class AnuLMConfig:
    # --- tokens / context -----------------------------------------------
    vocab_size: int = 259           # 256 bytes + BOS + EOS + PAD (byte mode)
    block_size: int = 256
    tokenizer_path: Optional[str] = None   # set when trained on BPE-encoded
                                           # data; sample.py reads it from the
                                           # checkpoint to encode/decode

    # --- trunk ------------------------------------------------------------
    n_layer: int = 8
    hidden_size: int = 384
    n_head: int = 6

    # --- attention ---------------------------------------------------------
    attn: str = "gqa"               # "gqa" (30B-style) | "mla" (105B-style)
    head_dim: Optional[int] = None  # gqa: defaults to hidden_size // n_head
    n_kv_head: int = 2              # gqa only
    kv_lora_rank: int = 48          # mla only
    qk_nope_head_dim: int = 48      # mla only
    qk_rope_head_dim: int = 24      # mla only
    v_head_dim: int = 48            # mla only
    use_qk_norm: bool = True
    # Sliding-window attention on layers [0, max_window_layers) -- Qwen2's
    # semantics, which the real 30B config inherits: it ships
    # `max_window_layers: 19` with no window set, i.e. off. None disables.
    sliding_window: Optional[int] = None
    max_window_layers: int = 0

    # --- MoE ---------------------------------------------------------------
    num_experts: int = 16
    num_experts_per_tok: int = 2
    num_shared_experts: int = 1
    moe_intermediate_size: int = 96
    intermediate_size: int = 768    # the dense layer-0 MLP
    first_k_dense_replace: int = 1
    n_group: int = 1                # group-limited routing; 1 disables it
    topk_group: int = 1
    routed_scaling_factor: float = 2.5
    bias_update_rate: float = 1e-3  # aux-loss-free balancing step size
    # "sparse": gather each expert's tokens and run one GEMM per expert. What a
    #   real MoE does, and 2.1-2.35x faster here in eager -- but it uses
    #   `nonzero`, whose output shape is data-dependent, so torch.compile breaks
    #   the graph 8 times per forward and buys you nothing.
    # "dense": run every expert on every token and combine with a one-hot
    #   weight matrix. Bit-identical output, 1 graph and 0 breaks under Dynamo,
    #   at the cost of num_experts/top_k times the expert FLOPs.
    # "grouped": sort the (token, slot) pairs by expert once, then run ALL
    #   experts as one grouped GEMM per projection (F.grouped_mm). No `nonzero`,
    #   so shapes are static and Dynamo captures it in one graph; on a GPU it
    #   turns num_experts small launches into one. bf16 only -- meant for
    #   autocast, where the sparse path's Linears run in bf16 anyway.
    moe_impl: str = "sparse"
    # DeepSeek-V3's complementary sequence-wise balance loss, and a router
    # z-loss. Both default OFF: the bias rule does the balancing and these are
    # the belt to its braces as data gets more diverse. They enter the training
    # loss only; train.py reports them separately from the LM loss.
    seq_balance_alpha: float = 0.0
    router_z_alpha: float = 0.0

    # --- RoPE --------------------------------------------------------------
    rope_theta: float = 10000.0
    yarn: bool = False
    yarn_factor: float = 8.0
    yarn_original_context: int = 256
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_mscale: float = 1.0
    yarn_mscale_all_dim: float = 1.0

    # --- misc --------------------------------------------------------------
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_word_embeddings: bool = False   # Sarvam unties both models
    initializer_range: float = 0.02

    def __post_init__(self):
        assert self.attn in ("gqa", "mla")
        assert self.moe_impl in ("sparse", "dense", "grouped")
        if self.moe_impl == "grouped":
            assert hasattr(F, "grouped_mm"), (
                "moe_impl='grouped' needs torch.nn.functional.grouped_mm (present in 2.11)")
        if self.sliding_window is not None:
            assert self.sliding_window >= 1 and self.max_window_layers >= 1, (
                "sliding_window needs max_window_layers >= 1 to apply to any layer")
        assert self.num_experts % self.n_group == 0, "n_group must divide num_experts"
        # The footgun that group-limited routing hides: you cannot select more
        # experts than the surviving groups actually contain.
        per_group = self.num_experts // self.n_group
        assert self.num_experts_per_tok <= per_group * self.topk_group, (
            f"top_k={self.num_experts_per_tok} exceeds the {per_group * self.topk_group} "
            f"experts reachable through {self.topk_group} of {self.n_group} groups"
        )
        assert self.topk_group <= self.n_group
        if self.attn == "gqa":
            if self.head_dim is None:
                self.head_dim = self.hidden_size // self.n_head
            assert self.n_head % self.n_kv_head == 0
        else:
            self.head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

    # --- presets: the two real models, scaled down ------------------------
    @staticmethod
    def nano_30b(**kw) -> "AnuLMConfig":
        """GQA + high rope_theta, no YaRN. Sarvam 30B's shape."""
        cfg = dict(
            attn="gqa", n_layer=8, hidden_size=384, n_head=6, head_dim=64,
            n_kv_head=2, num_experts=16, num_experts_per_tok=2,
            moe_intermediate_size=96,      # hidden/4, as in the real model
            intermediate_size=768,         # hidden*2
            n_group=1, topk_group=1,       # grouping is a no-op, like the 30B
            rope_theta=1e6, yarn=False,
        )
        cfg.update(kw)
        return AnuLMConfig(**cfg)

    @staticmethod
    def nano_350m(**kw) -> "AnuLMConfig":
        """353M total / 152M active. The largest that trains comfortably on 8 GB.

        Sized by measurement, not arithmetic. Memory is the binding constraint:
        optimizer state covers ALL parameters even though only 43% are active
        per token, so an MoE is memory-hungry to train however sparse its
        forward pass is. Measured peaks at batch 8 x 512, bf16 autocast,
        gradient checkpointing on, AdamW:

            443M -> 9.74 GB   spills to system RAM on an 8 GB card (12x slower)
            353M -> 6.34 GB   fits, 4955 tok/s      <- this preset
            281M -> 5.15 GB   fits, 5887 tok/s

        Requires `--grad-ckpt`. Without it the same model peaks at 15.75 GB and
        thrashes. Muon fits a larger model in the same memory but costs 2.6x per
        step here -- see muon.py and docs/RESULTS.md.
        """
        cfg = dict(
            attn="gqa", n_layer=20, hidden_size=1024, n_head=16, head_dim=64,
            n_kv_head=4,                   # 4:1, closer to the real 30B's 16:1
            num_experts=12, num_experts_per_tok=3,
            moe_intermediate_size=384,
            intermediate_size=2816,
            n_group=1, topk_group=1,
            rope_theta=1e6, yarn=False,
            block_size=512,
        )
        cfg.update(kw)
        return AnuLMConfig(**cfg)

    @staticmethod
    def nano_105b(**kw) -> "AnuLMConfig":
        """MLA + YaRN-ready. Sarvam 105B's shape."""
        cfg = dict(
            attn="mla", n_layer=12, hidden_size=384, n_head=6,
            kv_lora_rank=48,               # hidden/8, as in the real model
            qk_nope_head_dim=48, qk_rope_head_dim=24, v_head_dim=48,
            num_experts=16, num_experts_per_tok=2,
            moe_intermediate_size=192,     # hidden/2
            intermediate_size=1536,        # hidden*4
            n_group=4, topk_group=2,       # group-limited, like the 105B
            rope_theta=10000.0, yarn=False,
        )
        cfg.update(kw)
        return AnuLMConfig(**cfg)


# =============================================================================
# NORM / MLP
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


class SwiGLU(nn.Module):
    """The expert body and the dense layer-0 MLP. 3 * D * I params."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# =============================================================================
# ROPE  (see sarvam/yarn_rotary_annotated.py)
# =============================================================================

def _yarn_correction_dim(num_rotations, dim, base, orig_ctx):
    return (dim * math.log(orig_ctx / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_correction_range(beta_fast, beta_slow, dim, base, orig_ctx):
    low = math.floor(_yarn_correction_dim(beta_fast, dim, base, orig_ctx))
    high = math.ceil(_yarn_correction_dim(beta_slow, dim, base, orig_ctx))
    return max(low, 0), min(high, dim - 1)


def yarn_mscale(scale: float, mscale: float) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class Rotary(nn.Module):
    """One instance at model level, shared by every layer (the 30B's pattern --
    the 105B rebuilds this per layer and pays 2.15 GB for it at load)."""

    def __init__(self, cfg: AnuLMConfig, dim: int):
        super().__init__()
        self.dim = dim
        self.cfg = cfg
        inv_freq = self._build_inv_freq(cfg, dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # YaRN's temperature, split the way DeepSeek (and so the 105B) splits
        # it: attention multiplies its softmax scale by attn_mscale**2, and the
        # cos/sin tables are scaled by mscale(f, mscale) / mscale(f, mscale_all_dim).
        # With the shipped 1.0 / 1.0 the table factor is exactly 1.0 and the
        # whole correction lands on the attention scale.
        if cfg.yarn:
            self.attn_mscale = yarn_mscale(cfg.yarn_factor, cfg.yarn_mscale_all_dim)
            self.table_mscale = (yarn_mscale(cfg.yarn_factor, cfg.yarn_mscale)
                                 / yarn_mscale(cfg.yarn_factor, cfg.yarn_mscale_all_dim))
        else:
            self.attn_mscale = self.table_mscale = 1.0
        self._cached_len = 0
        self._cached_key = None            # (device, dtype) the tables were built for
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)

    @staticmethod
    def _build_inv_freq(cfg: AnuLMConfig, dim: int) -> torch.Tensor:
        idx = torch.arange(0, dim, 2, dtype=torch.float32) / dim
        freq_extra = 1.0 / (cfg.rope_theta ** idx)
        if not cfg.yarn:
            return freq_extra
        # NTK-by-parts: keep the fast dims, divide the slow dims by `factor`,
        # ramp linearly between the two correction dims.
        freq_inter = 1.0 / (cfg.yarn_factor * (cfg.rope_theta ** idx))
        low, high = _yarn_correction_range(
            cfg.yarn_beta_fast, cfg.yarn_beta_slow, dim,
            cfg.rope_theta, cfg.yarn_original_context,
        )
        if low == high:
            high += 0.001
        ramp = (torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)
        ramp = ramp.clamp(0, 1)
        keep = 1.0 - ramp                     # 1 -> extrapolate, 0 -> interpolate
        return freq_inter * (1 - keep) + freq_extra * keep

    def _grow(self, seq_len: int, device, dtype):
        # Keyed on device and dtype as well as length: a model moved with .to()
        # or run under a different autocast dtype must not be handed stale tables.
        key = (device, dtype)
        if seq_len <= self._cached_len and self._cached_key == key:
            return
        n = max(seq_len, self._cached_len)
        t = torch.arange(n, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))          # (T, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)                   # (T, dim)
        self.cos_cached = (emb.cos() * self.table_mscale).to(dtype)
        self.sin_cached = (emb.sin() * self.table_mscale).to(dtype)
        self._cached_len = n
        self._cached_key = key

    def forward(self, seq_len: int, device, dtype, start: int = 0):
        """cos/sin for absolute positions [start, start + seq_len). `start` is
        the number of tokens already in the KV cache when decoding; 0 otherwise."""
        self._grow(start + seq_len, device, dtype)
        return self.cos_cached[start:start + seq_len], self.sin_cached[start:start + seq_len]


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """q, k: (B, H, S, Dr). cos/sin: (S, Dr) -> broadcast over batch and heads.

    Split-halves (GPT-NeoX) convention. Sarvam's 105B checkpoint stores its rope
    dims interleaved and permutes before rotating; we train from scratch, so we
    pick one convention and stay in it.
    """
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def _attn_mask(S: int, T: int, start: int, window: Optional[int], device):
    """Boolean (S, T) mask, True = may attend. The S queries sit at absolute
    positions start..start+S-1; the T keys at 0..T-1 (T == start + S).

    Returns None whenever SDPA's built-in `is_causal` (or no mask at all, for a
    single decode step) is enough, so the fused kernels keep their fast path.
    An explicit mask is needed for a multi-token step against a non-empty cache
    -- `is_causal` aligns top-left, which is wrong there -- and for windows.
    """
    if window is None and (start == 0 or S == 1):
        return None
    q_pos = torch.arange(start, start + S, device=device)[:, None]
    k_pos = torch.arange(T, device=device)[None, :]
    mask = k_pos <= q_pos
    if window is not None:
        mask &= k_pos > q_pos - window
    return mask


def _sdpa(q, k, v, mask, S, scale, dropout_p):
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, is_causal=(mask is None and S > 1),
        scale=scale, dropout_p=dropout_p,
    )


# =============================================================================
# ATTENTION -- GQA  (see sarvam/gqa_forward_annotated.py)
# =============================================================================

class GQAttention(nn.Module):
    def __init__(self, cfg: AnuLMConfig, rotary: Rotary):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout
        # One fused projection: (H + 2*Hkv) * Dh, exactly as the 30B does.
        self.query_key_value = nn.Linear(
            cfg.hidden_size, (cfg.n_head + 2 * cfg.n_kv_head) * cfg.head_dim, bias=False
        )
        self.dense = nn.Linear(cfg.n_head * cfg.head_dim, cfg.hidden_size, bias=False)
        self.use_qk_norm = cfg.use_qk_norm
        if cfg.use_qk_norm:
            self.query_layernorm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
            self.key_layernorm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        # YaRN's temperature term lives here, not in the rotary tables --
        # the same place MLA puts it. Without this, `--yarn` on a GQA model
        # would apply the band blend and silently drop half of YaRN.
        self.scale = cfg.head_dim ** -0.5 * (rotary.attn_mscale ** 2)

    def forward(self, x, cos, sin, cache: Optional[dict] = None, start: int = 0,
                window: Optional[int] = None):
        B, S, _ = x.shape
        qkv = self.query_key_value(x)
        qkv = qkv.view(B, S, self.n_head + 2 * self.n_kv_head, self.head_dim)
        q, k, v = qkv.split([self.n_head, self.n_kv_head, self.n_kv_head], dim=-2)
        q = q.transpose(1, 2)                         # (B, H,   S, Dh)
        k = k.transpose(1, 2)                         # (B, Hkv, S, Dh)
        v = v.transpose(1, 2)                         # (B, Hkv, S, Dh)

        if self.use_qk_norm:                          # before RoPE, as in the 30B
            q = self.query_layernorm(q)
            k = self.key_layernorm(k)
        q, k = apply_rope(q, k, cos, sin)

        if cache is not None:
            # Cache the Hkv heads, not their H broadcast copies -- that ratio
            # (16:1 in the real 30B) is GQA's entire inference saving.
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        T = k.shape[2]

        # Broadcast each KV head across its query group. (torch>=2.5 could do
        # this inside SDPA with enable_gqa=True; done explicitly for version safety.)
        k = k.repeat_interleave(self.n_rep, dim=1)    # (B, H, T, Dh)
        v = v.repeat_interleave(self.n_rep, dim=1)

        y = _sdpa(q, k, v, _attn_mask(S, T, start, window, x.device), S,
                  self.scale, self.dropout if self.training else 0.0)     # (B, H, S, Dh)
        y = y.transpose(1, 2).contiguous().view(B, S, -1)
        return self.dense(y)


# =============================================================================
# ATTENTION -- MLA  (see sarvam/mla_forward_annotated.py)
# =============================================================================

class MLAttention(nn.Module):
    def __init__(self, cfg: AnuLMConfig, rotary: Rotary):
        super().__init__()
        self.n_head = cfg.n_head
        self.qk_nope = cfg.qk_nope_head_dim
        self.qk_rope = cfg.qk_rope_head_dim
        self.q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        self.v_head_dim = cfg.v_head_dim
        self.kv_lora_rank = cfg.kv_lora_rank
        self.dropout = cfg.dropout

        self.q_proj = nn.Linear(cfg.hidden_size, cfg.n_head * self.q_head_dim, bias=False)
        # The one projection that produces both the shared latent and the single
        # MQA-style positional key. Width = kv_lora_rank + qk_rope_head_dim.
        self.kv_a_proj_with_mqa = nn.Linear(
            cfg.hidden_size, cfg.kv_lora_rank + cfg.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = RMSNorm(cfg.kv_lora_rank, cfg.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            cfg.kv_lora_rank, cfg.n_head * (cfg.qk_nope_head_dim + cfg.v_head_dim), bias=False
        )
        self.o_proj = nn.Linear(cfg.n_head * cfg.v_head_dim, cfg.hidden_size, bias=False)

        self.use_qk_norm = cfg.use_qk_norm
        if cfg.use_qk_norm:
            self.q_layernorm = RMSNorm(self.q_head_dim, cfg.rms_norm_eps)

        # YaRN's temperature term lives here, not in the rotary tables.
        self.scale = self.q_head_dim ** -0.5 * (rotary.attn_mscale ** 2)

    def forward(self, x, cos, sin, cache: Optional[dict] = None, start: int = 0,
                window: Optional[int] = None):
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.n_head, self.q_head_dim).transpose(1, 2)
        if self.use_qk_norm:
            q = self.q_layernorm(q)
        q_nope, q_pe = q.split([self.qk_nope, self.qk_rope], dim=-1)

        compressed = self.kv_a_proj_with_mqa(x)                   # (B, S, Lkv+Rk)
        compressed, k_pe = compressed.split([self.kv_lora_rank, self.qk_rope], dim=-1)
        k_pe = k_pe.view(B, S, 1, self.qk_rope).transpose(1, 2)   # (B, 1, S, Rk)

        # RoPE touches only the rope slice; the latent stays position-free --
        # which is exactly what lets the latent be cached below.
        q_pe, k_pe = apply_rope(q_pe, k_pe, cos, sin)
        compressed = self.kv_a_layernorm(compressed)              # (B, S, Lkv)

        if cache is not None:
            # MLA's cache IS the point of MLA: kv_lora_rank + qk_rope_head_dim
            # numbers per token (48 + 24 here; 512 + 64 in the 105B) regardless
            # of head count, against GQA's 2 * Hkv * Dh. K and V are re-expanded
            # from it through kv_b_proj every step.
            if "c" in cache:
                compressed = torch.cat([cache["c"], compressed], dim=1)
                k_pe = torch.cat([cache["k_pe"], k_pe], dim=2)
            cache["c"], cache["k_pe"] = compressed, k_pe
        T = compressed.shape[1]

        kv = self.kv_b_proj(compressed)
        kv = kv.view(B, T, self.n_head, self.qk_nope + self.v_head_dim).transpose(1, 2)
        k_nope, v = kv.split([self.qk_nope, self.v_head_dim], dim=-1)
        k_pe = k_pe.expand(-1, self.n_head, -1, -1)               # (B, H, T, Rk)

        q = torch.cat([q_nope, q_pe], dim=-1)                     # (B, H, S, Qd)
        k = torch.cat([k_nope, k_pe], dim=-1)                     # (B, H, T, Qd)

        # SDPA allows V's head dim to differ from Q/K's -- which MLA needs.
        y = _sdpa(q, k, v, _attn_mask(S, T, start, window, x.device), S,
                  self.scale, self.dropout if self.training else 0.0)     # (B, H, S, Vd)
        y = y.transpose(1, 2).contiguous().view(B, S, self.n_head * self.v_head_dim)
        return self.o_proj(y)


# =============================================================================
# MoE  (see sarvam/moe_routing_annotated.py)
# =============================================================================

class Router(nn.Module):
    """Sigmoid scoring + aux-loss-free bias balancing + group-limited top-k.

    The bias is NOT learned by gradient descent. It is nudged after each
    optimizer step by `Model.update_expert_biases()`, using the load counts this
    module accumulates. That is the whole aux-loss-free trick: balancing steers
    selection without ever entering the loss.
    """

    def __init__(self, cfg: AnuLMConfig):
        super().__init__()
        self.top_k = cfg.num_experts_per_tok
        self.num_experts = cfg.num_experts
        self.n_group = cfg.n_group
        self.topk_group = cfg.topk_group
        self.routed_scaling_factor = cfg.routed_scaling_factor

        self.weight = nn.Parameter(torch.empty(cfg.num_experts, cfg.hidden_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Buffers, not parameters: no grad, but they must ride along in the
        # state dict and follow .to(device).
        self.register_buffer("expert_bias", torch.zeros(cfg.num_experts))
        self.register_buffer("load_counts", torch.zeros(cfg.num_experts))
        self.seq_balance_alpha = cfg.seq_balance_alpha
        self.router_z_alpha = cfg.router_z_alpha

    def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D)
        """Returns (topk_idx, combine weights, aux) -- aux is None unless one of
        the regularisers is on and the module is training. `seq_len` lets the
        sequence-wise balance loss see the (B, S) layout of a flat batch."""
        # fp32 router: top-k is discrete, and bf16 rounding would flip choices.
        # Casting the inputs is not enough -- autocast re-casts F.linear to
        # bf16 whatever dtype it is handed, and every GPU run before this fix
        # routed in bf16 (3.2% of expert sets differed from fp32 on a real
        # checkpoint). Autocast has to be off for the whole gate.
        with torch.autocast(device_type=x_flat.device.type, enabled=False):
            topk_idx, w, aux = self._gate(x_flat.float(), seq_len)
        return topk_idx, w.to(x_flat.dtype), aux

    def _gate(self, x_flat, seq_len: Optional[int]):
        logits = F.linear(x_flat, self.weight.float())             # (N, E)
        scores = torch.sigmoid(logits)                             # independent, not softmax

        scores_for_choice = scores + self.expert_bias[None, :]
        if self.n_group > 1:
            N = scores.shape[0]
            per_group = self.num_experts // self.n_group
            grouped = scores_for_choice.view(N, self.n_group, per_group)
            # A group is worth its best two experts.
            g_score = grouped.topk(min(2, per_group), dim=-1)[0].sum(-1)   # (N, G)
            g_idx = g_score.topk(self.topk_group, dim=-1)[1]               # (N, Gk)
            g_mask = torch.zeros_like(g_score).scatter_(1, g_idx, 1.0)
            mask = g_mask.unsqueeze(-1).expand(N, self.n_group, per_group).reshape(N, -1)
            scores_for_choice = scores_for_choice.masked_fill(~mask.bool(), float("-inf"))

        topk_idx = scores_for_choice.topk(self.top_k, dim=-1)[1]           # (N, k)
        # Gather from the UNBIASED scores: the bias picks, it never weights.
        w = scores.gather(1, topk_idx)                                     # (N, k)
        if self.top_k > 1:
            w = w / (w.sum(-1, keepdim=True) + 1e-20)
        w = w * self.routed_scaling_factor

        aux = None
        if self.training:
            with torch.no_grad():
                self.load_counts += torch.bincount(
                    topk_idx.flatten(), minlength=self.num_experts
                ).to(self.load_counts.dtype)
            if self.seq_balance_alpha > 0 or self.router_z_alpha > 0:
                aux = logits.new_zeros(())
            if self.seq_balance_alpha > 0:
                # DeepSeek-V3's complementary loss (their eqs. 17-20), per
                # sequence: f_i is expert i's share of the sequence's k*T slots
                # scaled so a uniform router scores 1.0; P_i is its mean
                # normalised sigmoid score. Only P carries gradient. Tiny alpha
                # (1e-4 in V3): it is there to stop one *sequence* collapsing
                # onto an expert, which the batch-level bias rule cannot see.
                T = seq_len or scores.shape[0]
                onehot = F.one_hot(topk_idx, self.num_experts).sum(1).float()      # (N, E)
                f = onehot.view(-1, T, self.num_experts).mean(1) * (self.num_experts / self.top_k)
                P = (scores / scores.sum(-1, keepdim=True)).view(-1, T, self.num_experts).mean(1)
                aux = aux + self.seq_balance_alpha * (f * P).sum(-1).mean()
            if self.router_z_alpha > 0:
                # Router z-loss (ST-MoE): keeps the logits from drifting to
                # magnitudes where sigmoid saturates and top-k stops moving.
                aux = aux + self.router_z_alpha * torch.logsumexp(logits, dim=-1).pow(2).mean()

        return topk_idx, w, aux


class MoE(nn.Module):
    def __init__(self, cfg: AnuLMConfig):
        super().__init__()
        self.top_k = cfg.num_experts_per_tok
        self.num_experts = cfg.num_experts
        self.impl = cfg.moe_impl
        self._stacked = None            # eval-mode cache for _route_grouped
        self.router = Router(cfg)
        self.experts = nn.ModuleList(
            [SwiGLU(cfg.hidden_size, cfg.moe_intermediate_size) for _ in range(cfg.num_experts)]
        )
        self.shared_expert = (
            SwiGLU(cfg.hidden_size, cfg.moe_intermediate_size * cfg.num_shared_experts)
            if cfg.num_shared_experts > 0 else None
        )

    def forward(self, x):                                     # (B, S, D)
        """Returns (y, aux): the routed + shared output, and the router's
        regulariser (None unless enabled and training)."""
        B, S, D = x.shape
        x_flat = x.view(-1, D)                                # (N, D)
        topk_idx, topk_w, aux = self.router(x_flat, seq_len=S)
        route = {"dense": self._route_dense,
                 "grouped": self._route_grouped}.get(self.impl, self._route_sparse)
        y = route(x_flat, topk_idx, topk_w).view(B, S, D)
        if self.shared_expert is not None:
            y = y + self.shared_expert(x)     # dense, every token, outside routing
        return y, aux

    def _route_sparse(self, x_flat, topk_idx, topk_w):
        """One GEMM per expert over its own tokens. Default."""
        N, D = x_flat.shape
        flat_expert = topk_idx.reshape(-1)                    # (N*k,)
        # Which token each (token, slot) assignment belongs to.
        tok_of_slot = torch.arange(N, device=x_flat.device).repeat_interleave(self.top_k)

        out = x_flat.new_zeros(N * self.top_k, D)
        for e in range(self.num_experts):
            sel = (flat_expert == e).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            # index_select + index_put with distinct indices: differentiable,
            # so unlike the released 105B this path trains. No repeat_interleave
            # of the hidden states, so peak memory stays at O(N*k) rows touched
            # rather than O(N*k) rows materialised up front.
            out[sel] = self.experts[e](x_flat[tok_of_slot[sel]]).to(out.dtype)

        return (out.view(N, self.top_k, D) * topk_w.unsqueeze(-1)).sum(1)

    def _route_dense(self, x_flat, topk_idx, topk_w):
        """Every expert on every token, combined through a one-hot weight
        matrix. Static shapes throughout, so Dynamo captures the whole block in
        one graph -- at num_experts/top_k times the expert FLOPs."""
        N, D = x_flat.shape
        weights = x_flat.new_zeros(N, self.num_experts).scatter_(1, topk_idx, topk_w)
        y = x_flat.new_zeros(N, D)
        for e in range(self.num_experts):
            y = y + weights[:, e: e + 1] * self.experts[e](x_flat)
        return y

    def _stacked_weights(self):
        """The ModuleList's expert weights as three (E, in, out) bf16 tensors.

        Stacking `.t()` views gives contiguous (E, D, I) directly -- the layout
        grouped_mm wants for x @ W -- and keeps the checkpoint layout identical
        to the sparse path. In training that is one copy of the expert weights
        per step (~1% of the step's bytes). In eval the weights do not change,
        so the stack is built once and reused: for decoding one token at a time
        the copy would otherwise cost more than the GEMMs it feeds.
        """
        # Keyed on the parameters' in-place version counters, so a
        # load_state_dict() after the first forward invalidates the copy too.
        key = sum(p._version for e in self.experts for p in e.parameters())
        if not self.training and self._stacked is not None and self._stacked[0] == key:
            return self._stacked[1]
        stacked = tuple(
            torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(torch.bfloat16)
            for name in ("gate_proj", "up_proj", "down_proj"))
        self._stacked = (key, stacked) if not self.training else None
        return stacked

    def train(self, mode: bool = True):
        self._stacked = None            # weights are about to move; drop the copy
        return super().train(mode)

    def _route_grouped(self, x_flat, topk_idx, topk_w):
        """Every expert in ONE grouped GEMM per projection (F.grouped_mm).

        Sort the N*k (token, slot) assignments by expert; the per-expert row
        ranges are then contiguous and `offs` (a cumsum of the bincount) tells
        the kernel where each group ends. argsort, bincount and cumsum all have
        static output shapes, so unlike `nonzero` this traces in one graph.
        bf16 only, by the kernel's contract -- run it under autocast.
        """
        N, D = x_flat.shape
        E, k = self.num_experts, self.top_k
        flat_expert = topk_idx.reshape(-1)                                  # (N*k,)
        order = torch.argsort(flat_expert, stable=True)                     # slots, grouped by expert
        # Per-expert counts via scatter_add rather than bincount: bincount's
        # output length depends on the data, which is itself a graph break.
        counts = torch.zeros(E, dtype=torch.int64, device=x_flat.device).scatter_add_(
            0, flat_expert, torch.ones_like(flat_expert))
        offs = counts.cumsum(0).to(torch.int32)
        xs = x_flat[order // k].to(torch.bfloat16)                          # (N*k, D) expert-sorted
        w_gate, w_up, w_down = self._stacked_weights()
        h = F.silu(F.grouped_mm(xs, w_gate, offs=offs)) * F.grouped_mm(xs, w_up, offs=offs)
        ys = F.grouped_mm(h, w_down, offs=offs)                              # (N*k, D) expert-sorted
        out = x_flat.new_zeros(N * k, D)
        out[order] = ys.to(out.dtype)                                        # back to slot order
        return (out.view(N, k, D) * topk_w.unsqueeze(-1)).sum(1)


# =============================================================================
# BLOCK / MODEL
# =============================================================================

class Block(nn.Module):
    def __init__(self, cfg: AnuLMConfig, layer_idx: int, rotary: Rotary):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = (
            GQAttention(cfg, rotary) if cfg.attn == "gqa" else MLAttention(cfg, rotary)
        )
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        # Layer 0 dense, the rest MoE -- first_k_dense_replace = 1 in both models.
        self.is_moe = layer_idx >= cfg.first_k_dense_replace
        self.mlp = MoE(cfg) if self.is_moe else SwiGLU(cfg.hidden_size, cfg.intermediate_size)
        # Sliding window on the first max_window_layers layers, full attention
        # on the rest (Qwen2 semantics). None = full attention.
        self.window = cfg.sliding_window if layer_idx < cfg.max_window_layers else None

    def forward(self, x, cos, sin, cache: Optional[dict] = None, start: int = 0):
        """Returns (x, aux). aux is the MoE regulariser or None."""
        x = x + self.attn(self.input_layernorm(x), cos, sin,
                          cache=cache, start=start, window=self.window)
        h = self.mlp(self.post_attention_layernorm(x))
        h, aux = h if self.is_moe else (h, None)
        return x + h, aux


class AnuLM(nn.Module):
    def __init__(self, cfg: AnuLMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim
        self.rotary = Rotary(cfg, rope_dim)
        self.layers = nn.ModuleList([Block(cfg, i, self.rotary) for i in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.grad_ckpt = False          # set via enable_gradient_checkpointing()
        self.apply(self._init_weights)
        # nanoGPT's scaled init for the residual output projections.
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "dense.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=cfg.initializer_range / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.initializer_range)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.initializer_range)

    def new_cache(self) -> list[dict]:
        """One dict per layer; each attention module owns its own entries."""
        return [dict() for _ in self.layers]

    def forward(self, idx, targets=None, cache: Optional[list] = None, start: int = 0):
        """`cache` (from `new_cache()`) holds K/V for the `start` tokens already
        seen; this call attends to them and appends the new ones. Positions are
        absolute -- start..start+S-1 -- so RoPE agrees with the cached keys.

        With targets, the returned loss is LM cross-entropy plus any router
        regulariser; the regulariser alone is kept in `self.aux_loss` (detached)
        so the training log can report the two separately.
        """
        B, S = idx.shape
        assert start + S <= self.cfg.block_size, \
            f"positions up to {start + S} exceed block_size {self.cfg.block_size}"
        x = self.embed_tokens(idx)                                # (B, S, D)
        cos, sin = self.rotary(S, x.device, x.dtype, start=start)
        aux_total = None
        for i, layer in enumerate(self.layers):
            if self.grad_ckpt and self.training:
                # Recompute activations in the backward pass instead of storing
                # them: roughly 30% slower per step, but frees the activation
                # memory that otherwise caps batch size on a small card.
                x, aux = torch.utils.checkpoint.checkpoint(
                    layer, x, cos, sin, use_reentrant=False)
            else:
                x, aux = layer(x, cos, sin, cache[i] if cache is not None else None, start)
            if aux is not None:
                aux_total = aux if aux_total is None else aux_total + aux
        x = self.norm(x)
        self.aux_loss = aux_total.detach() if aux_total is not None else None

        if targets is None:
            # Inference shortcut: only the last position matters.
            logits = self.lm_head(x[:, -1:, :])
            return logits, None
        logits = self.lm_head(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_index=-1
        )
        if aux_total is not None:
            loss = loss + aux_total
        return logits, loss

    def enable_gradient_checkpointing(self, enable: bool = True):
        """Unlike the released Sarvam 105B file -- which advertises support and
        then never calls the checkpoint function -- this actually takes effect.
        See sarvam/decoder_and_model_annotated.py note 3."""
        self.grad_ckpt = enable

    # --- aux-loss-free load balancing -------------------------------------
    @torch.no_grad()
    def update_expert_biases(self) -> float:
        """Call once per optimizer step. Returns the load-imbalance ratio
        (max expert load / mean load; 1.0 is perfect, top_k*... is collapse).

        DeepSeek-V3's rule: nudge each expert's bias by a fixed step against the
        sign of its load error. Overloaded experts get harder to pick, starved
        ones easier -- and because the bias never touches the combine weights,
        none of this perturbs the language-modelling gradient.
        """
        ratios = []
        for m in self.modules():
            if not isinstance(m, Router):
                continue
            load = m.load_counts
            total = load.sum()
            if total <= 0:
                continue
            mean = load.mean()
            m.expert_bias += self.cfg.bias_update_rate * torch.sign(mean - load)
            ratios.append((load.max() / mean).item())
            load.zero_()
        return max(ratios) if ratios else 1.0

    # --- bookkeeping --------------------------------------------------------
    def num_params(self) -> tuple[int, int]:
        """(total, active-per-token). Mirrors the count in the annotated files."""
        total = sum(p.numel() for p in self.parameters())
        cfg = self.cfg
        inactive = 0
        for layer in self.layers:
            if isinstance(layer.mlp, MoE):
                per_expert = sum(p.numel() for p in layer.mlp.experts[0].parameters())
                inactive += per_expert * (cfg.num_experts - cfg.num_experts_per_tok)
        return total, total - inactive

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, eos_id: int | None = None, use_cache: bool = True,
                 repetition_penalty: float = 1.0):
        """Incremental decoding. The prompt is prefilled once; every later step
        runs a single token against the KV cache. Once the context is full
        (block_size tokens) the cache is rebuilt from the last block_size tokens
        each step, which is exactly what the uncached path computes -- so the
        two paths agree token-for-token, and the tests hold them to it.

        Stops early when every row has produced `eos_id`; finished rows keep
        receiving eos so the batch stays rectangular.

        `repetition_penalty` > 1 (CTRL's rule: divide a positive logit, multiply
        a negative one, for every token already in the sequence) discourages the
        loops a small model falls into at low temperature. 1.0 is off, and the
        default, so nothing above changes.
        """
        B = idx.shape[0]
        bs = self.cfg.block_size
        done = torch.zeros(B, dtype=torch.bool, device=idx.device)
        cache, start, pending = None, 0, idx
        for _ in range(max_new_tokens):
            if not use_cache:
                logits, _ = self(idx[:, -bs:])
            else:
                if cache is None or start + pending.shape[1] > bs:
                    cache, start, pending = self.new_cache(), 0, idx[:, -bs:]
                logits, _ = self(pending, cache=cache, start=start)
                start += pending.shape[1]
            logits = logits[:, -1, :].float()
            if repetition_penalty != 1.0:
                seen = logits.gather(1, idx)
                seen = torch.where(seen > 0, seen / repetition_penalty, seen * repetition_penalty)
                logits = logits.scatter(1, idx, seen)
            logits = logits / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)              # (B, 1)
            if eos_id is not None:
                nxt = torch.where(done[:, None], torch.full_like(nxt, eos_id), nxt)
            idx = torch.cat((idx, nxt), dim=1)
            pending = nxt
            if eos_id is not None:
                done |= nxt[:, 0] == eos_id
                if bool(done.all()):
                    break
        return idx


if __name__ == "__main__":
    for name, cfg in [("nano_30b (GQA)", AnuLMConfig.nano_30b()),
                      ("nano_105b (MLA)", AnuLMConfig.nano_105b())]:
        # Seed per preset, not once up front: every number below is then
        # reproducible and quotable in the README, and stays that way if the
        # list is reordered or another preset is added. It matters most for the
        # "params without grad" count, which is a property of which experts this
        # particular 128-token batch happened to miss.
        torch.manual_seed(0)
        model = AnuLM(cfg)
        total, active = model.num_params()
        print(f"{name:18s}  total {total/1e6:6.2f}M   active/token {active/1e6:6.2f}M "
              f"({100*active/total:.1f}%)   layers {cfg.n_layer}")
        x = torch.randint(0, cfg.vocab_size, (2, 64))
        model.train()
        logits, loss = model(x, x)
        loss.backward()
        # Smoke test: an untrained model should sit at ln(V), every parameter
        # should have received a gradient, and the bias update should run.
        no_grad = [n for n, p in model.named_parameters() if p.grad is None]
        imbalance = model.update_expert_biases()
        print(f"{'':18s}  logits {tuple(logits.shape)}  loss {loss.item():.4f}  "
              f"(ln(V) = {math.log(cfg.vocab_size):.4f})")
        print(f"{'':18s}  backward OK  params without grad: {len(no_grad)}"
              f"{' -> ' + ', '.join(no_grad[:3]) if no_grad else ''}")
        print(f"{'':18s}  expert load imbalance after 1 step: {imbalance:.2f}x\n")

# The project was called nanosarvam until 2026-09-18. Checkpoints saved before
# then pickle their config as ``model.NanoSarvamConfig``; keep both names
# resolvable so torch.load keeps working on them.
NanoSarvamConfig = AnuLMConfig
NanoSarvam = AnuLM


def load_checkpoint(path, map_location="cpu") -> dict:
    """Load a training checkpoint (``.pt``) or a folder written by
    ``export_hf.py`` (``config.json`` + ``model.safetensors``) into the same
    dict shape: ``model`` (state dict), ``cfg`` (AnuLMConfig), and whatever
    ``step``, ``val_loss``, ``qa_templates`` and ``base_ckpt`` were saved."""
    import json
    from pathlib import Path as _P
    p = _P(path)
    if not p.is_dir():
        return torch.load(str(p), map_location=map_location, weights_only=False)
    from safetensors.torch import load_file
    meta = json.loads((p / "config.json").read_text(encoding="utf-8"))
    cfg_d = dict(meta["config"])
    if cfg_d.get("tokenizer_path"):
        cfg_d["tokenizer_path"] = str(p / cfg_d["tokenizer_path"])
    known = {f.name for f in dataclasses.fields(AnuLMConfig)}
    cfg = AnuLMConfig(**{k: v for k, v in cfg_d.items() if k in known})
    dev = map_location if isinstance(map_location, str) else str(map_location)
    sd = load_file(str(p / "model.safetensors"), device=dev)
    ck = {"model": sd, "cfg": cfg}
    for k in ("step", "val_loss", "qa_templates", "base_ckpt"):
        if meta.get(k) is not None:
            ck[k] = meta[k]
    return ck
