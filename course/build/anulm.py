"""
Your AnuLM, built chapter by chapter.

Each chapter adds pieces to this file. When it is finished (Chapter 8) it has
the same interface as the repository's model.py, and the course's checks
compare the two, weight for weight.

    python ../checks/ch04.py            # check what you have built so far
    python ../checks/ch04.py --impl solution    # see the checks pass on the reference

Replace each `raise NotImplementedError` with your code. Stuck? The reference
for each chapter is in solutions/, and the real thing is in model.py.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Chapter 4: the skeleton ----------------------------------------------------

class RMSNorm(nn.Module):
    """Scale each vector so its root-mean-square is 1, then multiply by a learned gain.

        rms(x) = sqrt(mean(x ** 2) + eps)      over the last dimension
        out    = (x / rms(x)) * weight

    Three details the checks look for:
      - `weight` is an nn.Parameter of shape (dim,) and starts at ones;
      - compute the statistics in float32 even when x is bfloat16, and return
        the input's dtype (x.float(), then .to(dtype));
      - use torch.rsqrt(... + eps), with eps inside the square root.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        raise NotImplementedError("Chapter 4, section 4.2")

    def forward(self, x):
        raise NotImplementedError("Chapter 4, section 4.2")


class SwiGLU(nn.Module):
    """A gated MLP: down_proj( silu(gate_proj(x)) * up_proj(x) ).

    Three Linear layers, all with bias=False and these exact names, so the
    checks can copy model.py's weights into yours:
        gate_proj: hidden -> intermediate
        up_proj:   hidden -> intermediate
        down_proj: intermediate -> hidden
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        raise NotImplementedError("Chapter 4, section 4.3")

    def forward(self, x):
        raise NotImplementedError("Chapter 4, section 4.3")


class MLPBlock(nn.Module):
    """One pre-norm residual block, before attention exists:

        x + mlp(post_attention_layernorm(x))

    Name the norm `post_attention_layernorm` and the MLP `mlp` (model.py's names).
    """

    def __init__(self, hidden_size: int, intermediate_size: int, eps: float = 1e-6):
        super().__init__()
        raise NotImplementedError("Chapter 4, section 4.4")

    def forward(self, x):
        raise NotImplementedError("Chapter 4, section 4.4")


class SkeletonLM(nn.Module):
    """embed_tokens -> n_layer MLPBlocks -> norm -> lm_head, returning (logits, loss).

    Initialise like model.py: every Linear and Embedding weight from N(0, 0.02),
    then every `down_proj.weight` again from N(0, 0.02 / sqrt(2 * n_layer)).
    The loss is cross-entropy between logits and `targets` (already shifted by
    the caller), ignoring positions where targets == -1.
    """

    def __init__(self, vocab_size: int = 259, hidden_size: int = 384, n_layer: int = 4,
                 intermediate_size: int = 768, initializer_range: float = 0.02,
                 tie_word_embeddings: bool = False):
        super().__init__()
        raise NotImplementedError("Chapter 4, section 4.5")

    def forward(self, idx, targets=None):
        raise NotImplementedError("Chapter 4, section 4.5")


# --- Chapter 5: positions -------------------------------------------------------

def rotate_half(x):
    """Split the last dimension into halves (x1, x2) and return (-x2, x1).

    This is the "split-halves" (GPT-NeoX) layout model.py uses: dimension i is
    paired with dimension i + dim/2, not with its neighbour i + 1.
    """
    raise NotImplementedError("Chapter 5, section 5.2")


def apply_rope(q, k, cos, sin):
    """Rotate q and k, shape (B, H, S, Dr), by angles given as cos/sin of shape (S, Dr):

        q * cos + rotate_half(q) * sin        (and the same for k)

    cos and sin need two leading dimensions added to broadcast over B and H.
    Return (q_rotated, k_rotated).
    """
    raise NotImplementedError("Chapter 5, section 5.2")


def yarn_mscale(scale: float, mscale: float = 1.0) -> float:
    """YaRN's attention temperature: 1.0 if scale <= 1, else 0.1 * mscale * ln(scale) + 1.0."""
    raise NotImplementedError("Chapter 5, section 5.4")


def yarn_correction_range(beta_fast, beta_slow, dim, base, orig_ctx):
    """Where YaRN's ramp starts and ends, as pair indices.

    The pair index that turns `r` full rotations within orig_ctx positions is
        d(r) = dim * ln(orig_ctx / (r * 2 * pi)) / (2 * ln(base))
    Return (max(floor(d(beta_fast)), 0), min(ceil(d(beta_slow)), dim - 1)).
    """
    raise NotImplementedError("Chapter 5, section 5.4")


class Rotary(nn.Module):
    """The cos/sin tables for rotary positions, shared by every layer.

    In __init__:
      - inv_freq[i] = 1 / rope_theta ** (2i / dim) for i in 0 .. dim/2 - 1, in float32,
        registered as a buffer (persistent=False);
      - with yarn=True, blend it with inv_freq / yarn_factor using a ramp from 0 to 1
        between yarn_correction_range(...) (if low == high, add 0.001 to high):
            inv_freq = (inv_freq / yarn_factor) * ramp + inv_freq * (1 - ramp)
        and set self.attn_mscale = yarn_mscale(yarn_factor, yarn_mscale_all_dim),
            self.table_mscale = yarn_mscale(yarn_factor, yarn_mscale) / attn_mscale;
        without YaRN both are 1.0.
        Watch out: inside __init__ the argument `yarn_mscale` hides the function of
        the same name. Keep a module-level alias (`_mscale = yarn_mscale`) and call that.

    forward(seq_len, device, dtype, start=0) returns (cos, sin) for absolute positions
    start .. start + seq_len - 1, each of shape (seq_len, dim):
      - angles = outer(positions, inv_freq), computed in float32 whatever `dtype` is;
      - emb = cat(angles, angles) (the split-halves layout), times table_mscale;
      - cos and sin of emb, cast to `dtype`.
    Cache the tables, and rebuild them when a longer length, another device or
    another dtype is asked for.
    """

    def __init__(self, dim: int, rope_theta: float = 10000.0, yarn: bool = False,
                 yarn_factor: float = 8.0, yarn_original_context: int = 256,
                 yarn_beta_fast: float = 32.0, yarn_beta_slow: float = 1.0,
                 yarn_mscale: float = 1.0, yarn_mscale_all_dim: float = 1.0):
        super().__init__()
        raise NotImplementedError("Chapter 5, section 5.3")

    def forward(self, seq_len: int, device, dtype, start: int = 0):
        raise NotImplementedError("Chapter 5, section 5.3")


# --- Chapter 6: attention -----------------------------------------------------------

def attn_mask(S: int, T: int, start: int, window, device):
    """A boolean (S, T) mask, True = this query may attend to this key, or None.

    The S queries sit at absolute positions start .. start+S-1, the T keys at 0 .. T-1.
      - Return None when window is None and (start == 0 or S == 1): then PyTorch's
        is_causal=True (or no mask, for a single token) is exactly right and faster.
      - Otherwise: True where key position <= query position, and, with a window,
        also key position > query position - window.
    """
    raise NotImplementedError("Chapter 6, section 6.2")


class GQAttention(nn.Module):
    """Grouped-query attention, with model.py's names:

      query_key_value: one Linear, hidden -> (n_head + 2 * n_kv_head) * head_dim, no bias
      dense:           Linear, n_head * head_dim -> hidden, no bias
      query_layernorm, key_layernorm: RMSNorm(head_dim) when use_qk_norm
      scale = head_dim ** -0.5 * attn_mscale ** 2   (YaRN's temperature, Chapter 5)

    forward(x, cos, sin, cache=None, start=0, window=None):
      1. project, view as (B, S, n_head + 2 * n_kv_head, head_dim), split into q, k, v,
         and move heads before positions: (B, heads, S, head_dim);
      2. QK-norm (if on), THEN rotary positions (apply_rope);
      3. with a cache dict: append k, v to cache["k"], cache["v"] (n_kv_head heads, not copies);
      4. mask = attn_mask(S, T, start, window, device), T = the number of keys now;
      5. repeat each kv head for its n_head // n_kv_head query heads (repeat_interleave on dim 1);
      6. F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
            is_causal=(mask is None and S > 1), scale=self.scale);
      7. heads back to (B, S, n_head * head_dim), then dense.
    """

    def __init__(self, hidden_size: int, n_head: int, n_kv_head: int, head_dim: int,
                 use_qk_norm: bool = True, eps: float = 1e-6, attn_mscale: float = 1.0):
        super().__init__()
        raise NotImplementedError("Chapter 6, section 6.3")

    def forward(self, x, cos, sin, cache=None, start: int = 0, window=None):
        raise NotImplementedError("Chapter 6, section 6.3")


class Block(nn.Module):
    """The full transformer layer (model.py's layer 0):

        x = x + attn(input_layernorm(x), cos, sin, cache, start, window=self.window)
        x = x + mlp(post_attention_layernorm(x))

    Modules: input_layernorm, attn (GQAttention), post_attention_layernorm, mlp (SwiGLU).
    """

    def __init__(self, hidden_size: int, n_head: int, n_kv_head: int, head_dim: int,
                 intermediate_size: int, use_qk_norm: bool = True, eps: float = 1e-6,
                 attn_mscale: float = 1.0, window=None):
        super().__init__()
        raise NotImplementedError("Chapter 6, section 6.4")

    def forward(self, x, cos, sin, cache=None, start: int = 0):
        raise NotImplementedError("Chapter 6, section 6.4")


class AttnLM(nn.Module):
    """Chapter 4's SkeletonLM with attention: embed_tokens, rotary (a Rotary(head_dim,
    rope_theta)), layers (Blocks; the first max_window_layers get window=sliding_window),
    norm, lm_head.

    Initialise as SkeletonLM did, scaling BOTH residual-writing projections,
    `dense.weight` and `down_proj.weight`, by 1 / sqrt(2 * n_layer).

    forward(idx, targets=None, cache=None, start=0) -> (logits for every position, loss):
      - refuse (assert) start + S > block_size;
      - cos, sin = self.rotary(S, device, dtype, start=start);
      - pass cache[i] (or None) and start to layer i.
    new_cache() returns one empty dict per layer.
    """

    def __init__(self, vocab_size: int = 259, hidden_size: int = 128, n_layer: int = 4, n_head: int = 4,
                 n_kv_head: int = 2, head_dim: int = 32, intermediate_size: int = 256,
                 rope_theta: float = 10000.0, block_size: int = 256, use_qk_norm: bool = True,
                 sliding_window=None, max_window_layers: int = 0, initializer_range: float = 0.02):
        super().__init__()
        raise NotImplementedError("Chapter 6, section 6.5")

    def new_cache(self):
        raise NotImplementedError("Chapter 6, section 6.5")

    def forward(self, idx, targets=None, cache=None, start: int = 0):
        raise NotImplementedError("Chapter 6, section 6.5")

    @torch.no_grad()
    def generate(self, idx, n: int, temperature: float = 1.0, use_cache: bool = True, generator=None):
        """Provided: sample n tokens, with or without the KV cache."""
        cache, pending, start = (self.new_cache() if use_cache else None), idx, 0
        for _ in range(n):
            if use_cache:
                logits, _ = self(pending, cache=cache, start=start)
                start += pending.shape[1]
            else:
                logits, _ = self(idx[:, -self.block_size:])
            p = F.softmax(logits[:, -1] / temperature, dim=-1)
            nxt = torch.multinomial(p, 1, generator=generator)
            idx, pending = torch.cat([idx, nxt], dim=1), nxt
        return idx


# --- Chapter 7: mixture of experts ------------------------------------------------------

class Router(nn.Module):
    """Chooses top_k of num_experts for every token (model.py's Router).

    __init__: `weight` an nn.Parameter (num_experts, hidden_size), kaiming_uniform_(a=sqrt(5));
      `expert_bias` and `load_counts` zeros(num_experts), registered as BUFFERS (not parameters).
    forward(x_flat (N, D), seq_len=None) -> (idx (N, k), weights (N, k), aux or None):
      do the work in float32 with autocast OFF (torch.autocast(device_type=..., enabled=False)):
      1. scores = sigmoid(x @ weight.T)                 every expert scored independently
      2. choice = scores + expert_bias                  the bias steers the choice ...
      3. if n_group > 1: keep only the topk_group groups whose two best `choice` values sum
         highest (others -> -inf)
      4. idx = top_k of choice
      5. weights = scores gathered at idx               ... but never the weights
         normalised to sum to 1 (if top_k > 1), then times routed_scaling_factor
      6. in training mode: load_counts += how many slots each expert received (no grad);
         aux = seq_balance_alpha * (DeepSeek-V3 sequence balance loss)
             + router_z_alpha * mean(logsumexp(logits) ** 2), when either alpha > 0.
      Return weights in x_flat's dtype.
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int, n_group: int = 1,
                 topk_group: int = 1, routed_scaling_factor: float = 2.5,
                 seq_balance_alpha: float = 0.0, router_z_alpha: float = 0.0):
        super().__init__()
        raise NotImplementedError("Chapter 7, section 7.2")

    def forward(self, x_flat, seq_len=None):
        raise NotImplementedError("Chapter 7, section 7.2")


class MoE(nn.Module):
    """router, experts (nn.ModuleList of num_experts SwiGLU(hidden, moe_intermediate_size)),
    shared_expert (SwiGLU(hidden, moe_intermediate_size * num_shared_experts), or None).

    forward(x (B, S, D)) -> (y, aux):
      route the flattened tokens; for each expert, run it ONCE on all the (token, slot)
      pairs sent to it; combine each token's k outputs with its weights; add the shared
      expert's output for every token. Extra keyword arguments go to the Router.
    """

    def __init__(self, hidden_size: int, moe_intermediate_size: int, num_experts: int, top_k: int,
                 num_shared_experts: int = 1, **router_kw):
        super().__init__()
        raise NotImplementedError("Chapter 7, section 7.3")

    def forward(self, x):
        raise NotImplementedError("Chapter 7, section 7.3")


class ExpertBlock(nn.Module):
    """Chapter 6's Block with `mlp = MoE(hidden_size, **moe_kw)`; forward returns (x, aux)."""

    def __init__(self, hidden_size: int, n_head: int, n_kv_head: int, head_dim: int, moe_kw: dict,
                 use_qk_norm: bool = True, eps: float = 1e-6, attn_mscale: float = 1.0, window=None):
        super().__init__()
        raise NotImplementedError("Chapter 7, section 7.4")

    def forward(self, x, cos, sin, cache=None, start: int = 0):
        raise NotImplementedError("Chapter 7, section 7.4")


class MoELM(nn.Module):
    """The whole model. Its weights must load into model.py's AnuLM, so use its names:
    embed_tokens, rotary, layers (a Block for i < first_k_dense_replace, ExpertBlocks after),
    norm, lm_head. Initialise as AttnLM did (the router's weight keeps its kaiming init).
    Refuse (assert) num_experts_per_tok > (num_experts // n_group) * topk_group.

    forward(idx, targets=None, cache=None, start=0) -> (logits, loss): as AttnLM, and add
      every layer's aux (when not None) to the loss.
    update_expert_biases(): for every Router with load: bias += bias_update_rate *
      sign(mean load - load); zero its load_counts; return the largest max/mean ratio.
    num_params(): (total, active), where active leaves out, in every MoE layer, the
      experts a token is not routed to: (num_experts - num_experts_per_tok) x one expert.
    The defaults are the `30b` preset: 17,440,384 parameters, 6,602,368 active.
    """

    def __init__(self, vocab_size: int = 259, block_size: int = 256, n_layer: int = 8, hidden_size: int = 384,
                 n_head: int = 6, head_dim: int = 64, n_kv_head: int = 2, use_qk_norm: bool = True,
                 num_experts: int = 16, num_experts_per_tok: int = 2, num_shared_experts: int = 1,
                 moe_intermediate_size: int = 96, intermediate_size: int = 768, first_k_dense_replace: int = 1,
                 n_group: int = 1, topk_group: int = 1, routed_scaling_factor: float = 2.5,
                 bias_update_rate: float = 1e-3, seq_balance_alpha: float = 0.0, router_z_alpha: float = 0.0,
                 rope_theta: float = 1e6, sliding_window=None, max_window_layers: int = 0,
                 rms_norm_eps: float = 1e-6, initializer_range: float = 0.02):
        super().__init__()
        raise NotImplementedError("Chapter 7, section 7.5")

    def new_cache(self):
        raise NotImplementedError("Chapter 7, section 7.5")

    def forward(self, idx, targets=None, cache=None, start: int = 0):
        raise NotImplementedError("Chapter 7, section 7.5")

    def update_expert_biases(self) -> float:
        raise NotImplementedError("Chapter 7, section 7.6")

    def num_params(self):
        raise NotImplementedError("Chapter 7, section 7.5")


# --- Chapter 8: generation and the machinery of pre-training --------------------------------

import os  # noqa: E402  (used by save_checkpoint)


@torch.no_grad()
def _moelm_generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k=None, eos_id=None,
                    use_cache: bool = True, repetition_penalty: float = 1.0):
    """MoELM.generate: sample max_new_tokens tokens after idx (B, T), doing exactly what
    model.py's AnuLM.generate does, in the same order, so the same seed gives the same text:

      per step:
        - logits for the last position: with the cache, feed only the new tokens (`pending`)
          at `start`; when start + len(pending) would pass block_size, start a new cache
          from the last block_size tokens. Without the cache, re-read idx[:, -block_size:].
        - repetition_penalty != 1: for every token already in idx, divide its logit if
          positive, multiply it if negative;
        - divide by max(temperature, 1e-5); with top_k, set all but the top_k logits to -inf;
        - torch.multinomial(softmax(logits), 1);
        - with eos_id: rows already finished get eos_id; stop when every row has finished.
    """
    raise NotImplementedError("Chapter 8, section 8.1")


MoELM.generate = _moelm_generate          # becomes a method of the MoELM you wrote in Chapter 7


def lr_at(step: int, steps: int, lr: float, min_lr: float, warmup: int, schedule: str = "cosine",
          decay_frac: float = 0.2) -> float:
    """The learning rate at `step` (counting from 0), as train.py's lr_at:
      - during warmup (step < warmup): lr * (step + 1) / warmup;
      - "cosine": from lr down to min_lr along half a cosine, over steps - warmup;
      - "wsd" (warmup-stable-decay): lr until int(steps * (1 - decay_frac)), then a straight
        line down to min_lr at the last step.
    """
    raise NotImplementedError("Chapter 8, section 8.2")


def make_optimizer(model, lr: float, weight_decay: float = 0.1):
    """AdamW(betas=(0.9, 0.95), eps=1e-8) with two groups: parameters with 2+ dimensions get
    weight_decay; 1-D ones (norm gains) get 0. Skip parameters with requires_grad False."""
    raise NotImplementedError("Chapter 8, section 8.3")


def accumulate(model, batches) -> float:
    """Zero the gradients, then backward each micro-batch's loss divided by len(batches), so the
    gradients are those of the MEAN loss over all of them. Return that mean loss (a float)."""
    raise NotImplementedError("Chapter 8, section 8.4")


def train_step(model, opt, batches, lr: float, clip: float = 1.0) -> float:
    """Set every param group's lr; accumulate; clip the gradient norm to `clip` (if > 0);
    opt.step(); then model.update_expert_biases(). Return the loss."""
    raise NotImplementedError("Chapter 8, section 8.4")


def save_checkpoint(path, model, opt, step: int, **extra) -> None:
    """torch.save {"model": state_dict (buffers included), "opt": opt.state_dict(), "step": step,
    **extra}. Write it to path + ".tmp" first, then os.replace() it onto path, so a crash
    in the middle of writing leaves the previous checkpoint intact."""
    raise NotImplementedError("Chapter 8, section 8.5")


def load_checkpoint(path, model, opt) -> dict:
    """Load (map_location="cpu"), restore the model and the optimizer in place, return the dict."""
    raise NotImplementedError("Chapter 8, section 8.5")
