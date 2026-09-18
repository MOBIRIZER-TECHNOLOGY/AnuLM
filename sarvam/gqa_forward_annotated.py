"""
Sarvam-30B  ::  SarvamMoEAttention.forward  -- annotated with tensor shapes.

Extract of sarvam-30b/modeling_sarvam_moe.py:407-511 (code verbatim, comments added).
Not runnable on its own; read alongside the original.
Companion to mla_forward_annotated.py -- same annotation conventions.

Shape symbols, with sarvam-30b/config.json values substituted:

    B    = bsz                     batch
    S    = q_len                   new tokens this step (S == 1 when decoding)
    T    = kv_seq_len              total keys attended = S + cached
    H    = num_attention_heads     = 64
    Hkv  = num_key_value_heads     = 4       <- 16x fewer than query heads
    G    = num_key_value_groups    = 64 / 4  = 16
    Dh   = head_dim                = 64      (explicit in config, == 4096/64 anyway)
    D    = hidden_size             = 4096

Contrast with the 105B: there, one 576-d latent served all heads and keys/values
had different widths (192 / 128). Here everything is a uniform 64-d head and the
saving comes only from having 4 KV heads instead of 64. Plain, well-trodden GQA.
"""


# Referenced by the forward pass; standard HF helper, unchanged.
def repeat_kv(hidden_states, n_rep):
    # (B, 4, T, 64) -> (B, 64, T, 64) by expand+reshape.
    # A view where possible, so it costs no copy in the FlashAttention/SDPA
    # subclasses -- but the eager path below does force materialisation.
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def forward(self, hidden_states, attention_mask=None, position_ids=None,
            past_key_value=None, output_attentions=False, use_cache=False,
            position_embeddings=None, **kwargs):

    bsz, q_len, _ = hidden_states.size()
    # hidden_states                                              (B, S, 4096)

    # ---- 1. FUSED QKV -------------------------------------------------------
    # One matmul for all three, unlike the 105B's separate q_proj / kv_a_proj.
    # Out features = (H + 2*Hkv) * Dh = (64 + 8) * 64 = 4608.
    # config.use_qkv_bias is false, so no bias.
    qkv = self.query_key_value(hidden_states)                 # (B, S, 4608)
    qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
    #                                                           (B, S, 72, 64)

    # Split along the HEAD axis (dim=-2), not the feature axis: 64 | 4 | 4.
    query_states, key_states, value_states = qkv.split(
        [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2)
    # query_states                                              (B, S, 64, 64)
    # key_states                                                (B, S,  4, 64)
    # value_states                                              (B, S,  4, 64)

    query_states = query_states.transpose(1, 2).contiguous()  # (B, 64, S, 64)
    key_states = key_states.transpose(1, 2).contiguous()      # (B,  4, S, 64)
    value_states = value_states.transpose(1, 2).contiguous()  # (B,  4, S, 64)

    # ---- 2. QK-NORM ---------------------------------------------------------
    # config.use_qk_norm is true. RMSNorm over the last axis (Dh=64), so it
    # normalises each head independently, with a learned 64-d gain shared by all
    # heads. Applied BEFORE RoPE -- rotation is norm-preserving, so the order
    # matters only for where the learned gain lands. This is the training
    # stability trick that lets attention logits stay bounded without clipping.
    if self.config.use_qk_norm:
        query_states = self.query_layernorm(query_states)     # (B, 64, S, 64)
        key_states = self.key_layernorm(key_states)           # (B,  4, S, 64)

    # ---- 3. ROPE ------------------------------------------------------------
    # cos/sin arrive precomputed from the model level (SarvamMoERotaryEmbedding),
    # not built per-layer as in the 105B.                    cos/sin (B, S, 64)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    # unchanged shapes; cos/sin unsqueeze at dim 1 and broadcast over heads,
    # which is how 4 KV heads and 64 Q heads share one table.
    #
    # Two things to know about this RoPE:
    #  * rope_theta = 8e6 and rope_scaling = null. No YaRN, no interpolation --
    #    the long context is trained in natively via the huge base frequency.
    #    (The 105B does the opposite: theta 1e4 + YaRN factor 40.)
    #  * apply_rotary_pos_emb supports partial rotary (it splits q into q_rot /
    #    q_pass at rotary_dim = cos.shape[-1]), and __init__:388 computes a
    #    `self.rope_dim` from a `partial_rotary_factor`. Both are inert here:
    #    the config has no partial_rotary_factor, rope_dim is never read, and
    #    compute_default_rope_parameters builds inv_freq over the full head_dim,
    #    so rotary_dim == 64 == Dh and q_pass is empty. Dead plumbing.

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs)
        # key_states                                            (B, 4, T, 64)
        # value_states                                          (B, 4, T, 64)
        # Only 4 heads are cached -- this is the whole point of GQA, and unlike
        # the 105B file, the shipped code really does get the saving.

    # ---- 4a. vLLM FAST PATH -------------------------------------------------
    # Vendor branch inside the reference model, unusual for an HF file. vLLM
    # sets config._attn_implementation = "vllm"; the kernel handles the KV-head
    # broadcast itself, so repeat_kv is skipped entirely and it consumes
    # (B,4,T,64) directly. Uses self.scaling (= Dh**-0.5, set in __init__).
    if self.config._attn_implementation == "vllm":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, **kwargs)
        # The rank-juggling that follows exists because the vLLM backend may
        # hand back (B,H,L,Dh), (B,L,hidden) or a flattened (B*L,hidden).
        ...  # normalise to (B, S, 4096)
        attn_output = self.dense(attn_output)                 # (B, S, 4096)
        return attn_output, attn_weights, past_key_value

    # ---- 4b. EAGER PATH -----------------------------------------------------
    # Materialise the 16x broadcast so a plain matmul works.
    key_states = repeat_kv(key_states, self.num_key_value_groups)    # (B, 64, T, 64)
    value_states = repeat_kv(value_states, self.num_key_value_groups)  # (B, 64, T, 64)

    # Note: divides by sqrt(head_dim) inline rather than using self.scaling,
    # which was computed in __init__ as the identical Dh**-0.5. Harmless
    # duplication -- but it means the eager and vLLM paths get their scale from
    # two different places, so a change to one would silently skip the other.
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    #     (B,64,S,64) @ (B,64,64,T)                      ->     (B, 64, S, T)
    #  scale = 64**-0.5 = 0.125, flat. No mscale correction, because no YaRN.

    kv_seq_len = key_states.shape[-2]                         # == T
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask          # mask (B, 1, S, T)
        # Mask is optional here; the 105B asserts it is not None.

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    # fp32 softmax, cast back                                   (B, 64, S, T)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

    attn_output = torch.matmul(attn_weights, value_states)
    #     (B,64,S,T) @ (B,64,T,64)                       ->     (B, 64, S, 64)

    # ---- 5. MERGE HEADS -----------------------------------------------------
    attn_output = attn_output.transpose(1, 2).contiguous()    # (B, S, 64, 64)
    attn_output = attn_output.reshape(bsz, q_len, -1)         # (B, S, 4096)
    # H * Dh == hidden_size here, so `dense` is square 4096->4096.
    # (In the 105B, o_proj is 8192->4096 because v_head_dim != head_dim.)
    attn_output = self.dense(attn_output)                     # (B, S, 4096)

    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value


# =============================================================================
# KV cache cost, side by side
# =============================================================================
# Values cached per token, per layer:
#
#   30B  GQA          2 * Hkv * Dh          = 2 * 4 * 64        =    512
#   105B MLA (real)   kv_lora_rank + Rk     = 512 + 64          =    576
#   105B MLA (as shipped in the HF file, which caches expanded K/V)
#                     H * (Qd + Vd)         = 64 * (192 + 128)  = 20480
#
# Whole model, per token (x num_hidden_layers):
#
#   30B   512 x 19 layers   =   9,728
#   105B  576 x 32 layers   =  18,432        (with absorbed kernels)
#
# So per token of context the 105B costs under 2x the 30B's cache despite being
# 3.3x the parameters -- that is the trade MLA buys, and why the 105B is the one
# given the 128K window while the 30B is evaluated at 65K. Two different answers
# to the same problem: GQA drops head count, MLA drops rank.
