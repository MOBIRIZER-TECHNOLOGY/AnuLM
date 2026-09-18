"""
Sarvam-105B  ::  SarvamMLAAttention.forward  -- annotated with tensor shapes.

Extract of sarvam-105b/modeling_sarvam_moe.py:558-644 (code verbatim, comments added).
Not runnable on its own; read alongside the original.

Shape symbols, with sarvam-105b/config.json values substituted:

    B    = bsz                     batch
    S    = q_len                   new tokens this step (S == 1 when decoding)
    T    = kv_seq_len              total keys attended = S + cached
    H    = num_attention_heads     = 64
    D    = hidden_size             = 4096
    Lkv  = kv_lora_rank            = 512     <- the compressed KV latent
    Rk   = qk_rope_head_dim        = 64      <- positional half of the key
    Nk   = qk_nope_head_dim        = 128     <- content half of the key
    Qd   = q_head_dim  = Nk + Rk   = 192
    Vd   = v_head_dim              = 128

Note `head_dim: 576` in config.json is NOT used by this forward. It is the
Lkv + Rk = 512 + 64 figure -- the per-token *cache* width that a true MLA
kernel keeps. See the closing note on the "absorbed" optimisation.
"""


def forward(self, hidden_states, attention_mask=None, position_ids=None,
            past_key_value=None, output_attentions=False, use_cache=False, **kwargs):

    bsz, q_len, _ = hidden_states.size()
    # hidden_states                                              (B, S, 4096)

    # ---- 1. QUERIES ---------------------------------------------------------
    # q_lora_rank is None in the shipped config, so queries take the plain path:
    # one big projection, no low-rank compression. (DeepSeek-V3 compresses Q too;
    # Sarvam ships the uncompressed variant.)
    if self.q_lora_rank is None:
        q = self.q_proj(hidden_states)                        # (B, S, 64*192 = 12288)
    else:
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))

    q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
    #                                                           (B, 64, S, 192)

    # Split each query head into a content part and a positional part.
    # Only q_pe will get RoPE applied; q_nope stays position-free.
    q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    # q_nope                                                    (B, 64, S, 128)
    # q_pe                                                      (B, 64, S,  64)

    # ---- 2. KEYS / VALUES: down-project once, then expand -------------------
    # THE core MLA move. One projection produces both the shared latent and the
    # single positional key. Width 512 + 64 = 576 per token, for ALL 64 heads.
    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)    # (B, S, 576)
    compressed_kv, k_pe = torch.split(
        compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    # compressed_kv                                             (B, S, 512)
    # k_pe                                                      (B, S,  64)

    # k_pe gets head dim 1, not 64: the positional key is MQA-style, shared
    # across every head. That is what the `_with_mqa` in the layer name means,
    # and it is why RoPE can be applied once, before the latent is expanded.
    k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    #                                                           (B, 1, S, 64)

    # Expand the 512-d latent back out to per-head content-keys AND values.
    # 64 * (128 + 128) = 16384 out features.
    kv = (
        self.kv_b_proj(self.kv_a_layernorm(compressed_kv))    # (B, S, 16384)
        .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        .transpose(1, 2)
    )                                                         # (B, 64, S, 256)

    k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    # k_nope                                                    (B, 64, S, 128)
    # value_states                                              (B, 64, S, 128)

    # ---- 3. ROPE ------------------------------------------------------------
    kv_seq_len = value_states.shape[-2]                       # == S (dim -2 of a 4-D tensor)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(...)
        kv_seq_len += _get_usable_past_kv_length(past_key_value, kv_seq_len, self.layer_idx)
        # kv_seq_len is now T = S + cached

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    # YaRN tables, built over the 64-d rope subspace only    cos/sin (T, 64)

    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
    # q_pe                                                      (B, 64, S, 64)
    # k_pe   (head axis stays 1)                                (B,  1, S, 64)

    # ---- 4. REASSEMBLE 192-d Q AND K ---------------------------------------
    # Content half and positional half are concatenated, not summed, so the
    # q.k dot product decomposes into  <q_nope, k_nope> + <q_pe, k_pe>:
    # a position-free content term plus a purely relative positional term.
    query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    query_states[:, :, :, : self.qk_nope_head_dim] = q_nope   # [..., :128]
    query_states[:, :, :, self.qk_nope_head_dim :] = q_pe     # [..., 128:]
    #                                                           (B, 64, S, 192)

    key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
    key_states[:, :, :, self.qk_nope_head_dim :] = k_pe       # (B,1,S,64) broadcasts
    #                                                           into all 64 heads
    #                                                           (B, 64, S, 192)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs)
        # key_states                                            (B, 64, T, 192)
        # value_states                                          (B, 64, T, 128)

    # ---- 5. ATTENTION -------------------------------------------------------
    # softmax_scale = q_head_dim ** -0.5 = 192**-0.5 = 0.07217, then multiplied
    # by mscale**2 back in __init__ (line 522) when YaRN is on:
    #   mscale = 0.1 * mscale_all_dim * ln(factor) + 1 = 0.1*ln(40)+1 = 1.36889
    #   softmax_scale = 0.07217 * 1.36889**2 = 0.13523   (1.874x the naive value)
    # The extra gain compensates for attention entropy rising as context is
    # stretched 40x; it is part of YaRN, not an ad-hoc stability hack.
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
    #     (B,64,S,192) @ (B,64,192,T)                    ->     (B, 64, S, T)

    assert attention_mask is not None           # this path requires an explicit mask
    attn_weights = attn_weights + attention_mask  # mask        (B, 1, S, T)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    # fp32 softmax, cast back                                   (B, 64, S, T)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

    attn_output = torch.matmul(attn_weights, value_states)
    #     (B,64,S,T) @ (B,64,T,128)                      ->     (B, 64, S, 128)
    #  NB values are 128-d while keys are 192-d -- asymmetric, unlike vanilla MHA.

    # ---- 6. MERGE HEADS -----------------------------------------------------
    attn_output = attn_output.transpose(1, 2).contiguous()    # (B, S, 64, 128)
    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
    #                                                           (B, S, 8192)
    attn_output = self.o_proj(attn_output)                    # (B, S, 4096)

    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value


# =============================================================================
# What this file does NOT do: the "absorbed" MLA optimisation
# =============================================================================
# Step 2 expands the 512-d latent into full per-head keys/values, and step 4
# caches THOSE. Per token, per layer, this reference implementation keeps
#
#     64 * (192 + 128) = 20480 values
#
# whereas the point of MLA is to cache only the latent plus the shared rope key:
#
#     512 + 64 = 576 values          (35.6x smaller -- and the config's head_dim)
#
# Production kernels (vLLM, SGLang) get there by folding kv_b_proj into the
# neighbouring matrices instead of materialising kv:
#   * absorb kv_b_proj's key half into q_proj, so q_nope is projected down into
#     the 512-d latent space and dotted directly against compressed_kv;
#   * absorb kv_b_proj's value half into o_proj, so attention outputs the latent
#     and the up-projection happens once, after the softmax.
# Mathematically identical -- the same two matmuls, reassociated. So read this
# file for the algebra, but do not read its memory behaviour as MLA's memory
# behaviour.
