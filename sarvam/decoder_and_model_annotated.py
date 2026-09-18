"""
Sarvam 30B / 105B  ::  decoder layer, model loop and LM head  -- annotated.

Extracts (code verbatim, comments added) from:
    sarvam-105b/modeling_sarvam_moe.py:645-699   SarvamMLADecoderLayer
    sarvam-105b/modeling_sarvam_moe.py:700-720   SarvamMLAPreTrainedModel
    sarvam-105b/modeling_sarvam_moe.py:721-842   SarvamMLAModel
    sarvam-105b/modeling_sarvam_moe.py:843-992   SarvamMLAForCausalLM
    sarvam-30b/modeling_sarvam_moe.py:718-772    SarvamMoEDecoderLayer
    sarvam-30b/modeling_sarvam_moe.py:773-795    SarvamMoEPreTrainedModel
    sarvam-30b/modeling_sarvam_moe.py:796-940    SarvamMoEModel
    sarvam-30b/modeling_sarvam_moe.py:941-1024   SarvamMoEForCausalLM
Fifth and last companion to mla_/gqa_/moe_routing_/yarn_rotary_annotated.py.

    B = bsz    S = q_len    T = total keys    D = hidden_size = 4096
    V = vocab_size = 262144    L = num_hidden_layers   105B: 32   30B: 19

The blocks themselves are textbook pre-norm transformer -- there is nothing
novel here and that is worth saying plainly. What IS worth reading is the
divergence: the 30B file is written against modern Transformers conventions and
the 105B file is not, and that gap is where the practical traps are.
"""


# =============================================================================
# PART 1 -- THE DECODER LAYER (identical in both, bar the MoE return)
# =============================================================================

class SarvamMLADecoderLayer(nn.Module):                      # 105B, line 645
    def __init__(self, config, layer_idx):
        self.self_attn = SarvamMLAAttention(config=config, layer_idx=layer_idx)

        # Dense-vs-MoE choice. first_k_dense_replace=1 in both models, so layer
        # 0 is a plain wide MLP (105B: intermediate 16384, 30B: 8192) and layers
        # 1..L-1 are MoE. moe_layer_freq defaults to 1 -> every layer after the
        # first. See moe_routing_annotated.py PART 3 for why layer 0 is dense.
        use_moe = (
            hasattr(config, "num_experts")
            and config.num_experts is not None
            and layer_idx >= getattr(config, "first_k_dense_replace", 0)
            and layer_idx % getattr(config, "moe_layer_freq", 1) == 0
        )
        self.mlp = SarvamMLAMoE(config) if use_moe else SarvamMLAMLP(config)
        self.input_layernorm = SarvamMLARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = SarvamMLARMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, ...):
        # Pre-norm, two residual blocks. Llama's shape exactly.
        residual = hidden_states                                # (B, S, 4096)
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(...)
        hidden_states = residual + hidden_states                # (B, S, 4096)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)                 # MoE or dense
        hidden_states = residual + hidden_states                # (B, S, 4096)

        outputs = (hidden_states,)
        if output_attentions: outputs += (self_attn_weights,)
        if use_cache:         outputs += (present_key_value,)
        return outputs
        # Positional tuple, so downstream indexing is order-dependent -- see the
        # layer_outputs[2 if output_attentions else 1] dance in the model loop.


# --- 30B deltas (SarvamMoEDecoderLayer, line 718) ---------------------------
#
#   self.attention = ATTENTION_CLASSES[config._attn_implementation](...)
#       Dispatch table at line 710: eager / flash_attention_2 / sdpa / vllm.
#       The 105B hardcodes one eager class -- see PART 4 note 1.
#
#   The MoE block returns (hidden, router_info), so the layer unpacks:
#       if isinstance(hidden_states, tuple):
#           hidden_states, router_logits = hidden_states
#       else:
#           router_logits = None                 # the dense layer 0
#       hidden_states = residual + hidden_states.to(residual.device)
#                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#       That .to() exists for naive pipeline sharding, where an expert can sit
#       on a different device than the residual stream.
#
#   It also takes `position_embeddings` as an argument -- the 30B computes RoPE
#   ONCE at model level and threads it down. The 105B rebuilds it per layer,
#   inside each attention module. See PART 4 note 2.


# =============================================================================
# PART 2 -- THE MODEL LOOP
# =============================================================================

class SarvamMLAModel(SarvamMLAPreTrainedModel):              # 105B, line 721
    def __init__(self, config):
        self.embed_tokens = nn.Embedding(V, D, self.padding_idx)   # (262144, 4096)
        self.layers = nn.ModuleList([SarvamMLADecoderLayer(config, i) for i in range(L)])
        self._use_flash_attention_2 = False    # "Not implemented yet"
        self.norm = SarvamMLARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(self, input_ids, ...):
        # input_ids                                              (B, S)
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = _get_usable_past_kv_length(past_key_values, seq_length)

        if position_ids is None:
            position_ids = torch.arange(past_key_values_length,
                                        seq_length + past_key_values_length, ...)
            position_ids = position_ids.unsqueeze(0)               # (1, S)

        inputs_embeds = self.embed_tokens(input_ids)               # (B, S, 4096)

        # ALWAYS materialises a dense float mask -- no FA2/SDPA path here.
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length)
        #                                                          (B, 1, S, T)

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:                          # 32 layers
            layer_outputs = decoder_layer(hidden_states, attention_mask=attention_mask,
                                          position_ids=position_ids,
                                          past_key_value=past_key_values, ...)
            hidden_states = layer_outputs[0]                       # (B, S, 4096)
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            # NOTE: no gradient-checkpointing branch anywhere in this loop,
            # despite supports_gradient_checkpointing=True on the parent class
            # and self.gradient_checkpointing being set in __init__. Enabling it
            # is a silent no-op. See PART 4 note 3.

        hidden_states = self.norm(hidden_states)                   # final RMSNorm
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, ...)


# --- 30B deltas (SarvamMoEModel, line 796) ----------------------------------
#
#   self.word_embeddings = nn.Embedding(...)   <- named `word_embeddings`, while
#       the 105B says `embed_tokens`. Different checkpoint key namespaces; any
#       tooling that assumes one name breaks on the other model.
#
#   self.rotary_emb = SarvamMoERotaryEmbedding(config=config)   <- model level
#       position_embeddings = self.rotary_emb(hidden_states, position_ids)
#       then passed into every layer. One table, not 19.
#
#   Three mask paths instead of one:
#       flash_attention_2 -> pass the 2-D mask through, or None if fully dense
#       sdpa              -> _prepare_4d_causal_attention_mask_for_sdpa
#       eager             -> _prepare_4d_causal_attention_mask
#
#   Real gradient checkpointing:
#       if self.gradient_checkpointing and self.training:
#           layer_outputs = self._gradient_checkpointing_func(decoder_layer.__call__, ...)
#       plus the standard use_cache=False warning when both are on.
#
#   Router logits are collected across layers into all_router_logits and
#   returned in SarvamMoEModelOutputWithPast.


# =============================================================================
# PART 3 -- THE LM HEAD
# =============================================================================

class SarvamMLAForCausalLM(SarvamMLAPreTrainedModel):        # 105B, line 843
    _tied_weights_keys = ["lm_head.weight"]     # <- inert; see PART 4 note 5
    def __init__(self, config):
        self.model = SarvamMLAModel(config)
        self.lm_head = nn.Linear(D, V, bias=False)      # 4096 -> 262144, untied

    def forward(self, ..., labels=None):
        outputs = self.model(...)
        hidden_states = outputs[0]                                 # (B, S, 4096)
        logits = self.lm_head(hidden_states)                       # (B, S, 262144)
        logits = logits.float()
        # That .float() is not free: at S=4096 one logits tensor is
        # 4096 * 262144 * 4 bytes = 4.3 GB per batch element. The 262144-token
        # Indic vocab makes the head unusually expensive at both ends -- the two
        # embedding matrices are 2.15 GB of the parameter count on their own.

        if labels is not None:
            # Hand-rolled shift + CrossEntropyLoss. The 30B calls
            # self.loss_function(...) instead, HF's modern pluggable path.
            shift_logits = logits[..., :-1, :].contiguous()        # (B, S-1, V)
            shift_labels = labels[..., 1:].contiguous()            # (B, S-1)
            loss = CrossEntropyLoss()(shift_logits.view(-1, V), shift_labels.view(-1))
        return CausalLMOutputWithPast(loss=loss, logits=logits, ...)

    # prepare_inputs_for_generation (931) and _reorder_cache (987) are the
    # pre-4.50 hand-written versions. _reorder_cache indexes past_key_values as
    # nested tuples, which is the LEGACY cache format -- it will not work on the
    # DynamicCache the model actually builds, so beam search is suspect.


# --- 30B deltas (SarvamMoEForCausalLM, line 941) ----------------------------
#
#   class SarvamMoEForCausalLM(SarvamMoEPreTrainedModel, GenerationMixin)
#                                                       ^^^^^^^^^^^^^^^^ present
#   loss = self.loss_function(logits, labels, self.config.vocab_size, **kwargs)
#   Returns SarvamMoECausalLMOutputWithPast, which carries an `aux_loss` field.
#
#   But: `aux_loss = None` at line 1006 and is never assigned again. The field
#   is plumbed through the dataclass and the tuple-return branch and is always
#   None. Not a bug so much as a vestige -- these models balance experts with
#   the router bias (aux-loss-FREE, see moe_routing_annotated.py), so there is
#   no auxiliary loss to report. The scaffolding outlived the technique.


# =============================================================================
# PART 4 -- WHAT ACTUALLY MATTERS HERE
# =============================================================================
#
# 1. THE 105B FILE IS EAGER-ATTENTION ONLY.
#      _supports_flash_attn_2 = False   # "Not implemented yet"
#      _use_flash_attention_2 = False   # "Not implemented yet"
#      no SDPA subclass, no ATTENTION_CLASSES table
#    So it materialises attn_weights of shape (B, 64, S, T) and a dense
#    (B, 1, S, T) mask. At the advertised 128K context that is
#      64 * 131072^2 * 2 bytes = 2.2 PB per batch element.
#    The 128K window is real, but it belongs to vLLM/SGLang, not to this file.
#    Combined with the KV-cache point in mla_forward_annotated.py: the HF
#    reference implementation is for reading and for short-prompt correctness
#    checks. It is not the thing that serves the model.
#
# 2. PER-LAYER ROPE IN THE 105B. Each SarvamMLAAttention calls _init_rope() and
#    owns a private rotary module; the 30B builds one at model level and passes
#    cos/sin down. Two consequences for the 105B: the 2.15 GB init-time table
#    spike from yarn_rotary_annotated.py note, and 32 redundant rebuilds the
#    first time a longer sequence arrives.
#
# 3. THE 105B CANNOT TRAIN, FOURTH INDEPENDENT REASON. Earlier three were in
#    the MoE (gate assert, @torch.no_grad on moe_infer, no real training
#    branch). Here: supports_gradient_checkpointing=True and
#    self.gradient_checkpointing is initialised, but the model loop never calls
#    _gradient_checkpointing_func. gradient_checkpointing_enable() will report
#    success and do nothing. The 30B implements it properly.
#
# 4. GenerationMixin ASYMMETRY -- CHECK THIS BEFORE CALLING .generate().
#    The 30B declares it explicitly:
#        class SarvamMoEForCausalLM(SarvamMoEPreTrainedModel, GenerationMixin)
#    The 105B does not:
#        class SarvamMLAForCausalLM(SarvamMLAPreTrainedModel)
#    Since Transformers v4.50, PreTrainedModel no longer inherits GenerationMixin,
#    and config.json pins transformers_version 4.57.2. On that version the 105B
#    class would have no .generate() of its own -- which squares with it still
#    carrying hand-written prepare_inputs_for_generation and _reorder_cache,
#    the pre-4.50 pattern. I could not verify this empirically (no transformers
#    installed here), so confirm on your install before concluding:
#        from transformers import PreTrainedModel
#        from transformers.generation import GenerationMixin
#        issubclass(PreTrainedModel, GenerationMixin)
#
# 5. _tied_weights_keys IS MISLEADING IN BOTH. Both declare
#    _tied_weights_keys = ["lm_head.weight"], but both configs set
#    tie_word_embeddings: false, and HF only acts on that attribute when the
#    config flag is true. So the head and the embedding really are two separate
#    262144 x 4096 matrices -- 1.07 GB each, 2.15 GB of the parameter count,
#    which is 6.7% of the 30B's 32.1 B total. Do not read the attribute as
#    evidence of tying.
#
# 6. THE NAMING SPLIT IS THE PRACTICAL HAZARD. Same company, same release,
#    same filename, and yet:
#        105B  embed_tokens / self_attn      SarvamMLA*     eager only
#         30B  word_embeddings / attention   SarvamMoE*     eager+FA2+SDPA+vLLM
#    Nothing that walks one model's state dict or module tree will walk the
#    other's. Treat them as two codebases that happen to ship under one name --
#    which, given the MLA-vs-GQA split underneath, is what they are.
