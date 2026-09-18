"""
Sarvam 30B / 105B  ::  MoE block and expert routing  -- annotated with tensor shapes.

Extracts (code verbatim, comments added) from:
    sarvam-105b/modeling_sarvam_moe.py:267-330   MoEGate
    sarvam-105b/modeling_sarvam_moe.py:331-458   SarvamMLAMoE + moe_infer
    sarvam-30b/modeling_sarvam_moe.py:196-247    SarvamMoEGate
    sarvam-30b/modeling_sarvam_moe.py:248-319    SarvamMoEExperts
    sarvam-30b/modeling_sarvam_moe.py:320-362    SarvamMoESparseMoeBlock
Third companion to mla_forward_annotated.py / gqa_forward_annotated.py.

Shape symbols:

    B    = bsz
    S    = seq_len
    N    = B * S              flattened token count -- the MoE works token-wise,
                              batch and sequence are irrelevant to it
    D    = hidden_size        = 4096 in both models
    E    = num_experts        = 128  in both models
    k    = num_experts_per_tok    105B: 8      30B: 6
    Emid = moe_intermediate_size  105B: 2048   30B: 1024

The two gates are the same algorithm (DeepSeek-V3 `noaux_tc`), but they are
configured to behave completely differently. See PART 1 and the closing note.
"""


# =============================================================================
# PART 1 -- THE GATE.  105B: MoEGate.forward (line 293)
# =============================================================================

def forward(self, hidden_states):
    bsz, seq_len, h = hidden_states.shape
    hidden_states = hidden_states.view(-1, h)                  # (N, 4096)

    # Router is a single bias-free linear, forced to fp32 for both operand and
    # weight. Routing decisions are discrete (a topk), so bf16 rounding could
    # flip which expert a token lands on and make the forward non-reproducible.
    logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32), None)
    #                                                           (N, 128)

    # SIGMOID, not softmax. Each expert is scored independently, so scores do
    # not compete to sum to 1 and one expert's logit rising cannot depress the
    # rest. That decoupling is what makes the bias trick below work.
    if self.scoring_func == "sigmoid":
        scores = logits.sigmoid()                              # (N, 128) in (0,1)
    else:
        raise NotImplementedError(...)

    if self.topk_method == "noaux_tc":
        assert not self.training      # <-- inference only; see closing note

        # AUX-LOSS-FREE LOAD BALANCING (the "noaux" in noaux_tc).
        # e_score_correction_bias is a per-expert scalar, updated OUT of band
        # during training: overloaded experts get their bias nudged down,
        # starved ones up. It steers SELECTION only -- note that `scores`, not
        # `scores_for_choice`, is what gets gathered as the weight further down.
        # So balancing never leaks into the output magnitudes, which is the
        # thing an auxiliary load-balancing loss does badly.
        scores_for_choice = scores.view(bsz * seq_len, -1) + self.e_score_correction_bias.unsqueeze(0)
        #                                                       (N, 128)

        # GROUP-LIMITED ROUTING. Experts are partitioned into n_group blocks;
        # each group is scored by the sum of its top-2 experts, the best
        # topk_group groups survive, everything else is masked to -inf.
        # Purpose is device placement: with experts sharded across nodes, an
        # unconstrained top-k could touch every node for a single token.
        # Capping the group count caps the all-to-all fan-out.
        group_scores = (
            scores_for_choice.view(bsz * seq_len, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        )                                                      # (N, n_group)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        #                                                       (N, topk_group)
        group_mask = torch.zeros_like(group_scores)            # (N, n_group)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(bsz * seq_len, -1)
        )                                                      # (N, 128) 0/1
        tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
        #                                                       (N, 8) int64

        # Gather from the UNBIASED scores -- the bias is a routing thumb on the
        # scale, never a contribution to the combine weights.
        topk_weight = scores.gather(1, topk_idx)               # (N, 8)

    # Renormalise the k selected sigmoid scores to sum to 1, then rescale.
    # routed_scaling_factor = 2.5 in both models: after normalisation the
    # routed branch would have unit gain, which is small next to the always-on
    # shared expert, so it is scaled back up. A fixed constant, not learned.
    if self.top_k > 1 and self.norm_topk_prob:
        denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
        topk_weight = topk_weight / denominator                # (N, 8) sums to 1
    topk_weight = topk_weight * self.routed_scaling_factor     # (N, 8) sums to 2.5

    return topk_idx, topk_weight
    #      (N, 8)    (N, 8)


# --- 30B deltas (SarvamMoEGate, line 236) -----------------------------------
# Same algorithm, four differences worth knowing:
#
#  1. n_group / topk_group are REAL CONFIG FIELDS here, both set to 1, so
#     group_limited_topk degenerates to a plain unconstrained top-6 over all
#     128 experts. In the 105B they are absent from config.json entirely and
#     fall through to getattr defaults -- see the closing note, this is the
#     single most consequential difference between the two routers.
#  2. expert_bias is an nn.Parameter with requires_grad=False, not a buffer.
#     The comment in the source says why: "vllm complains about it."
#  3. No `assert not self.training` -- the 30B router runs in training mode.
#  4. It also returns `logits`, so the block can surface router logits for
#     aux-loss or load-monitoring. The 105B gate drops them.


# =============================================================================
# PART 2 -- DISPATCH.  105B: moe_infer (line 390)
# =============================================================================
# The problem: 128 experts are 128 different weight matrices, and each token
# wants 8 of them. Looping tokens is hopeless. So sort tokens BY EXPERT and run
# one dense matmul per expert over its contiguous slice.

@torch.no_grad()                     # <-- see closing note
def moe_infer(self, x, topk_ids, topk_weight):
    # x (N, 4096)   topk_ids (N, 8)   topk_weight (N, 8)

    # How many tokens each expert must handle.
    cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))   # (N, 128)
    cnts.scatter_(1, topk_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)                        # (128,)

    # THE KEY LINE. argsort over the flattened (N*8,) assignment list groups all
    # work for expert 0 first, then expert 1, and so on.
    idxs = topk_ids.view(-1).argsort()                         # (N*8,)
    # Integer-divide by k to turn an assignment index back into a token index,
    # then gather. Each token now appears 8 times, once per chosen expert.
    sorted_tokens = x[idxs // topk_ids.shape[1]]               # (N*8, 4096)

    if self.ep_size > 1:
        ...  # expert parallelism: dist.all_to_all the token blocks to the rank
             # owning each expert, run locally, all_to_all back. This is the
             # branch group-limited routing exists to bound.

    tokens_per_expert = tokens_per_expert.cpu().numpy()   # forces a GPU sync

    # One dense GEMM per expert over its contiguous slice. Sequential, so on a
    # single device this is 128 launches -- correct, but the reason production
    # serving uses fused grouped-GEMM kernels instead of this file.
    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue                # NB: start_idx is not advanced here, which
                                    # is fine only because the slice was empty
        expert = self.experts[i + self.ep_rank * self.experts_per_rank]
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]   # (n_i, 4096)
        outputs.append(expert(tokens_for_this_expert))              # (n_i, 4096)
        start_idx = end_idx

    outs = torch.cat(outputs, dim=0)                           # (N*8, 4096)

    # UNSORT: scatter back to assignment order, reshape so the k axis reappears,
    # scale each copy by its gate weight, and sum the k contributions per token.
    new_x = torch.empty_like(outs)
    new_x[idxs] = outs                                         # (N*8, 4096)
    final_out = (
        new_x.view(*topk_ids.shape, -1)                        # (N, 8, 4096)
        .type(topk_weight.dtype)
        .mul_(topk_weight.unsqueeze(dim=-1))                   # x (N, 8, 1)
        .sum(dim=1)                                            # (N, 4096)
        .type(new_x.dtype)
    )
    return final_out                                           # (N, 4096)


# --- 30B deltas (SarvamMoEExperts.forward, line 259) ------------------------
# Inference path is the same sort/GEMM/unsort. It adds a real TRAINING path:
#
#     x = hidden_states.repeat_interleave(k, dim=0)      # (N*6, 4096)
#     y = torch.empty_like(x)
#     for i, expert in enumerate(self):
#         mask = flat_topk_idx == i
#         if mask.any():
#             y[mask] = expert(x[mask])                  # boolean-mask gather
#     y = (y.view(*top_k_weights.shape, -1) * top_k_weights.unsqueeze(-1)).sum(dim=1)
#
# Boolean masking instead of argsort, so it stays differentiable and no
# @torch.no_grad() is in the way. Memory-hungry (materialises N*6 copies of the
# hidden state up front) but correct.


# =============================================================================
# PART 3 -- BLOCK WIRING AND THE SHARED EXPERT
# =============================================================================
# Both blocks are the same shape:
#
#     def forward(self, hidden_states):                  # (B, S, 4096)
#         identity = hidden_states
#         topk_idx, topk_weight = self.gate(hidden_states)
#         y = <dispatch>(flat_hidden, topk_idx, topk_weight).view(B, S, 4096)
#         if self.shared_experts is not None:
#             y = y + self.shared_experts(identity)      # DENSE, every token
#         return y
#
# num_shared_experts = 1 in both. The shared expert is an ordinary MLP outside
# the routing entirely -- always on, sees every token, and its output is ADDED
# to the routed sum rather than being one of the k. It absorbs whatever is
# common to all tokens, freeing the 128 routed experts to specialise instead of
# each having to relearn the same general-purpose transform.
#
# Its width is moe_intermediate_size * num_shared_experts:
#     105B  2048 * 1 = 2048          30B  1024 * 1 = 1024
# (The 30B's config.json also carries moe_shared_expert_intermediate_size=1024,
# which no code path reads. Same value, so inert -- but do not trust it.)
#
# Layer 0 is not an MoE layer at all. Both decoder layers pick with
# `layer_idx >= first_k_dense_replace` (105B:654, 30B:725), and
# first_k_dense_replace = 1, so the first block is a plain dense MLP of width
# intermediate_size (105B 16384, 30B 8192). Standard practice: the first layer's
# representations are not yet differentiated enough for routing to mean much,
# and an early bad router poisons everything downstream.


# =============================================================================
# CLOSING NOTES
# =============================================================================
#
# 1. THE n_group TRAP.  Same gate code, opposite behaviour:
#
#      30B   n_group=1,  topk_group=1   (explicit in config.json AND in the
#            config class) -> grouping is a no-op, plain top-6 of 128.
#
#      105B  n_group and topk_group appear NOWHERE -- not in config.json, not
#            in SarvamMLAConfig.__init__. The gate reads them via
#              getattr(config, "n_group", self.n_routed_experts // 8)  -> 16
#              getattr(config, "topk_group", 2)                        ->  2
#            so the shipped model routes into 16 groups of 8, keeps the best 2
#            groups = 16 candidate experts, then takes top-8 of those.
#
#    Read that again: in the 105B, all 8 experts for a token must come from just
#    2 of the 16 groups. That is a hard constraint on what the router can
#    express, and it is load-bearing for how the model was trained -- yet it
#    exists only as a getattr default. Add an `n_group` field to that config for
#    any other purpose and you silently change the model's routing.
#
# 2. THE 105B FILE CANNOT TRAIN.  Three independent blocks:
#      - MoEGate.forward line 305:  `assert not self.training`
#      - moe_infer is decorated     `@torch.no_grad()`
#      - SarvamMLAMoE.forward's if/else picks moe_infer on BOTH branches, under
#        a comment conceding "in practice, you'd want a more sophisticated
#        training implementation"
#    It is an inference reference. The 30B file, by contrast, has a working
#    training path. Do not fine-tune from the 105B modelling file as shipped.
#
# 3. THE PARAMETER MATH CHECKS OUT.  Expert MLP = 3 * D * Emid (SwiGLU: gate,
#    up, down), 18 or 31 MoE layers plus one dense layer, attention as counted
#    in the companion files, untied embeddings 2 * 262144 * 4096:
#
#                  total      active (non-embedding)     published
#      105B      106.0 B            10.25 B            106B / 10.3B active
#       30B       32.1 B             2.36 B             32B / 2.4B active
#
#    So "30B" and "105B" are round-number product names; the configs give 32B
#    and 106B. Per token only 9 of 129 expert MLPs fire in the 105B (8 routed +
#    1 shared), 7 of 129 in the 30B -- that ratio is the whole economic argument
#    for the architecture.
