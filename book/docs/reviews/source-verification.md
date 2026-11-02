# Source verification

Written by `book/tools/verify_source_map.py` against `@773b6a8`.

Source-map rows checked: 94. Matching their line range (±1): **91**. Not matching: **3**.
Test-map entries: 61. At the `def` line given: **61**. Not: **0**.

Rows whose range deliberately starts with leading comments or constants (accepted): Native-GQA switch (+3), Weight init (+6), Persistence (+7), Vocabulary growth (+28,672 audio, +5 control) (+9).

## Rows that don't match

| Component | File | Symbols | Map says | Finding |
|---|---|---|---|---|
| QLoRA, 4-bit base | `lora.py` |  | – | no range: an absent feature or an uncommitted one (documented in the row) |
| DPO, RLHF, GRPO | `bidirectional.py` |  | – | no range: an absent feature or an uncommitted one (documented in the row) |
| Projector norm matching | `omni_train.py` | Omni._match_scale | – | no range: an absent feature or an uncommitted one (documented in the row) |

## Test map entries that don't match

- none

Test functions defined in `test_model.py` at the pin: 69.

## Drift between the pin and the working tree

- `omni_train.py`:  1 file changed, 40 insertions(+), 7 deletions(-)

## Claims recomputed from the code and the released checkpoint

`book/tools/verify_claims.py`, against `@773b6a8` and `ckpt_base_v2.pt`.

| Claim | Holds | Evidence |
|---|---|---|
| model.py has 1,224 lines | yes | 1224 lines at @773b6a8 |
| test_model.py has 66 tests | yes | 66 functions carry a test decorator (test) at @773b6a8 |
| released model: 397.7M parameters | yes | num_params() total 397,659,648; state dict 397,659,648 |
| about 173M active per token | yes | num_params() active 173,526,528 |
| 32k vocabulary | yes | vocab_size 32,768 |
| RoPE θ = 1e6 | yes | rope_theta 1e+06 |
| GQA 16:4 with QK-norm | yes | n_head 16, n_kv_head 4, use_qk_norm True, attention None |
| 24 experts, top-4 | yes | num_experts 24, top-k 4, shared 0 |
| sliding window on half the layers (0–9 of 20) | yes | sliding_window 256, max_window_layers 10, n_layer 20 |
| 20 KiB of KV cache per token (bf16) | yes | 20 layers × K,V × 4 heads × 64 × 2 bytes = 20,480 bytes |
| embedding + head 67.1M | yes | 67,108,864 |
| attention 52.4M | yes | 52,428,800 |
| dense layer 0: 8.7M | yes | 8,650,752 (intermediate_size 2816) |
| experts 269.0M | yes | 268,959,744 (moe_intermediate_size 192) |
| untied head | yes | tie_word_embeddings False |
| nano_30b is the 17M preset | yes | total 17,440,384, active 6,602,368 |
| router in fp32 because bf16 flipped 3.2% of choices (lines 580–586) | yes | # fp32 router: top-k is discrete, and bf16 rounding would flip choices.; # Casting the inputs is not enough -- autocast re-casts F.linear to; # bf16 whatever dtype it is handed, and every GPU run before this fix; # route |
| atomic save exists because a kill at step 265,000 destroyed a checkpoint (train.py 32–46) | yes | 265,000 of the coder run on 2026-09-12 and cost the optimizer state. |
| api.py accepts top_p and ignores it | yes | top_p on lines [188, 207]; check that none reaches generate() |
| dropout wired in but 0.0 by default | yes | dropout: float = 0.0 |
| the book's own code examples (planned for Stage 4) | yes | 6 planned; not written yet: arithmetic_400m.py, by_hand.py, history_positions.py, layernorm_vs_rmsnorm.py, top_p_patch.py |
| every file docs/authoring/book-plan.md names exists (repo at the pin, or book/) | yes | 40 named; missing: none |
| every test docs/authoring/book-plan.md names exists in test_model.py | yes | 44 named; missing: none |
