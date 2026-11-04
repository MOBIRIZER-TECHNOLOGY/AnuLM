# Repository source map

The table every chapter cites from. Line numbers are taken at `@b42998d`, the
tip of the `restructure` branch (the `anulm/` package layout), from the root
copy of the repository.

The pin is on [GitHub](https://github.com/MOBIRIZER-TECHNOLOGY/AnuLM), on the
branch `port-bundle-fixes`. The default branch, `main` (`ae59238`), still has
the old flat layout, whose line numbers differ. `tools/build_preview.py` links
every citation whose file is identical at this commit:

Public commit: `b42998d817806f4502a54ac5ed4d82c91821b023`

Re-pinned from `@773b6a8` on 2026-11-03, and from `@fdbb09a` on 2026-11-04,
by `tools/repin.py`; every citation the last re-pin moved, with the old and
new code line, is in `technical-review/repin-report.md`.
`.claude/worktrees/api-demo/` holds a second copy whose numbers differ, so
**never cite from that copy**.

Status values:

- **implemented**: the code exists and is wired into a runnable path.
- **partial**: it exists but defaults off, or has a gap the note names.
- **external**: the code delegates to a third-party library.
- **absent**: nothing in the repository implements it.

The "verified" column shows how each row was checked:

- **read**: the author read the code.
- **agent**: an inventory agent read the code, and spot-checks passed (8 of 8).
- **pending**: not yet checked.

**Stage 3 (2026-10-31):** every row with a file, a symbol and a range was
machine-checked against the code at the pin by `tools/verify_source_map.py`
(the symbol's span, from Python's own parser, must fall in the row's range), and
the arithmetic and narrative claims about the released model were recomputed by
`tools/verify_claims.py`. Results and the remaining documented deviations are in
`technical-review/source-verification.md`. This applies to "agent" rows too.

---

## 1. Core model: `anulm/model.py`

| Component | Symbol | Lines | Status | Concepts | Verified |
|---|---|---|---|---|---|
| Configuration and presets | `AnuLMConfig`, `nano_30b`, `nano_350m`, `nano_105b` | 39–218 | implemented | hyperparameters, GQA/MLA switch, MoE sizes | read |
| Config validation | `AnuLMConfig.__post_init__` | 126–156 | implemented | top-k reachable through groups | read |
| RMSNorm | `RMSNorm` | 225–239 | implemented | computed in fp32, learned gain, no mean subtraction | read |
| SwiGLU MLP / expert body | `SwiGLU` | 242–253 | implemented | gate · up, SiLU, 3·D·I parameters | read |
| YaRN helpers | `_yarn_correction_dim`, `_yarn_correction_range`, `yarn_mscale` | 260–275 | implemented | NTK-by-parts, temperature | read |
| Rotary tables | `Rotary` (`_build_inv_freq` 298–315, `_grow` 317–330, `forward` 332–336) | 278–343 | implemented | θ_i = base^(−2i/d), cached cos/sin, absolute `start` | read |
| Rotation | `rotate_half`, `apply_rope` | 346–362 | implemented | split-halves (NeoX) convention | read |
| Causal and window mask | `_attn_mask` | 365–381 | implemented | causal offset for the cache, sliding window | read |
| Attention kernel call | `_sdpa` | 414–418 | external (PyTorch SDPA) | softmax(QKᵀ·scale + mask)V | read |
| Native-GQA switch | `NATIVE_GQA` | 387–387 | partial (off by default) | `enable_gqa` | read |
| Merge-tree tables | `merge_tree_matrix` | 390–411 | partial (off unless `merge_tree`) | sparse reparameterisation from BPE merges | read |
| GQA attention | `GQAttention.__init__`, `GQAttention.forward` | 430–494 | implemented | fused QKV, QK-norm, RoPE, KV cache, repeat_interleave | read |
| MLA attention | `MLAttention` | 501–578 | implemented | latent KV `kv_lora_rank`, decoupled rope key, cache = latent | read |
| Router | `Router` (`forward` 611–622, `_gate` 624–672) | 585–672 | implemented | sigmoid scores, bias for selection only, group-limited top-k, seq-balance loss, z-loss | read |
| MoE layer | `MoE.__init__`, `MoE.forward` | 681–707 | implemented | routed experts, plus a shared expert when `num_shared_experts > 0`; the released model sets it to 0 | read |
| MoE dispatch paths | `_route_sparse` / `_route_dense` / `_route_grouped` | 709–790 | implemented | gather-scatter, one-hot, `F.grouped_mm` | read |
| Transformer block | `Block` | 797–824 | implemented | pre-norm residual; layer 0 dense, rest MoE; per-layer window | read |
| Model | `AnuLM.__init__` | 833–863 | implemented | embedding, shared rotary, stack, final norm, untied head, init | read |
| Weight init | `AnuLM._init_weights` + scaled residual init | 865–871 | implemented | N(0, 0.02); output projections ÷ √(2L) | read |
| KV cache container | `AnuLM.new_cache` | 873–875 | implemented | one dict per layer | read |
| Forward pass and loss | `AnuLM.forward` | 877–928 | implemented | embeds bypass, grad checkpointing, last-position shortcut, CE with `ignore_index=-1` | read |
| Chunked loss | `enable_chunked_loss`, `_chunked_loss`, `_ce_sum` | 930–956 | partial (opt-in) | never materialises B·S·V logits | read |
| Token tables | `_compose`, `_table`, `train`, `_head`, `bake_merge_tree` | 959–1013 | partial | merge-tree; eval cache | read |
| Decoding from embeddings | `AnuLM.decode_from_embeds` | 1015–1062 | implemented | greedy, n-gram block, repetition penalty | read |
| Gradient checkpointing | `enable_gradient_checkpointing` | 1064–1068 | implemented | activation recomputation | read |
| Aux-loss-free balancing | `AnuLM.update_expert_biases` | 1071–1101 | implemented | sign rule on the bias, imbalance ratio | read |
| Parameter count | `AnuLM.num_params` | 1104–1113 | implemented | total vs active | read |
| Generation | `AnuLM.generate` | 1115–1180 | implemented | prefill/decode, context slide, repetition penalty, temperature, top-k, EOS, streaming | read |
| Checkpoint loading | `load_checkpoint` | 1280–1342 | implemented | `.pt`, exported folder, Hub id, adapters | read |
| Memory guard | `fit_memory` | 1345–1364 | implemented | auto checkpointing, chunked CE | read |
| Smoke test | `__main__` | 1183–1208 | implemented | loss ≈ ln V, every parameter gets a gradient | read |

## 2. Tokenizer: `anulm/bpe.py`

| Component | Symbol | Lines | Status | Verified |
|---|---|---|---|---|
| Pre-tokenisation regex, document separator | `CHUNK_RE`, `DOC_SEP` | 39–45 | implemented | read |
| Vocabulary and ranks, EOS above the merges | `BPE.__init__`, `vocab_size` | 64–83 | implemented | read |
| Training (pair counts, pair→chunk index, lazy max-heap) | `BPE.train` | 87–162 | implemented | read |
| Encoding (lowest-rank merge first, memoised) | `_encode_chunk`, `encode`, `encode_documents` | 166–206 | implemented | read |
| Decoding | `decode` | 208–212 | implemented | read |
| Persistence | `save`, `load` | 221–243 | implemented | read |
| Parallel corpus encoding into int16 | `iter_documents`, `cmd_encode` | 259–344 | implemented | read |
| Statistics (bytes/token, round trip) | `_stats`, `cmd_stats` | 347–364 | implemented | read |

The byte-level path, which needs no tokenizer, lives in `anulm/training/train.py`:
`load_bytes` (lines 62–98) and `BYTE_EOS = 257` (line 101).

## 3. Training: `anulm/training/train.py`, `anulm/muon.py`, `anulm/bigram_teacher.py`

| Component | Symbol | Lines | Status | Verified |
|---|---|---|---|---|
| Crash-safe save | `atomic_save` | 32–46 | implemented | read |
| Window sampling without replacement | `WindowSampler` | 104–160 | implemented | read |
| Memory-mapped corpus | `load_memmap` | 163–173 | implemented | read |
| Interleaved validation split | `interleaved_split` | 202–216 | partial (flag) | read |
| Spill warning | `warn_if_spilling` | 219–234 | implemented | read |
| Batches (y = x shifted by one) | `make_batch` | 237–250 | implemented | read |
| Fixed eval windows / strided windows | `eval_windows`, `strided_windows` | 338–372 | implemented | read |
| LR schedule (warmup, cosine or WSD, rewarmup) | `lr_at` | 375–400 | implemented | read |
| Multi-optimizer handle | `OptimizerSet` | 403–444 | implemented | read |
| AdamW parameter groups (decay on 2-D only) | `make_optimizer` | 447–474 | implemented | read |
| Evaluation with standard error | `evaluate` | 477–533 | implemented | read |
| CLI, data, model, compile, resume, loop | `main` | 536–924 | implemented | read |
| ↳ gradient accumulation | in `main` | 854–870 | implemented | read |
| ↳ clip, step, bias update | in `main` | 874–878 | implemented | read |
| ↳ eval, best and `.last` checkpoints, bits/byte | in `main` | 888–905 | implemented | read |
| Muon Newton–Schulz | `zeropower_via_newtonschulz5`, `zeropower_batched` | `anulm/muon.py` 35–82 | implemented | agent |
| Muon optimizer | `Muon` | `anulm/muon.py` 85–141 | implemented | agent |
| Muon/AdamW split | `build_optimizers` | `anulm/muon.py` 144–182 | implemented | agent |
| Bigram teacher KL | `build_bigram_teacher`, `teacher_kl`, `teacher_weight` | `anulm/bigram_teacher.py` 34–75 | partial (flag) | agent |

## 4. Post-training, adaptation and retrieval

| Component | File · symbol | Lines | Status | Verified |
|---|---|---|---|---|
| SFT: prompt tokens labelled −1 | `anulm/training/finetune.py` `pack` | 91–118 (mask at 113) | implemented | agent + spot-check |
| SFT driver | `anulm/training/finetune.py` `main` | 222–422 | implemented | agent |
| LoRA layer | `anulm/lora.py` `LoRALinear` | 50–84 (scale 61, B=0 at 63, merge 80) | implemented | agent + spot-check |
| LoRA apply / save routers / merge | `apply_lora`, `router_state_dict`, `merge_lora` | 87–166 | implemented | agent |
| QLoRA, 4-bit base | — | — | **absent** in the root (course stub only) | agent |
| Chat data builder | `anulm/data/make_chat.py` `main` | 148–184 | implemented (data only) | agent |
| Option-scoring classifier | `anulm/tasks/decide.py` `Decider` | 105–246 | implemented | agent |
| Temperature scaling | `anulm/tasks/decide.py` `fit_temperature` | 368–374 | implemented | agent + spot-check |
| ECE, Brier, selective accuracy | `anulm/tasks/decide.py` `metrics` | 377–397 | implemented | agent |
| BM25 index (from scratch) | `anulm/inference/rag.py` `build` | 103–168 (weights 137–139) | implemented | agent + spot-check |
| Dense retrieval (e5-small) | `anulm/inference/rag.py` `Dense` | 174–230 | external (`transformers`) | agent |
| Hybrid RRF | `anulm/inference/rag.py` `Retriever.search` | 261–281 | implemented | agent |
| Generative reader | `anulm/inference/rag.py` `RagQA` | 324–375 | implemented | agent |
| Extractive span reader | `anulm/tasks/rc_pointer.py` `Pointer`, `best_span` | 49–72, 120–133 (loss 190) | implemented | agent + spot-check |
| Causal → bidirectional encoder (MNTP) | `anulm/bidirectional.py` `bidir_hidden`, `adapt` | 40–105 | implemented (CUDA only) | agent |
| DPO, RLHF, GRPO | — | — | **absent** in the root | agent |

## 5. Multimodal, serving and export

| Component | File · symbol | Lines | Status | Verified |
|---|---|---|---|---|
| Frozen vision tower (SigLIP ViT-B/16) | `anulm/multimodal/vision_encoder.py` `VisionTower` | 69–106 | external (`timm`) | agent |
| Patch pooling 2×2 + MLP projector | `ImageProjector` | 109–141 | implemented | agent + spot-check |
| Image sequence layout, answer-only targets | `VisionLanguage` (`build` 153–183) | 144–183 | implemented | agent |
| Frozen-feature cache | `build_cache` | 298–361 | implemented | agent |
| Multiple-choice VQA scoring | `mc_accuracy` | 537–564 | implemented | agent |
| Captioning (KV-cached greedy) | `caption` | 587–617 | implemented | agent |
| Audio frame stacking + projector | `anulm/multimodal/speech_encoder.py` `AudioProjector` | 64–92 | implemented | agent + spot-check |
| ASR sequence layout | `ContinuousSpeech` | 95–134 | implemented | agent |
| Whisper encoder (log-mel inside) | `anulm/multimodal/audio_codec.py` `WhisperEncoder` | 215–259 | external (`faster-whisper`, `ctranslate2`) | agent |
| SNAC codec, 7-slot flatten | `SnacCodec`, `flatten`, `unflatten` | 125–208 | external codec; flattening implemented | agent |
| Vocabulary growth (+28,672 audio, +5 control) | `anulm/multimodal/speech_vocab.py` `SpeechVocab`, `grow_checkpoint` | 69–159 | implemented | agent + spot-check |
| Joint multitask training | `anulm/multimodal/omni_train.py` `Omni`, `train` | 177–481 | implemented: `--accum` (line 507), `--proj-scale` (504), seeding and the resume recipe check (`recipe_of` 353) | Re-pinned 2026-11-03 against @fdbb09a; moved to @b42998d on 2026-11-04: `Omni` 177–350, `train` 362–481 |
| Projector norm matching | `anulm/multimodal/omni_train.py` `Omni._match_scale` | 249–263 | implemented; off by default (`--proj-scale embed`, line 504; `experiments/omni_one.py` does not pass it) | Re-pinned 2026-11-03 against @fdbb09a; moved to @b42998d on 2026-11-04 (Chapter 36, 36.5 Details) |
| Inference facade | `anulm/multimodal/omni.py` `Omni`, `Reader`, `Intent` | 33–164 | implemented | agent |
| Stdlib HTTP server, model engine | `anulm/inference/serve.py` `Engine`, `make_handler` | 55–245 | implemented | agent |
| OpenAI-compatible API, SSE streaming | `anulm/inference/api.py` `Generation`, `stream_generation` | 389–512 | implemented (`fastapi`) | agent |
| HF export (safetensors, bf16 storage with fp32 router bias) | `anulm/hf/export_hf.py` `main` | 211–383 (dtype note 328–338) | implemented | agent + spot-check |
| `transformers` wrapper over the native modules | `anulm/hf/modeling_anulm.py` `AnuLMForCausalLM`, `AnuLMCache` | 45–291 | implemented; no beam search, no padding mask, no `inputs_embeds` | agent + spot-check |
| HF config adapter | `anulm/hf/configuration_anulm.py` `AnuLMConfig` | 34–84 | implemented | agent |
| Environment check and first run | `anulm/quickstart.py` `check`, `report`, `run`, `train`, `demo` | 57–217 | implemented | agent |

**Confirmed absent from serving:** paged KV cache, continuous batching (`anulm/inference/api.py`
serialises all device work behind one lock), and quantised weights. The only
quantisation anywhere is ctranslate2's int8 Whisper on CPU
(`anulm/multimodal/audio_codec.py` lines 231–232).

**Stale docstrings to avoid quoting:**

- `anulm/multimodal/speech_vocab.py` says it adds 3 control tokens and 28,675 ids. The code
  adds 5 and 28,677.
- `anulm/multimodal/vision_encoder.py` lines 26–37 say the model was never trained on real
  captions. It later was (`docs/RESULTS.md`).

## 6. Tests that pin the concepts: `tests/test_model.py`

Each row maps a test to the chapter that should use it. They are the
"silent failure" checks: each catches a component that still trains while
it is wrong.

| Concept | Test (line) |
|---|---|
| Causal mask | `causal_mask_does_not_leak_future` (57), `causality_holds_under_gradients_too` (78) |
| RoPE | `rope_is_relative` (101), `rope_preserves_norms` (120), `rotary_cache_tracks_dtype_and_positions` (733) |
| YaRN | `yarn_blends_the_right_bands` (131), `yarn_temperature_reaches_attention` (144) |
| MoE equivalence | `moe_matches_naive_per_token_loop` (180), `moe_with_all_experts_selected_is_dense` (199), `dense_and_sparse_moe_agree` (571), `grouped_moe_matches_sparse_under_bf16` (827) |
| Router | `router_respects_group_limits` (218), `router_weights_come_from_unbiased_scores` (233), `router_choices_survive_autocast` (302), `config_rejects_unreachable_topk` (520) |
| Balancing | `bias_update_moves_in_the_right_direction` (331), `bias_update_recovers_from_collapse` (361), `expert_bias_is_not_trainable` (400), `router_regularisers_are_off_by_default_and_work_when_on` (883), `seq_balance_loss_matches_the_formula` (907) |
| Norm and FFN formulas | `rmsnorm_and_swiglu_match_their_formulas` (252) |
| Initialisation and loss | `tied_embeddings_share_one_table` (287), `all_parameters_receive_gradients` (415), `initial_loss_is_ln_vocab` (428), `chunked_loss_matches_the_full_loss_and_its_gradients` (1713) |
| MLA | `mla_shares_one_rope_key_across_heads` (529) |
| Training mechanics | `gradient_accumulation_equals_large_batch` (545), `window_sampler_covers_every_window_once_per_epoch` (931), `window_sampler_resumes_exactly` (956), `wsd_schedule_is_flat_then_decays_to_min_lr` (1800), `memmap_batches_match_in_memory_batches` (1922), `interleaved_split_samples_the_whole_corpus` (1944), `eval_windows_are_frozen_and_strided_windows_cover_the_split` (1113) |
| KV cache and generation | `incremental_forward_matches_full_forward` (658), `cache_holds_what_it_should` (679), `cached_generation_matches_uncached` (701), `generation_stops_at_eos` (717), `generation_is_deterministic_at_top_k_1` (495), `block_size_boundary` (481), `cached_embeds_decoding_matches_prefix_replay` (1742) |
| Sliding window | `sliding_window_limits_reach` (750), `sliding_window_agrees_with_cache` (776) |
| GQA | `native_gqa_matches_repeat_interleave` (1767) |
| Compile | `dense_moe_compiles_without_graph_breaks` (588), `sparse_moe_breaks_the_graph_as_documented` (603), `compiled_forward_matches_eager` (619), `compiled_backward_produces_gradients` (639), `grouped_moe_compiles_without_graph_breaks` (866), `grouped_moe_trains` (848) |
| Tokenizer | `bpe_documents_get_eos_and_the_vocab_is_exact` (970), `bpe_files_load_under_both_format_ids` (1544), `hf_tokenizer_conversion_is_exact` (1517), `byte_loader_inserts_eos_at_document_boundaries` (1595) |
| Checkpoints and export | `checkpoint_roundtrip_is_exact` (460), `an_exported_folder_loads_everywhere_a_pt_does` (1057), `transformers_wrapper_matches_the_native_model` (1450), `an_export_keeps_what_makes_a_checkpoint_more_than_a_language_model` (1342) |
| Fine-tuning | `lora_starts_as_the_base_model_and_merges_back_exactly` (1163), `lora_scale_is_alpha_over_r` (1262), `qa_packing_masks_prompts_and_keeps_examples_whole` (1676), `decide_scores_each_option_at_its_last_token_and_calibrates` (1875) |
| Merge tree, Muon | `merge_tree_composes_trains_and_bakes_losslessly` (1813), `merge_tree_eval_cache_sees_optimizer_updates` (1849), `batched_muon_matches_per_matrix_muon` (1902) |
| Parameter count | `param_count_matches_analysis` (507) |
