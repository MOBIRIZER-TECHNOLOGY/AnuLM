# Repository source map

**Layout note.** This is the historical citation map, not the current folder
inventory. The organized repository uses the `anulm/` package; its
`docs/source-paths.json` maps these filenames to their new locations. Exact
line checks still use the recorded historical revision. Do not compare a
historical line number against a new import or compatibility wrapper.

The table every chapter cites from. Line numbers are taken at `@773b6a8` on
the `course-book` branch, from the root copy of the repository.

Public commit: `ae592387dc373cc6f579abce444c7f08c9e373a3` (the tip of `main` on
[GitHub](https://github.com/MOBIRIZER-TECHNOLOGY/AnuLM), checked 2026-10-31).
The pin itself is not on GitHub. Every code file it cites is byte-identical at
the public commit; the two differ only in `course/` and an editor setting. The
HTML preview links citations to the public commit, and re-checks that each
cited file is identical before it does.
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
`docs/reviews/source-verification.md`. This applies to "agent" rows too.

---

## 1. Core model: `model.py`

| Component | Symbol | Lines | Status | Concepts | Verified |
|---|---|---|---|---|---|
| Configuration and presets | `AnuLMConfig`, `nano_30b`, `nano_350m`, `nano_105b` | 39–206 | implemented | hyperparameters, GQA/MLA switch, MoE sizes | read |
| Config validation | `AnuLMConfig.__post_init__` | 121–144 | implemented | top-k reachable through groups | read |
| RMSNorm | `RMSNorm` | 213–223 | implemented | computed in fp32, learned gain, no mean subtraction | read |
| SwiGLU MLP / expert body | `SwiGLU` | 226–236 | implemented | gate · up, SiLU, 3·D·I parameters | read |
| YaRN helpers | `_yarn_correction_dim`, `_yarn_correction_range`, `yarn_mscale` | 243–256 | implemented | NTK-by-parts, temperature | read |
| Rotary tables | `Rotary` (`_build_inv_freq` 286–303, `_grow` 305–318, `forward` 320–324) | 259–324 | implemented | θ_i = base^(−2i/d), cached cos/sin, absolute `start` | read |
| Rotation | `rotate_half`, `apply_rope` | 327–341 | implemented | split-halves (NeoX) convention | read |
| Causal and window mask | `_attn_mask` | 344–360 | implemented | causal offset for the cache, sliding window | read |
| Attention kernel call | `_sdpa` | 393–397 | external (PyTorch SDPA) | softmax(QKᵀ·scale + mask)V | read |
| Native-GQA switch | `NATIVE_GQA` | 363–366 | partial (off by default) | `enable_gqa` | read |
| Merge-tree tables | `merge_tree_matrix` | 369–390 | partial (off unless `merge_tree`) | sparse reparameterisation from BPE merges | read |
| GQA attention | `GQAttention.__init__`, `GQAttention.forward` | 404–465 | implemented | fused QKV, QK-norm, RoPE, KV cache, repeat_interleave | read |
| MLA attention | `MLAttention` | 472–543 | implemented | latent KV `kv_lora_rank`, decoupled rope key, cache = latent | read |
| Router | `Router` (`forward` 576–587, `_gate` 589–637) | 550–637 | implemented | sigmoid scores, bias for selection only, group-limited top-k, seq-balance loss, z-loss | read |
| MoE layer | `MoE.__init__`, `MoE.forward` | 640–667 | implemented | routed experts, plus a shared expert when `num_shared_experts > 0`; the released model sets it to 0 | read |
| MoE dispatch paths | `_route_sparse` / `_route_dense` / `_route_grouped` | 669–749 | implemented | gather-scatter, one-hot, `F.grouped_mm` | read |
| Transformer block | `Block` | 756–779 | implemented | pre-norm residual; layer 0 dense, rest MoE; per-layer window | read |
| Model | `AnuLM.__init__` | 782–808 | implemented | embedding, shared rotary, stack, final norm, untied head, init | read |
| Weight init | `AnuLM._init_weights` + scaled residual init | 804–816 | implemented | N(0, 0.02); output projections ÷ √(2L) | read |
| KV cache container | `AnuLM.new_cache` | 818–820 | implemented | one dict per layer | read |
| Forward pass and loss | `AnuLM.forward` | 822–873 | implemented | embeds bypass, grad checkpointing, last-position shortcut, CE with `ignore_index=-1` | read |
| Chunked loss | `enable_chunked_loss`, `_chunked_loss`, `_ce_sum` | 875–901 | partial (opt-in) | never materialises B·S·V logits | read |
| Token tables | `_compose`, `_table`, `train`, `_head`, `bake_merge_tree` | 904–956 | partial | merge-tree; eval cache | read |
| Decoding from embeddings | `AnuLM.decode_from_embeds` | 958–1005 | implemented | greedy, n-gram block, repetition penalty | read |
| Gradient checkpointing | `enable_gradient_checkpointing` | 1007–1011 | implemented | activation recomputation | read |
| Aux-loss-free balancing | `AnuLM.update_expert_biases` | 1014–1044 | implemented | sign rule on the bias, imbalance ratio | read |
| Parameter count | `AnuLM.num_params` | 1047–1056 | implemented | total vs active | read |
| Generation | `AnuLM.generate` | 1058–1116 | implemented | prefill/decode, context slide, repetition penalty, temperature, top-k, EOS, streaming | read |
| Checkpoint loading | `load_checkpoint` | 1153–1202 | implemented | `.pt`, exported folder, Hub id, adapters | read |
| Memory guard | `fit_memory` | 1205–1224 | implemented | auto checkpointing, chunked CE | read |
| Smoke test | `__main__` | 1119–1144 | implemented | loss ≈ ln V, every parameter gets a gradient | read |

## 2. Tokenizer: `bpe.py`

| Component | Symbol | Lines | Status | Verified |
|---|---|---|---|---|
| Pre-tokenisation regex, document separator | `CHUNK_RE`, `DOC_SEP` | 39–45 | implemented | read |
| Vocabulary and ranks, EOS above the merges | `BPE.__init__`, `vocab_size` | 51–67 | implemented | read |
| Training (pair counts, pair→chunk index, lazy max-heap) | `BPE.train` | 71–143 | implemented | read |
| Encoding (lowest-rank merge first, memoised) | `_encode_chunk`, `encode`, `encode_documents` | 147–186 | implemented | read |
| Decoding | `decode` | 188–190 | implemented | read |
| Persistence | `save`, `load` | 194–213 | implemented | read |
| Parallel corpus encoding into int16 | `iter_documents`, `cmd_encode` | 227–312 | implemented | read |
| Statistics (bytes/token, round trip) | `_stats`, `cmd_stats` | 315–330 | implemented | read |

The byte-level path, which needs no tokenizer, lives in `train.py`:
`load_bytes` (lines 62–98) and `BYTE_EOS = 257` (line 101).

## 3. Training: `train.py`, `muon.py`, `bigram_teacher.py`

| Component | Symbol | Lines | Status | Verified |
|---|---|---|---|---|
| Crash-safe save | `atomic_save` | 32–46 | implemented | read |
| Window sampling without replacement | `WindowSampler` | 104–148 | implemented | read |
| Memory-mapped corpus | `load_memmap` | 151–161 | implemented | read |
| Interleaved validation split | `interleaved_split` | 164–178 | partial (flag) | read |
| Spill warning | `warn_if_spilling` | 181–196 | implemented | read |
| Batches (y = x shifted by one) | `make_batch` | 199–208 | implemented | read |
| Fixed eval windows / strided windows | `eval_windows`, `strided_windows` | 211–245 | implemented | read |
| LR schedule (warmup, cosine or WSD, rewarmup) | `lr_at` | 248–269 | implemented | read |
| Multi-optimizer handle | `OptimizerSet` | 272–307 | implemented | read |
| AdamW parameter groups (decay on 2-D only) | `make_optimizer` | 310–337 | implemented | read |
| Evaluation with standard error | `evaluate` | 340–396 | implemented | read |
| CLI, data, model, compile, resume, loop | `main` | 399–750 | implemented | read |
| ↳ gradient accumulation | in `main` | 691–707 | implemented | read |
| ↳ clip, step, bias update | in `main` | 708–712 | implemented | read |
| ↳ eval, best and `.last` checkpoints, bits/byte | in `main` | 725–742 | implemented | read |
| Muon Newton–Schulz | `zeropower_via_newtonschulz5`, `zeropower_batched` | `muon.py` 35–82 | implemented | agent |
| Muon optimizer | `Muon` | `muon.py` 85–137 | implemented | agent |
| Muon/AdamW split | `build_optimizers` | `muon.py` 140–178 | implemented | agent |
| Bigram teacher KL | `build_bigram_teacher`, `teacher_kl`, `teacher_weight` | `bigram_teacher.py` 34–75 | partial (flag) | agent |

## 4. Post-training, adaptation and retrieval

| Component | File · symbol | Lines | Status | Verified |
|---|---|---|---|---|
| SFT: prompt tokens labelled −1 | `finetune.py` `pack` | 84–111 (mask at 106) | implemented | agent + spot-check |
| SFT driver | `finetune.py` `main` | 181–365 | implemented | agent |
| LoRA layer | `lora.py` `LoRALinear` | 50–81 (scale 61, B=0 at 63, merge 78) | implemented | agent + spot-check |
| LoRA apply / save routers / merge | `apply_lora`, `router_state_dict`, `merge_lora` | 84–158 | implemented | agent |
| QLoRA, 4-bit base | — | — | **absent** in the root (course stub only) | agent |
| Chat data builder | `make_chat.py` `main` | 97–119 | implemented (data only) | agent |
| Option-scoring classifier | `decide.py` `Decider` | 101–233 | implemented | agent |
| Temperature scaling | `decide.py` `fit_temperature` | 344–350 | implemented | agent + spot-check |
| ECE, Brier, selective accuracy | `decide.py` `metrics` | 353–370 | implemented | agent |
| BM25 index (from scratch) | `rag.py` `build` | 99–159 (weights 128–130) | implemented | agent + spot-check |
| Dense retrieval (e5-small) | `rag.py` `Dense` | 165–215 | external (`transformers`) | agent |
| Hybrid RRF | `rag.py` `Retriever.search` | 239–257 | implemented | agent |
| Generative reader | `rag.py` `RagQA` | 298–342 | implemented | agent |
| Extractive span reader | `rc_pointer.py` `Pointer`, `best_span` | 51–69, 111–124 (loss 177) | implemented | agent + spot-check |
| Causal → bidirectional encoder (MNTP) | `bidirectional.py` `bidir_hidden`, `adapt` | 40–101 | implemented (CUDA only) | agent |
| DPO, RLHF, GRPO | — | — | **absent** in the root | agent |

## 5. Multimodal, serving and export

| Component | File · symbol | Lines | Status | Verified |
|---|---|---|---|---|
| Frozen vision tower (SigLIP ViT-B/16) | `vision_encoder.py` `VisionTower` | 67–97 | external (`timm`) | agent |
| Patch pooling 2×2 + MLP projector | `ImageProjector` | 100–132 | implemented | agent + spot-check |
| Image sequence layout, answer-only targets | `VisionLanguage` (`build` 144–165) | 135–165 | implemented | agent |
| Frozen-feature cache | `build_cache` | 257–287 | implemented | agent |
| Multiple-choice VQA scoring | `mc_accuracy` | 463–490 | implemented | agent |
| Captioning (KV-cached greedy) | `caption` | 513–543 | implemented | agent |
| Audio frame stacking + projector | `speech_encoder.py` `AudioProjector` | 65–93 | implemented | agent + spot-check |
| ASR sequence layout | `ContinuousSpeech` | 96–126 | implemented | agent |
| Whisper encoder (log-mel inside) | `audio_codec.py` `WhisperEncoder` | 198–229 | external (`faster-whisper`, `ctranslate2`) | agent |
| SNAC codec, 7-slot flatten | `SnacCodec`, `flatten`, `unflatten` | 108–191 | external codec; flattening implemented | agent |
| Vocabulary growth (+28,672 audio, +5 control) | `speech_vocab.py` `SpeechVocab`, `grow_checkpoint` | 62–161 | implemented | agent + spot-check |
| Joint multitask training | `omni_train.py` `Omni`, `train` | 127–330 | implemented | agent; Stage 3: 127–330 at the pin (`train` 236–330). The working tree has uncommitted edits (`--accum`, `--proj-scale`) that move `train` to 256–357 |
| Projector norm matching | `Omni._match_scale` | — | **uncommitted**: not in the code at the pin | Stage 3: exists only in the working tree (lines 196–211, behind `--proj-scale embed`). Cite it only after those edits are committed and the map is re-pinned |
| Inference facade | `omni.py` `Omni`, `Reader`, `Intent` | 33–144 | implemented | agent |
| Stdlib HTTP server, model engine | `serve.py` `Engine`, `make_handler` | 50–200 | implemented | agent |
| OpenAI-compatible API, SSE streaming | `api.py` `Generation`, `stream_generation` | 334–449 | implemented (`fastapi`) | agent |
| HF export (safetensors, bf16 storage with fp32 router bias) | `export_hf.py` `main` | 169–321 (dtype note 271–279) | implemented | agent + spot-check |
| `transformers` wrapper over the native modules | `modeling_anulm.py` `AnuLMForCausalLM`, `AnuLMCache` | 45–264 | implemented; no beam search, no padding mask, no `inputs_embeds` | agent + spot-check |
| HF config adapter | `configuration_anulm.py` `AnuLMConfig` | 34–74 | implemented | agent |
| Environment check and first run | `quickstart.py` `check`, `report`, `run`, `train`, `demo` | 55–214 | implemented | agent |

**Confirmed absent from serving:** paged KV cache, continuous batching (`api.py`
serialises all device work behind one lock), and quantised weights. The only
quantisation anywhere is ctranslate2's int8 Whisper on CPU
(`audio_codec.py` lines 214–215).

**Stale docstrings to avoid quoting:**

- `speech_vocab.py` says it adds 3 control tokens and 28,675 ids. The code
  adds 5 and 28,677.
- `vision_encoder.py` lines 26–37 say the model was never trained on real
  captions. It later was (`docs/RESULTS.md`).

## 6. Tests that pin the concepts: `test_model.py`

Each row maps a test to the chapter that should use it. They are the
"silent failure" checks: each catches a component that still trains while
it is wrong.

| Concept | Test (line) |
|---|---|
| Causal mask | `causal_mask_does_not_leak_future` (46), `causality_holds_under_gradients_too` (67) |
| RoPE | `rope_is_relative` (90), `rope_preserves_norms` (109), `rotary_cache_tracks_dtype_and_positions` (625) |
| YaRN | `yarn_blends_the_right_bands` (120), `yarn_temperature_reaches_attention` (133) |
| MoE equivalence | `moe_matches_naive_per_token_loop` (169), `moe_with_all_experts_selected_is_dense` (188), `dense_and_sparse_moe_agree` (463), `grouped_moe_matches_sparse_under_bf16` (683) |
| Router | `router_respects_group_limits` (207), `router_weights_come_from_unbiased_scores` (222), `config_rejects_unreachable_topk` (412) |
| Balancing | `bias_update_moves_in_the_right_direction` (241), `bias_update_recovers_from_collapse` (271), `expert_bias_is_not_trainable` (310), `router_regularisers_are_off_by_default_and_work_when_on` (739), `seq_balance_loss_matches_the_formula` (763) |
| Initialisation and loss | `all_parameters_receive_gradients` (325), `initial_loss_is_ln_vocab` (338), `chunked_loss_matches_the_full_loss_and_its_gradients` (1531) |
| MLA | `mla_shares_one_rope_key_across_heads` (421) |
| Training mechanics | `gradient_accumulation_equals_large_batch` (437), `window_sampler_covers_every_window_once_per_epoch` (787), `window_sampler_resumes_exactly` (812), `wsd_schedule_is_flat_then_decays_to_min_lr` (1610), `memmap_batches_match_in_memory_batches` (1732), `interleaved_split_samples_the_whole_corpus` (1754), `eval_windows_are_frozen_and_strided_windows_cover_the_split` (971) |
| KV cache and generation | `incremental_forward_matches_full_forward` (550), `cache_holds_what_it_should` (571), `cached_generation_matches_uncached` (593), `generation_stops_at_eos` (609), `generation_is_deterministic_at_top_k_1` (387), `block_size_boundary` (373), `cached_embeds_decoding_matches_prefix_replay` (1560) |
| Sliding window | `sliding_window_limits_reach` (642), `sliding_window_agrees_with_cache` (668) |
| GQA | `native_gqa_matches_repeat_interleave` (1585) |
| Compile | `dense_moe_compiles_without_graph_breaks` (480), `sparse_moe_breaks_the_graph_as_documented` (495), `compiled_forward_matches_eager` (511), `compiled_backward_produces_gradients` (531), `grouped_moe_compiles_without_graph_breaks` (722), `grouped_moe_trains` (704) |
| Tokenizer | `bpe_documents_get_eos_and_the_vocab_is_exact` (826), `bpe_files_load_under_both_format_ids` (1367), `hf_tokenizer_conversion_is_exact` (1339), `byte_loader_inserts_eos_at_document_boundaries` (1413) |
| Checkpoints and export | `checkpoint_roundtrip_is_exact` (352), `an_exported_folder_loads_everywhere_a_pt_does` (914), `transformers_wrapper_matches_the_native_model` (1271), `an_export_keeps_what_makes_a_checkpoint_more_than_a_language_model` (1170) |
| Fine-tuning | `lora_starts_as_the_base_model_and_merges_back_exactly` (1021), `qa_packing_masks_prompts_and_keeps_examples_whole` (1494), `decide_scores_each_option_at_its_last_token_and_calibrates` (1685) |
| Merge tree, Muon | `merge_tree_composes_trains_and_bakes_losslessly` (1623), `merge_tree_eval_cache_sees_optimizer_updates` (1659), `batched_muon_matches_per_matrix_muon` (1712) |
| Parameter count | `param_count_matches_analysis` (399) |
