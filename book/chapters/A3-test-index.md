---
title: "Appendix C · Test index"
part: "Appendices"
accent: slate
pinned: "@084cbdd"
---

# Appendix C  Test index

`tests/test_model.py` holds 76 tests at @084cbdd. They need no test framework:
`@test` (line 28) appends each function to the list `TESTS`, and `main`
(line 2213) runs them in file order, printing PASS or FAIL and a count. Most
build `tiny()` (line 33) with a few fields changed. On the book's machine, a
clean export of the pin passed all 76 in 41 s on a CPU (Chapter 41.5;
`book/measurements/part_x_tests.json`), with the GPU hidden. `README.md` and
`docs/TUTORIAL.md` give the same count, 76.

The table lists every test in file order, with the line of its `def`, what it
checks, and the chapter that uses it. "Both" means the test runs for GQA and
for MLA. A test marked *optional* returns early, and is counted as passing,
when a package or PyTorch function it needs is missing.

## C.1  The model

| Test | Line | What it checks | Ch |
|---|---|---|---|
| `causal_mask_does_not_leak_future` | 46 | rewriting tokens 8–15 moves no logit at positions 0–7 by 10⁻⁵ or more (both) | 12 |
| `causality_holds_under_gradients_too` | 67 | the gradient of position 5's logits into the embeddings of positions 6–11 is exactly 0, and into positions 0–5, its own included, is not all zero (MLA) | 6, 12 |
| `rope_is_relative` | 90 | q·k after rotation depends only on the gap between positions, to 10⁻⁴ | 14 |
| `rope_preserves_norms` | 109 | rotation leaves vector norms unchanged, to 10⁻⁵ | 14 |
| `yarn_blends_the_right_bands` | 120 | YaRN leaves the fastest frequency alone, divides the slowest by the factor, and ramps monotonically between | 29 |
| `yarn_temperature_reaches_attention` | 133 | both attention classes multiply their scale by mscale² with YaRN on, and use the plain 1/√d with it off | 29 |
| `moe_matches_naive_per_token_loop` | 169 | the MoE layer equals a Python loop over tokens and their chosen experts, plus the shared expert | 18 |
| `moe_with_all_experts_selected_is_dense` | 188 | with top-k = E, scaling 1 and no shared expert, the output is the sum of all experts weighted by their sigmoid scores normalised to sum to 1 | 17, 18 |
| `router_respects_group_limits` | 207 | with 4 groups and topk_group 2, no token's experts span more than 2 groups; weights sum to `routed_scaling_factor` | 17 |
| `router_weights_come_from_unbiased_scores` | 222 | a large bias changes which experts are picked but not the weights of unchanged picks | 17 |
| `bias_update_moves_in_the_right_direction` | 241 | every update lowers overloaded experts' bias and raises starved ones' by exactly `bias_update_rate` | 18 |
| `bias_update_recovers_from_collapse` | 271 | from a router forced onto two experts, 200 updates cut the mean imbalance by at least 40% | 18 |
| `expert_bias_is_not_trainable` | 310 | the balancing bias is a buffer in the state dict, not a parameter AdamW could touch | 17, 31 |
| `all_parameters_receive_gradients` | 325 | after one backward pass every parameter has a gradient (both) | 6, 20 |
| `initial_loss_is_ln_vocab` | 338 | the untrained loss is within 0.35 of ln V (both) | 3, 20, 41 |
| `checkpoint_roundtrip_is_exact` | 352 | `torch.save` and reload reproduce the logits exactly (MLA) | 25 |
| `block_size_boundary` | 373 | a sequence of exactly `block_size` runs; one token more raises an assertion | 20 |
| `generation_is_deterministic_at_top_k_1` | 387 | greedy decoding twice gives identical ids, of shape (1, 4 + 12) | 28, 41 |
| `param_count_matches_analysis` | 399 | `nano_30b` and `nano_105b` have the analytical totals and active counts, within 1,000 | 20, 41 |
| `config_rejects_unreachable_topk` | 412 | a top-k larger than the reachable groups hold is refused at construction | 17 |
| `mla_shares_one_rope_key_across_heads` | 421 | MLA projects one rope key per token, not one per head, and `kv_b_proj` has the right width | 19 |
| `gradient_accumulation_equals_large_batch` | 437 | two half-batches, each loss halved, give the full batch's gradients to 2·10⁻⁵ | 22 |

## C.2  Dispatch paths, compilation and the cache

| Test | Line | What it checks | Ch |
|---|---|---|---|
| `dense_and_sparse_moe_agree` | 463 | the `sparse` and `dense` dispatch paths agree to 10⁻⁵ (both) | 18 |
| `dense_moe_compiles_without_graph_breaks` | 480 | `torch._dynamo.explain` finds one graph and no breaks with `dense` | 23 |
| `sparse_moe_breaks_the_graph_as_documented` | 495 | `sparse` does break the graph, as the config comment says | 23 |
| `compiled_forward_matches_eager` | 511 | `torch.compile` with the `aot_eager` backend gives exactly the eager output, for `sparse` and `dense` | 23 |
| `compiled_backward_produces_gradients` | 531 | a backward pass through the compiled model reaches every parameter | 23 |
| `incremental_forward_matches_full_forward` | 550 | feeding a sequence in chunks through the cache gives the full pass's logits, to 10⁻⁴ (both) | 27 |
| `cache_holds_what_it_should` | 571 | GQA caches H<sub>kv</sub> heads, not H; MLA caches the latent and one rope key | 19, 27 |
| `cached_generation_matches_uncached` | 593 | greedy decoding with and without the cache agree token for token, past `block_size` (both) | 27 |
| `generation_stops_at_eos` | 609 | `generate` stops once every row has produced `eos_id`, and runs to the end without one | 28 |
| `rotary_cache_tracks_dtype_and_positions` | 625 | the rotary tables follow the requested dtype, and `start` indexes absolute positions | 14 |
| `sliding_window_limits_reach` | 642 | on a windowed layer a token W back has no effect and one W − 1 back does; an unwindowed layer still reaches (both) | 12, 29 |
| `sliding_window_agrees_with_cache` | 668 | windowed attention decodes the same tokens with and without the cache | 27, 29 |
| `grouped_moe_matches_sparse_under_bf16` | 683 | the `grouped` path matches `sparse` under bf16, within 5% of the largest output (both); *optional* (`F.grouped_mm`) | 18 |
| `grouped_moe_trains` | 704 | gradients flow through `grouped_mm` to every parameter and are finite; *optional* | 18 |
| `grouped_moe_compiles_without_graph_breaks` | 722 | `grouped` compiles with no graph breaks; *optional* | 23 |
| `router_regularisers_are_off_by_default_and_work_when_on` | 739 | both regularisers are off by default; on, they add exactly `aux_loss` to the loss, reach the router weights, and vanish in eval mode | 18 |
| `seq_balance_loss_matches_the_formula` | 763 | the sequence-balance loss equals the hand computation with uniform scores | 18 |

## C.3  Data, tokenizers, training and evaluation

| Test | Line | What it checks | Ch |
|---|---|---|---|
| `window_sampler_covers_every_window_once_per_epoch` | 787 | each epoch is a permutation of a shifted window grid, with no window repeated and offsets that move | 21 |
| `window_sampler_resumes_exactly` | 812 | a sampler restored from `state()` continues the same sequence across an epoch boundary | 21, 25 |
| `bpe_documents_get_eos_and_the_vocab_is_exact` | 826 | `--vocab N` counts the EOS id; one EOS per document; decode drops it | 9 |
| `the_demo_page_offers_the_right_modes_for_each_checkpoint` | 845 | `anulm/apps/app.py` picks the right modes and labels per checkpoint kind; *optional* (`gradio`, `safetensors`) | 38 |
| `an_exported_folder_loads_everywhere_a_pt_does` | 914 | an exported folder loads through `load_checkpoint` with identical logits, and no script reads a checkpoint with `torch.load`; *optional* (`safetensors`) | 39 |
| `eval_windows_are_frozen_and_strided_windows_cover_the_split` | 971 | `eval_windows` returns its historic indices; strided windows cover the split whatever the batch; `evaluate` returns a mean and a standard error | 24 |
| `lora_starts_as_the_base_model_and_merges_back_exactly` | 1021 | LoRA starts at the base model, trains only adapters, saves the router bias with them, and merges back to 10⁻⁴. Its last check, that a merge without the saved bias differs, is conditional and never runs: the test never calls `update_expert_biases`, so the bias does not move | 31 |
| `an_exported_qa_checkpoint_still_offers_its_question_mode` | 1110 | an exported question-answering checkpoint keeps its question mode in `anulm/inference/serve.py`; *optional* (`gradio`, `safetensors`) | 38, 39 |
| `an_export_keeps_what_makes_a_checkpoint_more_than_a_language_model` | 1170 | `anulm/hf/export_hf.py` keeps adapters, heads and labels, and an adapter export finds its backbone; *optional* (`safetensors`) | 39 |
| `eval_code_sandbox_contains_generated_code` | 1219 | `anulm/eval/eval_code.py` runs generated code with a timeout, a private directory and no surviving children | 24 |
| `transformers_wrapper_matches_the_native_model` | 1271 | the `transformers` wrapper gives the native logits and greedy tokens, and survives a save and reload; *optional* (`transformers`) | 39 |
| `hf_tokenizer_conversion_is_exact` | 1339 | the converted `tokenizer.json` encodes every probe exactly as `anulm/bpe.py` does; *optional* (`tokenizers`) | 9, 39 |
| `bpe_files_load_under_both_format_ids` | 1367 | tokenizer files load under the old and new format ids, refuse foreign ones, and every committed tokenizer loads | 9 |
| `byte_loader_inserts_eos_at_document_boundaries` | 1413 | `load_bytes` puts one EOS per document, normalises CRLF, and stays uint8 without separators | 8 |
| `prose_filter_drops_lists_and_keeps_prose` | 1435 | the prose filter keeps prose lines and the title and drops list-shaped articles | 21 |
| `golden_items_are_well_formed_and_the_scorer_runs` | 1448 | golden-set items satisfy their contract and the scorer runs on a byte checkpoint | 24 |
| `qa_packing_masks_prompts_and_keeps_examples_whole` | 1494 | SFT packing labels exactly the answer tokens, never splits an example, and drops one longer than the block | 30 |
| `chunked_loss_matches_the_full_loss_and_its_gradients` | 1531 | chunked cross-entropy gives the same loss and gradients, with ignored targets and a short last chunk | 23 |
| `cached_embeds_decoding_matches_prefix_replay` | 1560 | `decode_from_embeds` with the cache emits what re-running the prefix emits (both) | 36 |
| `native_gqa_matches_repeat_interleave` | 1585 | `NATIVE_GQA` and `repeat_interleave` agree on full, cached and windowed paths | 13 |
| `wsd_schedule_is_flat_then_decays_to_min_lr` | 1610 | over 100 steps: warmup to the peak by step 9, flat over steps 10–79, strictly falling over 80–99, and `min_lr` at step 100 | 7 |
| `merge_tree_composes_trains_and_bakes_losslessly` | 1623 | merge-tree rows compose as defined, gradients reach ancestors, and baking changes no output | 10 |
| `merge_tree_eval_cache_sees_optimizer_updates` | 1659 | the eval-mode table cache never outlives a weight update | 10, 26 |
| `decide_scores_each_option_at_its_last_token_and_calibrates` | 1685 | each option is read at its last token; ECE is 0 when calibrated; temperature never changes the argmax | 32 |
| `batched_muon_matches_per_matrix_muon` | 1712 | batched Newton–Schulz gives the same steps as per-matrix Muon, to 10⁻⁶ | 7 |
| `memmap_batches_match_in_memory_batches` | 1732 | a memory-mapped corpus yields the same (x, y) batches as a tensor | 21 |
| `interleaved_split_samples_the_whole_corpus` | 1754 | the interleaved split holds out both sources; the tail split holds out only the last | 21 |

## C.4  Speech, images, serving and the pipeline

These ten tests came in with the package layout. Most guard a fix to the
multimodal stack or to a cache that could go stale without an error.

| Test | Line | What it checks | Ch |
|---|---|---|---|
| `whisper_audio_chunks_cover_long_recordings` | 1774 | `audio_chunks` splits two 30 s windows plus 17 samples into chunks of 30 s, 30 s and 17 samples that rejoin exactly; a clip that fits stays one chunk | 35 |
| `continuous_asr_decodes_long_chunks_in_order` | 1789 | `transcribe` encodes a 65 s clip as 30 s, 30 s and 5 s chunks and joins their texts in order; a short clip is one chunk, cut at its first newline | 35 |
| `image_feature_cache_reuses_invalidates_and_recovers` | 1853 | `build_cache` reuses the feature cache without re-encoding, and rebuilds it after an edited image, a missing sidecar, short bytes or leftover `.tmp` files | 36 |
| `sft_packed_cache_invalidates_same_count_content_edits` | 1917 | `load_packed` reuses its cache for unchanged data, and re-encodes when an answer changes but the row count does not | 30 |
| `omni_api_availability_and_missing_reader_fallback` | 1955 | one Omni checkpoint lists the chat, answer, vision and ASR models; with no reader, `answer_question` falls back to labelled memory, or returns no answer when the fallback is off; *optional* (the API's dependencies) | 37, 38 |
| `legacy_half_keeps_routers_fp32_and_projectors_runnable` | 2005 | `Models._half` keeps router weights and biases fp32, with every bias still distinct, makes a projector's weight and bias bf16, and leaves the module on the CPU; *optional* (the app's dependencies) | 23, 37 |
| `downloader_restarts_when_server_ignores_range` | 2033 | a resumed download answered with a full 200 response replaces the stale `.part` bytes instead of appending to them | 37 |
| `training_fingerprint_preserves_resume_and_invalidates_downstream` | 2085 | a changed Omni recipe archives the old checkpoint, resume point, curve and reports and clears the training and evaluation stages; an unchanged one keeps everything | 25, 37 |
| `experiment_scripts_resolve_root_to_the_repository` | 2145 | every `experiments/*.py` that sets `ROOT` from its own path lands on the repository root, and at least five do | 37 |
| `from_scratch_pipeline_runs_and_skips_completed_stages` | 2168 | `anulm.training.train_from_scratch` runs tokenizer, encoding, pretraining, vocabulary growth and Omni training on a tiny corpus, and a second run skips all five as verified | 22, 37 |

## C.5  What the suite does not catch

A test proves only what a wrong implementation would fail. The book's reviews
found these gaps. Each was confirmed here, by reading the test or by running
it against a deliberately broken model on an export of the pin.

| Test | Gap | How confirmed |
|---|---|---|
| `causal_mask_does_not_leak_future` | `tiny()` has no window, so `_sdpa` gets `mask=None` and uses `is_causal`; the explicit mask `_attn_mask` builds is never tested by it (`sliding_window_limits_reach` covers windowed layers) | `ch41_shapes.py` prints `mask None` for `tiny()` (Appendix A.1) |
| `sliding_window_agrees_with_cache` | compares cached with uncached decoding, which share `_attn_mask`; a window one key too wide passes | run with `_attn_mask` patched to `window + 1`: passes |
| `initial_loss_is_ln_vocab` | the 0.35 tolerance lets a badly scaled embedding through | run with the embedding initialised at std 1.0: passes |
| `native_gqa_matches_repeat_interleave` | uses `n_kv_head=1`, where every query head shares the one key/value head, so a wrong head pairing cannot show | line 1590 |
| `lora_starts_as_the_base_model_and_merges_back_exactly` | its router-bias check never runs (C.3) | lines 1102–1106 |
| `grouped_moe_*` (three tests) | return silently, and pass, without `F.grouped_mm` | lines 688, 709, 727 |
| `transformers_wrapper_matches_the_native_model`, `hf_tokenizer_conversion_is_exact` | return, and pass, without `transformers` or `tokenizers` (they print `(skipped: ...)`) | lines 1282–1286, 1342–1346 |
| `the_demo_page_offers_the_right_modes_for_each_checkpoint` | returns silently, and passes, after its `gradio` check if `safetensors` is missing | lines 888–891 |

*verified*: names, lines and order against `tests/test_model.py` at @084cbdd by
`book/code-examples/ch41_audit.py` (Appendix F). The "what it checks" column
was written from each test's assertions, not its docstring.
