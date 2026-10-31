# Course outline: *Build Your Own LLM, from Scratch to Omni*

The topic plan, chapter by chapter, before more chapters are written. It is
built on this repository's own code: the transformer in `model.py` and
everything around it was written from scratch, and the course teaches
**that** code.

## The idea that runs through the book

> *Loss going down proves almost nothing about correctness.*
> (`test_model.py`)

A leaky causal mask still converges: it just cheats. Broken rotary positions
still converge: they just lose long-range structure. A mixture-of-experts layer
that silently drops experts still converges. So every chapter that builds a
component also teaches the **test that catches its silent failure**. The
repository already has 66 such tests; each is assigned to a chapter below.

## How every build chapter works

1. **Concept.** What the component does, with figures (STYLE.md).
2. **Build it.** The learner writes it in `course/build/anulm.py`, a skeleton
   that grows chapter by chapter into a working model with `model.py`'s
   interface.
3. **Catch it.** Two kinds of check, both runnable on a CPU in seconds:
   - *equivalence:* load the same weights into the learner's class and
     `model.py`'s, and the outputs must match;
   - *silent-failure tests* from `test_model.py`, run against the learner's
     code.
4. **Run it.** Plug it into the real `train.py` on the `30b` preset
   (17M parameters, about 20 minutes on a CPU) and watch it learn.
5. **Code tour.** The same component in `model.py`, line by line, including
   the choices a teaching version leaves out.
6. **From the lab notebook.** One real bug or result from this project
   (`docs/RESULTS.md` section given).
7. **Exercises**, last one open-ended; **key takeaways**.

## Tracks

- **CPU track:** every default. Toy and `30b`-preset models, minutes per run.
- **GPU track (one 16 GB card):** the real 398M model, starting from the
  released base `toonist/AnuLM-Base-v2-400M`, so nobody has to pre-train
  for a week.

---

## Part I · Foundations

### 1 · The smallest language model
- **Topics:** text as bytes; the one job (a probability for every next byte);
  loss as average surprise, nats and bits; the blind-guess baseline ln 256;
  a bigram table; a tiny transformer; the training loop.
- **Code:** `course/ch01-smallest-model/lab.py` (self-contained, warm-up).
- **Catch it:** `initial_loss_is_ln_vocab`: a fresh model must start at
  ln(vocab). If it does not, initialisation is wrong before training starts.
- **Lab notebook:** the same loop, scaled to 398M and 7.8B tokens.

### 2 · Your own data
- **Topics:** collecting local text (files, exports, Wikipedia dumps, web);
  cleaning and prose filtering; deduplication; train/validation splits that
  don't leak; **decontaminating** against your benchmarks; **licences**: what
  you may train on and sell; memory-mapped corpora for data bigger than RAM.
- **Code:** `fetch_hindi.py`, `fetch_hf.py`, `fetch_code.py`, `fetch_web.py`,
  `filter_prose.py`, `mix_corpus.py`, `tools_contamination.py`,
  `experiments/build_v2.py`, `docs/DATASETS.md`.
- **Catch it:** `prose_filter_drops_lists_and_keeps_prose`,
  `interleaved_split_samples_the_whole_corpus`,
  `memmap_batches_match_in_memory_batches`,
  `window_sampler_covers_every_window_once_per_epoch`.
- **Lab notebook:** §8 corpus cleaning; §11 the whole dump at the same
  compute; §13 prose filtering, a clean negative; §14 and §17 the Wikisource
  register gap; §40 more Indian data that did not help; GPL code that reached
  base v2 and cannot be removed afterwards.

### 3 · Build the tokenizer
- **Topics:** why bytes are expensive (2.48 tokens per Devanagari character);
  byte-level BPE from scratch: counting pairs, merging, the vocabulary;
  special tokens (EOS between documents); encoding a corpus to `.bin`;
  measuring fertility; one tokenizer for three scripts; exporting to the
  Hugging Face format.
- **Code:** `bpe.py` (349 lines, standard library only), `hf_tok.py`,
  `tools_hf_tokenizer.py`.
- **Catch it:** `bpe_documents_get_eos_and_the_vocab_is_exact`,
  `byte_loader_inserts_eos_at_document_boundaries`,
  `hf_tokenizer_conversion_is_exact`, `bpe_files_load_under_both_format_ids`.
- **Lab notebook:** §7 bytes vs BPE at matched compute; §22 a 32k tokenizer
  for Hindi, English and Python.

## Part II · Build AnuLM from scratch

### 4 · The skeleton: embeddings, RMSNorm, SwiGLU
- **Topics:** the config object and its presets (`AnuLMConfig.nano_30b`,
  `nano_105b`, `nano_350m`); token embeddings and the output head (tied or
  untied); RMSNorm vs LayerNorm; the SwiGLU MLP; pre-norm residual blocks;
  counting parameters by hand.
- **Code:** `model.py`: `AnuLMConfig`, `RMSNorm`, `SwiGLU`, `Block`.
- **Catch it:** `all_parameters_receive_gradients`,
  `param_count_matches_analysis`, `initial_loss_is_ln_vocab`.
- **Lab notebook:** float32 is not optional: rounding the router's bias to
  bfloat16 changed which experts tokens went to, and the output collapsed into
  repeated tokens (`docs/DEVELOPING.md`).

### 5 · Positions: RoPE and YaRN
- **Topics:** why attention needs positions; rotary embeddings as rotations;
  why RoPE is *relative*; extending context past training length with YaRN
  (frequency bands, attention temperature); the rotary cache.
- **Code:** `model.py`: `Rotary`, `apply_rope`, `rotate_half`, `yarn_mscale`,
  `_yarn_correction_range`.
- **Catch it:** `rope_is_relative`, `rope_preserves_norms`,
  `yarn_blends_the_right_bands`, `yarn_temperature_reaches_attention`,
  `rotary_cache_tracks_dtype_and_positions`, `block_size_boundary`.
- **Lab notebook:** RoPE angles in bfloat16 drift by about 0.8 rad by
  position 200 (`app_omni.py`); §3 does YaRN earn its place; §27 the 400M
  model past its training length; §28 what training at 2,048 bought.

### 6 · Attention
- **Topics:** queries, keys and values; the causal mask; multi-head attention;
  **grouped-query attention** (fewer key/value heads); QK-norm; the **KV
  cache** and incremental decoding; sliding-window attention; **multi-head
  latent attention** (MLA), the compressed alternative.
- **Code:** `model.py`: `GQAttention`, `MLAttention`, `_attn_mask`, `_sdpa`,
  the cache in `AnuLM.forward`.
- **Catch it:** `causal_mask_does_not_leak_future`,
  `causality_holds_under_gradients_too`, `native_gqa_matches_repeat_interleave`,
  `mla_shares_one_rope_key_across_heads`,
  `incremental_forward_matches_full_forward`, `cache_holds_what_it_should`,
  `cached_generation_matches_uncached`, `sliding_window_limits_reach`,
  `sliding_window_agrees_with_cache`.
- **Lab notebook:** §10 architecture ablations; §15 what a sliding window
  costs past the training length.

### 7 · Mixture of experts
- **Topics:** why sparse models (more parameters, same compute per token);
  shared and routed experts; the **router** with sigmoid scores and top-k;
  group-limited routing; **load balancing without an auxiliary loss** (the
  bias rule); the optional sequence-balance loss and router z-loss; three
  ways to dispatch tokens (`sparse`, `dense`, `grouped`) and what
  `torch.compile` makes of each.
- **Code:** `model.py`: `Router`, `MoE`, `AnuLM.update_expert_biases`.
- **Catch it:** `moe_matches_naive_per_token_loop`,
  `moe_with_all_experts_selected_is_dense`, `router_respects_group_limits`,
  `router_weights_come_from_unbiased_scores`,
  `bias_update_moves_in_the_right_direction`,
  `bias_update_recovers_from_collapse`, `expert_bias_is_not_trainable`,
  `config_rejects_unreachable_topk`, `dense_and_sparse_moe_agree`,
  `grouped_moe_matches_sparse_under_bf16`, `seq_balance_loss_matches_the_formula`,
  `router_regularisers_are_off_by_default_and_work_when_on`.
- **Lab notebook:** §4 the three dispatch paths (grouped is 1.48× on a GPU);
  §5 load-balancing behaviour; §16 the bias update rate for fine-grained
  experts; reading imbalance by median as well as max.

### 8 · Assemble and pre-train
- **Topics:** the full `AnuLM` model; generation (temperature, top-k, EOS);
  the training loop in `train.py`: AdamW with a decay/no-decay split, **Muon**,
  warmup and the warmup-stable-decay schedule, gradient clipping, gradient
  accumulation, bf16 autocast, gradient checkpointing, chunked loss;
  evaluation and its own error bar; **checkpoints and exact resume**;
  sizing a run to one GPU; keeping it alive for days (scheduled restarts,
  backups, reboots).
- **Code:** `model.py`: `AnuLM`; `train.py`, `muon.py`, `sample.py`.
- **Catch it:** `gradient_accumulation_equals_large_batch`,
  `checkpoint_roundtrip_is_exact`, `window_sampler_resumes_exactly`,
  `generation_is_deterministic_at_top_k_1`, `generation_stops_at_eos`,
  `wsd_schedule_is_flat_then_decays_to_min_lr`,
  `batched_muon_matches_per_matrix_muon`,
  `chunked_loss_matches_the_full_loss_and_its_gradients`,
  `compiled_forward_matches_eager`, `eval_windows_are_frozen_and_strided_windows_cover_the_split`,
  `merge_tree_eval_cache_sees_optimizer_updates`.
- **Lab notebook:** a cache keyed on the weights' version served stale
  weights after fused AdamW steps: validation loss went 6.37 → 10.66 while
  training loss fell to 4.43; §26 reading the validation loss's own error;
  §32 what learns faster on one GPU (AdamW vs Muon); §35 a demo base in a day
  by decaying a copy of the run; base v2: 1.9M steps, Windows Update reboots,
  and how the run survived them.

## Part III · Post-training

### 9 · Supervised fine-tuning
- **Topics:** prompt templates; loss on the answer only; packing examples
  into rows without leaking across them; held-out loss as the guide; chat,
  question answering, translation and code; building SFT data from your own
  documents.
- **Code:** `finetune.py`, `make_qa.py`, `make_chat.py`, `eval_qa.py`.
- **Catch it:** `qa_packing_masks_prompts_and_keeps_examples_whole`,
  `golden_items_are_well_formed_and_the_scorer_runs`.
- **Lab notebook:** chat v3, held-out 3.44 → 3.145; §20, §23 questions in
  three languages; §24 translation; §25 the Python coder; benchmark
  contamination in code data.

### 10 · LoRA and QLoRA
- **Topics:** the memory bill of a full fine-tune; LoRA's frozen W and thin
  B·A; rank, alpha, which layers (attention, MLP, not the experts); merging;
  **QLoRA**: NF4 4-bit storage, blockwise scales, from scratch; learning vs
  forgetting, measured.
- **Code:** `lora.py`; new: QLoRA for the real model.
- **Catch it:** `lora_starts_as_the_base_model_and_merges_back_exactly`;
  new: *NF4 round trip stays within its bound*, *QLoRA step 0 equals the
  4-bit base*.
- **Lab notebook:** §29 LoRA vs a full fine-tune at 398M; LoRA recipes A–H
  on BANKING77 (`logs/post_v2`): recipe D 90.2% at 259 ms, recipe F (dense
  layers, rank 64) 89.0% at 56 ms, the one shipped for speed.

### 11 · Preference tuning (DPO)
- **Topics:** what SFT can't teach; preference pairs (chosen, rejected); DPO's
  loss and its reference model; building pairs from local data (your own
  ratings, or a stronger answer vs the model's own); judging the result.
- **Code:** new: `dpo.py`.
- **Catch it:** new: *DPO loss is zero-gradient when policy equals reference*,
  *chosen log-probability rises*.
- **Lab notebook:** new experiment for the book.

### 12 · Decisions and lookup: Decide and RAG
- **Topics:** when you need a decision, not prose: scoring every option in
  one pass; **calibration** (0.9 means right 90% of the time); handing low
  confidence to a person; retrieval with BM25 written in numpy; a reader that
  points at the answer; one chat model plus adapters per task.
- **Code:** `decide.py`, `rag.py`, `rc_pointer.py`, `make_rc.py`.
- **Catch it:** `decide_scores_each_option_at_its_last_token_and_calibrates`.
- **Lab notebook:** §33 looking facts up; §34–§38 Decide: speed, decoder vs
  encoder, Hindi 84.5% vs ModernBERT's 67.5%.

## Part IV · Multimodal and shipping

### 13 · Hearing and seeing
- **Topics:** growing the vocabulary (`speech_vocab.py`); audio as discrete
  codes (SNAC) vs continuous states; a frozen Whisper encoder plus a
  projector; a frozen SigLIP tower plus a projector; feeding embeddings with no
  ids; measuring speech (WER) and images (VQA accuracy, caption loss).
- **Code:** `speech_vocab.py`, `audio_codec.py`, `speech_encoder.py`,
  `vision_encoder.py`, `eval_speech.py`.
- **Catch it:** `cached_embeds_decoding_matches_prefix_replay`; new:
  *a projector's output scale is reported against the embeddings'*.
- **Lab notebook:** §39 speech, vision and chat on base v2; English WER 7.1%.

### 14 · One model for everything, and debugging it
- **Topics:** joint multi-task training; task weights; what a "score" across
  tasks hides; diagnosing a regression: probes, controls, A/B runs; why
  one-batch steps drift a shared model; **gradient accumulation** across
  tasks; protecting a skill with share and learning rate; release gates.
- **Code:** `omni_train.py`, `omni_eval.py`, `experiments/omni_one.py`.
- **Catch it:** new: *an accumulated step equals one large mixed batch*;
  *a gate fails when one task regresses even if the mean improves*.
- **Lab notebook:** the full story from 30 Oct 2026: chat held-out
  3.27 → 5.39, a wrong first theory (projector scale), chat-only probes,
  `--accum` fixing it, and choosing chat share and learning rate.

### 15 · Shipping
- **Topics:** release gates; model cards; licences and attribution; export to
  safetensors and a Hugging Face wrapper; a server, an API and a demo app;
  what to measure before publishing; operating a home training box.
- **Code:** `export_hf.py`, `modeling_anulm.py`, `configuration_anulm.py`,
  `serve.py`, `api.py`, `app_omni.py`, `docs/PUBLISHING.md`.
- **Catch it:** `an_exported_folder_loads_everywhere_a_pt_does`,
  `an_export_keeps_what_makes_a_checkpoint_more_than_a_language_model`,
  `transformers_wrapper_matches_the_native_model`,
  `the_demo_page_offers_the_right_modes_for_each_checkpoint`,
  `eval_code_sandbox_contains_generated_code`.
- **Lab notebook:** why AnuLM-Omni was not published on its first run.

---

## What has to be built for the course

| piece | for | status |
|---|---|---|
| `course/build/anulm.py` skeleton, filled in chapter by chapter | Ch 4–8 | done |
| `course/checks/`: equivalence checks against the repository's code, and the silent-failure tests pointed at the learner's code | Ch 2–15 | done |
| QLoRA for the real model, and its tests | Ch 10 | done (`build/solutions/ch10.py`) |
| DPO and its tests | Ch 11 | done (`build/solutions/ch11.py`; no `dpo.py` in the repository yet) |
| A small Omni lab that reproduces the drift and the fix on CPU | Ch 14 | done (`ch14-one-model/lab.py`) |
| Phone layouts for figures | all | new |

## Open decisions

- **Audience:** students, working developers, or both?
- **Language:** English with Hindi examples, or a Hindi edition too?
- **Publishing:** free online, a printed book, or a paid course?
- **Chapter 1:** keep the self-contained warm-up (current), or build it on
  `model.py` from the start?
