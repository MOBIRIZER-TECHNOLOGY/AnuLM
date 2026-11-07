# Repository architecture (Stage 1)

What this repository implements, traced through the code rather than read off
the README. Every claim cites a file and symbol, and line numbers are taken at
commit `@fdbb09a` on the `course-book` branch. The working tree had
uncommitted edits to eight files when this was written (`app_omni.py`,
`convert_parquet.py`, `anulm/eval/eval_code.py`, `anulm/multimodal/omni_train.py` and others), but not to
`anulm/model.py`, `anulm/training/train.py`, `anulm/bpe.py` or `anulm/inference/sample.py`. The core line numbers below are
therefore stable.

---

## 1. What kind of model this is

**A decoder-only, sparse Mixture-of-Experts transformer, written from
scratch in PyTorch.** Its pieces come from published papers: grouped-query
attention, multi-head latent attention (DeepSeek-V2), and DeepSeek-V3's
sigmoid router with bias-only balancing. `anulm/model.py` lines 1–17 list
them, and `docs/ARCHITECTURE.md` gives the paper behind each.

It is **not** a GPT-2 clone. The textbook model the book's prompt assumes
(absolute positional embeddings, LayerNorm, a dense GELU MLP, multi-head
attention, tied embeddings) differs from this one in nearly every
component.

| Component | Textbook GPT-2 | This repository |
|---|---|---|
| Tokens | 50k BPE | byte-level (259 ids) **or** the repo's own byte-level BPE (`anulm/bpe.py`); released models use `multi32k`, 32,768 ids |
| Position | learned absolute table | **RoPE**, split-halves convention, plus optional **YaRN** (`Rotary`, `apply_rope`) |
| Norm | LayerNorm, post or pre | **RMSNorm**, pre-norm, plus **QK-norm** inside attention |
| Attention | MHA | **GQA** (fused QKV) **or MLA** (latent KV), one config switch; optional **sliding window** per layer |
| FFN | dense, GELU, 4× | **SwiGLU**. Layer 0 is dense; layers 1+ are **MoE** with a sigmoid router, top-k, an optional shared expert and aux-loss-free bias balancing |
| Output | tied to the embedding | **untied** by default (`tie_word_embeddings=False`) |
| Attention kernel | hand-written | `F.scaled_dot_product_attention` (SDPA) |

### The released configuration (real numbers for the book)

Read from `ckpt_base_v2.pt`, the `AnuLM-Base-v2-400M` checkpoint
(`docs/MODEL_CARD_BASE_V2.md`). Evidence level: **verified**, loaded on
2026-10-31.

| field | value |
|---|---|
| `vocab_size` | 32,768 (`data/multi32k.json`) |
| `block_size` | 1,024 |
| `n_layer` / `hidden_size` | 20 / 1,024 |
| `n_head` / `n_kv_head` / `head_dim` | 16 / 4 / 64 (GQA 4:1) |
| `sliding_window` / `max_window_layers` | 256 on layers 0–9, full attention on layers 10–19 |
| `intermediate_size` (dense layer 0) | 2,816 |
| `num_experts` / `num_experts_per_tok` / `moe_intermediate_size` | 24 / 4 / 192, no shared expert |
| `routed_scaling_factor`, `bias_update_rate` | 2.5, 3e-3 |
| `seq_balance_alpha`, `router_z_alpha` | 1e-4, 1e-3 |
| `rope_theta` | 1e6, YaRN off |
| floating-point parameters in the state dict | 397,660,560 |
| step / val loss | 1,899,999 / 3.3523 |

The parameter split below was computed by hand from the config. The audit
script still has to check it.

- Embedding and output table: 2 × 32,768 × 1,024 = 67.1M.
- Attention: 20 layers × (24 × 64 × 1,024 + 1,024²) = 52.4M.
- Dense layer 0: 3 × 1,024 × 2,816 = 8.7M.
- Experts: 19 layers × 24 × 3 × 1,024 × 192 = 269.0M, plus routers.
- Total: ≈ 397.7M, which matches the state dict.
- Active per token: about 173M, since 20 of the 24 experts sit idle in each of the 19 MoE layers.
- KV cache: 2 × 20 × 4 × 64 × 2 bytes = **20 KiB per token** in bf16.

---

## 2. Layout of the repository

The repository root is flat. Grouped by role:

| Role | Files |
|---|---|
| **Core model** | `anulm/model.py` (1,224 lines) |
| **Tokenizer** | `anulm/bpe.py`; `hf_tok.py`, `anulm/hf/hf_tokenizer.py` (conversion to the HF format) |
| **Pre-training** | `anulm/training/train.py`, `anulm/muon.py`, `anulm/bigram_teacher.py`; corpus builders `experiments/build_v2.py`, `anulm/data/mix_corpus.py`, `anulm/data/filter_prose.py`, `fetch_*.py`, `convert_*.py` |
| **Generation** | `AnuLM.generate`, `AnuLM.decode_from_embeds` in `anulm/model.py`; `anulm/inference/sample.py`, `anulm/inference/sample_many.py`, `anulm/inference/ask.py` |
| **Post-training** | `anulm/training/finetune.py` (SFT), `anulm/lora.py`, `anulm/data/make_chat.py`, `make_qa*.py`, `anulm/tasks/decide.py`, `anulm/tasks/rc_pointer.py`, `anulm/bidirectional.py` |
| **Retrieval** | `anulm/inference/rag.py` |
| **Multimodal** | `anulm/multimodal/vision_encoder.py`, `anulm/multimodal/speech_encoder.py`, `anulm/multimodal/speech_vocab.py`, `anulm/multimodal/audio_codec.py`, `omni*.py` |
| **Serving / UI** | `anulm/inference/serve.py`, `anulm/inference/api.py` (OpenAI-compatible), `app*.py` (Gradio), `anulm/inference/langchain_anulm.py`, `web/`, `space/` |
| **Export** | `anulm/hf/export_hf.py`, `anulm/hf/modeling_anulm.py`, `anulm/hf/configuration_anulm.py` (a Hugging Face `transformers` wrapper) |
| **Evaluation** | `eval_*.py`, `anulm/data/make_golden.py`, `anulm/data/make_bench.py`, `anulm/multimodal/omni_eval.py`, `tools_contamination.py`, `tools/eval_noise.py` |
| **Tests** | `tests/test_model.py`: 66 named tests, each run through a `@test` decorator; there is no pytest dependency |
| **Records** | `docs/RESULTS.md` (experiment log, §1–§39+), `docs/ARCHITECTURE.md`, model cards, `logs/` |
| **Course (existing book)** | `course/`: *Build Your Own LLM, from Scratch to Omni*, with 15 chapters, labs, checks and stub code |

Dependencies (`requirements.txt`): the training path needs **only `torch`**.
`safetensors` is used for export, `pyarrow` for parquet. Gradio, FastAPI,
timm, snac and faster-whisper are optional and are imported only by the demo
and multimodal scripts.

---

## 3. Execution path: text in, loss out (training)

```mermaid
flowchart TD
  A["text file / .bin / meta.json"] --> B["train.py main()<br/>load_bytes · torch.load · load_memmap"]
  B --> C["WindowSampler.next()<br/>block_size windows, no replacement"]
  C --> D["make_batch() → x (B,S), y (B,S) = x shifted by 1"]
  D --> E["AnuLM.forward(idx, targets)"]
  E --> F["F.embedding(idx, _table('in')) → (B,S,D)"]
  F --> G["Rotary(S) → cos, sin (S, Dh)"]
  G --> H["Block × n_layer"]
  H --> I["RMSNorm → _head → logits (B,S,V)"]
  I --> J["F.cross_entropy(ignore_index=-1) + router aux"]
  J --> K["backward · clip_grad_norm_ · AdamW/Muon step"]
  K --> L["update_expert_biases()  (outside the loss)"]
  L --> M["evaluate() · atomic_save(best, .last)"]
```

| Stage | Where | Shape |
|---|---|---|
| Corpus → ids | `anulm/training/train.py` `load_bytes` lines 62–98 (byte mode, EOS 257 at `DOC_SEP`); `.bin` from `bpe.py cmd_encode` lines 395–449; memmap `load_memmap` lines 163–173 | 1-D id tensor |
| Window choice | `WindowSampler` lines 104–160 | `ix` (B,) start offsets |
| Batch | `make_batch` lines 237–250 | `x`, `y`: (B, S) int64; `y[t] = x[t+1]` |
| Embedding | `AnuLM.forward` line 896, `_table` lines 966–984 | (B, S, D) |
| Positions | `Rotary.forward` lines 339–343; `_build_inv_freq` 304–322 | cos, sin (S, Dh) |
| Block | `Block.forward` lines 816–824: `x + attn(norm(x))`, then `x + mlp(norm(x))` | (B, S, D) |
| Attention | `GQAttention.forward` lines 451–494 | q (B,H,S,Dh); k,v (B,Hkv,S,Dh) → (B,S,D) |
| Mask | `_attn_mask` lines 365–381; `_sdpa` lines 414–418 (`is_causal`) | (S, T) bool or None |
| FFN | `SwiGLU.forward` line 253 / `MoE.forward` lines 696–707 | (B, S, D) |
| Router | `Router._gate` lines 624–672 | topk_idx (N,k), weights (N,k) |
| Head + loss | `AnuLM.forward` lines 912–928 | logits (B,S,V); scalar loss |
| Optimiser | `make_optimizer` lines 447–474 (decay on 2-D weights only); `anulm/muon.py` | — |
| LR | `lr_at` lines 375–400 (warmup → cosine, or WSD) | — |
| Step | `main` loop lines 850–908 | — |

## 4. Execution path: prompt in, text out (inference)

```mermaid
flowchart LR
  P["prompt"] --> T["BPE.encode / UTF-8 bytes<br/>sample.py"]
  T --> G1["AnuLM.generate()"]
  G1 -->|"prefill: whole prompt, cache=new_cache()"| F1["forward(pending, cache, start)"]
  F1 --> L1["logits[:, -1]  (only the last position)"]
  L1 --> S1["repetition penalty → ÷ temperature → top-k mask → softmax → multinomial"]
  S1 -->|"decode: one token, start += 1"| F1
  S1 --> O["BPE.decode / bytes.decode"]
```

- **KV cache.** It is a list of dicts, one per layer (`new_cache`, line 873).
  GQA stores `k` and `v` for the **Hkv** heads, before they are repeated
  across query groups (lines 470–476). MLA stores the latent `c` and the
  shared rope key `k_pe` (lines 555–563). The cache grows with `torch.cat`,
  and there is no paging.
- **Position continuity.** `Rotary.forward(start=...)` gives absolute
  positions. When the cache is non-empty and a step adds several tokens,
  `_attn_mask` builds the mask explicitly, because SDPA's `is_causal`
  aligns top-left (lines 365–381).
- **Context overflow.** When the context is full, `generate` rebuilds the
  cache from the last `block_size` tokens on every step (lines 1148–1151),
  so cached and uncached decoding agree token for token. The test
  `cached_generation_matches_uncached` holds them to it.
- **Sampling controls in the model.** Temperature, top-k, the CTRL
  repetition penalty, EOS and a streaming callback. There is **no top-p**:
  `anulm/inference/api.py` line 219 accepts `top_p` for OpenAI compatibility and ignores it.
  There is no beam search; `anulm/hf/modeling_anulm.py` lines 285–290 raise on it.
- **Inputs that are not tokens.** `forward(embeds=...)` bypasses the
  embedding table (lines 891–894), and `decode_from_embeds` (lines 1016–1062)
  decodes greedily after an audio or image prefix.

---

## 5. Engineering features beyond the textbook (all present)

| Feature | Where |
|---|---|
| Three interchangeable MoE dispatch paths: sparse, dense, grouped-GEMM | `MoE._route_sparse` / `_route_dense` / `_route_grouped`, lines 709–790 |
| Router computed in fp32, with autocast off | `Router.forward` lines 611–622 |
| Aux-loss-free balancing, sequence balance loss, router z-loss | `Router._gate` lines 647–670; `AnuLM.update_expert_biases` lines 1071–1101 |
| Gradient checkpointing | `AnuLM.forward` lines 902–907 |
| Chunked cross-entropy, so the full logit tensor never exists | `enable_chunked_loss`, `_chunked_loss`, lines 930–956 |
| Merge-tree reparameterisation of the token tables | `merge_tree_matrix` lines 390–411; `_compose`, `_table`, `bake_merge_tree` |
| Scaled init of the residual output projections | `AnuLM.__init__` lines 860–863 |
| bf16 autocast, fused AdamW, `torch.compile` probe | `anulm/training/train.py` lines 645–650, 463, 750–770 |
| Gradient accumulation | `anulm/training/train.py` lines 857–873 |
| Atomic checkpoints, resume with optimizer and sampler state | `atomic_save` lines 32–46; resume lines 788–840 |
| Eval noise accounting (standard error) and strided eval windows | `evaluate` lines 477–533; `strided_windows` 351–372 |
| Interleaved validation split | `interleaved_split` lines 202–216 |
| Muon optimiser (Newton–Schulz) | `anulm/muon.py` |
| Bigram-teacher KL warm start | `anulm/bigram_teacher.py` |
| A memory guard for large models | `fit_memory` lines 1345–1364 |

## 6. Not implemented in this repository

The book has to present these as background or extensions, never as
features of this code.

| Topic | Status |
|---|---|
| Learned absolute or sinusoidal positional embeddings | **Absent.** RoPE only. Teach them as history. |
| LayerNorm | **Absent.** RMSNorm only. |
| Plain multi-head attention as its own class | GQA with `n_kv_head == n_head` *is* MHA (a config choice, not a class). MQA is `n_kv_head = 1`. |
| Hand-written attention kernel or FlashAttention | Absent. SDPA chooses the kernel (`IMPLEMENTATION-DEPENDENT`). |
| Top-p (nucleus) sampling | **Absent** from `generate`. `anulm/inference/api.py` ignores `top_p`. |
| Beam search, speculative decoding, multi-token prediction | Absent. |
| PagedAttention or a paged KV cache, continuous batching | **Absent.** The cache is a per-layer `torch.cat`. `anulm/inference/api.py` serialises requests behind one GPU lock. |
| Quantised inference (INT8/INT4/FP8) | **Absent** for AnuLM. The export stores bf16 weights, and the only int8 anywhere is ctranslate2's Whisper on CPU. A QLoRA-style `quantize` path exists only in the course stub `course/build/post.py`. |
| Distributed training (DDP, FSDP, tensor or pipeline parallel) | **Absent.** Single-GPU by design (`anulm/model.py` line 19). |
| fp16 with `GradScaler` | Absent. Only bf16 autocast. |
| DPO, RLHF or GRPO in the core code | Absent from the repo root. DPO appears in the course stubs (`course/build/post.py`) and RL in `course/build/further.py`. |
| Dropout | The field exists, defaults to 0.0 and is wired into SDPA. It is effectively off. |

## 7. How this was established

- `anulm/model.py`, `anulm/training/train.py`, `anulm/bpe.py`, `anulm/inference/sample.py` were read in full.
- `anulm/muon.py` and `anulm/bigram_teacher.py` were read in part.
- The test names in `tests/test_model.py` were listed.
- `requirements.txt`, the headers of `docs/ARCHITECTURE.md`, the index of
  `docs/RESULTS.md` and the base-v2 model card were read.
- The config was loaded from `ckpt_base_v2.pt`.
- The post-training, multimodal, serving and export modules (25 files) were
  read in full by two read-only inventory agents. Their line numbers were
  spot-checked against the files, and 14 of 14 matched. See
  `REPOSITORY_SOURCE_MAP.md` §§4–5.

**Not yet inspected:** `experiments/` (about 40 scripts), `eval_*.py`,
`app*.py` beyond their headers, `web/` and `space/`.
