# How Sarvam 30B and 105B are built

Findings from reading Sarvam AI's released modelling code and configs. Every
number here comes from `sarvam/sarvam-*/config.json` or the modelling files,
not from marketing material. Where a claim comes from a blog post or press
coverage instead, it says so.

Line references are into the downloaded originals under `sarvam/`. The annotated
companions in that directory walk through the same code with tensor shapes on
every line.

Sections 1–9 are the reading. Section 10 is what happened when the same
architecture was rebuilt at 1/1000th scale as AnuLM (the code at the root of this repository) and trained —
which of these mechanisms held up, which did not, and how much any of them
matter next to the tokenizer and the data.

---

## 1. Provenance

| | |
| --- | --- |
| Released | February 2026, Apache 2.0 |
| Trained | Entirely in India on IndiaAI Mission compute — 4,096 NVIDIA H100 SXM via Yotta |
| Pretraining | 16T tokens (30B), ~12T (105B) |
| Post-training | In-house SFT, then **asynchronous GRPO with KL regularization deliberately dropped** |
| Tokenizer | 262,144 tokens, 22 scheduled Indian languages across 12 scripts |
| Lineage | Sarvam-1 (2B, Oct 2024, from scratch) → Sarvam-M (24B, May 2025, a Mistral-Small post-train) → these two (from scratch) |

The models are **DeepSeek-lineage**. The 105B file says so in its header:
*"based on Llama and Deepseek MoE implementations… modified to accommodate
Sarvam's MLA MoE architecture."* MLA, fine-grained MoE with a shared expert, the
sigmoid `noaux_tc` router — all DeepSeek-V2/V3 mechanisms. The original
contributions are the tokenizer, the Indic data pipeline, and doing the whole
run domestically.

---

## 2. Shape

Both are sparse Mixture-of-Experts decoders. Shared: `hidden_size` 4096, 128
experts + 1 shared expert, layer 0 dense (`first_k_dense_replace: 1`), QK-norm,
sigmoid router with aux-loss-free balancing, `routed_scaling_factor` 2.5, vocab
262,144, untied embeddings.

| | **Sarvam 30B** | **Sarvam 105B** |
| --- | --- | --- |
| class | `SarvamMoEForCausalLM` | `SarvamMLAForCausalLM` |
| total params | ~32 B | ~106 B |
| active per token | **2.4 B** non-embedding | **10.3 B** |
| layers | 19 | 32 |
| attention | **GQA**, 64 q-heads / 4 kv-heads, `head_dim` 64 | **MLA**, 64 heads |
| MLA dims | — | `kv_lora_rank` 512, `q_head_dim` 192 (128 noPE + 64 RoPE), `v_head_dim` 128 |
| experts per token | top-**6** of 128 | top-**8** of 128 |
| `moe_intermediate_size` | 1024 (hidden/4) | 2048 (hidden/2) |
| dense layer-0 FFN | 8192 | 16384 |
| long context | `rope_theta` **8e6**, no scaling, 65K eval | `rope_theta` 1e4 + **YaRN factor 40**, 128K |
| group-limited routing | disabled (`n_group=1`) | **16 groups, keep 2** — see §5 |

Parameter arithmetic checks out against the published figures. Counting SwiGLU
experts as `3·D·I` over 18/31 MoE layers plus one dense layer, attention, and
untied embeddings: **106.0 B / 10.25 B active** and **32.1 B / 2.36 B**. So
"30B" and "105B" are round-number product names; the configs give 32B and 106B.

Per token only **9 of 129** expert MLPs fire in the 105B (8 routed + 1 shared),
**7 of 129** in the 30B. That ratio is the whole economic argument.

---

## 3. Attention: two different answers

### MLA (105B) — `sarvam-105b/modeling_sarvam_moe.py:467`

One projection, `kv_a_proj_with_mqa`, emits `kv_lora_rank + qk_rope_head_dim`
= 512 + 64 = **576 values per token, total, for all 64 heads**. That is the
entire KV cache. `kv_b_proj` expands 512 → `64 × (128 + 128)` at compute time.

Two details that matter:

- **The positional key has head dimension 1.** It is MQA-style, one copy shared
  by every head — that's the `_with_mqa` in the projection's name. It is also
  why RoPE can be applied *before* the latent is expanded: the rotation touches
  only the 64-d shared slice, so the 512-d latent stays position-free and
  cacheable.
- **Query and key are concatenated, not summed**: `q = [q_nope ‖ q_pe]`. So
  `q·k` decomposes into `⟨q_nope, k_nope⟩ + ⟨q_pe, k_pe⟩` — a position-free
  content term plus a purely relative positional term. Keys are 192-d, values
  128-d; that asymmetry breaks any code assuming one `head_dim`.

**Caveat that will mislead you if you benchmark the released file:** it caches
the *expanded* keys and values (line 610) — `64 × (192+128) = 20,480` values per
token per layer. Real MLA caches 576, which is 35.6× smaller and is where
`config.json`'s otherwise-unused `head_dim: 576` comes from. vLLM and SGLang get
there by absorbing `kv_b_proj` into `q_proj` and `o_proj` so `kv` is never
materialised. Same algebra, reassociated. **The HF file shows the math, not
MLA's memory profile.**

### GQA (30B) — `sarvam-30b/modeling_sarvam_moe.py:371`

Conventional by comparison. Fused QKV of width `(64 + 2·4) × 64 = 4608`, split
on the head axis, QK-norm (RMSNorm over the 64-d head, before RoPE), then
`repeat_kv` broadcasts 4 KV heads to 64. Unlike the 105B file, the shipped code
genuinely realises its saving: it caches 4 heads and expands afterward.

**KV cache per token per layer:**

| | values |
| --- | --- |
| 30B GQA | 512 |
| 105B MLA (absorbed, as served) | 576 |
| 105B MLA (as the HF file runs) | 20,480 |

Across full depth: 9,728/token for the 30B against 18,432 for the 105B — under
2× the cache for 3.3× the parameters. That is what MLA buys, and why the 105B
gets the 128K window.

---

## 4. Long context: opposite strategies

RoPE's frequency ladder is two mechanisms, not one. Fast dims wrap many times
inside the training window, so they encode *local offset*; squash them and
neighbouring tokens blur. Slow dims don't complete one rotation, so they encode
*absolute position*; extend past training and you request phases never seen.

- **30B: pure extrapolation.** `rope_theta = 8e6`, `rope_scaling: null`. Spread
  the ladder wide enough during training that long positions are in
  distribution. Works only because they trained that way.
- **105B: YaRN (NTK-by-parts).** `rope_theta = 1e4`, factor 40, from a 4096 base
  window to 131072. Keep the fast dims, divide the slow dims by 40, ramp
  linearly between.

Worked out for the 105B's config (`qk_rope_head_dim` 64 → 32 frequency pairs):

```
correction_dim(beta_fast=32) = 10.4722 → low  = 10
correction_dim(beta_slow=1)  = 22.5134 → high = 23
```

11 of 32 dims untouched, 13 blended, 8 fully interpolated. Dim 10 turns 36.7×
over the 4096 window (just past `beta_fast=32`); dim 23 turns 0.9× (just under
`beta_slow=1`). The cutoffs land exactly where asked.

**Three traps here:**

1. **The temperature term is in a different class.** `_mscale` in the rotary is
   `get(40, mscale) / get(40, mscale_all_dim)` = **exactly 1.0**, so the cos/sin
   tables are unscaled. The correction lives in `SarvamMLAAttention.__init__:522`
   instead: `softmax_scale = 192^-0.5 × 1.36889² = 0.13523`, a 1.874× gain
   compensating for attention entropy at 40× context. Read only the rotary file
   and you would conclude the term was unused.
2. **`beta_fast`/`beta_slow` are measured against `original_max_position_embeddings`
   (4096), never the 131072 target.** Most-often-mis-set field when copying a
   `rope_scaling` block between models.
3. **`factor=40` for a 32× extension** (131072/4096). Deliberate margin; nothing
   cross-checks them.

---

## 5. MoE routing

`MoEGate.forward` (105B :293) is DeepSeek-V3's `noaux_tc`, and the **sigmoid is
load-bearing**. Because experts are scored independently rather than competing
in a softmax, you can add a per-expert `e_score_correction_bias` — nudged down
for overloaded experts, up for starved ones — without distorting the others.

Crucially the bias steers **selection only**: `topk_weight` is gathered from the
*unbiased* `scores`. Balancing never enters the loss. No auxiliary
load-balancing term fighting the language-modelling objective.

Dispatch is one line: `idxs = topk_ids.view(-1).argsort()`, then `x[idxs // k]`.
Sorting assignments by expert makes each expert's work contiguous — 128 dense
GEMMs instead of per-token gather — then `new_x[idxs] = outs` unsorts.

### The `n_group` trap

Same gate code, opposite behaviour:

- **30B**: `n_group=1, topk_group=1`, explicit in both `config.json` and the
  config class. Grouping is a no-op — plain top-6 of 128.
- **105B**: `n_group` and `topk_group` appear **nowhere** — not in
  `config.json`, not in `SarvamMLAConfig.__init__`. The gate reads them via
  `getattr(config, "n_group", num_experts // 8)` → **16** and
  `getattr(config, "topk_group", 2)` → **2**.

So the shipped 105B routes into 16 groups of 8, keeps the best 2 groups (16
candidate experts), then takes top-8 of those. **All 8 experts for a token must
come from 2 of 16 groups** — a hard constraint on the router that exists only as
a getattr default. Add an `n_group` field to that config for any other purpose
and you silently change the model's routing.

Group-limited routing exists for device placement: with experts sharded across
nodes, unconstrained top-k could touch every node for one token. Capping groups
caps the all-to-all fan-out.

---

## 6. The released 105B file cannot train

Four independent blocks, worth knowing before anyone plans a fine-tune:

1. `MoEGate.forward:305` — `assert not self.training`
2. `moe_infer` is decorated `@torch.no_grad()`
3. `SarvamMLAMoE.forward`'s if/else calls `moe_infer` on **both** branches, under
   a comment conceding *"in practice, you'd want a more sophisticated training
   implementation"*
4. `supports_gradient_checkpointing = True` and `self.gradient_checkpointing` is
   initialised, but the model loop **never calls** `_gradient_checkpointing_func`.
   `gradient_checkpointing_enable()` reports success and does nothing.

The 30B file has a working training path (boolean-mask gather over
`repeat_interleave`) and real gradient checkpointing.

---

## 7. Other things that will bite you

- **The 105B is eager-attention only.** `_supports_flash_attn_2 = False # Not
  implemented yet`, no SDPA subclass, no dispatch table, and
  `_prepare_4d_causal_attention_mask` unconditionally. It materialises
  `(B, 64, S, T)` attention weights — at 128K that is ~2.2 PB per batch element.
  The 128K window is real but belongs to vLLM/SGLang, not this file.
- **`.generate()` asymmetry.** The 30B declares `GenerationMixin`; the 105B does
  not. Since Transformers v4.50 `PreTrainedModel` no longer provides it, and
  `config.json` pins 4.57.2 — consistent with the 105B still carrying
  hand-written `prepare_inputs_for_generation` and `_reorder_cache`. *Not
  verified empirically here* (no `transformers` install); check on yours.
- **`_tied_weights_keys = ["lm_head.weight"]` is inert in both** — both configs
  set `tie_word_embeddings: false`, and HF only acts on the attribute when the
  flag is true. Head and embedding are two separate 262144×4096 matrices, 2.15 GB,
  6.7% of the 30B's parameters.
- **Per-layer RoPE in the 105B.** Each attention layer owns a rotary module and
  builds a table at `max_position_embeddings` on construction: 67.1 MB per layer
  × 32 = **2.15 GB allocated and immediately orphaned** at load.
- **The 105B permutes rope dims before rotating** (`view(...,32,2).transpose(4,3)`),
  the 30B does not. The 105B checkpoint stores rope dims interleaved (GPT-J
  convention) while `rotate_half` expects split-halves (GPT-NeoX). Port the
  weights without that permutation and the model looks fine on short prompts and
  decays with distance.
- **`max_window_layers: 19` in the 30B config is a dead field.** It is
  Qwen2's knob for "sliding window on the first K layers", and the 30B ships it
  set to its own layer count — but with no `sliding_window` alongside it, and
  `modeling_sarvam_moe.py` never reads either. Every layer attends to the full
  context. AnuLM implements the semantics the field implies
  (`--sliding-window W --max-window-layers K`) and §10 has what it costs: nothing.
- **Naming is the practical hazard.** Same company, same release, same filename:
  `embed_tokens`/`self_attn`/`SarvamMLA*` versus
  `word_embeddings`/`attention`/`SarvamMoE*`. Nothing that walks one state dict
  walks the other. Treat them as two codebases.

---

## 8. Reported benchmarks

From the model cards — not reproduced here.

| | MMLU | Math500 | HumanEval | BrowseComp | τ²-bench | SWE-Bench |
| --- | --- | --- | --- | --- | --- | --- |
| Sarvam 105B | 90.6 | 98.6 | — | 49.5 | 68.3 | 45.0 |
| Sarvam 30B | 85.1 | 97.0 | 92.1 | 35.5 | — | — |

---

## 9. The wider stack

The LLMs are one layer. The Indic-specific work is arguably the moat: **Saaras
v3** (ASR, 23 languages, with transcribe/translate/verbatim/translit/codemix
modes), **Bulbul v3** (TTS tuned for Indian prosody), **Mayura** and
**Sarvam-Translate**, **Sarvam Vision** (OCR, 23 languages), **Sarvam Edge**
(on-device). Products: **Samvaad** (voice agents, on the 30B) and **Indus**
(agent suite, on the 105B).

The tokenizer deserves emphasis. Sarvam-1's card reports fertility of 1.4–2.1
tokens per word across Indic languages, 2–4× better than Llama/Gemma tokenizers.
Every saved token is attention cost, KV cache, and latency. `RESULTS.md`
measures both sides of that. The *absence* of such a tokenizer (§2): byte-level
Hindi runs at 2.48 bytes per character, so a 512-byte window holds only ~206
Devanagari characters. And the *presence* of even a crude one (§7): a 16k-token
byte-level BPE trained in ten seconds on 12 MB of Hindi reaches 1.34 tokens per
word — inside Sarvam's reported range — and at identical compute is worth 0.162
bits/byte, more than a 20× parameter increase bought (0.110). It is the largest
single lever in the whole experiment log, ahead of anything in §2–§5.

---

## 10. Rebuilt at 1/1000th scale: what held up

AnuLM (`model.py` and friends at the repository root) re-implements §2–§5 as trainable pure PyTorch (no HuggingFace)
and trains it, mostly on Hindi Wikipedia and Wikisource. The presets keep each
real model's *ratios* — `moe_intermediate_size` as a fraction of hidden, the
dense-layer FFN, the KV-to-query ratio, `n_group`, `rope_theta` — not its size.
`nano_350m` is the GPU workhorse (sized to an 8 GB card, RESULTS §6); the
"combo" column is the configuration the ablations converged on, run as `--cfg`
overrides of that preset, and is the model to build on.

| | 30B → `nano_30b` | 105B → `nano_105b` | `nano_350m` | combo (RESULTS §12) |
| --- | --- | --- | --- | --- |
| total / active params | 32.1B / 2.36B → 17.4M / 6.6M | 106B / 10.3B → 47.4M / 13.3M | 386M / 185M † | 364M / 140M |
| layers | 19 → 8 | 32 → 12 | 20 | 20 |
| attention | GQA 64q/4kv → 6q/2kv, `head_dim` 64 | MLA, latent 512 + rope 64 → 48 + 24 | GQA 16q/4kv, d 64 | same |
| experts | 128 top-6 +1 shared → 16 top-2 +1 | 128 top-8 +1 → 16 top-2 +1 | 12×384 top-3 +1 | **24×192 top-4, no shared** |
| `moe_intermediate_size` | hidden/4 → hidden/4 | hidden/2 → hidden/2 | 3·hidden/8 | hidden·3/16 |
| dense layer-0 FFN | 2·hidden → 2·hidden | 4·hidden → 4·hidden | 2816 | 2816 |
| group-limited routing | off → off | 16 keep 2 (getattr) → 4 keep 2 | off | off |
| RoPE | θ 8e6, no scaling → θ 1e6 | θ 1e4 + YaRN 40 → θ 1e4, YaRN on demand | θ 1e6 | θ 1e6 |
| sliding window | dead field (§7) → off | — | off | **256 on layers 0–9** |
| tokenizer | 262k, 22 languages → bytes or 16k BPE | same | 16k BPE | 16k BPE |

† with the 16k BPE head; 353M / 152M with the byte vocabulary. Sparsity is
milder than the originals throughout — 16 experts top-2 is 12.5% where the real
models route ~5% — because top-1 of 16 would be degenerate.

Faithful: the sigmoid router, the bias-only balancer with the bias kept out of
the combine weights, group-limited top-k, the shared expert and dense layer 0,
MLA's single `kv_a_proj_with_mqa` with RoPE applied before expansion, QK-norm
before RoPE, untied embeddings, fp32 router logits, and YaRN with the
temperature term on the attention scale. Deliberately different: it trains,
its KV cache actually caches the 576-style latent rather than the expanded
heads, RoPE is split-halves throughout, and there is no expert parallelism.
The repository `README.md` has the full list and the module-to-line mapping.

### What the small-scale runs say about each choice

Every row is a measurement in `RESULTS.md`; the section numbers below
refer to that file. Read the deltas as rankings, not as absolute quality —
these are 17M–386M models.

| design choice (above) | what AnuLM measured | RESULTS |
| --- | --- | --- |
| **MLA vs GQA** (§3) | At matched steps MLA wins by 0.034 bits/byte, as 2× the active parameters should. At matched wall clock GQA wins outright, and on Hindi MLA's per-step edge vanished. The same trade the two real models make, landing the same way round. | §1, §2 |
| **KV cache** (§3) | Both caches implemented and held to the uncached path token-for-token, including after the window fills. MLA caches `kv_lora_rank + qk_rope_head_dim` per token regardless of head count. The cache is worth ~1.35× at decode; the rest is ~1,000 kernel launches per token (33 tok/s at 386M). | §9 |
| **YaRN vs a wide `rope_theta`** (§4) | Reproduced exactly. The θ=1e4 preset collapses at 4× its training length without YaRN (loss 1.54 → 3.03 across the window) and is flat with it (1.68 → 1.71). The θ=1e6 preset extrapolates gently on its own; YaRN is a small net loss at 17M and a gain at 386M (−0.08 at 4×), where slow-frequency heads carry more. The temperature term has to reach the attention scale, not the tables — a GQA bug did exactly what trap 1 in §4 warns of and was caught by the position-bucketed eval. | §3, §15 |
| **Sigmoid + bias balancing** (§5) | Works, and never enters the loss. Imbalance *rises* before it falls — 3–8× in the first 100 steps, 1.1–1.4× by the end — because a ±1e-3 step needs time to overtake the router's early drift. The `sign()` rule oscillates around its fixed point rather than converging (period 2, measured), and raising the rate 10× makes balance *worse* (2.59×). With 24 experts the floor is ~1.8× and costs no measurable loss. | §5, §16 |
| **Group-limited routing** (§5) | 4 groups keep 2 makes balancing slower (peak 7×) but not worse: it finishes lowest of the three presets at 1.13×. On Hindi it damped collapse rather than amplifying it, against the prediction going in. The config rejects a top-k larger than the surviving groups can supply — the footgun the 105B's `getattr` defaults leave open. | §1, §2 |
| **Shared expert** (§2) | Not earning its 22M parameters at 386M on this data: removing it *improved* the benchmark by 0.003 bits/byte. Kept in both real models; at 128 experts and 16T tokens the trade may differ. | §10 |
| **Fine-grained experts** (§2) | 24×192 top-6 against 12×384 top-3 at the same active FLOPs: −0.008 bits/byte, the largest single architecture move, at 1.57× the wall clock from doubled launch count. Combined with no shared expert, top-4 and the window: the same loss at 76% of the active compute. DeepSeek-V3's argument, reproduced. | §10, §12 |
| **Sliding window on the lower layers** (§7) | The dead `max_window_layers` field, made real: a 256 window on 10 of 20 layers costs nothing at the 512 training length and extrapolates *better* to 2048 (5.37 vs 5.44 naive), because half the layers never meet an unseen phase. The cheapest KV-cache reduction available. | §10, §15 |
| **Expert dispatch** (§5) | The `argsort` dispatch as one grouped GEMM per projection (`F.grouped_mm`): 1.48× faster than per-expert GEMMs on a GPU, one Dynamo graph where `nonzero` gives 8 breaks. | §9 |
| **A training path** (§6) | Trains, with differentiable `index_select` dispatch; gradient checkpointing takes effect (6.34 GB against 15.75 GB at 353M) instead of silently doing nothing. | §6 |
| **The tokenizer** (§9) | 16k byte-level BPE, 1.34 tokens/word, −0.162 bits/byte at matched compute. More than 20× parameters. | §7 |
| **Data over architecture** | Best model: the combo configuration on Wikipedia + Wikisource for 36k steps, 0.588 / 0.648 bits/byte on the two benchmarks. Mixing in a second register was worth 0.042 averaged over both; 3× steps on mostly-unseen data 0.089; doubling the steps again, warm-restarted from the finished run, another 0.030. | §17, §18, §21 |
| **Accuracy, not just compression** | On a 600-item Hindi cloze golden set built from the held-out text, the best model picks the right word from four 85.8% of the time (chance 25%) and reproduces the exact word 11.2%; the Wikipedia-only model 76.7% / 6.3%, with the whole gap on literature. Fluent, not factual: every date and figure in its samples is invented. | §19 |
| **Post-training** (§1: SFT then GRPO) | The SFT half, at nano scale: 56k question-answer pairs mined from the corpus's own lead sentences, loss on answers only, 15–20 minutes. The model learns the format completely (names the subject asked about 86% of the time, from 12%) and learns few new facts (answer-recognition 43% → 51% against a 25% floor). Doubling the pretraining underneath it moved compression by 0.03 bits/byte and recognition by one point: instruction tuning reveals what pretraining put in, and what pretraining puts in about any one subject is bounded by how often the corpus mentions it. | §20, §21 |

The ledger, largest lever first: tokenizer 0.162, more steps on new data
0.089, 20× scale 0.110 (on one benchmark), corpus cleaning 0.054, a second
register 0.042, the training recipe 0.022, fine-grained experts 0.008, 3.3×
unique data at fixed compute 0.007. Every mechanism in §2–§5 is real and
reproduces at nano scale; none of them is within an order of magnitude of the
tokenizer and the data. That is consistent with what §1 says about where
Sarvam's own contribution lies.

## Sources

Downloaded configs and modelling code (`sarvam/`);
[sarvam-105b](https://huggingface.co/sarvamai/sarvam-105b) and
[sarvam-30b](https://huggingface.co/sarvamai/sarvam-30b) model cards;
[Sarvam's release blog](https://www.sarvam.ai/blogs/sarvam-30b-105b);
[Sarvam docs](https://docs.sarvam.ai/api/getting-started/models);
[TechCrunch coverage](https://techcrunch.com/2026/02/18/indian-ai-lab-sarvams-new-models-are-a-major-bet-on-the-viability-of-open-source-ai/).
