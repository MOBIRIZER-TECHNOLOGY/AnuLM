# Research references

This file is the book's register of **external** sources: the papers and
official documents that explain why the methods exist and how they were
developed. It is governed by the research requirements of 2026-10-31
(sections 16–22 of the master prompt), summarised in §1–§3 below.

`tools/research_audit.py` reads this file and writes
`technical-review/research-audit.md`. Run it after every change.

---

## 1. Two sources of truth, kept apart

| | Repository facts | External references |
|---|---|---|
| Answers | what *this* implementation does | why a method exists, how it works, where it came from |
| Primary source | `REPOSITORY_SOURCE_MAP.md` | this file |
| Supporting sources | the Python sources, configs, training and inference scripts, `tests/test_model.py`, `docs/` | original papers, official reports, official framework documentation |
| Evidence level on the page | **verified** or **measured** (`STYLE.md` §7) | **cited**, and only once the reference is CHECKED |

Rules:

1. A paper never establishes that a feature exists in the repository. Only
   the source map does.
2. The source map never establishes how a method was meant to work or who
   introduced it. Only a CHECKED reference does.
3. A chapter connects the two explicitly, and only where the implementation
   supports the connection. Where the code departs from the paper, the
   chapter says so (§7 records each departure).

## 2. Status lifecycle

| Status | Meaning | May a chapter use it? |
|---|---|---|
| **CITED** | Identified from a bibliography or from discovery. Metadata not yet checked at the source. | Listed only. Never quoted, never cited as authority. |
| **CHECKED** | The original source was opened; title, first author, year, identifier and venue match; it is relevant; and it supports each claim in its row. | Yes. |
| **CORRECTED** | The metadata was wrong or incomplete and has been fixed from the authoritative source. The change is in §6. | After the fix is validated, it is promoted to CHECKED. Until then, no. |
| **REJECTED** | Invalid, irrelevant, unverifiable or misattributed. | Never. The row stays, with the reason, so the history is kept. |

Corrections never overwrite silently: the original identifier and details
stay in §6 beside the fix.

## 3. Verification procedure (Stage 3)

Open every reference independently and record, in its row:

1. the exact publication title;
2. the first author's name;
3. the publication year (for arXiv, the year of the first version; note a
   later venue year separately);
4. the identifier: arXiv id, DOI, publisher URL or official report id;
5. the venue, where there is one;
6. its relevance to the chapters listed;
7. whether it actually supports each statement in **Claims to support**.
   A claim the source does not support is removed from the row, or the
   reference is replaced.

Use the original paper, the official publication page or the publisher's
record. Search snippets, secondary summaries, AI-generated descriptions and
copied bibliographies are never enough on their own. Then set the status and
the verification date.

### Register fields

| Field | Description |
|---|---|
| ID | Stable key a chapter cites, e.g. `[R12]` |
| Topic (chapters) | The concept, and the `BOOK_PLAN.md` chapters that use it |
| Title | The publication title (verified at CHECKED) |
| First author | Verified first author |
| Year | Publication year |
| Identifier | arXiv, DOI, official URL or report id |
| Primary source | The URL of the original |
| Status | CITED / CHECKED / CORRECTED / REJECTED |
| Verified | Date checked (blank until then) |
| Claims to support | The specific statements the chapters will rest on it |
| Notes | Corrections, limitations, provenance |

---


## 4. Inventory

Stage 3 ran on 2026-10-31. Every entry was opened at its source:

- **arXiv papers (59):** metadata from arXiv's own API (title, authors, first-version date, journal reference,
  comments), and every claim searched for and read in the paper's full text, extracted from the arXiv PDF.
- **Publisher records (3):** Crossref, by title, for R36, R37 and S05.
- **The rest:** the original GPT-2 PDF (R04), the Muon post (R24), the Hugging Face model repositories and their
  commit history (R49, withdrawn 2026-11-07), and the PyTorch 2.11 documentation pages (R50–R53).
- **Added 2026-11-03 (R54–R63, S14–S17):** from the writers' proposals, checked as §4.12 describes.

Venues are recorded only where a primary source states them. **Claims to support** holds only claims found
in the source; claims that were repository facts or unsupported were moved out or narrowed (§6). arXiv
primary sources are `https://arxiv.org/abs/<id>`.

### 4.1 Language models and transformer foundations

| ID | Topic (chapters) | Title | First author | Year | Identifier | Primary source | Status | Verified | Claims to support | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| R01 | Transformer (2, 11–13) | Attention Is All You Need | Vaswani | 2017 | arXiv:1706.03762 | arxiv.org/abs/1706.03762 | CHECKED | 2026-10-31 | scaled dot-product and multi-head attention; the 1/√d_k scaling, argued from unit-variance components giving q·k variance d_k (footnote 4), with the authors' suspicion that large dot products push the softmax into regions of extremely small gradients (§3.2.1); sinusoidal positional encoding; learned positional embeddings tried instead, with nearly identical results; the encoder–decoder structure; post-norm: a residual connection around each sub-layer followed by layer normalisation, LayerNorm(x + Sublayer(x)) (§3.1) | NeurIPS 2017, confirmed at papers.nips.cc. Scaling claims added 2026-10-31 for Ch 11, from the arXiv v7 full text. Post-norm claim added 2026-11-03 for Ch 15 (integration #27), read in the arXiv v7 PDF text §3.1 |
| R02 | Attention for translation (2) | Neural Machine Translation by Jointly Learning to Align and Translate | Bahdanau | 2014 | arXiv:1409.0473 | arxiv.org/abs/1409.0473 | CHECKED | 2026-10-31 | a model that learns to (soft-)search the source for the parts relevant to each target word | ICLR 2015 (arXiv comment) |
| R03 | word2vec (2) | Efficient Estimation of Word Representations in Vector Space | Mikolov | 2013 | arXiv:1301.3781 | arxiv.org/abs/1301.3781 | CHECKED | 2026-10-31 | continuous vector representations of words learned from large corpora |  |
| R04 | GPT-2 (2, 10) | Language Models are Unsupervised Multitask Learners | Radford | 2019 | OpenAI report (no arXiv id) | cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf | CHECKED | 2026-10-31 | a decoder-only language model trained on WebText; byte-level BPE with a base vocabulary of 256; layer normalisation moved to the input of each sub-block; an additional layer normalisation after the final self-attention block; the weights of residual layers scaled at initialisation by 1/√N, N the number of residual layers (§2.3) | year from the PDF's creation date, 2019-02-14. The PDF metadata says 'Proceedings of ICML 2019', a LaTeX template leftover: it is not an ICML paper; cite it as an OpenAI report. Final-norm and 1/√N claims added 2026-11-03 for Ch 15 (integration #27), read in the PDF text §2.3 |
| R05 | BPE for NMT (9) | Neural Machine Translation of Rare Words with Subword Units | Sennrich | 2015 | arXiv:1508.07909 | arxiv.org/abs/1508.07909 | CHECKED | 2026-10-31 | byte pair encoding adapted to segment words into subword units | ACL 2016 (arXiv comment) |

### 4.2 Positional encoding, normalisation and architecture

| ID | Topic (chapters) | Title | First author | Year | Identifier | Primary source | Status | Verified | Claims to support | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| R06 | RoPE (14) | RoFormer: Enhanced Transformer with Rotary Position Embedding | Su | 2021 | arXiv:2104.09864 | arxiv.org/abs/2104.09864 | CHECKED | 2026-10-31 | a rotation matrix applied to queries and keys makes attention depend on relative position; θᵢ = 10000^(−2(i−1)/d); the rotation acts on consecutive pairs (eq. 14–15) | the repo uses split-halves pairing and θ = 1e6 (§7) |
| R07 | YaRN (29) | YaRN: Efficient Context Window Extension of Large Language Models | Peng | 2023 | arXiv:2309.00071 | arxiv.org/abs/2309.00071 | CHECKED | 2026-10-31 | NTK-by-parts interpolation; α = 1, β = 32 found good for the Llama family; temperature √(1/t) = 0.1·ln(s) + 1 (eq. 15), recommended for LLaMA and Llama 2 | the temperature is a fitted recommendation, not a derived law: say so on the page |
| R08 | RMSNorm (15) | Root Mean Square Layer Normalization | Zhang | 2019 | arXiv:1910.07467 | arxiv.org/abs/1910.07467 | CHECKED | 2026-10-31 | normalising by the root mean square, without LayerNorm's re-centring; also proposes partial RMSNorm (pRMSNorm) | NeurIPS 2019 (arXiv comment) |
| R09 | LayerNorm (15) | Layer Normalization | Ba | 2016 | arXiv:1607.06450 | arxiv.org/abs/1607.06450 | CHECKED | 2026-10-31 | normalising with the mean and variance of the summed inputs within a layer, for each training case | not used in `anulm/model.py` (RMSNorm throughout); the speech and image projectors end in `nn.LayerNorm` (`anulm/multimodal/speech_encoder.py` 82, `anulm/multimodal/vision_encoder.py` 115). Note corrected 2026-11-01 (it said "not implemented in the repo") |
| R10 | Pre-norm (15) | On Layer Normalization in the Transformer Architecture | Xiong | 2020 | arXiv:2002.04745 | arxiv.org/abs/2002.04745 | CHECKED | 2026-10-31 | Pre-LN Transformers without the warm-up stage reach results comparable to baselines with less training time | ICML 2020 (arXiv journal-ref). An analysis of pre-norm, which GPT-2 [R04] already used in 2019: not its invention |
| R11 | SwiGLU (16) | GLU Variants Improve Transformer | Shazeer | 2020 | arXiv:2002.05202 | arxiv.org/abs/2002.05202 | CHECKED | 2026-10-31 | gated FFN variants, SwiGLU among them; d_ff reduced by a factor of 2/3 to keep parameters and computation constant |  |
| R12 | GQA (13) | GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints | Ainslie | 2023 | arXiv:2305.13245 | arxiv.org/abs/2305.13245 | CHECKED | 2026-10-31 | grouped-query attention uses an intermediate number of key-value heads, between one and the number of query heads; uptraining from MHA checkpoints by mean-pooling key and value heads, with 5% of the original pre-training compute | EMNLP 2023 (arXiv comment). The repo trains GQA from scratch (§7) |
| R13 | MQA (13) | Fast Transformer Decoding: One Write-Head is All You Need | Shazeer | 2019 | arXiv:1911.02150 | arxiv.org/abs/1911.02150 | CHECKED | 2026-10-31 | keys and values shared across all attention heads, reducing the memory bandwidth of incremental decoding |  |
| R14 | QK-norm (13) | Query-Key Normalization for Transformers | Henry | 2020 | arXiv:2010.04245 | arxiv.org/abs/2010.04245 | CHECKED | 2026-10-31 | ℓ2 normalisation of each query and key along the head dimension before their product, then scaling by a learnable parameter | Findings of EMNLP 2020 (arXiv comment). The repo uses a per-head RMSNorm with a learned gain instead (§7) |

### 4.3 Mixture of experts and advanced architectures

| ID | Topic (chapters) | Title | First author | Year | Identifier | Primary source | Status | Verified | Claims to support | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| R15 | Sparse MoE (17) | Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer | Shazeer | 2017 | arXiv:1701.06538 | arxiv.org/abs/1701.06538 | CHECKED | 2026-10-31 | a softmax gating network with noisy top-k gating; an importance loss to balance the experts | venue not confirmed at a primary source (OpenReview blocked automated access): cite the arXiv version |
| R16 | Switch Transformer (17–18) | Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity | Fedus | 2021 | arXiv:2101.03961 | arxiv.org/abs/2101.03961 | CHECKED | 2026-10-31 | routing each token to a single expert; a load balancing loss | JMLR (arXiv comment) |
| R17 | Router z-loss (18) | ST-MoE: Designing Stable and Transferable Sparse Expert Models | Zoph | 2022 | arXiv:2202.08906 | arxiv.org/abs/2202.08906 | CHECKED | 2026-10-31 | the router z-loss for training stability |  |
| R18 | Aux-loss-free balancing (18) | Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts | Wang | 2024 | arXiv:2408.15664 | arxiv.org/abs/2408.15664 | CHECKED | 2026-10-31 | an expert-wise bias added to the routing scores before top-K, used only for selection; Algorithm 1 updates it by bᵢ ← bᵢ + u·sign(eᵢ) from each expert's load error | the repo's update rule is this one (§7) |
| R19 | MLA (19) | DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model | DeepSeek-AI | 2024 | arXiv:2405.04434 | arxiv.org/abs/2405.04434 | CHECKED | 2026-10-31 | multi-head latent attention; the decoupled RoPE strategy with a shared rope key; at inference W^UK can be absorbed into W^Q and W^UV into W^O, so keys and values need not be computed (§2.1.2); RoPE is incompatible with that absorption, because a position-dependent RoPE matrix would lie between W^Q and W^UK, forcing the keys of all prefix tokens to be recomputed, which is why RoPE is decoupled (§2.1.3); low-rank compression of the queries, to reduce activation memory in training (§2.1.2, eq. 12–13) | corporate author. Absorption, RoPE-incompatibility and query-compression claims added 2026-11-03 for Ch 19 (integration #19), read in the arXiv v1 PDF text; the proposal had been written from memory |
| R20 | DeepSeek-V3 (17–18) | DeepSeek-V3 Technical Report | DeepSeek-AI | 2024 | arXiv:2412.19437 | arxiv.org/abs/2412.19437 | CHECKED | 2026-10-31 | sigmoid affinity scores (eq. 15) with shared and routed experts; node-limited routing; a sequence-wise balance loss | corporate author |

### 4.4 Optimisers and training methodology

| ID | Topic (chapters) | Title | First author | Year | Identifier | Primary source | Status | Verified | Claims to support | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| R21 | Adam (7) | Adam: A Method for Stochastic Optimization | Kingma | 2014 | arXiv:1412.6980 | arxiv.org/abs/1412.6980 | CHECKED | 2026-10-31 | exponential moving averages of the gradient and its square (moment estimates) with initialisation bias correction | ICLR 2015 (arXiv comment) |
| R22 | AdamW (7) | Decoupled Weight Decay Regularization | Loshchilov | 2017 | arXiv:1711.05101 | arxiv.org/abs/1711.05101 | CHECKED | 2026-10-31 | weight decay decoupled from the gradient-based update | ICLR 2019 (arXiv comment) |
| R23 | WSD schedule (7) | MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies | Hu | 2024 | arXiv:2404.06395 | arxiv.org/abs/2404.06395 | CHECKED | 2026-10-31 | the Warmup-Stable-Decay (WSD) learning rate scheduler, for continuous training |  |
| R24 | Muon (7) | Muon: An optimizer for hidden layers in neural networks | Jordan | 2024 | blog post | kellerjordan.github.io/posts/muon | CHECKED | 2026-10-31 | an optimiser for hidden layers whose update is orthogonalised by a Newton–Schulz iteration | dated December 8, 2024 on the page; a blog post, not peer-reviewed: say so on the page |

### 4.5 Training efficiency and scaling

| ID | Topic (chapters) | Title | First author | Year | Identifier | Primary source | Status | Verified | Claims to support | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| R25 | Gradient checkpointing (23) | Training Deep Nets with Sublinear Memory Cost | Chen | 2016 | arXiv:1604.06174 | arxiv.org/abs/1604.06174 | CHECKED | 2026-10-31 | O(√n) memory for an n-layer network at the cost of one extra forward pass, by recomputation |  |
| R26 | Mixed precision (23) | Mixed Precision Training | Micikevicius | 2017 | arXiv:1710.03740 | arxiv.org/abs/1710.03740 | CHECKED | 2026-10-31 | half-precision training with a single-precision copy of the weights and loss scaling | ICLR 2018 (arXiv comment). The repo uses bf16 autocast, no loss scaling (§7) |
| R27 | Scaling laws (24) | Scaling Laws for Neural Language Models | Kaplan | 2020 | arXiv:2001.08361 | arxiv.org/abs/2001.08361 | CHECKED | 2026-10-31 | loss follows power laws in model size, dataset size and compute |  |
| R28 | Chinchilla scaling (24) | Training Compute-Optimal Large Language Models | Hoffmann | 2022 | arXiv:2203.15556 | arxiv.org/abs/2203.15556 | CHECKED | 2026-10-31 | for compute-optimal training, model size and training tokens should be scaled equally: double one, double the other |  |
| R29 | FlashAttention (23, 40) | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | Dao | 2022 | arXiv:2205.14135 | arxiv.org/abs/2205.14135 | CHECKED | 2026-10-31 | an IO-aware exact attention algorithm that uses tiling to reduce reads and writes to GPU high-bandwidth memory | the repo calls PyTorch SDPA, which may choose such a kernel (§7) |

### 4.6 Generation and fine-tuning

| ID | Topic (chapters) | Title | First author | Year | Identifier | Primary source | Status | Verified | Claims to support | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| R30 | Repetition penalty (28) | CTRL: A Conditional Transformer Language Model for Controllable Generation | Keskar | 2019 | arXiv:1909.05858 | arxiv.org/abs/1909.05858 | CHECKED | 2026-10-31 | penalised sampling: discounting the scores of previously generated tokens |  |
| R31 | Nucleus sampling (28) | The Curious Case of Neural Text Degeneration | Holtzman | 2019 | arXiv:1904.09751 | arxiv.org/abs/1904.09751 | CHECKED | 2026-10-31 | Nucleus Sampling (top-p) | ICLR 2020 (arXiv comment). Top-p is absent in the repo (§7) |
| R32 | LoRA (31) | LoRA: Low-Rank Adaptation of Large Language Models | Hu | 2021 | arXiv:2106.09685 | arxiv.org/abs/2106.09685 | CHECKED | 2026-10-31 | frozen weights with trainable rank decomposition matrices; random Gaussian A and zero B, so BA is zero at the start; scaling by α/r; no additional inference latency |  |
| R33 | QLoRA (31, 40) | QLoRA: Efficient Finetuning of Quantized LLMs | Dettmers | 2023 | arXiv:2305.14314 | arxiv.org/abs/2305.14314 | CHECKED | 2026-10-31 | 4-bit NormalFloat (NF4); double quantisation; paged optimisers | absent in the repo root; course code only (§7) |

### 4.7 Evaluation, embeddings and retrieval

| ID | Topic (chapters) | Title | First author | Year | Identifier | Primary source | Status | Verified | Claims to support | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| R34 | Calibration (32) | On Calibration of Modern Neural Networks | Guo | 2017 | arXiv:1706.04599 | arxiv.org/abs/1706.04599 | CHECKED | 2026-10-31 | modern neural networks are poorly calibrated; temperature scaling, a single-parameter variant of Platt scaling, is effective; expected calibration error | ICML 2017 (arXiv comment) |
| R35 | LLM2Vec / MNTP (32) | LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders | BehnamGhader | 2024 | arXiv:2404.05961 | arxiv.org/abs/2404.05961 | CHECKED | 2026-10-31 | bidirectional attention, masked next token prediction, and unsupervised contrastive learning | COLM 2024 (arXiv comment) |
| R36 | BM25 (33) | The Probabilistic Relevance Framework: BM25 and Beyond | Robertson | 2009 | doi:10.1561/1500000019; Foundations and Trends in Information Retrieval 4(1–2) | doi.org/10.1561/1500000019 | CHECKED | 2026-10-31 | the probabilistic relevance framework and BM25 | CORRECTED: the inventory gave FnTIR 3(4); Crossref gives 4(1–2) (§6) |
| R37 | Reciprocal-rank fusion (33) | Reciprocal rank fusion outperforms condorcet and individual rank learning methods | Cormack | 2009 | doi:10.1145/1571941.1572114; SIGIR 2009 | doi.org/10.1145/1571941.1572114 | CHECKED | 2026-10-31 | reciprocal rank fusion of several rankings | CORRECTED: DOI added (§6). The formula is to be quoted from the paper itself, not from this row |
| R38 | Multilingual E5 (33) | Multilingual E5 Text Embeddings: A Technical Report | Wang | 2024 | arXiv:2402.05672 | arxiv.org/abs/2402.05672 | CHECKED | 2026-10-31 | multilingual E5 embedding models in small, base and large sizes | claim narrowed: that `anulm/inference/rag.py` uses the small one is a repository fact (§6) |

### 4.8 Alignment and post-training

| ID | Topic (chapters) | Title | First author | Year | Identifier | Primary source | Status | Verified | Claims to support | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| R39 | InstructGPT / RLHF (34) | Training language models to follow instructions with human feedback | Ouyang | 2022 | arXiv:2203.02155 | arxiv.org/abs/2203.02155 | CHECKED | 2026-10-31 | supervised fine-tuning, a reward model, and PPO with a per-token KL penalty from the SFT model |  |
| R40 | DPO (34) | Direct Preference Optimization: Your Language Model is Secretly a Reward Model | Rafailov | 2023 | arXiv:2305.18290 | arxiv.org/abs/2305.18290 | CHECKED | 2026-10-31 | a preference loss optimised directly against a reference policy, with no explicit reward model | absent in the repo root (§7) |
| R41 | GRPO (34) | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models | Shao | 2024 | arXiv:2402.03300 | arxiv.org/abs/2402.03300 | CHECKED | 2026-10-31 | Group Relative Policy Optimization, a PPO variant that foregoes the critic model and estimates the baseline from group scores | absent in the repo root (§7) |

### 4.9 Multimodal models

| ID | Topic (chapters) | Title | First author | Year | Identifier | Primary source | Status | Verified | Claims to support | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| R42 | SigLIP (36) | Sigmoid Loss for Language Image Pre-Training | Zhai | 2023 | arXiv:2303.15343 | arxiv.org/abs/2303.15343 | CHECKED | 2026-10-31 | a pairwise sigmoid loss for language–image pre-training | ICCV 2023 (arXiv comment) |
| R43 | ViT (36) | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale | Dosovitskiy | 2020 | arXiv:2010.11929 | arxiv.org/abs/2010.11929 | CHECKED | 2026-10-31 | a pure transformer applied directly to sequences of image patches |  |
| R44 | Whisper (36) | Robust Speech Recognition via Large-Scale Weak Supervision | Radford | 2022 | arXiv:2212.04356 | arxiv.org/abs/2212.04356 | CHECKED | 2026-10-31 | speech recognition trained on 680,000 hours of multilingual, multitask weak supervision | claim narrowed: which encoder the repo freezes is a repository fact (§6) |
| R45 | SNAC (35–36) | SNAC: Multi-Scale Neural Audio Codec | Siuzdak | 2024 | arXiv:2410.14411 | arxiv.org/abs/2410.14411 | CHECKED | 2026-10-31 | an RVQ codec whose quantisers operate at different temporal resolutions |  |

### 4.10 Inference and distributed training

| ID | Topic (chapters) | Title | First author | Year | Identifier | Primary source | Status | Verified | Claims to support | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| R46 | PagedAttention (40) | Efficient Memory Management for Large Language Model Serving with PagedAttention | Kwon | 2023 | arXiv:2309.06180 | arxiv.org/abs/2309.06180 | CHECKED | 2026-10-31 | the KV cache divided into blocks of a fixed number of tokens; vLLM | SOSP 2023 (arXiv comment). Absent in the repo (§7) |
| R47 | Speculative decoding (40) | Fast Inference from Transformers via Speculative Decoding | Leviathan | 2022 | arXiv:2211.17192 | arxiv.org/abs/2211.17192 | CHECKED | 2026-10-31 | 2–3× faster decoding of T5-XXL with identical outputs | ICML 2023 (arXiv comment). Absent in the repo (§7) |
| R48 | ZeRO (40) | ZeRO: Memory Optimizations Toward Training Trillion Parameter Models | Rajbhandari | 2019 | arXiv:1910.02054 | arxiv.org/abs/1910.02054 | CHECKED | 2026-10-31 | model states (optimiser states, gradients and parameters) dominate memory, and ZeRO partitions them | absent in the repo (§7) |

### 4.11 Repository-specific and framework references

| ID | Topic (chapters) | Title | First author | Year | Identifier | Primary source | Status | Verified | Claims to support | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| R49 | withdrawn (none) | withdrawn 2026-11-07: an entry for a released model's configuration files, no longer cited | — | — | — | — | REJECTED | 2026-11-07 | none | AnuLM's architecture was first studied against openly released MoE models; each idea is now cited to its original paper (R07, R08, R11, R12, R14, R19, R20) |
| R50 | PyTorch SDPA (11, 23) | torch.nn.functional.scaled_dot_product_attention | PyTorch | 2026 | PyTorch 2.11 documentation | docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html | CHECKED | 2026-10-31 | the SDPA API the repo calls; its reference formula (QKᵀ × scale, a −∞ bias where a boolean mask is False, softmax over the last dimension, dropout, × V); scale defaults to 1/√E; `True` in a boolean mask means take part, the inverse of `nn.MultiheadAttention`'s padding mask; `dropout_p` is always applied, so callers pass 0.0 at evaluation; three implementations (FlashAttention-2, memory-efficient, a C++ reference), chosen automatically, optimised ones on CUDA only; with `enable_gqa=True` the reference implementation repeats key and value heads with `repeat_interleave(query.size(-3)//key.size(-3), -3)`, and the docstring calls GQA "an experimental feature" that works only for the flash and math kernels on CUDA tensors | the `enable_gqa` claims added 2026-11-03 for Ch 13 (integration #27), read from the installed 2.11.0 docstring. Pinned to torch 2.11.0+cu128, the version installed; behaviour claims are quoted from that page. The claims added 2026-10-31 for Ch 11 were read from the installed function's docstring, which the page is generated from (the site refused scripted requests that day). The docstring is out of date about kernels: on this machine's CPU, torch 2.11.0 chooses a fused flash kernel (`_scaled_dot_product_flash_attention_for_cpu`) unless dropout is on, and `SDPBackend` also lists CUDNN_ATTENTION (checked 2026-10-31 by `code-examples/attention_by_hand.py`). Chapters cite R50 only for what the documentation says, and state the observed behaviour separately |
| R51 | PyTorch autocast (23) | Automatic Mixed Precision package - torch.amp | PyTorch | 2026 | PyTorch 2.11 documentation | docs.pytorch.org/docs/2.11/amp.html | CHECKED | 2026-10-31 | the autocast API the repo uses | as R50 |
| R52 | PyTorch torch.compile (11, 23) | torch.compile | PyTorch | 2026 | PyTorch 2.11 documentation | docs.pytorch.org/docs/2.11/generated/torch.compile.html | CHECKED | 2026-10-31 | the compile API the repo uses | as R50 |
| R53 | PyTorch grouped_mm (11, 23) | torch.nn.functional.grouped_mm | PyTorch | 2026 | PyTorch 2.11 documentation | docs.pytorch.org/docs/2.11/generated/torch.nn.functional.grouped_mm.html | CHECKED | 2026-10-31 | the grouped GEMM `_route_grouped` calls | as R50; `anulm/model.py` line 137 also names 2.11 as the version that has it |

### 4.12 Added 2026-11-03 from the writers' register proposals

Proposed in `technical-review/register-additions/` and checked on 2026-11-03: arXiv metadata from arXiv's API;
claims from the abstract, or, where the row says so, from the full text extracted from the arXiv or publisher
PDF; RFC 3629 from the RFC Editor's text. Venues only where the arXiv comment or the publisher states them.

| ID | Topic (chapters) | Title | First author | Year | Identifier | Primary source | Status | Verified | Claims to support | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| R54 | UTF-8 (8) | UTF-8, a transformation format of ISO 10646 | Yergeau | 2003 | RFC 3629; doi:10.17487/RFC3629 | rfc-editor.org/rfc/rfc3629 | CHECKED | 2026-11-03 | UTF-8 encodes each character as a sequence of 1 to 4 octets; the first octet's high-order bits (0, 110, 1110, 11110) give the length, and every following octet has the form 10xxxxxx (§3, the table of octet sequences) | STD 63, Standards Track, November 2003; obsoletes RFC 2279. Integration #9 |
| R55 | Cosine schedule (7) | SGDR: Stochastic Gradient Descent with Warm Restarts | Loshchilov | 2016 | arXiv:1608.03983 | arxiv.org/abs/1608.03983 | CHECKED | 2026-11-03 | a warm restart technique for SGD; within each run the learning rate decays from η_max to η_min by cosine annealing, η_min + ½(η_max − η_min)(1 + cos(π·T_cur/T_i)) (eq. 5, from the full text) | ICLR 2017 (arXiv comment). The abstract does not say 'cosine'; the claim is from §3 of the PDF. The repo's cosine is one run with no restarts. Integration #37 |
| R56 | Gradient clipping (7) | On the difficulty of training Recurrent Neural Networks | Pascanu | 2012 | arXiv:1211.5063 | arxiv.org/abs/1211.5063 | CHECKED | 2026-11-03 | a gradient norm clipping strategy to deal with exploding gradients (abstract) | venue not recorded (not in the arXiv metadata). Integration #37 |
| R57 | bfloat16 (23) | A Study of BFLOAT16 for Deep Learning Training | Kalamkar | 2019 | arXiv:1905.12322 | arxiv.org/abs/1905.12322 | CHECKED | 2026-11-03 | BFLOAT16 represents the same range of values as FP32, and keeping that range is why no hyper-parameter tuning is needed for convergence, unlike FP16 (abstract) | the abstract does not mention loss scaling: never cite it for 'bf16 needs no loss scaler'. Integration #9 |
| R58 | Continuous batching (40) | Orca: A Distributed Serving System for Transformer-Based Generative Models | Yu | 2022 | OSDI '22, pp. 521–538 (no arXiv id) | usenix.org/conference/osdi22/presentation/yu | CHECKED | 2026-11-03 | iteration-level scheduling: the scheduler invokes the engine to run a single iteration of the model on the batch; after each iteration a finished request returns to the client at once and a newly arrived request can be selected to run next (abstract and the full text, USENIX PDF) | 'continuous batching' is the later industry name; the paper does not use it. Integration #32 |
| R59 | Int8 inference (40) | LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale | Dettmers | 2022 | arXiv:2208.07339 | arxiv.org/abs/2208.07339 | CHECKED | 2026-11-03 | Int8 matrix multiplication for the feed-forward and attention projection layers of transformers; vector-wise quantization with separate normalisation constants for each inner product (abstract) | NeurIPS 2022 (arXiv comment). The paper also has a mixed-precision decomposition for outlier features, which the book's per-row int8 row does not model. Integration #32 |
| R60 | 4-bit weights (40) | GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | Frantar | 2022 | arXiv:2210.17323 | arxiv.org/abs/2210.17323 | CHECKED | 2026-11-03 | one-shot post-training weight quantisation of GPT models down to 3 or 4 bits per weight, based on approximate second-order information (abstract) | ICLR 2023 (arXiv comment). Integration #32 |
| R61 | Tensor parallelism (40) | Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism | Shoeybi | 2019 | arXiv:1909.08053 | arxiv.org/abs/1909.08053 | CHECKED | 2026-11-03 | a simple, efficient intra-layer model-parallel approach for training transformer models with billions of parameters (abstract) | 'tensor parallelism' is the later name; the abstract says 'intra-layer model parallel'. Integration #32 |
| R62 | Pipeline parallelism (40) | GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism | Huang | 2018 | arXiv:1811.06965 | arxiv.org/abs/1811.06965 | CHECKED | 2026-11-03 | a pipeline parallelism library that pipelines sub-sequences of layers on separate accelerators, with a batch-splitting pipelining algorithm (abstract) | venue not recorded (not in the arXiv metadata). Integration #32 |
| R63 | Indirect prompt injection (40) | Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection | Greshake | 2023 | arXiv:2302.12173 | arxiv.org/abs/2302.12173 | CHECKED | 2026-11-03 | LLM-integrated applications blur the line between data and instructions; adversaries can inject prompts into data likely to be retrieved (indirect prompt injection) (abstract) | venue not recorded (not in the arXiv metadata). Integration #32 |

---

## 5. Supplementary references for the historical progressions

Proposed in the gap review against requirement 19 and checked in Stage 3 like the rest.

| ID | Topic (chapters) | Title | First author | Year | Identifier | Primary source | Status | Verified | Claims to support | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| S01 | Position embeddings in an encoder (14) | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | Devlin | 2018 | arXiv:1810.04805 | arxiv.org/abs/1810.04805 | CHECKED | 2026-10-31 | input representations summing token, segment and position embeddings; BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers (abstract) | claim narrowed: the text checked doesn't call them 'learned'; use R01 for learned positional embeddings (§6). Bidirectional claim added 2026-11-03 for Ch 2.5 (integration #37), from the arXiv abstract; the abstract does not use the words 'encoder-only' |
| S02 | Position interpolation (29) | Extending Context Window of Large Language Models via Positional Interpolation | Chen | 2023 | arXiv:2306.15595 | arxiv.org/abs/2306.15595 | CHECKED | 2026-10-31 | Position Interpolation: linearly down-scaling position indices to extend RoPE models' context | June 2023, before YaRN (September 2023) |
| S03 | ALiBi (14, 29) | Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation | Press | 2021 | arXiv:2108.12409 | arxiv.org/abs/2108.12409 | CHECKED | 2026-10-31 | attention with linear biases (ALiBi), for input length extrapolation | context only; not in the repo |
| S04 | GELU, the dense FFN (16) | Gaussian Error Linear Units (GELUs) | Hendrycks | 2016 | arXiv:1606.08415 | arxiv.org/abs/1606.08415 | CHECKED | 2026-10-31 | the GELU activation |  |
| S05 | MoE origins (17) | Adaptive Mixtures of Local Experts | Jacobs | 1991 | doi:10.1162/neco.1991.3.1.79; Neural Computation 3(1) | doi.org/10.1162/neco.1991.3.1.79 | CHECKED | 2026-10-31 | the original mixture-of-experts idea | CORRECTED: DOI added (§6). Claims beyond this are to be quoted from the paper |
| S06 | MoE at scale (17) | GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding | Lepikhin | 2020 | arXiv:2006.16668 | arxiv.org/abs/2006.16668 | CHECKED | 2026-10-31 | a position-wise MoE layer with top-2 gating in every other transformer layer |  |
| S07 | Fine-grained and shared experts (17) | DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models | Dai | 2024 | arXiv:2401.06066 | arxiv.org/abs/2401.06066 | CHECKED | 2026-10-31 | fine-grained expert segmentation; shared experts that are always activated | the code supports a shared expert (`num_shared_experts`); the released model uses none |
| S08 | FlashAttention-2 (23, 40) | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | Dao | 2023 | arXiv:2307.08691 | arxiv.org/abs/2307.08691 | CHECKED | 2026-10-31 | better parallelism and work partitioning for the FlashAttention kernel |  |
| S09 | Adapters, before LoRA (31) | Parameter-Efficient Transfer Learning for NLP | Houlsby | 2019 | arXiv:1902.00751 | arxiv.org/abs/1902.00751 | CHECKED | 2026-10-31 | adapter modules for parameter-efficient transfer |  |
| S10 | Speculative sampling (40) | Accelerating Large Language Model Decoding with Speculative Sampling | Chen | 2023 | arXiv:2302.01318 | arxiv.org/abs/2302.01318 | CHECKED | 2026-10-31 | speculative sampling: several tokens generated from each target-model call | February 2023, after R47 (November 2022) |
| S11 | Distillation (Part VII extensions) | Distilling the Knowledge in a Neural Network | Hinton | 2015 | arXiv:1503.02531 | arxiv.org/abs/1503.02531 | CHECKED | 2026-10-31 | soft targets at a raised temperature; gradients from soft targets scale as 1/T², so they are multiplied by T² | NIPS 2014 Deep Learning Workshop (arXiv comment) |
| S12 | PPO (34) | Proximal Policy Optimization Algorithms | Schulman | 2017 | arXiv:1707.06347 | arxiv.org/abs/1707.06347 | CHECKED | 2026-10-31 | an objective with clipped probability ratios |  |
| S13 | PyTorch DDP (40) | PyTorch Distributed: Experiences on Accelerating Data Parallel Training | Li | 2020 | arXiv:2006.15704 | arxiv.org/abs/2006.15704 | CHECKED | 2026-10-31 | bucketing gradients, overlapping computation with communication, skipping gradient synchronisation | VLDB 2020 (arXiv comment) |
| S14 | Recurrent networks (2) | Finding Structure in Time | Elman | 1990 | doi:10.1207/s15516709cog1402_1; Cognitive Science 14(2):179–211 | doi.org/10.1207/s15516709cog1402_1 | CHECKED | 2026-11-03 | recurrent links in which hidden unit patterns are fed back to themselves, giving the network a dynamic memory of prior internal states (abstract) | metadata and abstract from Crossref (publisher deposit); the full text was not opened. The abstract credits the proposal to Jordan (1986): do not call Elman its inventor. Integration #37 |
| S15 | LSTM (2) | Long Short-Term Memory | Hochreiter | 1997 | doi:10.1162/neco.1997.9.8.1735; Neural Computation 9(8):1735–1780 | doi.org/10.1162/neco.1997.9.8.1735 | CHECKED | 2026-11-03 | a recurrent method whose multiplicative gate units control constant error flow, bridging time lags in excess of 1000 discrete-time steps (abstract) | metadata and abstract from Crossref (publisher deposit); the full text was not opened. Integration #37 |
| S16 | Residual connections (15) | Deep Residual Learning for Image Recognition | He | 2015 | arXiv:1512.03385 | arxiv.org/abs/1512.03385 | CHECKED | 2026-11-03 | layers reformulated as learning residual functions with reference to the layer inputs (abstract) | arXiv comment: 'Tech report'; no venue recorded. Integration #27 |
| S17 | Weight tying (10) | Using the Output Embedding to Improve Language Models | Press | 2016 | arXiv:1608.05859 | arxiv.org/abs/1608.05859 | CHECKED | 2026-11-03 | tying a language model's input embedding and output embedding (the topmost weight matrix) is recommended (abstract) | EACL 2017 (arXiv comment: 'To appear'). The repo unties by default (`tie_word_embeddings` False). Integration #9 |

**KV caching has no single originating paper.** It is an engineering practice of autoregressive decoding. The
book describes it from the repository's code and tests and does not attribute an invention to anyone.

---

## 6. Corrections log

The original stays beside the fix. Metadata corrections:

| Date | ID | Field | Was | Now | Source used |
|---|---|---|---|---|---|
| 2026-10-31 | R36 | identifier | FnTIR 3(4) | Foundations and Trends in IR 4(1–2), doi:10.1561/1500000019 | Crossref |
| 2026-10-31 | R37 | identifier | SIGIR 2009, DOI to verify | doi:10.1145/1571941.1572114 | Crossref |
| 2026-10-31 | S05 | identifier | Neural Computation 3(1), DOI to record | doi:10.1162/neco.1991.3.1.79 | Crossref |
| 2026-10-31 | R04 | primary source | official OpenAI PDF, URL to record | cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf | the PDF |
| 2026-10-31 | R49 | identifier | local copy; upstream to identify | the Hugging Face repositories at the revisions named | Hugging Face API and file hashes |
| 2026-10-31 | R50, R51, R52, R53 | identifier | pytorch.org/docs, version to pin | PyTorch 2.11 documentation URLs | the pages themselves |
| 2026-10-31 | R01, R15 | notes | NeurIPS 2017; ICLR 2017 (from memory) | R01 confirmed at papers.nips.cc; R15 venue dropped, unconfirmed | NeurIPS proceedings; OpenReview unreachable |

Claim corrections (requirement 17, step 7):

| Date | ID | Was | Now | Why |
|---|---|---|---|---|
| 2026-10-31 | R04 | decoder-only LM at scale; learned absolute positions; byte-level BPE | decoder-only LM on WebText; byte-level BPE (base vocabulary 256); layer norm moved to each sub-block's input | 'learned absolute positions' not found in the report; R01 supports learned positional embeddings |
| 2026-10-31 | S01 | learned position embeddings | token, segment and position embeddings summed | 'learned' not found in the text checked |
| 2026-10-31 | R07 | attention temperature 0.1·ln(s)+1 | √(1/t) = 0.1·ln(s)+1, recommended for LLaMA and Llama 2 | the paper presents it as a fitted recommendation |
| 2026-10-31 | R10 | pre-norm trains stably without warmup sensitivity | Pre-LN without warm-up reaches comparable results | the paper's own wording |
| 2026-10-31 | R38 | the e5-small model `anulm/inference/rag.py` embeds with | multilingual E5 in small, base and large sizes | the first is a repository fact (requirement 16) |
| 2026-10-31 | R44 | the encoder `anulm/multimodal/speech_encoder.py` uses, frozen | ASR trained on 680,000 hours of weak supervision | the first is a repository fact (requirement 16) |
| 2026-11-03 | R01, R04, R19, R50, S01 | (claims as on 2026-10-31) | widened, each addition dated in the row's notes | writers' proposals (integration #19, #27, #37), each checked at the source before adding |
| 2026-11-03 | R58 | proposed: requests can join and leave between iterations, from the abstract | the same, sourced to the full text | the abstract states only iteration-level scheduling; joining and early return are in the paper body |
| 2026-11-03 | S01 | proposed widening: '(an encoder-only model)' | dropped | the abstract does not say it |

Proposals not added on 2026-11-03: Gage, 'A New Algorithm for Data Compression' (1994) has no primary source
reachable (only secondary pages, which disagree on the venue) and no chapter makes its claim; the R01 FFN
extension (§3.3) was withdrawn by its writer. Both are recorded in `technical-review/register-additions/`.

---

## 7. Research and implementation

The **Repository** column comes from `REPOSITORY_SOURCE_MAP.md`, read at
`@b42998d`, and is verified. The paper side of each difference was confirmed
against the source in Stage 3 (2026-10-31); the claim it rests on is in that
reference's row.

| Feature | Original research | Repository implementation | Difference (confirmed) | Book treatment |
|---|---|---|---|---|
| GQA | R12 (with R13 MQA) | `GQAttention`, `anulm/model.py` 425–494: fused QKV, QK-norm, RoPE, KV cache, `repeat_interleave`; `NATIVE_GQA` switch off by default (377–380) | the paper's GQA is reached by uptraining MHA checkpoints (key and value heads mean-pooled, 5% of pre-training compute); AnuLM trains GQA from scratch. MHA and MQA are config cases (`n_kv_head`) | implemented |
| RoPE | R06 | `Rotary` 278–343, `rotate_half` / `apply_rope` 346–362: split-halves (NeoX) layout, θ = 1e6 in the released model | the paper rotates consecutive pairs with θᵢ = 10000^(−2(i−1)/d); AnuLM pairs dimension i with i + d/2 (an equivalent rotation under a fixed permutation of dimensions) and uses base 1e6 | implemented |
| YaRN | R07 (with S02) | `_yarn_correction_*` 260–268, `yarn_mscale` 270–275 (0.1·mscale·ln(scale) + 1); defaults β_fast 32, β_slow 1 (config lines 109–110); the temperature reaches attention as scale × mscale² (line 449); tests `yarn_blends_the_right_bands`, `yarn_temperature_reaches_attention` | none found: the band edges match the paper's α = 1, β = 32 for Llama, and mscale² = 1/t matches √(1/t) = 0.1·ln(s) + 1. The paper presents the temperature as fitted for LLaMA | implemented |
| RMSNorm | R08 (with R09, R10, R04) | `RMSNorm` 225–239: computed in fp32, learned gain, no mean subtraction; pre-norm blocks (`Block` 797–824) | the paper also proposes partial RMSNorm (pRMSNorm), not used here; LayerNorm is not implemented | implemented |
| SwiGLU | R11 (with S04) | `SwiGLU` 242–253, used as the dense FFN of layer 0 and as each expert's body | the paper shrinks d_ff by 2/3 to match a two-matrix FFN's parameters; AnuLM sets the hidden size by configuration, so the book reports the configured value, not the 2/3 rule | implemented |
| QK-norm | R14 | `GQAttention`: `query_layernorm` / `key_layernorm` are per-head `RMSNorm`s (lines 443–445), applied before RoPE (465–468) | the paper applies ℓ2 normalisation along the head dimension and then one learnable scale; AnuLM uses RMSNorm with a learned gain per dimension, before RoPE | implemented, with a different normaliser |
| MoE | R15, R16, R20 (with S05–S07) | `Router` 585–672 (sigmoid scores, bias for selection only, group-limited top-k), `MoE` 675–708 (routed experts; a shared expert only if configured), three dispatch paths 709–790 | R15's gate is a softmax with noisy top-k and an importance loss; AnuLM follows the later DeepSeek-V3 design (sigmoid affinities, eq. 15 of R20); the code supports a shared expert (S07) but the released model has `num_shared_experts = 0`. Group-limited top-k follows DeepSeek-V3's node-limited routing (R20) | implemented |
| Aux-loss-free balancing | R18 | the per-expert bias, updated after each step; tests `bias_update_*`, `expert_bias_is_not_trainable` | none in the rule: Algorithm 1's bᵢ ← bᵢ + u·sign(eᵢ) is the repo's update; the rate is a configuration value | implemented |
| Router z-loss, sequence balance loss | R17, R20 | in `Router._gate`, off by default; test `router_regularisers_are_off_by_default_and_work_when_on` | off by default in the code; the released run turned both on (`seq_balance_alpha` 1e-4, `router_z_alpha` 1e-3, from `ckpt_base_v2.pt`'s configuration) | partial (off by default; on in the released run) |
| MLA | R19 | `MLAttention` 501–578: latent KV (`kv_lora_rank`), decoupled rope key, cache = latent; test `mla_shares_one_rope_key_across_heads` | the released model uses GQA; MLA is a config option | implemented, not in the released model |
| FlashAttention | R29 (with S08) | `_sdpa` 414–418 calls PyTorch SDPA, which picks a kernel | the repository implements no FlashAttention kernel; whether one runs depends on PyTorch, device and dtype | external (theoretical in the book) |
| Mixed precision | R26 | bf16 autocast | the paper is fp16 with loss scaling; there is no GradScaler here | partial |
| LoRA | R32 (with S09) | `anulm/lora.py` `LoRALinear` 50–84, `apply_lora` / `router_state_dict` / `merge_lora` 87–166; test `lora_starts_as_the_base_model_and_merges_back_exactly` | attention projections by default; saves the experts' biases with the adapters, which a dense-model paper has no reason to discuss | implemented |
| QLoRA | R33 | absent in the repository root; NF4 written from scratch in `course/build/solutions/ch10.py` | no double quantisation or paged optimisers in the course code | course code only |
| Top-p sampling | R31 | absent: `anulm/inference/api.py` accepts the argument and ignores it | | theoretical, with an `EXTENSION` patch |
| PagedAttention | R46 | absent: one dict per layer (`AnuLM.new_cache` 873–875); `anulm/inference/api.py` serialises device work behind one lock | | theoretical |
| Speculative decoding | R47 (with S10) | absent | | theoretical |
| DPO, RLHF, GRPO | R39, R40, R41 (with S12) | absent in the repository root; written from scratch in `course/` Chapters 11 and 19 | | theoretical here; cross-link the course |
| Distillation | S11 | `anulm/bigram_teacher.py` and `train.py --bigram-teacher` (a bigram teacher, no model teacher); a model teacher in `course/ch17` | a teacher with no teacher model is the repo's own variant | implemented (bigram); course code (model teacher) |
| Distributed training | R48 (with S13) | absent in the repository root; data parallelism from scratch in `course/ch20` | | theoretical here; cross-link the course |

`course/` is the repository's lab book, not its implementation: a chapter
that cites `course/` code must label it as course code, never as what
AnuLM does.

---

## 8. Audit status

Written by `tools/research_audit.py` into
`technical-review/research-audit.md`. The book is **not research-complete**
while any CITED reference remains, or any significant claim lacks evidence.
