# AnuLM architecture

AnuLM is a decoder-only mixture-of-experts transformer, written in plain PyTorch
(`anulm/model.py`, no Hugging Face dependency) and built to train on one GPU.
This page says what each part is, which paper it comes from, and what the
project measured when it trained it. Every measurement cites a section of
`docs/RESULTS.md`.

---

## 1. The model in one table

| part | AnuLM | from |
|---|---|---|
| blocks | pre-norm decoder blocks, RMSNorm, residual stream | Zhang & Sennrich 2019 (RMSNorm) |
| attention | grouped-query attention (GQA) or multi-head latent attention (MLA), chosen per model | Ainslie et al. 2023; DeepSeek-V2 (2024) |
| query/key norm | RMSNorm over each head before the rotation | Dehghani et al. 2023 |
| positions | rotary embeddings, split-halves layout; optional YaRN for longer context | Su et al. 2021; Peng et al. 2023 |
| feed-forward | layer 0 a dense SwiGLU; every later layer a sparse mixture of SwiGLU experts | Shazeer 2020; Shazeer et al. 2017 |
| router | sigmoid scores, top-k, aux-loss-free bias balancing, optional group-limited top-k | DeepSeek-V3 (2024) |
| shared expert | optional, off in the trained models (§4) | DeepSeek-MoE (2024) |
| sliding window | optional, on the lower layers only | Mistral 7B (2023); Qwen2 |
| embeddings | untied input and output tables | |
| tokenizer | the project's own byte-level BPE (`anulm/bpe.py`), 32,768 ids for Hindi + English + Python | Sennrich et al. 2016; Radford et al. 2019 |

## 2. Presets

`AnuLMConfig` in `anulm/model.py`; `python -m anulm train --preset NAME`.

| preset | total / active per token | layers | attention | experts | used for |
|---|---|---|---|---|---|
| `tiny-gqa` | 17.4M / 6.6M | 8 | GQA 6 query / 2 key-value heads | 16 x 96 wide, top-2 + 1 shared | quickstart, tests, CPU |
| `tiny-mla` | 47.4M / 13.3M | 12 | MLA, latent 48 + rotary 24 | 16 x 192, top-2 + 1 shared, 4 groups keep 2 | comparing the two attention types |
| `350m` | 353M / 152M (byte vocab) | 20 | GQA 16 / 4 | 12 x 384, top-3 + 1 shared | the GPU workhorse of the ablations |
| base v2 (`350m` + `--cfg`, `experiments/v2_train.sh`) | 398M / 174M | 20 | GQA 16 / 4, sliding window 256 on layers 0-9 | **24 x 192, top-4, no shared** | AnuLM-Base-v2-400M and everything released on it |
| `1b` | 1,008M / 284M | 24 | as base v2 | 24 x 512, top-4, no shared | the from-scratch 1B base (`docs/PLAN_1B.md`) |

The base v2 column is the configuration the ablations converged on
(`RESULTS.md` §10, §12): fine-grained experts, no shared expert, top-4, and a
sliding window on the lower half.

## 3. Attention

**GQA.** One fused projection makes the queries and a smaller set of key and
value heads; each key/value head serves several query heads (4 in base v2), so
the KV cache is that many times smaller. QK-norm runs before the rotation.

**MLA.** Keys and values are rebuilt from a small shared latent
(`kv_lora_rank` wide) plus one rotary key shared by every head; only the latent
and the rotary key are cached, whatever the head count. Queries and keys are
concatenated, not summed, `q = [q_content | q_rotary]`, so a dot product splits
into a position-free content term and a purely relative position term. AnuLM's
cache stores the latent itself, not the expanded heads.

**What training them showed.** At matched steps MLA wins by 0.034 bits/byte,
as twice the active parameters should; at matched wall-clock time GQA wins
outright, and on Hindi MLA's per-step edge vanished (`RESULTS.md` §1, §2). Both
caches are held to the uncached path token for token, including after a sliding
window fills (§9). The trained models use GQA.

## 4. Positions and long context

Rotary embeddings rotate pairs of dimensions at a ladder of frequencies: fast
pairs wrap many times inside the training window and encode local offset; slow
pairs never complete a turn and encode absolute position. Two ways to reach
longer context:

- **A wide base** (`rope_theta` 1e6): spread the ladder so long positions stay
  in distribution. The GQA presets use it.
- **YaRN** (`--yarn`): keep the fast pairs, divide the slow ones by the
  extension factor, blend linearly between, and raise the attention
  temperature. The temperature belongs on the attention scale, not in the
  cos/sin tables; a bug that put it in the wrong place was caught by the
  position-bucketed evaluation (`RESULTS.md` §15).

AnuLM uses one rotary layout everywhere, split halves (GPT-NeoX). The other
common layout stores pairs interleaved; loading weights from one into the other
without a permutation looks fine on short prompts and decays with distance.

**What training showed.** A θ=1e4 model collapses at 4x its training length
without YaRN (loss 1.54 -> 3.03 across the window) and stays flat with it
(1.68 -> 1.71). A θ=1e6 model extrapolates gently on its own; YaRN is a small
loss at 17M and a gain at 386M (-0.08 at 4x) (§3, §15). A 256-token sliding
window on 10 of 20 layers costs nothing at the training length and extrapolates
better to 2,048 (5.37 vs 5.44), because half the layers never meet an unseen
phase (§10, §15).

## 5. Mixture of experts

Each MoE layer scores every expert with an independent sigmoid, picks the top
k, and combines their outputs with the normalised scores times a fixed scale.

**Aux-loss-free balancing.** Each expert has a bias added to its score for
selection only, never for the combine weights. After every optimizer step the
bias of an overloaded expert goes down and a starved one's goes up
(`update_expert_biases`). Balancing never enters the loss, so it never pulls
against language modelling. A tiny sequence-level balance loss and a router
z-loss (`--seq-balance-alpha`, `--router-z-alpha`) only stop one sequence from
collapsing onto one expert. The router computes in fp32: in bf16, rounding
flipped 3.2% of expert choices on a real checkpoint.

**Group-limited top-k** (`n_group`, `topk_group`): experts in groups, a token
keeps the best groups first, then its top k inside them. It exists to cap
cross-device traffic; on one GPU it is off. The config refuses a top-k larger
than the kept groups can supply.

**Dispatch.** Three implementations route identically: one GEMM per expert
(`sparse`), every expert on every token (`dense`, one compile graph), and
`grouped`, which sorts the token-slot pairs by expert and runs one grouped GEMM
per projection (`F.grouped_mm`): 1.48x faster than per-expert GEMMs on a GPU
(§9).

**What training showed.**

| choice | measured | RESULTS |
|---|---|---|
| bias balancing | works and never enters the loss; imbalance rises to 3-8x in the first 100 steps, then falls to 1.1-1.4x; raising the rate 10x makes it worse (2.59x) | §5, §16 |
| group-limited routing | slower to balance (peak 7x) but finishes lowest (1.13x); on Hindi it damped collapse | §1, §2 |
| shared expert | not earning its 22M parameters at 386M: removing it improved the benchmark by 0.003 bits/byte | §10 |
| fine-grained experts | 24 x 192 top-6 against 12 x 384 top-3 at equal active compute: -0.008 bits/byte; with no shared expert, top-4 and the window, the same loss at 76% of the active compute | §10, §12 |
| a training path | differentiable dispatch; gradient checkpointing takes effect (6.34 GB against 15.75 GB at 353M) | §6 |

## 6. What matters most

Every mechanism above is real and measurable at this scale. None of them comes
within an order of magnitude of the tokenizer and the data. The ledger,
largest lever first (bits/byte, `RESULTS.md`):

| lever | gain |
|---|---|
| a 16k BPE tokenizer instead of bytes, at matched compute | 0.162 (§7) |
| 20x the parameters | 0.110 (§2) |
| more steps on new data | 0.089 (§18) |
| cleaning the corpus | 0.054 |
| a second register (Wikisource beside Wikipedia) | 0.042 (§17) |
| the training recipe | 0.022 |
| fine-grained experts | 0.008 (§10) |

The same 398M shape with a 32k three-script tokenizer learns Hindi, English
and Python on one budget (§22); fine-tuned from that one base it translates
(chrF 41.5 / 43.4 on FLORES-200), answers questions in all three, and as
AnuLM-Omni-400M handles chat, reading, intents, English and Hindi speech,
captions and photo questions in one set of weights (`docs/OMNI.md`). The
architecture is not the limit at this scale; the token budget is. That is why
the next model (`docs/PLAN_1B.md`) is the same design, larger, on a clean corpus.

## Sources

- J. Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models
  from Multi-Head Checkpoints*, 2023.
- DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient
  Mixture-of-Experts Language Model*, 2024 (multi-head latent attention).
- DeepSeek-AI, *DeepSeek-V3 Technical Report*, 2024 (sigmoid routing,
  aux-loss-free balancing, group-limited top-k).
- D. Dai et al., *DeepSeekMoE*, 2024 (fine-grained and shared experts).
- N. Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated
  Mixture-of-Experts Layer*, 2017; N. Shazeer, *GLU Variants Improve
  Transformer*, 2020.
- J. Su et al., *RoFormer: Enhanced Transformer with Rotary Position
  Embedding*, 2021; B. Peng et al., *YaRN: Efficient Context Window Extension
  of Large Language Models*, 2023.
- M. Dehghani et al., *Scaling Vision Transformers to 22 Billion Parameters*,
  2023 (QK-norm); B. Zhang and R. Sennrich, *Root Mean Square Layer
  Normalization*, 2019.
- A. Q. Jiang et al., *Mistral 7B*, 2023 (sliding-window attention).
- R. Sennrich et al., *Neural Machine Translation of Rare Words with Subword
  Units*, 2016; A. Radford et al., *Language Models are Unsupervised Multitask
  Learners*, 2019 (byte-level BPE).
