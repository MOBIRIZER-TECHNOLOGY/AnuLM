---
title: "Appendix J · Revision questions"
part: "Appendices"
accent: slate
pinned: "@2e80b3e"
---

# Appendix J  Revision questions

Sixty questions, each answered in a sentence or two from the chapters' own key points,
with the chapter or section that explains it. Cover the answer column and test yourself,
or use the list to find the way back into the book. Every number and every code name in a
question or answer is checked by `book/code-examples/ch41_audit.py` against the chapters
the row cites; the sentences themselves were checked by reading.

| # | Question | Short answer | See |
|---|---|---|---|
| 1 | How far do counting, a bigram and a small transformer get on Shakespeare bytes? | Unigram 3.35 nats per byte, bigram 2.49, the lab book's transformer 1.75. | 2.3 |
| 2 | What is AnuLM with zero blocks? | A learned bigram: on bytes it lands on the count table's loss (2.487 vs 2.493). | 2.3 |
| 3 | What does one window of S tokens give in training? | S predictions: y is x shifted by one. | 3.1 |
| 4 | What is the released model's held-out loss? | 3.379 nats, 4.875 bits, perplexity 29.3, top-1 right 40.1%. | 3.3 |
| 5 | Where is most of the loss? | In a tail: 21% of tokens at 6+ nats carry 53% of it; the median is 2.54. | 3.3 |
| 6 | Where should a fresh model's first loss sit? | Near ln V, because small initial logits make the distribution near uniform. | 3.4 |
| 7 | Why does `nn.Linear(in, out)` store W as (out, in)? | It computes x Wᵀ over the last axis; every leading axis passes through. | 4 |
| 8 | When does `.view` after a transpose fail? | When strides are not contiguous;`.contiguous()` copies only if needed. | 4 |
| 9 | Why subtract the maximum before softmax? | It changes nothing mathematically and prevents overflow. | 5 |
| 10 | Which temperature fits held-out text best? | About 0.96 (3.372 vs 3.379 at T = 1); sampling at 0.8 trades 0.12 nats for sharper choices. | 5 |
| 11 | What does T ≤ 0 do in `generate`? | Exact greedy: argmax, no draw. | 5, 28 |
| 12 | Why do gradients reach the embeddings cleanly? | The residual stream hands each block's incoming gradient straight to its input. | 6 |
| 13 | Which parameters does `make_optimizer` decay? | Only matrices; norm gains are skipped, and the router bias is a buffer. | 7 |
| 14 | How much memory does AdamW training need for the released model? | 16 bytes per parameter: 6.36 GB, idle experts included. | 7 |
| 15 | What schedule trained the released model? | WSD at 6×10⁻⁴, decaying to 6×10⁻⁵ over the last 16.3%. | 7 |
| 16 | Why do 256 byte ids cover any text? | UTF-8 writes every character as 1–4 bytes. | 8 |
| 17 | Why does Hindi fit less text into a byte window? | 2.48 bytes per character, so under half the text of English. | 8 |
| 18 | How does BPE encode? | Apply the lowest-rank pair first; this equals replaying every merge in order. | 9 |
| 19 | What is `multi32k`? | 32,511 merges over 256 bytes, EOS 32,767, V = 32,768. | 9 |
| 20 | Are AnuLM's embedding tables tied? | No: two (V, D) tables, 67.1M parameters together. | 10 |
| 21 | What happens to rare tokens' rows? | Input rows shrink almost to zero; output rows do not. | 10 |
| 22 | Why divide scores by √d? | Unscaled scores grow as √d and make softmax nearly one-hot at width 64. | 11 |
| 23 | What does QK-norm bound? | Scaled scores, to 8 at initialisation. | 11 |
| 24 | Where does the mask enter, and why there? | As −∞ before the softmax, so each row still sums to 1. | 12 |
| 25 | Which layers use a sliding window? | Layers 0–9, the last 256 positions; the cache still keeps every position. | 12 |
| 26 | How many heads, and how wide? | 16 query heads of 64; 4 shared key/value heads. | 13 |
| 27 | What does the cache hold per token? | The 4 KV heads, normalised and rotated: 20 KiB in bf16; 30 KiB as served. | 13, 27 |
| 28 | Why `repeat_interleave`, not `repeat`? | Query head h must read KV head h // 4; tiling pairs them wrongly and silently. | 13 |
| 29 | Why do RoPE scores depend only on the gap? | Rotating query and key by their positions keeps their relative angle. | 14 |
| 30 | How many RoPE pairs turn once within 1,024 tokens at θ = 10⁶? | 12 of 32. | 14 |
| 31 | How does RMSNorm differ from LayerNorm? | It divides by the RMS and multiplies by a gain; it does not subtract the mean. | 15 |
| 32 | What does pre-norm mean here? | Each sublayer reads a normalised copy and adds to a stream that is never normalised until the end. | 15 |
| 33 | What is SwiGLU? | (SiLU(uWgate) ⊙ uWup) Wdown: gates that scale and leak rather than switch. | 16 |
| 34 | Which layer is dense? | Only layer 0; layers 1–19 hold 456 experts of width 192, 4 used per token. | 16 |
| 35 | How does the router choose? | Independent sigmoid scores; a balancing bias chooses the top 4; weights come from the unbiased scores, normalised, ×2.5. | 17 |
| 36 | Why does the gate run in fp32? | In bf16, 3.17% of expert choices change (10.95% at layer 1). | 17, 23 |
| 37 | How is load balanced? | After each step, each expert's bias moves 3×10⁻³ toward the mean load; the bias never enters the loss. | 18 |
| 38 | What does MLA cache? | A latent and one shared rotary key per token and layer; heads' keys and values are rebuilt from it. | 19 |
| 39 | How many parameters, and how many active? | 397.7M stored; 173.5M used per token. | 1, 20 |
| 40 | What does the model return without targets? | Logits for the last position only, (B, 1, V). | 20 |
| 41 | What is the training mixture? | 53% English, 40% Hindi, 7% Python; 11.31B training tokens. | 21 |
| 42 | Why hold out by document hash? | A tail split of an ordered file tests one source; for base v2 it would have been two-thirds Python. | 21 |
| 43 | Is the logged training loss a measure? | No: one step's batch, noisy. The validation loss with its standard error is. | 22 |
| 44 | What stays fp32 under autocast? | Weights, gradients, AdamW moments, norms and the loss. | 23 |
| 45 | What did gradient checkpointing buy? | Peak 12.26 → 6.64 GiB for about 30% more time. | 23 |
| 46 | How much do default 80-window val sets vary? | 3.2508 to 3.4436 on the released model; ± about 0.067. | 24 |
| 47 | Why is the resume point 3× the model? | AdamW keeps two moments per weight: 4.86 GB vs 1.59 GB. | 25 |
| 48 | What does `atomic_save` guard against, and not? | A kill mid-write; not power loss (no fsync). | 25 |
| 49 | Why can't a falling loss prove the code right? | Without its causal mask, `tiny()` reaches 0.218 on random ids; an honest model stays at 4.159. | 26 |
| 50 | Why is a query never cached? | It is used once, by its own position; past keys and values are final under the mask. | 27 |
| 51 | Can top-k let more than k tokens through? | Yes: it keeps every logit equal to the k-th; on the released model at a third of positions. | 28 |
| 52 | What does YaRN buy at 2,048 tokens? | The last quarter is 2.1 nats better than plain RoPE; flat to 4,096 at 0.012–0.040 nats inside the trained range. | 29 |
| 53 | What is instruction tuning here? | The pre-training loss on prompt–answer pairs, with prompt targets set to −1. | 30 |
| 54 | Why must the answer keep its leading space and end with EOS? | So training ids match serving ids, and the model learns to stop. | 30 |
| 55 | How big are the released model's attention adapters at r = 16? | 1,474,560 numbers: 0.37% of the model, about 5.9 MB in fp32. | 31 |
| 56 | LoRA or full fine-tune at 398M? | A full fine-tune beat every LoRA run on quality. | 31.7 |
| 57 | Does temperature scaling change a classifier's predictions? | No, only its confidence; one τ fitted on held-out NLL. | 32 |
| 58 | What does reciprocal-rank fusion add? | 1/(60 + rank) from each list; it ignores scores. | 33 |
| 59 | What does every preference method need? | An answer's log-probability: Chapter 30's masked loss, summed. | 34, H |
| 60 | Does bf16 export hurt? | Held-out loss moved −0.00004 nats; rounding router biases changed half the expert choices for 0.003 nats. | 39 |

`VERIFIED` · numbers and code names against the cited chapters, by `ch41_audit.py`.
