---
title: "Appendix K · AnuLM on one page"
part: "Appendices"
accent: slate
pinned: "@2e80b3e"
---

# Appendix K  AnuLM on one page

The released model, AnuLM-Base-v2-400M (`ckpt_base_v2.pt`), in the numbers the book
returns to most. Every value is the book's; the last column gives the chapter or appendix
section that derives it, and `book/code-examples/ch41_audit.py` checks that each number
appears there.

## K.1  Shape

| Quantity | Value | See |
|---|---|---|
| Vocabulary V (`multi32k` BPE) | 32,768 (EOS 32,767) | 9, B.1 |
| Context `block_size` | 1,024; YaRN flat to 4,096 | 29 |
| Layers L, width D | 20, 1,024 | B.1 |
| Query heads × width; KV heads | 16 × 64; 4 (GQA) | 13 |
| Sliding window | 256 tokens, layers 0–9 | 12 |
| Feed-forward | layer 0 dense SwiGLU, I = 2,816; layers 1–19: 24 experts of width 192, 4 per token, no shared expert | 16–17 |
| Router | sigmoid scores, bias-only balancing (3×10⁻³ per step), weights ×2.5, fp32 | 17–18 |
| Position | RoPE, θ = 10⁶, split halves; QK-norm on | 11, 14 |
| Embedding tables | untied, 67.1M together | 10 |
| Parameters | 397.7M stored; 173.5M active per token | 1, 20 |

## K.2  Numbers to remember

| Quantity | Value | See |
|---|---|---|
| Held-out loss | 3.379 nats · 4.875 bits · perplexity 29.3 | 3 |
| Top-1 accuracy | 40.1% | 3 |
| KV cache per token | 20 KiB bf16; 30 KiB as served | 13, 27 |
| Training memory (AdamW) | 16 B/param = 6.36 GB; step peak 12.26 GiB (6.64 with checkpointing) | 7, 23 |
| Checkpoint sizes | model 1.59 GB; resume point 4.86 GB | 25 |
| Corpus | 11.31B tokens: 53% EN, 40% HI, 7% Python | 21 |
| Schedule | WSD, 6×10⁻⁴ → 6×10⁻⁵ over the last 16.3% | 7 |
| Router in bf16 | 3.17% of expert choices change | 17, 23 |
| LoRA r = 16, attention | 1,474,560 numbers (0.37%) | 31 |
| Tiny CPU run (`tiny-gqa`, 1,000 steps) | 17,440,384 params; val 1.52 ± 0.03 | 41 |

## K.3  Units, once more

Nats ÷ ln 2 = bits; e<sup>loss</sup> = perplexity. Compare models with different
tokenizers only in bits per byte, on the same clean text (Chapter 24), and across scripts in
bits per character (Chapter 8).

`VERIFIED` · every number against the cited chapter, by `ch41_audit.py`.
