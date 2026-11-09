---
title: "Appendix B · The 400M arithmetic"
part: "Appendices"
accent: slate
pinned: "@2e80b3e"
---

# Appendix B  The 400M arithmetic

This appendix collects the model's configuration and every count that follows
from it: parameters, active parameters, KV cache and training memory. All
numbers were printed by `book/code-examples/ch41_arithmetic.py` (**Book example,
not repository code**), which reuses `trace_forward.arithmetic` from Chapter 20.
Each count is checked against `num_params()`, and the memory figures against the
files the released run wrote.

```bash
python book/code-examples/ch41_arithmetic.py            # B.2 to B.6, into ./ch41_arithmetic.json
python book/code-examples/ch41_arithmetic.py --config   # B.1's table; needs ckpt_base_v2.pt
```

The script needs no data for B.2–B.4 and B.6. The check in B.5 against the
released run's files runs only when `ckpt_base_v2.pt` and its resume point
`ckpt_base_v2.pt.last` (4.9 GB, not distributed) are present; otherwise the
script says it skipped it. The book's own run used `--save`, which writes
`book/measurements/part_x_arithmetic.json`.

**Units.** Byte counts are exact. Sizes are given in GB (10⁹ bytes), the unit
`docs/RESULTS.md` prints, with GiB (2³⁰ bytes) beside them where a reader may
compare with a tool that reports GiB.

## B.1  The configuration

`AnuLMConfig` (`anulm/model.py` lines 35–228) is a dataclass of 42 fields. The table
gives each field's line and declared default at @2e80b3e, the two presets this
book uses, and the value stored in `ckpt_base_v2.pt`. "=" means the same as
the default.

| Field | Line | Default | `tiny_gqa` | `base_350m` | Released | Meaning |
|---|---|---|---|---|---|---|
| `vocab_size` | 47 | `259` | = | = | `32768` | V: token ids, 256 bytes + BOS, EOS, PAD by default |
| `block_size` | 48 | `256` | = | `512` | `1024` | longest context |
| `tokenizer_path` | 49 | `None` | = | = | `'data/multi32k.json'` | BPE file; `None` for bytes |
| `n_layer` | 54 | `8` | = | `20` | `20` | L |
| `hidden_size` | 55 | `384` | = | `1024` | `1024` | D |
| `n_head` | 56 | `6` | = | `16` | `16` | H, query heads |
| `attn` | 59 | `'gqa'` | = | = | = | `"gqa"` or `"mla"` |
| `head_dim` | 60 | `None` | `64` | `64` | `64` | D<sub>h</sub>; `None` becomes D/H in `__post_init__` |
| `n_kv_head` | 61 | `2` | = | `4` | `4` | H<sub>kv</sub> (GQA only) |
| `kv_lora_rank` | 62 | `48` | = | = | = | MLA latent width r |
| `qk_nope_head_dim` | 63 | `48` | = | = | = | MLA non-rotated q/k width |
| `qk_rope_head_dim` | 64 | `24` | = | = | = | MLA rotated q/k width |
| `v_head_dim` | 65 | `48` | = | = | = | MLA value width |
| `use_qk_norm` | 66 | `True` | = | = | = | RMSNorm on each head's q and k |
| `sliding_window` | 70 | `None` | = | = | `256` | window, in tokens |
| `max_window_layers` | 71 | `0` | = | = | `10` | layers 0 … n−1 use the window |
| `num_experts` | 74 | `16` | = | `12` | `24` | E, routed experts per MoE layer |
| `num_experts_per_tok` | 75 | `2` | = | `3` | `4` | A |
| `num_shared_experts` | 76 | `1` | = | = | `0` | always-on experts |
| `moe_intermediate_size` | 77 | `96` | = | `384` | `192` | I<sub>e</sub> |
| `intermediate_size` | 78 | `768` | = | `2816` | `2816` | I, the dense layer 0 |
| `first_k_dense_replace` | 79 | `1` | = | = | = | dense layers before the MoE ones |
| `n_group` | 80 | `1` | = | = | = | expert groups (1: no grouping) |
| `topk_group` | 81 | `1` | = | = | = | groups a token may use |
| `routed_scaling_factor` | 82 | `2.5` | = | = | = | the routing weights' sum |
| `bias_update_rate` | 83 | `0.001` | = | = | `0.003` | sign-rule step for the balancing bias |
| `moe_impl` | 96 | `'sparse'` | = | = | `'grouped'` | dispatch path |
| `seq_balance_alpha` | 101 | `0.0` | = | = | `0.0001` | sequence-balance loss weight |
| `router_z_alpha` | 102 | `0.0` | = | = | `0.001` | router z-loss weight |
| `rope_theta` | 105 | `10000.0` | `1000000.0` | `1000000.0` | `1000000.0` | RoPE base θ |
| `yarn` | 106 | `False` | = | = | = | YaRN on or off |
| `yarn_factor` | 107 | `8.0` | = | = | = | YaRN scale s |
| `yarn_original_context` | 108 | `256` | = | = | `1024` | the training length YaRN extends |
| `yarn_beta_fast` | 109 | `32.0` | = | = | = | YaRN band edge β |
| `yarn_beta_slow` | 110 | `1.0` | = | = | = | YaRN band edge α |
| `yarn_mscale` | 111 | `1.0` | = | = | = | YaRN temperature coefficient |
| `yarn_mscale_all_dim` | 112 | `1.0` | = | = | = | YaRN temperature coefficient |
| `rms_norm_eps` | 115 | `1e-06` | = | = | = | ε in RMSNorm |
| `dropout` | 116 | `0.0` | = | = | = | attention dropout |
| `tie_word_embeddings` | 117 | `False` | = | = | = | share the input and output tables |
| `initializer_range` | 118 | `0.02` | = | = | = | std of the initial weights |
| `merge_tree` | 124 | `''` | = | = | = | merge-tree tables (Chapter 10) |

*verified*: printed by `ch41_arithmetic.py --config`; the released column is the
`cfg` object pickled in `ckpt_base_v2.pt`. `__post_init__` (lines 121–151) also
sets `head_dim` for MLA to `qk_nope_head_dim + qk_rope_head_dim` and refuses an
unreachable top-k with a `ValueError` (not an `assert`, which `python -O`
would strip).

**`released()` is not the stored configuration.** `trace_forward.released()`
builds `base_350m` with the released run's architecture. It differs from the
checkpoint in six fields, none of which changes a parameter's shape:
`tokenizer_path`, `bias_update_rate`, `moe_impl`, `seq_balance_alpha`,
`router_z_alpha` and `yarn_original_context`
(`ch41_checkpoints.py released`, Chapter 41, 41.5). Every count below holds for
both.

## B.2  Parameters, term by term

No layer has a bias, a SwiGLU has 3·D·I weights, and the MoE layers are
L − 1 of the L layers (`first_k_dense_replace` = 1).

| Term | Formula | `tiny()` | `tiny_gqa` | `base_350m` | Released |
|---|---|---|---|---|---|
| embedding + output head | 2·V·D | 4,096 | 198,912 | 530,432 | 67,108,864 |
| attention | L·((H + 2H<sub>kv</sub>)·D<sub>h</sub>·D + H·D<sub>h</sub>·D) | 9,216 | 3,145,728 | 52,428,800 | 52,428,800 |
| QK-norm gains | L·2·D<sub>h</sub> | 48 | 1,024 | 2,560 | 2,560 |
| RMSNorm gains | 2·L·D + D | 224 | 6,528 | 41,984 | 41,984 |
| dense layer 0 | 3·D·I | 6,144 | 884,736 | 8,650,752 | 8,650,752 |
| routed experts | (L − 1)·E·3·D·I<sub>e</sub> | 24,576 | 12,386,304 | 268,959,744 | 268,959,744 |
| shared experts | (L − 1)·3·D·I<sub>e</sub>·n<sub>shared</sub> | 3,072 | 774,144 | 22,413,312 | 0 |
| routers | (L − 1)·E·D | 512 | 43,008 | 233,472 | 466,944 |
| **total = `num_params()`** | | **47,888** | **17,440,384** | **353,261,056** | **397,659,648** |

*verified*: for every model, the script's sum equals `num_params()` exactly.
`base_350m` keeps the default byte vocabulary, which is why its tables are tiny.

The released checkpoint's state dict holds 397,660,560 numbers. The extra 912
are buffers, not parameters: each of the 19 routers keeps a 24-number
balancing bias and a 24-number load count.

## B.3  Active parameters

An MoE layer runs A of its E routed experts per token. `num_params()` subtracts
the idle ones: (L − 1)·(E − A)·3·D·I<sub>e</sub>.

| | `tiny()` | `tiny_gqa` | `base_350m` | Released |
|---|---|---|---|---|
| idle per token | 2·6·1,536 = 18,432 | 7·14·110,592 = 10,838,016 | 19·9·1,179,648 = 201,719,808 | 19·20·589,824 = 224,133,120 |
| active | 29,456 | 6,602,368 | 151,541,248 | 173,526,528 |
| active / total | 61.5% | 37.9% | 42.9% | 43.6% |

*verified*: printed by the script. "Active" counts the whole input embedding
table, of which a token reads one row (Chapter 20).

## B.4  The KV cache

Per token of context, each layer caches one key and one value for each
key/value head: L·2·H<sub>kv</sub>·D<sub>h</sub> numbers.

| | numbers per token | bf16 | fp32 keys, bf16 values | fp32 | full context in bf16 |
|---|---|---|---|---|---|
| `tiny()` | 96 | 192 B | 288 B | 384 B | 32 tokens: 0.01 MiB |
| `tiny_gqa` | 2,048 | 4,096 B | 6,144 B | 8,192 B | 256 tokens: 1.00 MiB |
| `base_350m` | 10,240 | 20,480 B | 30,720 B | 40,960 B | 512 tokens: 10.00 MiB |
| Released | 10,240 | 20,480 B | 30,720 B | 40,960 B | 1,024 tokens: 20.00 MiB |

*verified*: printed by the script. Chapter 20 explains the middle column: under
bf16 autocast, QK-norm's fp32 gain returns the keys to fp32. The sliding
window does not shrink the cache, because the cache keeps every position.

## B.5  Training memory

AdamW in fp32 keeps four numbers per parameter: the weight, its gradient, and
two moment estimates. That is 16 bytes per parameter before any activation is
stored.

| | parameters | weights | gradients | moments | total |
|---|---|---|---|---|---|
| `tiny_gqa` | 17,440,384 | 0.07 GB | 0.07 GB | 0.14 GB | 279,046,144 B: 0.28 GB (0.26 GiB) |
| `base_350m` | 353,261,056 | 1.41 GB | 1.41 GB | 2.83 GB | 5,652,176,896 B: 5.65 GB (5.26 GiB) |
| Released | 397,659,648 | 1.59 GB | 1.59 GB | 3.18 GB | 6,362,554,368 B: 6.36 GB (5.93 GiB) |

*verified*: arithmetic, printed by the script.

**Against a measurement.** `docs/RESULTS.md` §6's table measured the peak of
the 353M `base_350m` preset with AdamW at 6.34 GB, at batch 8 × 512 with bf16
autocast and gradient checkpointing. What it holds beyond the 5.65 GB of state
is activations and workspace, which this arithmetic does not count. *measured*,
`docs/RESULTS.md` §6 (table).

**The released run's own files agree.** Read memory-mapped, without loading:

```
released checkpoint: state-dict numbers 397,660,560 = parameters 397,659,648 + buffers 912
  ckpt_base_v2.pt            1,591,281,456 bytes on disk
  ckpt_base_v2.pt.last       4,862,324,329 bytes on disk
  .last holds: model 1,590,642,240 B, AdamW state 3,181,283,236 B (= 8 x 397,659,648 + 4 x 1513 step counters: 3,181,283,236), sampler 88,381,176 B
  sampler order: 11,047,015 window starts, int64
```

*verified*: the script asserts that the AdamW state is exactly 8 bytes per
parameter, plus one 4-byte step counter for each of the 1,513 parameter
tensors. The resume point is about three times the size of the best
checkpoint: weights, two moments, and the sampler's shuffled list of
11,047,015 window starts (Chapter 25). Gradients are not saved.

## B.6  Three extensions, as arithmetic

None of these is in the repository (Chapter 41, 41.6).

```
paged KV cache: a 16-token block, all 20 layers, holds 327,680 B (320 KiB) in bf16, 491,520 B (480 KiB) with the fp32 keys autocast leaves; a full 1024-token context 20 MiB in bf16
int8 weights: 397,148,160 matrix numbers at 1 B + 765,440 row scales at 4 B + 511,488 router and norm numbers and 456 router-bias numbers kept fp32 = 402,257,696 B (384 MiB), against fp32 1,590,638,592 B (1517 MiB) and bf16 758 MiB
DPO: a frozen fp32 reference copy adds 1.59 GB (1.48 GiB) to the 6.36 GB (5.93 GiB) of AdamW state: 7.95 GB (7.41 GiB) before activations
```

*verified*: printed by the script. A paged block covers every layer: 16 tokens
× 20,480 bytes in bf16, or × 30,720 with the fp32 keys B.4 describes. "Matrix
numbers" are the weights of every `nn.Linear` and `nn.Embedding`; the 511,488
left over are the 19 routers' 466,944 weights and the 44,544 norm gains. The
456 router-bias numbers are buffers, kept in fp32 because they decide which
experts fire; Chapter 39 (39.2) measures what rounding them to bf16 does.
