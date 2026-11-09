---
title: "Appendix A · Tensor shapes for every module"
part: "Appendices"
accent: slate
pinned: "@2e80b3e"
---

# Appendix A  Tensor shapes for every module

Every shape below was printed by running `anulm/model.py` unchanged at @2e80b3e,
with `book/code-examples/ch41_shapes.py` (**Book example, not repository
code**). The script puts a forward hook on each module and wraps the two
module-level functions `apply_rope` and `_sdpa`. It records what goes in and
what comes out, and checks the cache shapes against the formulas in A.4.

```bash
python book/code-examples/ch41_shapes.py          # three models, about 4 s on a CPU
python book/code-examples/ch41_shapes.py --tiny   # the two small ones only
```

Each model gets a prompt of B = 2 sequences of S = 5 tokens, with a KV cache
and no targets. Then it takes one decode step: S = 1 new token against
T = 6 key positions, the 5 cached and the new one. The symbols are Chapter 20's:

| | B | S | T | D | H | H<sub>kv</sub> | D<sub>h</sub> | V | E | A | I | I<sub>e</sub> |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `tiny()` | 2 | 5 | 5 | 32 | 4 | 2 | 8 | 64 | 8 | 2 | 64 | 16 |
| released 400M | 2 | 5 | 5 | 1,024 | 16 | 4 | 64 | 32,768 | 24 | 4 | 2,816 | 192 |

N = B·S is the number of rows the router sees, and n<sub>e</sub> is the number
of (token, slot) pairs that reach expert e, which depends on the data.

## A.1  The GQA model, prefill

Layer 0 is dense; layers 1 to L − 1 are mixtures of experts. A row marked
"all layers" has the same shapes in every layer.

| Module (`model.named_modules()` name) | Class | In | Out | `tiny()` out | Released out |
|---|---|---|---|---|---|
| `rotary` | `Rotary` | S, start | cos, sin (S, D<sub>h</sub>) | (5, 8) | (5, 64) |
| `layers.i.input_layernorm` (all layers) | `RMSNorm` | (B, S, D) | (B, S, D) | (2, 5, 32) | (2, 5, 1024) |
| `layers.i.attn.query_key_value` | `Linear` | (B, S, D) | (B, S, (H + 2H<sub>kv</sub>)·D<sub>h</sub>) | (2, 5, 64) | (2, 5, 1536) |
| `layers.i.attn.query_layernorm` | `RMSNorm` | (B, H, S, D<sub>h</sub>) | same | (2, 4, 5, 8) | (2, 16, 5, 64) |
| `layers.i.attn.key_layernorm` | `RMSNorm` | (B, H<sub>kv</sub>, S, D<sub>h</sub>) | same | (2, 2, 5, 8) | (2, 4, 5, 64) |
| `apply_rope` (function) | | q (B, H, S, D<sub>h</sub>), k (B, H<sub>kv</sub>, S, D<sub>h</sub>) | same | q (2, 4, 5, 8), k (2, 2, 5, 8) | q (2, 16, 5, 64), k (2, 4, 5, 64) |
| `_sdpa` (function) | | q (B, H, S, D<sub>h</sub>); k, v (B, H, T, D<sub>h</sub>) after `repeat_interleave` | (B, H, S, D<sub>h</sub>) | k, v (2, 4, 5, 8); mask `None` | k, v (2, 16, 5, 64); mask (5, 5) |
| `layers.i.attn.dense` | `Linear` | (B, S, H·D<sub>h</sub>) | (B, S, D) | (2, 5, 32) | (2, 5, 1024) |
| `layers.i.attn` | `GQAttention` | (B, S, D) | (B, S, D) | (2, 5, 32) | (2, 5, 1024) |
| `layers.i.post_attention_layernorm` | `RMSNorm` | (B, S, D) | (B, S, D) | (2, 5, 32) | (2, 5, 1024) |
| `layers.0.mlp.gate_proj`, `up_proj` | `Linear` | (B, S, D) | (B, S, I) | (2, 5, 64) | (2, 5, 2816) |
| `layers.0.mlp.down_proj` | `Linear` | (B, S, I) | (B, S, D) | (2, 5, 32) | (2, 5, 1024) |
| `layers.0.mlp` | `SwiGLU` | (B, S, D) | (B, S, D) | (2, 5, 32) | (2, 5, 1024) |
| `layers.1.mlp.router` | `Router` | (N, D) | ids (N, A), weights (N, A), aux | (10, 2), (10, 2) | (10, 4), (10, 4) |
| `layers.1.mlp.experts.e` | `SwiGLU` | (n<sub>e</sub>, D) | (n<sub>e</sub>, D) | expert 0: (6, 32) | expert 1: (4, 1024) |
| `layers.1.mlp.shared_expert` | `SwiGLU` | (B, S, D) | (B, S, D); inner width I<sub>e</sub> | (2, 5, 32) | absent (0 shared) |
| `layers.1.mlp` | `MoE` | (B, S, D) | (B, S, D), aux | (2, 5, 32) | (2, 5, 1024) |
| `layers.i` | `Block` | (B, S, D) | (B, S, D), aux | (2, 5, 32) | (2, 5, 1024) |
| `norm` | `RMSNorm` | (B, S, D) | (B, S, D) | (2, 5, 32) | (2, 5, 1024) |
| `lm_head` | `Linear` | (B, 1, D) without targets; (B, S, D) with | (B, 1, V) or (B, S, V) | (2, 1, 64) | (2, 1, 32768) |

*verified*: printed by `ch41_shapes.py`. Notes on four rows:

- **`embed_tokens` never runs as a module.** `AnuLM.forward` reads its table
  through `F.embedding(idx, self._table("in"))` (`anulm/model.py` line 906), so no
  hook fires; the stream it produces is the (B, S, D) input of
  `layers.0.input_layernorm`.
- **The mask.** `tiny()` has no window, so `_sdpa` gets `mask=None` and uses
  `is_causal`. The released model's layer 0 has a 256-token window, so it gets
  an explicit boolean mask of shape (S, T) (Chapter 12).
- **Experts.** Only experts that receive tokens run. The script prints the
  first one that ran; `n_e` was 6 for `tiny()`'s expert 0 and 4 for the
  released model's expert 1 (random weights).
- **With targets.** `lm_head` gets every position and returns (B, S, V). The
  one exception is a model in training mode with `enable_chunked_loss` on: the
  head then runs a chunk at a time and `forward` returns `None` for the logits
  (`anulm/model.py` line 929; Chapter 23). In evaluation mode the chunked path is
  skipped and the full logits come back.

## A.2  The decode step

After the prefill, one token per sequence goes in with `start = 5`:

| What | `tiny()` | Released |
|---|---|---|
| ids in, logits out | (2, 1) → (2, 1, 64) | (2, 1) → (2, 1, 32768) |
| `rotary` cos, sin | (1, 8) | (1, 64) |
| every (B, S, ·) tensor above | (2, 1, ·) | (2, 1, ·) |
| `apply_rope` q, k | (2, 4, 1, 8), (2, 2, 1, 8) | (2, 16, 1, 64), (2, 4, 1, 64) |
| `_sdpa` q; k, v | (2, 4, 1, 8); (2, 4, 6, 8) | (2, 16, 1, 64); (2, 16, 6, 64), mask (1, 6) |
| router in → out | (2, 32) → (2, 2) | (2, 1024) → (2, 4) |

*verified*: printed by `ch41_shapes.py`. The query covers the 1 new position;
the keys and values cover all T = 6, cached and new (Chapter 27).

## A.3  The MLA model

`tiny()` with `attn="mla"` and the sizes `cache_holds_what_it_should` uses:
H = 4, q/k "nope" width d<sub>n</sub> = 8, rope width d<sub>r</sub> = 8, value
width d<sub>v</sub> = 8, latent rank r = 16. The MoE half is as in A.1.

| Module | In | Out (symbolic) | `tiny` MLA, prefill | decode |
|---|---|---|---|---|
| `attn.q_proj` | (B, S, D) | (B, S, H·(d<sub>n</sub> + d<sub>r</sub>)) | (2, 5, 64) | (2, 1, 64) |
| `attn.q_layernorm` | (B, H, S, d<sub>n</sub> + d<sub>r</sub>) | same | (2, 4, 5, 16) | (2, 4, 1, 16) |
| `attn.kv_a_proj_with_mqa` | (B, S, D) | (B, S, r + d<sub>r</sub>) | (2, 5, 24) | (2, 1, 24) |
| `attn.kv_a_layernorm` | (B, S, r) | (B, S, r) | (2, 5, 16) | (2, 1, 16) |
| `apply_rope` | q<sub>pe</sub> (B, H, S, d<sub>r</sub>), k<sub>pe</sub> (B, 1, S, d<sub>r</sub>) | same | (2, 4, 5, 8), (2, 1, 5, 8) | (2, 4, 1, 8), (2, 1, 1, 8) |
| `attn.kv_b_proj` | (B, T, r) | (B, T, H·(d<sub>n</sub> + d<sub>v</sub>)) | (2, 5, 64) | in (2, 6, 16), out (2, 6, 64) |
| `_sdpa` | q (B, H, S, d<sub>n</sub> + d<sub>r</sub>); k (B, H, T, d<sub>n</sub> + d<sub>r</sub>); v (B, H, T, d<sub>v</sub>) | (B, H, S, d<sub>v</sub>) | q (2, 4, 5, 16), k (2, 4, 5, 16), v (2, 4, 5, 8) | k (2, 4, 6, 16), v (2, 4, 6, 8) |
| `attn.o_proj` | (B, S, H·d<sub>v</sub>) | (B, S, D) | (2, 5, 32) | (2, 1, 32) |

*verified*: printed by `ch41_shapes.py`. Note `kv_b_proj` in the decode step:
it expands the whole cached latent, all T = 6 positions, on every step
(Chapter 19).

## A.4  What the cache holds

`AnuLM.new_cache()` makes one dict per layer. After the prefill and one decode
step:

| Attention | Keys | Shape | `tiny()` after decode | Released after decode |
|---|---|---|---|---|
| GQA | `k`, `v` | (B, H<sub>kv</sub>, T, D<sub>h</sub>) each | (2, 2, 6, 8) | (2, 4, 6, 64) |
| MLA | `c` | (B, T, r) | (2, 6, 16) | (no MLA) |
| MLA | `k_pe` | (B, 1, T, d<sub>r</sub>) | (2, 1, 6, 8) | |

Per token and layer, GQA caches 2·H<sub>kv</sub>·D<sub>h</sub> numbers (512
for the released model) and MLA r + d<sub>r</sub> (24 for this `tiny` MLA).
Appendix B turns these into bytes. *verified*: the script asserts both shapes.

**How this page is checked.** `ch41_audit.py` (Appendix F) confirms that every
concrete shape on this page occurs somewhere in `ch41_shapes.py`'s output. That
is a membership check: it would not catch a correct shape placed in the wrong
row. The rows were transcribed by hand from the script's output, module by
module.
