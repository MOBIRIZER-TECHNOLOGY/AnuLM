# Register proposals: Part VI (Chapters 27–29)

Neither source below is cited in a chapter. Chapter 29 describes the sliding
window and the trimmed (rolling) cache from the code only, and says in its
References section that sources are proposed here. If the integrator accepts
them, the citations would go where marked.

## P-VI-1 · Longformer (sliding-window attention)

- **Title:** Longformer: The Long-Document Transformer
- **First author:** Iz Beltagy (with Peters, Cohan)
- **Year:** 2020 (arXiv v1, 10 April 2020)
- **Identifier:** arXiv:2004.05150 · arxiv.org/abs/2004.05150
- **Claim to support:** an attention that "combines a local windowed attention
  with a task motivated global attention" and scales linearly with sequence
  length; i.e. local (sliding) window attention as a published mechanism.
- **Where:** Chapter 29, 29.5, first paragraph after the `Block.__init__`
  excerpt ("Layers below `max_window_layers` attend only to ...").
- **How verified:** arxiv.org/abs/2004.05150 fetched 2026-11-01; title, first
  author, v1 date and the abstract sentence quoted above read from the page.
- **Note:** AnuLM's window follows Qwen2's per-layer semantics
  (`max_window_layers`, `model.py` lines 62–66 and 767–769), not Longformer's
  global-plus-local pattern; the §7 matrix row would say so.

## P-VI-2 · Mistral 7B (sliding window attention; rolling buffer cache)

- **Title:** Mistral 7B
- **First author:** Albert Q. Jiang
- **Year:** 2023 (arXiv v1, 10 October 2023)
- **Identifier:** arXiv:2310.06825 · arxiv.org/abs/2310.06825
- **Claims to support:**
  1. sliding window attention (SWA) in a decoder LLM (abstract: "sliding window
     attention (SWA) to effectively handle sequences of arbitrary length");
  2. the rolling buffer cache: "The cache has a fixed size of W, and the keys
     and values for the timestep i are stored in position i mod W of the
     cache" (§2, "Rolling Buffer Cache"), i.e. a windowed layer's cache can be
     bounded at W positions.
- **Where:** Chapter 29, 29.6 (the `EXTENSION`), on the sentence "A real
  version would keep a fixed buffer per windowed layer", and the matrix
  (`docs/reference/research-references.md` §7) as a new row "Sliding window / rolling cache":
  repository = mask only, cache never trimmed (`GQAttention.forward` 441–447);
  difference = the paper bounds the cache at W, AnuLM keeps every position;
  treatment = implemented (window), `EXTENSION` (rolling cache).
- **How verified:** the abstract page and the v1 PDF fetched 2026-11-01; the
  claim-2 sentence was extracted from the PDF text (§2). Not used: the paper's
  "8x" memory figure at 32k tokens, which is about its own model.

## Matrix difference found (for `docs/reference/research-references.md` §7)

No difference between R07 and the code beyond what the YaRN row already says.
Chapter 29 re-derived the band edges for the released configuration
(θ = 10⁶, D<sub>h</sub> = 64, original context 1,024): `low` = 3, `high` = 12,
printed by `code-examples/ch29_long_context.py`.
