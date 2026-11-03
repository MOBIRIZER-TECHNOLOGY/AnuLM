# Register proposals: Part IV, first half after Chapter 11 (Chapters 12–15)

None of these is cited with a register ID in a chapter. Where a chapter makes
the claim, it states it without a citation, as `WRITER_BRIEF.md` requires. The
integrator decides whether to add them to `RESEARCH_REFERENCES.md`.

| Proposed for | Title | Author(s) | Year | Identifier | Exact claim it would support | How it was checked (2026-11-01) |
|---|---|---|---|---|---|---|
| Ch 15, 15.5 — **extend [R04]'s claims** | Language Models are Unsupervised Multitask Learners | A. Radford et al. | 2019 | OpenAI report (as [R04]) | "We scale the weights of residual layers at initialization by a factor of 1/√N where N is the number of residual layers"; and "an additional layer normalization was added after the final self-attention block" (the final norm, `model.py` line 790) | Fetched the PDF at the [R04] URL (cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) and extracted its text with PyMuPDF; both sentences are in §2.3, next to the pre-norm sentence [R04] already supports. Chapter 15 currently says "GPT-2's report describes the same 1/√N rule" without an ID; with this extension it can cite [R04] |
| Ch 15, 15.3 — **extend [R01]'s claims** | Attention Is All You Need | A. Vaswani et al. | 2017 | arXiv:1706.03762 (as [R01]) | "We employ a residual connection around each of the two sub-layers, followed by layer normalization. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x))" (post-norm) | Fetched the arXiv v7 HTML (arxiv.org/html/1706.03762v7, v7 dated 2 August 2023), §3.1; both sentences found. Chapter 15 now describes post-norm without a citation; with this extension it can say the 2017 transformer used it [R01] |
| Ch 15, 15.3 (optional) | Deep Residual Learning for Image Recognition | K. He, X. Zhang, S. Ren, J. Sun | 2015 | arXiv:1512.03385 | layers reformulated "as learning residual functions with reference to the layer inputs" (the residual connection) | arXiv abstract page fetched: title, four authors and v1 date 10 December 2015 match; the quoted sentence is in the abstract. The chapter makes no historical claim about residual connections now; this would support one |
| Ch 13, 13.4 — **extend [R50]'s claims** | torch.nn.functional.scaled_dot_product_attention | PyTorch | 2026 | PyTorch 2.11 docs (as [R50]) | with `enable_gqa=True` the documented reference implementation repeats key and value heads with `repeat_interleave(query.size(-3)//key.size(-3), -3)`; the docstring calls GQA "an experimental feature" that "currently works only for Flash_attention and math kernel on CUDA tensor" | Read from the installed torch 2.11.0 function's docstring (`F.scaled_dot_product_attention.__doc__`), the source [R50] already records for its other claims. Chapter 13 states the grouping from its own runnable check, not from R50 |

Not proposed, but noted: the sliding window's origin (Longformer, Mistral) is
not cited in Chapter 12; the code comment attributes the semantics to Qwen2,
and the chapter quotes only that comment.
