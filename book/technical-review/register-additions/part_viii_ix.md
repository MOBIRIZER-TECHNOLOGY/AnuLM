# Register proposals: Parts VIII and IX (Chapters 35–40)

None of these is cited in a chapter. Each chapter states the claim without a
citation, as WRITER_BRIEF.md requires. If the integrator accepts an entry, the
sentence named under "where" can take the ID. All were checked on 2026-11-01
against the source page named (abstract or publisher page only, not the full
text), so the claim cell is limited to what the abstract says.

## Proposed: continuous batching, Orca (Chapter 40)

| field | value |
|---|---|
| title | Orca: A Distributed Serving System for Transformer-Based Generative Models |
| first author | Gyeong-In Yu (with Jeong, Kim, Kim, Chun) |
| year | 2022 |
| identifier | OSDI '22, pp. 521–538 (no arXiv id) |
| source | usenix.org/conference/osdi22/presentation/yu |
| claim to support | iteration-level scheduling: the scheduler runs a single iteration of the model on the batch at a time, so requests can join and leave between iterations |
| how verified | the USENIX presentation page, fetched 2026-11-01: title, authors, venue and pages as above; the abstract states the iteration-level scheduling claim. "Continuous batching" is the later industry name; the abstract does not use it |
| where | 40.2, "Continuous batching, its usual partner, lets requests join and leave the running batch at every step." |

## Proposed: LLM.int8() (Chapter 40)

| field | value |
|---|---|
| title | LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale |
| first author | Tim Dettmers |
| year | 2022 (arXiv v1, 15 August 2022) |
| identifier | arXiv:2208.07339 |
| source | arxiv.org/abs/2208.07339 |
| claim to support | an Int8 matrix multiplication procedure for transformer feed-forward and attention projections, with vector-wise quantization (separate normalisation constants per inner product) |
| how verified | arXiv abstract page, fetched 2026-11-01 |
| where | 40.3, the int8 row (per-row scales) |

## Proposed: GPTQ (Chapter 40)

| field | value |
|---|---|
| title | GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers |
| first author | Elias Frantar |
| year | 2022 (arXiv v1, 31 October 2022) |
| identifier | arXiv:2210.17323 |
| source | arxiv.org/abs/2210.17323 |
| claim to support | post-training quantisation of GPT models down to 3 or 4 bits per weight |
| how verified | arXiv abstract page, fetched 2026-11-01 |
| where | 40.3, the int4 row |

## Proposed: Megatron-LM, tensor parallelism (Chapter 40)

| field | value |
|---|---|
| title | Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism |
| first author | Mohammad Shoeybi |
| year | 2019 (arXiv v1, 17 September 2019) |
| identifier | arXiv:1909.08053 |
| source | arxiv.org/abs/1909.08053 |
| claim to support | a simple, efficient intra-layer model-parallel approach for training transformers with billions of parameters |
| how verified | arXiv abstract page, fetched 2026-11-01 |
| where | 40.6, "Tensor and pipeline parallelism are absent too." |

## Proposed: GPipe, pipeline parallelism (Chapter 40)

| field | value |
|---|---|
| title | GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism |
| first author | Yanping Huang |
| year | 2018 (arXiv v1, 16 November 2018) |
| identifier | arXiv:1811.06965 |
| source | arxiv.org/abs/1811.06965 |
| claim to support | a batch-splitting pipelining algorithm for a model partitioned across accelerators |
| how verified | arXiv abstract page, fetched 2026-11-01 |
| where | 40.6, same sentence |

## Proposed: indirect prompt injection (Chapter 40)

| field | value |
|---|---|
| title | Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection |
| first author | Kai Greshake |
| year | 2023 (arXiv v1, 23 February 2023) |
| identifier | arXiv:2302.12173 |
| source | arxiv.org/abs/2302.12173 |
| claim to support | LLM-integrated applications blur the line between data and instructions; adversaries can inject prompts into data likely to be retrieved |
| how verified | arXiv abstract page, fetched 2026-11-01; venue not checked |
| where | 40.7, the sentence on retrieved passages and prompt injection |

## Not proposed, noted

- Server-sent events (Chapter 38) are described only by what `api.py` writes
  (`data:` lines, `data: [DONE]`); no source is needed.
- safetensors (Chapter 39) is described only by the header the book example
  reads from the released file.
