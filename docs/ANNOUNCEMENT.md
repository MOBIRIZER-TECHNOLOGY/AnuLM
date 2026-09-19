# Announcement drafts

Ready-to-post copy for LinkedIn and X, plus the source of every number in
them. The last section is the important one: if someone asks in the comments
where a figure comes from, it is there with a file and a section number.

Nothing here is marketing copy in the sense of being unfalsifiable. Every
claim below is measured in `docs/RESULTS.md` and reproducible from this
repository.

---

## LinkedIn, long form

> I spent the last few weeks reading Sarvam AI's open-weight 30B and 105B
> code line by line — and then rebuilding it at 1/1000th the size to check I
> had actually understood it.
>
> The result is **AnuLM** (अणु, "atom"): a 398M-parameter mixture-of-experts
> model carrying the same architecture — grouped-query attention, 24 routed
> experts with aux-loss-free load balancing, sliding-window attention, YaRN —
> written from scratch in pure PyTorch and trained on a single consumer GPU
> for Hindi, English and Python.
>
> Five checkpoints are on Hugging Face:
> → a Python coder — 12.5% pass@1 on MBPP
> → an English ↔ Hindi translator — chrF 41.5 / 43.4 on FLORES-200
> → a Hindi/English/Python question answerer
> → a three-language base, and that base extended to a 2,048-token context
>
> The most useful thing I learned was not architectural. Across 28 documented
> experiments, the single largest lever was the **tokenizer**. Training a 16k
> byte-level BPE for Hindi was worth −0.162 bits/byte; a 20× increase in
> parameters bought 0.110. Reading Devanagari as raw bytes costs about 2.5
> tokens *per character*, so a 512-token window holds roughly 200 characters
> of Hindi. The trained tokenizer fits about 3.8 characters into each token
> instead.
>
> That is a large part of why serving Indic languages has been expensive, and
> it is exactly where Sarvam's real contribution sits: a 262,144-token
> tokenizer spanning 22 languages and 12 scripts. The architecture is
> DeepSeek-lineage and openly so. The tokenizer and the data pipeline are the
> moat.
>
> Everything is open — the code, the data recipes, and the full experiment
> log with every number that did not work alongside the ones that did. You
> can try it in Colab on the free tier without installing anything.
>
> 🔗 github.com/MOBIRIZER-TECHNOLOGY/AnuLM
>
> A caveat worth stating plainly: this is a 400M model trained on one GPU. It
> invents facts and it reads like what it is. It is published as a study, not
> a product. Independent work, not affiliated with Sarvam AI.
>
> #MachineLearning #OpenSource #LLM #IndicNLP #PyTorch

## LinkedIn, short form

> Can you understand a 105B-parameter model by rebuilding it at 1/1000th the
> size?
>
> I read Sarvam AI's open-weight code line by line and reimplemented the
> architecture — sparse MoE routing, grouped-query attention, YaRN — as a
> 398M model in pure PyTorch, trained from scratch on one consumer GPU for
> Hindi, English and Python.
>
> Five open checkpoints, 28 documented experiments. A Python coder at 12.5%
> pass@1 on MBPP; an English ↔ Hindi translator at chrF 41.5 / 43.4 on
> FLORES-200.
>
> The biggest finding: the tokenizer mattered more than 20× the parameters.
>
> 🔗 github.com/MOBIRIZER-TECHNOLOGY/AnuLM
>
> Independent study, not affiliated with Sarvam AI. It is a 400M model — it
> invents things.

## X / Twitter

> Rebuilt Sarvam AI's 30B/105B architecture at 1/1000th scale — 398M params,
> pure PyTorch, one consumer GPU, Hindi + English + Python.
>
> 5 open checkpoints. MBPP 12.5%. chrF 41.5/43.4 on FLORES-200.
>
> Biggest lever across 28 experiments wasn't the architecture. It was the
> tokenizer.
>
> github.com/MOBIRIZER-TECHNOLOGY/AnuLM

---

## Every number, and where it comes from

| claim | value | source |
| --- | --- | --- |
| parameters | 397.7M total, 173.5M active per token | `docs/MODEL_CARD.md`, "Architecture"; `release/*/config.json` |
| trained on one consumer GPU | RTX 5050 8 GB, then RTX 5070 Ti 16 GB | `docs/RESULTS.md` §6 onward |
| MBPP pass@1 | **12.5%** (32/257), instruction-tuned, asked as a question | §25; `docs/MODEL_CARD.md`, "Capability" |
| HumanEval pass@1 | 4.9% (8/164), base model continuing a signature | §25 — quote the mode, the tuned model scores 3.0% continuing |
| FLORES-200 chrF | **41.5** en→hi, **43.4** hi→en, all 1,012 sentences | §24; `docs/MODEL_CARD_TRANSLATE.md` |
| number of experiments | 28 sections | `docs/RESULTS.md` index |
| tokenizer lever | −0.162 bits/byte | §7; ledger in `docs/ARCHITECTURE.md` §10 |
| 20× parameters lever | 0.110 bits/byte, on one benchmark | same ledger |
| Devanagari as bytes | 2.48 bytes per character, so ~2.5 tokens per character | `docs/ARCHITECTURE.md` §9; `bpe.py` docstring |
| trained tokenizer | 9.5 bytes per token on Hindi, ≈3.8 characters | `README.md`, "Three languages"; §7, §22 |
| Sarvam's tokenizer | 262,144 tokens, 22 languages, 12 scripts | `docs/ARCHITECTURE.md` §2, §9 |
| DeepSeek lineage | Sarvam's own header says so | `docs/ARCHITECTURE.md` §1, quoting the 105B file |
| five checkpoints | Coder, Translate, Hindi-QA, Base, Base-2K | the table in `README.md` |
| runs in Colab free tier | `demo_colab.ipynb`, two cells | `docs/TUTORIAL.md` §10 |

**Two numbers that are easy to state backwards.** The byte-level tokenizer
costs ~2.5 tokens per Devanagari *character*; the trained BPE gets 9.5
*bytes* per token. Bytes are 1.0 byte per token by definition. An earlier
draft of this post had the comparison inverted, which is the kind of error a
commenter will find.

## Before posting

- **Attach an image.** The easiest honest one is a screenshot of the Colab
  demo mid-generation, or the results table in `README.md`. Posts with an
  image reliably outperform plain text.
- **Check the links resolve** — all nine in `README.md` were live when this
  was written, including the five Hugging Face repos and the Colab link.
- **Keep the disclaimers.** "Not affiliated with Sarvam AI" and "it invents
  facts" cost nothing and read as competence rather than modesty. `NOTICE`
  and every model card carry the same language.
- **Do not claim** state of the art, production readiness, or any endorsement
  by Sarvam AI, AI4Bharat, BharatGen or the Government of India. The project
  is a re-implementation of published architecture, in the sense that nanoGPT
  is a re-implementation of GPT-2.
- **A DOI reads well** and is not there yet: `TODO.md` item 1 mints one
  through Zenodo in about fifteen minutes, and only the repository owner can
  start it.

## Likely questions, and honest answers

**"Is this a Sarvam model?"** No. No Sarvam weights are used or
redistributed. Their modelling code sits unmodified under `sarvam/` with its
Apache 2.0 licence, as the object of study; everything else was written from
scratch. `NOTICE` §1.

**"Can I use it commercially?"** Depends which checkpoint. The base and Q&A
models are CC BY-SA 4.0; the translator is CC BY-NC 4.0 because Samanantar
is non-commercial; the coder is CC BY-NC-SA 4.0 because it learned from
ChatGPT-generated exercises. `NOTICE` §2 is the table to read.

**"How does it compare to a real model?"** It does not, and the docs say so.
For scale: CodeParrot-110M reaches ~4% on HumanEval after 25B tokens, this
reaches 4.9% after 2.87B; NLLB-600M scores about chrF 55 where this scores
41.5. The interesting part is what the experiments say, not the ranking.

**"Why does it repeat itself?"** It is a 398M base model. Raise the
repetition penalty to about 1.2 in the demo. The model card for
`AnuLM-Base-2K-400M` measures the degeneration rather than hiding it.
