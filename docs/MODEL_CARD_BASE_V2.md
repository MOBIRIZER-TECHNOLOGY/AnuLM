# Model card: AnuLM-Base-v2-400M (`ckpt_base_v2.pt`)

The second three-language base model: pretrained from scratch on Hindi,
English and Python with the `multi32k` tokenizer, 1.9M steps at batch 4 ×
1,024 tokens (about 7.8B tokens) over an 11.3B-token corpus, on one
consumer GPU. It is the backbone under every model the AnuLM demo app now
serves: chat, speech recognition, captions, image questions and the Decide
adapters. Its predecessor, `AnuLM-Base-400M`, had read about 150M tokens;
this one has read about fifty times as much.

Part of AnuLM: https://github.com/MOBIRIZER-TECHNOLOGY/AnuLM. The demo app
that uses it is `app_omni.py` in that repository.

**Licence: CC BY-SA 4.0.** Hindi Wikipedia is CC BY-SA; fineweb-edu and
fineweb-2 are ODC-BY; the Python slice includes codeparrot-clean, scraped
GitHub code under its files' own mixed licences (including GPL and AGPL).
Not affiliated with Sarvam AI, AI4Bharat, BharatGen or the Government of
India.

## What it does, and does not

Continues text in Hindi, English or Python, in the register it is given.
It is a **base model**: not tuned to answer questions, follow
instructions, chat or translate. For those, start from
`AnuLM-Omni-400M`, which is this model instruction-tuned and taught to
hear and see as well (docs/OMNI.md). Everything
factual in its output may be invented; treat names, dates and numbers as
unverified.

| held-out split of its own corpus (1% of documents, by hash) | value |
| --- | --- |
| val loss at step 1,900,000 | **3.3523** ± 0.0491 |
| bits per byte | **0.756** |
| the 800k-step demo base it replaced | 3.393 / 0.765 |

The held-out split samples every source in proportion, so it is in
distribution. Per-language bits/byte, code pass@1 and any external
benchmark: **not measured** for this checkpoint. What it bought
downstream, measured on each task's own held-out set before and after the
swap (`docs/RESULTS.md` §39): English ASR WER 8.4% → 7.1%, Hindi ASR
51.0% → 35.0% (50 FLEURS clips), A-OKVQA 40.9% → 43.5%, chat answer loss
3.2000 → 3.1445.

## Data

`experiments/build_v2.py` builds the corpus; `data/v2/meta.json` records
it. Nothing is redistributed: every source is re-fetched from its public
home.

| part | source | licence | train tokens |
| --- | --- | --- | --- |
| English | HuggingFaceFW/fineweb-edu, `sample/10BT` | ODC-BY | 6.00B |
| Hindi web | HuggingFaceFW/fineweb-2, `hin_Deva` | ODC-BY | 4.50B |
| Hindi encyclopaedia | Hindi Wikipedia (`wikimedia/wikipedia` 20231101.hi) | CC BY-SA 4.0 | 0.07B |
| Python | codeparrot/codeparrot-clean | mixed, per source file (incl. GPL/AGPL) | 0.75B in all |
| Python | nvidia/OpenCodeInstruct | CC BY 4.0 | (in the 0.75B) |
| Python | OpenCoder-LLM/opc-sft-stage1 | model-generated; upstream terms apply | (in the 0.75B) |

jinaai/code_exercises was deliberately left out: its CC BY-NC-SA terms
would have made the whole base non-commercial.

## Training

| | |
| --- | --- |
| steps | 1,900,000 at batch 4 × block 1,024 |
| schedule | warmup-stable-decay, AdamW, lr 6e-4 → 6e-5 over the last 16% |
| context | 1,024 |
| hardware | one RTX 5070 Ti (16 GB), grouped-GEMM MoE |
| result | val 3.3523, bits/byte 0.756 at the final step |

The run was planned for 2.75M steps and shortened to 1.9M after a
machine outage; the decay started at step 1,590,000.

## Architecture

The same network as every 400M AnuLM checkpoint: 20 layers, hidden 1,024,
GQA with 16 query / 4 key-value heads and QK-norm, RoPE, sliding window
256 on layers 0–9, a dense SwiGLU MLP on layer 0 and 24 routed experts of
192 with top-4 routing on layers 1–19, aux-loss-free bias balancing, no
shared expert, 32,768-entry vocabulary, untied embeddings. 397.7M
parameters, 173.5M active per token. An independent re-implementation of
the Sarvam 30B design.

## Intended use

Research and teaching on small mixture-of-experts models, Hindi language
modelling, and as a starting point for fine-tuning in Hindi, English or
Python. Out of scope: anything that relies on its output being true,
code that will run unreviewed, and use as a chatbot without further
tuning.

## Limitations

- **A base model.** It continues text; it does not answer.
- **Small.** 174M active parameters that have read about 7.8B tokens.
  It degenerates into repetition on long free-running generations; use a
  repetition penalty and short outputs.
- **It invents facts.** Every name, date and number is unverified.
- **Hindi means Devanagari Hindi from the web and Wikipedia.** Other Indic
  languages, romanised Hindi and code-mixed text were not targeted.
- **Code licences.** Its Python came partly from GitHub files under
  copyleft licences; what it writes may resemble them.
- **Load it in float32.** The router keeps a per-expert bias; rounding it
  to 16 bits changes which experts fire.

## Citation

```bibtex
@software{anulm2026,
  title  = {AnuLM: a small, trainable mixture-of-experts language model in the shape of Sarvam 30B, trained from scratch on one GPU for Hindi, English and Python},
  author = {{The AnuLM authors}},
  year   = {2026},
  url    = {https://github.com/MOBIRIZER-TECHNOLOGY/AnuLM}
}
```
