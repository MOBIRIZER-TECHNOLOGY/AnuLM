# Model card: AnuLM-Base-400M (`ckpt_multi36k.pt`)

The three-language base model: pretrained from scratch on Hindi, English
and Python with the `multi32k` tokenizer, 36,000 steps at batch 8 × 512
(about 150M tokens), on one consumer GPU. It is the checkpoint the
translation and question-answering models were fine-tuned from, and the
one to continue Hindi or English prose with. Full log: `docs/RESULTS.md`
§22; the Hindi-only ladder that led to it is §1–21.

**Licence: CC BY-SA 4.0.** Hindi Wikipedia and Wikisource are CC BY-SA;
C4 is ODC-BY; the Python slice is codeparrot-clean, de-duplicated GitHub
Python with mixed licences. Not affiliated with Sarvam AI, AI4Bharat,
BharatGen or the Government of India.

## What it does

Continues text in the register it was given. A Hindi title followed by a
blank line yields a Wikipedia- or Wikisource-style article; a narrative
phrase yields narrative; a Python signature yields code. Val loss 4.441 at
step 36,000 on the mixed held-out split. Everything factual in its output
is invented; treat names, dates and numbers as fiction.

## Data

| part | source | licence |
| --- | --- | --- |
| Hindi | Hindi Wikipedia + Wikisource dumps (`fetch_hindi.py`, `filter_prose.py`) | CC BY-SA |
| English | C4 (allenai), a small slice (`fetch_web.py`) | ODC-BY |
| Python | codeparrot-clean (`fetch_code.py`) | mixed, GitHub |

Corpus and tokenizer build: `experiments/build_multi.sh`; training:
`experiments/run_multi.sh`. Both re-fetch everything from public sources;
no data is redistributed.

## Architecture

The configuration the ablations in `docs/RESULTS.md` §12 and §16 converged
on: 20 layers, hidden 1,024, GQA with 16 query / 4 key-value heads and
QK-norm, RoPE θ = 1,000,000, sliding window 256 on layers 0–9, a dense
SwiGLU MLP on layer 0 and 24 routed experts of 192 with top-4 routing on
layers 1–19, aux-loss-free bias balancing (`bias_update_rate` 3e-3), no
shared expert, context 512, 32,768-entry vocabulary, untied embeddings.
397.7M parameters, 173.5M active per token. An independent
re-implementation of the Sarvam 30B design; `docs/ARCHITECTURE.md` §10
says which of Sarvam's choices held up at this scale.

## Use

```bash
python sample.py --ckpt <this folder> --prompt "हिन्दी साहित्य

"
python serve.py  --ckpt <this folder>        # continue-the-text page
```

To fine-tune it, `finetune.py --ckpt <this folder> --qa pairs.jsonl` takes
question/answer JSONL; `docs/DEVELOPING.md` describes the format.
