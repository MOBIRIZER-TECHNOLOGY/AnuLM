# Model card: AnuLM-Translate-400M (`ckpt_translate.pt`)

An English ↔ Hindi sentence translator: the three-language AnuLM base
(`AnuLM-Base-400M`, step 36,000) fine-tuned for one pass over two million
sentence pairs. Full log and samples: `docs/RESULTS.md` §24 and
`docs/TRANSLATE_PLAN.md`.

**Licence: CC BY-NC 4.0, research use only.** Samanantar is CC BY-NC 4.0
and the IIT Bombay corpus is released for research; the weights inherit
those terms. Not affiliated with Sarvam AI, AI4Bharat, BharatGen or the
Government of India.

## Results

chrF on FLORES-200 devtest, all 1,012 sentences per direction, greedy:

| direction | this model | untuned base | NLLB-600M (reference) |
| --- | --- | --- | --- |
| English → Hindi | **41.5** | 0.1 | ~55 |
| Hindi → English | **43.4** | 3.1 | ~55 |

Held-out answer loss fell 4.05 → 2.30 over the pass and was still
improving at the last eval. Everyday sentences of 8 to 25 words translate
well; rare names, numbers and long clauses are where it slips.

## Data

| source | licence | used |
| --- | --- | --- |
| [ai4bharat/samanantar](https://huggingface.co/datasets/ai4bharat/samanantar), Hindi split | CC BY-NC 4.0 | most of the 2M pairs |
| [cfilt/iitb-english-hindi](https://huggingface.co/datasets/cfilt/iitb-english-hindi) | research use | the rest |
| FLORES-200 devtest | CC BY-SA 4.0 | evaluation only |

Pairs were filtered for length, script and duplicates and used in both
directions as `English: …` newline `Hindi: …` and the reverse, loss on the
target sentence only. 4M items packed into 493,370 rows of 512; 61,672
steps at batch 8, learning rate 5e-5, about six hours on one RTX 5070 Ti.

## Architecture and tokenizer

Same network as every 400M AnuLM checkpoint: 20 layers, hidden 1,024,
GQA 16 query / 4 key-value heads, 24 routed experts of 192 with top-4
routing and aux-loss-free balancing, sliding window 256 on layers 0–9,
context 512, 32k vocabulary. 397.7M parameters, 173.5M active per token.
Tokenizer `multi32k`, a byte-level BPE trained on the Hindi + English +
Python mix (`docs/RESULTS.md` §22).

## Use

The page picks the direction from the script you type; the API takes
`mode: "translate"`. One sentence at a time; temperature 0.3 or lower.
`python serve.py --ckpt <this folder>` then open http://127.0.0.1:8000.
"Continue the text" and "answer a question" on this checkpoint are not
useful: fine-tuning on translation pairs eroded the base model's free-text
ability. Use `AnuLM-Base-400M` or `AnuLM-Hindi-QA-400M` for those.
