# Model card: AnuLM-Base-2K-400M (`ckpt_ctx2k.pt`)

The three-language base with a **2,048-token context**: `AnuLM-Base-400M`
continued for 10,000 steps at block 2,048 with YaRN, on 46M tokens of the
same Hindi / English / Python proportions it was originally trained on. Five
hours on one RTX 5070 Ti. Full log and the honest reading of what it bought:
`docs/RESULTS.md` §28, with the zero-shot measurement it is compared against
in §27.

**Licence: CC BY-SA 4.0**, the same as the base and for the same reason:
Hindi and English Wikipedia are CC BY-SA, C4 is ODC-BY, and the Python slice
is codeparrot-clean, de-duplicated GitHub Python with mixed licences. Not
affiliated with Sarvam AI, AI4Bharat, BharatGen or the Government of India.

## Why this exists

Every other checkpoint in this project is trained at 512 tokens. This one
answers what happens if you train at four times that with the architecture's
own long-context mechanism switched on — and the answer, stated plainly
because it is the interesting part, is **less than you would hope**:

| loss by position in a 2,048-token window | 0–512 | 1536–2048 | last − first |
| --- | --- | --- | --- |
| `AnuLM-Base-400M`, plain RoPE | **4.0400** | 4.1128 | **+0.0728** |
| `AnuLM-Base-400M`, YaRN, no training | 4.0743 | 4.0065 | **−0.0678** |
| **this model**, trained at 2,048 | 3.6693 | **3.5839** | **−0.0854** |

A model that gets *worse* the further into the window it goes is failing to
extend; the sign of that last column is the whole question. Zero-shot YaRN
already flips it, for free. Five GPU-hours of training moved it from −0.068
to −0.085. **If all you want is sane behaviour at 2,048 tokens, apply YaRN to
`AnuLM-Base-400M` and skip this model.**

What the training did buy is general quality, because it is 82M tokens on top
of the base's 148M — 55% more compute:

| bits/byte, held-out | Hindi | English | Python |
| --- | --- | --- | --- |
| `AnuLM-Base-400M` | 0.6001 | 1.5208 | 1.0433 |
| **this model** | **0.5323** | **1.3897** | **0.8102** |

Nothing regressed in any language, which is the usual risk of a continuation.
Python gains most because the base saw the least of it.

## What it does

Continues text in Hindi, English or Python, in the register it is given, over
a window four times longer than any other checkpoint here. It is a **base
model**: not tuned to answer questions, follow instructions, chat or
translate. Everything factual in its output is invented.

## What it sounds like

Better compression did not make it a better writer. Six prompts across the
three languages, temperature 0.8, top-k 50, 120 tokens each:

| | distinct-token ratio | repeated 4-grams |
| --- | --- | --- |
| `AnuLM-Base-400M` | 0.335 | 32.5% |
| this model | 0.425 | 33.3% |

More varied by one measure, the same by the other. **Both degenerate**: a
third of their 4-grams are repeats, which is what a 398M base model does when
asked to free-run. Greedy decoding is worse for both. Samples from this
model, at temperature 0.8:

```
मुंबई          -> बिग साइड (Red Dock) एक ऐसा एक प्रकार के साइड है, जो एक ही साइड के
                  एक साथ काम करता है। यह साइड सबसे पहले 1880 में सामने आया।
The history of -> microscopic computers, which are part of the CuOi and are used for
computing         the development of a complex network of computers.
def quicksort  -> x = x - 1 / y = y - 1 / y = X - y - 1 / y = X - y - 1 / ...
(arr):
```

Fluent-shaped, factually invented, and prone to looping — the English holds
together longest, the Python loops fastest. Use a repetition penalty and a
short generation if you want something readable; `serve.py` does.

## Data

46.0M tokens (306 MB) mixed in the base's own proportions by `mix_corpus.py`:

| part | source | size |
| --- | --- | --- |
| Hindi | Hindi Wikipedia dump (`fetch_hindi.py --keep-prob 0.3`) | 189 MB, 21,897 articles |
| English | C4 (`fetch_web.py`) | 79 MB, 23,611 pages |
| Python | codeparrot-clean (`fetch_code.py`) | 38 MB, 4,902 files |

`mix_corpus.py` holds out `data/bench_{hindi,english,python}.txt`, which is
what every number above is measured on. No data is redistributed;
`docs/DATASETS.md` lists each source with its licence, and the exact
commands are in `docs/RESULTS.md` §28.

## Training

| | |
| --- | --- |
| from | `AnuLM-Base-400M`, upcast to float32 |
| steps | 10,000 at batch 2 × grad-accum 4 × block 2,048 = 8,192 tokens/step |
| tokens | 82M, about 2 epochs of the mix |
| context | **2,048**, with YaRN: `yarn_original_context=512`, `yarn_factor=4.0` |
| learning rate | 1e-4 cosine to 1e-5, 200 warmup |
| precision | bf16 autocast over float32 weights, gradient checkpointing, grouped-GEMM MoE |
| hardware | one RTX 5070 Ti, ~1.7 s/step, 4 h 52 min unattended |
| result | val 4.3031 at step 250 → **4.0501** best (`ctx2k_curve.csv`, 40 points) |

`yarn_original_context` is pinned to 512 rather than left at the block size.
That is the field `docs/ARCHITECTURE.md` §4 calls the most-often-mis-set part
of a `rope_scaling` block: the NTK-by-parts correction range has to be
measured against the length the model was *trained* at, not the one it is
being extended to.

**The selected checkpoint is step 7,499, not 10,000**, and the difference is
0.0004 against a standard error of 0.0289 — best-val selection choosing
between numbers inside their own noise, exactly as `docs/RESULTS.md` §26
describes. Treat them as the same model.

## Architecture

Identical to every 400M AnuLM checkpoint except the context: 20 layers,
hidden 1,024, GQA with 16 query / 4 key-value heads and QK-norm, RoPE
θ = 1,000,000, sliding window 256 on layers 0–9, a dense SwiGLU MLP on layer 0
and 24 routed experts of 192 with top-4 routing on layers 1–19, aux-loss-free
bias balancing, no shared expert, 32,768-entry vocabulary, untied embeddings.
**397.7M parameters, 173.5M active per token.** Context 2,048 with YaRN
enabled in the config, where the others are 512 without it. Tokenizer
`multi32k`, unchanged from the base.

## How to load

### With `transformers`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("toonist/AnuLM-Base-2K-400M",
                                             trust_remote_code=True)
tok = AutoTokenizer.from_pretrained("toonist/AnuLM-Base-2K-400M")

ids = tok("भारत की राजधानी", return_tensors="pt")
print(tok.decode(model.generate(**ids, max_new_tokens=80, do_sample=False)[0]))
```

**Load it in float32**, which the config asks for by default. Do not force
`dtype=torch.bfloat16`: the router keeps a per-expert bias whose differences
decide which experts fire, and rounding it to 16 bits changes the routing —
the output collapses into repeated tokens rather than degrading. Use
`torch.autocast` over float32 weights for speed.

### With the AnuLM repository

```bash
hf download toonist/AnuLM-Base-2K-400M --local-dir AnuLM-Base-2K-400M
python serve.py  --ckpt AnuLM-Base-2K-400M
python sample.py --ckpt AnuLM-Base-2K-400M --prompt "हिन्दी साहित्य

"
```

## Limitations

- **A base model.** No instruction, chat, QA or translation ability. For
  those, `AnuLM-Hindi-QA-400M` and `AnuLM-Translate-400M`.
- **The long-context gain is mostly YaRN, not this training.** See the table
  at the top; the honest version of the claim is in `docs/RESULTS.md` §28.
- **Benchmarks flatter it slightly.** Its held-out benches are a slice of its
  own corpus, and a fresh out-of-sample draw for the base it is compared
  against, though both come from the same public sources.
- **It invents facts.** Every date, name and number in its output is fiction.
- Trained on a single consumer GPU for a total of about 9 GPU-hours across
  both stages. It is a 398M model, and reads like one.
