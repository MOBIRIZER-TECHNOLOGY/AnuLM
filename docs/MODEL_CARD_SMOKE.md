# Model card: AnuLM-Smoke-30B (`ckpt_smoke.pt`)

**This is not a usable language model.** It is a 17.4M-parameter
`nano_30b` preset trained for **200 steps on 1 MB of tinyshakespeare**
(29 seconds on one RTX 5070 Ti) for one purpose: to exercise the whole
pipeline — train, sample, export, upload, download, load, generate — end
to end after the project was renamed from *nanosarvam* to AnuLM and its
tokenizer format id changed. It exists so that the release machinery is
tested by something other than the four real checkpoints.

If you want a model that does something, use one of those instead:

| checkpoint | what it does |
| --- | --- |
| [toonist/AnuLM-Coder-400M](https://huggingface.co/toonist/AnuLM-Coder-400M) | writes small Python functions (MBPP 12.5% pass@1) |
| [toonist/AnuLM-Translate-400M](https://huggingface.co/toonist/AnuLM-Translate-400M) | English ↔ Hindi (chrF 41.5 / 43.4 on FLORES-200) |
| [toonist/AnuLM-Hindi-QA-400M](https://huggingface.co/toonist/AnuLM-Hindi-QA-400M) | answers questions in Hindi, English and Python |
| [toonist/AnuLM-Base-400M](https://huggingface.co/toonist/AnuLM-Base-400M) | the three-language base those were tuned from |

**Licence: MIT.** tinyshakespeare is public domain, so unlike every other
checkpoint in this project nothing constrains this one. Not affiliated
with Sarvam AI, AI4Bharat, BharatGen or the Government of India.

## What it is

The reference `nano_30b` preset from `README.md`: the shape of Sarvam 30B
at 1/1000th the size — GQA with 6 query / 2 key-value heads, 8 layers, 16
routed experts top-2 plus a shared expert on 7 MoE layers, a dense layer 0,
RoPE θ = 1,000,000, QK-norm, sigmoid router with aux-loss-free bias
balancing. **17.44M parameters, 6.60M active per token (37.9%).**
Byte-level vocabulary (256 bytes + BOS/EOS/PAD = 259), context 256, so
there is no BPE tokenizer file beside the weights.

```
python train.py --preset 30b --steps 200 --eval-every 50 --out ckpt_smoke.pt
```

| | |
| --- | --- |
| data | tinyshakespeare, 1.12 MB, 3 documents (`train.py` downloads it when given no `--data`) |
| steps | 200 at batch 16 × block 256, 0.82 of one epoch |
| optimizer | fused AdamW, lr 3e-4 cosine, bf16 autocast |
| wall clock | **29 s** on one RTX 5070 Ti, ~29k tokens/s |
| val loss | 5.6205 at step 0 → **1.8515** at step 199 (bits/byte 2.671) |
| expert imbalance | 4.68× → 7.97× peak at step 20 → **1.80×** at the end |

That imbalance curve is the point of keeping this run's numbers: load
imbalance **rises before it falls**, exactly as `docs/RESULTS.md` §5 and
`README.md` describe, because at `bias_update_rate=1e-3` the balancer needs
~100 steps to overtake the router's early collapse toward favourite
experts. A 200-step run catches the whole shape.

For comparison, the same preset at 1,000 steps reaches val 1.5190 and
writes recognisable pastiche; `README.md` "Results" has that run.

## What it writes

Prompt `KING RICHARD II:`, temperature 0.8, 150 tokens:

```
KING RICHARD II:
Why, I cere shate, and of there's at man some had-

HARDIZE:
That nor here, honour heart so follies should then, sost;
He condalge more, rether she b
```

Play-shaped, speaker-tagged, and made of non-words. At 200 steps the model
has learned the format of the corpus and little of its vocabulary. Do not
read this as a defect: it is what 0.8 of an epoch over 1 MB buys.

## Use

The architecture is not in `transformers`. Clone
[the repository](https://github.com/MOBIRIZER-TECHNOLOGY/AnuLM) and point
its scripts at this folder:

```bash
hf download toonist/AnuLM-Smoke-30B --local-dir AnuLM-Smoke-30B
python sample.py --ckpt AnuLM-Smoke-30B --prompt "KING RICHARD II:" --tokens 150
python serve.py  --ckpt AnuLM-Smoke-30B        # the web page, continue-the-text only
```

```python
from model import load_checkpoint, AnuLM
ck = load_checkpoint("AnuLM-Smoke-30B")
m = AnuLM(ck["cfg"]).eval(); m.load_state_dict(ck["model"])
```

Weights are `model.safetensors` in bfloat16, no pickle. Being a byte-level
model it decodes one byte at a time and emits invalid UTF-8 mid-sequence;
decode with `errors="replace"`, which `sample.py` does.

## Provenance

Trained 2026-09-18 from scratch on one machine, on public-domain text, by
the pipeline in the repository. No Sarvam weights are used or
redistributed; AnuLM is an independent re-implementation of a published
architecture. `docs/PUBLISHING.md` is how anything here gets released and
`TASKS.md` is the operations log.
