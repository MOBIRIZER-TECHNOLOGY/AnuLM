# Speech

How AnuLM is being taught to listen and to talk, what is built, what is
measured, and what is still only a plan. Numbers here were measured on the
RTX 5070 Ti this project trains on; where a number is an estimate it says so.

## Two roads, and why both are open

**The cascade** (`voice.py`, `app_voice.py`) puts three separate models in a
row: faster-whisper hears, AnuLM answers in text, Piper speaks. It works
today, it is fast, and none of the speech is AnuLM's. It exists to be the
baseline — the end-to-end work is only interesting if it is measured against
something, and this is the something.

**End-to-end** (`audio_codec.py`, `speech_vocab.py`, `speech_data.py`,
`eval_speech.py`) puts speech *into the vocabulary*. SNAC turns audio into
discrete codes, those codes become 28,672 new token ids, and the same 400M MoE
backbone that writes Python learns to emit them. One model, one softmax.

## The architecture, and how it differs from Qwen3-Omni

Qwen3-Omni is the obvious reference, and the design here borrows one half of
it and refuses the other.

| | Qwen3-Omni | AnuLM |
| --- | --- | --- |
| audio in | AuT encoder, continuous | Whisper encoder, continuous |
| audio out | codec tokens | SNAC tokens |
| speech module | a separate **Talker** on the Thinker's hidden states | the same backbone |
| vocoder | Code2Wav, streaming | SNAC decoder (42 ms) |
| backbone | sparse MoE | sparse MoE |
| active params | ~3B | 232M after the vocabulary grows |

Borrowed: the **asymmetry**. Audio arrives as continuous encoder states,
because quantising the input throws away detail an ear needs, and leaves as
discrete tokens, because generation needs a vocabulary to put a softmax over.

Refused: the **Talker**. A second autoregressive module conditioned on the
first one's hidden states is how you get low-latency duplex conversation at
30B. At 232M active it doubles the training surface to buy something this
project cannot yet use. If streaming duplex ever becomes the goal, that is the
piece to add, and this document should be the thing that argued against it
first.

## Measured

A 3.87 s Hindi clip through both input paths (`python audio_codec.py --probe`):

| | rate | shape | notes |
| --- | --- | --- | --- |
| SNAC | 83.3 tok/s | 46 frames × 7 slots | vocab 4,096 per level, decode 42 ms |
| Whisper-small encoder | 50.2 fr/s | 194 × 768 | frozen, fp16 |

At a 2,048-token context that is **24.6 s** of audio discrete, **40.8 s**
continuous.

**The codec is not the ceiling.** Round-tripping that clip through SNAC and
back left it *more* legible to Whisper, not less:

```
original            नमस्ते मैं आनु हु, यहा एक परीक्षन है
after SNAC          नमस्ते मैं अनु हूँ यह एक परीक्षा है
```

**The vocabulary is not free.** Growing `ckpt_ctx2k.pt` from 32,768 to 61,443
ids costs 58.7M parameters across the untied `embed_tokens` and `lm_head`:

| | before | after |
| --- | --- | --- |
| total | 398M | 456M |
| active | 173.5M | **232.3M** |

15% more parameters but **34% more active** ones, because embeddings are dense
and the experts they sit beside are not. Active parameters set tokens/second,
so budget the audio runs at about three quarters of the 10.2k tok/s the text
runs managed (§27), not at parity.

**Budget.** An hour of speech is ~300k audio tokens. At ~7.5k tok/s that is
roughly **90 hours of audio per GPU-hour**, one epoch. LibriSpeech
train-clean-100 is ~30M tokens: a little over an hour per epoch.

## The pipeline

```
speech_vocab.py   grow a checkpoint's vocabulary          (once, ~1 min)
speech_data.py    corpus -> SNAC tokens + manifest        (once per corpus)
finetune.py --speech --task tts|asr                       (the run)
eval_speech.py    WER, judged by Whisper                  (the number)
```

The tasks are ordinary SFT pairs; only the alphabet is new:

```
tts   <|text|> transcript <|speech|>  ->  audio … <|/speech|>
asr   <|speech|> audio <|text|>       ->  transcript … EOS
```

which is why `finetune.py` needed a data branch and nothing else — the packer,
the answer-only masking and the loop are the text path untouched.

**They are not equally efficient.** A TTS row is ~50% masked; an ASR row is
~98%, because the audio sits in the prompt and the transcript is a dozen
tokens. Expect ASR to need far more wall clock per unit of learning, and
consider giving the audio prompt its own unmasked loss so the compute is not
thrown away. Not yet run.

## Scoring

Whisper is a much better listener than this model is a speaker, so it judges:
generate speech, transcribe it, compare to the text that asked for it. That
gives a word error rate, which can sit in RESULTS.md beside MBPP and chrF.

Two caveats that belong beside any number this produces:

* It measures **intelligibility, not quality**. A voice can be flat, badly
  paced and robotic and still score well. Use `--keep` and listen.
* **The floor is not zero.** Whisper's own errors are in every score.
  `eval_speech.py floor` measures the judge on the reference audio; on the
  synthetic corpus it is **2.8%**, and no run should be expected to beat it.

Generation is deliberately unconstrained. A trained model should learn to stay
inside the audio block after `<|speech|>`, and `stray` counts how often it
escapes — masking the text logits would make a weak checkpoint sound better
than it is, and hide the failure worth seeing.

## The continuous path

`speech_encoder.py` is the other input road, and the one Qwen3-Omni takes.
Whisper's encoder is frozen — ctranslate2 runs it as a C++ graph, so no
gradient could flow back even if it should — and a 12.6M projector stacks four
50 Hz states into one and maps them to the model's hidden size.

Measured on the same clip:

| | rate | 2,048 positions buy |
| --- | --- | --- |
| discrete (SNAC) | 85.0 Hz | **24 s** |
| continuous (stack 4) | 12.1 Hz | **169 s** |

Seven times the audio per position, and the same rate as SNAC's coarse level,
which is what makes the two comparable per position rather than per second.

It needs its own training loop, and that is not stubbornness: `finetune.py`
packs many pairs into one row of *ids*, and a continuous prompt has no ids to
pack. Rows here are one clip each, padded, loss masked to the transcript —
less efficient than packing, and the smallest honest way to train something
whose prompt is a matrix.

One change reaches into the text path: `AnuLM.forward` takes an optional
`embeds=` that bypasses the embedding table. Twelve lines, and the text path's
tests still pass.

## Vision

`vision_encoder.py` is the same mechanism again, which is the point of having
built it twice. A frozen SigLIP ViT-B/16 gives 196 patch vectors, 2×2 pooling
takes that to 49, and a projector maps them in:

```
<|image|> [49 projected patches] <|/image|> caption … EOS
```

One image costs **49 of 2,048 positions (2.4%)**. The image control tokens were
added to the vocabulary at the same time as the audio ones, because growing it
twice would shift every id after the audio block and invalidate anything
trained before the change.

**There is no judge here.** Speech could lean on Whisper. Captioning's
automatic metrics reward copying the reference's phrasing, so this file reports
held-out loss and prints samples, and does not claim a headline number. A
vision result worth publishing needs a real metric harness or a stated human
protocol, and that decision has not been made.

## The first real run (LibriSpeech dev-clean)

4.97 h of read English, 2,642 clips, 1.48M audio tokens. Three hours of
training on one RTX 5070 Ti.

**The floors first**, because a TTS number means nothing without them:

| | WER |
| --- | --- |
| Whisper-small on the original audio | **3.78%** |
| the same audio after a SNAC round trip | **7.45%** |
| what the codec costs | **+3.67 points** |

That corrects an earlier claim in this file's history. On one synthetic clip
the round trip *improved* legibility; over 200 real clips it nearly doubles the
error rate. One clip was not evidence.

**The run:**

| | |
| --- | --- |
| held-out loss | 12.83 → **7.27** |
| best step | 1099 of 1852 (~4.7 of 8 epochs) |
| wall clock | 10,718 s (2 h 59 m) |
| **TTS WER** | **100.4%** over 50 clips |
| silent clips | 0 |
| stray non-audio tokens | 45 across 50 clips |

**It learned the structure and not the content, and the two numbers say so
separately.** Zero silent clips and 45 stray tokens over ~20,000 generated
means the model reliably enters the audio block, emits ids that live inside it,
and stops at `<|/speech|>` — the frame grammar is there. But 7.27 nats against
`ln(28672)` = 10.26 is the signature of *unconditional* audio modelling: it
learned what English speech sounds like, not what this sentence says. Played
back it is fluent, well-paced babble, and Whisper gamely parses it as French or
Danish.

The structural reason is in the packing. 77% of positions are audio, and the
text prompt is ~15 tokens against a ~500-token answer, so the cheapest descent
direction is to predict audio from preceding audio and ignore the prompt
entirely. The text signal is drowned. More data is the first lever; making the
model pay for ignoring the prompt is the real one.

WER above 100% is not a bug: insertions count, and the model says more than it
was asked to.

## The second run: 20x the data (train-clean-100)

100.58 h, 28,538 clips, 29.80M audio tokens, encoded in 9 minutes at 691x real
time on a free GPU. 1,500 steps -- 0.27 of an epoch, so every step saw audio
the model had never met, against dev-clean's eight passes over five hours.

| | dev-clean (5 h, 8 epochs) | probe (100 h, 0.27 epochs) |
| --- | --- | --- |
| best held-out loss | 7.2663 @ step 1099 | **7.1067 @ step 1500** |
| end of run | overfitting since 1099 | still improving |
| TTS WER | 100.4% | **100.0%** |
| stray non-audio tokens | 45 | 10 |
| codes out of slot | not measured | 1,353 |
| mean output rms | 0.0136 | **0.0049** |
| inaudible clips (<0.005) | 30 / 50 | **39 / 50** |
| real speech, for scale | | rms 0.0636 |

**The loss improved and the audio got worse, and that is the finding.**

Cross-entropy rewards hedging. Speech is a high-variance signal around
near-silence, so the distributional average of audio codes is *quiet*: a model
that is unsure lowers its loss by predicting the mean, and the mean is
inaudible. Eight passes over dev-clean forced that model to commit to specific
loud codes; 0.27 of an epoch never forced the probe to commit to anything, so
it settled into the hedge -- better loss, ten times quieter, and 39 of 50
clips below the threshold of audibility.

So **held-out cross-entropy on audio tokens is not a proxy for audio quality**,
and this project should stop reading it as one. The loudness columns above
exist because the first three clips came back from Whisper as `(nothing)`
rather than as the babble the previous run produced, which was the only reason
anyone looked. `eval_speech.py` now reports mean rms and counts inaudible
clips, so a quiet model cannot score well unnoticed again.

Two smaller results. The `stray` count fell from 45 to 10, so more data does
teach the model to stay inside the audio block. But `off_slot` -- 1,353 codes
landing in the wrong slot of their frame across 50 clips -- says it still has
not learned SNAC's 7-slot frame grammar, which is a different and more basic
failure than not knowing the words.

**What neither run was.** dev-clean converged on too little data; the probe saw
plenty of data and converged on none of it. Neither is the experiment that
settles whether this architecture can speak. That needs enough steps on the
large corpus to force commitment -- 5 to 10 epochs of train-clean-100, roughly
27,000 to 55,000 steps, 40 to 80 GPU-hours uncontended. That is a project, not
a probe, and it should be started deliberately.


## Status

Every stage runs end to end on real data. The pipeline is done; the model is
not. Vision and the continuous path have been exercised only on synthetic data
(six Piper clips, six drawn shapes) and are shape-correct, not trained.

Two bugs worth remembering, both found by running the thing rather than reading
it:

* `unflatten` subtracted each slot's offset positionally, which assumes the
  model emits codes in slot order. A trained model does; an undertrained one
  sends a slot-0 code to position 3, the index goes negative, and SNAC dies
  several frames later with `CUDA error: device-side assert triggered` inside a
  noise layer that has nothing to do with the mistake. Codes are now clamped
  into their slot and counted (`off_slot`).
* `finetune.py` wrote its curve to stdout and nowhere else, so a pipe through a
  block-buffered `grep` left three hours of evals invisible until the process
  exited. It now appends each point to `<out>_curve.csv` as it is measured.

train-clean-100 has now been run (above) and the answer was no: fewer epochs on
far more data trades intelligibility for a better loss. The held-out set was
raised to 5% (1,185 rows) via `--heldout-frac`, which is the one change from
that plan worth keeping.

Next is a decision rather than a command: either commit 40-80 GPU-hours to a
converged run on train-clean-100, or accept that 232M active parameters is too
small for this recipe and say so with the numbers above.

## Open questions

* Does the dense audio embedding limit the run? If so, tie the audio block's
  `lm_head` to `embed_tokens`, or factorise it into 7 slot vectors plus 4,096
  code vectors, and halve the bill.
* Is 83.3 tok/s worth it? Modelling only SNAC's coarse level is 11.9 tok/s and
  seven times the context, at the cost of needing something else to restore the
  fine levels.
* Does the continuous input path actually beat the discrete one at this scale,
  or does a frozen 768-dim encoder simply overwhelm a 1024-dim model?
