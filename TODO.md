# To do

Open work, ordered by how much it would help other people. Each item is
also a GitHub issue; discuss there, and open a PR against `main`. Items
marked **(no GPU)** can be done on a laptop.

## Release

1. **Zenodo DOI (no GPU).** The maintainer signs in to
   https://zenodo.org with GitHub, enables `MOBIRIZER-TECHNOLOGY/AnuLM`
   under GitHub settings, and publishes a `v0.1.1` release; Zenodo mints
   the DOI. Then add the DOI badge to `README.md` and the `doi:` field to
   `CITATION.cff`. Only the account owner can do the first step.
2. **Hosted demo (no GPU).** Hugging Face requires a PRO subscription
   to host Gradio Spaces even on the free CPU tier (the API returns 402),
   so no Space is up; `app.py` and `space/` are ready for anyone with
   PRO or ZeroGPU access, and `demo_colab.ipynb` is the free way to try
   the models today. The checkpoint dropdown is done.
3. **Linux reproduction report (no GPU for the small parts).** CI now
   runs the tests and a 50-step training run on Ubuntu for every push, so
   the small path is covered; nobody has yet run the *tutorial* end to end
   on Linux. Run §1–§4 of `docs/TUTORIAL.md` on a Linux box, note every
   place the instructions were unclear, and open one issue or PR with the
   fixes.

## Model and training

4. **A chat fine-tune.** None of the checkpoints can hold a conversation
   (`docs/MODEL_CARD_HINDI_QA.md` shows why). A dialogue stage on the
   three-language base with a public instruction set (OpenAssistant
   oasst2, Dolly-15k; Aya or indic-align for Hindi), a few hundred
   identity examples, and a turn-marked template. `finetune.py` already
   handles question/answer pairs; multi-turn packing within 512 tokens is
   the new part. Expect 1–3 GPU-hours and modest quality at 400M.
5. **Better coder data — rethink before running.** §31 ran the textbook
   probe this item proposed, and it did not do what the coder plan expected:
   after the §25 SFT recipe the textbook corpus scored MBPP 14.4% (up from
   12.5%) and HumanEval **0.0%** (down from 4.9%). The synthetic exercises
   specialised the model toward MBPP-shaped problems and cost it HumanEval
   entirely, so adding `python-edu` — more exercise-style data, and 3.8M
   Software Heritage fetches — would most likely deepen that trade rather
   than reverse it. The open question is now what data moves *HumanEval* at
   this size, and a mixed corpus that keeps the §25 GitHub share while adding
   exercises is the cheaper test.

## Speech and vision

The pattern from `docs/SPEECH.md` sets the order here. A task works when the
model only has to *look* or *listen* and answer in the vocabulary it already
speaks — captioning needs a 12.6M projector and twelve output tokens. It fails
when the model must *produce* a new modality — speech out needed 58.7M new
embeddings and six hundred tokens in a language it had never spoken. So the
input-side items come first; they are the likely wins.

9. **Speech recognition, continuous path.** Train `speech_encoder.py` on
   LibriSpeech train-clean-100 (28,538 clips, already encoded and on disk):
   frozen Whisper encoder, a projector, output in the existing text
   vocabulary. Structurally the same as captioning, which works. Measure WER
   on the held-out slice with `speech_encoder.py eval`, against Whisper's own
   3.78% floor on the same audio. ~2.4 GPU-hours per epoch; the trainer now
   checkpoints every 500 steps, so intermediate checkpoints can be scored
   rather than waiting for the end. **In progress.**
10. **A spoken turn with the project's own ear.** Once 9 works, route
    `voice.py` through it instead of Whisper: own ASR -> a reasoning
    checkpoint -> Piper. No new training beyond 9. The cascade currently
    borrows both the ear and the mouth; this would make the ear the
    project's.
11. **Visual question answering.** Captioning works; answering a question
    about an image does not exist. Same frozen tower and projector with an
    image + question -> answer template. Needs VQA data — VQAv2 or GQA for
    English, and a Hindi VQA set if one is open. The projector from
    `ckpt_caption_p1.pt` is a warm start.
12. **Hindi speech recognition.** After 9 works in English, train on the
    IndicTTS speaker already encoded (11.84 h). Two speakers only, so it
    will overfit to those voices; `ai4bharat/IndicVoices` (~1,600 h) and
    `ai4bharat/Shrutilipi` (~6,400 h) are the real corpora and are gated —
    someone with a Hugging Face account should request access.
13. **Intelligible speech out: model only SNAC's coarse level.** The Hindi
    TTS run solved the acoustics (loudness matched to source within 1.3%,
    zero inaudible clips) and not the alignment (WER 131%). SNAC at 83.3
    tok/s makes a seven-second utterance ~600 target tokens against ~30
    text tokens. Modelling only the 11.9 Hz coarse level cuts that to ~85 —
    seven times shorter — with a small upsampler restoring the fine levels.
    Design this before spending compute: both more data (5 h -> 100 h) and
    more epochs have already been shown not to help, and past ~4.7 epochs
    the runs collapse to the uniform line.
14. **More caption data.** Flickr8k is 6,000 images. Flickr30k
    (`nlphuji/flickr30k`, 4.39 GB) ships images as a zip with captions in a
    separate file, so `vision_encoder.py prep` needs a second reader. Use
    the known-good recipe: `--pool 1`, three epochs (six was worse), and a
    batch of at least 8 — batch size moved the loss more than epoch count.

## Evaluation

15. **Re-score MBPP with the entry-point fix.** 14 of the 257 sanitized
    problems derived the wrong function name from a wrapped assertion
    (`assert math.isclose(volume_sphere(10), ...)` -> "math.isclose"), so in
    question mode they were asked to "write a Python function
    `math.isclose`". Fixed in `eval_code.py`; §25's 12.5% and §31's 14.4%
    are both understated by it. The A/B between them survives because the
    bug hit both, but the absolute numbers should be re-measured — on an
    otherwise idle machine, since the timeout counter shows concurrent load
    moves pass@1 too.
16. **Finish the evaluation review (no GPU).** `eval_code.py` is reviewed.
    Still to read with the same eye: `eval_translate.py` (behind chrF
    41.5 / 43.4), `eval_qa.py`, `eval_golden.py`, `eval_context.py`, then
    `train.py` and `bpe.py`. Three evaluations misled rather than the model
    in the last two days — a format confound, a missing loudness check, and
    a four-clip sample — so every published number deserves a second look.
17. **Log median and max router imbalance, not just max (no GPU).**
    `train.py` logs `max()` over layers, which reported 4.33x for a model
    whose median layer is 2.15x. Measured on the textbook checkpoint the
    balancing is healthy — 83% of top-k picks survive removing the bias —
    but deep layers do specialise hard (layers 15–18 at 3.6–4.2x, layer
    18's top three experts taking 47% of tokens). Worth logging the
    distribution so that shape is visible without a separate probe.

## Tooling

6. **A GGUF conversion.** llama.cpp has no loader for this architecture;
    the MoE with sigmoid routing and QK-norm would need a new arch entry.
    Large, but it would put the models on phones.

## Documentation

7. **Translate the tutorial into Hindi (no GPU).** `docs/TUTORIAL.md` in
    Hindi would match the project's audience.
8. **Diagrams for `docs/ARCHITECTURE.md` (no GPU).** The MLA / GQA
    comparison and the MoE routing are explained in prose; two figures
    would help.

## Done

- 2026-09-26: router balancing restored in both encoder trainers.
  `vision_encoder.py` and `speech_encoder.py` never called
  `update_expert_biases()`, so the aux-loss-free balancing was silently off
  in every captioning and continuous-speech run — biases frozen at the init
  checkpoint's values, load counts growing without bound. `speech_encoder.py`
  also gained the periodic eval-and-save the vision trainer already had.
- 2026-09-26: `eval_code.py` MBPP entry points taken from the reference
  solution's `def` rather than the first assertion; 14 of 257 were wrong.
- 2026-09-26: patch pooling settled by a controlled run. `--pool 1` beats
  2x2 on loss by 0.059 at matched batch and epochs, but not clearly on
  captions; the first captioner's colour errors were one epoch of training,
  not pooling. Three epochs is the ceiling — six peaked at 2.8 and got worse.
- 2026-09-25: §31 answered. After the §25 SFT recipe the textbook corpus
  scored MBPP 14.4% (12.5% before) and HumanEval 0.0% (4.9%): a trade, not a
  win. The earlier 0.0% on both in continue mode was the corpus's markdown
  problem-statement format, not ability. Item 5 rewritten in light of it.
- 2026-09-25: image captioning works (`vision_encoder.py`, `app_vision.py`).
  Frozen SigLIP, a projector, Flickr8k; held-out captions such as "A brown
  dog is running on the beach" for "A brown dog is running along a beach".
- 2026-09-25: speech generation built and measured end to end. Acoustics
  work, alignment does not — details and every negative result in
  `docs/SPEECH.md`. The cascade (`voice.py`, `app_voice.py`) works and is
  3–6 s per spoken turn, fully local.

- 2026-09-19: long context, both halves (old item 5). §27 measured the
  released base at 2x and 4x its training length zero-shot; §28 continued it
  at 2,048 with YaRN for 10,000 unattended steps. Everything improved
  (Hindi −0.068, English −0.131, Python −0.233 bits/byte) but the *context*
  part of that is small: zero-shot YaRN already turns the loss-by-position
  slope from +0.073 to −0.068, and five GPU-hours took it to −0.085. The
  first evaluation was contaminated and thrown away; §28 says how.

- 2026-09-19: `app.py` holds all four checkpoints in a dropdown and loads one
  on demand, evicting the previous one so the page stays inside Colab's free
  tier. The modes, examples and defaults follow whatever is loaded, and
  `demo_colab.ipynb` is two cells now instead of three.

- 2026-09-19: val loss reports its own standard error, and `--eval-windows N`
  spreads N windows evenly over the split instead of drawing random batches
  (old item 7). Measured 4-11x closer to the true mean for the same number of
  windows, and unlike the random set it does not change when `--batch-size`
  does. `docs/RESULTS.md` §26. The default is untouched, so every number
  already in that file still reproduces.

- 2026-09-19: `eval_code.py` sandboxing (old item 7). A fresh directory per
  problem, the process tree killed on timeout, output to a capped file
  rather than a pipe, and `setrlimit` caps on address space, CPU, file size
  and process count on Linux and macOS. Windows has no rlimits and the
  module says so. Tested against an infinite loop, an orphaned grandchild
  and a program that shadows the standard library.

- 2026-09-18: the four released checkpoints load with
  `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` and
  `AutoTokenizer.from_pretrained(...)`, with nothing cloned. Closes the old
  items 9 and 10. The tokenizer conversion is verified token-for-token
  against `bpe.py` on 497 passages of real Hindi, Python and English before
  it is written; the model wrapper is held to the native path's logits and
  greedy output by `test_model.py`. Two traps found on the way, both now
  tested: the non-persistent rotary buffers came back as uninitialised
  memory, and bfloat16 weights change the router's expert selection badly
  enough to collapse the output.

- 2026-09-18: code, docs, recipes on GitHub (v0.1.0); four checkpoints on
  Hugging Face; Colab demo notebook; tutorial, dataset table, model cards,
  contributor guide; one GitHub issue per item above.
- 2026-09-18: onboarding — `quickstart.py` (environment check, then
  `--train` a 17M model or `--demo` a released one), GitHub Actions CI on
  Ubuntu and Windows with the CPU wheel, and issue forms that ask for the
  environment block `quickstart.py` prints. Closes the old item 3.
- 2026-09-18: documentation pass over the finished project — the two plan
  documents carry their final status, `docs/RESULTS.md` opens with an index
  of its 25 sections, `docs/ARCHITECTURE.md` §10 covers §22–§25, the layout
  in `docs/DEVELOPING.md` lists every shipped file, and `NOTICE` §3 lists
  every runtime dependency. `python test_model.py`: 47 passed, 0 failed.
