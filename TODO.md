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
   the models today. Also wanted: a dropdown in `app.py` that switches
   between the four checkpoints, loading each on first use (1.6 GB per
   model on CPU).
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
5. **Longer context.** Every checkpoint is trained at 512 tokens. The
   model already implements YaRN (`docs/ARCHITECTURE.md` §4, `README.md`
   "Long-context extension"); a continued-pretraining run at 2,048 with
   YaRN on the coder corpus, measured on `eval_context.py`, would show
   whether the design's long-context claim holds at this scale.
6. **Better coder data.** The coder plan's own analysis
   (`docs/CODER_PLAN.md`, "If the mix is the ceiling") says textbook-style
   Python is what moves pass@1 at this size. `python-edu` from the SmolLM
   corpus is the open match and needs a Software Heritage download per
   file. Adding it and re-running the 250k-step probe would be a clean
   experiment.

## Tooling

7. **A GGUF conversion.** llama.cpp has no loader for this architecture;
    the MoE with sigmoid routing and QK-norm would need a new arch entry.
    Large, but it would put the models on phones.

## Documentation

8. **Translate the tutorial into Hindi (no GPU).** `docs/TUTORIAL.md` in
    Hindi would match the project's audience.
9. **Diagrams for `docs/ARCHITECTURE.md` (no GPU).** The MLA / GQA
    comparison and the MoE routing are explained in prose; two figures
    would help.

## Done

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
