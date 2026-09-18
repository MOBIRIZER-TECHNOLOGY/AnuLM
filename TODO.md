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
3. **Continuous integration (no GPU).** A GitHub Actions workflow that
   runs `python test_model.py` on Ubuntu and Windows with the CPU torch
   wheel, and a 200-step `--preset 30b` smoke run, on every PR.
4. **Linux reproduction report (no GPU for the small parts).** Nobody has
   yet run the tutorial end to end on Linux. Run §1–§4 of
   `docs/TUTORIAL.md`, note every place the instructions were unclear,
   and open one issue or PR with the fixes.

## Model and training

5. **A chat fine-tune.** None of the checkpoints can hold a conversation
   (`docs/MODEL_CARD_HINDI_QA.md` shows why). A dialogue stage on the
   three-language base with a public instruction set (OpenAssistant
   oasst2, Dolly-15k; Aya or indic-align for Hindi), a few hundred
   identity examples, and a turn-marked template. `finetune.py` already
   handles question/answer pairs; multi-turn packing within 512 tokens is
   the new part. Expect 1–3 GPU-hours and modest quality at 400M.
6. **Longer context.** Every checkpoint is trained at 512 tokens. The
   model already implements YaRN (`docs/ARCHITECTURE.md` §4, `README.md`
   "Long-context extension"); a continued-pretraining run at 2,048 with
   YaRN on the coder corpus, measured on `eval_context.py`, would show
   whether the design's long-context claim holds at this scale.
7. **Better coder data.** The coder plan's own analysis
   (`docs/CODER_PLAN.md`, "If the mix is the ceiling") says textbook-style
   Python is what moves pass@1 at this size. `python-edu` from the SmolLM
   corpus is the open match and needs a Software Heritage download per
   file. Adding it and re-running the 250k-step probe would be a clean
   experiment.
8. **HumanEval and MBPP as a training-free eval harness (no GPU).**
   `eval_code.py` executes model output in a subprocess with a timeout.
   Hardening it (resource limits, a temp working directory per problem,
   Windows and Linux parity) would make it safe to run on untrusted
   checkpoints.
9. **Less noisy validation.** Val loss is read on 20 windows and carries
   about ±0.03 of noise (`docs/RESULTS.md` §25, "A review of the
   training loop"). Raising `--eval-iters` costs time at every eval; a
   fixed, larger, cached validation set would make small differences
   readable.

## Tooling

10. **`transformers`-loadable export (no GPU).** `export_hf.py` writes
    safetensors that only this repository's `model.py` can load. A
    `modeling_anulm.py` + `configuration_anulm.py` pair with
    `trust_remote_code` would let people load the weights with
    `AutoModelForCausalLM`.
11. **Tokenizer in `tokenizers` format (no GPU).** `bpe.py` uses its own
    JSON. Emitting a `tokenizer.json` that the `tokenizers` library reads
    would remove the last custom piece from the export.
12. **A GGUF conversion.** llama.cpp has no loader for this architecture;
    the MoE with sigmoid routing and QK-norm would need a new arch entry.
    Large, but it would put the models on phones.

## Documentation

13. **Translate the tutorial into Hindi (no GPU).** `docs/TUTORIAL.md` in
    Hindi would match the project's audience.
14. **Diagrams for `docs/ARCHITECTURE.md` (no GPU).** The MLA / GQA
    comparison and the MoE routing are explained in prose; two figures
    would help.

## Done

- 2026-09-18: code, docs, recipes on GitHub (v0.1.0); four checkpoints on
  Hugging Face; Colab demo notebook; tutorial, dataset table, model cards,
  contributor guide; one GitHub issue per item above.
- 2026-09-18: documentation pass over the finished project — the two plan
  documents carry their final status, `docs/RESULTS.md` opens with an index
  of its 25 sections, `docs/ARCHITECTURE.md` §10 covers §22–§25, the layout
  in `docs/DEVELOPING.md` lists every shipped file, and `NOTICE` §3 lists
  every runtime dependency. `python test_model.py`: 47 passed, 0 failed.
