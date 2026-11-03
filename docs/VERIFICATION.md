# Verification: what has been checked, and how to check it yourself

This page answers "does this code work?" with commands you can run, the results
they gave on the development machine, what has **not** been checked, and the
known issues still open. Everything here can be re-run from a fresh clone.

## How to check it

From the repository root (or after `pip install -e .`, from anywhere):

```bash
python -m tests.test_model                                                   # 80 tests, under a minute on CPU
python -m unittest tests.test_training_pipeline tests.test_multimodal_contracts tests.test_release_fixes   # 79 tests, seconds
python -m unittest discover -s tests -t . -p "test_quality_*.py"               # 63 tests, seconds
python -m tools.smoke                                                        # 11 end-to-end steps, ~65-80 s on CPU
```

`tools.smoke` is the one to run first. Each step runs as its own process,
exactly as the docs tell a newcomer to type it, and the run ends with a
pass/fail table:

| step | what it proves |
| --- | --- |
| train bytes | the training loop runs and fetches its sample data |
| train a BPE tokenizer, encode the corpus | the tokenizer trains, round-trips exactly, and encodes |
| train 30 steps on BPE | training on an encoded corpus |
| sample | generation from a checkpoint |
| fine-tune on Q&A, ask questions | supervised fine-tuning (including dropping a held-out row that repeats a training question) and the question mode |
| score held-out text | evaluation in bits per byte |
| export to HF folder, load export without repo | the Hugging Face export loads with `transformers` alone, with this package unimportable |
| serve over HTTP | the web server answers `/info` and `/generate` |

> **Gotcha.** If another copy of AnuLM is `pip install -e`'d on the same
> machine, a process started outside this checkout imports that copy. The tests
> and `tools.smoke` set `PYTHONPATH` to this checkout for their children; do
> the same if you run commands from another directory.

## Results on the development machine (2026-11-03)

Windows 11, Python 3.14.7, torch 2.11 (CUDA build, run on CPU), transformers 5.15.

| check | result |
| --- | --- |
| `tests.test_model` | 80 passed, 0 failed (also under `python -m unittest tests.test_model`) |
| `tests.test_training_pipeline`, `tests.test_multimodal_contracts`, `tests.test_release_fixes` | 79 passed, 1 skipped (the installed-command check, because another copy of AnuLM is installed on this machine) |
| `tests.test_quality_*` (layout, loading, serving) | 63 passed |
| `tools.smoke` | 11 of 11 steps passed |
| `ruff check .` (the configuration in `pyproject.toml`, which CI runs) | clean |
| every `python -m anulm` command with `--help` | 53 of 57 print their help; see "Known issues" for the four |
| restructure regression | a 50-step training run before and after the move into `anulm/`: identical loss and router imbalance at every step, val 2.7870 both |
| old checkpoints | all 89 local checkpoints pickle only `model.AnuLMConfig`, which the root `model.py` alias resolves; `ckpt_chat_v5.pt` loaded |
| Omni model layer, CPU, local `ckpt_omni.pt` | chat answers (Hindi "the capital is Delhi" correct; English answer repetitive), intent "my card has not arrived" -> `card_arrival` at 1.0, reader retrieved the right passage but extracted a wrong answer |

The Omni rows are a **smoke check of the code path, not a quality score**: the
local Omni checkpoint predates the current training recipe (see
`docs/OMNI.md`), and its chat and reading quality are known to be weak.

## Not checked

- Anything on the GPU since the restructure: the card was occupied by other
  processes. CPU and GPU share every code path except `--moe-impl grouped`,
  which needs `torch.nn.functional.grouped_mm`, and bf16 autocast.
- Real Whisper and SigLIP runs (speech and image tabs) inside the demo.
- A full Omni training run with the current recipe, and resume after an
  interruption (the code is read and unit-tested, not run end to end).
- `python -m anulm api` against all its models; `python -m anulm app_omni` in a
  browser.

## Known issues (open)

Found while documenting and reviewing; none is caused by the restructure.

- `make_rc` has no `--help` and starts building immediately; `hf_tokenizer`
  and `hf_upgrade` mishandle `--help`; `python -m anulm model --help` runs the
  module's shape demo instead of printing help (harmless, no files written).
- `rc_pointer train --max-len` is ignored when the starting checkpoint already
  records a `max_len`.
- `decide train-general` ignores a loaded head/LoRA mode and always optimises
  the full model.
- `voice --hear` always uses Whisper and ignores `--ear own`.
- The reviewed-cases page (`validated_demo`) ships with an empty catalogue:
  no Omni result has been recorded and reviewed yet (docs/DEMO_VALIDATION.md).
- `ask` does not support byte-level checkpoints (no tokenizer). It loads the
  project's BPE and Hugging Face `tokenizer.json` checkpoints.
- `rag`, `finetune` and the `eval_*` scripts still load tokenizers with
  `BPE.load`, so they need a checkpoint with the project's own BPE tokenizer.
- `eval_bench` divides the whole file's bytes by the tokens kept after
  truncation to whole windows, slightly understating bits per byte on short files.
- The fetchers and `mix_corpus` take budgets in MiB but print MB (about 5% apart).
- `make_rc.window` centres on an answer's start, so answers longer than about
  60 words can be cut.
- `AnuLMDecide.batch` (LangChain helper) ignores `return_exceptions`.
- The Omni demo's default model, `toonist/AnuLM-Omni-400M`, is not published,
  so a fresh clone's Omni demo has no model until one is trained or published.
- A local SigLIP folder path is saved into checkpoints as `tower`, so such a
  checkpoint's image tab only works on a machine with the same folder.

## Tests checked by mutation (2026-11-03)

21 single, plausible bugs were planted in `anulm/model.py` and `anulm/lora.py`,
one at a time, and `tests.test_model` was run against each. 8 of the 21 passed
unnoticed: RMSNorm's epsilon outside the square root, SwiGLU's gate and up
weights swapped, GQA heads tiled instead of interleaved, embeddings initialised
50 times too large, the router gate not run in fp32, `tie_word_embeddings`
ignored, the LoRA scale set to alpha instead of alpha/r, and `router_state_dict`
returning zeros. Four tests were strengthened and four added (76 -> 80); now
every one of the 21 fails at least one test, and each new or changed test
passes on the real code. No library bug was found.

## Fixed in the code-quality pass (2026-11-03)

A four-part review (structure, core library, serving, tests and CI) found
these; each fix has a test in `tests/test_quality_*.py` that fails on the code
before it, unless noted.

- **`/v1/decide` failed on every call** with the joint (Omni) model, the
  default: the API passed `options` to an intent classifier that did not take
  them. Trained labels are now scored; an unknown option is a 400.
- **Loading is weights-only.** Every checkpoint and resume file goes through
  `anulm.model.safe_load` (`torch.load(weights_only=True)` with the config
  class allow-listed under its three pickled names); a file it refuses raises
  `UnsafeCheckpointError` naming the explicit escape (`trust_pickle=True` or
  `ANULM_TRUST_PICKLE=1`). All local checkpoints and resume files load
  unchanged. A downloaded folder can no longer point `tokenizer_path`,
  `hf_backbone` or `base_model` outside itself.
- **Input checks survive `python -O`:** `assert`s that validated user input or
  files raise `ValueError` (or a CLI error); `hf_upgrade`'s parity checks are
  real errors.
- **API hardening:** constant-time key check; a start-up warning when it is
  reachable without a key; upload limits (audio 25 MB and 60 s, images 10 MB,
  decide 64 inputs, 256 options, 8,000 characters; `serve` 1 MB); 500 replies
  carry an error id instead of the exception text. Limits are in `docs/API.md`.
- **The shared demo is locked:** one lock covers model loading and every call
  on the shared decoder, and a visitor's threshold can no longer change another
  visitor's verdict. The retriever honours `--device`; a second reading model
  no longer changes the first one's length limit.
- **Tests:** the 76 core tests also run under `unittest` and pytest; 15 tests
  that returned early when an optional package was missing (and counted as
  passed) now report SKIP; two timing-dependent checks were fixed.
- **CI:** read-only permissions, Python 3.10, the smoke test, a `ruff` lint
  job, the quality tests, and one step per test command.
- **Hygiene:** `.demo_auth` (the demo's password file) and logs are
  git-ignored; six commands that ran nothing are removed; tools no longer
  hard-code a local path or write outside the repository; dated one-off
  scripts are in `experiments/archive/`; version 0.2.0.

## Fixed after the three-way comparison (2026-11-03)

The clean code was compared with V3 and with a later bundle built on another
machine (2026-11-02); these came out of that comparison. Each has a test in
`tests/test_release_fixes.py` that fails on the code before the fix.

- The API lost the last characters of a streamed reply when the client set
  `stop` and the model ended with its end token; `stop=""` emptied every reply;
  `top_k=0` was silently turned into 40.
- Streamed text never shows half a Devanagari character as U+FFFD, and a real
  U+FFFD is no longer deleted (`decode_complete`, holding an incomplete UTF-8
  tail back).
- `temperature=0` is exact greedy decoding (`AnuLM.generate`); it used to be a
  sample at temperature 1e-5, which changed with the seed on a tie.
- `finetune` read its packed-row cache with `weights_only=False` and swallowed
  every error, so a file beside the data could run code; it now refuses
  anything but tensors and checks their shape and dtype.
- `finetune` drops held-out rows whose question is also in training (the
  release chat files share 4), and `make_chat` splits by question
  (`question_key`) so a rebuild cannot put one question on both sides.
- `omni_train` seeds torch before building fresh projectors, so a rerun starts
  from the same weights, and refuses to resume a `.last` written under a
  different recipe.
- `load_tokenizer` decides by content rather than by the file name
  `tokenizer.json`; `serve`, `sample`, `sample_many` and `ask` accept Hugging
  Face tokenizers, and replies stop before the end token's text.
- `sample_many --temp 0` fell back to each prompt's own temperature.
- CI installs the packages the API, demo and tokenizer tests need, instead of
  letting those tests skip, and installs the package to test the command from
  outside the checkout. `pyproject.toml` gains an `api` extra.
- A stray `anulm/training/train.py.orig` from the restructure was removed.

Second pass, the same day:

- Five FLEURS Hindi training clips are longer than Whisper's 30-second window
  and stopped the first Omni training step; training now drops such clips with
  a count (cropping would pair half the audio with the whole transcript).
  Whisper refuses NaN or infinite audio, and `transcribe` asks for at most the
  tokens that fit beside the audio in the context.
- Decontamination checked every 4th 13-word window of a document, which missed
  6 of 8 positions of an exact 13-word overlap; it checks every window now.
- `export_hf` reloads every export and stops unless it is bit-identical to what
  it meant to write; `--weights-dtype float32` makes an export exactly the
  `.pt`. Exported model cards no longer show flat-layout imports.
- `python -m anulm validated_demo` shows only recorded, reviewed cases and runs
  no model (docs/DEMO_VALIDATION.md).

## Fixed in the documentation round

- `experiments/omni_one.py` and `experiments/chat_samples.py` resolved `ROOT` one
  level above the repository; a test now checks every experiment script's `ROOT`.
- The release data's explicit held-out files repeat a few training prompts
  (4 chat, 1 reading); `omni_train` now drops those from the held-out side
  instead of refusing to train, so held-out scores no longer include seen prompts.
- Omni training no longer runs Whisper or SigLIP over every clip and image at
  each start to check context length; the media length is bounded instead.
- The demo's bf16 path kept projector biases in fp32 beside bf16 weights,
  breaking the speech and image tabs; only the routers now stay fp32.
- `convert_parquet --decontaminate` silently filtered nothing outside the
  repository root; it now resolves its data from the repository and fails if
  the benchmark files are missing.
- `train_from_scratch` killed any stage after 2 hours by default and showed no
  output until a stage ended; it now has no default limit and streams output.
- `omni_train --accum 0` trained nothing and saved a "best" checkpoint; it is
  now rejected. The API's 404 for a missing speech model is restored.
- `sample`, `sample_many` and `ask` crashed on checkpoints without a recorded
  loss or step; the voice demo passed a dropdown label instead of a model id;
  `finetune` raised a traceback when its default held-out file was absent.
- `fetch_speech` checked tar members with a string prefix (a sibling folder
  passed) and did not refuse links; it now compares paths and refuses links.
- `make_chat` had no argument parser: `make_chat --help` rebuilt
  `data/chat_*.jsonl`, and with `data/dolly_train.jsonl` or `data/qa_lora.jsonl`
  absent it silently built a 4,781-row set in place of the 23,022-row release
  set. It now has `--help`, refuses to overwrite without `--force`, and fails
  when a local source is missing; `omni_data fetch chat` passes `--force`.
