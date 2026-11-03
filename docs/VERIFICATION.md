# Verification: what has been checked, and how to check it yourself

This page answers "does this code work?" with commands you can run, the results
they gave on the development machine, what has **not** been checked, and the
known issues still open. Everything here can be re-run from a fresh clone.

## How to check it

From the repository root (or after `pip install -e .`, from anywhere):

```bash
python -m tests.test_model                                                   # 76 tests, ~2 min CPU
python -m unittest tests.test_training_pipeline tests.test_multimodal_contracts tests.test_release_fixes   # 65 tests, seconds
python -m tools.smoke                                                        # 11 end-to-end steps, ~1-2 min CPU
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
| `tests.test_model` | 76 passed, 0 failed |
| `tests.test_training_pipeline`, `tests.test_multimodal_contracts`, `tests.test_release_fixes` | 65 passed, 1 skipped (the installed-command check, because another copy of AnuLM is installed on this machine) |
| `tools.smoke` | 11 of 11 steps passed |
| every `python -m anulm` command with `--help` | 58 of 61 run; see "Known issues" for the three |
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
  and `hf_upgrade` mishandle `--help`.
- `rc_pointer train --max-len` is ignored when the starting checkpoint already
  records a `max_len`.
- `decide train-general` ignores a loaded head/LoRA mode and always optimises
  the full model.
- `voice --hear` always uses Whisper and ignores `--ear own`.
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
