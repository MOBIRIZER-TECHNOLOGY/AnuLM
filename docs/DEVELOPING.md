# Developing

## Setup

```bash
python -m anulm quickstart                   # what this machine can run, and how to fix what it cannot
pip install -r requirements.txt        # torch, safetensors; pyarrow for the parquet converters
python -m tests.test_model                   # 66 tests, ~2 min — run this first
```

Everything runs from the repository root (this file lives in `docs/`, the
annotated Sarvam code in `sarvam/`, the model beside them). One third-party
dependency for the model, training, sampling, serving and every evaluation:
**torch**. No transformers, no datasets, no numpy. **safetensors** is used
only to read and write the exported checkpoint folders (`export_hf.py`,
and `load_checkpoint()` when handed a folder instead of a `.pt`). The only
other package is **pyarrow**, and only three scripts import it —
`convert_parquet.py`, `convert_translate.py` and `eval_code.py` — to read
Hugging Face parquet shards; the fetchers themselves are plain `urllib`.
`app.py`, the optional Gradio demo, additionally needs `gradio` and
`huggingface_hub`. New here? `docs/TUTORIAL.md` is the guided path. Developed on torch
2.14.0+cpu / Python 3.14.5 / Windows, later on torch 2.11.0+cu128 / Python
3.14.7 with an RTX 5070 Ti (RESULTS.md §9 onward).
`torch>=2.1` is the floor — both attention paths pass `scale=` to
`F.scaled_dot_product_attention`, which older versions don't accept;
`--moe-impl grouped` additionally needs `torch.nn.functional.grouped_mm`
(present in 2.11) and falls under an assertion in `AnuLMConfig` otherwise.

For a CPU-only install (much smaller than the default CUDA wheel):

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

## Layout

The code is one package, `anulm/`. Every program in it runs from the
repository root as `python -m anulm COMMAND`; `python -m anulm` lists the
commands, and `python -m anulm.training.train` is the same program as
`python -m anulm train`. `pip install -e .` adds an `anulm` command that
does the same from anywhere.

```
.
├── anulm/
│   ├── model.py          architecture. Config, RMSNorm, SwiGLU, Rotary(+YaRN),
│   │                     GQAttention, MLAttention, Router, MoE, Block, AnuLM
│   ├── bpe.py            byte-level BPE, DOC_SEP / EOS, the .bin encoder (train / encode / stats)
│   ├── hf_tok.py         a Hugging Face tokenizer.json behind the same members as bpe.BPE
│   ├── muon.py           Muon optimizer (+ AdamW companion)
│   ├── lora.py           low-rank adapters: apply/merge/save, and a merge CLI
│   ├── bigram_teacher.py the corpus's own bigram statistics as soft targets, faded out
│   ├── bidirectional.py  the decoder as a bidirectional encoder (the LLM2Vec recipe)
│   ├── quickstart.py     environment check, then --train a small model or --demo a released one
│   ├── __main__.py       python -m anulm COMMAND
│   │
│   ├── training/         train.py: the training loop, WindowSampler, LR schedule, checkpointing,
│   │                     --resume / --stop-at;  finetune.py: SFT on any jsonl set, loss on answer
│   │                     tokens only; resumable; writes qa_templates
│   ├── inference/        sample.py, sample_many.py: generation (KV cache, EOS, grouped dispatch);
│   │                     ask.py: questions the way serve.py asks them;  serve.py: a stdlib HTTP
│   │                     server + web/index.html;  api.py: OpenAI-compatible endpoints;
│   │                     rag.py: retrieval;  voice.py: speech in and out;  langchain_anulm.py
│   ├── tasks/            decide.py: typed, calibrated decisions (intents);  rc_pointer.py: the
│   │                     extractive reader rag.py points with
│   ├── data/             fetch_*.py: Wikimedia, C4, codeparrot, any HF dataset, speech;
│   │                     convert_parquet.py, convert_translate.py, mix_corpus.py, filter_prose.py;
│   │                     make_qa*.py, make_chat.py, make_rc.py: supervised sets;
│   │                     make_bench.py, make_golden.py: the shared held-out file and the golden set
│   ├── eval/             eval_bench.py (bits/byte), eval_context.py (long context / YaRN),
│   │                     eval_golden.py, eval_qa.py, eval_translate.py (chrF), eval_code.py
│   │                     (pass@1, in a subprocess), eval_speech.py
│   ├── hf/               export_hf.py: a .pt -> a Hub-ready folder;  configuration_anulm.py and
│   │                     modeling_anulm.py: the trust_remote_code pair;  hf_tokenizer.py: bpe.py's
│   │                     JSON -> tokenizer.json, verified exact;  hf_upgrade.py: adds all of it
│   │                     to an export made before it existed
│   ├── multimodal/       audio_codec.py, speech_vocab.py, speech_data.py, speech_encoder.py,
│   │                     vision_encoder.py; omni_train.py: one decoder for every task;
│   │                     omni.py: loading it; omni_data.py, omni_images.py, omni_eval.py
│   └── apps/             Gradio demos: app.py, app_omni.py, app_voice.py, app_vision.py,
│                         app_benchmark.py
├── tests/test_model.py   the test suite, plain asserts, no pytest: python -m tests.test_model
├── tools/                maintenance for the machine the runs were made on: update_tasks.py,
│                         eval_noise.py (RESULTS.md §25), backfill_curve.py, sweep.py, samples.py,
│                         screenshot.py, qwen.py; windows/: the scheduled-task launchers
├── model.py              the one alias at the root: old checkpoints pickle model.AnuLMConfig
├── experiments/          one script per RESULTS.md section from §9 on; see its README
├── examples/             the API demo and demo_colab.ipynb
├── course/               the lab book that builds the model from stubs
├── docs/                 this file, ARCHITECTURE.md, RESULTS.md, the model cards, TUTORIAL.md,
│                         DATASETS.md, PUBLISHING.md, TASKS.md (the operations log), TODO.md,
│                         curves/ (val-loss curves of the recorded runs)
├── golden/               the golden set itself; committed, unlike data/
├── sarvam/               Sarvam's released modelling code + the five annotated walkthroughs
├── space/                README front matter + requirements for a Gradio Space running app.py
├── web/index.html        the page serve.py serves
├── .github/              CI (tests on Ubuntu and Windows, CPU wheel) and the issue forms
├── data/                 gitignored except the tokenizers; regenerate, never commit
└── runs/, checkpoints/, release/   gitignored outputs; nothing is written to the root
```

`python -m anulm model` is a standalone smoke test: builds both presets, runs forward
and backward, prints parameter counts and loss against `ln(V)`.

## Testing

```bash
python -m tests.test_model
```

The tests target **what training would not catch**. A leaky causal mask still
converges — it just cheats. Broken RoPE still converges — it just loses
long-range structure. "Loss went down" proves almost nothing.

The ones that earn their keep:

| test | catches |
| --- | --- |
| `causal_mask_does_not_leak_future` | future leakage; asserts `∂loss[t]/∂embedding[t+1] == 0.0` exactly |
| `rope_is_relative` | `q·k` depending on absolute rather than relative position |
| `moe_matches_naive_per_token_loop` | dispatch/unsort index errors |
| `gradient_accumulation_equals_large_batch` | `--grad-accum` silently differing from a big batch |
| `dense_and_sparse_moe_agree` | the two `moe_impl` paths diverging |
| `checkpoint_roundtrip_is_exact` | save/load drift |

Add tests to `test_model.py` by decorating with `@test` — the runner collects
them automatically. Use the `tiny()` helper for a fast config.

`.github/workflows/tests.yml` runs all of it on Ubuntu and Windows for every
push and pull request, from the CPU wheel, plus a 50-step training run, a
sample, a checkpoint reload and a load of all eight shipped tokenizers — the
pipeline breakages a unit test cannot see. The first run of it (2026-09-18)
was the first time this project had executed on Linux at all; everything
passed unchanged, and `docs/TUTORIAL.md` §13 records how closely the
platforms agree.

**One flake seen so far**, worth knowing before you chase a red Windows
job: the suite exited 1 on `windows-latest` after 17 seconds having printed
*nothing*, and the identical commit passed on an immediate re-run. Empty
output is the diagnosis problem, not the bug — CI's stdout is a pipe, python
block-buffers it, and a process that dies without flushing loses its whole
log. Every python step in the workflow now runs under `-u` so the next one
says where it died. If you hit a red job with no output, re-run it once; if
it reproduces, the `-u` log will show the last test that started.

**Two tests assert current *limitations* on purpose.**
`sparse_moe_breaks_the_graph_as_documented` fails if a future torch stops
breaking on `nonzero`, and `param_count_matches_analysis` fails if preset shapes
drift. Both are meant to fail loudly so the docs get updated rather than
silently rotting.

## Training

```bash
python -m anulm train --preset 30b --steps 2000
python -m anulm train --preset 105b --data data/hindi.txt --block-size 512 \
                --batch-size 8 --steps 2000
```

Useful flags: `--grad-accum`, `--grad-ckpt` (mandatory for `350m` on 8 GB),
`--moe-impl {sparse,dense,grouped}` (grouped on a GPU),
`--seq-balance-alpha` / `--router-z-alpha`, `--sliding-window` /
`--max-window-layers`, `--cfg key=value ...` (override any config field on
top of the preset — how the ablations run, and how the §12 combo
configuration is expressed; see `experiments/run_combo.sh`),
`--sample-with-replacement` (the old sampler, for A/B), `--yarn`, `--compile`,
`--resume` and `--rewarmup` (extend a finished run), `--stop-at N` (end this
invocation at step N after an eval and a save, so a multi-day run proceeds
in phases between which the machine may sleep), `--optimizer
{adamw,muon}` with `--muon-lr` (not comparable to
`--lr`), `--lr` / `--min-lr` / `--warmup` / `--weight-decay` / `--grad-clip`,
`--eval-every` / `--log-every`, `--seed`, `--out`. The log prints the LM loss
and the regulariser `aux` separately, plus ms/step and the epoch position.

**Reading the val loss.** It is an estimate, and the log now prints its
standard error beside it (`val loss 1.8597 +/- 0.0058`) so you can see
whether a difference you are about to act on is larger than the noise. The
default draws `--eval-iters` random batches, which is what every number in
`docs/RESULTS.md` was measured with and is therefore left alone;
`--eval-windows N` instead spreads N windows evenly over the split, which
lands 4-11x closer to the true mean for the same N (§26) and does not change
when `--batch-size` does. Use it on new runs. The method is recorded in each
checkpoint as `eval_spec`, and a `--resume` that changes it warns, because
two halves of a curve measured differently are not one curve.

**Checkpoints.** Two files are written:

- `<out>` — best-val weights. Sample from this.
- `<out>.last` — newest step **plus optimizer state**. This is what `--resume`
  reads. AdamW's moments matter as much as the weights; resuming from weights
  alone puts a visible dent in the loss curve.

Both are written at every eval, so `--eval-every` sets your worst-case loss on
an interruption. A long run that gets killed:

```bash
python -m anulm train ... --steps 6000 --out ckpt.pt --resume
```

**Extending a finished run** is the same flag with a larger `--steps`, plus
`--rewarmup N`: the cosine over the new total lands mid-schedule, several
times the lr the weights had annealed to, and the ramp brings it back over
N steps instead of jumping. Expect held-out loss to get *worse* for the
first few thousand steps and to cross back below the old best around the
midpoint of the extension — RESULTS.md §21 has the curve. Copy the `.pt`
and `.pt.last` to a new name first so the original stays put.

**Budget.** On this CPU, `nano_30b` runs ~2200–3000 tok/s and `nano_105b`
~1300–1400. A 2000-step run at batch 8 × block 512 is roughly an hour. Plan
long runs with `--resume` in mind. On the RTX 5070 Ti the combo preset runs
~10k tok/s: 36k steps at batch 8 × 512 is four hours, and the coder plan's
700k steps is three and a half days, which is what `--stop-at` is for.

**LoRA** (`lora.py`, `finetune.py --lora`) freezes the checkpoint and trains
a rank-r correction on the attention projections: `W + (alpha/r)·B·A`, B
initialised to zero so step 0 is the base model to float noise. 1.5M
trainable of 397.7M at r=16, a 6 MB adapter instead of a 1.6 GB checkpoint,
and no `--grad-ckpt` needed on 8 GB. `python -m anulm lora merge <adapter> <out>`
folds it back into an ordinary checkpoint.

Two things worth knowing before trusting an adapter. The experts are **not**
adapted by default, and §29 measures why rather than asserting it: 1,411
adapters buy +0.049 of held-out loss over attention-only for five times the
wall clock, because each expert sees only the tokens the router sends it and
because adapters force `--moe-impl sparse` (the grouped path stacks expert
weights directly, and `apply_lora` now refuses that combination with an
explanation instead of an AttributeError from inside the kernel). Raising the
rank on attention is cheaper and better. §29 also finds that a **full
fine-tune beats every LoRA setting on both quality and time at this scale**,
so LoRA here is for artefact size and memory, not for results.
And the aux-loss-free balancer's `expert_bias` is a *buffer*: it has no
gradient, LoRA does not freeze it, and it keeps moving every step. Merging
adapters into a pristine base would therefore restore the base's routing and
quietly change which experts a token reaches, so the adapter carries the bias
and `merge` restores it. A test holds the merged model to the trained one.

**Fine-tuning** (`finetune.py`) reuses all of the above — same optimizer
split, cosine, bf16, grouped dispatch, `--resume` / `--stop-at` — on a
jsonl of `{question, answer, lang}` pairs, with the loss masked to the
answer tokens. `lang` picks the template (`make_qa.PROMPTS`: `hi`, `en`,
`py`, `en-hi`, `hi-en`), the packed rows are cached beside the jsonl
(`<file>.<tokenizer>.b512.pt`, built by `--workers` processes), and the best
checkpoint carries `qa_templates`, which `serve.py`, `eval_qa.py` and
`eval_translate.py` read to know how to ask. Two million translation pairs
pack in two minutes and train through in ~7 hours.

## Extending

**A new preset** — add a `@staticmethod` to `AnuLMConfig` beside
`nano_30b` / `nano_105b` / `nano_350m`, then add it to `train.py`'s `--preset`
choices. For a one-off shape, `--cfg key=value` on an existing preset is
enough and is what every ablation and the §12 combo configuration use; a
preset is worth adding once a configuration is reused across runs.
`__post_init__` validates the combination; in particular it rejects a
`num_experts_per_tok` larger than `topk_group` groups can supply, which is a
footgun the real Sarvam code leaves unguarded. `param_count_matches_analysis`
pins the `30b` and `105b` shapes to their analytical counts; add a line for a
new preset so its shape cannot drift silently either.

**A new attention type** — implement
`forward(x, cos, sin, cache=None, start=0, window=None) -> (B, S, D)` and wire
it in `Block.__init__`. `cache` is a per-layer dict you own (append K/V for
the new tokens, attend to all of it), `start` is how many tokens it already
holds, `window` is the sliding window or None; `_attn_mask` / `_sdpa` build the
right mask for every combination. Look at `GQAttention` for the simple case.
The tests `incremental_forward_matches_full_forward` and
`cached_generation_matches_uncached` will tell you whether the cache is right.

**The `transformers` path.** `export_hf.py` writes `configuration_anulm.py`,
`modeling_anulm.py` and `model.py` into every export, plus `auto_map` in
`config.json`, so `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`
works with nothing cloned. The wrapper does not reimplement the architecture:
it builds the real modules from `model.py` and adopts them under their own
names, so the parameter names match `model.safetensors` exactly and a
mismatch would be a missing key rather than a silent half-load. Two things it
has to get right, both learned the hard way and both now tested:

- **The rotary tables are not in the checkpoint.** They are computed, so
  `model.py` registers them non-persistently. transformers builds the model on
  the meta device and materialises whatever the checkpoint does not cover,
  which for those buffers means uninitialised memory. Every weight loaded,
  nothing warned, and the logits were wrong by 5.4. `AnuLMForCausalLM`
  rebuilds them after every load.
- **float32 is not optional.** The exports store big tensors in bfloat16 and
  small ones — the router's `expert_bias`, the norms — in float32, and
  transformers otherwise infers bfloat16 for everything. Rounding that bias
  changes which experts a token is routed to, and the output collapses into
  repeated tokens rather than degrading gracefully. The config declares
  float32 and the wrapper warns if you override it.

**A different tokenizer** — the byte tokenizer (`vocab_size=259`) and the
from-scratch BPE (`bpe.py`, `bpe.py train / encode`) both exist. A tokenizer
file is `{"type": "anulm-bpe", "version": 1, "merges": [...]}`; `BPE.load`
also accepts `"nanosarvam-bpe"`, the id written before the 2026-09-18
rename, because every tokenizer trained until then carries it — including
the four uploaded beside the released weights. `bpe_files_load_under_both_format_ids`
pins that, and the committed tokenizers under `data/` were migrated to the
new id in place, merges byte-for-byte unchanged. `train.py`
takes either a text file or a `.bin` from `bpe.py encode`, which carries the
vocab size, the EOS id and the bytes-per-token needed to report bits/byte.
Anything else needs to produce the same `.bin` dict. Read the fertility
numbers in [RESULTS.md §7](RESULTS.md) before assuming bytes are fine for
Indic text.

**A corpus** — one document per article, `DOC_SEP` (three newlines) between
them, title on the first line; `fetch_hindi.py`, `fetch_web.py`,
`fetch_code.py` and `convert_parquet.py` all write that shape,
`mix_corpus.py` interleaves several by MB budget without loading them, and
`filter_prose.py` cleans it. Python source needs one extra rule, which
`fetch_code.py` applies: runs of three or more newlines collapse to two,
because PEP 8 puts exactly `DOC_SEP` between top-level definitions. `make_bench.py` must be re-run whenever a new
corpus is added to `CORPORA`, or the benchmark silently contains training text.
The golden set is built *from* the benchmark files, so it inherits that
guarantee; re-run `make_golden.py` after `make_bench.py`, and note that any
change to its filters or seed produces a different set, so scores across
versions of the file are not comparable.

**An accuracy metric** — `eval_golden.py` is the template: items live in a
JSONL file with `prefix`, `answer`, `choices` and `answer_idx`, and the scorer
handles BPE and byte checkpoints through one `Codec`. The three task scorers
that followed each own their format: `eval_qa.py` (question-answer pairs,
per language), `eval_translate.py` (FLORES pairs, chrF) and `eval_code.py`
(HumanEval / MBPP, which executes what the model wrote). That last one is
the only place this project runs unreviewed model output, so it is worth
knowing what holds it: a fresh directory per problem, deleted afterwards,
so nothing one program writes can shadow an import for the next; `-I -B`
with no inherited environment and stdin closed; output to a size-capped
file rather than a pipe; the whole process *tree* killed when the timeout
fires; and on Linux and macOS `setrlimit` caps on address space, CPU time,
file size and process count. **Windows has no rlimits**, so memory and
process count are bounded only by the timeout there. Note what is
deliberately absent: `-S`, which would drop site-packages and fail any
solution importing numpy, changing the pass@1 being measured. None of this
is a boundary against code that is trying to escape — point it only at
checkpoints you trained.

## GPU setup

The default `pip install torch` on this project's original setup was the
**CPU-only wheel**, which silently reports `torch.cuda.is_available() == False`
even on a machine with a working NVIDIA GPU. Check `nvidia-smi` first, then:

```bash
pip install --index-url https://download.pytorch.org/whl/cu130 --force-reinstall torch
```

Verified working: RTX 5050 Laptop (8 GB, Blackwell sm_120), driver 592.82,
torch 2.14.0+cu130, bf16 at ~21 TFLOP/s; later an RTX 5070 Ti (16 GB) on
torch 2.11.0+cu128, where the 350M presets run at 10–12k tok/s with
`--moe-impl grouped` and `--grad-ckpt` is optional (the 16 GB card holds
the un-checkpointed 350M, but other GPU tenants — an Ollama server was
running alongside some of the §9–13 runs — make `--grad-ckpt` the safe
default there too). On CUDA, `train.py` automatically uses bf16 autocast and
fused AdamW.

Traps specific to small-VRAM Windows machines:

- **WDDM spillover masks OOM.** Exceeding VRAM on Windows does not raise
  `OutOfMemoryError` — the driver spills to system RAM and the run silently gets
  ~12x slower (measured: 443M params peaked 9.74 GB on an 8 GB card and crawled
  at 130 tok/s). If tok/s is inexplicably terrible, check
  `torch.cuda.max_memory_allocated()` against actual VRAM.
- **`--grad-ckpt` is mandatory for `nano_350m`.** Without it the model peaks at
  15.75 GB; with it, 6.34 GB and 4955 tok/s. The ~30% recompute cost is far
  cheaper than spilling.
- **Optimizer choice is a memory knob.** AdamW costs 16 bytes/param
  (fp32 weight+grad+2 moments), Muon 12. But see the Muon measurement in
  [RESULTS.md](RESULTS.md) before assuming Muon is the answer — on this MoE it
  costs 2.6x per step.

## Known gaps

Stated plainly so nobody rediscovers them:

- **`torch.compile`'s Inductor backend is untested.** It generates C++ and needs
  a host compiler (MSVC `cl` on Windows); the dev machine had none. Compile
  tests use the `aot_eager` backend, which exercises Dynamo tracing and
  AOTAutograd — where model-specific risk lives — but not codegen. `train.py
  --compile` probes with a throwaway forward and falls back to eager with a
  one-line message rather than dying mid-run.
- **`--compile` with `--moe-impl sparse` buys little** (8 graph breaks/forward).
  Use `grouped` (one graph, and 1.48× faster on a GPU anyway) or `dense`; see
  [RESULTS.md §4](RESULTS.md).
- **Decoding is launch-bound.** There is a KV cache now, and it is exact, but
  a 386M MoE at batch 1 issues ~1,000 small kernels per token; on this machine
  that is ~33 tok/s with `--moe-impl grouped` and there is no C++ compiler for
  Inductor / CUDA graphs to fuse them. That, not the model, is the ceiling.
- **No distributed training.** No expert parallelism, unlike the real 105B.
- **The Hindi corpus has minor wikitext leakage** (~6 `{{` per MB, some
  `ref /ref` where HTML-escaped tags were entity-decoded after tag-stripping).
  Roughly 0.01% of text. `strip_wikitext` in `fetch_hindi.py` is a scrubber, not
  a parser.

## Gotchas

- **Windows consoles are cp1252** and cannot encode Devanagari. `sample.py`
  reconfigures stdout to UTF-8; if you write a new script that prints Indic
  text, do the same or set `PYTHONIOENCODING=utf-8`.
- **Piping training output through `grep` hides it.** grep block-buffers when
  not attached to a terminal, so a backgrounded `train.py | grep ...` produces an
  empty file until the run ends. Use `python -u` and no pipe.
- **Expert load imbalance rises before it falls.** Expected — see
  [RESULTS.md §5](RESULTS.md). Don't tune `bias_update_rate` on the first 100
  steps.
- **`expert_bias` is a buffer, not a parameter**, deliberately: it must not be
  touched by AdamW or weight decay. A test enforces this.
- **Byte-level models emit invalid UTF-8 mid-sequence**, especially early in
  training and always at a truncated tail. Decode with `errors="replace"`.

## Reading the architecture study

`sarvam/` at the repository root holds Sarvam's released code plus annotated
walkthroughs. Read [ARCHITECTURE.md](ARCHITECTURE.md) first for the summary,
then the annotated files beside the originals. Each AnuLM module names
its counterpart:

| AnuLM | annotated in `sarvam/` |
| --- | --- |
| `GQAttention` | `gqa_forward_annotated.py` |
| `MLAttention` | `mla_forward_annotated.py` |
| `Router`, `MoE` | `moe_routing_annotated.py` |
| `Rotary` + YaRN | `yarn_rotary_annotated.py` |
| `Block`, `AnuLM` | `decoder_and_model_annotated.py` |
