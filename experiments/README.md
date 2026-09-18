# Experiment scripts

One script per section of `docs/RESULTS.md` from §9 on. Each runs the training
command(s), then `eval_bench.py` on the shared benchmarks and a few samples.
They `cd` to the repository root themselves, so run them from anywhere:

```bash
bash experiments/run_ab.sh        # §9   old recipe vs v4 recipe, matched compute
bash experiments/run_ablate.sh    # §10  shared expert / fine-grained / top-k / window
bash experiments/run_full.sh      # §11  the whole dump at fixed compute
bash experiments/run_combo.sh     # §12  the ablation winners together
bash experiments/run_prose.sh     # §13-15  prose filter, Wikisource bench, window context
bash experiments/run_bias.sh      # §16  bias_update_rate for 24 experts
bash experiments/build_mixed.sh   # §17  Wikipedia + Wikisource corpus, tokenizer, .bin
bash experiments/run_mixed.sh     # §17  the mixed run, 6k steps
bash experiments/run_long.sh      # §18  the mixed run, 18k steps
```

§19 (the golden set) and §20 (question answering) have no scripts: each is
a few commands, run from the repository root:

```bash
python make_golden.py                                   # deterministic, --seed 7
python eval_golden.py ckpt_hindi_mixed18k.pt ckpt_hindi_full.pt \
                      ckpt_hindi_v4.pt ckpt_hindi_350m.pt --device cuda

python make_qa.py                                       # 56k pairs + 1,129 held out
python finetune.py --ckpt ckpt_hindi_mixed18k.pt --out ckpt_hindi_qa.pt --epochs 2 --grad-ckpt
python eval_qa.py ckpt_hindi_mixed18k.pt ckpt_hindi_qa.pt --device cuda
```

§21 continues the §18 run to 36k steps: copy `ckpt_hindi_mixed18k.pt` and
its `.last` to `ckpt_hindi_mixed36k.pt[.last]`, then `run_long.sh`'s
command with `--steps 36000 --out ckpt_hindi_mixed36k.pt --resume
--rewarmup 500`; then the three §20 commands from the new checkpoint
(`--out ckpt_hindi_qa36k.pt`), plus `eval_bench.py` and `eval_golden.py`.

## Three languages, translation, code (§22 on)

```bash
bash experiments/build_multi.sh     # §22  Hindi + English + Python corpus, 32k tokenizer, .bin
bash experiments/run_multi.sh       # §22  ckpt_multi36k: the combo preset on it, 36k steps, 4 benchmarks
bash experiments/run_qa_multi.sh    # §23  ckpt_multi_qa: QA pairs in all three languages, scored per language

bash experiments/translate_build.sh        # §24  2M en-hi pairs both ways + FLORES-200 devtest
bash experiments/translate_train.sh 25000  # §24  fine-tune ckpt_multi36k to step 25k, save, exit
bash experiments/translate_train.sh        #      resume from ckpt_translate.pt.last, run to the end (61,672 steps)
bash experiments/translate_eval.sh         #      chrF on 300 FLORES sentences per direction, ~40 min
LIMIT= bash experiments/translate_eval.sh  #      all 1,012 sentences, ~2.5 h -- the number to quote

bash experiments/coder_fetch.sh            # docs/CODER_PLAN.md phase 1: ~15 GB of Python, English, exercises, tests
bash experiments/coder_build.sh            # phase 2: convert, mix (65/30/5), code32k tokenizer, encode (done)
                                           #   long runs belong to a scheduled task now -- see TASKS.md
bash experiments/coder_train.sh 250000     # phases 3-6: pretrain to step N and exit; rerun to continue; no arg = to 700k
                                           #   (step 100k done 2026-09-11, val 3.436; RESULTS.md 25)
bash experiments/coder_sft.sh              # phase 7: instruction-tune, then pass@1 on HumanEval and MBPP
```

The three long-running scripts (`translate_train.sh`, `coder_train.sh`,
and `finetune.py` / `train.py` underneath them) take an optional stop step.
Each call resumes from `<checkpoint>.last` if it exists, trains to the step
given, evaluates, saves, and exits, so the machine can sleep between calls;
a call that dies mid-way loses at most one `--eval-every` interval. Where
the translation run stands: `python -c "import torch; print(torch.load('ckpt_translate.pt.last', map_location='cpu', mmap=True, weights_only=False)['step'])"`.

Prerequisites, in order: `fetch_hindi.py` for `data/hindi_v4.txt` (30% of the
dump, `--keep-prob 0.3`) and `data/hindi_full.txt` (`--keep-prob 1.0`), the
Wikisource dump via `--dump`, `make_bench.py` for the Wikipedia benchmark,
`bpe.py train` / `encode` for each corpus, `filter_prose.py` for §13,
`build_mixed.sh` for `data/mixed.txt` (which `build_multi.sh` and
`coder_build.sh` both take as their Hindi). The translation and coder data
come from public Hugging Face repos through `fetch_hf.py` (Samanantar, the
IIT Bombay corpus, FLORES-200, fineweb-edu, code_exercises, OpenCodeInstruct,
glaive, HumanEval, MBPP) into `data/raw/`; reading their parquet shards needs
`pyarrow`. The scripts assume the data files those produce, under `data/`.

Two Windows notes. They export `/usr/bin` onto PATH because a Git Bash
launched detached (PowerShell `Start-Process`) has no MSYS tools; and the
`| grep -v` filters block-buffer, so a run's log fills in 4 KB chunks and a
run killed mid-phase leaves the log short of what the checkpoint holds --
read the checkpoint's `step` if you need progress mid-run.
