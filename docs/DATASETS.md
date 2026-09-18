# Datasets: every source, where to get it, what it is licensed for

No training data is redistributed in this repository. Every corpus is
rebuilt from a public source by the scripts named below, so anyone can
reproduce each checkpoint from scratch. `data/` is gitignored except for
the trained tokenizers (`data/*.json`), which are small and needed to run
the checkpoints.

Sizes are what the scripts actually download and keep; most sources are
streamed and cut at a byte budget, so you never fetch a whole dump.
Licences are as declared on the source at the time of writing (2026-09);
verify on the source before any use beyond research.

## Pretraining corpora

| corpus | link | licence | fetched by | kept on disk | used by |
| --- | --- | --- | --- | --- | --- |
| Hindi Wikipedia (articles dump) | https://dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-pages-articles.xml.bz2 | CC BY-SA 4.0 | `fetch_hindi.py --mb N` (streams the bz2, stops at N MB of clean prose) | 8 MB (first experiments) to ~1 GB | every Hindi model, §1–21; base §22; coder (Hindi slice) |
| Hindi Wikisource (literature) | https://dumps.wikimedia.org/hiwikisource/latest/ | CC BY-SA 4.0 / public domain texts | `fetch_hindi.py --dump <wikisource url>` via `experiments/build_mixed.sh` | ~0.3 GB | `ckpt_hindi_mixed*`, base, coder |
| English Wikipedia (articles dump) | https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 | CC BY-SA 4.0 | `fetch_hindi.py --dump <enwiki url> --min-devanagari 0` via `experiments/build_multi.sh` | 500 MB | base `ckpt_multi36k` |
| C4, English | https://huggingface.co/datasets/allenai/c4 (`en/` shards) | ODC-BY 1.0 (over Common Crawl) | `fetch_web.py --mb 350` | 350 MB | base `ckpt_multi36k` |
| codeparrot-clean (GitHub Python) | https://huggingface.co/datasets/codeparrot/codeparrot-clean | source files' own licences, mixed (includes GPL/AGPL) | `fetch_code.py --mb N --shards …` (keeps files that look hand-written) | 250 MB (base) / 7.3 GB (coder, shards 1–10) | base; coder |
| fineweb-edu, `sample/10BT` | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu | ODC-BY 1.0 | `fetch_hf.py HuggingFaceFW/fineweb-edu --prefix sample/10BT/000_ sample/10BT/001_` | 4.3 GB raw → 4.1 GB text | coder |
| code_exercises (phi-1-style Python exercises) | https://huggingface.co/datasets/jinaai/code_exercises | **CC BY-NC-SA 4.0**; synthetic, generated with ChatGPT 3.5 | `fetch_hf.py jinaai/code_exercises --prefix data/` | 0.5 GB raw → 1.0 GB text + 1.1 GB SFT pairs | coder pretraining (7%) and instruction tuning (93% of pairs) |
| tinyshakespeare | https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt | public domain | `train.py` downloads it when no `--data` is given | 1 MB | the CPU smoke runs in `README.md` "Results" |

## Instruction-tuning and translation pairs

| corpus | link | licence | fetched by | used by |
| --- | --- | --- | --- | --- |
| Hindi / English question pairs | built from the Wikipedia corpora above by `make_qa.py`, `make_qa_en.py` | CC BY-SA 4.0 (derived) | `experiments/run_qa_multi.sh` | `ckpt_multi_qa` |
| Python docstring pairs | built from codeparrot-clean by `make_qa_py.py` | as codeparrot-clean | `experiments/run_qa_multi.sh`, `coder_sft.sh` | `ckpt_multi_qa`, `ckpt_coder_sft` |
| Samanantar, Hindi | https://huggingface.co/datasets/ai4bharat/samanantar | **CC BY-NC 4.0** | `fetch_hf.py ai4bharat/samanantar --prefix hi/` | `ckpt_translate` |
| IIT Bombay English–Hindi corpus | https://huggingface.co/datasets/cfilt/iitb-english-hindi | **research use** (CC BY-NC 4.0 on the Hub card) | `fetch_hf.py cfilt/iitb-english-hindi` | `ckpt_translate` |
| OpenCodeInstruct | https://huggingface.co/datasets/nvidia/OpenCodeInstruct | CC BY 4.0 | `fetch_hf.py nvidia/OpenCodeInstruct --prefix data/train-00000-` | `ckpt_coder_sft` (43,728 Python pairs with test score ≥ 0.9) |
| glaive-code-assistant | https://huggingface.co/datasets/glaiveai/glaive-code-assistant | Apache 2.0 | `fetch_hf.py glaiveai/glaive-code-assistant` | `ckpt_coder_sft` (40,000 Python pairs) |

## Evaluation sets (never trained on)

| set | link | licence | fetched by | used by |
| --- | --- | --- | --- | --- |
| FLORES-200 devtest | https://github.com/facebookresearch/flores/tree/main/flores200 (also https://huggingface.co/datasets/openlanguagedata/flores_plus) | CC BY-SA 4.0 | `experiments/translate_build.sh` (curl) | `eval_translate.py`, chrF |
| HumanEval | https://huggingface.co/datasets/openai/openai_humaneval | MIT | `fetch_hf.py openai/openai_humaneval --prefix openai_humaneval/` | `eval_code.py --bench humaneval` |
| MBPP, sanitized split | https://huggingface.co/datasets/google-research-datasets/mbpp | CC BY 4.0 | `fetch_hf.py google-research-datasets/mbpp --prefix sanitized/` | `eval_code.py --bench mbpp` |
| Hindi golden cloze set (600 items) | `golden/golden_hindi.jsonl`, in this repository | CC BY-SA 4.0 (derived from the Hindi dumps) | `make_golden.py` | `eval_golden.py` |
| held-out benchmark files `data/bench_*.txt` | built by the `build_*.sh` scripts from the corpora above (last 2% of documents) | as their source | the build scripts | bits/byte in every results section |

## Which checkpoint saw what, and the licence that follows

| checkpoint | pretraining data | tuning data | released as |
| --- | --- | --- | --- |
| `ckpt_multi36k` (AnuLM-Base-400M) | Hindi Wikipedia + Wikisource, English Wikipedia + C4, codeparrot-clean | — | CC BY-SA 4.0 |
| `ckpt_multi_qa` (AnuLM-Hindi-QA-400M) | as the base | Wikipedia question pairs, docstring pairs | CC BY-SA 4.0 |
| `ckpt_translate` (AnuLM-Translate-400M) | as the base | Samanantar + IIT Bombay | CC BY-NC 4.0, research only |
| `ckpt_coder`, `ckpt_coder_sft` (AnuLM-Coder-400M) | codeparrot-clean, code_exercises, fineweb-edu, Hindi mix | code_exercises, OpenCodeInstruct, glaive, docstring pairs | CC BY-NC-SA 4.0 |

`NOTICE` §2 is the legal version of this table. Two points deserve
repeating: a model trained on GitHub code can reproduce licensed snippets
verbatim, and the coder learned mostly from ChatGPT-generated exercises,
so anything derived from it inherits non-commercial, share-alike terms.

## Disk and time to rebuild everything

| what | download | on disk after build | wall clock |
| --- | --- | --- | --- |
| Hindi ladder (§1–21) | < 1 GB | ~3 GB | minutes per corpus |
| three-language base corpus | ~1.5 GB | ~3 GB | 20 min |
| translation pairs + FLORES | ~3 GB | ~4 GB | 15 min |
| coder corpus | ~15 GB | ~25 GB (13.7 GB text + 5.6 GB tokens + SFT pairs) | 1 h download + 2 h CPU build |

All fetchers skip files already present, so an interrupted download
resumes where it stopped.
