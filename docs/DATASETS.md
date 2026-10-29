# Datasets: every source, where to get it, what it is licensed for

No training data is redistributed in this repository. Every corpus is
rebuilt from a public source by the scripts named below, so anyone can
reproduce each checkpoint from scratch. `data/` is gitignored except for
the trained tokenizers (`data/*.json`), which are small and needed to run
the checkpoints.

Sizes are what the scripts actually download and keep; most sources are
streamed and cut at a byte budget, so you never fetch a whole dump.
Licences are as declared on the source at the time of writing (2026-09,
and 2026-10 for the sources added for base v2 and AnuLM-Omni-400M);
verify on the source before any use beyond research.

AnuLM-Omni-400M, the one jointly trained model of the second release, is
meant for commercial use, so it learns only from the sources below whose
terms allow that. `omni_data.py` (text and speech) and `omni_images.py`
(photos, captions, image questions) fetch every one of them from its
primary source; `python omni_data.py list` prints each with its licence.
The rows below say, for every source, whether that model uses it.

## Pretraining corpora

| corpus | link | licence | fetched by | kept on disk | used by |
| --- | --- | --- | --- | --- | --- |
| Hindi Wikipedia (articles dump) | https://dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-pages-articles.xml.bz2 | CC BY-SA 4.0 | `fetch_hindi.py --mb N` (streams the bz2, stops at N MB of clean prose) | 8 MB (first experiments) to ~1 GB | every Hindi model, §1–21; base §22; coder (Hindi slice) |
| Hindi Wikisource (literature) | https://dumps.wikimedia.org/hiwikisource/latest/ | CC BY-SA 4.0 / public domain texts | `fetch_hindi.py --dump <wikisource url>` via `experiments/build_mixed.sh` | ~0.3 GB | `ckpt_hindi_mixed*`, base, coder |
| English Wikipedia (articles dump) | https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 | CC BY-SA 4.0 | `fetch_hindi.py --dump <enwiki url> --min-devanagari 0` via `experiments/build_multi.sh` | 500 MB | base `ckpt_multi36k` |
| C4, English | https://huggingface.co/datasets/allenai/c4 (`en/` shards) | ODC-BY 1.0 (over Common Crawl) | `fetch_web.py --mb 350` | 350 MB | base `ckpt_multi36k` |
| codeparrot-clean (GitHub Python) | https://huggingface.co/datasets/codeparrot/codeparrot-clean | source files' own licences, mixed (includes GPL/AGPL) | `fetch_code.py --mb N --shards …` (keeps files that look hand-written) | 250 MB (base) / 7.3 GB (coder, shards 1–10) | base; coder; base v2 (`gh_python.clean`, decontaminated), and so AnuLM-Omni-400M |
| fineweb-edu, `sample/10BT` | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu | ODC-BY 1.0 | `fetch_hf.py HuggingFaceFW/fineweb-edu --prefix sample/10BT/000_ sample/10BT/001_` | 4.3 GB raw → 4.1 GB text | coder; base v2 (14 shards, 6.0B tokens) |
| fineweb-2, Hindi (`hin_Deva`) | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 | ODC-BY 1.0 (over Common Crawl) | `fetch_hf.py` (shards 000–003 of `data/hin_Deva/train/`), read by `experiments/build_v2.py` | 4.5B tokens | base v2 |
| Hindi Wikipedia, `20231101.hi` (and `20231101.simple` for retrieval) | https://huggingface.co/datasets/wikimedia/wikipedia | CC BY-SA 4.0 | `rag.py build` (163k Hindi + 242k Simple English articles) | 66M tokens of Hindi in base v2 | base v2 (all of it); the reader's retrieval index and hard negatives; AnuLM-Omni-400M's reading passages |
| OpenCodeInstruct, as text | https://huggingface.co/datasets/nvidia/OpenCodeInstruct | CC BY 4.0; model-generated | `experiments/textbook_build.sh` (`convert_parquet.py textbook`: Python only, all ten tests passed, decontaminated against MBPP/HumanEval) | `data/tb_opencode.txt` | base v2; the §31 textbook probe |
| opc-sft-stage1 (OpenCoder) | https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage1 | MIT on the card, but see below; model-generated | `experiments/textbook_build.sh` (as above) | `data/tb_opc.txt` | base v2; the §31 textbook probe |
| code_exercises (phi-1-style Python exercises) | https://huggingface.co/datasets/jinaai/code_exercises | **CC BY-NC-SA 4.0**; synthetic, generated with ChatGPT 3.5 | `fetch_hf.py jinaai/code_exercises --prefix data/` | 0.5 GB raw → 1.0 GB text + 1.1 GB SFT pairs | coder pretraining (7%) and instruction tuning (93% of pairs) |
| tinyshakespeare | https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt | public domain | `train.py` downloads it when no `--data` is given | 1 MB | the CPU smoke runs in `README.md` "Results" |

Base v2 (`experiments/build_v2.py`, recorded in `data/v2/meta.json`) is 11.3B
training tokens: 6.0B English (fineweb-edu), 4.5B Hindi web (fineweb-2), 66M
Hindi Wikipedia and 0.75B Python (codeparrot-clean, OpenCodeInstruct,
opc-sft-stage1, every source decontaminated against MBPP and HumanEval by
13-gram overlap). `code_exercises` is deliberately left out, so its
non-commercial terms do not reach the base or anything tuned from it.

Two open questions stay in base v2, and AnuLM-Omni-400M inherits both;
neither is settled. codeparrot-clean is GitHub code and includes GPL and AGPL
files. opc-sft-stage1's card says MIT, but it includes
`filtered_infinity_instruct`, derived from BAAI/Infinity-Instruct
(CC BY-SA 4.0), and `realuser_instruct`, drawn from ShareGPT and WildChat
(ChatGPT conversations).

## Speech and image corpora

Added for the speech and vision work (`docs/SPEECH.md`) and for
AnuLM-Omni-400M. Same rule as above: nothing is redistributed, every set is
re-fetched from its source.

| corpus | link | licence | fetched by | kept on disk | used by |
| --- | --- | --- | --- | --- | --- |
| LibriSpeech `dev-clean` | https://huggingface.co/datasets/openslr/librispeech_asr (`clean/validation`) | CC BY 4.0 | `speech_data.py encode --parquet …` after a `snapshot_download` | 326 MB parquet → 4.97 h, 1.48M SNAC tokens | the first TTS probe, `docs/SPEECH.md` |
| LibriSpeech `train-clean-100` | same repo, `clean/train.100` (14 shards); also https://www.openslr.org/12/ | CC BY 4.0 | as above; `omni_data.py fetch librispeech` (the tarball's flac, no SNAC encoding) | 6.0 GB parquet → 100.58 h, 29.80M SNAC tokens | the 20x-data TTS probe; English ASR, 3 epochs (`ckpt_asr_en_v3`, whose audio projector AnuLM-Omni-400M starts from); AnuLM-Omni-400M |
| IndicTTS Hindi | https://huggingface.co/datasets/SPRINGLab/IndicTTS-Hindi | **no licence stated** on the card, which defers to the IIT Madras Indic TTS licence; commercial terms unclear | `speech_data.py encode --parquet … --where gender=0` | 7.7 GB parquet → 11.84 h one speaker, 3.52M SNAC tokens | `ckpt_tts_hi.pt`; Hindi ASR v3–v5, mixed with FLEURS. **Not used by AnuLM-Omni-400M** (`INCLUDE_INDICTTS = False` in `omni_data.py`) |
| FLEURS, Hindi (`hi_in`) | https://huggingface.co/datasets/google/fleurs | CC BY 4.0 | `experiments/omni_v3.py`, `experiments/indic_v4.py`; `omni_data.py fetch fleurs_hi` | train split for training; dev + test (~650 clips) held out | Hindi ASR (`ckpt_asr_hi_v3` and later); AnuLM-Omni-400M; Hindi WER |
| Kathbath, Hindi | https://huggingface.co/datasets/ai4bharat/Kathbath | CC BY 4.0; **gated**, access granted on Hugging Face | `experiments/speech_v5.py`; `omni_data.py fetch kathbath_hi` (m4a in parquet → 16 kHz clips) | 32 train shards; `valid` held out | Hindi ASR v5; AnuLM-Omni-400M |
| MUCS 2021 Hindi–English | https://www.openslr.org/104/ | CC BY-SA 4.0 | `experiments/speech_v5.py`; `omni_data.py fetch mucs_hi` (Kaldi segments → utterance clips; code-switched, Hindi in Devanagari, English in Latin) | train; test held out | Hindi ASR v5; AnuLM-Omni-400M |
| Flickr8k | https://huggingface.co/datasets/jxie/flickr8k | research use; images are Flickr-sourced, see the dataset card | `vision_encoder.py prep --parquet …` | 828 MB parquet → 6,000 jpgs + 30,000 caption pairs + 1.81 GB cached tower features | `ckpt_caption_f8k.pt`; not used by AnuLM-Omni-400M |
| Flickr30k | https://huggingface.co/datasets/nlphuji/flickr30k | **non-commercial research and education only** (Flickr30k terms); images are Flickr-sourced | `vision_encoder.py prep` via `experiments/omni_v3.py` | images, captions and cached SigLIP features | captioner, 2 epochs (`ckpt_caption_v3`), and the VQA model through it. **Not used by AnuLM-Omni-400M**, nor is the vision projector learned on it |
| A-OKVQA | https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA | Apache 2.0 upstream; the images are COCO's, under their Flickr owners' terms, some non-commercial | `vision_encoder.py prep` via `experiments/omni_v3.py` | questions, choices and cached features | VQA, 3 epochs from the captioner (`ckpt_vqa_v3`). **Not used by AnuLM-Omni-400M** |
| Localized Narratives (Google), Open Images train annotations | https://google.github.io/localized-narratives/ | annotations CC BY 4.0 | `omni_images.py fetch` (spoken opener cut, first two sentences kept; one narrative per photo) | 10 annotation shards, read until 40,000 photos (`--ln-images`) | AnuLM-Omni-400M (captions) |
| Open Images V7 photos | https://storage.googleapis.com/openimages/web/index.html | **CC BY 2.0** only: photos Open Images lists under any other licence are dropped; every kept photo's author and URL go to `data/omni/ATTRIBUTION_open_images.csv` | `omni_images.py fetch` | ~40,000 jpgs in `data/omni_img/open_images/` | AnuLM-Omni-400M (captions, with Localized Narratives) |
| DOCCI (Google) | https://google.github.io/docci/ | CC BY 4.0, images and descriptions (all photos taken by one researcher) | `omni_images.py fetch` (first two sentences of each description) | the images tarball, unpacked to `data/omni_img/docci/` | AnuLM-Omni-400M (captions) |
| TextVQA | https://textvqa.org | CC BY 4.0; the photos are Open Images' | `omni_images.py fetch` (most common of ten answers as the target, all ten for scoring) | train + val images, unpacked to `data/omni_img/textvqa/` | AnuLM-Omni-400M (image questions) |
| VizWiz-VQA | https://vizwiz.org/tasks-and-datasets/vqa/ | CC BY 4.0; photos and questions from blind users | `omni_images.py fetch --vizwiz` (opt-in) | ~15 GB more | AnuLM-Omni-400M only with `--vizwiz`; **off by default** |

`omni_images.py fetch` at its defaults is ~26 GB; the training run adds a
~28 GB feature cache. Its outputs are `data/omni/caption_{train,test}.jsonl`
and `data/omni/vqa_{train,test}.jsonl`. The Hindi speech sets are joined into
`data/omni/hindi_train.jsonl` (Kathbath, MUCS 2021 and FLEURS train,
146,350 clips) by `omni_data.py`.

Two Hindi speech sets worth naming because they are the obvious next step and
are **not** openly downloadable: `ai4bharat/IndicVoices` (~1,600 h) and
`ai4bharat/Shrutilipi` (~6,400 h) both list their files publicly but return
403 on the data itself and need access granted on Hugging Face. That is why the
Hindi TTS run used IndicTTS, which is two orders of magnitude smaller.
Kathbath is gated the same way, and access to it was granted.

Note on the IndicTTS card: it states ~10.33 h total and 5.18 h for the female
speaker. Encoding that speaker alone yielded **11.84 h across 5,851 clips**, so
the card understates the duration and the full set is nearer 24 h.

Pretrained models used as components, not trained here: `openai/whisper-small`
(via faster-whisper, MIT), `hubertsiuzdak/snac_24khz` (MIT),
`rhasspy/piper-voices` (MIT code, per-voice licences on the card), and
`vit_base_patch16_siglip_224` from timm (Apache 2.0). Whisper also acts as the
judge for TTS word error rate, which `eval_speech.py` documents as a bias to
account for rather than a neutral measurement. The ASR models and
AnuLM-Omni-400M keep Whisper-small's encoder frozen underneath, and the
caption and VQA models and AnuLM-Omni-400M keep SigLIP's; the words come from
the AnuLM backbone. Piper speaks the demo app's replies and is in no AnuLM
weights; its `en_US-lessac` (research only) and `hi_IN-pratham` (CC BY-NC-SA)
voices are not for commercial use, `en_US-ljspeech` is public domain
(`NOTICE` §4). Two more components arrived with the per-task models of
2026-10: `answerdotai/ModernBERT-base` (Apache 2.0), fine-tuned as the Decide
comparison baseline, and `intfloat/multilingual-e5-small` (MIT), the reader's
dense retriever.

## Instruction-tuning and translation pairs

| corpus | link | licence | fetched by | used by |
| --- | --- | --- | --- | --- |
| Hindi / English question pairs | built from the Wikipedia corpora above by `make_qa.py`, `make_qa_en.py` | CC BY-SA 4.0 (derived) | `experiments/run_qa_multi.sh`; `omni_data.py fetch chat` | `ckpt_multi_qa`; 4,000 Hindi pairs in chat v3 and AnuLM-Omni-400M |
| databricks-dolly-15k | https://huggingface.co/datasets/databricks/databricks-dolly-15k | CC BY-SA 3.0 | read by `make_chat.py` → `data/chat_train.jsonl` (`omni_data.py fetch chat` rebuilds it) | `ckpt_chat_v3`; AnuLM-Omni-400M |
| OpenAssistant oasst2 | https://huggingface.co/datasets/OpenAssistant/oasst2 | Apache 2.0 | `make_chat.py` (first turn of each tree, its rank-0 reply) | `ckpt_chat_v3`; AnuLM-Omni-400M |
| Aya dataset, Hindi rows | https://huggingface.co/datasets/CohereForAI/aya_dataset | Apache 2.0 | `make_chat.py` (native Hindi, ~1.1k rows) | `ckpt_chat_v3`; AnuLM-Omni-400M |
| indic-instruct-data-v0.1, Hindi subsets | https://huggingface.co/datasets/ai4bharat/indic-instruct-data-v0.1 | per subset; `anudesh` holds Llama 2 outputs | `experiments/indic_v4.py` | `ckpt_chat_v4`, `ckpt_chat_v5` only, never released; **not used by AnuLM-Omni-400M** |
| Python docstring pairs | built from codeparrot-clean by `make_qa_py.py` | as codeparrot-clean | `experiments/run_qa_multi.sh`, `coder_sft.sh` | `ckpt_multi_qa`, `ckpt_coder_sft` |
| Samanantar, Hindi | https://huggingface.co/datasets/ai4bharat/samanantar | **CC BY-NC 4.0** | `fetch_hf.py ai4bharat/samanantar --prefix hi/` | `ckpt_translate` |
| IIT Bombay English–Hindi corpus | https://huggingface.co/datasets/cfilt/iitb-english-hindi | **research use** (CC BY-NC 4.0 on the Hub card) | `fetch_hf.py cfilt/iitb-english-hindi` | `ckpt_translate` |
| OpenCodeInstruct | https://huggingface.co/datasets/nvidia/OpenCodeInstruct | CC BY 4.0 | `fetch_hf.py nvidia/OpenCodeInstruct --prefix data/train-00000-` | `ckpt_coder_sft` (43,728 Python pairs with test score ≥ 0.9) |
| glaive-code-assistant | https://huggingface.co/datasets/glaiveai/glaive-code-assistant | Apache 2.0 | `fetch_hf.py glaiveai/glaive-code-assistant` | `ckpt_coder_sft` (40,000 Python pairs) |

## Classification (Decide) and reading comprehension

| corpus | link | licence | fetched by | used by |
| --- | --- | --- | --- | --- |
| BANKING77 (PolyAI) | https://huggingface.co/datasets/PolyAI/banking77 | CC BY 4.0 | `decide.py` (`data/banking77`; 10,003 train / 3,080 test); `omni_data.py fetch banking77` | `ckpt_decide_lora_F.pt`, `ckpt_decide_modernbert.pt`; AnuLM-Omni-400M, as generated label text (`data/omni/intent_*.jsonl`) |
| MASSIVE, `hi-IN` (Amazon Science) | https://huggingface.co/datasets/AmazonScience/massive | CC BY 4.0 | `decide.py --task massive-hi` (`data/decide_general/massive_hi`); `omni_data.py fetch massive_hi` | `ckpt_decide_lora_hi_F.pt`, `ckpt_decide_mb_hi.pt`; AnuLM-Omni-400M, as generated label text |
| synthetic Indian banking (UPI, KYC, Aadhaar, EMI; English, Hindi, Hinglish; 20 intents) | `experiments/make_india_banking.py`, in this repository | generated here; no third-party text. The templates were written with Claude; real bank and app names appear only as vocabulary | `experiments/make_india_banking.py` | `ckpt_decide_lora_india_F.pt`. **Not used by AnuLM-Omni-400M**: Anthropic's terms forbid using Claude's outputs to train models without authorization |
| SQuAD v1.1, train | https://huggingface.co/datasets/rajpurkar/squad | CC BY-SA 4.0 | `make_rc.py` (30k); `omni_data.py fetch read` | `ckpt_rc_pointer_v2`; AnuLM-Omni-400M |
| MLQA translate-train, Hindi | https://huggingface.co/datasets/facebook/mlqa (`mlqa-translate-train.hi`; SQuAD machine-translated) | CC BY-SA 3.0 | `make_rc.py` (30k) | `ckpt_rc_pointer_v2`; AnuLM-Omni-400M |
| Wikipedia hard negatives | the `wikimedia/wikipedia` 20231101 index above | CC BY-SA 4.0 | `make_rc.py` over `rag.py`'s index | `ckpt_rc_pointer_v2`; AnuLM-Omni-400M |

The Decide adapters are LoRA weights plus a classification head on
`ckpt_chat_v3.pt` and attach to that exact model only. The two ModernBERT
deciders are fine-tunes of `answerdotai/ModernBERT-base` (Apache 2.0), kept as
a comparison baseline; they are not trained from scratch here. None of these
is released; AnuLM-Omni-400M learns the BANKING77 and MASSIVE intents as text
instead.

## Evaluation sets (never trained on)

| set | link | licence | fetched by | used by |
| --- | --- | --- | --- | --- |
| FLORES-200 devtest | https://github.com/facebookresearch/flores/tree/main/flores200 (also https://huggingface.co/datasets/openlanguagedata/flores_plus) | CC BY-SA 4.0 | `experiments/translate_build.sh` (curl) | `eval_translate.py`, chrF |
| HumanEval | https://huggingface.co/datasets/openai/openai_humaneval | MIT | `fetch_hf.py openai/openai_humaneval --prefix openai_humaneval/` | `eval_code.py --bench humaneval` |
| MBPP, sanitized split | https://huggingface.co/datasets/google-research-datasets/mbpp | CC BY 4.0 | `fetch_hf.py google-research-datasets/mbpp --prefix sanitized/` | `eval_code.py --bench mbpp` |
| Hindi golden cloze set (600 items) | `golden/golden_hindi.jsonl`, in this repository | CC BY-SA 4.0 (derived from the Hindi dumps) | `make_golden.py` | `eval_golden.py` |
| held-out benchmark files `data/bench_*.txt` | built by the `build_*.sh` scripts from the corpora above (last 2% of documents) | as their source | the build scripts | bits/byte in every results section |
| SQuAD v1.1 dev; MLQA `hi.hi` test | the SQuAD and MLQA repos above | CC BY-SA 4.0; CC BY-SA 3.0 | `make_rc.py` (500 each) | `rc_pointer.py eval`; AnuLM-Omni-400M's reading F1 |
| XQuAD, Hindi | https://huggingface.co/datasets/google/xquad (`xquad.hi`) | CC BY-SA 4.0 | `make_rc.py` (500) | `rc_pointer.py eval` |
| IndicQA, Hindi | https://huggingface.co/datasets/ai4bharat/IndicQA (`indicqa.hi`) | see the dataset card | `make_rc.py` (500) | `rc_pointer.py eval` |
| FLEURS Hindi dev + test; Kathbath `valid`; MUCS 2021 test | the speech repos above | as above | `experiments/indic_v4.py`, `experiments/speech_v5.py`; `omni_data.py` | Hindi WER |
| caption and VQA test sets | DOCCI test and 500 Localized Narratives photos; 1,000 TextVQA val questions (and VizWiz val, if fetched) | as their sources above | `omni_images.py fetch` (`data/omni/caption_test.jsonl`, `vqa_test.jsonl`) | AnuLM-Omni-400M's caption loss and VQA accuracy |
| Monsoon en-IN, public test | https://huggingface.co/datasets/VoiceArena/MonsoonASR-Open-ASR-leaderboard-en-IN | CC BY 4.0 | `experiments/speech_v5.py` | English WER on Indian-accented speech. The parquet also carries speakers' personal fields; only the `audio`, `id` and `text` columns are read, the rest never |

## Which checkpoint saw what, and the licence that follows

Hugging Face repos are named only for checkpoints that are published. The
per-task checkpoints of 2026-10 were exported locally and checked but are not
released; their rows record provenance only.

| checkpoint | pretraining data | tuning data | released as |
| --- | --- | --- | --- |
| `ckpt_multi36k` (AnuLM-Base-400M) | Hindi Wikipedia + Wikisource, English Wikipedia + C4, codeparrot-clean | — | CC BY-SA 4.0 |
| `ckpt_ctx2k` (AnuLM-Base-2K-400M) | as the base, then continued at 2,048 on a fresh Hindi Wikipedia + C4 + codeparrot mix (§28) | — | CC BY-SA 4.0 |
| `ckpt_multi_qa` (AnuLM-Hindi-QA-400M) | as the base | Wikipedia question pairs, docstring pairs | CC BY-SA 4.0 |
| `ckpt_translate` (AnuLM-Translate-400M) | as the base | Samanantar + IIT Bombay | CC BY-NC 4.0, research only |
| `ckpt_coder`, `ckpt_coder_sft` (AnuLM-Coder-400M) | codeparrot-clean, code_exercises, fineweb-edu, Hindi mix | code_exercises, OpenCodeInstruct, glaive, docstring pairs | CC BY-NC-SA 4.0 |
| `ckpt_base_v2` (AnuLM-Base-v2-400M) | fineweb-edu, fineweb-2 Hindi, Hindi Wikipedia, codeparrot-clean, OpenCodeInstruct, opc-sft-stage1 | — | CC BY-SA 4.0 (published 2026-10-29) |
| `ckpt_omni` (AnuLM-Omni-400M) | base v2, through chat v3; frozen Whisper-small (MIT) and SigLIP ViT-B/16 (Apache 2.0) encoders | chat v3's data; SQuAD, MLQA translate-train, Wikipedia passages; BANKING77 and MASSIVE `hi-IN` as label text; LibriSpeech train-clean-100; Kathbath, MUCS 2021, FLEURS Hindi; Localized Narratives on CC BY 2.0 Open Images photos, DOCCI; TextVQA (VizWiz-VQA optional) | CC BY-SA 4.0; published by `experiments/omni_one.py` only if its release gates pass |
| `ckpt_chat_v3` | base v2 | dolly-15k, oasst2, Aya Hindi, Wikipedia question pairs | not released (CC BY-SA 4.0 would follow) |
| `ckpt_chat_v4`, `ckpt_chat_v5` | base v2 | chat v3's data plus indic-instruct Hindi subsets (Llama 2 outputs in `anudesh`) | not released |
| `ckpt_decide_lora_F`, `_hi_F`, `_india_F` | base v2, via `ckpt_chat_v3` | chat v3's data, then BANKING77 / MASSIVE `hi-IN` / synthetic Indian banking | not released |
| `ckpt_decide_modernbert`, `ckpt_decide_mb_hi` | ModernBERT-base (Apache 2.0; not trained here) | BANKING77 / MASSIVE `hi-IN` | not released |
| `ckpt_rc_pointer_v2` | base v2 (a branch of the run at 700k steps, decayed) | SQuAD, MLQA translate-train, Wikipedia hard negatives | not released (CC BY-SA 4.0 would follow) |
| `ckpt_asr_en_v3` | base v2; frozen Whisper-small encoder (MIT) | LibriSpeech train-clean-100 | not released (CC BY-SA 4.0 would follow) |
| `ckpt_asr_hi_v3` (`_v4`, `_v5`) | as the English ASR model, which it starts from | FLEURS Hindi + IndicTTS Hindi (v5 adds Kathbath and MUCS 2021) | not released; IndicTTS's terms are unclear |
| `ckpt_caption_v3` | base v2; frozen SigLIP ViT-B/16 (Apache 2.0) | Flickr30k | not released; **non-commercial** (Flickr30k) |
| `ckpt_vqa_v3` | as the captioner, which it starts from | Flickr30k (through the captioner), A-OKVQA | not released; **non-commercial** (Flickr30k, COCO photos) |

`NOTICE` §2 is the legal version of this table. Three points deserve
repeating: a model trained on GitHub code can reproduce licensed snippets
verbatim; the coder learned mostly from ChatGPT-generated exercises,
so anything derived from it inherits non-commercial, share-alike terms; and
the Flickr30k caption and VQA checkpoints are non-commercial, which is why
AnuLM-Omni-400M learns captions and image questions from Localized
Narratives, DOCCI and TextVQA instead.

## Disk and time to rebuild everything

| what | download | on disk after build | wall clock |
| --- | --- | --- | --- |
| Hindi ladder (§1–21) | < 1 GB | ~3 GB | minutes per corpus |
| three-language base corpus | ~1.5 GB | ~3 GB | 20 min |
| translation pairs + FLORES | ~3 GB | ~4 GB | 15 min |
| coder corpus | ~15 GB | ~25 GB (13.7 GB text + 5.6 GB tokens + SFT pairs) | 1 h download + 2 h CPU build |

All fetchers skip files already present, so an interrupted download
resumes where it stopped. `docs/OMNI.md` gives the disk and time for
AnuLM-Omni-400M's data (~150 GB free, most of it speech and images).
