# AnuLM-Omni-400M: improvement plan

Written 2026-11-04 from the November run (kept checkpoint: step 9,375 of
12,500; `logs/omni_one/results.json`) and the 117-sample demo check
(`examples/samples/`, CPU smoke run). Every "now" number is measured; every
target is a goal, not a result. Update the status column as work lands.

## The model, skill by skill

| Skill | Now | Target | Why it falls short | Plan | Status |
|---|---|---|---|---|---|
| Chat | held-out loss 3.475, fails the 3.30 gate; Python code from memory broken; invented details | <= 3.30, ideally ~3.15 (where it started) | re-read the same 22,873 chat pairs 7.4 times after chat v3 had learned them | 1. distil from chat v3 on chat batches (`--distill chat`): probed, no help (step 625: 3.358 vs 3.351 without), so the drift is the other tasks, not repetition. 2. a shorter run (3,125 steps) keeping only checkpoints within `--max-ratio chat=1.049`. 3. more human-written chat data (multi-turn oasst2, Aya Hindi); short correct Python answers | 2 done: run b, 3.309 (misses by 0.009); a 5% weight blend gives 3.299 (exported copy 3.2985). Next: 3, more chat steps |
| Reading, English | SQuAD F1 48.4 (gate >= 30) | >= 55 (the separate reader: 57.2) | 12% of steps; answers are generated, not pointed to | raise reading's share; a short-answer prompt format | next run |
| Reading, Hindi | MLQA F1 11.7 | >= 25 | trained only on machine-translated MLQA | native Hindi extractive QA; eval files always linked (a missing file used to skip silently) | data being collected |
| Wikipedia lookup | 4/10 demo questions | >= 7/10 | the retriever picks the wrong page ("Hamlet" -> the 1964 film) | the misses were the reader repeating the question: retry with the passages rotated, cite the passage used; `rag build-dense` builds the index | 6/10 demo, done |
| Image questions | TextVQA 17.3% (gate >= 10); demo 2/12 | >= 25% | memorises: training-batch loss ~1.9 vs held-out 3.6 | more varied, commercially usable photo questions; stop at the best VQA step | data being collected |
| Captions | held-out loss 2.725; fluent, wrong details | <= 2.5; right object counts | same memorisation | more Localized Narratives captions on Open Images | data being collected |
| Speech, English | WER 5.5% (old model 7.1%); 11.4% on dev-clean demo clips | <= 5% gate set; <= 9% dev-clean | names and rare words | LibriSpeech train-clean-360 | data being collected |
| Speech, Hindi | mean WER 32.0% (old 40.0%); MUCS 46.1% | mean <= 28%; MUCS <= 40% | Hindi-English code-mixed speech | more code-mixed and native Hindi audio (CC0 / CC BY) | data being collected |
| Intents | BANKING77 85.0%, MASSIVE-hi 80.7% (3,125 steps) | >= 89% / >= 84% | 5% of steps was enough in 12,500 steps (87.7%), not in 3,125 (69.7%) | 15% of steps | done 2026-11-05 |

## The system around the model

| Area | Now | Target | Plan | Status |
|---|---|---|---|---|
| Model selection | the combined score kept a checkpoint whose chat got worse | never keep a checkpoint that fails a gate | keep the best checkpoint among those within the chat gate (`--max-ratio`) | done |
| Release | not published: 6 of 7 gates (run b); the 5% blend passes 7 of 7, exported copy included (chat 3.2985, by 0.0015) | all 7 gates, on the exported copy too, with a margin | export, re-run every gate (reading included), publish only with approval | after the chat fix |
| Demo | 117 one-click samples, CPU-checked; Benchmark tab records the old separate models | GPU-checked; Benchmark re-recorded with Omni | `experiments/demo_samples_check.py` on GPU; re-record; Chat tab keeps code lines | Chat tab done |
| Landing page | `site/index.html` drafted | live on GitHub Pages and a Hugging Face Space | publish with the release, with approval | waiting on release |
| Training reliability | GPU shared with other apps: 13% slower, one out-of-memory crash | the GPU to itself | quit other GPU apps during runs; the pipeline waits for a free GPU | ongoing |
| Book | pinned to b42998d; the November run in Chapter 37 | complete, print-ready | Chapter 37 gets the chat-fix result; worked examples by hand; a real capstone; print layout | ongoing |
| Branches | work on `restructure`; `main` is old | `main` = this code | point `main` here; remove extra working folders, with approval | after release |

## Order of work

1. Chat A/B probes (distillation did not help), then 4 h runs with gate-aware
   selection (`omni_one.py --retrain`): run a failed intents, run b (intents
   15%) missed chat by 0.009, a 5% weight blend passes (see `docs/OMNI.md`).
   Next: chat at ~25% with the new human-written chat pairs (`data/next/`).
2. During the run, no GPU: lookup fixes and the index command, gate-aware model
   selection, the Chat tab's code display, new data under `data/next/`, the book.
3. After the run: gates on the exported model, the GPU demo check, the
   benchmark re-recording, then a release decision.
4. Next run: reading (English and Hindi), image questions and captions, more
   speech, on top of the chat fix.
