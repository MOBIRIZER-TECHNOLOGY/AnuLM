# Publishing AnuLM: code, weights, citation

The end-to-end checklist for releasing this project as open source for
academic use. Steps 1–3 are done and committed in `C:\workspace\AnuLM` (tag v0.1.0);
4–8 need the project's GitHub and Hugging Face accounts. Nothing here uploads anything
by itself.

## Release status (2026-09-18)

| step | state |
| --- | --- |
| code, docs, recipes on GitHub | **done**: https://github.com/MOBIRIZER-TECHNOLOGY/AnuLM, public, branch `main`, tag and release `v0.1.0` |
| commit identity | GitHub no-reply address; no personal e-mail in history |
| weights exported | **done**: four safetensors folders, 0.8 GB each, in `release/` (gitignored) |
| weights on Hugging Face | **done**: toonist/AnuLM-Coder-400M, -Translate-400M, -Hindi-QA-400M, -Base-400M, and -Base-2K-400M (2026-09-19), each with its model card |
| v0.2.0: the second release | **Base-v2 published 2026-10-29** as toonist/AnuLM-Base-v2-400M (public, CC BY-SA 4.0); the 11 per-task exports in `release/` were checked (§3c) but superseded by one jointly trained model, toonist/AnuLM-Omni-400M, which `experiments/omni_one.py` exports and publishes when its gates pass |
| training data | **not uploaded, by design**: every corpus is re-fetched from its public source by the scripts; `docs/DATASETS.md` links each one with its licence |
| Zenodo DOI | open: enable the repository at https://zenodo.org/account/settings/github/ and re-publish the release, or publish v0.1.1 |
| hosted demo | Gradio Spaces need a PRO subscription (API returns 402 on free cpu-basic, 2026-09); `app.py` + `space/` are ready if that changes. `demo_colab.ipynb` is the free hosted route and is linked from the README |
| release pipeline tested end to end | **done 2026-09-18**: a 200-step tinyshakespeare checkpoint trained, exported, uploaded, downloaded and generated from, as [toonist/AnuLM-Smoke-30B](https://huggingface.co/toonist/AnuLM-Smoke-30B) (`docs/MODEL_CARD_SMOKE.md`). It is a deliberately useless 17M model whose card says so in its first line; it exists so this machinery is exercised without touching a released repo |
| hygiene | any Hugging Face token that was ever pasted into a chat or a terminal history should be revoked at https://huggingface.co/settings/tokens; the CLI login used here is a browser OAuth token that refreshes itself |

Local layout: `C:\workspace\AnuLM` is the only checkout, and the one that is
pushed. Until 2026-09-18 there were two — this one and a separate training
rig holding the checkpoints, corpora, logs and scheduled tasks, from which
changes were copied across with `git checkout-index -a -f
--prefix=C:/workspace/AnuLM/`. That rig has been deleted along with its
scheduled tasks, so edit, test, commit and push here. What it held that
still matters was published before it went: the four exported checkpoints
(Hugging Face and `release/`), the tokenizers under `data/`,
`coder_curve.csv` and the numbers in `docs/RESULTS.md`. The corpora are
re-fetchable by the scripts; the `.pt` training checkpoints, with their
optimizer state, are not — `TASKS.md` records what that means.

## 1. What is released, under which licence

| artefact | where | licence | why |
| --- | --- | --- | --- |
| code, docs, data recipes, tokenizers | GitHub repository | MIT (`LICENSE`) | original work |
| `sarvam/sarvam-30b/`, `sarvam/sarvam-105b/` | in the repository, unmodified | Apache 2.0 (Sarvam AI, `NOTICE` §1) | redistributed reference code |
| `AnuLM-Coder-400M` (`ckpt_coder_sft.pt`) and its base | Hugging Face | **CC BY-NC-SA 4.0** | tuned on jinaai/code_exercises, which is CC BY-NC-SA 4.0 and ChatGPT-generated |
| `AnuLM-Translate-400M` (`ckpt_translate.pt`) | Hugging Face | **CC BY-NC 4.0**, research only | Samanantar is CC BY-NC 4.0; the IIT Bombay corpus is research-only |
| `AnuLM-Hindi-QA-400M` (`ckpt_multi_qa.pt`) and `AnuLM-Base-400M` (`ckpt_multi36k.pt`) | Hugging Face | **CC BY-SA 4.0** | Wikipedia / Wikisource are CC BY-SA; C4 and fineweb-edu are ODC-BY |
| `AnuLM-Base-v2-400M` (`ckpt_base_v2.pt`) | Hugging Face, published 2026-10-29 | **CC BY-SA 4.0** | Hindi Wikipedia (CC BY-SA) in pretraining; fineweb-edu and fineweb-2 are ODC-BY. Two open questions, disclosed in `NOTICE` §2: GPL/AGPL files in codeparrot-clean, and opc-sft-stage1's Infinity-Instruct (CC BY-SA 4.0) and ShareGPT/WildChat-derived subsets |
| `AnuLM-Omni-400M` (`ckpt_omni.pt`) | Hugging Face, only if `experiments/omni_one.py`'s release gates pass | **CC BY-SA 4.0**, commercial use allowed | built for commercial use, from commercially usable data only (`NOTICE` §2): the share-alike sources are Wikipedia, SQuAD, MLQA, Dolly and MUCS 2021; the rest are CC BY, CC BY 2.0 photos (authors listed) or Apache 2.0. Flickr30k, COCO photos, IndicTTS and the synthetic Indian-banking set are left out. Inherits base v2's two open questions |
| training data | not redistributed | each source's own | `experiments/*_fetch.sh` and `*_build.sh` rebuild every corpus from its public source |

Each model card states the licence in its front matter and repeats the
non-commercial condition in prose where it applies. `NOTICE` §2 is the
full provenance table; keep it in sync with this one.

## 2. Repository hygiene (done 2026-09-18)

- Project renamed from nanosarvam to AnuLM. Class names `NanoSarvam` and
  `NanoSarvamConfig` remain as aliases in `model.py` so existing
  checkpoints, which pickle the config class by name, still load, and
  `BPE.load` still accepts the old `"nanosarvam-bpe"` format id, which the
  tokenizers uploaded with the weights carry; `bpe.py` writes `"anulm-bpe"`
  now and the committed tokenizers under `data/` were migrated in place
  (only the type string changed — the merges are byte-for-byte identical,
  so re-uploading the Hub copies is optional). Everything else carrying the
  old name was renamed with the project: the corpus fetchers'
  User-Agent, the scheduled tasks (`anulm_*`) and the paths in `TASKS.md`
  and `update_tasks.py`. The old working directory and its five scheduled
  tasks were deleted on 2026-09-18; `TASKS.md` records what that took with
  it and what had already been published.
- Local machine specifics removed from anything that ships: the Windows
  wrappers use `%LOCALAPPDATA%`, no user names, e-mails or host names remain
  outside `data/` and the logs. Logs, guard markers, editor backups and the
  `release/` staging folder are gitignored; tokenizers under `data/*.json`
  are the one thing kept from `data/`.
- `CITATION.cff` added. `README.md` opens with what the project is and the
  affiliation disclaimer.
- `python test_model.py`: 46 tests passed after the rename; 55 now, after
  one for the tokenizer format id below and two for the `transformers`
  export.

Before the first commit, decide the identity that will appear in history.
GitHub offers a no-reply address (`<id>+<user>@users.noreply.github.com`)
if the real e-mail should stay private:

```bash
git config user.name  "<name or handle>"
git config user.email "<id>+<user>@users.noreply.github.com"
git add -A && git status --short          # ~105 files, ~5 MB, no checkpoints, no data
git commit -m "AnuLM 0.1.0: code, docs, recipes, tokenizers"
git tag -a v0.1.0 -m "First public release"
```

## 3. Export the weights

`export_hf.py` converts a training checkpoint into a Hub-ready folder:
bfloat16 `model.safetensors` (no pickle, half the size), `config.json`,
the tokenizer, and a `README.md` model card with front matter. The
architecture is not in `transformers`; the card tells people to load the
folder with this repository's `model.py`, and `serve.py` / `sample.py`
accept a folder in place of a `.pt`.

```bash
python export_hf.py ckpt_coder_sft.pt release/AnuLM-Coder-400M \
    --license cc-by-nc-sa-4.0 --card docs/MODEL_CARD.md \
    --datasets codeparrot/codeparrot-clean jinaai/code_exercises HuggingFaceFW/fineweb-edu \
               nvidia/OpenCodeInstruct glaiveai/glaive-code-assistant \
    --repo toonist/AnuLM-Coder-400M
python export_hf.py ckpt_translate.pt release/AnuLM-Translate-400M \
    --license cc-by-nc-4.0 --card docs/MODEL_CARD_TRANSLATE.md --datasets ai4bharat/samanantar cfilt/iitb-english-hindi \
    --base-model toonist/AnuLM-Base-400M --repo toonist/AnuLM-Translate-400M
python export_hf.py ckpt_multi_qa.pt release/AnuLM-Hindi-QA-400M \
    --license cc-by-sa-4.0 --card docs/MODEL_CARD_HINDI_QA.md --base-model toonist/AnuLM-Base-400M --repo toonist/AnuLM-Hindi-QA-400M
python export_hf.py ckpt_multi36k.pt release/AnuLM-Base-400M \
    --license cc-by-sa-4.0 --card docs/MODEL_CARD_BASE.md --repo toonist/AnuLM-Base-400M
python export_hf.py ckpt_ctx2k.pt release/AnuLM-Base-2K-400M \
    --license cc-by-sa-4.0 --card docs/MODEL_CARD_BASE2K.md \
    --base-model toonist/AnuLM-Base-400M --repo toonist/AnuLM-Base-2K-400M
python serve.py --ckpt release/AnuLM-Coder-400M      # check an export loads and answers before uploading
```

To rehearse all of it without risking a released repo, train a throwaway
and push it to a separate model id — 29 seconds on a GPU, five minutes on
a CPU, and it exercises every step above plus the download:

```bash
python train.py --preset 30b --steps 200 --eval-every 50 --out ckpt_smoke.pt
python export_hf.py ckpt_smoke.pt release/AnuLM-Smoke-30B \
    --license mit --card docs/MODEL_CARD_SMOKE.md --repo toonist/AnuLM-Smoke-30B
python sample.py --ckpt release/AnuLM-Smoke-30B --prompt "KING RICHARD II:" --tokens 100
# then create_repo + upload_folder, and load it back from the Hub copy
```

**Never point a rehearsal at a released repo id.** The four above are the
only copies of those weights outside the Hub; the training rig that held
their `.pt` files is gone.

The cards are `docs/MODEL_CARD.md` (coder), `MODEL_CARD_TRANSLATE.md`,
`MODEL_CARD_HINDI_QA.md`, `MODEL_CARD_BASE.md` and `MODEL_CARD_BASE2K.md`. All four exports were
produced on 2026-09-18 into `release/` (0.8 GB each, gitignored) and each
was loaded back and generated from before being kept.

## 3b. The `transformers` layer

`export_hf.py` now writes it as part of any export. For a folder made before
it did -- which is all four on the Hub -- add it in place, without touching
`model.safetensors`:

```bash
python tools_hf_upgrade.py release/AnuLM-Coder-400M release/AnuLM-Translate-400M \
                           release/AnuLM-Hindi-QA-400M release/AnuLM-Base-400M
```

That converts the tokenizer (refusing to write one that disagrees with
`bpe.py` on any of 497 probe passages), copies `configuration_anulm.py`,
`modeling_anulm.py` and `model.py` in, adds `architectures`, `auto_map` and
`dtype: float32` to `config.json` beside the existing keys, and then loads the
folder back through `AutoModelForCausalLM` to check the logits and the greedy
continuation against this repository's own model before it reports success.

Uploading it is six small files per repo and no weights:

```python
api.upload_folder(folder_path="release/AnuLM-Coder-400M", repo_id="toonist/AnuLM-Coder-400M",
                  allow_patterns=["tokenizer.json", "tokenizer_config.json", "config.json",
                                  "configuration_anulm.py", "modeling_anulm.py", "model.py",
                                  "README.md"])
```

**Never re-upload the weights to fix a small file.** The `.pt` checkpoints
those exports came from no longer exist, so `release/` and the Hub are the
only copies.

## 3c. Models that are more than a language model (v0.2.0)

The speech, vision, reader and Decide checkpoints, and the jointly trained
Omni model, carry parts beside the backbone: a projector (`proj`,
`vision_proj`), a pointer or classifier head, LoRA weights, labels and a
calibration temperature. `export_hf.py --kind` writes those to
`extras.safetensors` (float32, keys `group/name`) and the rest to
`config.json["extras"]`; `model.load_checkpoint` puts them back as the
top-level keys the `.pt` had, so `speech_encoder.load_trained`,
`vision_encoder.load_trained`, `rc_pointer.PointerQA` and `decide.Decide`
take an exported folder, or a Hub id, wherever they took a `.pt`.

- `--adapter-of ckpt_chat_v3.pt` keeps only the adapter (27 MB instead of
  1.6 GB), after checking the adapter's copy of the backbone equals that
  file; on load the backbone is fetched from `--base-model`, or taken from a
  folder of that name beside the adapter.
- A ModernBERT decider ships its fine-tuned encoder in `backbone/` in
  `transformers` format, beside a copy of the original config and tokenizer.
- Only `--kind lm` gets the `transformers` layer (`auto_map`): for the others
  `AutoModelForCausalLM` would load the backbone and silently drop the rest.

AnuLM-Base-v2-400M is a plain `--kind lm` export carrying
`docs/MODEL_CARD_BASE_V2.md`. The second release's other model is exported
by `experiments/omni_one.py` (step 9), with the card it writes from
`docs/MODEL_CARD_OMNI.md` plus the run's measured table:

```bash
python export_hf.py ckpt_omni.pt release/AnuLM-Omni-400M --kind omni --license cc-by-sa-4.0 --card logs/omni_one/card.md --base-model toonist/AnuLM-Base-v2-400M --repo toonist/AnuLM-Omni-400M
```

The 11 per-task folders that came before it (chat, three Decide adapters,
two ModernBERT deciders, reader, English and Hindi ASR, caption, VQA) were
exported to `release/` the same way and checked, then superseded by the Omni
model; they are not published. What the check found is kept as the record
that `--kind` and `--adapter-of` round-trip correctly.

Checked on 2026-10-29 before anything was uploaded: each folder loaded back
with `load_checkpoint` holds the same keys and metadata as its `.pt`, the
heads, projectors and adapters bit-exact and the backbones within bfloat16
rounding (max relative difference 4e-3); the banking adapter, loaded from
its folder with the chat model beside it, made the same decisions as the
`.pt` on 100 test queries (95.40% answered at confidence >= 0.7); the
English speech model transcribed 20 clips word for word as the `.pt` did
(WER 8.8%); the caption model's held-out loss on 3 Flickr30k test images went
2.8643 -> 2.8612 and one greedy caption ran on by a few words -- the
bfloat16 rounding, the same trade the 0.1.0 exports made.

## 4. Create the remotes

- GitHub: create `MOBIRIZER-TECHNOLOGY/AnuLM` (public, no template, no licence
  picker: the repository already has `LICENSE`). Then
  `git remote add origin git@github.com:MOBIRIZER-TECHNOLOGY/AnuLM.git && git push -u origin main --tags`.
- Hugging Face: `pip install -U huggingface_hub`, `hf auth login` (paste a
  write token yourself; never store it in the repository), then one model
  repo per export:

```bash
hf repo create toonist/AnuLM-Coder-400M --type model
hf upload toonist/AnuLM-Coder-400M release/AnuLM-Coder-400M . --commit-message "AnuLM-Coder-400M v0.1.0"
# repeat for AnuLM-Translate-400M, AnuLM-Hindi-QA-400M, AnuLM-Base-400M
```

Then fill the placeholders `MOBIRIZER-TECHNOLOGY` and `toonist` in
`README.md`, `CITATION.cff` and the exported cards, and push again.

## 5. Make it citable

Enable the repository in Zenodo's GitHub integration, then publish a GitHub
release from the `v0.1.0` tag. Zenodo archives the release and mints a
DOI; put the DOI badge in `README.md` and the `doi:` field in
`CITATION.cff`. GitHub shows a "Cite this repository" button from the CFF
file automatically.

## 6. Optional: a public demo

As of 2026-09 Hugging Face requires a PRO subscription to host Gradio
Spaces even on the free CPU tier, so this is not deployed; the free
alternative is `demo_colab.ipynb`, which clones the repo, downloads a
checkpoint and launches `app.py` with a public link. If a PRO or ZeroGPU
account becomes available, `app.py` is the Space: it imports `Engine` from `serve.py`, shows the modes the
loaded checkpoint supports, and downloads the weights from the Hub when
the Space variable `ANULM_REPO` is set. Create a Gradio Space, add
`app.py`, `serve.py`, `model.py`, `bpe.py`, `make_qa.py` and a
`requirements.txt` with a CPU torch wheel, `safetensors`, `gradio` and
`huggingface_hub`, set `ANULM_REPO=toonist/AnuLM-Coder-400M`, done.
One Space per checkpoint, or one Space with a dropdown that reloads.

## 7. Optional: a write-up

`docs/RESULTS.md` is most of a technical report already. A short arXiv
note (cs.CL) with the architecture study (`docs/ARCHITECTURE.md` §10), the
ablations (§12, §16) and the three headline numbers would give people a
paper to cite alongside the DOI.

## 8. After release

- Add the "not affiliated" line to every model card, not only the README.
- Answer issues about reproduction with the exact script and seed; every
  run in `docs/RESULTS.md` records both.
- Do not merge anyone's training data into the repository; keep `data/`
  ignored and recipes only.
