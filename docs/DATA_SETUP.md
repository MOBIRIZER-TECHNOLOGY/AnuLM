# Reproducible small multimodal dataset setup

For the full India-focused run on another GPU computer, use
[START_TRAINING.md](../START_TRAINING.md). The portable ZIP includes downloader
code and pinned sources; it does not include the large datasets themselves.
The commands below are the separate small mechanics-check pack.

From the existing AnuLM source checkout, in a virtual environment:

```bash
python -m pip install -e ".[data]"
python -m anulm.data.setup_multimodal fetch
python -m anulm.data.setup_multimodal check
```

This downloads **64 real examples**, not full training corpora: eight train and
eight validation examples for each of image captioning, English speech, Hindi
speech and decision labels. It does not train a model, download encoders, or
approve demo cases. No comparison model is required. A later inference check
still needs the author's trained checkpoints and matching runtime assets.
This initial pack covers captions, ASR and decision labels; it does not cover
VQA, speech generation or all multilingual tasks. Those need separate datasets
and evaluations before being exposed as validated features.

Local verification on 1 November 2026: the default pack occupied about 47 MB,
all 64 records passed data checks, existing image/speech manifest readers loaded
them, a separate fresh 16-record setup succeeded, and repeat setup passed with
network access blocked. Five offline setup tests and 13 multimodal regression
checks passed. These results describe this tested environment, not a guarantee
that upstream servers or every developer environment will remain available.

## Sources and commercial-use terms

Verified against the publishers on 1 November 2026:

| Data | Official source | Pinned version | Terms |
|---|---|---|---|
| Image + caption | [Google DOCCI](https://google.github.io/docci/) | GCS generations recorded in the source config | Google licenses images and annotations under CC BY 4.0 |
| English/Hindi speech + transcript | [Google FLEURS](https://huggingface.co/datasets/google/fleurs) | Dataset commit in the source config | CC BY 4.0 |
| Decision labels | [PolyAI BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets) | Repository commit in the source config | [CC BY 4.0](https://github.com/PolyAI-LDN/task-specific-datasets/blob/master/LICENSE) |

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) permits commercial
sharing and adaptation subject to its conditions, including credit, a license
link and indicating modifications. The generated `ATTRIBUTION.md`, source
records and license files must accompany use/distribution as applicable.
The license gives no warranty and does not grant every possible third-party
right, such as privacy/publicity rights. This is a verified license selection,
not an unconditional guarantee for every commercial product.

The selection excludes research-only Flickr30k, mixed-license COCO photos,
unclear IndicTTS terms and gated datasets. New downloads **do not clear the
lineage of existing weights**. Earlier base/coder/projector checkpoints have
separate dataset histories and unresolved terms documented in [DATASETS.md](DATASETS.md).
Encoder weights and other runtime dependencies also retain their own terms.

## Files and portability

The default output is `data/commercial-smoke/`, ignored by Git:

```text
commercial-smoke/
  image_train.jsonl, image_validation.jsonl
  speech_en_us_train.jsonl, speech_en_us_validation.jsonl
  speech_hi_in_train.jsonl, speech_hi_in_validation.jsonl
  decide_train.jsonl, decide_validation.jsonl
  banking77/                 small CSV subsets and the full label catalogue
  media/                     actual JPEG images and WAV recordings
  raw/                       source annotations and CSV snapshots
  licenses/                  license text and publisher/card snapshots
  SOURCE_CONFIG.json         exact source URLs, versions and credits
  DATA_LOCK.json             local SHA256/size inventory and sample counts
  READY.json                 only written after data checks pass
  ATTRIBUTION.md             creators, source links, terms and changes
```

`config/dataset-sources.json` and `anulm/data/setup_multimodal.py` are committed
source files for developers; the downloaded pack is not committed. Images/audio
paths in the manifests resolve beside those manifests, so the pack can be moved.
The existing speech loader also retains compatibility with older repo-relative
manifests. These paths load through the existing vision/speech manifest readers.
Decision JSONL rows carry questions and label names; their small CSV counterparts
are isolated from the project's full BANKING77 data. They do not exercise all
77 labels and are insufficient to train or benchmark a complete decision system.

The image validation subset comes from DOCCI's official test partition; speech
validation comes from FLEURS dev; decision validation comes from BANKING77 test.
Keep these partitions out of training. Checks reject equal media hashes and
normalized decision questions across train/validation. This is exact-content
checking, not semantic/perceptual deduplication or a speaker-independent claim.
Files decode successfully; selected speech is mono 16 kHz and at most 30 seconds.
These checks validate data, not model predictions or complete manifest semantics.

## Bounded downloads and recovery

The script reads only an initial compressed prefix from each large official
media archive and closes the connection once enough eligible files are found.
It never downloads the entire 7.1 GB DOCCI archive for this small pack. The
default cap is 64 MiB **per archive**, plus small annotations. Sampling follows
the pinned archive order; it is not representative random sampling. The larger
FLEURS archives are likewise streamed only as far as needed.

```bash
python -m anulm.data.setup_multimodal fetch --out data/smoke-small --per-split 2
python -m anulm.data.setup_multimodal check --out data/smoke-small
```

Repeated fetches verify and reuse a complete matching pack without network
access. Interrupted fetches retain a configuration marker and can be retried
with the same command; bounded prefixes are read again, not byte-resumed.
Unrelated nonempty output directories are refused. Different counts/source
versions need a new output directory. Errors never mark partial data ready.
Requests have timeouts and limited retries; a failed connection during a stream
requires rerunning the command. No access token or gated-dataset approval is used.

If a prefix budget is exhausted, use a larger `--max-stream-mib` (at most 1024),
or reduce `--per-split`. Missing optional media libraries are caught before
downloads; reinstall the data extra. A 403/404 identifies the exact failing
source rather than silently substituting another dataset. Publisher availability
cannot be guaranteed by the setup script.

For larger datasets, review the existing [multimodal guide](OMNI.md) and
[dataset inventory](DATASETS.md) before choosing a source. The older `fetch all`
commands can download tens of gigabytes and include other licensing conditions;
they are deliberately not the default setup command here.

## Offline regression checks

```bash
python -m unittest tests.test_dataset_setup
```

These exercise byte limits, oversized metadata, modified files, split overlap
and portable/legacy speech paths without downloading data or weights.
