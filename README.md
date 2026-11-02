# AnuLM

Current companion update (2 November 2026): see [current status](docs/CURRENT_STATUS.md) and [language coverage](docs/LANGUAGE_SUPPORT.md). Historical model cards, experiment numbers and commands below must not be presented as measurements of the completed India research run.

A small, trainable language model for learning, experimentation, and measured
Hindi/English/Python work. Start with the book to understand the implementation,
or use the command line to train and run a model. The demonstration applications
and multimodal extensions are separate from the core text model.

## Start here

- **New training and evaluation workflows:** [selective V2 improvements](docs/V2_IMPROVEMENTS.md), including custom-corpus training, accumulation and a research evaluation app.
- **Set up on another GPU computer:** [portable training setup](START_TRAINING.md), including Windows/Linux instructions and the pinned environment.
- **Learn from the book:** [Inside AnuLM](book/README.md), with 41 chapters and six appendices.
- **Run your first example:** [installation and first CPU model](book/docs/reader/start-here.md).
- **Find a component:** [repository structure and migration guide](docs/STRUCTURE.md).
- **Multimodal demo and evidence:** [validated examples and remaining checks](docs/DEMO_VALIDATION.md).
- **Download a small real dataset pack:** [pinned sources, setup and commercial-use terms](docs/DATA_SETUP.md).
- **India-focused training:** [languages, overnight stages, recovery and limits](docs/INDIA_TRAINING.md).
- **Contribute:** [CONTRIBUTING.md](CONTRIBUTING.md) and the [book contribution guide](book/CONTRIBUTING.md).
- **Model releases, measurements and detailed usage:** [project reference](docs/project-reference.md).

## Install and run

Use Python 3.10 or newer and a virtual environment. From this checkout:

```bash
python -m pip install -e .
python -m anulm quickstart
python book/tools/first_prediction.py
```

The core install includes PyTorch and NumPy. Install the PyTorch build appropriate
to your hardware before the editable install if you need a particular CUDA build.
The first book example runs on CPU without pretrained weights or downloaded data.

Train the small model on sample text, then generate a continuation:

```bash
python -m anulm train --preset 30b --steps 50 --device cpu --out runs/first.pt
python -m anulm sample --ckpt runs/first.pt --device cpu --prompt "The king " --tokens 40
```

The trainer downloads Tiny Shakespeare when no data is supplied, with a small
built-in fallback if the download fails. Its default checkpoint is `runs/ckpt.pt`.
Use `--data PATH` for your own corpus. These are learning runs, not production
quality claims. Read the [training tutorial](docs/TUTORIAL.md) before a longer run.

Optional applications and export support:

```bash
python -m pip install -e ".[demo]"
python -m pip install -e ".[api]"
python -m pip install -e ".[export]"
python -m pip install -e ".[data]"
```

Install only the extras you need. `python -m anulm --help` lists the main commands;
`python -m anulm train --help` lists training options. Individual data/evaluation
modules can be run with `python -m anulm.data.fetch_hindi --help`, for example.

## Folder guide

| Folder | Purpose |
|---|---|
| `anulm/model.py` | Core model implementation |
| `anulm/training/`, `tokenization/`, `inference/` | Training, text encoding, generation and export |
| `anulm/data/`, `evaluation/`, `tasks/` | Dataset preparation, evaluation and task-specific components |
| `anulm/apps/`, `multimodal/`, `hf/`, `integrations/` | Optional applications, audio/vision and interoperability |
| `book/` | Manuscript, reader guides, examples, figures and book tools |
| `tests/` | Model, package-layout, multimodal and demo-evidence checks |
| `config/` | Approved demo catalogue; empty until real results are reviewed |
| `docs/` | Project reference, usage guides and roadmap |
| `scripts/windows/` | Optional Windows launch and maintenance helpers |
| `measurements/training/` | Recorded training curves |
| `experiments/`, `examples/`, `golden/` | Research scripts, application examples and evaluation fixtures |
| `data/`, `runs/`, `release/` | Local data, checkpoints and exports; large artifacts are ignored |

The single root `model.py` is a compatibility import for old pickled checkpoints.
New code uses `from anulm import AnuLM, AnuLMConfig` or `from anulm.model import ...`.
The core model and standalone Hugging Face templates retain their implementation;
see the [path map](docs/source-paths.json) for historical filenames.

## Verify a contribution

```bash
python -m tests.test_model
python -m unittest tests.test_layout
python -m unittest tests.test_multimodal_contracts tests.test_validated_demo
```

The original model suite reports its optional dependency skips in the output.
Book checks are documented in [book/CONTRIBUTING.md](book/CONTRIBUTING.md).
The [launch checklist](book/docs/reader/launch-checklist.md) separates a working
model from a production service. Current limitations and pending reviews remain
visible in the book; reorganizing the files does not certify those paths.

Code licence: [MIT](LICENSE). Data and checkpoint conditions are recorded in
[NOTICE](NOTICE); [CITATION.cff](CITATION.cff) describes attribution.
