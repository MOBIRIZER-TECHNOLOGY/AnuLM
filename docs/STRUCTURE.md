# Repository structure and migration guide

The project has one model package and one book package. Start at the repository
README; book readers can go directly to [the book](../book/README.md). Keep
generated checkpoints, corpora and build output separate from source documents.

```text
AnuLM/
  README.md                 project entry point
  CONTRIBUTING.md           development and pull-request workflow
  pyproject.toml            install metadata and anulm command
  model.py                  compatibility import for old checkpoints
  anulm/
    model.py                native model implementation
    training/               trainer, SFT, LoRA and optimizers
    tokenization/           byte-level BPE and tokenizer adapters
    inference/              sampling, serving and checkpoint export
    data/                   fetchers, converters and dataset builders
    evaluation/             evaluation commands
    tasks/                  retrieval, readers and classification
    apps/                   optional demo applications and API
    multimodal/             optional audio, image and joint-model work
    hf/                     standalone export templates and conversion tools
    integrations/           optional external-framework adapters
    tools/                  environment checks and maintenance helpers
  book/                     reader guides, chapters, examples and book tools
  tests/                    model, layout, multimodal and demo checks
  config/                   approved demo-evidence catalogue
  docs/                     usage guides, model cards, reference and roadmap
  scripts/windows/          optional Windows launch/maintenance helpers
  measurements/training/    committed historical loss curves
  experiments/              measured research scripts
  examples/                 application examples
  golden/                   committed evaluation fixtures
  data/                     local corpora and committed small tokenizers
  runs/                     local training checkpoints, ignored by Git
  release/                  local exported models, ignored by Git
```

## Commands and imports

The [demo evidence guide](DEMO_VALIDATION.md) documents approved cases and the
explicit opt-in for live research controls. Decide serving uses AnuLM only.

Install from this checkout with `python -m pip install -e .`. Then use:

| Previous command | Current command |
|---|---|
| `python train.py ...` | `python -m anulm train ...` |
| `python sample.py ...` | `python -m anulm sample ...` |
| `python serve.py ...` | `python -m anulm serve ...` |
| `python api.py ...` | `python -m anulm api ...` |
| `python export_hf.py ...` | `python -m anulm export ...` |
| `python finetune.py ...` | `python -m anulm finetune ...` |
| `python quickstart.py` | `python -m anulm quickstart` |
| `python test_model.py` | `python -m tests.test_model` |

Data and evaluation modules use their complete module name, for example
`python -m anulm.data.fetch_hindi --help`. The
[complete path map](source-paths.json) includes every relocated root source
file, Windows helper and recorded curve.

New Python code imports `AnuLM` and `AnuLMConfig` from `anulm`, or imports a
specific component from `anulm.model`. Import training helpers from
`anulm.training.train`. Do not add the old flat files back to the repository
root to make an import work.

## Checkpoint and export compatibility

The root `model.py` intentionally aliases the native `anulm.model` module.
Checkpoints pickled with `model.AnuLMConfig` can still be read. It also preserves
legacy access to module flags and private helpers; it is not a second model
implementation.

The native model source and the standalone Hugging Face templates retain their
original contents. Export copies the actual native model and wrapper templates,
not the compatibility import. Exported folders still contain their own
`model.py`, `configuration_anulm.py` and `modeling_anulm.py`, and do not require
the repository package to implement the model.

Relative dataset/output paths supplied on the command line still resolve from
the caller's working directory. Built-in repository resources are found through
`anulm.paths.PROJECT_ROOT`; moving a module no longer changes where it looks for
`web/`, `docs/`, `data/` or evaluation fixtures. Use the editable source checkout
for the optional applications that depend on those repository resources.

## Historical book citations

The book's line citations describe the original source at its recorded revision.
They are not line numbers in the reorganized training/import wrappers. Use the
path map to locate the current component and use the historical revision when
checking an exact quoted excerpt. The unpublished course pin remains a known
limitation, not a successful verification.

## Windows helpers

The helpers now live together under `scripts/windows/`. They return to the
repository root before launching module commands, and training-process guards
recognize both the old filenames and new module names. Existing scheduled tasks
that reference the old helper paths must be updated by their operator. This
refactor does not change scheduled tasks or start maintenance/training services.

## Verification

Run `python -m tests.test_model` and `python -m unittest tests.test_layout`.
Book changes additionally use [the book contribution checks](../book/CONTRIBUTING.md).
The CI workflow checks the model on Linux/Windows and the complete HTML book
structure. Tests and example runners report their optional exclusions explicitly.
