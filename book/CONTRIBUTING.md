# Contributing to Inside AnuLM

Start with [the reader guide](docs/reader/start-here.md) if you want to learn
the model. Use [the authoring plan](docs/authoring/book-plan.md) and
[style guide](docs/authoring/style.md) when changing the book.

## Where a change belongs

| Change | Folder |
|---|---|
| Chapter explanations, exercises and answers | `chapters/` |
| Small executable teaching examples | `code-examples/` |
| Figures and their generators | `figures/` and `tools/figures/` |
| Checked measurement records | `measurements/` |
| Reader setup, reading routes and launch guidance | `docs/reader/` |
| Source maps and research references | `docs/reference/` |
| Review findings and chapter review records | `docs/reviews/` |
| Chapter order and entry-card metadata | `config/` |

Keep local weights, raw corpora, environments, temporary files and generated
editions out of Git. The ignore file lists those locations. Do not replace
measurement records with your machine's results unless the change deliberately
updates the corresponding chapter and records the method.

## Verify a book change

From the `book/` folder:

```bash
python -m pip install -r tools/requirements.txt
python tools/build_preview.py
python tools/build_preview.py --print --html-only
python tools/verify_preview.py
```

For print-layout changes, also run `python tools/build_preview.py --print`,
then `python tools/verify_preview.py --pdf` and inspect the rendered pages.
Browser printing requires an installed supported browser; the HTML-only check
does not.

From the companion repository root, the selected CPU examples run with:

```bash
python book/code-examples/ch41_audit.py --examples-only
```

The explicit example manifest reports exclusions separately. Historical audits
still require the unpublished pin and named artifacts; report missing inputs
as unavailable. Do not relabel a chapter reviewed merely because its formatting
or excerpt checks pass.

## Open a contribution

Create a branch from `main`, make one coherent change, and open a pull request.
Explain the reader-visible result, the checks you ran, and any unavailable
checks. Add a review record for changed technical claims. New chapters must
also update the book manifest, chapter entry cards, references, and example
inventory where applicable.
