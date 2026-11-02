# Inside AnuLM

*A language model you can read, line by line.*

This technical preview contains **41 chapters and six appendices**. It is for
Python programmers who want to understand and change a language model.

## Read

Current package: [workflow](docs/reader/current-workflow.md), [language coverage](docs/reader/language-coverage.md), and [course route](docs/reader/course-route.md), updated 2 November 2026. These guides distinguish the current local run from historical chapter results.

1. [Start here](docs/reader/start-here.md): install the project and train your first small CPU model.
2. [Choose a learning path](docs/reader/learning-paths.md): dense model first, advanced architecture, or deployment evaluation.
3. [Browse the chapters](chapters/README.md): the complete table of contents.
4. [Check launch requirements](docs/reader/launch-checklist.md): what still needs evidence before release.

## Find a file

| Folder | What belongs here |
|---|---|
| `chapters/` | Book chapters and appendices |
| `code-examples/` | Small executable teaching examples |
| `figures/` | Figures used by the manuscript |
| `measurements/` | Recorded evidence for numerical claims |
| `docs/reader/` | Setup, reading paths, front matter and launch guidance |
| `docs/authoring/` | Writing plan, brief and style guide |
| `docs/reference/` | Architecture, source map and research references |
| `docs/reviews/` | Review records, findings and integration notes |
| `config/` | Chapter order, entry cards and example inventory |
| `tools/` | Build, checking and figure-generation tools |
| `preview/` | Generated HTML/PDF editions; not committed |

Local environments, training checkpoints, temporary output and reference assets
are ignored. They are not prerequisites for reading the book.

## Build the editions

From this book folder:

```bash
python -m pip install -r tools/requirements.txt
python tools/build_preview.py
python tools/build_preview.py --print --html-only
python tools/verify_preview.py
```

Open `preview/index.html` to read the complete local edition. To create the PDF:

```bash
python tools/build_preview.py --print
python tools/verify_preview.py --pdf
```

PDF printing needs a supported installed browser. Generated editions include the
reader guides, chapter entry cards and appendices. The build checks the explicit
manifest so a partial selection cannot masquerade as the full book.

## Contribute

Read [CONTRIBUTING.md](CONTRIBUTING.md) for the authoring workflow, checks and
pull-request expectations. Changes to technical claims need evidence, not just
a formatting pass.

Historical citations use the original filenames at the recorded revisions.
The reorganized project's [structure guide](../docs/STRUCTURE.md) and
[path map](../docs/source-paths.json) show their current locations. The private
historical pin and optional course labs remain unavailable; read the front matter
before attempting the historical audits.

The book is still a technical preview. Outstanding expert/reader reviews, operational service checks, corrected chat
evaluation and publication details are listed in the
[launch checklist](docs/reader/launch-checklist.md).


## Text license

Original book prose and reader/course guides: [CC BY 4.0](LICENSE-TEXT.md). Companion code retains MIT; separately identified third-party material retains its own terms.
