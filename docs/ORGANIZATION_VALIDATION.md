# Organization change: validation record

Validated locally on 1 November 2026 using Windows, Python 3.14.5, PyTorch
2.14.0+cu130, and an RTX 5050 Laptop GPU. This records checks of the directory
and package migration; it is not a model-quality or production certification.

## Preserved implementation

Compared directly with public base revision
`ae592387dc373cc6f579abce444c7f08c9e373a3`:

- `anulm/model.py` is byte-for-byte the original native `model.py`.
- `anulm/hf/configuration_anulm.py` and `modeling_anulm.py` retain the original
  standalone wrapper template contents.
- The root `model.py` is a module alias for old pickled checkpoint references;
  it is not another implementation.

Other modules have explicit package imports, resource locations, updated
commands and checkpoint-output defaults. The trainer creates missing output
directories before its existing atomic save. Export copies the real native
model and templates, not the compatibility alias.

## Checks completed

| Check | Local result |
|---|---|
| Editable install | `pip install --no-deps --no-build-isolation -e .` succeeded in an isolated environment using the installed model dependencies. |
| Model suite | 66 passed, 0 failed. Gradio was installed for this run, so the two UI checks that normally skip without it executed. |
| Layout regressions | Five passed: native/legacy import identity and flags, old config pickle loading, migration-map resources, installed CLI outside the checkout, and migrated experiment launch argument lists. |
| Previously trained checkpoint | The 17M checkpoint trained before the move loaded through the installed CLI and generated a continuation; its stored step was 599 and validation loss 1.6116. |
| New CLI training | 50 GPU steps from random weights completed through `python -m anulm train`, called outside the repository. It saved/reloaded a checkpoint and generated text through `python -m anulm sample`. This was a pipeline smoke run, not a convergence experiment. |
| Book first example | Two-layer CPU example completed; loss fell from 5.6009 to 0.0802 on its toy text. |
| Selected book examples | Eight passed, zero failed, 58 explicitly excluded by the CPU smoke manifest. |
| Book structure | Complete HTML and print-HTML editions contained 41 chapters, six appendices and 41 entry cards; internal anchors, unique IDs and figure paths passed. |
| Reader file links | Project/book entry points and book Markdown file links resolved in the assembled repository. |
| Static source checks | Project-owned package, test, experiment, book-tool and book-example Python files parsed successfully. Annotated third-party reference files were not treated as executable package code. |

The short new-CLI training run used the earlier local sample corpus, batch 16,
context 128, sparse expert dispatch and 50 steps. Its validation loss was 2.8129.
This corpus split and schedule differ from the book's longer reference runs;
the number is not a before/after quality comparison for the refactor.

## Checks not claimed

- Cross-platform CI is configured for the pull request; its remote result is
  separate from these local Windows checks.
- Long-running research experiments and Windows maintenance/scheduled tasks
  were not executed. Their imports, command argument lists, paths and process
  matching were migrated; operators must retarget existing scheduled-task paths.
- Space/Colab deployment was not launched. Their package installation and entry
  point instructions were updated for the complete source checkout.
- Historical book audits still require the unpublished pin and named artifacts.
  Three course solution excerpts are unavailable in the public base revision.
- The book's outstanding chapter reviews, human reader/expert review, API fixes,
  corrected chat evaluation and release permissions remain open as documented
  in its launch checklist.

Generated PDFs/HTML, local logs, corpora, environments and model weights are
excluded from the contribution. Reproduce the checks using the project and book
contribution guides rather than relying on machine-specific output paths.
