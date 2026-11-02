# Course route

This is a practical study route for the current companion code, updated
2 November 2026. It uses the available book examples and package commands.
It does not claim that the historical unpublished `course/` stub labs exist.
Start with [Current companion workflow](current-workflow.md).

| Session | Read | Practical work | Evidence to keep |
|---|---|---|---|
| 1. First model | Chapters 1, 3, 8, 20 | Run `python book/tools/first_prediction.py` on CPU. Explain the input/output shapes. | Environment, seed, observed loss and a short sample. |
| 2. Data and tokenizer | Chapters 9, 21, 24 | Prepare distinct local documents, plan a custom run with `--dry-run`, then run a small CPU experiment. | Source/terms, disjoint document groups and training-only tokenizer evidence. |
| 3. Training and recovery | Chapters 22, 23, 25, 26 | Inspect `pipeline.json`, stage logs and `.last` files. Rerun unchanged to verify reuse; change a step budget in a separate experiment and verify refusal. | Recipe/file hashes and explanation of the refusal. |
| 4. Multimodal inputs | Chapters 35, 36, 37 | Follow the pinned India setup and real preflight; distinguish frozen encoders from newly trained parameters. | Per-language/task mechanics results and eligibility counts. |
| 5. Evaluate one checkpoint | Chapters 24, 28, 30, 37 | Use the research evaluation app and a declared validation plan. Inspect per-task regressions and lexical metric limitations. | Checkpoint/input identities, language/split/source fields, metrics and failures. |
| 6. Export and launch decision | Chapters 38, 39, 40, 41 | Repeat relevant held-out checks on the export and complete the launch worksheet. | Native/export differences, unresolved serving gates and a scoped release decision. |

Commands run from the companion repository with its environment active. Session 1
needs no trained checkpoint. Later work needs the dependencies, data and artifacts
named in the [current workflow](current-workflow.md). Do not start a second GPU
exercise while the full training job is using the same small GPU.

## Questions you should be able to answer

1. Which parameters start randomly, and which encoders are pretrained?
2. Why must duplicate documents/media be grouped before train/validation splitting?
3. What is an optimizer step when accumulation is greater than one?
4. Why can the average validation score improve while one language regresses?
5. Why do token filtering and a small validation sample limit a language claim?
6. Why does passing a numeric evaluation gate not approve a demo case or service?

For multilingual work, use [Language coverage](language-coverage.md). For the
publication decision, use [Launch checklist](launch-checklist.md). New-reader
trials, accessibility review and expert review of the complete manuscript are
still required; this route is an organized learning plan, not proof those reviews
have occurred.
