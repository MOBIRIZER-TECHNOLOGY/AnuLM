# Learning paths

For the revised multilingual/multimodal package, read [Current companion workflow](current-workflow.md), [Language coverage](language-coverage.md) and [Course route](course-route.md). This 2 November 2026 update does not replace historical measurements or certify a production release.

Keep the chapter numbers as reference points. You do not have to read all 41 chapters in sequence before running a model. Start with [Start here](start-here.md), then choose a path.

## Path 1: understand and train your first model

For readers new to transformers:

1. **Orientation:** Chapters 1 and 3. Skim Chapter 1's repository inventory; you can return when you need a particular file. Chapter 2's history is optional on this first pass.
2. **Mathematical tools:** Chapters 4–7. Read the worked example before the implementation details. If the notation is familiar, use the exercises to check yourself.
3. **Text becomes vectors:** Chapters 8–10.
4. **A dense transformer:** Chapters 11–16. Run the small examples. Then read Chapter 20 for the full forward path, following the dense layer and skipping the expert-specific arithmetic on this pass. The first example in Start here supplies a complete dense model.
5. **Training and evaluation:** Chapters 21–26. Begin with synthetic examples; the measurements on the released corpus are a separate activity.
6. **Generation:** Chapters 27–28.
7. **A working checkpoint:** Chapter 41.1–41.4. Use the small preset, trace it, and add top-p. The released 400M comparison is optional and needs additional artifacts.

**Finish line:** explain one token's path, train a small model, interpret its loss, and test a decoding change. This path does not require the optional course labs.

## Path 2: understand AnuLM's advanced architecture

After Path 1's dense-transformer and generation sections:

- Read Chapters 17–18 for expert routing and dispatch. Return to Chapter 20 for the released model's complete parameter arithmetic.
- Read Chapter 19 for MLA **after Chapter 27's KV cache**. MLA is an alternative architecture; the released model uses GQA.
- Read Chapter 29 for long context after understanding RoPE (Chapter 14) and the cache (Chapter 27).
- Read Chapters 30–33 for SFT, LoRA, classification, and retrieval. These are different ways to adapt or use a model; they are not interchangeable quality improvements.
- Treat Chapter 34's preference learning and Chapters 35–37's multimodal work as separate extensions. Read the dependency notes before installing packages or downloading assets.

**Finish line:** explain which design choices AnuLM makes, compare alternatives, and identify which claims have measurements rather than only arithmetic or citations.

## Path 3: evaluate a model for deployment

For engineers already comfortable with transformers:

1. Read Chapters 20, 23–25, and 27–28 for configuration, precision, evaluation, artifact handling, and decoding.
2. Read Chapters 30–33 if your use case needs adaptation or retrieval. Check the disclosed chat-split contamination before relying on its validation results.
3. Read Chapters 38–40 and the [Launch checklist](launch-checklist.md). Separate source behaviour, proposed extensions, and measured workload results.
4. Complete the operational checks for your actual use case before exposing the service to users.

**Finish line:** produce evidence for a deployment decision. Reading the chapters or passing tiny-model tests alone does not establish service readiness.

## A repeatable way to study a chapter

Read the entry card's goal and prerequisites. Try its smallest example. Predict an output before running it. Compare what happened with the explanation, then follow the source excerpt. Finish one exercise without looking at the answer.

The commands use three different kinds of evidence:

| Kind | What you need | What success establishes |
|---|---|---|
| Small CPU example | Public companion source and the named packages | A mechanism works on the example's inputs. |
| Released-model measurement | Exact checkpoint, tokenizer, dataset, precision, and seed | A result for the specified model and sample. |
| Historical audit or course lab | The cited revision and course files, some unpublished | Consistency with that historical source. Unavailable material must be reported as unavailable. |

Within a chapter, **By hand** teaches the mechanism, **In the code** locates it, and **At real scale** asks what it does on a trained model. Read them as separate steps. Use the glossary in Appendix D when a symbol or term interrupts your reading.
