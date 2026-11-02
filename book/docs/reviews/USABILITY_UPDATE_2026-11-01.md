# Usability revision: 1 November 2026

## Result

The local book is now a complete technical preview with an explicit starting
point and reading routes. Both HTML editions contain all 41 chapters, all six
appendices, three reader guides, and 41 chapter entry cards. The rebuilt PDF
contains 748 pages at 7 by 9.25 inches. This is not a declaration of final
publication or service readiness.

## Changes made

- Added `docs/reader/start-here.md`: public companion revision, folder layout, environment
  setup, a first CPU model, expected behaviour, artifact requirements and
  troubleshooting.
- Added `docs/reader/learning-paths.md`: a dense-model-first route, an advanced architecture
  route, and a deployment-evaluation route. Chapter numbers remain stable.
- Added `docs/reader/launch-checklist.md`: book, data, model, request correctness and
  operational acceptance gates, with explicit pass/fail/not-run reporting.
- Added `tools/first_prediction.py`: two dense layers trained on repeated toy
  text, no external corpus or checkpoint, with an explicit companion path.
- Added goals, prerequisites, optional-path labels and example requirements for
  every chapter through `config/chapter-checkpoints.json` and the renderer.
- Added first-reading guidance to Chapters 1, 19 and 20; marked the chat split
  as historical faulty behaviour in Chapter 30.
- Made the trained-checkpoint portion of `attention_by_hand.py` opt-in with
  `--released`, with a clear missing-file message.
- Corrected Chapter 38's incomplete EOS repair advice. Added the separate
  byte-mode streaming repair reference, with seven checked cases. The actual
  historical API source remains unchanged.
- Reworded Chapter 40's concurrency calculation as a weights-and-cache upper
  bound, including exclusions and the need for workload measurements.
- Cross-referenced Chapter 23's routing-precision measurement from Chapter 20.
- Replaced the partial-edition process with a declared book manifest and
  complete default builds. Invalid or missing chapter selections fail clearly.
- Fixed SVG identifier collisions, local reader-guide anchors, UTF-8 document
  metadata, print page numbering, long-URL/table print overflow, and phone-width
  text overflow. The PDF no longer depends on remote font downloads.
- Added `tools/verify_preview.py` for structural edition checks, and an explicit
  CPU example manifest. `ch41_audit.py --examples-only` runs that smoke set
  without requiring historical private Git objects. It reports exclusions
  rather than treating them as passes.
- Refreshed the README and reproduction appendix. Preserved earlier README
  stage notes separately and marked the original readiness report as historical.

## Verification completed

| Check | Result |
|---|---|
| First-model CPU run | Passed: 35,200 parameters; loss 5.6009 to 0.0802 after 80 toy steps. Python 3.14.5, PyTorch 2.14.0+cu130, CPU. This checks memorisation of the toy text, not task quality. |
| Curated CPU examples | 8 passed, 0 failed, 58 excluded. Attention, shape tracing, softmax, attention from scratch, causal masks, the repair reference, top-p, and the forward trace. |
| Streaming repair reference | 7 cases passed, including held suffix on EOS, unequal stop lengths, split stop strings, Hindi, incomplete UTF-8, genuine replacement-character text, and a short reply with a long stop string. Both streamed and returned text are asserted. |
| Script syntax | All 67 example scripts and the nine tools parsed successfully. |
| Public excerpt comparison | No quoted-line mismatches in available public source; three Chapter 34 course excerpts remain unavailable. This does not verify the unpublished revision itself. |
| HTML structure | Both editions: 41 chapters, 6 appendices, 41 entry cards; expected order, unique IDs, internal anchors and missing-figure checks passed. |
| PDF content | 748 pages, 41 chapter entry cards, reader guides and back matter present; no empty text pages. |
| Layout | Visually inspected 12 representative final PDF pages, including cover, front matter, setup, chapter entry cards, serving/export/production chapters and back matter. This is sampling, not a visual inspection of every page. |
| Responsive layout | Body fits a 390-pixel phone viewport and 1,280-pixel desktop viewport. Print content fits its 547-pixel CSS content width. Wide figures and code blocks can scroll inside their own screen containers. |

The structural checks do not establish technical accuracy of every claim,
successful installation on every platform, or production-server readiness.
The CPU checks used the review's isolated public-source copy. Logs and visual
samples are in `tmp/review/`; they are not required reader inputs.

## Open launch gates

1. Publish the historical pin/course material, or replace every dependency on
   it. The public core path is documented, but exact historical audits and
   three course excerpts are still unavailable publicly.
2. Finish review of Chapters 27–29 and 35–40. These records remain drafted;
   this usability revision does not relabel them as fully reviewed.
3. Have a human expert review the technical work and have new target readers
   complete the first-run path without author assistance.
4. Correct the actual companion API's streaming faults and accepted-but-ignored
   options; test the supported request contract and live HTTP behaviour.
5. Rebuild the chat split with deduplication/grouping, rerun affected training
   and evaluation, and update the affected claims. No replacement scores were
   invented in this revision.
6. Complete workload-specific deployment, overload, monitoring, security and
   recovery checks. The new launch worksheet specifies the evidence to collect;
   it does not supply missing operational results.
7. Settle the book text's licence and publication identity. The preview label
   remains deliberate.

## Suggested reader trial

Give a Python programmer who has not studied transformers the book and a fresh
companion checkout. Ask them to install the environment, run the first model,
explain why its loss falls, follow the dense-model reading route, and complete
one exercise. Record their operating system/runtime, every interruption, time
to first successful run, and any instruction they could not follow. Use the
existing `REVIEWER_GUIDE.md` for the expert and target-reader feedback record.
