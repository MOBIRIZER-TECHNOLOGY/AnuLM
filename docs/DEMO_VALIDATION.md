# Showing reviewed results: `validated_demo`

```bash
python -m anulm validated_demo            # http://127.0.0.1:7864
python -m anulm validated_demo --check    # print which cases pass, and why the others do not
```

`app_omni` answers whatever a visitor types. That is the way to explore a
model, and the wrong way to show someone what it can be relied on to do. One
bad answer to an improvised question is what people remember.
`validated_demo` shows a fixed set of cases instead, and it runs no model:
each case is a recorded result that a person reviewed.

The catalogue, `examples/approved/catalogue.json`, ships empty, so the page
says that no case has passed yet. A checkpoint that loads, and tests that
pass, do not approve a capability. Recording real cases is a deliberate step.

## What a case must have

A case is shown only if every check passes. The checks fail closed: anything
missing, changed, or outside the repository excludes that case, and the page
lists the case with the reason.

- **A report file** whose SHA-256 matches the one in the catalogue, with:
  - `status: "passed"` and `approved: true`;
  - a non-empty `reviewed_by`, `validated_at`, `source_revision` (the exact
    commit that produced the output), `criteria` and `output`.
- **A held-out input.** `dataset.held_out` is `true`, and `dataset.source` and
  `dataset.license` are filled in. Never use a training example.
- **The files that produced it,** listed under `artifacts` with their hashes:
  - the `checkpoint` and the `tokenizer`;
  - for audio or image cases, also the `encoder` (Whisper, SigLIP).

  If one of these files is retrained or replaced, the case disappears until
  it is reviewed again.
- **The input itself:**
  - for a `text` case, the prompt as a string;
  - for an `audio` or `image` case, the file with its hash.

Hashes prove the files are the ones that were reviewed. They don't prove the
review was right, so approval stays a person's decision.

## Recording a case

1. Pick a held-out input and write the acceptance criterion **before** running
   the model, for example "the transcript has WER at most 10% against the
   reference" or "the answer names New Delhi". Keep failures: a case that
   fails its criterion stays in your evaluation notes with `status: "failed"`.
2. Run the exact checkpoint, and record its output verbatim.
3. Have a reviewer check the output against the criterion.
4. Write the report under `examples/approved/`:

```json
{
  "case_id": "hindi-capital-001",
  "task": "text",
  "status": "passed",
  "approved": true,
  "reviewed_by": "NAME",
  "validated_at": "2026-11-03",
  "source_revision": "COMMIT THE OUTPUT WAS PRODUCED AT",
  "criteria": "Names New Delhi as the capital. Result: it does.",
  "input": "भारत की राजधानी क्या है?",
  "output": "THE RECORDED OUTPUT, VERBATIM",
  "dataset": {"source": "data/chat_heldout.jsonl, row 17", "license": "CC BY-SA 3.0", "held_out": true},
  "artifacts": [
    {"role": "checkpoint", "path": "ckpt_omni.pt", "sha256": "..."},
    {"role": "tokenizer", "path": "data/multi32k.json", "sha256": "..."}
  ]
}
```

5. Add the case to the catalogue, with the report's own hash:

```json
{"version": 1, "cases": [
  {"id": "hindi-capital-001", "report": {"path": "examples/approved/hindi-capital-001.json", "sha256": "..."}}
]}
```

Compute each hash with `python -c "from anulm.apps.validated_demo import digest; print(digest(__import__('pathlib').Path('FILE')))"`,
or `sha256sum FILE`. Then run `python -m anulm validated_demo --check`.

An image case uses `"task": "image"`, an `input` of the form
`{"path": ..., "sha256": ...}`, an optional `question` (for VQA), and an
`encoder` artifact. An audio case is the same, with the recording as its input.

Large files (checkpoints, encoders) stay outside git as before. The catalogue
and the reports are small and belong in the repository.
