# Validated multimodal demo


Completed-run update (2 November 2026): text and joint training, native evaluations and both FP32 export checks are finished. Quality is inadequate for production. See [current status](CURRENT_STATUS.md) and the detailed assessment linked there. This guide describes reproducible setup, not an instruction to restart the completed run.

Decide serving uses AnuLM backbones and adapters only. The live demo has no
comparison-model selector, the API does not advertise or accept ModernBERT
decision models, and the active benchmark runner excludes that comparison.
Historical experiment code and recorded comparisons remain reference material;
they are not required dependencies of AnuLM Decide. This choice does not remove
the separate pretrained image/audio encoders used by the multimodal path.

The default `python -m anulm.apps.omni` displays **recorded approved results**.
It does not run inference or offer uploads, microphone access, editable prompts,
model pickers, unvalidated examples, benchmark controls or automatic model downloads.
The original live research interface is available explicitly:

```bash
python -m anulm.apps.omni --experimental
```

That interface accepts new inputs and retains its earlier limitations; it is
not covered by approval of the curated examples. Other standalone research
apps and API endpoints are unchanged by this demo restriction.

## Why the catalogue is empty

The [dataset setup guide](DATA_SETUP.md) now provides small real image, speech
and decision-label subsets with checked sources. Downloading them does not
approve any demo result; model inference and reviewer approval are still needed.

Training has completed and local multimodal checkpoints now exist. Exploratory
trained-model validation shows poor speech and text-task quality; full records are
kept under `runs/india-v1/post-training/`. A checkpoint and passing mechanics
tests do not approve real capability. Existing image/audio samples and historical
metrics alone are insufficient. No example was marked passed merely to populate
the UI. The source decoder remains the project's original implementation.

## Approving a recorded example

Run the actual model with a small held-out input. Record the exact model,
tokenizer, pretrained encoder files and dataset source/license. Set the
acceptance criterion before inspecting outputs, and retain failures in the
evaluation report. A project reviewer must approve the result; hashes establish
file consistency, not independent trust. A curated demo is not a benchmark.

For each approved case, create a report with this shape. All artifact paths are
relative to the checkout; replace placeholders with real evidence. Never mark
a synthetic fixture as real model validation.

```json
{
  "case_id": "english-transcription-001",
  "task": "audio",
  "status": "passed",
  "approved": true,
  "reviewed_by": "REVIEWER",
  "validated_at": "VALIDATION DATE",
  "source_revision": "EXACT TESTED COMMIT",
  "criteria": "PREDECLARED ACCEPTANCE CRITERION AND MEASURED RESULT",
  "output": "ACTUAL RECORDED TRANSCRIPT",
  "dataset": {"source": "DATASET AND SPLIT", "license": "VERIFIED TERMS", "held_out": true},
  "input": {"path": "examples/approved/clip.wav", "sha256": "ACTUAL SHA256"},
  "artifacts": [
    {"role": "checkpoint", "path": "runs/tested.pt", "sha256": "ACTUAL SHA256"},
    {"role": "tokenizer", "path": "data/tested-tokenizer.json", "sha256": "ACTUAL SHA256"},
    {"role": "encoder", "path": "runs/encoder/model.bin", "sha256": "ACTUAL SHA256"}
  ]
}
```

Use `task: image` with a hashed image input and optional `question` for image
cases. Use `task: text` with a string input for text cases. Include every runtime
artifact, encoder configuration and preprocessing configuration in `artifacts`,
not just the representative files above. Text cases require checkpoint and
tokenizer roles; media cases also require an encoder role. All listed files
are checked. Large weights stay outside Git as before.

Then add a matching entry to `config/demo-validation.json`:

```json
{
  "version": 1,
  "cases": [{
    "id": "english-transcription-001",
    "report": {"path": "examples/approved/report.json", "sha256": "ACTUAL REPORT SHA256"}
  }]
}
```

Missing, changed, failed, pending, unapproved or incomplete records are excluded.
The catalogue is loaded when the page is built; restart the demo after changing
evidence. Recorded outputs are not assertions that a newly loaded model will
reproduce them, nor permission to train on held-out inputs. The optional
`--validation-manifest` selects another catalogue with the same checks.

## Tests and remaining multimodal work

```bash
python -m unittest tests.test_validated_demo tests.test_multimodal_contracts
```

The demo tests cover absent/malformed evidence, approval status, changed model
and input files, dependencies, path containment, default UI isolation and
explicit experimental opt-in. Multimodal contracts cover modality ranges,
audio-frame roundtrips, pooling, answer masks, shared-decoder gradients,
checkpoint preservation, cache provenance and grouped splits. They use
synthetic features and do not download pretrained encoders.

The joint trainer now groups input paths/prompts before splitting and rejects
overlapping explicit splits. Feature caches bind images and encoder state to
metadata. Audio longer than 30 seconds is explicitly rejected instead of
silently truncated. Existing cache files rebuild; previous validation numbers
must be rebaselined after split changes. Copies under different media paths
still need deduplication. Use one writer per cache.

Next, run real image/audio integration on small licensed held-out subsets;
verify transcript accuracy, caption/VQA relevance and language coverage; test
combined inputs and checkpoint resumption; record peak memory and latency.
Current architecture uses pretrained frozen Whisper/SigLIP encoders. A random
decoder initialization check is not a fully pretrained-free multimodal run.
