# AnuLM India research checkpoint

Status: local research artifact; **quality not approved**. This is the native
AnuLM decoder/projector run, with frozen pretrained Whisper and SigLIP encoders.
The decoder uses eight layers, hidden size 384, eight routed experts and a shared
16,384-token text tokenizer, with a 512-token context. Vocabulary growth adds the
existing multimodal/control IDs; this is not a 30-billion-parameter trained model.

The text stage ran 10,000 updates and the joint stage ran 60,000. The original
loss selector saved its selected artifact at step 1,750; the final step is
preserved separately. See [the measured assessment](reviews/INDIA_MODEL_ASSESSMENT.md)
for actual validation/test results, sample sizes and selection limitations.

Data: pinned Wikipedia snapshots for 13 text configurations (Simple English for
English), FLEURS speech, DOCCI general images and explicitly synthetic Hindi/
English price-card/QA examples. The tokenizer fits training text only. See
[language coverage](LANGUAGE_SUPPORT.md) and the pinned configuration/source cards.
This does not validate broad chat, factual QA, Indian-script OCR, noisy/dialectal
speech, code switching, reasoning, tools, multi-turn chat or speech generation.

Local exports use FP32 weights and bundle the pinned frozen encoder snapshots.
They remain local. Their weight-license field is `other`: no distribution
license for weights is granted by the book's text license. Source code remains
MIT, original book prose is CC BY 4.0, and data/encoder obligations remain separate.
This card does not clear historical checkpoint lineage or certify commercial use.
