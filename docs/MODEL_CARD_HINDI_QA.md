# Model card: AnuLM-Hindi-QA-400M (`ckpt_multi_qa.pt`)

A Hindi / English / Python question answerer in the encyclopaedia register:
the three-language AnuLM base (`AnuLM-Base-400M`, step 36,000) fine-tuned
on question → lead-paragraph pairs built from Wikipedia and from Python
docstrings. Full log and decoded samples: `docs/RESULTS.md` §23.

**Licence: CC BY-SA 4.0.** The pairs derive from Hindi and English
Wikipedia (CC BY-SA); the base model also saw C4 (ODC-BY) and GitHub Python
via codeparrot-clean (mixed licences). Not affiliated with Sarvam AI,
AI4Bharat, BharatGen or the Government of India.

## What it does, and does not

Ask `X क्या है?`, `X के बारे में बताइए।`, `What is X?` or `What does this
function do?` and it answers with one or two definition-style sentences.
It is **not a chatbot**: it has never seen a conversation, has no
identity, and treats "What is your name?" as a topic to define. It invents
details freely; check anything that matters.

Held-out answer loss 3.3150 at the selected step. Manual scoring in
`docs/RESULTS.md` §23: sensible, on-topic answers for well-known subjects
in all three languages, degrading quickly on rare ones.

## Data

| pairs | built by | from |
| --- | --- | --- |
| Hindi questions | `make_qa.py` | Hindi Wikipedia lead paragraphs (CC BY-SA) |
| English questions | `make_qa_en.py` | English Wikipedia lead paragraphs (CC BY-SA) |
| Python questions | `make_qa_py.py` | docstrings of functions in codeparrot-clean |

Prompt formats are stored in the checkpoint as `qa_templates`:
`प्रश्न: {q}` newline `उत्तर:` for Hindi and `Question: {q}` newline
`Answer:` for English and Python; loss on the answer tokens only.

## Architecture and tokenizer

Same network as every 400M AnuLM checkpoint: 20 layers, hidden 1,024,
GQA 16 query / 4 key-value heads, 24 routed experts of 192 with top-4
routing and aux-loss-free balancing, sliding window 256 on layers 0–9,
context 512, 32k vocabulary. 397.7M parameters, 173.5M active per token.
Tokenizer `multi32k` (`docs/RESULTS.md` §22).

## Use

`python serve.py --ckpt <this folder>` then open http://127.0.0.1:8000
and pick *answer a question*, or `python ask.py --ckpt <this folder>` on
the command line. Temperature 0.3 and repetition penalty 1.3 are the
settings the samples were produced with.
