# AnuLM API demo: sample data, requests and real responses

A runnable tour of every endpoint in [`api.py`](../../api.py), with the
responses the server actually returned. The full guide is
[`docs/API.md`](../../docs/API.md).

| file | what it is |
| --- | --- |
| [`TRANSCRIPT.md`](TRANSCRIPT.md) | **start here**: each request (curl + JSON) and the response it got |
| `requests/*.json` | 18 requests, one per file: chat, streaming, completion, lookup, Decide (banking, Hindi, zero-shot), vision, text to speech, speech to text (synthetic, real English, real Hindi), and two errors |
| `responses/*.json` | the recorded response to each request: status, seconds, body |
| `responses/eval_*.json` | the sample data scored through the API |
| `data/banking_queries.csv` | 16 banking messages with their BANKING77 intent |
| `data/hindi_queries.csv` | 8 Hindi assistant commands with their MASSIVE intent |
| `data/questions.txt` | 4 questions for `/v1/answer`, one unanswerable on purpose |
| `data/astronaut.jpg` | the vision sample: NASA portrait of Eileen Collins, public domain (via scikit-image's sample data), resized to 384 px |
| `data/speech_*.wav` | real human speech the recognisers never trained on: a held-out LibriSpeech train-clean-100 clip (374-180298-0046) and a FLEURS Hindi dev clip, both CC BY 4.0; reference transcripts in `data/speech_refs.json` |
| `run_demo.py` | sends every request, saves the responses, scores the data, writes `TRANSCRIPT.md` |
| `openai_client.py` | the same API through the official `openai` SDK and LangChain (output in `responses/openai_client.txt`) |

## Benchmark: 1,480 timed requests on held-out data

`benchmark.py` sends every request one at a time to the running API and
saves each one (request, response, HTTP status, client-side milliseconds,
score) to `benchmark/<test>.jsonl`, with totals in `benchmark/summary.json`.
Recorded 2026-10-18 on an RTX 5070 Ti; latency excludes each model's first
(cold) request, which is reported separately.

| test | data | n | score | p50 | p95 |
| --- | --- | ---: | ---: | ---: | ---: |
| banking intents (AnuLM + LoRA) | BANKING77 test | 300 | 86.0% | 58 ms | 82 ms |
| banking intents (ModernBERT reference) | same 300 | 300 | 91.7% | 19 ms | 28 ms |
| Hindi intents | MASSIVE hi-IN | 300 | 84.3% | 62 ms | 81 ms |
| zero-shot intents, all 77 labels | BANKING77 test | 50 | 38.0% | 52 ms | 67 ms |
| Wikipedia lookup | `data/rag_demo_questions.jsonl` | 20 | 30.0% found | 57 ms | 1.2 s |
| speech to text, English | LibriSpeech held-out | 100 | WER 7.0% | 1.5 s | 1.8 s |
| speech to text, Hindi | FLEURS dev | 100 | WER 37.4% | 0.95 s | 1.9 s |
| image captions | Flickr30k val | 50 | (read them) | 507 ms | 807 ms |
| image questions, free-form exact match | A-OKVQA val | 200 | 13.5% | 122 ms | 187 ms |
| chat, English + Hindi | `chat_heldout.jsonl` | 40 | 30 tokens/s | 1.7 s | 3.9 s |
| text to speech | 20 sentences | 20 | 0.05 × real time | 47 ms | 58 ms |

Speech recognition runs at about 0.11 × real time (a 12 s clip in 1.3 s).
WER uses `eval_speech.normalise`, which keeps Devanagari vowel signs; image
questions here are open-ended, so the score is far below the 43.5% four-way
multiple choice. Run the models one group at a time on a 16 GB card: with
every model loaded at once the card spills into shared memory and each
request slows down by an order of magnitude.

```bash
python examples/api_demo/benchmark.py --only decide_banking,decide_hindi
python examples/api_demo/benchmark.py --n-scale 0.1        # a 2-minute smoke test
```

## Run it

```bash
python api.py                                   # terminal 1, repo root: http://127.0.0.1:8000
python examples/api_demo/run_demo.py            # terminal 2: all 18 requests + the sample data
python examples/api_demo/openai_client.py       # the OpenAI SDK / LangChain version
```

`python api.py --device cpu` works too (the recording was made on the GPU,
with the speech and vision models retrained on base v2; first use of each
model takes a few seconds to load it). Other options:

```bash
python examples/api_demo/run_demo.py --only decide              # just the Decide requests
python examples/api_demo/run_demo.py --only vision --image me.jpg
python examples/api_demo/run_demo.py --base-url http://10.0.0.5:8000 --key "$ANULM_API_KEY"
```

To add a request, drop a file in `requests/`:

```json
{"title": "What it shows", "method": "POST", "path": "/v1/decide",
 "body": {"model": "anulm-decide-banking", "input": "my card hasn't arrived"}}
```

`"@data/file.jpg"` in an `image_url` is replaced with a base64 data URL; in a
`multipart` field it is uploaded as the file. Plain curl cannot do this, so
for curl send `data:image/jpeg;base64,...` yourself.

## What the recording shows

The **classifiers and lookup** are the strong part:

- `anulm-decide-banking`: 16 of 16 sample messages right, every one above the
  0.7 threshold. "hi, quick question" gets 0.18 and `send to human`, which is
  the point of calibrated confidence.
- `anulm-decide-hindi`: 8 of 8 Hindi commands right.
- `anulm-decide-general` (zero-shot): word the options as descriptions
  ("complaint about a delivery", not "complaint"). With bare one-word labels
  it called the late-delivery complaint an "order request". With descriptions
  it gets the complaint and the praise, and sends the shipping question,
  which it got wrong at 0.67, to a human.
- `/v1/answer`: "Shah Jahan" and "नई दिल्ली" with their Wikipedia sources.

The **generators are a 398M model**, and the recording keeps their real output:

- `anulm-chat` is fluent but loose: the tips are vague and the Hindi answer
  drifts off the question. Use `temperature` 0.7 with
  `repetition_penalty` 1.3 (an AnuLM extension). At the default 0.3 / 1.15
  it repeats one line over and over.
- `anulm-answer` always finds *some* span: "What is the capital of Japan?" came
  back "Japan", and "What did I eat for breakfast today?" got a confident
  "Breakfast cereal". Check `sources`.
- `anulm-vision` captions only when the message has an image and **no text**
  ("A woman in a red uniform is standing next to a black and white flag." --
  the suit is orange and the flag is the US one). Any text makes it answer a
  short question ("What color is the flag?" → "blue"). Captions use a
  repetition penalty of 1.2, which ends greedy loops like "curling curling
  curling".
- `anulm-asr` (WER 7.1% English, 35.0% Hindi on held-out sets):
  - real English: "i have travelled a good deal and seen much grand**d**er
    things", one slip in eleven words;
  - real Hindi: "अमेरिकी राष्ट्रपति **और की** राष्ट्रपति **डुए** बुश ने **कोसना**
    का स्वागत किया" for "...जॉर्ज डब्ल्यू बुश ने घोषणा का स्वागत किया": the
    sentence is there, the name and "घोषणा" are not;
  - the TTS round trip came back exactly ("my card was stolen this morning
    please block it"), and the SDK's "Please send me a new debit card." as
    "please send me a new devot heart". Synthetic Piper speech is out of
    domain; real speech is the fairer test.

Sampled text, ids and timings change from run to run; re-run `run_demo.py` to
refresh the recording.
