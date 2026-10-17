# AnuLM API demo: sample data, requests and real responses

A runnable tour of every endpoint in [`api.py`](../../api.py), with the
responses the server actually returned. The full guide is
[`docs/API.md`](../../docs/API.md).

| file | what it is |
| --- | --- |
| [`TRANSCRIPT.md`](TRANSCRIPT.md) | **start here**: each request (curl + JSON) and the response it got |
| `requests/*.json` | 16 requests, one per file: chat, streaming, completion, lookup, Decide (banking, Hindi, zero-shot), vision, text to speech, speech to text, and two errors |
| `responses/*.json` | the recorded response to each request: status, seconds, body |
| `responses/eval_*.json` | the sample data scored through the API |
| `data/banking_queries.csv` | 16 banking messages with their BANKING77 intent |
| `data/hindi_queries.csv` | 8 Hindi assistant commands with their MASSIVE intent |
| `data/questions.txt` | 4 questions for `/v1/answer`, one unanswerable on purpose |
| `data/astronaut.jpg` | the vision sample: NASA portrait of Eileen Collins, public domain (via scikit-image's sample data), resized to 384 px |
| `run_demo.py` | sends every request, saves the responses, scores the data, writes `TRANSCRIPT.md` |
| `openai_client.py` | the same API through the official `openai` SDK and LangChain (output in `responses/openai_client.txt`) |

## Run it

```bash
python api.py                                   # terminal 1, repo root: http://127.0.0.1:8000
python examples/api_demo/run_demo.py            # terminal 2: all 16 requests + the sample data
python examples/api_demo/openai_client.py       # the OpenAI SDK / LangChain version
```

`python api.py --device cpu` works too (the recording was made on CPU; first
use of each model takes a few seconds to load it). Other options:

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
  which it got wrong at 0.66, to a human.
- `/v1/answer`: "Shah Jahan" and "नई दिल्ली" with their Wikipedia sources.

The **generators are a 398M model**, and the recording keeps their real output:

- `anulm-chat` is fluent but loose. Asked for three tips it lists six; the Hindi
  answer drifts off the question. Use `temperature` 0.7 with
  `repetition_penalty` 1.3 (an AnuLM extension). At the default 0.3 / 1.15
  it repeats one line over and over.
- `anulm-answer` always finds *some* span: "What is the capital of Japan?" came
  back "Japan", and "What did I eat for breakfast today?" got a confident
  "Breakfast cereal". Check `sources`.
- `anulm-vision` captions only when the message has an image and **no text**
  ("A woman in a orange jumpsuit is smiling and smiling."). Any text makes it
  answer a short question ("What color is the flag?" → "red").
- Speech round trip: the TTS audio is clean (peak-normalised, not clipped), but
  synthetic Piper speech is out of domain for the recogniser (WER 8.2% on
  LibriSpeech). The recording heard "My card was stolen this morning" as
  "my card was strollin this morning"; other runs got it exactly. The SDK run's
  "Please send me a new debit card." came back as nonsense. Expect better
  on recorded human speech.

Sampled text, ids and timings change from run to run; re-run `run_demo.py` to
refresh the recording.
