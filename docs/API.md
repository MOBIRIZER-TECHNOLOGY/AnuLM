# AnuLM API

Every model in the AnuLM demo behind one HTTP server that speaks **OpenAI's
protocol**. The official `openai` client, LangChain's `ChatOpenAI` and
`OpenAI`, and anything else built for OpenAI work by changing the base URL.
Two endpoints with no OpenAI equivalent add typed decisions (`/v1/decide`)
and Wikipedia lookup with sources (`/v1/answer`).

```bash
python api.py                        # GPU if there is one; http://127.0.0.1:8000
python api.py --device cpu --port 8000 --preload
```

* Interactive reference, every endpoint runnable in the browser: **`/docs`**
  (also `/redoc`, and the schema at `/openapi.json`).
* Health: `GET /health` (no key needed) lists the device and loaded models.
* End-to-end check of everything below: `python experiments/api_check.py`
  (51 checks, all passing on 2026-10-02).

Models load on first use and stay loaded. Requests are served one at a time:
the models share one GPU, and the chat model's weights are shared with the
Decide adapters.

## Authentication

Unset by default (any key is accepted, so OpenAI clients that insist on a key
still work). To require one:

```bash
ANULM_API_KEY=your-secret python api.py --host 0.0.0.0
```

Clients then send `Authorization: Bearer your-secret`; the OpenAI client does
this from `api_key=`. A missing or wrong key returns 401 `invalid_api_key`.
Bind to `0.0.0.0` only with a key set.

## Models

`GET /v1/models` returns each model with `from_scratch` and a description.
`from_scratch` is true only when every learned part is this project's own.

| model | endpoint | what it is | from scratch |
| --- | --- | --- | --- |
| `anulm-chat` | chat, completions | the chat model: 398M-total / 174M-active MoE pretrained here, instruction-tuned on 23k pairs (22% Hindi) | yes |
| `anulm-answer` | chat, `/v1/answer` | Wikipedia lookup: multilingual-e5-small retrieval (borrowed) + AnuLM's own reader | no (retriever) |
| `anulm-vision` | chat | captions and image questions: AnuLM decoder on a frozen SigLIP ViT-B/16 (Google) | no (image encoder) |
| `anulm-asr` | audio/transcriptions | English and Hindi speech: AnuLM decoder on a frozen Whisper-small encoder (OpenAI) | no (audio encoder) |
| `system-tts` | audio/speech | Piper voices (or Windows' own) | no |
| `anulm-decide-banking` | `/v1/decide` | 77 BANKING77 intents: the chat model + a 6.6 MB LoRA adapter | yes |
| `anulm-decide-hindi` | `/v1/decide` | 60 MASSIVE intents in Hindi: the chat model + a 6.6 MB adapter | yes |
| `anulm-decide-general` | `/v1/decide` | any option list, zero-shot (a separate 398M model) | yes |
| `modernbert-decide-banking` | `/v1/decide` | reference: ModernBERT-base fine-tuned on BANKING77 | no |
| `modernbert-decide-hindi` | `/v1/decide` | reference: ModernBERT-base fine-tuned on MASSIVE Hindi | no |

Measured quality (docs/RESULTS.md sections 33-38):

| model | result |
| --- | --- |
| `anulm-decide-banking` | 86.4% on 3,080 test queries; at confidence >= 0.7 it answers 80% at 94.7% |
| `anulm-decide-hindi` | 82.6% on 2,033 Hindi test messages (ModernBERT: 67.5%) |
| `anulm-decide-general` | 43.2% on BANKING77 never seen in training; calibrated (ECE 3.7%) |
| `modernbert-decide-banking` | 90.8%, about 4x faster than AnuLM |
| `anulm-answer` | SQuAD F1 57.2 |
| `anulm-asr` | WER 8.2% English (LibriSpeech), 48.4% Hindi (FLEURS) |

## Chat: `POST /v1/chat/completions`

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")
r = client.chat.completions.create(
    model="anulm-chat",
    messages=[{"role": "user", "content": "स्वस्थ रहने के तीन उपाय बताइए।"}],
    max_tokens=120)
print(r.choices[0].message.content, r.usage)
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "anulm-chat", "messages": [{"role": "user", "content": "Who are you?"}]}'
```

```json
{"id": "chatcmpl-...", "object": "chat.completion", "created": 1759420000, "model": "anulm-chat",
 "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant",
   "content": "I am AnuLM, a small language model trained from scratch on one consumer GPU. I read Hindi, English and Python."}}],
 "usage": {"prompt_tokens": 8, "completion_tokens": 27, "total_tokens": 35}}
```

**Streaming.** `stream=True` sends server-sent events token by token, in
OpenAI's chunk format, ending with `data: [DONE]`. Add
`stream_options={"include_usage": true}` for a final usage chunk. Hindi is
never cut mid-character, and greedy streamed text is identical to the
non-streamed reply. A client that hangs up stops the generation at once.

```python
for chunk in client.chat.completions.create(model="anulm-chat", stream=True,
        messages=[{"role": "user", "content": "भारत के बारे में बताइए"}]):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

| parameter | default | notes |
| --- | --- | --- |
| `model` | `anulm-chat` | or `anulm-answer`, `anulm-vision` |
| `messages` | required | the **last user message** is answered (see Limits) |
| `max_tokens` / `max_completion_tokens` | 200 | capped by the 1,024-token context |
| `temperature` | 0.3 | 0 is greedy |
| `top_k` | 40 | AnuLM extension (OpenAI has no top_k) |
| `repetition_penalty` | 1.15 | AnuLM extension |
| `stop` | none | string or list; `finish_reason` is then `stop` |
| `seed` | none | same seed and inputs give the same text |
| `stream`, `stream_options` | false | as OpenAI |
| `top_p`, `frequency_penalty`, `presence_penalty`, `user` | | accepted and ignored |
| `n` | 1 | only 1 |
| `tools`, `response_format` other than text | | rejected with 400 |

`finish_reason` is `stop` (end of answer or a stop sequence) or `length`.

### Facts: `model="anulm-answer"`

Looks the answer up instead of answering from memory, and adds a top-level
`sources` list (Wikipedia article titles) to the response:

```python
r = client.chat.completions.create(model="anulm-answer",
        messages=[{"role": "user", "content": "भारत की राजधानी क्या है?"}])
r.choices[0].message.content      # 'नई दिल्ली'
r.model_extra["sources"]          # ['दिल्ली', 'भारत की राजधानियों की सूची', 'नई दिल्ली']
```

When the reader finds nothing it falls back to the chat model and the reply
starts with `(not found in Wikipedia; from memory)`.

### Images: `model="anulm-vision"`

OpenAI's image format, with the image as a **base64 data URL** (the server
does not fetch remote URLs). No text gives a caption; text asks a question.
An image part also selects `anulm-vision` when another model is named.

```python
import base64
url = "data:image/jpeg;base64," + base64.b64encode(open("dog.jpg", "rb").read()).decode()
r = client.chat.completions.create(model="anulm-vision", messages=[{"role": "user", "content": [
        {"type": "text", "text": "What animal is this?"},
        {"type": "image_url", "image_url": {"url": url}}]}])
r.choices[0].message.content      # 'dog'  (no text: 'A dog runs across the grass.')
```

English only. The answer is computed whole, so a streamed vision reply
arrives as a single chunk.

## Completions: `POST /v1/completions`

Raw continuation with the chat model, no prompt template (OpenAI's legacy
completions; LangChain's `OpenAI` class). Defaults: `max_tokens` 64,
`temperature` 0.7, no repetition penalty. Supports `stream`, `stop`, `seed`
and `echo`.

```python
client.completions.create(model="anulm-chat", prompt="The capital of France is", max_tokens=12)
```

## Speech to text: `POST /v1/audio/transcriptions`

```python
with open("question.wav", "rb") as f:
    r = client.audio.transcriptions.create(model="anulm-asr", file=f, language="hi")
r.text                            # 'भारत के राजधानी का है'
```

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions -F file=@question.wav -F language=en
```

`language` is `en` (default) or `hi`; the server does not detect it.
`response_format` is `json` (default), `text` or `verbose_json` (adds
`duration`; `segments` is empty). Files: WAV, FLAC or OGG.

## Text to speech: `POST /v1/audio/speech`

```python
r = client.audio.speech.create(model="system-tts", voice="alloy",
                               input="The capital of India is New Delhi.", response_format="wav")
open("answer.wav", "wb").write(r.read())
```

`response_format` is `wav` (default) or `pcm` (16-bit mono; the rate is in
the `X-Sample-Rate` header). `voice` is accepted and ignored. The language is
taken from the script of `input`. Hindi needs Piper to be allowed by Windows
Smart App Control; when it is not and no Windows Hindi voice is installed,
Hindi returns 422.

## Decisions: `POST /v1/decide`

Typed, calibrated decisions in one forward pass, like TypeSafe's Jev: no
text is generated, the answer can only be one of the options, and the
confidence is calibrated so that 0.9 means right about 90% of the time.

```bash
curl http://127.0.0.1:8000/v1/decide -H "Content-Type: application/json" \
  -d '{"model": "anulm-decide-banking", "input": "I still haven'\''t received my new card"}'
```

```json
{"object": "decision", "model": "anulm-decide-banking", "from_scratch": true, "seconds": 0.06,
 "results": [{"input": "I still haven't received my new card", "choice": "card_arrival",
   "probabilities": {"card_arrival": 0.9212, "get_physical_card": 0.0131, "card_delivery_estimate": 0.0094},
   "confidence": 0.9212, "action": "auto-route"}]}
```

| field | default | notes |
| --- | --- | --- |
| `model` | `anulm-decide-banking` | any `anulm-decide-*` or `modernbert-decide-*` |
| `input` | required | a message, or a list (one result each, in order) |
| `options` | the model's labels | a subset or another list; **required** for `anulm-decide-general`. Options outside a model's trained labels are zero-shot and much weaker |
| `threshold` | 0.7 | below it, `action` is `send to human` |
| `top` | 3 | how many probabilities to return |

`GET /v1/models/anulm-decide-banking` describes each model; the labels are
the BANKING77 categories and MASSIVE intents (`alarm_set`, `weather_query`,
...). Each decision is one forward pass: about 60 ms on the RTX 5070 Ti for
AnuLM, 15 ms for ModernBERT.

## Lookup: `POST /v1/answer`

```bash
curl http://127.0.0.1:8000/v1/answer -H "Content-Type: application/json" \
  -d '{"question": "Who built the Taj Mahal?"}'
```

```json
{"object": "answer", "question": "Who built the Taj Mahal?", "answer": "Shah Jahan", "found": true,
 "sources": ["Taj Mahal", "Taj Mahal", "Taj Mahal"], "from": "wikipedia", "seconds": 0.5}
```

`fallback` (default true) answers from the chat model's memory when the
reader finds nothing; `from` then says `memory (anulm-chat; unreliable)`.

## LangChain

Chat needs nothing AnuLM-specific:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="anulm-chat", base_url="http://127.0.0.1:8000/v1", api_key="none", max_tokens=120)
chain = ChatPromptTemplate.from_template("Explain {topic} in one sentence.") | llm | StrOutputParser()
chain.invoke({"topic": "gravity"})
for chunk in llm.stream("Tell me about the moon."):
    print(chunk.content, end="")

facts = ChatOpenAI(model="anulm-answer", base_url="http://127.0.0.1:8000/v1", api_key="none")
facts.invoke("Who built the Taj Mahal?").content          # 'Shah Jahan'
```

`langchain_anulm.py` adds `anulm_chat()` (that ChatOpenAI, preconfigured)
and **`AnuLMDecide`**, `/v1/decide` as a Runnable: `invoke` takes a message
and returns the decision; `batch` sends a list in one request. With
`RunnableBranch` it routes each message by its decision:

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_anulm import AnuLMDecide, anulm_chat

route = AnuLMDecide(model="anulm-decide-banking", threshold=0.7)
answer = anulm_chat(max_tokens=80)
support = (RunnableLambda(lambda m: {"input": m, "d": route.invoke(m)})
           | RunnableBranch(
               (lambda x: x["d"]["action"] == "send to human", RunnableLambda(lambda x: "escalate")),
               (lambda x: x["d"]["choice"] == "card_arrival", RunnableLambda(lambda x: "card team")),
               RunnableLambda(lambda x: answer.invoke(x["input"]).content)))
support.invoke("I still haven't received my new card")             # 'card team'
support.invoke("Someone is using my card right now, block it!")    # 'escalate' (0.50, unsure)
```

`ANULM_BASE_URL` and `ANULM_API_KEY` set the defaults for both helpers.

What does not work: LangChain agents and anything else that needs **tool
calling** or **structured output** (`bind_tools`, `with_structured_output`):
the chat model was not trained for them, and the server returns 400 rather
than pretending. For classification and routing use `AnuLMDecide`.

## Errors

OpenAI's shape, so OpenAI clients raise their usual exceptions
(`NotFoundError`, `BadRequestError`, `AuthenticationError`):

```json
{"error": {"message": "Tool calling is not supported: the model was not trained for it.",
           "type": "invalid_request_error", "param": "tools", "code": null}}
```

| status | when |
| --- | --- |
| 400 | unsupported parameter (`tools`, `n` > 1, JSON mode), empty or missing input, unreadable audio or image, remote image URL, too many options for the context |
| 401 | `ANULM_API_KEY` is set and the key is missing or wrong |
| 404 | unknown model, or a model whose checkpoint is not on this server |
| 422 | malformed request body; or no voice for the language (speech) |
| 500 | anything else, with the exception in `message` |

## Limits

* **Single-turn chat.** The model was tuned on one question and one answer.
  System messages and earlier turns are accepted, so OpenAI and LangChain
  conversations work, but only the last user message is answered.
* **Facts from memory are unreliable** ("The capital of India is Kolkata").
  Use `anulm-answer` or `/v1/answer` for facts. The reader nearly always
  finds some span, so a question Wikipedia cannot answer ("What did I eat for
  breakfast?") gets a confident wrong one ("Breakfast cereal").
* **No tool calling, JSON mode, embeddings or multiple choices (`n`).**
* **Context 1,024 tokens**; long prompts keep their last 1,023 tokens.
* **One request at a time.** Concurrent requests queue. On the RTX 5070 Ti
  the chat model wrote 27 tokens in 0.85 s (about 30 tokens/s) and a decision
  takes ~60 ms; on CPU, expect several times slower.
* `top_p` and the penalties other than `repetition_penalty` are ignored.
