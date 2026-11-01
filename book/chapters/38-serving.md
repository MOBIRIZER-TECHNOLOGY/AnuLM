---
title: "38 · Serving: engine, API and streaming"
part: "IX · To production"
accent: teal
pinned: "@773b6a8"
---

# 38  Serving: engine, API and streaming

A checkpoint becomes a service in two files. `serve.py` is a small web server
built only on Python's standard library. `api.py` speaks OpenAI's HTTP
protocol, so existing clients work unchanged, and it streams the chat model's
replies token by token. Both are thin: they load the model once, hold one lock around the
device, and call Chapter 27's `generate`. This chapter reads both, and shows
what streaming has to get right that a batch evaluation never sees: half a
character, half a stop sequence, and a client that hangs up. It also finds two
bugs at the end of a streamed reply.

Prerequisites: generation and the KV cache (Chapters 27–28), the grown
vocabulary (Chapter 35).

**New words for Part IX.**

| Term | Meaning |
|---|---|
| engine | the object that owns one loaded model and turns a prompt into a reply |
| endpoint | one URL path and HTTP method a server answers, such as `POST /v1/chat/completions` |
| lock | a flag only one thread can hold at a time; others wait until it is released |
| streaming | sending a reply in pieces as it is generated, instead of all at the end |
| SSE | server-sent events: an HTTP response that stays open and carries `data:` lines, one per event |
| throughput | work finished per second across all requests; latency is the wait for one |

---

## 38.1  The engine: load once, hold one lock

> **In the repo** · `serve.py` · `Engine.__init__` · lines 53–63 · @773b6a8

```python
    def __init__(self, ckpt: str, device: str):
        ck = load_checkpoint(ckpt, device)          # a .pt, or a folder from export_hf.py
        cfg = ck["cfg"]
        # grouped = one grouped GEMM per projection, bf16, CUDA only; sparse
        # is the portable path. A checkpoint trained with either decodes with
        # either, so pick by device rather than by what the checkpoint says.
        cfg = replace(cfg, moe_impl="grouped" if device.startswith("cuda") else "sparse")
        self.cfg, self.device = cfg, device
        self.model = AnuLM(cfg).to(device).eval()
        self.model.load_state_dict(ck["model"])
        self.model.text_vocab = ck.get("speech_text_vocab")      # set on the one joint model (omni_train.py)
```

`load_checkpoint` reads a `.pt`, an exported folder or a Hub id (Chapter 39).
The mixture-of-experts dispatch path is chosen by device, not by the
checkpoint: the grouped matrix multiply on a GPU, the portable sparse path on
a CPU (Chapter 18). `text_vocab` keeps a grown model writing text
(Chapter 35). The constructor ends:

> **In the repo** · `serve.py` · `Engine.__init__` · lines 105–107 · @773b6a8

```python
        self.lock = threading.Lock()
        # Warm up: the first CUDA call and the grouped-GEMM weight stack are slow.
        self.generate("def f(" if self.coder else "भारत", 4, 0.8, 50, 0)
```

One lock per engine, and a four-token generation at start-up so the first real
request does not pay for CUDA initialisation.

`IN THE REPO`

## 38.2  One request

> **In the repo** · `serve.py` · `Engine.generate` · lines 129–141 · @773b6a8

```python
        ids = self.encode(prompt)[-(self.cfg.block_size - 1):]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        with self.lock:
            torch.manual_seed(seed)
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
                out = self.model.generate(x, max_tokens, temperature=max(temperature, 1e-3),
                                          top_k=top_k or None, eos_id=self.eos_id,
                                          repetition_penalty=repetition_penalty)
            dt = time.time() - t0
        new = out[0, len(ids):].tolist()
        stopped = self.eos_id is not None and self.eos_id in new
        text = self.decode(new)
```

In order:

1. **Truncate from the left.** Keep the last `block_size − 1` prompt ids, so at
   least one new token fits. The released model keeps 1,023.
2. **One sequence.** `x` is (1, S): a request is never batched with another.
3. **Under the lock.** Seeding and generation happen while holding it, so two
   threads can neither interleave on the device nor reseed each other's
   sampling.
4. **bf16 autocast on CUDA.** Weights stay fp32; matrix multiplies run in bf16
   (Chapter 23). On a CPU autocast is disabled.
5. **Temperature floor.** `max(temperature, 1e-3)`: temperature 0 means
   nearly greedy, never a division by zero.

Before line 129, the prompt is wrapped in the template the checkpoint was
tuned with, when the mode asks for one (lines 123–128).

`IN THE REPO`

## 38.3  The standard-library server

> **In the repo** · `serve.py` · `make_handler` · lines 175–195 · @773b6a8

```python
        def do_POST(self):
            if self.path != "/generate":
                return self._json({"error": "not found"}, 404)
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
                prompt = str(req.get("prompt", "")).strip()
                if not prompt:
                    return self._json({"error": "empty prompt"}, 400)
                res = engine.generate(
                    prompt,
                    max_tokens=max(1, min(int(req.get("max_tokens", 150)), 400)),
                    temperature=float(req.get("temperature", 0.8)),
                    top_k=int(req.get("top_k", 50)),
                    seed=int(req.get("seed", 1337)),
                    mode=req.get("mode") if req.get("mode") in ("question", "translate") else "continue",
                    repetition_penalty=float(req.get("repetition_penalty", 1.0)),
                )
                self._json(res)
            except Exception as e:                       # report, don't die
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)
```

The handler reads a JSON body, clamps `max_tokens` to 1–400, fills defaults,
and turns any exception into a 500 reply instead of a dead server. `main`
serves it with `ThreadingHTTPServer` (line 217): one thread per connection.
The threads overlap on parsing and sending, but every generation waits for the
engine's lock, so the device runs one request at a time.

`IN THE REPO`

## 38.4  An OpenAI-compatible API

`api.py` uses FastAPI and accepts OpenAI's request shapes:

> **In the repo** · `api.py` · `ChatRequest` · lines 187–193 · @773b6a8

```python
    temperature: Optional[float] = Field(None, ge=0, le=2, description="Default 0.3. 0 = greedy.")
    top_p: Optional[float] = Field(None, description="Accepted for compatibility; AnuLM samples with top_k.")
    top_k: Optional[int] = Field(None, description="AnuLM extension. Default 40.")
    repetition_penalty: Optional[float] = Field(None, description="AnuLM extension. Default 1.15.")
    stop: Union[str, list[str], None] = None
    seed: Optional[int] = None
    n: int = Field(1, description="Only 1 is supported.")
```

`top_p` is accepted so OpenAI clients that send it do not fail, and then never
used: nucleus sampling [R31] is `NOT IN THE REPO` (Chapter 28). `top_k` and
`repetition_penalty` are additions. The handler is single-turn:

> **In the repo** · `api.py` · `chat_completions` · lines 503–508 · @773b6a8

```python
    users = [m for m in req.messages if m.role == "user"]
    if not users:
        raise APIError(400, "messages must contain a user message.", param="messages")
    last = users[-1]
    question, images = text_of(last.content), images_of(last.content)
    mid = "anulm-vision" if images else req.model
```

Only the last user message is read. Earlier turns and any system message are
dropped, because the chat model was tuned on single questions. An image part
routes the request to the vision model. All device work shares one lock,
declared once for the whole server:

> **In the repo** · `api.py` · `GPU` · lines 259–260 · @773b6a8

```python
models: A.Models | None = None
GPU = threading.Lock()                # one request on the device at a time
```

The book example confirms the first point, and the lock in 38.6:

```
5. ChatRequest(top_p=0.1).top_p = 0.1; Generation takes ['prompt', 'max_tokens', 'temperature', 'top_k', 'repetition_penalty', 'seed', 'stops']
```

*verified*: `ch38_streaming.py`.

`IN THE REPO` · `NOT IN THE REPO` (top-p)

## 38.5  Streaming: release only what is safe to show

`generate` takes an `on_token` callback, called with each new (B, 1) token;
returning True stops (lines 1110–1111). `Generation.run` passes one:

> **In the repo** · `api.py` · `Generation.run` · lines 371–389 · @773b6a8

```python
        def on_token(nxt):
            t = int(nxt[0, 0])
            if eos is not None and t == eos:
                self.finish = "stop"
                return True
            self.new.append(t)
            delta = self._release(final=False)
            if delta and emit:
                emit(delta)
            return self.finish == "stop" or self.cancelled

        with GPU:
            if self.seed is not None:
                torch.manual_seed(self.seed)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=self.eng.device.startswith("cuda")):
                self.eng.model.generate(x, self.max_tokens, temperature=max(self.temperature, 1e-3),
                                        top_k=self.top_k or None, eos_id=eos,
                                        repetition_penalty=self.rep, on_token=on_token)
```

For each token: EOS ends the reply without being shown; anything else is
appended, and `_release` decides how much new text may go out. Generation stops
on a stop sequence or a cancellation. The release rule:

> **In the repo** · `api.py` · `Generation._release` · lines 350–364 · @773b6a8

```python
    def _release(self, final: bool) -> str:
        text = self.eng.decode(self.new)
        cut = len(text)
        for s in self.stops:
            i = text.find(s, max(0, len(self.sent) - len(s)))
            if i >= 0:
                cut, self.finish = min(cut, i), "stop"
        text = text[:cut]
        if self.finish != "stop" and not final:
            hold = max((len(s) - 1 for s in self.stops), default=0)
            text = text[:len(text) - hold] if hold else text
            text = text.rstrip("�")
        delta, self.sent = text[len(self.sent):], text if len(text) > len(self.sent) else self.sent
        self.text = self.sent
        return delta
```

Step by step:

1. **Decode everything so far.** A token can end inside a multi-byte
   character, so the tail of `text` may be the replacement character `�`.
2. **Cut at a stop sequence.** The search starts a little before what was
   already sent, so a stop string split across two tokens is still found.
3. **Hold back.** While not final, keep the last `len(s) − 1` characters, which
   could be the start of a stop string, and strip trailing `�`.
4. **Send the difference.** `delta` is what is new since `self.sent`.

The book example feeds a Hindi word one byte at a time through
`serve.Engine`'s byte-level decoder:

```
1. 'नमस्ते': 6 characters, 18 UTF-8 bytes, one byte per token
   byte  0 0xe0  decode -> '�'     released ''
   byte  1 0xa4  decode -> '�'     released ''
   byte  2 0xa8  decode -> 'न'     released 'न'
   ...
   byte 17 0x87  decode -> 'े'     released 'े'

2. stop sequence '\n\n' over 'Hi.\n\nNext': released ['', 'H', 'i', '.', '', '', '', '', '']; finish 'stop'; text 'Hi.'
```

*verified*: `ch38_streaming.py` (rows 3–16 elided; every third byte completes a
character). In the second case each character goes out one token late, the
one-character hold, and nothing from the stop string on is ever sent.

Only `anulm-chat`, and `anulm-answer` when it falls back to the chat model,
stream this way. `anulm-vision` and a found `anulm-answer` compute the whole
reply first and send it as one piece (`one_shot_reply`, lines 473–490).

![Figure 38.1 · Releasing text safely](../figures/fig38-1-utf8-release.svg)

*Figure 38.1 · Releasing text safely.* Bytes arrive one token at a time; text
leaves only when a character is complete. *verified*: `ch38_streaming.py`.

### Two bugs at the end of a reply

The hold-back is undone only by the final flush, and `run` skips that flush
whenever `finish` is `"stop"`:

> **In the repo** · `api.py` · `Generation.run` · lines 390–394 · @773b6a8

```python
        if self.finish != "stop":
            delta = self._release(final=True)
            if delta and emit:
                emit(delta)
        return self.text
```

But `on_token` sets `finish = "stop"` on EOS without calling `_release`. So when
a reply ends with EOS and the client sent any `stop` strings, the last
`max(len(s)) − 1` characters are held and never sent, in the stream or in the
non-streamed text. The final flush has the opposite fault: `final=True` skips
the `rstrip`, so a reply cut by `max_tokens` inside a character ends in `�`.
The book example drives `Generation.run` with a stand-in model that emits
fixed bytes:

```python
    for stops in ([], ["###"], ["\n\n", "User:"]):
        eng = byte_engine(Scripted(list(b"Hello") + [eos]))
        eng.eos_id = eos
        g = generation(eng, prompt="hi", stops=stops)
        streamed = []
        text = g.run(emit=streamed.append)
```

```
6. a model that writes 'Hello' then EOS, through Generation.run:
   stops []                   streamed 'Hello'  returned text 'Hello'  finish 'stop'
   stops ['###']              streamed 'Hel'    returned text 'Hel'    finish 'stop'
   stops ['\n\n', 'User:']    streamed 'He'     returned text 'He'     finish 'stop'
   max_tokens=5 over 'Hi न' (bytes 48 69 20 e0 a4): streamed ['H', 'i', ' ', '�']
```

*verified*: `ch38_streaming.py`, against `api.py` at @773b6a8. Calling
`self._release(final=True)` on EOS is not enough for streaming: it updates
`self.sent` and returns a delta, but does not send that delta. With that change
alone the returned text becomes `Hello` while the client still receives `Hel`.
The EOS branch must also emit the returned delta:

```python
# EXTENSION: the EOS branch inside on_token, not the pinned implementation.
if eos is not None and t == eos:
    delta = self._release(final=True)
    if delta and emit:
        emit(delta)
    self.finish = "stop"
    return True
```

Keep EOS distinct from a matched stop string: EOS must release a safe held
suffix, while an actual matched stop string must remain hidden. Test both the
joined streamed pieces and the returned text. Token-limit handling needs a
separate repair: use incremental UTF-8 decoding for the byte tokenizer so an
incomplete final character is held rather than replaced; stripping every
trailing `�` would also erase a genuine replacement character in the text.
Include complete Hindi text, incomplete UTF-8, stop strings split across
tokens, and legitimate `�` text in the checks. These repairs are extensions;
the pinned repository still has both faults.

For an executable byte-mode repair reference, run
`python book/code-examples/ch38_streaming_repair.py`. It needs no model or
third-party package and asserts both streamed and returned text in seven
cases, including a genuine `�`. It is a separate book extension, not a
patched API; another tokenizer must supply its actual token bytes.

`IN THE REPO` (both bugs are repository behaviour)

## 38.6  Streaming over HTTP, and hanging up

> **In the repo** · `api.py` · `stream_generation` · lines 413–447 · @773b6a8

```python
    q: queue.Queue = queue.Queue()

    def work():
        try:
            gen.run(emit=lambda d: q.put(("delta", d)))
            q.put(("done", None))
        except Exception as e:                                   # pragma: no cover
            q.put(("error", f"{type(e).__name__}: {e}"))

    async def events():
        threading.Thread(target=work, daemon=True).start()
        try:
            # ...
            while True:
                try:
                    kind, val = q.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.005)
                    continue
                # ...
            yield "data: [DONE]\n\n"
        finally:
            gen.cancelled = True
```

Generation runs in its own thread and puts each released piece on a queue. The
async generator `events` polls the queue every 5 ms and yields each piece as an
SSE `data:` line. The cancellation path depends on the web stack. By reading
Starlette 1.6.0's `StreamingResponse`, the version installed here with uvicorn
0.52.2: under ASGI HTTP spec versions below 2.4 it listens for the disconnect
and cancels the stream, so `events` is cancelled at its `await`, `finally` sets
`gen.cancelled`, and `on_token` returns True at the next token. Under spec 2.4
or later it waits for a send to fail instead, so the stop comes at the next
piece sent. The book example tests only the last step, setting `cancelled`
itself; the HTTP path was not tested end to end.

```
3. generate(max_new_tokens=50) with on_token returning True on the 5th token: 5 new tokens
   Generation.run, cancelled after 3 deltas: 8 tokens generated of max 40
4. two requests sent at once: on the device 0.000-0.321 s and 0.322-0.496 s; overlap: False
```

*verified*: `ch38_streaming.py` on `tiny()` and a CPU. The random model emits
bytes that are not valid UTF-8, so three deltas took eight tokens. The times
in line 4 vary from run to run; that the spans never overlap does not.

![Figure 38.2 · The serving path](../figures/fig38-2-serving-path.svg)

*Figure 38.2 · The serving path.* Many connections, one lock, one sequence on
the device. Dashed: what `api.py` does not have. *verified*: from `api.py`
and `serve.py` at @773b6a8.

`IN THE REPO` · `IMPLEMENTATION-DEPENDENT` (disconnect handling)

## 38.7  What this serving path is not

- **No batching across requests.** The lock serialises device work, so
  throughput is one sequence's speed. `docs/RESULTS.md` §39 records the chat
  model at about 30 tokens/s end to end through the API benchmark (*measured*).
- **No top-p, no multi-turn, no tool calls.** `top_p` and earlier turns are
  accepted and silently ignored; `tools` is refused with a 400 (line 500).
- **What it has been held to.** `docs/RESULTS.md` §30 drove `serve.Engine` with
  4,131 prompts: no errors and no empty replies, and translations within 0.3
  chrF of the evaluation script's scores (*measured*).

Chapter 40 sizes what a serving stack adds. `NOT IN THE REPO`

> **Build it.** The lab book's Chapter 15 (`course/ch15-shipping`) takes a
> model from checkpoint to release.

## 38.8  Exercises

1. A prompt is 1,500 tokens long. How many reach the released model, and how
   many new tokens can `Generation` then produce?
2. With stops `["###", "\n"]`, how many characters does `_release` hold back
   while streaming?
3. Why does `serve.py` seed inside the lock and not before it?

**Answers.** 1. The last 1,023; `max_tokens` is clamped to
1,024 − 1,023 = 1 (`api.py` line 343). 2. Two, from the longest stop string;
and because of 38.5's bug, if the reply then ends with EOS those two are never
sent.
3. Another thread could reseed between the seeding and the generation, and
both replies would stop being reproducible.

*verified*: answers 1 and 2 from the code at @773b6a8; the EOS case by the
example's step 6.

---

## Key points

- `serve.Engine` loads the model once, picks the MoE path by device, sets
  `text_vocab`, and warms up with a four-token generation.
- Every generation runs under one lock, with one sequence, bf16 autocast on
  CUDA, and the prompt cut to `block_size − 1` from the left.
- `api.py` speaks OpenAI's request shapes, reads only the last user message,
  and accepts `top_p` without using it.
- Streaming uses `generate`'s `on_token`: while a reply runs, only complete
  characters before any possible stop string are sent.
- Two repository bugs at the end: after EOS the held-back characters are lost
  when stops are set, and a `max_tokens` cut inside a character ends in `�`.
- There is no batching across requests: throughput is one sequence's speed.

> **In one breath.** One engine loads the model once and holds one lock;
> requests, over a standard-library server or OpenAI-shaped FastAPI routes,
> each run alone through `generate`, whose per-token callback streams complete
> characters and stops at stop strings or a cancelled stream, though the end
> of a reply can lose held characters or show a broken one.

## References

Research references cited in this chapter (`docs/reference/research-references.md`), each
used only for the claim in its register row:

| ID | Used for | Status |
|---|---|---|
| [R31] | nucleus (top-p) sampling, which the API accepts and ignores | CHECKED |

Repository evidence, all at @773b6a8:
- `serve.py`: `Engine` 50–151 (`__init__` 53–107, `generate` 121–151),
  `make_handler` 154–200, `main` 203–222.
- `api.py`: `ChatRequest` 181–198, `GPU` 260, `Generation` 334–394,
  `one_shot_reply` 473–490,
  `stream_generation` 408–449, `chat_completions` 493–571.
- `model.py`: `AnuLM.generate` 1058–1116 (`on_token` 1110–1111).
- `docs/RESULTS.md` §30, §39.
- Book example: `book/code-examples/ch38_streaming.py`.
