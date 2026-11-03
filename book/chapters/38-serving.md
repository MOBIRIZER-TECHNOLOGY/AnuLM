---
title: "38 · Serving: engine, API and streaming"
part: "IX · To production"
accent: teal
pinned: "@084cbdd"
---

# 38  Serving: engine, API and streaming

A checkpoint becomes a service in two files. `anulm/inference/serve.py` is a small web server
built only on Python's standard library. `anulm/inference/api.py` speaks OpenAI's HTTP
protocol, so existing clients work unchanged, and it streams the chat model's
replies token by token. Both are thin: they load the model once, hold one lock around the
device, and call Chapter 27's `generate`. This chapter reads both, and shows
what streaming has to get right that a batch evaluation never sees: half a
character, half a stop sequence, and a client that hangs up. It ends with the
end of a streamed reply, where the code once lost text.

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

> **In the repo** · `anulm/inference/serve.py` · `Engine.__init__` · lines 53–63 · @084cbdd

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

> **In the repo** · `anulm/inference/serve.py` · `Engine.__init__` · lines 105–107 · @084cbdd

```python
        self.lock = threading.Lock()
        # Warm up: the first CUDA call and the grouped-GEMM weight stack are slow.
        self.generate("def f(" if self.coder else "भारत", 4, 0.8, 50, 0)
```

One lock per engine, and a four-token generation at start-up so the first real
request does not pay for CUDA initialisation.

`IN THE REPO`

## 38.2  One request

> **In the repo** · `anulm/inference/serve.py` · `Engine.generate` · lines 147–161 · @084cbdd

```python
        ids = self.encode(prompt)[-(self.cfg.block_size - 1):]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        with self.lock:
            torch.manual_seed(seed)
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
                out = self.model.generate(x, max_tokens, temperature=temperature,   # 0 = exact greedy
                                          top_k=top_k or None, eos_id=self.eos_id,
                                          repetition_penalty=repetition_penalty)
            dt = time.time() - t0
        new = out[0, len(ids):].tolist()
        stopped = self.eos_id is not None and self.eos_id in new
        # Decode only what came before EOS: a HF tokenizer would otherwise
        # write the control token ("<|endoftext|>") into the reply.
        text = self.decode(new[:new.index(self.eos_id)] if stopped else new)
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
5. **Temperature as sent.** It is passed unclamped: `generate` treats
   `temperature <= 0` as exact greedy, an argmax with no random draw
   (Chapter 27). `top_k=0` becomes `None`, no top-k filter.
6. **Decode up to EOS.** Only the ids before EOS are decoded.

Before line 147, the prompt is wrapped in the template the checkpoint was
tuned with, when the mode asks for one (lines 141–146).

`IN THE REPO`

## 38.3  The standard-library server

> **In the repo** · `anulm/inference/serve.py` · `make_handler` · lines 198–218 · @084cbdd

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
serves it with `ThreadingHTTPServer` (line 242): one thread per connection.
The threads overlap on parsing and sending, but every generation waits for the
engine's lock, so the device runs one request at a time.

`IN THE REPO`

## 38.4  An OpenAI-compatible API

`anulm/inference/api.py` uses FastAPI and accepts OpenAI's request shapes:

> **In the repo** · `anulm/inference/api.py` · `ChatRequest` · lines 192–198 · @084cbdd

```python
    temperature: Optional[float] = Field(None, ge=0, le=2, description="Default 0.3. 0 = greedy.")
    top_p: Optional[float] = Field(None, description="Accepted for compatibility; AnuLM samples with top_k.")
    top_k: Optional[int] = Field(None, ge=0, description="AnuLM extension. Default 40; 0 turns it off.")
    repetition_penalty: Optional[float] = Field(None, description="AnuLM extension. Default 1.15.")
    stop: Union[str, list[str], None] = None
    seed: Optional[int] = None
    n: int = Field(1, description="Only 1 is supported.")
```

`top_p` is accepted so OpenAI clients that send it do not fail, and then never
used: nucleus sampling [R31] is `NOT IN THE REPO` (Chapter 28). `top_k` and
`repetition_penalty` are additions. The handler is single-turn:

> **In the repo** · `anulm/inference/api.py` · `chat_completions` · lines 518–523 · @084cbdd

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

> **In the repo** · `anulm/inference/api.py` · `GPU` · lines 264–265 · @084cbdd

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
returning True stops (lines 1162–1163). `Generation.run` passes one:

> **In the repo** · `anulm/inference/api.py` · `Generation.run` · lines 383–401 · @084cbdd

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
                self.eng.model.generate(x, self.max_tokens, temperature=self.temperature,  # 0 = exact greedy
                                        top_k=self.top_k or None, eos_id=eos,
                                        repetition_penalty=self.rep, on_token=on_token)
```

For each token: EOS ends the reply without being shown; anything else is
appended, and `_release` decides how much new text may go out. Generation stops
on a stop sequence or a cancellation. The release rule:

> **In the repo** · `anulm/inference/api.py` · `Generation._release` · lines 360–376 · @084cbdd

```python
    def _release(self, final: bool) -> str:
        # Whole characters only: a Devanagari letter split across tokens is held
        # back until complete, while a genuine U+FFFD in the text is kept.
        text = self.eng.decode_complete(self.new)
        cut = len(text)
        for s in self.stops:
            i = text.find(s, max(0, len(self.sent) - len(s)))
            if i >= 0:
                cut, self.finish, self.matched_stop = min(cut, i), "stop", True
        text = text[:cut]
        if not self.matched_stop and not final:
            # Hold back what could still turn out to be the start of a stop string.
            hold = max((len(s) - 1 for s in self.stops), default=0)
            text = text[:len(text) - hold] if hold else text
        delta, self.sent = text[len(self.sent):], text if len(text) > len(self.sent) else self.sent
        self.text = self.sent
        return delta
```

Step by step:

1. **Decode the complete characters so far.** A token can end inside a
   multi-byte character. `decode_complete` decodes only up to the last complete
   UTF-8 character and holds the unfinished bytes back; bytes that can never
   be valid, and a real U+FFFD in the text, still come out as `�`.
2. **Cut at a stop sequence.** The search starts a little before what was
   already sent, so a stop string split across two tokens is still found. A
   match sets `finish = "stop"` and `matched_stop = True`.
3. **Hold back.** While no stop string has matched and the call is not final,
   keep back the last `len(s) − 1` characters, which could be the start of a
   stop string.
4. **Send the difference.** `delta` is what is new since `self.sent`.

For a byte-level model the decoder is one call into the standard library:

> **In the repo** · `anulm/bpe.py` · `complete_utf8` · lines 51–61 · @084cbdd

```python
def complete_utf8(data: bytes) -> str:
    # ...
    import codecs
    return codecs.getincrementaldecoder("utf-8")(errors="replace").decode(data, final=False)
```

An incremental decoder called with `final=False` keeps an incomplete trailing
sequence in its buffer instead of replacing it. The engine picks the decoder:

> **In the repo** · `anulm/inference/serve.py` · `Engine.decode_complete` · lines 124–132 · @084cbdd

```python
    def decode_complete(self, ids) -> str:
        # ...
        if self.tok:
            fn = getattr(self.tok, "decode_complete", None)
            return fn(ids) if fn else self.tok.decode(ids)
        from anulm.bpe import complete_utf8
        return complete_utf8(bytes(i for i in ids if i < 256))
```

A tokenizer with its own `decode_complete` uses it; one without falls back to
plain `decode`, with no hold. A byte model drops ids 256 and up (BOS, EOS,
PAD) and calls `complete_utf8`. The book example feeds a Hindi word one byte
at a time; the `decode` column is the plain decoder, the `released` column is
`_release`:

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
character). Plain `decode` shows `�` for the half-written character;
`_release` sends nothing until the character is whole. In the second case
each character goes out one token late, the one-character hold, and nothing
from the stop string on is ever sent.

Only `anulm-chat`, and `anulm-answer` when it falls back to the chat model,
stream this way. `anulm-vision` and a found `anulm-answer` compute the whole
reply first and send it as one piece (`one_shot_reply`, lines 488–505).

![Figure 38.1 · Releasing text safely](../figures/fig38-1-utf8-release.svg)

*Figure 38.1 · Releasing text safely.* Bytes arrive one token at a time; text
leaves only when a character is complete. *verified*: `ch38_streaming.py`.

### The end of a reply

While the reply runs, up to `max(len(s)) − 1` characters sit in the hold. The
end of `run` must send them, unless a stop string is what ended the reply:

> **In the repo** · `anulm/inference/api.py` · `Generation.run` · lines 402–409 · @084cbdd

```python
        # Release the held-back tail unless a stop string ended the reply. Ending
        # on EOS also sets finish to "stop", which once skipped this and lost the
        # last characters of every streamed reply that used `stop`.
        if not self.matched_stop:
            delta = self._release(final=True)
            if delta and emit:
                emit(delta)
        return self.text
```

Both EOS and a stop string set `finish` to `"stop"`, what the client is told;
only a stop string sets `matched_stop` (False from line 358). After EOS or
`max_tokens`, the final `_release` skips the hold and the rest is emitted. It
still uses `decode_complete`, so a `max_tokens` cut inside a character drops
the unfinished bytes instead of ending in `�`.

**What went wrong.** Until November 2026 the flush ran only when
`finish != "stop"`, and `on_token` sets `finish = "stop"` on EOS without a
release, so a reply that ended on EOS with `stop` strings set lost its
held-back tail, in the stream and in the returned text. The fix this book first
suggested, calling `self._release(final=True)` on EOS, was incomplete:
`_release` only returns the delta, so `on_token` would also have had to emit
it. Keying the final release on the stop match rather than on `finish` fixes
it in one place, after the loop, where the delta is emitted. The old release also stripped trailing `�` with `rstrip`,
which deleted a genuine U+FFFD and was skipped in the final flush;
`decode_complete` replaced it.

One more edge: OpenAI's `stop` may be an empty string, and `"".find("")` is 0,
which would end every reply before its first character. The request's `stop`
goes through `stops_of` first:

> **In the repo** · `anulm/inference/api.py` · `stops_of` · lines 335–340 · @084cbdd

```python
def stops_of(stop) -> list[str]:
    # An empty string is no stop at all: "".find("") is 0, so treating it as
    # one ended every reply before its first character.
    if isinstance(stop, str):
        return [stop] if stop else []
    return [s for s in (stop or []) if s]
```

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
   stops ['###']              streamed 'Hello'  returned text 'Hello'  finish 'stop'
   stops ['\n\n', 'User:']    streamed 'Hello'  returned text 'Hello'  finish 'stop'
   max_tokens=5 over 'Hi न' (bytes 48 69 20 e0 a4): streamed ['H', 'i', ' ']
   a real U+FFFD (bytes ef bf bd) in the reply: returned text 'a�b'
   stops_of(''): []; stops_of(['', '###']): ['###']
```

*verified*: `ch38_streaming.py`, against `anulm/inference/api.py` at @084cbdd.
The same run on the code at @773b6a8 streamed `'Hel'` and `'He'` for the two
stop lists, and `['H', 'i', ' ', '�']` for the cut.

`IN THE REPO`

## 38.6  Streaming over HTTP, and hanging up

> **In the repo** · `anulm/inference/api.py` · `stream_generation` · lines 428–462 · @084cbdd

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
   Generation.run, cancelled after 3 deltas: 3 tokens generated of max 40
4. two requests sent at once: on the device 0.000-0.359 s and 0.360-0.564 s; overlap: False
```

*verified*: `ch38_streaming.py` on `tiny()` and a CPU. The random model emits
bytes that can never be valid UTF-8; they decode to `�` at once, so every token
is a delta and the stop comes on the third token. The times
in line 4 vary from run to run; that the spans never overlap does not.

![Figure 38.2 · The serving path](../figures/fig38-2-serving-path.svg)

*Figure 38.2 · The serving path.* Many connections, one lock, one sequence on
the device. Dashed: what `anulm/inference/api.py` does not have. *verified*: from `anulm/inference/api.py`
and `anulm/inference/serve.py` at @084cbdd.

`IN THE REPO` · `IMPLEMENTATION-DEPENDENT` (disconnect handling)

## 38.7  What this serving path is not

- **No batching across requests.** The lock serialises device work, so
  throughput is one sequence's speed. `docs/RESULTS.md` §39 records the chat
  model at about 30 tokens/s end to end through the API benchmark (*measured*).
- **No top-p, no multi-turn, no tool calls.** `top_p` and earlier turns are
  accepted and silently ignored; `tools` is refused with a 400 (line 515).
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
3. Why does `anulm/inference/serve.py` seed inside the lock and not before it?

**Answers.** 1. The last 1,023; `max_tokens` is clamped to
1,024 − 1,023 = 1 (`anulm/inference/api.py` line 352). 2. Two, from the longest stop string;
if the reply then ends with EOS, the final release in `run` sends them.
3. Another thread could reseed between the seeding and the generation, and
both replies would stop being reproducible.

*verified*: answers 1 and 2 from the code at @084cbdd; the EOS case by the
example's step 6.

---

## Key points

- `serve.Engine` loads the model once, picks the MoE path by device, sets
  `text_vocab`, and warms up with a four-token generation.
- Every generation runs under one lock, with one sequence, bf16 autocast on
  CUDA, and the prompt cut to `block_size − 1` from the left.
- `anulm/inference/api.py` speaks OpenAI's request shapes, reads only the last user message,
  and accepts `top_p` without using it.
- Streaming uses `generate`'s `on_token`: only complete characters
  (`decode_complete`) before any possible stop string are sent.
- At the end, the held-back tail is sent unless a stop string matched
  (`matched_stop`); an empty `stop` is no stop.
- There is no batching across requests: throughput is one sequence's speed.

> **In one breath.** One engine loads the model once and holds one lock;
> requests, over a standard-library server or OpenAI-shaped FastAPI routes,
> each run alone through `generate`, whose per-token callback streams complete
> characters, holds back what could start a stop string, and sends the held
> tail when EOS or the token limit ends the reply.

## References

Research references cited in this chapter (`RESEARCH_REFERENCES.md`), each
used only for the claim in its register row:

| ID | Used for | Status |
|---|---|---|
| [R31] | nucleus (top-p) sampling, which the API accepts and ignores | CHECKED |

Repository evidence, all at @084cbdd:
- `anulm/inference/serve.py`: `Engine` 50–171 (`__init__` 53–107, `decode_complete` 124–132,
  `generate` 134–171),
  `make_handler` 174–223, `main` 226–247.
- `anulm/inference/api.py`: `ChatRequest` 186–203, `GPU` 265, `stops_of` 335–340,
  `Generation` 343–409 (`_release` 360–376, `run` 378–409),
  `one_shot_reply` 488–505,
  `stream_generation` 423–464, `chat_completions` 508–586.
- `anulm/model.py`: `AnuLM.generate` 1103–1168 (`on_token` 1162–1163).
- `anulm/bpe.py`: `complete_utf8` 51–61.
- `docs/RESULTS.md` §30, §39.
- Book example: `book/code-examples/ch38_streaming.py`.
