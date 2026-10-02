"""
AnuLM API: every model in the demo behind OpenAI-compatible endpoints.

    python api.py                          # http://127.0.0.1:8000
    python api.py --device cpu --port 8000 --host 0.0.0.0

Interactive reference at /docs (try every endpoint in the browser) and /redoc;
the written guide, with OpenAI SDK, curl and LangChain examples, is
docs/API.md. Set ANULM_API_KEY to require `Authorization: Bearer <key>`.

OpenAI-compatible (the official `openai` client and LangChain's `ChatOpenAI`
/ `OpenAI` work unchanged, pointed at base_url=http://host:port/v1):

    GET  /v1/models                    every model, and which parts are borrowed
    POST /v1/chat/completions          anulm-chat, anulm-answer, anulm-vision; stream=true
    POST /v1/completions               raw continuation with the chat model
    POST /v1/audio/transcriptions      anulm-asr (English, Hindi)
    POST /v1/audio/speech              system-tts (WAV)

AnuLM-specific (no OpenAI equivalent):

    POST /v1/decide                    typed, calibrated decisions (Jev-style)
    POST /v1/answer                    Wikipedia lookup with sources

Requests run one at a time on the device: the models share one GPU, and the
chat model's weights are shared with the Decide adapters.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import queue
import tempfile
import threading
import time
import uuid
import wave
from pathlib import Path
from typing import Any, Literal, Optional, Union

import torch
from fastapi import Depends, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

import app_omni as A

HERE = Path(__file__).parent
CREATED = int(time.time())

# ------------------------------------------------------------------ the catalogue
# "from_scratch" is true only when every learned part is this project's own.
MODELS: dict[str, dict] = {
    "anulm-chat": {
        "endpoints": ["/v1/chat/completions", "/v1/completions"],
        "from_scratch": True,
        "description": "The chat model: AnuLM (398M total / 174M active MoE) pretrained from scratch, "
                       "instruction-tuned on 23k human-written pairs (22% Hindi). Fluent and on topic; "
                       "facts from memory are unreliable -- use anulm-answer for facts. Single-turn: "
                       "it answers the last user message. Context 1,024 tokens.",
    },
    "anulm-answer": {
        "endpoints": ["/v1/chat/completions", "/v1/answer"],
        "from_scratch": False,
        "description": "Looks the answer up: retrieves Simple English + Hindi Wikipedia passages with "
                       "multilingual-e5-small (borrowed, MIT) and AnuLM's own pointer reader picks the "
                       "answer span (SQuAD F1 57.2). Falls back to anulm-chat, and says so, when nothing "
                       "is found. The reader nearly always finds some span: questions Wikipedia cannot "
                       "answer (\"what did I eat today?\") get a confident wrong one.",
    },
    "anulm-vision": {
        "endpoints": ["/v1/chat/completions"],
        "from_scratch": False,
        "description": "Captions an image, or answers a question about it. AnuLM language model trained "
                       "on Flickr30k / A-OKVQA, reading a frozen pretrained SigLIP ViT-B/16 image encoder "
                       "(Google, borrowed). English. Send the image as a base64 data URL.",
    },
    "anulm-asr": {
        "endpoints": ["/v1/audio/transcriptions"],
        "from_scratch": False,
        "description": "Speech recognition, English (WER 8.2% LibriSpeech) and Hindi (WER 48.4% FLEURS). "
                       "AnuLM decoder trained here, reading a frozen pretrained Whisper-small encoder "
                       "(OpenAI, borrowed).",
    },
    "system-tts": {
        "endpoints": ["/v1/audio/speech"],
        "from_scratch": False,
        "description": "Text to speech with Piper voices (English and Hindi; borrowed, not an AnuLM "
                       "model). When Windows Smart App Control blocks Piper it falls back to Windows' own "
                       "voices, which are English only unless a Hindi voice is installed; Hindi then "
                       "returns 422.",
    },
    "anulm-decide-banking": {
        "endpoints": ["/v1/decide"],
        "from_scratch": True, "task": A.TASK_BANK, "backbone": A.DECIDE_OWN,
        "description": "BANKING77 intents (77, English): the chat model plus a 6.6 MB LoRA adapter. "
                       "86.4% accurate; at confidence >= 0.7 it answers 80% of queries at 94.7%.",
    },
    "anulm-decide-hindi": {
        "endpoints": ["/v1/decide"],
        "from_scratch": True, "task": A.TASK_HI, "backbone": A.DECIDE_OWN,
        "description": "MASSIVE assistant intents (60) in Hindi: the chat model plus a 6.6 MB LoRA "
                       "adapter. 82.6% accurate (ModernBERT: 67.5%).",
    },
    "anulm-decide-general": {
        "endpoints": ["/v1/decide"],
        "from_scratch": True, "ckpt": "ckpt_decide_general.pt",
        "description": "Any option list, zero-shot: trained on CLINC, MASSIVE and DBpedia label sets. "
                       "Weak: 43.2% on BANKING77 never seen in training, but calibrated (ECE 3.7%) -- "
                       "trust only high-confidence answers. A separate 398M model.",
    },
    "modernbert-decide-banking": {
        "endpoints": ["/v1/decide"],
        "from_scratch": False, "task": A.TASK_BANK, "backbone": A.DECIDE_MB,
        "description": "Reference: ModernBERT-base (Answer.AI, Apache 2.0, borrowed) fine-tuned on "
                       "BANKING77. 90.8%, about 4x faster than AnuLM.",
    },
    "modernbert-decide-hindi": {
        "endpoints": ["/v1/decide"],
        "from_scratch": False, "task": A.TASK_HI, "backbone": A.DECIDE_MB,
        "description": "Reference: ModernBERT-base (borrowed) fine-tuned on MASSIVE Hindi. 67.5%: its "
                       "English tokenizer reads Hindi as byte fragments.",
    },
}
CHAT_MODELS = ("anulm-chat", "anulm-answer", "anulm-vision")


def available(mid: str) -> bool:
    m = MODELS[mid]
    if "ckpt" in m:
        return (HERE / m["ckpt"]).exists()
    if "task" in m:
        return (HERE / A.DECIDE_TASKS[m["task"]][m["backbone"]]).exists()
    if mid == "anulm-chat":
        return (HERE / A.BRAIN).exists()
    if mid == "anulm-answer":
        return (HERE / A.READER).exists()
    if mid == "anulm-vision":
        return all((HERE / A.CKPTS[k]).exists() for k in ("caption", "vqa"))
    if mid == "anulm-asr":
        return any((HERE / A.CKPTS[k]).exists() for k in ("asr_en", "asr_hi"))
    return True


# ------------------------------------------------------------------ errors and auth
class APIError(Exception):
    def __init__(self, status: int, message: str, type_: str = "invalid_request_error",
                 param: str | None = None, code: str | None = None):
        self.status, self.message, self.type, self.param, self.code = status, message, type_, param, code


def check_key(authorization: Optional[str] = Header(default=None)):
    key = os.environ.get("ANULM_API_KEY")
    if key and authorization != f"Bearer {key}":
        raise APIError(401, "Missing or wrong API key: send 'Authorization: Bearer <ANULM_API_KEY>'.",
                       "authentication_error", code="invalid_api_key")


# ------------------------------------------------------------------ request schemas
class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str = Field(..., examples=["user"])
    content: Union[str, list[dict[str, Any]], None] = Field(
        None, description="A string, or OpenAI content parts: {type: text, text} and "
                          "{type: image_url, image_url: {url: 'data:image/jpeg;base64,...'}}.")


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = Field("anulm-chat", description="anulm-chat, anulm-answer or anulm-vision.")
    messages: list[ChatMessage] = Field(..., examples=[[{"role": "user", "content": "भारत की राजधानी क्या है?"}]])
    max_tokens: Optional[int] = Field(None, description="Default 200; capped by the 1,024-token context.")
    max_completion_tokens: Optional[int] = Field(None, description="Same as max_tokens (newer OpenAI name).")
    temperature: Optional[float] = Field(None, ge=0, le=2, description="Default 0.3. 0 = greedy.")
    top_p: Optional[float] = Field(None, description="Accepted for compatibility; AnuLM samples with top_k.")
    top_k: Optional[int] = Field(None, description="AnuLM extension. Default 40.")
    repetition_penalty: Optional[float] = Field(None, description="AnuLM extension. Default 1.15.")
    stop: Union[str, list[str], None] = None
    seed: Optional[int] = None
    n: int = Field(1, description="Only 1 is supported.")
    stream: bool = False
    stream_options: Optional[dict[str, Any]] = None
    tools: Optional[list[dict[str, Any]]] = Field(None, description="Not supported: the model was not "
                                                                    "trained for tool calls.")
    response_format: Optional[dict[str, Any]] = Field(None, description="Only {type: text}.")


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = "anulm-chat"
    prompt: Union[str, list[str]] = Field(..., examples=["भारत एक"])
    max_tokens: Optional[int] = Field(None, description="Default 64.")
    temperature: Optional[float] = Field(None, ge=0, le=2)
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    stop: Union[str, list[str], None] = None
    seed: Optional[int] = None
    n: int = 1
    stream: bool = False
    stream_options: Optional[dict[str, Any]] = None
    echo: bool = False


class DecideRequest(BaseModel):
    model: str = Field("anulm-decide-banking", description="An anulm-decide-* or modernbert-decide-* model.")
    input: Union[str, list[str]] = Field(..., description="One message, or a list of messages.",
                                         examples=["I still haven't received my new card"])
    options: Optional[list[str]] = Field(
        None, description="The options to choose between. Default: the model's trained labels. "
                          "Required for anulm-decide-general. Options outside the trained labels are "
                          "zero-shot and much less accurate.")
    threshold: float = Field(0.7, ge=0, le=1, description="Below this confidence, action is 'send to human'.")
    top: int = Field(3, ge=1, le=20, description="How many options to return probabilities for.")


class AnswerRequest(BaseModel):
    question: str = Field(..., examples=["Who built the Taj Mahal?"])
    fallback: bool = Field(True, description="Answer from the chat model's memory when Wikipedia has nothing.")


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = "system-tts"
    input: str = Field(..., examples=["The capital of India is New Delhi."])
    voice: Optional[str] = Field(None, description="Accepted for compatibility; the system voice is used.")
    response_format: str = Field("wav", description="wav or pcm (16-bit mono).")


# ------------------------------------------------------------------ app
DESCRIPTION = """
Every model in the AnuLM demo behind **OpenAI-compatible** endpoints, so the official
`openai` client and LangChain's `ChatOpenAI` work unchanged with
`base_url="http://127.0.0.1:8000/v1"`. Two AnuLM-specific endpoints add typed
decisions (`/v1/decide`) and Wikipedia lookup with sources (`/v1/answer`).

`GET /v1/models` says, for every model, which parts are this project's own and which
are borrowed. The guide with worked examples is `docs/API.md` in the repository.

**Limits.** Context 1,024 tokens. The chat model is single-turn and has no tool
calling or JSON mode. Its facts from memory are unreliable. Requests are served one
at a time.
"""
app = FastAPI(title="AnuLM API", version="1.0", description=DESCRIPTION,
              openapi_tags=[{"name": "OpenAI-compatible"}, {"name": "AnuLM"}, {"name": "service"}])
models: A.Models | None = None
GPU = threading.Lock()                # one request on the device at a time


@app.exception_handler(APIError)
async def _api_error(request: Request, e: APIError):
    return JSONResponse(status_code=e.status, content={"error": {
        "message": e.message, "type": e.type, "param": e.param, "code": e.code}})


@app.exception_handler(Exception)
async def _any_error(request: Request, e: Exception):
    return JSONResponse(status_code=500, content={"error": {
        "message": f"{type(e).__name__}: {e}", "type": "server_error", "param": None, "code": None}})


def need(mid: str, allowed) -> dict:
    if mid not in MODELS or mid not in allowed:
        raise APIError(404, f"Model '{mid}' is not served by this endpoint; use one of: "
                            + ", ".join(m for m in allowed if available(m)), param="model",
                       code="model_not_found")
    if not available(mid):
        raise APIError(404, f"Model '{mid}' has no checkpoint on this server.", param="model",
                       code="model_not_found")
    return MODELS[mid]


def model_card(mid: str) -> dict:
    m = MODELS[mid]
    return {"id": mid, "object": "model", "created": CREATED, "owned_by": "anulm",
            "from_scratch": m["from_scratch"], "endpoints": m["endpoints"], "description": m["description"]}


@app.get("/health", tags=["service"])
def health():
    return {"status": "ok", "device": models.device if models else None,
            "loaded": sorted(models._m) if models else []}


@app.get("/v1/models", tags=["OpenAI-compatible"], dependencies=[Depends(check_key)])
def list_models():
    return {"object": "list", "data": [model_card(m) for m in MODELS if available(m)]}


@app.get("/v1/models/{model_id}", tags=["OpenAI-compatible"], dependencies=[Depends(check_key)])
def get_model(model_id: str):
    if model_id not in MODELS or not available(model_id):
        raise APIError(404, f"Model '{model_id}' not found.", code="model_not_found")
    return model_card(model_id)


# ------------------------------------------------------------------ generation
def text_of(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return " ".join(p.get("text", "") for p in content if p.get("type") == "text").strip()


def images_of(content) -> list[str]:
    if not isinstance(content, list):
        return []
    out = []
    for p in content:
        if p.get("type") == "image_url":
            u = p.get("image_url")
            out.append(u.get("url") if isinstance(u, dict) else u)
    return out


def stops_of(stop) -> list[str]:
    return [stop] if isinstance(stop, str) else [s for s in (stop or []) if s]


class Generation:
    """Token-by-token generation with the chat model. Text is released as it
    becomes safe to show: never half a Devanagari character, never the start
    of a stop sequence."""

    def __init__(self, prompt: str, max_tokens: int, temperature: float, top_k: int,
                 repetition_penalty: float, seed: int | None, stops: list[str]):
        self.eng = models.brain()
        self.ids = self.eng.encode(prompt)[-(self.eng.cfg.block_size - 1):]
        self.max_tokens = max(1, min(max_tokens, self.eng.cfg.block_size - len(self.ids)))
        self.temperature, self.top_k, self.rep = temperature, top_k, repetition_penalty
        self.seed, self.stops = seed, stops
        self.new: list[int] = []
        self.sent, self.text, self.finish = "", "", "length"
        self.cancelled = False

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

    def run(self, emit=None):
        """Generate; `emit(delta)` receives text as it is released."""
        eos = self.eng.eos_id
        x = torch.tensor([self.ids], dtype=torch.long, device=self.eng.device)

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
        if self.finish != "stop":
            delta = self._release(final=True)
            if delta and emit:
                emit(delta)
        return self.text


def chat_prompt(question: str) -> str:
    """The template the chat model was tuned on, by the question's language."""
    from make_qa import template_for
    eng = models.brain()
    return template_for(question.strip(), eng.qa_templates).format(q=question.strip())


def sse(obj) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def stream_generation(gen: Generation, chunk, final_chunk, usage_chunk=None):
    """SSE stream: generation runs in a thread, deltas pass through a queue.
    The event loop is async so that a client hanging up cancels it at once
    (CancelledError at the await), which stops the generation at the next
    token and frees the device for the next request."""
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
            for s in chunk(None):                                # role / first chunk
                yield s
            while True:
                try:
                    kind, val = q.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.005)
                    continue
                if kind == "delta":
                    for s in chunk(val):
                        yield s
                elif kind == "error":
                    yield sse({"error": {"message": val, "type": "server_error"}})
                    break
                else:
                    for s in final_chunk():
                        yield s
                    for s in (usage_chunk() if usage_chunk else ()):
                        yield s
                    break
            yield "data: [DONE]\n\n"
        finally:
            gen.cancelled = True

    return StreamingResponse(events(), media_type="text/event-stream")


def usage(prompt_tokens: int, completion_tokens: int) -> dict:
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens}


def save_image(url: str) -> str:
    if not url or not url.startswith("data:image/"):
        raise APIError(400, "Images must be base64 data URLs (data:image/jpeg;base64,...); this server "
                            "does not fetch remote URLs.", param="messages")
    try:
        raw = base64.b64decode(url.split(",", 1)[1])
        from PIL import Image
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise APIError(400, "Could not decode the image data URL.", param="messages")
    f = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(f, quality=92)
    f.close()
    return f.name


def one_shot_reply(req: ChatRequest, content: str, extra: dict | None = None):
    """A reply that is computed whole (vision, lookup), as a response or a stream."""
    rid, now = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())
    if req.stream:
        def events():
            base = {"id": rid, "object": "chat.completion.chunk", "created": now, "model": req.model}
            yield sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                                            "finish_reason": None}]})
            yield sse({**base, "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]})
            yield sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            if (req.stream_options or {}).get("include_usage"):
                yield sse({**base, "choices": [], "usage": usage(0, 0)})
            yield "data: [DONE]\n\n"
        return StreamingResponse(events(), media_type="text/event-stream")
    return {"id": rid, "object": "chat.completion", "created": now, "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": usage(0, 0), **(extra or {})}


@app.post("/v1/chat/completions", tags=["OpenAI-compatible"], dependencies=[Depends(check_key)])
def chat_completions(req: ChatRequest):
    """Chat with `anulm-chat`, look facts up with `anulm-answer`, or ask about an image
    with `anulm-vision` (an image part in the last user message also selects it)."""
    if req.n != 1:
        raise APIError(400, "Only n=1 is supported.", param="n")
    if req.tools:
        raise APIError(400, "Tool calling is not supported: the model was not trained for it.", param="tools")
    if req.response_format and req.response_format.get("type") not in (None, "text"):
        raise APIError(400, "Only response_format {type: text} is supported.", param="response_format")
    users = [m for m in req.messages if m.role == "user"]
    if not users:
        raise APIError(400, "messages must contain a user message.", param="messages")
    last = users[-1]
    question, images = text_of(last.content), images_of(last.content)
    mid = "anulm-vision" if images else req.model
    need(mid, CHAT_MODELS)

    if mid == "anulm-vision":
        if not images:
            raise APIError(400, "anulm-vision needs an image_url part in the last user message.",
                           param="messages")
        from vision_encoder import answer, caption
        path = save_image(images[0])
        try:
            with GPU:
                if question:
                    vl, tower, tok = models.eye("vqa")
                    out = answer(vl, tower, tok, path, A.as_question(question), device=models.device)
                else:
                    vl, tower, tok = models.eye("caption")
                    out = caption(vl, tower, tok, path, device=models.device)
        finally:
            os.unlink(path)
        req.model = mid
        return one_shot_reply(req, out or "")

    if not question.strip():
        raise APIError(400, "The last user message is empty.", param="messages")

    if mid == "anulm-answer":
        qa = models.reader()
        with GPU:
            out = qa.ask(question) if qa else {"answer": "", "sources": []}
        if out["answer"]:
            return one_shot_reply(req, out["answer"], {"sources": out["sources"]})
        mid = "anulm-chat"                                       # fall back, and say so below
        prefix = "(not found in Wikipedia; from memory) "
    else:
        prefix = ""

    gen = Generation(chat_prompt(question), req.max_completion_tokens or req.max_tokens or 200,
                     0.3 if req.temperature is None else req.temperature, req.top_k or 40,
                     1.15 if req.repetition_penalty is None else req.repetition_penalty,
                     req.seed, stops_of(req.stop))
    rid, now = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())
    base = {"id": rid, "object": "chat.completion.chunk", "created": now, "model": req.model}

    if req.stream:
        def chunk(delta):
            if delta is None:
                yield sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": prefix},
                                                "finish_reason": None}]})
            else:
                yield sse({**base, "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]})

        def final():
            yield sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": gen.finish}]})

        def use():
            if (req.stream_options or {}).get("include_usage"):
                yield sse({**base, "choices": [], "usage": usage(len(gen.ids), len(gen.new))})
        return stream_generation(gen, chunk, final, use)

    text = gen.run()
    return {"id": rid, "object": "chat.completion", "created": now, "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": prefix + text.strip()},
                         "finish_reason": gen.finish}],
            "usage": usage(len(gen.ids), len(gen.new))}


@app.post("/v1/completions", tags=["OpenAI-compatible"], dependencies=[Depends(check_key)])
def completions(req: CompletionRequest):
    """Raw continuation of `prompt` with the chat model, no template (legacy OpenAI
    completions; LangChain's `OpenAI` LLM class)."""
    need(req.model, ("anulm-chat",))
    if req.n != 1 or (isinstance(req.prompt, list) and len(req.prompt) != 1):
        raise APIError(400, "Only one prompt and n=1 are supported.", param="prompt")
    prompt = req.prompt[0] if isinstance(req.prompt, list) else req.prompt
    gen = Generation(prompt, req.max_tokens or 64, 0.7 if req.temperature is None else req.temperature,
                     req.top_k or 40, 1.0 if req.repetition_penalty is None else req.repetition_penalty,
                     req.seed, stops_of(req.stop))
    rid, now = f"cmpl-{uuid.uuid4().hex[:24]}", int(time.time())
    base = {"id": rid, "object": "text_completion", "created": now, "model": req.model}
    if req.stream:
        def chunk(delta):
            if delta is None:
                if req.echo:
                    yield sse({**base, "choices": [{"index": 0, "text": prompt, "finish_reason": None}]})
                return
            yield sse({**base, "choices": [{"index": 0, "text": delta, "finish_reason": None}]})

        def final():
            yield sse({**base, "choices": [{"index": 0, "text": "", "finish_reason": gen.finish}]})

        def use():
            if (req.stream_options or {}).get("include_usage"):
                yield sse({**base, "choices": [], "usage": usage(len(gen.ids), len(gen.new))})
        return stream_generation(gen, chunk, final, use)
    text = gen.run()
    return {"id": rid, "object": "text_completion", "created": now, "model": req.model,
            "choices": [{"index": 0, "text": (prompt if req.echo else "") + text, "logprobs": None,
                         "finish_reason": gen.finish}],
            "usage": usage(len(gen.ids), len(gen.new))}


# ------------------------------------------------------------------ AnuLM-specific
def decider_for(mid: str):
    m = MODELS[mid]
    if "ckpt" in m:
        from decide import Decide
        return models._get(f"decide:{mid}", lambda: Decide(str(HERE / m["ckpt"]), models.device))
    return models.decider(m["backbone"], m["task"])


@app.post("/v1/decide", tags=["AnuLM"], dependencies=[Depends(check_key)])
def decide(req: DecideRequest):
    """Typed, calibrated decisions in one forward pass (no text is generated): the
    chosen option, the top probabilities, the confidence, and whether to act on it
    (`auto-route`) or hand it to a person (`send to human`)."""
    m = need(req.model, [k for k in MODELS if "/v1/decide" in MODELS[k]["endpoints"]])
    if "ckpt" in m and not req.options:
        raise APIError(400, f"{req.model} needs an options list.", param="options")
    if req.options is not None and len(set(req.options)) < 2:
        raise APIError(400, "options needs at least two distinct entries.", param="options")
    inputs = [req.input] if isinstance(req.input, str) else req.input
    if not inputs or any(not s or not s.strip() for s in inputs):
        raise APIError(400, "input must be non-empty text.", param="input")
    d = decider_for(req.model)
    t0, results = time.time(), []
    with GPU:
        d.threshold = req.threshold                         # the decider is shared between requests
        for s in inputs:
            try:
                out = d(s.strip(), req.options, top=req.top)
            except AssertionError as e:                      # too many options for the context
                raise APIError(400, str(e), param="options")
            results.append({"input": s, **out})
    return {"object": "decision", "model": req.model, "from_scratch": m["from_scratch"],
            "results": results, "seconds": round(time.time() - t0, 3)}


@app.post("/v1/answer", tags=["AnuLM"], dependencies=[Depends(check_key)])
def answer_question(req: AnswerRequest):
    """Look the answer up in Simple English + Hindi Wikipedia; returns the answer
    span and the articles it came from."""
    need("anulm-answer", ("anulm-answer",))
    if not req.question.strip():
        raise APIError(400, "question is empty.", param="question")
    qa = models.reader()
    t0 = time.time()
    with GPU:
        out = qa.ask(req.question.strip())
    if out["answer"] or not req.fallback:
        return {"object": "answer", "question": req.question, "answer": out["answer"] or None,
                "found": bool(out["answer"]), "sources": out["sources"] if out["answer"] else [],
                "from": "wikipedia" if out["answer"] else None, "seconds": round(time.time() - t0, 3)}
    text = Generation(chat_prompt(req.question), 120, 0.3, 40, 1.15, None, []).run().strip()
    return {"object": "answer", "question": req.question, "answer": text, "found": False, "sources": [],
            "from": "memory (anulm-chat; unreliable)", "seconds": round(time.time() - t0, 3)}


@app.post("/v1/audio/transcriptions", tags=["OpenAI-compatible"], dependencies=[Depends(check_key)])
def transcriptions(file: UploadFile = File(..., description="WAV, FLAC or OGG audio."),
                   model: str = Form("anulm-asr"),
                   language: str = Form("en", description="en or hi."),
                   response_format: str = Form("json", description="json, text or verbose_json."),
                   prompt: Optional[str] = Form(None), temperature: Optional[float] = Form(None)):
    """Speech to text, English or Hindi (OpenAI `audio.transcriptions`)."""
    need(model, ("anulm-asr",))
    if language not in ("en", "hi"):
        raise APIError(400, "language must be 'en' or 'hi'.", param="language")
    if response_format not in ("json", "text", "verbose_json"):
        raise APIError(400, "response_format must be json, text or verbose_json.", param="response_format")
    if not (HERE / A.CKPTS["asr_" + language]).exists():
        raise APIError(404, f"No {language} speech model on this server.", param="language")
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.write(file.file.read())
    f.close()
    try:
        from audio_codec import read_wav
        try:
            audio = read_wav(f.name, 16000)
        except Exception:
            raise APIError(400, "Could not read the audio; send WAV, FLAC or OGG.", param="file")
        with GPU:
            h = models.ear(language).hear(f.name)
    finally:
        os.unlink(f.name)
    if response_format == "text":
        return Response(h["text"], media_type="text/plain; charset=utf-8")
    if response_format == "verbose_json":
        return {"task": "transcribe", "language": language, "duration": round(len(audio) / 16000, 2),
                "text": h["text"], "segments": []}
    return {"text": h["text"]}


@app.post("/v1/audio/speech", tags=["OpenAI-compatible"], dependencies=[Depends(check_key)],
          response_class=Response, responses={200: {"content": {"audio/wav": {}}}})
def speech(req: SpeechRequest):
    """Text to speech (OpenAI `audio.speech`), WAV or raw 16-bit PCM. Uses the system
    voice, not an AnuLM model."""
    need(req.model, ("system-tts",))
    if req.response_format not in ("wav", "pcm"):
        raise APIError(400, "response_format must be wav or pcm.", param="response_format")
    if not req.input.strip():
        raise APIError(400, "input is empty.", param="input")
    rate, samples = models.mouth().say(req.input)
    if not len(samples):
        raise APIError(422, "No voice is installed for this language on the server.", param="input")
    pcm = samples.astype("<i2").tobytes()
    if req.response_format == "pcm":
        return Response(pcm, media_type="audio/pcm", headers={"X-Sample-Rate": str(rate)})
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return Response(buf.getvalue(), media_type="audio/wav")


def main():
    global models
    p = argparse.ArgumentParser(description="AnuLM API (OpenAI-compatible).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--preload", action="store_true", help="load the chat model at start, not on first use")
    a = p.parse_args()
    os.chdir(HERE)                     # checkpoints use repo-relative paths
    models = A.Models(a.device, A.BRAIN)
    if a.preload:
        models.brain()
    import uvicorn
    print(f"AnuLM API on http://{a.host}:{a.port}  (docs: /docs)  device {a.device}"
          + ("  [API key required]" if os.environ.get("ANULM_API_KEY") else ""), flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
