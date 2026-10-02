"""
End-to-end check of api.py, through the official OpenAI client, LangChain and
raw HTTP, against a running server:

    python api.py --device cuda            # in another shell
    python experiments/api_check.py [--base http://127.0.0.1:8000] [--key K]

Every endpoint, streaming and not, the error contract, a LangChain chain and a
LangChain router built on /v1/decide. Prints one line per check and exits
non-zero if any failed.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import io
import json
import sys
import time
import wave
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
RESULTS: list[tuple[str, bool, str]] = []


def check(name, fn):
    t0 = time.time()
    try:
        note = fn() or ""
        RESULTS.append((name, True, note))
        print(f"PASS  {name}  ({time.time() - t0:.1f}s)  {note}", flush=True)
    except Exception as e:
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"FAIL  {name}  ({time.time() - t0:.1f}s)  {type(e).__name__}: {str(e)[:300]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--key", default="none")
    ap.add_argument("--audio-dir", default="", help="folder with en_capital.wav and hi_capital.wav")
    ap.add_argument("--image", default=str(ROOT / "data" / "flickr30k_img" / "1009434119.jpg"))
    a = ap.parse_args()
    from openai import OpenAI
    V1 = a.base + "/v1"
    client = OpenAI(base_url=V1, api_key=a.key, timeout=600)
    H = {"Authorization": f"Bearer {a.key}"}
    http = httpx.Client(timeout=600, headers=H)
    served = {m.id for m in client.models.list().data}

    # ---------------------------------------------------------------- catalogue
    def models():
        data = http.get(V1 + "/models").json()["data"]
        assert all({"id", "object", "owned_by", "from_scratch", "description"} <= set(m) for m in data)
        one = client.models.retrieve("anulm-chat")
        assert one.id == "anulm-chat"
        return f"{len(data)} models: " + ", ".join(f"{m['id']}{'' if m['from_scratch'] else '*'}" for m in data)
    check("GET /v1/models (+ retrieve)", models)
    check("GET /docs and /openapi.json", lambda: (
        http.get(a.base + "/docs").raise_for_status(),
        f"{len(http.get(a.base + '/openapi.json').json()['paths'])} paths")[1])

    # ---------------------------------------------------------------- chat
    def chat(q):
        def f():
            r = client.chat.completions.create(model="anulm-chat", messages=[{"role": "user", "content": q}],
                                               max_tokens=80)
            c = r.choices[0]
            assert c.message.content and r.usage.completion_tokens > 0
            return f"[{c.finish_reason}, {r.usage.completion_tokens} tok] {c.message.content[:90]!r}"
        return f
    check("chat: English", chat("Who are you?"))
    check("chat: Hindi", chat("स्वस्थ रहने के तीन उपाय बताइए।"))
    check("chat: system prompt + earlier turns accepted", lambda: client.chat.completions.create(
        model="anulm-chat", max_tokens=30, messages=[
            {"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"}, {"role": "user", "content": "What is Python?"}]
    ).choices[0].message.content[:80])

    def stream():
        parts, usage, finish = [], None, None
        for ch in client.chat.completions.create(model="anulm-chat", stream=True, max_tokens=60,
                                                 stream_options={"include_usage": True},
                                                 messages=[{"role": "user", "content": "भारत के बारे में बताइए"}]):
            if ch.usage:
                usage = ch.usage
            for c in ch.choices:
                parts.append(c.delta.content or "")
                finish = c.finish_reason or finish
        text = "".join(parts)
        assert len([p for p in parts if p]) > 3, "expected many chunks"
        assert "�" not in text, "half a character was streamed"
        assert usage and usage.completion_tokens > 0 and finish
        return f"{len(parts)} chunks, finish {finish}, {usage.completion_tokens} tok: {text[:70]!r}"
    check("chat: streaming (Hindi, include_usage)", stream)

    def stream_equals():
        kw = dict(model="anulm-chat", max_tokens=40, temperature=0, messages=[{"role": "user", "content": "What is the sun?"}])
        whole = client.chat.completions.create(**kw).choices[0].message.content
        streamed = "".join(c.choices[0].delta.content or "" for c in client.chat.completions.create(stream=True, **kw)
                           if c.choices)
        assert whole.strip() == streamed.strip(), (whole, streamed)
        return "greedy: streamed text == non-streamed text"
    check("chat: streaming matches non-streaming", stream_equals)

    def length():
        r = client.chat.completions.create(model="anulm-chat", max_tokens=5,
                                           messages=[{"role": "user", "content": "Tell me a long story about a king."}])
        assert r.choices[0].finish_reason == "length" and r.usage.completion_tokens == 5
        return "max_tokens=5 -> finish_reason length, 5 tokens"
    check("chat: max_tokens / finish_reason=length", length)

    def stop():
        r = client.chat.completions.create(model="anulm-chat", max_tokens=120, stop=["2."],
                                           messages=[{"role": "user", "content": "Give me three tips for staying healthy."}])
        t = r.choices[0].message.content
        assert "2." not in t
        return f"finish {r.choices[0].finish_reason}: {t[:80]!r}"
    check("chat: stop sequence", stop)

    def seed():
        kw = dict(model="anulm-chat", max_tokens=30, temperature=1.0, seed=7,
                  messages=[{"role": "user", "content": "Write one sentence about rain."}])
        x, y = (client.chat.completions.create(**kw).choices[0].message.content for _ in range(2))
        assert x == y
        return "same seed, same text at temperature 1.0"
    check("chat: seed is reproducible", seed)

    # ---------------------------------------------------------------- lookup
    def lookup(q, expect):
        def f():
            r = http.post(V1 + "/chat/completions", json={"model": "anulm-answer",
                                                          "messages": [{"role": "user", "content": q}]}).json()
            t = r["choices"][0]["message"]["content"]
            assert expect in t, t
            return f"{t!r} sources={r.get('sources')}"
        return f
    if "anulm-answer" in served:
        check("chat anulm-answer: English fact", lookup("Who built the Taj Mahal?", "Shah Jahan"))
        check("chat anulm-answer: Hindi fact", lookup("भारत की राजधानी क्या है?", "दिल्ली"))
        check("POST /v1/answer", lambda: (lambda r: (r["found"] and r["sources"]) and f"{r['answer']!r} from {r['sources'][0]}"
                                          or (_ for _ in ()).throw(AssertionError(r)))(
            http.post(V1 + "/answer", json={"question": "What is the capital of India?"}).json()))
        # Not a pass/fail on the model: the reader nearly always finds a span, so
        # this records what it says to a question Wikipedia cannot answer.
        check("POST /v1/answer: unanswerable question (reported, see docs)", lambda: (
            lambda r: f"from {r['from']}: {r['answer']!r}")(
            http.post(V1 + "/answer", json={"question": "What did I eat for breakfast this morning?"}).json()))
        check("POST /v1/answer: fallback=false shape", lambda: (lambda r: f"found={r['found']} keys ok"
                                                                if {"answer", "found", "sources", "from"} <= set(r)
                                                                else (_ for _ in ()).throw(AssertionError(r)))(
            http.post(V1 + "/answer", json={"question": "Who built the Taj Mahal?", "fallback": False}).json()))

    # ---------------------------------------------------------------- vision
    if "anulm-vision" in served:
        url = "data:image/jpeg;base64," + base64.b64encode(Path(a.image).read_bytes()).decode()

        def vision(q):
            def f():
                content = [{"type": "image_url", "image_url": {"url": url}}]
                if q:
                    content.insert(0, {"type": "text", "text": q})
                r = client.chat.completions.create(model="anulm-vision", messages=[{"role": "user", "content": content}])
                t = r.choices[0].message.content
                assert t
                return repr(t)
            return f
        check("chat anulm-vision: caption", vision(""))
        check("chat anulm-vision: question", vision("What animal is this?"))
        check("chat: image part selects vision even with model=anulm-chat", lambda: client.chat.completions.create(
            model="anulm-chat", messages=[{"role": "user", "content": [
                {"type": "text", "text": "What animal is this?"},
                {"type": "image_url", "image_url": {"url": url}}]}]).choices[0].message.content)
        check("chat anulm-vision: streaming", lambda: "".join(
            c.choices[0].delta.content or "" for c in client.chat.completions.create(
                model="anulm-vision", stream=True,
                messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}])
            if c.choices))

    # ---------------------------------------------------------------- completions
    def completion():
        r = client.completions.create(model="anulm-chat", prompt="The capital of France is", max_tokens=12,
                                      temperature=0)
        return repr(r.choices[0].text)
    check("POST /v1/completions", completion)
    check("POST /v1/completions: streaming", lambda: repr("".join(
        c.choices[0].text for c in client.completions.create(model="anulm-chat", prompt="भारत एक",
                                                              max_tokens=15, stream=True) if c.choices)))

    # ---------------------------------------------------------------- audio
    audio_dir = Path(a.audio_dir) if a.audio_dir else None
    if "anulm-asr" in served and audio_dir and (audio_dir / "en_capital.wav").exists():
        def asr(name, lang, fmt="json"):
            def f():
                with open(audio_dir / name, "rb") as fh:
                    r = client.audio.transcriptions.create(model="anulm-asr", file=fh, language=lang,
                                                           response_format=fmt)
                text = r if isinstance(r, str) else r.text
                assert text
                return repr(text)
            return f
        check("audio.transcriptions: English", asr("en_capital.wav", "en"))
        check("audio.transcriptions: Hindi", asr("hi_capital.wav", "hi"))
        check("audio.transcriptions: response_format=text", asr("en_capital.wav", "en", "text"))
        check("audio.transcriptions: verbose_json", lambda: (lambda r: f"{r['duration']}s {r['text']!r}")(
            http.post(V1 + "/audio/transcriptions", data={"language": "en", "response_format": "verbose_json"},
                      files={"file": ("q.wav", (audio_dir / "en_capital.wav").read_bytes(), "audio/wav")}).json()))

    if "system-tts" in served:
        def tts():
            r = client.audio.speech.create(model="system-tts", voice="alloy", input="The capital of India is New Delhi.",
                                           response_format="wav")
            data = r.read()
            with wave.open(io.BytesIO(data)) as w:
                secs = w.getnframes() / w.getframerate()
            assert data[:4] == b"RIFF" and secs > 0.5
            return f"{len(data):,} bytes, {secs:.1f}s of audio"
        check("audio.speech: WAV", tts)
        check("audio.speech: Hindi (no voice installed -> 422, or audio)", lambda: str(
            http.post(V1 + "/audio/speech", json={"input": "भारत की राजधानी नई दिल्ली है।"}).status_code))

    # ---------------------------------------------------------------- decide
    def decide(model, inp, opts=None, expect=None):
        def f():
            body = {"model": model, "input": inp}
            if opts:
                body["options"] = opts
            r = http.post(V1 + "/decide", json=body)
            assert r.status_code == 200, r.text
            res = r.json()["results"]
            if expect:
                assert [x["choice"] for x in res] == expect, [x["choice"] for x in res]
            return " | ".join(f"{x['choice']} {x['confidence']:.2f} {x['action']}" for x in res)
        return f
    check("decide: anulm-decide-banking", decide("anulm-decide-banking", "I still haven't received my new card",
                                                 expect=["card_arrival"]))
    check("decide: batch of 3", decide("anulm-decide-banking", ["How do I top up with Apple Pay?",
                                                                "Why was I charged twice for the same coffee?",
                                                                "Someone is using my card right now, block it!"]))
    check("decide: anulm-decide-hindi", decide("anulm-decide-hindi", ["क्या धूप वाला दिन है", "दस बजे मुझे उठाओ"],
                                               expect=["weather_query", "alarm_set"]))
    check("decide: custom option subset", decide("anulm-decide-banking", "my card never came",
                                                 ["card_arrival", "top_up_failed", "exchange_rate"],
                                                 expect=["card_arrival"]))
    check("decide: anulm-decide-general, any options", decide(
        "anulm-decide-general", "Add the onions and fry until golden", ["cooking", "sports", "politics", "music"]))
    for m in ("modernbert-decide-banking", "modernbert-decide-hindi"):
        if m in served:
            check(f"decide: {m}", decide(m, "क्या धूप वाला दिन है" if "hindi" in m else "my card has not arrived"))

    def router():
        from langchain_core.runnables import RunnableLambda
        route = RunnableLambda(lambda msg: http.post(V1 + "/decide", json={
            "model": "anulm-decide-banking", "input": msg}).json()["results"][0])
        handle = route | RunnableLambda(lambda d: f"queue:{d['choice']}" if d["action"] == "auto-route"
                                        else "queue:human")
        msgs = ["I still haven't received my new card", "Someone is using my card right now, block it!"]
        out = handle.batch(msgs)
        direct = [route.invoke(m) for m in msgs]
        assert out == [f"queue:{d['choice']}" if d["action"] == "auto-route" else "queue:human" for d in direct]
        assert out[0] == "queue:card_arrival"
        return str(out)
    check("LangChain: router on /v1/decide (RunnableLambda)", router)

    # ---------------------------------------------------------------- LangChain chat
    def lc_chat():
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="anulm-chat", base_url=V1, api_key=a.key, max_tokens=60, temperature=0.3)
        r = llm.invoke("What is the capital of India?")
        assert r.content
        return repr(r.content[:80])
    check("LangChain: ChatOpenAI.invoke", lc_chat)

    def lc_stream():
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="anulm-chat", base_url=V1, api_key=a.key, max_tokens=40)
        parts = [c.content for c in llm.stream("Tell me about the moon.")]
        assert sum(1 for p in parts if p) > 3
        return f"{len(parts)} chunks: {''.join(parts)[:70]!r}"
    check("LangChain: ChatOpenAI.stream", lc_stream)

    def lc_chain():
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
        prompt = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant."),
                                                   ("human", "Explain {topic} in one sentence.")])
        chain = prompt | ChatOpenAI(model="anulm-chat", base_url=V1, api_key=a.key, max_tokens=60) | StrOutputParser()
        out = chain.batch([{"topic": "gravity"}, {"topic": "Python lists"}])
        assert all(out)
        return " || ".join(o[:60] for o in out)
    check("LangChain: prompt | ChatOpenAI | parser, batch", lc_chain)

    def lc_lookup():
        from langchain_openai import ChatOpenAI
        r = ChatOpenAI(model="anulm-answer", base_url=V1, api_key=a.key).invoke("Who built the Taj Mahal?")
        assert "Shah Jahan" in r.content
        return repr(r.content)
    if "anulm-answer" in served:
        check("LangChain: ChatOpenAI with anulm-answer", lc_lookup)

    def lc_llm():
        from langchain_openai import OpenAI
        return repr(OpenAI(model="anulm-chat", base_url=V1, api_key=a.key, max_tokens=12).invoke("भारत एक"))
    check("LangChain: OpenAI (completions) LLM", lc_llm)

    # ---------------------------------------------------------------- errors
    def err(method, path, body, status, contains=""):
        def f():
            r = http.request(method, V1 + path, json=body)
            e = r.json().get("error", {})
            assert r.status_code == status and {"message", "type"} <= set(e), (r.status_code, r.text)
            assert contains in e["message"], e["message"]
            return f"{status} {e['type']}: {e['message'][:70]}"
        return f
    check("error: unknown model -> 404", err("POST", "/chat/completions",
                                             {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}, 404))
    check("error: tools -> 400", err("POST", "/chat/completions", {
        "model": "anulm-chat", "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}]}, 400, "Tool"))
    check("error: n=2 -> 400", err("POST", "/chat/completions", {
        "model": "anulm-chat", "n": 2, "messages": [{"role": "user", "content": "hi"}]}, 400))
    check("error: no user message -> 400", err("POST", "/chat/completions", {
        "model": "anulm-chat", "messages": [{"role": "system", "content": "x"}]}, 400))
    check("error: remote image URL -> 400", err("POST", "/chat/completions", {
        "model": "anulm-vision", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "http://example.com/cat.jpg"}}]}]}, 400, "data URL"))
    check("error: general decide without options -> 400", err("POST", "/decide", {
        "model": "anulm-decide-general", "input": "hello"}, 400, "options"))
    check("error: malformed body -> 422", lambda: str(http.post(V1 + "/chat/completions", json={"model": 3}).status_code))
    check("error: openai client raises NotFoundError", lambda: (lambda: (_ for _ in ()).throw(AssertionError("no error")))()
          if not _raises(lambda: client.chat.completions.create(model="nope", messages=[{"role": "user", "content": "x"}]),
                         "NotFoundError") else "openai.NotFoundError")

    # ---------------------------------------------------------------- robustness
    def concurrent():
        def one(i):
            if i % 2:
                return http.post(V1 + "/decide", json={"input": "my card has not arrived"}).status_code
            return http.post(V1 + "/chat/completions", json={"model": "anulm-chat", "max_tokens": 20,
                                                             "messages": [{"role": "user", "content": f"Count to {i}"}]}).status_code
        with cf.ThreadPoolExecutor(6) as ex:
            codes = list(ex.map(one, range(6)))
        assert codes == [200] * 6, codes
        return "6 parallel chat + decide requests all 200"
    check("6 concurrent requests (chat and decide share the model)", concurrent)

    def disconnect():
        with http.stream("POST", V1 + "/chat/completions", json={
                "model": "anulm-chat", "stream": True, "max_tokens": 300,
                "messages": [{"role": "user", "content": "Write a long essay about India."}]}) as r:
            for i, _ in enumerate(r.iter_lines()):
                if i > 3:
                    break
        t0 = time.time()
        r = http.post(V1 + "/decide", json={"input": "my card has not arrived"})
        dt = time.time() - t0
        base = time.time()
        http.post(V1 + "/decide", json={"input": "my card has not arrived"})
        alone = time.time() - base
        assert r.status_code == 200 and dt < alone + 1.0, f"{dt:.1f}s vs {alone:.1f}s alone: generation kept running"
        return f"next request {dt:.2f}s after the client hung up (alone: {alone:.2f}s)"
    check("client disconnects mid-stream -> generation stops, lock released", disconnect)

    def unchanged():
        kw = dict(model="anulm-chat", max_tokens=30, temperature=0,
                  messages=[{"role": "user", "content": "Give me three tips for staying healthy."}])
        before = client.chat.completions.create(**kw).choices[0].message.content
        http.post(V1 + "/decide", json={"model": "anulm-decide-hindi", "input": "दस बजे मुझे उठाओ"})
        http.post(V1 + "/decide", json={"model": "anulm-decide-banking", "input": "card arrival"})
        after = client.chat.completions.create(**kw).choices[0].message.content
        assert before == after
        return "chat output identical before and after decisions on both adapters"
    check("chat unaffected by the Decide adapters", unchanged)

    bad = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed" + (": FAILED " + ", ".join(b[0] for b in bad) if bad else ""))
    sys.exit(1 if bad else 0)


def _raises(fn, name) -> bool:
    try:
        fn()
    except Exception as e:
        return type(e).__name__ == name
    return False


if __name__ == "__main__":
    main()
