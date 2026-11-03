"""
Book example, not repository code.

Chapter 38: serving, streaming and cancellation.

    python book/code-examples/ch38_streaming.py

It imports `api.py`, `serve.py` and `model.py` unchanged (api.py needs fastapi
and pydantic) and drives `api.Generation` with the small test model on the
CPU, through a stand-in for the chat engine that has `api.py`'s expected
attributes and `serve.Engine`'s own byte-level encode and decode. It checks:

1. that `_release` never shows half a UTF-8 character, one byte at a time;
2. that it holds back the start of a stop sequence and cuts at it;
3. that `generate`'s `on_token` stops decoding when it returns True, and that
   cancelling a `Generation` mid-stream stops it at the next token;
4. that the `GPU` lock serialises two requests sent at once;
5. that `top_p` is accepted by the request schema and never reaches decoding;
6. the end of a reply: ending on EOS still sends the characters held back for
   a stop string, a reply cut by max_tokens in the middle of a character does
   not end in U+FFFD, a real U+FFFD in the text is kept, and `stop=""` is no
   stop. (Until November 2026 the first two failed: see the chapter.)
7. the guards at the door: the API key check, the exposure warning, a
   `/v1/decide` request over its caps (422), a 500 that carries an error id
   and not the exception text, the joint model refusing options it was never
   trained on, and `serve.py`'s 1 MB body cap (413) and error id, over real
   HTTP on this machine.
"""

from __future__ import annotations

import inspect
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm.inference import api                                            # noqa: E402  repository code, unchanged
from anulm import model as M                                     # noqa: E402
from anulm.inference import serve                                          # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def tiny(**kw) -> M.AnuLMConfig:
    base = dict(vocab_size=259, block_size=64, n_layer=3, hidden_size=32, n_head=4, head_dim=8,
                n_kv_head=2, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16,
                intermediate_size=64, n_group=1, topk_group=1)
    base.update(kw)
    return M.AnuLMConfig(**base)


def byte_engine(model=None):
    """What api.Generation reads from models.brain(), with serve.Engine's byte-level codec."""
    eng = SimpleNamespace(tok=None, model=model, device="cpu", eos_id=None,
                          cfg=model.cfg if model is not None else tiny())
    eng.encode = lambda s: serve.Engine.encode(eng, s)
    eng.decode = lambda ids: serve.Engine.decode(eng, ids)
    eng.decode_complete = lambda ids: serve.Engine.decode_complete(eng, ids)
    return eng


def generation(eng, prompt="", max_tokens=20, stops=()) -> api.Generation:
    """api.Generation without __init__ (which asks the server's model cache for the engine)."""
    g = api.Generation.__new__(api.Generation)
    g.eng = eng
    g.ids = eng.encode(prompt)[-(eng.cfg.block_size - 1):] if prompt else []
    g.max_tokens, g.temperature, g.top_k, g.rep, g.seed = max_tokens, 1.0, None, 1.0, 0
    g.stops, g.new = list(stops), []
    g.sent, g.text, g.finish, g.cancelled = "", "", "length", False
    g.matched_stop = False
    return g


def utf8_holdback() -> None:
    word = "नमस्ते"
    data = word.encode("utf-8")
    g = generation(byte_engine())
    print(f"1. '{word}': {len(word)} characters, {len(data)} UTF-8 bytes, one byte per token")
    deltas = []
    for i, b in enumerate(data):
        g.new.append(b)
        d = g._release(final=False)
        deltas.append(d)
        print(f"   byte {i:2d} 0x{b:02x}  decode -> {serve.Engine.decode(g.eng, g.new)[-1:]!r:6s}  released {d!r}")
    assert "".join(deltas) == word and not any("�" in d for d in deltas)


def stop_holdback() -> None:
    g = generation(byte_engine(), stops=["\n\n"])
    out = []
    for b in "Hi.\n\nNext".encode("utf-8"):
        g.new.append(b)
        out.append(g._release(final=False))
    print(f"\n2. stop sequence '\\n\\n' over 'Hi.\\n\\nNext': released {out}; finish {g.finish!r}; text {g.text!r}")
    assert "".join(out) == "Hi." and g.finish == "stop"


def on_token_stops() -> None:
    torch.manual_seed(0)
    m = M.AnuLM(tiny()).eval()
    x = torch.randint(0, 256, (1, 4))
    calls = []
    out = m.generate(x, 50, on_token=lambda nxt: calls.append(int(nxt)) or len(calls) == 5)
    print(f"\n3. generate(max_new_tokens=50) with on_token returning True on the 5th token: "
          f"{out.shape[1] - 4} new tokens")
    assert out.shape[1] - 4 == 5
    g = generation(byte_engine(m), prompt="abc", max_tokens=40)
    got = []

    def emit(delta):                                    # a client that hangs up after three pieces
        got.append(delta)
        if len(got) == 3:
            g.cancelled = True
    g.run(emit=emit)
    print(f"   Generation.run, cancelled after 3 deltas: {len(g.new)} tokens generated of max 40")
    assert len(g.new) < 40


def lock_serialises() -> None:
    torch.manual_seed(0)
    m = M.AnuLM(tiny(block_size=256)).eval()
    spans = {}
    orig = m.generate

    def timed(*a, **k):                                 # record when each request is on the device
        t0 = time.perf_counter()
        r = orig(*a, **k)
        spans[threading.current_thread().name] = (t0, time.perf_counter())
        return r
    m.generate = timed
    gens = [generation(byte_engine(m), prompt="request " + c, max_tokens=120) for c in "AB"]
    ts = [threading.Thread(target=g.run, name=f"request {c}") for g, c in zip(gens, "AB")]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    (a0, a1), (b0, b1) = sorted(spans.values())
    base = a0
    print(f"\n4. two requests sent at once: on the device {a0 - base:.3f}-{a1 - base:.3f} s and "
          f"{b0 - base:.3f}-{b1 - base:.3f} s; overlap: {b0 < a1}")
    assert b0 >= a1


def top_p_ignored() -> None:
    req = api.ChatRequest(model="anulm-chat", messages=[{"role": "user", "content": "hello"}], top_p=0.1)
    params = list(inspect.signature(api.Generation.__init__).parameters)[1:]
    print(f"\n5. ChatRequest(top_p=0.1).top_p = {req.top_p}; Generation takes {params}")
    assert req.top_p == 0.1 and "top_p" not in params


class Scripted:
    """A stand-in model whose generate() emits a fixed list of byte tokens through on_token."""

    def __init__(self, tokens):
        self.tokens, self.cfg = tokens, tiny()

    def generate(self, x, max_new_tokens, on_token=None, **kw):
        for t in self.tokens[:max_new_tokens]:
            if on_token(torch.tensor([[t]])):
                break
        return x


def end_of_reply() -> None:
    eos = 257
    print("\n6. a model that writes 'Hello' then EOS, through Generation.run:")
    for stops in ([], ["###"], ["\n\n", "User:"]):
        eng = byte_engine(Scripted(list(b"Hello") + [eos]))
        eng.eos_id = eos
        g = generation(eng, prompt="hi", stops=stops)
        streamed = []
        text = g.run(emit=streamed.append)
        print(f"   stops {stops!s:20s} streamed {''.join(streamed)!r:8s} returned text {text!r:8s} finish {g.finish!r}")
        assert "".join(streamed) == text == "Hello" and g.finish == "stop" and not g.matched_stop
    eng = byte_engine(Scripted(list("Hi न".encode("utf-8"))[:5]))
    g = generation(eng, prompt="hi", max_tokens=5)
    streamed = []
    g.run(emit=streamed.append)
    print(f"   max_tokens=5 over 'Hi न' (bytes 48 69 20 e0 a4): streamed {streamed!r}")
    assert "".join(streamed) == "Hi " and "�" not in "".join(streamed)
    eng = byte_engine(Scripted(list("a�b".encode("utf-8"))))   # a genuine U+FFFD
    g = generation(eng, prompt="hi")
    text = g.run()
    print(f"   a real U+FFFD (bytes ef bf bd) in the reply: returned text {text!r}")
    assert text == "a�b"
    print(f"   stops_of(''): {api.stops_of('')}; stops_of(['', '###']): {api.stops_of(['', '###'])}")
    assert api.stops_of("") == [] and api.stops_of(["", "###"]) == ["###"]


def guards() -> None:
    import contextlib
    import io
    import json
    import os
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer
    from fastapi.testclient import TestClient
    from anulm.multimodal import omni

    print("\n7. the guards at the door:")
    os.environ["ANULM_API_KEY"] = "s3cret"
    results = []
    for header in (None, "Bearer wrong", "Bearer s3cret"):
        try:
            api.check_key(header)
            results.append("ok")
        except api.APIError as e:
            results.append(e.status)
    print(f"   check_key with no header, a wrong key, the right key: {results}")
    assert results == [401, 401, "ok"]
    warned_with_key = api.exposure_warning("0.0.0.0")
    del os.environ["ANULM_API_KEY"]
    print(f"   exposure_warning: 127.0.0.1 {api.exposure_warning('127.0.0.1')}, "
          f"0.0.0.0 with a key {warned_with_key}, 0.0.0.0 without one: "
          f"{api.exposure_warning('0.0.0.0')[:44]!r}...")
    assert warned_with_key is None and api.exposure_warning("0.0.0.0")

    client = TestClient(api.app, raise_server_exceptions=False)
    r = client.post("/v1/decide", json={"input": ["hi"] * 65})
    print(f"   /v1/decide with 65 inputs: {r.status_code}")
    assert r.status_code == 422
    log = io.StringIO()
    with contextlib.redirect_stdout(log):     # no model is loaded here, so generation fails
        r = client.post("/v1/completions", json={"model": "anulm-chat", "prompt": "hi"})
    err = r.json()["error"]
    eid = err["code"]
    print(f"   a failing /v1/completions: {r.status_code}, message "
          f"{err['message'].replace(eid, '<id>')!r}")
    print(f"   the server log has the id and the exception: "
          f"{f'error {eid}' in log.getvalue()} {'AttributeError' in log.getvalue()}; "
          f"the reply has the exception: {'AttributeError' in r.text}")
    assert r.status_code == 500 and f"error {eid}" in log.getvalue() and "AttributeError" not in r.text

    it = omni.Intent.__new__(omni.Intent)            # choices() reads only the trained labels
    it.labels = ["card_arrival", "lost_or_stolen_card", "card_payment_fee_charged"]
    print(f"   omni.Intent.choices(['lost_or_stolen_card', 'card_arrival']): "
          f"{it.choices(['lost_or_stolen_card', 'card_arrival'])}")
    try:
        it.choices(["card_arrival", "pizza"])
        refused = None
    except ValueError as e:
        refused = type(e).__name__
    print(f"   omni.Intent.choices(['card_arrival', 'pizza']): {refused}")
    assert refused == "ValueError"

    class Failing:                                     # an engine whose generation fails
        def generate(self, *a, **k):
            raise RuntimeError("C:/models/secret.pt is corrupt")
    server = ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler(Failing()))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/generate"

    def post(body: bytes) -> tuple[int, str]:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=body), timeout=10) as resp:
                return resp.status, resp.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()
    err_log = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err_log):
        big = post(b" " * ((1 << 20) + 1))
        bad = post(json.dumps({"prompt": "hi", "max_tokens": "many"}).encode())
        fail = post(json.dumps({"prompt": "hi"}).encode())
    server.shutdown()
    print(f"   serve.py, a 1 MB + 1 byte body: {big[0]}; max_tokens 'many': {bad[0]}; "
          f"a failing generation: {fail[0]}, exception text in the reply: {'secret' in fail[1]}, "
          f"in the server log: {'secret' in err_log.getvalue()}")
    assert (big[0], bad[0], fail[0]) == (413, 400, 500) and "secret" not in fail[1] and "secret" in err_log.getvalue()


if __name__ == "__main__":
    utf8_holdback()
    stop_holdback()
    on_token_stops()
    lock_serialises()
    top_p_ignored()
    end_of_reply()
    guards()
    print("\nall checks passed")
