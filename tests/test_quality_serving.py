"""Regression tests for serving and the apps: api.py, serve.py, rag.py,
voice.py, omni.py and the Gradio apps' shared state.

No network, no GPU, no weights: every model is a stand-in, and the one real
forward pass (omni.Intent) runs a tiny random AnuLM on the CPU.

    python -m unittest tests.test_quality_serving
"""

from __future__ import annotations

import http.client
import io
import os
import threading
import time
import unittest
import wave
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from anulm.model import AnuLM, AnuLMConfig

LABELS = ["card_arrival", "lost_or_stolen_card", "top_up_failed"]


def tiny_model() -> AnuLM:
    cfg = AnuLMConfig(vocab_size=64, block_size=512, n_layer=2, hidden_size=32, n_head=4, head_dim=8,
                      n_kv_head=2, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
                      intermediate_size=64, n_group=1, topk_group=1)
    torch.manual_seed(0)
    return AnuLM(cfg).eval()


class ByteTok:
    """Bytes folded into the tiny model's 64 ids; 63 is the end token."""
    eos_id = 63

    def encode(self, s):
        return [b % 63 for b in s.encode("utf-8")]


def stand_in_intent():
    """A real omni.Intent over a tiny model, without the checkpoint or the label files."""
    from anulm.multimodal.omni import Intent
    it = Intent.__new__(Intent)
    it.om = SimpleNamespace(tok=ByteTok(), model=tiny_model(), device="cpu")
    it.threshold, it.lang, it.template, it.labels = 0.7, "en", "Which banking intent is this? {m}", list(LABELS)
    return it


def wav_bytes(seconds: float, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.zeros(int(seconds * rate), dtype="<i2").tobytes())
    return buf.getvalue()


class ApiBase(unittest.TestCase):
    def setUp(self):
        try:
            from fastapi.testclient import TestClient
            from anulm.inference import api
        except ImportError as e:                     # fastapi / gradio absent
            self.skipTest(f"API dependency unavailable: {e}")
        self.api, self.A = api, api.A
        self.client = TestClient(api.app, raise_server_exceptions=False)
        p = patch.object(self.A, "USE_OMNI", True)        # Omni mode, the default
        p.start()
        self.addCleanup(p.stop)
        env = patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("ANULM_API_KEY", None)

    def serve(self, **parts):
        p = patch.object(self.api, "models", SimpleNamespace(device="cpu", **parts))
        p.start()
        self.addCleanup(p.stop)


class DecideOmniTests(ApiBase):
    def test_decide_works_in_omni_mode(self):
        # Bug: api called d(text, options, top=...) but Intent.__call__ had no
        # `options`, so every /v1/decide in Omni mode was a TypeError -> 500.
        intent = stand_in_intent()
        self.serve(decider=lambda which, task: intent)
        r = self.client.post("/v1/decide", json={"model": "anulm-decide-banking",
                                                 "input": ["my card has not arrived", "top up failed"]})
        self.assertEqual(r.status_code, 200, r.text)
        res = r.json()["results"]
        self.assertEqual(len(res), 2)
        self.assertIn(res[0]["choice"], LABELS)
        self.assertAlmostEqual(sum(intent(res[0]["input"], top=3)["probabilities"].values()), 1.0, places=2)

    def test_a_subset_of_trained_labels_is_scored_over_that_subset(self):
        self.serve(decider=lambda which, task: stand_in_intent())
        r = self.client.post("/v1/decide", json={"model": "anulm-decide-banking", "input": "lost it",
                                                 "options": LABELS[:2]})
        self.assertEqual(r.status_code, 200, r.text)
        out = r.json()["results"][0]
        self.assertEqual(set(out["probabilities"]), set(LABELS[:2]))

    def test_custom_options_are_a_clear_400_in_omni_mode(self):
        self.serve(decider=lambda which, task: stand_in_intent())
        r = self.client.post("/v1/decide", json={"model": "anulm-decide-banking", "input": "hello",
                                                 "options": ["greeting", "farewell"]})
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("not supported by the joint model", r.json()["error"]["message"])

    def test_value_error_from_the_decider_is_a_400(self):
        # decide.encode's "too many options" becomes a ValueError (was an assert).
        def d(message, options=None, top=3):
            raise ValueError("2000 tokens; too many options for the context")
        d.threshold = 0.7
        self.serve(decider=lambda which, task: d)
        r = self.client.post("/v1/decide", json={"model": "anulm-decide-banking", "input": "x",
                                                 "options": ["a", "b"]})
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("too many options", r.json()["error"]["message"])

    def test_request_size_is_capped(self):
        self.serve(decider=lambda which, task: stand_in_intent())
        too_many = ["x"] * (self.api.MAX_DECIDE_INPUTS + 1)
        r = self.client.post("/v1/decide", json={"model": "anulm-decide-banking", "input": too_many})
        self.assertEqual(r.status_code, 422)
        opts = [f"o{i}" for i in range(self.api.MAX_DECIDE_OPTIONS + 1)]
        r = self.client.post("/v1/decide", json={"model": "anulm-decide-banking", "input": "x", "options": opts})
        self.assertEqual(r.status_code, 422)
        r = self.client.post("/v1/decide", json={"model": "anulm-decide-banking",
                                                 "input": "x" * (self.api.MAX_TEXT_CHARS + 1)})
        self.assertEqual(r.status_code, 422)


class ApiHardeningTests(ApiBase):
    def test_server_errors_do_not_leak_exception_text(self):
        def d(message, options=None, top=3):
            raise RuntimeError(r"secret detail C:\private\ckpt_x.pt")
        d.threshold = 0.7
        self.serve(decider=lambda which, task: d)
        with patch("builtins.print"):                  # the traceback goes to the log
            r = self.client.post("/v1/decide", json={"model": "anulm-decide-banking", "input": "x"})
        self.assertEqual(r.status_code, 500)
        err = r.json()["error"]
        self.assertNotIn("secret", r.text)
        self.assertNotIn("private", r.text)
        self.assertNotIn("RuntimeError", r.text)
        self.assertTrue(err["code"])
        self.assertIn(err["code"], err["message"])

    def test_api_key_is_compared_in_constant_time(self):
        os.environ["ANULM_API_KEY"] = "s3cret"
        calls = []
        real = self.api.hmac.compare_digest

        def spy(a, b):
            calls.append((a, b))
            return real(a, b)
        with patch.object(self.api.hmac, "compare_digest", spy):
            self.assertEqual(self.client.get("/v1/models").status_code, 401)
            self.assertEqual(self.client.get("/v1/models", headers={"Authorization": "Bearer nope"}).status_code, 401)
            self.assertEqual(self.client.get("/v1/models", headers={"Authorization": "Bearer s3cret"}).status_code, 200)
        self.assertEqual(len(calls), 3)

    def test_exposure_warning(self):
        self.assertIsNone(self.api.exposure_warning("127.0.0.1"))
        self.assertIsNone(self.api.exposure_warning("localhost"))
        self.assertIn("ANULM_API_KEY", self.api.exposure_warning("0.0.0.0"))
        os.environ["ANULM_API_KEY"] = "k"
        self.assertIsNone(self.api.exposure_warning("0.0.0.0"))

    def test_brain_is_loaded_under_the_device_lock(self):
        # Bug: models.brain() ran outside GPU, so a first request loaded the
        # model onto the device while another generated on it.
        held = []

        def brain():
            held.append(self.api.GPU.locked())
            return SimpleNamespace(encode=lambda s: [1, 2], cfg=SimpleNamespace(block_size=64))
        self.serve(brain=brain)
        self.api.Generation("Q", 10, 0.3, 40, 1.0, None, [])
        self.assertEqual(held, [True])

    def test_oversized_image_is_refused_before_decoding(self):
        with patch.object(self.api, "MAX_IMAGE_BYTES", 100):
            with self.assertRaises(self.api.APIError) as cm:
                self.api.save_image("data:image/png;base64," + "A" * 1000)
        self.assertEqual(cm.exception.status, 413)

    def transcribe(self, data: bytes):
        heard = []

        class Ear:
            def hear(self, audio):
                heard.append(audio)
                return {"text": "hello"}
        self.serve(ear=lambda lang: Ear())
        r = self.client.post("/v1/audio/transcriptions", files={"file": ("a.wav", data, "audio/wav")},
                             data={"model": "anulm-asr", "language": "en"})
        return r, heard

    def test_audio_byte_cap(self):
        with patch.object(self.api, "MAX_AUDIO_BYTES", 1000):
            r, heard = self.transcribe(wav_bytes(1.0))
        self.assertEqual(r.status_code, 413, r.text)
        self.assertEqual(heard, [])

    def test_audio_duration_cap(self):
        r, heard = self.transcribe(wav_bytes(self.api.MAX_AUDIO_SECONDS + 1))
        self.assertEqual(r.status_code, 413, r.text)
        self.assertIn(f"{self.api.MAX_AUDIO_SECONDS} s", r.json()["error"]["message"])
        self.assertEqual(heard, [])

    def test_audio_is_decoded_once_and_the_samples_passed_on(self):
        r, heard = self.transcribe(wav_bytes(1.0))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json(), {"text": "hello"})
        # The decoded samples, not the temp file's path (which made the ear read it again).
        self.assertIsInstance(heard[0], tuple)
        self.assertEqual(heard[0][0], 16000)
        self.assertEqual(len(heard[0][1]), 16000)


class ServeTests(unittest.TestCase):
    """serve.py's stdlib handler, on a real socket."""

    def setUp(self):
        from http.server import ThreadingHTTPServer
        from anulm.inference import serve
        self.serve = serve
        self.engine = SimpleNamespace(info={}, generate=lambda prompt, **kw: {"completion": prompt})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler(self.engine))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def post(self, body: bytes, length: int | None = None):
        c = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=10)
        c.putrequest("POST", "/generate")
        c.putheader("Content-Type", "application/json")
        c.putheader("Content-Length", str(len(body) if length is None else length))
        c.endheaders()
        if body:
            c.send(body)
        r = c.getresponse()
        out = r.status, r.read().decode("utf-8")
        c.close()
        return out

    def test_content_length_is_capped(self):
        # Bug: rfile.read(n) with n straight from the header.
        with patch("builtins.print"):
            status, _ = self.post(b"", length=self.serve.MAX_BODY_BYTES + 1)
        self.assertEqual(status, 413)

    def test_errors_do_not_leak_exception_text(self):
        def boom(prompt, **kw):
            raise RuntimeError(r"secret C:\private\path")
        self.engine.generate = boom
        with patch("builtins.print"):
            status, text = self.post(b'{"prompt": "hi"}')
        self.assertEqual(status, 500)
        self.assertNotIn("secret", text)
        self.assertNotIn("RuntimeError", text)
        self.assertIn("id", text)

    def test_bad_numbers_are_a_400(self):
        with patch("builtins.print"):
            status, _ = self.post(b'{"prompt": "hi", "max_tokens": "lots"}')
            ok, text = self.post(b'{"prompt": "hi"}')
        self.assertEqual(status, 400)
        self.assertEqual(ok, 200, text)


class AppOmniStateTests(unittest.TestCase):
    def setUp(self):
        try:
            from anulm.apps import app_omni
        except ImportError as e:
            self.skipTest(f"gradio unavailable: {e}")
        self.A = app_omni

    def test_two_first_requests_load_once(self):
        # Bug: _get had no lock; two threads both saw the key missing.
        m = self.A.Models("cpu", "x")
        made = []

        def make():
            made.append(1)
            time.sleep(0.2)
            return object()
        with patch("builtins.print"):
            ts = [threading.Thread(target=m._get, args=("k", make)) for _ in range(2)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
        self.assertEqual(len(made), 1)

    def test_threshold_is_per_call_not_shared(self):
        # Bug: d.threshold = th ran outside the lock, so a second visitor's
        # slider could change the first visitor's verdict mid-call.
        m = self.A.Models("cpu", "x")

        class Slow:
            threshold = 0.5

            def __call__(self, message, options=None, top=3):
                th = self.threshold
                time.sleep(0.1)
                return {"seen": th, "now": self.threshold}
        d = Slow()
        m.decider = lambda which=None, task=None: d
        outs = {}
        ts = [threading.Thread(target=lambda th=th: outs.__setitem__(th, self.A.decide(m, "msg", th)))
              for th in (0.3, 0.9)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.assertEqual(set(outs), {0.3, 0.9})
        for th, out in outs.items():
            self.assertEqual(out["seen"], th)
            self.assertEqual(out["now"], th)       # nobody changed it during the call

    def test_hearing_and_looking_hold_the_shared_lock(self):
        m = self.A.Models("cpu", "x")
        held = []
        m.ear = lambda lang: SimpleNamespace(hear=lambda a: held.append(m.shared._is_owned()) or {"text": ""})
        self.A.hear(m, "en", None)
        self.assertEqual(held, [True])

    def test_dead_code_is_gone_and_public_names_stay(self):
        self.assertFalse(hasattr(self.A, "build_visitor"))
        self.assertFalse(hasattr(self.A, "_HERE"))
        self.assertTrue(callable(self.A.Models))
        from anulm.apps import app
        from anulm.inference import api
        self.assertTrue(callable(app.Loader) and callable(app.build) and callable(api.available))


class AppTests(unittest.TestCase):
    def setUp(self):
        try:
            from anulm.apps import app
        except ImportError as e:
            self.skipTest(f"gradio unavailable: {e}")
        self.app = app

    def test_user_penalty_wins(self):
        # Bug: max(penalty, default) -- below 1.3 was impossible in question mode.
        self.assertEqual(self.app.repetition_penalty("qa", "question", 1.0), 1.0)
        self.assertEqual(self.app.repetition_penalty("qa", "question", 1.5), 1.5)
        self.assertEqual(self.app.repetition_penalty("qa", "question", None), 1.3)
        self.assertEqual(self.app.repetition_penalty("coder", "question", None), 1.0)

    def test_exposure_warning(self):
        w = self.app.exposure_warning
        self.assertIsNone(w("127.0.0.1"))
        self.assertIsNone(w(None))
        self.assertIsNotNone(w("0.0.0.0"))
        self.assertIsNotNone(w("127.0.0.1", share=True))
        self.assertIsNone(w("0.0.0.0", auth=("u", "p")))

    def test_voice_default_is_named(self):
        from anulm.apps import app_voice
        self.assertEqual(self.app.MODELS[app_voice.DEFAULT_MODEL], "toonist/AnuLM-Hindi-QA-400M")


class RetrievalDeviceTests(unittest.TestCase):
    def test_retriever_passes_its_device_to_dense(self):
        # Bug: Dense() always picked CUDA, whatever device the app ran on.
        from anulm.inference import rag
        seen = []

        class FakeDense:
            def __init__(self, device=None):
                seen.append(device)

            def search(self, q, k):
                return [0], [1.0]
        r = rag.Retriever.__new__(rag.Retriever)
        r.device, r.dense, r.mode, r.chunks = "cpu", None, "dense", [{"title": "T", "text": "t", "lang": "en"}]
        with patch.object(rag, "Dense", FakeDense):
            hits = r.search("q", 1)
        self.assertEqual(seen, ["cpu"])
        self.assertEqual(hits[0]["title"], "T")

    def test_omni_reader_builds_its_retriever_on_the_model_device(self):
        from anulm.inference import rag
        from anulm.multimodal.omni import Reader
        made = []

        class FakeRetriever:
            def __init__(self, root=None, device=None):
                made.append(device)

            def search(self, q, k):
                return [{"title": "T", "text": "passage"}]
        om = SimpleNamespace(device="cpu", engine=SimpleNamespace(generate=lambda *a, **k: {"completion": " A "}))
        with patch.object(rag, "Retriever", FakeRetriever):
            out = Reader(om).ask("Who?")
        self.assertEqual(made, ["cpu"])
        self.assertEqual(out["answer"], "A")


class VoiceTests(unittest.TestCase):
    def test_no_powershell_fallback_off_windows(self):
        from anulm.inference import voice
        tts = voice.TTS()

        def blocked(lang):
            raise ImportError("no piper")
        tts.voice = blocked
        with patch.object(voice.sys, "platform", "linux"), \
                patch.object(voice, "windows_say", side_effect=AssertionError("powershell called")):
            with self.assertRaises(RuntimeError) as cm:
                tts.say("hello", "en")
        self.assertIn("piper", str(cm.exception).lower())


if __name__ == "__main__":
    unittest.main()
