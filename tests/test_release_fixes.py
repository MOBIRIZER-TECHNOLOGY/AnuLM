"""
Regression tests for the fixes ported after the three-way code comparison
(CLEAN vs V3 vs the 2026-11-02 bundle). Each test names the bug it pins.

    python -m unittest tests.test_release_fixes

Nothing here needs model weights, a GPU or a download. The API tests drive
`api.Generation` with a stand-in engine; they skip, visibly, without fastapi.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from anulm.bpe import BPE, complete_utf8
from anulm.model import AnuLM, AnuLMConfig

ROOT = Path(__file__).resolve().parents[1]
EOS = 255          # the stand-in engine's end token: one byte value that never occurs in the test text


def tiny(**kw) -> AnuLMConfig:
    base = dict(vocab_size=64, block_size=32, n_layer=2, hidden_size=32, n_head=4, head_dim=8,
                n_kv_head=2, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
                intermediate_size=64, n_group=1, topk_group=1)
    base.update(kw)
    return AnuLMConfig(**base)


class FakeEngine:
    """Byte-level stand-in for serve.Engine: `generate` replays fixed ids."""
    device, eos_id = "cpu", EOS
    cfg = SimpleNamespace(block_size=128)

    def __init__(self, ids):
        self.ids, self.model, self.kwargs = ids, self, None

    def encode(self, text):
        return list(text.encode()) or [10]

    def decode(self, ids):
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")

    def decode_complete(self, ids):
        return complete_utf8(bytes(i for i in ids if i < 256))

    def generate(self, x, max_tokens, **kwargs):
        self.kwargs = kwargs
        for t in self.ids[:max_tokens]:
            if kwargs["on_token"](torch.tensor([[t]])):
                break


class StreamingTests(unittest.TestCase):
    """api.Generation: what a streaming client receives."""

    def setUp(self):
        try:
            from anulm.inference import api
        except ImportError as e:            # fastapi / gradio absent
            self.skipTest(f"API dependency unavailable: {e}")
        self.api = api

    def stream(self, ids, stops=(), temperature=0.3):
        eng = FakeEngine(ids)
        with patch.object(self.api, "models", SimpleNamespace(brain=lambda: eng)):
            gen = self.api.Generation("Q", 100, temperature, 0, 1.0, None, list(stops))
            chunks = []
            text = gen.run(chunks.append)
        # What was streamed must be exactly what is returned.
        self.assertEqual("".join(chunks), text)
        return text, gen, eng

    def test_eos_releases_the_text_held_back_for_a_stop_string(self):
        # Bug: EOS set finish="stop", which skipped the final release, so a
        # client passing `stop` lost the last len(stop)-1 characters.
        text, gen, _ = self.stream(list(b"Hello") + [EOS], stops=["###"])
        self.assertEqual(text, "Hello")
        self.assertEqual(gen.finish, "stop")
        self.assertFalse(gen.matched_stop)

    def test_a_stop_string_across_tokens_is_cut_and_never_streamed(self):
        text, gen, _ = self.stream(list(b"Hello###tail") + [EOS], stops=["###"])
        self.assertEqual(text, "Hello")
        self.assertTrue(gen.matched_stop)

    def test_empty_stop_string_is_no_stop(self):
        # Bug: stop="" became [""], "".find("") == 0, and every reply was empty.
        self.assertEqual(self.api.stops_of(""), [])
        self.assertEqual(self.api.stops_of(["", "x"]), ["x"])
        self.assertEqual(self.api.stops_of(None), [])
        self.assertEqual(self.stream(list(b"OK") + [EOS], stops=self.api.stops_of(""))[0], "OK")

    def test_utf8_split_across_tokens_is_held_back_and_a_real_fffd_survives(self):
        # Bug: half a Devanagari character streamed as U+FFFD, and the old
        # rstrip of U+FFFD also deleted a genuine one at the end of the text.
        self.assertEqual(self.stream(list(b"Hi ") + [0xE0])[0], "Hi ")
        self.assertEqual(self.stream(list("नमस्ते".encode()) + [EOS])[0], "नमस्ते")
        self.assertEqual(self.stream(list("Hi �".encode()) + [EOS])[0], "Hi �")

    def test_temperature_zero_reaches_the_model_unclamped(self):
        # It used to become max(t, 1e-3): a near-greedy *sample*.
        _, _, eng = self.stream(list(b"OK") + [EOS], temperature=0.0)
        self.assertEqual(eng.kwargs["temperature"], 0.0)


class GreedyTests(unittest.TestCase):
    def test_temperature_zero_is_exact_and_seed_independent(self):
        torch.manual_seed(0)
        m = AnuLM(tiny()).eval()
        x = torch.tensor([[1, 2, 3]])
        outs = []
        for seed in (1, 2, 3):
            torch.manual_seed(seed)
            outs.append(m.generate(x, 8, temperature=0.0)[0].tolist())
        self.assertEqual(outs[0], outs[1])
        self.assertEqual(outs[0], outs[2])
        # ... and it is the argmax at every step, which the uncached path recomputes.
        ids = x
        for _ in range(8):
            logits, _ = m(ids)
            ids = torch.cat([ids, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        self.assertEqual(outs[0], ids[0].tolist())

    def test_temperature_zero_breaks_ties_deterministically(self):
        # Bug: temperature 0 became 1e-5 and still *sampled*: on a tie every
        # tied id was equally likely, so "greedy" output changed with the seed.
        torch.manual_seed(0)
        m = AnuLM(tiny()).eval()
        with torch.no_grad():
            m.lm_head.weight.zero_()                            # every logit equal: a 64-way tie
        x = torch.tensor([[1, 2, 3]])
        for seed in range(5):
            torch.manual_seed(seed)
            self.assertEqual(m.generate(x, 6, temperature=0.0)[0, 3:].tolist(), [0] * 6)

    def test_sampling_still_depends_on_the_seed(self):
        torch.manual_seed(0)
        m = AnuLM(tiny()).eval()
        x = torch.tensor([[1, 2, 3]])
        runs = set()
        for seed in range(6):
            torch.manual_seed(seed)
            runs.add(tuple(m.generate(x, 8, temperature=1.5)[0].tolist()))
        self.assertGreater(len(runs), 1)


class Utf8Tests(unittest.TestCase):
    def test_complete_utf8(self):
        na = "न".encode()                                    # three bytes
        self.assertEqual(complete_utf8(na[:2]), "")
        self.assertEqual(complete_utf8(b"a" + na), "aन")
        self.assertEqual(complete_utf8("x�".encode()), "x�")
        self.assertEqual(complete_utf8(b"a\xffb"), "a�b")   # never valid: still replaced

    def test_bpe_decode_complete(self):
        tok = BPE([])                                         # bytes only
        ids = list("नमस्ते".encode())
        for k in range(len(ids) + 1):
            self.assertNotIn("�", tok.decode_complete(ids[:k]))
        self.assertEqual(tok.decode_complete(ids), "नमस्ते")


class TokenizerDetectionTests(unittest.TestCase):
    def test_project_bpe_named_tokenizer_json_loads_as_bpe(self):
        # Bug: the file *name* "tokenizer.json" routed the project's own format
        # to HFTokenizer, which cannot read it.
        from anulm.hf_tok import load_tokenizer
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tokenizer.json"
            BPE([(104, 105)]).save(p)
            self.assertIsInstance(load_tokenizer(p), BPE)

    def test_hf_tokenizer_under_any_name_and_its_streaming_decode(self):
        try:
            from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        except ImportError:
            self.skipTest("tokenizers not installed")
        from anulm.hf_tok import HFTokenizer, load_tokenizer
        tk = Tokenizer(models.BPE())
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
        tk.train_from_iterator(["नमस्ते दुनिया", "hello world"] * 20,
                               trainers.BpeTrainer(vocab_size=300, special_tokens=["<|endoftext|>"],
                                                   initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "renamed_vocab.json"
            tk.save(str(p))
            tok = load_tokenizer(p)
            self.assertIsInstance(tok, HFTokenizer)
            text = "नमस्ते hello"
            ids = tok.encode(text)
            self.assertEqual(tok.decode_complete(ids), text)
            for k in range(len(ids) + 1):
                self.assertNotIn("�", tok.decode_complete(ids[:k]))
            self.assertEqual(tok.decode_complete(ids + [tok.eos_id]), text + "<|endoftext|>")


class ServeEosTests(unittest.TestCase):
    def test_reply_stops_before_the_eos_control_token(self):
        # A HF tokenizer decodes EOS to "<|endoftext|>"; it must not reach the reply.
        from anulm.inference.serve import Engine
        eng = Engine.__new__(Engine)
        eng.cfg, eng.device, eng.eos_id = SimpleNamespace(block_size=32), "cpu", 0
        eng.lock, eng.qa_templates, eng.coder = threading.Lock(), None, False
        eng.tok = SimpleNamespace(encode=lambda s: [2],
                                  decode=lambda ids: "".join({1: "A", 0: "<|endoftext|>"}[i] for i in ids))
        eng.model = SimpleNamespace(generate=lambda x, *a, **k: torch.cat([x, torch.tensor([[1, 0]])], 1))
        out = eng.generate("Q", 10, 0.0, 0, 0)
        self.assertEqual(out["completion"], "A")
        self.assertTrue(out["stopped_at_eos"])


class ChatSplitTests(unittest.TestCase):
    def test_question_key(self):
        from anulm.data.make_qa import question_key
        self.assertEqual(question_key("  SAME   question "), question_key("same QUESTION"))
        self.assertEqual(question_key("ﬁne"), question_key("fine"))       # NFKC
        self.assertNotEqual(question_key("a b"), question_key("ab"))

    def test_same_question_never_crosses_the_split(self):
        from anulm.data.make_chat import split_chat_rows
        from anulm.data.make_qa import question_key
        rows = [{"question": f"q {i}", "answer": "a", "lang": "en", "source": "s"} for i in range(50)]
        rows += [{"question": f"Q  {i}", "answer": "other", "lang": "en", "source": "t"} for i in range(50)]
        rows += [{"question": "Who are you?", "answer": "AnuLM", "lang": "en", "source": "identity"}] * 20
        rows += [{"question": "who are you?", "answer": "a copy", "lang": "en", "source": "s"}]
        train, held = split_chat_rows(rows, held_groups=10)
        tk, hk = {question_key(r["question"]) for r in train}, {question_key(r["question"]) for r in held}
        self.assertFalse(tk & hk)
        self.assertEqual(len(hk), 10)
        self.assertNotIn(question_key("Who are you?"), hk)              # identity questions stay in training
        self.assertEqual(sum(r["source"] == "identity" for r in train), 20)   # deliberate repeats kept

    def test_exact_duplicates_are_kept_once(self):
        from anulm.data.make_chat import split_chat_rows
        row = {"question": "x", "answer": "y", "lang": "en", "source": "s"}
        rows = [dict(row), dict(row), {"question": "z", "answer": "w", "lang": "en", "source": "s"}]
        train, held = split_chat_rows(rows, held_groups=1)
        self.assertEqual(len(train) + len(held), 2)


class FinetuneCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.tok = self.tmp / "tok.json"
        BPE([]).save(self.tok)
        self.qa = self.tmp / "qa.jsonl"
        self.qa.write_text("".join(json.dumps({"question": f"q{i}", "answer": f"a{i}", "lang": "en"}) + "\n"
                                   for i in range(6)), encoding="utf-8")

    def load(self):
        from anulm.training import finetune
        with patch("builtins.print"):
            return finetune.load_packed(str(self.qa), str(self.tok), 64, 256, 1)

    def cache(self):
        return Path(f"{self.qa}.{self.tok.stem}.b64.pt")

    def test_cache_round_trip_and_shape(self):
        x, y = self.load()
        self.assertEqual(x.shape[1], 64)
        self.assertTrue(self.cache().exists())
        x2, y2 = self.load()                                   # served from the cache
        self.assertTrue(torch.equal(x, x2) and torch.equal(y, y2))

    def test_a_malformed_cache_is_rebuilt(self):
        x, y = self.load()
        c = torch.load(self.cache(), weights_only=True)
        c["x"] = c["x"][:, :10]                                # right key, wrong shape
        torch.save(c, self.cache())
        x2, _ = self.load()
        self.assertTrue(torch.equal(x, x2))

    def test_a_planted_pickle_is_never_executed(self):
        # Bug: the cache was read with weights_only=False, so a file beside the
        # data could run code. Now unpickling it is refused and it is rebuilt.
        marker = self.tmp / "pwned"

        class Payload:
            def __reduce__(self):
                return (Path.write_text, (marker, "ran"))
        with open(self.cache(), "wb") as f:
            torch.save({"x": Payload()}, f)
        self.load()
        self.assertFalse(marker.exists())


class OmniRecipeTests(unittest.TestCase):
    def test_recipe_records_what_a_resume_must_share(self):
        from anulm.multimodal.omni_train import recipe_of
        args = SimpleNamespace(accum=8, proj_scale="ln", lr=2.5e-5, proj_lr=1e-3, steps=100, warmup=10)
        tasks = [SimpleNamespace(name="chat", weight=0.25, batch=8)]
        base = recipe_of(args, tasks)
        self.assertEqual(base, recipe_of(SimpleNamespace(**vars(args)), tasks))
        for change in ({"accum": 1}, {"lr": 5e-5}, {"steps": 200}):
            self.assertNotEqual(base, recipe_of(SimpleNamespace(**{**vars(args), **change}), tasks))
        self.assertNotEqual(base, recipe_of(args, [SimpleNamespace(name="chat", weight=0.07, batch=8)]))


class ValidatedDemoTests(unittest.TestCase):
    """validated_demo shows a case only if every piece of its evidence checks out."""

    def setUp(self):
        from anulm.apps import validated_demo as vd
        self.vd = vd
        self.root = Path(tempfile.mkdtemp())
        self.catalogue = self.root / "catalogue.json"
        self.report = {"case_id": "c1", "task": "text", "status": "passed", "approved": True,
                       "reviewed_by": "reviewer", "validated_at": "2026-11-03", "criteria": "exact match",
                       "output": "Delhi", "source_revision": "abc1234", "input": "Capital of India?",
                       "dataset": {"source": "a held-out set", "license": "CC BY 4.0", "held_out": True},
                       "artifacts": []}
        for role in ("checkpoint", "tokenizer", "encoder"):
            (self.root / role).write_bytes(role.encode())
            self.report["artifacts"].append({"role": role, "path": role, "sha256": vd.digest(self.root / role)})

    def save(self, cases=None):
        rp = self.root / "report.json"
        rp.write_text(json.dumps(self.report), encoding="utf-8")
        entry = {"id": "c1", "report": {"path": "report.json", "sha256": self.vd.digest(rp)}}
        self.catalogue.write_text(json.dumps({"version": 1, "cases": cases or [entry]}), encoding="utf-8")

    def reasons(self):
        cases, excluded = self.vd.load_cases(self.catalogue, self.root)
        self.assertEqual(cases, [])
        return " | ".join(excluded)

    def test_a_complete_case_is_shown(self):
        self.save()
        cases, excluded = self.vd.load_cases(self.catalogue, self.root)
        self.assertEqual([c["id"] for c in cases], ["c1"])
        self.assertEqual(excluded, [])

    def test_missing_empty_and_malformed_catalogues_show_nothing(self):
        self.assertIn("no catalogue", self.reasons())
        for text in ("{}", "[]", "not json", '{"version": 2, "cases": []}'):
            self.catalogue.write_text(text, encoding="utf-8")
            self.assertIn("invalid", self.reasons())
        self.catalogue.write_text('{"version": 1, "cases": []}', encoding="utf-8")
        self.assertEqual(self.vd.load_cases(self.catalogue, self.root), ([], []))

    def test_each_failure_is_excluded_with_its_own_reason(self):
        for change, reason in [({"status": "pending"}, "not 'passed'"),
                               ({"approved": False}, "not approved"),
                               ({"reviewed_by": " "}, "reviewed_by"),
                               ({"case_id": "other"}, "for case 'other'"),
                               ({"task": "video"}, "task 'video'"),
                               ({"dataset": {"source": "x", "license": "y", "held_out": False}}, "held-out")]:
            saved = dict(self.report)
            self.report.update(change)
            self.save()
            with self.subTest(change=change):
                self.assertIn(f"c1: ", self.reasons())
                self.assertIn(reason, self.reasons())
            self.report = saved

    def test_a_changed_or_missing_file_or_one_outside_the_repo_is_excluded(self):
        self.save()
        (self.root / "checkpoint").write_bytes(b"retrained")
        self.assertIn("checkpoint has changed", self.reasons())
        self.report["artifacts"][0]["path"] = "../elsewhere"
        self.save()
        self.assertIn("outside the repository", self.reasons())
        self.report["artifacts"] = self.report["artifacts"][1:]
        self.save()
        self.assertIn("lack checkpoint", self.reasons())

    def test_media_cases_need_their_input_and_an_encoder(self):
        img = self.root / "photo.jpg"
        img.write_bytes(b"jpeg")
        self.report.update(task="image", input={"path": "photo.jpg", "sha256": self.vd.digest(img)})
        self.save()
        self.assertEqual(len(self.vd.load_cases(self.catalogue, self.root)[0]), 1)
        img.write_bytes(b"another photo")
        self.assertIn("image input", self.reasons())

    def test_duplicate_ids_and_each_file_hashed_once(self):
        self.save()
        entry = json.loads(self.catalogue.read_text(encoding="utf-8"))["cases"][0]
        self.save(cases=[entry, entry])
        cases, excluded = self.vd.load_cases(self.catalogue, self.root)
        self.assertEqual(len(cases), 1)
        self.assertEqual(excluded, ["c1: duplicate id"])
        h = self.vd.Hasher()
        with patch.object(Path, "open", wraps=Path.open, autospec=True) as opened:
            h(self.root / "checkpoint")
            h(self.root / "checkpoint")
        self.assertEqual(opened.call_count, 1)

    def test_the_shipped_catalogue_is_valid(self):
        cases, excluded = self.vd.load_cases()
        self.assertEqual(excluded, [])


class DecontaminationTests(unittest.TestCase):
    def test_a_13_word_overlap_at_any_offset_is_caught(self):
        # Documents were checked every 4th window: an overlap of exactly 13
        # words is one window, and 3 offsets in 4 were never looked at.
        from anulm.data.convert_parquet import NGRAM, ngrams
        bench_words = [f"b{i}" for i in range(NGRAM)]
        bench = ngrams(" ".join(bench_words))
        for offset in range(8):
            doc = " ".join([f"x{i}" for i in range(offset)] + bench_words + ["tail", "words"])
            with self.subTest(offset=offset):
                self.assertTrue(ngrams(doc) & bench)
        self.assertFalse(ngrams(" ".join(bench_words[:-1] + ["other"])) & bench)


class SpeechSafetyTests(unittest.TestCase):
    def test_training_drops_clips_longer_than_whispers_window(self):
        # Five FLEURS Hindi train clips (31-36 s, float WAVs with no `seconds`)
        # stopped the joint run at its first step.
        import numpy as np
        # Only this test writes WAVs. The skip used to sit in setUp, which took
        # the two tests below -- neither touches soundfile -- down with it.
        try:
            import soundfile as sf
        except ImportError:
            self.skipTest("soundfile not installed")
        from anulm.multimodal.speech_encoder import whisper_window_rows
        d = Path(tempfile.mkdtemp())
        for name, secs in (("short", 2.0), ("long", 31.0)):
            sf.write(d / f"{name}.wav", np.zeros(int(secs * 16000), np.float32), 16000, subtype="FLOAT")
        rows = [{"audio": str(d / "short.wav"), "text": "a"},            # duration from the header
                {"audio": str(d / "long.wav"), "text": "b"},
                {"audio": str(d / "short.wav"), "text": "c", "seconds": 29.9},
                {"audio": str(d / "short.wav"), "text": "d", "seconds": 30.5}]
        with patch("builtins.print"):
            kept = whisper_window_rows(rows, "test")
        self.assertEqual([r["text"] for r in kept], ["a", "c"])

    def test_whisper_refuses_nan_audio_and_oversized_chunks(self):
        import numpy as np
        from anulm.multimodal.audio_codec import ASR_RATE, WhisperEncoder, audio_chunks
        for bad in (np.array([0.0, np.nan], np.float32), np.array([np.inf] * 10, np.float32)):
            with self.assertRaises(ValueError):
                WhisperEncoder.encode(SimpleNamespace(), bad)          # refused before the model is touched
        with self.assertRaises(ValueError):
            audio_chunks(np.zeros(10), 31 * ASR_RATE)
        self.assertEqual(len(audio_chunks(np.zeros(70 * ASR_RATE))), 3)

    def test_transcript_budget_fits_the_context(self):
        # decode_from_embeds asserts past block_size; transcribe now asks for at
        # most what fits beside the audio.
        import numpy as np
        from anulm.multimodal import speech_encoder as se
        asked = []
        model = SimpleNamespace(cfg=SimpleNamespace(block_size=50), embed_tokens=torch.nn.Embedding(10, 4),
                                decode_from_embeds=lambda prefix, n, **k: asked.append((prefix.shape[0], n)) or [])
        sm = SimpleNamespace(model=model, proj=torch.nn.Linear(4, 4),
                             vocab=SimpleNamespace(speech_bos=1, text_bos=2, text=8))
        enc = SimpleNamespace(encode=lambda chunk: np.zeros((40, 4), np.float32))
        tok = SimpleNamespace(eos_id=3, decode=lambda ids: "")
        se.transcribe(sm, enc, tok, np.zeros(16000, np.float32), max_tokens=64, device="cpu")
        self.assertEqual(asked, [(42, 8)])                              # 50 - (1 + 40 + 1)
        enc.encode = lambda chunk: np.zeros((49, 4), np.float32)
        with self.assertRaises(ValueError):
            se.transcribe(sm, enc, tok, np.zeros(16000, np.float32), device="cpu")


class ExportParityTests(unittest.TestCase):
    """export_hf --weights-dtype float32 is exact, and every export checks itself."""

    @classmethod
    def setUpClass(cls):
        try:
            import safetensors  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("safetensors not installed")
        cls.tmp = Path(tempfile.mkdtemp())
        tok = cls.tmp / "tok.json"
        BPE([(104, 105), (256, 33)]).save(tok)
        torch.manual_seed(0)
        cfg = tiny(vocab_size=259, tokenizer_path=str(tok))
        cls.state = AnuLM(cfg).state_dict()
        cls.ckpt = cls.tmp / "ck.pt"
        torch.save({"model": cls.state, "cfg": cfg, "step": 7, "val_loss": 1.5}, cls.ckpt)

    def export(self, name, *extra):
        import os
        out = self.tmp / name
        r = subprocess.run([sys.executable, "-m", "anulm", "export_hf", str(self.ckpt), str(out), "--license", "mit",
                            *extra], cwd=ROOT, capture_output=True, text=True, timeout=300,
                           env={**os.environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return out, r.stdout

    def test_float32_export_reloads_bit_identical(self):
        from anulm.model import load_checkpoint
        out, log = self.export("fp32", "--weights-dtype", "float32")
        self.assertIn("parity: reloaded", log)
        back = load_checkpoint(str(out))["model"]
        self.assertEqual(set(back), set(self.state))
        for k, v in self.state.items():
            self.assertEqual(back[k].dtype, v.dtype, k)
            self.assertTrue(torch.equal(back[k], v), k)
        self.assertIn("float32 on disk", (out / "config.json").read_text(encoding="utf-8"))

    def test_default_export_passes_its_own_parity_check(self):
        out, log = self.export("mixed")
        self.assertIn("parity: reloaded", log)
        self.assertIn("tokenizer agrees", log)

    def test_parity_check_catches_a_changed_tensor(self):
        from anulm.hf.export_hf import check_parity, to_bf16
        out, _ = self.export("tamper")
        want = to_bf16(self.state)
        k = max(want, key=lambda k: want[k].numel())               # the largest tensor, whatever its dtype
        want[k] = want[k] + 1
        with self.assertRaises(SystemExit) as cm:
            check_parity(out, want, {}, None, None)
        self.assertIn(k, str(cm.exception))


class LayoutTests(unittest.TestCase):
    def test_root_model_is_the_package_module(self):
        import model as legacy
        from anulm import model as native
        self.assertIs(legacy, native)

    def test_a_checkpoint_pickled_under_the_old_module_name_loads(self):
        # 89 local checkpoints pickle `model.AnuLMConfig` from the flat layout.
        import model  # noqa: F401  the alias must be importable for this to work
        from anulm import model as native
        cfg = native.AnuLMConfig()
        buf = io.BytesIO()
        old = native.AnuLMConfig.__module__
        try:
            native.AnuLMConfig.__module__ = "model"
            torch.save({"cfg": cfg}, buf)
        finally:
            native.AnuLMConfig.__module__ = old
        buf.seek(0)
        restored = torch.load(buf, weights_only=False)["cfg"]
        self.assertIsInstance(restored, native.AnuLMConfig)
        self.assertEqual(restored.hidden_size, cfg.hidden_size)

    def test_the_installed_command_works_outside_the_checkout(self):
        import os
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}     # the install, not a path hack
        with tempfile.TemporaryDirectory() as work:
            where = subprocess.run([sys.executable, "-c", "import anulm; print(anulm.__file__)"], cwd=work,
                                   capture_output=True, text=True, timeout=60, env=env)
            if where.returncode or Path(where.stdout.strip()).resolve().parents[1] != ROOT:
                self.skipTest("anulm is not installed from this checkout (`pip install -e .`)")
            r = subprocess.run([sys.executable, "-m", "anulm", "--help"], cwd=work,
                               capture_output=True, text=True, timeout=60, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("train", r.stdout)


if __name__ == "__main__":
    unittest.main()
