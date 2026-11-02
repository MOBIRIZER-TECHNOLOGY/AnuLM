"""Focused tests for demo_test.py; no checkpoint or encoder is downloaded."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from anulm.apps import evaluate as demo_test


class FakeEngine:
    def __init__(self, prediction="New Delhi!"):
        self.prediction = prediction
        self.info = {"ckpt": "fake", "device": "cpu"}
        self.calls = []

    def generate(self, question, max_tokens, temperature, top_k, seed, mode="continue",
                 repetition_penalty=1.0):
        self.calls.append((question, max_tokens, temperature, top_k, seed, mode, repetition_penalty))
        return {"completion": self.prediction, "tokens": 2, "seconds": 0.01,
                "stopped_at_eos": True, "mode": mode}


class FakeOmni:
    def __init__(self):
        self.engine = FakeEngine()
        self.proj = object()
        self.vision_proj = object()
        self.meta = {"tower": "fake-tower"}

    def ear(self):
        return mock.Mock(hear=mock.Mock(return_value={
            "text": "hello world", "language": "en", "seconds": 0.02}))

    def eye(self):
        return object(), object(), object()


class DemoTestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_jsonl(self, rows, name="rows.jsonl"):
        path = self.root / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def test_text_scores_normalize_exact_match_and_token_f1(self):
        scores = demo_test.text_scores("New Delhi!", "new delhi")
        self.assertEqual(scores, {"exact_match": 1.0, "f1": 1.0})
        partial = demo_test.text_scores("the blue dog", "the dog")
        self.assertEqual(partial["exact_match"], 0.0)
        self.assertAlmostEqual(partial["f1"], 0.8)
        self.assertEqual(demo_test.text_scores("anything", None),
                         {"exact_match": None, "f1": None})

    def test_jsonl_validation_and_batch_cap(self):
        path = self.write_jsonl([{"question": "Capital?", "answer": "Delhi"}])
        self.assertEqual(demo_test.read_text_jsonl(str(path)),
                         [{"question": "Capital?", "answer": "Delhi"}])
        too_many = self.write_jsonl(
            [{"question": f"Q{i}", "answer": "A"} for i in range(3)], "many.jsonl")
        with self.assertRaisesRegex(ValueError, "more than 2 rows"):
            demo_test.read_text_jsonl(str(too_many), limit=2)
        bad = self.root / "bad.jsonl"
        bad.write_text('{"question": "missing answer"}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "question and answer"):
            demo_test.read_text_jsonl(str(bad))
        invalid = self.root / "invalid.jsonl"
        invalid.write_text("not json\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            demo_test.read_text_jsonl(str(invalid))

    def test_generation_and_source_validation(self):
        local = self.root / 'folder with spaces' / 'model.pt'
        local.parent.mkdir()
        local.write_bytes(b'path validation only')
        self.assertEqual(demo_test.validate_source(str(local)), str(local.resolve()))
        settings = demo_test.generation_settings(32, 0, 40, 0, 1.0)
        self.assertEqual(settings["max_tokens"], 32)
        with self.assertRaisesRegex(ValueError, "max tokens"):
            demo_test.generation_settings(demo_test.MAX_NEW_TOKENS + 1, 0, 40, 0, 1)
        with self.assertRaisesRegex(ValueError, "finite"):
            demo_test.generation_settings(32, float("nan"), 40, 0, 1)
        with self.assertRaisesRegex(ValueError, "does not exist"):
            demo_test.validate_source("definitely-missing.pt")
        self.assertEqual(demo_test.validate_source("org/model"), "org/model")
        with self.assertRaisesRegex(ValueError, "image generation"):
            demo_test.vision_token_limit(demo_test.MAX_VISION_TOKENS + 1)

    def test_manager_loads_lazily_and_reuses_one_shared_omni(self):
        manager = demo_test.ModelManager()
        model = FakeOmni()
        self.assertFalse(manager.loaded)
        with mock.patch("anulm.multimodal.omni.Omni", return_value=model) as constructor:
            first = manager.load("org/model", "cpu")
            second = manager.load("org/model", "cpu")
        self.assertIs(first, second)
        self.assertTrue(manager.loaded)
        constructor.assert_called_once_with("org/model", "cpu")
        info = demo_test.model_metadata(model)
        self.assertEqual(info["modalities"], {"text": True, "asr": True, "vision": True})

    def test_missing_index_error_is_actionable(self):
        manager = demo_test.ModelManager()
        with mock.patch("anulm.multimodal.omni.Omni", side_effect=FileNotFoundError("model.safetensors.index.json")):
            with self.assertRaisesRegex(RuntimeError, "index.*missing|file.*index.*missing"):
                manager.load("org/model", "cpu")

    def test_single_text_scores_and_writes_download(self):
        manager = demo_test.ModelManager()
        model = FakeOmni()
        with mock.patch("anulm.multimodal.omni.Omni", return_value=model):
            result, download = demo_test.run_text(
                manager, "org/model", "cpu", "Capital?", "new delhi", "question", 32, 0, 1, 7, 1)
        self.assertEqual(result["exact_match"], 1.0)
        self.assertEqual(result["f1"], 1.0)
        self.assertEqual(model.engine.calls[0][-2:], ("question", 1.0))
        payload = json.loads(Path(download).read_text(encoding="utf-8"))
        self.assertEqual(payload["results"][0]["prediction"], "New Delhi!")
        Path(download).unlink()

    def test_batch_uses_same_model_and_aggregates_scores(self):
        manager = demo_test.ModelManager()
        model = FakeOmni()
        rows = self.write_jsonl([
            {"question": "Capital?", "answer": "New Delhi"},
            {"question": "Again?", "answer": "Something else"},
        ])
        with mock.patch("anulm.multimodal.omni.Omni", return_value=model) as constructor:
            payload, download = demo_test.run_text_batch(
                manager, "org/model", "cpu", str(rows), "question", 32, 0, 1, 0, 1)
        constructor.assert_called_once()
        self.assertEqual(len(model.engine.calls), 2)
        self.assertEqual(payload["summary"]["rows"], 2)
        self.assertEqual(payload["summary"]["exact_match"], 0.5)
        self.assertEqual(len(payload["results"]), 2)
        Path(download).unlink()

    def test_asr_wer_and_missing_audio_projector(self):
        audio = self.root / "clip.wav"
        audio.write_bytes(b"fake audio; the ASR model is mocked")
        manager = demo_test.ModelManager()
        model = FakeOmni()
        with mock.patch("anulm.multimodal.omni.Omni", return_value=model):
            result, download = demo_test.run_asr(manager, "org/model", "cpu", str(audio), "Hello, world!")
        self.assertEqual(result["transcription"], "hello world")
        self.assertEqual(result["wer"], 0.0)
        Path(download).unlink()

        model.proj = None
        with mock.patch("anulm.multimodal.omni.Omni", return_value=model):
            with self.assertRaisesRegex(ValueError, "audio projector"):
                demo_test.run_asr(demo_test.ModelManager(), "org/model", "cpu", str(audio), "hello")

    def test_vision_caption_and_vqa_dispatch(self):
        image = self.root / "image.png"
        image.write_bytes(b"fake image; the vision model is mocked")
        manager = demo_test.ModelManager()
        model = FakeOmni()
        with mock.patch("anulm.multimodal.omni.Omni", return_value=model), \
                mock.patch("anulm.multimodal.vision_encoder.caption", return_value="a dog") as caption:
            result, download = demo_test.run_vision(
                manager, "org/model", "cpu", str(image), "caption", "", "A dog.", 16)
        self.assertEqual(result["exact_match"], 1.0)
        caption.assert_called_once()
        Path(download).unlink()

        with mock.patch("anulm.multimodal.omni.Omni", return_value=model), \
                mock.patch("anulm.multimodal.vision_encoder.answer", return_value="yes") as answer:
            result, download = demo_test.run_vision(
                demo_test.ModelManager(), "org/model", "cpu", str(image), "vqa", "Is this a dog?", "yes", 8)
        self.assertEqual(result["prediction"], "yes")
        answer.assert_called_once()
        Path(download).unlink()

    def test_ui_constructs_without_loading_and_cli_is_localhost_only(self):
        manager = demo_test.ModelManager()
        with mock.patch("anulm.multimodal.omni.Omni") as constructor:
            app = demo_test.build(manager)
        self.assertIsNotNone(app)
        self.assertFalse(manager.loaded)
        constructor.assert_not_called()
        args = demo_test.make_parser().parse_args([])
        self.assertEqual(args.host, "127.0.0.1")
        self.assertFalse(args.share)


if __name__ == "__main__":
    unittest.main()
