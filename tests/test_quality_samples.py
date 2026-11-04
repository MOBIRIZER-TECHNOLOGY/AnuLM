"""The demo's sample library (examples/samples/) is complete and credited.

No model, no GPU, no network: it reads the manifest and checks the files it
names. A sample whose file went missing would vanish from the demo's gallery
without a word (app_omni.sample_library skips it), so the check is here.

    python -m unittest tests.test_quality_samples
"""

from __future__ import annotations

import json
import unittest
import wave
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "examples" / "samples"
TABS = {"listen", "look", "chat", "decide"}
KINDS = {"asr_en": "listen", "asr_hi": "listen", "caption": "look", "vqa": "look", "chat": "chat",
         "lookup": "chat", "intent_banking": "decide", "intent_massive_hi": "decide"}
# Not for commercial use: the one model leaves them out, so must its demo.
EXCLUDED_SOURCES = ("flickr", "coco", "a-okvqa", "aokvqa", "indictts")


class SampleLibrary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = json.loads((SAMPLES / "samples.json").read_text(encoding="utf-8"))["items"]

    def test_at_least_100_with_unique_ids(self):
        self.assertGreaterEqual(len(self.items), 100)
        ids = [i["id"] for i in self.items]
        self.assertEqual(len(ids), len(set(ids)), [k for k, n in Counter(ids).items() if n > 1])

    def test_tabs_and_kinds_are_known(self):
        for i in self.items:
            with self.subTest(i["id"]):
                self.assertIn(i["tab"], TABS)
                self.assertEqual(KINDS.get(i["kind"]), i["tab"])
                self.assertIn(i["language"], {"en", "hi"})
        self.assertEqual({i["tab"] for i in self.items}, TABS, "every tab has samples")

    def test_licence_attribution_and_source(self):
        for i in self.items:
            with self.subTest(i["id"]):
                for key in ("license", "attribution", "source"):
                    self.assertTrue(str(i.get(key) or "").strip(), key)
                self.assertFalse(any(x in i["source"].lower() for x in EXCLUDED_SOURCES), i["source"])
        credits = (SAMPLES / "ATTRIBUTION.md").read_text(encoding="utf-8")
        for i in self.items:
            if i["kind"] in ("vqa", "caption"):        # every photo is credited by file
                self.assertIn(i["inputs"]["image"], credits, i["id"])

    def test_inputs_and_references(self):
        for i in self.items:
            inp = i["inputs"]
            with self.subTest(i["id"]):
                if i["tab"] == "listen":
                    self.assertIn(inp["lang"], {"en", "hi"})
                elif i["tab"] == "look":
                    self.assertIn("question", inp)
                elif i["tab"] == "chat":
                    self.assertIn(inp["mode"], {"chat", "lookup"})
                    self.assertTrue(inp["question"].strip())
                else:
                    self.assertIn(inp["task"], {"banking", "massive-hi"})
                    self.assertTrue(inp["message"].strip())
                if i["kind"] != "chat":                 # everything but open chat has an answer to score
                    self.assertTrue(str(i.get("reference") or "").strip())

    def test_media_exist_and_are_small(self):
        for i in self.items:
            for key in ("audio", "image"):
                if key not in i["inputs"]:
                    continue
                p = SAMPLES / i["inputs"][key]
                with self.subTest(i["id"]):
                    self.assertTrue(p.is_file(), p)
                    self.assertEqual(p.resolve().parent, (SAMPLES / "media").resolve())
                    if key == "audio":
                        with wave.open(str(p)) as w:
                            self.assertEqual((w.getframerate(), w.getnchannels()), (16000, 1))
                    else:
                        self.assertEqual(p.read_bytes()[:2], b"\xff\xd8", "a JPEG")
        size = sum(p.stat().st_size for p in SAMPLES.rglob("*") if p.is_file())
        self.assertLess(size, 30e6, "the library ships in the repository")

    def test_mix(self):
        n = Counter(i["kind"] for i in self.items)
        for kind, least in {"asr_en": 20, "asr_hi": 20, "caption": 5, "vqa": 5, "chat": 10, "lookup": 5,
                            "intent_banking": 5, "intent_massive_hi": 5}.items():
            self.assertGreaterEqual(n[kind], least, kind)


if __name__ == "__main__":
    unittest.main()
