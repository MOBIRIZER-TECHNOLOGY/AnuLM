"""
Regression tests for safe loading and input validation.

    python -m unittest tests.test_quality_loading

Each test names the bug it pins. Nothing here needs model weights, a GPU or a
download: checkpoints are tiny ones written to a temporary folder, and the
Hub is replaced by a stand-in.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

import anulm.model as model_mod
from anulm.bpe import BPE
from anulm.model import AnuLM, AnuLMConfig, load_checkpoint

try:
    from anulm.model import UnsafeCheckpointError, safe_load
except ImportError:                     # the code before the fix: the tests below fail, not error
    UnsafeCheckpointError, safe_load = None, None


def tiny(**kw) -> AnuLMConfig:
    """The same small shape tests/test_model.py uses."""
    base = dict(vocab_size=64, block_size=32, n_layer=2, hidden_size=32, n_head=4,
                head_dim=8, n_kv_head=2, num_experts=4, num_experts_per_tok=2,
                moe_intermediate_size=16, intermediate_size=64, n_group=1, topk_group=1)
    base.update(kw)
    return AnuLMConfig(**base)


class Payload:
    """An arbitrary object: what a hostile .pt would carry. Unpickling it with
    a full unpickler runs __setstate__; the safe loader must refuse it."""
    ran = False

    def __setstate__(self, state):
        Payload.ran = True
        self.__dict__.update(state)


class SafeLoad(unittest.TestCase):
    """torch.load(weights_only=False) ran arbitrary code from any .pt,
    including ones downloaded from the Hub."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        Payload.ran = False

    def tearDown(self):
        self.tmp.cleanup()

    def test_project_checkpoint_shape_loads(self):
        """Weights, config, optimizer state and sampler generator state: what a
        `.last` resume point holds, all inside the allow-list."""
        self.assertIsNotNone(safe_load, "anulm.model.safe_load is missing")
        m = AnuLM(tiny())
        opt = torch.optim.AdamW(m.parameters())
        m(torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 4, dtype=torch.long))[1].backward()
        opt.step()
        g = torch.Generator().manual_seed(0)
        path = self.d / "ck.pt.last"
        torch.save({"model": m.state_dict(), "cfg": m.cfg, "opt": [opt.state_dict()], "step": 3,
                    "best_val": float("inf"), "sampler": {"epoch": 0, "pos": 2,
                                                          "order": torch.arange(5), "g": g.get_state()}},
                   path)
        ck = safe_load(path)
        self.assertEqual(ck["cfg"], m.cfg)
        self.assertEqual(ck["step"], 3)
        self.assertTrue(torch.equal(ck["model"]["lm_head.weight"], m.state_dict()["lm_head.weight"]))

    def test_legacy_config_names_load(self):
        """Checkpoints written before the package layout pickle the config as
        model.AnuLMConfig (the root alias): it must stay loadable."""
        self.assertIsNotNone(safe_load, "anulm.model.safe_load is missing")
        for name in ("AnuLMConfig",):
            cfg = tiny()
            saved = (AnuLMConfig.__module__, AnuLMConfig.__qualname__, AnuLMConfig.__name__)
            had = sys.modules.get("model")
            try:
                sys.modules["model"] = model_mod
                AnuLMConfig.__module__, AnuLMConfig.__qualname__, AnuLMConfig.__name__ = "model", name, name
                torch.save({"cfg": cfg}, self.d / f"{name}.pt")
            finally:
                AnuLMConfig.__module__, AnuLMConfig.__qualname__, AnuLMConfig.__name__ = saved
                if had is None:
                    sys.modules.pop("model", None)
                else:
                    sys.modules["model"] = had
            # the pickle (stored uncompressed in the zip) names the legacy class
            self.assertIn(name.encode(), (self.d / f"{name}.pt").read_bytes())
            ck = safe_load(self.d / f"{name}.pt")
            self.assertIsInstance(ck["cfg"], AnuLMConfig)
            self.assertEqual(ck["cfg"].hidden_size, 32)

    def test_arbitrary_object_is_refused_with_the_escape_named(self):
        self.assertIsNotNone(safe_load, "anulm.model.safe_load is missing")
        torch.save({"x": Payload()}, self.d / "evil.pt")
        Payload.ran = False
        with patch.dict(os.environ, {model_mod.TRUST_PICKLE_ENV: ""}):
            with self.assertRaises(UnsafeCheckpointError) as cm:
                safe_load(self.d / "evil.pt")
            self.assertFalse(Payload.ran, "the object was unpickled anyway")
            self.assertIn("trust_pickle=True", str(cm.exception))
            self.assertIn("ANULM_TRUST_PICKLE", str(cm.exception))
            # load_checkpoint is the path every tool uses: it must refuse too
            with self.assertRaises(UnsafeCheckpointError):
                load_checkpoint(self.d / "evil.pt")
            self.assertFalse(Payload.ran)

    def test_explicit_escapes_load_a_trusted_legacy_file(self):
        self.assertIsNotNone(safe_load, "anulm.model.safe_load is missing")
        torch.save({"x": Payload()}, self.d / "legacy.pt")
        self.assertIsInstance(safe_load(self.d / "legacy.pt", trust_pickle=True)["x"], Payload)
        with patch.dict(os.environ, {model_mod.TRUST_PICKLE_ENV: "1"}):
            self.assertIsInstance(safe_load(self.d / "legacy.pt")["x"], Payload)

    def test_lora_merge_refuses_an_unsafe_adapter(self):
        """lora merge read the adapter with a full unpickle."""
        from types import SimpleNamespace
        from anulm import lora
        torch.save({"lora": {}, "x": Payload()}, self.d / "adapter.pt")
        with patch.dict(os.environ, {model_mod.TRUST_PICKLE_ENV: ""}):
            with self.assertRaises(Exception) as cm:
                lora._merge_cli(SimpleNamespace(adapter=str(self.d / "adapter.pt"), base=None,
                                                out=str(self.d / "out.pt")))
        self.assertFalse(Payload.ran, "lora merge unpickled an arbitrary object")
        self.assertIn("ANULM_TRUST_PICKLE", str(cm.exception))


class FolderContainment(unittest.TestCase):
    """A downloaded folder's config.json could point tokenizer_path,
    extras.hf_backbone or extras.base_model anywhere on the machine."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def folder(self, name="repo", config=None, extras=None, weights=False):
        p = self.root / name
        p.mkdir()
        meta = {"config": config if config is not None else {}, "extras": extras or {}}
        if weights:
            from safetensors.torch import save_file
            cfg = tiny()
            meta["config"] = {"vocab_size": 64, "block_size": 32, "n_layer": 2, "hidden_size": 32,
                              "n_head": 4, "head_dim": 8, "n_kv_head": 2, "num_experts": 4,
                              "num_experts_per_tok": 2, "moe_intermediate_size": 16,
                              "intermediate_size": 64, "n_group": 1, "topk_group": 1}
            save_file({k: v.contiguous() for k, v in AnuLM(cfg).state_dict().items()},
                      str(p / "model.safetensors"))
        (p / "config.json").write_text(json.dumps(meta), encoding="utf-8")
        return p

    def test_tokenizer_path_escapes_are_refused(self):
        for bad in ("../outside.json", "/etc/passwd", "C:/Windows/win.ini", "a/../../outside.json",
                    "..\\outside.json"):
            p = self.folder(f"t{abs(hash(bad))}", config={"tokenizer_path": bad})
            with self.assertRaises(ValueError, msg=bad):
                load_checkpoint(p)

    def test_tokenizer_path_inside_is_kept(self):
        p = self.folder(config={"tokenizer_path": "tok/bpe.json"})
        ck = load_checkpoint(p)
        self.assertEqual(Path(ck["cfg"].tokenizer_path), p / "tok" / "bpe.json")

    def test_hf_backbone_escapes_are_refused(self):
        for bad in ("../elsewhere", "/opt/model"):
            p = self.folder(f"b{abs(hash(bad))}", extras={"hf_backbone": bad})
            with self.assertRaises(ValueError, msg=bad):
                load_checkpoint(p)
        ok = self.folder("good", extras={"hf_backbone": "backbone"})
        self.assertEqual(Path(load_checkpoint(ok)["hf_backbone"]), ok / "backbone")

    def test_base_model_must_be_a_hub_id(self):
        # a local path or a traversal must never be followed from a config.json
        side = self.folder("secret", weights=True)
        for bad in (str(side), "../secret", "a/..", "a/b/c"):
            p = self.folder(f"m{abs(hash(bad))}", extras={"base_model": bad})
            with self.assertRaises(ValueError, msg=bad):
                load_checkpoint(p)

    def test_base_model_side_by_side_still_loads(self):
        self.folder("AnuLM-Base", weights=True)
        p = self.folder("adapter", extras={"base_model": "toonist/AnuLM-Base"})
        ck = load_checkpoint(p)
        self.assertIn("lm_head.weight", ck["model"])
        self.assertEqual(ck["cfg"].hidden_size, 32)

    def test_hub_fallback_is_announced(self):
        """A mistyped local path turned silently into a Hub download."""
        target = self.folder("downloaded", config={})
        buf = io.StringIO()
        with patch("huggingface_hub.snapshot_download", return_value=str(target)) as dl, \
                redirect_stdout(buf):
            load_checkpoint("no-such-owner/no-such-model")
        dl.assert_called_once()
        self.assertIn("not found locally, trying the Hugging Face Hub", buf.getvalue())


class ValidationSurvivesOptimize(unittest.TestCase):
    """Input and file validation written as `assert` disappears under
    `python -O`; each check must raise ValueError instead."""

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            tiny(num_experts=16, n_group=4, topk_group=1, num_experts_per_tok=8)
        with self.assertRaises(ValueError):
            tiny(attn="nope")
        with self.assertRaises(ValueError):
            tiny(n_kv_head=3)

    def test_positions_beyond_block_size(self):
        m = AnuLM(tiny()).eval()
        with self.assertRaises(ValueError), torch.no_grad():
            m(torch.zeros(1, 33, dtype=torch.long))

    def test_foreign_tokenizer_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.json"
            p.write_text(json.dumps({"type": "gpt2", "merges": []}), encoding="utf-8")
            with self.assertRaises(ValueError):
                BPE.load(p)

    def test_decide_prompt_too_long(self):
        from anulm.tasks import decide
        tok = BPE([])
        with self.assertRaises(ValueError) as cm:
            decide.encode(tok, "x" * 50, ["a", "b"], max_len=20)
        self.assertIn("too many options for the context", str(cm.exception))

    def test_decide_uses_the_backbone_block_size(self):
        """The limit was a hard-coded 1024 whatever the model's context."""
        from anulm.tasks import decide
        d = decide.Decider(AnuLM(tiny(vocab_size=300)), "causal", 0)
        dec = decide.Decide.__new__(decide.Decide)
        dec.d, dec.tok, dec.device, dec.threshold, dec.labels = d, BPE([]), "cpu", 0.7, ["a", "b"]
        with self.assertRaises(ValueError) as cm:
            dec("a message long enough to pass thirty-two byte tokens", ["a", "b"])
        self.assertIn("too many options for the context", str(cm.exception))

    def test_decide_on_refuses_a_full_finetune(self):
        from anulm.tasks import decide
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "full.pt"
            torch.save({"mode": "full"}, path)
            with self.assertRaises(ValueError):
                decide.Decide.on(None, None, str(path), "cpu")


class ModelInit(unittest.TestCase):
    def test_attributes_exist_before_first_use(self):
        """aux_loss, imbalance_median and text_vocab only existed after the
        first forward / bias update / loader set them."""
        m = AnuLM(tiny())
        self.assertIsNone(m.aux_loss)
        self.assertIsNone(m.imbalance_median)
        self.assertIsNone(m.text_vocab)


class PointerMaxLen(unittest.TestCase):
    """rc_pointer.load() rewrote the module-global MAX_LEN, so loading a
    second reader changed how the first one truncated its input."""

    def test_two_readers_keep_their_own_max_len(self):
        from anulm.tasks import rc_pointer
        before = rc_pointer.MAX_LEN
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            tok = BPE([])
            tok.save(d / "bpe.json")
            cfg = tiny(vocab_size=257, tokenizer_path=str(d / "bpe.json"))
            m = AnuLM(cfg)
            head = rc_pointer.Pointer(m).head.state_dict()
            torch.save({"model": m.state_dict(), "cfg": cfg, "pointer_head": head, "max_len": 12},
                       d / "short.pt")
            torch.save({"model": m.state_dict(), "cfg": cfg}, d / "plain.pt")
            a, _, _ = rc_pointer.load(str(d / "short.pt"), "cpu")
            b, _, _ = rc_pointer.load(str(d / "plain.pt"), "cpu", max_len=20)
            self.assertEqual(rc_pointer.MAX_LEN, before, "load() changed the module global")
            self.assertEqual(getattr(a, "max_len", None), 12)
            self.assertEqual(getattr(b, "max_len", None), 20)
            ids, _, _ = rc_pointer.encode(tok, "Question", "q", "a long passage " * 10, max_len=a.max_len)
            self.assertEqual(len(ids), 12)


if __name__ == "__main__":
    unittest.main()
