import copy
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from anulm.training import train
from anulm.training import train_from_scratch as pipeline


ROOT = Path(__file__).resolve().parents[1]


class DocumentSplitTests(unittest.TestCase):
    def test_duplicate_groups_never_cross_splits_and_train_is_tokenizer_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [f"unique-marker-{i} " + ("document text " * 20) for i in range(8)]
            corpus = root / "corpus.txt"
            corpus.write_text("\n\n\n".join([docs[0], docs[1], docs[0], *docs[2:]]),
                              encoding="utf-8", newline="\n")
            metadata_path = pipeline.prepare_document_splits(corpus, root / "splits", 7)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["mode"], "document-grouped")
            self.assertEqual(set(metadata["splits"]), {"train", "validation", "test"})
            self.assertEqual(metadata["documents"], {"total": 9, "unique": 8})

            locations = {}
            for name in metadata["splits"]:
                text = (root / "splits" / f"{name}.txt").read_text(encoding="utf-8")
                for i, doc in enumerate(docs):
                    if f"unique-marker-{i}" in text:
                        locations.setdefault(i, set()).add(name)
            self.assertTrue(all(len(names) == 1 for names in locations.values()))
            duplicate_split = locations[0].pop()
            duplicate_text = (root / "splits" / f"{duplicate_split}.txt").read_text(encoding="utf-8")
            self.assertEqual(duplicate_text.count("unique-marker-0"), 2)
            pipeline.validate_document_splits(metadata_path)

    def test_large_corpus_uses_proportional_holdouts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.txt"
            corpus.write_text("\n\n\n".join(f"document {i}" for i in range(20)), encoding="utf-8")
            path = pipeline.prepare_document_splits(corpus, root / "splits", 0)
            metadata = json.loads(path.read_text(encoding="utf-8"))
            counts = {name: record["documents"] for name, record in metadata["splits"].items()}
            self.assertEqual(counts, {"train": 16, "validation": 2, "test": 2})

    def test_single_unique_document_preserves_legacy_one_step_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.txt"
            corpus.write_text("same document\n\n\nsame document\n\n\n", encoding="utf-8")
            path = pipeline.prepare_document_splits(corpus, root / "splits", 0)
            metadata = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["mode"], "legacy-single-document")
            self.assertEqual(list(metadata["splits"]), ["train"])

    def test_changed_source_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.txt"
            corpus.write_text("\n\n\n".join(f"document {i}" for i in range(4)), encoding="utf-8")
            path = pipeline.prepare_document_splits(corpus, root / "splits", 0)
            corpus.write_text("changed input", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "source corpus changed"):
                pipeline.validate_document_splits(path)

    def test_encoded_manifest_loads_separate_validation_and_test_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_metadata = root / "splits.json"
            split_metadata.write_text("{}", encoding="utf-8")
            encoded = {}
            expected = {}
            for number, name in enumerate(("train", "validation", "test")):
                path = root / f"{name}.bin"
                ids = torch.arange(number * 20, number * 20 + 20, dtype=torch.int16)
                torch.save({"ids": ids, "vocab_size": 100, "tokenizer": str(root / "bpe.json"),
                            "bytes_per_token": 2.0}, path)
                encoded[name] = path
                expected[name] = ids
            manifest = root / "corpus.splits.json"
            pipeline.create_encoded_manifest(split_metadata, encoded, manifest)
            train_ids, val_ids, test_ids, meta = train.load_encoded_splits(str(manifest))
            self.assertTrue(torch.equal(train_ids, expected["train"]))
            self.assertTrue(torch.equal(val_ids, expected["validation"]))
            self.assertTrue(torch.equal(test_ids, expected["test"]))
            self.assertEqual(meta["vocab_size"], 100)
            encoded["test"].write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "encoded test split changed"):
                train.load_encoded_splits(str(manifest))


class SafetyTests(unittest.TestCase):
    def test_completed_stage_rejects_changed_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "artifact"
            output.write_text("complete", encoding="utf-8")
            state_path = root / "pipeline.json"
            state = {"version": pipeline.STATE_VERSION, "stages": {}}
            identity = {"source_sha256": "old"}
            pipeline.save_stage(state_path, state, "encode", identity, output)
            self.assertTrue(pipeline.completed(state, "encode", identity, output, lambda path: None))
            with self.assertRaisesRegex(RuntimeError, "different recipe"):
                pipeline.completed(state, "encode", {"source_sha256": "new"}, output,
                                   lambda path: None)

    def test_duplicate_run_lock_is_exclusive_and_reusable(self):
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / ".lock"
            with pipeline.RunLock(lock_path):
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    with pipeline.RunLock(lock_path):
                        pass
            with pipeline.RunLock(lock_path):
                pass

    def test_disk_budget_is_checked_before_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(pipeline.shutil, "disk_usage", return_value=SimpleNamespace(free=10)):
                with self.assertRaisesRegex(RuntimeError, "only"):
                    pipeline.ensure_disk_space(Path(directory), 11, "encode")

    def test_run_without_log_executes_command(self):
        pipeline.run([sys.executable, "-c", "print('console output')"], timeout=10)

    def test_run_with_log_captures_command_output(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "nested" / "success.log"
            pipeline.run([sys.executable, "-c", "print('logged output')"], log_path=log, timeout=10)
            self.assertEqual(log.read_text(encoding="utf-8").strip(), "logged output")

    def test_timeout_terminates_command_and_keeps_log(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "timeout.log"
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                pipeline.run([sys.executable, "-c", "import time; print('started', flush=True); time.sleep(30)"],
                             log_path=log, timeout=0.2)
            self.assertLess(time.monotonic() - started, 10)
            self.assertIn("started", log.read_text(encoding="utf-8"))

    def test_dry_run_reads_no_input_and_creates_no_work_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "not-created"
            result = subprocess.run(
                [sys.executable, "-m", "anulm.training.train_from_scratch", "--corpus",
                 str(Path(directory) / "missing.txt"), "--work-dir", str(work), "--dry-run",
                 "--device", "cpu"],
                cwd=ROOT, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("splits/train.txt", result.stdout.replace("\\", "/"))
            self.assertFalse(work.exists())


class SplitValidationTests(unittest.TestCase):
    def test_short_splits_fail_with_actionable_messages(self):
        train_ids = torch.arange(64)
        with self.assertRaisesRegex(RuntimeError, "training split has 8 tokens.*at least 9"):
            train.validate_split_lengths(torch.arange(8), None, None, 8)
        with self.assertRaisesRegex(RuntimeError, "validation split has 9 tokens.*at least 10"):
            train.validate_split_lengths(train_ids, torch.arange(9), None, 8)
        with self.assertRaisesRegex(RuntimeError, "test split has 9 tokens.*at least 10"):
            train.validate_split_lengths(train_ids, torch.arange(10), torch.arange(9), 8)
        train.validate_split_lengths(train_ids, torch.arange(10), torch.arange(10), 8)

    def test_zero_window_training_split_fails_before_logging(self):
        with self.assertRaisesRegex(RuntimeError, "zero windows.*block size 8.*--block-size"):
            train.WindowSampler(9, 8, 2, 0)


class PreflightTests(unittest.TestCase):
    @staticmethod
    def make_model():
        from anulm.model import AnuLM, AnuLMConfig

        config = AnuLMConfig(vocab_size=64, block_size=8, n_layer=1, hidden_size=32,
                             n_head=4, head_dim=8, n_kv_head=2, num_experts=2,
                             num_experts_per_tok=1, num_shared_experts=1,
                             moe_intermediate_size=16, intermediate_size=64,
                             first_k_dense_replace=0)
        return AnuLM(config)

    @staticmethod
    def preflight_args():
        return SimpleNamespace(block_size=8, batch_size=2, device="cpu",
                               autocast=torch.autocast("cpu", enabled=False),
                               report_gpu_memory=False)

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for a, e in zip(actual, expected):
                self.assert_nested_equal(a, e)
        elif torch.is_tensor(expected):
            self.assertTrue(torch.equal(actual, expected))
        else:
            self.assertEqual(actual, expected)

    def router_buffers(self, model):
        return {name: buffer.detach().clone() for name, buffer in model.named_buffers()
                if name.endswith(("load_counts", "expert_bias"))}

    def test_real_data_forward_backward_has_finite_loss_and_gradients(self):
        torch.manual_seed(0)
        model = self.make_model()
        model.eval()
        data = torch.randint(0, 64, (64,), dtype=torch.int16)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        before_buffers = self.router_buffers(model)
        before_rng = torch.random.get_rng_state().clone()
        self.assertTrue(before_buffers)
        report = train.real_data_preflight(model, data, self.preflight_args())
        self.assertGreater(report["gradient_tensors"], 0)
        self.assertTrue(torch.isfinite(torch.tensor(report["loss"])))
        self.assertTrue(torch.isfinite(torch.tensor(report["gradient_norm"])))
        for parameter, original in zip(model.parameters(), before):
            self.assertTrue(torch.equal(parameter, original))
        for name, buffer in self.router_buffers(model).items():
            self.assertTrue(torch.equal(buffer, before_buffers[name]), name)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before_rng))
        self.assertFalse(model.training)
        self.assertFalse(hasattr(model, "aux_loss"))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_preflight_restores_resumed_router_and_optimizer_state(self):
        torch.manual_seed(0)
        data = torch.randint(0, 64, (64,), dtype=torch.int16)
        source = self.make_model()
        router = next(m for m in source.modules() if isinstance(m, train.Router))
        with torch.no_grad():
            router.expert_bias.copy_(torch.tensor([100.0, -100.0]))
        source_opt = torch.optim.AdamW(source.parameters(), lr=1e-3)
        x, y = train.make_batch(data, torch.arange(2), 8, "cpu")
        _, loss = source(x, y)
        loss.backward()
        source_opt.step()
        source_opt.zero_grad(set_to_none=True)
        source.update_expert_biases()

        model = self.make_model()
        model.load_state_dict(source.state_dict())
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        opt.load_state_dict(source_opt.state_dict())
        sentinel = torch.tensor(17.0)
        model.aux_loss = sentinel
        before_buffers = self.router_buffers(model)
        before_opt = copy.deepcopy(opt.state_dict())
        before_params = [parameter.detach().clone() for parameter in model.parameters()]

        train.real_data_preflight(model, data, self.preflight_args())

        for name, buffer in self.router_buffers(model).items():
            self.assertTrue(torch.equal(buffer, before_buffers[name]), name)
        self.assert_nested_equal(opt.state_dict(), before_opt)
        for parameter, original in zip(model.parameters(), before_params):
            self.assertTrue(torch.equal(parameter, original))
        self.assertIs(model.aux_loss, sentinel)
        self.assertTrue(model.training)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_nonfinite_preflight_loss_is_rejected(self):
        class BadModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))

            def forward(self, x, y):
                return None, self.weight * torch.tensor(float("nan"))

        args = SimpleNamespace(block_size=4, batch_size=1, device="cpu",
                               autocast=torch.autocast("cpu", enabled=False),
                               report_gpu_memory=False)
        with self.assertRaisesRegex(FloatingPointError, "non-finite loss"):
            train.real_data_preflight(BadModel(), torch.arange(20, dtype=torch.int16), args)


if __name__ == "__main__":
    unittest.main()
