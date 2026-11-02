"""train.py's safety checks: split lengths, and the real-data preflight before step 0.

From V3 (tests/test_training_pipeline.py); the tests of train_from_scratch.py stay
with that script."""

import copy
import unittest
from types import SimpleNamespace

import torch

from anulm.training import train


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
        from model import AnuLM, AnuLMConfig

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
