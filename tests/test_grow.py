"""Tests for anulm/training/grow.py: a grown model must start as the same function.

    python -m unittest tests.test_grow

Growth that silently changes the function still trains (the loss just starts
higher and the pretraining it was meant to keep is partly lost), so these
check the property directly: identical logits before any step, gradients that
reach every new weight, and refusals for layouts that would break it.
"""
from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

import torch

from anulm.model import AnuLM, AnuLMConfig
from anulm.training import grow as G


def tiny(**kw) -> AnuLMConfig:
    base = dict(vocab_size=64, block_size=32, n_layer=6, hidden_size=32, n_head=4, head_dim=8,
                n_kv_head=2, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16,
                intermediate_size=64, n_group=1, topk_group=1, sliding_window=4,
                max_window_layers=2, first_k_dense_replace=1, moe_impl="sparse", use_qk_norm=True)
    base.update(kw)
    return AnuLMConfig(**base)


def trained_like(cfg: AnuLMConfig) -> AnuLM:
    """A tiny model with non-default weights everywhere, including expert biases."""
    torch.manual_seed(1)
    m = AnuLM(cfg)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
        for name, b in m.named_buffers():
            if name.endswith("expert_bias"):
                b.copy_(0.1 * torch.randn_like(b))
    return m.eval()


class GrowTests(unittest.TestCase):
    INSERT = [2, 4, 4]          # two new layers after old layer 4, one after 2

    def grown(self, width=40):
        old_cfg = tiny()
        old = trained_like(old_cfg)
        new_cfg = G.grown_config(old_cfg, width, self.INSERT)
        new = G.grow(old.state_dict(), old_cfg, new_cfg, self.INSERT).eval()
        return old, new, old_cfg, new_cfg

    def test_same_logits_at_start(self):
        old, new, _, new_cfg = self.grown()
        self.assertEqual(new_cfg.n_layer, 9)
        self.assertEqual(new_cfg.moe_intermediate_size, 40)
        idx = torch.randint(0, 64, (3, 24))       # longer than the sliding window
        with torch.no_grad():
            a, _ = old(idx)
            b, _ = new(idx)
        self.assertLess((a - b).abs().max().item(), 1e-5)
        self.assertGreater(a.abs().max().item(), 0.1)   # not trivially equal (all zeros)

    def test_more_parameters(self):
        old, new, _, _ = self.grown()
        self.assertGreater(new.num_params()[0], old.num_params()[0])
        self.assertGreater(new.num_params()[1], old.num_params()[1])

    def test_every_new_weight_gets_gradient(self):
        _, new, old_cfg, _ = self.grown()
        new.train()
        idx = torch.randint(0, 64, (4, 24))
        _, loss = new(idx, idx)
        loss.backward()
        I = old_cfg.moe_intermediate_size
        plan = G.layer_plan(old_cfg.n_layer, self.INSERT)
        # Widened experts: the zero new columns of down_proj must move.
        old_layer = next(i for i, o in enumerate(plan) if o is not None and o >= 1)
        downs = [new.layers[old_layer].mlp.experts[e].down_proj.weight.grad for e in range(8)]
        self.assertTrue(any(g is not None and g[:, I:].abs().sum() > 0 for g in downs))
        # Inserted layers: the zeroed output projection must move.
        for i, o in enumerate(plan):
            if o is None:
                g = new.layers[i].attn.dense.weight.grad
                self.assertIsNotNone(g)
                self.assertGreater(g.abs().sum().item(), 0)

    def test_new_layers_are_identity_and_keep_attention_types(self):
        _, new, old_cfg, new_cfg = self.grown()
        plan = G.layer_plan(old_cfg.n_layer, self.INSERT)
        windows = [blk.window for blk in new.layers]
        old_windows = [old_cfg.sliding_window if i < old_cfg.max_window_layers else None
                       for i in range(old_cfg.n_layer)]
        for i, o in enumerate(plan):
            if o is not None:
                self.assertEqual(windows[i], old_windows[o], f"layer {i} (old {o}) changed type")
            else:
                self.assertEqual(new.layers[i].attn.dense.weight.abs().sum().item(), 0)

    def test_refuses_layouts_that_break_it(self):
        with self.assertRaises(ValueError):
            G.grown_config(tiny(), 40, [0])          # would shift a windowed layer
        with self.assertRaises(ValueError):
            G.grown_config(tiny(), 8, [4])           # narrower experts
        with self.assertRaises(ValueError):
            G.layer_plan(6, [6])                     # no such layer

    def test_cfg_overrides_rebuild_the_config(self):
        old_cfg = tiny()
        new_cfg = G.grown_config(old_cfg, 40, self.INSERT)
        kw = {}
        for item in G.cfg_overrides(old_cfg, new_cfg):     # parsed exactly as train.py --cfg does
            key, _, val = item.partition("=")
            field = AnuLMConfig.__dataclass_fields__[key]
            cast = {"int": int, "float": float, "bool": lambda s: s.lower() in ("1", "true", "yes"),
                    "str": str}.get(str(field.type).replace("Optional[", "").rstrip("]"), str)
            kw[key] = None if val.lower() == "none" else cast(val)
        rebuilt = AnuLMConfig.base_350m(**kw, vocab_size=new_cfg.vocab_size,
                                       tokenizer_path=new_cfg.tokenizer_path,
                                       block_size=new_cfg.block_size,
                                       yarn_original_context=new_cfg.yarn_original_context,
                                       moe_impl=new_cfg.moe_impl)
        self.assertEqual(dataclasses.asdict(rebuilt), dataclasses.asdict(new_cfg))

    def test_cli_writes_a_resume_point(self):
        old_cfg = tiny()
        old = trained_like(old_cfg)
        with tempfile.TemporaryDirectory() as d:
            src, out = Path(d) / "small.pt", Path(d) / "big.pt.last"
            torch.save({"model": old.state_dict(), "cfg": old_cfg, "step": 99, "val_loss": 1.0}, src)
            import sys
            argv = sys.argv
            try:
                sys.argv = ["grow", str(src), str(out), "--experts-width", "40",
                            "--insert-after", *map(str, self.INSERT)]
                G.main()
            finally:
                sys.argv = argv
            ck = torch.load(out, weights_only=False)
            self.assertEqual(ck["step"], -1)                 # train.py --resume starts at 0
            self.assertNotIn("opt", ck)                      # fresh optimizer
            m = AnuLM(ck["cfg"])
            m.load_state_dict(ck["model"])                   # strict: every key matches


if __name__ == "__main__":
    unittest.main()
