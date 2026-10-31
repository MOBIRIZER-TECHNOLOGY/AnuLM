"""
Chapter 14 checks: one model for everything, and debugging it.

    python ch14.py                  # your course/build/post.py
    python ch14.py --impl solution  # the reference solution
    python ch14.py --self-test      # proves each check catches the bug it is for

A multi-task run reports one number, and that number can improve while a
task gets worse. These checks hold the step, the score and the release gates
to omni_train.py and omni_one.py, and replay the run where the score hid a
regression.
"""

from __future__ import annotations

import copy
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, main, solution, with_replaced  # noqa: E402

spec = importlib.util.spec_from_file_location("omni_one", ROOT / "experiments" / "omni_one.py")
omni_one = importlib.util.module_from_spec(spec)
spec.loader.exec_module(omni_one)

CHECKS = []
B = solution(8)
KW = dict(vocab_size=64, block_size=32, n_layer=2, hidden_size=32, n_head=4, head_dim=8, n_kv_head=2,
          num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16, intermediate_size=64,
          rope_theta=10000.0, bias_update_rate=0.01)
CURVE = ROOT / "ckpt_omni_curve.csv"          # the first AnuLM-Omni run, step 0 to 100,000


def check(fn):
    CHECKS.append(fn)
    return fn


@check
def tasks_are_drawn_like_omni_train(M):
    """Same generator, same shares, same sequence of tasks as omni_train.py's loop."""
    shares = [0.25, 0.14, 0.06, 0.12, 0.32, 0.17, 0.12]          # omni_one.TASKS
    a, b = np.random.default_rng(7), np.random.default_rng(7)
    w = np.array(shares) / sum(shares)
    got = [M.pick_task(shares, a) for _ in range(2000)]
    want = [int(b.choice(len(w), p=w)) for _ in range(2000)]
    assert got == want, "the sequence of tasks differs from omni_train's draws"


def model():
    torch.manual_seed(0)
    return B.MoELM(**KW).train()


@check
def an_accumulated_step_is_one_large_batch(M):
    """Four batches summed into one step move the weights exactly as one batch four times the size."""
    torch.manual_seed(1)
    parts = [torch.randint(0, 64, (2, 17)) for _ in range(4)]
    a, b = model(), model()
    opt_a, opt_b = torch.optim.SGD(a.parameters(), lr=0.5), torch.optim.SGD(b.parameters(), lr=0.5)
    it = iter(parts)
    M.omni_step(a, opt_a, lambda: (lambda p: (p[:, :-1], p[:, 1:]))(next(it)), accum=4, clip=1e9)
    big = torch.cat(parts)
    _, loss = b(big[:, :-1], big[:, 1:])
    loss.backward()
    opt_b.step()
    b.update_expert_biases()
    for (n, p), q in zip(a.named_parameters(), b.parameters()):
        torch.testing.assert_close(p, q, rtol=1e-4, atol=1e-6, msg=f"{n} differs from one large batch's step")
    for (n, x), y in zip(a.named_buffers(), b.buffers()):
        torch.testing.assert_close(x, y, msg=f"buffer {n} differs: the balancer was not moved like a normal step")


@check
def the_step_leaves_no_gradient_behind(M):
    a = model()
    opt = torch.optim.SGD(a.parameters(), lr=0.1)
    x = torch.randint(0, 64, (2, 17))
    M.omni_step(a, opt, lambda: (x[:, :-1], x[:, 1:]), accum=2)
    left = [n for n, p in a.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert not left, f"gradients left after the step ({left[:2]}): the next step would add to them"


@check
def gradients_are_clipped_once_over_the_sum(M):
    """Clipping applies to the accumulated gradient, after the last batch, never to each batch."""
    torch.manual_seed(2)
    parts = [torch.randint(0, 64, (2, 17)) for _ in range(2)]
    a = model()
    norms = []
    real = torch.nn.utils.clip_grad_norm_

    def spy(params, max_norm, *args, **kw):
        params = list(params)
        norms.append(torch.norm(torch.stack([p.grad.norm() for p in params if p.grad is not None])).item())
        return real(params, max_norm, *args, **kw)
    torch.nn.utils.clip_grad_norm_ = spy
    try:
        it = iter(parts)
        M.omni_step(a, torch.optim.SGD(a.parameters(), lr=0.0), lambda: (lambda p: (p[:, :-1], p[:, 1:]))(next(it)), 2)
    finally:
        torch.nn.utils.clip_grad_norm_ = real
    b = model()
    big = torch.cat(parts)
    b(big[:, :-1], big[:, 1:])[1].backward()
    want = torch.norm(torch.stack([p.grad.norm() for p in b.parameters() if p.grad is not None])).item()
    assert len(norms) == 1, f"clipped {len(norms)} times in one step"
    assert abs(norms[0] - want) < 1e-4 * max(1, want), "clipped a gradient that was not the whole step's"


def curve_rows():
    with open(CURVE, encoding="utf-8") as f:
        return list(csv.DictReader(f))


@check
def score_matches_omni_train(M):
    """Recompute the first Omni run's logged score from its logged losses."""
    rows = curve_rows()
    tasks = [k for k in rows[0] if k not in ("step", "score")]
    first = {k: float(rows[0][k]) for k in tasks}
    for r in rows[1:]:
        got = M.score({k: float(r[k]) for k in tasks}, first)
        assert abs(got - float(r["score"])) < 2e-4, f"step {r['step']}: score {got:.4f}, the run logged {r['score']}"


@check
def gates_match_omni_one(M):
    rules = omni_one.GATES
    cases = [{"wer_en": 12.2, "wer_hi_fleurs": 48.8, "wer_hi_kathbath": 35.5, "wer_hi_mucs": 54.4,
              "chat_heldout_loss": 5.3181, "read_squad_en": 24.6, "intent_banking77": 90.0, "vqa_acc": 17.6},
             {"wer_en": 9.0, "wer_hi_fleurs": 43.0, "wer_hi_kathbath": 50.0, "wer_hi_mucs": 60.0,
              "chat_heldout_loss": 3.30, "read_squad_en": 30.0, "intent_banking77": 80.0, "vqa_acc": 10.0},
             {"wer_en": 7.0, "chat_heldout_loss": 3.1}]
    for c in cases:
        got, want = M.gates(dict(c), rules), omni_one.gates(dict(c))
        assert got == want, f"gates {got} differ from omni_one's {want} for {c}"


@check
def a_gate_fails_when_one_task_regresses_even_if_the_score_improves(M):
    """The first Omni run, as logged: the score fell from 1.0 to 0.587, a 41% improvement, while chat's
    held-out loss rose from 3.27 to 5.39. The score must say "better" and the chat gate "fail"."""
    rows = curve_rows()
    tasks = [k for k in rows[0] if k not in ("step", "score")]
    first = {k: float(rows[0][k]) for k in tasks}
    last = {k: float(rows[-1][k]) for k in tasks}
    assert M.score(last, first) < 0.6, "the score does not show the improvement the run logged"
    g = M.gates({"chat_heldout_loss": last["chat"]}, {"chat_heldout_loss": ("<=", 3.30)})
    assert g == {"chat_heldout_loss": False}, "the chat gate passed a loss of 5.39"


# --- broken versions for --self-test ----------------------------------------------------

NAMES = ("pick_task", "omni_step", "score", "gates")


def MUTANTS(sol):
    def uniform(shares, rng):
        return int(rng.choice(len(shares)))

    def no_divide(model, opt, draw, accum=1, clip=1.0):
        losses = []
        for _ in range(accum):
            x, y = draw()
            _, loss = model(x, y)
            loss.backward()
            losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        model.update_expert_biases()
        return losses

    def no_balancer(model, opt, draw, accum=1, clip=1.0):
        rate, model.bias_update_rate = model.bias_update_rate, 0.0
        try:
            return sol.omni_step(model, opt, draw, accum, clip)
        finally:
            model.bias_update_rate = rate

    def no_clear(model, opt, draw, accum=1, clip=1.0):
        losses = []
        for _ in range(accum):
            x, y = draw()
            _, loss = model(x, y)
            (loss / accum).backward()
            losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        model.update_expert_biases()
        return losses

    def clip_each(model, opt, draw, accum=1, clip=1.0):
        losses = []
        for _ in range(accum):
            x, y = draw()
            _, loss = model(x, y)
            (loss / accum).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            losses.append(loss.item())
        opt.step()
        opt.zero_grad(set_to_none=True)
        model.update_expert_biases()
        return losses

    def raw_mean(losses, first):
        return float(np.mean(list(losses.values()))) / float(np.mean(list(first.values())))

    def missing_passes(results, rules):
        res = dict(results)
        out = {}
        for k, (op, thr) in rules.items():
            v = res.get(k)
            out[k] = v is None or (v <= thr if op == "<=" else v >= thr)
        return out

    def no_hindi_mean(results, rules):
        out = {}
        for k, (op, thr) in rules.items():
            v = results.get(k)
            out[k] = v is not None and (v <= thr if op == "<=" else v >= thr)
        return out

    def strict(results, rules):
        res = dict(results)
        hi = [res.get(k) for k in ("wer_hi_fleurs", "wer_hi_kathbath", "wer_hi_mucs")]
        if all(v is not None for v in hi):
            res["wer_hi_mean"] = round(sum(hi) / 3, 1)
        return {k: res.get(k) is not None and (res[k] < thr if op == "<=" else res[k] > thr)
                for k, (op, thr) in rules.items()}

    bugs = [("tasks drawn uniformly, ignoring the shares", dict(pick_task=uniform)),
            ("accumulated losses not divided", dict(omni_step=no_divide)),
            ("the balancer not moved", dict(omni_step=no_balancer)),
            ("gradients not cleared after the step", dict(omni_step=no_clear)),
            ("each batch clipped on its own", dict(omni_step=clip_each)),
            ("score as a ratio of raw means", dict(score=raw_mean)),
            ("a missing measurement passes its gate", dict(gates=missing_passes)),
            ("the Hindi mean never computed", dict(gates=no_hindi_mean)),
            ("a value exactly at the threshold fails", dict(gates=strict))]
    return [(d, with_replaced(sol, NAMES, **r)) for d, r in bugs]


if __name__ == "__main__":
    main(14, CHECKS, MUTANTS, __doc__, learner="post.py")
