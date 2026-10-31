"""
Chapter 8 checks: generation, the schedule, the optimizer, accumulation, checkpoints.

    python ch08.py                  # your course/build/anulm.py
    python ch08.py --impl solution  # the reference solution
    python ch08.py --self-test      # proves each check catches the bug it is for

Pre-training runs for days. The bugs here don't show in a five-minute test: a
resume that silently resets the optimizer, a save that a crash can corrupt, a
schedule that is a little off. Each one costs a run.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import sys
import tempfile
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import main, with_replaced  # noqa: E402
import model as ref  # noqa: E402
import train as ref_train  # noqa: E402  the repository's train.py

CHECKS = []
KW = dict(vocab_size=64, block_size=24, n_layer=3, hidden_size=32, n_head=4, head_dim=8, n_kv_head=2,
          num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16, intermediate_size=64,
          rope_theta=10000.0)


def check(fn):
    CHECKS.append(fn)
    return fn


def models(M):
    torch.manual_seed(0)
    mine = M.MoELM(**KW).eval()
    theirs = ref.AnuLM(ref.AnuLMConfig(**KW)).eval()
    theirs.load_state_dict(mine.state_dict())
    return mine, theirs


# --- generation ------------------------------------------------------------------------------

@check
def generate_matches_model(M):
    """Same weights, same seed: exactly model.py's tokens, with top-k, a repetition penalty,
    a temperature, without the cache, and past block_size (where the cache must be rebuilt)."""
    mine, theirs = models(M)
    p = torch.randint(0, 64, (2, 10), generator=torch.Generator().manual_seed(3))
    for kw in (dict(), dict(top_k=5, repetition_penalty=1.3), dict(repetition_penalty=2.0),   # no top-k: every
               dict(temperature=0.6), dict(use_cache=False, top_k=8)):                     # penalised logit counts
        torch.manual_seed(5)
        a = mine.generate(p, 30, **kw)
        torch.manual_seed(5)
        b = theirs.generate(p, 30, **kw)
        assert torch.equal(a, b), f"different tokens from model.py with {kw or 'defaults'}"


@check
def generation_stops_at_eos_like_model(M):
    """With eos_id: finished rows keep receiving EOS, and generation stops once every row is done."""
    mine, theirs = models(M)
    p = torch.randint(0, 64, (3, 6), generator=torch.Generator().manual_seed(4))
    for seed in range(200):                                   # a seed where EOS really appears
        torch.manual_seed(seed)
        b = theirs.generate(p, 40, eos_id=7)
        if (b[:, 6:] == 7).any():
            break
    torch.manual_seed(seed)
    a = mine.generate(p, 40, eos_id=7)
    assert torch.equal(a, b), "EOS handling differs from model.py (finished rows, or when to stop)"


# --- the schedule and the optimizer --------------------------------------------------------

@check
def lr_at_matches_train_py(M):
    for sched in ("cosine", "wsd"):
        args = types.SimpleNamespace(lr=3e-4, min_lr=3e-5, warmup=10, steps=100, schedule=sched, decay_frac=0.25)
        for step in range(0, 100):
            mine = M.lr_at(step, 100, 3e-4, 3e-5, 10, sched, 0.25)
            want = ref_train.lr_at(step, args)
            assert abs(mine - want) < 1e-12, f"{sched} at step {step}: {mine:.3e}, train.py gives {want:.3e}"


@check
def weight_decay_spares_norms(M):
    """Decaying RMSNorm gains pulls every one of them towards 0 all run long; the model trains
    anyway, against a constant drag on its own normalisation."""
    torch.manual_seed(0)
    m = M.MoELM(**KW)
    opt = M.make_optimizer(m, 3e-4, 0.1)
    with contextlib.redirect_stdout(io.StringIO()):            # train.py prints a line when it builds one
        want = ref_train.make_optimizer(m, types.SimpleNamespace(weight_decay=0.1, lr=3e-4, device="cpu",
                                                                 optimizer="adamw")).opts[0]
    for g_mine, g_want in zip(opt.param_groups, want.param_groups):
        assert {id(p) for p in g_mine["params"]} == {id(p) for p in g_want["params"]} \
            and g_mine["weight_decay"] == g_want["weight_decay"], \
            "parameter groups differ from train.py's (matrices decayed at 0.1, 1-D parameters at 0)"
    assert opt.param_groups[0]["betas"] == (0.9, 0.95) and opt.param_groups[0]["eps"] == 1e-8, \
        f"betas {opt.param_groups[0]['betas']}, eps {opt.param_groups[0]['eps']}: want (0.9, 0.95), 1e-8"


# --- accumulation and the step --------------------------------------------------------------

def batches(n, size, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [(x, x) for x in torch.randint(0, 64, (n, size, 16), generator=g)]


@check
def accumulation_gives_large_batch_gradients(M):
    """Four micro-batches of 2 must give the gradients of one batch of 8. Forgetting to divide
    by 4 is hidden by AdamW and clipping (see the chapter), so this compares gradients."""
    torch.manual_seed(0)
    m = M.MoELM(**KW).train()
    small = batches(4, 2)
    M.accumulate(m, small)
    acc = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True)
    x = torch.cat([b[0] for b in small])
    m(x, x)[1].backward()
    for n, p in m.named_parameters():
        if p.grad is not None:
            torch.testing.assert_close(acc[n], p.grad, rtol=1e-4, atol=1e-6,
                                       msg=f"{n}: accumulated gradient differs from the large batch's")


@check
def train_step_updates_the_balancer(M):
    """After the optimizer step, the balancing bias moves and the load counts reset."""
    torch.manual_seed(0)
    m = M.MoELM(**KW).train()
    opt = M.make_optimizer(m, 1e-3)
    r = next(mod for mod in m.modules() if type(mod).__name__ == "Router")
    before = r.expert_bias.clone()
    M.train_step(m, opt, batches(2, 4), 1e-3)
    assert not torch.equal(before, r.expert_bias), "the balancing bias did not move"
    assert r.load_counts.sum().item() == 0, "load counts were not reset by the update"


# --- checkpoints ---------------------------------------------------------------------------

def run(M, steps, start=0, m=None, opt=None):
    if m is None:
        torch.manual_seed(0)
        m = M.MoELM(**KW).train()
        opt = M.make_optimizer(m, 1e-3)
    for step in range(start, steps):
        M.train_step(m, opt, batches(2, 4, seed=100 + step), M.lr_at(step, 6, 1e-3, 1e-4, 2))
    return m, opt


@check
def resume_is_exact(M):
    """Six steps straight must equal three steps, a save, a fresh process, a load, and three
    more: same weights, same balancing biases, same optimizer state. Losing the optimizer's
    moments still trains, with a dent that looks like noise."""
    straight, _ = run(M, 6)
    m, opt = run(M, 3)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "run.last"
        M.save_checkpoint(path, m, opt, step=2)
        torch.manual_seed(123)                                  # a different start, as after a crash
        m2 = M.MoELM(**KW).train()
        opt2 = M.make_optimizer(m2, 1e-3)
        ck = M.load_checkpoint(path, m2, opt2)
        resumed, _ = run(M, 6, start=ck["step"] + 1, m=m2, opt=opt2)
    for (n, a), b in zip(straight.state_dict().items(), resumed.state_dict().values()):
        assert torch.equal(a, b), f"{n} differs after resuming"


@check
def a_crash_mid_save_keeps_the_old_checkpoint(M):
    """torch.save straight onto the file leaves it truncated if the process dies mid-write;
    the resume point is then gone. Write elsewhere and rename."""
    torch.manual_seed(0)
    m = M.MoELM(**KW)
    opt = M.make_optimizer(m, 1e-3)

    class Unpicklable:
        def __reduce__(self):
            raise RuntimeError("simulated crash while writing")

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "run.last"
        M.save_checkpoint(path, m, opt, step=41)
        try:
            M.save_checkpoint(path, m, opt, step=42, crash=Unpicklable())
        except Exception:  # noqa: BLE001 -- the crash we asked for
            pass
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: BLE001
            raise AssertionError(f"the previous checkpoint was destroyed by the failed save ({type(e).__name__})")
        assert ck["step"] == 41, f"the checkpoint holds step {ck['step']}, expected the old 41"


# --- broken versions for --self-test -------------------------------------------------------

NAMES = ("MoELM", "lr_at", "make_optimizer", "accumulate", "train_step", "save_checkpoint", "load_checkpoint")


def MUTANTS(sol):
    import torch.nn.functional as F

    def generate_variant(**bug):
        class Buggy(sol.MoELM):
            @torch.no_grad()
            def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=None, use_cache=True,
                         repetition_penalty=1.0):
                B, bs = idx.shape[0], self.block_size
                done = torch.zeros(B, dtype=torch.bool)
                cache, start, pending = None, 0, idx
                for _ in range(max_new_tokens):
                    if not use_cache:
                        logits, _ = self(idx[:, -bs:])
                    else:
                        if cache is None or (start + pending.shape[1] > bs and not bug.get("no_rebuild")):
                            cache, start, pending = self.new_cache(), 0, idx[:, -bs:]
                        logits, _ = self(pending, cache=cache, start=start)
                        start += pending.shape[1]
                    logits = logits[:, -1, :].float()
                    if repetition_penalty != 1.0:
                        seen = logits.gather(1, idx)
                        seen = seen / repetition_penalty if bug.get("always_divide") else \
                            torch.where(seen > 0, seen / repetition_penalty, seen * repetition_penalty)
                        logits = logits.scatter(1, idx, seen)
                    logits = logits / max(temperature, 1e-5)
                    if top_k is not None and not bug.get("no_top_k"):
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = -float("inf")
                    nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                    if eos_id is not None and not bug.get("eos_ignored"):
                        nxt = torch.where(done[:, None], torch.full_like(nxt, eos_id), nxt)
                    idx = torch.cat((idx, nxt), dim=1)
                    pending = nxt
                    if eos_id is not None and not bug.get("eos_ignored"):
                        done |= nxt[:, 0] == eos_id
                        if bool(done.all()):
                            break
                return idx
        return Buggy

    def wsd_wrong_start(step, steps, lr, min_lr, warmup, schedule="cosine", decay_frac=0.2):
        return sol.lr_at(step, steps, lr, min_lr, warmup, schedule, 1 - decay_frac)

    def warmup_from_zero(step, steps, lr, min_lr, warmup, schedule="cosine", decay_frac=0.2):
        return lr * step / max(warmup, 1) if step < warmup else sol.lr_at(step, steps, lr, min_lr, warmup, schedule,
                                                                           decay_frac)

    def cosine_to_zero(step, steps, lr, min_lr, warmup, schedule="cosine", decay_frac=0.2):
        return sol.lr_at(step, steps, lr, 0.0, warmup, schedule, decay_frac)

    def decay_everything(model, lr, weight_decay=0.1):
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay)

    def default_betas(model, lr, weight_decay=0.1):
        opt = sol.make_optimizer(model, lr, weight_decay)
        for g in opt.param_groups:
            g["betas"] = (0.9, 0.999)
        return opt

    def no_division(model, bs):
        model.zero_grad(set_to_none=True)
        tot = 0.0
        for x, y in bs:
            _, loss = model(x, y)
            loss.backward()
            tot += loss.item() / len(bs)
        return tot

    def no_balancer(model, opt, bs, lr, clip=1.0):
        for g in opt.param_groups:
            g["lr"] = lr
        loss = sol.accumulate(model, bs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        return loss

    def save_in_place(path, model, opt, step, **extra):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, **extra}, path)

    def save_without_optimizer(path, model, opt, step, **extra):
        tmp = Path(str(path) + ".tmp")
        torch.save({"model": model.state_dict(), "step": step, **extra}, tmp)
        os.replace(tmp, path)

    def load_without_optimizer(path, model, opt):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        return ck

    def save_parameters_only(path, model, opt, step, **extra):
        tmp = Path(str(path) + ".tmp")
        torch.save({"model": {n: p.detach() for n, p in model.named_parameters()}, "opt": opt.state_dict(),
                    "step": step, **extra}, tmp)
        os.replace(tmp, path)

    def load_non_strict(path, model, opt):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        opt.load_state_dict(ck["opt"])
        return ck

    bugs = [("repetition penalty dividing negative logits too", dict(MoELM=generate_variant(always_divide=True))),
            ("top_k ignored", dict(MoELM=generate_variant(no_top_k=True))),
            ("finished rows not held at EOS", dict(MoELM=generate_variant(eos_ignored=True))),
            ("cache not rebuilt past block_size", dict(MoELM=generate_variant(no_rebuild=True))),
            ("WSD decaying at the wrong point", dict(lr_at=wsd_wrong_start)),
            ("warmup starting at lr 0", dict(lr_at=warmup_from_zero)),
            ("cosine ending at 0, not min_lr", dict(lr_at=cosine_to_zero)),
            ("weight decay on the norms too", dict(make_optimizer=decay_everything)),
            ("AdamW's default betas (0.9, 0.999)", dict(make_optimizer=default_betas)),
            ("accumulation without dividing", dict(accumulate=no_division)),
            ("balancing bias never updated", dict(train_step=no_balancer)),
            ("checkpoint written in place", dict(save_checkpoint=save_in_place)),
            ("optimizer state not saved", dict(save_checkpoint=save_without_optimizer,
                                               load_checkpoint=load_without_optimizer)),
            ("buffers (balancing bias) not saved", dict(save_checkpoint=save_parameters_only,
                                                        load_checkpoint=load_non_strict))]
    return [(d, with_replaced(sol, NAMES, **r)) for d, r in bugs]


if __name__ == "__main__":
    main(8, CHECKS, MUTANTS, __doc__)
