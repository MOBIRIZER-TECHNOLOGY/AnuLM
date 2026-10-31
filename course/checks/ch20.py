"""
Chapter 20 checks: data-parallel training, from scratch.

    python ch20.py                  # your course/build/further.py
    python ch20.py --impl solution  # the reference solution
    python ch20.py --self-test      # proves each check catches the bug it is for

The checks run two copies of the model in two threads that talk through a small
in-process stand-in for torch.distributed. A copy that skips a call waits for
the other forever, so every collective here gives up after a few seconds and
reports a hang.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import main, solution, with_replaced  # noqa: E402

CHECKS = []
B = solution(8)
KW = dict(vocab_size=64, block_size=32, n_layer=2, hidden_size=32, n_head=4, head_dim=8, n_kv_head=2,
          num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16, intermediate_size=64,
          rope_theta=10000.0, bias_update_rate=0.01)


class Hang(RuntimeError):
    pass


class ThreadComm:
    """torch.distributed's two calls, for ranks that are threads of one process."""

    def __init__(self, rank, shared):
        self.rank, self.world, self.s = rank, shared["world"], shared

    def _sync(self):
        try:
            self.s["barrier"].wait(timeout=5)
        except threading.BrokenBarrierError:
            raise Hang("the ranks made different calls and would wait for each other forever") from None

    def all_reduce_sum(self, t):
        self.s["slots"][self.rank] = t.detach().clone()
        self._sync()
        total = sum(self.s["slots"])
        self._sync()
        t.copy_(total)

    def broadcast(self, t, src):
        self.s["slots"][self.rank] = t.detach().clone()
        self._sync()
        value = self.s["slots"][src].clone()
        self._sync()
        t.copy_(value)


def run_ranks(world, fn):
    """fn(rank, comm) in `world` threads; returns their results, or raises what any of them raised."""
    shared = {"world": world, "slots": [None] * world, "barrier": threading.Barrier(world)}
    torch.set_num_threads(1)              # two ranks in threads: keep each one's own math single-threaded
    results, errors = [None] * world, []

    def body(r):
        try:
            results[r] = fn(r, ThreadComm(r, shared))
        except Exception as e:  # noqa: BLE001
            errors.append(e)
            shared["barrier"].abort()
    threads = [threading.Thread(target=body, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return results


def check(fn):
    CHECKS.append(fn)
    return fn


@check
def every_example_is_read_once_per_epoch(M):
    parts = [M.shard(103, r, 4, seed=7, epoch=2) for r in range(4)]
    flat = [i for p in parts for i in p]
    assert sorted(flat) == list(range(103)), "the ranks' shares overlap or miss examples"
    assert max(map(len, parts)) - min(map(len, parts)) <= 1, "the shares are uneven"
    assert M.shard(103, 0, 4, 7, 3) != parts[0], "the same order every epoch"


@check
def every_copy_starts_from_rank_0(M):
    models = []
    for rank in range(2):                                # built before the threads: they share one RNG
        torch.manual_seed(rank)                          # deliberately different starting weights
        models.append(B.MoELM(**KW))
    with torch.no_grad():                                # and different expert biases, as a resumed copy might have
        for name, buf in models[1].named_buffers():
            if name.endswith("expert_bias"):
                buf.normal_()

    def fn(rank, comm):
        M.broadcast_params(models[rank], comm)
        return models[rank].state_dict()
    a, b = run_ranks(2, fn)
    torch.manual_seed(0)
    want = B.MoELM(**KW).state_dict()
    for k in want:
        assert torch.equal(a[k], want[k]) and torch.equal(b[k], want[k]), f"{k}: not rank 0's starting value"


def batches(steps=3):
    g = torch.Generator().manual_seed(5)
    return [torch.randint(0, 64, (4, 17), generator=g) for _ in range(steps)]


def single(clip, steps=3):
    torch.manual_seed(0)
    m = B.MoELM(**KW).train()
    opt = torch.optim.SGD(m.parameters(), lr=0.5)
    for full in batches(steps):
        opt.zero_grad(set_to_none=True)
        _, loss = m(full[:, :-1], full[:, 1:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), clip)
        opt.step()
        m.update_expert_biases()
    return m.state_dict()


@check
def two_copies_take_the_same_steps_as_one_big_batch(M):
    """Two ranks, each with half of every batch, end with exactly the weights AND expert biases one
    process gets from the whole batch: without clipping (which would hide a gradient twice too big,
    as in Chapter 8), and with it."""
    for clip in (1e9, 1.0):
        models = []
        for _ in range(2):
            torch.manual_seed(0)
            models.append(B.MoELM(**KW).train())

        def fn(rank, comm):
            m = models[rank]
            opt = torch.optim.SGD(m.parameters(), lr=0.5)
            for full in batches():
                half = full[rank * 2: rank * 2 + 2]
                M.ddp_step(m, opt, half[:, :-1], half[:, 1:], comm, clip)
            return m.state_dict()
        a, b = run_ranks(2, fn)
        want = single(clip)
        for k in want:
            assert torch.allclose(a[k].float(), b[k].float(), atol=1e-6), f"{k}: the two copies disagree"
            assert torch.allclose(a[k].float(), want[k].float(), rtol=1e-4, atol=1e-5), \
                f"{k}: differs from one process stepping on the whole batch (clip {clip:g})"


@check
def an_expert_nobody_used_does_not_hang_the_run(M):
    """Rank 1's tokens never reach some experts, so those experts have no gradient there."""
    models = []
    for _ in range(2):
        torch.manual_seed(0)
        models.append(B.MoELM(**KW).train())
    xs = [torch.randint(0, 64, (2, 16)), torch.zeros(1, 2, dtype=torch.long)]

    def fn(rank, comm):
        m = models[rank]
        opt = torch.optim.SGD(m.parameters(), lr=0.1)
        M.ddp_step(m, opt, xs[rank], xs[rank], comm)
        return {k: v.clone() for k, v in m.state_dict().items()}
    a, b = run_ranks(2, fn)
    for k in a:
        assert torch.allclose(a[k].float(), b[k].float(), atol=1e-6), f"{k}: the copies drifted apart"


# --- broken versions for --self-test ----------------------------------------------------

NAMES = ("shard", "broadcast_params", "all_reduce_grads", "ddp_step")


def MUTANTS(sol):
    def same_data(n, rank, world, seed, epoch):
        return sol.shard(n, 0, world, seed, epoch)

    def own_order(n, rank, world, seed, epoch):
        return sol.shard(n, rank, world, seed + rank, epoch)

    def every_epoch_alike(n, rank, world, seed, epoch):
        return sol.shard(n, rank, world, seed, 0)

    def no_broadcast(model, comm):
        return None

    def params_only(model, comm):
        with torch.no_grad():
            for p in model.parameters():
                comm.broadcast(p.data, 0)

    @torch.no_grad()
    def summed(model, comm):
        for p in model.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            comm.all_reduce_sum(p.grad)

    @torch.no_grad()
    def skips_unused(model, comm):
        for p in model.parameters():
            if p.grad is not None:
                comm.all_reduce_sum(p.grad)
                p.grad /= comm.world

    def counts_not_synced(model, opt, x, y, comm, clip=1.0):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        sol.all_reduce_grads(model, comm)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        model.update_expert_biases()
        return loss.item()

    def clip_before_sync(model, opt, x, y, comm, clip=1.0):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        sol.all_reduce_grads(model, comm)
        with torch.no_grad():
            for name, buf in model.named_buffers():
                if name.endswith("load_counts"):
                    comm.all_reduce_sum(buf)
        opt.step()
        model.update_expert_biases()
        return loss.item()

    def no_grad_sync(model, opt, x, y, comm, clip=1.0):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        with torch.no_grad():
            for name, buf in model.named_buffers():
                if name.endswith("load_counts"):
                    comm.all_reduce_sum(buf)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        model.update_expert_biases()
        return loss.item()

    bugs = [("every rank reads the same examples", dict(shard=same_data)),
            ("each rank shuffles with its own seed", dict(shard=own_order)),
            ("the same order every epoch", dict(shard=every_epoch_alike)),
            ("no broadcast: copies start different", dict(broadcast_params=no_broadcast)),
            ("buffers not broadcast", dict(broadcast_params=params_only)),
            ("gradients summed, not averaged", dict(all_reduce_grads=summed)),
            ("unused experts skipped (a hang)", dict(all_reduce_grads=skips_unused)),
            ("expert load counts not shared", dict(ddp_step=counts_not_synced)),
            ("clipped before averaging", dict(ddp_step=clip_before_sync)),
            ("gradients never shared", dict(ddp_step=no_grad_sync))]
    out = []
    for d, r in bugs:
        ns = with_replaced(sol, NAMES, **r)
        if "all_reduce_grads" in r:                       # ddp_step must call the broken one
            def step(model, opt, x, y, comm, clip=1.0, _f=r["all_reduce_grads"]):
                opt.zero_grad(set_to_none=True)
                _, loss = model(x, y)
                loss.backward()
                _f(model, comm)
                with torch.no_grad():
                    for name, buf in model.named_buffers():
                        if name.endswith("load_counts"):
                            comm.all_reduce_sum(buf)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
                model.update_expert_biases()
                return loss.item()
            ns.ddp_step = step
        out.append((d, ns))
    return out


if __name__ == "__main__":
    main(20, CHECKS, MUTANTS, __doc__, learner="further.py")
