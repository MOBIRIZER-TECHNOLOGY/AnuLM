"""
Chapter 20 lab: two copies of the model, two processes, one training run.

    python lab.py                      # your course/build/further.py
    python lab.py --impl solution      # the reference solution

Needs ../ch08-pretrain/base.pt (for its sizes). CPU, about one minute. Real
torch.distributed: two processes on this machine, talking through the "gloo"
backend, exactly as two GPUs would through NCCL.

Part 1. Train Chapter 8's model from scratch for 150 steps, batches of 16
windows, three ways: one process; two processes with 8 each (ddp_step); and
two processes that share gradients but NOT the experts' load counts. Compare
the weights, the expert biases, and the time.

Part 2. In the run that didn't share load counts: how far apart are the two
copies' expert biases, how many routing choices differ on the same text, and
does the "same" model now score differently on each copy?
Writes results.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = Path(__file__).resolve().parent
COURSE = HERE.parent
STEPS, GLOBAL, CTX, LR = 150, 16, 128, 1e-3
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DistComm:
    """The two calls further.py uses, on torch.distributed."""

    def __init__(self):
        self.rank, self.world = dist.get_rank(), dist.get_world_size()

    def all_reduce_sum(self, t):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)

    def broadcast(self, t, src):
        dist.broadcast(t, src)


class OneComm:
    rank, world = 0, 1

    def all_reduce_sum(self, t):
        pass

    def broadcast(self, t, src):
        pass


def train(rank, world, impl, mode, rendezvous, out_dir, threads):
    torch.set_num_threads(threads)
    if world > 1:
        dist.init_process_group("gloo", init_method="file:///" + rendezvous.replace(os.sep, "/"),
                                rank=rank, world_size=world)
    sol = impl == "solution"
    lab1 = load(COURSE / "ch01-smallest-model" / "lab.py")
    Bm = load(COURSE / "build" / ("solutions/ch08.py" if sol else "anulm.py"))
    S = load(COURSE / "build" / ("solutions/ch20.py" if sol else "further.py"))
    sizes = torch.load(COURSE / "ch08-pretrain" / "base.pt", weights_only=False)["sizes"]
    train_data, val = lab1.load_bytes(str(COURSE.parent / "data" / "input.txt"))
    comm = DistComm() if world > 1 else OneComm()
    torch.manual_seed(100 + rank)                        # each copy starts different ...
    m = Bm.MoELM(**sizes).train()
    S.broadcast_params(m, comm)                          # ... until rank 0's weights are sent to all
    opt = Bm.make_optimizer(m, LR)
    g = torch.Generator().manual_seed(0)                 # every copy draws the same global batch
    t0 = time.time()
    for step in range(1, STEPS + 1):
        for grp in opt.param_groups:
            grp["lr"] = LR * min(1.0, step / 20)
        x, y = lab1.batch(train_data, CTX, GLOBAL, g)
        mine = S.shard(GLOBAL, rank, world, seed=0, epoch=step)  # this copy's share of it
        if mode == "no shared counts":
            opt.zero_grad(set_to_none=True)
            _, loss = m(x[mine], y[mine])
            loss.backward()
            S.all_reduce_grads(m, comm)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            m.update_expert_biases()
        else:
            S.ddp_step(m, opt, x[mine], y[mine], comm)
        if step == 1:
            after_one = {k: v.detach().clone() for k, v in m.state_dict().items()}
    secs = time.time() - t0
    m.eval()
    vx, _ = lab1.batch(val, CTX, 8, torch.Generator().manual_seed(1))
    seen = []
    hooks = [r.register_forward_hook(lambda mod, inp, o: seen.append(o[0].sort(-1).values))
             for r in m.modules() if type(r).__name__ == "Router"]
    with torch.no_grad():
        m(vx)
    for h in hooks:
        h.remove()
    torch.save({"state": m.state_dict(), "after_one": after_one, "routes": torch.cat([s.reshape(-1, s.shape[-1]) for s in seen]),
                "val": lab1.evaluate(m, val, CTX, torch.Generator().manual_seed(0)), "secs": secs},
               Path(out_dir) / f"{mode.replace(' ', '_')}_{rank}.pt")
    if world > 1:
        dist.destroy_process_group()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=["build", "solution"], default="build")
    a = p.parse_args()
    out_dir = tempfile.mkdtemp()
    threads = max(1, torch.get_num_threads())
    try:
        load(COURSE / "build" / ("solutions/ch20.py" if a.impl == "solution" else "further.py")).shard(4, 0, 1, 0, 0)
    except NotImplementedError as e:
        sys.exit(f"course/build/further.py is not finished yet ({e}). Run course/checks/ch20.py, or use --impl solution.")
    runs = [("one process", 1), ("two processes", 2), ("no shared counts", 2)]
    for mode, world in runs:
        rdv = os.path.join(tempfile.mkdtemp(), "rendezvous")
        if world == 1:
            train(0, 1, a.impl, mode, rdv, out_dir, threads)
        else:
            mp.spawn(train, args=(world, a.impl, mode, rdv, out_dir, max(1, threads // world)), nprocs=world)
        print(f"finished: {mode}")
    get = lambda mode, r: torch.load(Path(out_dir) / f"{mode.replace(' ', '_')}_{r}.pt", weights_only=False)
    one, two, loose = get("one process", 0), [get("two processes", r) for r in (0, 1)], \
        [get("no shared counts", r) for r in (0, 1)]

    def gap(a, b, buffers=False):
        keys = [k for k in a if k.endswith("expert_bias") == buffers and a[k].is_floating_point()]
        return max((a[k].float() - b[k].float()).abs().max().item() for k in keys)

    out = {"steps": STEPS, "global_batch": GLOBAL, "threads": threads, "part1": {}, "part2": {}}
    out["part1"] = {
        "one_vs_two_after_one_step": gap(one["after_one"], two[0]["after_one"]),
        "one_vs_two_weights": gap(one["state"], two[0]["state"]),
        "one_vs_two_biases": gap(one["state"], two[0]["state"], True),
        "two_ranks_weights": gap(two[0]["state"], two[1]["state"]),
        "val": {"one process": one["val"], "two processes": two[0]["val"]},
        "secs": {"one process": one["secs"], "two processes": max(t["secs"] for t in two)}}
    out["part2"] = {
        "ranks_weights": gap(loose[0]["state"], loose[1]["state"]),
        "ranks_biases": gap(loose[0]["state"], loose[1]["state"], True),
        "routing_differs": (loose[0]["routes"] != loose[1]["routes"]).any(-1).float().mean().item(),
        "val": [loose[0]["val"], loose[1]["val"]],
        "biases_vs_shared": gap(loose[0]["state"], two[0]["state"], True)}
    p1, p2 = out["part1"], out["part2"]
    print(f"\none process vs two: largest weight difference {p1['one_vs_two_after_one_step']:.2e} after one step, "
          f"{p1['one_vs_two_weights']:.2e} after {STEPS}, "
          f"expert biases {p1['one_vs_two_biases']:.2e}; the two copies differ by {p1['two_ranks_weights']:.2e}")
    print(f"validation loss {p1['val']['one process']:.4f} vs {p1['val']['two processes']:.4f}; "
          f"{STEPS} steps took {p1['secs']['one process']:.0f}s vs {p1['secs']['two processes']:.0f}s")
    print(f"without shared load counts: weights still differ by {p2['ranks_weights']:.2e}, but the expert biases "
          f"by {p2['ranks_biases']:.3f}; {p2['routing_differs']:.1%} of expert choices differ on the same text; "
          f"validation loss {p2['val'][0]:.4f} on copy 0, {p2['val'][1]:.4f} on copy 1")
    (HERE / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("saved results.json")


if __name__ == "__main__":
    main()
