"""
Speed benchmark for AnuLM-Decide (decide.py).

    python experiments/decide_speed.py --device cuda      # in a pause of the main run
    python experiments/decide_speed.py --device cpu

Measures, for the trained BANKING77 checkpoint:
  1. cold start: loading the model
  2. latency against the number of options (2 .. 150) at batch 1
  3. throughput (queries/s) and peak memory at batch 1 .. 64, 77 options
  4. the ceiling for a cached-options redesign: one pass over the message alone
     (~25 tokens). The current model reads options after the message, so their
     work cannot be reused; this is what a model that encoded options once
     could at best approach. An estimate, not a measurement of a built model.

Each latency is the median of repeated runs after warm-up, with the GPU
synchronised around the timed region.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from decide import batchify, encode, labels, load, load_split  # noqa: E402


def sync(dev):
    if dev.startswith("cuda"):
        torch.cuda.synchronize()


def timed(fn, dev, reps):
    for _ in range(3):
        fn()
    times = []
    for _ in range(reps):
        sync(dev)
        t0 = time.perf_counter()
        fn()
        sync(dev)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_decide.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev, gpu = a.device, a.device.startswith("cuda")
    reps = 30 if gpu else 8
    res = {"device": dev, "gpu": torch.cuda.get_device_name(0) if gpu else "cpu"}

    t0 = time.perf_counter()
    d, tok, _ = load(str(ROOT / a.ckpt), dev)
    d.eval()
    sync(dev)
    res["cold_start_s"] = round(time.perf_counter() - t0, 2)
    print(f"cold start (load model to {dev}): {res['cold_start_s']} s", flush=True)

    # a pool of distinct option names for counts beyond 77
    extra = json.load(open(ROOT / "data" / "decide_general" / "clinc" / "intent_names.json"))
    pool = labels() + [n for n in extra if n not in labels() and n != "oos"]
    msgs = [t for t, _ in load_split("test")[:64]]
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=gpu)

    def run_batch(examples):
        x, pos, _ = batchify(examples, tok, d.pad_id)
        with torch.no_grad(), ac:
            return d(x.to(dev), pos.to(dev))

    print("\nlatency vs number of options (batch 1, median ms):", flush=True)
    res["latency_by_options"] = {}
    for k in (2, 5, 10, 20, 50, 77, 100, 150):
        opts = pool[:k]
        try:
            ids, pos = encode(tok, msgs[0], opts)
        except AssertionError:
            print(f"  {k:4d} options: exceeds the 1,024-token context")
            break
        ms = timed(lambda: run_batch([(ids, pos, 0, opts)]), dev, reps)
        res["latency_by_options"][k] = {"ms": round(ms, 1), "tokens": len(ids)}
        print(f"  {k:4d} options, {len(ids):4d} tokens: {ms:7.1f} ms", flush=True)

    print("\nthroughput, 77 options:", flush=True)
    res["throughput"] = {}
    opts = labels()
    for bs in ((1, 8, 32, 64) if gpu else (1, 8)):
        batch = []
        for m in msgs[:bs]:
            ids, pos = encode(tok, m, opts)
            batch.append((ids, pos, 0, opts))
        if gpu:
            torch.cuda.reset_peak_memory_stats()
        try:
            ms = timed(lambda: run_batch(batch), dev, max(5, reps // 3))
        except torch.OutOfMemoryError:
            print(f"  batch {bs:3d}: out of memory")
            break
        peak = torch.cuda.max_memory_allocated() / 2**30 if gpu else None
        qps = bs / (ms / 1000)
        res["throughput"][bs] = {"ms_per_batch": round(ms, 1), "queries_per_s": round(qps, 1),
                                 "peak_gb": round(peak, 2) if peak else None}
        print(f"  batch {bs:3d}: {ms:7.1f} ms per batch, {qps:7.1f} queries/s"
              + (f", peak {peak:.2f} GB" if peak else ""), flush=True)

    print("\nceiling for a cached-options design (one pass over the message alone):", flush=True)
    ids = [tok.eos_id] + tok.encode(f"Message: {msgs[0]}")
    x = torch.tensor([ids], device=dev)

    def message_only():
        with torch.no_grad(), ac:
            return d.hidden(x)
    ms = timed(message_only, dev, reps)
    full = res["latency_by_options"].get(77, {}).get("ms")
    res["message_only"] = {"ms": round(ms, 1), "tokens": len(ids),
                           "possible_speedup_vs_77": round(full / ms, 1) if full else None}
    print(f"  {len(ids)} tokens: {ms:.1f} ms" + (f"  (vs {full} ms with 77 options in the prompt: "
                                                 f"up to {full / ms:.1f}x if options were encoded once)" if full else ""))

    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
