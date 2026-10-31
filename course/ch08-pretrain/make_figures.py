"""Draws Chapter 8's figures; the evidence ones from the lab's results.json.

    python lab.py && python make_figures.py
"""

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "figures"))
from fig import Fig, P  # noqa: E402

OUT = HERE / "figures"
RES = HERE / "results.json"
R = json.loads(RES.read_text(encoding="utf-8")) if RES.exists() else None


def fig_sampling():
    """8.1: from the last position's logits to one sampled token."""
    f = Fig(720, 210)
    steps = [("logits", "one per token", P.indigo_soft, P.indigo),
             ("repetition penalty", "seen tokens pushed down", P.paper, P.ink),
             ("÷ temperature", "<1 sharper, >1 flatter", P.paper, P.ink),
             ("top-k", "keep the k best", P.paper, P.ink),
             ("softmax, sample", "one token", P.saffron_soft, P.saffron)]
    x, w, gap = 16, 124, 14
    for i, (label, sub, fill, stroke) in enumerate(steps):
        f.box(x, 60, w, 56, label, fill, stroke, 12, sub=sub)
        if i < len(steps) - 1:
            f.arrow(x + w + 1, 88, x + w + gap - 1, 88, P.stone)
        x += w + gap
    f.text(16, 150, "Then append the token, feed only it through the KV cache, and repeat. When the cache is full,",
           12, P.ink)
    f.text(16, 168, "rebuild it from the last block_size tokens: exactly what re-reading them would give.", 12, P.ink)
    f.caption("Figure 8.1  One generation step, in model.py's order. The same seed gives the same text in your "
              "model and in model.py, which the checks prove.")
    f.save(OUT / "fig8_1_sampling.svg")


def fig_loop():
    """8.2: one optimizer step of the training loop, and what is saved."""
    f = Fig(720, 300)
    row = [("micro-batch 1 … n", P.teal_soft, P.teal), ("forward, loss ÷ n,\nbackward", P.indigo_soft, P.indigo),
           ("clip the\ngradient norm", P.paper, P.ink), ("AdamW step", P.saffron_soft, P.saffron),
           ("update expert\nbiases", P.paper, P.ink)]
    x, w = 16, 122
    for i, (label, fill, stroke) in enumerate(row):
        f.rect(x, 50, w, 64, fill, stroke)
        for k, line in enumerate(label.split("\n")):
            f.text(x + w / 2, 78 + k * 16 - (8 if "\n" in label else 0), line, 12, P.ink, "middle", 600)
        if i < len(row) - 1:
            f.arrow(x + w + 1, 82, x + w + 19, 82, P.stone)
        x += w + 20
    f.parts.append(f'<path d="M 640 116 C 640 160, 380 170, 120 150 L 80 120" fill="none" stroke="{P.stone}" '
                   f'stroke-width="1.4" stroke-dasharray="5 4"/>')
    f.text(360, 182, "next step", 11, P.stone, "middle")
    f.box(160, 210, 400, 44, "every N steps: evaluate, then save the resume point", P.mist, P.ink, 12,
          sub="weights + buffers, optimizer moments, step, data order")
    f.caption("Figure 8.2  One optimizer step. The resume point holds everything needed to continue as if the "
              "run had never stopped.")
    f.save(OUT / "fig8_2_loop.svg")


def fig_schedule():
    """8.4: the WSD schedule, the main run's loss, and the decayed copy."""
    f = Fig(720, 520)
    lr = [(s, v) for s, v in R["schedule"]]
    X, Y = f.line_chart(110, 30, 560, 120, [("", lr, P.indigo)], ylabel="learning rate", ymin=0, ymax=R["lr"] * 1.1,
                        yticks=[0, 0.001])
    ds = R["decay_start"]
    f.text(X(ds) + 6, Y(R["lr"]) - 6, f"decay starts at {ds}", 11, P.indigo, weight=600)
    first = R["save_at"]
    copy_lr = [(first, R["lr"]), (first + R["copy_steps"], R["min_lr"])]
    d = f"M{X(copy_lr[0][0]):.1f},{Y(copy_lr[0][1]):.1f} L{X(copy_lr[1][0]):.1f},{Y(copy_lr[1][1]):.1f}"
    f.parts.append(f'<path d="{d}" fill="none" stroke="{P.saffron}" stroke-width="2.4" stroke-dasharray="6 4"/>')
    main = [p for p in R["main"] if p[0] >= 200]
    copy = R["copy"]
    X2, Y2 = f.line_chart(110, 210, 560, 230, [("", main, P.indigo), ("", copy, P.saffron)],
                          xlabel="training step", ylabel="validation loss", ymin=1.4)   # room for labels below
    end = copy[-1]
    main_at = dict(R["main"])[end[0]]
    f.text(X2(end[0]) - 8, Y2(end[1]) + 20, f"decayed copy: {end[1]:.3f}", 12, P.saffron, "end", 600)   # below-left: empty
    f.text(X2(end[0]) + 8, Y2(main_at) - 8, f"main run, still at peak: {main_at:.3f}", 12, P.indigo, weight=600)
    last = R["main"][-1]
    f.text(X2(last[0]) - 6, Y2(last[1]) + 22, f"after its own decay: {last[1]:.3f}", 12, P.indigo, "end", 600)
    f.caption(f"Figure 8.4  Warmup-stable-decay. Top: the learning rate (blue) and the copy's (saffron, dashed). "
              f"Bottom: validation loss. The copy branches from the resume point at step {first}. Measured in the lab.")
    f.save(OUT / "fig8_4_schedule.svg")


def fig_atomic():
    """8.3: why a checkpoint is written beside the old one and renamed."""
    f = Fig(720, 300)
    f.text(30, 40, "Writing in place", 13, P.rose, weight=700)
    f.box(30, 56, 190, 40, "run.last (step 41)", P.mist, P.ink, 12)
    f.arrow(222, 76, 270, 76, P.stone)
    f.box(272, 56, 190, 40, "run.last, half written", P.rose_soft, P.rose, 12)
    f.text(480, 72, "crash here: both the old and", 12, P.rose)
    f.text(480, 90, "the new resume point are gone", 12, P.rose)
    f.text(30, 150, "Write beside it, then rename", 13, P.teal, weight=700)
    f.box(30, 166, 190, 40, "run.last (step 41)", P.mist, P.ink, 12)
    f.box(272, 166, 190, 40, "run.last.tmp, writing", P.teal_soft, P.teal, 12)
    f.arrow(462, 186, 510, 186, P.stone)
    f.box(512, 166, 180, 40, "os.replace → run.last", P.teal_soft, P.teal, 12)
    f.text(272, 228, "crash while writing: run.last is untouched (step 41).", 12, P.teal)
    f.text(272, 246, "os.replace swaps the file in one indivisible move.", 12, P.teal)
    f.caption("Figure 8.3  Saving a checkpoint so that a crash can't destroy the one you already have.")
    f.save(OUT / "fig8_3_atomic.svg")


def fig_errorbar():
    """8.5: the same model, six window sets."""
    v = R["window_sets"]
    mean, sd = statistics.mean(v), statistics.stdev(v)
    f = Fig(720, 260)
    lo, hi = min(v) - 0.02, max(v) + 0.02
    x0, x1, y = 90, 650, 120
    X = lambda t: x0 + (t - lo) / (hi - lo) * (x1 - x0)
    f.line(x0, y, x1, y, P.ink, 1.2)
    for t in (round(lo + (hi - lo) * k / 4, 3) for k in range(5)):
        f.line(X(t), y, X(t), y + 5, P.ink, 1)
        f.text(X(t), y + 20, f"{t:.3f}", 11, P.stone, "middle")
    f.rect(X(mean - sd), y - 22, X(mean + sd) - X(mean - sd), 16, P.indigo_soft, r=3, sw=0)
    for t in v:
        f.parts.append(f'<circle cx="{X(t):.1f}" cy="{y - 14:.1f}" r="6" fill="{P.saffron}"/>')
    f.text(X(mean), y - 36, f"mean {mean:.3f}, standard deviation {sd:.3f}", 12, P.ink, "middle", 600)
    f.text(x0, y + 52, "validation loss of the SAME model, measured on six different sets of 20 random windows",
           12, P.ink)
    f.caption("Figure 8.5  One model, six numbers. A difference smaller than this spread says nothing about the "
              "model. Measured in the lab.")
    f.save(OUT / "fig8_5_errorbar.svg")


if __name__ == "__main__":
    for fn in (fig_sampling, fig_loop, fig_atomic):
        fn()
    if R:
        for fn in (fig_schedule, fig_errorbar):
            fn()
    print("figures written to", OUT)
