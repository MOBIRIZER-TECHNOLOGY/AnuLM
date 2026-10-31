"""Draws Chapter 20's figures; the evidence one from the lab's results.json.

    python lab.py && python make_figures.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "figures"))
from fig import Fig, P  # noqa: E402

OUT = HERE / "figures"
RES = HERE / "results.json"
R = json.loads(RES.read_text(encoding="utf-8")) if RES.exists() else None


def fig_ddp():
    """20.1: two copies, two halves of the batch, gradients and load counts averaged."""
    f = Fig(720, 370)
    f.box(260, 20, 200, 40, "a batch of 16 windows", P.mist, P.stone, 12)
    for k, x in enumerate((60, 420)):
        f.arrow(360, 62, x + 120, 96, P.stone, 1.2, 6)
        f.box(x, 98, 240, 40, f"rank {k}: its 8 windows (shard)", P.indigo_soft, P.indigo, 12)
        f.box(x, 152, 240, 40, "forward, backward on its copy", P.paper, P.ink, 12)
        f.arrow(x + 120, 140, x + 120, 150, P.stone, 1.2, 5)
    f.rect(60, 210, 600, 46, P.saffron_soft, P.saffron, r=6)
    f.text(360, 230, "all-reduce: average the gradients, sum the experts' load counts", 12, P.ink, "middle", 700)
    f.text(360, 248, "every parameter, in the same order on every rank; zeros for an expert that got no tokens", 11,
           P.stone, "middle")
    for x in (60, 420):
        f.arrow(x + 120, 194, x + 120, 208, P.stone, 1.2, 5)
        f.arrow(x + 120, 258, x + 120, 276, P.stone, 1.2, 5)
        f.box(x, 278, 240, 26, "the same step, the same bias update", P.teal_soft, P.teal, 11)
    f.caption("Figure 20.1  Data-parallel training. The copies never diverge because every step starts from the "
              "same averaged numbers.")
    f.save(OUT / "fig20_1_ddp.svg")


def fig_numbers():
    """20.2: what the lab measured."""
    a, b = R["part1"], R["part2"]
    rows = [("one process vs two, after 1 step", f"{a['one_vs_two_after_one_step']:.1e}", "largest weight difference"),
            (f"one process vs two, after {R['steps']} steps", f"{a['one_vs_two_weights']:.1e}",
             "rounding, grown by flipped routing choices"),
            ("the two copies, as they train", f"{a['two_ranks_weights']:.2g}", "identical, bit for bit"),
            ("validation loss: one process / two", f"{a['val']['one process']:.3f} / {a['val']['two processes']:.3f}", ""),
            ("load counts not shared: weights", f"{b['ranks_weights']:.2g}", "still identical"),
            ("load counts not shared: expert biases", f"{b['ranks_biases']:.3f}", "drifted apart"),
            ("load counts not shared: routing", f"{100 * b['routing_differs']:.1f}%", "of expert choices differ")]
    f = Fig(720, 330)
    for i, (label, value, note) in enumerate(rows):
        y = 40 + i * 36
        f.rect(16, y - 20, 688, 30, P.teal_soft if i < 4 else P.rose_soft, "none", r=4)
        f.text(28, y, label, 12, P.ink)
        f.text(420, y, value, 12, P.ink, "end", 700, mono=True)
        f.text(436, y, note, 11, P.stone)
    f.caption("Figure 20.2  Chapter 8's model trained from scratch, batches of 16, in one process and in two "
              "(torch.distributed, gloo, on the CPU). Measured in the lab.")
    f.save(OUT / "fig20_2_numbers.svg")


if __name__ == "__main__":
    fig_ddp()
    if R:
        fig_numbers()
    print("figures written to", OUT)
