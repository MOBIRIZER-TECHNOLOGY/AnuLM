"""Draws Chapter 6's figures; the evidence ones from the lab's results.

    python lab.py && python make_figures.py
"""

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "figures"))
from fig import Fig, P  # noqa: E402

OUT = HERE / "figures"
R = json.loads((HERE / "results.json").read_text(encoding="utf-8"))


def show(ch):
    return {" ": "␣", "\n": "↵"}.get(ch, ch)


def fig_attention():
    """6.1: one query compares itself with every earlier key and mixes their values."""
    f = Fig(720, 330)
    word = "To be or not to b"
    weights = [0.02, 0.03, 0.02, 0.30, 0.06, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.02, 0.04, 0.05, 0.10, 0.20]
    x0, y = 24, 190
    xs = []
    for ch in word:
        w = f.chip(x0, y, show(ch))
        xs.append(x0 + w / 2)
        x0 += w + 3
    qx = xs[-1]
    f.text(qx, y + 46, "query", 11, P.saffron, "middle", 700)
    f.text(qx, y + 60, "(position 16)", 10, P.stone, "middle")
    f.text(24, 62, "attention weight (bar height)", 11, P.stone, weight=600)
    for cx, wgt in zip(xs, weights):
        h = wgt * 260
        f.rect(cx - 7, y - 10 - h, 14, h, P.indigo, r=2, sw=0)
    f.text(xs[3], y - 10 - weights[3] * 260 - 6, "the earlier b", 11, P.indigo, "middle", 600)
    tx = 470
    f.text(tx, 40, "For the query at the last b:", 13, P.ink, weight=700)
    for k, line in enumerate(["1. score each earlier key: q · k", "2. softmax the scores into", "   weights that sum to 1",
                              "3. output = the weighted sum", "   of the values v", "",
                              "Keys after the query are never", "scored: the causal mask."]):
        f.text(tx, 66 + 19 * k, line, 12, P.ink)
    f.caption("Figure 6.1  Attention for one query. Every earlier position gets a weight; the output mixes their "
              "values by those weights. The weights here are illustrative.")
    f.save(OUT / "fig6_1_attention.svg")


def fig_masks():
    """6.2: three masks: training, a sliding window, and decoding against a cache."""
    f = Fig(720, 350)
    cell = 16

    def grid(x0, y0, S, T, start, window, title, wrong=None):
        f.text(x0 + T * cell / 2, y0 - 26, title, 12, P.ink, "middle", 700)
        f.text(x0 + T * cell / 2, y0 - 10, "keys →", 10, P.stone, "middle")
        f.text(x0 - 6, y0 + S * cell / 2, "queries ↓", 10, P.stone, "end")
        for i in range(S):
            qp = start + i
            for j in range(T):
                ok = j <= qp and (window is None or j > qp - window)
                col = P.indigo if ok else P.mist
                if wrong is not None and ok and wrong(i, j):      # visible, but is_causal would hide it
                    col = P.rose
                f.rect(x0 + j * cell + 1, y0 + i * cell + 1, cell - 2, cell - 2, col, r=2, sw=0)

    grid(70, 70, 8, 8, 0, None, "training: causal")
    grid(300, 70, 8, 8, 0, 3, "sliding window, 3")
    grid(530, 70, 3, 8, 5, None, "3 new tokens, 5 cached", wrong=lambda i, j: j > i)   # top-left: key j <= row i
    for k, line in enumerate(["PyTorch's is_causal draws its", "triangle from the top-left:",
                              "it would hide the red keys,", "which these tokens must see.",
                              "An explicit mask is needed."]):
        f.text(530, 70 + 3 * cell + 22 + 16 * k, line, 11, P.rose, weight=600 if k == 0 else 400)
    f.rect(70, 248, 14, 12, P.indigo, r=2, sw=0)
    f.text(90, 258, "may attend", 11, P.ink)
    f.rect(190, 248, 14, 12, P.mist, r=2, sw=0)
    f.text(210, 258, "masked", 11, P.ink)
    f.caption("Figure 6.2  Which keys each query may see. Training uses the plain causal triangle; a window also "
              "hides keys too far back; decoding against a cache needs the triangle shifted by what is cached.")
    f.save(OUT / "fig6_2_masks.svg")


def fig_gqa():
    """6.3: query heads share key/value heads; the cache stores only the shared ones."""
    f = Fig(720, 360)
    f.text(30, 34, "AnuLM 400M: 16 query heads share 4 key/value heads", 13, P.ink, weight=700)
    for g in range(4):
        kx = 60 + g * 160
        f.box(kx, 150, 110, 34, f"k,v head {g}", P.saffron_soft, P.saffron, 12)
        for j in range(4):
            qx = kx - 18 + j * 37
            f.box(qx, 60, 32, 30, f"q{g * 4 + j}", P.indigo_soft, P.indigo, 11)
            f.line(qx + 16, 90, kx + 55, 150, P.stone, 1)
    mem = R["cache_bytes_per_token"]
    rows = [(name.split(" (")[0], b / 1024, P.saffron if "AnuLM" in name else P.stone_soft) for name, b in mem.items()]
    f.text(30, 222, "KV cache per token (20 layers, head dim 64, bf16)", 12, P.ink, weight=700)
    f.hbars(30, 236, 640, rows, label_w=170, fmt="{:.0f} KiB", bar_h=20, gap=8)
    f.caption("Figure 6.3  Grouped-query attention. Each key/value head serves a group of query heads, and only "
              "the shared heads go in the cache: a quarter of multi-head attention's memory at 4 groups.")
    f.save(OUT / "fig6_3_gqa.svg")


def fig_curve():
    rows = list(csv.DictReader(open(HERE / "curve.csv", encoding="utf-8")))
    pts = [(int(r["step"]), float(r["val_loss"])) for r in rows]
    ch4 = HERE.parent / "ch04-skeleton" / "curve.csv"
    series = [("", pts, P.saffron)]
    if ch4.exists():
        sk = [(int(r["step"]), float(r["val_loss"])) for r in csv.DictReader(open(ch4, encoding="utf-8"))]
        series.insert(0, ("", sk, P.stone))
    f = Fig(720, 372)
    X, Y = f.line_chart(90, 30, 580, 250, series, xlabel="training step",
                        ylabel="validation loss (nats per byte)", ymin=1.0, ymax=5.8,
                        hlines=[(R["bigram_loss"], f"bigram table {R['bigram_loss']:.2f}", P.teal)])
    f.text(X(900), Y(R["skeleton_loss"]) - 26, f"Chapter 4's skeleton: {R['skeleton_loss']:.2f}", 12, P.stone, weight=600)
    f.text(X(900), Y(R["final_loss"]) + 22, f"the same skeleton with attention: {R['final_loss']:.2f}", 12,
           P.saffron, weight=600)
    f.caption(f"Figure 6.4  Same sizes, data and steps as Chapter 4; only attention added. It passes the bigram "
              f"ceiling within 100 steps. Measured in the lab ({R['seconds']} s on CPU).")
    f.save(OUT / "fig6_4_curve.svg")


def fig_head():
    """6.5: one trained head's attention over a line of text."""
    hd = R["head"]
    text, w = hd["text"], hd["weights"]
    n = len(text)
    cell = 8.5
    x0, y0 = 70, 40
    f = Fig(720, int(y0 + n * cell + 80))
    hi = max(max(r) for r in w)
    for i in range(n):
        for j in range(i + 1):
            v = w[i][j] / hi
            if v > 0.02:
                f.rect(x0 + j * cell, y0 + i * cell, cell - 0.6, cell - 0.6, P.indigo, r=1, sw=0)
                f.parts[-1] = f.parts[-1].replace(f'fill="{P.indigo}"', f'fill="{P.indigo}" fill-opacity="{v:.2f}"')
    for i, ch in enumerate(text):
        f.text(x0 - 4, y0 + i * cell + 8, show(ch), 8, P.ink, "end", mono=True)
        f.text(x0 + i * cell + 4.5, y0 - 4, show(ch), 8, P.ink, "middle", mono=True)
    tx = x0 + n * cell + 24
    f.text(tx, 60, f"Layer {hd['layer']}, head {hd['head']}", 13, P.ink, weight=700)
    for k, line in enumerate(["Row: a query byte.", "Column: a key byte.", "Darker: more weight.", "",
                              f"This head puts {hd['prev_weight']:.0%} of its", "weight on the byte",
                              "just before: the dark", "line under the diagonal.", "",
                              "It has learned to look", "at the previous character."]):
        f.text(tx, 86 + 19 * k, line, 12, P.ink)
    f.caption("Figure 6.5  Attention weights of one trained head over the start of a line of validation text. "
              "Measured in the lab; of the 16 heads, the one most focused on the previous byte.")
    f.save(OUT / "fig6_5_head.svg")


if __name__ == "__main__":
    for fn in (fig_attention, fig_masks, fig_gqa, fig_curve, fig_head):
        fn()
    print("figures written to", OUT)
