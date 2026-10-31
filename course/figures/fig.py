"""
The course's figure library: plain SVG, no dependencies, one visual language.

    from fig import Fig, P
    f = Fig(720, 300)
    f.box(40, 40, 160, 60, "embedding", fill=P.indigo_soft, stroke=P.indigo)
    f.arrow(200, 70, 280, 70)
    f.save("figures/fig1_3.svg")

Colours and type follow course/STYLE.md. Saffron means "this is what learns".
"""

from __future__ import annotations

import html
import math
from pathlib import Path


class P:
    """The palette (STYLE.md)."""
    ink = "#1F2433"
    paper = "#FBF8F3"
    saffron = "#E8833A"
    saffron_soft = "#FBE3D0"
    indigo = "#3B4CCA"
    indigo_soft = "#DDE1F7"
    teal = "#1F9E89"
    teal_soft = "#D3EEE9"
    rose = "#D1495B"
    rose_soft = "#F6D9DD"
    stone = "#8A8F98"
    stone_soft = "#E9E9EB"
    mist = "#ECE7DF"


SANS = "Inter, 'Noto Sans Devanagari', 'Segoe UI', sans-serif"
MONO = "'JetBrains Mono', Consolas, monospace"


def esc(s) -> str:
    return html.escape(str(s), quote=True)


class Fig:
    def __init__(self, w: int = 720, h: int = 320, bg: str = P.paper):
        self.w, self.h = w, h
        self.parts = [f'<rect width="{w}" height="{h}" fill="{bg}"/>']

    # --- primitives ----------------------------------------------------
    def text(self, x, y, s, size=13, fill=P.ink, anchor="start", weight=400, mono=False, italic=False):
        style = ' font-style="italic"' if italic else ""
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{MONO if mono else SANS}" '
                          f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
                          f'text-anchor="{anchor}"{style}>{esc(s)}</text>')

    def rect(self, x, y, w, h, fill=P.mist, stroke="none", r=8, sw=1.5, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" '
                          f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def line(self, x1, y1, x2, y2, stroke=P.ink, sw=1.5, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                          f'stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def arrow(self, x1, y1, x2, y2, stroke=P.ink, sw=1.6, head=7):
        self.line(x1, y1, x2, y2, stroke, sw)
        a = math.atan2(y2 - y1, x2 - x1)
        pts = [(x2, y2),
               (x2 - head * math.cos(a - 0.45), y2 - head * math.sin(a - 0.45)),
               (x2 - head * math.cos(a + 0.45), y2 - head * math.sin(a + 0.45))]
        self.parts.append(f'<polygon points="{" ".join(f"{px:.1f},{py:.1f}" for px, py in pts)}" fill="{stroke}"/>')

    def box(self, x, y, w, h, label="", fill=P.indigo_soft, stroke=P.indigo, size=13, sub=None,
            weight=600, color=P.ink, dash=None):
        """A labelled block, the basic unit of concept and anatomy figures."""
        self.rect(x, y, w, h, fill, stroke, dash=dash)
        cy = y + h / 2 + (size * 0.35 if not sub else -2)
        if label:
            self.text(x + w / 2, cy, label, size, color, "middle", weight)
        if sub:
            self.text(x + w / 2, cy + size + 2, sub, size - 2, P.stone, "middle")

    def chip(self, x, y, s, fill=P.teal_soft, stroke=P.teal, size=13):
        """A token: one byte or character in a small rounded square. Returns its width."""
        w = max(22, 9 * len(s) + 12)
        self.rect(x, y, w, 26, fill, stroke, r=5, sw=1.2)
        self.text(x + w / 2, y + 18, s, size, P.ink, "middle", 500, mono=True)
        return w

    def caption(self, s, width=112):
        """The numbered caption, wrapped to the figure's width, at the bottom."""
        import textwrap
        lines = textwrap.wrap(s, width)
        for i, line in enumerate(lines):
            self.text(16, self.h - 12 - (len(lines) - 1 - i) * 15, line, 11, P.stone, italic=True)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        body = "\n".join(self.parts)
        Path(path).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
                              f'width="{self.w}" height="{self.h}" role="img">\n{body}\n</svg>\n',
                              encoding="utf-8")

    # --- evidence ------------------------------------------------------
    def line_chart(self, x, y, w, h, series, xlabel="", ylabel="", ymin=None, ymax=None,
                   xticks=None, yticks=None, hlines=()):
        """series: [(label, [(x, y), ...], colour)], labelled at their last point
        instead of a legend. hlines: [(y, label, colour)] for reference levels."""
        xs = [p[0] for _, pts, _ in series for p in pts]
        ys = [p[1] for _, pts, _ in series for p in pts] + [v for v, _, _ in hlines]
        x0, x1 = min(xs), max(xs)
        y0 = min(ys) if ymin is None else ymin
        y1 = max(ys) if ymax is None else ymax
        X = lambda v: x + (v - x0) / (x1 - x0 or 1) * w
        Y = lambda v: y + h - (v - y0) / (y1 - y0 or 1) * h
        for t in yticks or _ticks(y0, y1):
            self.line(x, Y(t), x + w, Y(t), P.mist, 1)
            self.text(x - 8, Y(t) + 4, _fmt(t), 11, P.stone, "end")
        for t in xticks or _ticks(x0, x1):
            self.text(X(t), y + h + 18, _fmt(t), 11, P.stone, "middle")
        self.line(x, y + h, x + w, y + h, P.ink, 1.2)
        for v, label, col in hlines:
            self.line(x, Y(v), x + w, Y(v), col, 1.4, dash="5 4")
            self.text(x + w - 4, Y(v) - 6, label, 11, col, "end", 600)
        for label, pts, col in series:
            d = " ".join(f"{'M' if i == 0 else 'L'}{X(a):.1f},{Y(b):.1f}" for i, (a, b) in enumerate(pts))
            self.parts.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2.4" '
                              f'stroke-linejoin="round" stroke-linecap="round"/>')
            ex, ey = pts[-1]
            self.parts.append(f'<circle cx="{X(ex):.1f}" cy="{Y(ey):.1f}" r="3.5" fill="{col}"/>')
            if label:
                self.text(X(ex) - 6, Y(ey) - 10, label, 12, col, "end", 600)
        if xlabel:
            self.text(x + w / 2, y + h + 38, xlabel, 12, P.ink, "middle")
        if ylabel:
            self.parts.append(f'<text transform="translate({x - 46:.1f},{y + h / 2:.1f}) rotate(-90)" '
                              f'font-family="{SANS}" font-size="12" fill="{P.ink}" '
                              f'text-anchor="middle">{esc(ylabel)}</text>')
        return X, Y

    def hbars(self, x, y, w, rows, vmax=None, bar_h=22, gap=10, label_w=130, fmt="{:.0%}"):
        """rows: [(label, value, colour)] as horizontal bars, value printed at the end."""
        vmax = vmax or max(v for _, v, _ in rows)
        for i, (label, v, col) in enumerate(rows):
            yy = y + i * (bar_h + gap)
            self.text(x + label_w - 10, yy + bar_h * 0.7, label, 13, P.ink, "end", 500,
                      mono=len(label) <= 3)
            bw = (w - label_w - 60) * v / vmax
            self.rect(x + label_w, yy, max(bw, 2), bar_h, col, r=4)
            self.text(x + label_w + bw + 8, yy + bar_h * 0.7, fmt.format(v), 12, P.ink)
        return y + len(rows) * (bar_h + gap)


def _ticks(a, b, n=5):
    span = b - a or 1
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    t = math.ceil(a / step) * step
    out = []
    while t <= b + 1e-9:
        out.append(round(t, 10))
        t += step
    return out


def _fmt(v):
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:g}"
