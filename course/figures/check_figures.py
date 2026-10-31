"""
Check every figure's text before a chapter ships.

    python course/figures/check_figures.py            # all chapters
    python course/figures/check_figures.py ch05       # chapters whose folder starts with ch05

Two rules, from bugs found by eye in earlier drafts:
  1. every label lies inside the figure (estimated from its length and size:
     Inter is about 0.56 em per character, JetBrains Mono 0.6);
  2. nothing but the caption comes within one line of the caption.
Exit status 1 if anything breaks a rule.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

COURSE = Path(__file__).resolve().parents[1]
TEXT = re.compile(r'<text x="([\d.]+)" y="([\d.]+)" font-family="([^"]+)" font-size="([\d.]+)"'
                  r'[^>]*text-anchor="(\w+)"([^>]*)>(.*?)</text>')


def problems(svg: Path) -> list[str]:
    s = svg.read_text(encoding="utf-8")
    w, h = map(float, re.search(r'viewBox="0 0 (\d+) (\d+)"', s).groups())
    texts = []
    for m in TEXT.finditer(s):
        x, y, fam, size, anchor, rest, t = (float(m[1]), float(m[2]), m[3], float(m[4]), m[5], m[6],
                                            html.unescape(m[7]))
        width = len(t) * size * (0.6 if "Mono" in fam else 0.56)
        x0 = x - width if anchor == "end" else x - width / 2 if anchor == "middle" else x
        texts.append((x0, x0 + width, y, size, 'font-style="italic"' in rest and x == 16.0, t))
    out = []
    for x0, x1, y, size, is_caption, t in texts:
        if x0 < 2 or x1 > w - 2 or y > h - 2 or y < size * 0.8:
            out.append(f"outside the figure: {t[:48]!r} spans x {x0:.0f}-{x1:.0f}, y {y:.0f} (figure {w:.0f}x{h:.0f})")
    caption_top = min((y - size for *_, y, size, cap, _ in texts if cap), default=None)
    if caption_top is not None:
        for x0, x1, y, size, is_caption, t in texts:
            if not is_caption and y > caption_top - 4:
                out.append(f"touches the caption: {t[:48]!r} at y {y:.0f}, caption starts at {caption_top:.0f}")
    return out


def main() -> None:
    prefix = sys.argv[1] if len(sys.argv) > 1 else "ch"
    bad = 0
    for svg in sorted(COURSE.glob(f"{prefix}*/figures/*.svg")):
        for p in problems(svg):
            bad += 1
            print(f"{svg.parent.parent.name}/{svg.name}: {p}")
    print(f"{bad} problem(s)" if bad else "all figures pass")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
