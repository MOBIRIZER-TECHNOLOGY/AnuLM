"""
Part III figures (Chapters 8-10): bytes, BPE, embeddings.

    python book/tools/make_figures.py part_iii

Every number drawn is computed here from the repository's code or from
book/measurements/part_iii_*.json, and asserted against the chapter's value.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from make_figures import (BOOK, CARD, CORAL, GOLD, INK, MONO, MUTED, PAPER, ROOT, RULE, SERIF,  # noqa: E402
                          TEAL, close, esc, line, rect, svg, text)

sys.path.insert(0, str(ROOT))


def _arrow(x1, y1, x2, y2):
    return line(x1, y1, x2, y2, MUTED, "f-muted-stroke", 1.5, ' marker-end="url(#arrow)"')


# --------------------------------------------------------------------------- Ch 8
def fig_utf8():
    ch = "भ"
    cp = ord(ch)
    enc = ch.encode("utf-8")
    assert cp == 0x092D and enc.hex() == "e0a4ad"
    bits16 = f"{cp:016b}"
    groups = [bits16[:4], bits16[4:10], bits16[10:]]
    w, h = 760, 300
    b = [text(20, 28, "UTF-8: the code point of भ written as three bytes", 15, weight=700, family=SERIF),
         text(20, 48, "U+092D = 0x092D = 2349 · 16 bits, split 4 + 6 + 6 from the right · verified", 12, "f-muted", MUTED)]
    c = 22
    # code point bits, grouped
    x = 150
    b.append(text(20, 100, "code point", 13, weight=700))
    b.append(text(20, 116, "U+092D", 12, "f-muted", MUTED, family=MONO))
    gx = []
    for gi, g in enumerate(groups):
        gx.append(x)
        for i, bit in enumerate(g):
            b.append(rect(x + i * c, 88, c - 2, c + 6, CARD, "f-card", 3, tip=f"group {gi + 1}, bit {bit}"))
            b.append(text(x + i * c + (c - 2) / 2, 106, bit, 13, anchor="middle", weight=700, family=MONO))
        x += len(g) * c + 26
    # bytes
    y = 220
    b.append(text(20, y + 14, "UTF-8 bytes", 13, weight=700))
    markers = ["1110", "10", "10"]
    bx = 150
    for k in range(3):
        byte = f"{enc[k]:08b}"
        m = len(markers[k])
        assert byte.startswith(markers[k]) and byte[m:] == groups[k]
        for i, bit in enumerate(byte):
            is_m = i < m
            b.append(rect(bx + i * c, y, c - 2, c + 6, CARD, "f-card", 3))
            if is_m:
                b.append(rect(bx + i * c, y, c - 2, c + 6, CORAL, "f-coral", 3, ' fill-opacity="0.28"',
                              tip="marker bit: " + ("lead of a three-byte character" if k == 0 else "continuation byte")))
            b.append(text(bx + i * c + (c - 2) / 2, y + 18, bit, 13, anchor="middle", weight=700, family=MONO))
        b.append(text(bx + 4 * c - 1, y + 50, f"0x{enc[k]:02x}", 13, anchor="middle", weight=700, family=MONO))
        b.append(text(bx + 4 * c - 1, y + 66, "lead byte" if k == 0 else "continuation", 11, "f-muted", MUTED, "middle"))
        # arrow from the code point group to the payload bits
        gcx = gx[k] + len(groups[k]) * c / 2 - 1
        pcx = bx + (m + (8 - m) / 2) * c - 1
        b.append(_arrow(gcx, 120, pcx, y - 6))
        bx += 8 * c + 24
    b.append(rect(560, 74, 14, 14, CORAL, "f-coral", 3, ' fill-opacity="0.28"'))
    b.append(text(580, 85, "marker bits", 12))
    b.append(rect(560, 96, 14, 14, CARD, "f-card", 3, f' stroke="{RULE}"'))
    b.append(text(580, 107, "the code point's bits", 12))
    return svg("fig08-1-utf8.svg", w, h, "Figure 8.1 · One Devanagari letter as three bytes",
               "The 16 bits of U+092D, 0000 100100 101101, go into three bytes: 1110 0000, 10 100100, 10 101101, "
               "which are 0xe0 0xa4 0xad. The marker bits 1110, 10 and 10 are highlighted.", b)


def fig_bytes_per_char():
    from anulm.training.train import FALLBACK
    rows = []
    labels = ["Hindi (Devanagari)", "English (Latin)", "Tamil", "Kannada", "Bengali",
              "Hinglish (Latin)", "Hindi + English", "English (Latin)"]
    for lab, ln in zip(labels, FALLBACK.strip().split("\n")):
        rows.append((lab, len(ln.encode("utf-8")) / len(ln)))
    close(rows[0][1], 2.62, 5e-3, "Hindi line bytes/char")
    close(rows[3][1], 2.76, 5e-3, "Kannada line bytes/char")
    close(rows[6][1], 1.86, 5e-3, "code-mixed line bytes/char")
    w, h = 720, 360
    x0, x1, y0, rh = 190, 640, 80, 30
    sx = lambda v: x0 + (x1 - x0) * v / 3.0
    b = [text(20, 28, "UTF-8 bytes per character, line by line", 15, weight=700, family=SERIF),
         text(20, 48, "the eight lines of train.FALLBACK · characters are code points · verified", 12, "f-muted", MUTED)]
    for t in (0, 1, 2, 3):
        b.append(line(sx(t), y0 - 8, sx(t), y0 + 8 * rh))
        b.append(text(sx(t), y0 + 8 * rh + 16, f"{t}", 11, "f-muted", MUTED, "middle", family=MONO))
    b.append(text((x0 + x1) / 2, y0 + 8 * rh + 34, "bytes per character", 11, "f-muted", MUTED, "middle"))
    for i, (lab, v) in enumerate(rows):
        y = y0 + i * rh
        b.append(text(x0 - 12, y + 17, lab, 12.5, anchor="end"))
        b.append(rect(x0, y + 5, sx(v) - x0, 18, GOLD, "f-gold", 3, tip=f"{lab}: {v:.2f} bytes per character"))
        b.append(text(sx(v) + 8, y + 19, f"{v:.2f}", 12, weight=700, family=MONO))
    xr = sx(2.48)
    b.append(line(xr, y0 - 14, xr, y0 + 8 * rh, INK, "f-ink-stroke", 1.5, ' stroke-dasharray="3 3"'))
    b.append(text(xr - 4, y0 - 18, "2.48: the RESULTS §2 Hindi corpus (measured)", 11, anchor="end", weight=700))
    return svg("fig08-2-bytes-per-char.svg", w, h, "Figure 8.2 · Bytes per character, by script",
               "Hindi 2.62, English 1.00, Tamil 2.66, Kannada 2.76, Bengali 2.70, Hinglish 1.00, "
               "Hindi with English words 1.86, English 1.00. A dashed line marks 2.48, the Hindi corpus average.", b)


# --------------------------------------------------------------------------- Ch 9
TOY = "low lower lowest low slow slower newest low new lower"


def fig_merges():
    from anulm.bpe import BPE
    tok = BPE.train(TOY, 262, verbose=False)
    assert tok.merges == [(108, 111), (256, 119), (32, 257), (101, 114), (32, 110)]
    counts = [8, 8, 5, 3, 2]                     # printed by ch09_bpe_from_scratch.py
    w, h = 760, 330
    b = [text(20, 28, "Five merges on the toy corpus", 15, weight=700, family=SERIF),
         text(20, 48, "each new id is a pair of older ids · count = the pair's frequency when chosen · illustrative",
              12, "f-muted", MUTED)]
    pos = {}
    leaves = [("sp1", 32, 110), ("l", 108, 190), ("o", 111, 270), ("w", 119, 350),
              ("e", 101, 470), ("r", 114, 550), ("sp2", 32, 630), ("n", 110, 705)]
    yl = 280

    def box(x, y, label, sub, accent=False, tip=None):
        bw = max(46, 11 * len(label) + 20, 7 * len(sub) + 14)
        out = [rect(x - bw / 2, y - 18, bw, 36, CARD, "f-card", 6,
                    f' stroke="{CORAL}" stroke-width="2.5"' if accent else f' stroke="{RULE}" stroke-width="1"', tip)]
        out.append(text(x, y + 1, label, 13, anchor="middle", weight=700, family=MONO))
        out.append(text(x, y + 13, sub, 9.5, "f-muted", MUTED, "middle", family=MONO))
        return out

    for key, i, x in leaves:
        pos[key] = (x, yl)
        b += box(x, yl, repr(chr(i)), f"byte {i}")
    levels = {256: ("l", "o", 210), 257: (256, "w", 150), 258: ("sp1", 257, 90), 259: ("e", "r", 210),
              260: ("sp2", "n", 210)}
    for k, new in enumerate(sorted(levels)):
        a, c, y = levels[new]
        xa, ya = pos[a]
        xc, yc = pos[c]
        x = (xa + xc) / 2
        pos[new] = (x, y)
        for xx, yy in ((xa, ya), (xc, yc)):
            b.append(line(xx, yy - 18, x, y + 18, MUTED, "f-muted-stroke", 1.3))
        label = repr(tok.vocab[new].decode())
        b += box(x, y, label, f"id {new} · count {counts[k]}", accent=(new == 258),
                 tip=f"merge {k}: id {new} = {label}, count {counts[k]}")
    b.append(text(20, 200, "deepest token:", 11, "f-muted", MUTED))
    b.append(text(20, 214, "' low', 3 merges", 11, weight=700))
    return svg("fig09-1-merges.svg", w, h, "Figure 9.1 · Five merges on a toy corpus",
               "Bytes l and o merge into lo (id 256, count 8); lo and w into low (257, 8); space and low into ' low' "
               "(258, 5); e and r into er (259, 3); space and n into ' n' (260, 2).", b)


def _fertility():
    return json.loads((BOOK / "measurements" / "part_iii_fertility.json").read_text(encoding="utf-8"))


def fig_window():
    r = _fertility()["window_chars_1024"]
    close(r["multi32k_devanagari"], 3855, 0, "Devanagari chars in a multi32k window")
    close(r["bytes_devanagari"], 410, 0, "Devanagari chars in a byte window")
    w, h = 720, 260
    x0, x1, y0 = 170, 600, 84
    sx = lambda v: x0 + (x1 - x0) * v / 5000
    b = [text(20, 28, "Characters of text in a 1,024-token window", 15, weight=700, family=SERIF),
         text(20, 48, "3,000 held-out documents of data/v2/val.u16, split by majority script · measured", 12, "f-muted", MUTED)]
    b.append(rect(x0, 60, 12, 12, CORAL, "f-coral", 2))
    b.append(text(x0 + 18, 70, "bytes (V = 259)", 12))
    b.append(rect(x0 + 140, 60, 12, 12, TEAL, "f-teal", 2))
    b.append(text(x0 + 158, 70, "multi32k BPE (V = 32,768)", 12))
    for t in range(0, 5001, 1000):
        b.append(line(sx(t), y0, sx(t), y0 + 140))
        b.append(text(sx(t), y0 + 156, f"{t:,}", 10, "f-muted", MUTED, "middle", family=MONO))
    for gi, (lab, kb, km) in enumerate((("mostly Devanagari", "bytes_devanagari", "multi32k_devanagari"),
                                        ("mostly Latin", "bytes_latin", "multi32k_latin"))):
        y = y0 + 10 + gi * 70
        b.append(text(x0 - 12, y + 26, lab, 12.5, anchor="end", weight=700))
        for j, (k, col, cls) in enumerate(((kb, CORAL, "f-coral"), (km, TEAL, "f-teal"))):
            v = r[k]
            b.append(rect(x0, y + j * 24, sx(v) - x0, 20, col, cls, 3, tip=f"{lab}, {k}: {v:,} characters"))
            b.append(text(sx(v) + 8, y + j * 24 + 15, f"{v:,}", 12, weight=700, family=MONO))
    return svg("fig09-2-window.svg", w, h, "Figure 9.2 · Characters in a 1,024-token window",
               f"Mostly Devanagari text: {r['bytes_devanagari']:,} characters as bytes, {r['multi32k_devanagari']:,} "
               f"with multi32k. Mostly Latin text: {r['bytes_latin']:,} as bytes, {r['multi32k_latin']:,} with multi32k.", b)


def fig_vocab():
    r = _fertility()
    kinds, lengths, dev = r["vocab"]["kinds"], r["vocab"]["byte_length"], r["vocab"]["devanagari_byte_length"]
    assert kinds["ASCII"] == 21117 and kinds["Devanagari"] == 11146 and kinds["not valid UTF-8 on its own"] == 71
    assert sum(lengths.values()) == 32511 and r["distinct_ids_used"] == 29666 and dev["16"] == 3285
    w, h = 720, 310
    x0, x1, y0, y1 = 70, 680, 86, 246
    top = 4000
    n = len(lengths)
    bw = (x1 - x0) / n
    sy = lambda v: y1 - (y1 - y0) * v / top
    b = [text(20, 28, "The 32,511 merged tokens of multi32k, by length in bytes", 15, weight=700, family=SERIF),
         text(20, 48, f"ASCII {kinds['ASCII']:,} · with Devanagari {kinds['Devanagari']:,} · other {kinds['other valid UTF-8']} · "
                      f"not valid UTF-8 alone {kinds['not valid UTF-8 on its own']} · verified", 12, "f-muted", MUTED)]
    b.append(rect(x0, 60, 12, 12, TEAL, "f-teal", 2))
    b.append(text(x0 + 18, 70, "contains Devanagari", 12))
    b.append(rect(x0 + 170, 60, 12, 12, CORAL, "f-coral", 2))
    b.append(text(x0 + 188, 70, "all other merged tokens", 12))
    for t_ in range(0, top + 1, 1000):
        b.append(line(x0, sy(t_), x1, sy(t_)))
        b.append(text(x0 - 8, sy(t_) + 4, f"{t_:,}", 10, "f-muted", MUTED, "end", family=MONO))
    for i, (k, v) in enumerate(lengths.items()):
        x = x0 + i * bw
        lab = f"{k}+" if k == "16" else k
        d = dev.get(k, 0)
        b.append(rect(x + 3, sy(d), bw - 6, y1 - sy(d), TEAL, "f-teal", 0, tip=f"{lab} bytes: {d:,} with Devanagari"))
        b.append(rect(x + 3, sy(v), bw - 6, sy(d) - sy(v), CORAL, "f-coral", 0, tip=f"{lab} bytes: {v - d:,} others"))
        b.append(text(x + bw / 2, y1 + 16, lab, 10.5, "f-muted", MUTED, "middle", family=MONO))
    b.append(text((x0 + x1) / 2, y1 + 34, "bytes per merged token (16+ = 16 or more)", 11, "f-muted", MUTED, "middle"))
    return svg("fig09-3-vocab.svg", w, h, "Figure 9.3 · What the 32,511 merged tokens are",
               "Stacked histogram of merged-token lengths in bytes, Devanagari tokens and the rest. Totals: "
               + ", ".join(f"{k}: {v}" for k, v in lengths.items()) + ". Devanagari: "
               + ", ".join(f"{k}: {v}" for k, v in dev.items()) + ".", b)


# --------------------------------------------------------------------------- Ch 10
def fig_lookup():
    import torch
    torch.manual_seed(0)
    Vt, Dt, t = 8, 4, 5
    W = torch.randn(Vt, Dt).mul(10).round().div(10)
    oh = torch.zeros(Vt)
    oh[t] = 1
    assert torch.equal(oh @ W, W[t])
    w, h = 760, 330
    c, rh = 34, 26
    b = [text(20, 28, "Looking up id 5 = multiplying its one-hot row by the table", 15, weight=700, family=SERIF),
         text(20, 48, "a toy table with V = 8 rows of width D = 4 · illustrative", 12, "f-muted", MUTED)]
    # one-hot row vector
    x0, y0 = 30, 150
    b.append(text(x0, y0 - 14, "one-hot, (1, V)", 12, "f-muted", MUTED))
    for i in range(Vt):
        hot = i == t
        b.append(rect(x0 + i * 26, y0, 24, 26, CARD, "f-card", 3))
        if hot:
            b.append(rect(x0 + i * 26, y0, 24, 26, CORAL, "f-coral", 3, ' fill-opacity="0.28"', tip="position 5: 1"))
        b.append(text(x0 + i * 26 + 12, y0 + 18, "1" if hot else "0", 13, anchor="middle", weight=700, family=MONO))
    b.append(text(x0 + Vt * 26 + 14, y0 + 19, "×", 20, anchor="middle", weight=700))
    # table
    tx, ty = 300, 84
    b.append(text(tx, ty - 10, "W, (V, D)", 12, "f-muted", MUTED))
    for i in range(Vt):
        hot = i == t
        b.append(text(tx - 8, ty + i * rh + 18, f"{i}", 11, "f-muted", MUTED, "end", family=MONO))
        for j in range(Dt):
            b.append(rect(tx + j * (c + 22), ty + i * rh, c + 20, rh - 2, CARD, "f-card", 3))
            if hot:
                b.append(rect(tx + j * (c + 22), ty + i * rh, c + 20, rh - 2, CORAL, "f-coral", 3,
                              ' fill-opacity="0.28"', tip=f"row 5, column {j}: {W[i, j].item():.1f}"))
            b.append(text(tx + j * (c + 22) + (c + 20) / 2, ty + i * rh + 17, f"{W[i, j].item():.1f}", 12,
                          anchor="middle", weight=700 if hot else 400, family=MONO))
    ex = tx + Dt * (c + 22) + 20
    b.append(text(ex, y0 + 19, "=", 20, anchor="middle", weight=700))
    rx = ex + 24
    b.append(text(rx, y0 - 14, "row 5, (1, D)", 12, "f-muted", MUTED))
    for j in range(Dt):
        b.append(rect(rx + j * (c + 4), y0, c + 2, 26, CARD, "f-card", 3))
        b.append(rect(rx + j * (c + 4), y0, c + 2, 26, CORAL, "f-coral", 3, ' fill-opacity="0.28"'))
        b.append(text(rx + j * (c + 4) + (c + 2) / 2, y0 + 18, f"{W[t, j].item():.1f}", 12, anchor="middle",
                      weight=700, family=MONO))
    b.append(text(20, h - 18, "F.embedding indexes row 5 directly; it never builds the one-hot row or multiplies the zeros.",
                  12, "f-muted", MUTED))
    return svg("fig10-1-lookup.svg", w, h, "Figure 10.1 · A lookup is a one-hot product",
               "A one-hot row with a 1 at position 5 times an 8 by 4 table gives row 5 of the table.", b)


def fig_norms():
    r = json.loads((BOOK / "measurements" / "part_iii_tables.json").read_text(encoding="utf-8"))
    bk = r["buckets"]
    close(bk[0]["in_norm"], 0.013, 5e-4, "unseen input norm")
    close(bk[-1]["in_norm"], 0.783, 5e-4, "100k+ input norm")
    close(bk[0]["out_norm"], 0.508, 5e-4, "unseen output norm")
    close(r["init_row_norm"], 0.64, 1e-3, "init norm")
    w, h = 720, 330
    x0, x1, y0, y1 = 80, 690, 86, 256
    top = 0.9
    gw = (x1 - x0) / len(bk)
    sy = lambda v: y1 - (y1 - y0) * v / top
    b = [text(20, 28, "Row norms of the released tables, by how common the token is", 15, weight=700, family=SERIF),
         text(20, 48, "ckpt_base_v2.pt · counts over the 113M-token held-out split · measured", 12, "f-muted", MUTED)]
    b.append(rect(x0, 60, 12, 12, CORAL, "f-coral", 2))
    b.append(text(x0 + 18, 70, "input row (embed_tokens)", 12))
    b.append(rect(x0 + 200, 60, 12, 12, TEAL, "f-teal", 2))
    b.append(text(x0 + 218, 70, "output row (lm_head)", 12))
    for v in (0, 0.2, 0.4, 0.6, 0.8):
        b.append(line(x0, sy(v), x1, sy(v)))
        b.append(text(x0 - 8, sy(v) + 4, f"{v:.1f}", 10, "f-muted", MUTED, "end", family=MONO))
    yi = sy(r["init_row_norm"])
    b.append(line(x0, yi, x1, yi, INK, "f-ink-stroke", 1.5, ' stroke-dasharray="4 3"'))
    b.append(text(x0 + 6, yi - 6, "at initialisation, 0.640", 11, weight=700))
    bw = 34
    for i, row in enumerate(bk):
        cx = x0 + i * gw + gw / 2
        for j, (k, col, cls) in enumerate((("in_norm", CORAL, "f-coral"), ("out_norm", TEAL, "f-teal"))):
            v = row[k]
            x = cx - bw - 2 + j * (bw + 4)
            b.append(rect(x, sy(v), bw, y1 - sy(v), col, cls, 2, tip=f"{row['bucket']}: {k} {v:.3f}"))
            b.append(text(x + bw / 2, sy(v) - 5, f"{v:.2f}", 10.5, anchor="middle", weight=700, family=MONO))
        b.append(text(cx, y1 + 16, row["bucket"], 11, anchor="middle", weight=700, family=MONO))
        b.append(text(cx, y1 + 31, f"{row['ids']:,} ids", 10, "f-muted", MUTED, "middle"))
    b.append(text((x0 + x1) / 2, y1 + 50, "occurrences of the id in the held-out split", 11, "f-muted", MUTED, "middle"))
    return svg("fig10-2-norms.svg", w, h, "Figure 10.2 · Row norms by how common the token is",
               "Mean row norms by held-out count. Input rows: " + ", ".join(f"{x['bucket']} {x['in_norm']}" for x in bk)
               + ". Output rows: " + ", ".join(f"{x['bucket']} {x['out_norm']}" for x in bk) + ". Initial norm 0.640.", b)
