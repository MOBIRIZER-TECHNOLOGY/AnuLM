"""
Render chapters of the book as one styled HTML page, in the design STYLE.md sets.

    python book/tools/build_preview.py                 # every chapter -> book/preview/index.html
    python book/tools/build_preview.py 11 20           # chosen chapters, in book order

It reads the chapter Markdown unchanged and adds only presentation:

- a cover band and a part divider for each part the chapters belong to;
- a contents rail built from the chapter and section headings;
- the page furniture of STYLE.md: "In the repo" citation strips, badges as
  pills, evidence labels (*verified*, *measured*, *illustrative*), the dark
  "In one breath" card, "Details" as a collapsed box, "Build it" cards;
- [Rnn]/[Snn] citations as chips whose tooltip is the register entry, and a
  list of every source the rendered chapters cite, from RESEARCH_REFERENCES.md;
- Python excerpts highlighted with Pygments in the book's palette; Mermaid
  diagrams left as `<pre class="mermaid">` for the viewer to draw.

The page is self-contained apart from Google Fonts.
"""

from __future__ import annotations

import html
import re
import subprocess
import sys
from pathlib import Path

import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

BOOK = Path(__file__).resolve().parent.parent
CHAPTERS = BOOK / "chapters"
OUT = BOOK / "preview" / "index.html"
REGISTER = BOOK / "RESEARCH_REFERENCES.md"
BADGES = {"IN THE REPO": "repo", "NOT IN THE REPO": "notrepo", "MEASURED": "measured", "ANALOGY": "analogy",
          "RULE OF THUMB": "rule", "EXTENSION": "extension", "IMPLEMENTATION-DEPENDENT": "impl"}
PARTS = {
    "I": ("Foundations", "What this book builds, where language models came from, and the one job they do."),
    "II": ("Just enough mathematics", "Tensors, the softmax, gradients and optimisers, each shown in the code that "
           "uses it."),
    "III": ("From text to vectors", "Byte-level BPE from scratch, and the tables that turn token ids into vectors."),
    "IV": ("Inside the transformer", "One forward pass, opened up: attention, positions, normalisation, "
           "experts, and the whole model traced end to end."),
    "V": ("Training", "Data, the training loop, precision, honest measurement, checkpoints and debugging."),
    "VI": ("Generation", "Decoding with a KV cache, choosing the next token, and stretching the context."),
    "VII": ("After pre-training", "Instruction tuning, LoRA, classification, retrieval and preference "
            "optimisation."),
    "VIII": ("Beyond text", "A wider vocabulary, frozen encoders for speech and images, and one model for many "
             "tasks."),
    "IX": ("To production", "Serving, exporting, and what separates this model from a production system."),
    "X": ("Capstone", "Train it, open it, change it."),
    "Appendices": ("Appendices", "Reference tables: shapes, parameter arithmetic, every test, a glossary, the "
                   "bibliography and the audit of every number."),
}
ROOT = BOOK.parent
FIGURES = BOOK / "figures"
GITHUB = "https://github.com/MOBIRIZER-TECHNOLOGY/AnuLM"
_MAP = (BOOK / "REPOSITORY_SOURCE_MAP.md").read_text(encoding="utf-8")
PIN = re.search(r"`@([0-9a-f]{7,40})`", _MAP).group(1)
_pub = re.search(r"Public commit: `([0-9a-f]{40})`", _MAP)
PUBLIC = _pub.group(1) if _pub else None
_same: dict[str, bool] = {}


def public_link(path: str, lines: str) -> str | None:
    """A GitHub link to these lines at the public commit, only if the file there is identical to the pin."""
    if not PUBLIC:
        return None
    if path not in _same:
        r = subprocess.run(["git", "diff", "--quiet", PIN, PUBLIC, "--", path], cwd=ROOT)
        _same[path] = r.returncode == 0
        if not _same[path]:
            print(f"  note: {path} differs between @{PIN} and the public commit; its citations are not linked")
    if not _same[path]:
        return None
    m = re.search(r"(\d+)(?:–(\d+))?", lines)
    anchor = f"#L{m.group(1)}" + (f"-L{m.group(2)}" if m and m.group(2) else "") if m else ""
    return f"{GITHUB}/blob/{PUBLIC}/{path}{anchor}"


_fig_n = [0]


def inline_figure(src: str, alt: str) -> str:
    path = (CHAPTERS / html.unescape(src)).resolve()
    if not path.is_file() or path.suffix != ".svg":
        return f'<p class="missing">Missing figure: {html.escape(src)}</p>'
    _fig_n[0] += 1
    p = f"f{_fig_n[0]}-"
    svg = path.read_text(encoding="utf-8")
    for ident in ("t", "d", "hatch", "arrow"):
        svg = svg.replace(f'id="{ident}"', f'id="{p}{ident}"').replace(f"url(#{ident})", f"url(#{p}{ident})")
    svg = svg.replace('aria-labelledby="t d"', f'aria-labelledby="{p}t {p}d"')
    return f'<figure class="fig">{svg}</figure>'


# --------------------------------------------------------------------------- input
def chapter_files(wanted: list[str]) -> list[Path]:
    """Numbered chapters in order, then the appendices (A1-*.md ...). `wanted` takes "11" or "A1"."""
    files = sorted(CHAPTERS.glob("[0-9][0-9]-*.md")) + sorted(CHAPTERS.glob("A[0-9]*-*.md"), key=lambda p: int(p.name[1:p.name.index("-")]))
    if wanted:
        keys = {w.upper().lstrip("0") for w in wanted}
        files = [f for f in files if f.name.split("-")[0].upper().lstrip("0") in keys]
    return files


def front_matter(text: str) -> tuple[dict, str]:
    meta = {}
    if text.startswith("---\n"):
        head, body = text[4:].split("\n---\n", 1)
        for line in head.splitlines():
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip().strip('"')
        return meta, body
    return meta, text


def register() -> dict[str, dict]:
    refs = {}
    for line in REGISTER.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| ([RS]\d{2}) \|", line)
        if not m:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 8:
            continue
        refs[cells[0]] = {"topic": cells[1], "title": cells[2], "author": cells[3], "year": cells[4],
                          "id": cells[5], "source": cells[6], "status": cells[7]}
    return refs


# --------------------------------------------------------------------------- markdown
def normalise(md: str) -> str:
    """GitHub-style Markdown to what Python-Markdown expects: a blank line before
    a list that follows a paragraph line, and four-space indents for nested lists."""
    out = []
    item = re.compile(r"^(> ?)?(\s*)([-*]|\d+\.) ")
    for line in md.splitlines():
        m = item.match(line)
        if m:
            quote, indent = m.group(1) or "", m.group(2)
            if indent and len(indent) % 4:
                line = quote + " " * (2 * len(indent)) + line[len(quote) + len(indent):]
            prev = out[-1] if out else ""
            prev_body = prev[len(quote):] if quote and prev.startswith(quote.strip()) else prev
            prev_body = prev_body.lstrip("> ").rstrip() if quote else prev_body.rstrip()
            if prev_body and not item.match(prev) and not prev_body.startswith(("|", "```")) \
                    and not prev.startswith(" ") and not indent:
                out.append(quote.rstrip() if quote else "")
        out.append(line)
    return "\n".join(out)


class PalettedStyle:
    pass


def python_html(code: str) -> str:
    return highlight(code, PythonLexer(), HtmlFormatter(nowrap=True))


def render_body(md: str, refs: dict, cited: set) -> str:
    md = normalise(md)
    # badges: a line made only of `BADGE` tokens (optionally with a parenthesis after each)
    def badge_line(m):
        pills = []
        for b, note in re.findall(r"`([A-Z][A-Z -]+)`(?:\s*\(([^)]*)\))?", m.group(0)):
            if b in BADGES:
                extra = f'<span class="badge-note">{html.escape(note)}</span>' if note else ""
                pills.append(f'<span class="badge b-{BADGES[b]}">{b}</span>{extra}')
        return '<div class="badges">' + "".join(pills) + "</div>" if pills else m.group(0)
    md = re.sub(r"(?m)^`(?:IN THE REPO|NOT IN THE REPO|MEASURED|ANALOGY|RULE OF THUMB|EXTENSION|"
                r"IMPLEMENTATION-DEPENDENT)`.*$", badge_line, md)
    # fenced code inside a blockquote (a Details box): fenced_code does not see fences behind "> ",
    # so lift each one out, leave a placeholder paragraph in the quote, and render it after markdown
    quoted_code: list[tuple[str, str]] = []

    def lift(m):
        lines = [re.sub(r"^[ \t]*> ?", "", l) for l in m.group(0).split("\n")]
        lang = lines[0].strip().lstrip("`").strip() or "text"
        quoted_code.append((lang, "\n".join(lines[1:-1])))
        return f"{m.group(1)}> XQQUOTEDCODE{len(quoted_code) - 1}XQ"
    md = re.sub(r"(?m)^([ \t]*)> ?```[^\n]*\n(?:[ \t]*>[^\n]*\n)*?[ \t]*> ?```[ \t]*$", lift, md)
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "md_in_html"], output_format="html5")
    body = re.sub(r"<p>XQQUOTEDCODE(\d+)XQ</p>",
                  lambda m: '<pre><code class="language-{}">{}</code></pre>'.format(
                      quoted_code[int(m.group(1))][0], html.escape(quoted_code[int(m.group(1))][1])), body)

    # code: python highlighted; mermaid handed to the viewer; plain blocks marked as output
    def code(m):
        lang, src = m.group(1), html.unescape(m.group(2))
        if lang == "mermaid":
            return f'<pre class="mermaid">{html.escape(src)}</pre>'
        if lang == "python":
            return f'<pre class="code"><code>{python_html(src)}</code></pre>'
        if lang in ("bash", "sh"):
            return f'<pre class="code shell"><code>{html.escape(src)}</code></pre>'
        return f'<pre class="output"><code>{html.escape(src)}</code></pre>'
    body = re.sub(r'<pre><code class="language-(\w+)">(.*?)</code></pre>', code, body, flags=re.S)
    body = re.sub(r"<pre><code>(.*?)</code></pre>",
                  lambda m: f'<pre class="output"><code>{m.group(1)}</code></pre>', body, flags=re.S)

    # figures: an image paragraph becomes the inlined SVG; the caption paragraph after it becomes its caption
    body = re.sub(r'<p><img alt="([^"]*)" src="([^"]+\.svg)"\s*/?></p>', lambda m: inline_figure(m.group(2), m.group(1)), body)
    body = re.sub(r'</figure>\s*<p><em>(Figure [^<]*)</em>(.*?)</p>',
                  r'<figcaption><strong>\1</strong>\2</figcaption></figure>', body, flags=re.S)
    body = body.replace("</figure><figcaption>", "<figcaption>")

    # the citation strip: "> **In the repo** · `file` · `symbol` · lines · @pin"
    def strip(m):
        parts = [re.sub(r"</?code>", "", p).strip() for p in m.group(1).split("·")]
        cells = "".join(f'<span class="cite-{k}">{p}</span>' for k, p in zip(("file", "sym", "lines", "pin"), parts))
        link = public_link(parts[0], parts[2]) if len(parts) >= 3 else None
        gh = f'<a class="cite-gh" href="{link}">view on GitHub</a>' if link else ""
        return f'<div class="cite"><span class="cite-label">In the repo</span>{cells}{gh}</div>'
    body = re.sub(r"<blockquote>\s*<p><strong>In the repo</strong> · (.*?)</p>\s*</blockquote>", strip, body,
                  flags=re.S)

    # asides
    def aside(m):
        inner = m.group(1)
        if inner.lstrip().startswith("<p><strong>In one breath.</strong>"):
            inner = inner.replace("<strong>In one breath.</strong>", "", 1)
            return f'<aside class="breath"><div class="breath-label">In one breath</div>{inner}</aside>'
        if inner.lstrip().startswith("<p><strong>Details.</strong>"):
            inner = inner.replace("<strong>Details.</strong>", "", 1)
            return f'<details class="details"><summary>Details</summary><div>{inner}</div></details>'
        if inner.lstrip().startswith("<p><strong>Build it.</strong>"):
            inner = inner.replace("<strong>Build it.</strong>", "", 1)
            return f'<aside class="build"><div class="build-label">Build it</div>{inner}</aside>'
        return f"<blockquote>{inner}</blockquote>"
    body = re.sub(r"<blockquote>(.*?)</blockquote>", aside, body, flags=re.S)

    # evidence labels at the start of a paragraph
    body = re.sub(r"<p><em>(verified|measured|illustrative|cited)</em>:?\s*",       # the label replaces the colon
                  lambda m: f'<p><span class="ev ev-{m.group(1)}">{m.group(1)}</span>', body)
    body = re.sub(r"<em>(verified on this machine, PyTorch [\d.]+)</em>",
                  r'<span class="ev ev-verified">\1</span>', body)

    # research citations
    def ref(m):
        rid = m.group(1)
        cited.add(rid)
        r = refs.get(rid, {})
        tip = html.escape(f'{r.get("title", "not in the register")} — {r.get("author", "")} {r.get("year", "")}'
                          f' · {r.get("status", "")}', quote=True)
        return f'<a class="ref" href="#src-{rid}" title="{tip}">{rid}</a>'
    body = re.sub(r"\[([RS]\d{2})\]", ref, body)

    body = re.sub(r"<table>", '<div class="table"><table>', body)
    body = re.sub(r"</table>", "</table></div>", body)
    return body


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", re.sub(r"<[^>]+>", "", s).lower()).strip("-")


def add_ids(body: str, prefix: str) -> tuple[str, list[tuple[str, str]]]:
    toc = []

    def h2(m):
        text = m.group(1)
        sid = f"{prefix}-{slug(text)}"
        label = re.sub(r"<[^>]+>", "", text)
        num = re.match(r"(\d+\.\d+)\s+(.*)", label)
        toc.append((sid, label))
        if num:
            text = f'<span class="secnum">{num.group(1)}</span>{html.escape(num.group(2))}'
        return f'<h2 id="{sid}">{text}</h2>'
    body = re.sub(r"<h2>(.*?)</h2>", h2, body)
    body = re.sub(r"<h1>.*?</h1>", "", body, count=1, flags=re.S)
    return body, toc


# --------------------------------------------------------------------------- page
CSS = """
:root {
  /* Layout: a reading column with a contents rail; part dividers and the cover are full-bleed night bands. */
  --night: #13232A; --ink: #17262E; --paper: #FFFFFF; --card: #F0F2F1; --slate: #56676C;
  --coral: #C4532F; --gold: #D9A13B; --gold-text: #8F6A1E; --teal: #2B8C82; --plum: #7B5E8E;
  --rule: #DCE1DF; --muted: #5E6E72; --code-bg: #13232A; --code-fg: #E8EEEC; --link: #1F6F67;
  --chart-coral: #C4532F; --chart-teal: #0E9488; --heat: #A8701A; --onheat: #FFFFFF;
  --display: "Lora", "Source Serif 4", Georgia, serif;
  --body: "Lato", "Source Sans 3", "Segoe UI", system-ui, sans-serif;
  --mono: "JetBrains Mono", Consolas, "SFMono-Regular", monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --ink: #E6ECEA; --paper: #0F1B20; --card: #18282F; --slate: #93A4A8; --rule: #26383F; --muted: #9AAAAE;
  --coral: #EE8A69; --gold-text: #E2B45A; --teal: #4FB3A8; --plum: #B095C2; --code-bg: #0A1418; --link: #6CC5BB;
  --chart-coral: #D9714F; --chart-teal: #139C8F; --heat: #E2B45A; --onheat: #13232A;
  color-scheme: dark } }
:root[data-theme="dark"] {
  --ink: #E6ECEA; --paper: #0F1B20; --card: #18282F; --slate: #93A4A8; --rule: #26383F; --muted: #9AAAAE;
  --coral: #EE8A69; --gold-text: #E2B45A; --teal: #4FB3A8; --plum: #B095C2; --code-bg: #0A1418; --link: #6CC5BB;
  --chart-coral: #D9714F; --chart-teal: #139C8F; --heat: #E2B45A; --onheat: #13232A;
  color-scheme: dark }
* { box-sizing: border-box }
body { background: var(--paper); color: var(--ink); font: 17px/1.65 var(--body); margin: 0 }
a { color: var(--link) }
.band { background: var(--night); color: #EEF2F0; padding-block: 56px 48px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)) }
.band-inner { max-width: 1120px; margin: 0 auto }
.eyebrow { font: 600 12px/1 var(--body); letter-spacing: .14em; text-transform: uppercase; color: var(--gold) }
.cover h1 { font: 600 clamp(36px, 6vw, 60px)/1.05 var(--display); margin: 14px 0 10px; text-wrap: balance; color: #FFFFFF }
.cover .sub { font: italic 400 clamp(18px, 2.4vw, 23px)/1.4 var(--display); color: #C9D4D1; margin: 0 0 22px; max-width: 34ch }
.cover .meta { font: 400 13px/1.6 var(--mono); color: #93A6A3 }
.cover .byline { font: 600 15px/1.4 var(--body); color: #EEF2F0; margin: 0 0 14px }
.cover .links { display: flex; flex-wrap: wrap; gap: 10px; margin: 0 0 20px }
.cover .links a { color: #13232A; background: var(--gold); font: 700 13px/1 var(--body); padding: 10px 14px;
  border-radius: 999px; text-decoration: none }
.cover .links a:hover, .cover .links a:focus-visible { background: #F0C46A; outline: 2px solid #F0C46A; outline-offset: 2px }
.about { padding-block: 44px 8px; padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); border-bottom: 1px solid var(--rule) }
.about-inner { max-width: 1120px; margin: 0 auto; max-width: 78ch }
.about h3 { font: 600 20px/1.3 var(--display); margin: 26px 0 10px }
.about .eyebrow { color: var(--gold-text) }
.cite-gh { margin-left: auto; font: 700 11px/1 var(--body); letter-spacing: .06em; color: var(--link); text-decoration: none;
  border: 1px solid var(--rule); padding: 5px 8px; border-radius: 999px; white-space: nowrap }
.cite-gh:hover, .cite-gh:focus-visible { border-color: var(--link) }
figure.fig { margin: 26px 0 28px; padding: 18px 18px 14px; background: var(--paper); border: 1px solid var(--rule);
  border-radius: 10px; overflow-x: auto }
figure.fig svg { display: block; width: 100%; min-width: 520px; height: auto }
figure.fig figcaption { font-size: 14.5px; line-height: 1.55; color: var(--muted); margin-top: 12px; max-width: 70ch }
figure.fig figcaption strong { color: var(--ink) }
.figure-svg .f-ink { fill: var(--ink) } .figure-svg .f-muted { fill: var(--muted) }
.figure-svg .f-onfill { fill: #13232A }
.figure-svg .f-rule { stroke: var(--rule) } .figure-svg .f-rule-fill { fill: var(--rule) }
.figure-svg .f-card { fill: var(--card) } .figure-svg .f-paper { fill: var(--paper) }
.figure-svg .f-gold { fill: var(--gold) } .figure-svg .f-heat { fill: var(--heat) } .figure-svg .f-onheat { fill: var(--onheat) } .figure-svg .f-gold-stroke { stroke: var(--gold) }
.figure-svg .f-coral { fill: var(--chart-coral) } .figure-svg .f-teal { fill: var(--chart-teal) }
.figure-svg .f-coral-stroke { stroke: var(--chart-coral) } .figure-svg .f-teal-stroke { stroke: var(--chart-teal) }
.figure-svg .f-heat-stroke { stroke: var(--heat) }
.figure-svg .f-ring { stroke: var(--paper) }
.figure-svg .f-ink-stroke { stroke: var(--ink) } .figure-svg .f-muted-stroke { stroke: var(--muted) }
.figure-svg .f-muted-fill { fill: var(--muted) }
.part { border-top: 6px solid var(--gold) }
.part h2 { font: 600 clamp(28px, 4vw, 40px)/1.15 var(--display); margin: 10px 0 8px; color: #FFFFFF }
.part p { color: #C9D4D1; max-width: 60ch; margin: 0 }
.wrap { max-width: 1120px; margin: 0 auto; display: grid; grid-template-columns: 240px minmax(0, 1fr); gap: 56px;
  padding-block: 40px 80px; padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)) }
nav.toc { position: sticky; top: calc(env(safe-area-inset-top, 0px) + 20px); align-self: start;
  max-height: calc(100vh - 40px); overflow: auto; font-size: 13.5px; line-height: 1.4 }
nav.toc .toc-ch { font: 600 14px/1.3 var(--display); margin: 18px 0 6px; color: var(--ink) }
nav.toc .toc-ch:first-child { margin-top: 0 }
nav.toc a { display: block; color: var(--muted); text-decoration: none; padding: 3px 0 3px 10px; border-left: 2px solid var(--rule) }
nav.toc a:hover, nav.toc a:focus-visible { color: var(--ink); border-left-color: var(--gold) }
main { min-width: 0 }
article.chapter { max-width: 70ch }
article.chapter + article.chapter { margin-top: 96px; padding-top: 48px; border-top: 1px solid var(--rule) }
.ch-head .eyebrow { color: var(--gold-text) }
.ch-head h1 { font: 600 clamp(30px, 4.4vw, 44px)/1.12 var(--display); margin: 10px 0 28px; text-wrap: balance }
.ch-head h1 .chnum { color: var(--gold-text); margin-right: 14px }
h2 { font: 600 25px/1.25 var(--display); margin: 56px 0 14px; text-wrap: balance; scroll-margin-top: 20px }
h2 .secnum { font: 500 15px/1 var(--mono); color: var(--gold-text); margin-right: 12px; vertical-align: 3px }
h3 { font: 600 19px/1.3 var(--display); margin: 34px 0 10px }
p { margin: 0 0 16px }
ul, ol { padding-left: 24px; margin: 0 0 16px }
li { margin: 4px 0 }
li > ul, li > ol { margin: 4px 0 }
hr { border: 0; border-top: 1px solid var(--rule); margin: 36px 0 }
code { font: 0.86em var(--mono); background: var(--card); padding: 1px 5px; border-radius: 3px }
pre { margin: 0 0 18px; overflow-x: auto; border-radius: 6px; font: 13.5px/1.55 var(--mono) }
pre code { background: none; padding: 0; font: inherit }
pre.code { background: var(--code-bg); color: var(--code-fg); padding: 16px 18px }
pre.code.shell::before { content: "$ "; color: var(--gold) }
/* print.html only: print_layout puts each code line in a .ln block (so it also reads right on screen) */
pre .ln { display: block }
pre.code.shell:has(.ln)::before { content: none }
pre.code.shell .ln:first-child::before { content: "$ "; color: var(--gold) }
pre.output { background: var(--card); color: var(--ink); padding: 14px 18px; border-left: 3px solid var(--teal) }
pre.mermaid { background: var(--card); padding: 18px; text-align: center; border-radius: 6px }
.cite { display: flex; flex-wrap: wrap; gap: 4px 14px; align-items: baseline; margin: 26px 0 8px;
  font: 13px/1.5 var(--mono); padding: 8px 12px; border-left: 3px solid var(--gold); background: var(--card) }
.cite-label { font: 700 11px/1 var(--body); letter-spacing: .12em; text-transform: uppercase; color: var(--gold-text) }
.cite-sym { color: var(--ink); font-weight: 600 }
.cite-file, .cite-lines, .cite-pin { color: var(--muted) }
.cite + pre.code { border-top-left-radius: 0 }
.badges { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 4px 0 8px }
.badge { font: 700 10.5px/1 var(--body); letter-spacing: .1em; padding: 5px 9px; border-radius: 999px; border: 1.5px solid }
.badge-note { font-size: 13px; color: var(--muted); margin: 0 6px 0 -2px }
.b-repo { color: var(--teal); border-color: var(--teal) }
.b-notrepo { color: var(--slate); border-color: var(--slate) }
.b-measured { color: var(--gold-text); border-color: var(--gold) }
.b-analogy, .b-rule { color: var(--plum); border-color: var(--plum) }
.b-extension { color: var(--plum); border-color: var(--plum); border-style: dashed }
.b-impl { color: var(--coral); border-color: var(--coral) }
.ev { font: 700 10.5px/1 var(--body); letter-spacing: .1em; text-transform: uppercase; padding: 3px 7px;
  border-radius: 3px; margin-right: 6px; vertical-align: 2px; color: var(--paper) }
.ev-verified { background: var(--teal) }
.ev-measured { background: var(--gold-text) }
.ev-illustrative { background: var(--plum) }
.ev-cited { background: var(--slate) }
.ref { font: 600 11px/1 var(--mono); text-decoration: none; color: var(--link); background: var(--card);
  border: 1px solid var(--rule); padding: 2px 5px; border-radius: 4px; white-space: nowrap }
.table { overflow-x: auto; margin: 0 0 20px }
table { border-collapse: collapse; font-size: 14.5px; width: 100%; font-variant-numeric: tabular-nums }
th { background: var(--night); color: #EEF2F0; text-align: left; font-weight: 700; padding: 8px 10px }
th code { background: #22363E; color: #EEF2F0 }
td { padding: 7px 10px; border-bottom: 1px solid var(--rule); vertical-align: top }
.breath { background: var(--night); color: #EEF2F0; border-radius: 8px; padding: 22px 26px; margin: 28px 0 }
.breath p { font: italic 400 19px/1.55 var(--display); margin: 0 }
.breath code { background: #22363E; color: #EEF2F0 }
.breath-label, .build-label { font: 700 11px/1 var(--body); letter-spacing: .14em; text-transform: uppercase; margin-bottom: 10px }
.breath-label { color: var(--gold) }
.build { border: 1.5px solid var(--teal); border-radius: 8px; padding: 16px 20px; margin: 24px 0 }
.build-label { color: var(--teal) }
.build p:last-child, .build ul:last-child { margin-bottom: 0 }
details.details { border: 1px solid var(--rule); border-radius: 6px; margin: 18px 0; background: var(--card) }
details.details summary { cursor: pointer; padding: 10px 14px; font: 700 12px/1 var(--body); letter-spacing: .12em;
  text-transform: uppercase; color: var(--slate) }
details.details summary:focus-visible { outline: 2px solid var(--gold); outline-offset: 2px }
details.details > div { padding: 0 16px 6px; font-size: 15.5px }
.sources { margin-top: 96px; padding-top: 40px; border-top: 1px solid var(--rule); max-width: 70ch }
.sources h2 { margin-top: 0 }
.sources li { margin: 10px 0; font-size: 15px }
.sources .sid { font: 600 12px var(--mono); color: var(--gold-text); margin-right: 8px }
.toc-mobile { display: none }
@media (max-width: 900px) {
  .wrap { grid-template-columns: minmax(0, 1fr); gap: 0 }
  nav.toc { display: none }
  .toc-mobile { display: block; margin-bottom: 28px; border: 1px solid var(--rule); border-radius: 6px }
  .toc-mobile summary { padding: 12px 14px; font: 700 12px/1 var(--body); letter-spacing: .12em; text-transform: uppercase; cursor: pointer }
  .toc-mobile nav { padding: 0 14px 12px; font-size: 14px }
  .toc-mobile a { display: block; padding: 4px 0; color: var(--muted); text-decoration: none }
  body { font-size: 16.5px }
}
/* Pygments, in the book's palette (code blocks are always dark) */
.code .k, .code .kn, .code .kc, .code .kd, .code .ow { color: #E5A76A }
.code .nf, .code .nc { color: #8FD1C9 }
.code .s, .code .s1, .code .s2, .code .sd, .code .si, .code .sa { color: #D7C08A }
.code .c, .code .c1, .code .cm { color: #8A9D9A; font-style: italic }
.code .mi, .code .mf, .code .m { color: #E3A9C9 }
.code .nb, .code .bp { color: #A9C7E8 }
.code .o, .code .p { color: #C9D4D1 }
@media (prefers-reduced-motion: no-preference) { html { scroll-behavior: smooth } }
.print-only { display: none }
/* Print: a 7 × 9.25 in technical-book page. Built with --print and rendered to PDF by Chrome. */
@page { size: 7in 9.25in; margin: 0.7in 0.65in 0.75in }
@page cover { margin: 0 }
@media print {
  :root { --paper: #FFFFFF; --ink: #17262E; --card: #F0F2F1; --rule: #DCE1DF; --muted: #4E5E62; --link: #1F6F67;
    --gold-text: #8F6A1E; --heat: #A8701A; --onheat: #FFFFFF; color-scheme: light }
  * { -webkit-print-color-adjust: exact; print-color-adjust: exact }
  /* Sizes are the printed sizes: nothing may be wider than the 5.7 in measure, or Chrome shrinks every page to fit. */
  body { font-size: 9.4pt; line-height: 1.45; orphans: 3; widows: 2 }
  .print-only { display: block }
  nav.toc, .toc-mobile, .cite-gh, .cover .links { display: none }
  .band { padding-left: 0.65in; padding-right: 0.65in }
  .band.cover { page: cover; height: 9.25in; padding-top: 2.6in; break-after: page }
  .cover h1 { font-size: 40pt } .cover .sub { font-size: 15pt }
  .about { padding: 0; border: 0; break-after: page; font-size: 9pt } .about-inner { max-width: none }
  .about h3 { font-size: 11.5pt; margin: 12px 0 5px } .about table { font-size: 7.8pt }
  .contents { break-after: page } .contents h2 { margin-top: 0 }
  .contents .c-ch { font: 600 10.5pt/1.3 var(--display); margin: 11px 0 3px; break-after: avoid }
  .contents .c-sec { font-size: 8.4pt; line-height: 1.35; color: var(--muted); margin: 1px 0 1px 18px }
  .band.part { page: cover; break-before: page; break-after: page; height: 9.25in; padding-top: 3.4in;
    border-top-width: 10px }
  .part h2 { font-size: 28pt } .part p { font-size: 11pt }
  .wrap { display: block; padding: 0; max-width: none }
  article.chapter { max-width: none; break-before: page }
  article.chapter + article.chapter { margin-top: 0; padding-top: 0; border: 0 }
  .ch-head h1 { font-size: 22pt; margin: 8px 0 20px }
  p { margin: 0 0 9pt } ul, ol { margin: 0 0 9pt } li { margin: 2pt 0 }
  /* A heading stays with what follows; orphans: 3 then keeps at least three lines of its first paragraph under it.
     (A zero-net-height ::after "reserve" under headings was tried: Chrome then pushed whole blocks to the next page.) */
  h2 { font-size: 13.5pt; margin: 20pt 0 6pt; break-after: avoid; break-inside: avoid }
  h3 { font-size: 11pt; margin: 14pt 0 5pt; break-after: avoid; break-inside: avoid }
  h2 .secnum { font-size: 10pt; vertical-align: 1pt; margin-right: 8pt }
  /* The citation strip stays with its excerpt; a section's closing badges stay with the section. Lead-in paragraphs
     are not chained to the block after them: with short blocks kept whole, that chain left half-empty pages. */
  .cite { break-after: avoid } .badges { break-before: avoid }
  p.lead { break-after: avoid; break-inside: avoid }      /* marked by print_layout: a short lead-in right under a heading */
  /* overflow: visible matters: Chrome never breaks inside a scroll container, so a long block jumped whole to the next page. */
  pre { font-size: 7pt; line-height: 1.42; white-space: pre-wrap; overflow-wrap: anywhere; margin-bottom: 10pt;
    overflow: visible; orphans: 1; widows: 1 }   /* the line rules below do this job; orphans > 1 here pushed short blocks */
  pre.code { padding: 7pt 8pt } pre.output { padding: 6pt 8pt }
  pre.short { break-inside: avoid }
  /* One block per source line, so a wrapped line hangs under its own indentation instead of at column 0. */
  pre > code { display: block }       /* line blocks inside an inline <code> would make the whole block unbreakable */
  pre .ln { display: block; padding-left: calc(var(--i, 0ch) + 3ch); text-indent: calc(-1 * (var(--i, 0ch) + 3ch)) }
  /* a long block breaks with at least three lines on each side; a short one is kept whole by break-inside alone
     (adding these line rules to a short block made Chrome push it to the next page even when it fitted) */
  pre:not(.short) .ln:nth-child(-n+3) { break-after: avoid } pre:not(.short) .ln:nth-last-child(-n+3) { break-before: avoid }
  .cite { font-size: 7pt; margin: 14pt 0 5pt; padding: 5pt 8pt }
  code { padding: 0 2pt }
  .badges { margin: 2pt 0 6pt } .badge { font-size: 7pt; padding: 3pt 6pt } .badge-note { font-size: 8.4pt; margin: 0 4pt 0 0 }
  .ev { font-size: 6.8pt; padding: 1.5pt 4pt; vertical-align: 1pt }
  .breath { padding: 12pt 16pt; margin: 16pt 0 } .breath p { font-size: 11pt; line-height: 1.45 }
  .build { padding: 9pt 12pt; margin: 14pt 0 }
  figure.fig { break-inside: avoid; padding: 4pt 0; margin: 12pt 0 14pt; border: 0 }
  figure.fig svg { min-width: 0; max-height: 6.2in }
  figure.fig figcaption { font-size: 8pt; line-height: 1.45; margin-top: 6pt; max-width: none }
  table { font-size: 8pt; line-height: 1.35 } th { padding: 4pt 6pt } td { padding: 3.5pt 6pt }
  .table { overflow: visible; margin-bottom: 12pt } .table.short { break-inside: avoid }
  tr, .breath, .build { break-inside: avoid }
  details.details { margin: 12pt 0 } details.details > div { font-size: 9pt; padding: 0 10pt 4pt }
  details.details summary { padding: 7pt 10pt; break-after: avoid } details.details summary::marker { content: "" }
  hr { margin: 18pt 0 }
  .sources { break-before: page; margin-top: 0; padding-top: 0; border: 0 } .sources li { font-size: 8.6pt; margin: 4pt 0 }
}
"""


SHORT_PRE, SHORT_TABLE = 12, 8          # lines / body rows: blocks this small are never split across pages


def print_layout(page: str) -> str:
    """Markup the printed page needs and the screen page does not:

    - each code line in its own block, with its indentation as --i, so a line
      too long for the measure wraps with a hanging indent;
    - short code blocks and short tables marked `short`, which keeps them on one page;
    - break opportunities after _ / . in inline code, so a long identifier or path
      breaks inside a table cell instead of widening the table past the page."""
    def lines(m):
        cls, src = m.group(1), m.group(2)
        out, open_tags = [], []
        rows = src.split("\n")
        if rows and rows[-1] == "":
            rows.pop()
        for row in rows:
            body = "".join(open_tags) + row
            for tag in re.findall(r"<span[^>]*>|</span>", row):
                if tag == "</span>":
                    if open_tags:
                        open_tags.pop()
                else:
                    open_tags.append(tag)
            body += "</span>" * len(open_tags)
            text = html.unescape(re.sub(r"<[^>]+>", "", row))
            indent = len(text) - len(text.lstrip(" "))
            style = f' style="--i:{indent}ch"' if indent else ""
            out.append(f'<span class="ln"{style}>{body if text.strip() else body + " "}</span>')
        short = " short" if len(rows) <= SHORT_PRE else ""
        return f'<pre class="{cls}{short}"><code>{"".join(out)}</code></pre>'
    page = re.sub(r'<pre class="((?:code|output)[^"]*)"><code>(.*?)</code></pre>', lines, page, flags=re.S)

    def long_tokens(text):                # text between tags: a token over 18 characters may break after . / _ :
        return re.sub(r"[^\s<>]{19,}", lambda t: re.sub(r"(?<=[_/.:])(?=[^\s&;])", "<wbr>", t.group(0)), text)

    def table(m):
        rows = len(re.findall(r"<tr>", m.group(1))) - 1
        inner = re.sub(r">([^<]+)<", lambda t: ">" + long_tokens(t.group(1)) + "<", m.group(1))
        return f'<div class="table{" short" if rows <= SHORT_TABLE else ""}"><table>{inner}</table></div>'
    page = re.sub(r'<div class="table"><table>(.*?)</table></div>', table, page, flags=re.S)

    # a one- or two-line paragraph between a heading and a block ("... in banner sections:") travels with both,
    # so a heading never ends a page with one line under it
    def lead(m):
        head, first, first_in, gap, second_in, after = m.groups()
        if any(len(re.sub(r"<[^>]+>", "", p)) >= 190 for p in (first_in or "", second_in)):
            return m.group(0)
        first = f'<p class="lead">{first_in}</p>{gap}' if first else ""
        return f'{head}{first}<p class="lead">{second_in}</p>{after}'
    page = re.sub(r'(</h[23]>\s*)(<p>([^\n]*?)</p>(\s*))?<p>([^\n]*?)</p>(\s*<(?:pre|ul|ol|div class="(?:table|cite|badges)))',
                  lead, page)

    def breakable(m):
        inner = m.group(1)
        if "<" in inner or len(inner) < 12:
            return m.group(0)
        return "<code>" + re.sub(r"(?<=[_/.])(?=[^\s&;])", "<wbr>", inner) + "</code>"
    parts = re.split(r"(<pre[ >].*?</pre>)", page, flags=re.S)
    return "".join(p if p.startswith("<pre") else re.sub(r"<code>(.*?)</code>", breakable, p) for p in parts)


def build(wanted: list[str], print_mode: bool = False) -> Path:
    refs = register()
    cited: set[str] = set()
    files = chapter_files(wanted)
    chapters, toc_all, parts = [], [], []
    for f in files:
        meta, md = front_matter(f.read_text(encoding="utf-8"))
        num, _, title = meta.get("title", f.stem).partition(" · ")
        anchor = "ch" + re.sub(r"[^A-Za-z0-9]+", "-", num).strip("-")          # "11", "Appendix-A"
        body, toc = add_ids(render_body(md, refs, cited), anchor)
        part = meta.get("part", "")
        roman = part.split(" · ")[0].strip()
        chapters.append((roman, part, num, title, body, anchor))
        toc_all.append((num, title, toc))
        if roman and roman not in parts:
            parts.append(roman)

    toc_html = "".join(
        f'<div class="toc-ch">{html.escape(n)} · {html.escape(t)}</div>'
        + "".join(f'<a href="#{sid}">{html.escape(label)}</a>' for sid, label in toc)
        for n, t, toc in toc_all)
    pin = re.search(r"`@([0-9a-f]{7,40})`", (BOOK / "REPOSITORY_SOURCE_MAP.md").read_text(encoding="utf-8")).group(1)
    names = ", ".join(n for n, _, _ in toc_all)
    # FRONT_MATTER.md's `author:` field is the one place the author is set
    fm_meta, fm_md = front_matter((BOOK / "FRONT_MATTER.md").read_text(encoding="utf-8"))
    author = fm_meta.get("author", "").strip()
    byline = "t" + author[1:] if author.startswith("The ") else author      # "By the AnuLM authors", "By Jane Doe"
    out = [f"""<title>Inside AnuLM</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,500;0,600;1,400&family=Lato:wght@400;700&family=JetBrains+Mono:wght@400;500;600&display=swap">
<style>{CSS}</style>
<header class="band cover"><div class="band-inner">
<div class="eyebrow">Reference book · preview</div>
<h1>Inside AnuLM</h1>
<p class="sub">A language model you can read, line by line</p>
<div class="byline">By {html.escape(byline)}</div>
<div class="links"><a href="{GITHUB}">Source code on GitHub</a><a href="https://huggingface.co/toonist/AnuLM-Base-v2-400M">The model on Hugging Face</a><a href="#about">About this book</a></div>
<div class="meta">Chapters {html.escape(names)} of 41 · code cited at @{pin} · every number printed by a script in book/code-examples</div>
</div></header>"""]
    fm_md = re.sub(r"(?s)^# .*?\n\*\*By [^*\n]+\*\*\n", "", fm_md.lstrip())
    about = render_body(fm_md, refs, set())
    about = re.sub(r"<h2>(.*?)</h2>", lambda m: f'<h3>{m.group(1)}</h3>', about)
    out.append(f'<section class="about" id="about"><div class="about-inner"><div class="eyebrow">About this book</div>'
               f'{about}</div></section>')
    contents = "".join(f'<div class="c-ch">{html.escape(n)} · {html.escape(t)}</div>'
                       + "".join(f'<div class="c-sec">{html.escape(label)}</div>' for _, label in toc)
                       for n, t, toc in toc_all)
    out.append(f'<section class="contents print-only band-inner"><h2>Contents</h2>{contents}</section>')
    current = None
    for roman, part, num, title, body, anchor in chapters:
        if roman != current:
            if current is not None:
                out.append("</main></div>")
            name, blurb = PARTS.get(roman, (part.partition(" · ")[2], ""))
            label = "Back matter" if roman == "Appendices" else f"Part {roman}"
            out.append(f'<section class="band part"><div class="band-inner"><div class="eyebrow">{label}</div>'
                       f'<h2>{html.escape(name)}</h2><p>{html.escape(blurb)}</p></div></section>')
            out.append(f'<div class="wrap"><nav class="toc" aria-label="Contents">{toc_html}</nav><main>'
                       f'<details class="toc-mobile"><summary>Contents</summary><nav>{toc_html}</nav></details>')
            current = roman
        eyebrow = "Appendices" if roman == "Appendices" else f"Part {html.escape(part)}"
        out.append(f'<article class="chapter" id="{anchor}"><header class="ch-head"><div class="eyebrow">'
                   f'{eyebrow}</div><h1><span class="chnum">{html.escape(num)}</span>{html.escape(title)}</h1>'
                   f'</header>{body}</article>')
    items = []
    for rid in sorted(cited):
        r = refs.get(rid, {})
        src = r.get("source", "")
        link = f' <a href="https://{html.escape(src)}">{html.escape(src)}</a>' if src and not src.startswith("http") else ""
        items.append(f'<li id="src-{rid}"><span class="sid">{rid}</span>{html.escape(r.get("author", ""))} '
                     f'({html.escape(r.get("year", ""))}). <em>{html.escape(r.get("title", ""))}</em>. '
                     f'{html.escape(r.get("id", ""))}.{link} <span class="ev ev-verified">{html.escape(r.get("status", ""))}</span></li>')
    out.append(f'<section class="sources"><h2>Sources cited in these chapters</h2><p>From the book\'s register, '
               f'<code>RESEARCH_REFERENCES.md</code>. Every entry was opened at its source and checked.</p>'
               f'<ul>{"".join(items)}</ul></section>')
    out.append("</main></div>")
    page = "\n".join(out)
    target = OUT
    if print_mode:                                   # a full HTML document, Details open, for Chrome's PDF printer
        page = print_layout(page.replace('<details class="details">', '<details class="details" open>'))
        page = ('<!doctype html><html lang="en"><head><meta charset="utf-8">' + page.replace("<style>", "<style>", 1)
                .replace("</style>", "</style></head><body>", 1) + "</body></html>")
        target = OUT.with_name("print.html")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(page, encoding="utf-8")
    return target


def to_pdf(html_path: Path) -> Path:
    """Print with headless Chrome or Edge, then count the pages."""
    pdf = html_path.with_name("inside-anulm.pdf")
    for exe in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", "google-chrome", "chromium"):
        try:
            subprocess.run([exe, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                            "--virtual-time-budget=15000", f"--print-to-pdf={pdf}", html_path.as_uri()],
                           check=True, capture_output=True, timeout=300)
            return pdf
        except (OSError, subprocess.CalledProcessError):
            continue
    raise SystemExit("no Chrome or Edge found to print the PDF")


def add_bookmarks(pdf: Path) -> int:
    """Give the printed PDF an outline: one bookmark per chapter and appendix,
    with its numbered sections beneath. Chrome prints none, and 500 pages with
    no bookmarks are hard to use. A heading is placed on the first page, at or
    after the previous heading's, whose text holds it as a line of its own."""
    import pymupdf
    doc = pymupdf.open(pdf)
    pages = []
    for i in range(doc.page_count):
        ls = [" ".join(l.split()) for l in doc[i].get_text().splitlines()]
        def join(*parts):                   # "multi-" + "head" wraps at the hyphen, without a space
            out = parts[0]
            for p in parts[1:]:
                out += p if out.endswith("-") else " " + p
            return out
        pages.append(set(ls) | {join(a, b) for a, b in zip(ls, ls[1:])}     # a long title wraps
                     | {join(a, b, c) for a, b, c in zip(ls, ls[1:], ls[2:])})
    # Skip the cover, front matter and contents: start at the page that opens Chapter 1.
    start = next((i for i, ls in enumerate(pages) if i > 3 and "1 What this book builds" in ls), 0)
    files = sorted(CHAPTERS.glob("[0-9][0-9]-*.md")) + sorted(CHAPTERS.glob("A[0-9]*-*.md"), key=lambda p: int(p.name[1:p.name.index("-")]))
    toc, at = [], start
    for f in files:
        fenced = False
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.startswith("```"):
                fenced = not fenced                 # a "# comment" in a code block is not a heading
                continue
            m = None if fenced else re.match(r"^(#{1,2})\s+(\S+)\s+(.+?)\s*$", line)
            if not m:
                continue
            level, num, title = len(m.group(1)), m.group(2), re.sub(r"[`*]", "", m.group(3)).strip()
            heading = f"{num} {' '.join(title.split())}"          # how Chrome lays it out: "1 What this book builds"
            for i in range(at, doc.page_count):
                if heading in pages[i]:
                    toc.append([level, f"{num}  {title}", i + 1])
                    at = i
                    break
    doc.set_toc(toc)
    doc.saveIncr()
    return len(toc)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    printing = "--print" in sys.argv
    p = build(args, print_mode=printing)
    print(f"wrote {p} ({p.stat().st_size / 1024:.0f} KB)")
    if printing:
        pdf = to_pdf(p)
        n_marks = add_bookmarks(pdf)
        import pymupdf
        print(f"wrote {pdf}: {pymupdf.open(pdf).page_count} pages at 7 × 9.25 in, {n_marks} bookmarks")
