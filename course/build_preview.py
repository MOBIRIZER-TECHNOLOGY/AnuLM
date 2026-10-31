"""
One self-contained HTML page with the whole book: cover, contents, part pages,
every chapter, and an index (book_parts.py), styled as the book.

    python build_preview.py                    # -> course/preview/sample.html
    python build_preview.py ch01-smallest-model ch10-lora-qlora

No dependencies: a small Markdown reader for the subset STYLE.md allows
(headings, paragraphs, emphasis, code, tables, lists, callouts, figures),
with every figure's SVG inlined so the page works offline and can be shared
as one file.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import book_parts  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT = ["ch01-smallest-model", "ch02-data", "ch03-tokenizer", "ch04-skeleton", "ch05-positions", "ch06-attention", "ch07-experts", "ch08-pretrain", "ch09-sft", "ch10-lora-qlora", "ch11-dpo", "ch12-decide-rag", "ch13-senses", "ch14-one-model", "ch15-shipping"]


def inline(s: str) -> str:
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![*\w])\*([^*\n]+)\*(?![*\w])", r"<em>\1</em>", s)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2">\1</a>', s)
    return s


def figure(path: Path) -> str:
    svg = path.read_text(encoding="utf-8")
    svg = re.sub(r'\swidth="\d+" height="\d+"', "", svg, count=1)          # scale to the column
    return f'<figure class="plate">{svg}</figure>'


def anchor(chapter: str, heading: str) -> str:
    """A section's id: unique across the book, since every chapter has a "Key takeaways"."""
    return chapter + "--" + re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")


def render(md: str, folder: Path) -> tuple[str, str]:
    title = ""
    m = re.match(r"---\n(.*?)\n---\n", md, re.S)
    if m:
        t = re.search(r'title:\s*"?(.*?)"?\s*$', m.group(1), re.M)
        title = t.group(1) if t else ""
        md = md[m.end():]
    out, lines, i = [], md.split("\n"), 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
        elif line.startswith("```"):
            lang, body = line[3:].strip(), []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                body.append(lines[i])
                i += 1
            i += 1
            out.append(f'<pre class="code" data-lang="{lang}"><code>{html.escape(chr(10).join(body))}</code></pre>')
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            text = line[level:].strip()
            out.append(f'<h{level} id="{anchor(folder.name, text)}">{inline(text)}</h{level}>')
            i += 1
        elif line.startswith("![]("):
            out.append(figure(folder / line[4:line.index(")")]))
            i += 1
        elif line.startswith(">"):
            body = []
            while i < len(lines) and lines[i].startswith(">"):
                body.append(lines[i][1:].strip())
                i += 1
            kind = re.match(r"\*\*(.+?)\*\*", body[0]).group(1) if body and body[0].startswith("**") else ""
            if kind:
                body = body[1:]
            paras = "\n".join(body).split("\n\n")
            cls = {"From the lab notebook": "notebook", "Watch out": "watch"}.get(kind, "note")
            out.append(f'<aside class="callout {cls}"><div class="callout-kind">{inline(kind)}</div>'
                       + "".join(f"<p>{inline(' '.join(p.split()))}</p>" for p in paras if p.strip())
                       + "</aside>")
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            head, body = rows[0], [r for r in rows[1:] if not set("".join(r)) <= set("-: ")]
            num = lambda c: ' class="num"' if re.fullmatch(r"[\d.,%×\-−+ ]+", c.replace("*", "")) else ""
            out.append('<div class="table-wrap"><table><thead><tr>'
                       + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr></thead><tbody>"
                       + "".join("<tr>" + "".join(f"<td{num(c)}>{inline(c)}</td>" for c in r) + "</tr>"
                                 for r in body) + "</tbody></table></div>")
        elif re.match(r"(\d+\.|-) ", line):
            ordered = line[0].isdigit()
            items = []
            while i < len(lines) and (re.match(r"(\d+\.|-) ", lines[i]) or lines[i].startswith("   ")
                                      or lines[i].startswith("  ")):
                if re.match(r"(\d+\.|-) ", lines[i]):
                    items.append(re.sub(r"^(\d+\.|-) ", "", lines[i]))
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{inline(t)}</li>" for t in items) + f"</{tag}>")
        else:
            para = []
            while i < len(lines) and lines[i].strip() and not re.match(r"(#|```|>|\||!\[\]|(\d+\.|-) )", lines[i]):
                para.append(lines[i].strip())
                i += 1
            out.append(f"<p>{inline(' '.join(para))}</p>")
    return title, "\n".join(out)


CSS = """
/* Layout: one reading column, figures as full-column plates, chapter openers set large. */
:root {
  --paper: #FBF8F3; --ink: #1F2433; --muted: #5E6472; --rule: #E2DCD2; --wash: #F2EDE5;
  --saffron: #C8651F; --indigo: #3B4CCA; --code-bg: #F2EDE5; --plate: #FBF8F3;
  --serif: "Source Serif 4", "Noto Serif Devanagari", Georgia, serif;
  --sans: "Inter", "Noto Sans Devanagari", "Segoe UI", system-ui, sans-serif;
  --mono: "JetBrains Mono", Consolas, "Courier New", monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --paper: #171A22; --ink: #E9E6E0; --muted: #A3A8B4; --rule: #2E3340; --wash: #20242E;
  --saffron: #F0954F; --indigo: #9AA6F2; --code-bg: #20242E; --plate: #FBF8F3; color-scheme: dark } }
:root[data-theme="dark"] {
  --paper: #171A22; --ink: #E9E6E0; --muted: #A3A8B4; --rule: #2E3340; --wash: #20242E;
  --saffron: #F0954F; --indigo: #9AA6F2; --code-bg: #20242E; --plate: #FBF8F3; color-scheme: dark }
body { background: var(--paper); color: var(--ink); font: 18px/1.65 var(--serif); }
.page { max-width: 44rem; margin: 0 auto; padding-block: 2.5rem 5rem;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
.cover { padding-block: 1rem 2.5rem; border-bottom: 1px solid var(--rule); margin-bottom: 1rem; }
.kicker { font: 600 0.78rem/1.4 var(--sans); letter-spacing: 0.12em; text-transform: uppercase; color: var(--saffron); }
.cover h1 { font: 700 clamp(2.1rem, 6vw, 3.1rem)/1.08 var(--serif); margin: 0.4rem 0 0.6rem; text-wrap: balance; }
.cover p { color: var(--muted); margin: 0; max-width: 36rem; }
.toc { display: flex; flex-wrap: wrap; gap: 0.5rem 1.25rem; margin-top: 1.25rem; font: 500 0.9rem var(--sans); }
.toc a { color: var(--indigo); text-decoration: none; border-bottom: 1px solid currentColor; }
.chapter { padding-top: 3rem; }
.chapter-num { font: 700 4.5rem/1 var(--serif); color: var(--saffron); }
.chapter h1 { font: 700 2.2rem/1.15 var(--serif); margin: 0.25rem 0 1.25rem; text-wrap: balance; }
h2 { font: 650 1.35rem/1.3 var(--sans); margin: 2.4rem 0 0.6rem; text-wrap: balance; }
p, li { max-width: 40rem; }
li { margin: 0.25rem 0; }
code { font: 0.86em var(--mono); background: var(--wash); padding: 0.08em 0.3em; border-radius: 3px; }
pre.code { background: var(--code-bg); border-left: 3px solid var(--indigo); padding: 0.9rem 1rem;
  overflow-x: auto; font: 0.8rem/1.55 var(--mono); border-radius: 0 6px 6px 0; }
pre.code code { background: none; padding: 0; font: inherit; }
figure.plate { margin: 1.75rem 0; background: var(--plate); border: 1px solid var(--rule); border-radius: 6px;
  overflow-x: auto; }
figure.plate svg { display: block; width: 100%; height: auto; min-width: 30rem; }  /* below 480px, scroll rather than shrink labels past reading */
.table-wrap { overflow-x: auto; margin: 1.25rem 0; }
table { border-collapse: collapse; font: 0.9rem/1.45 var(--sans); min-width: 100%; }
th { text-align: left; font-weight: 650; border-bottom: 2px solid var(--ink); padding: 0.45rem 0.7rem; }
td { border-bottom: 1px solid var(--rule); padding: 0.45rem 0.7rem; vertical-align: top; }
td.num { font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }
.callout { margin: 1.75rem 0; padding: 1rem 1.2rem; background: var(--wash); border-radius: 6px; }
.callout p { margin: 0.4rem 0; font-size: 0.95rem; }
.callout-kind { font: 650 0.75rem var(--sans); letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }
.callout.notebook { background: transparent; border: 1.5px solid var(--saffron); }
.callout.notebook .callout-kind { color: var(--saffron); }
a:focus-visible { outline: 2px solid var(--indigo); outline-offset: 2px; }
footer { margin-top: 4rem; padding-top: 1rem; border-top: 1px solid var(--rule); color: var(--muted);
  font: 0.85rem var(--sans); }
@media (max-width: 30rem) { body { font-size: 17px; } .chapter-num { font-size: 3.4rem; } }
"""


def main() -> None:
    names = sys.argv[1:] or DEFAULT
    chapters = []
    for name in names:
        folder = HERE / name
        md = (folder / "chapter.md").read_text(encoding="utf-8")
        title, body = render(md, folder)
        num, _, rest = title.partition("·")
        chapters.append({"name": name, "num": num.strip(), "title": rest.strip() or title, "body": body,
                         "sections": book_parts.sections(md)})
    parts = [
        '<meta charset="utf-8">',                       # also right when the file is opened or served on its own
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">',
        "<title>Build Your Own LLM</title>",
        '<meta name="claude:viewport-fit-sides" content="cover">',
        '<link rel="preconnect" href="https://fonts.googleapis.com">',
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;'
        '0,8..60,600;0,8..60,700;1,8..60,700&family=Inter:wght@500;600;700&family=JetBrains+Mono:wght@400;600'
        '&family=Noto+Serif+Devanagari:wght@400;600&display=swap">',
        f"<style>{CSS}{book_parts.CSS}</style>",
        book_parts.cover(),
        '<main class="page">',
        book_parts.contents(chapters, anchor),
    ]
    for c in chapters:
        parts.append(book_parts.part_page(c["name"]))
        parts.append(f'<section class="chapter" id="{c["name"]}"><div class="chapter-num">{c["num"]}</div>'
                     f'<h1>{html.escape(c["title"])}</h1>{c["body"]}</section>')
    parts.append(book_parts.index(chapters, anchor))
    parts.append('<footer>Draft edition, October 2026. Code, labs and checks: <code>course/</code> in the AnuLM '
                 "repository. Lab data: Shakespeare (public domain), Hindi Wikipedia, XQuAD and SQuAD (CC BY-SA 4.0), "
                 "BANKING77 and Localized Narratives (CC BY 4.0).</footer></main>")
    out = HERE / "preview" / "sample.html"
    out.parent.mkdir(exist_ok=True)
    # Chapter 1 shows the replacement character on purpose; write it as an entity
    # so it survives tools that treat a raw U+FFFD as damage.
    out.write_text("\n".join(parts).replace("\ufffd", "&#xFFFD;"), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
