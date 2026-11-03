"""
Move every code citation in the book from one commit to another.

    python book/tools/repin.py 773b6a8 aae3fdd            # dry run: writes the report only
    python book/tools/repin.py 773b6a8 aae3fdd --write    # rewrite the book files too

The book cites code as file + line numbers at a pinned commit. When the code
moves (files renamed, docstrings added, functions rewritten), the citations
must follow. This tool does the mechanical part exactly and reports the rest:

- **Line map.** For each cited file, the old and new contents are diffed
  (difflib, no junk heuristics). A line in an unchanged block maps exactly; a
  line in a changed block of equal length maps by position (an edited import,
  say) and is reported as `edited`; a line in a block that changed length has
  no new home and is reported as `UNMAPPED`, never guessed.
- **Which file a number belongs to.** Excerpt headers and "`name` (`file`
  line N)" name their file. In prose, a number after "line"/"lines" belongs to
  the last file the text named (in that paragraph, else earlier in the
  chapter). In a chapter's "Repository evidence" list, every number after a
  symbol or "line(s)" belongs to that item's file. If the sentence names a
  symbol (`Name` or `Class.method`) whose old span does not contain the line,
  the citation is reported as `CHECK` rather than trusted.
- **Excerpts.** After the range moves, every quoted code line must still occur
  in the new range; a quoted line whose code was edited in place is replaced
  by the new text and reported. A quoted line that is gone is `UNMAPPED`.
- **Paths and commands.** `model.py` becomes `anulm/model.py`, etc. (the
  restructure's renames), `python train.py` becomes `python -m anulm train`,
  and `@old` becomes `@new`.

The report, `technical-review/repin-report.md`, lists every change with the
old and new code line side by side, so a reviewer can scan it; every
`UNMAPPED` and `CHECK` row needs a person. Afterwards run
`tools/verify_chapters.py` and `tools/verify_source_map.py` at the new pin.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import re
import subprocess
from pathlib import Path

BOOK = Path(__file__).resolve().parent.parent
ROOT = BOOK.parent
REPORT = BOOK / "technical-review" / "repin-report.md"
# where <TAB> numbers as written <TAB> old-pin file it belongs to (or "-" to leave
# it alone), one per line; written by a person after reading the report.
OVERRIDES = BOOK / "technical-review" / "repin-overrides.tsv"
# Files rewritten: the chapters and the book's own maps. The review records in
# technical-review/ are history at the old pin and are left alone.
TARGETS = sorted((BOOK / "chapters").glob("*.md")) + [BOOK / n for n in (
    "REPOSITORY_SOURCE_MAP.md", "REPOSITORY_ARCHITECTURE.md", "FRONT_MATTER.md", "README.md",
    "BOOK_PLAN.md", "STYLE.md", "WRITER_BRIEF.md", "RESEARCH_REFERENCES.md")]
# The old root model.py still exists at the new pin as a two-line alias for
# old checkpoints, so git records the move as an edit, not a rename.
EXTRA_RENAMES = {"model.py": "anulm/model.py"}


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        raise SystemExit(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


class Pins:
    """Old and new trees, the path map between them, and per-file line maps."""

    def __init__(self, old: str, new: str):
        self.old, self.new = old, new
        self.old_files = set(git("ls-tree", "-r", "--name-only", old).splitlines())
        self.new_files = set(git("ls-tree", "-r", "--name-only", new).splitlines())
        self.paths: dict[str, str] = {}
        for line in git("diff", "-M", "--name-status", old, new).splitlines():
            parts = line.split("\t")
            if parts[0].startswith("R"):
                self.paths[parts[1]] = parts[2]
        self.paths.update(EXTRA_RENAMES)
        self._src: dict[tuple[str, str], list[str]] = {}
        self._maps: dict[str, list[tuple[int | None, str]]] = {}
        self._defs: dict[tuple[str, str], dict[str, list[tuple[int, int]]]] = {}

    def new_path(self, old_path: str) -> str:
        return self.paths.get(old_path, old_path)

    def resolve(self, token: str) -> str | None:
        """An old-pin path for a backticked token: exact, or a unique basename."""
        if token in self.old_files:
            return token
        hits = [f for f in self.old_files if f.rsplit("/", 1)[-1] == token and "/" not in token]
        root = [f for f in hits if "/" not in f]
        if root:
            return root[0]
        return hits[0] if len(hits) == 1 else None

    def src(self, rev: str, path: str) -> list[str]:
        key = (rev, path)
        if key not in self._src:
            r = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            self._src[key] = r.stdout.splitlines() if r.returncode == 0 else []
        return self._src[key]

    def line_map(self, path: str) -> list[tuple[int | None, str]]:
        """Index i (0-based old line) -> (new 1-based line or None, kind)."""
        if path not in self._maps:
            a, b = self.src(self.old, path), self.src(self.new, self.new_path(path))
            out: list[tuple[int | None, str]] = [(None, "UNMAPPED")] * len(a)
            sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "equal":
                    for k in range(i2 - i1):
                        out[i1 + k] = (j1 + k + 1, "same")
                elif tag == "replace" and i2 - i1 == j2 - j1:
                    for k in range(i2 - i1):
                        out[i1 + k] = (j1 + k + 1, "edited")
                elif tag == "replace":
                    # A block that changed length: a line that was only moved or
                    # re-indented inside it (its stripped text occurs exactly once
                    # in the new block) still has a home; the rest has none.
                    new_block = [b[j].strip() for j in range(j1, j2)]
                    for i in range(i1, i2):
                        t = a[i].strip()
                        if t and len(t) > 3 and new_block.count(t) == 1:
                            out[i] = (j1 + new_block.index(t) + 1, "reindented")
            self._maps[path] = out
        return self._maps[path]

    def map_line(self, path: str, n: int) -> tuple[int | None, str]:
        m = self.line_map(path)
        return m[n - 1] if 1 <= n <= len(m) else (None, "OUT OF FILE")

    def defs(self, rev: str, path: str) -> dict[str, list[tuple[int, int]]]:
        """Qualified name (`f`, `Class`, `Class.method`) -> [(first, last line)]."""
        key = (rev, path)
        if key not in self._defs:
            out: dict[str, list[tuple[int, int]]] = {}
            src = "\n".join(self.src(rev, path if rev == self.old else self.new_path(path)))
            try:
                tree = ast.parse(src) if path.endswith(".py") and src else None
            except SyntaxError:
                tree = None

            def walk(node, prefix=""):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        first = min([child.lineno] + [d.lineno for d in child.decorator_list])
                        span = (first, child.end_lineno or child.lineno, child.lineno)
                        out.setdefault(child.name, []).append(span)
                        if prefix:
                            out.setdefault(prefix + child.name, []).append(span)
                        walk(child, child.name + "." if isinstance(child, ast.ClassDef) else prefix)
                    elif isinstance(child, (ast.Assign, ast.AnnAssign)) and not prefix:
                        targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                        for t in targets:
                            if isinstance(t, ast.Name):
                                out.setdefault(t.id, []).append((child.lineno, child.end_lineno or child.lineno, child.lineno))
            if tree:
                walk(tree)
            self._defs[key] = out
        return self._defs[key]


class Report:
    def __init__(self):
        self.rows: list[tuple[str, str, str, str, str, str, str]] = []

    def add(self, where, path, old, new, kind, old_code="", new_code=""):
        self.rows.append((where, path, old, new, kind, old_code.strip()[:70], new_code.strip()[:70]))

    def write(self, pins: Pins):
        def esc(s):
            return s.replace("|", "\\|").replace("`", "'")
        flagged = [r for r in self.rows if r[4] in ("UNMAPPED", "CHECK", "OUT OF FILE", "EXCERPT", "STALE OVERRIDE")]
        kinds: dict[str, int] = {}
        for r in self.rows:
            kinds[r[4]] = kinds.get(r[4], 0) + 1
        lines = [f"# Re-pin report: @{pins.old} -> @{pins.new}", "",
                 "Written by `book/tools/repin.py`. Every line citation it moved, with the code at",
                 "the old line and at the new one. Rows marked UNMAPPED, CHECK, EXCERPT or OUT OF",
                 "FILE were **not** resolved automatically and need a person.", "",
                 "| kind | count |", "|---|---|"] + [f"| {k} | {v} |" for k, v in sorted(kinds.items())]
        lines += ["", f"## Needs a person ({len(flagged)})", "",
                  "| where | file | old | new | kind | old code | new code |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {esc(r[0])} | {esc(r[1])} | {r[2]} | {r[3]} | {r[4]} | {esc(r[5])} | {esc(r[6])} |" for r in flagged]
        lines += ["", "## Every change", "",
                  "| where | file | old | new | kind | old code | new code |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {esc(r[0])} | {esc(r[1])} | {r[2]} | {r[3]} | {r[4]} | {esc(r[5])} | {esc(r[6])} |" for r in self.rows]
        REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        return kinds, len(flagged)


DASH = "–"
NUM = r"\d+(?:[–-]\d+)?"
EXCERPT = re.compile(r"(\*\*In the repo\*\* · `)([\w./-]+)(` · `)([^`]+(?:`, `[^`]+)*)(` · lines? )(\d+)(?:–(\d+))?( · @)(\w+)(\n\n```\w*\n)(.*?)(```)", re.S)
LINE_REF = re.compile(r"`(\w+)` \((?:`([\w./-]+)` )?line (\d+)\)")
TOKEN = re.compile(r"`([^`\n]+)`")
LINES_WORD = re.compile(r"\b(lines?) ((?:" + NUM + r")(?:(?:, | and | or |, and )(?:" + NUM + r"))*)")
SYMBOL = re.compile(r"^[A-Za-z_][\w]*(?:\.[A-Za-z_]\w*)?$")


class Rewriter:
    def __init__(self, pins: Pins, report: Report):
        self.p, self.r = pins, report
        self.overrides: dict[tuple[str, str], str] = {}
        if OVERRIDES.exists():
            for row in OVERRIDES.read_text(encoding="utf-8").splitlines():
                if row.strip() and not row.startswith("#"):
                    where, numbers, f = row.split("\t")
                    self.overrides[(where, numbers.replace("-", DASH))] = f.strip()
        self.used: set[tuple[str, str]] = set()

    # -- numbers ----------------------------------------------------------
    def move_range(self, path, a, b, where, symbol=None) -> str | None:
        """New 'a' or 'a–b' text for an old range in `path`, or None if unresolved."""
        old = self.p.src(self.p.old, path)
        new = self.p.src(self.p.new, self.p.new_path(path))
        na, ka = self.p.map_line(path, a)
        nb, kb = self.p.map_line(path, b) if b else (na, ka)
        oc = old[a - 1] if 0 < a <= len(old) else ""
        if na is None or nb is None:
            self.r.add(where, path, f"{a}" + (f"{DASH}{b}" if b else ""), "?", ka if na is None else kb, oc)
            return None
        if symbol:
            spans = self.p.defs(self.p.old, path).get(symbol) or self.p.defs(self.p.old, path).get(symbol.split(".")[-1])
            if spans and not any(s <= a <= e for s, e, _ in spans):
                self.r.add(where, path, str(a), str(na), "CHECK", oc, f"`{symbol}` spans {spans} at the old pin")
                return None
        text = f"{na}" + (f"{DASH}{nb}" if b else "")
        kind = "edited" if "edited" in (ka, kb) else "reindented" if "reindented" in (ka, kb) else "same"
        if text != f"{a}" + (f"{DASH}{b}" if b else "") or kind != "same":
            self.r.add(where, path, f"{a}" + (f"{DASH}{b}" if b else ""), text, kind, oc, new[na - 1] if na <= len(new) else "")
        return text

    def move_numbers(self, numbers: str, path, where, symbol=None) -> str:
        def one(m):
            s = m.group(0)
            a, _, b = s.replace("-", DASH).partition(DASH)
            res = self.move_range(path, int(a), int(b) if b else None, where, symbol)
            return res if res is not None else s
        return re.sub(NUM, one, numbers)

    # -- excerpts ---------------------------------------------------------
    def excerpt(self, m, where) -> str:
        path = self.p.resolve(m.group(2))
        if not path:
            return m.group(0)
        a, b = int(m.group(6)), int(m.group(7)) if m.group(7) else None
        moved = self.move_range(path, a, b, where)
        if moved is None:
            self.r.add(where, path, f"{a}", "?", "EXCERPT", "header range unresolved: " + m.group(4))
            return m.group(0)
        na, _, nb = moved.partition(DASH)
        na, nb = int(na), int(nb or na)
        new_src = self.p.src(self.p.new, self.p.new_path(path))
        window = [l.strip() for l in new_src[na - 1:nb]]
        body = []
        for line in m.group(11).split("\n"):
            s = line.strip()
            if not s or s.startswith("# ...") or s == "..." or s in window:
                body.append(line)
                continue
            # The quoted line changed: find its old position in the old range, map it.
            old_src = self.p.src(self.p.old, path)
            hits = [i for i in range(a, (b or a) + 1) if 0 < i <= len(old_src) and old_src[i - 1].strip() == s]
            if hits:
                nl, kind = self.p.map_line(path, hits[0])
                if nl and na <= nl <= nb:
                    indent = line[:len(line) - len(line.lstrip())]
                    body.append(indent + new_src[nl - 1].strip())
                    self.r.add(where, path, str(hits[0]), str(nl), "edited-quote", s, new_src[nl - 1])
                    continue
            self.r.add(where, path, moved, "", "EXCERPT", s, "quoted line not in the new range")
            body.append(line)
        return (m.group(1) + self.p.new_path(path) + m.group(3) + m.group(4) + m.group(5) + moved
                + m.group(8) + self.p.new + m.group(10) + "\n".join(body) + m.group(12))

    # -- prose ------------------------------------------------------------
    def owner(self, a: int, b: int, path: str | None, strong: bool, symbol: str | None, hints: list[str],
              doc_files: list[str], where: str, numbers: str) -> str | None:
        """The file a prose line number belongs to, or None (reported) if unsure.

        Trusted: the file is named earlier on the same line; or a symbol named
        in the sentence spans the line; or a code fragment quoted in the
        sentence occurs on (or right next to) the line. Evidence pointing at a
        different file than the running context reattributes the number.
        """
        def spans_in(f):
            d = self.p.defs(self.p.old, f)
            return d.get(symbol) or (d.get(symbol.split(".")[-1]) if "." in symbol else None)

        def hint_at(f):
            src = self.p.src(self.p.old, f)
            near = " ".join(src[max(0, a - 3):b + 2])
            return any(h in near for h in hints)

        if symbol:
            owners = [f for f in self.p.old_files if f.endswith(".py") and
                      any(a <= e and b >= s for s, e, _ in (spans_in(f) or []))]
            if path in owners:
                return path
            preferred = [f for f in owners if f in doc_files] or owners
            if len(preferred) == 1:
                if path != preferred[0]:
                    self.r.add(where, preferred[0], numbers, "", "reattributed",
                               f"`{symbol}` is in {preferred[0]}, not {path}")
                return preferred[0]
            if path and spans_in(path):
                self.r.add(where, path, numbers, "?", "CHECK", self.p.src(self.p.old, path)[a - 1]
                           if a <= len(self.p.src(self.p.old, path)) else "",
                           f"`{symbol}` spans {spans_in(path)} at the old pin")
                return None
        if path and strong:
            return path
        if path and hints and hint_at(path):
            return path
        others = [f for f in doc_files if f != path and hints and hint_at(f)]
        if len(others) == 1:
            self.r.add(where, others[0], numbers, "", "reattributed", f"quoted code is in {others[0]}, not {path}")
            return others[0]
        if path or others:
            old = self.p.src(self.p.old, path) if path else []
            self.r.add(where, path or "?", numbers, "?", "CHECK", old[a - 1] if path and a <= len(old) else "",
                       "no file named on the line and nothing quoted confirms it")
        return None

    def prose_line(self, line: str, ctx: dict, where: str, evidence: bool, doc_files: list[str],
                   prev: str = "") -> str:
        """Rewrite line numbers in one line of prose; ctx['file'] carries over."""
        out, pos = [], 0
        events = [(m.start(), "tok", m) for m in TOKEN.finditer(line)] + \
                 [(m.start(), "num", m) for m in LINES_WORD.finditer(line)]
        # Bare paths outside backticks (code comments: "# train.py line 740").
        for m in re.finditer(r"(?<![`\w/])([\w/]+\.(?:py|md|json|toml|yml|cmd))(?![`\w])", line):
            events.append((m.start(), "bare", m))
        refs = [(m.start(), m.end()) for m in LINE_REF.finditer(line)]
        events = [e for e in events if not (e[1] == "num" and any(a <= e[0] < b for a, b in refs))]
        if evidence:
            # In a "Repository evidence" list every other number outside code is
            # a line of the item's file too: "(loop 1083-1109)", "defaults 31-32",
            # "(411)", a wrapped line's "740-742". Not: chapter or section
            # numbers, a flag's value ("`--steps` 2000"), or a quantity with a unit.
            spans = [(m.start(), m.end()) for m in TOKEN.finditer(line)]
            for m in re.finditer(r"(?<![\w.,–\"-])" + NUM + r"(?![\w%×\"]|\.\d|,\d|[–-][A-Za-z]|\s+(?:tests?|steps?|tokens?|rows?|files?|times|of))", line):
                before = line[:m.start()]
                clause = re.split(r"[;(]", before)[-1]
                if (any(a <= m.start() < b for a, b in spans)
                        or re.search(r"(Chapters?|§|Part|Figure|Table|Appendix|\[R|\[S)\s*(\d+[–-]?\d*,?\s*(and\s+)?)*$", clause)
                        or re.search(r"`-[^`]*` $", before)):
                    continue
                events.append((m.start(), "sweep", m))
        events.sort(key=lambda e: e[0])
        context = prev + " " + line
        hints = [t for t in TOKEN.findall(context) if len(t) >= 4 and not re.search(r"\.(py|md|json|toml|yml|cmd|txt)$", t)
                 and not SYMBOL.match(t)]
        hints += [t.split(".")[-1] for t in TOKEN.findall(context) if SYMBOL.match(t) and len(t.split(".")[-1]) >= 5]
        hints += re.findall(r'"([^"]{4,40})"', context)
        symbol, strong, taken_until = None, False, 0

        def move(numbers):
            key = (where, numbers.replace("-", DASH))
            if key in self.overrides:
                self.used.add(key)
                f = self.overrides[key]
                if f == "-":
                    return numbers
                self.r.add(where, f, numbers, "", "override", "decided in repin-overrides.tsv")
                return self.move_numbers(numbers, f, where, None)
            nums = [int(x) for x in re.findall(r"\d+", numbers)]
            f = self.owner(min(nums), max(nums), ctx.get("file"), strong, symbol, hints, doc_files, where, numbers)
            return self.move_numbers(numbers, f, where, None) if f else numbers

        for start, kind, m in events:
            if start < taken_until:
                continue
            if kind in ("tok", "bare"):
                tok = m.group(1)
                if re.search(r"\.(py|md|json|toml|yml|cmd|txt)$", tok) and " " not in tok:
                    ctx["file"] = self.p.resolve(tok)       # None for files not at the pin
                    symbol, strong = None, ctx["file"] is not None
                elif SYMBOL.match(tok):
                    symbol = tok
                    if evidence and ctx.get("file"):
                        # "`load_bytes` 62–98" in an evidence list: the numbers right after it.
                        tail = re.match(r"` ((?:" + NUM + r")(?:(?:, | and )(?:" + NUM + r"))*)(?![\w%]|\.\d)",
                                        line[m.end() - 1:])
                        if tail and not re.match(r"` \d+(?:\.\d|,\d{3}| ?(?:GB|MB|KB|ms|M|B|k|%|×|x)\b)",
                                                 line[m.end() - 1:]):
                            s0 = m.end() + len("` ") - 1
                            s1 = s0 + len(tail.group(1))
                            out.append(line[pos:s0])
                            out.append(move(tail.group(1)))
                            pos = taken_until = s1
                continue
            if kind == "num" and (ctx.get("file") or symbol or hints):
                s0 = m.start(2)
                out.append(line[pos:s0])
                out.append(move(m.group(2)))
                pos = taken_until = m.end(2)
            elif kind == "sweep" and ctx.get("file"):
                strong_was, strong = strong, True      # the list item names its file
                moved = move(m.group(0))
                strong = strong_was
                self.r.add(where, ctx["file"], m.group(0), moved, "sweep",
                           line[max(0, m.start() - 45):m.end() + 5])
                out.append(line[pos:m.start()])
                out.append(moved)
                pos = taken_until = m.end()
        out.append(line[pos:])
        return "".join(out)

    def line_refs(self, text: str, ctx_file, where) -> str:
        def one(m):
            name, f, n = m.group(1), m.group(2), int(m.group(3))
            path = self.p.resolve(f) if f else ctx_file
            if not f:
                def has(fp):
                    return any(d == n for _, _, d in self.p.defs(self.p.old, fp).get(name, []))
                if not (path and has(path)):
                    hits = [fp for fp in self.p.old_files if fp.endswith(".py") and has(fp)]
                    if len(hits) == 1:
                        if hits[0] != path:
                            self.r.add(where, hits[0], str(n), "", "reattributed", f"`{name}` is defined there")
                        path = hits[0]
            if not path:
                return m.group(0)
            new_defs = self.p.defs(self.p.new, path).get(name)
            nl, kind = self.p.map_line(path, n)
            if new_defs and (nl is None or nl not in [d for _, _, d in new_defs]):
                nl = new_defs[0][2] if len(new_defs) == 1 else nl
            if nl is None:
                self.r.add(where, path, str(n), "?", "UNMAPPED", f"`{name}` (line {n})")
                return m.group(0)
            if nl != n:
                self.r.add(where, path, str(n), str(nl), "def", f"`{name}`",
                           self.p.src(self.p.new, self.p.new_path(path))[nl - 1])
            return m.group(0).replace(f"line {n})", f"line {nl})")
        return LINE_REF.sub(one, text)

    # -- files ------------------------------------------------------------
    def file(self, path: Path) -> str:
        text = path.read_text(encoding="utf-8")
        name = path.relative_to(BOOK).as_posix()
        # 1. excerpts (they name their file), protected from the prose pass.
        excerpts: list[str] = []
        doc_files = sorted({f for t in TOKEN.findall(text) if (f := self.p.resolve(t))} | {"model.py"})

        def keep(m):
            where = f"{name}:{text[:m.start()].count(chr(10)) + 1}"
            excerpts.append(self.excerpt(m, where))
            # One marker line per original line, so report line numbers stay true.
            return f"\x00EXCERPT{len(excerpts) - 1}\x00" + "\n\x01" * m.group(0).count("\n")
        text = EXCERPT.sub(keep, text)
        # 2. prose, line by line, carrying the current file.
        ctx: dict = {"file": None}
        out, evidence, prev = [], False, ""
        for i, line in enumerate(text.split("\n"), 1):
            where = f"{name}:{i}"
            if line == "\x01":
                out.append(line)
                continue
            if "\x00EXCERPT" in line:
                k = int(re.search(r"\x00EXCERPT(\d+)\x00", line).group(1))
                hdr = re.search(r"· `([\w./-]+)` ·", excerpts[k])
                ctx["file"] = self.p.resolve(hdr.group(1)) or self.p.resolve(
                    next((o for o, n in self.p.paths.items() if n == hdr.group(1)), hdr.group(1)))
                out.append(line)
                continue
            if re.match(r"(Repository evidence|Repository sources|Code cited)", line):
                evidence = True
            elif not line.strip():
                evidence = False
            if evidence and re.match(r"- `", line):
                ctx["file"] = None
            line = self.line_refs(line, ctx.get("file"), where)
            out.append(self.prose_line(line, ctx, where, evidence, doc_files, prev))
            prev = line
        text = "\n".join(out)
        # 3. put the excerpts back; paths, commands and the pin.
        text = re.sub(r"\x00EXCERPT(\d+)\x00(?:\n\x01)*", lambda m: excerpts[int(m.group(1))], text)
        return self.rename(text)

    def rename(self, text: str) -> str:
        p = self.p
        for old, new in sorted(p.paths.items(), key=lambda kv: -len(kv[0])):
            if not old.endswith(".py") and not old.endswith(".md"):
                continue
            text = re.sub(r"(?<=`)" + re.escape(old) + r"(?=`)", new, text)
            text = re.sub(r"(?<![\w/.-])" + re.escape(old) + r"(?= line)", new, text)
            mod = old[:-3]
            if old.endswith(".py") and "/" not in old and new.startswith("anulm/"):
                text = re.sub(r"\bpython (?:-u )?" + re.escape(old) + r"\b", f"python -m anulm {mod}", text)
            elif old.endswith(".py") and new.startswith("tests/"):
                text = re.sub(r"\bpython (?:-u )?" + re.escape(old) + r"\b",
                              "python -m tests." + Path(new).stem, text)
        text = text.replace(f"@{p.old}", f"@{p.new}").replace(f'pinned: "@{p.old}"', f'pinned: "@{p.new}"')
        return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--write", action="store_true", help="rewrite the book files (default: report only)")
    a = ap.parse_args()
    pins = Pins(git("rev-parse", "--short", a.old).strip(), git("rev-parse", "--short", a.new).strip())
    report = Report()
    rw = Rewriter(pins, report)
    changed = 0
    for path in TARGETS:
        if not path.exists():
            continue
        before = path.read_text(encoding="utf-8")
        after = rw.file(path)
        if after != before:
            changed += 1
            if a.write:
                path.write_text(after, encoding="utf-8", newline="\n")
    for key in sorted(set(rw.overrides) - rw.used):
        report.add(key[0], rw.overrides[key], key[1], "", "STALE OVERRIDE", "no citation matches this row")
    kinds, flagged = report.write(pins)
    print(f"@{pins.old} -> @{pins.new}: {changed} files {'rewritten' if a.write else 'would change'}; "
          f"{sum(kinds.values())} citations touched {dict(sorted(kinds.items()))}; {flagged} need a person")
    print(f"report: {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
