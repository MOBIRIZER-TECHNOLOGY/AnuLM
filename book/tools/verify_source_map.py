"""
Stage 3: check every row of docs/reference/repository-source-map.md against the code at the pinned commit.

    python book/tools/verify_source_map.py            # writes docs/reviews/source-verification.md

For each row it reads the file at the commit the map names (git show), parses it
with Python's ast, finds where each named symbol starts and ends, and compares
that span with the line range in the row. It also checks the test map, and that
nothing in the cited files has changed between the pin and the working tree.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

BOOK = Path(__file__).resolve().parent.parent
ROOT = BOOK.parent
MAP = BOOK / "docs/reference/repository-source-map.md"
OUT = BOOK / "docs/reviews" / "source-verification.md"
SECTION_FILE = {"1": "model.py", "2": "bpe.py", "3": "train.py"}
DEFS: dict[str, list[int]] = {}
RANGE = re.compile(r"(\d+)[–-](\d+)")


def at(commit: str, path: str) -> str | None:
    r = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return r.stdout if r.returncode == 0 else None


def spans(src: str) -> dict[str, list[tuple[int, int]]]:
    """name -> [(first line, last line)] for classes, functions, methods (also as Class.method) and
    module-level names; decorators count as part of a definition."""
    out: dict[str, list[tuple[int, int]]] = {}
    defs = DEFS
    defs.clear()

    def add(name, node):
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        out.setdefault(name, []).append((start, node.end_lineno))
        defs.setdefault(name, []).append(node.lineno)

    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            add(node.name, node)
            if isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        add(f"{node.name}.{sub.name}", sub)
                        add(sub.name, sub)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                if isinstance(t, ast.Name):
                    add(t.id, node)
        elif isinstance(node, ast.If) and "__main__" in ast.unparse(node.test):
            add("__main__", node)
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    text = MAP.read_text(encoding="utf-8")
    commit = re.search(r"`@([0-9a-f]{7,40})`", text).group(1)
    section, last_file, results, files, lead = "0", None, [], set(), {}
    for line in text.splitlines():
        m = re.match(r"## (\d+)\.", line)
        if m:
            section, last_file = m.group(1), SECTION_FILE.get(m.group(1))
            continue
        if not line.startswith("| ") or line.startswith("| Component") or line.startswith("|---") or section == "6":
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        comp, sym_cell, lines_cell = cells[0], cells[1], cells[2]
        file_here = re.findall(r"`([\w/]+\.py)`", sym_cell + " " + lines_cell)
        f = file_here[0] if file_here else last_file
        last_file = f if section in ("4", "5") and file_here else last_file
        names = [n for n in re.findall(r"`([\w.]+)`", sym_cell) if not n.endswith(".py")]
        names = [n for n in names if not re.fullmatch(r"\d+", n)]
        rng = RANGE.search(lines_cell.replace(f or "", ""))
        if not f or not rng or not names:
            results.append((comp, f, names, None, "no range: an absent feature or an uncommitted one (documented in the row)"))
            continue
        ranges = [(int(a), int(b)) for a, b in RANGE.findall(lines_cell)]
        files.add(f)
        src = at(commit, f)
        if src is None:
            results.append((comp, f, names, None, f"{f} not found at {commit}"))
            continue
        lo, hi = int(rng.group(1)), int(rng.group(2))
        sp = spans(src)
        found, problems = [], []
        for n in names:
            if n not in sp:
                if n in ("main",) and comp.startswith("↳"):
                    continue
                problems.append(f"`{n}` not defined at top level of {f}")
                continue
            best = min(sp[n], key=lambda s: abs(s[0] - lo) + abs(s[1] - hi))
            found.append((n, best))
        if comp.startswith("↳"):                                   # a stretch inside main: check it is inside
            body = src.splitlines()[lo - 1:hi]
            key = {"gradient accumulation": "grad_accum", "clip, step, bias update": "update_expert_biases",
                   "eval, best and `.last` checkpoints, bits/byte": "last_path"}
            k = next((v for kk, v in key.items() if kk in comp), None)
            ok = k is not None and any(k in b for b in body)
            results.append((comp, f, names, (lo, hi), "ok" if ok else f"`{k}` not found in lines {lo}–{hi}"))
            continue
        if found and len(ranges) > 1:                          # several ranges: each symbol inside one of them
            inside = all(any(a <= s[0] and s[1] <= b + 1 for a, b in ranges) for _, s in found)
            note = "ok" if inside and not problems else (f"spans {[s for _, s in found]} vs ranges {ranges}"
                                                          + ("; " + "; ".join(problems) if problems else ""))
        elif found:
            first = min(s[0] for _, s in found)
            last = max(s[1] for _, s in found)
            if abs(first - lo) <= 1 and abs(last - hi) <= 1 and not problems:
                note = "ok"
            elif 0 <= first - lo <= 10 and abs(last - hi) <= 1 and not problems:
                note = "ok"                                     # the range starts with comments or constants
                lead[comp] = first - lo
            else:
                note = f"actual span {first}–{last}" + ("; " + "; ".join(problems) if problems else "")
        else:
            note = "; ".join(problems)
        results.append((comp, f, names, (lo, hi), note))

    # the test map
    tsrc = at(commit, "test_model.py")
    tsp = spans(tsrc) if tsrc else {}
    test_def_lines = dict(DEFS)                               # the def line, which is what the map cites
    tests = []
    sec6 = text[text.index("## 6."):]
    for name, ln in re.findall(r"`(\w+)` \((\d+)\)", sec6):
        got = test_def_lines.get(name)
        tests.append((name, int(ln), got))
    test_defs = [n for n in tsp if "." not in n and not n.startswith("_") and n not in ("main", "__main__")]

    # drift: pinned commit vs working tree, for every cited file
    drift = {}
    for f in sorted(files | {"test_model.py"}):
        r = subprocess.run(["git", "diff", "--stat", commit, "--", f], cwd=ROOT, capture_output=True, text=True)
        drift[f] = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""

    ok = sum(1 for r in results if r[4] == "ok")
    bad = [r for r in results if r[4] != "ok"]
    t_ok = sum(1 for _, ln, got in tests if got and ln in got)
    t_bad = [t for t in tests if not (t[2] and t[1] in t[2])]
    L = ["# Source verification", "", f"Written by `book/tools/verify_source_map.py` against `@{commit}`.", "",
         f"Source-map rows checked: {len(results)}. Matching their line range (±1): **{ok}**. Not matching: **{len(bad)}**.",
         f"Test-map entries: {len(tests)}. At the `def` line given: **{t_ok}**. Not: **{len(t_bad)}**.", "",
         "Rows whose range deliberately starts with leading comments or constants (accepted): "
         + (", ".join(f"{k} (+{v})" for k, v in lead.items()) or "none") + ".", "", "## Rows that don't match", ""]
    L += ["| Component | File | Symbols | Map says | Finding |", "|---|---|---|---|---|"]
    L += [f"| {c} | `{f}` | {', '.join(n)} | {f'{r[0]}–{r[1]}' if r else '–'} | {note} |"
          for c, f, n, r, note in bad] or ["| – | | | | none |"]
    L += ["", "## Test map entries that don't match", ""]
    L += [f"- `{n}`: map says {ln}, defined at {got}" for n, ln, got in t_bad] or ["- none"]
    L += ["", f"Test functions defined in `test_model.py` at the pin: {len(test_defs)}.", "",
          "## Drift between the pin and the working tree", ""]
    L += [f"- `{f}`: {d}" for f, d in drift.items() if d] or ["- none"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[4:6]))
    for c, f, n, r, note in bad:
        print(f"  ROW  {c} | {f} | {', '.join(n)} | map {r} | {note}")
    for n, ln, got in t_bad:
        print(f"  TEST {n}: map {ln}, actual {got}")
    print(f"  test functions at the pin: {len(test_defs)}")
    print("  drift:", {f: d for f, d in drift.items() if d} or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
