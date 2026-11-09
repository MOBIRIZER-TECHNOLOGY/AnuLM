"""
Book example, not repository code.

Appendix F: the book's own audit. It checks four appendix tables against
their sources, takes an inventory of the evidence labels in every chapter, and,
on request, runs the book's self-contained examples and reports which ones
exit cleanly. It does not recompute the numbers in the chapters: each
chapter's own example does that.

    python book/code-examples/ch41_audit.py                  # appendix checks + evidence inventory
    python book/code-examples/ch41_audit.py --bibliography   # print Appendix E's table from the register
    python book/code-examples/ch41_audit.py --run            # also run the self-contained examples (minutes)

The checks:

- Appendix C: every test in `tests/test_model.py` at the pin, with its `def` line,
  in file order, and no other;
- Appendix B.1: every configuration row, recomputed by
  `ch41_arithmetic.py --config` (needs `ckpt_base_v2.pt`);
- Appendix E: every row, regenerated from `RESEARCH_REFERENCES.md`;
- Appendix A: every concrete shape the page prints occurs somewhere in
  `ch41_shapes.py`'s output. This is membership only: it does not check that
  a shape sits in the right row.

`--run` runs each example in a temporary directory, except those in SKIP below:
an explicit list of the examples that need the released checkpoint (most of
them also the GPU and the v2 corpus), the v2 corpus itself, the base run's
log, or a local export in release/. It lists what it skipped and why, and
reports a SKIP entry that names no file. No example writes `book/measurements/` unless given `--save`, so running
the rest is safe.

It exits 1 if a check fails or an example run fails.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOOK = HERE.parent
ROOT = BOOK.parent
CH = BOOK / "chapters"
PIN = "2e80b3e"
GPU = "needs ckpt_base_v2.pt and data/v2; measured on the GPU"

# Examples --run does not run, with the reason. Everything else runs, from a temporary directory.
SKIP = {
    "ch09_bpe_from_scratch.py": "needs release/AnuLM-Base-v2-400M (the local export's tokenizers)",
    "ch09_measure_fertility.py": "needs data/v2 (the base run's corpus)",
    "ch10_measure_tables.py": "needs ckpt_base_v2.pt and data/v2",
    "ch12_measure_masks.py": GPU,
    "ch13_measure_groups.py": GPU,
    "ch14_measure_shift.py": GPU,
    "ch15_measure_stream.py": GPU,
    "ch16_measure_trained.py": "needs ckpt_base_v2.pt, data/v2 and v2_train_run.log; measured on the GPU",
    "ch21_windows.py": "needs release/AnuLM-Base-400M/tokenizer.json (experiments/build_v2.py reads it)",
    "ch21_measure_corpus.py": "needs data/v2 (the base run's corpus)",
    "ch22_read_run_log.py": "needs v2_train_run.log (the base run's log, local only)",
    "ch23_measure.py": GPU,
    "ch24_measure_val_noise.py": GPU,
    "ch25_measure_resume_point.py": "needs ckpt_base_v2.pt and ckpt_base_v2.pt.last",
    "ch26_measure_router_bf16.py": GPU,
    "ch27_measure_cache.py": GPU,
    "ch28_measure_sampling.py": GPU,
    "ch29_measure_context.py": GPU,
    "ch30_measure_chat_split.py": "needs ckpt_base_v2.pt and data/chat_*.jsonl",
    "ch31_measure_lora.py": "needs ckpt_base_v2.pt; measured on the GPU",
    "ch35_wider_vocab.py": GPU,
    "ch36_projectors.py": "needs ckpt_base_v2.pt, ckpt_omni.pt and data/omni",
    "ch39_export_roundtrip.py": "needs ckpt_base_v2.pt, data/v2 and release/AnuLM-Base-v2-400M",
    "ch41_checkpoints.py": "needs ckpt_base_v2.pt, data/v2 and release/AnuLM-Base-v2-400M",
    "measure_attention.py": GPU,
    "measure_part_i_ii.py": GPU,
}

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def at_pin(path: str) -> str:
    return subprocess.run(["git", "show", f"{PIN}:{path}"], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8").stdout


def run(script: str, *args: str) -> str:
    r = subprocess.run([sys.executable, str(HERE / script), *args], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    assert r.returncode == 0, f"{script} {' '.join(args)} failed:\n{r.stderr[-2000:]}"
    return r.stdout


def tests_at_pin() -> list[tuple[str, int]]:
    tree = ast.parse(at_pin("tests/test_model.py"))
    return [(n.name, n.lineno) for n in tree.body if isinstance(n, ast.FunctionDef)
            and any(isinstance(d, ast.Name) and d.id == "test" for d in n.decorator_list)]


def check_test_index() -> list[str]:
    want = tests_at_pin()
    text = (CH / "A3-test-index.md").read_text(encoding="utf-8")
    got = [(m.group(1), int(m.group(2))) for m in re.finditer(r"^\| `(\w+)` \| (\d+) \|", text, re.M)]
    out = [] if got == want else [f"Appendix C lists {len(got)} tests; the pin has {len(want)}; "
                                  f"first difference: {next((a, b) for a, b in zip(got + [None] * 99, want) if a != b)}"]
    print(f"Appendix C: {len(got)} rows against {len(want)} tests at @{PIN}: {'ok' if not out else 'MISMATCH'}")
    return out


def check_config_table() -> list[str]:
    rows = [l for l in run("ch41_arithmetic.py", "--config").splitlines() if l.startswith("| `")]
    text = (CH / "A2-parameter-arithmetic.md").read_text(encoding="utf-8")
    page = {l.split(" | ")[0]: l for l in text.splitlines() if l.startswith("| `") and l.count(" | ") == 6}
    out = []
    for r in rows:
        key = r.split(" | ")[0]
        if key not in page or not page[key].startswith(r[:-2]):
            out.append(f"Appendix B.1 row differs: {r}")
    print(f"Appendix B.1: {len(rows)} configuration rows recomputed: {'ok' if not out else 'MISMATCH'}")
    return out


def bibliography() -> list[str]:
    reg = (BOOK / "RESEARCH_REFERENCES.md").read_text(encoding="utf-8")
    rows = []
    for line in reg.splitlines():
        m = re.match(r"^\| ([RS]\d\d) \| [^|]*\| ([^|]*)\| ([^|]*)\| (\d{4}) \| ([^|]*)\| [^|]*\| (\w+) \|", line)
        if m:
            i, title, author, year, ident, status = (g.strip() for g in m.groups())
            rows.append(f"| {i} | {author} | {year} | {title} | {ident} | {status} |")
    return rows


def check_bibliography() -> list[str]:
    want = bibliography()
    text = (CH / "A5-bibliography.md").read_text(encoding="utf-8")
    got = [l for l in text.splitlines() if re.match(r"^\| [RS]\d\d \|", l)]
    out = [] if got == want else [f"Appendix E differs from the register ({len(got)} rows, register {len(want)})"]
    print(f"Appendix E: {len(got)} rows against {len(want)} register entries: {'ok' if not out else 'MISMATCH'}")
    return out


def check_shapes() -> list[str]:
    printed = run("ch41_shapes.py")
    text = (CH / "A1-shape-trace.md").read_text(encoding="utf-8")
    shapes = set(re.findall(r"\((?:\d+, )+\d+\)", text))
    missing = sorted(s for s in shapes if s not in printed)
    print(f"Appendix A: {len(shapes)} distinct concrete shapes, {len(missing)} not in ch41_shapes.py's output"
          f"{': ' + ', '.join(missing) if missing else ''}")
    return [f"Appendix A shape not printed: {s}" for s in missing]


def evidence_inventory() -> None:
    print("\nEvidence labels per chapter (verified / measured / illustrative / cited references):")
    for p in sorted(CH.glob("[0-9][0-9]-*.md")):
        t = p.read_text(encoding="utf-8")
        n = {k: len(re.findall(rf"\*{k}\*", t)) for k in ("verified", "measured", "illustrative")}
        refs = len(set(re.findall(r"\[([RS]\d{2})\]", t)))
        print(f"  {p.name:<48} {n['verified']:>3} {n['measured']:>3} {n['illustrative']:>3}   {refs:>2}")
    print("\nParagraphs labelled illustrative, with the script each names:")
    for p in sorted(CH.glob("[0-9][0-9]-*.md")):
        for para in p.read_text(encoding="utf-8").split("\n\n"):
            if "*illustrative*" in para:
                scripts = sorted(set(re.findall(r"`(?:code-examples/|book/code-examples/)?(\w+\.py)`", para)))
                print(f"  {p.name[:2]}: {', '.join(scripts) or 'NO SCRIPT NAMED IN THE PARAGRAPH'}")


def run_examples() -> list[str]:
    import tempfile
    print("\nSelf-contained book examples, each run in a temporary directory:")
    failed, skipped = [], []
    for p in sorted(HERE.glob("*.py")):
        if p.name == Path(__file__).name:
            continue
        if p.name in SKIP:
            skipped.append(f"{p.name} ({SKIP[p.name]})")
            continue
        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                r = subprocess.run([sys.executable, str(p)], cwd=tmp, capture_output=True, timeout=600)
                status = "ok" if r.returncode == 0 else f"exit {r.returncode}"
            except subprocess.TimeoutExpired:
                status = "timeout"
        if status != "ok":
            failed.append(f"example {p.name}: {status}")
        print(f"  {p.name:<34} {status:<8} {time.time() - t0:6.1f} s")
    print("skipped:")
    for x in skipped:
        print(f"  {x}")
    failed += [f"SKIP names a missing example: {n}" for n in sorted(SKIP) if not (HERE / n).exists()]
    return failed


if __name__ == "__main__":
    if "--bibliography" in sys.argv:
        print("\n".join(bibliography()))
        sys.exit(0)
    problems = check_test_index() + check_config_table() + check_bibliography() + check_shapes()
    for x in problems:
        print("  PROBLEM:", x)
    evidence_inventory()
    if "--run" in sys.argv:
        problems += run_examples()
    print(f"\n{len(problems)} problems")
    sys.exit(1 if problems else 0)
