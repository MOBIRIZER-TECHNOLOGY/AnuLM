"""
Shared machinery for the chapter checks: load the learner's file or a reference
solution, run every check, and prove the checks work with --self-test.

A chapter's check file defines CHECKS (via @check) and MUTANTS, a function
returning (description, namespace) pairs: deliberately broken versions of the
reference solution, each of which at least one check must catch.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "course" / "build"
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def load(path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def solution(chapter: int) -> types.ModuleType:
    return load(BUILD / "solutions" / f"ch{chapter:02d}.py")


def with_replaced(module, names, **replace) -> types.SimpleNamespace:
    """A namespace with `names` taken from `module`, some of them replaced."""
    ns = types.SimpleNamespace(**{n: getattr(module, n) for n in names})
    for k, v in replace.items():
        setattr(ns, k, v)
    return ns


def run(checks, M, quiet=False) -> list[str]:
    failed = []
    for fn in checks:
        try:
            fn(M)
            if not quiet:
                print(f"  pass  {fn.__name__}")
        except NotImplementedError as e:
            failed.append(fn.__name__)
            if not quiet:
                print(f"  todo  {fn.__name__}  ({e})")
        except Exception as e:  # noqa: BLE001 -- report every failure and keep going
            failed.append(fn.__name__)
            if not quiet:
                msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
                print(f"  FAIL  {fn.__name__}\n        {msg}")
    return failed


def main(chapter: int, checks, mutants, doc: str, learner: str = "anulm.py") -> None:
    """learner: the file in course/build/ that the chapter fills in (data.py for Chapters 2-3)."""
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=["build", "solution"], default="build")
    p.add_argument("--self-test", action="store_true", help="prove each check catches its bug")
    a = p.parse_args()
    sol = solution(chapter)
    if a.self_test:
        print("reference solution must pass everything:")
        bad = run(checks, sol, quiet=True)
        print("  ok" if not bad else f"  FAILED: {bad}")
        print("each broken version must be caught by at least one check:")
        missed = 0
        for name, ns in mutants(sol):
            caught = run(checks, ns, quiet=True)
            missed += not caught
            print(f"  {'caught' if caught else 'MISSED'}  {name:46s} by {', '.join(caught) or '-'}")
        sys.exit(1 if bad or missed else 0)
    M = sol if a.impl == "solution" else load(BUILD / learner)
    where = "the reference solution" if a.impl == "solution" else f"course/build/{learner}"
    print(f"Chapter {chapter} checks on {where}:")
    failed = run(checks, M)
    print(f"\n{len(checks) - len(failed)} of {len(checks)} pass")
    sys.exit(1 if failed else 0)
