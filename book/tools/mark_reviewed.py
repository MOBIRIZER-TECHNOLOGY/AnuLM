"""
Mark chapters as reviewed in their claim records, after the integrator has checked the fixes.

    python book/tools/mark_reviewed.py 08 09 10 --note "independent review 2026-11-01; fixes applied and re-verified"

A chapter is marked only if `verify_chapters.py` finds no problem in it, its record has a
"Review fixes" section, and the chapter meets the code standard (appendices are exempt).
The record's "**Status:** drafted" line becomes "**Status:** reviewed (<date>; <note>)".
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_chapters as V  # noqa: E402

BOOK = Path(__file__).resolve().parent.parent
CLAIMS = BOOK / "technical-review" / "claims"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    note = sys.argv[sys.argv.index("--note") + 1] if "--note" in sys.argv else "reviewed"
    args = [a for a in args if a != note]
    failed = 0
    for key in args:
        key = key.upper().zfill(2)
        chapter = next(iter(sorted(V.CHAPTERS.glob(f"{key}-*.md"))), None)
        record = CLAIMS / (f"{key}.md" if key[0].isdigit() else "appendices.md")
        if chapter is None or not record.exists():
            print(f"{key}: chapter or record missing"); failed += 1; continue
        problems, notes, stats = V.check(chapter)
        below = [n for n in notes[1:] if "below" in n]
        text = record.read_text(encoding="utf-8")
        if problems or below or "Review fixes" not in text:
            print(f"{key}: NOT marked: problems {problems}, standard {below}, "
                  f"review fixes {'present' if 'Review fixes' in text else 'missing'}"); failed += 1; continue
        new, n = re.subn(r"^\*\*Status:\*\* drafted.*$", f"**Status:** reviewed ({date.today().isoformat()}; {note})",
                         text, count=1, flags=re.M)
        if not n:
            print(f"{key}: no 'Status: drafted' line (already marked?)"); continue
        record.write_text(new, encoding="utf-8")
        print(f"{key}: marked reviewed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
