"""
The research audit (requirement 22): read RESEARCH_REFERENCES.md and the
chapters, and write technical-review/research-audit.md.

    python book/tools/research_audit.py

It counts what can be counted: references by status, missing identifiers,
which references the chapters cite and which they don't, and citations of
references that are not CHECKED. Claims without evidence can't be found by a
script; the report lists them as a Stage 5 review item, never as "none".
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

BOOK = Path(__file__).resolve().parent.parent
REGISTER = BOOK / "RESEARCH_REFERENCES.md"
CHAPTERS = BOOK / "chapters"
OUT = BOOK / "technical-review" / "research-audit.md"
FIELDS = ["id", "topic", "title", "author", "year", "identifier", "source", "status", "verified", "claims", "notes"]
STATUSES = ("CITED", "CHECKED", "CORRECTED", "REJECTED")
UNRESOLVED = re.compile(r"to (verify|record|pin|identify|confirm)|missing|<version>", re.I)
CITE = re.compile(r"\[([RS]\d{2})\]")


def register() -> list[dict]:
    rows = []
    for line in REGISTER.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == len(FIELDS) and re.fullmatch(r"[RS]\d{2}", cells[0]):
            row = dict(zip(FIELDS, cells))
            row["state"] = next((s for s in STATUSES if row["status"].upper().startswith(s)), "UNKNOWN")
            rows.append(row)
    return rows


def citations() -> dict[str, list[str]]:
    used: dict[str, list[str]] = {}
    if CHAPTERS.exists():
        for f in sorted(CHAPTERS.rglob("*.md")):
            for ref in set(CITE.findall(f.read_text(encoding="utf-8"))):
                used.setdefault(ref, []).append(f.relative_to(BOOK).as_posix())
    return used


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    rows = register()
    ids = [r["id"] for r in rows]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    count = Counter(r["state"] for r in rows)
    used = citations()
    by_id = {r["id"]: r for r in rows}
    missing_id = [r for r in rows if UNRESOLVED.search(r["identifier"] + " " + r["source"])]
    unknown_cites = sorted(set(used) - set(by_id))
    unchecked_cites = sorted(i for i in used if i in by_id and by_id[i]["state"] != "CHECKED")
    unused = [r for r in rows if r["id"] not in used and r["state"] != "REJECTED"]
    no_claim = [r for r in rows if not r["claims"]]
    text = REGISTER.read_text(encoding="utf-8")
    log = text[text.find("## 6. Corrections log"): text.find("## 7.")]
    corrected = sorted({i for line in log.splitlines() if line.startswith("| 20")
                        for i in re.findall(r"[RS]\d{2}", line.split("|")[2])})
    review = BOOK / "technical-review" / "claims-review.md"           # Stage 5 writes it, chapter by chapter
    refs_done = (count["CHECKED"] == len(rows) - count["REJECTED"] and not unchecked_cites and not unknown_cites
                 and not missing_id and len(rows) > 0)
    # Stage 5 is per chapter: a "## Chapter NN" section in claims-review.md whose status line says "reviewed".
    rtext = review.read_text(encoding="utf-8") if review.exists() else ""
    for extra in sorted((BOOK / "technical-review" / "claims").glob("*.md")):   # one file per chapter
        rtext += "\n" + extra.read_text(encoding="utf-8")
    reviewed = {int(n) for n, body in re.findall(r"^## Chapter (\d+)\b(.*?)(?=^## |\Z)", rtext, re.M | re.S)
                if re.search(r"^\*\*Status:\*\* reviewed", body, re.M)}
    written = sorted(int(f.name[:2]) for f in CHAPTERS.glob("[0-9][0-9]-*.md")) if CHAPTERS.exists() else []
    planned = len(re.findall(r"^\*\*(\d+)\. ", (BOOK / "BOOK_PLAN.md").read_text(encoding="utf-8"), re.M))
    unreviewed = [n for n in written if n not in reviewed]
    complete = refs_done and planned > 0 and len(written) == planned and not unreviewed

    L = [f"# Research audit", "", f"Written {date.today().isoformat()} by `book/tools/research_audit.py` from "
         "`RESEARCH_REFERENCES.md` and `book/chapters/`.", "",
         f"**Research-complete: {'yes' if complete else 'no'}.** References verified: {'yes' if refs_done else 'no'}. "
         f"Claim review (`technical-review/claims-review.md`): {len(written) - len(unreviewed)} of {len(written)} written "
         f"chapters reviewed; {len(written)} of {planned} planned chapters written"
         + (f"; not yet reviewed: {', '.join(map(str, unreviewed))}" if unreviewed else "") + ".", "",
         "## References", "",
         "| | count |", "|---|---|", f"| total | {len(rows)} |"]
    L += [f"| {s.lower()} | {count[s]} |" for s in STATUSES]
    if count["UNKNOWN"]:
        L.append(f"| status not recognised | {count['UNKNOWN']} |")
    L += ["", f"Inventory entries (R): {sum(i.startswith('R') for i in ids)}; supplementary (S): "
          f"{sum(i.startswith('S') for i in ids)}.", "",
          f"Corrected at some point (from the corrections log, now promoted): {len(corrected)} — "
          + (", ".join(f"`{i}`" for i in corrected) or "none") + "."]
    if dup:
        L += ["", f"**Duplicate IDs:** {', '.join(dup)}"]
    L += ["", "## Missing or unresolved identifiers", ""]
    L += [f"- `{r['id']}` {r['topic']}: {r['identifier']} / {r['source']}" for r in missing_id] or ["- none"]
    L += ["", "## References without a claim to support", ""]
    L += [f"- `{r['id']}` {r['topic']}" for r in no_claim] or ["- none"]
    L += ["", "## Use in chapters", "",
          f"Chapter files scanned: {len(list(CHAPTERS.rglob('*.md'))) if CHAPTERS.exists() else 0}.", ""]
    L += ["**Cited in a chapter but not CHECKED** (must be fixed before publication):", ""]
    L += [f"- `{i}` ({by_id[i]['state']}) in {', '.join(used[i])}" for i in unchecked_cites] or ["- none"]
    L += ["", "**Cited in a chapter but not in the register:**", ""]
    L += [f"- `{i}` in {', '.join(used[i])}" for i in unknown_cites] or ["- none"]
    L += ["", f"**In the register, not cited by any chapter:** {len(unused)}", ""]
    L += [", ".join(f"`{r['id']}`" for r in unused) or "none"]
    L += ["", "## Claims (Stage 5, by review)", "",
          "A script cannot tell whether a sentence needs evidence. Stage 5 reviews each chapter for:", "",
          "- repository claims without a source-map row or a line citation;",
          "- external claims without a CHECKED reference;",
          "- quality claims without a `docs/RESULTS.md` section (*measured here*) or the words *not measured here*.",
          "", "Each chapter's review is recorded in `technical-review/claims-review.md`. For a chapter without "
          "one, these counts are **unknown**, not zero.", "",
          f"Reviewed: {', '.join(f'Ch {n}' for n in sorted(reviewed)) or 'none'}. "
          f"Written but not reviewed: {', '.join(f'Ch {n}' for n in unreviewed) or 'none'}."]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"{len(rows)} references: " + ", ".join(f"{count[s]} {s.lower()}" for s in STATUSES)
          + f"; {len(corrected)} corrected along the way; {len(missing_id)} unresolved identifiers; "
          f"references verified: {'yes' if refs_done else 'no'}; chapters reviewed {len(written) - len(unreviewed)}/{len(written)} "
          f"written, {len(written)}/{planned} planned; research-complete: {'yes' if complete else 'no'}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
