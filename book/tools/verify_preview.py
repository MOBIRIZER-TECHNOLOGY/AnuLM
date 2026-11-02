"""Check the complete generated editions against the book manifest.

    python tools/verify_preview.py
    python tools/verify_preview.py --pdf
"""
from __future__ import annotations

import argparse
from collections import Counter
from html.parser import HTMLParser
import json
from pathlib import Path

BOOK = Path(__file__).resolve().parents[1]


class Inventory(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.links, self.chapters, self.cards, self.missing = [], [], [], 0, 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.append(a["id"])
        if tag == "a" and a.get("href", "").startswith("#"):
            self.links.append(a["href"][1:])
        classes = a.get("class", "").split()
        if tag == "article" and "chapter" in classes:
            self.chapters.append(a.get("id"))
        self.cards += "reader-card" in classes
        self.missing += "missing" in classes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((BOOK / "config/book-manifest.json").read_text(encoding="utf-8"))
    expected = ["ch" + str(int(name[:2])) for name in manifest["chapters"]]
    expected += ["chAppendix-" + chr(65 + i) for i, _ in enumerate(manifest["appendices"])]
    for name in ("index.html", "print.html"):
        src = (BOOK / "preview" / name).read_text(encoding="utf-8")
        inv = Inventory()
        inv.feed(src)
        assert inv.chapters == expected, (name, "chapter manifest/order mismatch")
        duplicates = [i for i, count in Counter(inv.ids).items() if count > 1]
        assert not duplicates, (name, "duplicate ids", duplicates)
        assert not set(inv.links) - set(inv.ids), (name, "broken anchors", set(inv.links) - set(inv.ids))
        assert inv.cards == len(manifest["chapters"]), (name, "missing chapter entry cards")
        assert not inv.missing, (name, "missing figure")
        assert '<meta charset="utf-8">' in src and src.startswith("<!doctype html>"), name
        print(f"PASS: {name}: 41 chapters, 6 appendices, 41 entry cards; anchors and figure paths checked")
    if args.pdf:
        from pypdf import PdfReader
        reader = PdfReader(BOOK / "preview" / "inside-anulm.pdf")
        text = "\n".join(p.extract_text() or "" for p in reader.pages)
        assert text.count("Before you begin") == 41, "PDF does not contain all chapter entry cards"
        for label in ("Start here", "Learning paths", "Launch checklist", "Appendix F"):
            assert label in text, ("PDF missing", label)
        print(f"PASS: PDF: {len(reader.pages)} pages; 41 chapter entry cards and reader/back matter present")
    print("These structural checks do not replace visual QA or technical review.")


if __name__ == "__main__":
    main()
