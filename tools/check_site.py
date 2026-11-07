"""Check the landing page before it is published (docs/LAUNCH.md).

    python tools/check_site.py                  # exit 1 if anything would ship broken
    python tools/check_site.py --draft          # skip the launch-only checks (placeholders, form wiring)
    python tools/check_site.py --dir _site      # check a deploy folder (.github/workflows/pages.yml)

Checks every page under site/:
  - every in-page link (#id) and every local file it references exists;
  - no script is loaded from another host (the page promises no trackers);
  - launch only: no "[TO FILL" placeholder is left (privacy notice), and the
    enquiry form is either connected (data-endpoint, or at least data-email) or,
    before it is, replaced by the public contact card with no privacy page
    deployed (the page then collects no personal data).
"""
from __future__ import annotations

import argparse
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / "site"


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.refs, self.scripts, self.form = set(), [], [], {}

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.add(a["id"])
        for k in ("href", "src"):
            if a.get(k):
                self.refs.append((tag, a[k]))
        if tag == "script" and a.get("src"):
            self.scripts.append(a["src"])
        if tag == "form" and a.get("id") == "enquiry-form":
            self.form = a


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")     # Hindi placeholders on a Windows console
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--dir", default=str(SITE), help="the folder to check (default: site/)")
    a = ap.parse_args()
    site = Path(a.dir)
    problems = []
    for page in sorted(site.glob("*.html")):
        text = page.read_text(encoding="utf-8")
        p = Page()
        p.feed(text)
        for tag, ref in p.refs:
            if ref.startswith(("http://", "https://", "mailto:", "data:")):
                continue
            target, _, frag = ref.partition("#")
            if target:
                f = site / target
                if not f.exists():
                    problems.append(f"{page.name}: <{tag}> points at missing file {target}")
                    continue
                ids = Page()
                ids.feed(f.read_text(encoding="utf-8")) if f.suffix == ".html" and frag else None
                if frag and f.suffix == ".html" and frag not in ids.ids:
                    problems.append(f"{page.name}: {ref} has no element with id {frag!r}")
            elif frag and frag not in p.ids:
                problems.append(f"{page.name}: #{frag} has no element with that id")
        for src in p.scripts:
            if re.match(r"(https?:)?//", src):
                problems.append(f"{page.name}: external script {src} (the page promises no trackers)")
        if not a.draft:
            for m in re.finditer(r"\[TO FILL[^\]]*\]", text):
                problems.append(f"{page.name}: placeholder {m.group(0)}")
            if p.form is not None and page.name == "index.html":
                connected = p.form.get("data-endpoint") or p.form.get("data-email")
                if not connected and ("contact-fallback" not in p.ids or (site / "privacy.html").exists()):
                    problems.append("index.html: the enquiry form is not connected (data-endpoint or data-email); "
                                    "deploy without privacy.html to use the public contact card instead")
    for line in problems:
        print(line)
    print(f"{len(problems)} problem(s) in {site}" + (" (draft mode)" if a.draft else ""))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
