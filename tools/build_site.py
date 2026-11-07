"""Copy site/ into a deploy folder, leaving out what is not ready to publish.

    python tools/build_site.py --out _site      # then: python tools/check_site.py --dir _site

Until the enquiry form is connected (data-endpoint or data-email on
<form id="enquiry-form">) and the privacy notice has no "[TO FILL" left, the
deployed page uses the public contact card instead: the form, the privacy
notice and every link to it are left out, so the live page collects no
personal data and shows no placeholder. Once both are done, everything ships.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "_site"))
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(SITE, out)
    index = (out / "index.html").read_text(encoding="utf-8")
    form = re.search(r'<form class="card enquiry" id="enquiry-form".*?</form>\n', index, re.S)
    if not form:
        raise SystemExit("site/index.html: <form id=\"enquiry-form\"> not found")
    connected = re.search(r'data-endpoint="[^"]+"|data-email="[^"]+"', form.group(0))
    privacy_ready = "[TO FILL" not in (out / "privacy.html").read_text(encoding="utf-8")
    if connected and privacy_ready:
        print(f"{out}: full site (form connected, privacy notice filled)")
        return
    index = index.replace(form.group(0), "")
    index = re.sub(r'\s*<li data-requires-form hidden>.*?</li>', "", index)
    (out / "index.html").write_text(index, encoding="utf-8", newline="\n")
    (out / "privacy.html").unlink()
    why = [w for w, ok in (("form not connected", connected), ("privacy notice not filled", privacy_ready)) if not ok]
    print(f"{out}: contact-card site, without the form and the privacy notice ({'; '.join(why)})")


if __name__ == "__main__":
    main()
