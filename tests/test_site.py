"""The landing page stays consistent: every link and file it uses exists, and no
script loads from another host (python tools/check_site.py --draft).

    python -m unittest tests.test_site

Launch-only checks (placeholders in the privacy notice, the form's connection)
are left to `tools/check_site.py` without --draft, run before deploying.
"""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Site(unittest.TestCase):
    def test_draft_check_passes(self):
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "check_site.py"), "--draft"],
                           capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_launch_check_catches_what_is_unfinished(self):
        """Until the notice is filled and the form connected, launch mode must fail."""
        text = (ROOT / "site" / "privacy.html").read_text(encoding="utf-8")
        if "[TO FILL" not in text:
            self.skipTest("the privacy notice is filled in")
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "check_site.py")],
                           capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(r.returncode, 1)
        self.assertIn("placeholder", r.stdout)


if __name__ == "__main__":
    unittest.main()
