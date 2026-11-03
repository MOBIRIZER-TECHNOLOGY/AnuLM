"""Repository layout, packaging and docs hygiene.

    python -m unittest tests.test_quality_layout

Each test pins down one thing that had drifted: a command table listing
modules that cannot run, tools with one machine's paths baked in, a .gitignore
whitelist wide enough to commit a contamination report, docs pointing at files
that moved. No network, no GPU, no model weights; `git` is used where present.
"""

import ast
import importlib
import re
import shutil
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def has_main_block(path: Path) -> bool:
    """True if the module has a top-level `if __name__ == "__main__":`."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"
                and len(node.test.comparators) == 1
                and isinstance(node.test.comparators[0], ast.Constant)
                and node.test.comparators[0].value == "__main__"):
            return True
    return False


def module_path(dotted: str) -> Path:
    return REPO.joinpath(*dotted.split(".")).with_suffix(".py")


class CommandTable(unittest.TestCase):
    """`python -m anulm NAME` runs NAME's module as __main__; a module without a
    main block runs, does nothing and exits 0, which looks like success."""

    def setUp(self):
        from anulm.__main__ import COMMANDS
        self.commands = COMMANDS

    def test_every_command_module_has_a_main_block(self):
        for name, dotted in self.commands.items():
            p = module_path(dotted)
            self.assertTrue(p.exists(), f"{name}: {dotted} has no file")
            self.assertTrue(has_main_block(p), f"{name}: {dotted} has no __main__ block, so the command does nothing")

    def test_every_runnable_module_is_a_command(self):
        listed = set(self.commands.values())
        for p in sorted((REPO / "anulm").rglob("*.py")):
            if "__pycache__" in p.parts or p.name in ("__init__.py", "__main__.py"):
                continue
            if has_main_block(p):
                dotted = ".".join(p.relative_to(REPO).with_suffix("").parts)
                self.assertIn(dotted, listed, f"{dotted} has a __main__ block but no `python -m anulm` command")


class Tools(unittest.TestCase):

    def test_no_machine_specific_paths(self):
        for p in sorted((REPO / "tools").rglob("*.py")):
            src = p.read_text(encoding="utf-8")
            self.assertNotRegex(src, r"(?i)C:[\\/]+workspace", f"{p.name} hard-codes a machine path")

    def test_backfill_curve_does_nothing_on_import(self):
        # Before the guard, importing it read (and rewrote) coder_curve.csv at once,
        # and raised FileNotFoundError on any checkout without that file.
        sys.modules.pop("tools.backfill_curve", None)
        mod = importlib.import_module("tools.backfill_curve")
        self.assertEqual(mod.CURVE.parent, REPO)
        self.assertTrue(callable(mod.main))

    def test_eval_noise_paths_follow_the_checkout(self):
        src = (REPO / "tools" / "eval_noise.py").read_text(encoding="utf-8")
        self.assertIn("ROOT = Path(__file__).resolve().parents[1]", src)

    def test_update_tasks_writes_only_inside_the_repository(self):
        src = (REPO / "tools" / "update_tasks.py").read_text(encoding="utf-8")
        self.assertNotIn("Path.home()", src)
        self.assertNotIn(".claude", src)
        self.assertNotIn("write_memory", src)
        self.assertIn('HERE / "docs" / "history" / "TASKS.md"', src)

    def test_samples_page_names_the_current_tools(self):
        src = (REPO / "tools" / "samples.py").read_text(encoding="utf-8")
        for old in ("tools_sweep.py", "tools_samples.py"):
            self.assertNotIn(old, src)


class Experiments(unittest.TestCase):

    ARCHIVE = REPO / "experiments" / "archive"

    def test_archived_scripts_resolve_root_to_the_repository(self):
        # tests/test_model.py checks experiments/*.py; the archive is one level deeper.
        scripts = sorted(self.ARCHIVE.glob("*.py"))
        self.assertGreaterEqual(len(scripts), 5)
        for f in scripts:
            m = re.search(r"^ROOT = Path\(__file__\)\.resolve\(\)\.parents\[(\d)\]", f.read_text(encoding="utf-8"), re.M)
            self.assertIsNotNone(m, f"{f.name}: no ROOT = Path(__file__).resolve().parents[N]")
            self.assertEqual(f.resolve().parents[int(m.group(1))], REPO, f"{f.name}: ROOT is not the repository")

    def test_archive_readme_lists_every_script(self):
        readme = (self.ARCHIVE / "README.md").read_text(encoding="utf-8")
        for f in self.ARCHIVE.iterdir():
            if f.suffix in (".py", ".cmd", ".sh"):
                self.assertIn(f"`{f.name}`", readme)

    def test_branch_scripts_match_the_package_training_run(self):
        hits = list(REPO.glob("experiments/**/decide_branch.py")) + \
            list(REPO.glob("experiments/**/decide_general_branch.py")) + \
            list(REPO.glob("experiments/**/demo_branch.py"))
        self.assertEqual(len(hits), 3)
        for f in hits:
            src = f.read_text(encoding="utf-8")
            self.assertFalse("CommandLine -like '*train.py*'" in src, f"{f.name} still matches only the old train.py")
            tree = ast.parse(src)
            pattern = next(ast.literal_eval(n.value) for n in tree.body
                           if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "TRAIN_RE")
            rx = re.compile(pattern)
            for cmd in ("python -u -m anulm train --preset 350m --out ckpt_base_v2.pt",
                        "python -u -m anulm.training.train --out ckpt_base_v2.pt",
                        "python -u train.py --out ckpt_base_v2.pt"):
                self.assertTrue(rx.search(cmd), f"{f.name}: TRAIN_RE misses {cmd!r}")
            for cmd in ("python -m anulm.multimodal.omni_train --out x.pt",
                        "python -m anulm train_from_scratch --work-dir w",
                        "python -m anulm.training.train_from_scratch --work-dir w"):
                self.assertFalse(rx.search(cmd), f"{f.name}: TRAIN_RE also matches {cmd!r}")


@unittest.skipUnless(shutil.which("git"), "git not on PATH")
class GitIgnore(unittest.TestCase):

    def ignored(self, path: str) -> bool:
        r = subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=REPO)
        return r.returncode == 0

    def test_private_and_generated_files_are_ignored(self):
        for p in (".demo_auth", "logs/omni_v3/pipeline.log", ".claude/launch.json",
                  "anulm.egg-info/PKG-INFO", "build/lib/x.py", "dist/anulm-0.2.0.tar.gz",
                  ".pytest_cache/v", ".ruff_cache/v", "course/preview/sample.html",
                  "data/contamination_github.json", "data/contamination_sft_code.json"):
            self.assertTrue(self.ignored(p), f"{p} is not ignored")

    def test_tracked_data_files_stay_tracked(self):
        out = subprocess.run(["git", "ls-files", "data"], cwd=REPO, capture_output=True, text=True).stdout.split()
        self.assertTrue(out)
        for p in out:
            self.assertFalse(self.ignored(p), f"{p} is tracked but the .gitignore would no longer admit it")


class Packaging(unittest.TestCase):

    def test_versions_agree(self):
        meta = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        cff = re.search(r"^version: (\S+)", (REPO / "CITATION.cff").read_text(encoding="utf-8"), re.M).group(1)
        self.assertEqual(meta["version"], cff)
        self.assertIn("huggingface_hub>=0.30", meta["optional-dependencies"]["demo"])
        self.assertIn("Repository", meta["urls"])

    def test_requirements_list_numpy(self):
        lines = [l.strip() for l in (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines()]
        self.assertIn("numpy", lines)


class Docs(unittest.TestCase):

    def test_readme_links_resolve(self):
        text = (REPO / "README.md").read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)#\s]+)\)", text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            self.assertTrue((REPO / target).exists(), f"README links to missing {target}")

    def test_readme_links_verification_and_speech(self):
        text = (REPO / "README.md").read_text(encoding="utf-8")
        self.assertIn("](docs/VERIFICATION.md)", text)
        self.assertIn("](docs/SPEECH.md)", text)
        self.assertNotIn("353M/151M-active", text)

    def test_moved_files_are_not_cited_at_their_old_paths(self):
        old = [f"docs/{n}.md" for n in ("TASKS", "ANNOUNCEMENT", "CODER_PLAN", "TRANSLATE_PLAN")]
        old += ["logs/decide_speed"]
        for n in ("demo_branch", "decide_branch", "decide_general_branch", "post_v2", "chat_v5",
                  "overnight_text", "prep_indic_extra"):
            old.append(f"experiments/{n}.py")
        roots = [REPO / "README.md", REPO / "docs", REPO / "experiments", REPO / "tools"]
        files = []
        for r in roots:
            files += [r] if r.is_file() else [p for p in r.rglob("*")
                                              if p.suffix in (".md", ".py", ".sh", ".cmd") and "__pycache__" not in p.parts]
        for p in files:
            text = p.read_text(encoding="utf-8", errors="replace")
            for o in old:
                self.assertFalse(o in text, f"{p.relative_to(REPO)} still cites {o}")

    def test_decide_speed_results_are_where_results_md_says(self):
        for name in ("cpu.json", "gpu.json"):
            self.assertTrue((REPO / "docs" / "results" / "decide_speed" / name).exists())
        self.assertIn("`docs/results/decide_speed/`", (REPO / "docs" / "RESULTS.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
