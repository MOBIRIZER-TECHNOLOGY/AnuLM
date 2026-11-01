"""Regression checks for imports, old checkpoints and installed CLI discovery."""
import io
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from anulm import model as native
from anulm.paths import PROJECT_ROOT, source_path


class LayoutTests(unittest.TestCase):
    def test_legacy_import_is_the_native_module(self):
        import model as legacy
        self.assertIs(legacy, native)
        self.assertIs(legacy._sdpa, native._sdpa)
        original = native.NATIVE_GQA
        try:
            legacy.NATIVE_GQA = not original
            self.assertEqual(native.NATIVE_GQA, not original)
        finally:
            native.NATIVE_GQA = original

    def test_old_model_config_pickle_still_loads(self):
        # Emulate the module name encoded by historical .pt checkpoints.
        import model  # noqa: F401
        config = native.AnuLMConfig()
        stream = io.BytesIO()
        old = native.AnuLMConfig.__module__
        try:
            native.AnuLMConfig.__module__ = 'model'
            torch.save({'cfg': config}, stream)
        finally:
            native.AnuLMConfig.__module__ = old
        stream.seek(0)
        restored = torch.load(stream, weights_only=False)['cfg']
        self.assertIsInstance(restored, native.AnuLMConfig)
        self.assertEqual(restored.hidden_size, config.hidden_size)

    def test_migration_map_points_to_real_files(self):
        mapping = json.loads((PROJECT_ROOT / 'docs/source-paths.json').read_text(encoding='utf-8'))
        for old, new in mapping.items():
            with self.subTest(old=old):
                self.assertEqual(source_path(old), PROJECT_ROOT / new)
                self.assertTrue(source_path(old).is_file())

    def test_installed_cli_works_outside_checkout(self):
        with tempfile.TemporaryDirectory() as work:
            result = subprocess.run([sys.executable, '-m', 'anulm', '--help'], cwd=work,
                                    text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('train', result.stdout)
        self.assertIn('quickstart', result.stdout)

    def test_experiment_launch_lists_do_not_name_removed_root_scripts(self):
        mapping = json.loads((PROJECT_ROOT / 'docs/source-paths.json').read_text(encoding='utf-8'))
        old_python = {name for name in mapping if name.endswith('.py')}
        for path in (PROJECT_ROOT / 'experiments').rglob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, ast.List) and node.elts and isinstance(node.elts[0], ast.Constant):
                    with self.subTest(file=path.name, line=node.lineno):
                        self.assertNotIn(node.elts[0].value, old_python)


if __name__ == '__main__':
    unittest.main()
