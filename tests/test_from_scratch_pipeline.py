"""Real bounded CPU pipeline test with no external model or dataset downloads."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from anulm.paths import PROJECT_ROOT


class Pipeline(unittest.TestCase):
    def test_train_skip_and_reject_changed_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corpus = root / 'corpus.txt'
            corpus.write_text('\n\n'.join(f'Document {i}: ' +
                'AnuLM learns from documents with independent validation examples. ' * 60
                for i in range(12)), encoding='utf-8')
            train, held = root / 'train.jsonl', root / 'held.jsonl'
            train.write_text(json.dumps({'question': 'Hi?', 'answer': 'Hello.', 'lang': 'en'}) + '\n')
            held.write_text(json.dumps({'question': 'Bye?', 'answer': 'Goodbye.', 'lang': 'en'}) + '\n')
            command = [sys.executable, '-m', 'anulm.training.from_scratch', '--corpus', str(corpus),
                '--work-dir', str(root / 'run'), '--source', 'procedural regression fixture',
                '--license', 'project test fixture', '--steps', '1', '--joint-steps', '1',
                '--vocab', '280', '--context', '64', '--eval-every', '1',
                '--task', f'chat=text:{train}:1:1', '--held', f'chat={held}', '--accum', '2']
            env = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONIOENCODING': 'utf-8'}
            def invoke(cmd):
                return subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True,
                                      text=True, encoding='utf-8', errors='replace', timeout=120)
            first = invoke(command)
            logs = '\n'.join(p.read_text(encoding='utf-8', errors='replace') for p in (root / 'run').glob('*.log'))
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr + logs)
            self.assertTrue((root / 'run/omni.pt.last').is_file())
            plan = root / 'plan.json'
            plan.write_text(json.dumps({'required_tasks': ['chat'], 'tasks': [{
                'name': 'chat', 'kind': 'text', 'language': 'en', 'split': 'validation',
                'path': str(held), 'source': 'procedural regression fixture',
                'license': 'project test fixture', 'limit': 1, 'criteria': []}]}))
            report_command = [sys.executable, '-m', 'anulm.evaluation.report', '--checkpoint',
                str(root / 'run/omni.pt'), '--plan', str(plan), '--out', str(root / 'report.json'),
                '--device', 'cpu', '--max-tokens', '4']
            evaluated = invoke(report_command)
            self.assertEqual(evaluated.returncode, 0, evaluated.stdout + evaluated.stderr)
            report = json.loads((root / 'report.json').read_text(encoding='utf-8'))
            self.assertFalse(report['all_declared_gates_passed'])
            self.assertEqual(report['tasks'][0]['evaluated_rows'], 1)
            self.assertEqual(report['approval'], 'unreviewed')
            second = invoke(command)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            for stage in ['prepare', 'text', 'grow', 'joint']:
                self.assertIn(f'skip {stage}: verified', second.stdout)
            changed = command.copy()
            changed[changed.index('--steps') + 1] = '2'
            rejected = invoke(changed)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('step budget or inputs changed', rejected.stderr)
            self.assertFalse((root / 'run/RUNNING.json').exists())


if __name__ == '__main__':
    unittest.main()
