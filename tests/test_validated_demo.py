"""Evidence gate regressions; fixtures are synthetic, not model validation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from anulm.apps.validated_demo import digest, load_cases


class ValidatedDemoTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = self.root / 'catalogue.json'
        self.report = {'case_id': 'fixture', 'task': 'text', 'status': 'passed',
                       'approved': True, 'reviewed_by': 'synthetic-test',
                       'validated_at': 'test-only', 'criteria': 'fixture condition',
                       'output': 'fixture output', 'source_revision': 'test-only',
                       'dataset': {'source': 'synthetic fixture', 'license': 'test', 'held_out': True},
                       'input': 'fixture prompt', 'artifacts': []}
        for role in ['checkpoint', 'tokenizer', 'encoder']:
            path = self.root / role
            path.write_bytes(b'synthetic fixture only')
            self.report['artifacts'].append({'role': role, 'path': role, 'sha256': digest(path)})

    def save(self):
        report = self.root / 'report.json'
        report.write_text(json.dumps(self.report), encoding='utf-8')
        self.manifest.write_text(json.dumps({'version': 1, 'cases': [{
            'id': 'fixture', 'report': {'path': 'report.json', 'sha256': digest(report)}}]}), encoding='utf-8')

    def test_missing_empty_and_malformed_catalogues_fail_closed(self):
        self.assertEqual(load_cases(self.manifest, self.root)[0], [])
        for content in ['{}', '[]', 'not json', '{"version": 1, "cases": []}']:
            self.manifest.write_text(content, encoding='utf-8')
            self.assertEqual(load_cases(self.manifest, self.root)[0], [])

    def test_complete_report_is_accepted(self):
        self.save()
        cases, reasons = load_cases(self.manifest, self.root)
        self.assertEqual([c['id'] for c in cases], ['fixture'])
        self.assertEqual(reasons, [])

    def test_pending_failed_and_unapproved_reports_are_excluded(self):
        for status, approved in [('pending', True), ('failed', True), ('passed', False)]:
            self.report.update(status=status, approved=approved)
            self.save()
            self.assertEqual(load_cases(self.manifest, self.root)[0], [])

    def test_changed_checkpoint_and_report_are_excluded(self):
        self.save()
        (self.root / 'checkpoint').write_bytes(b'changed')
        self.assertEqual(load_cases(self.manifest, self.root)[0], [])
        self.report['artifacts'][0]['sha256'] = digest(self.root / 'checkpoint')
        self.save()
        (self.root / 'report.json').write_text('{}', encoding='utf-8')
        self.assertEqual(load_cases(self.manifest, self.root)[0], [])

    def test_missing_dependency_and_training_data_are_excluded(self):
        self.report['dataset']['held_out'] = False
        self.save()
        self.assertEqual(load_cases(self.manifest, self.root)[0], [])
        self.report['dataset']['held_out'] = True
        self.report['artifacts'] = self.report['artifacts'][:1]
        self.save()
        self.assertEqual(load_cases(self.manifest, self.root)[0], [])

    def test_media_requires_matching_input_and_encoder(self):
        image = self.root / 'image.fake'
        image.write_bytes(b'fixture image')
        self.report.update(task='image', input={'path': image.name, 'sha256': digest(image)})
        self.save()
        self.assertEqual(len(load_cases(self.manifest, self.root)[0]), 1)
        image.write_bytes(b'changed image')
        self.assertEqual(load_cases(self.manifest, self.root)[0], [])

    def test_outside_paths_and_case_identity_mismatch_are_excluded(self):
        self.report['case_id'] = 'different'
        self.save()
        self.assertEqual(load_cases(self.manifest, self.root)[0], [])
        self.report['case_id'] = 'fixture'
        self.report['artifacts'][0]['path'] = '../outside'
        self.save()
        self.assertEqual(load_cases(self.manifest, self.root)[0], [])

    def test_default_build_never_loads_models_or_experimental_ui(self):
        from anulm.apps import omni
        with patch.object(omni, 'build_experimental', side_effect=AssertionError('unvalidated UI')):
            demo = omni.build(object(), self.manifest)
        self.assertFalse(demo.config['dependencies'])

    def test_experimental_ui_requires_explicit_opt_in(self):
        from anulm.apps import omni
        with patch.object(omni, 'build_experimental', return_value='research UI') as build:
            model = object()
            self.assertEqual(omni.build(model, experimental=True), 'research UI')
            build.assert_called_once_with(model)


if __name__ == '__main__':
    unittest.main()
