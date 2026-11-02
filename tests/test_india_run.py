"""Offline safety checks for resumable data and the overnight run helpers."""
import io
from contextlib import contextmanager
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock
import subprocess
import os

from anulm.data import india_corpus as D
from anulm.data.india_examples import partition
from anulm.training import india_run as R


class IndiaRunTests(unittest.TestCase):
    def test_timeout_stops_only_the_owned_subprocess_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            process = MagicMock(pid=123456)
            process.wait.side_effect = [subprocess.TimeoutExpired('fixture', 1), 0]
            with patch.object(R, 'LOGS', Path(directory)), \
                 patch.object(R.subprocess, 'Popen', return_value=process), \
                 patch.object(R.subprocess, 'run') as killer, \
                 patch.object(R.os, 'killpg', create=True) as kill_group:
                with self.assertRaisesRegex(RuntimeError, 'wall-clock budget'):
                    R.command('fixture_timeout', ['-c', 'pass'], .001)
                if os.name == 'nt':
                    self.assertEqual(killer.call_args.args[0], ['taskkill', '/PID', '123456', '/T', '/F'])
                else:
                    self.assertEqual(kill_group.call_args.args[0], 123456)
            self.assertFalse((Path(directory) / 'active-child.json').exists())

    def test_archive_paths_and_links_cannot_escape_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / 'archive.tar.gz'
            with tarfile.open(archive_path, 'w:gz') as archive:
                member = tarfile.TarInfo('../../allowed.jpg')
                member.size = 3
                archive.addfile(member, io.BytesIO(b'abc'))
                link = tarfile.TarInfo('link.jpg')
                link.type, link.linkname = tarfile.SYMTYPE, '../../outside'
                archive.addfile(link)
            D.extract_selected(archive_path, root / 'media', {'allowed.jpg', 'link.jpg'})
            self.assertEqual((root / 'media/allowed.jpg').read_bytes(), b'abc')
            self.assertFalse((root / 'media/link.jpg').exists())
            self.assertFalse((root / 'allowed.jpg').exists())

    def test_weight_resume_checks_range_and_preserves_prefix(self):
        class Response(io.BytesIO):
            status = 206
            headers = {'Content-Range': 'bytes 2-5/6'}
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'model.bin'
            destination.with_name('model.bin.part').write_bytes(b'ab')
            with patch('urllib.request.urlopen', return_value=Response(b'cdef')) as request:
                R.download_weight('https://fixture.invalid/model', destination)
                self.assertEqual(request.call_args.args[0].get_header('Range'), 'bytes=2-8388609')
            self.assertEqual(destination.read_bytes(), b'abcdef')
            with patch('urllib.request.urlopen', side_effect=AssertionError('must reuse offline')):
                R.download_weight('https://fixture.invalid/model', destination)

    def test_invalid_encoder_range_never_appends_or_completes(self):
        class Response(io.BytesIO):
            status = 206
            headers = {'Content-Range': 'bytes 0-3/4'}
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'model.bin'
            destination.with_name('model.bin.part').write_bytes(b'ab')
            with patch('urllib.request.urlopen', side_effect=lambda *a, **k: Response(b'abcd')), patch.object(R.time, 'sleep'):
                with self.assertRaisesRegex(ValueError, 'range'):
                    R.download_weight('https://fixture.invalid/model', destination)
            self.assertFalse(destination.exists())
            self.assertEqual(destination.with_name('model.bin.part').read_bytes(), b'ab')

    def test_license_document_uses_bounded_full_transfer_and_offline_reuse(self):
        class Response(io.BytesIO):
            status = 200
            headers = {'Content-Length': '12'}
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / 'LICENSE.txt'
            with patch('urllib.request.urlopen', return_value=Response(b'license text')) as request:
                record = D.download_document('https://fixture.invalid/license', dest, reserve_gib=0)
                self.assertIsNone(request.call_args.args[0].get_header('Range'))
            self.assertEqual(dest.read_bytes(), b'license text')
            self.assertEqual(record['transfer'], 'bounded_full_document')
            with patch('urllib.request.urlopen', side_effect=AssertionError('offline reuse expected')):
                D.download_document('https://fixture.invalid/license', dest)

    def test_license_document_rejects_partial_http_range_response(self):
        class Response(io.BytesIO):
            status = 206
            headers = {'Content-Range': 'bytes 0-2/3'}
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / 'LICENSE.txt'
            with patch('urllib.request.urlopen', side_effect=lambda *a, **k: Response(b'abc')), patch.object(D.time, 'sleep'):
                with self.assertRaisesRegex(ValueError, 'HTTP 200'):
                    D.download_document('https://fixture.invalid/license', dest, reserve_gib=0)
            self.assertFalse(dest.exists())

    def test_license_document_rejects_incomplete_body(self):
        class Response(io.BytesIO):
            status = 200
            headers = {'Content-Length': '20'}
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / 'LICENSE.txt'
            with patch('urllib.request.urlopen', side_effect=lambda *a, **k: Response(b'abc')), patch.object(D.time, 'sleep'):
                with self.assertRaisesRegex(ValueError, 'Incomplete'):
                    D.download_document('https://fixture.invalid/license', dest, reserve_gib=0)
            self.assertFalse(dest.exists())

    def test_synthetic_groups_have_stable_partitions(self):
        assignments = {partition(f'price:{item}:{price}') for item in range(8) for price in range(100)}
        self.assertEqual(assignments, {'train', 'validation', 'test'})
        self.assertEqual(partition('price:0:123'), partition('price:0:123'))

    def test_generated_questions_and_translated_cards_do_not_cross_splits(self):
        from anulm.data import india_examples as E
        @contextmanager
        def no_render(font):
            yield lambda *args: None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(E, 'renderer', no_render), patch.object(E, 'download'):
                E.generate(root, 600)
            questions, groups, languages = {}, {}, {}
            for split in ['train', 'validation', 'test']:
                records = [json.loads(line) for line in (root / f'india_qa_{split}.jsonl').read_text(encoding='utf-8').splitlines()]
                questions[split] = {row['question'] for row in records}
                self.assertEqual(len(records), len(questions[split]))
                cards = [json.loads(line) for line in (root / f'india_caption_{split}.jsonl').read_text(encoding='utf-8').splitlines()]
                groups[split] = {row['group'] for row in cards}
                for row in cards:
                    languages.setdefault(row['group'], set()).add(row['lang'])
            for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
                self.assertFalse(questions[a] & questions[b])
                self.assertFalse(groups[a] & groups[b])
            self.assertTrue(all(value == {'en', 'hi'} for value in languages.values()))

    def test_no_usable_batches_raises_instead_of_looping_forever(self):
        from anulm.multimodal.omni_train import Omni
        omni = Omni.__new__(Omni)
        omni.args = SimpleNamespace(seed=0)
        omni.batches = lambda *a, **k: iter([])
        task = SimpleNamespace(name='empty', _it=None, epoch=0, rows=[], batch=1)
        with self.assertRaisesRegex(ValueError, 'no usable'):
            omni.next_batch(task)


if __name__ == '__main__':
    unittest.main()
