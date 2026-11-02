"""Offline regressions for bounded dataset setup and portable manifests."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from anulm.data import setup_multimodal as D


class DatasetSetupTests(unittest.TestCase):
    def test_archive_budget_is_enforced(self):
        reader = D.BoundedReader(io.BytesIO(b'123456789'), 4)
        self.assertEqual(reader.read(20), b'1234')
        with self.assertRaisesRegex(ValueError, 'budget'):
            reader.read(1)

    def test_oversized_metadata_never_becomes_a_completed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metadata'
            with patch.object(D, 'request', return_value=io.BytesIO(b'12345')):
                with self.assertRaisesRegex(ValueError, 'limit'):
                    D.download('https://fixture.invalid', path, limit=4)
            self.assertFalse(path.exists())

    def test_integrity_and_split_checks_detect_modified_files_and_overlap(self):
        try:
            from PIL import Image
            import soundfile  # noqa: F401
        except ImportError:
            self.skipTest('Install the data extra for media checks.')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, value in [('train', 0), ('validation', 1)]:
                image = root / f'{split}.png'
                Image.new('RGB', (8, 8), (value * 255, 0, 0)).save(image)
                audio = root / f'{split}.wav'
                with wave.open(str(audio), 'wb') as stream:
                    stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                    stream.writeframes(bytes([value, 0]) * 160)
                D.jsonl(root / f'image_{split}.jsonl', [{'image': image.name}])
                for language in ['en_us', 'hi_in']:
                    D.jsonl(root / f'speech_{language}_{split}.jsonl', [{'audio': audio.name}])
                D.jsonl(root / f'decide_{split}.jsonl', [{'question': split}])
            config = {'license_url': 'fixture', 'sources': {}}
            D.write_lock(root, config, b'{}', 1, 0)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(sum(D.check(root).values()), 8)
            (root / 'train.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'modified'):
                D.check(root)
            Image.new('RGB', (8, 8)).save(root / 'train.png')
            D.jsonl(root / 'decide_validation.jsonl', [{'question': ' TRAIN '}])
            D.write_lock(root, config, b'{}', 1, 0)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                D.check(root)

    def test_speech_manifest_resolves_beside_it(self):
        from anulm.multimodal.speech_encoder import load_manifest
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'clip.wav').write_bytes(b'path fixture only')
            D.jsonl(root / 'speech.jsonl', [{'audio': 'clip.wav', 'text': 'fixture'}])
            self.assertEqual(load_manifest(str(root / 'speech.jsonl'))[0]['audio'], str((root / 'clip.wav').resolve()))

    def test_speech_manifest_keeps_legacy_repo_relative_paths(self):
        from anulm.multimodal.speech_encoder import load_manifest
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root = Path(directory)
            audio = root / 'clip.wav'
            audio.write_bytes(b'path fixture only')
            D.jsonl(root / 'speech.jsonl', [{'audio': str(audio), 'text': 'fixture'}])
            self.assertEqual(Path(load_manifest(str(root / 'speech.jsonl'))[0]['audio']).resolve(), audio.resolve())


if __name__ == '__main__':
    unittest.main()
