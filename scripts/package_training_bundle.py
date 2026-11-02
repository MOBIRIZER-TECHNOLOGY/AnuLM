"""Package current tracked/unignored source, or verify a produced ZIP."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.git', '.venv', 'venv', '__pycache__', '.cache', 'data',
            'runs', 'logs', 'release', '.env', '.aws', '.codex', '.agents'}


def verify(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)), 'Duplicate archive entries'
        manifest_name = 'AnuLM/BUNDLE_MANIFEST.json'
        record = json.loads(archive.read(manifest_name))
        expected = {'AnuLM/' + name for name in record['files']} | {manifest_name}
        assert set(names) == expected, 'Archive contents differ from manifest'
        for name, info in record['files'].items():
            member = PurePosixPath(name)
            assert not member.is_absolute() and '..' not in member.parts and ':' not in name
            value = archive.read('AnuLM/' + name)
            assert len(value) == info['bytes']
            assert hashlib.sha256(value).hexdigest() == info['sha256'], name
    return record


def build(output):
    output = output.resolve()
    assert not output.exists(), 'Choose a new ZIP filename; existing packages are not overwritten.'
    paths = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT
    ).decode('utf-8').split('\0')
    record = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'base_git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
              'includes_local_changes': True,
              'note': 'Source snapshot; no corpora, checkpoints, environments or live job state.',
              'files': {}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(set(paths)):
            if not name:
                continue
            relative = PurePosixPath(name)
            source = ROOT / name
            if (relative.parts[0] in EXCLUDED or any(part in {'__pycache__', '.git', '.venv'} for part in relative.parts)
                    or source.suffix.lower() in {'.pyc', '.pyo', '.pt', '.pth', '.safetensors', '.zip', '.pem', '.key'}
                    or source.name.startswith('.env') or not source.is_file()):
                continue
            if source.is_symlink():
                raise ValueError(f'Unexpected symlink: {name}')
            value = source.read_bytes()
            record['files'][name] = {'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}
            archive.writestr('AnuLM/' + name, value)
        archive.writestr('AnuLM/BUNDLE_MANIFEST.json', json.dumps(record, indent=2).encode())
    verified = verify(output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + '.sha256').write_text(digest + '  ' + output.name + '\n', encoding='utf-8')
    print(json.dumps({'zip': str(output), 'files': len(verified['files']),
                      'bytes': output.stat().st_size, 'sha256': digest, 'verified': True}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output', type=Path)
    group.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        result = verify(args.verify)
        print(f"Verified {len(result['files'])} source files.")
    else:
        build(args.output)
