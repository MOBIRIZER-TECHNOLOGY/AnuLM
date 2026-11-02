"""Immutable file identities for research reports; never grants demo approval."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from anulm.paths import PROJECT_ROOT


def file_record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    before = path.stat()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'Artifact changed while hashing: {path}')
    return {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': after.st_size}


def path_record(value):
    path = Path(value).expanduser()
    if path.is_file():
        return {'status': 'hashed', 'files': [file_record(path)]}
    if path.is_dir():
        files = [p for p in sorted(path.rglob('*')) if p.is_file()]
        if any(p.is_symlink() for p in files):
            raise ValueError('Use a local artifact directory without symlinks.')
        return {'status': 'hashed' if files else 'unresolved',
                'files': [file_record(p) for p in files]}
    return {'status': 'unresolved', 'source': str(value),
            'reason': 'Use a pinned downloaded local snapshot for reproducible evidence.'}


def runtime_record():
    # Works from a ZIP without Git installed or a repository clone.
    files = {p.relative_to(PROJECT_ROOT).as_posix(): file_record(p)['sha256']
             for p in sorted((PROJECT_ROOT / 'anulm').rglob('*.py'))}
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {'sha256': identity, 'files': files}


def stamp_result(payload):
    result = dict(payload)
    result.update(schema_version=1, generated_utc=datetime.now(timezone.utc).isoformat(),
                  approval='unreviewed', purpose='research_evaluation',
                  source=runtime_record(), model_artifacts=path_record(payload['model']),
                  dataset={'split': 'undeclared', 'language': 'undeclared',
                           'note': 'Declare provenance and held-out status before quality reporting.'})
    loaded = payload.get('loaded_artifacts', {})
    if loaded.get('evaluation_artifacts', result['model_artifacts']) != result['model_artifacts']:
        raise ValueError('Model artifacts changed since loading; result rejected.')
    for artifact in [loaded.get('evaluation_tokenizer', {}), *loaded.get('evaluation_encoders', {}).values()]:
        for record in artifact.get('files', []):
            if file_record(record['path']) != record:
                raise ValueError('Tokenizer or encoder artifact changed since loading.')
    inputs = []
    for row in payload.get('results', []):
        for field in ('audio', 'image'):
            if row.get(field):
                inputs.append(file_record(row[field]))
    result['media_artifacts'] = inputs
    inputs = [{key: row[key] for key in ('question', 'reference_answer', 'audio', 'image',
               'reference_transcript', 'task') if key in row} for row in payload.get('results', [])]
    result['inputs_sha256'] = hashlib.sha256(json.dumps(inputs,
        sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return result
