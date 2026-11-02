"""Download bounded, pinned real-data subsets; no training or model downloads.

python -m anulm.data.setup_multimodal fetch
python -m anulm.data.setup_multimodal check
Install the data extra for Pillow and SoundFile media validation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time
import urllib.error
import urllib.request

from anulm.paths import PROJECT_ROOT

DEFAULT_OUT = PROJECT_ROOT / 'data' / 'commercial-smoke'
CONFIG = PROJECT_ROOT / 'config' / 'dataset-sources.json'
MIB = 1024 * 1024


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(MIB), b''):
            value.update(chunk)
    return value.hexdigest()


def request(url):
    for attempt in range(3):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={
                'User-Agent': 'AnuLM-reproducible-dataset-setup/1'}), timeout=30)
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError(f'Dataset source returned HTTP {error.code}: {url}') from error
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise RuntimeError(f'Dataset connection failed: {url}')
        time.sleep(attempt + 1)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.part')
    temporary.write_bytes(data)
    temporary.replace(path)


def download(url, path, limit=16 * MIB):
    with request(url) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f'Metadata exceeded the bounded download limit: {url}')
    save(path, data)
    return data


class BoundedReader:
    def __init__(self, stream, limit):
        self.stream, self.limit, self.bytes_read = stream, limit, 0

    def read(self, size=-1):
        remaining = self.limit - self.bytes_read
        if remaining <= 0:
            raise ValueError('Archive prefix budget exhausted; rerun with a larger --max-stream-mib.')
        data = self.stream.read(min(size if size >= 0 else remaining, remaining))
        self.bytes_read += len(data)
        return data


def archive_subset(url, select, limit):
    """Read a bounded archive prefix; write only selected regular file bytes.

    Never extract member paths, links or executable code. The caller chooses
    fixed output locations. Stopping early intentionally avoids a full archive.
    """
    with request(url) as response:
        bounded = BoundedReader(response, limit)
        with tarfile.open(fileobj=bounded, mode='r|gz') as archive:
            for member in archive:
                if not member.isfile() or member.size > 8 * MIB:
                    continue
                stream = archive.extractfile(member)
                if stream is not None and select(Path(member.name).name, stream):
                    break
        return bounded.bytes_read


def jsonl(path, rows):
    save(path, ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode('utf-8'))


def images(out, config, count, budget):
    source = config['sources']['docci']
    data = download(source['descriptions'], out / 'raw' / 'docci-descriptions.jsonl')
    annotations = {r['image_file']: r for r in map(json.loads, data.splitlines())}
    rows = {'train': [], 'validation': []}
    def select(name, stream):
        annotation = annotations.get(name)
        if annotation and annotation['split'] in ('train', 'test'):
            split = 'train' if annotation['split'] == 'train' else 'validation'
            if len(rows[split]) < count:
                relative = f'media/docci/{name}'
                save(out / relative, stream.read())
                rows[split].append({'image': relative, 'text': annotation['description'],
                                    'source': 'docci', 'source_split': annotation['split'],
                                    'source_id': annotation['example_id'], 'license': config['license']})
        return all(len(part) >= count for part in rows.values())
    transferred = archive_subset(source['images'], select, budget)
    for split, part in rows.items():
        if len(part) != count:
            raise ValueError('DOCCI archive did not supply the requested sample count.')
        jsonl(out / f'image_{split}.jsonl', part)
    return transferred


def speech(out, config, language, count, budget):
    import soundfile as sf
    base = config['sources']['fleurs']['base'] + f'data/{language}/'
    transferred = 0
    for original, split in [('train', 'train'), ('dev', 'validation')]:
        data = download(base + original + '.tsv', out / 'raw' / f'fleurs-{language}-{original}.tsv')
        annotations = {r[1]: r for r in csv.reader(io.StringIO(data.decode('utf-8')), delimiter='\t') if len(r) >= 4}
        rows = []
        def select(name, stream):
            annotation = annotations.get(name)
            if annotation and name.endswith('.wav'):
                payload = stream.read()
                audio = sf.info(io.BytesIO(payload))
                duration = audio.duration
                if audio.channels != 1 or audio.samplerate != 16000 or not 0 < duration <= 30:
                    return False
                relative = f'media/fleurs/{language}/{original}/{name}'
                save(out / relative, payload)
                rows.append({'audio': relative, 'text': annotation[3], 'source': 'fleurs',
                             'lang': language, 'source_split': original, 'source_id': annotation[0],
                             'seconds': duration, 'license': config['license']})
            return len(rows) >= count
        transferred += archive_subset(base + 'audio/' + original + '.tar.gz', select, budget)
        if len(rows) != count:
            raise ValueError(f'FLEURS {language}/{original} has too few usable clips.')
        jsonl(out / f'speech_{language}_{split}.jsonl', rows)
    return transferred


def question_key(text):
    return ' '.join(text.casefold().split())


def decisions(out, config, count):
    base = config['sources']['banking77']['base']
    download(base + 'LICENSE', out / 'licenses' / 'BANKING77-LICENSE.txt')
    categories = json.loads(download(base + 'banking_data/categories.json', out / 'banking77' / 'categories.json'))
    parts = {}
    seen = set()
    # Hold out official test examples first, then exclude equal normalized
    # questions from training. No comparison-model data or weights are used.
    for original, split in [('test', 'validation'), ('train', 'train')]:
        data = download(base + f'banking_data/{original}.csv', out / 'raw' / f'banking77-{original}.csv')
        rows = []
        for item in csv.DictReader(io.StringIO(data.decode('utf-8'))):
            key = question_key(item['text'])
            if key in seen:
                continue
            seen.add(key)
            rows.append({'question': item['text'], 'answer': item['category'], 'lang': 'en',
                         'source': 'banking77', 'source_split': original, 'license': config['license']})
            if len(rows) >= count:
                break
        if len(rows) != count or any(row['answer'] not in categories for row in rows):
            raise ValueError('BANKING77 labels or sample count do not match the pinned source.')
        parts[split] = rows
        jsonl(out / f'decide_{split}.jsonl', rows)
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=['text', 'category'])
        writer.writeheader()
        writer.writerows({'text': row['question'], 'category': row['answer']} for row in rows)
        save(out / 'banking77' / (original + '.csv'), buffer.getvalue().encode('utf-8'))
    return parts


def write_lock(out, config, config_bytes, count, transferred):
    attribution = ['# Dataset attribution', '', 'All selected sources: CC BY 4.0.', config['license_url'],
                   '', 'Changes: bounded subsets, normalized BANKING77 question deduplication;',
                   'FLEURS clips over 30 seconds excluded; original captions/transcripts retained.', '']
    for name, source in config['sources'].items():
        attribution += [f"- {name}: {source['creator']}; {source['page']}"]
    save(out / 'ATTRIBUTION.md', '\n'.join(attribution).encode('utf-8'))
    save(out / 'SOURCE_CONFIG.json', config_bytes)
    files = {str(path.relative_to(out)).replace('\\', '/'): {
        'sha256': file_sha(path), 'bytes': path.stat().st_size}
        for path in out.rglob('*') if path.is_file() and path.name not in ('DATA_LOCK.json', 'READY.json', 'FETCHING.json')}
    save(out / 'DATA_LOCK.json', json.dumps({'version': 1, 'source_config_sha256': sha(config_bytes),
         'per_split': count, 'archive_prefix_bytes_read': transferred, 'files': files}, indent=2).encode())


def check(out):
    lock = json.loads((out / 'DATA_LOCK.json').read_text(encoding='utf-8'))
    from PIL import Image
    import soundfile as sf
    for name, info in lock['files'].items():
        path = (out / name).resolve()
        if not path.is_relative_to(out.resolve()) or not path.is_file() or file_sha(path) != info['sha256']:
            raise ValueError(f'Missing or modified data: {name}')
    counts = {}
    for group, key in [('image', 'image'), ('speech_en_us', 'audio'), ('speech_hi_in', 'audio'), ('decide', 'question')]:
        partitions = {}
        for split in ['train', 'validation']:
            path = out / f'{group}_{split}.jsonl'
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            if len(rows) != lock['per_split']:
                raise ValueError(f'Unexpected sample count: {path.name}')
            partitions[split] = set()
            for row in rows:
                if key == 'question':
                    identity = question_key(row[key])
                else:
                    media = out / row[key]
                    identity = file_sha(media)
                    if key == 'image':
                        with Image.open(media) as image:
                            image.verify()
                    else:
                        audio = sf.info(media)
                        if audio.samplerate != 16000 or audio.channels != 1 or not 0 < audio.duration <= 30:
                            raise ValueError(f'Invalid audio: {media.name}')
                partitions[split].add(identity)
            counts[path.name] = len(rows)
        if partitions['train'] & partitions['validation']:
            raise ValueError(f'Train/validation content overlap: {group}')
    print(json.dumps({'status': 'data checks passed', 'counts': counts}, indent=2))
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['fetch', 'check'])
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--per-split', type=int, default=8)
    parser.add_argument('--max-stream-mib', type=int, default=64,
                        help='maximum compressed bytes per streamed media archive')
    args = parser.parse_args()
    if not 1 <= args.per_split <= 100 or not 1 <= args.max_stream_mib <= 1024:
        parser.error('per-split must be 1..100; max-stream-mib must be 1..1024')
    out = args.out.resolve()
    if args.action == 'check':
        check(out)
        return
    config_bytes = CONFIG.read_bytes()
    config = json.loads(config_bytes)
    # Preflight optional libraries before starting any download.
    import PIL.Image  # noqa: F401
    import soundfile  # noqa: F401
    if (out / 'READY.json').exists():
        lock = json.loads((out / 'DATA_LOCK.json').read_text(encoding='utf-8'))
        if lock['source_config_sha256'] != sha(config_bytes) or lock['per_split'] != args.per_split:
            raise ValueError('Existing pack has different sources/counts. Choose a new --out directory.')
        check(out)
        print('Existing checked pack reused; no downloads.')
        return
    expected = {'source_config_sha256': sha(config_bytes), 'per_split': args.per_split}
    marker = out / 'FETCHING.json'
    if out.exists() and any(out.iterdir()):
        if not marker.exists() or json.loads(marker.read_text(encoding='utf-8')) != expected:
            raise ValueError('Output directory belongs to another setup. Choose a new --out.')
        print('Retrying the interrupted bounded download; incomplete data is not trusted.', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    save(marker, json.dumps(expected).encode())
    print('Fetching license and pinned image annotations/images...', flush=True)
    download('https://creativecommons.org/licenses/by/4.0/legalcode.txt', out / 'licenses' / 'CC-BY-4.0.txt')
    download(config['sources']['docci']['page'], out / 'licenses' / 'DOCCI-source.html')
    download(config['sources']['fleurs']['base'] + 'README.md', out / 'licenses' / 'FLEURS-card.md')
    transferred = images(out, config, args.per_split, args.max_stream_mib * MIB)
    for language in config['sources']['fleurs']['languages']:
        print(f'Fetching bounded FLEURS {language} train/dev clips...', flush=True)
        transferred += speech(out, config, language, args.per_split, args.max_stream_mib * MIB)
    print('Fetching BANKING77 labels and examples...', flush=True)
    decisions(out, config, args.per_split)
    write_lock(out, config, config_bytes, args.per_split, transferred)
    counts = check(out)
    save(out / 'READY.json', json.dumps({'status': 'data checks passed', 'counts': counts}, indent=2).encode())
    marker.unlink(missing_ok=True)
    print(f'Ready: {out}. No model training or demo approval was performed.')


if __name__ == '__main__':
    main()
