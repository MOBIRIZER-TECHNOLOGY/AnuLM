"""Full pinned corpora for the initial India-focused AnuLM run.

Downloads are resumable; public archives are never extracted with extractall.
The run is a research experiment, not production or commercial certification.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import time
import urllib.request
import re

from anulm.paths import PROJECT_ROOT
from anulm.data.setup_multimodal import file_sha, jsonl, save

CONFIG = PROJECT_ROOT / 'config' / 'india-training.json'
SOURCE_CONFIG = PROJECT_ROOT / 'config' / 'dataset-sources.json'
OUT = PROJECT_ROOT / 'data' / 'india-v1'


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def download(url, dest, reserve_gib=40):
    """Resume only after checking the server's Content-Range start."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    meta = dest.with_name(dest.name + '.source.json')
    if dest.exists() and meta.exists():
        record = json.loads(meta.read_text(encoding='utf-8'))
        if record['url'] == url and record['sha256'] == file_sha(dest):
            return record
        raise ValueError(f'Changed existing source: {dest}')
    part = dest.with_name(dest.name + '.part')
    failures = 0
    while True:
        try:
            if shutil.disk_usage(dest.parent).free < reserve_gib * 1024**3:
                raise OSError('Free disk space reached the configured reserve; download stopped.')
            offset = part.stat().st_size if part.exists() else 0
            chunk_size = 8 * 1024**2
            headers = {'User-Agent': 'AnuLM-India-v1', 'Range': f'bytes={offset}-{offset + chunk_size - 1}'}
            separator = '&' if '?' in url else '?'
            request = urllib.request.Request(url + separator + f'anulm_offset={offset}', headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                if response.status == 206:
                    if match is None or int(match[1]) != offset:
                        raise ValueError('Server returned an invalid resume offset.')
                    payload = response.read(chunk_size + 1)
                    if len(payload) != int(match[2]) - offset + 1 or len(payload) > chunk_size:
                        raise ValueError('Incomplete bounded dataset range.')
                    total = int(match[3])
                    with part.open('ab') as stream:
                        stream.write(payload)
                elif response.status == 200 and offset == 0:
                    # Small license/card endpoints may ignore Range. Refuse
                    # an unbounded large response rather than silently stall.
                    payload = response.read(chunk_size + 1)
                    if len(payload) > chunk_size:
                        raise ValueError('Large source did not honor bounded range requests.')
                    total = len(payload)
                    save(part, payload)
                else:
                    raise ValueError('Source did not honor the requested resume range.')
            failures = 0
            if part.stat().st_size < total:
                if part.stat().st_size % (64 * 1024**2) == 0:
                    log(f'{dest.name}: {part.stat().st_size / 1024**2:.0f}/{total / 1024**2:.0f} MiB')
                continue
            if part.stat().st_size != total:
                raise ValueError('Downloaded dataset size exceeds source size.')
            part.replace(dest)
            record = {'url': url, 'sha256': file_sha(dest), 'bytes': dest.stat().st_size}
            save(meta, json.dumps(record, indent=2).encode())
            log(f'Downloaded {dest.name}: {record["bytes"] / 1024**2:.1f} MiB')
            return record
        except Exception:
            failures += 1
            if failures >= 4:
                raise
            time.sleep(2 ** (failures - 1))


def download_document(url, dest, reserve_gib=40):
    """Bounded full transfer for small license/cards; never append range bytes.

    Some document endpoints/proxies retain compressed Content-Range metadata
    while returning decoded text. Keep strict range checks for corpus archives;
    fetch these small documents independently without a Range request.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    meta = dest.with_name(dest.name + '.source.json')
    if dest.exists() and meta.exists():
        record = json.loads(meta.read_text(encoding='utf-8'))
        if record['url'] == url and record['sha256'] == file_sha(dest):
            return record
        raise ValueError(f'Changed existing source: {dest}')
    for attempt in range(4):
        try:
            if shutil.disk_usage(dest.parent).free < reserve_gib * 1024**3:
                raise OSError('Free disk space reached the configured reserve.')
            separator = '&' if '?' in url else '?'
            request = urllib.request.Request(url + separator + 'anulm_document=1',
                headers={'User-Agent': 'AnuLM-India-v1', 'Accept-Encoding': 'identity'})
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 200:
                    raise ValueError('License/card full transfer requires HTTP 200.')
                payload = response.read(8 * 1024**2 + 1)
                if not payload or len(payload) > 8 * 1024**2:
                    raise ValueError('License/card is empty or exceeds the bounded document limit.')
                length = response.headers.get('Content-Length')
                if length is not None and len(payload) != int(length):
                    raise ValueError('Incomplete license/card full transfer.')
                payload.decode('utf-8')
            save(dest, payload)
            record = {'url': url, 'sha256': file_sha(dest), 'bytes': len(payload),
                      'transfer': 'bounded_full_document'}
            save(meta, json.dumps(record, indent=2).encode())
            log(f'Downloaded document {dest.name}: {len(payload)} bytes')
            return record
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def catalogue(config, sources, out):
    fleurs = sources['sources']['fleurs']['base']
    items = []
    for language in config['languages']:
        for split in ['train', 'dev', 'test']:
            for suffix in [f'{split}.tsv', f'audio/{split}.tar.gz']:
                items.append((fleurs + f'data/{language}/' + suffix, out / 'raw' / 'fleurs' / language / suffix))
    docci = sources['sources']['docci']
    items += [(docci['descriptions'], out / 'raw' / 'docci-descriptions.jsonl'),
              (docci['images'], out / 'raw' / 'docci-images.tar.gz')]
    wiki = config['wikipedia']
    request = urllib.request.Request('https://huggingface.co/api/datasets/' + wiki['repo'] + '/revision/' + wiki['revision'],
                                     headers={'User-Agent': 'AnuLM-India-v1'})
    with urllib.request.urlopen(request, timeout=60) as response:
        listing = json.load(response)
    for item in listing['siblings']:
        name = item['rfilename']
        if name.endswith('.parquet') and any(name.startswith('20231101.' + lang + '/') for lang in config['languages'].values()):
            items.append((f'https://huggingface.co/datasets/{wiki["repo"]}/resolve/{wiki["revision"]}/{name}', out / 'raw' / 'wiki' / name))
    items += [(sources['license_url'] + 'legalcode.txt', out / 'licenses' / 'CC-BY-4.0.txt'),
              ('https://creativecommons.org/licenses/by-sa/3.0/legalcode', out / 'licenses' / 'CC-BY-SA-3.0.html'),
              (f'https://huggingface.co/datasets/{wiki["repo"]}/resolve/{wiki["revision"]}/README.md', out / 'licenses' / 'WIKIPEDIA-card.md'),
              (fleurs + 'README.md', out / 'licenses' / 'FLEURS-card.md'),
              (docci['page'], out / 'licenses' / 'DOCCI-source.html')]
    return items


def fetch(out=OUT):
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    sources = json.loads(SOURCE_CONFIG.read_text(encoding='utf-8'))
    out.mkdir(parents=True, exist_ok=True)
    save(out / 'RUN_CONFIG.json', CONFIG.read_bytes())
    save(out / 'SOURCE_CONFIG.json', SOURCE_CONFIG.read_bytes())
    items = catalogue(config, sources, out)
    log(f'Fetching {len(items)} pinned source files with four download workers.')
    records = []
    failures = []
    with ThreadPoolExecutor(max_workers=4) as workers:
        futures = {workers.submit(download_document if 'licenses' in path.relative_to(out).parts else download,
                    url, path, config['minimum_free_gib']): path for url, path in items}
        for future in as_completed(futures):
            try:
                records.append(future.result())
            except Exception as error:
                failures.append({'path': str(futures[future]), 'error': f'{type(error).__name__}: {error}'})
                log(f'Failed {futures[future]}: {error}')
            save(out / 'DOWNLOAD_PROGRESS.json', json.dumps({'complete': len(records), 'total': len(items),
                 'failed': len(failures), 'last': str(futures[future])}, indent=2).encode())
    if failures:
        save(out / 'DOWNLOAD_FAILURES.json', json.dumps(failures, indent=2).encode())
        raise RuntimeError(f'{len(failures)} source downloads failed; see {out / "DOWNLOAD_FAILURES.json"}')
    (out / 'DOWNLOAD_FAILURES.json').unlink(missing_ok=True)
    save(out / 'DOWNLOADS.json', json.dumps(records, indent=2).encode())
    log('All selected full source files downloaded and hashed.')


def extract_selected(archive_path, dest, wanted):
    """Copy named regular files only; discard archive paths and links."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, 'r:gz') as archive:
        for member in archive:
            name = Path(member.name).name
            if member.isfile() and name in wanted and 0 < member.size <= 16 * 1024**2:
                target = dest / name
                if target.exists() and target.stat().st_size == member.size:
                    continue
                stream = archive.extractfile(member)
                if stream is not None:
                    save(target, stream.read())


def split_for(key):
    return 'validation' if int(hashlib.sha256(key.encode('utf-8')).hexdigest()[:8], 16) % 100 < 2 else 'train'


def prepare(out=OUT):
    import soundfile as sf
    import pyarrow.parquet as pq
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    if not (out / 'DOWNLOADS.json').exists():
        raise ValueError('Complete the pinned download stage before preparation.')
    counts = {}
    # Captions preserve official train/test boundaries. Qualitative splits stay
    # unused; test rows are never used for training or model selection.
    annotations = [json.loads(line) for line in (out / 'raw/docci-descriptions.jsonl').read_text(encoding='utf-8').splitlines()]
    selected = [r for r in annotations if r['split'] in ('train', 'test')]
    extract_selected(out / 'raw/docci-images.tar.gz', out / 'media/docci', {r['image_file'] for r in selected})
    captions = {'train': [], 'validation': [], 'test': []}
    seen_images = set()
    for row in sorted(selected, key=lambda record: record['split'] != 'test'):
        image_path = out / 'media/docci' / row['image_file']
        identity = file_sha(image_path)
        if identity in seen_images:
            continue
        seen_images.add(identity)
        split = 'test' if row['split'] == 'test' else split_for(identity)
        captions[split].append({'image': str((out / 'media/docci' / row['image_file']).resolve()),
                               'text': ' '.join(row['description'].split()[:70]), 'source_id': row['example_id'],
                               'source': 'docci', 'license': 'CC-BY-4.0'})
    for split, rows in captions.items():
        jsonl(out / f'caption_{split}.jsonl', rows)
        counts[f'caption_{split}'] = len(rows)
    # Keep all FLEURS official splits distinct; never train from dev or test.
    # Some utterance IDs/transcripts repeat across recordings. We remove exact
    # normalized transcript overlap from training, retaining the held-out sets.
    for language in config['languages']:
        held_text = set()
        pending = {}
        for original, split in [('dev', 'validation'), ('test', 'test'), ('train', 'train')]:
            base = out / 'raw/fleurs' / language
            records = [r for r in csv.reader(io.StringIO((base / f'{original}.tsv').read_text(encoding='utf-8')), delimiter='\t') if len(r) >= 4]
            extract_selected(base / 'audio' / f'{original}.tar.gz', out / 'media/fleurs' / language / original, {r[1] for r in records})
            rows = []
            for row in records:
                path = out / 'media/fleurs' / language / original / row[1]
                if not path.is_file():
                    raise ValueError(f'Missing speech file: {path}')
                info = sf.info(path)
                text = ' '.join(row[3].split())
                key = text.casefold()
                if split == 'train' and key in held_text:
                    continue
                if info.channels != 1 or info.samplerate != 16000 or not 0 < info.duration <= 24 or not text:
                    continue
                rows.append({'audio': str(path.resolve()), 'text': text, 'lang': language,
                             'seconds': info.duration, 'source': 'fleurs', 'source_split': original,
                             'source_id': row[0], 'license': 'CC-BY-4.0'})
                if split != 'train':
                    held_text.add(key)
            if not rows:
                raise ValueError(f'No eligible speech rows: {language}/{split}')
            jsonl(out / f'asr_{language}_{split}.jsonl', rows)
            counts[f'asr_{language}_{split}'] = len(rows)
        log(f'Prepared speech: {language}')
    # Text document split is content based globally across languages, keeping
    # exact repeated content together. Train-only files feed tokenizer fitting.
    text_dir = out / 'text'
    text_dir.mkdir(exist_ok=True)
    for language in config['languages'].values():
        seen = set()
        outputs = {split: (text_dir / f'{language}_{split}.jsonl').open('w', encoding='utf-8') for split in ['train', 'validation']}
        sizes = {split: 0 for split in outputs}
        try:
            for shard in sorted((out / 'raw/wiki' / ('20231101.' + language)).glob('*.parquet')):
                parquet = pq.ParquetFile(shard)
                for batch in parquet.iter_batches(batch_size=256):
                    for row in batch.to_pylist():
                        text = ' '.join(row['text'].split())
                        if len(text) < 200:
                            continue
                        text = text[:6000]
                        key = hashlib.sha256(text.encode('utf-8')).hexdigest()
                        if key in seen:
                            continue
                        seen.add(key)
                        split = split_for(key)
                        # Bound each language's first-stage corpus at 32 MiB;
                        # complete raw corpora remain available for later runs.
                        if sizes[split] >= (32 if split == 'train' else 2) * 1024**2:
                            continue
                        record = {'text': text, 'url': row['url'], 'title': row['title'],
                                  'lang': language, 'license': 'CC-BY-SA-3.0', 'document_sha256': key}
                        outputs[split].write(json.dumps(record, ensure_ascii=False) + '\n')
                        sizes[split] += len(text.encode('utf-8'))
            counts[f'text_{language}_bytes'] = sizes
        finally:
            for stream in outputs.values():
                stream.close()
        log(f'Prepared text: {language}, {sizes}')
    save(out / 'PREPARED.json', json.dumps({'counts': counts, 'notes': config['notes']}, indent=2).encode())
    log('Prepared all languages and modalities. No training has run in this stage.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['fetch', 'prepare'])
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    {'fetch': fetch, 'prepare': prepare}[args.action](args.out.resolve())


if __name__ == '__main__':
    main()
