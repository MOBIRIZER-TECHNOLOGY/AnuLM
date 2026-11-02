"""Bounded local post-training validation, export and exploratory test report.

Does not retrain, publish, choose an artifact using test scores, or approve a demo.
The training-selected best checkpoint is fixed before any evaluation begins.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anulm.data.setup_multimodal import file_sha, jsonl, save

DATA = ROOT / 'data/india-v1'
OUT = ROOT / 'runs/india-v1/post-training'
EXPORT = ROOT / 'release/AnuLM-India-research-fp32'


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').split('\n') if line.strip()]


def status(stage, **extra):
    value = {'stage': stage, 'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}
    save(OUT / 'STATUS.json', json.dumps(value, indent=2).encode())
    save(ROOT / 'logs/india-v1/status.json', json.dumps(value, indent=2).encode())
    print(json.dumps(value), flush=True)


def command(stage, args, timeout=3600):
    status(stage)
    with (OUT / f'{stage}.log').open('a', encoding='utf-8') as stream:
        process = subprocess.Popen([sys.executable, '-u', *args], cwd=ROOT, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=os.name != 'nt')
        try:
            result = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
            else:
                import signal
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
            raise RuntimeError(f'{stage} timed out; owned process tree stopped.')
        if result:
            raise RuntimeError(f'{stage} failed; see {OUT / (stage + ".log")}')


def prepare_plans(config):
    # Validation diagnostics use existing validation manifests. Final text cases
    # come from raw documents absent from BOTH prepared training and validation.
    from anulm.data.india_corpus import split_for
    import hashlib
    import pyarrow.parquet as pq
    manifests = OUT / 'manifests'
    manifests.mkdir(exist_ok=True)
    plans = {'validation': [], 'test': [], 'export': []}
    def add(split, name, kind, language, path, source, license_, limit):
        plans[split].append({'name': name, 'kind': kind, 'language': language,
            'split': 'validation' if split == 'export' else split, 'path': str(path),
            'source': source, 'license': license_, 'limit': limit, 'criteria': []})
    read_validation = rows(DATA / 'read_validation.jsonl')
    old_prompts = {r['question'] for split in ['train', 'validation'] for r in rows(DATA / f'read_{split}.jsonl')}
    for fleurs, wiki in config['languages'].items():
        for split, limit in [('validation', 20), ('test', 50), ('export', 2)]:
            actual = 'validation' if split == 'export' else split
            add(split, 'asr_' + fleurs, 'asr', fleurs, DATA / f'asr_{fleurs}_{actual}.jsonl',
                'Pinned google/fleurs; official split with recorded eligibility filters', 'CC-BY-4.0', limit)
        validation = [r for r in read_validation if urlparse(r['source_url']).hostname == wiki + '.wikipedia.org']
        if not validation:
            raise ValueError('Missing reading validation language: ' + wiki)
        path = manifests / f'read_{wiki}_validation.jsonl'
        jsonl(path, validation)
        add('validation', 'read_' + wiki, 'text', wiki, path, config['wikipedia']['page'], config['wikipedia']['license'], 10)
        add('export', 'read_' + wiki, 'text', wiki, path, config['wikipedia']['page'], config['wikipedia']['license'], 2)
        known = {r['document_sha256'] for split in ['train', 'validation']
                 for r in rows(DATA / 'text' / f'{wiki}_{split}.jsonl')}
        unseen = []
        for shard in sorted((DATA / 'raw/wiki' / ('20231101.' + wiki)).glob('*.parquet')):
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=256):
                for record in batch.to_pylist():
                    text = ' '.join(record['text'].split())[:6000]
                    if len(text) < 200:
                        continue
                    key = hashlib.sha256(text.encode()).hexdigest()
                    if key in known:
                        continue
                    known.add(key)
                    question = ('इस अनुच्छेद का विषय क्या है?\n' if wiki == 'hi' else
                                'What is the topic of this passage?\n') + text[:500]
                    if question in old_prompts:
                        continue
                    old_prompts.add(question)
                    unseen.append({'question': question, 'answer': record['title'],
                        'lang': 'hi' if wiki == 'hi' else 'en', 'document_sha256': key,
                        'source_url': record['url'], 'license': config['wikipedia']['license']})
                    if len(unseen) >= 10:
                        break
                if len(unseen) >= 10:
                    break
            if len(unseen) >= 10:
                break
        if len(unseen) < 10:
            raise ValueError('Not enough untouched raw reading documents: ' + wiki)
        path = manifests / f'read_{wiki}_test.jsonl'
        jsonl(path, unseen)
        add('test', 'read_' + wiki, 'text', wiki, path, 'Derived title task; raw documents absent from prepared splits', config['wikipedia']['license'], 10)
    for split in plans:
        original = 'validation' if split == 'export' else split
        limit = 2 if split == 'export' else 20
        add(split, 'docci_caption', 'caption', 'en', DATA / f'caption_{original}.jsonl', 'Pinned DOCCI', 'CC-BY-4.0', limit)
        for prefix, kind in [('india_caption', 'caption'), ('india_vqa', 'vqa'), ('india_qa', 'text')]:
            for language in ['en', 'hi']:
                selected = [row for row in rows(DATA / f'{prefix}_{original}.jsonl') if row.get('lang') == language]
                if not selected:
                    raise ValueError('Missing synthetic language cohort')
                path = manifests / f'{prefix}_{language}_{original}.jsonl'
                jsonl(path, selected)
                add(split, prefix + '_' + language, kind, language, path,
                    'Explicitly synthetic procedural Indian-context examples', 'Project-generated synthetic', limit)
    for name, tasks in plans.items():
        save(OUT / f'{name}-plan.json', json.dumps({'required_tasks': [t['name'] for t in tasks],
            'tasks': tasks, 'note': 'Exploratory fixed sample plan, not full benchmark or release certification. '
            'No automatic quality approval; thresholds are undeclared. Never tune against final test output.'}, indent=2).encode())


def main():
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / 'RUNNING.json'
    from anulm.training.india_run import active
    if lock.exists() and active(json.loads(lock.read_text())['pid']):
        raise RuntimeError('Post-training job already running.')
    save(lock, json.dumps({'pid': os.getpid()}).encode())
    try:
        config = json.loads((ROOT / 'config/india-training.json').read_text())
        best = ROOT / 'runs/india-v1/omni.pt'
        native = torch.load(best, map_location='cpu', weights_only=False)
        save(OUT / 'selection.json', json.dumps({'selected_checkpoint': str(best),
            'sha256': file_sha(best), 'selected_training_step': native['step'],
            'selection_rule': 'Original training minimum mean relative validation loss; fixed before test evaluation.',
            'training_completed_steps': 60000, 'post_training_script_sha256': file_sha(Path(__file__))}, indent=2).encode())
        del native
        final = ROOT / 'runs/india-v1/final-step-60000.pt'
        if not final.exists():
            ck = torch.load(str(best) + '.last', map_location='cpu', weights_only=False)
            if ck['step'] != 60000:
                raise ValueError('Training last checkpoint has not reached its target.')
            for key in ('opt', 'first', 'best', 'seen'):
                ck.pop(key, None)
            from anulm.training.train import atomic_save
            atomic_save(ck, final)
            del ck
        status('preparing_evaluation_cohorts')
        prepare_plans(config)
        for label, checkpoint, plan in [('best_validation', best, 'validation'),
                                         ('final_validation', final, 'validation')]:
            command(label, ['-m', 'anulm.evaluation.report', '--checkpoint', str(checkpoint),
                '--plan', str(OUT / f'{plan}-plan.json'), '--out', str(OUT / f'{label}.json'), '--device', 'cuda'])
        if not (EXPORT / 'config.json').exists():
            command('export', ['-m', 'anulm.inference.export_hf', str(best), str(EXPORT),
                '--kind', 'omni', '--weights-dtype', 'float32', '--bundle-encoders',
                '--license', 'other', '--name', 'AnuLM India local research export'])
        command('export_validation', ['-m', 'anulm.evaluation.report', '--checkpoint', str(EXPORT),
            '--plan', str(OUT / 'export-plan.json'), '--out', str(OUT / 'export_validation.json'), '--device', 'cuda'])
        command('best_test', ['-m', 'anulm.evaluation.report', '--checkpoint', str(best),
            '--plan', str(OUT / 'test-plan.json'), '--out', str(OUT / 'best_test.json'), '--device', 'cuda'])
        from anulm.model import load_checkpoint
        original, exported = load_checkpoint(best, 'cpu'), load_checkpoint(EXPORT, 'cpu')
        mismatches = []
        for group in ('model', 'proj', 'vision_proj'):
            left, right = original[group], exported[group]
            if set(left) != set(right):
                mismatches.append(group + ':keys')
            else:
                mismatches += [group + ':' + key for key in left if not torch.equal(left[key], right[key])]
        from anulm.tokenization.hf import load_tokenizer
        a = load_tokenizer(original['cfg'].tokenizer_path)
        b = load_tokenizer(exported['cfg'].tokenizer_path)
        for text in ['Hello भारत', 'नमस्ते दुनिया', 'தமிழ் বাংলা ಕನ್ನಡ', 'اردو ਪੰਜਾਬੀ']:
            if a.encode(text) != b.encode(text):
                mismatches.append('tokenizer:' + text)
        save(OUT / 'export-parity.json', json.dumps({'exact_state_and_tokenizer_parity': not mismatches,
            'mismatches': mismatches, 'precision': 'float32', 'bundled_encoders': True}, indent=2).encode())
        if mismatches:
            raise ValueError('Export parity failed.')
        status('post_training_completed_quality_not_approved', selected_checkpoint=str(best),
            reports=str(OUT), export=str(EXPORT), note='Exploratory metrics complete; human quality/demo and publication decisions remain.')
    except Exception as error:
        status('post_training_failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
