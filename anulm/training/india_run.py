"""One local, logged India-focused research run using the existing trainers.

python -m anulm.training.india_run --download-pid PID
Run with the same interpreter/environment used for setup. Does not publish.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from anulm.paths import PROJECT_ROOT
from anulm.data.setup_multimodal import save, jsonl

ROOT = PROJECT_ROOT
DATA = ROOT / 'data/india-v1'
RUN = ROOT / 'runs/india-v1'
LOGS = ROOT / 'logs/india-v1'
CONFIG = ROOT / 'config/india-training.json'
IST = timezone(timedelta(hours=5, minutes=30))


def status(stage, **details):
    LOGS.mkdir(parents=True, exist_ok=True)
    record = {'stage': stage, 'at_ist': datetime.now(IST).isoformat(), **details}
    save(LOGS / 'status.json', json.dumps(record, indent=2).encode())
    print(json.dumps(record), flush=True)


def active(pid):
    if os.name == 'nt':
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def command(stage, arguments, timeout_hours=12):
    status(stage, log=str(LOGS / f'{stage}.log'))
    with (LOGS / f'{stage}.log').open('a', encoding='utf-8') as stream:
        process = subprocess.Popen([sys.executable, '-u', *arguments], cwd=ROOT,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=os.name != 'nt',
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'GRADIO_ANALYTICS_ENABLED': 'False'})
        child_record = LOGS / 'active-child.json'
        save(child_record, json.dumps({'pid': process.pid, 'stage': stage, 'arguments': arguments}).encode())
        try:
            returncode = process.wait(timeout=timeout_hours * 3600)
        except subprocess.TimeoutExpired:
            # A Windows virtualenv launcher can have a second Python child.
            # Terminate this owned process tree, never unrelated Python jobs.
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
            else:
                import signal
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
            raise RuntimeError(f'{stage} exceeded its wall-clock budget; its process tree was stopped.')
        finally:
            child_record.unlink(missing_ok=True)
    if returncode:
        raise RuntimeError(f'{stage} failed; see {LOGS / (stage + ".log")}')


def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def source_signature():
    from anulm.data.setup_multimodal import file_sha
    paths = list((ROOT / 'anulm').rglob('*.py')) + [CONFIG, ROOT / 'config/dataset-sources.json']
    return {str(path.relative_to(ROOT)): file_sha(path) for path in sorted(paths)}


def tokenize(config):
    import numpy as np
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    directory = DATA / 'tokens'
    directory.mkdir(exist_ok=True)
    tokenizer_path = directory / 'tokenizer.json'
    languages = list(config['languages'].values())
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    def training_text():
        iterators = [iter(rows(DATA / 'text' / f'{lang}_train.jsonl')) for lang in languages]
        # Round robin across available languages; no validation text is fitted.
        while iterators:
            remaining = []
            for iterator in iterators:
                try:
                    yield next(iterator)['text']
                    remaining.append(iterator)
                except StopIteration:
                    pass
            iterators = remaining
    status('tokenizer_fit', languages=languages, train_only=True)
    tokenizer.train_from_iterator(training_text(), trainer=trainers.BpeTrainer(
        vocab_size=config['text_vocab'], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=['<|endoftext|>'], show_progress=False))
    tokenizer.save(str(tokenizer_path))
    eos = tokenizer.token_to_id('<|endoftext|>')
    counts = {}
    total_bytes = total_tokens = 0
    for split in ['train', 'validation']:
        with (directory / (split + '.u16')).open('wb') as stream:
            count = 0
            for language in languages:
                for record in rows(DATA / 'text' / f'{language}_{split}.jsonl'):
                    ids = tokenizer.encode(record['text']).ids + [eos]
                    stream.write(np.asarray(ids, dtype=np.uint16).tobytes())
                    count += len(ids)
                    if split == 'train':
                        total_bytes += len(record['text'].encode('utf-8'))
                        total_tokens += len(ids)
            counts[split] = count
    if min(counts.values()) < 4096:
        raise ValueError('Too little tokenized text for train/validation.')
    metadata = {'train': 'train.u16', 'val': 'validation.u16', 'vocab_size': tokenizer.get_vocab_size(),
                'tokenizer': str(tokenizer_path.resolve()), 'bytes_per_token': total_bytes / total_tokens,
                'sources': config['languages'], 'counts': counts}
    save(directory / 'meta.json', json.dumps(metadata, indent=2).encode())
    # Reserve the full 30-second Whisper window (1500 states / stack 4),
    # even though eligible clips are shorter. Predeclare this resource filter
    # for every split; raw data remains intact for later larger-context runs.
    filters = {}
    for language in config['languages']:
        for split in ['train', 'validation', 'test']:
            path = DATA / f'asr_{language}_{split}.jsonl'
            original = list(rows(path))
            eligible = [row for row in original if len(tokenizer.encode(' ' + row['text']).ids) + 1 <= config['context'] - 375 - 8]
            if not eligible:
                raise ValueError(f'No speech rows fit context: {language}/{split}')
            jsonl(path, eligible)
            filters[f'{language}/{split}'] = {'accepted': len(eligible), 'dropped_for_context': len(original) - len(eligible)}
    save(DATA / 'TOKEN_FILTERS.json', json.dumps(filters, indent=2).encode())
    # Reading supervision is an explicitly derived title-identification task,
    # not synthetic factual Q&A or a claim of conversational alignment.
    for split in ['train', 'validation']:
        with (DATA / f'read_{split}.jsonl').open('w', encoding='utf-8') as stream:
            for language in languages:
                for index, record in enumerate(rows(DATA / 'text' / f'{language}_{split}.jsonl')):
                    if index >= (3000 if split == 'train' else 200):
                        break
                    prompt = ('इस अनुच्छेद का विषय क्या है?\n' if language == 'hi' else
                              'What is the topic of this passage?\n') + record['text'][:500]
                    item = {'question': prompt, 'answer': record['title'], 'lang': 'hi' if language == 'hi' else 'en',
                            'source_url': record['url'], 'license': record['license']}
                    stream.write(json.dumps(item, ensure_ascii=False) + '\n')
    return metadata


def download_encoders(config):
    from anulm.data.india_corpus import download
    for key, files in [('whisper', ['config.json', 'model.bin', 'tokenizer.json', 'vocabulary.txt', 'README.md']),
                       ('vision', ['config.json', 'model.safetensors', 'README.md'])]:
        source = config[key]
        path = RUN / 'encoders' / key
        for name in files:
            url = f'https://huggingface.co/{source["repo"]}/resolve/{source["revision"]}/{name}'
            if name in ('model.bin', 'model.safetensors'):
                download_weight(url, path / name)
            else:
                download(url, path / name)
        save(RUN / (key + '-snapshot.json'), json.dumps({'path': str(path.resolve()), **source}, indent=2).encode())


def download_weight(url, destination):
    """Bounded HTTP ranges avoid a stalled unbounded model transfer."""
    import urllib.request
    import re
    from anulm.data.setup_multimodal import file_sha
    destination.parent.mkdir(parents=True, exist_ok=True)
    provenance = destination.with_name(destination.name + '.source.json')
    if destination.exists() and provenance.exists():
        record = json.loads(provenance.read_text())
        if record['url'] == url and record['sha256'] == file_sha(destination):
            return
        raise ValueError('Changed local encoder weights: ' + str(destination))
    part = destination.with_name(destination.name + '.part')
    while True:
        offset = part.stat().st_size if part.exists() else 0
        end = offset + 8 * 1024**2 - 1
        for attempt in range(4):
            try:
                request = urllib.request.Request(url + f'?download=true&anulm_range={offset}',
                    headers={'Range': f'bytes={offset}-{end}', 'User-Agent': 'AnuLM-India-v1'})
                with urllib.request.urlopen(request, timeout=60) as response:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if response.status != 206 or match is None or int(match[1]) != offset:
                        raise ValueError('Encoder server did not honor the bounded range.')
                    payload = response.read(8 * 1024**2 + 1)
                    if len(payload) != int(match[2]) - offset + 1 or len(payload) > 8 * 1024**2:
                        raise ValueError('Incomplete encoder range.')
                    with part.open('ab') as stream:
                        stream.write(payload)
                    total = int(match[3])
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        print(f'{destination.name}: {part.stat().st_size / 1024**2:.1f}/{total / 1024**2:.1f} MiB', flush=True)
        if part.stat().st_size == total:
            part.replace(destination)
            save(provenance, json.dumps({'url': url, 'sha256': file_sha(destination), 'bytes': total}, indent=2).encode())
            return


def configure_hub():
    # Use the operating system's trusted certificates, matching urllib's
    # verified downloads. Never disable certificate verification.
    import ssl
    import httpx
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    from huggingface_hub import set_client_factory
    set_client_factory(lambda: httpx.Client(verify=ssl.create_default_context(), follow_redirects=True))


def preflight(config, checkpoint, tokenizer, out_dir, sample_data=None):
    """Real encoders + native decoder: one update for every task kind/language."""
    import math
    from types import SimpleNamespace
    import torch
    from anulm import model as M
    from anulm.multimodal.speech_vocab import SpeechVocab
    from anulm.multimodal.omni_train import Omni
    configure_hub()
    torch.manual_seed(config['seed'])
    out_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; unattended GPU training was not launched.')
    cfg = M.AnuLMConfig.nano_30b(vocab_size=SpeechVocab(config['text_vocab']).total,
        tokenizer_path=str(tokenizer), block_size=config['context'], num_experts=8)
    native = M.AnuLM(cfg)
    from anulm.training.train import atomic_save
    atomic_save({'model': native.state_dict(), 'cfg': cfg, 'speech_text_vocab': config['text_vocab']}, checkpoint)
    del native
    tasks, held = [], []
    if sample_data is None:
        specs = [('read', 'text', 'read'), ('caption', 'image', 'caption'),
                 ('india_qa', 'text', 'india_qa'), ('india_vqa', 'image', 'india_vqa')]
        specs += [('asr_' + lang, 'asr', 'asr_' + lang) for lang in config['languages']]
        base = DATA
    else:
        base = sample_data
        specs = [('read', 'text', 'decide'), ('caption', 'image', 'image'),
                 ('asr_en', 'asr', 'speech_en_us'), ('asr_hi', 'asr', 'speech_hi_in')]
    for name, kind, prefix in specs:
        for split in ['train', 'validation']:
            examples = list(rows(base / f'{prefix}_{split}.jsonl'))[:2]
            if kind == 'image':
                for item in examples:
                    item['image'] = str((base / item['image']).resolve())
            if kind == 'asr':
                for item in examples:
                    item['audio'] = str((base / item['audio']).resolve())
            jsonl(out_dir / f'{name}_{split}.jsonl', examples)
        tasks.append(f'{name}={kind}:{out_dir / (name + "_train.jsonl")}::1')
        held.append(f'{name}={out_dir / (name + "_validation.jsonl")}')
    whisper = json.loads((RUN / 'whisper-snapshot.json').read_text())['path']
    tower = json.loads((RUN / 'vision-snapshot.json').read_text())['path']
    args = SimpleNamespace(device='cuda', ckpt=str(checkpoint), grad_ckpt=True, task=tasks, held=held,
        whisper=whisper, encoder_device='cpu', stack=4, tower=tower, pool=2,
        audio_proj_from=None, vision_proj_from=None, seed=config['seed'], eval_n=1)
    omni = Omni(args)
    parameters = list(omni.model.parameters()) + list(omni.sm.proj.parameters()) + list(omni.vl.proj.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-5)
    losses = {}
    for task in omni.tasks:
        optimizer.zero_grad(set_to_none=True)
        embeds, targets = omni.next_batch(task)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, loss = omni.model(None, targets, embeds=embeds)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite real-data preflight loss: ' + task.name)
        loss.backward()
        if any(not torch.isfinite(p.grad).all() for p in parameters if p.grad is not None):
            raise ValueError('Non-finite real-data preflight gradient: ' + task.name)
        optimizer.step()
        losses[task.name] = loss.item()
        status('preflight_task_passed', task=task.name, loss=loss.item())
    result = {'status': 'passed', 'losses': losses, 'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
              'gpu_bytes': torch.cuda.get_device_properties(0).total_memory,
              'claims': 'Real encoder/projector/decoder forward-backward updates only; not trained quality.'}
    save(out_dir / 'RESULT.json', json.dumps(result, indent=2).encode())
    return result


def evaluation_plan(config):
    """Validation plan only; final test sets remain untouched by the trainer."""
    tasks = []
    for language in config['languages']:
        tasks.append({'name': 'asr_' + language, 'kind': 'asr', 'language': language,
            'split': 'validation', 'path': str(DATA / f'asr_{language}_validation.jsonl'),
            'source': 'google/fleurs pinned in config/dataset-sources.json', 'license': 'CC-BY-4.0',
            'limit': 20, 'criteria': []})
    for name, kind, language, source, license_ in [
            ('caption', 'caption', 'en', 'DOCCI pinned in config/dataset-sources.json', 'CC-BY-4.0'),
            ('read', 'text', 'multilingual', config['wikipedia']['page'], config['wikipedia']['license']),
            ('india_caption', 'caption', 'hi/en synthetic', 'procedural price cards', 'project-generated synthetic'),
            ('india_vqa', 'vqa', 'hi/en synthetic', 'procedural price cards', 'project-generated synthetic'),
            ('india_qa', 'text', 'hi/en synthetic', 'computed integer arithmetic', 'project-generated synthetic')]:
        tasks.append({'name': name, 'kind': kind, 'language': language, 'split': 'validation',
            'path': str(DATA / f'{name}_validation.jsonl'), 'source': source, 'license': license_,
            'limit': 20, 'criteria': []})
    plan = {'required_tasks': [t['name'] for t in tasks], 'tasks': tasks,
        'note': 'Validation diagnostics only. Set criteria before a quality decision. '
                'Prepare a separate predeclared final test plan; never tune on final test results.'}
    path = RUN / 'validation-plan.json'
    save(path, json.dumps(plan, indent=2).encode())
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-pid', type=int)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    RUN.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
    os.environ['HF_HUB_CACHE'] = str(ROOT / '.cache/huggingface/hub')
    lock = LOGS / 'RUNNING.json'
    if not args.preflight_only:
        if lock.exists():
            owner = json.loads(lock.read_text())
            if active(owner['pid']):
                raise RuntimeError('Another orchestrator already owns this run; no duplicate was started.')
            lock.unlink()
        with lock.open('x', encoding='utf-8') as stream:
            json.dump({'pid': os.getpid(), 'config_sha256': hashlib.sha256(CONFIG.read_bytes()).hexdigest()}, stream)
    if os.name == 'nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    initial_source = source_signature()
    if not args.preflight_only:
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, capture_output=True).stdout.strip()
        save(RUN / 'source-provenance.json', json.dumps({'base_revision': revision,
            'local_changes': True, 'files': initial_source}, indent=2).encode())
    try:
        status('encoder_downloads')
        download_encoders(config)
        if args.preflight_only:
            # Existing project tokenizer is used only to test the mechanics.
            # The actual run fits a new multilingual tokenizer on train data.
            preflight(config, RUN / 'preflight-init.pt', ROOT / 'data/hi16k_v4.json',
                      RUN / 'initial-preflight', ROOT / 'data/commercial-smoke')
            return
        while not (DATA / 'DOWNLOADS.json').exists():
            if args.download_pid is None or not active(args.download_pid):
                raise RuntimeError('Full dataset download did not complete. Check download-error.log; rerun fetch then this runner.')
            status('waiting_for_downloads', progress=json.loads((DATA / 'DOWNLOAD_PROGRESS.json').read_text()) if (DATA / 'DOWNLOAD_PROGRESS.json').exists() else {})
            time.sleep(30)
        prepared = DATA / 'PIPELINE_PREPARED.json'
        from anulm.data.setup_multimodal import file_sha
        signature = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
        if prepared.exists():
            recorded = json.loads(prepared.read_text())
            if recorded['config_sha256'] != signature or any(file_sha(DATA / name) != value for name, value in recorded['files'].items()):
                raise ValueError('Prepared training data/config changed; do not resume against changed tokens.')
            status('reusing_prepared_data')
        else:
            if (RUN / 'text.pt.last').exists() or (RUN / 'omni.pt.last').exists():
                raise ValueError('Resume checkpoints exist without the matching prepared-data signature.')
            command('prepare', ['-m', 'anulm.data.india_corpus', 'prepare'], 12)
            command('procedural_india_data', ['-m', 'anulm.data.india_examples'], 1)
            tokenize(config)
            files = list((DATA / 'tokens').glob('*')) + list(DATA.glob('*.jsonl'))
            save(prepared, json.dumps({'config_sha256': signature,
                'files': {str(path.relative_to(DATA)): file_sha(path) for path in files if path.is_file()}}, indent=2).encode())
        command('real_preflight', ['-m', 'anulm.training.india_run', '--training-preflight-internal'], 1)
        target = datetime.fromisoformat(config['start_not_before_ist'])
        while datetime.now(IST) < target:
            status('ready_waiting_for_tonight', starts_not_before_ist=target.isoformat())
            time.sleep(min(30, (target - datetime.now(IST)).total_seconds()))
        if source_signature() != initial_source:
            raise ValueError('Runtime source changed after launch; review and restart the run before training.')
        # State markers only follow successful subprocess completion. Training
        # resumption belongs to the original trainers' atomic .last checkpoints.
        deadline = time.monotonic() + config['max_training_hours'] * 3600
        text_args = ['-m', 'anulm.training.train', '--preset', '30b', '--data', str(DATA / 'tokens/meta.json'),
            '--steps', str(config['text_steps']), '--batch-size', '8', '--block-size', str(config['context']),
            '--grad-ckpt', '--moe-impl', 'grouped', '--cfg', 'num_experts=8',
            '--out', str(RUN / 'text.pt'), '--eval-every', '250', '--eval-windows', '32',
            '--log-every', '25', '--warmup', '500', '--device', 'cuda', '--seed', str(config['seed']), '--resume']
        command('text_training', text_args, min(20, (deadline - time.monotonic()) / 3600))
        command('grow_vocabulary', ['-m', 'anulm.multimodal.speech_vocab', '--ckpt', str(RUN / 'text.pt'),
                                   '--out', str(RUN / 'text-mm.pt')], 1)
        whisper = json.loads((RUN / 'whisper-snapshot.json').read_text())['path']
        joint = ['-m', 'anulm.multimodal.omni_train', '--ckpt', str(RUN / 'text-mm.pt'),
            '--out', str(RUN / 'omni.pt'), '--steps', str(config['joint_steps']),
            '--accum', str(config.get('joint_accum', 1)),
            '--whisper', whisper, '--encoder-device', 'cpu', '--tower',
            json.loads((RUN / 'vision-snapshot.json').read_text())['path'],
            '--grad-ckpt', '--device', 'cuda', '--eval-every', '250', '--eval-n', '8',
            '--log-every', '25', '--warmup', '500', '--seed', str(config['seed']), '--resume']
        specs = [('read', 'text', 'read', .20), ('caption', 'image', 'caption', .15),
                 ('india_caption', 'image', 'india_caption', .05),
                 ('india_vqa', 'image', 'india_vqa', .05), ('india_qa', 'text', 'india_qa', .05)]
        specs += [('asr_' + lang, 'asr', 'asr_' + lang, .5 / len(config['languages'])) for lang in config['languages']]
        for name, kind, prefix, weight in specs:
            joint += ['--task', f'{name}={kind}:{DATA / (prefix + "_train.jsonl")}:{weight}:1',
                      '--held', f'{name}={DATA / (prefix + "_validation.jsonl")}']
        command('joint_training', joint, max(.1, (deadline - time.monotonic()) / 3600))
        status('training_completed_needs_quality_evaluation', checkpoint=str(RUN / 'omni.pt'),
               validation_plan=str(evaluation_plan(config)),
               note='Not approved for demo or production; per-language/task quality evaluation is next.')
    except Exception as error:
        status('failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        if not args.preflight_only and lock.exists() and json.loads(lock.read_text())['pid'] == os.getpid():
            lock.unlink()


if __name__ == '__main__':
    # Private subprocess entry isolates preflight GPU/optimizer allocations.
    if '--training-preflight-internal' in sys.argv:
        configuration = json.loads(CONFIG.read_text(encoding='utf-8'))
        preflight(configuration, RUN / 'training-preflight-init.pt', DATA / 'tokens/tokenizer.json', RUN / 'training-preflight')
    else:
        main()
