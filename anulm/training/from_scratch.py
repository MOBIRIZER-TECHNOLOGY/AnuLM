"""Custom-corpus pipeline using the native AnuLM trainers, with grouped splits."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from anulm.data.setup_multimodal import file_sha, save
from anulm.evaluation.evidence import path_record, runtime_record
from anulm.paths import PROJECT_ROOT


def split_documents(text, seed=1337):
    from anulm.tokenization.bpe import DOC_SEP
    docs = text.replace('\r\n', '\n').split(DOC_SEP) if DOC_SEP in text else re.split(r'\n\s*\n', text)
    unique = {hashlib.sha256(d.strip().encode()).hexdigest(): d.strip() for d in docs if d.strip()}
    if len(unique) < 2:
        raise ValueError('Supply at least two distinct documents separated by blank lines or DOC_SEP.')
    ordered = sorted(unique, key=lambda key: hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest())
    held = set(ordered[:max(1, len(ordered) // 10)])
    return ([unique[k] for k in ordered if k not in held], [unique[k] for k in ordered if k in held])


@contextmanager
def owner(work):
    from anulm.training.india_run import active
    lock = work / 'RUNNING.json'
    if lock.exists():
        record = json.loads(lock.read_text())
        if active(record['pid']):
            raise RuntimeError('Another process owns this run.')
        lock.unlink()
    with lock.open('x') as stream:
        json.dump({'pid': os.getpid()}, stream)
    try:
        yield
    finally:
        if lock.exists() and json.loads(lock.read_text())['pid'] == os.getpid():
            lock.unlink()


def prepare(corpus, work, vocab, seed, context):
    import numpy as np
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    train, held = split_documents(corpus.read_text(encoding='utf-8'), seed)
    tok = Tokenizer(models.BPE(unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(train, trainers.BpeTrainer(vocab_size=vocab,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=['[PAD]', '<|endoftext|>', '[UNK]']))
    tok.save(str(work / 'tokenizer.json'))
    counts = {}
    for name, docs in [('train', train), ('validation', held)]:
        ids = [token for doc in docs for token in tok.encode(doc).ids + [tok.token_to_id('<|endoftext|>')]]
        if len(ids) < context * 2 + 2:
            raise ValueError(f'{name}: too few tokens for stable training/evaluation windows.')
        np.asarray(ids, dtype=np.uint16).tofile(work / f'{name}.u16')
        counts[name] = {'documents': len(docs), 'tokens': len(ids)}
    metadata = {'train': 'train.u16', 'val': 'validation.u16', 'vocab_size': tok.get_vocab_size(),
        'tokenizer': str(work / 'tokenizer.json'), 'bytes_per_token': 1.0,
        'split': 'deduplicated document groups, deterministic 10% validation', 'counts': counts}
    save(work / 'meta.json', json.dumps(metadata, indent=2).encode())


def task_manifest(spec):
    from anulm.multimodal.omni_train import Task
    task = Task(spec)
    path = Path(task.path).expanduser().resolve()
    task.path = str(path)
    return task, path.with_suffix('.jsonl') if task.kind == 'asr' else path


def run(args):
    work = Path(args.work_dir).expanduser().resolve()
    corpus = Path(args.corpus).expanduser().resolve()
    if args.accum < 1 or args.steps < 1 or args.joint_steps < 1 or args.batch_size < 1 or args.context < 8 or args.eval_every < 1:
        raise ValueError('Steps, batches and accumulation must be positive; context must be at least 8.')
    if not math.isfinite(args.max_hours) or args.max_hours <= 0:
        raise ValueError('max-hours must be finite and positive.')
    normalized = []
    for spec in args.task:
        task, manifest = task_manifest(spec)
        if task.weight is not None and (not math.isfinite(task.weight) or task.weight <= 0):
            raise ValueError('Task weights must be finite and positive.')
        if not isinstance(task.batch, int) or task.batch < 1:
            raise ValueError('Task batches must be positive.')
        normalized.append(f'{task.name}={task.kind}:{task.path}:{task.weight if task.weight is not None else 1}:{task.batch}')
    args.task = normalized
    args.held = [name + '=' + str(Path(path).expanduser().resolve())
                 for name, path in (spec.split('=', 1) for spec in args.held)]
    if not 259 <= args.vocab <= 32768:
        raise ValueError('Vocabulary must be between 259 and 32768.')
    commands = [['-m', 'anulm.training.train', '--preset', args.preset,
        '--data', str(work / 'meta.json'), '--out', str(work / 'text.pt'),
        '--steps', str(args.steps), '--batch-size', str(args.batch_size), '--block-size', str(args.context),
        '--device', args.device, '--seed', str(args.seed), '--eval-every', str(args.eval_every),
        '--eval-windows', '2', '--warmup', '1', '--log-every', '1']]
    if args.grad_ckpt:
        commands[0].append('--grad-ckpt')
    if args.task:
        commands.append(['-m', 'anulm.multimodal.speech_vocab', '--ckpt', str(work / 'text.pt'),
                         '--out', str(work / 'multimodal-init.pt')])
        commands.append(['-m', 'anulm.multimodal.omni_train', '--ckpt', str(work / 'multimodal-init.pt'),
            '--out', str(work / 'omni.pt'), '--steps', str(args.joint_steps), '--accum', str(args.accum),
            '--device', args.device, '--encoder-device', 'cpu', '--seed', str(args.seed),
            '--eval-every', str(args.eval_every), '--eval-n', '2', '--log-every', '1', '--warmup', '1'])
        for flag in ['task', 'held']:
            for value in getattr(args, flag):
                commands[-1] += ['--' + flag, value]
        for flag in ['whisper', 'tower']:
            if getattr(args, flag):
                commands[-1] += ['--' + flag, getattr(args, flag)]
        if args.grad_ckpt:
            commands[-1].append('--grad-ckpt')
    if args.dry_run:
        print(json.dumps({'preparation': 'Document split first; tokenizer fitted on training documents only.',
                          'commands': [[sys.executable, *c] for c in commands]}, indent=2))
        return
    if not args.source or not args.license:
        raise ValueError('Record --source and --license for your corpus before training.')
    dependencies = {str(corpus): path_record(corpus)}
    names = []
    for spec in args.task:
        task, manifest = task_manifest(spec)
        names.append(task.name)
        dependencies[str(manifest)] = path_record(manifest)
        rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').split('\n') if line.strip()]
        for row in rows:
            media = row.get('image') if task.kind == 'image' else row.get('audio') if task.kind == 'asr' else None
            if media:
                candidate = (manifest.parent / media).resolve()
                if not candidate.is_file():
                    raise FileNotFoundError(candidate)
                dependencies[str(candidate)] = path_record(candidate)
        if task.kind in {'asr', 'image'}:
            encoder = args.whisper if task.kind == 'asr' else args.tower
            if not encoder or not Path(encoder).is_dir():
                raise ValueError('Use an explicit pinned local encoder directory for media tasks.')
            dependencies[encoder] = path_record(encoder)
    held = {}
    for spec in args.held:
        name, path = spec.split('=', 1)
        if name in held:
            raise ValueError('Duplicate held task.')
        held[name] = path
        dependencies[str(Path(path).resolve())] = path_record(path)
        # Include held media in the recipe identity as well.
        task = next((task_manifest(s)[0] for s in args.task if s.split('=', 1)[0] == name), None)
        if task is not None and task.kind in {'asr', 'image'}:
            for line in Path(path).read_text(encoding='utf-8').split('\n'):
                if line.strip():
                    row = json.loads(line)
                    media = (Path(path).resolve().parent / row['audio' if task.kind == 'asr' else 'image']).resolve()
                    if not media.is_file():
                        raise FileNotFoundError(media)
                    dependencies[str(media)] = path_record(media)
    if len(names) != len(set(names)) or set(names) != set(held):
        raise ValueError('Each unique task needs its own explicit --held name=manifest.')
    recipe = {k: v for k, v in vars(args).items() if k not in {'resume', 'dry_run'}}
    identity = {'recipe': recipe, 'inputs': dependencies, 'runtime': runtime_record()['sha256']}
    work.mkdir(parents=True, exist_ok=True)
    with owner(work):
        state_path = work / 'pipeline.json'
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state['identity'] != identity:
                raise ValueError('Recipe, source, step budget or inputs changed; choose a new work directory.')
        else:
            if any(p.name != 'RUNNING.json' for p in work.iterdir()):
                raise ValueError('Unrecognized existing artifacts; use a new work directory.')
            state = {'identity': identity, 'stages': {}}
            save(state_path, json.dumps(state, indent=2).encode())
        stage_outputs = [('prepare', ['tokenizer.json', 'train.u16', 'validation.u16', 'meta.json']),
                         ('text', ['text.pt'])]
        if args.task:
            stage_outputs += [('grow', ['multimodal-init.pt']), ('joint', ['omni.pt'])]
        deadline = None
        for index, (name, outputs) in enumerate(stage_outputs):
            if name in state['stages']:
                if any(file_sha(work / output) != sha for output, sha in state['stages'][name].items()):
                    raise ValueError('A completed artifact changed; choose a new work directory.')
                print(f'skip {name}: verified', flush=True)
                continue
            if name == 'prepare':
                prepare(corpus, work, args.vocab, args.seed, args.context)
            else:
                if deadline is None:
                    deadline = time.monotonic() + args.max_hours * 3600
                if runtime_record()['sha256'] != identity['runtime']:
                    raise ValueError('Runtime source changed during this run; stop and review the recipe.')
                command = commands[index - 1].copy()
                if name in {'text', 'joint'}:
                    last = work / (outputs[0] + '.last')
                    if last.exists() and not args.resume:
                        raise ValueError('Interrupted stage found; rerun the same recipe with --resume.')
                    if args.resume:
                        command.append('--resume')
                with (work / f'{name}.log').open('a', encoding='utf-8') as stream:
                    process = subprocess.Popen([sys.executable, '-u', *command], cwd=PROJECT_ROOT,
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=os.name != 'nt')
                    try:
                        result = process.wait(timeout=max(.01, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        if os.name == 'nt':
                            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
                        else:
                            import signal
                            os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=30)
                        raise RuntimeError(f'{name}: training budget exceeded; owned process tree stopped.')
                    if result:
                        raise RuntimeError(f'{name} failed: inspect {work / (name + ".log")}')
            state['stages'][name] = {output: file_sha(work / output) for output in outputs}
            save(state_path, json.dumps(state, indent=2).encode())
        print(f'Completed mechanics pipeline: {work}. Quality evaluation remains required.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', required=True)
    p.add_argument('--work-dir', required=True)
    p.add_argument('--source')
    p.add_argument('--license')
    p.add_argument('--vocab', type=int, default=16384)
    p.add_argument('--context', type=int, default=256)
    p.add_argument('--preset', choices=['30b', '105b', '350m'], default='30b')
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--joint-steps', type=int, default=2000)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--accum', type=int, default=1)
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--max-hours', type=float, default=40)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--device', default='cpu')
    p.add_argument('--task', action='append', default=[])
    p.add_argument('--held', action='append', default=[])
    p.add_argument('--whisper')
    p.add_argument('--tower')
    p.add_argument('--grad-ckpt', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    run(p.parse_args())


if __name__ == '__main__':
    main()
