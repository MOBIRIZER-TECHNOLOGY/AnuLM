"""Evaluate declared held-out manifests; no training or publication.

python -m anulm.evaluation.report --checkpoint runs/india-v1/omni.pt --plan PLAN.json --out REPORT.json
Repeat against an exported local checkpoint directory to validate that artifact.
"""
import argparse
import json
import math
from pathlib import Path
import tempfile

from anulm.evaluation.evidence import file_record, path_record, runtime_record


def gate(metrics, criteria):
    if not criteria:
        return False
    for rule in criteria:
        value, threshold = metrics.get(rule['metric']), rule['threshold']
        if (not isinstance(value, (int, float)) or not math.isfinite(value)
                or not isinstance(threshold, (int, float)) or not math.isfinite(threshold)):
            return False
        if rule['op'] == '<=':
            passed = value <= threshold
        elif rule['op'] == '>=':
            passed = value >= threshold
        else:
            raise ValueError('Only <= and >= criteria are supported.')
        if not passed:
            return False
    return True


def load_plan(path):
    path = Path(path).resolve()
    plan = json.loads(path.read_text(encoding='utf-8'))
    tasks = plan['tasks']
    names = [t['name'] for t in tasks]
    if not tasks or len(names) != len(set(names)):
        raise ValueError('Evaluation needs nonempty uniquely named tasks.')
    if set(plan['required_tasks']) != set(names):
        raise ValueError('Declared required task coverage differs from supplied tasks.')
    for task in tasks:
        if (task['kind'] not in {'text', 'asr', 'caption', 'vqa'}
                or task['split'] not in {'validation', 'test'}
                or not all(task.get(key) for key in ('language', 'source', 'license'))):
            raise ValueError('Each task needs a held-out split, language, source, license and supported kind.')
        limit = task.get('limit', 100)
        if not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError('Task limit must be between 1 and 1000.')
        task['path'] = str((path.parent / task['path']).resolve())
        if not Path(task['path']).is_file():
            raise FileNotFoundError(task['path'])
        if not isinstance(task.get('criteria', []), list):
            raise ValueError('criteria must be a list.')
    return plan


def evaluate(checkpoint, plan_path, out, device='cpu', max_tokens=64):
    from anulm.apps import evaluate as app
    from anulm.multimodal.speech_encoder import load_manifest as speech_rows
    from anulm.multimodal.vision_encoder import load_manifest as image_rows
    plan = load_plan(plan_path)
    plan_hash = file_record(plan_path)
    original = path_record(checkpoint)
    manager = app.ModelManager()
    model = manager.load(str(checkpoint), device)
    initial_runtime = runtime_record()
    results = []
    for task in plan['tasks']:
        print(f"Evaluating {task['name']} ({task['language']}, {task['split']})", flush=True)
        path = Path(task['path'])
        identity = file_record(path)
        if task['kind'] == 'asr':
            rows = speech_rows(str(path))
        elif task['kind'] in {'caption', 'vqa'}:
            rows = image_rows(str(path))
        else:
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').split('\n') if line.strip()]
        if not rows:
            raise ValueError(f"{task['name']}: empty evaluation manifest")
        cases = []
        for row in rows[:task.get('limit', 100)]:
            media_key = 'audio' if task['kind'] == 'asr' else 'image' if task['kind'] in {'caption', 'vqa'} else None
            media = file_record(row[media_key]) if media_key else None
            if task['kind'] == 'asr':
                from anulm.evaluation.eval_speech import wer
                app.require_modality(model, 'asr')
                prediction = model.ear().hear(row['audio']).get('text', '')
                edits, words = wer(row['text'], prediction)
                case = {'audio': row['audio'], 'reference_transcript': row['text'],
                        'transcription': prediction, 'edits': edits, 'reference_words': words,
                        'wer': edits / max(words, 1)}
            elif task['kind'] in {'caption', 'vqa'}:
                from anulm.multimodal.vision_encoder import caption, answer
                app.require_modality(model, 'vision')
                vl, tower, tok = model.eye()
                if task['kind'] == 'vqa':
                    prediction = answer(vl, tower, tok, row['image'], row['question'],
                        max_tokens=min(max_tokens, app.MAX_VISION_TOKENS), device=device)
                else:
                    prediction = caption(vl, tower, tok, row['image'],
                        max_tokens=min(max_tokens, app.MAX_VISION_TOKENS), device=device)
                case = {'image': row['image'], 'question': row.get('question'),
                        'reference_answer': row['text'], 'prediction': prediction,
                        **app.text_scores(prediction, row['text'])}
            else:
                settings = app.generation_settings(max_tokens, 0, 1, 0, 1)
                output = app._generate(model, row['question'], 'question', settings, row.get('lang'))
                prediction = output['completion']
                case = {'question': row['question'], 'reference_answer': row['answer'],
                        'prediction': prediction, **app.text_scores(prediction, row['answer'])}
            if media is not None and file_record(row[media_key]) != media:
                raise ValueError('Media changed during evaluation.')
            case['media_artifacts'] = [media] if media else []
            cases.append(case)
        if file_record(path) != identity:
            raise ValueError('Manifest changed during evaluation.')
        if task['kind'] == 'asr':
            words = sum(c['reference_words'] for c in cases)
            if words <= 0:
                raise ValueError('ASR references have no words.')
            metrics = {'wer': sum(c['edits'] for c in cases) / words}
        else:
            metrics = {key: sum(c[key] for c in cases) / len(cases) for key in ('exact_match', 'f1')}
        results.append({'task': task, 'manifest': identity, 'available_rows': len(rows),
            'evaluated_rows': len(cases), 'sampling': 'first eligible rows in declared manifest order',
            'metrics': metrics, 'gate_passed': gate(metrics, task.get('criteria', [])), 'cases': cases})
        from anulm.data.setup_multimodal import save
        save(Path(out).with_suffix('.progress.json'), json.dumps({
            'completed_tasks': len(results), 'total_tasks': len(plan['tasks']),
            'last': task['name'], 'metrics': metrics, 'complete': False}, indent=2).encode())
    if path_record(checkpoint) != original or file_record(plan_path) != plan_hash or runtime_record() != initial_runtime:
        raise ValueError('Model or evaluation plan changed during evaluation.')
    report = {'schema_version': 1, 'approval': 'unreviewed', 'model_artifacts': original,
        'loaded_artifacts': app.model_metadata(manager._model),
        'plan': plan_hash, 'source': initial_runtime, 'tasks': results,
        'all_declared_gates_passed': all(r['gate_passed'] for r in results),
        'note': 'Research metrics only. Lexical caption F1 is not a caption-quality judgment. '
                'Criteria and sample coverage require review; test data must not drive repeated tuning.'}
    from anulm.data.setup_multimodal import save
    save(Path(out), json.dumps(report, ensure_ascii=False, indent=2).encode())
    save(Path(out).with_suffix('.progress.json'), json.dumps({
        'completed_tasks': len(results), 'total_tasks': len(results), 'complete': True,
        'all_declared_gates_passed': report['all_declared_gates_passed']}, indent=2).encode())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-tokens', type=int, default=64)
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.plan, args.out, args.device, args.max_tokens)
    print(json.dumps({'report': str(args.out), 'all_declared_gates_passed': result['all_declared_gates_passed']}))


if __name__ == '__main__':
    main()
