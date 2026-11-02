"""Finish local checks, preserving pre-correction reports and provenance."""
import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anulm.data.setup_multimodal import save, file_sha
from anulm.evaluation.evidence import path_record
from anulm.evaluation.report import gate

OUT = ROOT / 'runs/india-v1/post-training'


def command(name, args):
    from post_training import command as bounded_command
    return bounded_command(name, args, timeout=1800)


def complete():
    from post_training import status
    if json.loads((OUT / 'STATUS.json').read_text())['stage'] != 'post_training_completed_quality_not_approved':
        raise RuntimeError('Finish the frozen original assessment before correcting text-only output records.')
    status('text_output_correction')
    corrected = OUT / 'text-output-correction'
    corrected.mkdir(exist_ok=True)
    best = ROOT / 'runs/india-v1/omni.pt'
    final = ROOT / 'runs/india-v1/final-step-60000.pt'
    export = ROOT / 'release/AnuLM-India-research-fp32'
    for label, checkpoint in [('best_validation', best), ('final_validation', final),
                              ('export_validation', export), ('best_test', best)]:
        path = OUT / (label + '.json')
        backup = OUT / (label + '.before-eos-fix.json')
        if not backup.exists():
            save(backup, path.read_bytes())
        original = json.loads(backup.read_text(encoding='utf-8'))
        if path_record(checkpoint) != original['model_artifacts']:
            raise ValueError('Checkpoint changed since the original assessment.')
        tasks = [t['task'] for t in original['tasks'] if t['task']['kind'] == 'text']
        plan = corrected / (label + '-plan.json')
        save(plan, json.dumps({'required_tasks': [t['name'] for t in tasks], 'tasks': tasks}, indent=2).encode())
        report_path = corrected / (label + '.json')
        command(label + '_text_rerun', ['-m', 'anulm.evaluation.report', '--checkpoint', str(checkpoint),
            '--plan', str(plan), '--out', str(report_path), '--device', 'cuda'])
        rerun = json.loads(report_path.read_text(encoding='utf-8'))
        new_tasks = {t['task']['name']: t for t in rerun['tasks']}
        merged = copy.deepcopy(original)
        merged['tasks'] = []
        for task in original['tasks']:
            if task['task']['kind'] == 'text':
                entry = new_tasks[task['task']['name']]
                entry['evaluation_source_key'] = 'text_rerun'
            else:
                entry = task
                entry['evaluation_source_key'] = 'unchanged_audio_image'
            merged['tasks'].append(entry)
        merged['component_sources'] = {'text_rerun': rerun['source'], 'unchanged_audio_image': original['source']}
        merged['text_correction'] = {'original_report': str(backup), 'original_report_sha256': file_sha(backup),
            'rerun_report': str(report_path), 'rerun_report_sha256': file_sha(report_path),
            'reason': 'EOS control token was being displayed/scored as text by the HF-tokenizer serving path. '
                      'Text outputs were rerun. Unchanged audio/image measurements retain their original code provenance.'}
        merged['all_declared_gates_passed'] = all(t['gate_passed'] for t in merged['tasks'])
        save(path, json.dumps(merged, ensure_ascii=False, indent=2).encode())
    status('final_checkpoint_export_validation')
    final_export = ROOT / 'release/AnuLM-India-final-60000-fp32'
    if not (final_export / 'config.json').exists():
        command('final_export', ['-m', 'anulm.inference.export_hf', str(final), str(final_export),
            '--kind', 'omni', '--weights-dtype', 'float32', '--bundle-encoders', '--license', 'other'])
    command('final_export_validation', ['-m', 'anulm.evaluation.report', '--checkpoint', str(final_export),
        '--plan', str(OUT / 'export-plan.json'), '--out', str(OUT / 'final_export_validation.json'), '--device', 'cuda'])
    import torch
    from anulm.model import load_checkpoint
    native, exported = load_checkpoint(final, 'cpu'), load_checkpoint(final_export, 'cpu')
    differences = [group + ':' + key for group in ['model', 'proj', 'vision_proj']
        for key in native[group] if key not in exported[group] or not torch.equal(native[group][key], exported[group][key])]
    save(OUT / 'final-export-parity.json', json.dumps({'exact_parameter_parity': not differences,
        'differences': differences}, indent=2).encode())
    if differences:
        raise ValueError('Final export parameter parity failed.')
    status('release_checks_completed_quality_not_approved', reports=str(OUT),
           note='Code/export checks finished. Trained quality and human demo/publication review remain separate.')


def main():
    from anulm.training.from_scratch import owner
    from post_training import status
    with owner(OUT):
        try:
            complete()
        except Exception as error:
            status('release_checks_failed', error=f'{type(error).__name__}: {error}')
            raise


if __name__ == '__main__':
    main()
