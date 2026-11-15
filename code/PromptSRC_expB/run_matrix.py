import fcntl
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from experiment import ROOT, file_sha256, write_json
from experiment_metrics import metrics

WORK = Path(__file__).resolve().parent
CONFIGS = {
    'maple': ('MaPLe', 'configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml', 5),
    'tcpt': ('MaPLe', 'configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml', 5),
    'promptsrc': ('PromptSRC', 'configs/trainers/PromptSRC/vit_b16_c2_ep20_batch4_4+4ctx.yaml', 20),
    'clip': ('ZeroshotCLIP', 'configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml', 5),
}


def command(dataset, method, seed, output, group='base', split='test', model_dir=None, export_dir=None):
    trainer, config, epochs = CONFIGS[method]
    arguments = [sys.executable, 'train.py', '--root', str(ROOT / 'datasets'), '--seed', str(seed),
                 '--trainer', trainer, '--dataset-config-file', f'configs/datasets/{dataset}.yaml',
                 '--config-file', config, '--output-dir', str(output)]
    if model_dir is not None:
        arguments += ['--eval-only', '--model-dir', str(model_dir), '--load-epoch', str(epochs)]
    elif method == 'clip':
        arguments += ['--eval-only']
    arguments += ['DATASET.NUM_SHOTS', '16', 'DATASET.SUBSAMPLE_CLASSES', group,
                  'OPTIM.MAX_EPOCH', str(epochs), 'TRAINER.MAPLE.USE_TCPT', str(method == 'tcpt'),
                  'TEST.SPLIT', split]
    if export_dir is not None:
        arguments += ['EXPERIMENT.EXPORT_DIR', str(export_dir)]
    return arguments


def execute(arguments, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
                       OUTDATED_IGNORE='1', PYTHONUNBUFFERED='1')
    write_json(log_path.with_suffix('.command.json'), dict(arguments=arguments, cwd=str(WORK)))
    with log_path.open('w', encoding='utf-8') as stream:
        result = subprocess.run(arguments, cwd=WORK, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    log_path.with_suffix('.exit').write_text(str(result.returncode) + '\n')
    assert result.returncode == 0, f'Command failed, see {log_path}'


def validate_export(directory):
    metadata = json.loads((directory / 'metadata.json').read_text())
    assert metadata['online_offline_check']
    assert file_sha256(directory / 'predictions.npz') == metadata['predictions_sha256']
    with np.load(directory / 'predictions.npz', allow_pickle=False) as data:
        assert data['logits'].dtype == np.float32
        assert data['logits'].shape == (metadata['n'], metadata['classes'])
        assert data['classnames'].tolist() == metadata['classnames']
        assert len(set(data['image_ids'].tolist())) == metadata['n']
        result = metrics(data['logits'], data['labels'])
    for field in ['accuracy', 'ece', 'nll']:
        assert abs(result[field] - metadata['online'][field]) < 1e-8
    return metadata


def next_attempt(parent):
    numbers = [int(path.name.removeprefix('attempt')) for path in parent.glob('attempt[0-9][0-9]')]
    destination = parent / f'attempt{max(numbers, default=0) + 1:02d}'
    destination.mkdir(parents=True)
    return destination


def paired_check(dataset, seed, selected):
    baseline = Path(selected[f'{dataset}/maple/seed{seed}'])
    calibrated = Path(selected[f'{dataset}/tcpt/seed{seed}'])
    first = json.loads((baseline / 'initialization.json').read_text())
    second = json.loads((calibrated / 'initialization.json').read_text())
    assert first['parameter_sha256'] == second['parameter_sha256']
    assert first['rng_sha256'] == second['rng_sha256']
    with (baseline / 'updates.jsonl').open() as stream:
        baseline_batches = [json.loads(line) for line in itertools.islice(stream, 3)]
    with (calibrated / 'updates.jsonl').open() as stream:
        calibrated_batches = [json.loads(line) for line in itertools.islice(stream, 3)]
    for left, right in zip(baseline_batches, calibrated_batches):
        assert left['image_ids'] == right['image_ids']
        assert left['input_sha256'] == right['input_sha256']
    return dict(passed=True, initial_parameter_sha256=first['parameter_sha256'],
                initial_rng_sha256=first['rng_sha256'], first_three_batches_equal=True)


def train_run(dataset, method, seed, recipe_sha):
    parent = ROOT / 'outputs' / 'B' / dataset / method / f'seed{seed}'
    for attempt in sorted(parent.glob('attempt[0-9][0-9]')):
        status_path = attempt / 'run_status.json'
        if status_path.exists():
            previous = json.loads(status_path.read_text())
            if previous['status'] == 'complete' and previous['recipe_sha256'] == recipe_sha:
                for group in ['test_base', 'test_new', 'val_base']:
                    validate_export(attempt / 'predictions' / group)
                return attempt
            if previous['status'] == 'running':
                previous.update(status='interrupted', recovery='Started a fresh attempt after acquiring the runner lock')
                write_json(status_path, previous)
    attempt = next_attempt(parent)
    status = dict(status='running', dataset=dataset, method=method, seed=seed,
                  run_id=str(attempt.relative_to(ROOT)), recipe_sha256=recipe_sha,
                  implementation_sha256=file_sha256(ROOT / 'records' / 'implementation_files.json'),
                  started_at=datetime.now().astimezone().isoformat())
    write_json(attempt / 'run_status.json', status)
    try:
        execute(command(dataset, method, seed, attempt), attempt / 'console.log')
        assert json.loads((attempt / 'checkpoint_check.json').read_text())['passed']
        for split, group in [('test', 'base'), ('test', 'new'), ('val', 'base')]:
            export_dir = attempt / 'predictions' / f'{split}_{group}'
            eval_dir = attempt / f'eval_{split}_{group}'
            execute(command(dataset, method, seed, eval_dir, group, split,
                            model_dir=attempt, export_dir=export_dir), eval_dir / 'console.log')
            validate_export(export_dir)
        with np.load(attempt / 'reload_probe.npz', allow_pickle=False) as probe:
            with np.load(attempt / 'predictions' / 'val_base' / 'predictions.npz', allow_pickle=False) as validation:
                assert validation['image_ids'][:len(probe['image_ids'])].tolist() == probe['image_ids'].tolist()
                assert np.array_equal(validation['logits'][:len(probe['logits'])], probe['logits']), 'Cross-process reload probe mismatch'
        status.update(status='complete', finished_at=datetime.now().astimezone().isoformat(),
                      prediction_exports=3, checkpoint_reload_passed=True, cross_process_probe_exact=True)
    except Exception as error:
        status.update(status='failed', error=str(error), finished_at=datetime.now().astimezone().isoformat())
        write_json(attempt / 'run_status.json', status)
        raise
    write_json(attempt / 'run_status.json', status)
    return attempt


def main():
    lock = (ROOT / 'records' / 'B_runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert json.loads((ROOT / 'records' / 'preflight.json').read_text())['passed']
    assert json.loads((ROOT / 'records' / 'data_checks.json').read_text())['passed']
    implementation = json.loads((ROOT / 'records' / 'implementation_files.json').read_text())
    for name, expected in implementation['files'].items():
        assert file_sha256(WORK / name) == expected, f'Implementation changed: {name}'
    implementation_sha = file_sha256(ROOT / 'records' / 'implementation_files.json')
    preflight_code = json.loads((ROOT / 'records' / 'preflight_code_sha256.json').read_text())
    for name, expected in preflight_code.items():
        assert file_sha256(WORK / name) == expected, f'Code changed since preflight: {name}'
    freeze = dict(implementation_sha256=implementation_sha,
                  data_manifest_sha256=file_sha256(ROOT / 'records' / 'data_manifest.json'),
                  pretrained_sha256=file_sha256(ROOT / 'weights' / 'ViT-B-16.pt'),
                  preflight_sha256=file_sha256(ROOT / 'records' / 'preflight.json'),
                  datasets=['dtd', 'oxford_flowers'], seeds=[1, 2, 3],
                  methods=['maple', 'tcpt', 'promptsrc'], train_batch=4,
                  shots=16, precision='fp16', base_validation_shots=4,
                  calibration_bins=[10, 15, 20], temperature_grid=dict(min=0.05, max=20.0, points=401),
                  method_configs={method: dict(trainer=spec[0], path=spec[1], epochs=spec[2],
                                               sha256=file_sha256(WORK / spec[1]))
                                  for method, spec in CONFIGS.items() if method != 'clip'})
    recipe_fields = dict(freeze)
    recipe_fields.pop('preflight_sha256')
    recipe_sha = hashlib.sha256(json.dumps(recipe_fields, sort_keys=True).encode()).hexdigest()
    freeze['recipe_sha256'] = recipe_sha
    write_json(ROOT / 'records' / 'config_manifest.json', freeze)
    progress = dict(status='running', completed=0, total=18, selected_runs={}, pairing={}, clip_references={})
    started = time.perf_counter()
    for seed in [1, 2, 3]:
        for dataset in ['dtd', 'oxford_flowers']:
            for method in ['maple', 'tcpt', 'promptsrc']:
                key = f'{dataset}/{method}/seed{seed}'
                progress.update(current=key, elapsed_seconds=time.perf_counter() - started)
                write_json(ROOT / 'records' / 'B_progress.json', progress)
                print('Starting', key, flush=True)
                attempt = train_run(dataset, method, seed, recipe_sha)
                progress['selected_runs'][key] = str(attempt)
                progress['completed'] = len(progress['selected_runs'])
                if method == 'tcpt':
                    progress['pairing'][f'{dataset}/seed{seed}'] = paired_check(dataset, seed, progress['selected_runs'])
                write_json(ROOT / 'records' / 'B_progress.json', progress)
                print('Completed', key, progress['completed'], '/18', flush=True)
        if seed == 1:
            write_json(ROOT / 'records' / 'seed1_acceptance.json', dict(passed=True, **progress))
    for dataset in ['dtd', 'oxford_flowers']:
        for group in ['base', 'new']:
            directory = ROOT / 'outputs' / 'B' / 'clip_reference' / dataset / group
            execute(command(dataset, 'clip', 1, directory, group=group,
                            export_dir=directory / 'predictions'), directory / 'console.log')
            validate_export(directory / 'predictions')
            progress['clip_references'][f'{dataset}/{group}'] = str(directory / 'predictions')
    progress.update(status='training_and_exports_complete', elapsed_seconds=time.perf_counter() - started,
                    completed_at=datetime.now().astimezone().isoformat())
    write_json(ROOT / 'records' / 'B_progress.json', progress)
    print('All 18 runs, 54 prediction exports and 4 CLIP references completed.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except BlockingIOError:
        raise
    except Exception as error:
        progress_path = ROOT / 'records' / 'B_progress.json'
        if progress_path.exists():
            state = json.loads(progress_path.read_text())
            state.update(status='failed', error=str(error), failed_at=datetime.now().astimezone().isoformat())
            write_json(progress_path, state)
        raise
