import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK / 'Dassl.pytorch'))
sys.path.append(str(WORK.parent / 'PromptSRC_expB'))

import numpy as np

from experiment import ROOT, file_sha256, write_json
from run_matrix import next_attempt, validate_export

CONFIGS = {
    'coop': ('CoOpE', 'configs/trainers/CoOp/vit_b16.yaml', 200),
    'dpc': ('DPCE', 'configs/trainers/SPLE/CoOp/vit_b16_ep20.yaml', 20),
}


def command(dataset, method, seed, output, source=None, group='base', split='test', evaluate=None):
    trainer, config, epochs = CONFIGS[method]
    args = [sys.executable, 'train.py', '--root', str(ROOT / 'datasets'), '--seed', str(seed),
            '--trainer', trainer, '--dataset-config-file', f'configs/datasets/{dataset}.yaml',
            '--config-file', config, '--output-dir', str(output)]
    if evaluate is not None:
        args += ['--eval-only', '--model-dir', str(evaluate), '--load-epoch', str(epochs)]
    args += ['DATASET.NUM_SHOTS', '16', 'DATASET.SUBSAMPLE_CLASSES', group,
             'TEST.SPLIT', split, 'TEST.NO_TEST', 'True', 'TRAIN.PRINT_FREQ', '50']
    if evaluate is not None:
        args += ['EXPERIMENT.EXPORT_DIR', str(evaluate / 'predictions' / f'{split}_{group}')]
    if method == 'dpc':
        assert source is not None
        args += ['SPLE.BACK_CKPT_PATH', str(source), 'SPLE.BACK_CKPT_EPOCH', '200',
                 'SPLE.PIC_LIB', str(ROOT / f'datasets/SPLE_Database/{dataset}_seed{seed}.json'),
                 'SPLE.STACK.WEIGHT', '0.5', 'SPLE.STACK.WEIGHT_FOR_NEW', '0.0']
    return args


def execute(args, log):
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
                       OUTDATED_IGNORE='1', PYTHONUNBUFFERED='1')
    log.parent.mkdir(parents=True, exist_ok=True)
    write_json(log.with_suffix('.command.json'), dict(arguments=args, cwd=str(WORK)))
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run(args, cwd=WORK, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    log.with_suffix('.exit').write_text(str(result.returncode) + '\n')
    assert result.returncode == 0, f'Command failed, see {log}'


def train_run(dataset, method, seed, recipe_sha, source=None):
    parent = ROOT / 'outputs/E' / dataset / method / f'seed{seed}'
    for previous in sorted(parent.glob('attempt*/run_status.json')):
        status = json.loads(previous.read_text())
        if status['status'] == 'complete' and status['recipe_sha256'] == recipe_sha:
            for group in ['test_base', 'test_new', 'val_base']:
                validate_export(previous.parent / 'predictions' / group)
            return previous.parent
    attempt = next_attempt(parent)
    status = dict(status='running', dataset=dataset, method=method, seed=seed,
                  run_id=str(attempt.relative_to(ROOT)), recipe_sha256=recipe_sha,
                  started_at=datetime.now().astimezone().isoformat())
    write_json(attempt / 'run_status.json', status)
    try:
        execute(command(dataset, method, seed, attempt, source), attempt / 'console.log')
        assert json.loads((attempt / 'checkpoint_check.json').read_text())['passed']
        for split, group in [('test', 'base'), ('test', 'new'), ('val', 'base')]:
            output = attempt / f'eval_{split}_{group}'
            execute(command(dataset, method, seed, output, source, group, split, attempt), output / 'console.log')
            validate_export(attempt / 'predictions' / f'{split}_{group}')
        with np.load(attempt / 'reload_probe.npz', allow_pickle=False) as probe:
            with np.load(attempt / 'predictions/val_base/predictions.npz', allow_pickle=False) as validation:
                assert np.array_equal(validation['logits'][:len(probe['logits'])], probe['logits'])
                assert validation['image_ids'][:len(probe['image_ids'])].tolist() == probe['image_ids'].tolist()
        if method == 'dpc':
            with np.load(source / 'predictions/test_new/predictions.npz', allow_pickle=False) as baseline:
                with np.load(attempt / 'predictions/test_new/predictions.npz', allow_pickle=False) as enhanced:
                    assert np.array_equal(baseline['logits'], enhanced['logits'])
                    assert np.array_equal(baseline['image_ids'], enhanced['image_ids'])
            status['new_zero_weight_matches_coop_exactly'] = True
        status.update(status='complete', finished_at=datetime.now().astimezone().isoformat(),
                      prediction_exports=3, checkpoint_reload_passed=True, cross_process_probe_exact=True)
    except Exception as error:
        status.update(status='failed', error=str(error), finished_at=datetime.now().astimezone().isoformat())
        write_json(attempt / 'run_status.json', status)
        raise
    write_json(attempt / 'run_status.json', status)
    return attempt


def main(pilot=False):
    lock = (ROOT / 'records/E_runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert json.loads((ROOT / 'records/E_preflight.json').read_text())['passed']
    assert json.loads((ROOT / 'records/E_annotation_checks.json').read_text())['passed']
    implementation = json.loads((ROOT / 'records/E_implementation_files.json').read_text())
    for name, expected in implementation['files'].items():
        assert file_sha256(WORK / name) == expected, name
    recipe = dict(implementation_sha256=file_sha256(ROOT / 'records/E_implementation_files.json'),
                  data_manifest_sha256=file_sha256(ROOT / 'records/data_manifest.json'),
                  annotations_sha256=file_sha256(ROOT / 'records/E_annotation_checks.json'),
                  configurations=CONFIGS, shots=16, context_length=16, class_specific_context=False,
                  initialization='random_normal_0.02', precision='fp16', base_weight=0.5,
                  new_weight=0.0, configured_topk=8, nonrepeat=True, learning_rate=0.002,
                  datasets=['dtd', 'oxford_flowers'], seeds=[1, 2, 3])
    recipe_sha = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    recipe['recipe_sha256'] = recipe_sha
    write_json(ROOT / 'records/E_config_manifest.json', recipe)
    progress = dict(status='running', completed=0, total=12, selected_runs={},
                    clip_references=json.loads((ROOT / 'records/B_progress.json').read_text())['clip_references'])
    started = time.perf_counter()
    combinations = [('dtd', 1)] if pilot else [(dataset, seed) for seed in [1, 2, 3]
                                               for dataset in ['dtd', 'oxford_flowers']]
    for dataset, seed in combinations:
        source = None
        for method in ['coop', 'dpc']:
            key = f'{dataset}/{method}/seed{seed}'
            progress.update(current=key, elapsed_seconds=time.perf_counter() - started)
            write_json(ROOT / 'records/E_progress.json', progress)
            print('Starting', key, flush=True)
            attempt = train_run(dataset, method, seed, recipe_sha, source)
            if method == 'coop':
                source = attempt
            progress['selected_runs'][key] = str(attempt)
            progress['completed'] = len(progress['selected_runs'])
            write_json(ROOT / 'records/E_progress.json', progress)
            print('Completed', key, flush=True)
    progress.update(status='pilot_complete' if pilot else 'training_and_exports_complete',
                    elapsed_seconds=time.perf_counter() - started,
                    completed_at=datetime.now().astimezone().isoformat())
    write_json(ROOT / 'records/E_progress.json', progress)
    if pilot:
        write_json(ROOT / 'records/E_pilot.json', dict(passed=True, **progress))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    main(parser.parse_args().pilot)
