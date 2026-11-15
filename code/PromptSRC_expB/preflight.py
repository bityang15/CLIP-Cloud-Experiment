import gc
import json
from types import SimpleNamespace

import numpy as np
import torch

from dassl.engine import build_trainer
from dassl.utils import set_random_seed
from train import setup_cfg
from experiment import ROOT, tensor_digest, write_json
from experiment_metrics import calibration_bins, fit_temperature, metrics


CONFIGS = {
    'maple': ('MaPLe', 'configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml'),
    'tcpt': ('MaPLe', 'configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml'),
    'promptsrc': ('PromptSRC', 'configs/trainers/PromptSRC/vit_b16_c2_ep20_batch4_4+4ctx.yaml'),
}


def configuration(dataset, method, output, group='base', export=''):
    trainer, config_file = CONFIGS[method]
    args = SimpleNamespace(root=str(ROOT / 'datasets'), output_dir=str(output), resume='', seed=1,
                           source_domains=None, target_domains=None, transforms=None,
                           trainer=trainer, backbone='', head='',
                           dataset_config_file=f'configs/datasets/{dataset}.yaml', config_file=config_file,
                           opts=['DATASET.NUM_SHOTS', '16', 'DATASET.SUBSAMPLE_CLASSES', group,
                                 'TRAINER.MAPLE.USE_TCPT', str(method == 'tcpt'),
                                 'EXPERIMENT.EXPORT_DIR', export])
    return setup_cfg(args)


def make_trainer(dataset, method, output, group='base', export=''):
    cfg = configuration(dataset, method, output, group, export)
    set_random_seed(1)
    return build_trainer(cfg)


def first_batches(trainer):
    iterator = iter(trainer.train_loader_x)
    batches = [next(iterator) for _ in range(3)]
    del iterator
    return batches


def finite_update(trainer, batch, epoch=0, last_batch=False):
    trainer.epoch = epoch
    trainer.batch_idx = 0
    trainer.num_batches = 1 if last_batch else len(trainer.train_loader_x)
    trainer.set_model_mode('train')
    summary = trainer.forward_backward(batch)
    assert all(np.isfinite(value) for value in summary.values())
    assert all(parameter.grad is None for parameter in trainer.model.parameters() if not parameter.requires_grad)
    assert all('prompt_learner' in name or 'VPT' in name for name, _ in trainer.trainable_parameters)
    assert not trainer.model.prompt_learner.ZS_image_encoder.training if trainer.method == 'promptsrc' else True
    return summary


def reload_check(trainer, dataset, method, output, epoch):
    trainer.save_model(epoch - 1, str(output))
    trainer.set_model_mode('eval')
    batch = next(iter(trainer.val_loader))
    images, _ = trainer.parse_batch_test(batch)
    with torch.no_grad():
        before = trainer.model_inference(images).float().cpu()
    replacement = make_trainer(dataset, method, output / 'reload')
    replacement.load_model(str(output), epoch=epoch)
    replacement.set_model_mode('eval')
    with torch.no_grad():
        after = replacement.model_inference(images).float().cpu()
    maximum_error = (before - after).abs().max().item()
    assert torch.equal(before, after), f'Checkpoint reload mismatch {maximum_error}'
    del replacement
    gc.collect()
    torch.cuda.empty_cache()
    return dict(passed=True, max_logit_difference=maximum_error, samples=len(before))


def regularizer_gradients(trainer, batch):
    trainer.set_model_mode('train')
    images, labels = trainer.parse_batch_train(batch)
    loss = trainer.model(images, labels)
    terms = trainer.model.loss_terms
    parameters = [parameter for _, parameter in trainer.trainable_parameters]
    result = {}
    for name in ['margin', 'moment']:
        gradients = torch.autograd.grad(terms[name], parameters, retain_graph=True, allow_unused=True)
        present = [gradient for gradient in gradients if gradient is not None]
        assert present and all(torch.isfinite(gradient).all().item() for gradient in present)
        norm = sum(gradient.float().square().sum().item() for gradient in present) ** 0.5
        assert norm > 0
        result[name] = dict(value=terms[name].item(), gradient_norm=norm, finite=True)
    assert torch.allclose(loss.float(), terms['ce'].float() + terms['margin'] + 5 * terms['moment'])
    assert not trainer.model.tcpt_reference.requires_grad
    return result


def official_baseline_check(trainer, batch):
    source_path = ROOT / 'code' / 'PromptSRC' / 'trainers' / 'maple.py'
    source = source_path.read_text().split('@TRAINER_REGISTRY.register()')[0]
    namespace = {'__name__': 'official_maple_reference'}
    exec(compile(source, str(source_path), 'exec'), namespace)
    with torch.random.fork_rng(devices=[0]):
        clip_model = namespace['load_clip_to_cpu'](trainer.cfg)
        reference = namespace['CustomCLIP'](trainer.cfg, trainer.dm.dataset.classnames, clip_model).cuda()
    reference.load_state_dict(trainer.model.state_dict(), strict=True)
    images, labels = trainer.parse_batch_train(batch)
    trainer.set_model_mode('train')
    reference.train()
    with torch.no_grad():
        current_ce = trainer.model(images, labels).float()
        reference_ce = reference(images, labels).float()
    assert torch.equal(current_ce, reference_ce)
    trainer.set_model_mode('eval')
    reference.eval()
    with torch.no_grad():
        current_logits = trainer.model(images).float()
        reference_logits = reference(images).float()
    assert torch.equal(current_logits, reference_logits)
    result = dict(ce_difference=(current_ce - reference_ce).abs().item(),
                  max_logit_difference=(current_logits - reference_logits).abs().max().item())
    del reference, clip_model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def metric_checks():
    boundary = calibration_bins(np.array([0.0, 0.1, 0.2, 1.0]), np.array([0, 1, 1, 1]), 10)
    assert boundary['counts'] == [2, 1, 0, 0, 0, 0, 0, 0, 0, 1]
    assert abs(boundary['ece'] - 42.5) < 1e-12
    extreme = metrics(np.array([[10000.0, -10000.0], [-10000.0, 10000.0]]), np.array([0, 1]))
    assert extreme['nll'] == 0.0
    uniform_fit = fit_temperature(np.zeros((4, 3)), np.array([0, 1, 2, 0]))
    assert uniform_fit['temperature'] == 1.0
    return dict(right_closed_bins=True, confidence_zero_in_first_bin=True,
                extreme_nll_stable=True, temperature_tie_prefers_one=True)


def main():
    assert json.loads((ROOT / 'records' / 'data_checks.json').read_text())['passed']
    torch.backends.cudnn.benchmark = True
    parent = ROOT / 'outputs' / 'B_preflight'
    numbers = [int(path.name.removeprefix('attempt')) for path in parent.glob('attempt[0-9][0-9]')]
    base = parent / f'attempt{max(numbers, default=0) + 1:02d}'
    base.mkdir(parents=True)
    write_json(base / 'started.json', {'started': True})
    report = dict(passed=False, preflight_attempt=str(base.relative_to(ROOT)),
                  metric_checks=metric_checks(), datasets={})
    for dataset in ['dtd', 'oxford_flowers']:
        dataset_report = {}
        baseline_output = base / dataset / 'maple'
        baseline = make_trainer(dataset, 'maple', baseline_output)
        initial_sha = baseline.initial_parameter_sha256
        initial_rng = baseline.initial_rng_sha256
        batches = first_batches(baseline)
        batch_ids = [batch['impath'] for batch in batches]
        batch_hashes = [tensor_digest([('images', batch['img'])]) for batch in batches]
        dataset_report['official_maple_equivalence'] = official_baseline_check(baseline, batches[0])
        baseline.set_model_mode('train')
        images, labels = baseline.parse_batch_train(batches[0])
        with torch.no_grad():
            baseline_initial_ce = baseline.model(images, labels).item()
        dataset_report['maple_update'] = finite_update(baseline, batches[0])
        dataset_report['maple_reload'] = reload_check(baseline, dataset, 'maple', baseline_output, 5)
        del baseline, batches, images, labels
        gc.collect()
        torch.cuda.empty_cache()

        tcpt_output = base / dataset / 'tcpt'
        tcpt = make_trainer(dataset, 'tcpt', tcpt_output)
        assert tcpt.initial_parameter_sha256 == initial_sha
        assert tcpt.initial_rng_sha256 == initial_rng
        tcpt_batches = first_batches(tcpt)
        assert [batch['impath'] for batch in tcpt_batches] == batch_ids
        assert [tensor_digest([('images', batch['img'])]) for batch in tcpt_batches] == batch_hashes
        tcpt.set_model_mode('train')
        images, labels = tcpt.parse_batch_train(tcpt_batches[0])
        tcpt.model.use_tcpt = False
        with torch.no_grad():
            tcpt_disabled_ce = tcpt.model(images, labels).item()
        assert tcpt_disabled_ce == baseline_initial_ce
        tcpt.model.use_tcpt = True
        reference_hash = tensor_digest([('reference', tcpt.model.tcpt_reference)])
        dataset_report['tcpt_regularizer_gradients'] = regularizer_gradients(tcpt, tcpt_batches[0])
        dataset_report['tcpt_update'] = finite_update(tcpt, tcpt_batches[0])
        assert tensor_digest([('reference', tcpt.model.tcpt_reference)]) == reference_hash
        dataset_report['pairing'] = dict(initial_parameter_sha256=initial_sha, initial_rng_sha256=initial_rng,
                                         first_three_batch_ids=batch_ids, first_three_input_sha256=batch_hashes,
                                         disabled_ce_difference=abs(tcpt_disabled_ce - baseline_initial_ce),
                                         fixed_reference=True, regularizer_precision='float32')
        dataset_report['tcpt_reload'] = reload_check(tcpt, dataset, 'tcpt', tcpt_output, 5)
        del tcpt, tcpt_batches, images, labels
        gc.collect()
        torch.cuda.empty_cache()

        prompt_output = base / dataset / 'promptsrc'
        prompt = make_trainer(dataset, 'promptsrc', prompt_output)
        batch = next(iter(prompt.train_loader_x))
        for epoch in range(20):
            dataset_report['promptsrc_last_update'] = finite_update(prompt, batch, epoch, last_batch=True)
        assert prompt.step_counter == 21
        assert all(torch.equal(value, prompt.previous_model_gpa[name])
                   for name, value in prompt.model.state_dict().items())
        dataset_report['gpa'] = dict(aggregation_passed=True, weights=prompt.gauss.tolist(), epochs=20,
                                     reload=reload_check(prompt, dataset, 'promptsrc', prompt_output, 20))
        del prompt, batch
        gc.collect()
        torch.cuda.empty_cache()
        report['datasets'][dataset] = dataset_report
        write_json(ROOT / 'records' / 'preflight_partial.json', report)

    for group in ['base', 'new']:
        output = base / 'export_check' / group
        trainer = make_trainer('dtd', 'maple', output, group=group, export=str(output / 'predictions'))
        trainer.load_model(str(base / 'dtd' / 'maple'), epoch=5)
        trainer.experiment_checkpoint = base / 'dtd' / 'maple' / 'MultiModalPromptLearner' / 'model.pth.tar-5'
        trainer.test()
        metadata = json.loads((output / 'predictions' / 'metadata.json').read_text())
        assert metadata['online_offline_check']
        report.setdefault('exports', {})[group] = metadata
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
    report['passed'] = True
    write_json(ROOT / 'records' / 'preflight.json', report)
    print('All preflight checks passed.', flush=True)


if __name__ == '__main__':
    main()
