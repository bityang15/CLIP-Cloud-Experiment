import gc
import json
import sys
from pathlib import Path
from types import SimpleNamespace

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK / 'Dassl.pytorch'))
sys.path.append(str(WORK.parent / 'PromptSRC_expB'))

import numpy as np
import torch

from dassl.engine import build_trainer
from dassl.utils import set_random_seed
from experiment import ROOT, tensor_digest, write_json
from train import setup_cfg


def configuration(dataset, trainer, output, group='base', source=None):
    config_file = ('configs/trainers/CoOp/vit_b16.yaml' if trainer == 'CoOpE'
                   else 'configs/trainers/SPLE/CoOp/vit_b16_ep20.yaml')
    options = ['DATASET.NUM_SHOTS', '16', 'DATASET.SUBSAMPLE_CLASSES', group,
               'OPTIM.MAX_EPOCH', '1', 'DATALOADER.TRAIN_X.BATCH_SIZE', '4',
               'DATALOADER.NUM_WORKERS', '2', 'TEST.NO_TEST', 'True']
    if source is not None:
        options += ['SPLE.BACK_CKPT_PATH', str(source), 'SPLE.BACK_CKPT_EPOCH', '1',
                    'SPLE.PIC_LIB', str(ROOT / f'datasets/SPLE_Database/{dataset}_seed1.json'),
                    'SPLE.STACK.WEIGHT', '0.5', 'SPLE.STACK.WEIGHT_FOR_NEW', '0.0']
    args = SimpleNamespace(root=str(ROOT / 'datasets'), seed=1, trainer=trainer,
        output_dir=str(output), resume='', source_domains=None, target_domains=None,
        transforms=None, backbone='', head='', config_file=config_file,
        dataset_config_file=str(WORK / f'configs/datasets/{dataset}.yaml'), opts=options)
    cfg = setup_cfg(args)
    cfg.defrost()
    cfg.EXPERIMENT.EXPORT_DIR = str(output / 'predictions' / f'test_{group}')
    cfg.freeze()
    return cfg


def check_data(trainer, dataset):
    for name, actual in [('train_base', trainer.dm.dataset.train_x),
                         ('val_base', trainer.dm.dataset.val),
                         ('test_base', trainer.dm.dataset.test)]:
        expected = json.loads((ROOT / f'records/manifests/{dataset}/seed1/{name}.json').read_text())
        rows = [dict(image_id=Path(item.impath).relative_to(ROOT / 'datasets').as_posix(),
                     label=item.label, classname=item.classname) for item in actual]
        assert rows == expected, name


def train_one_batch(cfg, dataset):
    set_random_seed(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    trainer = build_trainer(cfg)
    check_data(trainer, dataset)
    trainer.train_loader_x = [next(iter(trainer.train_loader_x))]
    initial = tensor_digest(trainer.trainable_parameters)
    trainer.train()
    assert tensor_digest(trainer.trainable_parameters) != initial
    assert json.loads((Path(cfg.OUTPUT_DIR) / 'checkpoint_check.json').read_text())['passed']
    return trainer


def check_prompt_weights(trainer):
    learner = trainer.model.prompt_learner
    assert learner.ctx.data_ptr() != learner.pretuned_ctx.data_ptr()
    source = learner.pretuned_ctx.detach().clone()
    mixed = learner.ctx.detach().clone()
    learner.train()
    training_ctx = learner()[0, 1:1 + learner.n_ctx].detach()
    expected = (mixed - 0.5 * source) * 2
    assert torch.equal(training_ctx, expected)
    learner.eval()
    evaluation_ctx = learner()[0, 1:1 + learner.n_ctx].detach()
    assert torch.equal(evaluation_ctx, mixed)
    return dict(separate_storage=True, training_decoupling_exact=True,
                base_inference_mixture_exact=True,
                first_stage_sha256=tensor_digest([('source', source)]))


def main():
    report = dict(passed=False, scope='One real update per stage and dataset, not formal runs', datasets={})
    for dataset in ['dtd', 'oxford_flowers']:
        directory = ROOT / 'outputs/E_preflight/attempt01' / dataset
        baseline_dir = directory / 'coop'
        baseline = train_one_batch(configuration(dataset, 'CoOpE', baseline_dir), dataset)
        baseline.test()
        del baseline
        gc.collect()
        torch.cuda.empty_cache()
        dpc_dir = directory / 'dpc'
        dpc = train_one_batch(configuration(dataset, 'DPCE', dpc_dir, source=baseline_dir), dataset)
        weight_checks = check_prompt_weights(dpc)
        dpc.test()
        sampling = json.loads((dpc_dir / 'negative_sampling.jsonl').read_text().splitlines()[0])
        assert sampling['negative_image_ids']
        assert all(size <= 8 for size in sampling['expanded_group_sizes'])
        del dpc
        gc.collect()
        torch.cuda.empty_cache()
        new_logits = {}
        for method, trainer_name, checkpoint in [('coop', 'CoOpE', baseline_dir), ('dpc', 'DPCE', dpc_dir)]:
            cfg = configuration(dataset, trainer_name, directory / f'{method}_new', 'new',
                                source=baseline_dir if method == 'dpc' else None)
            set_random_seed(1)
            trainer = build_trainer(cfg)
            trainer.load_model(str(checkpoint), epoch=1)
            trainer.test()
            with np.load(Path(cfg.EXPERIMENT.EXPORT_DIR) / 'predictions.npz', allow_pickle=False) as prediction:
                new_logits[method] = prediction['logits'].copy()
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
        assert np.array_equal(new_logits['coop'], new_logits['dpc'])
        report['datasets'][dataset] = dict(passed=True, data_manifests_exact=True,
            prompt_weights=weight_checks, real_update_each_stage=True,
            negative_sampling_budget_passed=True, new_weight_zero_logits_exact=True)
        write_json(ROOT / 'records/E_preflight.json', report)
    report['passed'] = True
    write_json(ROOT / 'records/E_preflight.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
