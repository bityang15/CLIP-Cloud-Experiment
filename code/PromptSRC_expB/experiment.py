import hashlib
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dassl.utils import save_checkpoint
from experiment_metrics import metrics

ROOT = Path('/root/autodl-tmp/clip-exp')


def file_sha256(path):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def tensor_digest(items):
    value = hashlib.sha256()
    for name, tensor in items:
        value.update(name.encode('utf-8'))
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def tcpt_regularizers(logits, labels, text_features, reference):
    logits = logits.float()
    labels = labels.long()
    rows = torch.arange(len(labels), device=labels.device)
    negative = logits.clone()
    negative[rows, labels] = -torch.inf
    margin = logits[rows, labels] - negative.max(dim=1).values
    margin_loss = -0.1 * margin.mean() + 0.01 * margin.var(unbiased=False)
    tuned = text_features.float()
    frozen = reference.float()
    tuned_mean = tuned.mean(dim=0, keepdim=True)
    frozen_mean = frozen.mean(dim=0, keepdim=True)
    tuned_centered = tuned - tuned_mean
    frozen_centered = frozen - frozen_mean
    count = len(tuned)
    tuned_covariance = tuned_centered.t() @ tuned_centered / count
    frozen_covariance = frozen_centered.t() @ frozen_centered / count
    moment_loss = F.mse_loss(tuned_mean, frozen_mean, reduction='sum')
    moment_loss = moment_loss + (tuned_covariance - frozen_covariance).square().sum()
    return margin_loss, moment_loss


def prepare_tcpt_reference(cfg, classnames, device):
    from clip import clip
    from trainers.promptsrc import load_clip_to_cpu

    templates = {'DescribableTextures': '{} texture.',
                 'OxfordFlowers': 'a photo of a {}, a type of flower.'}
    template = templates[cfg.DATASET.NAME]
    started = time.perf_counter()
    with torch.random.fork_rng(devices=[device.index or 0]), torch.no_grad():
        frozen = load_clip_to_cpu(cfg, zero_shot_model=True).to(device).eval()
        frozen.requires_grad_(False)
        tokens = torch.cat([clip.tokenize(template.format(name.replace('_', ' '))) for name in classnames])
        features = F.normalize(frozen.encode_text(tokens.to(device)).float(), dim=-1)
    return features.detach(), dict(template=template, classnames=classnames,
                                   feature_sha256=tensor_digest([('reference', features)]),
                                   seconds=time.perf_counter() - started, precision='float32',
                                   evaluation_mode=True, requires_grad=False)


class ExperimentMixin:
    def __init__(self, cfg):
        started = time.perf_counter()
        super().__init__(cfg)
        self.initialization_seconds = time.perf_counter() - started
        self.method = {'MaPLe': 'tcpt' if cfg.TRAINER.MAPLE.USE_TCPT else 'maple',
                       'PromptSRC': 'promptsrc', 'ZeroshotCLIP': 'clip'}.get(
                           cfg.TRAINER.NAME, cfg.TRAINER.NAME.lower())
        self.trainable_parameters = [(name, parameter) for name, parameter in self.model.named_parameters()
                                     if parameter.requires_grad]
        self.initial_parameter_sha256 = tensor_digest(self.trainable_parameters)
        self.initial_rng_sha256 = tensor_digest([('cpu', torch.get_rng_state()),
                                                ('cuda', torch.cuda.get_rng_state())])
        self.experiment_checkpoint = None
        self.output_path = Path(cfg.OUTPUT_DIR)
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / 'config.yaml').write_text(cfg.dump(), encoding='utf-8')
        self.config_sha256 = hashlib.sha256(cfg.dump().encode('utf-8')).hexdigest()
        self.dataset_key = {'DescribableTextures': 'dtd', 'OxfordFlowers': 'oxford_flowers'}.get(
            cfg.DATASET.NAME, cfg.DATASET.NAME)
        write_json(self.output_path / 'initialization.json', dict(
            method=self.method, seed=cfg.SEED, initialization_seconds=self.initialization_seconds,
            parameter_sha256=self.initial_parameter_sha256, rng_sha256=self.initial_rng_sha256,
            trainable_parameters=[dict(name=name, count=parameter.numel(), dtype=str(parameter.dtype))
                                  for name, parameter in self.trainable_parameters],
            trainable_count=sum(parameter.numel() for _, parameter in self.trainable_parameters),
            config_sha256=self.config_sha256,
            tcpt_reference=getattr(self, 'tcpt_reference_record', None)))

    def set_model_mode(self, mode='train', names=None):
        super().set_model_mode(mode, names)
        teacher = getattr(getattr(self.model, 'prompt_learner', None), 'ZS_image_encoder', None)
        if teacher is not None:
            teacher.eval()

    def record_update(self, batch, summary):
        assert all(math.isfinite(float(value)) for value in summary.values()), summary
        gradients = [parameter.grad for _, parameter in self.trainable_parameters]
        assert all(gradient is not None for gradient in gradients), 'Missing trainable gradient'
        assert torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all().item()
        assert torch.stack([torch.isfinite(parameter).all() for _, parameter in self.trainable_parameters]).all().item()
        now = time.perf_counter()
        record = dict(epoch=self.epoch + 1, batch=self.batch_idx + 1, losses=summary,
                      seconds=now - getattr(self, '_previous_update', now),
                      image_ids=[Path(path).relative_to(self.cfg.DATASET.ROOT).as_posix() for path in batch['impath']],
                      labels=batch['label'].tolist(), gradients_finite=True)
        if self.epoch == 0 and self.batch_idx < 3:
            record['input_sha256'] = tensor_digest([('images', batch['img'])])
        with (self.output_path / 'updates.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        self._previous_update = now

    def run_epoch(self):
        torch.cuda.synchronize()
        started = time.perf_counter()
        self._previous_update = started
        super().run_epoch()
        torch.cuda.synchronize()
        record = dict(epoch=self.epoch + 1, seconds=time.perf_counter() - started,
                      updates=self.num_batches, peak_memory_gib=torch.cuda.max_memory_allocated() / 1024**3)
        with (self.output_path / 'epochs.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')

    def before_train(self):
        torch.cuda.reset_peak_memory_stats()
        super().before_train()

    def save_model(self, epoch, directory, is_best=False, val_result=None, model_name=''):
        for name in self.get_model_names():
            save_checkpoint(dict(
                state_dict=self._models[name].state_dict(), epoch=epoch + 1,
                optimizer=self._optims[name].state_dict() if self._optims[name] is not None else None,
                scheduler=self._scheds[name].state_dict() if self._scheds[name] is not None else None,
                val_result=val_result,
                experiment=dict(classnames=self.dm.dataset.classnames, method=self.method,
                                config_sha256=self.config_sha256,
                                checkpoint_type='GPA' if self.method == 'promptsrc' else 'final_epoch'),
                random_state=dict(python=random.getstate(), numpy=np.random.get_state(),
                                  torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())),
                str(Path(directory) / name), is_best=is_best, model_name=model_name)

    def after_train(self):
        super().after_train()
        training_seconds = time.time() - self.time_start
        training_peak = torch.cuda.max_memory_allocated() / 1024**3
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        self.set_model_mode('eval')
        batch = next(iter(self.val_loader))
        images, _ = self.parse_batch_test(batch)
        with torch.no_grad():
            before = self.model_inference(images).float().cpu()
        self.load_model(str(self.output_path), epoch=self.cfg.OPTIM.MAX_EPOCH)
        self.set_model_mode('eval')
        with torch.no_grad():
            after = self.model_inference(images).float().cpu()
        assert torch.equal(before, after), 'Final checkpoint failed the fixed-batch reload check'
        np.savez_compressed(self.output_path / 'reload_probe.npz', logits=before.numpy(),
                            image_ids=np.asarray([Path(path).relative_to(self.cfg.DATASET.ROOT).as_posix()
                                                  for path in batch['impath']], dtype=np.str_))
        write_json(self.output_path / 'checkpoint_check.json', dict(
            passed=True, exact_logit_match=True, max_logit_difference=(before - after).abs().max().item(),
            samples=len(before), checkpoint_type='GPA' if self.method == 'promptsrc' else 'final_epoch'))
        write_json(self.output_path / 'training_resources.json', dict(
            finished_at=datetime.now().astimezone().isoformat(),
            initialization_seconds=self.initialization_seconds,
            training_seconds=training_seconds, peak_memory_gib=training_peak,
            checkpoint_probe_seconds=time.perf_counter() - started,
            checkpoint_probe_peak_memory_gib=torch.cuda.max_memory_allocated() / 1024**3,
            peak_memory_method='torch.cuda.max_memory_allocated',
            epochs=self.cfg.OPTIM.MAX_EPOCH,
            checkpoint_type='GPA' if self.method == 'promptsrc' else 'final_epoch'))

    @torch.no_grad()
    def test(self, split=None):
        if not self.cfg.EXPERIMENT.EXPORT_DIR:
            return super().test(split)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.set_model_mode('eval')
        self.evaluator.reset()
        split = split or self.cfg.TEST.SPLIT
        assert split in ['val', 'test']
        loader = self.val_loader if split == 'val' else self.test_loader
        assert loader is not None
        predictions, labels, image_ids = [], [], []
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for batch in loader:
            image, target = self.parse_batch_test(batch)
            output = self.model_inference(image).float()
            assert torch.isfinite(output).all().item()
            self.evaluator.process(output, target)
            predictions.append(output.cpu())
            labels.append(target.cpu())
            image_ids.extend(Path(path).relative_to(self.cfg.DATASET.ROOT).as_posix() for path in batch['impath'])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        logits = torch.cat(predictions)
        targets = torch.cat(labels).long()
        assert len(image_ids) == len(set(image_ids))
        online_logits = logits.double()
        probabilities = online_logits.softmax(dim=1)
        confidence, predicted = probabilities.max(dim=1)
        correct = predicted.eq(targets)
        online_nll = F.cross_entropy(online_logits, targets).item()
        bins = torch.clamp(torch.ceil(confidence * 10).long() - 1, 0, 9)
        counts = torch.bincount(bins, minlength=10)
        confidence_sum = torch.bincount(bins, weights=confidence, minlength=10)
        correct_sum = torch.bincount(bins, weights=correct.double(), minlength=10)
        online_ece = 100.0 * (correct_sum - confidence_sum).abs().sum().item() / len(targets)
        online_accuracy = 100.0 * correct.double().mean().item()
        offline = metrics(logits.numpy(), targets.numpy())
        assert counts.tolist() == offline['counts']
        assert abs(online_nll - offline['nll']) < 1e-10
        assert abs(online_ece - offline['ece']) < 1e-8
        assert abs(online_accuracy - offline['accuracy']) < 1e-10
        framework = self.evaluator.evaluate()
        assert abs(framework['accuracy'] - online_accuracy) < 1e-8
        group = self.cfg.DATASET.SUBSAMPLE_CLASSES
        classnames = self.dm.dataset.classnames
        expected_path = ROOT / 'records' / 'manifests' / self.dataset_key / f'seed{self.cfg.SEED}' / f'{split}_{group}.json'
        expected = json.loads(expected_path.read_text(encoding='utf-8'))
        assert image_ids == [row['image_id'] for row in expected]
        assert targets.tolist() == [row['label'] for row in expected]
        assert all(classnames[row['label']] == row['classname'] for row in expected)
        directory = Path(self.cfg.EXPERIMENT.EXPORT_DIR)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / 'predictions.tmp.npz'
        np.savez_compressed(temporary, logits=logits.numpy().astype(np.float32),
                            labels=targets.numpy(), image_ids=np.asarray(image_ids, dtype=np.str_),
                            classnames=np.asarray(classnames, dtype=np.str_))
        temporary.replace(directory / 'predictions.npz')
        checkpoint = Path(self.experiment_checkpoint) if self.experiment_checkpoint else ROOT / 'weights' / 'ViT-B-16.pt'
        write_json(directory / 'metadata.json', dict(
            dataset=self.dataset_key, method=self.method, seed=self.cfg.SEED,
            run_id=str(checkpoint.parent.parent.relative_to(ROOT)) if self.experiment_checkpoint else 'frozen_clip',
            candidate_group=group, split=split, checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
            config_sha256=self.config_sha256, data_manifest_sha256=file_sha256(ROOT / 'records' / 'data_manifest.json'),
            sample_manifest_sha256=file_sha256(expected_path), classnames=classnames,
            raw_logit_scale=True, logits_dtype='float32', n=len(targets), classes=len(classnames),
            text_template={'dtd': '{} texture.', 'oxford_flowers': 'a photo of a {}, a type of flower.'}.get(
                self.dataset_key) if self.method == 'clip' else None,
            evaluation_seconds=elapsed, online=dict(accuracy=online_accuracy, ece=online_ece, nll=online_nll),
            peak_memory_gib=torch.cuda.max_memory_allocated() / 1024**3,
            offline=offline, online_offline_check=True,
            predictions_sha256=file_sha256(directory / 'predictions.npz')))
        print('Exported predictions', directory, 'metrics', offline, flush=True)
        return online_accuracy
