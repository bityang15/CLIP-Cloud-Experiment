import json
from pathlib import Path

import torch

from dassl.engine import TRAINER_REGISTRY
from dassl.utils import load_checkpoint
from experiment import ExperimentMixin, ROOT, tensor_digest, write_json
from trainers.coop import CoOp
from trainers.sple_coop import StackSPLE_CoOp


class ExperimentMixinE(ExperimentMixin):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.method = {'CoOpE': 'coop', 'DPCE': 'dpc'}[cfg.TRAINER.NAME]
        initialization = json.loads((self.output_path / 'initialization.json').read_text())
        initialization['method'] = self.method
        write_json(self.output_path / 'initialization.json', initialization)
        manifest = ROOT / 'records/manifests' / self.dataset_key / f'seed{cfg.SEED}/train_base.json'
        self.allowed_negative_paths = {
            (ROOT / 'datasets' / row['image_id']).as_posix()
            for row in json.loads(manifest.read_text())
        }
        self.sampled_negative_ids = []
        self.sampled_group_sizes = []

    def load_model(self, directory, epoch=None):
        if not directory:
            return
        prompt_dir = Path(directory) / 'prompt_learner'
        if epoch is not None and epoch < 0:
            epoch = max(int(path.name.rsplit('-', 1)[1]) for path in prompt_dir.glob('model.pth.tar-*'))
        filename = f'model.pth.tar-{epoch}' if epoch is not None else 'model-best.pth.tar'
        checkpoint_path = prompt_dir / filename
        checkpoint = load_checkpoint(str(checkpoint_path))
        state = dict(checkpoint['state_dict'])
        for name in ['token_prefix', 'token_suffix', 'tokenized_prompts']:
            state.pop(name, None)
        if self.cfg.TRAINER.NAME == 'DPCE' and self.cfg.DATASET.SUBSAMPLE_CLASSES == 'new':
            assert self.cfg.SPLE.STACK.MODE == 'converse'
            source = self.model.prompt_learner.pretuned_ctx
            base_weight = self.cfg.SPLE.STACK.WEIGHT
            new_weight = self.cfg.SPLE.STACK.WEIGHT_FOR_NEW
            parallel = (state['ctx'].to(source.device) - (1 - base_weight) * source) / base_weight
            state['ctx'] = new_weight * parallel + (1 - new_weight) * source
            if new_weight == 0:
                assert torch.equal(state['ctx'], source)
        missing, unexpected = self.model.prompt_learner.load_state_dict(state, strict=False)
        assert set(missing) <= {'token_prefix', 'token_suffix'}
        assert not unexpected
        self.experiment_checkpoint = str(checkpoint_path)


@TRAINER_REGISTRY.register()
class CoOpE(ExperimentMixinE, CoOp):
    def forward_backward(self, batch):
        summary = super().forward_backward(batch)
        self.record_update(batch, summary)
        return summary


@TRAINER_REGISTRY.register()
class DPCE(ExperimentMixinE, StackSPLE_CoOp):
    def forward_backward(self, batch):
        self.sampled_negative_ids = []
        self.sampled_group_sizes = []
        source_before = tensor_digest([('pretuned', self.model.prompt_learner.pretuned_ctx)])
        summary = super().forward_backward(batch)
        source_after = tensor_digest([('pretuned', self.model.prompt_learner.pretuned_ctx)])
        assert source_before == source_after, 'The first-stage prompt changed during DPC training'
        self.record_update(batch, summary)
        record = dict(epoch=self.epoch + 1, batch=self.batch_idx + 1,
                      positive_image_ids=[Path(path).relative_to(self.cfg.DATASET.ROOT).as_posix()
                                          for path in batch['impath']],
                      negative_image_ids=self.sampled_negative_ids,
                      expanded_group_sizes=self.sampled_group_sizes,
                      configured_topk=self.cfg.SPLE.INFER_TOPK,
                      first_stage_prompt_unchanged=True)
        with (self.output_path / 'negative_sampling.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        return summary
