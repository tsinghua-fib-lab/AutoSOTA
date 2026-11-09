import torch
from torch.distributions.beta import Beta
from kuma_utils.torch.utils import freeze_module
from kuma_utils.torch.hooks import SimpleHook
from kuma_utils.torch.callbacks import CallbackTemplate


class MixupTrain(SimpleHook):

    def __init__(self, evaluate_in_batch=False, alpha=0.4, hard_label=False, lor_label=False):
        super().__init__(evaluate_in_batch=evaluate_in_batch)
        self.alpha = alpha
        self.beta = Beta(alpha, alpha)
        self.hard_label = hard_label
        self.lor_label = lor_label

    def forward_train(self, trainer, inputs):
        target = inputs[-1]
        bs = target.shape[0]
        lam = self.beta.rsample(sample_shape=(bs,)).to(target.device)
        idx = torch.randperm(bs).to(target.device)
        approx, lam = trainer.model(*inputs[:-1], lam=lam, idx=idx)
        if self.lor_label:
            target = target + target[idx] - target * target[idx]
        else:
            target = target * lam[:, None] + target[idx] * (1-lam)[:, None]
            if self.hard_label:
                target = (target > self.hard_label).float()
        loss = trainer.criterion(approx, target)
        return loss, approx.detach()

    def forward_valid(self, trainer, inputs):
        return super().forward_train(trainer, inputs)

    def __repr__(self) -> str:
        return f'MixUp(alpha={self.alpha}, hard_label={self.hard_label}, lor_label={self.lor_label})'


class MixupTrain2(SimpleHook):

    def __init__(self, evaluate_in_batch=False, alpha=0.4, pos_pos=True, neg_neg=True):
        super().__init__(evaluate_in_batch=evaluate_in_batch)
        self.alpha = alpha
        self.beta = Beta(alpha, alpha)
        self.pos = pos_pos
        self.neg = neg_neg

    def forward_train(self, trainer, inputs):
        target = inputs[-1]
        bs = target.shape[0]
        lam = self.beta.rsample(sample_shape=(bs,)).to(target.device)
        idx = torch.arange(bs).to(target.device)
        if self.pos:
            pos_idx = torch.where(target == 1.0)[0]
            pos_rand = pos_idx[torch.randperm(len(pos_idx))]
            idx[pos_idx] = pos_rand
        if self.neg:
            neg_idx = torch.where(target == 0.0)[0]
            neg_rand = neg_idx[torch.randperm(len(neg_idx))]
            idx[neg_idx] = neg_rand
        approx, lam = trainer.model(*inputs[:-1], lam=lam, idx=idx)
        loss = trainer.criterion(approx, target)
        return loss, approx.detach()

    def forward_valid(self, trainer, inputs):
        return super().forward_train(trainer, inputs)

    def __repr__(self) -> str:
        return f'MixUp2(alpha={self.alpha}, pos_pos={self.pos}, neg_neg={self.neg})'


def make_tta_dataloader(loader, dataset, dataset_params):
    skip_keys = ['dataset', 'sampler', 'batch_sampler', 'dataset_kind']
    dl_args = {
        k: v for k, v in loader.__dict__.items()
        if not k.startswith('_') and k not in skip_keys
    }
    dl_args['dataset'] = dataset(**dataset_params)
    return type(loader)(**dl_args)
