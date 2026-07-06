import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


class MedianRecovery(Metric):
    def __init__(self, ignore_index: int = -100, **kwargs):
        super().__init__(**kwargs)
        self.ignore_index = ignore_index
        self.add_state("acc_list", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds = preds.argmax(dim=-1)
        assert preds.shape == target.shape
        bsz = target.shape[0]

        preds = preds.view(bsz, -1)
        target = target.view(bsz, -1)
        mask = (target != self.ignore_index).float()
        acc = ((preds == target) * mask).sum(1) / mask.sum(1)
        self.acc_list.append(acc)

    def compute(self):
        # parse inputs
        acc_list = dim_zero_cat(self.acc_list)
        return torch.median(acc_list)
