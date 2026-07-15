import torch
from torch.nn.modules.loss import _Loss


class BaselineLoss(_Loss):
    def __init__(self):
        super().__init__()
        self.bce = torch.nn.BCELoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return self.bce(input, target)
