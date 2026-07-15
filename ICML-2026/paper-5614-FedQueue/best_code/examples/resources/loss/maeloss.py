import torch.nn as nn


class MAELoss(nn.Module):
    """Mean Absolute Error Loss"""

    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss(reduction="mean")

    def forward(self, prediction, target):
        return self.criterion(prediction, target)
