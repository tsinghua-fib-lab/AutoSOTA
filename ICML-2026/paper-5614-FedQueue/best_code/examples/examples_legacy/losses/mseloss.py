import torch.nn as nn


class MSELoss(nn.Module):
    """Mean Squared Error Loss"""

    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss(reduction="mean")

    def forward(self, prediction, target):
        return self.criterion(prediction, target)
