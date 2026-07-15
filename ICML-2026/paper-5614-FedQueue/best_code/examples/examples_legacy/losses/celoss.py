import torch.nn as nn


class CELoss(nn.Module):
    """Cross Entroy Loss"""

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, prediction, target):
        target = target if len(target.shape) == 1 else target.squeeze(1)
        return self.criterion(prediction, target)
