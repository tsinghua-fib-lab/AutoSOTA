import torch
import torch.nn as nn


class MASELoss(nn.Module):
    """Mean Absolute Scaled Error Loss"""

    def __init__(self):
        super().__init__()
        self.min_number = 1e-8  # floor for denominator to prevent inf losses

    def forward(self, prediction, target):
        numerator = torch.mean(torch.abs(prediction - target))
        denominator = torch.mean(torch.abs(torch.diff(target), n=1))
        denominator = torch.maximum(
            denominator, torch.mul(torch.ones_like(denominator), self.min_number)
        )
        return torch.divide(numerator, denominator)
