"""
Loss functions for PyTorch.
"""

import torch
import torch.nn as nn

def mmd_loss(source, target):
    """
    source: [Ns, D]
    target: [Nt, D]
    """
    kernels = gaussian_kernel(source, target)
    Ns = source.size(0)
    Nt = target.size(0)

    XX = kernels[:Ns, :Ns]
    YY = kernels[Ns:, Ns:]
    XY = kernels[:Ns, Ns:]
    YX = kernels[Ns:, :Ns]

    loss = XX.mean() + YY.mean() - XY.mean() - YX.mean()
    return loss

def coral_loss(source, target):
    """
    source: [Ns, D]
    target: [Nt, D]
    """
    d = source.size(1)

    # center
    source = source - source.mean(dim=0, keepdim=True)
    target = target.float() - target.float().mean(dim=0, keepdim=True)
    
    # covariance
    cov_s = (source.T @ source) / (source.size(0) - 1)
    cov_t = (target.T @ target) / (target.size(0) - 1)

    # Frobenius norm
    loss = torch.mean((cov_s - cov_t) ** 2)

    return loss / (4 * d * d)

import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.
    Down-weights easy examples, focuses on hard ones.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
