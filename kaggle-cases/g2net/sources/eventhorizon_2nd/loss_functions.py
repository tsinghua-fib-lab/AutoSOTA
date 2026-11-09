import torch
import torch.nn as nn
import torch.nn.functional as F


def binary_cross_entropy(inputs, target, weight=None, reduction='mean', smooth_eps=None, from_logits=False):
    """cross entropy loss, with support for label smoothing https://arxiv.org/abs/1512.00567"""
    smooth_eps = smooth_eps or 0
    if smooth_eps > 0:
        _target = target.clone().float()
        # _target.add_(smooth_eps).div_(2.)
        _target.mul_(1-smooth_eps).add(0.5*smooth_eps)
    else:
        _target = target
    if from_logits:
        return F.binary_cross_entropy_with_logits(inputs, _target, weight=weight, reduction=reduction)
    else:
        return F.binary_cross_entropy(inputs, _target, weight=weight, reduction=reduction)


def binary_cross_entropy_with_logits(inputs, target, weight=None, reduction='mean', smooth_eps=None, from_logits=True):
    return binary_cross_entropy(inputs, target, weight, reduction, smooth_eps, from_logits)


class BCELoss(nn.BCELoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', smooth_eps=None, from_logits=False):
        super(BCELoss, self).__init__(weight, size_average, reduce, reduction)
        self.smooth_eps = smooth_eps
        self.from_logits = from_logits

    def forward(self, input, target):
        return binary_cross_entropy(input, target,
                                    weight=self.weight, reduction=self.reduction,
                                    smooth_eps=self.smooth_eps, from_logits=self.from_logits)

    def __repr__(self):
        return f'BCELoss(smooth={self.smooth_eps})'


class BCEWithLogitsLoss(BCELoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', smooth_eps=None, from_logits=True):
        super(BCEWithLogitsLoss, self).__init__(weight, size_average,
                                                reduce, reduction, smooth_eps=smooth_eps, from_logits=from_logits)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        if not isinstance(smoothing, torch.Tensor):
            self.smoothing = nn.Parameter(
                torch.tensor(smoothing), requires_grad=False)
        else:
            self.smoothing = nn.Parameter(
                smoothing, requires_grad=False)
        assert 0 <= self.smoothing.min() and self.smoothing.max() < 1
    
    @staticmethod
    def _smooth(targets:torch.Tensor, smoothing:torch.Tensor):
        with torch.no_grad():
            if smoothing.shape != targets.shape:
                _smoothing = smoothing.expand_as(targets)
            else:
                _smoothing = smoothing
            return targets * (1.0 - _smoothing) + 0.5 * _smoothing

    def forward(self, inputs, targets):
        targets = self._smooth(targets, self.smoothing)
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1. - pt)**self.gamma * bce_loss
        return focal_loss.mean()

    def __repr__(self):
        return f'FocalLoss(smoothing={self.smoothing})'
