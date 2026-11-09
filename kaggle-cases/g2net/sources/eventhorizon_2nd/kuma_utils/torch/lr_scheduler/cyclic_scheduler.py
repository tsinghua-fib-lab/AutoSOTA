'''
Implementation of Cyclic learning rate schedulers
https://github.com/bluesky314/Cyclical_LR_Scheduler_With_Decay_Pytorch
'''
import math
from bisect import bisect_right, bisect_left

import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


class CyclicCosAnnealingLR(_LRScheduler):
    r"""
    
    Implements reset on milestones inspired from CosineAnnealingLR pytorch
    
    Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))
    When last_epoch > last set milestone, lr is automatically set to \eta_{min}
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list of ints): List of epoch indices. Must be increasing.
        decay_milestones(list of ints):List of increasing epoch indices. Ideally,decay values should overlap with milestone points
        gamma (float): factor by which to decay the max learning rate at each decay milestone
        eta_min (float): Minimum learning rate. Default: 1e-6
        last_epoch (int): The index of last epoch. Default: -1.
        
        
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, milestones, decay_milestones=None, gamma=0.5, eta_min=1e-6, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.eta_min = eta_min
        self.milestones = milestones
        self.milestones2 = decay_milestones

        self.gamma = gamma
        super(CyclicCosAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        if self.last_epoch >= self.milestones[-1]:
            return [self.eta_min for base_lr in self.base_lrs]

        idx = bisect_right(self.milestones, self.last_epoch)

        left_barrier = 0 if idx == 0 else self.milestones[idx-1]
        right_barrier = self.milestones[idx]

        width = right_barrier - left_barrier
        curr_pos = self.last_epoch - left_barrier

        if self.milestones2:
            return [self.eta_min + (base_lr * self.gamma ** bisect_right(self.milestones2, self.last_epoch) - self.eta_min) *
                    (1 + math.cos(math.pi * curr_pos / width)) / 2
                    for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * curr_pos / width)) / 2
                    for base_lr in self.base_lrs]


class CyclicLinearLR(_LRScheduler):
    r"""
    Implements reset on milestones inspired from Linear learning rate decay
    
    Set the learning rate of each parameter group using a linear decay
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart:
    .. math::
        \eta_t = \eta_{min} + (\eta_{max} - \eta_{min})(1 -\frac{T_{cur}}{T_{max}})
    When last_epoch > last set milestone, lr is automatically set to \eta_{min}
  
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list of ints): List of epoch indices. Must be increasing.
        decay_milestones(list of ints):List of increasing epoch indices. Ideally,decay values should overlap with milestone points
        gamma (float): factor by which to decay the max learning rate at each decay milestone
        eta_min (float): Minimum learning rate. Default: 1e-6
        last_epoch (int): The index of last epoch. Default: -1.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, milestones, decay_milestones=None, gamma=0.5, eta_min=1e-6, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.eta_min = eta_min

        self.gamma = gamma
        self.milestones = milestones
        self.milestones2 = decay_milestones
        super(CyclicLinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        if self.last_epoch >= self.milestones[-1]:
            return [self.eta_min for base_lr in self.base_lrs]

        idx = bisect_right(self.milestones, self.last_epoch)

        left_barrier = 0 if idx == 0 else self.milestones[idx-1]
        right_barrier = self.milestones[idx]

        width = right_barrier - left_barrier
        curr_pos = self.last_epoch - left_barrier

        if self.milestones2:
            return [self.eta_min + (base_lr * self.gamma ** bisect_right(self.milestones2, self.last_epoch) - self.eta_min) *
                    (1. - 1.0*curr_pos / width)
                    for base_lr in self.base_lrs]

        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1. - 1.0*curr_pos / width)
                    for base_lr in self.base_lrs]
