import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, logit, target):
        return focal_loss(F.cross_entropy(logit, target, reduction='none', weight=self.weight), self.gamma)


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, s=30):
        super().__init__()
        m_list = 1.0 / torch.sqrt(torch.sqrt(cls_num_list))
        m_list = m_list * (max_m / torch.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        self.s = s

    def forward(self, logit, target):
        index = torch.zeros_like(logit, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        logit_m = logit - batch_m * self.s  # scale only the margin, as the logit is already scaled.

        output = torch.where(index, logit_m, logit)
        return F.cross_entropy(output, target)


class ClassBalancedLoss(nn.Module):
    def __init__(self, cls_num_list, beta=0.9999):
        super().__init__()
        per_cls_weights = (1.0 - beta) / (1.0 - (beta ** cls_num_list))
        per_cls_weights = per_cls_weights / torch.mean(per_cls_weights)
        self.per_cls_weights = per_cls_weights
    
    def forward(self, logit, target):
        logit = logit.to(self.per_cls_weights.dtype)
        return F.cross_entropy(logit, target, weight=self.per_cls_weights)


class GeneralizedReweightLoss(nn.Module):
    def __init__(self, cls_num_list, exp_scale=1.0):
        super().__init__()
        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
        per_cls_weights = 1.0 / (cls_num_ratio ** exp_scale)
        per_cls_weights = per_cls_weights / torch.mean(per_cls_weights)
        self.per_cls_weights = per_cls_weights
    
    def forward(self, logit, target):
        logit = logit.to(self.per_cls_weights.dtype)
        return F.cross_entropy(logit, target, weight=self.per_cls_weights)


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, cls_num_list):
        super().__init__()
        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
        log_cls_num = torch.log(cls_num_ratio)
        self.log_cls_num = log_cls_num

    def forward(self, logit, target):
        logit_adjusted = logit + self.log_cls_num.unsqueeze(0)
        return F.cross_entropy(logit_adjusted, target)


class LogitAdjustedLoss(nn.Module):
    def __init__(self, cls_num_list, tau=1.0):
        super().__init__()
        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
        log_cls_num = torch.log(cls_num_ratio)
        self.log_cls_num = log_cls_num
        self.tau = tau

    def forward(self, logit, target):
        logit_adjusted = logit + self.tau * self.log_cls_num.unsqueeze(0)
        return F.cross_entropy(logit_adjusted, target)


class LADELoss(nn.Module):
    def __init__(self, cls_num_list, remine_lambda=0.1, estim_loss_weight=0.1):
        super().__init__()
        self.num_classes = len(cls_num_list)
        self.prior = cls_num_list / torch.sum(cls_num_list)

        self.balanced_prior = torch.tensor(1. / self.num_classes).float().to(self.prior.device)
        self.remine_lambda = remine_lambda

        self.cls_weight = cls_num_list / torch.sum(cls_num_list)
        self.estim_loss_weight = estim_loss_weight

    def mine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        N = x_p.size(-1)
        first_term = torch.sum(x_p, -1) / (num_samples_per_cls + 1e-8)
        second_term = torch.logsumexp(x_q, -1) - np.log(N)
 
        return first_term - second_term, first_term, second_term

    def remine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        loss, first_term, second_term = self.mine_lower_bound(x_p, x_q, num_samples_per_cls)
        reg = (second_term ** 2) * self.remine_lambda
        return loss - reg, first_term, second_term

    def forward(self, logit, target):
        logit_adjusted = logit + torch.log(self.prior).unsqueeze(0)
        ce_loss =  F.cross_entropy(logit_adjusted, target)

        per_cls_pred_spread = logit.T * (target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target))  # C x N
        pred_spread = (logit - torch.log(self.prior + 1e-9) + torch.log(self.balanced_prior + 1e-9)).T  # C x N

        num_samples_per_cls = torch.sum(target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target), -1).float()  # C
        estim_loss, first_term, second_term = self.remine_lower_bound(per_cls_pred_spread, pred_spread, num_samples_per_cls)
        estim_loss = -torch.sum(estim_loss * self.cls_weight)

        return ce_loss + self.estim_loss_weight * estim_loss