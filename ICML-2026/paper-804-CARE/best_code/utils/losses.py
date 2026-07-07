import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import normal


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.2, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        num_class = pred.size()[-1]
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (num_class-1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class GCLLoss(nn.Module):

    def __init__(self, cls_num_list, m=0.5, weight=None, s=30, train_cls=False, noise_mul=1., gamma=0.):
        super(GCLLoss, self).__init__()
        cls_list = torch.cuda.FloatTensor(cls_num_list)
        m_list = torch.log(cls_list)
        m_list = m_list.max() - m_list
        self.m_list = m_list
        assert s > 0
        self.m = m
        self.s = s
        self.weight = weight
        self.simpler = normal.Normal(0, 1 / 3)
        self.train_cls = train_cls
        self.noise_mul = noise_mul
        self.gamma = gamma

    def forward(self, cosine, target):
        index = torch.zeros_like(cosine, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        noise = self.simpler.sample(cosine.shape).clamp(-1, 1).to(
            cosine.device)  # self.scale(torch.randn(cosine.shape).to(cosine.device))

        # cosine = cosine - self.noise_mul * noise/self.m_list.max() *self.m_list
        cosine = cosine - self.noise_mul * noise.abs() / self.m_list.max() * self.m_list
        output = torch.where(index, cosine - self.m, cosine)
        if self.train_cls:
            return focal_loss(F.cross_entropy(self.s * output, target, reduction='none', weight=self.weight),
                              self.gamma)
        else:
            return F.cross_entropy(self.s * output, target, weight=self.weight)

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


# class LogitAdjustedLoss(nn.Module):
#     def __init__(self, cls_num_list, tau=1.0):
#         super().__init__()
#         # cls_num_list: 这是一个包含每个类别样本数量的张量。它的长度应该等于类别的数量。
#         # tau: 这是一个调整因子的超参数，用于控制类别调整的强度。默认为1.0。
#         # cls_num_ratio: 计算每个类别的样本比例。cls_num_list 是每个类别的样本数量，torch.sum(cls_num_list) 是所有类别样本数量的总和。将每个类别的样本数量除以总数量得到类别的样本比例。
#         # log_cls_num: 计算每个类别样本比例的对数。torch.log(cls_num_ratio) 是对类别比例进行对数变换的结果，这个对数值用于调整 logits。
#         cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
#         log_cls_num = torch.log(cls_num_ratio)
#         self.log_cls_num = log_cls_num
#         self.tau = tau
#
#     def forward(self, logit, target):
#         # logit_adjusted: 调整后的 logits，通过将原始 logits 与类别调整项相加来计算。self.log_cls_num.unsqueeze(0) 将类别调整项的形状从 (num_classes,) 扩展到 (1, num_classes)，
#         # 这样可以进行广播，确保它与 logit 形状相匹配。
#         # 要将形状为 (batch_size, num_classes) 的 logits 和形状为 (1, num_classes) 的类别调整项相加，可以使用广播机制。具体操作是将 (1, num_classes) 形状的调整项广播到 (batch_size, num_classes)，
#         # 然后进行逐元素相加。在大多数深度学习框架中，如 PyTorch 或 TensorFlow，广播会自动处理这种情况。
#         logit_adjusted = logit + self.tau * self.log_cls_num.unsqueeze(0)
#         return F.cross_entropy(logit_adjusted, target)
# # unsqueeze(0) 操作是在张量的第一个维度（即索引为0的维度）上插入一个新的维度。这将改变张量的形状。例如，如果原始张量的形状是 (num_classes,)，
# # unsqueeze(0) 会将其形状变为 (1, num_classes)。这样做可以使得张量在广播过程中与其他形状不同的张量进行相加或其他操作。

class LogitAdjustedLoss(nn.Module):
    def __init__(self, cls_num_list, tau=1.0):
        super().__init__()

        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
        log_cls_num = torch.log(cls_num_ratio)
        self.log_cls_num = log_cls_num
        self.tau = tau

    def forward(self, logit, target, cls_num_list, flag=True):
        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
        log_cls_num = torch.log(cls_num_ratio)
        logit_adjusted = logit + self.tau * log_cls_num.unsqueeze(0)
        # if epoch > 17:
        #     per_cls_weights = 1.0 / np.array(cls_num_list.cpu())
        #     per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        #     per_cls_weights = torch.tensor(per_cls_weights).cuda()
        #     return F.cross_entropy(logit_adjusted, target, weight=per_cls_weights)
        # else:
        if flag:
            return F.cross_entropy(logit_adjusted, target)
        else:
            return F.cross_entropy(logit, target)


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