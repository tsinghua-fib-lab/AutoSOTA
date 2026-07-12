import torch
import torch.nn as nn
import torch.optim as optim
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.1, gamma=3.0, reduction='mean'):
        """
        alpha=0.1: 即使正样本难分，我也只给它 0.1 的基础关注，强迫模型优先保证负样本对。
        gamma=3.0: 强力挖掘困难样本。
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)

        # 动态调整 alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # 核心公式
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, reduction='mean'):
        """
        Asymmetric Loss (非对称损失) - 专为极度不平衡和难易样本设计

        Args:
            gamma_neg (int): 负样本(Non-Member)的聚焦参数。设大一点(如4)以严厉惩罚假阳性。
            gamma_pos (int): 正样本(Member)的聚焦参数。设小一点(如1或0)以保持召回。
            clip (float): 非对称截断阈值。如果负样本预测概率 > (1-clip)，则Loss归零。
            eps (float): 防止 log(0) 的极小值。
            reduction (str): 'mean' | 'sum' | 'none'。
                             设置为 'none' 时返回向量，用于后续手动加权。
        """
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        """
        x: Logits (未经过 Sigmoid 的模型输出), shape [Batch_Size, 1] 或 [Batch_Size]
        y: Labels (0 或 1), shape 必须与 x 一致
        """

        # 1. 计算概率 (Probabilities)
        # xs_pos: 属于正类(Member)的概率
        xs_pos = torch.sigmoid(x)
        # xs_neg: 属于负类(Non-Member)的概率
        xs_neg = 1 - xs_pos

        # 2. Asymmetric Clipping (非对称截断)
        # 作用：彻底忽略那些非常容易区分的负样本 (Easy Negatives)
        if self.clip > 0:
            # 逻辑：如果 xs_neg 本来是 0.99 (很确信是负类)，加上 clip 后变成 >1，
            # clamp(max=1) 后变成 1.0。
            # 下一步 log(1.0) = 0，梯度消失，不再学习该样本。
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # 3. 计算基础 Log Loss (Basic Cross Entropy)
        # 加上 eps 防止数值不稳定
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # 4. 计算 Focal Weights (自适应权重)
        # 核心思想：(1 - p_correct) ^ gamma
        # 预测越准，p_correct 越接近 1，权重越接近 0 (不关注)
        # 预测越差，p_correct 越接近 0，权重越接近 1 (重点关注)

        # 正样本权重: (1 - p_pos) ^ gamma_pos
        weight_pos = (1 - xs_pos) ** self.gamma_pos

        # 负样本权重: (1 - p_neg) ^ gamma_neg
        # [逻辑修正]: 这里必须是 (1 - xs_neg)。
        # 如果 xs_neg 很低(比如0.1，意味着模型把它误判为正类了)，
        # 那么权重 = (1 - 0.1)^4 = 0.65 (很大)，给与重罚。
        weight_neg = (1 - xs_neg) ** self.gamma_neg

        # 5. 组合 Loss
        # 公式: Loss = - Weight * LogLoss
        loss = - weight_pos * los_pos - weight_neg * los_neg

        # 6. 根据 reduction 返回结果
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # 返回形状为 [Batch_Size] 的向量，方便后续手动加权

# --- 使用方法 ---
# class AsymmetricLoss(nn.Module):
#     def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
#         super(AsymmetricLoss, self).__init__()
#         self.gamma_neg = gamma_neg
#         self.gamma_pos = gamma_pos
#         self.clip = clip
#         self.eps = eps
#
#     def forward(self, x, y):
#         # x: logits, y: labels
#
#         # 1. 计算概率
#         xs_pos = torch.sigmoid(x)
#         xs_neg = 1 - xs_pos
#
#         # 2. Asymmetric Clipping (处理掉极简单的负样本)
#         if self.clip > 0:
#             xs_neg = (xs_neg + self.clip).clamp(max=1)
#
#         # 3. 计算基础 Log Loss
#         # 加上 eps 防止 log(0)
#         los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
#         los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
#
#         # 4. 计算 Focal Weights (关注难样本)
#         # 正样本权重: 预测越接近0(错得越离谱)，权重越大
#         weight_pos = (1 - xs_pos) ** self.gamma_pos
#
#         # 负样本权重: 预测越接近0(即 xs_neg越低，意味着被误判为正)，权重越大
#         # [核心修正]: 这里改为 (1 - xs_neg)
#         weight_neg = (1 - xs_neg) ** self.gamma_neg
#
#         # 5. 组合 Loss
#         loss = - weight_pos * los_pos - weight_neg * los_neg
#
#         return loss.sum()

