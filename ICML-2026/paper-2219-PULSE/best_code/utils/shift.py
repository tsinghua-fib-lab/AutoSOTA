import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
def shifting_score_torch_btc_fast(X: torch.Tensor, m: int = 24, eps: float = 1e-8) -> torch.Tensor:
    """
    批量计算 [B, T, C] 的 Shifting 指标，返回 [B, C] 的 tensor。
    与你之前的定义保持一致，但用向量化实现，大幅减少 Python for 循环。

    X: [B, T, C]
    """
    assert X.dim() == 3, "只支持 [B, T, C]，其他情况可以先自己 reshape 一下"
    X = X.to(dtype=torch.float32)
    B, T, C = X.shape
    device = X.device

    # 1. 沿时间维 Z-score 归一化: [B, T, C]
    mean = X.mean(dim=1, keepdim=True)             # [B, 1, C]
    std  = X.std(dim=1, keepdim=True) + eps        # [B, 1, C]
    Z = (X - mean) / std                           # [B, T, C]

    # 2. 把 (B, C) 拉平成一个 batch 维度 N = B*C，后面统一按 N 处理
    N = B * C
    Z_flat = Z.permute(0, 2, 1).reshape(N, T)      # [N, T]

    zmin = Z_flat.min(dim=1, keepdim=True).values  # [N, 1]
    zmax = Z_flat.max(dim=1, keepdim=True).values  # [N, 1]
    span = zmax - zmin                             # [N, 1]

    # 常数序列：直接给 0
    valid_seq = span > eps
    span_safe = torch.where(valid_seq, span, torch.ones_like(span))

    # 3. 为每条序列生成自己的 m 个阈值 s_list: [N, m]
    grid = torch.arange(m, device=device, dtype=Z_flat.dtype).view(1, m) / m  # [1, m]
    s_list = zmin + span_safe * grid                                         # [N, m]

    # 4. 对所有 (N, m, T) 一次性比较 z > s
    z_exp = Z_flat.unsqueeze(1)           # [N, 1, T]
    s_exp = s_list.unsqueeze(-1)         # [N, m, 1]
    mask = z_exp > s_exp                 # [N, m, T] bool

    # 每个 (样本, 阈值) 下的元素个数
    count = mask.sum(dim=-1)             # [N, m]
    valid_thresh = count > 0             # 哪些阈值有非空集合 K_i

    # 5. 用累积和 + argmax 求“时间中位数索引”
    cum = mask.cumsum(dim=-1)            # [N, m, T]
    half = (count + 1) // 2              # [N, m]
    half_clamped = torch.clamp(half, min=1)
    cond = cum >= half_clamped.unsqueeze(-1)       # [N, m, T] bool
    median_idx = cond.float().argmax(dim=-1)       # [N, m]，对 count==0 的行结果是 0，后面会 mask 掉

    # 6. 对每条序列，只在 valid_thresh 的位置上算 Mmin、Mmax 和归一化
    # Masked min/max
    large = torch.full_like(median_idx, T - 1)
    small = torch.zeros_like(median_idx)

    Mmin = torch.where(valid_thresh, median_idx, large).min(dim=1).values  # [N]
    Mmax = torch.where(valid_thresh, median_idx, small).max(dim=1).values  # [N]

    # 对完全无有效阈值的序列（理论上只有 span<=eps 的情况）
    has_valid = valid_thresh.any(dim=1) & valid_seq.view(-1)

    denom = (Mmax - Mmin).clamp_min(1)   # 防止除 0，[N]

    # 归一化: [N, m]
    M_norm = (median_idx - Mmin.unsqueeze(-1)) / denom.unsqueeze(-1)

    # 无效阈值位置改成 NaN，方便 nanmedian 忽略
    M_norm = torch.where(valid_thresh, M_norm, torch.full_like(M_norm, float('nan')))

    # 7. 对每条序列的 M_norm 做中位数 -> shifting_flat: [N]
    shifting_flat = torch.nanmedian(M_norm, dim=1).values
    shifting_flat = torch.nan_to_num(shifting_flat, nan=0.0)

    # 对 span 太小的序列强制设为 0
    shifting_flat = torch.where(valid_seq.view(-1), shifting_flat, torch.zeros_like(shifting_flat))

    # 保留 5 位小数
    shifting_flat = torch.round(shifting_flat * 1e5) / 1e5

    # 8. reshape 回 [B, C]
    shifting = shifting_flat.view(B, C)
    return shifting