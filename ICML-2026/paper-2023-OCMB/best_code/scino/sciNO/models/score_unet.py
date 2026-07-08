# models/score_unet.py
"""分数估计网络：基于DiffAN的MLP/UNet混合架构"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.math_utils import GaussianFourierProjection


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + 0.5 * self.net(x)


class ScoreUNet(nn.Module):
    """
    简化的MLP/UNet混合架构用于分数估计
    可以替换为DiffAN的完整UNet以获得更好性能
    """
    def __init__(self, dim, hidden=256, time_emb_dim=128, n_blocks=2):
        super().__init__()
        self.dim = dim
        self.time_proj = GaussianFourierProjection(embed_dim=time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden),
            nn.ReLU()
        )

        self.input = nn.Linear(dim, hidden)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(n_blocks)])
        self.out = nn.Linear(hidden, dim)

    def forward(self, x, t):
        """
        Args:
            x: (B, D) 输入数据
            t: (B,) 时间步

        Returns:
            score: (B, D) 分数估计
        """
        # 时间嵌入
        te = self.time_proj(t)  # (B, time_emb_dim)
        te = self.time_mlp(te)  # (B, hidden)

        # 主干网络
        h = self.input(x)  # (B, hidden)
        h = h + te  # 时间条件
        h = self.blocks(h)
        return self.out(h)


class DiffMLP(nn.Module):
    """
    原始DiffAN的MLP架构（兼容性接口）
    """
    def __init__(self, n_nodes: int) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        big_layer = max(1024, 5 * self.n_nodes)
        small_layer = max(128, 3 * self.n_nodes)

        self.main_block = nn.Sequential(
            nn.Linear(self.n_nodes + 1, small_layer, bias=False),
            nn.LeakyReLU(),
            nn.LayerNorm([small_layer]),
            nn.Dropout(0.2),
            nn.Linear(small_layer, big_layer),
            nn.LeakyReLU(),
            nn.LayerNorm([big_layer]),
            nn.Linear(big_layer, big_layer),
            nn.LeakyReLU(),
            nn.Linear(big_layer, small_layer),
            nn.LeakyReLU(),
            nn.Linear(small_layer, self.n_nodes),
        )

    def forward(self, X, t):
        X_t = torch.cat([X, t.unsqueeze(1)], axis=1)
        return self.main_block(X_t)
