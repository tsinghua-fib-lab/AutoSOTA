# models/fno_operator.py
"""
傅里叶神经算子 (FNO) - SciNO的核心创新

性能优化版本：
- 使用torch.einsum替代Python循环进行批量复数矩阵乘法
- 大幅提升GPU利用率

论文核心思想：
- 在频域进行卷积操作，天然保证输出平滑性
- 平滑性对于计算二阶导数（Hessian）至关重要
- 多尺度谱分支处理不同频率特征

API:
    FNOOperator(in_dim, width=128, modes_list=[32,16,8], depth=4, time_emb_dim=64)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..utils.math_utils import GaussianFourierProjection


class SpectralConv1d(nn.Module):
    """
    1D谱卷积层（优化版）

    使用rFFT进行频域卷积，对前`modes`个傅里叶系数应用可学习的复数权重。
    这种操作保证了全局感受野和输出平滑性。

    优化：使用批量矩阵乘法替代Python循环
    """
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # 复数权重参数化为两个实张量
        self.scale = 1 / (in_channels * out_channels)
        self.weight_real = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes)
        )
        self.weight_imag = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes)
        )

    def compl_mul1d_batched(self, input_ft, weight_r, weight_i):
        """
        批量复数矩阵乘法（优化版）

        使用einsum一次性处理所有模式，避免Python循环

        Args:
            input_ft: (B, F, C_in) 复数频域输入
            weight_r: (C_in, C_out, modes) 权重实部
            weight_i: (C_in, C_out, modes) 权重虚部

        Returns:
            (B, F, C_out) 复数频域输出
        """
        B, F, C_in = input_ft.shape
        modes = min(self.modes, F)

        # 构造复数权重张量 (C_in, C_out, modes)
        weight = torch.complex(weight_r, weight_i)

        # 取需要处理的模式
        input_modes = input_ft[:, :modes, :]  # (B, modes, C_in)

        # 使用einsum进行批量复数矩阵乘法
        # input: (B, modes, C_in), weight: (C_in, C_out, modes)
        # 我们需要对每个模式m: out[:, m, :] = input[:, m, :] @ weight[:, :, m]
        # 等价于: out[b, m, c_out] = sum_c_in input[b, m, c_in] * weight[c_in, c_out, m]
        out_modes = torch.einsum('bmc,com->bmo', input_modes, weight[:, :, :modes])

        # 创建输出并填充
        out_ft = torch.zeros(B, F, self.out_channels, dtype=torch.cfloat, device=input_ft.device)
        out_ft[:, :modes, :] = out_modes

        return out_ft

    def forward(self, x):
        """
        Args:
            x: (B, D, C) 输入张量

        Returns:
            (B, D, C) 频域卷积后的输出
        """
        B, D, C = x.shape

        # 傅里叶变换
        x_ft = torch.fft.rfft(x, dim=1)  # (B, Df, C) 复数

        # 应用可学习权重（批量化）
        out_ft = self.compl_mul1d_batched(x_ft, self.weight_real, self.weight_imag)

        # 逆傅里叶变换
        x_out = torch.fft.irfft(out_ft, n=D, dim=1)
        return x_out.real


class FNOBlock(nn.Module):
    """FNO基础块：谱卷积 + 1x1卷积 + 残差 + LayerNorm"""
    def __init__(self, width, modes):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.w_conv = nn.Conv1d(width, width, kernel_size=1)
        self.norm = nn.LayerNorm(width)
        # 谱归一化提升稳定性
        nn.utils.spectral_norm(self.w_conv)

    def forward(self, x):
        """
        Args:
            x: (B, D, width)

        Returns:
            (B, D, width)
        """
        y = self.spectral(x) + self.w_conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        y = F.gelu(y)
        return self.norm(x + y)


class MultiScaleSpectral(nn.Module):
    """
    多尺度谱卷积模块

    结合不同模式数量的谱卷积分支，捕获多尺度特征
    """
    def __init__(self, width, modes_list):
        super().__init__()
        self.branches = nn.ModuleList([
            SpectralConv1d(width, width, modes=m) for m in modes_list
        ])
        self.fuse = nn.Linear(len(modes_list) * width, width)

    def forward(self, x):
        """
        Args:
            x: (B, D, width)

        Returns:
            (B, D, width)
        """
        outs = [branch(x) for branch in self.branches]
        concat = torch.cat(outs, dim=-1)  # (B, D, width * n_branches)

        B, D, Ctot = concat.shape
        fused = self.fuse(concat.view(B * D, Ctot)).view(B, D, -1)
        return fused


class FNOOperator(nn.Module):
    """
    傅里叶神经算子 (SciNO核心模块)

    学习从分数场到Hessian对角线的映射：
        G_θ: s(x) -> diag(∇²log p(x))

    核心优势：
    - 频域操作保证输出平滑性
    - 避免直接求导带来的数值不稳定
    - 全局感受野捕获长程依赖
    """
    def __init__(
        self,
        in_dim,
        width=128,
        modes_list=(32, 16, 8),
        depth=4,
        time_emb_dim=64,
        pos_emb_dim=16,
        use_time=True
    ):
        super().__init__()
        self.in_dim = in_dim
        self.width = width
        self.use_time = use_time
        self.depth = depth
        self.pos_emb_dim = pos_emb_dim

        # 时间嵌入
        self.time_proj = GaussianFourierProjection(embed_dim=time_emb_dim) if use_time else None

        # 位置嵌入（打破节点维度上的平移不变性，让谱卷积有意义）
        assert pos_emb_dim % 2 == 0, "pos_emb_dim must be even"
        self.pos_freqs = nn.Parameter(torch.randn(pos_emb_dim // 2) * math.pi, requires_grad=False)

        # 输入投影
        input_dim = 1 + pos_emb_dim + (time_emb_dim if use_time else 0)  # 每个节点1个score + 位置编码
        self.in_proj = nn.Linear(input_dim, width)

        # FNO块堆叠
        self.blocks = nn.ModuleList()
        for i in range(depth):
            modes = modes_list[min(i, len(modes_list) - 1)]
            self.blocks.append(nn.Sequential(
                MultiScaleSpectral(width, modes_list=modes_list),
                FNOBlock(width, modes=modes)
            ))

        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, in_dim)

    def forward(self, score, time=None):
        """
        Args:
            score: (B, D) 分数估计
            time: (B,) 时间步 (可选)

        Returns:
            hessian_diag: (B, D) 预测的Hessian对角线
        """
        B, D = score.shape

        # 节点位置编码
        pos = torch.linspace(0, 1, D, device=score.device, dtype=score.dtype)  # (D,)
        pos_emb = pos.unsqueeze(-1) * self.pos_freqs  # (D, pos_emb_dim/2)
        pos_emb = torch.cat([pos_emb.sin(), pos_emb.cos()], dim=-1)  # (D, pos_emb_dim)
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)  # (B, D, pos_emb_dim)

        # 逐节点特征: score + 位置 + (可选时间)
        score_feat = score.unsqueeze(-1)  # (B, D, 1)
        feats = [score_feat, pos_emb]
        if self.use_time and time is not None:
            te = self.time_proj(time)  # (B, time_emb_dim)
            te = te.unsqueeze(1).expand(-1, D, -1)  # (B, D, time_emb_dim)
            feats.append(te)
        inp = torch.cat(feats, dim=-1)  # (B, D, input_dim)

        # 投影到隐藏空间
        h = self.in_proj(inp)  # (B, D, width)

        # FNO块处理
        for block in self.blocks:
            h = block(h)

        # 输出
        h = self.norm(h)
        out = self.out(h.mean(dim=1))  # 全局池化

        return out


class LightFNOOperator(nn.Module):
    """
    轻量级FNO算子 - 适用于高维数据

    相比完整版本：
    - 更少的层数和宽度
    - 单尺度谱卷积
    - 更激进的降维
    """
    def __init__(self, in_dim, width=64, modes=16, depth=2, time_emb_dim=32, pos_emb_dim=16, use_time=True):
        super().__init__()
        self.in_dim = in_dim
        self.width = width
        self.use_time = use_time
        self.pos_emb_dim = pos_emb_dim

        assert pos_emb_dim % 2 == 0, "pos_emb_dim must be even"
        self.pos_freqs = nn.Parameter(torch.randn(pos_emb_dim // 2) * math.pi, requires_grad=False)
        self.time_proj = GaussianFourierProjection(embed_dim=time_emb_dim) if use_time else None
        input_dim = 1 + pos_emb_dim + (time_emb_dim if use_time else 0)

        self.in_proj = nn.Linear(input_dim, width)

        self.blocks = nn.ModuleList([
            FNOBlock(width, modes=modes) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, in_dim)

    def forward(self, score, time=None):
        B, D = score.shape

        pos = torch.linspace(0, 1, D, device=score.device, dtype=score.dtype)
        pos_emb = pos.unsqueeze(-1) * self.pos_freqs
        pos_emb = torch.cat([pos_emb.sin(), pos_emb.cos()], dim=-1).unsqueeze(0).expand(B, -1, -1)

        score_feat = score.unsqueeze(-1)
        feats = [score_feat, pos_emb]
        if self.use_time and time is not None:
            te = self.time_proj(time).unsqueeze(1).expand(-1, D, -1)
            feats.append(te)
        inp = torch.cat(feats, dim=-1)

        h = self.in_proj(inp)

        for block in self.blocks:
            h = block(h)

        h = self.norm(h)
        return self.out(h.mean(dim=1))
