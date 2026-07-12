import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
class SimpleMIA(nn.Module):
    def __init__(self, input_dim):
        super(SimpleMIA, self).__init__()

        # 1. 第一层：加宽到 128，捕捉更多特征组合
        self.layer_1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)  # BatchNorm 加速收敛

        # 2. 第二层：逐渐降维
        self.layer_2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)

        # 3. 第三层 (新增)：增加非线性深度
        self.layer_3 = nn.Linear(64, 32)

        # 4. 输出层
        self.output_layer = nn.Linear(32, 1)

        # 激活函数
        self.relu = nn.ReLU()

        # Dropout: 每次随机丢弃 20% 的神经元，防止过拟合
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # Layer 1
        x = self.layer_1(x)
        x = self.bn1(x)  # 归一化
        x = self.relu(x)
        x = self.dropout(x)  # 丢弃

        # Layer 2
        x = self.layer_2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Layer 3
        x = self.layer_3(x)
        x = self.relu(x)

        # Output (注意：这里不加 Sigmoid，因为用了 BCEWithLogitsLoss)
        x = self.output_layer(x)
        return x


class GapGatedMIA(nn.Module):
    def __init__(self, input_dim):
        super(GapGatedMIA, self).__init__()

        # input_dim = 特征数 + Gap (即 N+1)
        self.feat_dim = input_dim - 1

        # [修改点 1] 门控网络输出维度 = input_dim (给所有特征+Gap都生成权重)
        self.gate_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),  # <--- 输出 N+1
            nn.Sigmoid()
        )

        # [修改点 2] 主网络输入维度 = input_dim (就是加权后的 x)
        self.main_net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, return_attention=False):
        # x: [Batch, N+1] (包含 Gap)
        gap = x[:, -1:]  # Gap 仍然作为控制信号提取出来

        # 1. 生成权重 (针对整个 x)
        weights = self.gate_net(gap)  # [Batch, N+1]

        # 2. 全局加权 (Gap 自己也被乘了权重)
        weighted_x = x * weights

        # 3. 输入主网络
        out = self.main_net(weighted_x)

        if return_attention:
            return out, weights  # weights 包含 Gap 的权重
        else:
            return out


class CrossAttnGapMIA(nn.Module):
    def __init__(self, input_dim, temperature=0.5):
        super(CrossAttnGapMIA, self).__init__()

        self.feat_dim = input_dim - 1
        self.attn_dim = input_dim  # 权重维度 = N+1
        self.temperature = temperature

        # Q: Gap 的意图 (输入 1, 输出 N+1)
        # Query 向量必须和 Key 向量维度一致才能做点积
        self.W_q = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, self.attn_dim)  # <--- 输出 N+1
        )

        # K: 特征的内容 (输入 N+1, 输出 N+1)
        # Key 现在看的是整个 x (包含 Gap)
        self.W_k = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.attn_dim)  # <--- 输出 N+1
        )

        # 主网络
        self.main_net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, return_attention=False):
        # x: [Batch, N+1]
        gap = x[:, -1:]

        # 1. QKV 计算
        # Query 来自 Gap
        Q = self.W_q(gap)  # [Batch, N+1]

        # Key 来自整个 x (包含 Gap)
        K = self.W_k(x)  # [Batch, N+1]

        # 2. 计算权重 (Scale)
        attn_logits = (Q * K) / np.sqrt(self.attn_dim)
        weights = torch.sigmoid(attn_logits / self.temperature)  # [Batch, N+1]

        # 3. 全局加权 (Gap 也被加权了)
        weighted_x = x * weights

        # 4. 主网络
        out = self.main_net(weighted_x)

        if return_attention:
            return out, weights
        else:
            return out
class AffineGapMIA(nn.Module):
    def __init__(self, input_dim, temperature=0.5):
        super(AffineGapMIA, self).__init__()

        self.attn_dim = input_dim # 维度 N+1
        self.temperature = temperature

        # Q: Gap 的意图 (输出 N+1)
        self.W_q = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, self.attn_dim)
        )

        # K: 特征的内容 (输入 N+1, 输出 N+1)
        self.W_k = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, self.attn_dim)
        )

        # Shift Network: (输入 1, 输出 N+1)
        # 给每个特征(包括Gap)生成一个偏置
        self.W_bias = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, self.attn_dim)
        )

        # 主网络
        self.main_net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x, return_attention=False):
        # x: [Batch, N+1]
        gap = x[:, -1:]

        # --- A. Scale (重要性) ---
        Q = self.W_q(gap)
        K = self.W_k(x) # Key 看整个 x
        attn_logits = (Q * K) / np.sqrt(self.attn_dim)
        scale = torch.sigmoid(attn_logits / self.temperature) # [Batch, N+1]

        # --- B. Shift (动态阈值) ---
        shift = self.W_bias(gap) # [Batch, N+1]

        # --- C. 全局仿射变换 ---
        # 核心公式: Output = x * Scale + Shift
        # Gap 自己也会变成: Gap * Scale_gap + Shift_gap
        transformed_x = x * scale + shift

        # 主网络
        out = self.main_net(transformed_x)

        if return_attention:
            return out, scale, shift # 这里的 scale/shift 包含 Gap 的分量
        else:
            return out
