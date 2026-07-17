"""
SlideSparse 权重转换脚本
========================

将 2:L 稀疏权重（如 2:8）通过滑动窗口机制转换为 2:4 硬件兼容格式。

稀疏格式说明：
-------------
- 2:8 表示：每8个连续元素中**至少有2个零**（稀疏度 ≥ 25%）
- 2:4 表示：每4个连续元素中**至少有2个零**（稀疏度 ≥ 50%）
- 转换后的权重天然满足 2:4 约束，可直接送入 cuSPARSELt 硬件加速

核心原理：
---------
SlideSparse 利用重叠滑动窗口将 2:L 稀疏模式映射到 2:4 硬件：
- 每 L 个输入元素为一组（Group）
- 每组使用步长为 2（Stride=2）的重叠窗口（Window=4）
- 每组输入 L 个元素，输出 (L/2 - 1) × 4 个元素

维度变化（以 2:8 为例，N=L/2=4）：
- 输入：  [out_features, in_features]
- 输出：  [out_features, in_features × (N-1)×4/L] = [out_features, in_features × 1.5]

映射示例（2:8 → 2:4）：
  Group 0 (输入 0-7):
    Window 0: input[0,1,2,3] → output[0,1,2,3]
    Window 1: input[2,3,4,5] → output[4,5,6,7]  
    Window 2: input[4,5,6,7] → output[8,9,10,11]
  Group 1 (输入 8-15): 重新开始（Group 边界重置）
    Window 0: input[8,9,10,11] → output[12,13,14,15]
    ...

目标层（与 prune 脚本一致）：
- wqkv: 分为 wq, wk, wv 三部分分别处理
- w13: 分为 w1, w3 两部分分别处理  
- w2, wo: 整体处理

使用示例：
---------
python convert_weights_slidesparse.py \\
    --input ./checkpoints/model_state_8I_pruned_2_8_magnitude.pt \\
    --Z 2 --L 8

依赖：
-----
- Python 3.8+
- PyTorch 2.0+
- 无其他重依赖（不需要 xformers、cuSPARSELt）
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass, asdict, field
import time

import torch
import numpy as np

# 尝试导入 numba 加速
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Warning: numba not installed, using slow Python implementation")


# ============================================================================
#                       Numba 加速核心函数
# ============================================================================

if HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _slide_greedy_allocation_numba(weight_np, out_np, 
                                       in_group_size, out_group_size, 
                                       num_windows, num_groups, k_out):
        """
        Numba 加速的贪婪残差分配算法
        
        Args:
            weight_np: 输入权重 [N, K_padded] (float32/float64)
            out_np: 输出权重 [N, K_out] (预分配，初始化为0)
            in_group_size: L (如 8)
            out_group_size: (N-1)*4 (如 12)
            num_windows: N-1 (如 3)
            num_groups: K_padded // L
            k_out: 输出维度
        """
        N_rows = weight_np.shape[0]
        MAX_NZ_PER_WINDOW = 2
        
        # 并行处理每一行
        for row_idx in prange(N_rows):
            # 分配掩码
            allocated = np.zeros(weight_np.shape[1], dtype=np.bool_)
            
            for g in range(num_groups):
                in_start = g * in_group_size
                out_start = g * out_group_size
                
                for w in range(num_windows):
                    win_in_start = in_start + w * 2
                    win_out_start = out_start + w * 4
                    nz_count = 0
                    
                    for lane in range(4):
                        in_pos = win_in_start + lane
                        out_pos = win_out_start + lane
                        
                        if out_pos >= k_out:
                            continue
                        
                        val = weight_np[row_idx, in_pos]
                        
                        if val != 0.0 and not allocated[in_pos] and nz_count < MAX_NZ_PER_WINDOW:
                            out_np[row_idx, out_pos] = val
                            allocated[in_pos] = True
                            nz_count += 1
                        # else: out_np already initialized to 0


# ============================================================================
#                           模型配置（与 prune 脚本保持一致）
# ============================================================================

@dataclass
class ModelArgs:
    """模型配置参数 - 与 model_allint8.py 中的 ModelArgs 保持一致"""
    dim: int = 2560
    n_layers: int = 30
    n_heads: int = 20
    n_kv_heads: int = 5
    vocab_size: int = 128256
    ffn_dim: int = 6912
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    use_kernel: bool = False


# ============================================================================
#                           SlideSparse 配置
# ============================================================================

@dataclass
class SlideSparseConfig:
    """
    SlideSparse 转换配置
    
    稀疏格式说明：
        2:L 表示每 L 个连续元素中至少有 2 个零
        例如 2:8 表示每 8 个元素至少 2 个零（稀疏度 ≥ 25%）
    
    Attributes:
        Z: 每组中至少的零元素数量（稀疏度分子），当前固定为 2
        L: 稀疏组的大小（稀疏度分母），如 8 表示 2:8 稀疏
        N: 内部参数，N = L // 2
        window_size: 滑动窗口大小，固定为 4（对应 2:4 硬件）
        stride: 滑动步长，固定为 2
        num_windows: 每组内的窗口数量，= N - 1
        expand_ratio: K 维度的扩展比例，= (num_windows × window_size) / L
    """
    Z: int
    L: int
    
    def __post_init__(self):
        # ===== 参数验证 =====
        if self.Z != 2:
            raise ValueError(f"当前仅支持 Z=2 的稀疏格式，收到 Z={self.Z}")
        if self.L % 2 != 0:
            raise ValueError(f"L 必须为偶数，收到 L={self.L}")
        if self.L < 4:
            raise ValueError(f"L 必须 >= 4，收到 L={self.L}")
        
        # ===== 派生参数计算 =====
        self.N = self.L // 2                           # 2:8 → N=4, 2:6 → N=3
        self.window_size = 4                           # 目标 2:4 硬件窗口大小
        self.stride = 2                                # 滑动步长（固定）
        self.num_windows = self.N - 1                  # 每组窗口数 = N-1
        self.expand_ratio = (self.num_windows * self.window_size) / self.L
        
        # ===== 输入/输出组大小 =====
        self.in_group_size = self.L                    # 每组输入元素数 = L
        self.out_group_size = self.num_windows * self.window_size  # 每组输出元素数 = (N-1)*4
    
    def __repr__(self):
        return (f"SlideSparseConfig(Z={self.Z}, L={self.L}, N={self.N}, "
                f"windows={self.num_windows}, expand_ratio={self.expand_ratio:.3f})")


# ============================================================================
#                           核心转换算法
# ============================================================================

def compute_output_k(k_in: int, config: SlideSparseConfig, align_to: int = 16) -> Tuple[int, int]:
    """
    计算滑动拓展后的 K 维度
    
    Args:
        k_in: 原始输入维度 K
        config: SlideSparse 配置
        align_to: 输出对齐要求（默认 16，满足 cuSPARSELt 要求）
    
    Returns:
        Tuple[k_padded, k_out]:
            - k_padded: padding 后的输入 K（必须是 L 的倍数）
            - k_out: 滑动拓展后的输出 K（对齐到 align_to）
    
    计算公式:
        k_padded = ceil(k_in / L) × L
        num_groups = k_padded / L
        k_out_raw = num_groups × (N-1) × 4
        k_out = ceil(k_out_raw / align_to) × align_to
    """
    L = config.L
    
    # Step 1: 将输入 K padding 到 L 的倍数
    k_padded = ((k_in + L - 1) // L) * L
    
    # Step 2: 计算组数
    num_groups = k_padded // L
    
    # Step 3: 计算原始输出大小
    k_out_raw = num_groups * config.out_group_size
    
    # Step 4: 对齐到 align_to
    k_out = ((k_out_raw + align_to - 1) // align_to) * align_to
    
    return k_padded, k_out


def build_slide_index_mapping(k_in: int, config: SlideSparseConfig) -> torch.Tensor:
    """
    构建滑动索引映射表（离线预计算）
    
    这是 SlideSparse 的核心：为每个输出位置计算对应的输入位置。
    
    Args:
        k_in: 原始输入维度 K
        config: SlideSparse 配置
    
    Returns:
        index_map: 形状 [k_out]，每个元素是对应的输入索引
                   -1 表示 padding 位置（填充 0）
    
    映射公式（对于每个输出索引 out_idx）：
        group_id = out_idx // OUT_GROUP_SIZE      # 当前属于第几组
        local_out = out_idx % OUT_GROUP_SIZE      # 组内的输出偏移
        local_block = local_out // 4              # 组内第几个窗口
        lane = local_out % 4                      # 窗口内的位置 (0-3)
        in_idx = group_id × IN_GROUP_SIZE + local_block × 2 + lane
    
    示例（2:8, N=4, IN_GROUP=8, OUT_GROUP=12）：
        out_idx=0:  group=0, local=0,  block=0, lane=0 → in=0×8+0×2+0=0
        out_idx=1:  group=0, local=1,  block=0, lane=1 → in=0×8+0×2+1=1
        out_idx=2:  group=0, local=2,  block=0, lane=2 → in=0×8+0×2+2=2
        out_idx=3:  group=0, local=3,  block=0, lane=3 → in=0×8+0×2+3=3
        out_idx=4:  group=0, local=4,  block=1, lane=0 → in=0×8+1×2+0=2  (重叠!)
        out_idx=5:  group=0, local=5,  block=1, lane=1 → in=0×8+1×2+1=3  (重叠!)
        out_idx=6:  group=0, local=6,  block=1, lane=2 → in=0×8+1×2+2=4
        out_idx=7:  group=0, local=7,  block=1, lane=3 → in=0×8+1×2+3=5
        out_idx=8:  group=0, local=8,  block=2, lane=0 → in=0×8+2×2+0=4  (重叠!)
        out_idx=9:  group=0, local=9,  block=2, lane=1 → in=0×8+2×2+1=5  (重叠!)
        out_idx=10: group=0, local=10, block=2, lane=2 → in=0×8+2×2+2=6
        out_idx=11: group=0, local=11, block=2, lane=3 → in=0×8+2×2+3=7
        --- Group 1 边界重置 ---
        out_idx=12: group=1, local=0,  block=0, lane=0 → in=1×8+0×2+0=8
        ...
    """
    # 计算输出维度
    k_padded, k_out = compute_output_k(k_in, config)
    
    # 提取配置参数
    IN_GROUP = config.in_group_size    # L
    OUT_GROUP = config.out_group_size  # (N-1)*4
    
    # 创建输出索引序列
    out_indices = torch.arange(k_out, dtype=torch.long)
    
    # ===== 核心映射公式 =====
    # 计算每个输出索引属于哪个组
    group_id = out_indices // OUT_GROUP
    
    # 计算组内的局部输出偏移
    local_out = out_indices % OUT_GROUP
    
    # 计算组内的窗口编号（每窗口4个元素）
    local_block = local_out // 4
    
    # 计算窗口内的位置（0-3）
    lane = local_out % 4
    
    # 计算对应的输入索引
    in_idx = group_id * IN_GROUP + local_block * 2 + lane
    
    # ===== 处理 padding =====
    # 超出 padding 后输入范围的索引标记为 -1
    in_idx = torch.where(in_idx < k_padded, in_idx, torch.tensor(-1, dtype=torch.long))
    
    # 如果输出 k_out 比原始计算的大（因为对齐），多余部分也标记为 -1
    num_groups = k_padded // IN_GROUP
    valid_out_size = num_groups * OUT_GROUP
    in_idx = torch.where(out_indices < valid_out_size, in_idx, torch.tensor(-1, dtype=torch.long))
    
    return in_idx


def slide_weight_tensor(weight: torch.Tensor, config: SlideSparseConfig, 
                        verbose: bool = False) -> Tuple[torch.Tensor, Dict]:
    """
    对整个权重张量执行滑动拓展 + 贪婪残差分配（Greedy Residual Allocation）
    
    核心思想：
        - 每个原始非零元素只在一个 Window 中出现
        - 重叠位置的元素在后续 Window 中置零
        - 这样保证每个 Window 都满足 2:4 稀疏约束
    
    贪婪分配策略：
        按窗口顺序遍历，对于每个位置：
        - 如果原始值是零 → 输出零
        - 如果原始值非零且未被之前窗口分配 → 输出该值，标记为已分配
        - 如果原始值非零但已被之前窗口分配 → 输出零
    
    Args:
        weight: 形状 [N, K] 的权重矩阵（N=out_features, K=in_features）
        config: SlideSparse 配置
        verbose: 是否打印调试信息
    
    Returns:
        Tuple[slided_weight, metadata]:
            - slided_weight: 形状 [N, K_out] 的滑动拓展后的权重
            - metadata: 包含转换信息的字典
    """
    N, K = weight.shape
    
    # Step 1: 计算输出维度并 padding 输入
    k_padded, k_out = compute_output_k(K, config)
    
    if verbose:
        print(f"    原始 K: {K}, Padding 后: {k_padded}, 输出 K: {k_out}")
        print(f"    扩展比例: {k_out / K:.4f} (理论值: {config.expand_ratio:.4f})")
    
    # Step 2: 对输入进行 padding（如果需要）
    if k_padded > K:
        padding_size = k_padded - K
        weight_padded = torch.cat([
            weight,
            torch.zeros(N, padding_size, dtype=weight.dtype, device=weight.device)
        ], dim=1)
    else:
        weight_padded = weight.clone()  # 需要 clone 因为后续会修改
    
    # Step 3: 贪婪残差分配（Greedy Residual Allocation）
    # 核心约束：
    #   1. 每个原始非零值只分配给一个 Window（由 allocated 掩码跟踪）
    #   2. 每个 Window 最多容纳 2 个非零值（2:4 约束）
    # 分配策略：
    #   按 Window 顺序遍历，在每个 Window 内按 lane 顺序分配，
    #   如果 Window 已满 2 个非零，剩余非零溢出到下一个 Window
    
    IN_GROUP = config.in_group_size    # L (如 8)
    OUT_GROUP = config.out_group_size  # (N-1)*4 (如 12)
    num_windows = config.num_windows   # N-1 (如 3)
    num_groups = k_padded // IN_GROUP
    
    # 创建输出张量
    slided_weight = torch.zeros(N, k_out, dtype=weight.dtype, device=weight.device)
    
    # 使用 Numba 加速（如果可用）
    if HAS_NUMBA:
        # 转换为 numpy 进行加速计算
        # 注意：FP8 (E5M2/E4M3) 不支持直接转 numpy，需要先转 float32
        original_dtype = weight.dtype
        weight_padded_fp32 = weight_padded.float()  # 转为 float32
        weight_np = weight_padded_fp32.numpy().astype(np.float64)
        out_np = np.zeros((N, k_out), dtype=np.float64)
        
        _slide_greedy_allocation_numba(
            weight_np, out_np,
            IN_GROUP, OUT_GROUP, num_windows, num_groups, k_out
        )
        
        slided_weight = torch.from_numpy(out_np).float().to(original_dtype)
    else:
        # 回退到纯 Python 实现（慢）
        for row_idx in range(N):
            row_weights = weight_padded[row_idx]
            allocated = torch.zeros(k_padded, dtype=torch.bool, device=weight.device)
            
            for g in range(num_groups):
                in_start = g * IN_GROUP
                out_start = g * OUT_GROUP
                
                for w in range(num_windows):
                    win_in_start = in_start + w * 2
                    win_out_start = out_start + w * 4
                    nz_count = 0
                    MAX_NZ_PER_WINDOW = 2
                    
                    for lane in range(4):
                        in_pos = win_in_start + lane
                        out_pos = win_out_start + lane
                        
                        if out_pos >= k_out:
                            continue
                        
                        val = row_weights[in_pos].item()
                        
                        if val != 0 and not allocated[in_pos].item() and nz_count < MAX_NZ_PER_WINDOW:
                            slided_weight[row_idx, out_pos] = val
                            allocated[in_pos] = True
                            nz_count += 1
    
    # Step 4: 构建元数据
    metadata = {
        "original_k": K,
        "padded_k": k_padded,
        "output_k": k_out,
        "expand_ratio": k_out / K,
    }
    
    return slided_weight, metadata


# ============================================================================
#                           2:4 稀疏验证
# ============================================================================

def verify_2to4_sparsity(tensor: torch.Tensor, tolerance: float = 0.0) -> Tuple[bool, float]:
    """
    验证张量是否满足 2:4 稀疏约束
    
    2:4 约束含义：每 4 个连续元素中**至少有 2 个零**
    
    Args:
        tensor: 需要验证的权重张量 [N, K]，K 应为 4 的倍数
        tolerance: 允许的违规组比例（默认 0 = 严格检查）
    
    Returns:
        Tuple[is_valid, violation_ratio]:
            - is_valid: 是否满足约束（考虑容忍度）
            - violation_ratio: 违规组的比例
    
    验证规则：
        每 4 个连续元素中，零的数量 >= 2（即非零数量 <= 2）
    """
    N, K = tensor.shape
    
    if K % 4 != 0:
        print(f"警告: K={K} 不是 4 的倍数，无法验证 2:4 稀疏")
        return False, 1.0
    
    # 重塑为 [N, K/4, 4] 进行分组检查
    grouped = tensor.view(N, K // 4, 4)
    
    # 统计每组的零元素数量
    zero_counts = (grouped == 0).sum(dim=2)  # [N, K/4]
    
    # 检查是否每组至少 2 个零（即零的数量 >= 2）
    violations = zero_counts < 2
    num_violations = violations.sum().item()
    total_groups = N * (K // 4)
    
    violation_ratio = num_violations / total_groups if total_groups > 0 else 0
    is_valid = violation_ratio <= tolerance
    
    return is_valid, violation_ratio


def analyze_sparsity_pattern(tensor: torch.Tensor, group_size: int = 8) -> Dict:
    """
    分析权重张量的稀疏模式统计
    
    Args:
        tensor: 权重张量 [N, K]
        group_size: 分组大小（默认 8，用于 2:8 分析）
    
    Returns:
        包含各种稀疏统计信息的字典
    """
    N, K = tensor.shape
    total_elements = N * K
    
    # 基本统计
    zero_count = (tensor == 0).sum().item()
    global_sparsity = zero_count / total_elements
    
    # 指定组大小分析
    group_stats = {}
    if K % group_size == 0:
        grouped = tensor.view(N, K // group_size, group_size)
        zeros_per_group = (grouped == 0).sum(dim=2)  # [N, K/group_size]
        
        # 统计满足 2:L 约束的组数（至少 2 个零）
        valid_groups = (zeros_per_group >= 2).sum().item()
        total_groups = N * (K // group_size)
        group_stats[f"2:{group_size}_valid_ratio"] = valid_groups / total_groups if total_groups > 0 else 0
        
        # 统计各种零数量的分布
        for nz in range(group_size + 1):
            count = (zeros_per_group == nz).sum().item()
            group_stats[f"groups_with_{nz}_zeros"] = count
    
    # 2:4 组分析
    if K % 4 == 0:
        grouped_4 = tensor.view(N, K // 4, 4)
        zeros_per_group_4 = (grouped_4 == 0).sum(dim=2)
        valid_2to4 = (zeros_per_group_4 >= 2).sum().item()
        total_groups_4 = N * (K // 4)
        group_stats["2:4_valid_ratio"] = valid_2to4 / total_groups_4 if total_groups_4 > 0 else 0
    
    return {
        "shape": (N, K),
        "total_elements": total_elements,
        "zero_count": zero_count,
        "global_sparsity": global_sparsity,
        "group_analysis": group_stats
    }


# ============================================================================
#                           模型转换主流程
# ============================================================================

def is_target_layer(key: str) -> bool:
    """
    判断 key 是否为目标层（需要进行 SlideSparse 转换）
    
    目标层（与 prune 脚本一致）：
        - wqkv: QKV 投影层
        - w13: FFN gate/up 投影层
        - w2: FFN down 投影层
        - wo: 注意力输出投影层
    """
    target_patterns = ['wqkv', 'w13', 'w2', 'wo']
    key_lower = key.lower()
    
    # 必须包含目标模式之一
    for pattern in target_patterns:
        if pattern in key_lower:
            # 确保是权重（不是 scale 等）
            if 'weight' in key_lower and 'scale' not in key_lower:
                return True
    
    return False


def convert_checkpoint(
    input_path: str,
    output_path: str,
    slide_config: SlideSparseConfig,
    model_config: ModelArgs,
    verify: bool = True,
    verbose: bool = True
) -> Dict:
    """
    转换整个模型检查点为 SlideSparse 格式
    
    处理逻辑（与 prune 脚本一致）：
        - wqkv: 分为 wq, wk, wv 三部分分别处理后拼接
        - w13: 分为 w1, w3 两部分分别处理后拼接
        - w2, wo: 整体处理
        - 其他层: 直接复制
    
    Args:
        input_path: 输入检查点路径（.pt 文件）
        output_path: 输出检查点路径
        slide_config: SlideSparse 配置
        model_config: 模型配置
        verify: 是否验证转换后的 2:4 稀疏性
        verbose: 是否打印详细信息
    
    Returns:
        转换报告字典
    """
    start_time = time.time()
    
    if verbose:
        print("=" * 70)
        print("SlideSparse 权重转换")
        print("=" * 70)
        print(f"配置: {slide_config}")
        print(f"输入: {input_path}")
        print(f"输出: {output_path}")
        print("-" * 70)
    
    # 加载检查点
    if verbose:
        print("加载检查点...")
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    
    # 准备输出
    output_checkpoint = {}
    conversion_report = {
        "slide_config": {
            "Z": slide_config.Z,
            "L": slide_config.L,
            "N": slide_config.N,
            "expand_ratio": slide_config.expand_ratio
        },
        "model_config": asdict(model_config),
        "converted_layers": [],
        "skipped_layers": [],
        "verification_results": {}
    }
    
    # 用于计算 wqkv 分割的参数
    dim = model_config.dim
    n_heads = model_config.n_heads
    n_kv_heads = model_config.n_kv_heads
    ffn_dim = model_config.ffn_dim
    
    # wqkv 的分割点
    wq_size = dim
    wk_size = dim // n_heads * n_kv_heads
    wv_size = dim // n_heads * n_kv_heads
    
    # 遍历所有张量
    for key, value in checkpoint.items():
        # ===== 非张量直接复制 =====
        if not torch.is_tensor(value):
            output_checkpoint[key] = value
            continue
        
        # ===== 非 2D 张量直接复制 =====
        if value.dim() != 2:
            output_checkpoint[key] = value
            conversion_report["skipped_layers"].append({
                "key": key,
                "reason": "not_2d",
                "shape": list(value.shape)
            })
            continue
        
        # ===== 检查是否为目标层 =====
        if not is_target_layer(key):
            output_checkpoint[key] = value
            if verbose:
                print(f"跳过 (非目标层): {key}, shape={list(value.shape)}")
            conversion_report["skipped_layers"].append({
                "key": key,
                "reason": "not_target",
                "shape": list(value.shape)
            })
            continue
        
        # ===== 执行 SlideSparse 转换 =====
        if verbose:
            print(f"\n转换: {key}")
            print(f"  输入 shape: {list(value.shape)}")
        
        # 根据层类型进行不同处理
        if 'wqkv' in key:
            # ===== wqkv 层：分为 wq, wk, wv 分别处理 =====
            wq = value[:wq_size]
            wk = value[wq_size:wq_size + wk_size]
            wv = value[wq_size + wk_size:]
            
            if verbose:
                print(f"  wqkv 分割: wq={list(wq.shape)}, wk={list(wk.shape)}, wv={list(wv.shape)}")
            
            # 分别进行 slide 转换
            wq_slided, _ = slide_weight_tensor(wq, slide_config, verbose=verbose)
            wk_slided, _ = slide_weight_tensor(wk, slide_config, verbose=verbose)
            wv_slided, meta = slide_weight_tensor(wv, slide_config, verbose=verbose)
            
            # 拼接结果
            slided_weight = torch.cat([wq_slided, wk_slided, wv_slided], dim=0)
            
        elif 'w13' in key:
            # ===== w13 层：分为 w1, w3 分别处理 =====
            w1 = value[:ffn_dim]
            w3 = value[ffn_dim:]
            
            if verbose:
                print(f"  w13 分割: w1={list(w1.shape)}, w3={list(w3.shape)}")
            
            # 分别进行 slide 转换
            w1_slided, _ = slide_weight_tensor(w1, slide_config, verbose=verbose)
            w3_slided, meta = slide_weight_tensor(w3, slide_config, verbose=verbose)
            
            # 拼接结果
            slided_weight = torch.cat([w1_slided, w3_slided], dim=0)
            
        elif 'w2' in key or 'wo' in key:
            # ===== w2/wo 层：整体处理 =====
            slided_weight, meta = slide_weight_tensor(value, slide_config, verbose=verbose)
        
        else:
            # 不应该到达这里
            output_checkpoint[key] = value
            continue
        
        if verbose:
            print(f"  输出 shape: {list(slided_weight.shape)}")
        
        # 验证 2:4 稀疏性
        if verify:
            is_valid, violation_ratio = verify_2to4_sparsity(slided_weight)
            if verbose:
                status = "✓ 通过" if is_valid else f"✗ 失败 (违规率: {violation_ratio:.2%})"
                print(f"  2:4 验证: {status}")
            
            conversion_report["verification_results"][key] = {
                "is_valid": is_valid,
                "violation_ratio": violation_ratio
            }
        
        # 保存转换后的权重
        output_checkpoint[key] = slided_weight
        
        # 记录转换信息
        conversion_report["converted_layers"].append({
            "key": key,
            "original_shape": list(value.shape),
            "output_shape": list(slided_weight.shape),
        })
    
    # 保存输出检查点
    if verbose:
        print("\n" + "-" * 70)
        print(f"保存到: {output_path}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    torch.save(output_checkpoint, output_path)
    
    # 保存转换报告（JSON 格式）
    report_path = output_path.replace(".pt", "_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(conversion_report, f, indent=2, ensure_ascii=False)
    
    elapsed_time = time.time() - start_time
    
    if verbose:
        print(f"转换报告: {report_path}")
        print("=" * 70)
        print(f"转换完成! 耗时: {elapsed_time:.2f}s")
        print(f"  转换层数: {len(conversion_report['converted_layers'])}")
        print(f"  跳过层数: {len(conversion_report['skipped_layers'])}")
        
        if verify:
            failed = [k for k, v in conversion_report["verification_results"].items() if not v["is_valid"]]
            passed = len(conversion_report["verification_results"]) - len(failed)
            print(f"  2:4 验证: {passed} 通过, {len(failed)} 失败")
    
    return conversion_report


# ============================================================================
#                           命令行接口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SlideSparse 权重转换工具 - 将 2:L 稀疏权重转换为 2:4 硬件兼容格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
稀疏格式说明:
  2:L 表示每 L 个连续元素中至少有 2 个零
  例如：2:8 表示每 8 个元素至少 2 个零（稀疏度 ≥ 25%）

示例:
  # 转换 2:8 稀疏模型
  python convert_weights_slidesparse.py \\
      --input ./checkpoints/model_pruned_2_8.pt \\
      --Z 2 --L 8

  # 转换 2:6 稀疏模型（不验证）
  python convert_weights_slidesparse.py \\
      --input ./checkpoints/model_pruned_2_6.pt \\
      --Z 2 --L 6 --no-verify

支持的稀疏格式和扩展比例:
  2:4  (N=2, expand_ratio=1.0)   - 直接兼容，无需 slide
  2:6  (N=3, expand_ratio=1.33)  - 需要 slide
  2:8  (N=4, expand_ratio=1.5)   - 需要 slide
  2:10 (N=5, expand_ratio=1.6)   - 需要 slide
  2:12 (N=6, expand_ratio=1.67)  - 需要 slide
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入检查点路径 (.pt 文件)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出检查点路径（默认: 输入文件名_slidesparse.pt）"
    )
    
    parser.add_argument(
        "--Z",
        type=int,
        default=2,
        help="稀疏度分子 - 每组至少零元素数（当前仅支持 Z=2）"
    )
    
    parser.add_argument(
        "--L",
        type=int,
        required=True,
        help="稀疏度分母 - 稀疏组大小（如 4, 6, 8, 10, 12）"
    )
    
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="跳过 2:4 稀疏性验证"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="静默模式（减少输出）"
    )
    
    args = parser.parse_args()
    
    # 构建输出路径
    if args.output is None:
        input_stem = Path(args.input).stem
        input_dir = Path(args.input).parent
        args.output = str(input_dir / f"{input_stem}_slidesparse_{args.Z}_{args.L}.pt")
    
    # 创建配置
    slide_config = SlideSparseConfig(Z=args.Z, L=args.L)
    model_config = ModelArgs()
    
    if not args.quiet:
        print(f"Model config: {asdict(model_config)}")
    
    # 执行转换
    try:
        report = convert_checkpoint(
            input_path=args.input,
            output_path=args.output,
            slide_config=slide_config,
            model_config=model_config,
            verify=not args.no_verify,
            verbose=not args.quiet
        )
        
        # 检查是否有验证失败
        if not args.no_verify:
            failed_layers = [
                k for k, v in report.get("verification_results", {}).items()
                if not v.get("is_valid", True)
            ]
            if failed_layers:
                print(f"\n警告: 以下 {len(failed_layers)} 层未通过 2:4 验证:")
                for layer in failed_layers[:10]:  # 只显示前10个
                    ratio = report["verification_results"][layer]["violation_ratio"]
                    print(f"  - {layer} (违规率: {ratio:.2%})")
                if len(failed_layers) > 10:
                    print(f"  ... 还有 {len(failed_layers) - 10} 层")
                return 1
        
        return 0
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
