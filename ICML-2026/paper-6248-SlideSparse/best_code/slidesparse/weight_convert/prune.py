#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse 剪枝脚本

对权重应用 Z:L 稀疏约束（如 2:8），支持两种输入模式：

1. FP8/INT8 量化模型（已量化）:
   - 仅执行剪枝，保持原有数据类型
   - 通过 magnitude 或 random 模式选择要剪枝的元素
   
2. BF16 Dense 模型（BitNet 场景）:
   - 执行 Quant + Prune 融合操作
   - 先量化为三元值 (-1, 0, 1)，利用量化产生的零
   - 再应用 Z:L 稀疏约束
   - 输出为指定格式（INT8 或 FP8）

Usage:
    # FP8 模型剪枝
    python prune.py --input /path/to/model --Z 2 --L 8 --mode magnitude
    
    # BF16 模型量化+剪枝（BitNet）
    python prune.py --input /path/to/model --Z 2 --L 8 --bitnet --output-dtype int8
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import torch

# 添加项目路径
_SCRIPT_DIR = Path(__file__).parent
_SLIDESPARSE_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _SLIDESPARSE_ROOT.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 从顶层导入模型名称处理工具
from slidesparse.utils import model_base_name

# 导入本地工具
from utils import (
    SlideSparseConfig,
    is_target_layer,
    get_layer_type,
    load_safetensors,
    save_safetensors,
    get_model_safetensors_path,
    get_all_safetensors_files,
    detect_weight_dtype,
    get_torch_dtype,
    verify_ZL_sparsity,
    print_header,
    print_info,
    print_success,
    print_warning,
    print_error,
    CHECKPOINT_DIR,
    CHECKPOINT_SLIDESPARSE_DIR,
)


# =============================================================================
# 剪枝核心函数
# =============================================================================

def build_prune_mask_magnitude(
    weight: torch.Tensor,
    Z: int,
    L: int,
) -> torch.Tensor:
    """
    基于幅度（magnitude）构建剪枝掩码
    
    策略：在每 L 个元素的组内，保留 L-Z 个幅度最大的元素
    
    Args:
        weight: 权重张量 [N, K]
        Z: 每组至少剪掉的元素数
        L: 组大小
    
    Returns:
        剪枝掩码，True 表示需要剪掉
    """
    N, K = weight.shape
    
    # Padding
    pad_cols = (L - (K % L)) % L
    if pad_cols > 0:
        weight_padded = torch.cat([
            weight,
            torch.zeros(N, pad_cols, dtype=weight.dtype, device=weight.device)
        ], dim=1)
    else:
        weight_padded = weight
    
    # 重塑为 [num_groups, L]
    grouped = weight_padded.view(-1, L)
    num_groups = grouped.shape[0]
    
    # 计算幅度（转换为 float 处理 FP8/INT8）
    importance = grouped.float().abs()
    
    # 找到每组中幅度最小的 Z 个位置
    _, prune_indices = importance.topk(k=Z, dim=1, largest=False)
    
    # 构建掩码
    prune_mask = torch.zeros_like(grouped, dtype=torch.bool)
    batch_indices = torch.arange(num_groups, device=weight.device).unsqueeze(1).expand(-1, Z)
    prune_mask[batch_indices, prune_indices] = True
    
    # 恢复形状并去除 padding
    prune_mask = prune_mask.view(N, -1)
    if pad_cols > 0:
        prune_mask = prune_mask[:, :K]
    
    return prune_mask


def build_prune_mask_random(
    weight: torch.Tensor,
    Z: int,
    L: int,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    基于随机选择构建剪枝掩码
    
    Args:
        weight: 权重张量 [N, K]
        Z: 每组至少剪掉的元素数
        L: 组大小
        seed: 随机种子
    
    Returns:
        剪枝掩码，True 表示需要剪掉
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    N, K = weight.shape
    
    # Padding
    pad_cols = (L - (K % L)) % L
    if pad_cols > 0:
        weight_padded = torch.cat([
            weight,
            torch.zeros(N, pad_cols, dtype=weight.dtype, device=weight.device)
        ], dim=1)
        K_padded = K + pad_cols
    else:
        weight_padded = weight
        K_padded = K
    
    # 重塑为 [num_groups, L]
    grouped = weight_padded.view(-1, L)
    num_groups = grouped.shape[0]
    
    # 为每组生成随机分数
    rand_scores = torch.rand(num_groups, L, device=weight.device)
    
    # 找到随机分数最小的 Z 个位置
    _, prune_indices = rand_scores.topk(k=Z, dim=1, largest=False)
    
    # 构建掩码
    prune_mask = torch.zeros_like(grouped, dtype=torch.bool)
    batch_indices = torch.arange(num_groups, device=weight.device).unsqueeze(1).expand(-1, Z)
    prune_mask[batch_indices, prune_indices] = True
    
    # 恢复形状并去除 padding
    prune_mask = prune_mask.view(N, -1)
    if pad_cols > 0:
        prune_mask = prune_mask[:, :K]
    
    return prune_mask


def prune_tensor(
    weight: torch.Tensor,
    Z: int,
    L: int,
    mode: str = "magnitude",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    对张量应用 Z:L 稀疏剪枝
    
    Args:
        weight: 权重张量 [N, K]
        Z: 每组至少剪掉的元素数
        L: 组大小
        mode: 剪枝模式 ("magnitude" 或 "random")
        seed: 随机种子（仅用于 random 模式）
    
    Returns:
        剪枝后的权重
    """
    if mode == "magnitude":
        prune_mask = build_prune_mask_magnitude(weight, Z, L)
    elif mode == "random":
        prune_mask = build_prune_mask_random(weight, Z, L, seed)
    else:
        raise ValueError(f"Unknown prune mode: {mode}")
    
    # 应用剪枝
    original_dtype = weight.dtype
    if original_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # FP8 不支持直接的布尔索引赋值，需要先转为 float
        pruned_weight = weight.float()
        pruned_weight[prune_mask] = 0
        pruned_weight = pruned_weight.to(original_dtype)
    else:
        pruned_weight = weight.clone()
        pruned_weight[prune_mask] = 0
    
    return pruned_weight


# =============================================================================
# BitNet 量化 + 剪枝融合
# =============================================================================

def _build_prune_mask_with_ternary(
    ternary_grouped: torch.Tensor,
    original_grouped: torch.Tensor,
    Z: int,
    L: int,
    mode: str,
) -> torch.Tensor:
    """
    基于 Ternary 权重的零分布，利用 Original 权重的幅度信息，生成剪枝掩码
    
    核心思想：
    1. 优先利用量化过程自然产生的零
    2. 如果组内零不足 Z 个，再根据幅度/随机选择额外剪枝
    """
    # 统计 Ternary 中的非零情况
    nonzero_mask = (ternary_grouped != 0)
    nonzero_count = nonzero_mask.sum(dim=1)
    
    # 计算需要额外剪掉的数量
    # 目标：每组最多保留 L-Z 个非零值
    prune_count = (nonzero_count - (L - Z)).clamp(min=0)
    
    max_prune = int(Z)
    if max_prune == 0 or prune_count.max() == 0:
        return torch.zeros_like(ternary_grouped, dtype=torch.bool)
    
    # 确定剪枝候选位置
    if mode == "magnitude":
        importance = original_grouped.float().abs()
        # 已经为 0 的位置不需要再剪，设为无限大防止被选中
        importance[~nonzero_mask] = float("inf")
        _, candidate_idx = importance.topk(k=max_prune, dim=1, largest=False)
    else:  # random
        rand_scores = torch.rand(ternary_grouped.shape, device=ternary_grouped.device)
        rand_scores[~nonzero_mask] = float("inf")
        _, candidate_idx = rand_scores.topk(k=max_prune, dim=1, largest=False)
    
    # 生成掩码
    prune_mask = torch.zeros_like(ternary_grouped, dtype=torch.bool)
    for k in range(max_prune):
        # 只对需要剪掉 > k 个元素的组进行操作
        needs = prune_count > k
        if needs.any():
            rows = torch.where(needs)[0]
            cols = candidate_idx[needs, k]
            prune_mask[rows, cols] = True
    
    return prune_mask


def quant_and_prune_tensor_bitnet(
    weight: torch.Tensor,
    Z: int,
    L: int,
    mode: str = "magnitude",
    output_dtype: str = "int8",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    BitNet 量化 + 剪枝融合
    
    流程：
    1. 计算 Scale（基于原始 Dense 权重）
    2. 量化为三元值 (-1, 0, 1)
    3. 应用 Z:L 稀疏约束
    4. 转换为指定输出格式
    
    Args:
        weight: BF16 Dense 权重 [N, K]
        Z: 每组至少零元素数
        L: 组大小
        mode: 剪枝模式
        output_dtype: 输出格式 ("int8", "fp8_e4m3", "fp8_e5m2")
    
    Returns:
        (pruned_weight, scale)
    """
    N, K = weight.shape
    
    # Step 1: 计算 Scale 并量化为三元值
    weight_float = weight.float()
    scale = 1.0 / weight_float.abs().mean().clamp(min=1e-5)
    ternary = (weight_float * scale).round().clamp(-1, 1)
    
    # Step 2: Padding 并分组
    pad_cols = (L - (K % L)) % L
    if pad_cols > 0:
        ternary_padded = torch.cat([
            ternary,
            torch.zeros(N, pad_cols, dtype=ternary.dtype, device=ternary.device)
        ], dim=1)
        original_padded = torch.cat([
            weight_float,
            torch.zeros(N, pad_cols, dtype=weight_float.dtype, device=weight_float.device)
        ], dim=1)
    else:
        ternary_padded = ternary
        original_padded = weight_float
    
    ternary_grouped = ternary_padded.view(-1, L)
    original_grouped = original_padded.view(-1, L)
    
    # Step 3: 应用剪枝（当 L <= Z 时跳过，表示纯量化模式）
    # 例如 Z=2, L=2 时，不做额外剪枝，只保留量化结果
    if L > Z:
        prune_mask = _build_prune_mask_with_ternary(
            ternary_grouped, original_grouped, Z, L, mode
        )
        
        if prune_mask.any():
            ternary_grouped[prune_mask] = 0
    
    # 恢复形状
    ternary_final = ternary_grouped.view(N, -1)
    if pad_cols > 0:
        ternary_final = ternary_final[:, :K]
    
    # Step 4: 转换输出格式
    scale_tensor = (1.0 / scale).to(torch.bfloat16).reshape(1)
    
    if output_dtype == "int8":
        return ternary_final.to(torch.int8), scale_tensor
    elif output_dtype == "fp8_e4m3":
        return ternary_final.to(torch.float8_e4m3fn), scale_tensor
    elif output_dtype == "fp8_e5m2":
        return ternary_final.to(torch.float8_e5m2), scale_tensor
    else:
        raise ValueError(f"Unsupported output dtype: {output_dtype}")


# =============================================================================
# 模型处理
# =============================================================================

def prune_safetensors(
    input_path: Path,
    output_path: Path,
    config: SlideSparseConfig,
    mode: str = "magnitude",
    bitnet_mode: bool = False,
    output_dtype: Optional[str] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    对 safetensors 文件执行剪枝
    
    Args:
        input_path: 输入 safetensors 文件路径
        output_path: 输出 safetensors 文件路径
        config: SlideSparse 配置
        mode: 剪枝模式
        bitnet_mode: 是否使用 BitNet 量化+剪枝模式
        output_dtype: BitNet 模式下的输出格式
        seed: 随机种子
        verbose: 是否打印详细信息
    
    Returns:
        处理报告
    """
    if verbose:
        print_info(f"Loading {input_path}")
    
    # 加载权重
    weights = load_safetensors(input_path)
    
    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "config": config.to_dict(),
        "mode": mode,
        "bitnet_mode": bitnet_mode,
        "processed_layers": [],
        "skipped_layers": [],
    }
    
    output_weights = {}
    
    for key, tensor in weights.items():
        # 检查是否为目标层
        if not is_target_layer(key):
            output_weights[key] = tensor
            report["skipped_layers"].append(key)
            continue
        
        # 检查是否为 2D 张量
        if tensor.dim() != 2:
            output_weights[key] = tensor
            report["skipped_layers"].append(key)
            continue
        
        if verbose:
            dtype_str = detect_weight_dtype(tensor)
            print_info(f"Processing {key}: shape={list(tensor.shape)}, dtype={dtype_str}")
        
        # 执行剪枝
        if bitnet_mode:
            # BitNet 模式：量化 + 剪枝
            if output_dtype is None:
                output_dtype = "int8"
            
            pruned_tensor, scale = quant_and_prune_tensor_bitnet(
                tensor, config.Z, config.L, mode, output_dtype
            )
            output_weights[key] = pruned_tensor
            
            # 保存 scale
            scale_key = key.replace(".weight", ".weight_scale")
            if ".weight" not in key:
                scale_key = key + "_scale"
            output_weights[scale_key] = scale
        else:
            # 普通模式：仅剪枝
            pruned_tensor = prune_tensor(
                tensor, config.Z, config.L, mode, seed
            )
            output_weights[key] = pruned_tensor
        
        # 验证剪枝结果
        is_valid, valid_ratio = verify_ZL_sparsity(
            pruned_tensor.float() if pruned_tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.int8) else pruned_tensor,
            config.Z, config.L
        )
        
        layer_info = {
            "key": key,
            "shape": list(pruned_tensor.shape),
            "dtype": detect_weight_dtype(pruned_tensor),
            "valid_ratio": valid_ratio,
            "is_valid": is_valid,
        }
        report["processed_layers"].append(layer_info)
        
        if verbose:
            status = "✓" if is_valid else f"✗ ({valid_ratio:.2%})"
            print_info(f"  {config.Z}:{config.L} validation: {status}")
    
    # 保存
    if verbose:
        print_info(f"Saving to {output_path}")
    
    save_safetensors(output_weights, output_path)
    
    return report


# =============================================================================
# 命令行接口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SlideSparse 剪枝工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入模型目录或 safetensors 文件",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出路径（默认: 输入路径 + _pruned 后缀）",
    )
    parser.add_argument(
        "--Z",
        type=int,
        default=2,
        help="每组至少零元素数（默认: 2）",
    )
    parser.add_argument(
        "--L",
        type=int,
        required=True,
        help="组大小（如 6, 8, 10）",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["magnitude", "random"],
        default="magnitude",
        help="剪枝模式（默认: magnitude）",
    )
    parser.add_argument(
        "--bitnet",
        action="store_true",
        help="启用 BitNet 量化+剪枝模式（用于 BF16 Dense 权重）",
    )
    parser.add_argument(
        "--output-dtype",
        type=str,
        choices=["int8", "fp8_e4m3", "fp8_e5m2"],
        default="int8",
        help="BitNet 模式下的输出格式（默认: int8）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（用于 random 模式）",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="静默模式",
    )
    
    args = parser.parse_args()
    
    # 创建配置
    config = SlideSparseConfig(Z=args.Z, L=args.L)
    
    input_path = Path(args.input)
    
    # 确定输入文件
    if input_path.is_dir():
        safetensors_files = get_all_safetensors_files(input_path)
        if not safetensors_files:
            print_error(f"No safetensors files found in {input_path}")
            return 1
    else:
        safetensors_files = [input_path]
    
    # 确定输出路径
    # 命名规范: {ModelName}_{mode}_Z{Z}L{L}_{dtype}
    # 例如: BitNet-2B_mag_Z2L8_INT8
    if args.output:
        output_base = Path(args.output)
    else:
        # 从输入路径提取模型名，使用顶层工具去除量化后缀
        raw_name = input_path.name if input_path.is_dir() else input_path.stem
        base_name = model_base_name(raw_name)
        
        # 构建输出目录名
        mode_short = args.mode[:3]  # "mag" or "ran"
        dtype_tag = args.output_dtype.upper().replace("_", "") if args.bitnet else "PRUNED"
        output_name = f"{base_name}_{mode_short}_Z{args.Z}L{args.L}_{dtype_tag}"
        
        # 放到 checkpoints_slidesparse 目录
        output_base = CHECKPOINT_SLIDESPARSE_DIR / output_name
    
    if not args.quiet:
        print_header(f"SlideSparse Pruning: {config.Z}:{config.L}")
        print_info(f"Mode: {args.mode}")
        print_info(f"BitNet: {args.bitnet}")
        print()
    
    # 处理每个文件
    for sf_path in safetensors_files:
        if len(safetensors_files) == 1 and not input_path.is_dir():
            out_path = output_base
        else:
            out_path = output_base / sf_path.name if output_base.suffix != ".safetensors" else output_base
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = prune_safetensors(
            sf_path,
            out_path,
            config,
            mode=args.mode,
            bitnet_mode=args.bitnet,
            output_dtype=args.output_dtype if args.bitnet else None,
            seed=args.seed,
            verbose=not args.quiet,
        )
        
        if not args.quiet:
            processed = len(report["processed_layers"])
            skipped = len(report["skipped_layers"])
            valid = sum(1 for l in report["processed_layers"] if l["is_valid"])
            print_success(f"Processed: {processed}, Valid: {valid}, Skipped: {skipped}")
            print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
