#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse 滑动扩展脚本

将 Z:L 稀疏权重通过滑动窗口机制转换为 2:4 硬件兼容格式。

核心原理：
- SlideSparse 利用重叠滑动窗口将 Z:L 稀疏模式映射到 2:4 硬件
- 每 L 个输入元素为一组（Group）
- 每组使用步长为 2（Stride=2）的重叠窗口（Window=4）
- 每组输入 L 个元素，输出 (N-1) × 4 个元素，其中 N = L/2

维度变化（以 2:8 为例，N=4）：
- 输入：  [out_features, in_features]
- 输出：  [out_features, in_features × (N-1)×4/L] = [out_features, in_features × 1.5]

Usage:
    python slide.py --input /path/to/pruned --output /path/to/slided --Z 2 --L 8
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import torch
import numpy as np

# 导入本地工具
from utils import (
    SlideSparseConfig,
    is_target_layer,
    load_safetensors,
    save_safetensors,
    get_all_safetensors_files,
    detect_weight_dtype,
    compute_output_k,
    verify_2to4_sparsity,
    print_header,
    print_info,
    print_success,
    print_warning,
    print_error,
    HAS_NUMBA,
)

# 尝试导入 numba 加速
if HAS_NUMBA:
    from numba import njit, prange


# =============================================================================
# Numba 加速核心函数
# =============================================================================

if HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _slide_greedy_allocation_numba(
        weight_np: np.ndarray,
        out_np: np.ndarray,
        in_group_size: int,
        out_group_size: int,
        num_windows: int,
        num_groups: int,
        k_out: int,
    ) -> None:
        """
        Numba 加速的贪婪残差分配算法
        
        核心约束：
        1. 每个原始非零值只分配给一个 Window
        2. 每个 Window 最多容纳 2 个非零值（2:4 约束）
        
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
            # 分配掩码：跟踪哪些输入位置已被分配
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
                        
                        # 贪婪分配：如果非零、未分配、且窗口未满
                        if val != 0.0 and not allocated[in_pos] and nz_count < MAX_NZ_PER_WINDOW:
                            out_np[row_idx, out_pos] = val
                            allocated[in_pos] = True
                            nz_count += 1
                        # else: out_np already initialized to 0


def _slide_greedy_allocation_python(
    weight: torch.Tensor,
    k_out: int,
    config: SlideSparseConfig,
) -> torch.Tensor:
    """
    纯 Python 实现的贪婪残差分配（慢，用于回退）
    """
    N, K_padded = weight.shape
    device = weight.device
    dtype = weight.dtype
    
    IN_GROUP = config.in_group_size
    OUT_GROUP = config.out_group_size
    num_windows = config.num_windows
    num_groups = K_padded // IN_GROUP
    
    slided_weight = torch.zeros(N, k_out, dtype=dtype, device=device)
    
    for row_idx in range(N):
        row_weights = weight[row_idx]
        allocated = torch.zeros(K_padded, dtype=torch.bool, device=device)
        
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
    
    return slided_weight


# =============================================================================
# 滑动核心函数
# =============================================================================

def slide_tensor(
    weight: torch.Tensor,
    config: SlideSparseConfig,
    align_to: int = 32,
    verbose: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    对权重张量执行滑动扩展 + 贪婪残差分配
    
    Args:
        weight: 形状 [N, K] 的权重矩阵
        config: SlideSparse 配置
        align_to: 输出对齐要求
        verbose: 是否打印调试信息
    
    Returns:
        (slided_weight, metadata)
    """
    N, K = weight.shape
    original_dtype = weight.dtype
    device = weight.device
    
    # Step 1: 计算输出维度并 padding 输入
    k_padded, k_out = compute_output_k(K, config, align_to)
    
    if verbose:
        print_info(f"  Original K: {K}, Padded: {k_padded}, Output K: {k_out}")
        print_info(f"  Expand ratio: {k_out / K:.4f} (theoretical: {config.expand_ratio:.4f})")
    
    # Step 2: 对输入进行 padding
    if k_padded > K:
        padding_size = k_padded - K
        # 需要先转换为 float 进行 cat（FP8 不支持 cat）
        if original_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            weight_float = weight.float()
            weight_padded = torch.cat([
                weight_float,
                torch.zeros(N, padding_size, dtype=torch.float32, device=device)
            ], dim=1)
        else:
            weight_padded = torch.cat([
                weight,
                torch.zeros(N, padding_size, dtype=original_dtype, device=device)
            ], dim=1)
    else:
        if original_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            weight_padded = weight.float()
        else:
            weight_padded = weight.clone()
    
    # Step 3: 执行贪婪残差分配
    IN_GROUP = config.in_group_size
    OUT_GROUP = config.out_group_size
    num_windows = config.num_windows
    num_groups = k_padded // IN_GROUP
    
    if HAS_NUMBA:
        # 使用 Numba 加速
        weight_np = weight_padded.float().cpu().numpy().astype(np.float64)
        out_np = np.zeros((N, k_out), dtype=np.float64)
        
        _slide_greedy_allocation_numba(
            weight_np, out_np,
            IN_GROUP, OUT_GROUP, num_windows, num_groups, k_out
        )
        
        slided_weight = torch.from_numpy(out_np).float().to(device)
    else:
        # 回退到纯 Python 实现
        slided_weight = _slide_greedy_allocation_python(
            weight_padded.float() if original_dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else weight_padded,
            k_out,
            config,
        )
    
    # Step 4: 转换回原始数据类型
    if original_dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.int8):
        # 先转到 float，然后转到目标类型
        if original_dtype == torch.int8:
            slided_weight = slided_weight.round().clamp(-128, 127).to(torch.int8)
        else:
            slided_weight = slided_weight.to(original_dtype)
    else:
        slided_weight = slided_weight.to(original_dtype)
    
    # 构建元数据
    metadata = {
        "original_k": K,
        "padded_k": k_padded,
        "output_k": k_out,
        "expand_ratio": k_out / K,
        "config": config.to_dict(),
    }
    
    return slided_weight, metadata


def build_slide_index_mapping(k_in: int, config: SlideSparseConfig) -> torch.Tensor:
    """
    构建滑动索引映射表（用于验证和调试）
    
    Returns:
        index_map: 形状 [k_out]，每个元素是对应的输入索引，-1 表示 padding
    """
    k_padded, k_out = compute_output_k(k_in, config)
    
    IN_GROUP = config.in_group_size
    OUT_GROUP = config.out_group_size
    
    out_indices = torch.arange(k_out, dtype=torch.long)
    
    # 核心映射公式
    group_id = out_indices // OUT_GROUP
    local_out = out_indices % OUT_GROUP
    local_block = local_out // 4
    lane = local_out % 4
    
    in_idx = group_id * IN_GROUP + local_block * 2 + lane
    
    # 处理 padding
    in_idx = torch.where(in_idx < k_padded, in_idx, torch.tensor(-1, dtype=torch.long))
    
    num_groups = k_padded // IN_GROUP
    valid_out_size = num_groups * OUT_GROUP
    in_idx = torch.where(out_indices < valid_out_size, in_idx, torch.tensor(-1, dtype=torch.long))
    
    return in_idx


# =============================================================================
# Safetensors 处理
# =============================================================================

def slide_safetensors(
    input_path: Path,
    output_path: Path,
    config: SlideSparseConfig,
    verify: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    对 safetensors 文件执行滑动扩展
    
    Args:
        input_path: 输入 safetensors 文件路径
        output_path: 输出 safetensors 文件路径
        config: SlideSparse 配置
        verify: 是否验证 2:4 稀疏性
        verbose: 是否打印详细信息
    
    Returns:
        处理报告
    """
    if verbose:
        print_info(f"Loading {input_path}")
    
    weights = load_safetensors(input_path)
    
    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "config": config.to_dict(),
        "processed_layers": [],
        "skipped_layers": [],
        "verification_results": {},
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
        
        original_shape = list(tensor.shape)
        dtype_str = detect_weight_dtype(tensor)
        
        if verbose:
            print_info(f"Processing {key}: shape={original_shape}, dtype={dtype_str}")
        
        # 执行滑动扩展
        slided_tensor, metadata = slide_tensor(tensor, config, verbose=verbose)
        output_weights[key] = slided_tensor
        
        # 验证 2:4 稀疏性
        if verify:
            # 需要先转为 float 进行验证
            tensor_for_verify = slided_tensor.float() if slided_tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.int8) else slided_tensor
            is_valid, violation_ratio = verify_2to4_sparsity(tensor_for_verify)
            
            report["verification_results"][key] = {
                "is_valid": is_valid,
                "violation_ratio": violation_ratio,
            }
            
            if verbose:
                status = "✓" if is_valid else f"✗ (violation: {violation_ratio:.2%})"
                print_info(f"  2:4 verification: {status}")
        
        layer_info = {
            "key": key,
            "original_shape": original_shape,
            "output_shape": list(slided_tensor.shape),
            "dtype": dtype_str,
            "metadata": metadata,
        }
        report["processed_layers"].append(layer_info)
    
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
        description="SlideSparse 滑动扩展工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
维度变化示例 (2:8, N=4, expand_ratio=1.5):
  输入:  [4096, 4096]
  输出:  [4096, 6144]  (4096 × 1.5)

映射原理:
  Group 0 (输入 0-7):
    Window 0: input[0,1,2,3] → output[0,1,2,3]
    Window 1: input[2,3,4,5] → output[4,5,6,7]  (重叠)
    Window 2: input[4,5,6,7] → output[8,9,10,11] (重叠)
  Group 1 (输入 8-15): 重新开始...
        """
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
        help="输出路径（默认: 输入路径 + _slided 后缀）",
    )
    parser.add_argument(
        "--Z",
        type=int,
        default=2,
        help="稀疏度分子（默认: 2）",
    )
    parser.add_argument(
        "--L",
        type=int,
        required=True,
        help="稀疏度分母（如 6, 8, 10）",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="跳过 2:4 稀疏性验证",
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
    if args.output:
        output_base = Path(args.output)
    else:
        if input_path.is_dir():
            output_base = input_path.parent / f"{input_path.name}_slided_{args.Z}_{args.L}"
        else:
            output_base = input_path.parent / f"{input_path.stem}_slided_{args.Z}_{args.L}.safetensors"
    
    if not args.quiet:
        print_header(f"SlideSparse Sliding: {config.Z}:{config.L}")
        print_info(f"Expand ratio: {config.expand_ratio:.3f}")
        print_info(f"Numba acceleration: {'enabled' if HAS_NUMBA else 'disabled'}")
        print()
    
    # 处理每个文件
    all_valid = True
    for sf_path in safetensors_files:
        if len(safetensors_files) == 1 and not input_path.is_dir():
            out_path = output_base
        else:
            out_path = output_base / sf_path.name if output_base.suffix != ".safetensors" else output_base
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = slide_safetensors(
            sf_path,
            out_path,
            config,
            verify=not args.no_verify,
            verbose=not args.quiet,
        )
        
        if not args.quiet:
            processed = len(report["processed_layers"])
            skipped = len(report["skipped_layers"])
            
            if not args.no_verify:
                valid = sum(
                    1 for v in report["verification_results"].values()
                    if v["is_valid"]
                )
                if valid != processed:
                    all_valid = False
                print_success(f"Processed: {processed}, Valid: {valid}/{processed}, Skipped: {skipped}")
            else:
                print_success(f"Processed: {processed}, Skipped: {skipped}")
            print()
    
    if not args.no_verify and not all_valid:
        print_warning("Some layers failed 2:4 verification!")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
