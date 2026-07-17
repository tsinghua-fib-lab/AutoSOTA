#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse cuSPARSELt 压缩脚本

将满足 2:4 稀疏约束的权重使用 cuSPARSELt 压缩为硬件加速格式。

压缩原理：
- cuSPARSELt 的 2:4 结构化稀疏将每 4 个元素压缩为 2 个非零值 + 元数据
- K 维度减半：[N, K] -> [N, K/2] (压缩数据) + [N, K/8] (元数据)

依赖：
- build/libbitnet_compress.so - 编译好的 cuSPARSELt 压缩库

Usage:
    python compress.py --input /path/to/slided --output /path/to/compressed
"""

import argparse
import ctypes
import os
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
    compute_compressed_k,
    verify_2to4_sparsity,
    print_header,
    print_info,
    print_success,
    print_warning,
    print_error,
    BUILD_DIR,
)


# =============================================================================
# libbitnet_compress.so 加载（自定义压缩库）
# =============================================================================

_compress_lib = None


def get_compress_lib():
    """
    加载 BitNet 压缩库 (libbitnet_compress.so)
    
    这是一个自定义编译的库，封装了 cuSPARSELt 的压缩功能。
    
    Returns:
        ctypes library object
    """
    global _compress_lib
    
    if _compress_lib is not None:
        return _compress_lib
    
    # 查找 SO 文件
    so_paths = [
        BUILD_DIR / "libbitnet_compress.so",
        Path(__file__).parent / "build" / "libbitnet_compress.so",
        Path("/usr/local/lib/libbitnet_compress.so"),
    ]
    
    so_path = None
    for p in so_paths:
        if p.exists():
            so_path = p
            break
    
    if so_path is None:
        raise FileNotFoundError(
            f"BitNet compress library not found. "
            f"Expected locations: {[str(p) for p in so_paths]}"
        )
    
    print_info(f"Loading compress library: {so_path}")
    
    _compress_lib = ctypes.CDLL(str(so_path))
    
    # 设置函数签名
    # bitlinear_get_compress_sizes(int M, int N, int K, size_t* compressed_size, size_t* temp_buffer_size)
    _compress_lib.bitlinear_get_compress_sizes.argtypes = [
        ctypes.c_int,  # M
        ctypes.c_int,  # N
        ctypes.c_int,  # K
        ctypes.POINTER(ctypes.c_size_t),  # compressed_size
        ctypes.POINTER(ctypes.c_size_t),  # temp_buffer_size
    ]
    _compress_lib.bitlinear_get_compress_sizes.restype = None
    
    # bitlinear_compress_weight(int8_t* input, void* compressed, void* temp, int M, int N, int K, cudaStream_t stream)
    _compress_lib.bitlinear_compress_weight.argtypes = [
        ctypes.c_void_p,  # input_weight
        ctypes.c_void_p,  # compressed_weight
        ctypes.c_void_p,  # temp_buffer
        ctypes.c_int,     # M
        ctypes.c_int,     # N
        ctypes.c_int,     # K
        ctypes.c_void_p,  # stream (可以是 None)
    ]
    _compress_lib.bitlinear_compress_weight.restype = None
    
    return _compress_lib


def check_compress_lib_available() -> bool:
    """检查压缩库是否可用"""
    try:
        get_compress_lib()
        return True
    except (FileNotFoundError, OSError) as e:
        print_warning(f"Compress library not available: {e}")
        return False


# =============================================================================
# 压缩核心函数
# =============================================================================

def get_compress_sizes(M: int, N: int, K: int) -> Tuple[int, int]:
    """
    查询压缩后的大小和临时缓冲区大小
    
    Args:
        M: 激活矩阵行数（用于构建计划，可以是 1）
        N: 权重行数（out_features）
        K: 权重列数（in_features，必须满足 2:4）
    
    Returns:
        (compressed_size, temp_buffer_size) in bytes
    """
    lib = get_compress_lib()
    
    compressed_size = ctypes.c_size_t()
    temp_buffer_size = ctypes.c_size_t()
    
    lib.bitlinear_get_compress_sizes(
        M, N, K,
        ctypes.byref(compressed_size),
        ctypes.byref(temp_buffer_size),
    )
    
    return compressed_size.value, temp_buffer_size.value


def compress_tensor(
    weight: torch.Tensor,
    M: int = 1,
    verbose: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    使用 cuSPARSELt 压缩 2:4 稀疏权重
    
    Args:
        weight: 满足 2:4 稀疏的权重 [N, K]，必须是 INT8
        M: 用于构建计划的激活行数
        verbose: 是否打印详细信息
    
    Returns:
        (compressed_weight, metadata)
        - compressed_weight: 压缩后的数据，伪装为 [N, K/2] 的 uint8 tensor
        - metadata: 压缩元数据
    """
    if weight.dtype != torch.int8:
        raise ValueError(f"cuSPARSELt compress requires INT8 weight, got {weight.dtype}")
    
    N, K = weight.shape
    
    if K % 4 != 0:
        raise ValueError(f"K must be multiple of 4 for 2:4 sparsity, got K={K}")
    
    # 验证 2:4 稀疏性
    is_valid, violation_ratio = verify_2to4_sparsity(weight.float())
    if not is_valid:
        raise ValueError(f"Weight does not satisfy 2:4 sparsity (violation: {violation_ratio:.2%})")
    
    # 查询压缩大小
    compressed_size, temp_buffer_size = get_compress_sizes(M, N, K)
    
    if verbose:
        print_info(f"  Compress sizes: compressed={compressed_size}, temp={temp_buffer_size}")
    
    # 分配 GPU 内存
    weight_gpu = weight.cuda().contiguous()
    compressed_gpu = torch.empty(compressed_size, dtype=torch.uint8, device="cuda")
    
    if temp_buffer_size > 0:
        temp_buffer_gpu = torch.empty(temp_buffer_size, dtype=torch.uint8, device="cuda")
        temp_ptr = temp_buffer_gpu.data_ptr()
    else:
        temp_ptr = None
    
    # 执行压缩
    lib = get_compress_lib()
    
    lib.bitlinear_compress_weight(
        ctypes.c_void_p(weight_gpu.data_ptr()),
        ctypes.c_void_p(compressed_gpu.data_ptr()),
        ctypes.c_void_p(temp_ptr) if temp_ptr else None,
        M, N, K,
        None,  # stream
    )
    
    # 同步
    torch.cuda.synchronize()
    
    # 将压缩数据移回 CPU
    compressed_cpu = compressed_gpu.cpu()
    
    # 构建元数据
    metadata = {
        "original_shape": [N, K],
        "compressed_size_bytes": compressed_size,
        "temp_buffer_size_bytes": temp_buffer_size,
        "sparsity_pattern": "2:4",
    }
    
    return compressed_cpu, metadata


def compress_tensor_fake(
    weight: torch.Tensor,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    模拟 2:4 压缩（不调用 cuSPARSELt，用于测试）
    
    简单地将每 4 个元素中的 2 个非零值提取出来，并生成对应的元数据。
    
    Args:
        weight: 满足 2:4 稀疏的权重 [N, K]
        verbose: 是否打印详细信息
    
    Returns:
        (values, metadata_tensor, info)
        - values: [N, K/2] 非零值
        - metadata_tensor: [N, K/8] 稀疏位置元数据 (2 bits per element)
        - info: 压缩信息字典
    """
    N, K = weight.shape
    
    if K % 4 != 0:
        raise ValueError(f"K must be multiple of 4, got K={K}")
    
    # 重塑为 [N, K/4, 4]
    grouped = weight.view(N, K // 4, 4)
    
    # 找到每组的非零位置
    nonzero_mask = (grouped != 0)
    
    # 提取非零值（每组 2 个）
    values = torch.zeros(N, K // 2, dtype=weight.dtype)
    metadata_bits = torch.zeros(N, K // 4, dtype=torch.uint8)
    
    for i in range(N):
        for j in range(K // 4):
            group = grouped[i, j]
            mask = nonzero_mask[i, j]
            
            # 获取非零位置
            nz_indices = torch.where(mask)[0]
            
            # 取前 2 个非零值
            num_nz = min(len(nz_indices), 2)
            for k in range(num_nz):
                values[i, j * 2 + k] = group[nz_indices[k]]
            
            # 编码位置（简化版：用 2 bits 表示每个非零位置）
            if len(nz_indices) >= 1:
                metadata_bits[i, j] |= nz_indices[0].item()
            if len(nz_indices) >= 2:
                metadata_bits[i, j] |= (nz_indices[1].item() << 2)
    
    # 将 metadata 打包为 [N, K/8]（每字节存储 4 个组的元数据）
    metadata_tensor = metadata_bits.view(N, -1, 4)
    # 简化：直接保留原始形状
    
    info = {
        "original_shape": [N, K],
        "values_shape": list(values.shape),
        "metadata_shape": list(metadata_bits.shape),
        "compression_type": "fake_2to4",
    }
    
    return values, metadata_bits, info


# =============================================================================
# Safetensors 处理
# =============================================================================

def compress_safetensors(
    input_path: Path,
    output_path: Path,
    use_real_cusparselt: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    对 safetensors 文件执行 cuSPARSELt 压缩
    
    Args:
        input_path: 输入 safetensors 文件路径
        output_path: 输出 safetensors 文件路径
        use_real_cusparselt: 是否使用真实的 cuSPARSELt（否则用模拟压缩）
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
        "use_real_cusparselt": use_real_cusparselt,
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
        
        original_shape = list(tensor.shape)
        dtype_str = detect_weight_dtype(tensor)
        
        if verbose:
            print_info(f"Processing {key}: shape={original_shape}, dtype={dtype_str}")
        
        # 需要转换为 INT8 进行压缩
        if tensor.dtype != torch.int8:
            if verbose:
                print_warning(f"  Converting {dtype_str} to INT8 for compression")
            
            # FP8/INT8 都可以表示 ternary 值，直接转换
            if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                tensor_int8 = tensor.float().round().clamp(-127, 127).to(torch.int8)
            elif tensor.dtype in (torch.bfloat16, torch.float16, torch.float32):
                tensor_int8 = tensor.round().clamp(-127, 127).to(torch.int8)
            else:
                tensor_int8 = tensor.to(torch.int8)
        else:
            tensor_int8 = tensor
        
        # 执行压缩
        if use_real_cusparselt:
            compressed, metadata = compress_tensor(tensor_int8, verbose=verbose)
            output_weights[key] = compressed
            
            # 保存压缩元数据
            meta_key = key.replace(".weight", ".weight_compressed_meta")
            # 将元数据信息编码（实际中可能需要单独存储）
            
            layer_info = {
                "key": key,
                "original_shape": original_shape,
                "compressed_size": compressed.shape[0],
                "metadata": metadata,
            }
        else:
            # 使用模拟压缩
            values, meta_tensor, info = compress_tensor_fake(tensor_int8, verbose=verbose)
            output_weights[key] = values
            
            # 保存元数据张量
            meta_key = key.replace(".weight", ".weight_meta")
            output_weights[meta_key] = meta_tensor
            
            layer_info = {
                "key": key,
                "original_shape": original_shape,
                "values_shape": list(values.shape),
                "meta_shape": list(meta_tensor.shape),
                "info": info,
            }
        
        report["processed_layers"].append(layer_info)
        
        if verbose:
            if use_real_cusparselt:
                print_info(f"  Compressed: {compressed.shape[0]} bytes")
            else:
                print_info(f"  Values: {list(values.shape)}, Meta: {list(meta_tensor.shape)}")
    
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
        description="SlideSparse cuSPARSELt 压缩工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
压缩说明：
  2:4 结构化稀疏压缩将 K 维度减半
  
  输入：[N, K] 满足 2:4 稀疏的权重
  输出：[N, K/2] 压缩数据 + 元数据

依赖：
  需要 build/libbitnet_compress.so
  使用 --fake 可以跳过真实压缩进行测试
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
        help="输出路径（默认: 输入路径 + _compressed 后缀）",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="使用模拟压缩（不调用 cuSPARSELt）",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="静默模式",
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # 检查压缩库可用性
    use_real = not args.fake
    if use_real and not check_compress_lib_available():
        print_warning("Compress library not available, falling back to fake compression")
        use_real = False
    
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
            output_base = input_path.parent / f"{input_path.name}_compressed"
        else:
            output_base = input_path.parent / f"{input_path.stem}_compressed.safetensors"
    
    if not args.quiet:
        print_header("SlideSparse cuSPARSELt Compression")
        print_info(f"Mode: {'real cuSPARSELt' if use_real else 'fake compression'}")
        print()
    
    # 处理每个文件
    for sf_path in safetensors_files:
        if len(safetensors_files) == 1 and not input_path.is_dir():
            out_path = output_base
        else:
            out_path = output_base / sf_path.name if output_base.suffix != ".safetensors" else output_base
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            report = compress_safetensors(
                sf_path,
                out_path,
                use_real_cusparselt=use_real,
                verbose=not args.quiet,
            )
            
            if not args.quiet:
                processed = len(report["processed_layers"])
                skipped = len(report["skipped_layers"])
                print_success(f"Processed: {processed}, Skipped: {skipped}")
                print()
        
        except Exception as e:
            print_error(f"Failed to compress {sf_path}: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
