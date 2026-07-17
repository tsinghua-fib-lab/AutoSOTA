#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse Benchmark Kernel 专用工具库

提供 benchmark_kernel 模块所需的通用功能：
- 数据精度配置与硬件兼容性检测
- Sparsity 计算与 K_slide 转换
- 数据准备与量化
- 结果整合与加速比计算
- 增量保存与原子写入

该工具库依赖顶层 slidesparse.utils，提供 benchmark 场景的封装。
"""

import base64
import ctypes
import datetime
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

# 导入顶层 utils
BENCHMARK_KERNEL_DIR = Path(__file__).parent.absolute()
SLIDESPARSE_DIR = BENCHMARK_KERNEL_DIR.parent
PROJECT_ROOT = SLIDESPARSE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from slidesparse.utils import (
    # 硬件信息
    hw_info,
    HardwareInfo,
    normalize_dtype,
    # 全局默认配置
    DEFAULT_M_LIST,
    M_QUICK_LIST,
    # 编译与加载
    load_cuda_extension,
    build_cuda_extension_direct,
    ensure_cublaslt_loaded,
    ensure_cusparselt_loaded,
    BACKEND_LDFLAGS,
    SUPPORTED_BACKENDS,
    # 系统库路径
    get_system_lib_path,
    # 文件命名
    build_filename,
    build_stem,
    build_tuned_filename,
    # 模型信息
    get_model_nk_sizes,
)


# =============================================================================
# Square 模式专用配置
# =============================================================================

# Square 模式: M=N=K 使用此列表
# 从 64 到 16384，每次翻倍（移除 32768/65536 避免 OOM）
SQUARE_M_LIST = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


# =============================================================================
# 数据精度配置 (根据 cuBLASLt/cuSPARSELt 硬件限制)
# =============================================================================

# 5 种数据精度的完整配置
# 低精度输出策略：输入 = 输出（减少 IO 开销）
# - FP16/BF16: 输入输出相同
# - INT8: cuBLASLt/cuSPARSELt 都输出 INT8
# - FP8: cuBLASLt/cuSPARSELt 都输出 FP8
# - FP4: cuBLASLt 输出 BF16, cuSPARSELt 输出 FP4
DTYPE_CONFIG = {
    "fp16": {
        "cuda_input": "CUDA_R_16F",
        "cuda_output": "CUDA_R_16F",
        "cuda_compute": "CUDA_R_32F",
        "cublaslt_compute": "CUBLAS_COMPUTE_32F",
        "cusparselt_compute": "CUSPARSE_COMPUTE_32F",
        "scale_type": "CUDA_R_32F",
        "elem_size": 2,
        "out_elem_size": 2,
        "min_cc": 60,
        "torch_dtype": torch.float16,
        "torch_out_dtype": torch.float16,
        "cublaslt_out_dtype": torch.float16,
        "cusparselt_out_dtype": torch.float16,
    },
    "bf16": {
        "cuda_input": "CUDA_R_16BF",
        "cuda_output": "CUDA_R_16BF",
        "cuda_compute": "CUDA_R_32F",
        "cublaslt_compute": "CUBLAS_COMPUTE_32F",
        "cusparselt_compute": "CUSPARSE_COMPUTE_32F",
        "scale_type": "CUDA_R_32F",
        "elem_size": 2,
        "out_elem_size": 2,
        "min_cc": 80,
        "torch_dtype": torch.bfloat16,
        "torch_out_dtype": torch.bfloat16,
        "cublaslt_out_dtype": torch.bfloat16,
        "cusparselt_out_dtype": torch.bfloat16,
    },
    "int8": {
        "cuda_input": "CUDA_R_8I",
        "cuda_output": "CUDA_R_8I",  # 低精度输出：INT8
        "cuda_compute": "CUDA_R_32I",
        "cublaslt_compute": "CUBLAS_COMPUTE_32I",
        "cusparselt_compute": "CUSPARSE_COMPUTE_32I",
        "scale_type": "CUDA_R_32I",
        "elem_size": 1,
        "out_elem_size": 1,  # INT8 = 1 byte
        "min_cc": 75,
        "torch_dtype": torch.int8,
        "torch_out_dtype": torch.int8,  # 低精度输出
        "cublaslt_out_dtype": torch.int8,  # cuBLASLt: INT8 输出
        "cusparselt_out_dtype": torch.int8,  # cuSPARSELt: INT8 低精度输出
    },
    "fp8e4m3": {
        "cuda_input": "CUDA_R_8F_E4M3",
        "cuda_output": "CUDA_R_8F_E4M3",  # 低精度输出：FP8
        "cuda_compute": "CUDA_R_32F",
        "cublaslt_compute": "CUBLAS_COMPUTE_32F",
        "cusparselt_compute": "CUSPARSE_COMPUTE_32F",
        "scale_type": "CUDA_R_32F",
        "elem_size": 1,
        "out_elem_size": 1,  # FP8 = 1 byte
        "min_cc": 89,  # Ada Lovelace+
        "torch_dtype": torch.float8_e4m3fn,
        "torch_out_dtype": torch.float8_e4m3fn,  # 低精度输出
        "cublaslt_out_dtype": torch.float8_e4m3fn,  # cuBLASLt: FP8 输出
        "cusparselt_out_dtype": torch.float8_e4m3fn,  # cuSPARSELt: FP8 低精度输出
    },
    "fp4e2m1": {
        "cuda_input": "CUDA_R_4F_E2M1",
        "cuda_output": "CUDA_R_4F_E2M1",  # 低精度输出：FP4 (打包格式)
        "cuda_compute": "CUDA_R_32F",
        "cublaslt_compute": "CUBLAS_COMPUTE_32F",
        "cusparselt_compute": "CUSPARSE_COMPUTE_32F",
        "scale_type": "CUDA_R_32F",
        "elem_size": 0.5,  # 4-bit packed (2 values per byte)
        "out_elem_size": 0.5,  # FP4 打包输出 (2 values per byte)
        "min_cc": 100,  # Blackwell+
        "torch_dtype": None,  # PyTorch 尚未原生支持 FP4
        "torch_out_dtype": torch.uint8,  # FP4 打包存储使用 uint8
        "cublaslt_out_dtype": torch.uint8,  # cuBLASLt FP4 打包输出
        "cusparselt_out_dtype": torch.bfloat16,  # cuSPARSELt: FP4 输出 BF16 (保持高精度)
        "requires_scale": True,  # FP4 强制要求 scale
    },
}

# 支持的数据类型列表
SUPPORTED_DTYPES = ["fp16", "bf16", "int8", "fp8e4m3", "fp4e2m1"]

# 默认稀疏度列表
DEFAULT_SPARSITY_LIST = ["2_4", "2_6", "2_8", "2_10"]

# 对齐要求
# - 标准对齐: 32（FP16/BF16/INT8/FP8）
# - FP4 对齐: 64（因为 FP4 使用 VEC32_UE4M3 scale mode，需要更大的 block 对齐）
ALIGNMENT = 32
ALIGNMENT_FP4 = 64  # FP4 需要 64 的倍数


# =============================================================================
# GPU 兼容性检测
# =============================================================================

def get_current_cc() -> int:
    """
    获取当前 GPU 的 Compute Capability (整数形式)
    例如: CC 9.0 -> 90, CC 10.0 -> 100
    """
    return hw_info.cc_major * 10 + hw_info.cc_minor


def check_dtype_support(dtype: str) -> Tuple[bool, str]:
    """
    检查当前 GPU 是否支持指定的数据类型
    
    Args:
        dtype: 数据类型 (fp16, bf16, int8, fp8e4m3, fp4e2m1)
    
    Returns:
        (supported, reason): 是否支持，及原因说明
    """
    dtype_lower = dtype.lower()
    
    if dtype_lower not in DTYPE_CONFIG:
        return False, f"Unknown dtype: {dtype}. Supported: {SUPPORTED_DTYPES}"
    
    config = DTYPE_CONFIG[dtype_lower]
    min_cc = config["min_cc"]
    current_cc = get_current_cc()
    
    if current_cc < min_cc:
        return False, (
            f"dtype={dtype} requires CC >= {min_cc // 10}.{min_cc % 10}, "
            f"current GPU: {hw_info.gpu_name} (CC {hw_info.cc_major}.{hw_info.cc_minor})"
        )
    
    return True, "OK"


def get_supported_dtypes_for_gpu() -> List[str]:
    """
    获取当前 GPU 支持的所有数据类型列表
    """
    supported = []
    for dtype in SUPPORTED_DTYPES:
        ok, _ = check_dtype_support(dtype)
        if ok:
            supported.append(dtype)
    return supported


def check_cusparselt_support() -> Tuple[bool, str]:
    """检查 cuSPARSELt 是否支持当前 GPU"""
    if get_current_cc() < 80:
        return False, f"cuSPARSELt requires CC >= 8.0 (Ampere+), current: {hw_info.cc_tag}"
    return True, "OK"


def check_segment_k_support() -> Tuple[bool, str]:
    """检查是否支持 Segment-K (SM90+)"""
    cc_major = hw_info.cc_major
    if cc_major >= 9:
        return True, "OK"
    return False, f"Segment-K requires SM >= 9.0 (Hopper+), current: {hw_info.cc_tag}"


# =============================================================================
# Sparsity 计算
# =============================================================================

def parse_sparsity_config(sparsity: str) -> Tuple[int, int]:
    """
    解析稀疏度配置字符串
    
    格式: "{Z}_{L}" 例如 "2_4", "2_6", "2_8", "2_10"
    
    含义: 每 L 个元素保留 Z 个非零值 (其余 L-Z 个为零)
    - 2_4: 保留 2/4 = 50% 非零 (标准 2:4 稀疏)
    - 2_6: 保留 2/6 ≈ 33% 非零
    - 2_8: 保留 2/8 = 25% 非零
    - 2_inf: 全零稀疏 (理论上限)
    
    Returns:
        (Z, L): 非零元素数, 窗口长度
    """
    parts = sparsity.split('_')
    if len(parts) != 2:
        raise ValueError(f"Invalid sparsity format: {sparsity}, expected Z_L (e.g., 2_4, 2_8)")
    
    Z = int(parts[0])
    if parts[1].lower() == 'inf':
        L = float('inf')
    else:
        L = int(parts[1])
    
    if Z <= 0:
        raise ValueError(f"Z must be positive, got {Z}")
    if L != float('inf') and L <= Z:
        raise ValueError(f"L must be greater than Z, got Z={Z}, L={L}")
    
    return Z, L


def calculate_k_slide(
    K: int, 
    sparsity: str, 
    align_to: Optional[int] = None,
    dtype: Optional[str] = None,
) -> int:
    """
    计算 SlideSparse 稀疏化后的 K 维度 (K_slide)
    
    对于 cuSPARSELt benchmark，我们测试的是：
    - Dense cuBLAS: W[N,K] @ A[M,K] (原始K)
    - Sparse cuSPARSELt: W'[N,K_slide] @ A'[M,K_slide] (slide后的K维度)
    
    稀疏度格式: {Z}_{L}
    计算公式: K_slide = ceil( 2 * (L - Z) / L * K / align_to ) * align_to
    
    示例:
    - 2_4:  ratio = 2*(4-2)/4 = 1.0,  K_slide = K
    - 2_6:  ratio = 2*(6-2)/6 = 4/3,  K_slide ≈ 1.33*K
    - 2_8:  ratio = 2*(8-2)/8 = 1.5,  K_slide = 1.5*K
    - 2_10: ratio = 2*(10-2)/10 = 1.6, K_slide = 1.6*K
    - 2_inf: ratio = 2.0, K_slide = 2*K (理论上限)
    
    注意: 
    - 标准类型 (FP16/BF16/INT8/FP8): K 必须是 32 的倍数
    - FP4 (E2M1): K 必须是 64 的倍数（因为 scale mode VEC32_UE4M3 需要更大的 block）
    
    Args:
        K: 原始 K 维度
        sparsity: 稀疏度配置 (如 "2_4", "2_8")
        align_to: 对齐要求 (默认根据 dtype 自动选择)
        dtype: 数据类型，用于确定对齐要求 (可选)
    
    Returns:
        K_slide: slide 后的 K 维度 (向上对齐)
    """
    Z, L = parse_sparsity_config(sparsity)
    
    # 特殊情况: inf 表示全稀疏
    if L == float('inf'):
        ratio = 2.0
    else:
        ratio = 2.0 * (L - Z) / L
    
    # 确定对齐要求
    if align_to is None:
        if dtype and dtype.lower() in ("fp4e2m1", "fp4"):
            align_to = ALIGNMENT_FP4  # 64 for FP4
        else:
            align_to = ALIGNMENT  # 32 for others
    
    K_slide_raw = K * ratio
    K_slide = ((int(K_slide_raw) + align_to - 1) // align_to) * align_to
    return K_slide


def get_k_expansion_factor(sparsity: str) -> float:
    """
    获取 K 扩展因子
    
    Returns:
        K_slide / K 的比例
    """
    Z, L = parse_sparsity_config(sparsity)
    if L == float('inf'):
        return 2.0
    return 2.0 * (L - Z) / L


def get_sparsity_ratio(sparsity: str) -> float:
    """
    获取稀疏度比例 (非零元素占比)
    
    Returns:
        Z/L (0.0 ~ 1.0 之间)
    """
    Z, L = parse_sparsity_config(sparsity)
    if L == float('inf'):
        return 0.0
    return Z / L


def pad_to_alignment(value: int, align_to: int = ALIGNMENT) -> int:
    """
    将值向上对齐到指定倍数
    
    Args:
        value: 原始值
        align_to: 对齐要求 (默认 32)
    
    Returns:
        对齐后的值
    """
    return ((value + align_to - 1) // align_to) * align_to


# =============================================================================
# CUDA 扩展编译与加载
# =============================================================================

def build_benchmark_extension(
    name: str,
    source_file: Path,
    build_dir: Path,
    backend: str,
    *,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """
    编译 benchmark 扩展 (extern "C" 接口)
    
    基于顶层 build_cuda_extension_direct，自动添加后端依赖库。
    
    Args:
        name: 扩展名称（如 "cublaslt_gemm"）
        source_file: 源文件路径 (.cu)
        build_dir: 构建目录
        backend: 后端类型 ("cublaslt" 或 "cusparselt")
        force: 是否强制重新编译
        verbose: 是否显示详细输出
    
    Returns:
        编译生成的 .so 文件路径
    """
    # 构建带完整硬件信息的名称
    full_name = (
        f"{name}_{hw_info.gpu_name}_{hw_info.cc_tag}"
        f"_{hw_info.python_tag}_{hw_info.cuda_tag}_{hw_info.arch_tag}"
    )
    
    # 获取后端链接库 (使用系统库路径，支持 x86_64/aarch64)
    backend_lower = backend.lower()
    lib_path = get_system_lib_path()
    if backend_lower == "cublaslt":
        extra_ldflags = [f"-L{lib_path}", "-lcublasLt", "-lcublas", "-lcuda"]
    elif backend_lower == "cusparselt":
        extra_ldflags = [f"-L{lib_path}", "-lcusparseLt", "-lcusparse", "-lcublas", "-lcuda"]
    else:
        raise ValueError(f"Unknown backend: {backend}. Supported: cublaslt, cusparselt")
    
    return build_cuda_extension_direct(
        name=full_name,
        source_file=source_file,
        build_dir=build_dir,
        extra_ldflags=extra_ldflags,
        force=force,
        verbose=verbose,
    )


def load_benchmark_extension(
    so_path: Path,
    backend: str,
    setup_func: Callable[[ctypes.CDLL], None],
) -> ctypes.CDLL:
    """
    加载 benchmark 扩展 (extern "C" 接口)
    
    Args:
        so_path: .so 文件路径
        backend: 后端类型 ("cublaslt" 或 "cusparselt")
        setup_func: 设置函数签名的回调函数
    
    Returns:
        加载的 ctypes.CDLL 对象
    """
    # 预加载后端库
    backend_lower = backend.lower()
    if backend_lower == "cublaslt":
        ensure_cublaslt_loaded()
    elif backend_lower == "cusparselt":
        ensure_cusparselt_loaded()
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    # 加载扩展
    lib = ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
    
    # 调用设置函数配置签名
    setup_func(lib)
    
    return lib


# =============================================================================
# 数据准备工具
# =============================================================================

def quantize_int8(x: torch.Tensor, inplace: bool = False) -> Tuple[torch.Tensor, float]:
    """
    将 BF16/FP16 张量量化到 INT8
    
    Args:
        x: 输入张量
        inplace: 是否原地操作（节省显存）
    
    Returns:
        (quantized_tensor, scale)
    """
    abs_max = x.abs().max().item()
    scale = 127.0 / abs_max if abs_max > 0 else 1.0
    
    if inplace:
        x.mul_(scale)
        x.round_()
        x.clamp_(-128, 127)
        q = x.to(torch.int8)
    else:
        q = (x * scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale


def to_fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    """转换为 FP8E4M3 格式"""
    return x.to(torch.float8_e4m3fn)


def to_fp16(x: torch.Tensor) -> torch.Tensor:
    """转换为 FP16 格式"""
    return x.to(torch.float16)


def to_bf16(x: torch.Tensor) -> torch.Tensor:
    """转换为 BF16 格式"""
    return x.to(torch.bfloat16)


def to_fp4_e2m1_packed(x: torch.Tensor) -> torch.Tensor:
    """
    将 BF16/FP16 张量转换为打包的 FP4E2M1 格式
    
    FP4 E2M1 格式：2-bit 指数 + 1-bit 尾数 = 4-bit
    每个字节存储 2 个 FP4 值（packed format）
    
    注意：这是一个用于 benchmark 的简化实现，使用以下策略：
    1. 将 FP32/BF16 值量化到 FP4 可表示的范围
    2. 将两个 FP4 值打包到一个 uint8 中
    
    FP4 E2M1 可表示的值（带符号）：
    - 正值：0, 0.5, 1, 1.5, 2, 3, 4, 6
    - 负值：-0, -0.5, -1, -1.5, -2, -3, -4, -6
    
    Args:
        x: 输入张量 (BF16/FP16)，形状 [N, K]
    
    Returns:
        打包后的张量，形状 [N, K//2]，dtype=uint8
    """
    # 转为 float32 进行计算
    x_fp32 = x.to(torch.float32)
    original_shape = x_fp32.shape
    
    # 展平处理
    x_flat = x_fp32.flatten()
    n_elements = x_flat.numel()
    
    # 确保元素数量是偶数（用于打包）
    if n_elements % 2 != 0:
        x_flat = torch.cat([x_flat, torch.zeros(1, device=x.device, dtype=torch.float32)])
        n_elements += 1
    
    # 对每个值进行量化：找到最近的 FP4 可表示值
    # 符号分离
    signs = (x_flat < 0).to(torch.uint8)  # 0 = 正, 1 = 负
    abs_vals = x_flat.abs()
    
    # 缩放到 FP4 范围（最大值 6.0）
    max_val = abs_vals.max().item()
    if max_val > 0:
        abs_vals = abs_vals * (6.0 / max_val)
    
    # 使用 bucketize 进行高效量化（内存友好）
    # FP4 E2M1 可表示的正值: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    # 相邻值的中点作为边界:  [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    # bucketize 返回的索引正好对应 FP4 值的索引
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], 
                              device=x.device, dtype=torch.float32)
    fp4_indices = torch.bucketize(abs_vals, boundaries).to(torch.uint8)
    
    # 组合符号位和数值位：sign(1bit) | value(3bits) = 4-bit
    fp4_codes = (signs << 3) | fp4_indices  # 4-bit 编码
    
    # 打包：每两个 FP4 值打包到一个 uint8
    # 低 4-bit 是第一个值，高 4-bit 是第二个值
    fp4_codes = fp4_codes.view(-1, 2)
    packed = (fp4_codes[:, 1] << 4) | fp4_codes[:, 0]
    
    # 重塑为原始形状（K 维度减半）
    new_shape = list(original_shape)
    new_shape[-1] = new_shape[-1] // 2
    
    return packed.view(new_shape)


def to_fp4_e2m1_unpacked(x: torch.Tensor) -> torch.Tensor:
    """
    将 BF16/FP16 张量转换为未打包的 FP4E2M1 格式 (用于 cuSPARSELt)
    
    cuSPARSELt 的 FP4 API 期望：
    - 输入数据已经是 packed format（每字节 2 个 FP4 值）
    - 但传递的维度是逻辑维度（未打包的元素数量）
    - cuSPARSELt 内部会正确处理 paired 4:8 稀疏约束
    
    本函数返回打包后的数据，但调用者应该使用原始逻辑维度。
    
    FP4 E2M1 可表示的值（带符号）：
    - 正值：0, 0.5, 1, 1.5, 2, 3, 4, 6
    - 负值：-0, -0.5, -1, -1.5, -2, -3, -4, -6
    
    Args:
        x: 输入张量 (BF16/FP16)，形状 [N, K]
    
    Returns:
        打包后的张量，形状 [N, K//2]，dtype=uint8
        
    Note:
        返回的张量形状是 [N, K//2]，但对于 cuSPARSELt 调用，
        应该传递逻辑维度 (N, K)，而不是打包后的维度。
    """
    # 实际上 cuSPARSELt 期望的就是 packed format
    # 只是调用时维度参数不同
    return to_fp4_e2m1_packed(x)


def quantize_tensor(x: torch.Tensor, dtype: str) -> torch.Tensor:
    """
    根据 dtype 量化/转换张量
    
    Args:
        x: 输入张量 (BF16/FP16)
        dtype: 目标类型
    
    Returns:
        量化后的张量
    """
    dtype_lower = dtype.lower()
    
    if dtype_lower == "int8":
        q, _ = quantize_int8(x)
        return q
    elif dtype_lower == "fp8e4m3":
        return to_fp8_e4m3(x)
    elif dtype_lower == "fp16":
        return to_fp16(x)
    elif dtype_lower == "bf16":
        return to_bf16(x)
    elif dtype_lower == "fp4e2m1":
        # FP4 E2M1：将 BF16/FP16 量化并打包为 uint8（每字节 2 个 FP4 值）
        return to_fp4_e2m1_packed(x)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def get_output_torch_dtype(dtype: str, backend: str = "cublaslt") -> torch.dtype:
    """
    获取输出对应的 PyTorch dtype
    
    注意：INT8 和 FP8/FP4 的输出类型取决于后端
    - cuBLASLt INT8: 输出 INT32
    - cuSPARSELt INT8: 输出 BF16
    - FP8/FP4: 输出 BF16
    
    Args:
        dtype: 输入数据类型
        backend: 后端类型 ("cublaslt" 或 "cusparselt")
    
    Returns:
        输出对应的 PyTorch dtype
    """
    dtype_lower = dtype.lower()
    backend_lower = backend.lower()
    
    if dtype_lower not in DTYPE_CONFIG:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    config = DTYPE_CONFIG[dtype_lower]
    
    # 根据后端选择输出类型
    if backend_lower == "cublaslt":
        torch_dtype = config.get("cublaslt_out_dtype")
    elif backend_lower == "cusparselt":
        torch_dtype = config.get("cusparselt_out_dtype")
    else:
        torch_dtype = config.get("torch_out_dtype")
    
    if torch_dtype is None:
        # FP4 等特殊类型，使用 BF16 作为输出
        return torch.bfloat16
    
    return torch_dtype


# =============================================================================
# NK 列表处理（复用 slidesparse.utils 的统一工具）
# =============================================================================

# 导入统一的 NK 获取工具
from slidesparse.utils import (
    get_nk_list_for_search,
    model_base_name,
)


def get_sparsity_list_for_benchmark(L_max: Optional[int] = None) -> List[str]:
    """
    生成用于 benchmark 的稀疏度列表
    
    与参考代码一致：
    - L_max=None: 只返回 ["2_4"]（标准 2:4 稀疏）
    - L_max 指定: 返回 ["2_4", "2_6", ..., "2_{L_max}", "2_inf"]
    
    特殊之处：在 L_max 时额外加上 2_inf
    
    Args:
        L_max: 最大 L 值。如果为 None，只返回标准 2:4 稀疏。
    
    Returns:
        稀疏度字符串列表，如 ["2_4", "2_6", "2_8", "2_inf"]
    
    Example:
        >>> get_sparsity_list_for_benchmark()
        ['2_4']
        >>> get_sparsity_list_for_benchmark(8)
        ['2_4', '2_6', '2_8', '2_inf']
        >>> get_sparsity_list_for_benchmark(10)
        ['2_4', '2_6', '2_8', '2_10', '2_inf']
    """
    if L_max is None:
        return ["2_4"]
    
    if L_max < 4:
        raise ValueError(f"L_max 必须 >= 4，当前值: {L_max}")
    
    sparsity_list = []
    for L in range(4, L_max + 1, 2):
        sparsity_list.append(f"2_{L}")
    
    # 特殊：在 L_max 时额外加上 2_inf（理论上限）
    sparsity_list.append("2_inf")
    
    return sparsity_list


def get_nk_list_for_benchmark(
    model: Optional[str] = None,
    L_max: Optional[int] = None,
    m_list: Optional[List[int]] = None,
    checkpoints_dir: Optional[Union[str, Path]] = None,
) -> Tuple[List[Tuple[int, int]], str, str]:
    """
    获取用于 benchmark 的 NK 列表（复用 get_nk_list_for_search）
    
    支持两种模式:
    1. Model-based 模式: 从模型 checkpoint 提取真实 NK
       - L_max=None: 获取原始 NK
       - L_max 指定: 获取 L=4,6,...,L_max 的所有 slided NK
    2. Square 模式: model=None 或 model="square"，M=N=K 使用 m_list
    
    Args:
        model: 模型名称（可选，None 或 "square" 进入 Square 模式）
               支持完整路径、checkpoint 目录名、或 base name
        L_max: 最大 L 值（用于 slide sparse），与 get_nk_list_for_search 一致
        m_list: 自定义 M 列表（仅 Square 模式有效，作为 M=N=K 的值）
        checkpoints_dir: 自定义 checkpoints 目录路径
    
    Returns:
        (nk_list, model_name, mode)
        - nk_list: [(N1, K1), (N2, K2), ...] 列表
        - model_name: 模型名称或 "SQUARE"
        - mode: "model" 或 "square"
    
    Note:
        本函数直接复用 slidesparse.utils.get_nk_list_for_search，
        保证与参考代码（offline_autotune_algsearch.py）的逻辑一致。
    """
    # Square 模式: model=None 或 model="square"
    if model is None or model.lower() == "square":
        # Square 模式: M=N=K 使用自定义 m_list 或默认 M_QUICK_LIST
        square_list = m_list if m_list else list(M_QUICK_LIST)
        nk_list = [(m, m) for m in square_list]
        return nk_list, "SQUARE", "square"
    
    # Model-based 模式：直接调用统一工具
    try:
        nk_list, model_name = get_nk_list_for_search(
            model=model,
            L_max=L_max,
            checkpoints_dir=checkpoints_dir,
        )
        
        # 对齐到 32
        nk_list = [(pad_to_alignment(n), pad_to_alignment(k)) for n, k in nk_list]
        
        return nk_list, model_name, "model"
    except ValueError as e:
        # 未找到模型，fallback 到 Square 模式
        print(f"[WARN] {e}")
        print("[INFO] Falling back to Square mode")
        square_list = m_list if m_list else list(M_QUICK_LIST)
        nk_list = [(m, m) for m in square_list]
        return nk_list, "SQUARE", "square"


# =============================================================================
# 输出目录与文件命名
# =============================================================================

def build_hw_folder_name() -> str:
    """
    构建硬件目录名称
    
    格式: {GPU}_{CC}_{PyVer}_{CUDAVer}_{arch}
    示例: H100_cc90_py312_cu124_x86_64
    """
    return (
        f"{hw_info.gpu_name}_{hw_info.cc_tag}"
        f"_{hw_info.python_tag}_{hw_info.cuda_tag}_{hw_info.arch_tag}"
    )


def build_dtype_folder_name(dtype: str) -> str:
    """
    构建 dtype 子文件夹名称
    
    格式: {dtype} (规范化后的大写)
    示例: FP16, BF16, INT8, FP8, FP4
    """
    dtype_upper = dtype.upper()
    if dtype_upper == "FP8E4M3":
        return "FP8"
    elif dtype_upper == "FP4E2M1":
        return "FP4"
    else:
        return dtype_upper


def build_output_dir(base_dir: Path, dtype: str) -> Path:
    """
    构建完整的输出目录路径（包含硬件和dtype子文件夹）
    
    结构: {base_dir}/{hw_folder}/{dtype_folder}/
    示例: alg_search_results/H100_cc90_py312_cu124_x86_64/FP8/
    
    Args:
        base_dir: 基础输出目录
        dtype: 数据类型
    
    Returns:
        完整的输出目录路径
    """
    hw_folder = build_hw_folder_name()
    dtype_folder = build_dtype_folder_name(dtype)
    return base_dir / hw_folder / dtype_folder


def build_result_filename(
    prefix: str,
    model_name: str,
    ext: str,
    sparsity: Optional[str] = None,
) -> str:
    """
    构建结果文件名
    
    格式: {prefix}_{model}[_{sparsity}].{ext}
    示例: alg_search_Qwen2.5-0.5B-FP8.csv
          speedup_Qwen2.5-0.5B-FP8_2_8.csv
    """
    
    if sparsity:
        return f"{prefix}_{model_name}_{sparsity}.{ext}"
    else:
        return f"{prefix}_{model_name}.{ext}"


# =============================================================================
# 结果整合与加速比计算
# =============================================================================

def compute_speedup(dense_tops: float, sparse_tops: float) -> float:
    """
    计算加速比 (Sparse / Dense)
    
    Args:
        dense_tops: Dense GEMM (cuBLASLt) 的 TOPS
        sparse_tops: Sparse GEMM (cuSPARSELt) 的 TOPS
    
    Returns:
        加速比 (sparse_tops / dense_tops)
    """
    if dense_tops <= 0:
        return 0.0
    return sparse_tops / dense_tops


def merge_benchmark_results(
    cublaslt_results: Dict,
    cusparselt_results: Dict,
    sparsity: str,
    output_dir: Path,
    model_name: str,
    dtype: str,
) -> Path:
    """
    整合 cuBLASLt 和 cuSPARSELt 的结果，计算加速比
    
    Args:
        cublaslt_results: cuBLASLt 搜索结果
        cusparselt_results: cuSPARSELt 搜索结果
        sparsity: 稀疏度配置
        output_dir: 输出目录
        model_name: 模型名称
        dtype: 数据类型
    
    Returns:
        保存的 CSV 文件路径
    """
    hw_folder = build_hw_folder_name()
    out_dir = output_dir / hw_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_filename = build_result_filename("speedup", model_name, "csv", sparsity)
    csv_path = out_dir / csv_filename
    
    k_factor = get_k_expansion_factor(sparsity)
    
    # 构建 CSV
    lines = [
        f"# GPU: {hw_info.gpu_full_name}",
        f"# Mode: {model_name}",
        f"# dtype: {dtype.upper()}",
        f"# Sparsity: {sparsity} (K_factor={k_factor:.3f})",
        f"# Time: {datetime.datetime.now().isoformat()}",
        "M,N,K,K_slide,cublaslt_tops,cusparselt_tops,speedup,cublaslt_lat_us,cusparselt_lat_us",
    ]
    
    # 合并结果
    cublaslt_data = {}
    for nk_res in cublaslt_results.get("results", []):
        N, K = nk_res["N"], nk_res["K"]
        for M, m_res in nk_res.get("m_results", {}).items():
            results = m_res.get("results", [])
            if results:
                best = results[0]
                cublaslt_data[(M, N, K)] = {
                    "tops": best["tops"],
                    "lat_us": best["lat_us"],
                }
    
    cusparselt_data = {}
    for nk_res in cusparselt_results.get("results", []):
        N, K = nk_res["N"], nk_res["K"]
        K_slide = calculate_k_slide(K, sparsity, dtype=dtype)
        for M, m_res in nk_res.get("m_results", {}).items():
            results = m_res.get("results", [])
            if results:
                best = results[0]
                cusparselt_data[(M, N, K_slide)] = {
                    "tops": best["tops"],
                    "lat_us": best["lat_us"],
                    "orig_K": K,
                }
    
    # 匹配并生成结果
    for (M, N, K), dense in sorted(cublaslt_data.items()):
        K_slide = calculate_k_slide(K, sparsity, dtype=dtype)
        sparse_key = (M, N, K_slide)
        
        if sparse_key in cusparselt_data:
            sparse = cusparselt_data[sparse_key]
            speedup = compute_speedup(dense["tops"], sparse["tops"])
            
            line = (
                f"{M},{N},{K},{K_slide},"
                f"{dense['tops']:.6f},{sparse['tops']:.6f},{speedup:.4f},"
                f"{dense['lat_us']:.3f},{sparse['lat_us']:.3f}"
            )
            lines.append(line)
    
    csv_path.write_text("\n".join(lines))
    print(f"Speedup results saved to: {csv_path}")
    
    return csv_path


# =============================================================================
# CSV 表头构建
# =============================================================================

def get_output_dtype_name(dtype: str, backend: str) -> str:
    """
    获取实际的输出类型名称
    """
    dtype_lower = dtype.lower()
    backend_lower = backend.lower()
    
    # FP16/BF16 输出与输入相同
    if dtype_lower in ("fp16", "bf16"):
        return dtype.upper()
    
    # INT8
    if dtype_lower == "int8":
        if backend_lower == "cublaslt":
            return "INT32"
        else:
            return "BF16"
    
    # FP8/FP4 输出 BF16
    if dtype_lower in ("fp8e4m3", "fp8", "fp4e2m1", "fp4"):
        return "BF16"
    
    return dtype.upper()


def build_csv_header_lines(
    *,
    model_name: str,
    dtype: str,
    mode: str,
    warmup: int,
    repeat: int,
    verify: bool,
    m_list: List[int],
    nk_list: List[Tuple[int, int]],
    backend: str,
    alg_count: int = 0,
    config_count: int = 0,
    sparsity: Optional[str] = None,
) -> List[str]:
    """
    构建 CSV 文件的元数据头部行
    """
    out_dtype_name = get_output_dtype_name(dtype, backend)
    
    lines = [
        f"# GPU: {hw_info.gpu_full_name}",
        f"# CC: {hw_info.cc_tag}",
        f"# Mode: {mode}",
        f"# Model: {model_name}",
        f"# Backend: {backend}",
        f"# dtype: {dtype.upper()} -> {out_dtype_name}",
        f"# alg_count: {alg_count}, config_count: {config_count}",
        f"# torch: {torch.__version__}",
        f"# CUDA driver: {hw_info.cuda_driver_version}, runtime: {hw_info.cuda_runtime_version}",
        f"# warmup={warmup}, repeat={repeat}, verify={verify}",
        f"# M_list: {m_list}",
        f"# NK_list: {nk_list}",
    ]
    
    if sparsity:
        k_factor = get_k_expansion_factor(sparsity)
        lines.append(f"# Sparsity: {sparsity} (K_factor={k_factor:.3f})")
    
    lines.append(f"# Time: {datetime.datetime.now().isoformat()}")
    
    return lines


# =============================================================================
# 增量保存与原子写入
# =============================================================================

def atomic_write_json(filepath: Path, data: Dict, indent: int = 2) -> None:
    """
    原子写入 JSON 文件（使用临时文件 + 重命名）
    
    防止写入中断导致文件损坏。
    
    Args:
        filepath: 目标文件路径
        data: 要写入的数据字典
        indent: JSON 缩进
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # 在同一目录创建临时文件，确保在同一文件系统上
    temp_fd, temp_path = tempfile.mkstemp(
        suffix='.tmp',
        prefix=filepath.stem + '_',
        dir=filepath.parent
    )
    
    try:
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())  # 确保数据写入磁盘
        
        # 原子重命名（在同一文件系统上是原子操作）
        shutil.move(temp_path, filepath)
    except Exception:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def atomic_write_csv(filepath: Path, lines: List[str]) -> None:
    """
    原子写入 CSV 文件（使用临时文件 + 重命名）
    
    防止写入中断导致文件损坏。
    
    Args:
        filepath: 目标文件路径
        lines: CSV 行列表
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # 在同一目录创建临时文件
    temp_fd, temp_path = tempfile.mkstemp(
        suffix='.tmp',
        prefix=filepath.stem + '_',
        dir=filepath.parent
    )
    
    try:
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
            f.flush()
            os.fsync(f.fileno())
        
        shutil.move(temp_path, filepath)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


class IncrementalResultSaver:
    """
    增量结果保存器
    
    支持在搜索过程中增量保存结果，防止长时间运行的任务因中断导致数据丢失。
    每完成一个 NK 组合的搜索后自动保存当前进度。
    
    特性：
    - 使用临时文件 + 原子重命名，防止写入中断导致文件损坏
    - 自动维护 CSV 和 JSON 两种格式
    - 最终保存的文件格式与原来完全一致
    
    使用示例:
        saver = IncrementalResultSaver(
            out_dir=out_dir,
            model_name="Qwen2.5-0.5B",
            dtype="fp8e4m3",
            backend="cuBLASLt",
            mode="model",
            warmup=25,
            repeat=100,
            m_list=[16, 128, 1024, 4096],
            nk_list=[(4096, 4096), (4096, 11008)],
        )
        
        for nk_res in search_results:
            saver.add_nk_result(nk_res)
        
        csv_path, json_path = saver.finalize()
    """
    
    def __init__(
        self,
        out_dir: Path,
        model_name: str,
        dtype: str,
        backend: str,
        mode: str,
        warmup: int,
        repeat: int,
        m_list: List[int],
        nk_list: List[Tuple[int, int]],
        sparsity: Optional[str] = None,
    ):
        """
        初始化增量保存器
        
        Args:
            out_dir: 输出目录（不含 hw_folder 和 dtype_folder）
            model_name: 模型名称
            dtype: 数据类型
            backend: 后端名称 (cuBLASLt / cuSPARSELt)
            mode: 运行模式 (model / square)
            warmup: 预热次数
            repeat: 测量次数
            m_list: M 列表
            nk_list: NK 列表
            sparsity: 稀疏度配置（仅 cuSPARSELt）
        """
        self.out_dir = Path(out_dir)
        self.model_name = model_name
        self.dtype = dtype
        self.backend = backend
        self.mode = mode
        self.warmup = warmup
        self.repeat = repeat
        self.m_list = m_list
        self.nk_list = nk_list
        self.sparsity = sparsity
        
        # 搜索结果
        self.results: List[Dict] = []
        self.max_alg_count = 0
        self.max_config_count = 0
        self.search_stats = {"total": 0, "success": 0, "failed": 0, "errors": 0}
        self.skipped_nk: List[Dict] = []
        
        # 构建输出路径
        self.subdir = build_output_dir(self.out_dir, self.dtype)
        self.subdir.mkdir(parents=True, exist_ok=True)
        
        csv_filename = build_result_filename("alg_search", model_name, "csv", sparsity)
        json_filename = build_result_filename("alg_search", model_name, "json", sparsity)
        self.csv_path = self.subdir / csv_filename
        self.json_path = self.subdir / json_filename
        
        # 进度 JSON 路径（用于增量保存）
        progress_filename = build_result_filename("alg_search", model_name, "progress.json", sparsity)
        self.progress_path = self.subdir / progress_filename
    
    def add_nk_result(
        self,
        nk_result: Dict,
        save_progress: bool = True,
    ) -> None:
        """
        添加一个 NK 组合的搜索结果
        
        Args:
            nk_result: NK 搜索结果，格式:
                {
                    "nk_id": int,
                    "N": int,
                    "K": int,
                    "m_results": {M: {...}, ...},
                    "skipped": bool,
                    "skip_reason": str or None,
                }
            save_progress: 是否立即保存进度文件
        """
        self.results.append(nk_result)
        
        # 更新统计信息
        if nk_result.get("skipped", False):
            self.skipped_nk.append({
                "N": nk_result["N"],
                "K": nk_result["K"],
                "reason": nk_result.get("skip_reason", "unknown"),
            })
        
        # 更新 alg_count 和搜索统计
        for M, m_res in nk_result.get("m_results", {}).items():
            alg_count = m_res.get("alg_count", 0)
            if alg_count > self.max_alg_count:
                self.max_alg_count = alg_count
            
            config_count = m_res.get("config_count", 0)
            if config_count > self.max_config_count:
                self.max_config_count = config_count
            
            # 更新搜索统计
            self.search_stats["total"] += 1
            if m_res.get("error"):
                self.search_stats["errors"] += 1
            elif m_res.get("num_valid", 0) > 0:
                self.search_stats["success"] += 1
            else:
                self.search_stats["failed"] += 1
        
        # 增量保存进度
        if save_progress:
            self._save_progress()
    
    def _convert_bytes_to_base64(self, obj: Any) -> Any:
        """递归将 bytes 类型转换为 base64 编码字符串，使其可 JSON 序列化"""
        if isinstance(obj, bytes):
            return base64.b64encode(obj).decode('ascii')
        elif isinstance(obj, dict):
            return {k: self._convert_bytes_to_base64(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_bytes_to_base64(item) for item in obj]
        else:
            return obj
    
    def _save_progress(self) -> None:
        """保存当前进度到临时 JSON 文件"""
        # 将 results 中的 bytes 类型转换为 base64
        serializable_results = self._convert_bytes_to_base64(self.results)
        
        progress_data = {
            "meta": self._build_meta(),
            "results": serializable_results,
            "completed_nk_count": len(self.results),
            "total_nk_count": len(self.nk_list),
            "search_stats": self.search_stats,
            "last_updated": datetime.datetime.now().isoformat(),
        }
        atomic_write_json(self.progress_path, progress_data)
    
    def _build_meta(self) -> Dict:
        """构建元数据"""
        meta = {
            "gpu_name": hw_info.gpu_full_name,
            "gpu_short_name": hw_info.gpu_name,
            "compute_capability": hw_info.cc_tag,
            "model_name": self.model_name,
            "mode": self.mode,
            "backend": self.backend,
            "dtype": self.dtype,
            "alg_count": self.max_alg_count,
            "warmup": self.warmup,
            "repeat": self.repeat,
            "torch_version": torch.__version__,
            "time": datetime.datetime.now().isoformat(),
            "M_list": self.m_list,
            "NK_list": [[n, k] for n, k in self.nk_list],
            "search_stats": self.search_stats,
        }
        
        if self.sparsity:
            meta["sparsity"] = self.sparsity
        
        if self.max_config_count > 0:
            meta["config_count"] = self.max_config_count
        
        if self.skipped_nk:
            meta["skipped_nk"] = self.skipped_nk
        
        return meta
    
    def _build_nk_entries(self) -> Dict:
        """构建 nk_entries 结构（与原格式一致）"""
        nk_entries = {}
        
        for nk_res in self.results:
            N, K = nk_res["N"], nk_res["K"]
            nk_key = f"({N},{K})"
            
            m_thresholds = []
            alg_by_m = {}
            errors_by_m = {}
            
            for M, m_res in nk_res.get("m_results", {}).items():
                results = m_res.get("results", [])
                error = m_res.get("error")
                
                if error:
                    errors_by_m[str(M)] = error
                
                if results:
                    m_thresholds.append(M)
                    r = results[0]  # top1
                    
                    entry_data = {
                        "workspace": r.get("workspace", 0),
                        "tops": r["tops"],
                        "lat_us": r["lat_us"],
                    }
                    
                    # cuBLASLt 特有: algo_data
                    if "algo_data" in r:
                        algo_b64 = base64.b64encode(r["algo_data"]).decode('ascii')
                        entry_data["algo_data"] = algo_b64
                    
                    # cuSPARSELt 特有: alg_id, split_k
                    if "alg_id" in r:
                        entry_data["alg_id"] = r["alg_id"]
                    if "split_k" in r:
                        entry_data["split_k"] = r.get("split_k", 1)
                    
                    alg_by_m[str(M)] = entry_data
            
            entry = {
                "m_thresholds": m_thresholds,
                "alg_by_m": alg_by_m,
            }
            if errors_by_m:
                entry["errors_by_m"] = errors_by_m
            
            nk_entries[nk_key] = entry
        
        return nk_entries
    
    def _build_csv_lines(self) -> List[str]:
        """构建 CSV 内容行"""
        header_lines = build_csv_header_lines(
            model_name=self.model_name,
            dtype=self.dtype,
            mode=self.mode,
            warmup=self.warmup,
            repeat=self.repeat,
            verify=False,
            m_list=self.m_list,
            nk_list=self.nk_list,
            backend=self.backend,
            alg_count=self.max_alg_count,
            config_count=self.max_config_count,
            sparsity=self.sparsity,
        )
        
        csv_lines = list(header_lines)
        
        # 根据 backend 确定 CSV 列
        if self.backend.lower() == "cusparselt":
            csv_lines.append(
                "M,N,K,alg_count,tops_1,lat_us_1,alg_id_1,split_k_1,workspace_1,"
                "tops_2,lat_us_2,alg_id_2,split_k_2,workspace_2,"
                "tops_3,lat_us_3,alg_id_3,split_k_3,workspace_3"
            )
        else:  # cuBLASLt
            csv_lines.append(
                "M,N,K,alg_count,tops_1,lat_us_1,alg_id_1,workspace_1,"
                "tops_2,lat_us_2,alg_id_2,workspace_2,"
                "tops_3,lat_us_3,alg_id_3,workspace_3"
            )
        
        csv_rows = []
        for nk_res in self.results:
            N, K = nk_res["N"], nk_res["K"]
            
            for M, m_res in nk_res.get("m_results", {}).items():
                results = m_res.get("results", [])
                values = [str(M), str(N), str(K), str(m_res.get("alg_count", 0))]
                
                for k in range(3):
                    if k < len(results):
                        r = results[k]
                        if self.backend.lower() == "cusparselt":
                            values.extend([
                                f"{r['tops']:.6f}",
                                f"{r['lat_us']:.3f}",
                                str(r.get('alg_id', -1)),
                                str(r.get('split_k', 1)),
                                str(r.get('workspace', 0)),
                            ])
                        else:  # cuBLASLt
                            values.extend([
                                f"{r['tops']:.6f}",
                                f"{r['lat_us']:.3f}",
                                str(r.get('alg_id', -1)),
                                str(r.get('workspace', 0)),
                            ])
                    else:
                        if self.backend.lower() == "cusparselt":
                            values.extend(["", "", "", "", ""])
                        else:
                            values.extend(["", "", "", ""])
                
                csv_rows.append((M, N, K, ",".join(values)))
        
        csv_rows.sort(key=lambda x: (x[0], x[1], x[2]))
        for _, _, _, line in csv_rows:
            csv_lines.append(line)
        
        return csv_lines
    
    def finalize(self, delete_progress: bool = True) -> Tuple[Path, Path]:
        """
        最终保存并返回文件路径
        
        Args:
            delete_progress: 是否删除 progress 文件（默认 True）
        
        Returns:
            (csv_path, json_path)
        """
        # 构建最终 JSON
        json_payload = {
            "meta": self._build_meta(),
            "nk_entries": self._build_nk_entries(),
        }
        
        # 原子写入最终文件
        atomic_write_json(self.json_path, json_payload)
        atomic_write_csv(self.csv_path, self._build_csv_lines())
        
        # 根据参数决定是否删除进度文件
        if delete_progress and self.progress_path.exists():
            self.progress_path.unlink()
        
        print(f"    CSV: {self.csv_path}")
        print(f"    JSON: {self.json_path}")
        
        return self.csv_path, self.json_path
    
    def get_output_dir(self) -> Path:
        """获取输出目录"""
        return self.subdir


def safe_ctypes_call(func: Callable, *args, error_msg: str = "CUDA call failed") -> Tuple[int, Optional[str]]:
    """
    安全的 ctypes 调用包装器
    
    捕获 ctypes 调用中可能发生的异常。
    
    Args:
        func: ctypes 函数
        *args: 函数参数
        error_msg: 错误消息前缀
    
    Returns:
        (return_code, error_message): 返回码和错误消息（成功时为 None）
    """
    try:
        ret = func(*args)
        return ret, None
    except OSError as e:
        return -1, f"{error_msg}: OSError - {e}"
    except Exception as e:
        return -1, f"{error_msg}: {type(e).__name__} - {e}"


# =============================================================================
# 导出接口
# =============================================================================

__all__ = [
    # 常量
    "DTYPE_CONFIG",
    "SUPPORTED_DTYPES",
    "DEFAULT_SPARSITY_LIST",
    "ALIGNMENT",
    "DEFAULT_M_LIST",
    "M_QUICK_LIST",
    "SQUARE_M_LIST",
    # 硬件检测
    "get_current_cc",
    "check_dtype_support",
    "get_supported_dtypes_for_gpu",
    "check_cusparselt_support",
    "check_segment_k_support",
    # Sparsity 计算
    "parse_sparsity_config",
    "calculate_k_slide",
    "get_k_expansion_factor",
    "get_sparsity_ratio",
    "pad_to_alignment",
    "get_sparsity_list_for_benchmark",
    # 编译与加载
    "build_benchmark_extension",
    "load_benchmark_extension",
    # 数据准备
    "quantize_int8",
    "to_fp8_e4m3",
    "to_fp16",
    "to_bf16",
    "quantize_tensor",
    "get_output_torch_dtype",
    # NK 列表
    "get_nk_list_for_benchmark",
    "get_nk_list_for_search",
    "model_base_name",
    # 文件命名与目录
    "build_hw_folder_name",
    "build_dtype_folder_name",
    "build_output_dir",
    "build_result_filename",
    # 结果整合
    "compute_speedup",
    "merge_benchmark_results",
    "build_csv_header_lines",
    # 增量保存与原子写入
    "atomic_write_json",
    "atomic_write_csv",
    "IncrementalResultSaver",
    "safe_ctypes_call",
    # 重导出
    "hw_info",
]
