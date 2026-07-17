#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Quant Only Kernel Autotune & Code Generation Script

基于 per-row kernel 设计：
- 每行一个 program（grid = M）
- BLOCK_K 是主要调优参数
- M 虽然不在 kernel 块参数中，但影响最佳 num_warps/num_stages
- autotune key = ['M', 'K']

统一 FP8/INT8 设计：
- 生成单一文件，包含 FP8 和 INT8 两个函数
- 只用 FP8 进行 autotune（IO-bound kernel，config 通用）
- 如果 FP8 不支持则 fallback 到 INT8 autotune
- 输出文件名不含 dtype 后缀

Usage:
    python3 autotune_autogen_quant_only.py           # 默认 FP8 autotune
    python3 autotune_autogen_quant_only.py --quick   # 快速模式

"""

import os
import sys
import argparse

# 优先使用系统 CUDA ptxas（支持更新的 GPU 架构如 sm_121）
# Triton 内置的 ptxas 版本可能较旧，不支持最新架构
_CUDA_PTXAS = "/usr/local/cuda/bin/ptxas"
if os.path.exists(_CUDA_PTXAS) and "TRITON_PTXAS_PATH" not in os.environ:
    os.environ["TRITON_PTXAS_PATH"] = _CUDA_PTXAS

import torch
import triton
import triton.language as tl
from pathlib import Path
from typing import Optional

# 设置路径以导入 slidesparse 模块
_SCRIPT_DIR = Path(__file__).parent
_SLIDESPARSE_ROOT = _SCRIPT_DIR.parent.parent  # slidesparse/
_PROJECT_ROOT = _SLIDESPARSE_ROOT.parent       # vllmbench/

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from slidesparse.utils import (
    build_filename,
    build_hw_dir_name,
    build_tuned_filename,
    get_python_version_tag,
    get_arch_tag,
    get_gpu_cc,
    get_gpu_name,
    get_nk_list_for_search,
    get_unique_k_values,
    model_base_name,
    DEFAULT_M_LIST,
    M_QUICK_LIST,
)

# 将 csrc 目录添加到 sys.path 以导入 utils
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from utils import get_quant_autotune_configs


def get_output_filename(model_name: Optional[str] = None) -> str:
    """Generate output filename: quant_only_tuned[_{base_model}].py
    
    使用 base name（去掉 -INT8/-FP8 后缀），因为 Triton autotune 结果
    对 INT8/FP8 相同，一个文件可以被两种量化类型共用。
    """
    if model_name:
        base_name = model_base_name(model_name)
        return build_tuned_filename("quant_only_tuned", base_name, ext=".py")
    return build_tuned_filename("quant_only_tuned", None, ext=".py")


# Get autotune configs
AUTOTUNE_CONFIGS = get_quant_autotune_configs()


# =============================================================================
# Test Matrix Sizes (默认值，可通过命令行参数覆盖)
# =============================================================================

# 使用顶层 DEFAULT_M_LIST 作为默认值
M_VALUES = list(DEFAULT_M_LIST)

# 默认 warmup/repeat 次数
DEFAULT_WARMUP = 25
DEFAULT_REPEAT = 100



# =============================================================================
# FP8 Autotune Kernel (per-row design)
# =============================================================================

@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=['M', 'K'],
    warmup=5,
    rep=30,
)
@triton.jit
def _quant_only_fp8_kernel_autotune(
    x_ptr, out_ptr, scale_ptr,
    M, K: tl.constexpr,
    stride_xm, stride_om,
    BLOCK_K: tl.constexpr,
):
    """FP8 per-row quantization kernel for autotuning"""
    row = tl.program_id(0)
    
    FP8_MAX: tl.constexpr = 448.0
    MIN_SCALE: tl.constexpr = 1.0 / (448.0 * 512.0)
    
    x_row_ptr = x_ptr + row * stride_xm
    out_row_ptr = out_ptr + row * stride_om
    
    # Pass 1: 计算 absmax
    absmax = tl.zeros((), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x_val = tl.load(x_row_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        absmax = tl.maximum(absmax, tl.max(tl.abs(x_val)))
    
    # 计算 scale
    absmax = tl.maximum(absmax, 1e-12)
    scale = tl.maximum(absmax / FP8_MAX, MIN_SCALE)
    inv_scale = FP8_MAX / absmax
    
    tl.store(scale_ptr + row, scale)
    
    # Pass 2: 量化
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x_val = tl.load(x_row_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        y_val = tl.clamp(x_val * inv_scale, -FP8_MAX, FP8_MAX)
        tl.store(out_row_ptr + offs_k, y_val.to(tl.float8e4nv), mask=mask_k)


# =============================================================================
# INT8 Autotune Kernel (per-row design)
# =============================================================================

@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=['M', 'K'],
    warmup=5,
    rep=30,
)
@triton.jit
def _quant_only_int8_kernel_autotune(
    x_ptr, out_ptr, scale_ptr,
    M, K: tl.constexpr,
    stride_xm, stride_om,
    BLOCK_K: tl.constexpr,
):
    """INT8 per-row quantization kernel for autotuning"""
    row = tl.program_id(0)
    
    INT8_MAX: tl.constexpr = 127.0
    MIN_SCALE: tl.constexpr = 1.0 / (127.0 * 512.0)
    
    x_row_ptr = x_ptr + row * stride_xm
    out_row_ptr = out_ptr + row * stride_om
    
    # Pass 1: 计算 absmax
    absmax = tl.zeros((), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x_val = tl.load(x_row_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        absmax = tl.maximum(absmax, tl.max(tl.abs(x_val)))
    
    # 计算 scale
    absmax = tl.maximum(absmax, 1e-12)
    scale = tl.maximum(absmax / INT8_MAX, MIN_SCALE)
    inv_scale = INT8_MAX / absmax
    
    tl.store(scale_ptr + row, scale)
    
    # Pass 2: 量化
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x_val = tl.load(x_row_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        y_val = tl.clamp(tl.extra.cuda.libdevice.rint(x_val * inv_scale), -128.0, 127.0)
        tl.store(out_row_ptr + offs_k, y_val.to(tl.int8), mask=mask_k)


# =============================================================================
# Autotune Wrapper Functions
# =============================================================================

def quant_only_fp8_autotune(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 quantization with autotune - grid = (M,)"""
    M, K = x.shape
    
    # Padding: K -> 32 aligned, M -> 16 aligned
    K_padded = ((K + 31) // 32) * 32
    M_padded = ((M + 15) // 16) * 16
    
    out = torch.zeros(M_padded, K_padded, dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.ones(M_padded, dtype=torch.float32, device=x.device)
    
    # 每行一个 program (只处理有效的 M 行)
    _quant_only_fp8_kernel_autotune[(M,)](
        x, out, scale,
        M, K,
        x.stride(0), K_padded,  # output stride 使用 K_padded
    )
    return out, scale


def quant_only_int8_autotune(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """INT8 quantization with autotune - grid = (M,)"""
    M, K = x.shape
    
    # Padding: K -> 32 aligned, M -> 16 aligned
    K_padded = ((K + 31) // 32) * 32
    M_padded = ((M + 15) // 16) * 16
    
    out = torch.zeros(M_padded, K_padded, dtype=torch.int8, device=x.device)
    scale = torch.ones(M_padded, dtype=torch.float32, device=x.device)
    
    # 每行一个 program (只处理有效的 M 行)
    _quant_only_int8_kernel_autotune[(M,)](
        x, out, scale,
        M, K,
        x.stride(0), K_padded,  # output stride 使用 K_padded
    )
    return out, scale


# =============================================================================
# Tuning Runner
# =============================================================================

def check_fp8_support():
    """Check if the GPU supports FP8"""
    cc = get_gpu_cc()
    # cc format is "cc90", "cc89", "cc100", etc.
    # FP8 requires SM89+ (Ada Lovelace) or SM90+ (Hopper)
    # SM80 (A100) does NOT have native FP8 support
    cc_num = cc.replace("cc", "")  # "cc90" -> "90"
    cc_major = int(cc_num)
    return cc_major >= 89


def run_tuning():
    """
    Run autotune and collect best configs for each (M, K)
    
    Strategy:
    - Use FP8 autotune by default (more representative for modern GPUs)
    - Fallback to INT8 if FP8 is not supported
    - Config is transferable between FP8 and INT8 (IO-bound kernel)
    """
    use_fp8 = check_fp8_support()
    dtype_name = "FP8" if use_fp8 else "INT8 (fallback)"
    quant_func = quant_only_fp8_autotune if use_fp8 else quant_only_int8_autotune
    kernel_cache = _quant_only_fp8_kernel_autotune if use_fp8 else _quant_only_int8_kernel_autotune
    
    print(f"\nTuning with {dtype_name}...")
    if not use_fp8:
        print("  Note: FP8 not supported on this GPU, using INT8 for tuning.")
        print("  The config will be applied to both FP8 and INT8 kernels.")
    print(f"K values: {K_VALUES}")
    print(f"M values: {M_VALUES}")
    print(f"Configs: {len(AUTOTUNE_CONFIGS)}")
    print("=" * 70)
    
    results = {}
    max_M, max_K = max(M_VALUES), max(K_VALUES)
    
    # Pre-allocate buffers
    x_buf = torch.randn(max_M, max_K, dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()
    
    for k in K_VALUES:
        results[k] = {}
        print(f"\n[K={k}]")
        
        for m in M_VALUES:
            x = x_buf[:m, :k].contiguous()
            
            # Warmup & trigger autotune
            for _ in range(3):
                _ = quant_func(x)
            torch.cuda.synchronize()
            
            # Extract best config from cache by matching (M, K) prefix
            # Cache key format varies by Triton version, so we iterate and match
            best_config = None
            for key, cfg in kernel_cache.cache.items():
                if isinstance(key, tuple) and len(key) >= 2:
                    cached_m, cached_k = key[0], key[1]
                    if cached_m == m and cached_k == k:
                        best_config = cfg
                        break
            
            if best_config:
                block_k = best_config.kwargs.get('BLOCK_K', 0)
                num_warps = best_config.num_warps
                num_stages = best_config.num_stages
                
                results[k][m] = (block_k, num_warps, num_stages)
                print(f"  M={m:<6} -> BLOCK_K={block_k:<5} num_warps={num_warps:<2} num_stages={num_stages}")
            else:
                print(f"  M={m:<6} -> [WARN] No cache entry")
    
    return results


def build_branches(results):
    """Analyze results and build interval-based branch strategy"""
    branches = {}
    
    for k, m_configs in results.items():
        sorted_ms = sorted(m_configs.keys())
        if not sorted_ms:
            continue
        
        intervals = []
        prev_key = None
        interval_start = None
        
        for m in sorted_ms:
            cfg = m_configs[m]
            key = cfg  # (BLOCK_K, num_warps, num_stages)
            
            if key != prev_key:
                if interval_start is not None:
                    intervals.append((interval_start, prev_m, prev_key))
                interval_start = m
                prev_key = key
            prev_m = m
        
        if interval_start is not None:
            intervals.append((interval_start, prev_m, prev_key))
        
        branches[k] = intervals
    
    return branches


# =============================================================================
# Code Generator
# =============================================================================

def generate_kernel_code(branches, results) -> str:
    """Generate the tuned kernel Python file with both FP8 and INT8 kernels"""
    
    # Generate config selector function
    def gen_config_selector():
        lines = ["def _get_config(M: int, K: int) -> tuple:"]
        lines.append('    """Returns (BLOCK_K, num_warps, num_stages)"""')
        
        k_values = sorted(branches.keys())
        for i, k in enumerate(k_values):
            cond = "if" if i == 0 else "elif"
            lines.append(f"    {cond} K == {k}:")
            
            intervals = branches[k]
            for j, (m_start, m_end, cfg) in enumerate(intervals):
                block_k, num_warps, num_stages = cfg
                if j == 0:
                    lines.append(f"        if M <= {m_end}:")
                elif j == len(intervals) - 1:
                    lines.append(f"        else:")
                else:
                    lines.append(f"        elif M <= {m_end}:")
                lines.append(f"            return {block_k}, {num_warps}, {num_stages}")
        
        # Default fallback
        lines.append("    # Default fallback")
        lines.append("    if K <= 4096:")
        lines.append("        return 4096, 8, 2")
        lines.append("    return 8192, 8, 2")
        
        return "\n".join(lines)
    
    config_selector = gen_config_selector()
    
    code = f'''# Auto-generated by autotune_autogen_quant_only.py
# Target: {get_gpu_name()} ({get_gpu_cc()})
# Design: Per-row kernel (grid = M), Unified FP8/INT8
# DO NOT EDIT

import torch
import triton
import triton.language as tl




{config_selector}


# =============================================================================
# FP8 Kernel
# =============================================================================

@triton.jit
def _quant_only_fp8_kernel(
    x_ptr, out_ptr, scale_ptr,
    M, K: tl.constexpr,
    stride_xm, stride_om,
    BLOCK_K: tl.constexpr,
):
    """Per-token FP8 quantization kernel - one program per row"""
    row = tl.program_id(0)
    
    FP8_MAX: tl.constexpr = 448.0
    MIN_SCALE: tl.constexpr = 1.0 / (448.0 * 512.0)
    
    x_row_ptr = x_ptr + row * stride_xm
    out_row_ptr = out_ptr + row * stride_om
    
    # Pass 1: 计算 absmax
    absmax = tl.zeros((), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x_val = tl.load(x_row_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        absmax = tl.maximum(absmax, tl.max(tl.abs(x_val)))
    
    # 计算 scale
    absmax = tl.maximum(absmax, 1e-12)
    scale = tl.maximum(absmax / FP8_MAX, MIN_SCALE)
    inv_scale = FP8_MAX / absmax
    
    tl.store(scale_ptr + row, scale)
    
    # Pass 2: 量化
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x_val = tl.load(x_row_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        y_val = tl.clamp(x_val * inv_scale, -FP8_MAX, FP8_MAX)
        tl.store(out_row_ptr + offs_k, y_val.to(tl.float8e4nv), mask=mask_k)


def quant_only_fp8_triton(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    assert x.is_cuda and x.is_contiguous()
    M, K = x.shape
    
    # Padding: K -> 32 aligned, M -> 16 aligned
    K_padded = ((K + 31) // 32) * 32
    M_padded = ((M + 15) // 16) * 16
    
    # 使用 zeros 分配，padding 区域天然为 0（torch.compile 友好）
    out = torch.zeros(M_padded, K_padded, dtype=torch.float8_e4m3fn, device=x.device)
    # scale padding 为 1.0，避免 dequant 时除以 0
    scale = torch.ones(M_padded, dtype=torch.float32, device=x.device)
    
    BLOCK_K, num_warps, num_stages = _get_config(M, K)
    
    # Per-row: grid = (M,) - 只处理有效的 M 行
    _quant_only_fp8_kernel[(M,)](
        x, out, scale,
        M, K,
        x.stride(0), K_padded,  # output stride 使用 K_padded
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, scale


# =============================================================================
# INT8 Kernel
# =============================================================================

@triton.jit
def _quant_only_int8_kernel(
    x_ptr, out_ptr, scale_ptr,
    M, K: tl.constexpr,
    stride_xm, stride_om,
    BLOCK_K: tl.constexpr,
):
    """Per-token INT8 quantization kernel - one program per row"""
    row = tl.program_id(0)
    
    INT8_MAX: tl.constexpr = 127.0
    MIN_SCALE: tl.constexpr = 1.0 / (127.0 * 512.0)
    
    x_row_ptr = x_ptr + row * stride_xm
    out_row_ptr = out_ptr + row * stride_om
    
    # Pass 1: 计算 absmax
    absmax = tl.zeros((), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x_val = tl.load(x_row_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        absmax = tl.maximum(absmax, tl.max(tl.abs(x_val)))
    
    # 计算 scale
    absmax = tl.maximum(absmax, 1e-12)
    scale = tl.maximum(absmax / INT8_MAX, MIN_SCALE)
    inv_scale = INT8_MAX / absmax
    
    tl.store(scale_ptr + row, scale)
    
    # Pass 2: 量化
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x_val = tl.load(x_row_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        y_val = tl.clamp(tl.extra.cuda.libdevice.rint(x_val * inv_scale), -128.0, 127.0)
        tl.store(out_row_ptr + offs_k, y_val.to(tl.int8), mask=mask_k)


def quant_only_int8_triton(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    assert x.is_cuda and x.is_contiguous()
    M, K = x.shape
    
    # Padding: K -> 32 aligned, M -> 16 aligned
    K_padded = ((K + 31) // 32) * 32
    M_padded = ((M + 15) // 16) * 16
    
    # 使用 zeros 分配，padding 区域天然为 0（torch.compile 友好）
    out = torch.zeros(M_padded, K_padded, dtype=torch.int8, device=x.device)
    # scale padding 为 1.0，避免 dequant 时除以 0
    scale = torch.ones(M_padded, dtype=torch.float32, device=x.device)
    
    BLOCK_K, num_warps, num_stages = _get_config(M, K)
    
    # Per-row: grid = (M,) - 只处理有效的 M 行
    _quant_only_int8_kernel[(M,)](
        x, out, scale,
        M, K,
        x.stride(0), K_padded,  # output stride 使用 K_padded
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, scale


__all__ = ['quant_only_fp8_triton', 'quant_only_int8_triton', '_get_config']
'''
    return code


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Quant Only Kernel Autotune & Codegen (Unified FP8/INT8)")
    parser.add_argument('--info', action='store_true', help='Show naming info only')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--model', type=str, default=None, help='Model name (e.g., BitNet-2B4T-INT8)')
    parser.add_argument('--Lmax', type=int, default=None, help='Max L for slide sparse (e.g., 10). If set, generates NK for L=4,6,8,...,Lmax')
    parser.add_argument('--M-quick', action='store_true', dest='m_quick',
                        help='M-quick mode: use fixed M values [16, 128, 1024, 4096, 16384]')
    parser.add_argument('--m_list', type=str, default=None, 
                        help='M list, comma separated (e.g., 16,128,512,2048,16384)')
    parser.add_argument('--warmup', type=int, default=DEFAULT_WARMUP,
                        help=f'Warmup iterations for autotune (default: {DEFAULT_WARMUP})')
    parser.add_argument('--repeat', type=int, default=DEFAULT_REPEAT,
                        help=f'Repeat iterations for autotune (default: {DEFAULT_REPEAT})')
    args = parser.parse_args()
    
    # M 列表优先级: --m_list > --M-quick > DEFAULT_M_LIST
    global M_VALUES, K_VALUES
    if args.m_list:
        M_VALUES = [int(x.strip()) for x in args.m_list.split(",")]
    elif args.m_quick:
        M_VALUES = list(M_QUICK_LIST)
    else:
        M_VALUES = list(DEFAULT_M_LIST)
    
    # 使用统一的 NK 获取工具
    nk_list, model_name = get_nk_list_for_search(args.model, args.Lmax)
    K_VALUES = get_unique_k_values(nk_list)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return 1
    
    use_fp8 = check_fp8_support()
    tune_dtype = "FP8" if use_fp8 else "INT8 (fallback)"
    output_filename = get_output_filename(model_name if args.model else None)
    
    print("=" * 70)
    print("Quant Only Kernel Autotune (Per-Row Design, Unified FP8/INT8)")
    print("=" * 70)
    print(f"GPU:     {get_gpu_name()} ({get_gpu_cc()})")
    print(f"Python:  {get_python_version_tag()}")
    print(f"Arch:    {get_arch_tag()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton:  {triton.__version__}")
    print(f"Tune dtype: {tune_dtype}")
    print(f"Warmup:  {args.warmup}, Repeat: {args.repeat}")
    if args.model:
        print(f"Model:   {model_name}")
    if args.Lmax:
        print(f"Lmax:    {args.Lmax} (slide sparse L=4,6,...,{args.Lmax})")
    print(f"M values: {M_VALUES}")
    print(f"K values: {K_VALUES}")
    print(f"Output file: {output_filename}")
    
    if args.info:
        return 0
    
    # Step 1: Run autotune
    print("\n" + "=" * 70)
    print("Step 1: Running autotune...")
    print("=" * 70)
    results = run_tuning()
    
    # Step 2: Build branches
    print("\n" + "=" * 70)
    print("Step 2: Building branch strategy...")
    print("=" * 70)
    branches = build_branches(results)
    
    for k, intervals in branches.items():
        print(f"\nK={k}: {len(intervals)} intervals")
        for m_start, m_end, cfg in intervals:
            block_k, num_warps, num_stages = cfg
            print(f"  M=[{m_start}, {m_end}] -> BLOCK_K={block_k}, warps={num_warps}, stages={num_stages}")
    
    # Step 3: Generate code
    print("\n" + "=" * 70)
    print("Step 3: Generating kernel code...")
    print("=" * 70)
    
    kernel_code = generate_kernel_code(branches, results)
    
    # Determine output path
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # 使用硬件信息作为子目录
        output_dir = Path(__file__).parent / "build" / build_hw_dir_name()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / output_filename
    
    with open(output_file, "w") as f:
        f.write(kernel_code)
    
    print(f"\nGenerated: {output_file}")
    print(f"Size: {len(kernel_code)} bytes")
    print("\nDone!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
