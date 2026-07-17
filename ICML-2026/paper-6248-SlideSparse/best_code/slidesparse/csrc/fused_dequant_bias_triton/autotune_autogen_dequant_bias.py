#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Dequant + Bias Kernel Autotune & Code Generation Script

生成的 kernel 支持 BF16, FP32 和 INT32 三种输入（自动转为 FP32 计算，输出 BF16）

Usage:
    python3 autotune_autogen_dequant_bias.py
    python3 autotune_autogen_dequant_bias.py --quick

"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

# 优先使用系统 CUDA ptxas（支持更新的 GPU 架构如 sm_121）
# Triton 内置的 ptxas 版本可能较旧，不支持最新架构
_CUDA_PTXAS = "/usr/local/cuda/bin/ptxas"
if os.path.exists(_CUDA_PTXAS) and "TRITON_PTXAS_PATH" not in os.environ:
    os.environ["TRITON_PTXAS_PATH"] = _CUDA_PTXAS

import torch
import triton
import triton.language as tl

# 设置路径以导入 slidesparse 模块
_SCRIPT_DIR = Path(__file__).parent
_SLIDESPARSE_ROOT = _SCRIPT_DIR.parent.parent  # slidesparse/
_PROJECT_ROOT = _SLIDESPARSE_ROOT.parent       # vllmbench/

# 将项目根目录添加到 sys.path以支持 "from slidesparse.utils import ..."
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
    get_unique_n_values,
    model_base_name,
    DEFAULT_M_LIST,
    M_QUICK_LIST,
)

# 将 csrc 目录添加到 sys.path 以导入 utils
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from utils import get_dequant_autotune_configs


def get_output_filename(model_name: Optional[str] = None) -> str:
    """Generate output filename: dequant_bias_tuned[_{base_model}].py
    
    使用 base name（去掉 -INT8/-FP8 后缀），因为 Triton autotune 结果
    对 INT8/FP8 相同，一个文件可以被两种量化类型共用。
    """
    if model_name:
        base_name = model_base_name(model_name)
        return build_tuned_filename("dequant_bias_tuned", base_name, ext=".py")
    return build_tuned_filename("dequant_bias_tuned", None, ext=".py")


# Get autotune configs from utils
AUTOTUNE_CONFIGS = get_dequant_autotune_configs()


# =============================================================================
# Test Matrix Sizes (默认值，可通过命令行参数覆盖)
# =============================================================================

# 使用顶层 DEFAULT_M_LIST 作为默认值
M_VALUES = list(DEFAULT_M_LIST)

# 默认 warmup/repeat 次数
DEFAULT_WARMUP = 25
DEFAULT_REPEAT = 100


# =============================================================================
# Autotune Kernel (warmup/rep 由 run_tuning 动态设置)
# =============================================================================

# 全局变量用于动态设置 warmup/rep
_AUTOTUNE_WARMUP = DEFAULT_WARMUP
_AUTOTUNE_REP = DEFAULT_REPEAT


def _get_autotune_configs():
    """返回带有动态 warmup/rep 的 autotune configs"""
    return AUTOTUNE_CONFIGS


@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=['M', 'N'],
    warmup=25,  # 默认值，实际值在 run_tuning 中通过 cache reset 调整
    rep=100,
)
@triton.jit
def _dequant_bias_kernel_autotune(
    gemm_ptr, scale_a_ptr, scale_b_ptr, bias_ptr, out_ptr,
    M, N,
    stride_gm, stride_gn, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    INPUT_FP32: tl.constexpr, INPUT_INT32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_2d = mask_m[:, None] & mask_n[None, :]
    
    scale_a = tl.load(scale_a_ptr + offs_m, mask=mask_m, other=1.0)
    scale_b = tl.load(scale_b_ptr + offs_n, mask=mask_n, other=1.0)
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    
    gemm_offs = offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
    val = tl.load(gemm_ptr + gemm_offs, mask=mask_2d, other=0.0)
    
    # 转换为 FP32:
    if INPUT_INT32:
        val = val.to(tl.float32)
    elif not INPUT_FP32:
        val = val.to(tl.float32)
    
    val = val * scale_a[:, None] * scale_b[None, :] + bias[None, :]
    
    out_offs = offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptr + out_offs, val.to(tl.bfloat16), mask=mask_2d)


def _prepare_scale(scale: torch.Tensor, size: int) -> torch.Tensor:
    """Prepare scale tensor: view as 1D, ensure float32, contiguous."""
    if scale.numel() == 1:
        scale = scale.view(1).expand(size)
    else:
        scale = scale.view(-1)
    # Only convert if not already float32 and contiguous
    if scale.dtype != torch.float32:
        return scale.contiguous().float()
    return scale.contiguous() if not scale.is_contiguous() else scale


def dequant_bias_autotune(
    gemm_output: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    M, N = gemm_output.shape
    input_fp32 = gemm_output.dtype == torch.float32
    input_int32 = gemm_output.dtype == torch.int32
    
    scale_a = _prepare_scale(scale_a, M)
    scale_b = _prepare_scale(scale_b, N)
    
    bias = bias.view(-1)
    if bias.dtype != torch.bfloat16:
        bias = bias.to(torch.bfloat16)
    bias = bias.contiguous() if not bias.is_contiguous() else bias
    
    output = torch.empty((M, N), dtype=torch.bfloat16, device=gemm_output.device)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    
    _dequant_bias_kernel_autotune[grid](
        gemm_output, scale_a, scale_b, bias, output,
        M, N,
        gemm_output.stride(0), gemm_output.stride(1),
        output.stride(0), output.stride(1),
        INPUT_FP32=input_fp32,
        INPUT_INT32=input_int32,
    )
    return output


# =============================================================================
# Tuning Runner
# =============================================================================

def run_tuning():
    """Run autotune and collect best configs for each (M, N)"""
    # 使用 BF16 进行 autotune（生成的 kernel 同样支持 FP32 输入）
    input_dtype = torch.bfloat16
    
    print(f"\nTuning (input: BF16, kernel supports both BF16/FP32)...")
    print(f"N values: {N_VALUES}")
    print(f"M values: {len(M_VALUES)} points")
    print("=" * 70)
    
    results = {}
    max_M, max_N = max(M_VALUES), max(N_VALUES)
    
    # Pre-allocate buffers
    gemm_buf = torch.randn(max_M, max_N, dtype=input_dtype, device="cuda")
    scale_a_buf = torch.rand(max_M, dtype=torch.float32, device="cuda") * 0.1 + 0.01
    scale_b_buf = torch.rand(max_N, dtype=torch.float32, device="cuda") * 0.1 + 0.01
    bias_buf = torch.randn(max_N, dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()
    
    for n in N_VALUES:
        results[n] = {}
        print(f"\n[N={n}]")
        
        for m in M_VALUES:
            gemm = gemm_buf[:m, :n].contiguous()
            scale_a = scale_a_buf[:m]
            scale_b = scale_b_buf[:n]
            bias = bias_buf[:n]
            
            try:
                # Run autotune
                dequant_bias_autotune(gemm, scale_a, scale_b, bias)
                torch.cuda.synchronize()
                
                # Extract best config from cache
                best_cfg = None
                for key, cfg in _dequant_bias_kernel_autotune.cache.items():
                    if isinstance(key, tuple) and len(key) >= 2:
                        cached_m, cached_n = key[0], key[1]
                        if cached_m == m and cached_n == n:
                            best_cfg = cfg
                            break
                
                if best_cfg:
                    results[n][m] = {
                        'BLOCK_M': best_cfg.kwargs['BLOCK_M'],
                        'BLOCK_N': best_cfg.kwargs['BLOCK_N'],
                        'num_warps': best_cfg.num_warps,
                        'num_stages': best_cfg.num_stages,
                    }
                    cfg = results[n][m]
                    print(f"  M={m:<6} -> ({cfg['BLOCK_M']:>3}, {cfg['BLOCK_N']:>3}) w={cfg['num_warps']:<2} s={cfg['num_stages']}")
                else:
                    print(f"  M={m:<6} -> [cache miss]")
                    
            except Exception as e:
                print(f"  M={m:<6} -> ERROR: {e}")
    
    return results


def build_branches(results):
    """Analyze results and build interval-based branch strategy"""
    branches = {}
    
    for n, m_configs in results.items():
        sorted_ms = sorted(m_configs.keys())
        if not sorted_ms:
            continue
        
        intervals = []
        prev_key = None
        interval_start = None
        
        for m in sorted_ms:
            cfg = m_configs[m]
            cfg_key = (cfg['BLOCK_M'], cfg['BLOCK_N'], cfg['num_warps'], cfg['num_stages'])
            
            if cfg_key != prev_key:
                if prev_key is not None:
                    intervals.append((interval_start, m, m_configs[interval_start]))
                interval_start = m
                prev_key = cfg_key
        
        if interval_start is not None:
            intervals.append((interval_start, None, m_configs[interval_start]))
        
        branches[n] = intervals
    
    return branches


# =============================================================================
# Code Generator
# =============================================================================

def generate_kernel_code(branches) -> str:
    """Generate the tuned kernel Python file"""
    
    # Generate config selector function
    def gen_config_selector():
        lines = ["def _get_config(M: int, N: int) -> tuple:"]
        lines.append('    """Returns (BLOCK_M, BLOCK_N, num_warps, num_stages)"""')
        
        n_values = sorted(branches.keys())
        for i, n in enumerate(n_values):
            cond = "if" if i == 0 else "elif"
            lines.append(f"    {cond} N == {n}:")
            
            intervals = branches.get(n, [])
            if not intervals:
                lines.append("        return 64, 64, 8, 4")
                continue
            
            for j, (m_start, m_end, cfg) in enumerate(intervals):
                ret = f"{cfg['BLOCK_M']}, {cfg['BLOCK_N']}, {cfg['num_warps']}, {cfg['num_stages']}"
                if j == 0:
                    if m_end is None:
                        lines.append(f"        return {ret}")
                    else:
                        lines.append(f"        if M < {m_end}:")
                        lines.append(f"            return {ret}")
                elif m_end is None:
                    lines.append(f"        return {ret}")
                else:
                    lines.append(f"        elif M < {m_end}:")
                    lines.append(f"            return {ret}")
        
        # Default fallback for unknown N
        lines.append("    if M <= 128:")
        lines.append("        return 32, 64, 4, 4")
        lines.append("    elif M <= 4096:")
        lines.append("        return 64, 64, 8, 4")
        lines.append("    return 128, 64, 8, 4")
        
        return "\n".join(lines)
    
    config_selector = gen_config_selector()
    
    code = f'''# Auto-generated by autotune_autogen_dequant_bias.py
# Target: {get_gpu_name()} ({get_gpu_cc()})
# Supports BF16, FP32 and INT32 input (auto-converts to FP32 for computation)
# DO NOT EDIT

import torch
import triton
import triton.language as tl


def _prepare_scale(scale: torch.Tensor, size: int) -> torch.Tensor:
    """Prepare scale tensor: view as 1D, ensure float32, contiguous."""
    if scale.numel() == 1:
        scale = scale.view(1).expand(size)
    else:
        scale = scale.view(-1)
    # Only convert if not already float32 and contiguous
    if scale.dtype != torch.float32:
        return scale.contiguous().float()
    return scale.contiguous() if not scale.is_contiguous() else scale


{config_selector}


@triton.jit
def _dequant_bias_kernel(
    gemm_ptr, scale_a_ptr, scale_b_ptr, bias_ptr, out_ptr,
    M, N, stride_gm, stride_gn, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_2d = mask_m[:, None] & mask_n[None, :]
    
    scale_a = tl.load(scale_a_ptr + offs_m, mask=mask_m, other=1.0)
    scale_b = tl.load(scale_b_ptr + offs_n, mask=mask_n, other=1.0)
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    
    gemm_offs = offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
    val = tl.load(gemm_ptr + gemm_offs, mask=mask_2d, other=0.0)
    val = val.to(tl.float32) * scale_a[:, None] * scale_b[None, :] + bias[None, :]
    
    out_offs = offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptr + out_offs, val.to(tl.bfloat16), mask=mask_2d)


def dequant_bias_triton(
    gemm_output: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    assert gemm_output.is_cuda and gemm_output.is_contiguous()
    M, N = gemm_output.shape
    
    scale_a = _prepare_scale(scale_a, M)
    scale_b = _prepare_scale(scale_b, N)
    
    bias = bias.view(-1)
    if bias.dtype != torch.bfloat16:
        bias = bias.to(torch.bfloat16)
    bias = bias.contiguous() if not bias.is_contiguous() else bias
    
    output = torch.empty((M, N), dtype=torch.bfloat16, device=gemm_output.device)
    
    BLOCK_M, BLOCK_N, num_warps, num_stages = _get_config(M, N)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    _dequant_bias_kernel[grid](
        gemm_output, scale_a, scale_b, bias, output,
        M, N,
        gemm_output.stride(0), gemm_output.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages,
    )
    return output.to(out_dtype) if out_dtype != torch.bfloat16 else output


__all__ = ['dequant_bias_triton', '_get_config']
'''
    return code


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Dequant+Bias Kernel Autotune & Codegen")
    parser.add_argument('--info', action='store_true', help='Show naming info only')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: ./build)')
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
    global M_VALUES, N_VALUES
    if args.m_list:
        M_VALUES = [int(x.strip()) for x in args.m_list.split(",")]
    elif args.m_quick:
        M_VALUES = list(M_QUICK_LIST)
    else:
        M_VALUES = list(DEFAULT_M_LIST)
    
    # 使用统一的 NK 获取工具
    nk_list, model_name = get_nk_list_for_search(args.model, args.Lmax)
    N_VALUES = get_unique_n_values(nk_list)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return 1
    
    output_filename = get_output_filename(model_name if args.model else None)
    
    print("=" * 70)
    print("Dequant + Bias Kernel Autotune")
    print("=" * 70)
    print(f"GPU:     {get_gpu_name()} ({get_gpu_cc()})")
    print(f"Python:  {get_python_version_tag()}")
    print(f"Arch:    {get_arch_tag()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton:  {triton.__version__}")
    print(f"Input:   BF16/FP32 (auto-handled)")
    print(f"Warmup:  {args.warmup}, Repeat: {args.repeat}")
    if args.model:
        print(f"Model:   {model_name}")
    if args.Lmax:
        print(f"Lmax:    {args.Lmax} (slide sparse L=4,6,...,{args.Lmax})")
    print(f"M values: {M_VALUES}")
    print(f"N values: {N_VALUES}")
    print(f"Output:  {output_filename}")
    
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
    
    for n, intervals in branches.items():
        print(f"\nN={n}: {len(intervals)} intervals")
        for m_start, m_end, cfg in intervals:
            end_str = f"< {m_end}" if m_end else "to max"
            print(f"  M >= {m_start:<5} {end_str:<12} -> ({cfg['BLOCK_M']}, {cfg['BLOCK_N']}) w={cfg['num_warps']} s={cfg['num_stages']}")
    
    # Step 3: Generate code
    print("\n" + "=" * 70)
    print("Step 3: Generating kernel code...")
    print("=" * 70)
    
    kernel_code = generate_kernel_code(branches)
    
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
