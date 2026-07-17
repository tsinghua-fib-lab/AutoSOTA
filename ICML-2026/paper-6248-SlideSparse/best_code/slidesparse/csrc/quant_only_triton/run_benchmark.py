#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Quant Only Kernel Benchmark

Usage:
    python3 run_benchmark.py --dtype fp8   # Test FP8 quantization
    python3 run_benchmark.py --dtype int8  # Test INT8 quantization
"""

import sys
import argparse
import importlib.util
from pathlib import Path
from types import ModuleType

import torch
import triton
import triton.language as tl
import triton.testing as testing

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
    get_gpu_cc,
    get_nk_list_for_search,
    get_unique_k_values,
)


# =============================================================================
# Test Configuration
# =============================================================================

# M values for throughput benchmark (same as autotune M-quick)
M_VALUES_BENCH = [16, 128, 1024, 4096, 16384]
# M/K values for correctness test (include non-aligned values)
M_VALUES_CORRECTNESS = [16, 17, 100, 128, 1024, 4096, 16384]
K_VALUES_CORRECTNESS = [2560, 2561, 6912]

# K values for benchmark will be set dynamically from model
K_VALUES_BENCH: list[int] = []

WARMUP = 25
REP = 100


# =============================================================================
# Theoretical Baseline (pure memory copy: BF16 -> FP8/INT8) - Per-row design
# =============================================================================

@triton.jit
def _memcpy_bf16_to_fp8_kernel(
    in_ptr, out_ptr, M, K: tl.constexpr,
    stride_im, stride_om,
    BLOCK_K: tl.constexpr,
):
    """Pure memory copy baseline: BF16 input -> FP8 output (per-row)"""
    row = tl.program_id(0)
    
    in_row_ptr = in_ptr + row * stride_im
    out_row_ptr = out_ptr + row * stride_om
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        val = tl.load(in_row_ptr + offs_k, mask=mask_k, other=0.0)
        tl.store(out_row_ptr + offs_k, val.to(tl.float8e4nv), mask=mask_k)


@triton.jit
def _memcpy_bf16_to_int8_kernel(
    in_ptr, out_ptr, M, K: tl.constexpr,
    stride_im, stride_om,
    BLOCK_K: tl.constexpr,
):
    """Pure memory copy baseline: BF16 input -> INT8 output (per-row)"""
    row = tl.program_id(0)
    
    in_row_ptr = in_ptr + row * stride_im
    out_row_ptr = out_ptr + row * stride_om
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        val = tl.load(in_row_ptr + offs_k, mask=mask_k, other=0.0)
        # Truncate to INT8 range without actual quantization
        val_f32 = val.to(tl.float32)
        val_clamped = tl.clamp(val_f32 * 100.0, -128.0, 127.0)  # dummy scale
        tl.store(out_row_ptr + offs_k, val_clamped.to(tl.int8), mask=mask_k)


# Tensor Cache for baseline (与 quant kernel 使用相同的分配策略以公平比较)
_baseline_fp8_out_cache: dict = {}
_baseline_int8_out_cache: dict = {}


def make_theoretical_baseline(get_config_func, dtype: str):
    """Create theoretical baseline function that uses tuned config (per-row design)
    
    Note: 使用 tensor cache + zero_() 与 quant kernel 使用相同的分配策略，确保公平比较
    """
    
    def theoretical_baseline_fp8(x: torch.Tensor) -> torch.Tensor:
        """Pure memory copy baseline: BF16 -> FP8 (with same allocation overhead as quant kernel)"""
        M, K = x.shape
        # 使用相同的 padding 策略
        K_padded = ((K + 31) // 32) * 32
        M_padded = ((M + 15) // 16) * 16
        
        # 使用 tensor cache + zero_()，与 quant kernel 相同
        key = (M_padded, K_padded, x.device.index if x.device.index is not None else 0)
        if key not in _baseline_fp8_out_cache:
            _baseline_fp8_out_cache[key] = torch.empty(M_padded, K_padded, dtype=torch.float8_e4m3fn, device=x.device)
        output = _baseline_fp8_out_cache[key]
        output.zero_()  # 与 quant kernel 相同的初始化开销
        
        BLOCK_K, num_warps, num_stages = get_config_func(M, K)
        
        # Per-row: grid = (M,) - 只处理有效的 M 行
        _memcpy_bf16_to_fp8_kernel[(M,)](
            x, output, M, K,
            x.stride(0), K_padded,  # 使用 K_padded 作为 output stride
            BLOCK_K=BLOCK_K,
            num_warps=num_warps, num_stages=num_stages,
        )
        return output
    
    def theoretical_baseline_int8(x: torch.Tensor) -> torch.Tensor:
        """Pure memory copy baseline: BF16 -> INT8 (with same allocation overhead as quant kernel)"""
        M, K = x.shape
        # 使用相同的 padding 策略
        K_padded = ((K + 31) // 32) * 32
        M_padded = ((M + 15) // 16) * 16
        
        # 使用 tensor cache + zero_()，与 quant kernel 相同
        key = (M_padded, K_padded, x.device.index if x.device.index is not None else 0)
        if key not in _baseline_int8_out_cache:
            _baseline_int8_out_cache[key] = torch.empty(M_padded, K_padded, dtype=torch.int8, device=x.device)
        output = _baseline_int8_out_cache[key]
        output.zero_()  # 与 quant kernel 相同的初始化开销
        
        BLOCK_K, num_warps, num_stages = get_config_func(M, K)
        
        # Per-row: grid = (M,) - 只处理有效的 M 行
        _memcpy_bf16_to_int8_kernel[(M,)](
            x, output, M, K,
            x.stride(0), K_padded,  # 使用 K_padded 作为 output stride
            BLOCK_K=BLOCK_K,
            num_warps=num_warps, num_stages=num_stages,
        )
        return output
    
    return theoretical_baseline_fp8 if dtype == "fp8" else theoretical_baseline_int8


# =============================================================================
# Load Kernels
# =============================================================================

def load_tuned_module(model_name: str | None = None) -> ModuleType | None:
    """Load the auto-tuned kernel module from build/hw_dir/ (unified FP8/INT8 file)"""
    # 使用硬件子目录
    build_dir = Path(__file__).parent / "build" / build_hw_dir_name()
    
    # autotune 生成的文件名：quant_only_tuned[_model].py
    filename = build_tuned_filename("quant_only_tuned", model_name, ext=".py")
    module_path = build_dir / filename
    
    if not module_path.exists():
        print(f"ERROR: Tuned kernel not found: {module_path}")
        print(f"Please run: python3 autotune_autogen_quant_only.py --model {model_name or 'MODEL'}")
        return None
    
    try:
        spec = importlib.util.spec_from_file_location("tuned_kernel", module_path)
        if spec is None or spec.loader is None:
            print(f"ERROR: Failed to load module: {module_path}")
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"ERROR: Failed to load tuned kernel: {e}")
        return None


def load_basic_module() -> ModuleType | None:
    """Load the basic kernel module (contains untuned kernel and pytorch reference)"""
    module_path = Path(__file__).parent / "basic_quant_only_triton.py"
    
    if not module_path.exists():
        print(f"ERROR: Basic kernel not found: {module_path}")
        return None
    
    spec = importlib.util.spec_from_file_location("basic_kernel", module_path)
    if spec is None or spec.loader is None:
        print(f"ERROR: Failed to load module: {module_path}")
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =============================================================================
# Correctness Test
# =============================================================================

def test_correctness(tuned_func, untuned_func, pytorch_ref_func, dtype: str):
    """Test correctness against PyTorch reference"""
    print("\n" + "=" * 70)
    print("Correctness Test")
    print("=" * 70)
    
    torch.manual_seed(42)
    all_passed = True
    
    # Sample shapes to avoid OOM (skip large M * K combinations)
    test_shapes = [(M, K) for M in M_VALUES_CORRECTNESS for K in K_VALUES_CORRECTNESS if M * K <= 64 * 1024 * 1024]
    
    for M, K in test_shapes:
        x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        
        ref_out, ref_scale = pytorch_ref_func(x)
        tuned_out, tuned_scale = tuned_func(x)
        untuned_out, untuned_scale = untuned_func(x)
        
        # Compare outputs (convert to float for comparison)
        # Note: Triton functions return padded output, extract valid region [:M, :K]
        ref_float = ref_out.float()
        tuned_float = tuned_out[:M, :K].float()
        untuned_float = untuned_out[:M, :K].float()
        
        diff_tuned_out = (ref_float - tuned_float).abs().max().item()
        diff_untuned_out = (ref_float - untuned_float).abs().max().item()
        diff_tuned_scale = (ref_scale - tuned_scale[:M]).abs().max().item()
        diff_untuned_scale = (ref_scale - untuned_scale[:M]).abs().max().item()
        
        # FP8 has limited precision. Due to floating-point rounding differences
        # in GPU division (1 ULP error in inv_scale), values very close to FP8 
        # quantization boundaries may round to adjacent representable values.
        # FP8E4M3 step sizes: 8 (64-128 range), 16 (128-256), 32 (256-448).
        # We allow up to 2x the step size to account for boundary cases.
        if dtype == "fp8":
            # Max step size in FP8E4M3 is 32 (for values near 448)
            # Allow 2x this for boundary rounding differences
            tol = 64.0
        else:
            # INT8 has uniform step size, smaller tolerance needed
            tol = 2.0
        scale_tol = 1e-5
        
        passed = (diff_tuned_out <= tol and diff_untuned_out <= tol and 
                  diff_tuned_scale < scale_tol and diff_untuned_scale < scale_tol)
        all_passed = all_passed and passed
        
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] M={M:<6} K={K:<5} "
              f"out_diff(tuned={diff_tuned_out:.4f}, untuned={diff_untuned_out:.4f}) "
              f"scale_diff(tuned={diff_tuned_scale:.2e}, untuned={diff_untuned_scale:.2e})")
        
        # Free memory
        del x, ref_out, ref_scale, tuned_out, tuned_scale, untuned_out, untuned_scale
        torch.cuda.empty_cache()
    
    print("-" * 70)
    print(f"Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


# =============================================================================
# Throughput Benchmark
# =============================================================================

def run_benchmark(tuned_func, untuned_func, baseline_func, dtype: str, k_values: list[int]):
    """Run throughput benchmark: theoretical vs untuned vs tuned"""
    print("\n" + "=" * 70)
    print("Throughput Benchmark")
    print("=" * 70)
    dtype_name = dtype.upper()
    print(f"Input: BF16, Output: {dtype_name}, Warmup: {WARMUP}, Rep: {REP}")
    print(f"Baseline: memcpy kernel (BF16 -> {dtype_name}) using same tuned config")
    print(f"K values: {k_values}")
    print()
    
    # Header
    print(f"{'M':<7} {'K':<6} | {'Baseline(us)':<12} {'Untuned(us)':<12} {'Tuned(us)':<11} | {'vs Base':<10} {'vs Untuned':<10}")
    print("-" * 90)
    
    results: list[dict] = []
    
    for M in M_VALUES_BENCH:
        for K in k_values:
            # Generate data
            x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
            
            # Benchmark baseline (pure memory copy with tuned config)
            t_baseline: float = testing.do_bench(
                lambda: baseline_func(x),
                warmup=WARMUP, rep=REP, return_mode="min"
            ) * 1000  # ms -> us
            
            # Benchmark untuned kernel
            t_untuned: float = testing.do_bench(
                lambda: untuned_func(x),
                warmup=WARMUP, rep=REP, return_mode="min"
            ) * 1000  # ms -> us
            
            # Benchmark tuned kernel
            t_tuned: float = testing.do_bench(
                lambda: tuned_func(x),
                warmup=WARMUP, rep=REP, return_mode="min"
            ) * 1000  # ms -> us
            
            # Speedup ratios
            speedup_vs_baseline = t_baseline / t_tuned  # <1 expected (quant slower than memcpy)
            speedup_vs_untuned = t_untuned / t_tuned    # >1 means tuned is faster
            
            results.append({
                'M': M, 'K': K,
                'baseline': t_baseline,
                'untuned': t_untuned,
                'tuned': t_tuned,
                'speedup_baseline': speedup_vs_baseline,
                'speedup_untuned': speedup_vs_untuned,
            })
            
            print(f"{M:<7} {K:<6} | {t_baseline:<12.2f} {t_untuned:<12.2f} {t_tuned:<11.2f} | {speedup_vs_baseline:<10.2f} {speedup_vs_untuned:<10.2f}")
            
            # Free memory
            del x
            torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "-" * 90)
    print("Summary:")
    avg_baseline = sum(r['speedup_baseline'] for r in results) / len(results)
    avg_untuned = sum(r['speedup_untuned'] for r in results) / len(results)
    max_untuned = max(r['speedup_untuned'] for r in results)
    min_untuned = min(r['speedup_untuned'] for r in results)
    print(f"  vs Baseline: Avg {avg_baseline:.2f}x  (expected <1, quant does compute, baseline only memcpy)")
    print(f"  vs Untuned:  Avg {avg_untuned:.2f}x  Min {min_untuned:.2f}x  Max {max_untuned:.2f}x")
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Quant Only Kernel Benchmark")
    parser.add_argument('--dtype', type=str, required=True, choices=['fp8', 'int8'],
                        help='Output dtype: fp8 or int8')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name (e.g., Qwen2.5-0.5B-INT8). If not specified, uses BitNet-2B-BF16 default.')
    parser.add_argument('--Lmax', type=int, default=None,
                        help='Max L for slide sparse (e.g., 10). If set, uses slided K values')
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return 1
    
    dtype = args.dtype
    dtype_tag = dtype.upper()
    
    # Get K values and model_name from unified tool
    nk_list, model_name = get_nk_list_for_search(args.model, args.Lmax)
    k_values = get_unique_k_values(nk_list)
    
    print("=" * 70)
    print("Quant Only Kernel Benchmark")
    print("=" * 70)
    print(f"GPU:     {torch.cuda.get_device_name()} ({get_gpu_cc()})")
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton:  {triton.__version__}")
    print(f"Output dtype: {dtype_tag}")
    print(f"Model:   {model_name}")
    if args.Lmax:
        print(f"Lmax:    {args.Lmax}")
    print(f"K values: {k_values}")
    tuned_filename = build_tuned_filename("quant_only_tuned", model_name, ext=".py")
    print(f"Kernel:  {tuned_filename}")
    
    # Load tuned module (unified file with both FP8 and INT8)
    tuned_module = load_tuned_module(model_name)
    if tuned_module is None:
        print("\nFalling back to basic kernel for benchmark...")
        # Load basic module first
        basic_module = load_basic_module()
        if basic_module is None:
            return 1
        # Use basic kernel as "tuned"
        if dtype == "fp8":
            tuned_func = basic_module.quant_only_fp8_triton
        else:
            tuned_func = basic_module.quant_only_int8_triton
        get_config_func = basic_module._get_config
    else:
        # Select tuned function based on dtype
        if dtype == "fp8":
            tuned_func = tuned_module.quant_only_fp8_triton
        else:
            tuned_func = tuned_module.quant_only_int8_triton
        get_config_func = tuned_module._get_config
        # Load basic module
        basic_module = load_basic_module()
        if basic_module is None:
            return 1
    
    # Create baseline using tuned config
    baseline_func = make_theoretical_baseline(get_config_func, dtype)
    
    # Select appropriate functions based on dtype
    if dtype == "fp8":
        untuned_func = basic_module.quant_only_fp8_triton
        pytorch_ref_func = basic_module.quant_only_fp8_pytorch
    else:
        untuned_func = basic_module.quant_only_int8_triton
        pytorch_ref_func = basic_module.quant_only_int8_pytorch
    
    # Correctness test
    if not test_correctness(tuned_func, untuned_func, pytorch_ref_func, dtype):
        print("\nERROR: Correctness test failed!")
        return 1
    
    # Throughput benchmark
    run_benchmark(tuned_func, untuned_func, baseline_func, dtype, k_values)
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
