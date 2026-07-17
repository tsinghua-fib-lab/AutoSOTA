#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Dequant + Bias Kernel Benchmark

Usage:
    python3 run_benchmark.py --dtype bf16   # Test BF16 input
    python3 run_benchmark.py --dtype fp32   # Test FP32 input
    python3 run_benchmark.py --dtype int32  # Test INT32 input
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
    get_unique_n_values,
    load_module,
)


# =============================================================================
# Test Configuration
# =============================================================================

# M values for throughput benchmark (aligned with M-quick autotune)
M_VALUES_BENCH = [16, 128, 1024, 4096, 16384]

# Correctness test uses fixed values (smaller set for quick verification)
M_VALUES_CORRECTNESS = [16, 128, 1024, 4096]
N_VALUES_CORRECTNESS = [2560, 3840, 13824]

WARMUP = 25
REP = 100


# =============================================================================
# Theoretical Baseline (pure memory copy with tuned config)
# =============================================================================

@triton.jit
def _memcpy_kernel(
    in_ptr, out_ptr, M, N,
    stride_im, stride_in, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    in_offs = offs_m[:, None] * stride_im + offs_n[None, :] * stride_in
    val = tl.load(in_ptr + in_offs, mask=mask, other=0.0)
    
    out_offs = offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptr + out_offs, val.to(tl.bfloat16), mask=mask)


def make_theoretical_baseline(get_config_func):
    """Create theoretical baseline function that uses tuned config"""
    
    def theoretical_baseline(input_tensor: torch.Tensor) -> torch.Tensor:
        """Pure memory copy baseline using tuned BLOCK_M, BLOCK_N, num_warps, num_stages"""
        M, N = input_tensor.shape
        output = torch.empty((M, N), dtype=torch.bfloat16, device=input_tensor.device)
        
        # Use the same config as tuned kernel
        BLOCK_M, BLOCK_N, num_warps, num_stages = get_config_func(M, N)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        
        _memcpy_kernel[grid](
            input_tensor, output, M, N,
            input_tensor.stride(0), input_tensor.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=num_warps, num_stages=num_stages,
        )
        return output
    
    return theoretical_baseline


# =============================================================================
# Load Kernels
# =============================================================================

def load_tuned_module(model_name: str | None = None) -> ModuleType | None:
    """Load the auto-tuned kernel module from build/"""
    hw_dir_name = build_hw_dir_name()
    build_dir = Path(__file__).parent / "build" / hw_dir_name
    
    # Try model-specific tuned kernel first
    filename = build_tuned_filename("dequant_bias_tuned", model_name, ext=".py")
    module_path = build_dir / filename
    
    if module_path.exists():
        # Use importlib to load the module directly
        spec = importlib.util.spec_from_file_location("dequant_bias_tuned", module_path)
        if spec is None or spec.loader is None:
            print(f"ERROR: Failed to load module spec: {module_path}")
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    # Try generic (non-model-specific) tuned kernel
    generic_filename = build_filename("dequant_bias_tuned", ext=".py")
    generic_path = build_dir / generic_filename
    
    if generic_path.exists():
        print(f"NOTE: Using generic tuned kernel (no model-specific kernel for {model_name})")
        spec = importlib.util.spec_from_file_location("dequant_bias_tuned", generic_path)
        if spec is None or spec.loader is None:
            print(f"ERROR: Failed to load module spec: {generic_path}")
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    print(f"ERROR: Tuned kernel not found in: {build_dir}")
    print(f"  Tried: {filename}")
    print(f"  Please run: python3 autotune_autogen_dequant_bias.py --model {model_name}")
    return None


def load_basic_module() -> ModuleType | None:
    """Load the basic kernel module (contains untuned kernel and pytorch reference)"""
    module_path = Path(__file__).parent / "basic_dequant_bias_triton.py"
    
    if not module_path.exists():
        print(f"ERROR: Basic kernel not found: {module_path}")
        return None
    
    # 使用 importlib 加载本地文件
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

def test_correctness(tuned_func, untuned_func, pytorch_ref_func, input_dtype: torch.dtype):
    """Test correctness against PyTorch reference"""
    print("\n" + "=" * 70)
    print("Correctness Test")
    print("=" * 70)
    
    torch.manual_seed(42)
    all_passed = True
    
    # Sample shapes to avoid OOM (skip large M * N combinations)
    test_shapes = [(M, N) for M in M_VALUES_CORRECTNESS for N in N_VALUES_CORRECTNESS if M * N <= 64 * 1024 * 1024]
    
    for M, N in test_shapes:
        # Generate test data based on input dtype
        if input_dtype == torch.int32:
            gemm = torch.randint(-1000, 1000, (M, N), dtype=torch.int32, device='cuda')
        else:
            gemm = torch.randn(M, N, dtype=input_dtype, device='cuda')
        scale_a = torch.rand(M, dtype=torch.float32, device='cuda') * 0.1 + 0.01
        scale_b = torch.rand(N, dtype=torch.float32, device='cuda') * 0.1 + 0.01
        bias = torch.randn(N, dtype=torch.bfloat16, device='cuda')
        
        ref = pytorch_ref_func(gemm, scale_a, scale_b, bias, torch.bfloat16)
        out_tuned = tuned_func(gemm, scale_a, scale_b, bias)
        out_untuned = untuned_func(gemm, scale_a, scale_b, bias)
        
        diff_tuned = (ref.float() - out_tuned.float()).abs().max().item()
        diff_untuned = (ref.float() - out_untuned.float()).abs().max().item()
        
        # INT32 输入 dequant 后值范围大 (~10), BF16 精度限制导致误差较大 (~0.03-0.06)
        # BF16/FP32 输入值范围小 (~1), 误差小 (~0.001-0.01)
        tolerance = 0.1 if input_dtype == torch.int32 else 1e-2
        passed = diff_tuned < tolerance and diff_untuned < tolerance
        all_passed = all_passed and passed
        
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] M={M:<6} N={N:<6} tuned_diff={diff_tuned:.6f} untuned_diff={diff_untuned:.6f}")
        
        # Free memory
        del gemm, scale_a, scale_b, bias, ref, out_tuned, out_untuned
        torch.cuda.empty_cache()
    
    print("-" * 70)
    print(f"Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


# =============================================================================
# Throughput Benchmark
# =============================================================================

def run_benchmark(tuned_func, untuned_func, baseline_func, input_dtype: torch.dtype, n_values: list[int]):
    """Run throughput benchmark: theoretical vs untuned vs tuned"""
    print("\n" + "=" * 70)
    print("Throughput Benchmark")
    print("=" * 70)
    dtype_name = "FP32" if input_dtype == torch.float32 else "BF16"
    print(f"Input: {dtype_name}, Output: BF16, Warmup: {WARMUP}, Rep: {REP}")
    print(f"Baseline: memcpy kernel using same tuned (BLOCK_M, BLOCK_N, warps, stages)")
    print()
    
    # Header
    print(f"{'M':<7} {'N':<7} | {'Baseline(us)':<12} {'Untuned(us)':<12} {'Tuned(us)':<11} | {'vs Base':<10} {'vs Untuned':<10}")
    print("-" * 90)
    
    results: list[dict] = []
    
    for M in M_VALUES_BENCH:
        for N in n_values:
            # Generate data based on input dtype
            # INT32: use randint to simulate cuBLASLt INT8 GEMM output
            if input_dtype == torch.int32:
                gemm = torch.randint(-1000, 1000, (M, N), dtype=torch.int32, device='cuda')
            else:
                gemm = torch.randn(M, N, dtype=input_dtype, device='cuda')
            scale_a = torch.rand(M, dtype=torch.float32, device='cuda') * 0.1 + 0.01
            scale_b = torch.rand(N, dtype=torch.float32, device='cuda') * 0.1 + 0.01
            bias = torch.randn(N, dtype=torch.bfloat16, device='cuda')
            
            # Benchmark baseline (pure memory copy with tuned config)
            t_baseline: float = testing.do_bench(
                lambda: baseline_func(gemm),
                warmup=WARMUP, rep=REP, return_mode="min"
            ) * 1000  # type: ignore  # ms -> us
            
            # Benchmark untuned kernel
            t_untuned: float = testing.do_bench(
                lambda: untuned_func(gemm, scale_a, scale_b, bias),
                warmup=WARMUP, rep=REP, return_mode="min"
            ) * 1000  # type: ignore  # ms -> us
            
            # Benchmark tuned kernel
            t_tuned: float = testing.do_bench(
                lambda: tuned_func(gemm, scale_a, scale_b, bias),
                warmup=WARMUP, rep=REP, return_mode="min"
            ) * 1000  # type: ignore  # ms -> us
            
            # Speedup ratios
            speedup_vs_baseline = t_baseline / t_tuned  # <1 expected (tuned slower than memcpy)
            speedup_vs_untuned = t_untuned / t_tuned  # >1 means tuned is faster
            
            results.append({
                'M': M, 'N': N,
                'baseline': t_baseline,
                'untuned': t_untuned,
                'tuned': t_tuned,
                'speedup_baseline': speedup_vs_baseline,
                'speedup_untuned': speedup_vs_untuned,
            })
            
            print(f"{M:<7} {N:<7} | {t_baseline:<12.2f} {t_untuned:<12.2f} {t_tuned:<11.2f} | {speedup_vs_baseline:<10.2f} {speedup_vs_untuned:<10.2f}")
            
            # Free memory
            del gemm, scale_a, scale_b, bias
            torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "-" * 90)
    print("Summary:")
    avg_baseline = sum(r['speedup_baseline'] for r in results) / len(results)
    avg_untuned = sum(r['speedup_untuned'] for r in results) / len(results)
    max_untuned = max(r['speedup_untuned'] for r in results)
    min_untuned = min(r['speedup_untuned'] for r in results)
    print(f"  vs Baseline: Avg {avg_baseline:.2f}x  (expected <1, tuned does compute, baseline only memcpy)")
    print(f"  vs Untuned:  Avg {avg_untuned:.2f}x  Min {min_untuned:.2f}x  Max {max_untuned:.2f}x")
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Dequant + Bias Kernel Benchmark")
    parser.add_argument('--dtype', type=str, required=True, choices=['bf16', 'fp32', 'int32'],
                        help='Input dtype: bf16, fp32 or int32')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name (e.g., Qwen2.5-0.5B-INT8). If not specified, uses BitNet-2B-BF16 default.')
    parser.add_argument('--Lmax', type=int, default=10,
                        help='Max L value for nk_list (default: 10)')
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return 1
    
    dtype_map = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'int32': torch.int32}
    input_dtype = dtype_map[args.dtype]
    
    # Get N values and model_name from unified tool
    nk_list, model_name = get_nk_list_for_search(args.model, args.Lmax)
    n_values = get_unique_n_values(nk_list)
    
    print("=" * 70)
    print("Dequant + Bias Kernel Benchmark")
    print("=" * 70)
    print(f"GPU:     {torch.cuda.get_device_name()} ({get_gpu_cc()})")
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton:  {triton.__version__}")
    print(f"Input:   {args.dtype.upper()}")
    print(f"Model:   {model_name}")
    print(f"N values: {n_values}")
    print(f"Kernel:  {build_tuned_filename('dequant_bias_tuned', model_name, ext='.py')}")
    
    # Load tuned module (supports BF16, FP32 and INT32 input)
    tuned_module = load_tuned_module(model_name)
    if tuned_module is None:
        return 1
    
    tuned_func = tuned_module.dequant_bias_triton
    get_config_func = tuned_module._get_config
    
    # Create baseline using tuned config
    baseline_func = make_theoretical_baseline(get_config_func)
    
    # Load basic module
    basic_module = load_basic_module()
    if basic_module is None:
        return 1
    
    untuned_func = basic_module.dequant_bias_triton
    pytorch_ref_func = basic_module.dequant_bias_pytorch
    
    # Correctness test
    if not test_correctness(tuned_func, untuned_func, pytorch_ref_func, input_dtype):
        print("\nERROR: Correctness test failed!")
        return 1
    
    # Throughput benchmark
    run_benchmark(tuned_func, untuned_func, baseline_func, input_dtype, n_values)
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
