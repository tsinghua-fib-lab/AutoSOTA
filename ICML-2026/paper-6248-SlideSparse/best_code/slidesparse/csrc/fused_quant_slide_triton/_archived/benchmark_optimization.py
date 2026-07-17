#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Quant + Slide Kernel 优化版本对比 Benchmark

比较三个版本的性能：
- Basic (archived): 原始实现，作为 baseline
- Version A: 2D Block Load + Register Slicing (Expert A)
- Version C: Output-Oriented Design (Expert C)

Usage:
    python3 benchmark_optimization.py --dtype fp8 --L 8
    python3 benchmark_optimization.py --dtype fp8 --L 10
    python3 benchmark_optimization.py --dtype int8 --L 8
"""

import argparse
import sys
from pathlib import Path

import torch
import triton

# 设置路径
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# =============================================================================
# 导入各版本实现
# =============================================================================

# Basic (baseline) - 从 archived 导入
from slidesparse.csrc.fused_quant_slide_triton._archived import basic_quant_slide_triton as basic_module

# Version A - 2D Block Load
from slidesparse.csrc.fused_quant_slide_triton import quant_slide_verA as verA_module

# Version C - Output-Oriented
from slidesparse.csrc.fused_quant_slide_triton import quant_slide_verC as verC_module


# =============================================================================
# Test Configuration
# =============================================================================

# M values: 包含非对齐值以测试 padding
M_VALUES = [16, 17, 100, 128, 1024, 4096, 16384]

# K values: 包含非对齐值
K_VALUES = [2560, 2561, 6912]

WARMUP = 25
REP = 100


# =============================================================================
# Correctness Test
# =============================================================================

def test_correctness(dtype: str, L: int) -> bool:
    """
    验证 verA 和 verC 的输出与 basic 一致
    """
    print("\n" + "=" * 70)
    print("Correctness Test")
    print("=" * 70)
    
    torch.manual_seed(42)
    all_passed = True
    
    # 根据 dtype 选择函数
    if dtype == "fp8":
        basic_func = basic_module.quant_slide_fp8_triton
        verA_func = verA_module.quant_slide_fp8_triton
        verC_func = verC_module.quant_slide_fp8_triton
        out_tol = 2.0  # FP8 容差
    else:
        basic_func = basic_module.quant_slide_int8_triton
        verA_func = verA_module.quant_slide_int8_triton
        verC_func = verC_module.quant_slide_int8_triton
        out_tol = 1.0  # INT8 容差
    
    scale_tol = 1e-5
    
    # 选择几个典型形状测试
    test_shapes = [(M, K) for M in M_VALUES[:5] for K in K_VALUES]
    
    for M, K in test_shapes:
        x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        
        # 运行三个版本
        basic_out, basic_scale = basic_func(x, L)
        verA_out, verA_scale = verA_func(x, L)
        verC_out, verC_scale = verC_func(x, L)
        
        # 计算输出维度
        _, K_out, _ = basic_module._compute_output_k(K, L)
        
        # 转换为 float 进行比较
        basic_float = basic_out[:M, :K_out].float()
        verA_float = verA_out[:M, :K_out].float()
        verC_float = verC_out[:M, :K_out].float()
        
        # 计算差异
        diff_A_out = (basic_float - verA_float).abs().max().item()
        diff_C_out = (basic_float - verC_float).abs().max().item()
        diff_A_scale = (basic_scale[:M] - verA_scale[:M]).abs().max().item()
        diff_C_scale = (basic_scale[:M] - verC_scale[:M]).abs().max().item()
        
        passed_A = diff_A_out <= out_tol and diff_A_scale <= scale_tol
        passed_C = diff_C_out <= out_tol and diff_C_scale <= scale_tol
        passed = passed_A and passed_C
        all_passed = all_passed and passed
        
        status = "✓" if passed else "✗"
        print(f"  M={M:<5} K={K:<5} | "
              f"verA: out_diff={diff_A_out:.4f} scale_diff={diff_A_scale:.2e} {'✓' if passed_A else '✗'} | "
              f"verC: out_diff={diff_C_out:.4f} scale_diff={diff_C_scale:.2e} {'✓' if passed_C else '✗'}")
    
    print("-" * 70)
    print(f"Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


# =============================================================================
# Performance Benchmark
# =============================================================================

def run_benchmark(dtype: str, L: int):
    """
    运行性能对比 benchmark
    """
    print("\n" + "=" * 70)
    print("Performance Benchmark")
    print("=" * 70)
    
    dtype_name = dtype.upper()
    num_windows = L // 2 - 1
    expand_ratio = (num_windows * 4) / L
    
    print(f"Input: BF16, Output: {dtype_name}, L={L} (2:{L} sparsity, expand={expand_ratio:.2f}x)")
    print(f"Warmup: {WARMUP}, Rep: {REP}")
    print()
    
    # 根据 dtype 选择函数
    if dtype == "fp8":
        basic_func = basic_module.quant_slide_fp8_triton
        verA_func = verA_module.quant_slide_fp8_triton
        verC_func = verC_module.quant_slide_fp8_triton
    else:
        basic_func = basic_module.quant_slide_int8_triton
        verA_func = verA_module.quant_slide_int8_triton
        verC_func = verC_module.quant_slide_int8_triton
    
    # Header
    print(f"{'M':<7} {'K':<6} {'K_out':<7} | "
          f"{'Basic(us)':<12} {'VerA(us)':<12} {'VerC(us)':<12} | "
          f"{'A/Basic':<10} {'C/Basic':<10}")
    print("-" * 95)
    
    results = []
    
    for M in M_VALUES:
        for K in K_VALUES:
            # 避免 OOM
            if M * K > 128 * 1024 * 1024:
                continue
            
            x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
            
            _, K_out, _ = basic_module._compute_output_k(K, L)
            
            # Benchmark 三个版本
            try:
                time_basic = triton.testing.do_bench(
                    lambda: basic_func(x, L),
                    warmup=WARMUP, rep=REP
                )
            except Exception as e:
                print(f"  M={M:<5} K={K:<5} | Basic FAILED: {e}")
                continue
            
            try:
                time_verA = triton.testing.do_bench(
                    lambda: verA_func(x, L),
                    warmup=WARMUP, rep=REP
                )
            except Exception as e:
                time_verA = float('inf')
                print(f"  Warning: verA failed for M={M}, K={K}: {e}")
            
            try:
                time_verC = triton.testing.do_bench(
                    lambda: verC_func(x, L),
                    warmup=WARMUP, rep=REP
                )
            except Exception as e:
                time_verC = float('inf')
                print(f"  Warning: verC failed for M={M}, K={K}: {e}")
            
            # 计算加速比 (相对于 basic)
            speedup_A = time_basic / time_verA if time_verA != float('inf') else 0.0
            speedup_C = time_basic / time_verC if time_verC != float('inf') else 0.0
            
            results.append({
                'M': M, 'K': K, 'K_out': K_out,
                'time_basic': time_basic,
                'time_verA': time_verA,
                'time_verC': time_verC,
                'speedup_A': speedup_A,
                'speedup_C': speedup_C,
            })
            
            # 格式化输出
            verA_str = f"{time_verA:.2f}" if time_verA != float('inf') else "FAIL"
            verC_str = f"{time_verC:.2f}" if time_verC != float('inf') else "FAIL"
            speedup_A_str = f"{speedup_A:.2f}x" if speedup_A > 0 else "N/A"
            speedup_C_str = f"{speedup_C:.2f}x" if speedup_C > 0 else "N/A"
            
            print(f"{M:<7} {K:<6} {K_out:<7} | "
                  f"{time_basic:<12.2f} {verA_str:<12} {verC_str:<12} | "
                  f"{speedup_A_str:<10} {speedup_C_str:<10}")
    
    # Summary
    print("\n" + "-" * 95)
    print("Summary:")
    
    valid_A = [r for r in results if r['speedup_A'] > 0]
    valid_C = [r for r in results if r['speedup_C'] > 0]
    
    if valid_A:
        avg_A = sum(r['speedup_A'] for r in valid_A) / len(valid_A)
        max_A = max(r['speedup_A'] for r in valid_A)
        min_A = min(r['speedup_A'] for r in valid_A)
        best_A = max(valid_A, key=lambda r: r['speedup_A'])
        print(f"  Version A: Avg {avg_A:.2f}x  Min {min_A:.2f}x  Max {max_A:.2f}x")
        print(f"             Best at M={best_A['M']}, K={best_A['K']}")
    else:
        print(f"  Version A: No valid results")
    
    if valid_C:
        avg_C = sum(r['speedup_C'] for r in valid_C) / len(valid_C)
        max_C = max(r['speedup_C'] for r in valid_C)
        min_C = min(r['speedup_C'] for r in valid_C)
        best_C = max(valid_C, key=lambda r: r['speedup_C'])
        print(f"  Version C: Avg {avg_C:.2f}x  Min {min_C:.2f}x  Max {max_C:.2f}x")
        print(f"             Best at M={best_C['M']}, K={best_C['K']}")
    else:
        print(f"  Version C: No valid results")
    
    # 分 M 范围分析
    print("\n" + "-" * 95)
    print("Analysis by M range:")
    
    ranges = [
        ("Small (M <= 128)", lambda r: r['M'] <= 128),
        ("Medium (128 < M <= 4096)", lambda r: 128 < r['M'] <= 4096),
        ("Large (M > 4096)", lambda r: r['M'] > 4096),
    ]
    
    for name, filter_fn in ranges:
        filtered = [r for r in results if filter_fn(r)]
        if filtered:
            avg_A = sum(r['speedup_A'] for r in filtered if r['speedup_A'] > 0) / max(1, len([r for r in filtered if r['speedup_A'] > 0]))
            avg_C = sum(r['speedup_C'] for r in filtered if r['speedup_C'] > 0) / max(1, len([r for r in filtered if r['speedup_C'] > 0]))
            print(f"  {name}: VerA avg {avg_A:.2f}x, VerC avg {avg_C:.2f}x")
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Quant+Slide Kernel Optimization Benchmark")
    parser.add_argument('--dtype', type=str, default='fp8', choices=['fp8', 'int8'],
                        help='Output dtype (default: fp8)')
    parser.add_argument('--L', type=int, default=8, choices=[6, 8, 10],
                        help='Group size L (default: 8)')
    parser.add_argument('--skip-correctness', action='store_true',
                        help='Skip correctness test')
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return 1
    
    print("=" * 70)
    print("Quant + Slide Kernel Optimization Benchmark")
    print("=" * 70)
    print(f"GPU:     {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton:  {triton.__version__}")
    print(f"Dtype:   {args.dtype.upper()}")
    print(f"L:       {args.L}")
    
    # Correctness test
    if not args.skip_correctness:
        if not test_correctness(args.dtype, args.L):
            print("\nERROR: Correctness test failed!")
            return 1
    
    # Performance benchmark
    run_benchmark(args.dtype, args.L)
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
