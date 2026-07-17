#!/usr/bin/env python3
"""
SlideSparse 性能对比测试

比较:
  - vLLM INT8 量化 (baseline)
  - Triton INT8 量化 (triton_quant)
  - Fused Quant + Slide (fused_quant_slide)

用法:
  python3 run_benchmark.py                         # 测试 L=6,8 int8/fp8
  python3 run_benchmark.py --L 8 --dtype int8      # 只测 L=8 int8
  python3 run_benchmark.py --warmup 50 --rep 200   # 调整 benchmark 参数
  python3 run_benchmark.py --tuned                 # 使用 autotuned 版本
"""

import torch
import triton.testing as testing
import argparse
import sys
import os
from pathlib import Path
import importlib.util

# ============================================================================
#                           加载 Kernel
# ============================================================================

def load_slide_kernel(L: int, dtype: str, tuned: bool = False):
    """动态加载 slide kernel"""
    kernels_dir = Path(__file__).parent
    if tuned:
        module_name = f"slide_L{L}_{dtype}_autotuned"
    else:
        module_name = f"slide_L{L}_{dtype}"
    module_path = kernels_dir / f"{module_name}.py"
    
    if not module_path.exists():
        return None
    
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fused_quant_slide


def load_triton_quant(dtype: str = "int8"):
    """加载 triton_quant (根据 dtype 选择 int8 或 fp8)"""
    kernels_dir = Path(__file__).parent
    
    if dtype == "fp8":
        module_path = kernels_dir / "triton_quant_fp8.py"
        func_name = "triton_quant_fp8"
    else:
        module_path = kernels_dir / "triton_quant_int8.py"
        func_name = "triton_quant_int8"
    
    if not module_path.exists():
        return None
    
    spec = importlib.util.spec_from_file_location(func_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func_name)


# vLLM
try:
    from vllm._custom_ops import scaled_int8_quant as vllm_scaled_int8_quant
    VLLM_INT8_AVAILABLE = True
except ImportError:
    VLLM_INT8_AVAILABLE = False

try:
    from vllm._custom_ops import scaled_fp8_quant as vllm_scaled_fp8_quant
    VLLM_FP8_AVAILABLE = True
except ImportError:
    VLLM_FP8_AVAILABLE = False

VLLM_AVAILABLE = VLLM_INT8_AVAILABLE or VLLM_FP8_AVAILABLE


def vllm_quant_int8(x: torch.Tensor):
    """vLLM INT8 量化"""
    return vllm_scaled_int8_quant(x, scale=None, symmetric=True)


def vllm_quant_fp8(x: torch.Tensor):
    """vLLM FP8 量化 - 使用 per-token 模式以公平对比"""
    return vllm_scaled_fp8_quant(x, scale=None, use_per_token_if_dynamic=True)


# ============================================================================
#                           Benchmark
# ============================================================================

def run_benchmark(L: int, dtype: str, warmup: int, rep: int, tuned: bool = False):
    """运行单个配置的 benchmark"""
    tuned_str = " (autotuned)" if tuned else ""
    print(f"\n{'='*100}")
    print(f"Benchmark: L={L}, dtype={dtype}{tuned_str}")
    print(f"{'='*100}")
    
    # 加载 kernel
    fused_quant_slide = load_slide_kernel(L, dtype, tuned=tuned)
    if fused_quant_slide is None:
        kernel_name = f"slide_L{L}_{dtype}_autotuned.py" if tuned else f"slide_L{L}_{dtype}.py"
        print(f"  ⚠ Kernel 不存在: {kernel_name}")
        return None
    
    triton_quant = load_triton_quant(dtype)  # 根据 dtype 加载对应的 triton quant
    
    # 根据 dtype 选择 vLLM baseline
    if dtype == "fp8":
        vllm_quant = vllm_quant_fp8 if VLLM_FP8_AVAILABLE else None
        vllm_available = VLLM_FP8_AVAILABLE
    else:
        vllm_quant = vllm_quant_int8 if VLLM_INT8_AVAILABLE else None
        vllm_available = VLLM_INT8_AVAILABLE
    
    # 计算扩展比例
    N = L // 2
    NUM_WINDOWS = N - 1
    expand_ratio = (NUM_WINDOWS * 4) / L
    
    print(f"稀疏格式: 2:{L} (N={N}, windows={NUM_WINDOWS}, expand_ratio={expand_ratio:.3f}x)")
    
    # 测试配置
    test_configs = [
        (1, 2560, "Wqkv"),
        (4, 2560, "Wqkv"),
        (16, 2560, "Wqkv"),
        (64, 2560, "Wqkv"),
        (128, 2560, "Wqkv"),
        (256, 2560, "Wqkv"),
        (512, 2560, "Wqkv"),
        (1024, 2560, "Wqkv"),
        (2048, 2560, "Wqkv"),
        (4096, 2560, "Wqkv"),
        (1, 6912, "W2"),
        (16, 6912, "W2"),
        (64, 6912, "W2"),
        (128, 6912, "W2"),
        (512, 6912, "W2"),
        (1024, 6912, "W2"),
        (4096, 6912, "W2"),
    ]
    
    header_parts = [f"{'Layer':<6}", f"{'M':>5}", f"{'K':>5}"]
    if vllm_available:
        header_parts.append(f"{'vLLM':>10}")
    if triton_quant is not None:
        header_parts.append(f"{'Tr.Quant':>10}")
    header_parts.extend([f"{'Tr.Slide':>10}", f"{'Slide/vLLM':>12}", f"{'Slide/Quant':>13}"])
    
    print(f"\n{' | '.join(header_parts)}")
    print("-" * 100)
    
    results = []
    
    for M, K, layer in test_configs:
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        
        # vLLM baseline (根据 dtype 使用对应的量化函数)
        if vllm_available and vllm_quant is not None:
            t_vllm = testing.do_bench(
                lambda: vllm_quant(x), warmup=warmup, rep=rep, return_mode="min"
            )
        else:
            t_vllm = float('nan')
        
        # Triton Quant baseline
        if triton_quant is not None:
            t_quant = testing.do_bench(
                lambda: triton_quant(x), warmup=warmup, rep=rep, return_mode="min"
            )
        else:
            t_quant = float('nan')
        
        # Fused Quant + Slide
        t_slide = testing.do_bench(
            lambda: fused_quant_slide(x), warmup=warmup, rep=rep, return_mode="min"
        )
        
        # 计算比率
        ratio_slide_vllm = t_slide / t_vllm if vllm_available else float('nan')
        ratio_slide_quant = t_slide / t_quant if triton_quant is not None else float('nan')
        
        # 格式化输出
        parts = [f"{layer:<6}", f"{M:>5}", f"{K:>5}"]
        if vllm_available:
            parts.append(f"{t_vllm*1000:>9.2f}µ")
        if triton_quant is not None:
            parts.append(f"{t_quant*1000:>9.2f}µ")
        parts.extend([
            f"{t_slide*1000:>9.2f}µ",
            f"{ratio_slide_vllm:>11.2f}x" if vllm_available else f"{'N/A':>11}",
            f"{ratio_slide_quant:>12.2f}x" if triton_quant is not None else f"{'N/A':>12}",
        ])
        print(" | ".join(parts))
        
        results.append({
            'M': M, 'K': K, 'layer': layer,
            't_vllm': t_vllm, 't_quant': t_quant, 't_slide': t_slide,
            'ratio_slide_vllm': ratio_slide_vllm, 'ratio_slide_quant': ratio_slide_quant,
        })
    
    # 统计
    print("-" * 100)
    
    valid_vllm = [r for r in results if not (r['t_vllm'] != r['t_vllm'])]  # not nan
    valid_quant = [r for r in results if not (r['t_quant'] != r['t_quant'])]
    
    if valid_vllm:
        avg_slide_vllm = sum(r['ratio_slide_vllm'] for r in valid_vllm) / len(valid_vllm)
        print(f"\n平均 Slide / vLLM = {avg_slide_vllm:.2f}x")
    
    if valid_quant:
        avg_slide_quant = sum(r['ratio_slide_quant'] for r in valid_quant) / len(valid_quant)
        print(f"平均 Slide / Triton Quant = {avg_slide_quant:.2f}x")
        print(f"  (Slide 做量化 + {expand_ratio:.2f}x 数据扩展，正常应该稍慢)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="SlideSparse Benchmark")
    parser.add_argument('--L', type=int, default=None, choices=[6, 8],
                        help='只测试指定 L 值')
    parser.add_argument('--dtype', type=str, default=None, choices=['int8', 'fp8'],
                        help='只测试指定 dtype')
    parser.add_argument('--warmup', type=int, default=25, help='Warmup iterations')
    parser.add_argument('--rep', type=int, default=100, help='Benchmark repetitions')
    parser.add_argument('--tuned', action='store_true',
                        help='使用 autotuned 版本的 kernel (slide_L*_*_autotuned.py)')
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("CUDA 不可用")
        return 1
    
    tuned_str = " [AUTOTUNED]" if args.tuned else ""
    print("=" * 100)
    print(f"SlideSparse Performance Benchmark{tuned_str}")
    print("=" * 100)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"vLLM INT8: {'✓ available' if VLLM_INT8_AVAILABLE else '✗ not available'}")
    print(f"vLLM FP8:  {'✓ available' if VLLM_FP8_AVAILABLE else '✗ not available'}")
    print(f"Warmup: {args.warmup}, Rep: {args.rep}")
    if args.tuned:
        print(f"Mode: Using autotuned kernels")
    
    # 确定测试配置
    if args.L is not None and args.dtype is not None:
        configs = [(args.L, args.dtype)]
    elif args.L is not None:
        configs = [(args.L, 'int8'), (args.L, 'fp8')]
    elif args.dtype is not None:
        configs = [(6, args.dtype), (8, args.dtype)]
    else:
        configs = [(6, 'int8'), (6, 'fp8'), (8, 'int8'), (8, 'fp8')]
    
    all_results = {}
    
    for L, dtype in configs:
        results = run_benchmark(L, dtype, args.warmup, args.rep, tuned=args.tuned)
        if results:
            all_results[(L, dtype)] = results
    
    # 总结
    print(f"\n{'='*100}")
    print("Summary")
    print(f"{'='*100}")
    
    if VLLM_AVAILABLE:
        print(f"\n{'Config':<12} | {'Avg Slide/vLLM':>15} | {'Avg Slide/Quant':>15}")
        print("-" * 50)
        for (L, dtype), results in all_results.items():
            valid_vllm = [r for r in results if not (r['t_vllm'] != r['t_vllm'])]
            valid_quant = [r for r in results if not (r['t_quant'] != r['t_quant'])]
            
            avg_vllm = sum(r['ratio_slide_vllm'] for r in valid_vllm) / len(valid_vllm) if valid_vllm else float('nan')
            avg_quant = sum(r['ratio_slide_quant'] for r in valid_quant) / len(valid_quant) if valid_quant else float('nan')
            
            print(f"L={L} {dtype:<5} | {avg_vllm:>14.2f}x | {avg_quant:>14.2f}x")
    
    print(f"\n{'='*100}")
    print("Benchmark 完成!")
    print(f"{'='*100}")
    
    return 0


if __name__ == "__main__":
    exit(main())
