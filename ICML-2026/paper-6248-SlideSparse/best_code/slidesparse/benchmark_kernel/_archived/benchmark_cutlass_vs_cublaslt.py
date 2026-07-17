#!/usr/bin/env python3
"""
CUTLASS vs cuBLASLt FP8 GEMM Benchmark

对比 vLLM 的 CUTLASS 和 cuBLASLt 在 FP8 GEMM 上的性能。
固定 Layout: TN+CC+Col，FP8E4M3 输入，FP32 计算，BF16 输出。

Usage:
    python benchmark_cutlass_vs_cublaslt.py
    python benchmark_cutlass_vs_cublaslt.py --compile  # 强制重新编译
"""

import os
import sys
import time
import csv
import ctypes
import ctypes.util
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.cpp_extension import load

# 项目根目录
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# 导入统一的库加载函数
sys.path.insert(0, str(PROJECT_ROOT))
from slidesparse.utils import ensure_cublaslt_loaded


# ============== CUDA 扩展加载 ==============
def load_cublaslt_extension(force_compile: bool = False):
    """加载 cuBLASLt benchmark CUDA 扩展"""
    ensure_cublaslt_loaded()
    
    src_path = SCRIPT_DIR / "cublaslt_gemm_benchmark.cu"
    build_dir = SCRIPT_DIR / "build"
    build_dir.mkdir(exist_ok=True)
    
    # 获取 GPU 信息用于命名
    prop = torch.cuda.get_device_properties(0)
    ext_name = f"cublaslt_benchmark_cc{prop.major}{prop.minor}"
    
    # 检查是否需要编译
    so_files = list(build_dir.glob(f"{ext_name}*.so"))
    need_compile = force_compile or not so_files
    
    if not need_compile and so_files:
        src_mtime = src_path.stat().st_mtime
        if any(f.stat().st_mtime < src_mtime for f in so_files):
            need_compile = True
    
    if need_compile:
        print(f"[INFO] Compiling cuBLASLt extension for cc{prop.major}{prop.minor}...")
    
    ext = load(
        name=ext_name,
        sources=[str(src_path)],
        build_directory=str(build_dir),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_ldflags=["-lcublasLt", "-lcublas"],
        verbose=need_compile,
    )
    
    # 清理中间文件，只保留 .so
    if need_compile:
        for f in build_dir.iterdir():
            if f.is_file() and not f.suffix == ".so":
                f.unlink()
    
    return ext


# ============== CUTLASS Benchmark ==============
def benchmark_cutlass_fp8(m: int, n: int, k: int, warmup: int = 25, iterations: int = 100) -> Dict:
    """使用 vLLM 的 cutlass_scaled_mm 进行 benchmark"""
    from vllm._custom_ops import cutlass_scaled_mm
    
    device = torch.device('cuda')
    
    # CUTLASS 期望:
    # A: (M, K) row-major, contiguous
    # B: (K, N) column-major (stride(0)==1)
    a_fp8 = torch.randn(m, k, dtype=torch.float16, device=device).to(torch.float8_e4m3fn)
    b_fp8 = torch.randn(n, k, dtype=torch.float16, device=device).to(torch.float8_e4m3fn).t()  # (K,N) col-major
    
    scale_a = torch.ones(1, dtype=torch.float32, device=device)
    scale_b = torch.ones(1, dtype=torch.float32, device=device)
    
    # Warmup
    for _ in range(warmup):
        cutlass_scaled_mm(a_fp8, b_fp8, scale_a, scale_b, torch.bfloat16)
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        cutlass_scaled_mm(a_fp8, b_fp8, scale_a, scale_b, torch.bfloat16)
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - start) * 1000
    
    lat_us = (total_ms * 1000) / iterations
    flops = 2 * m * n * k
    tflops = flops / (lat_us * 1e6)
    
    return {"lat_us": lat_us, "tflops": tflops}


# ============== Benchmark 配置 ==============
# M: batch size (外层循环)
# (N, K, name): 权重矩阵维度 (内层循环)

M_LIST = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

# 基于 LLM 模型的典型 NK 配置
NK_CONFIGS = [
    # Qwen2.5-0.5B: hidden=896, intermediate=4864
    (4864, 896, "Qwen0.5B_W13"),    # gate/up proj
    (896, 4864, "Qwen0.5B_W2"),     # down proj
    
    # Llama3.2-1B: hidden=2048, intermediate=8192  
    (8192, 2048, "Llama1B_W13"),    # gate/up proj
    (2048, 8192, "Llama1B_W2"),     # down proj
]


def align_to_16(x: int) -> int:
    """将维度对齐到 16 的倍数"""
    return ((x + 15) // 16) * 16


# ============== Main ==============
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="Force recompile CUDA extension")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--max-algos", type=int, default=5, help="Max cuBLASLt algorithms to test")
    args = parser.parse_args()
    
    print("=" * 80)
    print("CUTLASS vs cuBLASLt FP8 GEMM Benchmark")
    print("=" * 80)
    
    # GPU 信息
    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    prop = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu_name}")
    print(f"Compute Capability: {prop.major}.{prop.minor}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Warmup: {args.warmup}, Repeat: {args.repeat}")
    print()
    
    # 加载 cuBLASLt 扩展
    print("[INFO] Loading cuBLASLt extension...")
    ext = load_cublaslt_extension(force_compile=args.compile)
    print("[INFO] Extension loaded successfully")
    print()
    
    # 结果存储
    results = []
    
    # 表头
    header = f"{'Config':<20} {'M':>6} {'N':>6} {'K':>6} {'CUTLASS':>12} {'cuBLASLt':>12} {'Speedup':>8}"
    print("-" * 90)
    print(header)
    print("-" * 90)
    
    # 遍历 NK 配置 (内层循环)
    for N_orig, K_orig, config_name in NK_CONFIGS:
        # 对齐到 16
        N = align_to_16(N_orig)
        K = align_to_16(K_orig)
        
        # 遍历 M (外层循环)
        for M in M_LIST:
            M_aligned = align_to_16(M)
            
            try:
                # CUTLASS benchmark
                cutlass_result = benchmark_cutlass_fp8(M_aligned, N, K, args.warmup, args.repeat)
                cutlass_tflops = cutlass_result["tflops"]
                
                # cuBLASLt benchmark
                cublaslt_result = ext.benchmark_fp8_gemm(M_aligned, N, K, args.warmup, args.repeat, args.max_algos)
                cublaslt_tflops = cublaslt_result["best_tflops"]
                
                # 加速比 (cuBLASLt / CUTLASS, >1 表示 cuBLASLt 更快)
                speedup = cublaslt_tflops / cutlass_tflops if cutlass_tflops > 0 else 0
                
                # 记录结果 (保留2位小数)
                results.append({
                    "config": config_name,
                    "M": M_aligned,
                    "N": N,
                    "K": K,
                    "cutlass_tflops": round(cutlass_tflops, 2),
                    "cublaslt_tflops": round(cublaslt_tflops, 2),
                    "speedup": round(speedup, 2),
                })
                
                # 打印
                speedup_str = f"{speedup:.2f}x"
                if speedup > 1.05:
                    speedup_str = f"\033[32m{speedup_str}\033[0m"  # 绿色
                elif speedup < 0.95:
                    speedup_str = f"\033[31m{speedup_str}\033[0m"  # 红色
                
                print(f"{config_name:<20} {M_aligned:>6} {N:>6} {K:>6} "
                      f"{cutlass_tflops:>10.2f}T {cublaslt_tflops:>10.2f}T {speedup_str:>8}")
                
            except Exception as e:
                print(f"{config_name:<20} {M_aligned:>6} {N:>6} {K:>6} ERROR: {e}")
    
    print("-" * 90)
    
    # 统计汇总
    if results:
        print("\n" + "=" * 80)
        print("Summary Statistics")
        print("=" * 80)
        
        speedups = [r["speedup"] for r in results]
        avg_speedup = sum(speedups) / len(speedups)
        max_speedup = max(speedups)
        min_speedup = min(speedups)
        
        faster_count = sum(1 for s in speedups if s > 1.0)
        
        print(f"Average Speedup (cuBLASLt/CUTLASS): {avg_speedup:.2f}x")
        print(f"Max Speedup: {max_speedup:.2f}x")
        print(f"Min Speedup: {min_speedup:.2f}x")
        print(f"cuBLASLt faster in {faster_count}/{len(results)} cases ({100*faster_count/len(results):.1f}%)")
        
        # 保存 CSV
        csv_path = SCRIPT_DIR / f"benchmark_results_cc{prop.major}{prop.minor}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {csv_path}")
    
    print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    main()
