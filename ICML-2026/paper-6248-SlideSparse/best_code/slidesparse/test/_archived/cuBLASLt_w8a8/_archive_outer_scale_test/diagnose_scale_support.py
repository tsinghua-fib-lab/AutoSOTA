#!/usr/bin/env python3
"""
诊断 SM120 (Blackwell/RTX 5080) 上的 cuBLASLt FP8 Scale 支持情况

运行: python3 slidesparse/cuBLASLt_test/outer_scale_test/diagnose_scale_support.py
"""

import os
import torch
from torch.utils.cpp_extension import load

def main():
    print("=" * 70)
    print("cuBLASLt FP8 Scale Mode Diagnostic")
    print("=" * 70)
    
    # 编译
    cuda_file = os.path.join(os.path.dirname(__file__), "diagnose_scale_support.cu")
    
    print(f"\nCompiling {cuda_file}...")
    ext = load(
        name="diagnose_scale_support",
        sources=[cuda_file],
        extra_cuda_cflags=[
            "-O3",
            "-gencode=arch=compute_89,code=sm_89",
            "-gencode=arch=compute_90,code=sm_90", 
            "-gencode=arch=compute_120,code=sm_120",
        ],
        extra_ldflags=["-lcublasLt", "-lcublas"],
        verbose=False,
    )
    
    print(f"\nGPU: {ext.get_gpu_info()}")
    print("\n" + "-" * 70)
    print("Running diagnostics...")
    print("-" * 70)
    
    results = ext.run_diagnostics()
    
    # 打印结果表格
    print(f"\n{'Test Case':<45} {'Result':<25}")
    print("=" * 70)
    
    for name, result in results:
        if result == "":  # GPU info line
            print(f"\n{name}")
            continue
        status = "✓ OK" if result == "OK" else f"✗ {result}"
        print(f"{name:<45} {status:<25}")
    
    print("\n" + "=" * 70)
    print("Analysis:")
    print("=" * 70)
    
    # 统计
    ok_count = sum(1 for _, r in results if r == "OK")
    fail_count = sum(1 for _, r in results if r.startswith("FAIL"))
    
    print(f"  Passed: {ok_count}")
    print(f"  Failed: {fail_count}")
    
    # 分析 outer_vec 结果
    outer_vec_results = [(n, r) for n, r in results if "outer_vec" in n]
    tensorwide_results = [(n, r) for n, r in results if "tensorwide" in n]
    no_scale_results = [(n, r) for n, r in results if "no scale" in n]
    
    print("\n  Per-category:")
    print(f"    No scale:    {sum(1 for _, r in no_scale_results if r == 'OK')}/{len(no_scale_results)} passed")
    print(f"    Tensorwide:  {sum(1 for _, r in tensorwide_results if r == 'OK')}/{len(tensorwide_results)} passed")
    print(f"    Outer Vec:   {sum(1 for _, r in outer_vec_results if r == 'OK')}/{len(outer_vec_results)} passed")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
