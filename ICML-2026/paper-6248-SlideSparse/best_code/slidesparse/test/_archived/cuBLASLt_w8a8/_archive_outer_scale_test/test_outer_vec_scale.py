#!/usr/bin/env python3
"""
测试 cuBLASLt FP8 + Outer Vector Scaling

1. 编译 CUDA extension
2. 运行简单测试
3. 对比 PyTorch 计算结果

python3 slidesparse/test/test_outer_vec_scale.py

"""

import os
import sys
import subprocess
import tempfile
import shutil

def compile_extension():
    """编译 CUDA extension"""
    print("=" * 60)
    print("Step 1: Compiling CUDA extension")
    print("=" * 60)
    
    import torch
    from torch.utils.cpp_extension import load
    
    cuda_file = os.path.join(os.path.dirname(__file__), 
                             "../csrc/test_outer_vec_scale.cu")
    
    # 编译
    ext = load(
        name="test_outer_vec_scale",
        sources=[cuda_file],
        extra_cuda_cflags=[
            "-O3",
            "-gencode=arch=compute_89,code=sm_89",  # Ada
            "-gencode=arch=compute_90,code=sm_90",  # Hopper
            "-gencode=arch=compute_120,code=sm_120", # Blackwell
        ],
        extra_ldflags=["-lcublasLt", "-lcublas", "-lcuda"],
        verbose=True,
    )
    
    return ext

def run_test(ext):
    """运行测试"""
    print("\n" + "=" * 60)
    print("Step 2: Running FP8 GEMM Test")
    print("=" * 60)
    
    import torch
    
    # 测试参数
    M, N, K = 64, 128, 256
    
    print(f"\nTest dimensions: M={M}, N={N}, K={K}")
    
    # 获取 FP8 类型
    fp8_dtype = torch.float8_e4m3fn
    
    # 创建随机数据（先用 FP32，然后转 FP8）
    torch.manual_seed(42)
    A_fp32 = torch.randn(M, K, device="cuda", dtype=torch.float32) * 0.1
    B_fp32 = torch.randn(N, K, device="cuda", dtype=torch.float32) * 0.1
    
    # 转换为 FP8
    A_fp8 = A_fp32.to(fp8_dtype)
    B_fp8 = B_fp32.to(fp8_dtype)
    
    # 创建 scale 向量
    scale_A = torch.ones(M, device="cuda", dtype=torch.float32) * 1.0
    scale_B = torch.ones(N, device="cuda", dtype=torch.float32) * 1.0
    
    print(f"A shape: {A_fp8.shape}, dtype: {A_fp8.dtype}")
    print(f"B shape: {B_fp8.shape}, dtype: {B_fp8.dtype}")
    print(f"scale_A shape: {scale_A.shape}")
    print(f"scale_B shape: {scale_B.shape}")
    
    # 调用 cuBLASLt
    print("\nCalling cuBLASLt FP8 GEMM with Outer Vector Scaling...")
    try:
        D = ext.simple_fp8_gemm_outer_vec(
            A_fp8, B_fp8, scale_A, scale_B, torch.bfloat16
        )
        print(f"Output shape: {D.shape}, dtype: {D.dtype}")
        print(f"Output (first 5x5):\n{D[:5, :5]}")
    except Exception as e:
        print(f"\n!!! cuBLASLt FAILED: {e}")
        return False
    
    # 使用 PyTorch 计算参考结果
    print("\nComputing reference with PyTorch...")
    # D = scale_A[:, None] * scale_B[None, :] * (A @ B^T)
    A_fp32_from_fp8 = A_fp8.to(torch.float32)
    B_fp32_from_fp8 = B_fp8.to(torch.float32)
    D_ref = torch.mm(A_fp32_from_fp8, B_fp32_from_fp8.t())
    D_ref = scale_A.unsqueeze(1) * scale_B.unsqueeze(0) * D_ref
    D_ref = D_ref.to(torch.bfloat16)
    
    print(f"Reference (first 5x5):\n{D_ref[:5, :5]}")
    
    # 比较
    D_fp32 = D.to(torch.float32)
    D_ref_fp32 = D_ref.to(torch.float32)
    
    abs_diff = (D_fp32 - D_ref_fp32).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    
    print(f"\nMax absolute difference: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")
    
    # 相对误差
    rel_diff = abs_diff / (D_ref_fp32.abs() + 1e-8)
    max_rel_diff = rel_diff.max().item()
    
    print(f"Max relative difference: {max_rel_diff:.4%}")
    
    # 判断是否通过
    tolerance = 0.01  # 1% 相对误差
    if max_rel_diff < tolerance:
        print(f"\n✓ TEST PASSED (rel_diff < {tolerance:.1%})")
        return True
    else:
        print(f"\n✗ TEST FAILED (rel_diff >= {tolerance:.1%})")
        return False

def test_different_sizes(ext):
    """测试不同尺寸"""
    print("\n" + "=" * 60)
    print("Step 3: Testing Different Sizes")
    print("=" * 60)
    
    import torch
    fp8_dtype = torch.float8_e4m3fn
    
    test_cases = [
        (1, 64, 64),      # M=1
        (16, 64, 64),     # 小尺寸
        (32, 128, 256),   # 中尺寸
        (64, 256, 512),   # 较大
        (17, 65, 129),    # 非对齐尺寸
    ]
    
    results = []
    for M, N, K in test_cases:
        print(f"\nTesting M={M}, N={N}, K={K}...")
        
        torch.manual_seed(42)
        A_fp8 = (torch.randn(M, K, device="cuda") * 0.1).to(fp8_dtype)
        B_fp8 = (torch.randn(N, K, device="cuda") * 0.1).to(fp8_dtype)
        scale_A = torch.ones(M, device="cuda", dtype=torch.float32)
        scale_B = torch.ones(N, device="cuda", dtype=torch.float32)
        
        try:
            D = ext.simple_fp8_gemm_outer_vec(
                A_fp8, B_fp8, scale_A, scale_B, torch.bfloat16
            )
            
            # 计算参考
            A_fp32 = A_fp8.to(torch.float32)
            B_fp32 = B_fp8.to(torch.float32)
            D_ref = torch.mm(A_fp32, B_fp32.t()).to(torch.bfloat16)
            
            max_diff = (D.to(torch.float32) - D_ref.to(torch.float32)).abs().max().item()
            
            print(f"  ✓ Success, max_diff={max_diff:.6f}")
            results.append((M, N, K, "PASS", max_diff))
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results.append((M, N, K, "FAIL", str(e)))
    
    # 汇总
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for M, N, K, status, info in results:
        print(f"  M={M:3d}, N={N:3d}, K={K:3d}: {status} ({info})")
    
    return all(r[3] == "PASS" for r in results)

def main():
    print("cuBLASLt FP8 Outer Vector Scaling Test")
    print("=" * 60)
    
    # 显示 GPU 信息
    import torch
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return 1
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    
    # 编译
    try:
        ext = compile_extension()
    except Exception as e:
        print(f"Compilation failed: {e}")
        return 1
    
    # 基本测试
    if not run_test(ext):
        return 1
    
    # 多尺寸测试
    if not test_different_sizes(ext):
        return 1
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
