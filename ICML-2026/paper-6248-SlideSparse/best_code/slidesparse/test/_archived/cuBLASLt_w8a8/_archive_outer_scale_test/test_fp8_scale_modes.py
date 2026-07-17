#!/usr/bin/env python3
"""
对比测试 cuBLASLt FP8 的两种 Scale 模式:

1. Tensorwide Scaling (标量 scale) - 应该在所有 FP8 GPU 上工作
2. Outer Vector Scaling (向量 scale) - 仅在 Hopper (SM 9.0) 上工作

在 RTX 5080 (Blackwell SM 12.0) 上:
- Tensorwide 应该工作
- Outer Vector 预期不工作

python3 slidesparse/test/test_fp8_scale_modes.py

"""

import os
import sys
import torch
from torch.utils.cpp_extension import load

def compile_extension():
    """编译 CUDA extension"""
    print("=" * 60)
    print("Compiling CUDA extension")
    print("=" * 60)
    
    cuda_file = os.path.join(os.path.dirname(__file__), 
                             "../csrc/test_fp8_scale_modes.cu")
    
    ext = load(
        name="test_fp8_scale_modes",
        sources=[cuda_file],
        extra_cuda_cflags=[
            "-O3",
            "-gencode=arch=compute_89,code=sm_89",
            "-gencode=arch=compute_90,code=sm_90",
            "-gencode=arch=compute_120,code=sm_120",
        ],
        extra_ldflags=["-lcublasLt", "-lcublas", "-lcuda"],
        verbose=True,
    )
    
    return ext

def test_tensorwide(ext):
    """测试 Tensorwide Scaling (标量)"""
    print("\n" + "=" * 60)
    print("Test 1: Tensorwide Scaling (scalar scale)")
    print("=" * 60)
    
    M, N, K = 64, 128, 256
    fp8_dtype = torch.float8_e4m3fn
    
    torch.manual_seed(42)
    A_fp8 = (torch.randn(M, K, device="cuda") * 0.1).to(fp8_dtype)
    B_fp8 = (torch.randn(N, K, device="cuda") * 0.1).to(fp8_dtype)
    
    scale_a = 1.0
    scale_b = 1.0
    
    print(f"A: {A_fp8.shape}, B: {B_fp8.shape}")
    print(f"scale_a: {scale_a}, scale_b: {scale_b}")
    
    try:
        D = ext.fp8_gemm_tensorwide(A_fp8, B_fp8, scale_a, scale_b, torch.bfloat16)
        print(f"Output: {D.shape}, dtype: {D.dtype}")
        
        # 验证
        A_fp32 = A_fp8.to(torch.float32)
        B_fp32 = B_fp8.to(torch.float32)
        D_ref = (scale_a * scale_b * torch.mm(A_fp32, B_fp32.t())).to(torch.bfloat16)
        
        max_diff = (D.float() - D_ref.float()).abs().max().item()
        print(f"Max diff vs reference: {max_diff:.6f}")
        
        print("✓ Tensorwide Scaling: WORKS!")
        return True
    except Exception as e:
        print(f"✗ Tensorwide Scaling: FAILED - {e}")
        return False

def test_outer_vec(ext):
    """测试 Outer Vector Scaling (向量)"""
    print("\n" + "=" * 60)
    print("Test 2: Outer Vector Scaling (per-row/per-col scale)")
    print("=" * 60)
    
    M, N, K = 64, 128, 256
    fp8_dtype = torch.float8_e4m3fn
    
    torch.manual_seed(42)
    A_fp8 = (torch.randn(M, K, device="cuda") * 0.1).to(fp8_dtype)
    B_fp8 = (torch.randn(N, K, device="cuda") * 0.1).to(fp8_dtype)
    
    scale_A = torch.ones(M, device="cuda", dtype=torch.float32)
    scale_B = torch.ones(N, device="cuda", dtype=torch.float32)
    
    print(f"A: {A_fp8.shape}, B: {B_fp8.shape}")
    print(f"scale_A: {scale_A.shape}, scale_B: {scale_B.shape}")
    
    try:
        D = ext.fp8_gemm_outer_vec(A_fp8, B_fp8, scale_A, scale_B, torch.bfloat16)
        print(f"Output: {D.shape}, dtype: {D.dtype}")
        
        # 验证
        A_fp32 = A_fp8.to(torch.float32)
        B_fp32 = B_fp8.to(torch.float32)
        D_ref = scale_A.unsqueeze(1) * scale_B.unsqueeze(0) * torch.mm(A_fp32, B_fp32.t())
        D_ref = D_ref.to(torch.bfloat16)
        
        max_diff = (D.float() - D_ref.float()).abs().max().item()
        print(f"Max diff vs reference: {max_diff:.6f}")
        
        print("✓ Outer Vector Scaling: WORKS!")
        return True
    except Exception as e:
        print(f"✗ Outer Vector Scaling: NOT SUPPORTED - {e}")
        return False

def main():
    print("cuBLASLt FP8 Scale Mode Comparison")
    print("=" * 60)
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    
    # 编译
    ext = compile_extension()
    
    # 检查 GPU 支持
    print("\n" + "=" * 60)
    print("Checking GPU capability")
    print("=" * 60)
    ext.check_outer_vec_support()
    
    # 测试
    tensorwide_ok = test_tensorwide(ext)
    outer_vec_ok = test_outer_vec(ext)
    
    # 总结
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Tensorwide Scaling (scalar): {'✓ SUPPORTED' if tensorwide_ok else '✗ NOT SUPPORTED'}")
    print(f"Outer Vector Scaling (vector): {'✓ SUPPORTED' if outer_vec_ok else '✗ NOT SUPPORTED'}")
    
    if tensorwide_ok and not outer_vec_ok:
        print("\n结论: 此 GPU 支持 FP8 GEMM，但不支持 Outer Vector Scaling")
        print("需要在 Hopper (H100/H200) 上测试 Outer Vector Scaling")
        print("\n替代方案:")
        print("  1. 使用 Tensorwide (标量) + 手动 broadcast scale")
        print("  2. 输出 FP32，然后 fused kernel 应用 scale 并转 BF16")
    
    return 0 if tensorwide_ok else 1

if __name__ == "__main__":
    sys.exit(main())
