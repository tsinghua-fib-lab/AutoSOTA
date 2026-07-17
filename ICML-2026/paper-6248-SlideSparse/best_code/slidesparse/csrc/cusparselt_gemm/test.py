#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
cuSPARSELt GEMM 测试脚本

测试 cuSPARSELt 2:4 稀疏 GEMM 的正确性和性能。

测试内容:
1. FP8E4M3 GEMM (inner_dtype = bf16/fp32)
2. INT8 GEMM (inner_dtype = bf16/fp32)
3. 与 torch.matmul 参考结果比对
4. 性能测试 (throughput)

布局约定 (TN-CC):
- W: [N, K] 行主序, 稀疏权重
- A: [M, K] 行主序, 稠密激活  
- D: [M, N] 行主序, 输出

使用方法:
    python3 test.py                    # 运行所有测试
    python3 test.py --correctness      # 仅正确性测试
    python3 test.py --performance      # 仅性能测试
    python3 test.py --verbose          # 详细输出
"""

import argparse
import ctypes
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# 添加项目路径
_SCRIPT_DIR = Path(__file__).parent.absolute()
_CSRC_DIR = _SCRIPT_DIR.parent
_SLIDESPARSE_ROOT = _CSRC_DIR.parent
_PROJECT_ROOT = _SLIDESPARSE_ROOT.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 使用顶层 utils 工具
from slidesparse.utils import (
    find_file,
    ensure_cusparselt_loaded,
    hw_info,
)

# 导入压缩工具
from slidesparse.weight_convert.compress import (
    compress_tensor_online,
    get_compress_sizes,
)


# =============================================================================
# 常量配置
# =============================================================================

# 默认测试维度 (必须是 32 的倍数)
DEFAULT_M = 128
DEFAULT_N = 256
DEFAULT_K = 512

# 2:4 稀疏 = 每 4 个元素中 2 个非零
SPARSITY_BLOCK = 4
SPARSITY_NONZERO = 2

# 误差容忍度说明:
# - BF16 输出: 累加后转 BF16 会丢失精度，相对误差约 0.1-0.5%
# - FP32/INT32 输出: 精度完美，几乎无误差
FP8_BF16_RTOL = 0.01   # FP8 -> BF16: 相对误差 ~0.4%
FP8_FP32_RTOL = 0.0001 # FP8 -> FP32: 几乎无误差
INT8_BF16_RTOL = 0.01  # INT8 -> BF16: 相对误差 ~0.4%
INT8_INT32_RTOL = 1e-6 # INT8 -> INT32: 完全精确 (允许极小浮点误差)

# 性能测试配置
WARMUP_ITERS = 10
BENCHMARK_ITERS = 100


# =============================================================================
# 工具函数
# =============================================================================

def print_header(msg: str) -> None:
    """打印标题"""
    print("\n" + "=" * 60)
    print(msg)
    print("=" * 60)


def print_info(msg: str) -> None:
    """打印信息"""
    print(f"  {msg}")


def print_success(msg: str) -> None:
    """打印成功信息"""
    print(f"  ✓ {msg}")


def print_error(msg: str) -> None:
    """打印错误信息"""
    print(f"  ✗ {msg}")


def create_2to4_sparse_tensor(
    shape: Tuple[int, int],
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> torch.Tensor:
    """
    创建满足 2:4 结构化稀疏的张量
    
    每 4 个连续元素中恰好有 2 个非零值。
    
    Args:
        shape: (N, K) 张量形状
        dtype: 数据类型
        device: 设备
        
    Returns:
        满足 2:4 稀疏的张量
    """
    N, K = shape
    assert K % SPARSITY_BLOCK == 0, f"K must be multiple of {SPARSITY_BLOCK}"
    
    # 创建稠密张量
    tensor = torch.randn(N, K, dtype=torch.float32, device=device)
    
    # 应用 2:4 稀疏掩码
    # 重塑为 [N, K/4, 4]
    grouped = tensor.view(N, K // SPARSITY_BLOCK, SPARSITY_BLOCK)
    
    # 对每组 4 个元素，保留绝对值最大的 2 个
    _, indices = grouped.abs().topk(SPARSITY_NONZERO, dim=-1)
    
    # 创建掩码
    mask = torch.zeros_like(grouped, dtype=torch.bool)
    mask.scatter_(-1, indices, True)
    
    # 应用掩码
    sparse_tensor = grouped * mask.float()
    sparse_tensor = sparse_tensor.view(N, K)
    
    # 转换到目标类型
    return sparse_tensor.to(dtype)


def quantize_to_fp8(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将 BF16/FP32 张量量化为 FP8E4M3 (per-tensor scale)
    
    Args:
        tensor: 输入张量 [*, K]
        
    Returns:
        (qout, scale) - 量化后的 FP8 张量和 scale
    """
    # FP8E4M3 范围: ~[-448, 448]
    amax = tensor.abs().max()
    scale = (448.0 / amax).clamp(max=1e4)
    
    # 量化
    qout = (tensor * scale).to(torch.float8_e4m3fn)
    
    # scale 用于反量化: original ≈ qout / scale
    return qout, scale


def quantize_to_int8(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将 BF16/FP32 张量量化为 INT8 (per-tensor scale)
    
    Args:
        tensor: 输入张量 [*, K]
        
    Returns:
        (qout, scale) - 量化后的 INT8 张量和 scale
    """
    # INT8 范围: [-127, 127]
    amax = tensor.abs().max()
    scale = (127.0 / amax).clamp(max=1e4)
    
    # 量化并截断
    qout = (tensor * scale).round().clamp(-127, 127).to(torch.int8)
    
    return qout, scale


def load_gemm_extension() -> ctypes.CDLL:
    """
    加载 cuSPARSELt GEMM 扩展
    
    Returns:
        ctypes 加载的库
    """
    # 确保 cuSPARSELt 库已加载
    ensure_cusparselt_loaded()
    
    # 查找 .so 文件
    build_dir = _SCRIPT_DIR / "build"
    so_path = find_file("cusparselt_gemm", search_dir=build_dir, ext=".so")
    
    if so_path is None:
        raise FileNotFoundError(
            f"cuSPARSELt GEMM extension not found in {build_dir}.\n"
            f"Please run: python3 build_cusparselt.py build"
        )
    
    print_info(f"Loading: {so_path.name}")
    
    lib = ctypes.CDLL(str(so_path))
    
    # 设置函数签名
    lib.cusparselt_gemm_get_last_error.argtypes = []
    lib.cusparselt_gemm_get_last_error.restype = ctypes.c_char_p
    
    # GEMM 签名: int fn(W_compressed, A, D, M, N, K, inner_dtype, stream)
    gemm_sig = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 3 + [ctypes.c_char_p, ctypes.c_void_p]
    for name in ["cusparselt_fp8_mm", "cusparselt_int8_mm"]:
        getattr(lib, name).argtypes = gemm_sig
        getattr(lib, name).restype = ctypes.c_int
    
    lib.cusparselt_is_available.argtypes = []
    lib.cusparselt_is_available.restype = ctypes.c_int
    
    return lib


def call_cusparselt_gemm(
    lib: ctypes.CDLL,
    W_compressed: torch.Tensor,
    A: torch.Tensor,
    M: int, N: int, K: int,
    input_dtype: str,
    inner_dtype: str,
) -> torch.Tensor:
    """
    调用 cuSPARSELt GEMM
    
    Args:
        lib: ctypes 库
        W_compressed: 压缩后的权重 (1D uint8)
        A: 激活张量 [M, K] (FP8/INT8)
        M, N, K: 矩阵维度
        input_dtype: "fp8" 或 "int8"
        inner_dtype: "bf16", "fp32" 或 "int32"
        
    Returns:
        D: 输出张量 [M, N] (BF16/FP32/INT32)
    """
    # 选择输出类型
    if inner_dtype == "fp32":
        out_torch_dtype = torch.float32
    elif inner_dtype == "bf16":
        out_torch_dtype = torch.bfloat16
    elif inner_dtype == "int32":
        out_torch_dtype = torch.int32
    else:
        raise ValueError(f"Unsupported inner_dtype: {inner_dtype}")
    
    # 分配输出张量
    D = torch.empty((M, N), dtype=out_torch_dtype, device=A.device)
    
    # 选择 GEMM 函数
    fn_name = "cusparselt_fp8_mm" if input_dtype == "fp8" else "cusparselt_int8_mm"
    fn = getattr(lib, fn_name)
    
    # 调用 GEMM
    ret = fn(
        W_compressed.data_ptr(),
        A.data_ptr(),
        D.data_ptr(),
        M, N, K,
        inner_dtype.encode(),
        torch.cuda.current_stream().cuda_stream
    )
    
    if ret != 0:
        err = lib.cusparselt_gemm_get_last_error()
        raise RuntimeError(f"{fn_name} failed: {err.decode() if err else 'Unknown'}")
    
    # 同步
    torch.cuda.synchronize()
    
    return D


# =============================================================================
# 正确性测试
# =============================================================================

def test_correctness(
    lib: ctypes.CDLL,
    M: int, N: int, K: int,
    input_dtype: str,
    inner_dtype: str,
    verbose: bool = False,
) -> bool:
    """
    测试 GEMM 正确性
    
    Args:
        lib: ctypes 库
        M, N, K: 矩阵维度
        input_dtype: "fp8" 或 "int8"
        inner_dtype: "bf16" 或 "fp32"
        verbose: 详细输出
        
    Returns:
        测试是否通过
    """
    print_info(f"Testing {input_dtype.upper()} GEMM (out={inner_dtype})...")
    print_info(f"  Dimensions: M={M}, N={N}, K={K}")
    
    # 1. 生成 BF16 数据
    # W: [N, K] 稀疏权重
    W_bf16 = create_2to4_sparse_tensor((N, K), dtype=torch.bfloat16, device="cuda")
    # A: [M, K] 稠密激活  
    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    
    if verbose:
        print_info(f"  W_bf16: {W_bf16.shape}, sparsity verified")
        print_info(f"  A_bf16: {A_bf16.shape}")
    
    # 2. 量化数据
    if input_dtype == "fp8":
        W_quant, W_scale = quantize_to_fp8(W_bf16.float())
        A_quant, A_scale = quantize_to_fp8(A_bf16.float())
    else:
        W_quant, W_scale = quantize_to_int8(W_bf16.float())
        A_quant, A_scale = quantize_to_int8(A_bf16.float())
    
    # 根据输入和输出类型选择误差容忍度
    if input_dtype == "fp8" and inner_dtype == "bf16":
        rtol = FP8_BF16_RTOL
    elif input_dtype == "fp8" and inner_dtype == "fp32":
        rtol = FP8_FP32_RTOL
    elif input_dtype == "int8" and inner_dtype == "bf16":
        rtol = INT8_BF16_RTOL
    else:  # int8 + int32
        rtol = INT8_INT32_RTOL
    
    if verbose:
        print_info(f"  W_quant: {W_quant.shape}, dtype={W_quant.dtype}, scale={W_scale.item():.4f}")
        print_info(f"  A_quant: {A_quant.shape}, dtype={A_quant.dtype}, scale={A_scale.item():.4f}")
    
    # 3. 计算参考结果 - 使用量化后的值计算，这才是真正的比对基准
    # D_ref = A_quant @ W_quant^T = [M, K] @ [K, N] = [M, N]
    # 注意：需要转换为 float 进行计算，因为 PyTorch 不直接支持 FP8/INT8 matmul
    D_ref = torch.matmul(A_quant.float(), W_quant.float().t())
    
    if verbose:
        print_info(f"  Reference D_ref (using quant values): {D_ref.shape}")
    
    # 4. 压缩权重 (W 必须是 contiguous)
    W_quant = W_quant.contiguous()
    W_compressed = compress_tensor_online(W_quant, verbose=verbose)
    
    if verbose:
        print_info(f"  W_compressed: {W_compressed.numel()} bytes (1D uint8)")
    
    # 5. 调用 cuSPARSELt GEMM
    # A 也必须是 contiguous
    A_quant = A_quant.contiguous()
    
    D_sparse = call_cusparselt_gemm(
        lib, W_compressed, A_quant, M, N, K,
        input_dtype, inner_dtype
    )
    
    if verbose:
        print_info(f"  D_sparse: {D_sparse.shape}, dtype={D_sparse.dtype}")
    
    # 6. 直接比较 GEMM 结果（不需要反量化，因为参考值也是用量化值算的）
    D_sparse_float = D_sparse.float()
    D_ref_float = D_ref.float()
    
    # 计算误差
    abs_diff = (D_sparse_float - D_ref_float).abs()
    rel_diff = abs_diff / (D_ref_float.abs() + 1e-6)
    
    max_abs_err = abs_diff.max().item()
    max_rel_err = rel_diff.max().item()
    mean_abs_err = abs_diff.mean().item()
    mean_rel_err = rel_diff.mean().item()
    
    print_info(f"  Max  Error: abs={max_abs_err:.4f}, rel={max_rel_err:.6f}")
    print_info(f"  Mean Error: abs={mean_abs_err:.4f}, rel={mean_rel_err:.6f}")
    
    # 检查是否通过
    # - BF16 输出: 有精度损失，用相对误差
    # - FP32/INT32 输出: 精度完美，几乎无误差
    passed = max_rel_err < rtol
    
    if passed:
        print_success(f"{input_dtype.upper()} GEMM (out={inner_dtype}) PASSED")
    else:
        print_error(f"{input_dtype.upper()} GEMM (out={inner_dtype}) FAILED")
        print_error(f"  Expected rel_err < {rtol}, got {max_rel_err:.6f}")
    
    return passed


# =============================================================================
# 性能测试
# =============================================================================

def test_performance(
    lib: ctypes.CDLL,
    M: int, N: int, K: int,
    input_dtype: str,
    inner_dtype: str,
    warmup: int = WARMUP_ITERS,
    iters: int = BENCHMARK_ITERS,
    verbose: bool = False,
) -> float:
    """
    测试 GEMM 性能
    
    Args:
        lib: ctypes 库
        M, N, K: 矩阵维度
        input_dtype: "fp8" 或 "int8"
        inner_dtype: "bf16" 或 "fp32"
        warmup: warmup 迭代次数
        iters: benchmark 迭代次数
        verbose: 详细输出
        
    Returns:
        平均延迟 (ms)
    """
    print_info(f"Benchmarking {input_dtype.upper()} GEMM (out={inner_dtype})...")
    print_info(f"  Dimensions: M={M}, N={N}, K={K}")
    print_info(f"  Warmup: {warmup}, Iters: {iters}")
    
    # 准备数据
    W_bf16 = create_2to4_sparse_tensor((N, K), dtype=torch.bfloat16, device="cuda")
    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    
    if input_dtype == "fp8":
        W_quant, _ = quantize_to_fp8(W_bf16.float())
        A_quant, _ = quantize_to_fp8(A_bf16.float())
    else:
        W_quant, _ = quantize_to_int8(W_bf16.float())
        A_quant, _ = quantize_to_int8(A_bf16.float())
    
    W_quant = W_quant.contiguous()
    A_quant = A_quant.contiguous()
    W_compressed = compress_tensor_online(W_quant, verbose=False)
    
    # Warmup
    for _ in range(warmup):
        call_cusparselt_gemm(lib, W_compressed, A_quant, M, N, K, input_dtype, inner_dtype)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        call_cusparselt_gemm(lib, W_compressed, A_quant, M, N, K, input_dtype, inner_dtype)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_latency_ms = (end - start) / iters * 1000
    
    # 计算 TFLOPS (2:4 稀疏有效 FLOPS = M * N * K, 实际计算量是一半)
    flops = 2 * M * N * K  # 乘加
    effective_flops = flops  # 2:4 稀疏的有效 FLOPS
    tflops = effective_flops / (avg_latency_ms / 1000) / 1e12
    
    print_info(f"  Avg Latency: {avg_latency_ms:.4f} ms")
    print_info(f"  Throughput:  {tflops:.2f} TFLOPS (effective)")
    
    return avg_latency_ms


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="cuSPARSELt GEMM 测试脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--correctness", action="store_true", help="仅运行正确性测试")
    parser.add_argument("--performance", action="store_true", help="仅运行性能测试")
    parser.add_argument("-M", type=int, default=DEFAULT_M, help=f"M 维度 (default: {DEFAULT_M})")
    parser.add_argument("-N", type=int, default=DEFAULT_N, help=f"N 维度 (default: {DEFAULT_N})")
    parser.add_argument("-K", type=int, default=DEFAULT_K, help=f"K 维度 (default: {DEFAULT_K})")
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERS, help="Warmup 迭代次数")
    parser.add_argument("--iters", type=int, default=BENCHMARK_ITERS, help="Benchmark 迭代次数")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 维度检查
    for dim, name in [(args.M, "M"), (args.N, "N"), (args.K, "K")]:
        if dim % 32 != 0:
            print_error(f"{name}={dim} must be multiple of 32 for cuSPARSELt")
            sys.exit(1)
    
    # 如果两个都没指定，则都运行
    run_correctness = args.correctness or (not args.correctness and not args.performance)
    run_performance = args.performance or (not args.correctness and not args.performance)
    
    print_header("cuSPARSELt GEMM Test")
    print_info(f"GPU: {hw_info.gpu_full_name}")
    print_info(f"CC: {hw_info.sm_code}")
    print_info(f"CUDA: {hw_info.cuda_tag}")
    
    # 加载扩展
    try:
        lib = load_gemm_extension()
    except FileNotFoundError as e:
        print_error(str(e))
        sys.exit(1)
    
    # 检查 cuSPARSELt 可用性
    if lib.cusparselt_is_available() != 1:
        print_error("cuSPARSELt is not available")
        sys.exit(1)
    print_success("cuSPARSELt is available")
    
    # 测试配置
    # cuSPARSELt 支持的组合：
    # - FP8 输入: out_dtype = bf16 或 fp32 (内部 FP32 累加)
    # - INT8 输入: out_dtype = bf16 或 int32 (内部 INT32 累加)
    test_configs = [
        ("fp8", "bf16"),   # FP8 GEMM, 输出 BF16
        ("fp8", "fp32"),   # FP8 GEMM, 输出 FP32
        ("int8", "bf16"),  # INT8 GEMM, 输出 BF16
        ("int8", "int32"), # INT8 GEMM, 输出 INT32
    ]
    
    # 正确性测试
    if run_correctness:
        print_header("Correctness Tests")
        
        all_passed = True
        for input_dtype, inner_dtype in test_configs:
            try:
                passed = test_correctness(
                    lib, args.M, args.N, args.K,
                    input_dtype, inner_dtype,
                    verbose=args.verbose
                )
                all_passed = all_passed and passed
            except Exception as e:
                print_error(f"{input_dtype.upper()} GEMM (out={inner_dtype}) ERROR: {e}")
                all_passed = False
            print()  # 空行分隔
        
        if all_passed:
            print_success("All correctness tests PASSED")
        else:
            print_error("Some correctness tests FAILED")
    
    # 性能测试
    if run_performance:
        print_header("Performance Tests")
        
        for input_dtype, inner_dtype in test_configs:
            try:
                test_performance(
                    lib, args.M, args.N, args.K,
                    input_dtype, inner_dtype,
                    warmup=args.warmup,
                    iters=args.iters,
                    verbose=args.verbose
                )
            except Exception as e:
                print_error(f"{input_dtype.upper()} GEMM (out={inner_dtype}) ERROR: {e}")
            print()  # 空行分隔
    
    print_header("Test Complete")


if __name__ == "__main__":
    main()
