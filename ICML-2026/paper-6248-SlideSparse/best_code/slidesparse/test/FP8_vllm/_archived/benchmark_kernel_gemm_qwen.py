#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_kernel_gemm_qwen.py - GEMM 调用开销分析

专门测试三个后端的 GEMM kernel 调用开销：
1. CUTLASS: ops.cutlass_scaled_mm vs torch.ops._C 直接调用
2. cuBLASLt: Python wrapper vs ctypes 直接调用
3. cuSPARSELt: Python wrapper vs ctypes 直接调用

目的：
    量化 Python 包装带来的开销，评估是否需要优化

Qwen2.5-0.5B 线性层配置：
    1. qkv_proj:  N=1152, K=896
    2. o_proj:    N=896,  K=896
    3. gate_up:   N=9728, K=896
    4. down_proj: N=896,  K=4864

使用方法：
    python3 benchmark_kernel_gemm_qwen.py
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Callable

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

_SCRIPT_DIR = Path(__file__).parent.absolute()
_SLIDESPARSE_TEST_DIR = _SCRIPT_DIR.parent
_SLIDESPARSE_DIR = _SLIDESPARSE_TEST_DIR.parent
_VLLMBENCH_DIR = _SLIDESPARSE_DIR.parent
_WEIGHT_CONVERT_DIR = _SLIDESPARSE_DIR / "weight_convert"

sys.path.insert(0, str(_WEIGHT_CONVERT_DIR))
sys.path.insert(0, str(_VLLMBENCH_DIR))

import torch


# ============================================================================
# 配置
# ============================================================================

@dataclass
class LinearLayerConfig:
    """线性层配置"""
    name: str
    N: int
    K: int


QWEN_LAYERS = [
    LinearLayerConfig("qkv_proj",  N=1152, K=896),
    LinearLayerConfig("o_proj",    N=896,  K=896),
    LinearLayerConfig("gate_up",   N=9728, K=896),
    LinearLayerConfig("down_proj", N=896,  K=4864),
]

M_DECODE = 32
M_PREFILL = 2048

SPARSITY_Z = 2
SPARSITY_L = 8

WARMUP = 100
REPEAT = 500


# ============================================================================
# 辅助函数
# ============================================================================

def get_fp8_dtype():
    return torch.float8_e4m3fn


def prune_weight(weight: torch.Tensor) -> torch.Tensor:
    """对权重应用 2:8 稀疏剪枝"""
    from slidesparse.weight_convert.prune import prune_tensor
    return prune_tensor(weight, SPARSITY_Z, SPARSITY_L, mode="magnitude")


def slide_weight(weight: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """对剪枝后的权重执行 slide 转换"""
    from slidesparse.utils import SlideSparseConfig
    from slidesparse.weight_convert.slide import slide_tensor
    
    config = SlideSparseConfig(Z=SPARSITY_Z, L=SPARSITY_L)
    slided, metadata = slide_tensor(weight, config, align_to=32, verbose=False)
    return slided, metadata["output_k"]


def compress_weight(slide_weight: torch.Tensor) -> torch.Tensor:
    """cuSPARSELt 在线压缩"""
    from slidesparse.weight_convert.compress import compress_tensor_online
    return compress_tensor_online(slide_weight, verbose=False)


def benchmark_kernel(fn: Callable, warmup: int = WARMUP, repeat: int = REPEAT) -> float:
    """精确的 kernel 计时，返回平均时间 (us)"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / repeat * 1000  # us


# ============================================================================
# 测试数据准备
# ============================================================================

@dataclass
class GemmTestData:
    """GEMM 测试数据"""
    name: str
    M: int
    N: int
    K: int
    K_slide: int
    # 输入（已量化）
    qinput: torch.Tensor          # [M, K] FP8
    scale_a: torch.Tensor         # [M, 1] FP32
    # CUTLASS 权重
    weight_cutlass: torch.Tensor  # [K, N] 列主序 FP8
    scale_b: torch.Tensor         # [N, 1] 或 [1] FP32
    # cuBLASLt 权重
    weight_cublaslt: torch.Tensor # [N, K] FP8
    # cuSPARSELt 权重
    weight_compressed: torch.Tensor  # [compressed_size] uint8
    qinput_slide: torch.Tensor       # [M, K_slide] FP8
    scale_a_slide: torch.Tensor      # [M, 1] FP32


def prepare_gemm_data(M: int, device: str = "cuda", seed: int = 42) -> List[GemmTestData]:
    """准备 GEMM 测试数据"""
    from vllm import _custom_ops as ops
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    data_list = []
    
    for layer in QWEN_LAYERS:
        N, K = layer.N, layer.K
        
        # 1. 生成输入并量化
        input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        qinput, scale_a = ops.scaled_fp8_quant(input_bf16, None, use_per_token_if_dynamic=True)
        
        # 2. 生成权重并剪枝
        weight_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.1
        weight_pruned = prune_weight(weight_bf16)
        
        # 3. 量化权重
        fp8_max = torch.finfo(get_fp8_dtype()).max
        weight_absmax = weight_pruned.abs().max(dim=1, keepdim=True).values
        weight_scale = (weight_absmax / fp8_max).to(torch.float32)
        weight_scale = torch.clamp(weight_scale, min=1e-12)
        weight_fp8 = (weight_pruned.float() / weight_scale).to(get_fp8_dtype())  # [N, K]
        
        # 4. CUTLASS 格式: [K, N] 列主序
        weight_cutlass = weight_fp8.t()  # [K, N]
        
        # 5. cuBLASLt 格式: [N, K]
        weight_cublaslt = weight_fp8.contiguous()
        
        # 6. cuSPARSELt: slide + compress
        weight_slide, K_slide = slide_weight(weight_fp8)
        weight_compressed = compress_weight(weight_slide)
        
        # 7. 为 cuSPARSELt 准备 slide 后的输入
        from slidesparse.core.SlideSparseLinearMethod_FP8 import quant_slide_fp8_kernel
        qinput_slide, scale_a_slide = quant_slide_fp8_kernel(input_bf16, L=SPARSITY_L)
        
        data_list.append(GemmTestData(
            name=layer.name,
            M=M, N=N, K=K, K_slide=K_slide,
            qinput=qinput,
            scale_a=scale_a,
            weight_cutlass=weight_cutlass,
            scale_b=weight_scale,
            weight_cublaslt=weight_cublaslt,
            weight_compressed=weight_compressed,
            qinput_slide=qinput_slide,
            scale_a_slide=scale_a_slide,
        ))
    
    return data_list


# ============================================================================
# CUTLASS GEMM 测试
# ============================================================================

def test_cutlass_gemm(data_list: List[GemmTestData], M: int):
    """测试 CUTLASS GEMM 调用开销"""
    from vllm import _custom_ops as ops
    
    print(f"\n{'='*80}")
    print(f"CUTLASS GEMM 调用开销 (M={M})")
    print(f"{'='*80}")
    
    results = []
    
    for data in data_list:
        # 预分配输出（用于直接调用）
        out_pre = torch.empty((data.M, data.N), dtype=torch.bfloat16, device="cuda")
        
        # 1. 通过 ops.cutlass_scaled_mm 调用（有 Python 包装）
        def wrapper_fn():
            return ops.cutlass_scaled_mm(
                data.qinput, data.weight_cutlass,
                out_dtype=torch.bfloat16,
                scale_a=data.scale_a, scale_b=data.scale_b
            )
        
        # 2. 直接调用 torch.ops._C（无输出分配开销）
        def direct_fn():
            torch.ops._C.cutlass_scaled_mm(
                out_pre, data.qinput, data.weight_cutlass,
                data.scale_a, data.scale_b, None
            )
        
        t_wrapper = benchmark_kernel(wrapper_fn)
        t_direct = benchmark_kernel(direct_fn)
        overhead = t_wrapper - t_direct
        
        results.append({
            "name": data.name,
            "N": data.N, "K": data.K,
            "wrapper": t_wrapper,
            "direct": t_direct,
            "overhead": overhead,
        })
        
        print(f"  {data.name:12s} (N={data.N:4d}, K={data.K:4d}): "
              f"wrapper={t_wrapper:7.2f}us, direct={t_direct:7.2f}us, "
              f"overhead={overhead:5.2f}us ({overhead/t_direct*100:5.1f}%)")
    
    avg_wrapper = sum(r["wrapper"] for r in results) / len(results)
    avg_direct = sum(r["direct"] for r in results) / len(results)
    avg_overhead = sum(r["overhead"] for r in results) / len(results)
    
    print(f"\n  平均: wrapper={avg_wrapper:.2f}us, direct={avg_direct:.2f}us, "
          f"overhead={avg_overhead:.2f}us ({avg_overhead/avg_direct*100:.1f}%)")
    
    return results


# ============================================================================
# cuBLASLt GEMM 测试
# ============================================================================

def test_cublaslt_gemm(data_list: List[GemmTestData], M: int):
    """测试 cuBLASLt GEMM 调用开销"""
    os.environ["USE_CUBLASLT"] = "1"
    os.environ["USE_CUSPARSELT"] = "0"
    
    from slidesparse.core.SlideSparseLinearMethod_FP8 import (
        _get_gemm_extension,
        get_inner_dtype_str,
    )
    
    print(f"\n{'='*80}")
    print(f"cuBLASLt GEMM 调用开销 (M={M})")
    print(f"{'='*80}")
    
    ext = _get_gemm_extension("cublaslt")
    lib = ext._lib
    inner_dtype = get_inner_dtype_str()
    inner_dtype_bytes = inner_dtype.encode()
    stream = torch.cuda.current_stream().cuda_stream
    
    results = []
    
    for data in data_list:
        # 预分配输出
        out_dtype = torch.float32 if inner_dtype == "fp32" else torch.bfloat16
        out_pre = torch.empty((data.M, data.N), dtype=out_dtype, device="cuda")
        
        # 1. 通过 Python wrapper 调用
        def wrapper_fn():
            return ext.cublaslt_fp8_mm(data.weight_cublaslt, data.qinput, inner_dtype)
        
        # 2. 直接调用 ctypes C 函数
        def direct_fn():
            lib.cublaslt_fp8_mm(
                data.weight_cublaslt.data_ptr(),
                data.qinput.data_ptr(),
                out_pre.data_ptr(),
                data.M, data.N, data.K,
                inner_dtype_bytes, stream
            )
        
        t_wrapper = benchmark_kernel(wrapper_fn)
        t_direct = benchmark_kernel(direct_fn)
        overhead = t_wrapper - t_direct
        
        results.append({
            "name": data.name,
            "N": data.N, "K": data.K,
            "wrapper": t_wrapper,
            "direct": t_direct,
            "overhead": overhead,
        })
        
        print(f"  {data.name:12s} (N={data.N:4d}, K={data.K:4d}): "
              f"wrapper={t_wrapper:7.2f}us, direct={t_direct:7.2f}us, "
              f"overhead={overhead:5.2f}us ({overhead/t_direct*100:5.1f}%)")
    
    avg_wrapper = sum(r["wrapper"] for r in results) / len(results)
    avg_direct = sum(r["direct"] for r in results) / len(results)
    avg_overhead = sum(r["overhead"] for r in results) / len(results)
    
    print(f"\n  平均: wrapper={avg_wrapper:.2f}us, direct={avg_direct:.2f}us, "
          f"overhead={avg_overhead:.2f}us ({avg_overhead/avg_direct*100:.1f}%)")
    
    return results


# ============================================================================
# cuSPARSELt GEMM 测试
# ============================================================================

def test_cusparselt_gemm(data_list: List[GemmTestData], M: int):
    """测试 cuSPARSELt GEMM 调用开销"""
    os.environ["USE_CUBLASLT"] = "0"
    os.environ["USE_CUSPARSELT"] = "1"
    os.environ["SPARSITY"] = f"{SPARSITY_Z}_{SPARSITY_L}"
    
    from slidesparse.core.SlideSparseLinearMethod_FP8 import (
        _get_gemm_extension,
        get_inner_dtype_str,
    )
    
    print(f"\n{'='*80}")
    print(f"cuSPARSELt GEMM 调用开销 (M={M}, sparsity={SPARSITY_Z}:{SPARSITY_L})")
    print(f"{'='*80}")
    
    ext = _get_gemm_extension("cusparselt")
    lib = ext._lib
    inner_dtype = get_inner_dtype_str()
    inner_dtype_bytes = inner_dtype.encode()
    stream = torch.cuda.current_stream().cuda_stream
    
    results = []
    
    for data in data_list:
        # 预分配输出
        out_dtype = torch.float32 if inner_dtype == "fp32" else torch.bfloat16
        M_pad = data.qinput_slide.shape[0]
        out_pre = torch.empty((M_pad, data.N), dtype=out_dtype, device="cuda")
        
        # 1. 通过 Python wrapper 调用
        def wrapper_fn():
            return ext.cusparselt_fp8_mm(
                data.weight_compressed, data.qinput_slide,
                N=data.N, K_slide=data.K_slide,
                inner_dtype=inner_dtype
            )
        
        # 2. 直接调用 ctypes C 函数
        def direct_fn():
            lib.cusparselt_fp8_mm(
                data.weight_compressed.data_ptr(),
                data.qinput_slide.data_ptr(),
                out_pre.data_ptr(),
                M_pad, data.N, data.K_slide,
                inner_dtype_bytes, stream
            )
        
        t_wrapper = benchmark_kernel(wrapper_fn)
        t_direct = benchmark_kernel(direct_fn)
        overhead = t_wrapper - t_direct
        
        results.append({
            "name": data.name,
            "N": data.N, "K_slide": data.K_slide,
            "wrapper": t_wrapper,
            "direct": t_direct,
            "overhead": overhead,
        })
        
        print(f"  {data.name:12s} (N={data.N:4d}, K_slide={data.K_slide:4d}): "
              f"wrapper={t_wrapper:7.2f}us, direct={t_direct:7.2f}us, "
              f"overhead={overhead:5.2f}us ({overhead/t_direct*100:5.1f}%)")
    
    avg_wrapper = sum(r["wrapper"] for r in results) / len(results)
    avg_direct = sum(r["direct"] for r in results) / len(results)
    avg_overhead = sum(r["overhead"] for r in results) / len(results)
    
    print(f"\n  平均: wrapper={avg_wrapper:.2f}us, direct={avg_direct:.2f}us, "
          f"overhead={avg_overhead:.2f}us ({avg_overhead/avg_direct*100:.1f}%)")
    
    return results


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 80)
    print("GEMM 调用开销分析 (Qwen2.5-0.5B 配置)")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  M_decode = {M_DECODE}")
    print(f"  M_prefill = {M_PREFILL}")
    print(f"  Sparsity = {SPARSITY_Z}:{SPARSITY_L}")
    print(f"  Warmup = {WARMUP}, Repeat = {REPEAT}")
    print(f"\n线性层配置:")
    for layer in QWEN_LAYERS:
        print(f"  {layer.name:12s}: N={layer.N:5d}, K={layer.K:4d}")
    
    results = {"decode": {}, "prefill": {}}
    
    # ========================================================================
    # Decode 阶段 (M=32)
    # ========================================================================
    print(f"\n\n{'#'*80}")
    print(f"# Decode 阶段 (M={M_DECODE})")
    print(f"{'#'*80}")
    
    print("\n准备测试数据...")
    data_decode = prepare_gemm_data(M_DECODE)
    
    results["decode"]["cutlass"] = test_cutlass_gemm(data_decode, M_DECODE)
    results["decode"]["cublaslt"] = test_cublaslt_gemm(data_decode, M_DECODE)
    results["decode"]["cusparselt"] = test_cusparselt_gemm(data_decode, M_DECODE)
    
    # ========================================================================
    # Prefill 阶段 (M=2048)
    # ========================================================================
    print(f"\n\n{'#'*80}")
    print(f"# Prefill 阶段 (M={M_PREFILL})")
    print(f"{'#'*80}")
    
    print("\n准备测试数据...")
    data_prefill = prepare_gemm_data(M_PREFILL)
    
    results["prefill"]["cutlass"] = test_cutlass_gemm(data_prefill, M_PREFILL)
    results["prefill"]["cublaslt"] = test_cublaslt_gemm(data_prefill, M_PREFILL)
    results["prefill"]["cusparselt"] = test_cusparselt_gemm(data_prefill, M_PREFILL)
    
    # ========================================================================
    # 汇总
    # ========================================================================
    print(f"\n\n{'='*80}")
    print("汇总对比")
    print("=" * 80)
    
    def summarize(phase_name, phase_results):
        print(f"\n{phase_name}:")
        print(f"{'Backend':<12} | {'Wrapper (us)':>12} | {'Direct (us)':>12} | "
              f"{'Overhead (us)':>14} | {'Overhead %':>10}")
        print("-" * 70)
        
        for backend in ["cutlass", "cublaslt", "cusparselt"]:
            r = phase_results[backend]
            avg_w = sum(x["wrapper"] for x in r) / len(r)
            avg_d = sum(x["direct"] for x in r) / len(r)
            avg_o = sum(x["overhead"] for x in r) / len(r)
            pct = avg_o / avg_d * 100 if avg_d > 0 else 0
            
            print(f"{backend.upper():<12} | {avg_w:>12.2f} | {avg_d:>12.2f} | "
                  f"{avg_o:>14.2f} | {pct:>9.1f}%")
    
    summarize(f"Decode (M={M_DECODE})", results["decode"])
    summarize(f"Prefill (M={M_PREFILL})", results["prefill"])
    
    # ========================================================================
    # 结论
    # ========================================================================
    print(f"\n\n{'='*80}")
    print("结论")
    print("=" * 80)
    print("""
GEMM 调用开销分析:

1. CUTLASS (ops.cutlass_scaled_mm):
   - wrapper 开销主要来自: torch.empty 分配输出, view reshape
   - 直接调用 torch.ops._C 可以消除输出分配开销（预分配）

2. cuBLASLt (ctypes wrapper):
   - wrapper 开销主要来自: torch.empty 分配输出, Python 方法调用
   - 直接调用 lib.cublaslt_fp8_mm 可以消除这些开销

3. cuSPARSELt (ctypes wrapper):
   - 与 cuBLASLt 类似

对比 quant 的 nn.Module 开销 (~30us):
   - GEMM 的 Python 包装开销通常 < 5us
   - 相对较小，优化收益有限
   - 但如果追求极致性能，可以考虑预分配输出 + 直接调用
""")


if __name__ == "__main__":
    main()
