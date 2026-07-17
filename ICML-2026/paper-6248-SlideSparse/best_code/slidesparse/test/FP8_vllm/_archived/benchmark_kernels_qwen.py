#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_kernels_qwen.py - 对齐端到端的 Kernel 基准测试

目标：
    精确模拟 Qwen2.5-0.5B 端到端推理中的 kernel 性能，
    与 SLIDESPARSE_PROFILE 的输出对齐。

Qwen2.5-0.5B 线性层配置（每层 4 个线性操作）：
    1. qkv_proj:  N=1152, K=896   (Q + KV 合并)
    2. o_proj:    N=896,  K=896
    3. gate_up:   N=9728, K=896   (gate + up 合并)
    4. down_proj: N=896,  K=4864

测试方法：
    - 每个 kernel 测试都遍历 4 个 NK 组合，取平均
    - 分别测试 M=32 (Decode) 和 M=2048 (Prefill)
    - 单独测试每个 kernel，不串联

对齐说明：
    端到端 profile 在 warmup 后会 reset，统计的是正式测试阶段的调用。
    由于 ProfileTimer 使用 CUDA events 单独测量每次调用，其结果会比
    批量测量（本脚本的方法）高出约 30-40us，这是 event 记录的固有开销。
    
    ProfileTimer 测量的是"端到端占用时间"，包含:
    - kernel 执行时间
    - CUDA event 记录开销
    - kernel 间的调度 gap
    
    本脚本的批量测量是"纯 kernel 执行时间"，不包含上述开销。

使用方法：
    python3 benchmark_kernels_qwen.py
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

_SCRIPT_DIR = Path(__file__).parent.absolute()
_SLIDESPARSE_TEST_DIR = _SCRIPT_DIR.parent
_SLIDESPARSE_DIR = _SLIDESPARSE_TEST_DIR.parent
_VLLMBENCH_DIR = _SLIDESPARSE_DIR.parent
_WEIGHT_CONVERT_DIR = _SLIDESPARSE_DIR / "weight_convert"

# 先加 weight_convert 目录，确保它的 utils.py 优先
sys.path.insert(0, str(_WEIGHT_CONVERT_DIR))
sys.path.insert(0, str(_VLLMBENCH_DIR))

import torch


# ============================================================================
# Qwen2.5-0.5B 配置
# ============================================================================

@dataclass
class LinearLayerConfig:
    """线性层配置"""
    name: str
    N: int
    K: int


# 4 组 NK 配置（vLLM 合并后的实际维度）
QWEN_LAYERS = [
    LinearLayerConfig("qkv_proj",  N=1152, K=896),
    LinearLayerConfig("o_proj",    N=896,  K=896),
    LinearLayerConfig("gate_up",   N=9728, K=896),
    LinearLayerConfig("down_proj", N=896,  K=4864),
]

# 测试的 M 值
M_DECODE = 32
M_PREFILL = 2048

# 稀疏配置
SPARSITY_Z = 2
SPARSITY_L = 8  # 2:8 稀疏


# ============================================================================
# 辅助函数
# ============================================================================

def get_fp8_dtype():
    return torch.float8_e4m3fn


def prune_weight_28(weight: torch.Tensor) -> torch.Tensor:
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


# ============================================================================
# 测试数据准备
# ============================================================================

@dataclass
class LayerTestData:
    """单层测试数据"""
    name: str
    N: int
    K: int
    K_slide: int  # slide 后的 K 维度
    # Dense 数据
    input_bf16: torch.Tensor       # [M, K]
    weight_fp8: torch.Tensor       # [N, K] for cuBLASLt, [K, N] for CUTLASS
    weight_fp8_t: torch.Tensor     # [K, N] CUTLASS 格式
    weight_scale: torch.Tensor     # [N, 1]
    bias: torch.Tensor             # [N]
    # cuSPARSELt 数据
    weight_slide_fp8: torch.Tensor    # [N, K_slide]
    weight_compressed: torch.Tensor   # [compressed_size] uint8


def prepare_test_data(M: int, device: str = "cuda", seed: int = 42) -> List[LayerTestData]:
    """
    准备 4 组测试数据
    
    所有权重都经过 2:8 剪枝，确保三种后端数学等价
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    data_list = []
    
    for layer in QWEN_LAYERS:
        N, K = layer.N, layer.K
        
        # 1. 生成 BF16 输入和权重
        input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        weight_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.1
        
        # 2. 剪枝
        weight_pruned = prune_weight_28(weight_bf16)
        
        # 3. 量化为 FP8
        fp8_max = torch.finfo(get_fp8_dtype()).max
        weight_absmax = weight_pruned.abs().max(dim=1, keepdim=True).values
        weight_scale = (weight_absmax / fp8_max).to(torch.float32)
        weight_scale = torch.clamp(weight_scale, min=1e-12)
        
        weight_scaled = weight_pruned.float() / weight_scale
        weight_fp8 = weight_scaled.to(get_fp8_dtype())  # [N, K]
        
        # 4. CUTLASS 格式 [K, N]
        weight_fp8_t = weight_fp8.t()  # 列主序视图
        
        # 5. Bias
        bias = torch.zeros(N, dtype=torch.bfloat16, device=device)
        
        # 6. Slide + Compress
        weight_slide_fp8, K_slide = slide_weight(weight_fp8)
        weight_compressed = compress_weight(weight_slide_fp8)
        
        data_list.append(LayerTestData(
            name=layer.name,
            N=N, K=K, K_slide=K_slide,
            input_bf16=input_bf16,
            weight_fp8=weight_fp8,
            weight_fp8_t=weight_fp8_t,
            weight_scale=weight_scale,
            bias=bias,
            weight_slide_fp8=weight_slide_fp8,
            weight_compressed=weight_compressed,
        ))
    
    return data_list


# ============================================================================
# 计时工具
# ============================================================================

def benchmark_kernel(
    fn: Callable,
    warmup: int = 100,
    repeat: int = 500,
) -> float:
    """
    精确的 kernel 计时
    
    Returns:
        平均时间 (ms)
    """
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    # 批量计时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / repeat


def benchmark_kernel_4layers(
    fn_factory: Callable[[LayerTestData], Callable],
    data_list: List[LayerTestData],
    warmup: int = 100,
    repeat: int = 500,
) -> Tuple[float, List[float]]:
    """
    对 4 层分别计时，返回平均和各层时间
    
    Args:
        fn_factory: 给定 LayerTestData，返回要测试的函数
        data_list: 4 层的测试数据
        warmup: 预热次数（每层）
        repeat: 重复次数（每层）
    
    Returns:
        (avg_ms, [layer1_ms, layer2_ms, layer3_ms, layer4_ms])
    """
    times = []
    
    for data in data_list:
        fn = fn_factory(data)
        t = benchmark_kernel(fn, warmup=warmup, repeat=repeat)
        times.append(t)
    
    avg = sum(times) / len(times)
    return avg, times


# ============================================================================
# CUTLASS 路径测试
# ============================================================================

def test_cutlass_kernels(data_list: List[LayerTestData], M: int):
    """测试 CUTLASS 路径的各个 kernel
    
    注意：直接调用 ops.scaled_fp8_quant 而不是 QuantFP8 类，
    避免 nn.Module.__call__ 带来的 ~30us Python 开销。
    这样测量的是纯 kernel 时间，与 cuBLASLt/cuSPARSELt 路径公平对比。
    """
    from vllm import _custom_ops as ops
    
    print(f"\n{'='*80}")
    print(f"CUTLASS 路径 (M={M})")
    print(f"{'='*80}")
    
    # 1. 测试 Quant - 直接调用 ops.scaled_fp8_quant（绑过 QuantFP8 类）
    print("\n[CUTLASS.quant] ops.scaled_fp8_quant (直接调用，无类包装)")
    def quant_factory(data):
        def fn():
            return ops.scaled_fp8_quant(data.input_bf16, None, use_per_token_if_dynamic=True)
        return fn
    
    avg_quant, times_quant = benchmark_kernel_4layers(quant_factory, data_list)
    print(f"  平均: {avg_quant:.4f} ms ({avg_quant*1000:.2f} us)")
    for i, (data, t) in enumerate(zip(data_list, times_quant)):
        print(f"    {data.name:12s} (K={data.K:4d}): {t:.4f} ms ({t*1000:.2f} us)")
    
    # 2. 测试 scaled_mm（需要先 quant 得到 qinput）
    print("\n[CUTLASS.scaled_mm] CUTLASS 融合 GEMM+Dequant")
    
    # 预先计算 qinput 和 scale_a（直接调用 ops）
    qinputs = []
    scale_as = []
    for data in data_list:
        qinput, scale_a = ops.scaled_fp8_quant(data.input_bf16, None, use_per_token_if_dynamic=True)
        qinputs.append(qinput)
        scale_as.append(scale_a)
    
    def scaled_mm_factory(idx):
        data = data_list[idx]
        qinput = qinputs[idx]
        scale_a = scale_as[idx]
        def fn():
            return ops.cutlass_scaled_mm(
                qinput, data.weight_fp8_t,
                out_dtype=torch.bfloat16,
                scale_a=scale_a, scale_b=data.weight_scale,
                bias=data.bias
            )
        return fn
    
    times_mm = []
    for i in range(len(data_list)):
        fn = scaled_mm_factory(i)
        t = benchmark_kernel(fn, warmup=100, repeat=500)
        times_mm.append(t)
    
    avg_mm = sum(times_mm) / len(times_mm)
    print(f"  平均: {avg_mm:.4f} ms ({avg_mm*1000:.2f} us)")
    for i, (data, t) in enumerate(zip(data_list, times_mm)):
        print(f"    {data.name:12s} (N={data.N:4d}, K={data.K:4d}): {t:.4f} ms ({t*1000:.2f} us)")
    
    # 汇总
    print(f"\n[CUTLASS 总计]")
    print(f"  quant + scaled_mm = {avg_quant + avg_mm:.4f} ms ({(avg_quant + avg_mm)*1000:.2f} us)")
    
    return {
        "quant": avg_quant,
        "scaled_mm": avg_mm,
        "total": avg_quant + avg_mm,
    }


# ============================================================================
# cuBLASLt 路径测试
# ============================================================================

def test_cublaslt_kernels(data_list: List[LayerTestData], M: int):
    """测试 cuBLASLt 路径的各个 kernel"""
    os.environ["USE_CUBLASLT"] = "1"
    os.environ["USE_CUSPARSELT"] = "0"
    
    from slidesparse.core.SlideSparseLinearMethod_FP8 import (
        quant_only_fp8_kernel,
        dequant_bias_kernel,
        _get_gemm_extension,
        get_inner_dtype_str,
    )
    
    print(f"\n{'='*80}")
    print(f"cuBLASLt 路径 (M={M})")
    print(f"{'='*80}")
    
    ext = _get_gemm_extension("cublaslt")
    
    # 1. 测试 Quant
    print("\n[cuBLASLt.quant] Triton quant_only_fp8")
    def quant_factory(data):
        def fn():
            return quant_only_fp8_kernel(data.input_bf16)
        return fn
    
    avg_quant, times_quant = benchmark_kernel_4layers(quant_factory, data_list)
    print(f"  平均: {avg_quant:.4f} ms ({avg_quant*1000:.2f} us)")
    for i, (data, t) in enumerate(zip(data_list, times_quant)):
        print(f"    {data.name:12s} (K={data.K:4d}): {t:.4f} ms ({t*1000:.2f} us)")
    
    # 2. 测试 GEMM
    print("\n[cuBLASLt.gemm] cuBLASLt FP8 GEMM")
    
    # 预先计算 qinput
    qinputs = []
    scale_as = []
    for data in data_list:
        qinput, scale_a = quant_only_fp8_kernel(data.input_bf16)
        qinputs.append(qinput)
        scale_as.append(scale_a)
    
    def gemm_factory(idx):
        data = data_list[idx]
        qinput = qinputs[idx]
        def fn():
            return ext.cublaslt_fp8_mm(data.weight_fp8, qinput, get_inner_dtype_str())
        return fn
    
    times_gemm = []
    for i in range(len(data_list)):
        fn = gemm_factory(i)
        t = benchmark_kernel(fn, warmup=100, repeat=500)
        times_gemm.append(t)
    
    avg_gemm = sum(times_gemm) / len(times_gemm)
    print(f"  平均: {avg_gemm:.4f} ms ({avg_gemm*1000:.2f} us)")
    for i, (data, t) in enumerate(zip(data_list, times_gemm)):
        print(f"    {data.name:12s} (N={data.N:4d}, K={data.K:4d}): {t:.4f} ms ({t*1000:.2f} us)")
    
    # 3. 测试 Dequant
    print("\n[cuBLASLt.dequant] Triton dequant_bias")
    
    # 预先计算 gemm_out
    gemm_outs = []
    for i, data in enumerate(data_list):
        gemm_out = ext.cublaslt_fp8_mm(data.weight_fp8, qinputs[i], get_inner_dtype_str())
        gemm_outs.append(gemm_out[:M, :])  # 截断到原始 M
    
    def dequant_factory(idx):
        data = data_list[idx]
        gemm_out = gemm_outs[idx]
        scale_a = scale_as[idx][:M]
        def fn():
            return dequant_bias_kernel(gemm_out, scale_a, data.weight_scale, data.bias, torch.bfloat16)
        return fn
    
    times_dequant = []
    for i in range(len(data_list)):
        fn = dequant_factory(i)
        t = benchmark_kernel(fn, warmup=100, repeat=500)
        times_dequant.append(t)
    
    avg_dequant = sum(times_dequant) / len(times_dequant)
    print(f"  平均: {avg_dequant:.4f} ms ({avg_dequant*1000:.2f} us)")
    for i, (data, t) in enumerate(zip(data_list, times_dequant)):
        print(f"    {data.name:12s} (N={data.N:4d}): {t:.4f} ms ({t*1000:.2f} us)")
    
    # 汇总
    print(f"\n[cuBLASLt 总计]")
    print(f"  quant + gemm + dequant = {avg_quant + avg_gemm + avg_dequant:.4f} ms ({(avg_quant + avg_gemm + avg_dequant)*1000:.2f} us)")
    
    return {
        "quant": avg_quant,
        "gemm": avg_gemm,
        "dequant": avg_dequant,
        "total": avg_quant + avg_gemm + avg_dequant,
    }


# ============================================================================
# cuSPARSELt 路径测试
# ============================================================================

def test_cusparselt_kernels(data_list: List[LayerTestData], M: int):
    """测试 cuSPARSELt 路径的各个 kernel"""
    os.environ["USE_CUBLASLT"] = "0"
    os.environ["USE_CUSPARSELT"] = "1"
    os.environ["SPARSITY"] = f"{SPARSITY_Z}_{SPARSITY_L}"
    
    from slidesparse.core.SlideSparseLinearMethod_FP8 import (
        quant_slide_fp8_kernel,
        dequant_bias_kernel,
        _get_gemm_extension,
        get_inner_dtype_str,
    )
    
    print(f"\n{'='*80}")
    print(f"cuSPARSELt 路径 (M={M}, sparsity={SPARSITY_Z}:{SPARSITY_L})")
    print(f"{'='*80}")
    
    ext = _get_gemm_extension("cusparselt")
    
    # 1. 测试 Quant + Slide
    print("\n[cuSPARSELt.quant_slide] Triton quant_slide_fp8")
    def quant_factory(data):
        def fn():
            return quant_slide_fp8_kernel(data.input_bf16, L=SPARSITY_L)
        return fn
    
    avg_quant, times_quant = benchmark_kernel_4layers(quant_factory, data_list)
    print(f"  平均: {avg_quant:.4f} ms ({avg_quant*1000:.2f} us)")
    for i, (data, t) in enumerate(zip(data_list, times_quant)):
        print(f"    {data.name:12s} (K={data.K:4d}): {t:.4f} ms ({t*1000:.2f} us)")
    
    # 2. 测试 GEMM
    print("\n[cuSPARSELt.gemm] cuSPARSELt 2:4 Sparse GEMM")
    
    # 预先计算 qinput
    qinputs = []
    scale_as = []
    for data in data_list:
        qinput, scale_a = quant_slide_fp8_kernel(data.input_bf16, L=SPARSITY_L)
        qinputs.append(qinput)
        scale_as.append(scale_a)
    
    def gemm_factory(idx):
        data = data_list[idx]
        qinput = qinputs[idx]
        def fn():
            return ext.cusparselt_fp8_mm(
                data.weight_compressed, qinput,
                N=data.N, K_slide=data.K_slide,
                inner_dtype=get_inner_dtype_str()
            )
        return fn
    
    times_gemm = []
    for i in range(len(data_list)):
        fn = gemm_factory(i)
        t = benchmark_kernel(fn, warmup=100, repeat=500)
        times_gemm.append(t)
    
    avg_gemm = sum(times_gemm) / len(times_gemm)
    print(f"  平均: {avg_gemm:.4f} ms ({avg_gemm*1000:.2f} us)")
    for i, (data, t) in enumerate(zip(data_list, times_gemm)):
        print(f"    {data.name:12s} (N={data.N:4d}, K_slide={data.K_slide:4d}): {t:.4f} ms ({t*1000:.2f} us)")
    
    # 3. 测试 Dequant
    print("\n[cuSPARSELt.dequant] Triton dequant_bias")
    
    # 预先计算 gemm_out
    gemm_outs = []
    for i, data in enumerate(data_list):
        gemm_out = ext.cusparselt_fp8_mm(
            data.weight_compressed, qinputs[i],
            N=data.N, K_slide=data.K_slide,
            inner_dtype=get_inner_dtype_str()
        )
        gemm_outs.append(gemm_out[:M, :])
    
    def dequant_factory(idx):
        data = data_list[idx]
        gemm_out = gemm_outs[idx]
        scale_a = scale_as[idx][:M]
        def fn():
            return dequant_bias_kernel(gemm_out, scale_a, data.weight_scale, data.bias, torch.bfloat16)
        return fn
    
    times_dequant = []
    for i in range(len(data_list)):
        fn = dequant_factory(i)
        t = benchmark_kernel(fn, warmup=100, repeat=500)
        times_dequant.append(t)
    
    avg_dequant = sum(times_dequant) / len(times_dequant)
    print(f"  平均: {avg_dequant:.4f} ms ({avg_dequant*1000:.2f} us)")
    for i, (data, t) in enumerate(zip(data_list, times_dequant)):
        print(f"    {data.name:12s} (N={data.N:4d}): {t:.4f} ms ({t*1000:.2f} us)")
    
    # 汇总
    print(f"\n[cuSPARSELt 总计]")
    print(f"  quant_slide + gemm + dequant = {avg_quant + avg_gemm + avg_dequant:.4f} ms ({(avg_quant + avg_gemm + avg_dequant)*1000:.2f} us)")
    
    return {
        "quant_slide": avg_quant,
        "gemm": avg_gemm,
        "dequant": avg_dequant,
        "total": avg_quant + avg_gemm + avg_dequant,
    }


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 80)
    print("Qwen2.5-0.5B Kernel Benchmark (对齐端到端 profile)")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  M_decode = {M_DECODE}")
    print(f"  M_prefill = {M_PREFILL}")
    print(f"  Sparsity = {SPARSITY_Z}:{SPARSITY_L}")
    print(f"\n线性层配置 (每层 4 组 NK):")
    for layer in QWEN_LAYERS:
        print(f"  {layer.name:12s}: N={layer.N:5d}, K={layer.K:4d}")
    
    results = {}
    
    # ========================================================================
    # Decode 阶段 (M=32)
    # ========================================================================
    print(f"\n\n{'#'*80}")
    print(f"# Decode 阶段 (M={M_DECODE})")
    print(f"{'#'*80}")
    
    print("\n准备 Decode 测试数据...")
    data_decode = prepare_test_data(M_DECODE)
    
    results["decode"] = {
        "cutlass": test_cutlass_kernels(data_decode, M_DECODE),
        "cublaslt": test_cublaslt_kernels(data_decode, M_DECODE),
        "cusparselt": test_cusparselt_kernels(data_decode, M_DECODE),
    }
    
    # ========================================================================
    # Prefill 阶段 (M=2048)
    # ========================================================================
    print(f"\n\n{'#'*80}")
    print(f"# Prefill 阶段 (M={M_PREFILL})")
    print(f"{'#'*80}")
    
    print("\n准备 Prefill 测试数据...")
    data_prefill = prepare_test_data(M_PREFILL)
    
    results["prefill"] = {
        "cutlass": test_cutlass_kernels(data_prefill, M_PREFILL),
        "cublaslt": test_cublaslt_kernels(data_prefill, M_PREFILL),
        "cusparselt": test_cusparselt_kernels(data_prefill, M_PREFILL),
    }
    
    # ========================================================================
    # 汇总对比
    # ========================================================================
    print(f"\n\n{'='*80}")
    print("汇总对比 (平均 4 层)")
    print("=" * 80)
    
    def print_comparison(phase, phase_name):
        print(f"\n{phase_name}:")
        print(f"{'Backend':<12} | {'Quant':>10} | {'GEMM':>10} | {'Dequant':>10} | {'Total':>10}")
        print("-" * 60)
        
        cutlass = phase["cutlass"]
        cublaslt = phase["cublaslt"]
        cusparselt = phase["cusparselt"]
        
        # CUTLASS
        print(f"{'CUTLASS':<12} | {cutlass['quant']*1000:>7.2f} us | "
              f"{cutlass['scaled_mm']*1000:>7.2f} us | {'(fused)':>10} | "
              f"{cutlass['total']*1000:>7.2f} us")
        
        # cuBLASLt
        print(f"{'cuBLASLt':<12} | {cublaslt['quant']*1000:>7.2f} us | "
              f"{cublaslt['gemm']*1000:>7.2f} us | {cublaslt['dequant']*1000:>7.2f} us | "
              f"{cublaslt['total']*1000:>7.2f} us")
        
        # cuSPARSELt
        print(f"{'cuSPARSELt':<12} | {cusparselt['quant_slide']*1000:>7.2f} us | "
              f"{cusparselt['gemm']*1000:>7.2f} us | {cusparselt['dequant']*1000:>7.2f} us | "
              f"{cusparselt['total']*1000:>7.2f} us")
        
        # 对比
        print()
        print(f"cuBLASLt vs CUTLASS:    {cublaslt['total']/cutlass['total']:.2f}x")
        print(f"cuSPARSELt vs CUTLASS:  {cusparselt['total']/cutlass['total']:.2f}x")
    
    print_comparison(results["decode"], f"Decode (M={M_DECODE})")
    print_comparison(results["prefill"], f"Prefill (M={M_PREFILL})")
    
    # ========================================================================
    # 与端到端 profile 对比
    # ========================================================================
    print(f"\n\n{'='*80}")
    print("与端到端 Profile 数据对比")
    print("=" * 80)
    print("""
端到端 Profile 数据 (from screenshot):

Decode (after 29000 calls, 增量 9000 calls):
  CUTLASS:     quant=11.1us, scaled_mm=34.6us           total≈45.7us
  cuBLASLt:    quant=33.6us, gemm=27.5us, dequant=30.3us  total≈91.4us + get_ext=5.1us + view=4.5us
  cuSPARSELt:  quant_slide=35.5us, gemm=40.1us, dequant=20.2us  total≈95.8us + get_ext=4.1us + view=3.6us

Prefill (after 20000 calls):
  CUTLASS:     quant=8.3us, scaled_mm=49.3us            total≈57.6us
  cuBLASLt:    quant=30.2us, gemm=54.2us, dequant=21.0us  total≈105.4us + get_ext=3.5us + view=2.7us
  cuSPARSELt:  quant_slide=31.8us, gemm=56.3us, dequant=19.3us  total≈107.4us + get_ext=1.9us + view=2.7us
""")
    
    print("\n本次测试数据:")
    for phase_name, phase in [("Decode", results["decode"]), ("Prefill", results["prefill"])]:
        print(f"\n{phase_name}:")
        cutlass = phase["cutlass"]
        cublaslt = phase["cublaslt"]
        cusparselt = phase["cusparselt"]
        
        print(f"  CUTLASS:     quant={cutlass['quant']*1000:.1f}us, scaled_mm={cutlass['scaled_mm']*1000:.1f}us  total={cutlass['total']*1000:.1f}us")
        print(f"  cuBLASLt:    quant={cublaslt['quant']*1000:.1f}us, gemm={cublaslt['gemm']*1000:.1f}us, dequant={cublaslt['dequant']*1000:.1f}us  total={cublaslt['total']*1000:.1f}us")
        print(f"  cuSPARSELt:  quant_slide={cusparselt['quant_slide']*1000:.1f}us, gemm={cusparselt['gemm']*1000:.1f}us, dequant={cusparselt['dequant']*1000:.1f}us  total={cusparselt['total']*1000:.1f}us")


if __name__ == "__main__":
    main()
