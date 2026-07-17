#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Quant + Slide Kernel Benchmark (Paper Version)

生成 Kernel 性能数据的基准测试脚本。

输出文件（放在 benchmark_result/{hw_dir}/ 目录）:
1. latency_fp8.csv       - FP8 延时绝对值 (us)
2. latency_int8.csv      - INT8 延时绝对值 (us)
3. latency_ratio_fp8.csv  - FP8 相对 baseline 的延时比 (kernel/baseline)
4. latency_ratio_int8.csv - INT8 相对 baseline 的延时比 (kernel/baseline)

CSV 格式:
- 第一列: M 值
- 第二列: baseline (memcpy kernel)
- 后续列: 2:4, 2:6, 2:8, 2:10, 2:12, 2:14, 2:16 等不同 L 值

Usage:
    python3 run_benchmark_paper.py                    # 使用默认 M 列表
    python3 run_benchmark_paper.py --m-fine           # 使用更细粒度的 M 列表
    python3 run_benchmark_paper.py --model Qwen2.5-7B # 指定模型
    python3 run_benchmark_paper.py --Lmax 12         # 最大 L 值为 12
    python3 run_benchmark_paper.py --K 2560           # 指定 K 值
    python3 run_benchmark_paper.py --dtype int8       # 只测试 INT8（A100 等不支持 FP8 E4M3 的 GPU）
    python3 run_benchmark_paper.py --dtype fp8        # 只测试 FP8
"""

import sys
import argparse
import csv
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Callable, Optional

import torch
import triton
import triton.language as tl
import triton.testing as testing

# 设置路径以导入 slidesparse 模块
_SCRIPT_DIR = Path(__file__).parent
_SLIDESPARSE_ROOT = _SCRIPT_DIR.parent.parent  # slidesparse/
_PROJECT_ROOT = _SLIDESPARSE_ROOT.parent       # vllmbench/

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from slidesparse.utils import (
    build_hw_dir_name,
    build_tuned_filename,
    get_gpu_cc,
    get_nk_list_for_search,
    get_unique_k_values,
    hw_info,
)


# =============================================================================
# Configuration
# =============================================================================

# M values: 默认列表（覆盖 decode 到 prefill 场景）
M_VALUES_DEFAULT = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

# M values: 更细粒度列表（用于论文图表）
M_VALUES_FINE = [
    1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512,
    768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384
]

# L values: 支持的稀疏配置
L_VALUES_ALL = [4, 6, 8, 10, 12, 14, 16]

# K value: 默认测试 K（BitNet-2B 的 K 维度）
K_DEFAULT = 2560

# Benchmark 参数
WARMUP = 25
REP = 100

# FP8 E4M3 (float8_e4m3fn) 需要 SM89+ (Ada Lovelace / Hopper)
# A100 是 SM80，不支持 float8_e4m3fn
FP8_E4M3_MIN_CC = 89


def check_fp8_support() -> bool:
    """
    检查当前 GPU 是否支持 FP8 E4M3 (float8_e4m3fn)
    
    Returns:
        True 如果支持，False 如果不支持
    """
    if not torch.cuda.is_available():
        return False
    
    cc = torch.cuda.get_device_capability()
    cc_int = cc[0] * 10 + cc[1]
    return cc_int >= FP8_E4M3_MIN_CC


# =============================================================================
# Baseline Kernel (Memory Copy)
# =============================================================================

@triton.jit
def _memcpy_slide_fp8_kernel(
    in_ptr, out_ptr, M, K_in, K_out,
    stride_im, stride_om,
    BLOCK_K: tl.constexpr,
):
    """Pure memory copy baseline with slide output ratio (FP8)"""
    row = tl.program_id(0)
    
    in_row_ptr = in_ptr + row * stride_im
    out_row_ptr = out_ptr + row * stride_om
    
    for k_start in range(0, K_out, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_out
        in_offs = offs_k % K_in
        val = tl.load(in_row_ptr + in_offs, mask=mask_k, other=0.0)
        tl.store(out_row_ptr + offs_k, val.to(tl.float8e4nv), mask=mask_k)


@triton.jit
def _memcpy_slide_int8_kernel(
    in_ptr, out_ptr, M, K_in, K_out,
    stride_im, stride_om,
    BLOCK_K: tl.constexpr,
):
    """Pure memory copy baseline with slide output ratio (INT8)"""
    row = tl.program_id(0)
    
    in_row_ptr = in_ptr + row * stride_im
    out_row_ptr = out_ptr + row * stride_om
    
    for k_start in range(0, K_out, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_out
        in_offs = offs_k % K_in
        val = tl.load(in_row_ptr + in_offs, mask=mask_k, other=0.0)
        val_f32 = val.to(tl.float32)
        val_clamped = tl.clamp(val_f32 * 100.0, -128.0, 127.0)
        tl.store(out_row_ptr + offs_k, val_clamped.to(tl.int8), mask=mask_k)


# Tensor cache for baseline
_baseline_fp8_cache: dict = {}
_baseline_int8_cache: dict = {}


def make_baseline_func(
    get_config_func: Callable,
    compute_output_k_func: Callable,
    dtype: str,
    L: int
) -> Callable:
    """
    创建 baseline 函数（纯内存拷贝 + slide 输出大小）
    
    与 tuned kernel 使用相同的 tensor cache 策略以确保公平比较。
    """
    
    def baseline_fp8(x: torch.Tensor) -> torch.Tensor:
        M, K_in = x.shape
        K_in_padded, K_out, _ = compute_output_k_func(K_in, L)
        K_out_padded = ((K_out + 31) // 32) * 32
        M_padded = ((M + 15) // 16) * 16
        
        key = (M_padded, K_out_padded, x.device.index or 0)
        if key not in _baseline_fp8_cache:
            _baseline_fp8_cache[key] = torch.empty(
                M_padded, K_out_padded, dtype=torch.float8_e4m3fn, device=x.device
            )
        output = _baseline_fp8_cache[key]
        output.zero_()
        
        config = get_config_func(M, K_in)
        if len(config) == 4:
            BLOCK_OUT, BLOCK_K, num_warps, num_stages = config
        else:
            BLOCK_OUT, num_warps, num_stages = config
            BLOCK_K = BLOCK_OUT * 8
        
        _memcpy_slide_fp8_kernel[(M,)](
            x, output, M, K_in, K_out,
            x.stride(0), K_out_padded,
            BLOCK_K=BLOCK_K,
            num_warps=num_warps, num_stages=num_stages,
        )
        return output
    
    def baseline_int8(x: torch.Tensor) -> torch.Tensor:
        M, K_in = x.shape
        K_in_padded, K_out, _ = compute_output_k_func(K_in, L)
        K_out_padded = ((K_out + 31) // 32) * 32
        M_padded = ((M + 15) // 16) * 16
        
        key = (M_padded, K_out_padded, x.device.index or 0)
        if key not in _baseline_int8_cache:
            _baseline_int8_cache[key] = torch.empty(
                M_padded, K_out_padded, dtype=torch.int8, device=x.device
            )
        output = _baseline_int8_cache[key]
        output.zero_()
        
        config = get_config_func(M, K_in)
        if len(config) == 4:
            BLOCK_OUT, BLOCK_K, num_warps, num_stages = config
        else:
            BLOCK_OUT, num_warps, num_stages = config
            BLOCK_K = BLOCK_OUT * 8
        
        _memcpy_slide_int8_kernel[(M,)](
            x, output, M, K_in, K_out,
            x.stride(0), K_out_padded,
            BLOCK_K=BLOCK_K,
            num_warps=num_warps, num_stages=num_stages,
        )
        return output
    
    return baseline_fp8 if dtype == "fp8" else baseline_int8


# =============================================================================
# Module Loading
# =============================================================================

def load_tuned_module(model_name: Optional[str] = None) -> Optional[ModuleType]:
    """Load the auto-tuned kernel module"""
    build_dir = Path(__file__).parent / "build" / build_hw_dir_name()
    filename = build_tuned_filename("quant_slide_tuned", model_name, ext=".py")
    module_path = build_dir / filename
    
    if not module_path.exists():
        print(f"[Warning] Tuned kernel not found: {module_path}")
        return None
    
    try:
        spec = importlib.util.spec_from_file_location("tuned_kernel", module_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"[Warning] Failed to load tuned kernel: {e}")
        return None


def load_basic_module() -> Optional[ModuleType]:
    """Load the basic kernel module"""
    module_path = Path(__file__).parent / "basic_quant_slide_triton.py"
    
    if not module_path.exists():
        print(f"[Error] Basic kernel not found: {module_path}")
        return None
    
    spec = importlib.util.spec_from_file_location("basic_kernel", module_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_single(
    func: Callable,
    x: torch.Tensor,
    L: int,
) -> float:
    """
    运行单个 benchmark 并返回延时（us）
    
    Returns:
        延时（微秒）
    """
    latency_ms: float = testing.do_bench(
        lambda: func(x, L),
        warmup=WARMUP, rep=REP, return_mode="min"
    )
    return latency_ms * 1000  # ms -> us


def benchmark_baseline(
    baseline_func: Callable,
    x: torch.Tensor,
) -> float:
    """
    运行 baseline benchmark 并返回延时（us）
    """
    latency_ms: float = testing.do_bench(
        lambda: baseline_func(x),
        warmup=WARMUP, rep=REP, return_mode="min"
    )
    return latency_ms * 1000  # ms -> us


def run_benchmark_for_dtype(
    dtype: str,
    tuned_func: Callable,
    get_config_func: Callable,
    compute_output_k_func: Callable,
    M_values: list[int],
    L_values: list[int],
    K: int,
) -> tuple[list[list[float]], list[list[float]]]:
    """
    为单个 dtype 运行完整 benchmark
    
    Returns:
        (latency_data, latency_ratio_data): 两个二维列表
        - latency_data[i] = [M, baseline, L4, L6, L8, ...]
        - latency_ratio_data[i] = [M, 1.0, ratio_L4, ratio_L6, ...] (kernel/baseline)
    """
    print(f"\n{'=' * 70}")
    print(f"Benchmarking {dtype.upper()} (K={K})")
    print(f"{'=' * 70}")
    print(f"M values: {len(M_values)} points")
    print(f"L values: {L_values}")
    print()
    
    latency_data: list[list[float]] = []
    latency_ratio_data: list[list[float]] = []
    
    # 选择一个参考 L 值来计算 baseline（使用最小的 L 值）
    L_ref = L_values[0]
    baseline_func = make_baseline_func(get_config_func, compute_output_k_func, dtype, L_ref)
    
    # 打印进度 header
    L_headers = " | ".join([f"2:{L:<3}" for L in L_values])
    print(f"{'M':<8} | {'Baseline':<10} | {L_headers}")
    print("-" * (20 + len(L_values) * 8))
    
    for M in M_values:
        # 创建输入数据
        x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        
        # Benchmark baseline
        t_baseline = benchmark_baseline(baseline_func, x)
        
        # Benchmark 各个 L 值
        t_L_values: list[float] = []
        for L in L_values:
            t_L = benchmark_single(tuned_func, x, L)
            t_L_values.append(t_L)
        
        # 计算延时比（kernel latency / baseline）
        # 注意：由于 kernel 做了更多计算，延时比通常 > 1
        latency_ratio_values = [t / t_baseline for t in t_L_values]
        
        # 保存数据
        latency_data.append([M, t_baseline] + t_L_values)
        latency_ratio_data.append([M, 1.0] + latency_ratio_values)
        
        # 打印进度
        L_latencies = " | ".join([f"{t:<7.2f}" for t in t_L_values])
        print(f"{M:<8} | {t_baseline:<10.2f} | {L_latencies}")
        
        # 清理内存
        del x
        torch.cuda.empty_cache()
    
    return latency_data, latency_ratio_data


def save_csv(
    filepath: Path,
    data: list[list[float]],
    L_values: list[int],
):
    """
    保存数据到 CSV 文件
    
    Header: M, baseline, 2:4, 2:6, 2:8, ...
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # 构建 header
    header = ["M", "baseline"] + [f"2:{L}" for L in L_values]
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in data:
            # 格式化数据：M 保持整数，其他保留 4 位小数
            formatted_row = [int(row[0])] + [f"{v:.4f}" for v in row[1:]]
            writer.writerow(formatted_row)
    
    print(f"  Saved: {filepath}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Quant + Slide Kernel Benchmark (Paper Version)"
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help='Model name (e.g., Qwen2.5-7B-FP8). Default: BitNet-2B-BF16'
    )
    parser.add_argument(
        '--K', type=int, default=None,
        help=f'K dimension to benchmark. Default: auto-detect from model or {K_DEFAULT}'
    )
    parser.add_argument(
        '--Lmax', type=int, default=16,
        help='Maximum L value (default: 16)'
    )
    parser.add_argument(
        '--m-fine', action='store_true',
        help='Use fine-grained M values for paper figures'
    )
    parser.add_argument(
        '--m-list', type=str, default=None,
        help='Custom M values (comma-separated, e.g., "16,128,1024,4096")'
    )
    parser.add_argument(
        '--dtype', type=str, default='auto',
        choices=['auto', 'all', 'fp8', 'int8'],
        help='Data type to benchmark: auto (detect GPU capability), all, fp8, or int8. '
             'A100 and older GPUs only support int8. (default: auto)'
    )
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("[Error] CUDA not available")
        return 1
    
    # 选择 M 值列表
    if args.m_list:
        M_values = [int(m.strip()) for m in args.m_list.split(',')]
    elif args.m_fine:
        M_values = M_VALUES_FINE
    else:
        M_values = M_VALUES_DEFAULT
    
    # 确定 L 值范围
    L_max = args.Lmax
    L_values = [L for L in L_VALUES_ALL if L <= L_max]
    
    # 获取模型的 K 值
    if args.K:
        K = args.K
        model_name = args.model
    else:
        try:
            nk_list, model_name = get_nk_list_for_search(args.model, L_max)
            k_values = get_unique_k_values(nk_list)
            K = k_values[0] if k_values else K_DEFAULT
        except Exception:
            K = K_DEFAULT
            model_name = args.model
    
    # 确定要测试的数据类型
    fp8_supported = check_fp8_support()
    
    if args.dtype == 'auto':
        run_fp8 = fp8_supported
        run_int8 = True
    elif args.dtype == 'all':
        run_fp8 = True
        run_int8 = True
    elif args.dtype == 'fp8':
        run_fp8 = True
        run_int8 = False
    else:  # int8
        run_fp8 = False
        run_int8 = True
    
    # 打印配置信息
    hw_dir = build_hw_dir_name()
    output_dir = _SCRIPT_DIR / "benchmark_result" / hw_dir
    
    print("=" * 70)
    print("Quant + Slide Kernel Benchmark (Paper Version)")
    print("=" * 70)
    print(f"GPU:         {torch.cuda.get_device_name()} ({get_gpu_cc()})")
    print(f"PyTorch:     {torch.__version__}")
    print(f"Triton:      {triton.__version__}")
    print(f"Model:       {model_name or 'Default (BitNet-2B-BF16)'}")
    print(f"K:           {K}")
    print(f"L range:     {L_values}")
    print(f"M values:    {len(M_values)} points ({M_values[0]} ~ {M_values[-1]})")
    print(f"Output dir:  {output_dir}")
    
    # 打印 FP8 支持状态
    if fp8_supported:
        print(f"FP8 E4M3:    Supported (SM{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]})")
    else:
        print(f"FP8 E4M3:    NOT supported (SM{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]} < SM89)")
    
    dtypes_to_run = []
    if run_fp8:
        dtypes_to_run.append('FP8')
    if run_int8:
        dtypes_to_run.append('INT8')
    print(f"Benchmarks:  {', '.join(dtypes_to_run)}")
    print()
    
    # 加载 kernel 模块
    basic_module = load_basic_module()
    if basic_module is None:
        return 1
    
    tuned_module = load_tuned_module(model_name)
    if tuned_module is None:
        print("[Info] Using basic kernel (no tuned kernel found)")
        get_config_func = basic_module._get_config
        tuned_fp8_func = basic_module.quant_slide_fp8_triton
        tuned_int8_func = basic_module.quant_slide_int8_triton
    else:
        get_config_func = tuned_module._get_config
        tuned_fp8_func = tuned_module.quant_slide_fp8_triton
        tuned_int8_func = tuned_module.quant_slide_int8_triton
    
    compute_output_k_func = basic_module._compute_output_k
    
    # 初始化结果变量
    fp8_latency: list[list[float]] = []
    fp8_latency_ratio: list[list[float]] = []
    int8_latency: list[list[float]] = []
    int8_latency_ratio: list[list[float]] = []
    
    # 运行 FP8 benchmark
    if run_fp8:
        print("\n" + "=" * 70)
        print("FP8 Benchmark")
        print("=" * 70)
        
        try:
            fp8_latency, fp8_latency_ratio = run_benchmark_for_dtype(
                dtype="fp8",
                tuned_func=tuned_fp8_func,
                get_config_func=get_config_func,
                compute_output_k_func=compute_output_k_func,
                M_values=M_values,
                L_values=L_values,
                K=K,
            )
        except Exception as e:
            print(f"\n[Error] FP8 benchmark failed: {e}")
            print("[Info] Skipping FP8 benchmark. Your GPU may not support FP8 E4M3.")
            print("[Info] Use --dtype int8 to run only INT8 benchmarks.")
            fp8_latency = []
            fp8_latency_ratio = []
    else:
        print("\n[Info] Skipping FP8 benchmark (not supported or not requested)")
    
    # 运行 INT8 benchmark
    if run_int8:
        print("\n" + "=" * 70)
        print("INT8 Benchmark")
        print("=" * 70)
        
        try:
            int8_latency, int8_latency_ratio = run_benchmark_for_dtype(
                dtype="int8",
                tuned_func=tuned_int8_func,
                get_config_func=get_config_func,
                compute_output_k_func=compute_output_k_func,
                M_values=M_values,
                L_values=L_values,
                K=K,
            )
        except Exception as e:
            print(f"\n[Error] INT8 benchmark failed: {e}")
            int8_latency = []
            int8_latency_ratio = []
    else:
        print("\n[Info] Skipping INT8 benchmark (not requested)")
    
    # 保存 CSV 文件
    print("\n" + "=" * 70)
    print("Saving Results")
    print("=" * 70)
    
    if fp8_latency:
        save_csv(output_dir / "latency_fp8.csv", fp8_latency, L_values)
        save_csv(output_dir / "latency_ratio_fp8.csv", fp8_latency_ratio, L_values)
    else:
        print("  [Skipped] FP8 results (no data)")
    
    if int8_latency:
        save_csv(output_dir / "latency_int8.csv", int8_latency, L_values)
        save_csv(output_dir / "latency_ratio_int8.csv", int8_latency_ratio, L_values)
    else:
        print("  [Skipped] INT8 results (no data)")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
