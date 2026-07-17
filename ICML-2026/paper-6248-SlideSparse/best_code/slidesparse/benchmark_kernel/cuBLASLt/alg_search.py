#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
cuBLASLt Dense GEMM 算法搜索

架构说明：
=========
- Python 端：负责外层 NK 循环、参数解析、GPU 检测、数据生成、结果落盘
- C++ 端：负责内层 M 循环、算法枚举、cuBLASLt API 调用、精确计时

支持的数据类型（输入/输出相同）:
==============================
- FP16:    FP16 输入, FP32 计算, FP16 输出
- BF16:    BF16 输入, FP32 计算, BF16 输出
- INT8:    INT8 输入, INT32 计算, INT8 输出
- FP8E4M3: FP8 输入, FP32 计算, FP8 输出
- FP4E2M1: FP4 输入, FP32 计算, FP4 输出

固定 Layout:
- T/N + Col/Col + Col (权重 W 在左)
- R[N,M]_col = W[K,N]^T_col @ A[K,M]_col

运行示例:
    # Model-based 模式
    python3 alg_search.py --dtype int8 --model Qwen2.5-0.5B
    python3 alg_search.py --dtype fp8e4m3 --model Llama3.2-1B
    
    # Square Matrix 模式 (不指定 --model)
    python3 alg_search.py --dtype bf16 --N 4096 --K 4096 --M-quick
"""

import argparse
import base64
import ctypes
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# 添加路径
SCRIPT_DIR = Path(__file__).parent.absolute()
BENCHMARK_KERNEL_DIR = SCRIPT_DIR.parent
SLIDESPARSE_DIR = BENCHMARK_KERNEL_DIR.parent
PROJECT_ROOT = SLIDESPARSE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from slidesparse.benchmark_kernel.utils import (
    # 常量
    DTYPE_CONFIG,
    SUPPORTED_DTYPES,
    ALIGNMENT,
    DEFAULT_M_LIST,
    M_QUICK_LIST,
    SQUARE_M_LIST,
    # 硬件检测
    hw_info,
    check_dtype_support,
    get_supported_dtypes_for_gpu,
    # 数据准备
    quantize_tensor,
    get_output_torch_dtype,
    pad_to_alignment,
    # NK 列表
    get_nk_list_for_benchmark,
    # 编译与加载
    build_benchmark_extension,
    load_benchmark_extension,
    # 文件命名与目录
    build_hw_folder_name,
    build_dtype_folder_name,
    build_output_dir,
    build_result_filename,
    build_csv_header_lines,
    # 增量保存
    IncrementalResultSaver,
)


# =============================================================================
# CUDA 扩展加载
# =============================================================================

def setup_lib_signatures(lib: ctypes.CDLL) -> None:
    """设置 CUDA 扩展的函数签名"""
    lib.cublaslt_search_single_m.argtypes = [
        ctypes.c_void_p,   # W_ptr
        ctypes.c_void_p,   # A_ptr
        ctypes.c_void_p,   # R_ptr
        ctypes.c_int64,    # N
        ctypes.c_int64,    # K
        ctypes.c_int64,    # M
        ctypes.c_char_p,   # dtype
        ctypes.c_int,      # warmup
        ctypes.c_int,      # repeat
        ctypes.c_int,      # topk
        ctypes.POINTER(ctypes.c_int),        # out_alg_ids
        ctypes.POINTER(ctypes.c_float),      # out_lat_us
        ctypes.POINTER(ctypes.c_float),      # out_tops
        ctypes.POINTER(ctypes.c_int64),      # out_workspace
        ctypes.POINTER(ctypes.c_float),      # out_waves_count
        ctypes.POINTER(ctypes.c_uint8),      # out_algo_data
        ctypes.POINTER(ctypes.c_uint8),      # out_valid
        ctypes.POINTER(ctypes.c_int),        # out_num_valid
        ctypes.POINTER(ctypes.c_int),        # out_alg_count
        ctypes.c_void_p,   # stream
    ]
    lib.cublaslt_search_single_m.restype = ctypes.c_int
    
    lib.cublaslt_alg_search_is_available.argtypes = []
    lib.cublaslt_alg_search_is_available.restype = ctypes.c_int
    
    lib.cublaslt_alg_search_get_last_error.argtypes = []
    lib.cublaslt_alg_search_get_last_error.restype = ctypes.c_char_p
    
    lib.cublaslt_alg_search_get_alignment.argtypes = [ctypes.c_char_p]
    lib.cublaslt_alg_search_get_alignment.restype = ctypes.c_int


# =============================================================================
# 搜索核心
# =============================================================================

def search_single_nk(
    lib: ctypes.CDLL,
    N: int, K: int, M: int,
    W_q_col: torch.Tensor,
    A_q_col: torch.Tensor,
    dtype: str,
    warmup: int,
    repeat: int,
    topk: int = 3,
) -> Dict[str, Any]:
    """
    搜索单个 (N, K, M) 组合的最佳算法
    
    添加了异常捕获，防止 ctypes 调用崩溃导致整个进程终止。
    """
    try:
        # 分配输出缓冲：直接按列主序 stride 分配
        # FP4 输出为 BF16（根据 cuBLASLt Table 4）
        R_torch_dtype = get_output_torch_dtype(dtype, backend="cublaslt")
        R_out = torch.empty_strided((N, M), (1, N), dtype=R_torch_dtype, device=W_q_col.device)
        R_out.zero_()
    except torch.cuda.OutOfMemoryError as e:
        return {
            "results": [],
            "num_valid": 0,
            "alg_count": 0,
            "error": f"CUDA OOM allocating output buffer: {e}",
        }
    except Exception as e:
        return {
            "results": [],
            "num_valid": 0,
            "alg_count": 0,
            "error": f"Error allocating output buffer: {e}",
        }
    
    # 分配输出数组
    out_alg_ids = (ctypes.c_int * topk)()
    out_lat_us = (ctypes.c_float * topk)()
    out_tops = (ctypes.c_float * topk)()
    out_workspace = (ctypes.c_int64 * topk)()
    out_waves_count = (ctypes.c_float * topk)()
    out_algo_data = (ctypes.c_uint8 * (topk * 64))()
    out_valid = (ctypes.c_uint8 * topk)()
    out_num_valid = ctypes.c_int(0)
    out_alg_count = ctypes.c_int(0)
    
    # 调用 C 函数（带异常捕获）
    try:
        ret = lib.cublaslt_search_single_m(
            W_q_col.data_ptr(),
            A_q_col.data_ptr(),
            R_out.data_ptr(),
            N, K, M,
            dtype.encode(),
            warmup,
            repeat,
            topk,
            out_alg_ids,
            out_lat_us,
            out_tops,
            out_workspace,
            out_waves_count,
            out_algo_data,
            out_valid,
            ctypes.byref(out_num_valid),
            ctypes.byref(out_alg_count),
            None,  # 使用默认 stream
        )
    except OSError as e:
        return {
            "results": [],
            "num_valid": 0,
            "alg_count": 0,
            "error": f"CUDA call OSError: {e}",
        }
    except Exception as e:
        return {
            "results": [],
            "num_valid": 0,
            "alg_count": 0,
            "error": f"CUDA call exception: {type(e).__name__} - {e}",
        }
    
    if ret != 0:
        error = lib.cublaslt_alg_search_get_last_error()
        error_msg = error.decode() if error else 'unknown error'
        return {
            "results": [],
            "num_valid": 0,
            "alg_count": 0,
            "error": error_msg,
        }
    
    # 转换结果
    results = []
    for i in range(topk):
        if out_valid[i]:
            algo_bytes = bytes(out_algo_data[i*64:(i+1)*64])
            results.append({
                "alg_id": out_alg_ids[i],
                "lat_us": out_lat_us[i],
                "tops": out_tops[i],
                "workspace": out_workspace[i],
                "waves_count": out_waves_count[i],
                "algo_data": algo_bytes,
            })
    
    return {
        "results": results,
        "num_valid": out_num_valid.value,
        "alg_count": out_alg_count.value,
    }


def run_search(
    lib: ctypes.CDLL,
    dtype: str,
    nk_list: List[Tuple[int, int]],
    m_list: List[int],
    warmup: int,
    repeat: int,
    topk: int = 3,
    verbose: bool = True,
    is_square_mode: bool = False,
    incremental_saver: Optional[IncrementalResultSaver] = None,
) -> Dict:
    """
    运行完整的算法搜索
    
    Args:
        is_square_mode: 如果为 True，只测试 M=N=K 的组合
                       （此时 nk_list 应该是 [(m,m) for m in m_list]）
        incremental_saver: 增量保存器（可选），用于每完成一个 NK 就保存进度
    """
    results = []
    total_nk = len(nk_list)
    max_alg_count = 0
    
    # 搜索统计
    search_stats = {"total": 0, "success": 0, "failed": 0, "errors": 0}
    
    # FP4 是打包格式（每字节 2 个值），需要特殊处理
    is_fp4 = dtype.lower() in ("fp4e2m1", "fp4")
    
    for nk_id, (N, K) in enumerate(nk_list):
        if verbose:
            print(f"    NK {nk_id+1}/{total_nk}: ({N}, {K})", flush=True)
        
        # Square 模式下只测试 M=N=K，否则测试所有 M
        effective_m_list = [N] if is_square_mode else m_list
        max_M = max(effective_m_list)
        
        nk_results = {
            "nk_id": nk_id,
            "N": N,
            "K": K,  # 报告原始 K，CUDA 端知道实际存储是 K/2
            "m_results": {},
            "skipped": False,
            "skip_reason": None,
        }
        
        try:
            # 生成随机数据
            W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
            A = torch.randn(max_M, K, device="cuda", dtype=torch.bfloat16)
        except torch.cuda.OutOfMemoryError as e:
            if verbose:
                print(f"      ⚠ CUDA OOM 生成数据，跳过 NK=({N}, {K})")
            nk_results["skipped"] = True
            nk_results["skip_reason"] = f"CUDA OOM: {e}"
            for M in effective_m_list:
                search_stats["total"] += 1
                search_stats["errors"] += 1
                nk_results["m_results"][M] = {"error": str(e), "results": [], "num_valid": 0, "alg_count": 0}
            results.append(nk_results)
            # 增量保存
            if incremental_saver:
                incremental_saver.add_nk_result(nk_results)
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            if verbose:
                print(f"      ⚠ 生成数据失败: {e}")
            nk_results["skipped"] = True
            nk_results["skip_reason"] = f"Data generation error: {e}"
            for M in effective_m_list:
                search_stats["total"] += 1
                search_stats["errors"] += 1
                nk_results["m_results"][M] = {"error": str(e), "results": [], "num_valid": 0, "alg_count": 0}
            results.append(nk_results)
            if incremental_saver:
                incremental_saver.add_nk_result(nk_results)
            torch.cuda.empty_cache()
            continue
        
        try:
            # 量化 (行主序)
            if dtype in ("fp16", "bf16"):
                # FP16/BF16 直接转换
                W_q = W.to(DTYPE_CONFIG[dtype]["torch_dtype"])
                A_q = A.to(DTYPE_CONFIG[dtype]["torch_dtype"])
            elif is_fp4:
                # FP4 打包：K 维度减半
                W_q = quantize_tensor(W, dtype)  # [N, K//2] packed
                A_q = quantize_tensor(A, dtype)  # [M, K//2] packed
            else:
                W_q = quantize_tensor(W, dtype)
                A_q = quantize_tensor(A, dtype)
        except Exception as e:
            if verbose:
                print(f"      ⚠ 量化失败: {e}")
            nk_results["skipped"] = True
            nk_results["skip_reason"] = f"Quantize error: {e}"
            for M in effective_m_list:
                search_stats["total"] += 1
                search_stats["errors"] += 1
                nk_results["m_results"][M] = {"error": str(e), "results": [], "num_valid": 0, "alg_count": 0}
            results.append(nk_results)
            if incremental_saver:
                incremental_saver.add_nk_result(nk_results)
            del W, A
            torch.cuda.empty_cache()
            continue
        
        # 转置为列主序供 cuBLASLt 使用
        # 对于 FP4，维度已经是打包后的
        W_q_col = W_q.t()  # [K_packed, N], stride (1, K_packed) 列主序
        A_q_col = A_q.t()  # [K_packed, Mmax], stride (1, K_packed) 列主序
        
        # FP4 打包后 K 维度减半，需要传递实际的 K_effective 给 CUDA
        # 但 TOPS 计算应该用原始的 K（逻辑维度）
        K_effective = K // 2 if is_fp4 else K
        K_logical = K  # 保留原始 K 用于 TOPS 计算
        
        for M in effective_m_list:
            # 列主序切片供 CUDA 使用
            A_slice_col = A_q_col[:, :M]
            
            out = search_single_nk(
                lib, N, K_effective, M,  # FP4 传 K/2，其他传 K
                W_q_col, A_slice_col,
                dtype,
                warmup, repeat, topk,
            )
            
            # FP4: 用原始 K 重新计算 TOPS（因为 CUDA 端用的是 K_effective）
            if is_fp4 and out.get("results"):
                for r in out["results"]:
                    if r.get("lat_us", 0) > 0:
                        ops = 2.0 * M * N * K_logical
                        r["tops"] = ops / (r["lat_us"] / 1e6) / 1e12
            
            nk_results["m_results"][M] = out
            
            if out["alg_count"] > max_alg_count:
                max_alg_count = out["alg_count"]
            
            # 更新搜索统计
            search_stats["total"] += 1
            if out.get("error"):
                search_stats["errors"] += 1
            elif out["num_valid"] > 0:
                search_stats["success"] += 1
            else:
                search_stats["failed"] += 1
        
        if verbose:
            first_m = effective_m_list[0]
            first_result = nk_results["m_results"][first_m]
            print(f"      → 算法数: {first_result['alg_count']}, 有效: {first_result['num_valid']}")
        
        results.append(nk_results)
        
        # 增量保存
        if incremental_saver:
            incremental_saver.add_nk_result(nk_results)
        
        # 释放
        del W, A, W_q, A_q
        torch.cuda.empty_cache()
    
    # 打印搜索统计汇总
    if verbose:
        print()
        print(f"    搜索统计: 总计={search_stats['total']}, "
              f"成功={search_stats['success']}, "
              f"失败={search_stats['failed']}, "
              f"错误={search_stats['errors']}")
        if search_stats["total"] > 0:
            success_rate = 100.0 * search_stats["success"] / search_stats["total"]
            print(f"    成功率: {success_rate:.1f}%")
    
    return {
        "dtype": dtype,
        "results": results,
        "M_list": m_list,
        "NK_list": nk_list,
        "max_alg_count": max_alg_count,
        "search_stats": search_stats,
    }


# =============================================================================
# 结果保存（使用增量保存器）
# =============================================================================

def save_results(
    out_dir: Path,
    model_name: str,
    dtype: str,
    mode: str,
    search_ret: Dict,
    warmup: int,
    repeat: int,
) -> Path:
    """
    保存搜索结果到 CSV 和 JSON 文件（使用增量保存器以支持原子写入）
    
    注意：此函数现在仅作为兼容接口。推荐在 run_search 时直接使用 IncrementalResultSaver。
    """
    # 创建增量保存器
    saver = IncrementalResultSaver(
        out_dir=out_dir,
        model_name=model_name,
        dtype=dtype,
        backend="cuBLASLt",
        mode=mode,
        warmup=warmup,
        repeat=repeat,
        m_list=search_ret["M_list"],
        nk_list=search_ret["NK_list"],
    )
    
    # 添加所有结果
    for nk_res in search_ret["results"]:
        saver.add_nk_result(nk_res, save_progress=False)
    
    # 最终保存
    saver.finalize()
    
    return saver.get_output_dir()


# =============================================================================
# 主流程
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="cuBLASLt Dense GEMM 算法搜索",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # Model-based 模式
  python3 alg_search.py --dtype int8 --model Qwen2.5-0.5B
  python3 alg_search.py --dtype fp8e4m3 --model Llama3.2-1B
  
  # Model-based 模式 + Lmax（生成 L=4,6,...,Lmax 的所有 slided NK）
  python3 alg_search.py --dtype fp8e4m3 --model Qwen2.5-0.5B --Lmax 8
  
  # Square 模式 (不指定 --model 或 --model square)
  python3 alg_search.py --dtype bf16
  python3 alg_search.py --dtype bf16 --model square
        """
    )
    p.add_argument("--dtype", required=True, choices=SUPPORTED_DTYPES,
                   help="数据类型 (fp16, bf16, int8, fp8e4m3, fp4e2m1)")
    p.add_argument("--model", default=None,
                   help="模型名称（不指定或指定 'square' 进入 Square 模式）")
    p.add_argument("--Lmax", type=int, default=None,
                   help="最大 L 值，用于 slide sparse。如果指定，会生成 L=4,6,...,Lmax 的所有 NK")
    p.add_argument("--M-quick", action="store_true", dest="m_quick",
                   help="使用快速 M 列表 (仅 Model-based 模式有效)")
    p.add_argument("--m_list", type=str, default=None,
                   help="M 列表，逗号分隔 (Square 模式: M=N=K 使用此列表)")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--repeat", type=int, default=100)
    p.add_argument("--compile", action="store_true",
                   help="强制重新编译 CUDA 扩展")
    p.add_argument("--out_dir", default=None,
                   help="输出目录")
    return p.parse_args()


def main():
    args = parse_args()
    
    if not torch.cuda.is_available():
        print("[ERROR] 需要 CUDA 环境")
        sys.exit(1)
    
    # 检查 dtype 支持
    supported, reason = check_dtype_support(args.dtype)
    if not supported:
        print(f"[SKIP] dtype={args.dtype}: {reason}")
        print(f"[INFO] 当前 GPU 支持的类型: {get_supported_dtypes_for_gpu()}")
        sys.exit(0)  # 跳过
    
    # 确定运行模式
    is_square_mode = (args.model is None or args.model.lower() == "square")
    
    # 确定 M 列表
    if args.m_list:
        m_list = [int(x.strip()) for x in args.m_list.split(",")]
    elif args.m_quick:
        m_list = list(M_QUICK_LIST)
    else:
        m_list = list(M_QUICK_LIST)  # 默认使用 M_QUICK_LIST
    
    # 确保 M 是 32 的倍数
    m_list = [pad_to_alignment(m) for m in m_list]
    
    # 获取 NK 列表（传入 L_max 支持 slide sparse）
    try:
        nk_list, model_name, mode = get_nk_list_for_benchmark(
            model=args.model,
            L_max=args.Lmax,  # 使用新的 Lmax 参数
            m_list=m_list if is_square_mode else None
        )
    except Exception as e:
        print(f"[ERROR] 获取 NK 列表失败: {e}")
        sys.exit(1)
        
    # 显示配置
    print("=" * 60)
    print("cuBLASLt Dense GEMM 算法搜索")
    print("=" * 60)
    print(f"GPU: {hw_info.gpu_full_name} ({hw_info.cc_tag})")
    print(f"Mode: {mode.upper()}")
    print(f"Model: {model_name}")
    if args.Lmax:
        print(f"Lmax: {args.Lmax} (slide sparse L=4,6,...,{args.Lmax})")
    print(f"dtype: {args.dtype} -> {args.dtype} (same input/output)")
    print(f"warmup={args.warmup}, repeat={args.repeat}")
    print()
    
    # 输出目录
    out_dir = Path(args.out_dir) if args.out_dir else SCRIPT_DIR / "alg_search_results"
    
    # 编译 CUDA 扩展
    print("[1/4] 编译 CUDA 扩展...")
    src_path = SCRIPT_DIR / "cublaslt_gemm.cu"
    build_dir = SCRIPT_DIR / "build"
    try:
        so_path = build_benchmark_extension(
            name="cublaslt_gemm",
            source_file=src_path,
            build_dir=build_dir,
            backend="cublaslt",
            force=args.compile,
        )
    except Exception as e:
        print(f"[ERROR] CUDA 扩展编译失败: {e}")
        sys.exit(1)
    
    print("[2/4] 加载 CUDA 扩展...")
    try:
        lib = load_benchmark_extension(so_path, backend="cublaslt", setup_func=setup_lib_signatures)
    except Exception as e:
        print(f"[ERROR] CUDA 扩展加载失败: {e}")
        sys.exit(1)
    
    if not lib.cublaslt_alg_search_is_available():
        print("[SKIP] cuBLASLt 不可用（可能需要更新驱动或 CUDA 版本）")
        sys.exit(0)  # 跳过
    print("✓ cuBLASLt 可用")
    
    print()
    print(f"[3/4] 开始算法搜索...")
    if is_square_mode:
        print(f"      M=N=K 列表: {m_list}")
    else:
        print(f"      NK 组合: {len(nk_list)} 个, M 列表: {m_list}")
    print()
    
    # 创建增量保存器（每完成一个 NK 就保存进度）
    incremental_saver = IncrementalResultSaver(
        out_dir=out_dir,
        model_name=model_name,
        dtype=args.dtype,
        backend="cuBLASLt",
        mode=mode,
        warmup=args.warmup,
        repeat=args.repeat,
        m_list=m_list,
        nk_list=nk_list,
    )
    
    ret = run_search(
        lib,
        args.dtype,
        nk_list,
        m_list,
        args.warmup,
        args.repeat,
        topk=3,
        verbose=True,
        is_square_mode=is_square_mode,
        incremental_saver=incremental_saver,
    )
    
    print()
    print("[4/4] 保存结果...")
    # 使用增量保存器的 finalize 方法进行原子写入
    csv_path, json_path = incremental_saver.finalize()
    saved_dir = incremental_saver.get_output_dir()
    
    print()
    print(f"✓ 完成! 结果已保存到: {saved_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
