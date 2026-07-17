#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
cuSPARSELt Sparse GEMM 算法搜索 (2:4 稀疏)

架构说明：
=========
- Python 端：负责外层 NK 循环、参数解析、GPU 检测、数据生成、结果落盘
- C++ 端：负责内层 M 循环、算法枚举、cuSPARSELt API 调用、精确计时

支持的数据类型（输入/输出相同）:
==============================
- FP16:    FP16 输入, FP32 计算, FP16 输出
- BF16:    BF16 输入, FP32 计算, BF16 输出
- INT8:    INT8 输入, INT32 计算, INT8 输出
- FP8E4M3: FP8 输入, FP32 计算, FP8 输出
- FP4E2M1: FP4 输入, FP32 计算, FP4 输出

固定 Layout:
- T/N + Col/Col + Col (权重 W 在左, 2:4 稀疏)
- R[N,M]_col = W_compressed[K,N]^T_col @ A[K,M]_col

搜索策略:
- 自适应 Split-K 倍增策略 (1, 2, 4, 8, ...)
- Segment-K 测试 (SM90+ 支持 split_k=-1)
- 官方 API 搜索对比 (cusparseLtMatmulSearch)

Sparsity 说明:
=============
对于 K_slide 测试，用户需要传入已经计算好的 K_slide 作为 K 参数。
例如：sparsity=2_8 时，K_slide = 1.5 * K_original

运行示例:
    # Model-based 模式
    python3 alg_search.py --dtype int8 --model Qwen2.5-0.5B
    python3 alg_search.py --dtype fp8e4m3 --model Llama3.2-1B
    
    # Square Matrix 模式 (不指定 --model)
    python3 alg_search.py --dtype bf16 --N 4096 --K 4096 --M-quick
    
    # 指定 sparsity (会自动计算 K_slide)
    python3 alg_search.py --dtype fp8e4m3 --N 4096 --K 4096 --sparsity 2_8
"""

import argparse
import datetime
import json
import ctypes
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
    DEFAULT_SPARSITY_LIST,
    ALIGNMENT,
    DEFAULT_M_LIST,
    M_QUICK_LIST,
    SQUARE_M_LIST,
    # 硬件检测
    hw_info,
    check_dtype_support,
    get_supported_dtypes_for_gpu,
    check_cusparselt_support,
    check_segment_k_support,
    # Sparsity 计算
    calculate_k_slide,
    get_k_expansion_factor,
    pad_to_alignment,
    # 数据准备
    quantize_int8,
    to_fp8_e4m3,
    to_fp4_e2m1_packed,
    get_output_torch_dtype,
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
    lib.cusparselt_search_single_m.argtypes = [
        ctypes.c_void_p,   # W_pruned_ptr
        ctypes.c_void_p,   # A_ptr
        ctypes.c_void_p,   # R_ptr
        ctypes.c_int64,    # N
        ctypes.c_int64,    # K
        ctypes.c_int64,    # M
        ctypes.c_char_p,   # dtype
        ctypes.c_int,      # warmup
        ctypes.c_int,      # repeat
        ctypes.c_int,      # topk
        ctypes.c_int,      # test_segment_k
        ctypes.c_int,      # do_api_search
        # 输出
        ctypes.POINTER(ctypes.c_int),        # out_alg_ids
        ctypes.POINTER(ctypes.c_int),        # out_split_k
        ctypes.POINTER(ctypes.c_float),      # out_lat_us
        ctypes.POINTER(ctypes.c_float),      # out_tops
        ctypes.POINTER(ctypes.c_int64),      # out_workspace
        ctypes.POINTER(ctypes.c_uint8),      # out_valid
        ctypes.POINTER(ctypes.c_int),        # out_num_valid
        ctypes.POINTER(ctypes.c_int),        # out_alg_count
        ctypes.POINTER(ctypes.c_int),        # out_config_count
        ctypes.POINTER(ctypes.c_int),        # out_api_alg_id
        ctypes.POINTER(ctypes.c_int),        # out_api_split_k
        ctypes.POINTER(ctypes.c_float),      # out_api_lat_us
        ctypes.POINTER(ctypes.c_int),        # out_api_rank
        ctypes.c_void_p,   # stream
    ]
    lib.cusparselt_search_single_m.restype = ctypes.c_int
    
    lib.cusparselt_prune_24.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_char_p,
        ctypes.c_void_p,
    ]
    lib.cusparselt_prune_24.restype = ctypes.c_int
    
    lib.cusparselt_compress.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_char_p,
        ctypes.c_void_p,
    ]
    lib.cusparselt_compress.restype = ctypes.c_int64
    
    lib.cusparselt_get_compressed_size.argtypes = [
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_char_p,
    ]
    lib.cusparselt_get_compressed_size.restype = ctypes.c_int64
    
    lib.cusparselt_supports_segment_k.argtypes = []
    lib.cusparselt_supports_segment_k.restype = ctypes.c_int
    
    lib.cusparselt_alg_search_is_available.argtypes = []
    lib.cusparselt_alg_search_is_available.restype = ctypes.c_int
    
    lib.cusparselt_alg_search_get_last_error.argtypes = []
    lib.cusparselt_alg_search_get_last_error.restype = ctypes.c_char_p


# =============================================================================
# 数据准备
# =============================================================================

def prepare_and_prune_weight(
    lib: ctypes.CDLL,
    W_bf16: torch.Tensor,
    dtype: str,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[str]]:
    """
    准备并剪枝权重矩阵
    
    对于 FP4 (E2M1)：
    - cuSPARSELt 期望 packed format（每字节 2 个 FP4 值）
    - 但维度参数使用逻辑维度（未打包的 K）
    - cuSPARSELt 会正确处理 paired 4:8 稀疏约束
    
    Returns:
        (W_pruned, W_q, error_msg): 
            W_pruned: 剪枝后的矩阵 [K_packed, N] (列主序)，K_packed = K//2 for FP4
            W_q: 原始量化权重 [N, K] or [N, K//2] for FP4
            error_msg: 错误信息，成功时为 None
    """
    try:
        N, K = W_bf16.shape  # 逻辑维度
        dtype_lower = dtype.lower()
        is_fp4 = dtype_lower in ("fp4e2m1", "fp4")
        
        # 量化
        if dtype_lower == "int8":
            W_q, _ = quantize_int8(W_bf16)
        elif dtype_lower in ("fp8e4m3", "fp8"):
            W_q = to_fp8_e4m3(W_bf16)
        elif dtype_lower == "fp16":
            W_q = W_bf16.to(torch.float16)
        elif dtype_lower == "bf16":
            W_q = W_bf16
        elif is_fp4:
            # FP4 打包：K 维度减半（每字节 2 个 FP4 值）
            W_q = to_fp4_e2m1_packed(W_bf16)  # [N, K//2]
        else:
            return None, None, f"Unsupported dtype: {dtype}"
        
        # 转置为列主序存储
        # 对于 FP4，W_q 形状是 [N, K//2]，转置后 [K//2, N]
        W_t = W_q.t()
        
        # Prune 2:4 (或 paired 4:8 for FP4)
        W_pruned = torch.empty_like(W_t)
        
        # ========================================================================
        # 关键：cuSPARSELt 期望逻辑维度，而不是打包后的维度
        # ========================================================================
        # 对于 FP4：传递逻辑 K（未打包），cuSPARSELt 内部会正确处理 packed format
        # 对于其他类型：正常传递 K
        # ========================================================================
        ret = lib.cusparselt_prune_24(
            W_t.data_ptr(),
            W_pruned.data_ptr(),
            K, N,  # 始终传递逻辑维度 K（不是 K//2）
            dtype.encode(),
            None,
        )
        if ret != 0:
            error = lib.cusparselt_alg_search_get_last_error()
            error_msg = error.decode() if error else 'unknown error'
            return None, None, f"Prune failed: {error_msg}"
        
        torch.cuda.synchronize()
        
        return W_pruned, W_q, None
        
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return None, None, f"CUDA OOM during prune: {e}"
    except Exception as e:
        return None, None, f"Exception during prune: {e}"


def prepare_activation(
    A_bf16: torch.Tensor,
    dtype: str,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[str]]:
    """
    准备激活矩阵
    
    对于 FP4 (E2M1)：
    - cuSPARSELt 期望 packed format（每字节 2 个 FP4 值）
    - 但维度参数使用逻辑维度（未打包的 K）
    
    Returns:
        (A_transposed, A_q, error_msg):
            A_transposed: 转置后的激活 [K_packed, M] or [K, M] (用于 CUDA)
            A_q: 原始量化激活 [M, K] or [M, K//2] for FP4
            error_msg: 错误信息，成功时为 None
    """
    try:
        dtype_lower = dtype.lower()
        is_fp4 = dtype_lower in ("fp4e2m1", "fp4")
        
        if dtype_lower == "int8":
            A_q, _ = quantize_int8(A_bf16)
        elif dtype_lower in ("fp8e4m3", "fp8"):
            A_q = to_fp8_e4m3(A_bf16)
        elif dtype_lower == "fp16":
            A_q = A_bf16.to(torch.float16)
        elif dtype_lower == "bf16":
            A_q = A_bf16
        elif is_fp4:
            # FP4 打包：K 维度减半（每字节 2 个 FP4 值）
            A_q = to_fp4_e2m1_packed(A_bf16)  # [M, K//2]
        else:
            return None, None, f"Unsupported dtype: {dtype}"
        
        # 转置为列主序
        # 对于 FP4，A_q 形状是 [M, K//2]，转置后 [K//2, M]
        A_transposed = A_q.t()
        
        return A_transposed, A_q, None
        
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return None, None, f"CUDA OOM during activation prep: {e}"
    except Exception as e:
        return None, None, f"Exception during activation prep: {e}"


# =============================================================================
# 搜索核心
# =============================================================================

def search_single_nk(
    lib: ctypes.CDLL,
    N: int, K: int, M: int,
    W_pruned: torch.Tensor,
    A_transposed: torch.Tensor,
    dtype: str,
    warmup: int,
    repeat: int,
    topk: int = 3,
    test_segment_k: bool = True,
    do_api_search: bool = True,
) -> Dict[str, Any]:
    """搜索单个 (N, K, M) 组合的最佳算法
    
    添加了异常捕获，防止 ctypes 调用崩溃导致整个进程终止。
    """
    try:
        # 分配输出缓冲
        # 注意：INT8 输出 BF16，FP8/FP4 输出 BF16
        R_torch_dtype = get_output_torch_dtype(dtype, backend="cusparselt")
        R_out = torch.empty_strided((N, M), (1, N), dtype=R_torch_dtype, device=A_transposed.device)
        R_out.zero_()
    except torch.cuda.OutOfMemoryError as e:
        return {
            "results": [],
            "num_valid": 0,
            "alg_count": 0,
            "config_count": 0,
            "api_result": None,
            "error": f"CUDA OOM allocating output buffer: {e}",
        }
    except Exception as e:
        return {
            "results": [],
            "num_valid": 0,
            "alg_count": 0,
            "config_count": 0,
            "api_result": None,
            "error": f"Error allocating output buffer: {e}",
        }
    
    # 分配输出数组
    out_alg_ids = (ctypes.c_int * topk)()
    out_split_k = (ctypes.c_int * topk)()
    out_lat_us = (ctypes.c_float * topk)()
    out_tops = (ctypes.c_float * topk)()
    out_workspace = (ctypes.c_int64 * topk)()
    out_valid = (ctypes.c_uint8 * topk)()
    out_num_valid = ctypes.c_int(0)
    out_alg_count = ctypes.c_int(0)
    out_config_count = ctypes.c_int(0)
    out_api_alg_id = ctypes.c_int(-1)
    out_api_split_k = ctypes.c_int(1)
    out_api_lat_us = ctypes.c_float(0.0)
    out_api_rank = ctypes.c_int(-1)
    
    # 调用 C 函数（带异常捕获）
    try:
        ret = lib.cusparselt_search_single_m(
            W_pruned.data_ptr(),
            A_transposed.data_ptr(),
            R_out.data_ptr(),
            N, K, M,
            dtype.encode(),
            warmup,
            repeat,
            topk,
            1 if test_segment_k else 0,
            1 if do_api_search else 0,
            out_alg_ids,
            out_split_k,
            out_lat_us,
            out_tops,
            out_workspace,
            out_valid,
            ctypes.byref(out_num_valid),
            ctypes.byref(out_alg_count),
            ctypes.byref(out_config_count),
            ctypes.byref(out_api_alg_id),
            ctypes.byref(out_api_split_k),
            ctypes.byref(out_api_lat_us),
            ctypes.byref(out_api_rank),
            None,
        )
    except OSError as e:
        return {
            "results": [],
            "num_valid": 0,
            "alg_count": 0,
            "config_count": 0,
            "api_result": None,
            "error": f"CUDA call OSError: {e}",
        }
    except Exception as e:
        return {
            "results": [],
            "num_valid": 0,
            "alg_count": 0,
            "config_count": 0,
            "api_result": None,
            "error": f"CUDA call exception: {type(e).__name__} - {e}",
        }
    
    if ret != 0:
        error = lib.cusparselt_alg_search_get_last_error()
        return {
            "results": [],
            "num_valid": 0,
            "alg_count": 0,
            "config_count": 0,
            "api_result": None,
            "error": error.decode() if error else "unknown error",
        }
    
    # 转换结果
    results = []
    for i in range(topk):
        if out_valid[i]:
            results.append({
                "alg_id": out_alg_ids[i],
                "split_k": out_split_k[i],
                "lat_us": out_lat_us[i],
                "tops": out_tops[i],
                "workspace": out_workspace[i],
            })
    
    return {
        "results": results,
        "num_valid": out_num_valid.value,
        "alg_count": out_alg_count.value,
        "config_count": out_config_count.value,
        "api_result": {
            "alg_id": out_api_alg_id.value,
            "split_k": out_api_split_k.value,
            "lat_us": out_api_lat_us.value,
            "rank": out_api_rank.value,
        } if out_api_alg_id.value >= 0 else None,
    }


def run_search(
    lib: ctypes.CDLL,
    dtype: str,
    nk_list: List[Tuple[int, int]],
    m_list: List[int],
    warmup: int,
    repeat: int,
    topk: int = 3,
    test_segment_k: bool = True,
    do_api_search: bool = True,
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
    max_config_count = 0
    supports_segment_k_hw = bool(lib.cusparselt_supports_segment_k())
    
    # 搜索统计
    search_stats = {"total": 0, "success": 0, "failed": 0, "errors": 0}
    
    for nk_id, (N, K) in enumerate(nk_list):
        if verbose:
            print(f"    NK {nk_id+1}/{total_nk}: ({N}, {K})", flush=True)
        
        # Square 模式下只测试 M=N=K，否则测试所有 M
        effective_m_list = [N] if is_square_mode else m_list
        max_M = max(effective_m_list)
        
        nk_results = {
            "nk_id": nk_id,
            "N": N,
            "K": K,
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
        
        # 剪枝权重
        W_pruned, W_q, prune_error = prepare_and_prune_weight(lib, W, dtype)
        if prune_error:
            if verbose:
                print(f"      ⚠ 权重剪枝失败: {prune_error}")
            nk_results["skipped"] = True
            nk_results["skip_reason"] = prune_error
            for M in effective_m_list:
                search_stats["total"] += 1
                search_stats["errors"] += 1
                nk_results["m_results"][M] = {"error": prune_error, "results": [], "num_valid": 0, "alg_count": 0}
            results.append(nk_results)
            if incremental_saver:
                incremental_saver.add_nk_result(nk_results)
            del W, A
            torch.cuda.empty_cache()
            continue
        
        # 准备激活
        A_transposed, A_q, act_error = prepare_activation(A, dtype)
        if act_error:
            if verbose:
                print(f"      ⚠ 激活准备失败: {act_error}")
            nk_results["skipped"] = True
            nk_results["skip_reason"] = act_error
            for M in effective_m_list:
                search_stats["total"] += 1
                search_stats["errors"] += 1
                nk_results["m_results"][M] = {"error": act_error, "results": [], "num_valid": 0, "alg_count": 0}
            results.append(nk_results)
            if incremental_saver:
                incremental_saver.add_nk_result(nk_results)
            del W, A, W_pruned, W_q
            torch.cuda.empty_cache()
            continue
        
        # ====================================================================
        # 关键：cuSPARSELt 期望逻辑维度，即使数据是 packed format
        # ====================================================================
        # 对于 FP4：
        # - 数据已经是 packed（每字节 2 个 FP4 值）
        # - 但传给 cuSPARSELt 的维度参数是逻辑维度 K（不是 K//2）
        # - cuSPARSELt 内部会正确处理 packed format 和 paired 4:8 稀疏
        # ====================================================================
        
        for M in effective_m_list:
            # 切片
            A_slice = A_transposed[:, :M]
            
            # 传递逻辑维度 K（不是 K//2）给 cuSPARSELt
            out = search_single_nk(
                lib, N, K, M,  # 始终使用逻辑维度 K
                W_pruned, A_slice,
                dtype,
                warmup, repeat, topk,
                test_segment_k=test_segment_k and supports_segment_k_hw,
                do_api_search=do_api_search,
            )
            
            nk_results["m_results"][M] = out
            
            if out["alg_count"] > max_alg_count:
                max_alg_count = out["alg_count"]
            if out.get("config_count", 0) > max_config_count:
                max_config_count = out["config_count"]
            
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
        del W, A, W_pruned, A_transposed, W_q, A_q
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
        "max_config_count": max_config_count,
        "supports_segment_k": supports_segment_k_hw,
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
    sparsity: Optional[str] = None,
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
        backend="cuSPARSELt",
        mode=mode,
        warmup=warmup,
        repeat=repeat,
        m_list=search_ret["M_list"],
        nk_list=search_ret["NK_list"],
        sparsity=sparsity,
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
        description="cuSPARSELt Sparse GEMM 算法搜索 (2:4 稀疏)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # Model-based 模式
  python3 alg_search.py --dtype int8 --model Qwen2.5-0.5B
  python3 alg_search.py --dtype fp8e4m3 --model Llama3.2-1B --sparsity 2_8
  
  # Model-based 模式 + Lmax（生成 L=4,6,...,Lmax 的所有 slided NK）
  python3 alg_search.py --dtype fp8e4m3 --model Qwen2.5-0.5B --Lmax 8
  
  # Square 模式 (不指定 --model 或 --model square)
  python3 alg_search.py --dtype bf16
  python3 alg_search.py --dtype bf16 --model square --sparsity 2_6
        """
    )
    p.add_argument("--dtype", required=True, choices=SUPPORTED_DTYPES,
                   help="数据类型 (fp16, bf16, int8, fp8e4m3, fp4e2m1)")
    p.add_argument("--model", default=None,
                   help="模型名称（不指定或指定 'square' 进入 Square 模式）")
    p.add_argument("--Lmax", type=int, default=None,
                   help="最大 L 值，用于 slide sparse。如果指定，会生成 L=4,6,...,Lmax 的所有 NK")
    p.add_argument("--sparsity", type=str, default=None,
                   help="稀疏度配置 (如 2_4, 2_6, 2_8)，会自动计算 K_slide")
    p.add_argument("--M-quick", action="store_true", dest="m_quick",
                   help="使用快速 M 列表 (仅 Model-based 模式有效)")
    p.add_argument("--m_list", type=str, default=None,
                   help="M 列表，逗号分隔 (Square 模式: M=N=K 使用此列表)")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--repeat", type=int, default=100)
    p.add_argument("--compile", action="store_true",
                   help="强制重新编译 CUDA 扩展")
    p.add_argument("--no_segment_k", action="store_true",
                   help="禁用 Segment-K 测试")
    p.add_argument("--no_api_search", action="store_true",
                   help="禁用官方 API 搜索对比")
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
    
    # 检查 cuSPARSELt 支持
    cusparselt_ok, cusparselt_reason = check_cusparselt_support()
    if not cusparselt_ok:
        print(f"[SKIP] cuSPARSELt: {cusparselt_reason}")
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
    
    # 如果指定了 sparsity，计算 K_slide
    sparsity = args.sparsity
    if sparsity:
        k_factor = get_k_expansion_factor(sparsity)
        # 传入 dtype 以便 FP4 使用 64 对齐
        nk_list = [(n, calculate_k_slide(k, sparsity, dtype=args.dtype)) for n, k in nk_list]
        print(f"[INFO] Sparsity={sparsity}, K_factor={k_factor:.3f}")
        print(f"[INFO] NK list adjusted: {nk_list[:5]}..." if len(nk_list) > 5 else f"[INFO] NK list adjusted: {nk_list}")
    
    test_segment_k = not args.no_segment_k
    do_api_search = not args.no_api_search
    
    segment_k_ok, segment_k_reason = check_segment_k_support()
    
    # 显示配置
    print("=" * 60)
    print("cuSPARSELt Sparse GEMM 算法搜索 (2:4 稀疏)")
    print("=" * 60)
    print(f"GPU: {hw_info.gpu_full_name} ({hw_info.cc_tag})")
    print(f"Mode: {mode.upper()}")
    print(f"Model: {model_name}")
    if args.Lmax:
        print(f"Lmax: {args.Lmax} (slide sparse L=4,6,...,{args.Lmax})")
    print(f"dtype: {args.dtype} -> {args.dtype} (same input/output)")
    if sparsity:
        print(f"Sparsity: {sparsity}")
    print(f"Segment-K 测试: {'开启' if test_segment_k and segment_k_ok else '关闭'}")
    print(f"API 搜索对比: {'开启' if do_api_search else '关闭'}")
    print(f"warmup={args.warmup}, repeat={args.repeat}")
    print()
    
    # 输出目录
    out_dir = Path(args.out_dir) if args.out_dir else SCRIPT_DIR / "alg_search_results"
    
    # 编译 CUDA 扩展
    print("[1/4] 编译 CUDA 扩展...")
    src_path = SCRIPT_DIR / "cusparselt_gemm.cu"
    build_dir = SCRIPT_DIR / "build"
    try:
        so_path = build_benchmark_extension(
            name="cusparselt_gemm",
            source_file=src_path,
            build_dir=build_dir,
            backend="cusparselt",
            force=args.compile,
        )
    except Exception as e:
        print(f"[ERROR] CUDA 扩展编译失败: {e}")
        sys.exit(1)
    
    print("[2/4] 加载 CUDA 扩展...")
    try:
        lib = load_benchmark_extension(so_path, backend="cusparselt", setup_func=setup_lib_signatures)
    except Exception as e:
        print(f"[ERROR] CUDA 扩展加载失败: {e}")
        sys.exit(1)
    
    if not lib.cusparselt_alg_search_is_available():
        print("[SKIP] cuSPARSELt 不可用（可能需要更新驱动或 CUDA 版本）")
        sys.exit(0)  # 优雅跳过
    print("✓ cuSPARSELt 可用")
    
    supports_segment_k_hw = bool(lib.cusparselt_supports_segment_k())
    print(f"✓ Segment-K 支持: {'是' if supports_segment_k_hw else '否'}")
    
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
        backend="cuSPARSELt",
        mode=mode,
        warmup=args.warmup,
        repeat=args.repeat,
        m_list=m_list,
        nk_list=nk_list,
        sparsity=sparsity,
    )
    
    ret = run_search(
        lib,
        args.dtype,
        nk_list,
        m_list,
        args.warmup,
        args.repeat,
        topk=3,
        test_segment_k=test_segment_k,
        do_api_search=do_api_search,
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
