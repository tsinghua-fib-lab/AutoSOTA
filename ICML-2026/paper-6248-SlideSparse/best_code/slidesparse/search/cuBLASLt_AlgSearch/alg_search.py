#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
cuBLASLt 算法离线搜索

架构说明：
=========
- Python 端：负责外层 NK 循环、参数解析、GPU 检测、数据生成、结果落盘
- C++ 端：负责内层 M 循环、算法枚举、cuBLASLt API 调用、精确计时

固定 Layout:
- T/N + Col/Col + Col (权重 W 在左)
- W[N,K]^T_col * A[K,M]_col = C[N,M]_col

运行示例:
    python3 alg_search.py --dtype int8 --outdtype int32 --model Qwen2.5-0.5B-INT8
    python3 alg_search.py --dtype fp8e4m3 --outdtype bf16 --model Qwen2.5-0.5B-FP8
"""

import argparse
import ctypes
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch

# 添加 search 目录到路径
SCRIPT_DIR = Path(__file__).parent.absolute()
SEARCH_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SEARCH_DIR))

from utils import (
    # 硬件信息
    hw_info,
    normalize_dtype,
    # 编译与加载
    build_search_extension,
    load_search_extension,
    # 模型 NK 工具
    get_nk_list_for_search,
    # 数据准备
    quantize_tensor,
    get_output_torch_dtype,
    # 结果保存
    save_alg_search_results,
    # 验证
    verify_gemm_result,
    # dtype 检测和验证
    SUPPORTED_DTYPES,
    SUPPORTED_OUTDTYPES,
    validate_dtype_outdtype_combination,
    get_default_outdtype,
    # 默认配置
    default_m_list,
)


# =============================================================================
# CUDA 扩展加载
# =============================================================================

def setup_lib_signatures(lib: ctypes.CDLL) -> None:
    """设置 CUDA 扩展的函数签名。"""
    lib.cublaslt_search_single_m.argtypes = [
        ctypes.c_void_p,   # W_ptr
        ctypes.c_void_p,   # A_ptr
        ctypes.c_void_p,   # C_ptr
        ctypes.c_int64,    # N
        ctypes.c_int64,    # K
        ctypes.c_int64,    # M
        ctypes.c_char_p,   # dtype
        ctypes.c_char_p,   # outdtype
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
    outdtype: str,
    warmup: int,
    repeat: int,
    topk: int = 3,
    verify: bool = False,
    W_q_for_verify: Optional[torch.Tensor] = None,
    A_q_for_verify: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    搜索单个 (N, K, M) 组合的最佳算法。
    """
    # 分配输出缓冲：直接按列主序 stride 分配 [N, M]，避免行主序写入错位
    R_torch_dtype = get_output_torch_dtype(outdtype)
    R_out = torch.empty_strided((N, M), (1, N), dtype=R_torch_dtype, device=W_q_col.device)
    R_out.zero_()
    
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
    
    # 调用 C 函数
    ret = lib.cublaslt_search_single_m(
        W_q_col.data_ptr(),
        A_q_col.data_ptr(),
        R_out.data_ptr(),
        N, K, M,
        dtype.encode(),
        outdtype.encode(),
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
    
    if ret != 0:
        error = lib.cublaslt_alg_search_get_last_error()
        raise RuntimeError(f"搜索失败: {error.decode() if error else 'unknown error'}")
    
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
    
    # 验证正确性
    verify_result = None
    if verify and W_q_for_verify is not None and A_q_for_verify is not None:
        verify_result = verify_gemm_result(
            W_q=W_q_for_verify,
            A_q=A_q_for_verify,
            R_out=R_out,
            M=M,
            # R_out 已按列主序 stride 分配为 [N, M]，直接与参考对齐
            is_col_major=False,
        )
        if verify_result["critical"]:
            print(f"    [CRITICAL] M={M}: {verify_result['message']}")
        elif not verify_result["passed"]:
            print(f"    [WARN] M={M}: {verify_result['message']}")
    
    return {
        "results": results,
        "num_valid": out_num_valid.value,
        "alg_count": out_alg_count.value,
        "verify_result": verify_result,
    }


def run_search(
    lib: ctypes.CDLL,
    dtype: str,
    outdtype: str,
    nk_list: List,
    m_list: List[int],
    warmup: int,
    repeat: int,
    topk: int = 3,
    verify: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    运行完整的算法搜索。
    """
    results = []
    max_M = max(m_list)
    total_nk = len(nk_list)
    
    max_alg_count = 0
    
    # verify 统计
    verify_stats = {"total": 0, "passed": 0, "warned": 0, "critical": 0}
    
    for nk_id, (N, K) in enumerate(nk_list):
        if verbose:
            print(f"    NK {nk_id+1}/{total_nk}: ({N}, {K})", flush=True)
        
        # 生成随机数据
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        A = torch.randn(max_M, K, device="cuda", dtype=torch.bfloat16)
        
        # 量化 (行主序)。随后通过转置视图提供列主序给 cuBLASLt
        W_q = quantize_tensor(W, dtype)
        A_q = quantize_tensor(A, dtype)
        W_q_col = W_q.t()          # [K, N], stride (1, K) 列主序
        A_q_col = A_q.t()          # [K, Mmax], stride (1, K) 列主序
        
        nk_results = {
            "nk_id": nk_id,
            "N": N,
            "K": K,
            "m_results": {},
        }
        
        for M in m_list:
            # 列主序切片供 CUDA 使用；保持转置视图的列主序 stride
            A_slice_col = A_q_col[:, :M]
            # verify 用的行主序切片 (M, K)
            A_q_slice = A_q[:M].contiguous() if verify else None
            
            out = search_single_nk(
                lib, N, K, M,
                W_q_col, A_slice_col,
                dtype, outdtype,
                warmup, repeat, topk,
                verify=verify,
                W_q_for_verify=W_q if verify else None,
                A_q_for_verify=A_q_slice,
            )
            
            nk_results["m_results"][M] = out
            
            if out["alg_count"] > max_alg_count:
                max_alg_count = out["alg_count"]
            
            # 更新 verify 统计
            if verify and out.get("verify_result"):
                vr = out["verify_result"]
                verify_stats["total"] += 1
                if vr["critical"]:
                    verify_stats["critical"] += 1
                elif vr["passed"]:
                    verify_stats["passed"] += 1
                else:
                    verify_stats["warned"] += 1
        
        if verbose:
            first_m = m_list[0]
            first_result = nk_results["m_results"][first_m]
            print(f"      → 启发式返回: {first_result['alg_count']} 算法, 有效: {first_result['num_valid']}")
        
        results.append(nk_results)
        
        # 释放
        del W, A, W_q, A_q
    
    torch.cuda.empty_cache()
    
    # 打印 verify 汇总
    if verify and verbose:
        print()
        print(f"    验证统计: 总计={verify_stats['total']}, "
              f"通过={verify_stats['passed']}, "
              f"警告={verify_stats['warned']}, "
              f"严重错误={verify_stats['critical']}")
    
    return {
        "dtype": dtype,
        "outdtype": outdtype,
        "results": results,
        "M_list": m_list,
        "NK_list": nk_list,
        "max_alg_count": max_alg_count,
    }


# =============================================================================
# 主流程
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="cuBLASLt 算法离线搜索")
    p.add_argument("--dtype", default="int8", choices=SUPPORTED_DTYPES, help="输入数据类型")
    p.add_argument("--outdtype", default="bf16", choices=SUPPORTED_OUTDTYPES, help="输出数据类型")
    p.add_argument("--model", default=None, help="模型名称（如 Qwen2.5-0.5B-INT8）或路径，必须与 checkpoints/ 目录下的文件夹名匹配。不指定则使用 BitNet-2B-BF16 默认配置")
    p.add_argument("--Lmax", type=int, default=None, help="最大 L 值（slide sparse），会为 L=4,6,...,Lmax 生成所有 NK")
    p.add_argument("--M-quick", action="store_true", dest="m_quick", help="M-quick 模式: 使用固定 M 列表 [16, 128, 1024, 4096, 16384]")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--repeat", type=int, default=100)
    p.add_argument("--verify", action="store_true", help="开启正确性校验")
    p.add_argument("--compile", action="store_true", help="强制重新编译 CUDA 扩展")
    p.add_argument("--out_dir", default=None, help="输出目录")
    p.add_argument("--m_list", type=str, default=None, help="M 列表，逗号分隔，如 16,128,512,2048,16384")
    return p.parse_args()


def main():
    args = parse_args()
    
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 环境")
    
    # 检查 FP8 硬件支持（CC >= 8.9，Ada/Hopper+）
    if args.dtype in ("fp8e4m3", "fp8") and not hw_info.supports_fp8:
        raise RuntimeError(
            f"GPU {hw_info.gpu_name} ({hw_info.cc_tag}) 不支持原生 FP8 GEMM。\n"
            f"FP8 需要 CC >= 8.9 (Ada Lovelace / Hopper / Blackwell)，"
            f"当前 GPU 为 {hw_info.arch_name} (CC {hw_info.cc_major}.{hw_info.cc_minor})。\n"
            f"请使用 --dtype int8 或在支持 FP8 的 GPU 上运行。"
        )
    
    # 验证并获取实际使用的 outdtype
    # cuBLASLt INT8 只支持 int32 输出，不支持 bf16/fp32
    actual_outdtype = validate_dtype_outdtype_combination(
        args.dtype, args.outdtype, backend="cublaslt"
    )
    
    # 获取 NK 列表和模型名称（统一使用 get_nk_list_for_search）
    nk_list, model_name = get_nk_list_for_search(args.model, args.Lmax)
    
    # === 显示配置信息 ===
    print("=" * 60)
    print("cuBLASLt 算法离线搜索")
    print("=" * 60)
    print(f"GPU: {hw_info.gpu_full_name} ({hw_info.cc_tag}, {hw_info.arch_name})")
    print(f"模型: {model_name}")
    print(f"参数: dtype={args.dtype}, outdtype={actual_outdtype}, warmup={args.warmup}, repeat={args.repeat}")
    print()
    
    # 输出目录 (脚本所在目录下)
    out_dir = Path(args.out_dir) if args.out_dir else SCRIPT_DIR / "alg_search_results"
    
    # 编译 CUDA 扩展
    print("[1/4] 编译 CUDA 扩展...")
    src_path = SCRIPT_DIR / "alg_search_cublaslt.cu"
    build_dir = SCRIPT_DIR / "build"
    so_path = build_search_extension(
        name="alg_search_cublaslt",
        source_file=src_path,
        build_dir=build_dir,
        backend="cublaslt",
        force=args.compile,
    )
    
    print("[2/4] 加载 CUDA 扩展...")
    lib = load_search_extension(so_path, backend="cublaslt", setup_func=setup_lib_signatures)
    
    if not lib.cublaslt_alg_search_is_available():
        raise RuntimeError("cuBLASLt 不可用")
    print("✓ cuBLASLt 可用")
    
    if args.Lmax:
        print(f"Lmax: {args.Lmax} (slide sparse L=4,6,...,{args.Lmax})")
    
    # 获取 M 列表
    if args.m_quick:
        m_list = [16, 128, 1024, 4096, 16384]
    elif args.m_list:
        m_list = [int(x.strip()) for x in args.m_list.split(",")]
    else:
        m_list = default_m_list()
    
    print()
    print(f"[3/4] 开始算法搜索...")
    print(f"      NK 组合: {len(nk_list)} 个, M 列表: {m_list}")
    print()
    
    ret = run_search(
        lib,
        args.dtype,
        actual_outdtype,  # 使用实际的 outdtype
        nk_list,
        m_list,
        args.warmup,
        args.repeat,
        topk=3,
        verify=args.verify,
        verbose=True,
    )
    
    saved_dir = save_alg_search_results(
        out_dir,
        model_name,
        args.dtype,
        actual_outdtype,  # 使用实际的 outdtype
        ret,
        args.warmup,
        args.repeat,
        args.verify,
        layout="TNCCcol",
        is_sparse=False,
        has_split_k=False,
    )
    
    print()
    print(f"[4/4] 完成! 结果已保存到:")
    print(f"      - {saved_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
