#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试脚本 v6：针对具体失败的 NK 配置进行深入分析

根据之前的调试日志，失败的配置是：
- (N=2048, K=8192) 使用 M=4096 搜索的 algo_data 执行 M=8192 失败
- workspace=33562783 (约 32MB)

这个脚本专门分析这个配置。
"""

import base64
import ctypes
import json
import struct
import sys
from pathlib import Path
from typing import Optional, Dict

import torch

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import (
    hw_info,
    build_search_extension,
    load_search_extension,
    quantize_tensor,
    get_output_torch_dtype,
)


def setup_lib_signatures_search(lib: ctypes.CDLL) -> None:
    lib.cublaslt_search_single_m.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p,
    ]
    lib.cublaslt_search_single_m.restype = ctypes.c_int
    lib.cublaslt_alg_search_is_available.argtypes = []
    lib.cublaslt_alg_search_is_available.restype = ctypes.c_int
    lib.cublaslt_alg_search_get_last_error.argtypes = []
    lib.cublaslt_alg_search_get_last_error.restype = ctypes.c_char_p


def setup_lib_signatures_gemm(lib: ctypes.CDLL) -> None:
    lib.cublaslt_gemm_get_last_error.argtypes = []
    lib.cublaslt_gemm_get_last_error.restype = ctypes.c_char_p
    
    gemm_sig = ([ctypes.c_void_p] * 3 + 
               [ctypes.c_int64] * 3 + 
               [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p])
    lib.cublaslt_fp8_mm.argtypes = gemm_sig
    lib.cublaslt_fp8_mm.restype = ctypes.c_int


def search_all_algos(search_lib, W_q, A_q, N, K, M, dtype, outdtype):
    """搜索并返回所有有效算法"""
    R_torch_dtype = get_output_torch_dtype(outdtype)
    R_out = torch.zeros(M, N, dtype=R_torch_dtype, device=W_q.device)
    
    topk = 20
    out_alg_ids = (ctypes.c_int * topk)()
    out_lat_us = (ctypes.c_float * topk)()
    out_tops = (ctypes.c_float * topk)()
    out_workspace = (ctypes.c_int64 * topk)()
    out_waves_count = (ctypes.c_float * topk)()
    out_algo_data = (ctypes.c_uint8 * (topk * 64))()
    out_valid = (ctypes.c_uint8 * topk)()
    out_num_valid = ctypes.c_int(0)
    out_alg_count = ctypes.c_int(0)
    
    ret = search_lib.cublaslt_search_single_m(
        W_q.data_ptr(), A_q.data_ptr(), R_out.data_ptr(),
        N, K, M, dtype.encode(), outdtype.encode(),
        3, 10, topk,
        out_alg_ids, out_lat_us, out_tops, out_workspace,
        out_waves_count, out_algo_data, out_valid,
        ctypes.byref(out_num_valid), ctypes.byref(out_alg_count),
        None,
    )
    
    if ret != 0:
        error = search_lib.cublaslt_alg_search_get_last_error()
        raise RuntimeError(f"搜索失败: {error.decode() if error else 'unknown error'}")
    
    results = []
    for i in range(topk):
        if out_valid[i]:
            results.append({
                "alg_id": out_alg_ids[i],
                "algo_data": bytes(out_algo_data[i*64:(i+1)*64]),
                "workspace": out_workspace[i],
                "lat_us": out_lat_us[i],
            })
    return results, out_alg_count.value


def call_gemm(gemm_lib, W_q, A_q, N, K, M, inner_dtype, algo_data, workspace):
    out_dtype = torch.float32 if inner_dtype == "fp32" else torch.bfloat16
    output = torch.empty((M, N), dtype=out_dtype, device=W_q.device)
    
    algo_ptr = None
    if algo_data is not None and len(algo_data) == 64:
        algo_ptr = (ctypes.c_uint8 * 64).from_buffer_copy(algo_data)
    
    ret = gemm_lib.cublaslt_fp8_mm(
        W_q.data_ptr(), A_q.data_ptr(), output.data_ptr(),
        M, N, K,
        inner_dtype.encode(),
        ctypes.cast(algo_ptr, ctypes.c_void_p) if algo_ptr else None,
        workspace,
        torch.cuda.current_stream().cuda_stream
    )
    
    if ret != 0:
        err = gemm_lib.cublaslt_gemm_get_last_error()
        raise RuntimeError(f"failed: {err.decode() if err else 'Unknown'}")
    
    return output


def main():
    print("=" * 70)
    print("cuBLASLt 问题配置深入分析 v6")
    print("=" * 70)
    print(f"GPU: {hw_info.gpu_full_name} ({hw_info.cc_tag})")
    print()
    
    dtype = "fp8e4m3"
    outdtype = "bf16"
    
    # 加载扩展
    print("[1] 加载扩展...")
    search_src = SCRIPT_DIR / "cuBLASLt_AlgSearch" / "alg_search_cublaslt.cu"
    search_build = SCRIPT_DIR / "cuBLASLt_AlgSearch" / "build"
    search_so = build_search_extension(
        name="alg_search_cublaslt",
        source_file=search_src,
        build_dir=search_build,
        backend="cublaslt",
        force=False,
    )
    search_lib = load_search_extension(search_so, backend="cublaslt", setup_func=setup_lib_signatures_search)
    
    gemm_so = SCRIPT_DIR.parent / "csrc" / "cublaslt_gemm" / "build"
    so_files = list(gemm_so.glob("cublaslt_gemm*.so"))
    gemm_lib = ctypes.CDLL(str(so_files[0]))
    setup_lib_signatures_gemm(gemm_lib)
    print("    ✓ 扩展加载完成")
    
    # 加载 JSON 配置
    print("\n[2] 加载 JSON 配置...")
    from slidesparse.utils import build_hw_dir_name
    hw_folder = build_hw_dir_name()
    json_path = (SCRIPT_DIR / "cuBLASLt_AlgSearch" / "alg_search_results" / 
                 hw_folder / "alg_search_Llama3.2-1B-FP8_out-BF16.json")
    
    with open(json_path) as f:
        config = json.load(f)
    
    print(f"    JSON: {json_path}")
    
    # 检查所有失败可能的 NK 组合
    nk_list = config["meta"]["NK_list"]
    m_list = config["meta"]["M_list"]
    print(f"    NK_list: {nk_list}")
    print(f"    M_list: {m_list}")
    
    # 测试所有 NK 组合
    print("\n" + "=" * 70)
    print("[3] 测试所有 NK x M 组合的交叉兼容性")
    print("=" * 70)
    
    max_M = max(m_list)
    max_K = max(k for n, k in nk_list)
    max_N = max(n for n, k in nk_list)
    
    W_full = torch.randn(max_N, max_K, device="cuda", dtype=torch.bfloat16)
    A_full = torch.randn(max_M, max_K, device="cuda", dtype=torch.bfloat16)
    W_q_full = quantize_tensor(W_full, dtype)
    A_q_full = quantize_tensor(A_full, dtype)
    
    problematic_configs = []
    
    for N, K in nk_list:
        W_q = W_q_full[:N, :K].contiguous()
        nk_key = f"({N},{K})"
        nk_entry = config["nk_entries"].get(nk_key)
        
        if not nk_entry:
            print(f"\n{nk_key}: 无配置，跳过")
            continue
        
        print(f"\n{nk_key}:")
        
        # 获取所有 M 的配置
        m_configs = {}
        for m_str, algo_cfg in nk_entry["alg_by_m"].items():
            M = int(m_str)
            algo_data = base64.b64decode(algo_cfg["algo_data"])
            workspace = algo_cfg.get("workspace", 0)
            alg_id = struct.unpack("<I", algo_data[:4])[0]
            m_configs[M] = {
                "algo_data": algo_data,
                "workspace": workspace,
                "alg_id": alg_id,
            }
        
        print(f"  JSON 中的 M 配置: {sorted(m_configs.keys())}")
        print(f"  alg_ids: {[(m, c['alg_id']) for m, c in sorted(m_configs.items())]}")
        print(f"  workspaces: {[(m, c['workspace']) for m, c in sorted(m_configs.items())]}")
        
        # 搜索几个 M 值对比
        print(f"\n  实时搜索对比:")
        for M in [1024, 4096, 8192]:
            if M > max_M:
                continue
            A_q = A_q_full[:M, :K].contiguous()
            
            # 搜索
            algos, total = search_all_algos(search_lib, W_q, A_q, N, K, M, dtype, outdtype)
            if algos:
                best = algos[0]
                print(f"    M={M:5d}: 搜索 alg_id={best['alg_id']}, workspace={best['workspace']}")
        
        # 交叉执行测试
        print(f"\n  交叉执行测试:")
        for M_from in sorted(m_configs.keys()):
            cfg = m_configs[M_from]
            
            for M_to in [1024, 4096, 8192]:
                if M_to > max_M:
                    continue
                
                A_q = A_q_full[:M_to, :K].contiguous()
                
                try:
                    out = call_gemm(gemm_lib, W_q, A_q, N, K, M_to, "bf16",
                                   cfg['algo_data'], cfg['workspace'])
                    status = "✓"
                except Exception as e:
                    status = "✗"
                    problematic_configs.append({
                        "N": N, "K": K,
                        "M_from": M_from, "M_to": M_to,
                        "alg_id": cfg["alg_id"],
                        "workspace": cfg["workspace"],
                        "error": str(e)[:50],
                    })
                
                marker = " ←问题" if status == "✗" else ""
                if status == "✗" or M_from != M_to:
                    print(f"    用 M={M_from} 的配置执行 M={M_to}: {status}{marker}")
        
        # 默认启发式测试
        print(f"\n  默认启发式测试:")
        for M in [1024, 4096, 8192]:
            if M > max_M:
                continue
            A_q = A_q_full[:M, :K].contiguous()
            try:
                out = call_gemm(gemm_lib, W_q, A_q, N, K, M, "bf16", None, 0)
                status = "✓"
            except Exception as e:
                status = "✗"
            print(f"    M={M:5d}: {status}")
    
    # 总结
    print("\n" + "=" * 70)
    print("[4] 问题配置总结")
    print("=" * 70)
    
    if problematic_configs:
        print(f"\n发现 {len(problematic_configs)} 个问题配置：")
        for cfg in problematic_configs:
            print(f"  ({cfg['N']},{cfg['K']}) M_from={cfg['M_from']} -> M_to={cfg['M_to']}: "
                  f"alg_id={cfg['alg_id']}, workspace={cfg['workspace']}")
            print(f"    错误: {cfg['error']}")
    else:
        print("\n✓ 未发现问题配置！")
        print("\n可能的解释：")
        print("  1. 问题配置不在这个 JSON 中")
        print("  2. 问题只在 vLLM 完整流程中出现")
        print("  3. 问题与线程/进程隔离有关")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
