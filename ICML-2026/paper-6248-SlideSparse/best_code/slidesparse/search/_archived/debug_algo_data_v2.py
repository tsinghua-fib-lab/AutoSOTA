#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试脚本 v2：深入分析 cuBLASLt algo_data 不兼容问题

这个脚本专门测试：
1. MatmulDesc 配置差异（epilogue 设置）
2. 不同 M 值是否影响算法兼容性
3. 模拟 vLLM 的实际调用路径
"""

import base64
import ctypes
import json
import sys
from pathlib import Path

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
    """设置搜索扩展的函数签名"""
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
    """设置 GEMM 扩展的函数签名"""
    lib.cublaslt_gemm_get_last_error.argtypes = []
    lib.cublaslt_gemm_get_last_error.restype = ctypes.c_char_p
    
    gemm_sig = ([ctypes.c_void_p] * 3 + 
               [ctypes.c_int64] * 3 + 
               [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p])
    lib.cublaslt_fp8_mm.argtypes = gemm_sig
    lib.cublaslt_fp8_mm.restype = ctypes.c_int
    
    lib.cublaslt_is_available.argtypes = []
    lib.cublaslt_is_available.restype = ctypes.c_int


def search_and_get_algo(search_lib, W_q, A_q, N, K, M, dtype, outdtype):
    """执行搜索并返回 algo_data"""
    R_torch_dtype = get_output_torch_dtype(outdtype)
    R_out = torch.zeros(M, N, dtype=R_torch_dtype, device=W_q.device)
    
    topk = 3
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
        N, K, M,
        dtype.encode(), outdtype.encode(),
        5, 20, topk,
        out_alg_ids, out_lat_us, out_tops, out_workspace,
        out_waves_count, out_algo_data, out_valid,
        ctypes.byref(out_num_valid), ctypes.byref(out_alg_count),
        None,
    )
    
    if ret != 0:
        error = search_lib.cublaslt_alg_search_get_last_error()
        raise RuntimeError(f"搜索失败: {error.decode() if error else 'unknown error'}")
    
    if out_valid[0]:
        algo_bytes = bytes(out_algo_data[0:64])
        return {
            "alg_id": out_alg_ids[0],
            "algo_data": algo_bytes,
            "workspace": out_workspace[0],
        }
    return None


def call_gemm_with_algo(gemm_lib, W_q, A_q, N, K, M, inner_dtype, algo_data, workspace):
    """使用指定的 algo_data 执行 GEMM"""
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
        raise RuntimeError(f"cublaslt_fp8_mm failed: {err.decode() if err else 'Unknown'}")
    
    return output


def main():
    print("=" * 70)
    print("cuBLASLt algo_data 深入调试 v2")
    print("=" * 70)
    print(f"GPU: {hw_info.gpu_full_name} ({hw_info.cc_tag})")
    print()
    
    dtype = "fp8e4m3"
    outdtype = "bf16"
    
    # === 加载扩展 ===
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
    if not so_files:
        print("    [ERROR] GEMM 库不存在")
        return
    gemm_lib = ctypes.CDLL(str(so_files[0]))
    setup_lib_signatures_gemm(gemm_lib)
    print("    ✓ 扩展加载完成")
    
    # === 从 JSON 加载配置 ===
    print("\n[2] 加载 JSON 配置...")
    from slidesparse.utils import build_hw_dir_name
    hw_folder = build_hw_dir_name()
    json_path = (SCRIPT_DIR / "cuBLASLt_AlgSearch" / "alg_search_results" / 
                 hw_folder / "alg_search_Llama3.2-1B-FP8_out-BF16.json")
    
    if not json_path.exists():
        print(f"    [ERROR] JSON 不存在: {json_path}")
        return
    
    with open(json_path) as f:
        config = json.load(f)
    
    nk_list = config["meta"]["NK_list"]
    m_list = config["meta"]["M_list"]
    print(f"    NK 列表: {nk_list}")
    print(f"    M 列表: {m_list}")
    
    # === 测试所有 NK x M 组合 ===
    print("\n[3] 测试所有 NK x M 组合...")
    
    # 先生成足够大的测试数据
    max_M = max(m_list)
    max_K = max(k for n, k in nk_list)
    max_N = max(n for n, k in nk_list)
    
    W_full = torch.randn(max_N, max_K, device="cuda", dtype=torch.bfloat16)
    A_full = torch.randn(max_M, max_K, device="cuda", dtype=torch.bfloat16)
    W_q_full = quantize_tensor(W_full, dtype)
    A_q_full = quantize_tensor(A_full, dtype)
    
    success_count = 0
    fail_count = 0
    fail_cases = []
    
    for N, K in nk_list:
        W_q = W_q_full[:N, :K].contiguous()
        nk_key = f"({N},{K})"
        nk_entry = config["nk_entries"].get(nk_key)
        
        if not nk_entry:
            print(f"  {nk_key}: 无配置，跳过")
            continue
        
        for M in m_list:
            A_q = A_q_full[:M, :K].contiguous()
            m_key = str(M)
            algo_cfg = nk_entry["alg_by_m"].get(m_key)
            
            if not algo_cfg:
                continue
            
            algo_data = base64.b64decode(algo_cfg["algo_data"])
            workspace = algo_cfg.get("workspace", 0)
            
            # 测试 JSON 中的 algo_data
            try:
                out = call_gemm_with_algo(
                    gemm_lib, W_q, A_q, N, K, M, "bf16", algo_data, workspace
                )
                success_count += 1
                status = "✓"
            except Exception as e:
                fail_count += 1
                fail_cases.append((N, K, M, str(e)))
                status = "✗"
            
            print(f"  {nk_key} M={M}: {status}")
    
    print()
    print(f"结果: {success_count} 成功, {fail_count} 失败")
    
    if fail_cases:
        print("\n失败案例:")
        for N, K, M, err in fail_cases:
            print(f"  ({N},{K}) M={M}: {err[:100]}...")
    
    # === 测试：搜索不同的 M 并立即使用 ===
    print("\n" + "=" * 70)
    print("[4] 测试：不同 M 值搜索后立即使用")
    print("=" * 70)
    
    # 固定 NK，变化 M
    N, K = 3072, 2048
    W_q = W_q_full[:N, :K].contiguous()
    
    test_ms = [16, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    
    for M in test_ms:
        if M > max_M:
            continue
        A_q = A_q_full[:M, :K].contiguous()
        
        # 搜索
        algo_info = search_and_get_algo(search_lib, W_q, A_q, N, K, M, dtype, outdtype)
        
        if algo_info:
            # 立即使用搜索到的 algo
            try:
                out = call_gemm_with_algo(
                    gemm_lib, W_q, A_q, N, K, M, "bf16",
                    algo_info['algo_data'], algo_info['workspace']
                )
                print(f"  M={M:5d}: ✓ 搜索+执行成功 (alg_id={algo_info['alg_id']})")
            except Exception as e:
                print(f"  M={M:5d}: ✗ 搜索成功但执行失败: {e}")
        else:
            print(f"  M={M:5d}: ⚠ 搜索无有效算法")
    
    # === 测试：用 M1 搜索的算法执行 M2 ===
    print("\n" + "=" * 70)
    print("[5] 测试：用不同 M 搜索的算法交叉执行")
    print("=" * 70)
    
    # 用 M=1024 搜索
    M_search = 1024
    A_q_search = A_q_full[:M_search, :K].contiguous()
    algo_info = search_and_get_algo(search_lib, W_q, A_q_search, N, K, M_search, dtype, outdtype)
    
    if algo_info:
        print(f"使用 M={M_search} 搜索到的算法 (alg_id={algo_info['alg_id']})")
        
        for M_exec in [16, 128, 512, 1024, 2048, 4096, 8192]:
            if M_exec > max_M:
                continue
            A_q_exec = A_q_full[:M_exec, :K].contiguous()
            
            try:
                out = call_gemm_with_algo(
                    gemm_lib, W_q, A_q_exec, N, K, M_exec, "bf16",
                    algo_info['algo_data'], algo_info['workspace']
                )
                print(f"  执行 M={M_exec:5d}: ✓ 成功")
            except Exception as e:
                print(f"  执行 M={M_exec:5d}: ✗ 失败: {str(e)[:80]}")
    
    print("\n" + "=" * 70)
    print("调试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
