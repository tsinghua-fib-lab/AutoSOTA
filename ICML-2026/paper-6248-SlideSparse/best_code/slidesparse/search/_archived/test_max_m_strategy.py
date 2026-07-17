#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证"向上取整"策略的完整性

从交叉矩阵可以看出：
1. ws>0 的配置：只能执行 M <= M_from 的情况（向下兼容）
2. ws=0 的配置：可以执行任意 M（完全兼容）

新假设验证：
- 对于任意 M_x，应该使用 M_upper = max{M in M_list}（最大的 M）的配置
- 因为最大 M 通常是 ws=0，可以兼容所有 M

这个脚本验证：
1. ws=0 配置是否总是出现在大 M 时
2. 使用最大 M 配置是否可以覆盖所有情况
3. 模拟离线-在线场景
"""

import ctypes
import struct
import sys
import os
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


def search_best_algo(search_lib, W_q, A_q, N, K, M, dtype, outdtype):
    R_torch_dtype = get_output_torch_dtype(outdtype)
    R_out = torch.zeros(M, N, dtype=R_torch_dtype, device=W_q.device)
    
    topk = 5
    out_alg_ids = (ctypes.c_int * topk)()
    out_lat_us = (ctypes.c_float * topk)()
    out_workspace = (ctypes.c_int64 * topk)()
    out_algo_data = (ctypes.c_uint8 * (topk * 64))()
    out_valid = (ctypes.c_uint8 * topk)()
    out_num_valid = ctypes.c_int(0)
    out_alg_count = ctypes.c_int(0)
    dummy = (ctypes.c_float * topk)()
    
    ret = search_lib.cublaslt_search_single_m(
        W_q.data_ptr(), A_q.data_ptr(), R_out.data_ptr(),
        N, K, M, dtype.encode(), outdtype.encode(),
        2, 5, topk,
        out_alg_ids, out_lat_us, dummy, out_workspace,
        dummy, out_algo_data, out_valid,
        ctypes.byref(out_num_valid), ctypes.byref(out_alg_count),
        None,
    )
    
    if ret != 0 or not out_valid[0]:
        return None
    
    algo_data = bytes(out_algo_data[0:64])
    u32 = struct.unpack("<16I", algo_data)
    
    return {
        "alg_id": out_alg_ids[0],
        "algo_data": algo_data,
        "workspace": out_workspace[0],
        "split_k": u32[3],
        "tile": u32[1],
    }


def call_gemm(gemm_lib, W_q, A_q, N, K, M, algo_data, workspace):
    output = torch.empty((M, N), dtype=torch.bfloat16, device=W_q.device)
    
    algo_ptr = None
    if algo_data is not None:
        algo_ptr = (ctypes.c_uint8 * 64).from_buffer_copy(algo_data)
    
    ret = gemm_lib.cublaslt_fp8_mm(
        W_q.data_ptr(), A_q.data_ptr(), output.data_ptr(),
        M, N, K, b"bf16",
        ctypes.cast(algo_ptr, ctypes.c_void_p) if algo_ptr else None,
        workspace,
        torch.cuda.current_stream().cuda_stream
    )
    
    if ret != 0:
        err = gemm_lib.cublaslt_gemm_get_last_error()
        raise RuntimeError(err.decode() if err else 'Unknown')
    
    return output


def main():
    print("=" * 80)
    print("验证'使用最大M配置'策略")
    print("=" * 80)
    print(f"GPU: {hw_info.gpu_full_name} ({hw_info.cc_tag})")
    print()
    
    dtype = "fp8e4m3"
    outdtype = "bf16"
    
    # 加载扩展
    search_src = SCRIPT_DIR / "cuBLASLt_AlgSearch" / "alg_search_cublaslt.cu"
    search_build = SCRIPT_DIR / "cuBLASLt_AlgSearch" / "build"
    search_so = build_search_extension(
        name="alg_search_cublaslt", source_file=search_src,
        build_dir=search_build, backend="cublaslt", force=False,
    )
    search_lib = load_search_extension(search_so, backend="cublaslt", setup_func=setup_lib_signatures_search)
    
    gemm_so = SCRIPT_DIR.parent / "csrc" / "cublaslt_gemm" / "build"
    so_files = list(gemm_so.glob("cublaslt_gemm*.so"))
    gemm_lib = ctypes.CDLL(str(so_files[0]))
    setup_lib_signatures_gemm(gemm_lib)
    
    # 测试多个 NK 配置
    nk_configs = [
        (3072, 2048),   # 预期全部 ws=0
        (2048, 2048),   # 预期全部 ws=0
        (16384, 2048),  # 预期全部 ws=0
        (2048, 8192),   # 有 split-K，部分 ws>0
    ]
    
    M_list = [16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    max_M = max(M_list)
    
    W_full = torch.randn(16384, 8192, device="cuda", dtype=torch.bfloat16)
    A_full = torch.randn(max_M, 8192, device="cuda", dtype=torch.bfloat16)
    W_q_full = quantize_tensor(W_full, dtype)
    A_q_full = quantize_tensor(A_full, dtype)
    
    for N, K in nk_configs:
        print("=" * 80)
        print(f"测试 (N={N}, K={K})")
        print("=" * 80)
        
        W_q = W_q_full[:N, :K].contiguous()
        
        # 搜索最大 M 的配置
        A_q_max = A_q_full[:max_M, :K].contiguous()
        cfg_max = search_best_algo(search_lib, W_q, A_q_max, N, K, max_M, dtype, outdtype)
        
        if not cfg_max:
            print(f"  无法搜索到配置，跳过")
            continue
        
        print(f"\n最大 M={max_M} 的配置:")
        print(f"  alg_id={cfg_max['alg_id']}, split_k={cfg_max['split_k']}, "
              f"ws={cfg_max['workspace']/1024:.1f}KB")
        
        # 测试所有 M
        print(f"\n使用 M={max_M} 的配置执行所有 M:")
        all_success = True
        
        for M_exec in M_list:
            A_q_exec = A_q_full[:M_exec, :K].contiguous()
            
            try:
                out = call_gemm(gemm_lib, W_q, A_q_exec, N, K, M_exec,
                               cfg_max['algo_data'], cfg_max['workspace'])
                status = "✓"
            except Exception as e:
                status = "✗"
                all_success = False
            
            # 同时搜索该 M 的最佳配置作为参考
            cfg_m = search_best_algo(search_lib, W_q, A_q_exec, N, K, M_exec, dtype, outdtype)
            if cfg_m:
                ws_info = f"(该M的最佳ws={cfg_m['workspace']/1024:.1f}KB)"
            else:
                ws_info = "(无法搜索)"
            
            print(f"  M={M_exec:5d}: {status} {ws_info}")
        
        if all_success:
            print(f"\n✓ 使用 M={max_M} 配置可以覆盖所有 M!")
        else:
            print(f"\n✗ 存在失败情况")
    
    # =========================================================================
    # 关键结论
    # =========================================================================
    print()
    print("=" * 80)
    print("关键发现")
    print("=" * 80)
    
    print("""
从交叉矩阵测试可以看出一个重要规律：

对于 ws>0 的配置（有 split-K）：
  - M_from 的配置只能执行 M_to <= M_from 的情况
  - 这是因为 workspace 大小与 M 成正比：ws ≈ M * N * split_k * dtype_size
  - 用小 M 的 workspace 不足以支撑大 M 的计算

对于 ws=0 的配置（无 split-K）：
  - 可以执行任意 M
  - 因为不需要额外的 workspace

正确的策略：
=============
1. 离线搜索时：搜索足够大的 M_max（如 16384 或更大）
2. 在线执行时：
   - 如果 M_x <= M_max 中最大的 ws=0 的 M：直接使用 ws=0 配置
   - 否则：找到 M_upper = min{M in M_list | M >= M_x}，使用该配置

更简单的策略：
=============
既然最大 M 通常是 ws=0（因为大 M 本身有足够并行度，不需要 split-K），
那么直接使用最大 M 的配置即可覆盖所有情况！
""")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
