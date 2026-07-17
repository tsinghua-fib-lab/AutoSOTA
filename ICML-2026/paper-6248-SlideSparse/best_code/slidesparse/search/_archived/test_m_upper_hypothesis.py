#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证假设：M_upper 配置能否应用于 M_lower

假设验证：
1. ws>0 且 M 不匹配时：用 M_upper 的配置执行 M_lower 是否可行？
2. ws=0 的配置是否可以跨 M 任意复用？
3. M_lower 的配置执行 M_upper 是否会失败？

测试矩阵：
- 固定 (N=2048, K=8192)，这是有 split-K 和 ws>0 的配置
- 搜索 M = [16, 128, 1024, 4096, 8192, 16384]
- 交叉测试所有组合
"""

import ctypes
import struct
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
        "reduction": u32[4],
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
    print("验证假设：M_upper 配置能否应用于 M_lower")
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
    
    # 测试配置：(N=2048, K=8192) - 有 split-K 和 ws>0 的配置
    N, K = 2048, 8192
    
    W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    A_full = torch.randn(16384, K, device="cuda", dtype=torch.bfloat16)
    W_q = quantize_tensor(W, dtype)
    A_q_full = quantize_tensor(A_full, dtype)
    
    # 搜索所有 M 的配置
    M_list = [16, 128, 512, 1024, 2048, 4096, 8192, 16384]
    
    print(f"[1] 搜索各 M 值的最佳配置 (N={N}, K={K})")
    print("-" * 80)
    
    configs = {}
    for M in M_list:
        A_q = A_q_full[:M, :].contiguous()
        cfg = search_best_algo(search_lib, W_q, A_q, N, K, M, dtype, outdtype)
        if cfg:
            configs[M] = cfg
            ws_kb = cfg['workspace'] / 1024
            print(f"  M={M:5d}: alg_id={cfg['alg_id']}, split_k={cfg['split_k']}, "
                  f"tile={cfg['tile']}, ws={ws_kb:8.1f}KB")
    
    # 识别 ws>0 和 ws=0 的边界
    ws_positive = [(M, cfg) for M, cfg in configs.items() if cfg['workspace'] > 0]
    ws_zero = [(M, cfg) for M, cfg in configs.items() if cfg['workspace'] == 0]
    
    print()
    print(f"  ws>0 的 M 值: {[m for m, _ in ws_positive]}")
    print(f"  ws=0 的 M 值: {[m for m, _ in ws_zero]}")
    
    # =========================================================================
    # 测试 1: 用 M_upper 的配置执行 M_lower（ws>0 情况）
    # =========================================================================
    print()
    print("=" * 80)
    print("[2] 测试：用 M_upper 的配置执行 M_lower（针对 ws>0 的情况）")
    print("=" * 80)
    
    if ws_positive:
        # 按 M 排序
        ws_positive_sorted = sorted(ws_positive, key=lambda x: x[0])
        
        # 取最大的 ws>0 配置作为 M_upper
        M_upper, cfg_upper = ws_positive_sorted[-1]
        
        print(f"\n使用 M_upper={M_upper} 的配置 (split_k={cfg_upper['split_k']}, ws={cfg_upper['workspace']})")
        print()
        
        # 测试所有更小的 M
        for M_exec in M_list:
            if M_exec > M_upper:
                continue
            
            A_q = A_q_full[:M_exec, :].contiguous()
            
            try:
                out = call_gemm(gemm_lib, W_q, A_q, N, K, M_exec,
                               cfg_upper['algo_data'], cfg_upper['workspace'])
                print(f"  M_upper={M_upper} -> M_exec={M_exec:5d}: ✓ 成功")
            except Exception as e:
                print(f"  M_upper={M_upper} -> M_exec={M_exec:5d}: ✗ 失败 ({str(e)[:40]})")
    
    # =========================================================================
    # 测试 2: 用 M_lower 的配置执行 M_upper（预期失败）
    # =========================================================================
    print()
    print("=" * 80)
    print("[3] 测试：用 M_lower 的配置执行 M_upper（预期失败）")
    print("=" * 80)
    
    if len(ws_positive) >= 2:
        ws_positive_sorted = sorted(ws_positive, key=lambda x: x[0])
        M_lower, cfg_lower = ws_positive_sorted[0]  # 最小的 ws>0 配置
        
        print(f"\n使用 M_lower={M_lower} 的配置 (split_k={cfg_lower['split_k']}, ws={cfg_lower['workspace']})")
        print()
        
        for M_exec in M_list:
            if M_exec < M_lower:
                continue
            
            A_q = A_q_full[:M_exec, :].contiguous()
            
            try:
                out = call_gemm(gemm_lib, W_q, A_q, N, K, M_exec,
                               cfg_lower['algo_data'], cfg_lower['workspace'])
                print(f"  M_lower={M_lower} -> M_exec={M_exec:5d}: ✓ 成功")
            except Exception as e:
                print(f"  M_lower={M_lower} -> M_exec={M_exec:5d}: ✗ 失败")
    
    # =========================================================================
    # 测试 3: ws=0 配置的跨 M 复用
    # =========================================================================
    print()
    print("=" * 80)
    print("[4] 测试：ws=0 配置的跨 M 复用")
    print("=" * 80)
    
    if ws_zero:
        # 取一个 ws=0 的配置
        M_ws0, cfg_ws0 = ws_zero[0]
        
        print(f"\n使用 M={M_ws0} 的 ws=0 配置 (split_k={cfg_ws0['split_k']})")
        print()
        
        # 测试所有 M
        for M_exec in M_list:
            A_q = A_q_full[:M_exec, :].contiguous()
            
            try:
                out = call_gemm(gemm_lib, W_q, A_q, N, K, M_exec,
                               cfg_ws0['algo_data'], cfg_ws0['workspace'])
                print(f"  M={M_ws0} (ws=0) -> M_exec={M_exec:5d}: ✓ 成功")
            except Exception as e:
                print(f"  M={M_ws0} (ws=0) -> M_exec={M_exec:5d}: ✗ 失败")
    
    # =========================================================================
    # 测试 4: 完整的交叉矩阵
    # =========================================================================
    print()
    print("=" * 80)
    print("[5] 完整交叉矩阵：用 M_from 的配置执行 M_to")
    print("=" * 80)
    
    # 打印表头
    header = "M_from\\M_to |" + "".join(f"{m:>7}" for m in M_list)
    print(header)
    print("-" * len(header))
    
    for M_from in M_list:
        if M_from not in configs:
            continue
        
        cfg = configs[M_from]
        ws_tag = "+" if cfg['workspace'] > 0 else "0"
        row = f"  {M_from:5d}({ws_tag}) |"
        
        for M_to in M_list:
            A_q = A_q_full[:M_to, :].contiguous()
            
            try:
                out = call_gemm(gemm_lib, W_q, A_q, N, K, M_to,
                               cfg['algo_data'], cfg['workspace'])
                row += "      ✓"
            except:
                row += "      ✗"
        
        print(row)
    
    # =========================================================================
    # 结论
    # =========================================================================
    print()
    print("=" * 80)
    print("[6] 结论")
    print("=" * 80)
    
    print("""
根据测试结果验证假设：

1. ws>0 且用 M_upper 配置执行 M_lower：
   - 预期：应该可以工作（大的 workspace 足够小 M 使用）
   - 实际：需要看上面的测试结果

2. ws>0 且用 M_lower 配置执行 M_upper：
   - 预期：会失败（workspace 不足）
   - 实际：需要看上面的测试结果

3. ws=0 配置跨 M 复用：
   - 预期：应该可以任意复用
   - 实际：需要看上面的测试结果

策略建议：
- 如果假设 1 和 3 成立，则可以采用 "向上取整" 策略
- 即：对于任意 M_x，找到 M_upper = min{M in M_list | M >= M_x}
- 使用 M_upper 的配置执行 M_x
""")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
