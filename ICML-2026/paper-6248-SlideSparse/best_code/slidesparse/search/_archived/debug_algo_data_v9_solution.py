#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cuBLASLt 离线配置问题 - 最终解决方案

根本原因：
==========
cublasLtMatmulAlgo_t 结构体包含了与问题规模相关的配置，特别是：
- split_k_num：K 维度分割数量
- reduction_scheme：归约方案
- tile_id：tile 配置

这些参数决定了：
1. 是否需要 workspace（split_k > 1 时需要）
2. workspace 大小（与 M 相关：≈ M * N * split_k * dtype_size）
3. 算法的计算方式

关键问题：
==========
当 K 较大（如 8192）且 M 较小时，cuBLASLt 会选择 split-K 策略来增加并行度。
这导致：
1. 不同 M 搜索出的 algo_data 内部 split_k 配置不同
2. workspace 需求随 M 变化
3. 用小 M 搜索的 algo_data 执行大 M 时会失败（workspace 不足或配置不兼容）

解决方案验证：
=============
本脚本验证三种解决方案的可行性：
1. 【方案 A】在线重新搜索：只保存 alg_id，在线用实际 M 搜索
2. 【方案 B】强制 split_k=1：搜索时限制只选择 split_k=1 的算法
3. 【方案 C】大 M 配置回退：用最大 M 搜索的配置应用于所有 M
"""

import ctypes
import struct
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List

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


def search_algos(search_lib, W_q, A_q, N, K, M, dtype, outdtype, topk=20):
    R_torch_dtype = get_output_torch_dtype(outdtype)
    R_out = torch.zeros(M, N, dtype=R_torch_dtype, device=W_q.device)
    
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
    
    if ret != 0:
        return []
    
    results = []
    for i in range(topk):
        if out_valid[i]:
            algo_data = bytes(out_algo_data[i*64:(i+1)*64])
            u32 = struct.unpack("<16I", algo_data)
            results.append({
                "alg_id": out_alg_ids[i],
                "algo_data": algo_data,
                "workspace": out_workspace[i],
                "lat_us": out_lat_us[i],
                "split_k": u32[3],
                "reduction": u32[4],
            })
    return results


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


def benchmark_gemm(gemm_lib, W_q, A_q, N, K, M, algo_data, workspace, warmup=3, repeat=10):
    """Benchmark GEMM 执行时间"""
    for _ in range(warmup):
        call_gemm(gemm_lib, W_q, A_q, N, K, M, algo_data, workspace)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        call_gemm(gemm_lib, W_q, A_q, N, K, M, algo_data, workspace)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    return (elapsed / repeat) * 1e6  # 返回微秒


def main():
    print("=" * 70)
    print("cuBLASLt 离线配置问题 - 解决方案验证")
    print("=" * 70)
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
    
    # 问题配置
    N, K = 2048, 8192
    W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    A_full = torch.randn(16384, K, device="cuda", dtype=torch.bfloat16)
    W_q = quantize_tensor(W, dtype)
    A_q_full = quantize_tensor(A_full, dtype)
    
    print(f"测试配置: (N={N}, K={K})")
    print()
    
    # =========================================================================
    # 方案 A：在线重新搜索（只保存 alg_id）
    # =========================================================================
    print("=" * 70)
    print("【方案 A】在线重新搜索")
    print("=" * 70)
    print("策略：离线只保存 alg_id，在线用实际 M 搜索，找到匹配 alg_id 的 algo")
    print()
    
    # 模拟离线：用 M=1024 搜索
    M_offline = 1024
    A_q_offline = A_q_full[:M_offline, :].contiguous()
    offline_algos = search_algos(search_lib, W_q, A_q_offline, N, K, M_offline, dtype, outdtype)
    
    if offline_algos:
        offline_best = offline_algos[0]
        saved_alg_id = offline_best['alg_id']
        print(f"离线保存: alg_id={saved_alg_id} (来自 M={M_offline})")
        
        # 模拟在线：用实际 M 执行
        for M_online in [1024, 4096, 8192]:
            A_q_online = A_q_full[:M_online, :].contiguous()
            
            # 在线搜索
            online_algos = search_algos(search_lib, W_q, A_q_online, N, K, M_online, dtype, outdtype)
            
            # 找到 alg_id 匹配的
            matched = None
            for algo in online_algos:
                if algo['alg_id'] == saved_alg_id:
                    matched = algo
                    break
            
            if matched:
                try:
                    call_gemm(gemm_lib, W_q, A_q_online, N, K, M_online,
                             matched['algo_data'], matched['workspace'])
                    lat = benchmark_gemm(gemm_lib, W_q, A_q_online, N, K, M_online,
                                        matched['algo_data'], matched['workspace'])
                    print(f"  M={M_online:5d}: ✓ 找到匹配 alg_id={saved_alg_id}, "
                          f"split_k={matched['split_k']}, ws={matched['workspace']}, "
                          f"lat={lat:.1f}us")
                except Exception as e:
                    print(f"  M={M_online:5d}: ✗ 匹配但执行失败")
            else:
                # 回退到最佳算法
                best = online_algos[0] if online_algos else None
                if best:
                    call_gemm(gemm_lib, W_q, A_q_online, N, K, M_online,
                             best['algo_data'], best['workspace'])
                    lat = benchmark_gemm(gemm_lib, W_q, A_q_online, N, K, M_online,
                                        best['algo_data'], best['workspace'])
                    print(f"  M={M_online:5d}: ⚠ 未找到 alg_id={saved_alg_id}, "
                          f"回退到 alg_id={best['alg_id']}, lat={lat:.1f}us")
    
    # =========================================================================
    # 方案 B：强制 split_k=1（搜索时只选择 split_k=1 的算法）
    # =========================================================================
    print()
    print("=" * 70)
    print("【方案 B】强制 split_k=1")
    print("=" * 70)
    print("策略：搜索时只保存 split_k=1 的算法配置")
    print()
    
    for M in [1024, 4096, 8192]:
        A_q = A_q_full[:M, :].contiguous()
        algos = search_algos(search_lib, W_q, A_q, N, K, M, dtype, outdtype)
        
        # 只选择 split_k=1 的算法
        no_split_algos = [a for a in algos if a['split_k'] == 1]
        split_algos = [a for a in algos if a['split_k'] > 1]
        
        if no_split_algos:
            best = no_split_algos[0]
            try:
                lat = benchmark_gemm(gemm_lib, W_q, A_q, N, K, M,
                                    best['algo_data'], best['workspace'])
                print(f"  M={M:5d}: ✓ 找到 split_k=1 算法, alg_id={best['alg_id']}, "
                      f"lat={lat:.1f}us")
            except Exception as e:
                print(f"  M={M:5d}: ✗ split_k=1 算法执行失败")
        else:
            print(f"  M={M:5d}: ⚠ 无 split_k=1 算法可用")
        
        # 对比 split_k>1 的性能
        if split_algos:
            best_split = split_algos[0]
            try:
                lat_split = benchmark_gemm(gemm_lib, W_q, A_q, N, K, M,
                                          best_split['algo_data'], best_split['workspace'])
                if no_split_algos:
                    best_no_split = no_split_algos[0]
                    lat_no_split = benchmark_gemm(gemm_lib, W_q, A_q, N, K, M,
                                                  best_no_split['algo_data'], best_no_split['workspace'])
                    speedup = lat_no_split / lat_split
                    print(f"            对比: split_k={best_split['split_k']} lat={lat_split:.1f}us "
                          f"({'快' if speedup > 1 else '慢'} {abs(speedup-1)*100:.1f}%)")
            except:
                pass
    
    # =========================================================================
    # 方案 C：大 M 配置回退
    # =========================================================================
    print()
    print("=" * 70)
    print("【方案 C】大 M 配置回退")
    print("=" * 70)
    print("策略：用最大 M (如 16384) 搜索的配置应用于所有 M")
    print()
    
    # 用最大 M 搜索
    M_max = 16384
    A_q_max = A_q_full[:M_max, :].contiguous()
    max_algos = search_algos(search_lib, W_q, A_q_max, N, K, M_max, dtype, outdtype)
    
    if max_algos:
        max_best = max_algos[0]
        print(f"使用 M={M_max} 的配置: alg_id={max_best['alg_id']}, "
              f"split_k={max_best['split_k']}, workspace={max_best['workspace']}")
        
        for M_exec in [1024, 4096, 8192, 16384]:
            A_q_exec = A_q_full[:M_exec, :].contiguous()
            
            try:
                lat = benchmark_gemm(gemm_lib, W_q, A_q_exec, N, K, M_exec,
                                    max_best['algo_data'], max_best['workspace'])
                print(f"  执行 M={M_exec:5d}: ✓ lat={lat:.1f}us")
            except Exception as e:
                print(f"  执行 M={M_exec:5d}: ✗ ({str(e)[:30]})")
    
    # =========================================================================
    # 方案比较
    # =========================================================================
    print()
    print("=" * 70)
    print("【方案比较】")
    print("=" * 70)
    
    print("""
方案 A：在线重新搜索
  优点：总是使用最优配置，workspace 和 split_k 都正确
  缺点：需要在线搜索开销（约 100-500ms），首次调用延迟
  适用：推理时 M 变化频繁，追求最优性能

方案 B：强制 split_k=1
  优点：配置可跨 M 复用，无 workspace 需求
  缺点：小 M 时可能性能不如 split_k>1
  适用：推理时 M 较大，不需要 split-K 优化

方案 C：大 M 配置回退
  优点：实现简单，配置可复用
  缺点：与方案 B 类似，小 M 性能可能次优
  适用：推理时 M 变化不大，主要是大 M

推荐策略：
=========
1. 离线搜索时：
   - 为每个 (N, K) 搜索多个 M 值
   - 优先保存 split_k=1 的配置（workspace=0）
   - 同时记录 alg_id 供在线匹配使用

2. 在线执行时：
   - 首先尝试使用离线配置
   - 如果失败或 M 不匹配，用实际 M 在线搜索
   - 缓存搜索结果，避免重复搜索
""")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
