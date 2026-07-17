#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试脚本 v7：深入分析 workspace 与 algo_data 的关系

关键发现：
1. (2048,8192) 需要非零 workspace
2. workspace 大小与 M 相关
3. M=16384 的 workspace=0 可以跨 M 复用

假设验证：
- algo_data 内部可能包含了 workspace 计算相关的信息
- 当 workspace 需求不匹配时会失败
- 解决方案：要么保证 workspace 足够大，要么用正确 M 的配置
"""

import base64
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


def search_all_algos(search_lib, W_q, A_q, N, K, M, dtype, outdtype):
    R_torch_dtype = get_output_torch_dtype(outdtype)
    R_out = torch.zeros(M, N, dtype=R_torch_dtype, device=W_q.device)
    
    topk = 20
    out_alg_ids = (ctypes.c_int * topk)()
    out_lat_us = (ctypes.c_float * topk)()
    out_workspace = (ctypes.c_int64 * topk)()
    out_algo_data = (ctypes.c_uint8 * (topk * 64))()
    out_valid = (ctypes.c_uint8 * topk)()
    out_num_valid = ctypes.c_int(0)
    out_alg_count = ctypes.c_int(0)
    
    dummy_tops = (ctypes.c_float * topk)()
    dummy_waves = (ctypes.c_float * topk)()
    
    ret = search_lib.cublaslt_search_single_m(
        W_q.data_ptr(), A_q.data_ptr(), R_out.data_ptr(),
        N, K, M, dtype.encode(), outdtype.encode(),
        2, 5, topk,
        out_alg_ids, out_lat_us, dummy_tops, out_workspace,
        dummy_waves, out_algo_data, out_valid,
        ctypes.byref(out_num_valid), ctypes.byref(out_alg_count),
        None,
    )
    
    if ret != 0:
        return []
    
    results = []
    for i in range(topk):
        if out_valid[i]:
            results.append({
                "alg_id": out_alg_ids[i],
                "algo_data": bytes(out_algo_data[i*64:(i+1)*64]),
                "workspace": out_workspace[i],
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


def compare_algo_data(data1, data2, label1, label2):
    """对比两个 algo_data"""
    diff = sum(1 for a, b in zip(data1, data2) if a != b)
    u1 = struct.unpack("<16I", data1)
    u2 = struct.unpack("<16I", data2)
    print(f"  {label1} vs {label2}: 差异字节={diff}/64")
    print(f"    {label1} uint32[0:6]: {u1[:6]}")
    print(f"    {label2} uint32[0:6]: {u2[:6]}")


def main():
    print("=" * 70)
    print("cuBLASLt workspace 与 algo_data 关系分析 v7")
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
    print(f"\n[1] 分析问题配置: (N={N}, K={K})")
    
    W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    A_full = torch.randn(16384, K, device="cuda", dtype=torch.bfloat16)
    W_q = quantize_tensor(W, dtype)
    A_q_full = quantize_tensor(A_full, dtype)
    
    # 搜索不同 M 的配置
    print("\n[2] 搜索不同 M 值的配置:")
    m_configs = {}
    for M in [16, 128, 1024, 4096, 8192, 16384]:
        A_q = A_q_full[:M, :].contiguous()
        algos = search_all_algos(search_lib, W_q, A_q, N, K, M, dtype, outdtype)
        if algos:
            best = algos[0]
            m_configs[M] = best
            print(f"  M={M:5d}: alg_id={best['alg_id']}, workspace={best['workspace']:10d}")
    
    # 分析 algo_data 差异
    print("\n[3] algo_data 内部差异分析:")
    ms = sorted(m_configs.keys())
    for i in range(len(ms) - 1):
        compare_algo_data(
            m_configs[ms[i]]['algo_data'],
            m_configs[ms[i+1]]['algo_data'],
            f"M={ms[i]}", f"M={ms[i+1]}"
        )
    
    # 实验：用不同 workspace 大小测试
    print("\n" + "=" * 70)
    print("[4] 实验：workspace 大小对执行的影响")
    print("=" * 70)
    
    M_from = 4096  # 搜索时的 M
    M_to = 8192    # 执行时的 M
    
    cfg_4096 = m_configs.get(M_from)
    cfg_8192 = m_configs.get(M_to)
    
    if cfg_4096 and cfg_8192:
        A_q_to = A_q_full[:M_to, :].contiguous()
        
        print(f"\n使用 M={M_from} 的配置 (workspace={cfg_4096['workspace']}) 执行 M={M_to}:")
        
        # 测试 1: 原始 workspace
        try:
            out = call_gemm(gemm_lib, W_q, A_q_to, N, K, M_to,
                           cfg_4096['algo_data'], cfg_4096['workspace'])
            print(f"  原始 workspace={cfg_4096['workspace']}: ✓")
        except Exception as e:
            print(f"  原始 workspace={cfg_4096['workspace']}: ✗ ({str(e)[:50]})")
        
        # 测试 2: 增大 workspace
        for ws_mult in [2, 4, 8, 16]:
            larger_ws = cfg_4096['workspace'] * ws_mult
            try:
                out = call_gemm(gemm_lib, W_q, A_q_to, N, K, M_to,
                               cfg_4096['algo_data'], larger_ws)
                print(f"  增大 workspace={larger_ws}: ✓")
            except Exception as e:
                print(f"  增大 workspace={larger_ws}: ✗")
        
        # 测试 3: 使用 M=8192 的正确 workspace
        print(f"\n使用 M={M_from} 的 algo_data + M={M_to} 的 workspace:")
        try:
            out = call_gemm(gemm_lib, W_q, A_q_to, N, K, M_to,
                           cfg_4096['algo_data'], cfg_8192['workspace'])
            print(f"  workspace={cfg_8192['workspace']}: ✓")
        except Exception as e:
            print(f"  workspace={cfg_8192['workspace']}: ✗")
        
        # 测试 4: 使用 M=8192 的完整配置
        print(f"\n使用 M={M_to} 的完整配置 (algo_data + workspace):")
        try:
            out = call_gemm(gemm_lib, W_q, A_q_to, N, K, M_to,
                           cfg_8192['algo_data'], cfg_8192['workspace'])
            print(f"  ✓ 正确配置可以工作")
        except Exception as e:
            print(f"  ✗")
    
    # 实验：M=16384 (workspace=0) 能否跨 M 复用
    print("\n" + "=" * 70)
    print("[5] 实验：workspace=0 的配置能否跨 M 复用")
    print("=" * 70)
    
    if 16384 in m_configs:
        cfg_16384 = m_configs[16384]
        print(f"M=16384 配置: alg_id={cfg_16384['alg_id']}, workspace={cfg_16384['workspace']}")
        
        for M_exec in [1024, 4096, 8192, 16384]:
            A_q_exec = A_q_full[:M_exec, :].contiguous()
            try:
                out = call_gemm(gemm_lib, W_q, A_q_exec, N, K, M_exec,
                               cfg_16384['algo_data'], cfg_16384['workspace'])
                print(f"  执行 M={M_exec:5d}: ✓")
            except Exception as e:
                print(f"  执行 M={M_exec:5d}: ✗")
    
    # 分析 workspace=0 vs workspace>0 的 algo_data 差异
    print("\n" + "=" * 70)
    print("[6] 分析 workspace=0 和 workspace>0 的 algo_data 差异")
    print("=" * 70)
    
    if 16384 in m_configs and 4096 in m_configs:
        compare_algo_data(
            m_configs[4096]['algo_data'],
            m_configs[16384]['algo_data'],
            "M=4096 (ws>0)", "M=16384 (ws=0)"
        )
    
    # 结论
    print("\n" + "=" * 70)
    print("[7] 结论与解决方案")
    print("=" * 70)
    
    print("""
关键发现：
1. cublasLtMatmulAlgo_t 结构体内部包含与问题规模相关的配置
2. workspace 需求与 M 相关，但不是简单的线性关系
3. workspace=0 的配置更容易跨 M 复用

可能的解决方案：
A. 在线重新搜索：放弃离线配置，每次用实际 M 搜索
B. 精确 M 匹配：只有当 M 精确匹配时才使用离线配置
C. 保守选择：只使用 workspace=0 的配置
D. 保存 alg_id：只保存 alg_id，在线根据实际 M 重新搜索找到匹配的 algo
""")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
