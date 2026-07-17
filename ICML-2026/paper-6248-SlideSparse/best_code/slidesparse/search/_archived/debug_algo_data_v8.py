#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试脚本 v8：分析 cublasLtMatmulAlgo_t 结构体

关键发现总结：
1. uint32[3] 和 uint32[4] 与 workspace 需求强相关
   - workspace>0: uint32[3]=4-5, uint32[4]=2
   - workspace=0: uint32[3]=1, uint32[4]=0
2. 增大 workspace 可以让失败的配置工作
3. workspace=0 的配置可以跨 M 复用

假设：
- uint32[3] 和 uint32[4] 可能是 split-K 或 tile 配置
- workspace>0 表示使用了 split-K 策略需要中间缓冲区
- workspace=0 表示不使用 split-K

验证策略：
- 使用 cublasLtMatmulAlgoConfigGetAttribute 获取算法属性
- 理解每个字段的含义
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


def analyze_algo_structure(algo_data: bytes) -> dict:
    """详细分析 algo_data 结构"""
    u32 = struct.unpack("<16I", algo_data)
    
    # 根据 cuBLAS 文档和实验推断：
    # uint32[0] = alg_id (CUBLASLT_ALGO_CONFIG_ID)
    # uint32[1] = tile_id (CUBLASLT_ALGO_CONFIG_TILE_ID)  
    # uint32[2] = stages_id (CUBLASLT_ALGO_CONFIG_STAGES_ID)
    # uint32[3] = split_k_factor (CUBLASLT_ALGO_CONFIG_SPLITK_NUM)
    # uint32[4] = reduction_scheme (CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME)
    # uint32[5] = cta_swizzle (CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING)
    # 后续字段可能是 custom options
    
    return {
        "raw_u32": u32,
        "alg_id": u32[0],
        "tile_id": u32[1],
        "stages_id": u32[2],
        "split_k_num": u32[3],  # 关键！
        "reduction_scheme": u32[4],  # 关键！
        "cta_swizzle": u32[5],
        "custom_opts": u32[6:12],
    }


def main():
    print("=" * 70)
    print("cublasLtMatmulAlgo_t 结构分析 v8")
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
    
    print(f"[1] 分析 (N={N}, K={K}) 不同 M 的 algo 结构")
    print("=" * 70)
    
    for M in [16, 128, 1024, 4096, 8192, 16384]:
        A_q = A_q_full[:M, :].contiguous()
        algos = search_all_algos(search_lib, W_q, A_q, N, K, M, dtype, outdtype)
        
        if algos:
            best = algos[0]
            info = analyze_algo_structure(best['algo_data'])
            
            ws_kb = best['workspace'] / 1024 if best['workspace'] > 0 else 0
            print(f"\nM={M:5d}, workspace={ws_kb:8.1f} KB:")
            print(f"  alg_id={info['alg_id']}, tile={info['tile_id']}, stages={info['stages_id']}")
            print(f"  split_k_num={info['split_k_num']}, reduction={info['reduction_scheme']}, swizzle={info['cta_swizzle']}")
    
    # 关键实验：修改 algo_data 中的 split_k 字段
    print("\n" + "=" * 70)
    print("[2] 实验：修改 algo_data 中的 split_k 字段")
    print("=" * 70)
    
    M_from = 4096
    M_to = 8192
    
    A_q_from = A_q_full[:M_from, :].contiguous()
    A_q_to = A_q_full[:M_to, :].contiguous()
    
    algos_from = search_all_algos(search_lib, W_q, A_q_from, N, K, M_from, dtype, outdtype)
    algos_to = search_all_algos(search_lib, W_q, A_q_to, N, K, M_to, dtype, outdtype)
    
    if algos_from and algos_to:
        algo_from = algos_from[0]
        algo_to = algos_to[0]
        
        info_from = analyze_algo_structure(algo_from['algo_data'])
        info_to = analyze_algo_structure(algo_to['algo_data'])
        
        print(f"\nM={M_from} 配置: split_k_num={info_from['split_k_num']}, reduction={info_from['reduction_scheme']}, workspace={algo_from['workspace']}")
        print(f"M={M_to} 配置: split_k_num={info_to['split_k_num']}, reduction={info_to['reduction_scheme']}, workspace={algo_to['workspace']}")
        
        # 尝试修改 algo_from 的 split_k 字段为 algo_to 的值
        print(f"\n尝试将 M={M_from} 配置的 split_k_num 和 reduction 修改为 M={M_to} 的值:")
        
        modified_algo = bytearray(algo_from['algo_data'])
        # 修改 uint32[3] (split_k_num) 和 uint32[4] (reduction_scheme)
        struct.pack_into("<I", modified_algo, 12, info_to['split_k_num'])  # offset 12 = 3*4
        struct.pack_into("<I", modified_algo, 16, info_to['reduction_scheme'])  # offset 16 = 4*4
        
        print(f"  原始: split_k_num={info_from['split_k_num']}, reduction={info_from['reduction_scheme']}")
        print(f"  修改后: split_k_num={info_to['split_k_num']}, reduction={info_to['reduction_scheme']}")
        
        # 测试修改后的配置
        for ws_to_try in [0, algo_from['workspace'], algo_to['workspace']]:
            try:
                out = call_gemm(gemm_lib, W_q, A_q_to, N, K, M_to,
                               bytes(modified_algo), ws_to_try)
                print(f"  用修改后的 algo_data + workspace={ws_to_try}: ✓")
            except Exception as e:
                print(f"  用修改后的 algo_data + workspace={ws_to_try}: ✗")
    
    # 分析所有 NK 配置，找出哪些需要 workspace
    print("\n" + "=" * 70)
    print("[3] 分析所有 NK 配置的 workspace 需求模式")
    print("=" * 70)
    
    nk_configs = [
        (3072, 2048),
        (2048, 2048),
        (16384, 2048),
        (2048, 8192),
    ]
    
    for N, K in nk_configs:
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        W_q = quantize_tensor(W, dtype)
        
        print(f"\n(N={N}, K={K}):")
        for M in [1024, 4096, 8192]:
            A_q = A_q_full[:M, :K].contiguous()
            algos = search_all_algos(search_lib, W_q, A_q, N, K, M, dtype, outdtype)
            
            if algos:
                best = algos[0]
                info = analyze_algo_structure(best['algo_data'])
                ws_kb = best['workspace'] / 1024
                print(f"  M={M:5d}: split_k={info['split_k_num']}, reduction={info['reduction_scheme']}, workspace={ws_kb:.1f}KB")
    
    # 结论
    print("\n" + "=" * 70)
    print("[4] 根本原因分析与解决方案")
    print("=" * 70)
    
    print("""
根本原因：
=========
cublasLtMatmulAlgo_t 结构体中的 split_k_num 和 reduction_scheme 字段
决定了算法是否需要 workspace，以及 workspace 的大小。

当 split_k_num > 1 时：
- 算法会将 K 维度分割成多个部分并行计算
- 需要 workspace 存储中间结果
- workspace 大小 ≈ M * N * num_splits * sizeof(output_type)

关键观察：
1. (2048, 8192) 配置：
   - 小 M (16-4096): split_k > 1, 需要 workspace
   - 大 M (8192-16384): split_k = 1, workspace = 0
   
2. 其他 NK 配置：
   - 所有 M: split_k = 1, workspace = 0

为什么 (2048, 8192) 不同？
- K=8192 很大，小 M 时 GPU 利用率低
- cuBLASLt 选择 split-K 策略来增加并行度
- 大 M 时自然有足够并行度，不需要 split-K

解决方案：
=========
1. 【推荐】在线重新搜索：放弃离线 algo_data，只保存 alg_id 作为提示
   - 用实际 M 在线搜索，找到 alg_id 匹配的算法
   - 这样可以获得正确的 split_k 和 workspace 配置

2. 【备选】workspace 动态计算：
   - 保存 algo_data，但在线根据实际 M 重新计算 workspace
   - 问题：algo_data 内部的 split_k 配置可能不适合新的 M

3. 【保守】只保存 workspace=0 的配置：
   - 优先选择不需要 workspace 的算法
   - 可能牺牲小 M 时的性能
""")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
