#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cuSPARSELt M_upper 策略验证

验证 cuSPARSELt 是否也有和 cuBLASLt 相同的问题：
- ws>0 时用 M_lower 配置执行 M_upper 会失败
- 应该使用 M_upper 策略
"""

import ctypes
import json
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


def setup_signatures_search(lib: ctypes.CDLL) -> None:
    lib.cusparselt_search_single_m.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p,
    ]
    lib.cusparselt_search_single_m.restype = ctypes.c_int
    lib.cusparselt_alg_search_get_last_error.argtypes = []
    lib.cusparselt_alg_search_get_last_error.restype = ctypes.c_char_p


def setup_signatures_gemm(lib: ctypes.CDLL) -> None:
    lib.cusparselt_gemm_get_last_error.argtypes = []
    lib.cusparselt_gemm_get_last_error.restype = ctypes.c_char_p
    
    gemm_sig = ([ctypes.c_void_p] * 3 + 
               [ctypes.c_int64] * 3 + 
               [ctypes.c_char_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_size_t,
                ctypes.c_void_p])
    lib.cusparselt_fp8_mm.argtypes = gemm_sig
    lib.cusparselt_fp8_mm.restype = ctypes.c_int


def search_algo(search_lib, W_q, A_q, N, K, M):
    """搜索 cuSPARSELt 配置"""
    R_out = torch.zeros(M, N, dtype=torch.bfloat16, device=W_q.device)
    topk = 5
    
    out_alg_ids = (ctypes.c_int * topk)()
    out_lat_us = (ctypes.c_float * topk)()
    out_tops = (ctypes.c_float * topk)()
    out_workspace = (ctypes.c_int64 * topk)()
    out_split_k = (ctypes.c_int * topk)()
    out_valid = (ctypes.c_uint8 * topk)()
    out_num_valid = ctypes.c_int(0)
    out_alg_count = ctypes.c_int(0)
    
    ret = search_lib.cusparselt_search_single_m(
        W_q.data_ptr(), A_q.data_ptr(), R_out.data_ptr(),
        N, K, M, b"fp8e4m3", b"bf16",
        2, 5, topk,
        out_alg_ids, out_lat_us, out_tops, out_workspace,
        out_split_k, out_valid,
        ctypes.byref(out_num_valid), ctypes.byref(out_alg_count),
        None,
    )
    
    if ret != 0 or not out_valid[0]:
        return None
    
    return {
        "alg_id": out_alg_ids[0],
        "split_k": out_split_k[0],
        "workspace": out_workspace[0],
    }


def call_gemm(gemm_lib, W_q, A_q, N, K, M, alg_id, split_k, workspace):
    """调用 cuSPARSELt GEMM"""
    output = torch.empty((M, N), dtype=torch.bfloat16, device=W_q.device)
    
    ret = gemm_lib.cusparselt_fp8_mm(
        W_q.data_ptr(), A_q.data_ptr(), output.data_ptr(),
        M, N, K, b"bf16",
        alg_id, split_k, workspace,
        torch.cuda.current_stream().cuda_stream
    )
    
    if ret != 0:
        err = gemm_lib.cusparselt_gemm_get_last_error()
        raise RuntimeError(err.decode() if err else 'GEMM failed')
    
    return output


def compress_weight(W_bf16):
    """压缩权重为 2:4 稀疏格式"""
    N, K = W_bf16.shape
    
    # 简单的 2:4 稀疏化：每4个元素保留2个最大的
    W_sparse = W_bf16.clone()
    for i in range(0, K, 4):
        block = W_sparse[:, i:i+4].abs()
        _, indices = block.topk(2, dim=1)
        mask = torch.zeros_like(block, dtype=torch.bool)
        mask.scatter_(1, indices, True)
        W_sparse[:, i:i+4][~mask] = 0
    
    return W_sparse.to(torch.float8_e4m3fn)


def main():
    print("=" * 80)
    print("cuSPARSELt M_upper 策略验证")
    print("=" * 80)
    print(f"GPU: {hw_info.gpu_full_name} ({hw_info.cc_tag})")
    print()
    
    # 加载搜索扩展
    search_src = SCRIPT_DIR / "cuSPARSELt_AlgSearch" / "alg_search_cusparselt.cu"
    search_build = SCRIPT_DIR / "cuSPARSELt_AlgSearch" / "build"
    
    if not search_src.exists():
        print(f"搜索扩展源码不存在: {search_src}")
        print("跳过 cuSPARSELt 验证")
        return
    
    try:
        search_so = build_search_extension(
            name="alg_search_cusparselt",
            source_file=search_src,
            build_dir=search_build,
            backend="cusparselt",
            force=False,
        )
        search_lib = load_search_extension(search_so, backend="cusparselt", 
                                           setup_func=setup_signatures_search)
    except Exception as e:
        print(f"加载搜索扩展失败: {e}")
        return
    
    # 加载 GEMM 扩展
    gemm_so = SCRIPT_DIR.parent / "csrc" / "cusparselt_gemm" / "build"
    so_files = list(gemm_so.glob("cusparselt_gemm*.so"))
    if not so_files:
        print("GEMM 库不存在，跳过")
        return
    
    gemm_lib = ctypes.CDLL(str(so_files[0]))
    setup_signatures_gemm(gemm_lib)
    
    # 测试配置 - 使用较小的尺寸
    N, K = 2048, 3072  # K 需要是 16 的倍数用于 2:4 稀疏
    
    print(f"测试配置: (N={N}, K={K})")
    
    # 准备数据
    W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    W_q = compress_weight(W)
    
    A_full = torch.randn(16384, K, device="cuda", dtype=torch.bfloat16)
    A_q_full = A_full.to(torch.float8_e4m3fn)
    
    # 搜索不同 M 的配置
    M_list = [16, 128, 512, 1024, 4096, 8192, 16384]
    
    print("\n[1] 搜索各 M 值的配置:")
    print("-" * 60)
    
    configs: Dict[int, dict] = {}
    for M in M_list:
        A_q = A_q_full[:M, :].contiguous()
        cfg = search_algo(search_lib, W_q, A_q, N, K, M)
        if cfg:
            configs[M] = cfg
            ws_kb = cfg['workspace'] / 1024 if cfg['workspace'] > 0 else 0
            print(f"  M={M:5d}: alg_id={cfg['alg_id']}, split_k={cfg['split_k']}, ws={ws_kb:.1f}KB")
        else:
            print(f"  M={M:5d}: 搜索失败")
    
    if not configs:
        print("没有搜索到任何配置，退出")
        return
    
    # 识别 ws>0 和 ws=0 的配置
    ws_positive = {m: c for m, c in configs.items() if c['workspace'] > 0}
    ws_zero = {m: c for m, c in configs.items() if c['workspace'] == 0}
    
    print(f"\n  ws>0 的 M: {list(ws_positive.keys())}")
    print(f"  ws=0 的 M: {list(ws_zero.keys())}")
    
    # 测试交叉矩阵
    print("\n" + "=" * 80)
    print("[2] 交叉矩阵：用 M_from 配置执行 M_to")
    print("=" * 80)
    
    sorted_ms = sorted(configs.keys())
    
    # 打印表头
    header = "M_from\\M_to |" + "".join(f"{m:>7}" for m in sorted_ms)
    print(header)
    print("-" * len(header))
    
    for M_from in sorted_ms:
        cfg = configs[M_from]
        ws_tag = "+" if cfg['workspace'] > 0 else "0"
        row = f"  {M_from:5d}({ws_tag}) |"
        
        for M_to in sorted_ms:
            A_q = A_q_full[:M_to, :].contiguous()
            
            try:
                out = call_gemm(gemm_lib, W_q, A_q, N, K, M_to,
                               cfg['alg_id'], cfg['split_k'], cfg['workspace'])
                row += "      ✓"
            except:
                row += "      ✗"
        
        print(row)
    
    # 验证 M_upper 策略
    print("\n" + "=" * 80)
    print("[3] 验证 M_upper 策略")
    print("=" * 80)
    
    test_M_x = [100, 200, 300, 600, 800, 2000, 3000, 5000, 10000]
    
    print("\n对于未搜索的 M_x，测试 M_upper vs M_lower:")
    print(f"M_list = {sorted_ms}")
    
    for M_x in test_M_x:
        if M_x > max(sorted_ms):
            print(f"\n  M_x={M_x}: 超过最大 M，需要 fallback")
            continue
        
        # 找 M_upper
        M_upper = None
        for m in sorted_ms:
            if m >= M_x:
                M_upper = m
                break
        
        # 找 M_lower
        M_lower = None
        for m in reversed(sorted_ms):
            if m <= M_x:
                M_lower = m
                break
        
        if M_upper is None or M_lower is None:
            continue
        
        A_q = A_q_full[:M_x, :].contiguous()
        
        # 测试 M_upper
        cfg_upper = configs[M_upper]
        try:
            call_gemm(gemm_lib, W_q, A_q, N, K, M_x,
                     cfg_upper['alg_id'], cfg_upper['split_k'], cfg_upper['workspace'])
            upper_ok = "✓"
        except:
            upper_ok = "✗"
        
        # 测试 M_lower
        cfg_lower = configs[M_lower]
        try:
            call_gemm(gemm_lib, W_q, A_q, N, K, M_x,
                     cfg_lower['alg_id'], cfg_lower['split_k'], cfg_lower['workspace'])
            lower_ok = "✓"
        except:
            lower_ok = "✗"
        
        print(f"\n  M_x={M_x:5d}:")
        print(f"    M_upper={M_upper:5d} (ws={cfg_upper['workspace']}): {upper_ok}")
        print(f"    M_lower={M_lower:5d} (ws={cfg_lower['workspace']}): {lower_ok}")
    
    # 结论
    print("\n" + "=" * 80)
    print("[4] 结论")
    print("=" * 80)
    print("""
如果交叉矩阵显示：
  - ws>0 配置只能执行 M_to <= M_from
  - ws=0 配置可以执行任意 M

则 cuSPARSELt 与 cuBLASLt 有相同的问题，需要使用 M_upper 策略。
""")


if __name__ == "__main__":
    main()
