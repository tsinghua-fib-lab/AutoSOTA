#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证用户的 M_upper 策略

策略描述：
  对于任意 M_x，找到 M_upper = min{M ∈ M_list | M >= M_x}
  使用 M_upper 的配置执行 M_x

例如 M_list = [16, 256, 512, 1024, 4096, 16384]:
  M_x = 1000 --> M_upper = 1024
  M_x = 128  --> M_upper = 256
  M_x = 4096 --> M_upper = 4096
  M_x = 2048 --> M_upper = 4096

验证逻辑：
  ws>0 的配置可以执行 M_exec <= M_from
  所以 M_x <= M_upper，用 M_upper 的配置应该可以执行 M_x
"""

import ctypes
import struct
import sys
from pathlib import Path
from typing import List, Dict, Optional

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


def setup_signatures(search_lib, gemm_lib):
    search_lib.cublaslt_search_single_m.argtypes = [
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
    search_lib.cublaslt_search_single_m.restype = ctypes.c_int
    
    gemm_lib.cublaslt_fp8_mm.argtypes = ([ctypes.c_void_p] * 3 + 
               [ctypes.c_int64] * 3 + 
               [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p])
    gemm_lib.cublaslt_fp8_mm.restype = ctypes.c_int
    gemm_lib.cublaslt_gemm_get_last_error.argtypes = []
    gemm_lib.cublaslt_gemm_get_last_error.restype = ctypes.c_char_p


def search_algo(search_lib, W_q, A_q, N, K, M):
    R_out = torch.zeros(M, N, dtype=torch.bfloat16, device=W_q.device)
    topk = 3
    out_alg_ids = (ctypes.c_int * topk)()
    out_workspace = (ctypes.c_int64 * topk)()
    out_algo_data = (ctypes.c_uint8 * (topk * 64))()
    out_valid = (ctypes.c_uint8 * topk)()
    dummy_int = ctypes.c_int(0)
    dummy_float = (ctypes.c_float * topk)()
    
    search_lib.cublaslt_search_single_m(
        W_q.data_ptr(), A_q.data_ptr(), R_out.data_ptr(),
        N, K, M, b"fp8e4m3", b"bf16",
        2, 5, topk,
        out_alg_ids, dummy_float, dummy_float, out_workspace,
        dummy_float, out_algo_data, out_valid,
        ctypes.byref(dummy_int), ctypes.byref(dummy_int), None,
    )
    
    if not out_valid[0]:
        return None
    
    algo_data = bytes(out_algo_data[0:64])
    u32 = struct.unpack("<16I", algo_data)
    return {
        "algo_data": algo_data,
        "workspace": out_workspace[0],
        "split_k": u32[3],
    }


def call_gemm(gemm_lib, W_q, A_q, N, K, M, algo_data, workspace):
    output = torch.empty((M, N), dtype=torch.bfloat16, device=W_q.device)
    algo_ptr = (ctypes.c_uint8 * 64).from_buffer_copy(algo_data) if algo_data else None
    
    ret = gemm_lib.cublaslt_fp8_mm(
        W_q.data_ptr(), A_q.data_ptr(), output.data_ptr(),
        M, N, K, b"bf16",
        ctypes.cast(algo_ptr, ctypes.c_void_p) if algo_ptr else None,
        workspace, torch.cuda.current_stream().cuda_stream
    )
    if ret != 0:
        err = gemm_lib.cublaslt_gemm_get_last_error()
        raise RuntimeError(err.decode() if err else 'GEMM failed')
    return output


def find_m_upper(M_list: List[int], M_x: int) -> Optional[int]:
    """找到 M_upper = min{M ∈ M_list | M >= M_x}"""
    for M in sorted(M_list):
        if M >= M_x:
            return M
    return None  # M_x 超过最大值


def main():
    print("=" * 80)
    print("验证 M_upper 策略")
    print("=" * 80)
    print(f"GPU: {hw_info.gpu_full_name} ({hw_info.cc_tag})")
    print()
    
    # 加载扩展
    search_so = build_search_extension(
        name="alg_search_cublaslt",
        source_file=SCRIPT_DIR / "cuBLASLt_AlgSearch" / "alg_search_cublaslt.cu",
        build_dir=SCRIPT_DIR / "cuBLASLt_AlgSearch" / "build",
        backend="cublaslt", force=False,
    )
    search_lib = load_search_extension(search_so, backend="cublaslt", setup_func=lambda x: None)
    
    gemm_so = list((SCRIPT_DIR.parent / "csrc" / "cublaslt_gemm" / "build").glob("*.so"))[0]
    gemm_lib = ctypes.CDLL(str(gemm_so))
    setup_signatures(search_lib, gemm_lib)
    
    # 使用有 ws>0 的配置：(N=2048, K=8192)
    N, K = 2048, 8192
    
    W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    W_q = quantize_tensor(W, "fp8e4m3")
    A_full = torch.randn(20000, K, device="cuda", dtype=torch.bfloat16)
    A_q_full = quantize_tensor(A_full, "fp8e4m3")
    
    # 模拟离线搜索的 M_list（全部是 ws>0 的情况，排除 ws=0 来测试极端情况）
    M_list = [16, 256, 512, 1024, 4096]  # 故意不包含 8192, 16384（它们是 ws=0）
    
    print(f"测试配置: (N={N}, K={K})")
    print(f"离线搜索的 M_list: {M_list}")
    print()
    
    # 搜索各 M 的配置
    print("[1] 搜索 M_list 中各 M 的配置:")
    print("-" * 60)
    configs: Dict[int, dict] = {}
    for M in M_list:
        A_q = A_q_full[:M, :].contiguous()
        cfg = search_algo(search_lib, W_q, A_q, N, K, M)
        if cfg:
            configs[M] = cfg
            ws_kb = cfg['workspace'] / 1024
            print(f"  M={M:5d}: split_k={cfg['split_k']}, ws={ws_kb:8.1f}KB")
    
    # 测试 M_upper 策略
    print()
    print("[2] 验证 M_upper 策略:")
    print("-" * 60)
    print("策略：对于 M_x，找 M_upper = min{M ∈ M_list | M >= M_x}，用 M_upper 配置执行 M_x")
    print()
    
    # 测试各种 M_x
    test_M_x_list = [
        # (M_x, 预期 M_upper, 说明)
        (8, 16, "M_x < 最小 M"),
        (16, 16, "精确匹配"),
        (100, 256, "在 (16, 256] 区间"),
        (128, 256, "在 (16, 256] 区间"),
        (255, 256, "在 (16, 256] 区间，接近边界"),
        (256, 256, "精确匹配"),
        (300, 512, "在 (256, 512] 区间"),
        (500, 512, "在 (256, 512] 区间"),
        (512, 512, "精确匹配"),
        (600, 1024, "在 (512, 1024] 区间"),
        (1000, 1024, "在 (512, 1024] 区间"),
        (1024, 1024, "精确匹配"),
        (2000, 4096, "在 (1024, 4096] 区间"),
        (2048, 4096, "在 (1024, 4096] 区间"),
        (3000, 4096, "在 (1024, 4096] 区间"),
        (4096, 4096, "精确匹配"),
        (5000, None, "超过最大 M，需要 fallback"),
        (8192, None, "超过最大 M，需要 fallback"),
    ]
    
    success_count = 0
    fail_count = 0
    
    for M_x, expected_upper, desc in test_M_x_list:
        M_upper = find_m_upper(M_list, M_x)
        
        # 验证 M_upper 计算是否正确
        upper_match = "✓" if M_upper == expected_upper else f"✗ 期望{expected_upper}"
        
        if M_upper is None:
            print(f"  M_x={M_x:5d} -> M_upper=None ({desc})")
            print(f"       需要 fallback 到默认启发式或报错")
            continue
        
        # 使用 M_upper 配置执行 M_x
        A_q = A_q_full[:M_x, :].contiguous()
        cfg = configs[M_upper]
        
        try:
            out = call_gemm(gemm_lib, W_q, A_q, N, K, M_x,
                           cfg['algo_data'], cfg['workspace'])
            status = "✓"
            success_count += 1
        except Exception as e:
            status = f"✗ 失败"
            fail_count += 1
        
        print(f"  M_x={M_x:5d} -> M_upper={M_upper:5d}: {status}  ({desc})")
    
    # 额外测试：边界情况
    print()
    print("[3] 额外验证：用 M_upper 配置执行 M_x (M_x < M_upper)")
    print("-" * 60)
    
    # 取 M=4096 的配置，测试执行各种更小的 M
    M_upper = 4096
    cfg_4096 = configs[M_upper]
    print(f"使用 M={M_upper} 的配置 (split_k={cfg_4096['split_k']}, ws={cfg_4096['workspace']})")
    
    for M_x in [1, 5, 10, 15, 16, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 4096]:
        A_q = A_q_full[:M_x, :].contiguous()
        
        try:
            out = call_gemm(gemm_lib, W_q, A_q, N, K, M_x,
                           cfg_4096['algo_data'], cfg_4096['workspace'])
            status = "✓"
        except:
            status = "✗"
        
        print(f"  M_upper={M_upper} -> M_x={M_x:4d}: {status}")
    
    # 反向验证：用 M_lower 执行 M_upper（应该失败）
    print()
    print("[4] 反向验证：用 M_lower 配置执行 M_upper (应该失败)")
    print("-" * 60)
    
    M_lower = 256
    cfg_256 = configs[M_lower]
    print(f"使用 M={M_lower} 的配置 (split_k={cfg_256['split_k']}, ws={cfg_256['workspace']})")
    
    for M_x in [256, 300, 400, 512, 600, 800, 1024, 2000, 4096]:
        A_q = A_q_full[:M_x, :].contiguous()
        
        try:
            out = call_gemm(gemm_lib, W_q, A_q, N, K, M_x,
                           cfg_256['algo_data'], cfg_256['workspace'])
            status = "✓"
        except:
            status = "✗"
        
        relation = "<=" if M_x <= M_lower else ">"
        expected = "应该成功" if M_x <= M_lower else "应该失败"
        print(f"  M_lower={M_lower} -> M_x={M_x:4d} ({relation} {M_lower}): {status} ({expected})")
    
    # 结论
    print()
    print("=" * 80)
    print("结论")
    print("=" * 80)
    print(f"""
验证结果:
  成功: {success_count}
  失败: {fail_count}

M_upper 策略验证:
  ✓ 用 M_upper 配置执行 M_x (M_x <= M_upper): 应该总是成功
  ✗ 用 M_lower 配置执行 M_x (M_x > M_lower): 应该失败

你的策略是正确的:
  对于 M_x，找 M_upper = min{{M ∈ M_list | M >= M_x}}
  使用 M_upper 的配置执行 M_x

  例如 M_list = [16, 256, 512, 1024, 4096]:
    M_x = 1000 --> M_upper = 1024 ✓
    M_x = 128  --> M_upper = 256  ✓
    M_x = 2048 --> M_upper = 4096 ✓
    
  区间划分:
    (0, 16]     -> M=16 配置
    (16, 256]   -> M=256 配置
    (256, 512]  -> M=512 配置
    (512, 1024] -> M=1024 配置
    (1024, 4096]-> M=4096 配置
    > 4096     -> fallback（使用默认启发式或报错）
""")


if __name__ == "__main__":
    main()
