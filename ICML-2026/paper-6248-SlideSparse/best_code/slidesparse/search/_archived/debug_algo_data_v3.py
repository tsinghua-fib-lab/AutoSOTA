#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试脚本 v3：模拟 vLLM 的首次 GEMM 调用

vLLM 首次调用时的 M 值可能非常小（如 profile run 时）。
这个脚本测试各种边界 M 值。
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


def setup_gemm_lib(lib_path):
    """设置 GEMM 扩展"""
    lib = ctypes.CDLL(str(lib_path))
    lib.cublaslt_gemm_get_last_error.argtypes = []
    lib.cublaslt_gemm_get_last_error.restype = ctypes.c_char_p
    
    gemm_sig = ([ctypes.c_void_p] * 3 + 
               [ctypes.c_int64] * 3 + 
               [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p])
    lib.cublaslt_fp8_mm.argtypes = gemm_sig
    lib.cublaslt_fp8_mm.restype = ctypes.c_int
    return lib


def call_gemm(gemm_lib, W_q, A_q, N, K, M, inner_dtype, algo_data, workspace):
    """执行 GEMM"""
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
    print("cuBLASLt algo_data 边界 M 值测试")
    print("=" * 70)
    print(f"GPU: {hw_info.gpu_full_name} ({hw_info.cc_tag})")
    print()
    
    dtype = "fp8e4m3"
    
    # 加载 GEMM 扩展
    gemm_dir = SCRIPT_DIR.parent / "csrc" / "cublaslt_gemm" / "build"
    so_files = list(gemm_dir.glob("cublaslt_gemm*.so"))
    if not so_files:
        print("[ERROR] GEMM 库不存在")
        return
    gemm_lib = setup_gemm_lib(so_files[0])
    print(f"GEMM 扩展: {so_files[0].name}")
    
    # 加载 JSON 配置
    from slidesparse.utils import build_hw_dir_name
    hw_folder = build_hw_dir_name()
    json_path = (SCRIPT_DIR / "cuBLASLt_AlgSearch" / "alg_search_results" / 
                 hw_folder / "alg_search_Llama3.2-1B-FP8_out-BF16.json")
    
    with open(json_path) as f:
        config = json.load(f)
    
    # 选择一个 NK 组合测试
    N, K = 3072, 2048
    nk_key = f"({N},{K})"
    nk_entry = config["nk_entries"][nk_key]
    
    # 获取 M=16 的算法配置（最小阈值）
    algo_cfg = nk_entry["alg_by_m"]["16"]
    algo_data = base64.b64decode(algo_cfg["algo_data"])
    workspace = algo_cfg.get("workspace", 0)
    
    print(f"\n使用 ({N},{K}) M=16 的算法配置")
    print(f"algo_data: {algo_data[:16].hex()}...")
    print(f"workspace: {workspace}")
    
    # 生成测试数据
    W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    A_large = torch.randn(16384, K, device="cuda", dtype=torch.bfloat16)
    W_q = quantize_tensor(W, dtype)
    A_q_large = quantize_tensor(A_large, dtype)
    
    # 测试各种 M 值
    print("\n测试各种 M 值:")
    test_ms = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 16, 17, 20, 24, 32, 64, 128, 256, 512, 1024]
    
    for M in test_ms:
        A_q = A_q_large[:M].contiguous()
        try:
            out = call_gemm(gemm_lib, W_q, A_q, N, K, M, "bf16", algo_data, workspace)
            status = "✓"
        except Exception as e:
            err_msg = str(e)[:50]
            status = f"✗ {err_msg}"
        
        print(f"  M={M:5d}: {status}")
    
    # 测试不用 algo_data（使用默认启发式）
    print("\n测试使用默认启发式 (algo_data=None):")
    for M in test_ms[:10]:
        A_q = A_q_large[:M].contiguous()
        try:
            out = call_gemm(gemm_lib, W_q, A_q, N, K, M, "bf16", None, 0)
            status = "✓"
        except Exception as e:
            err_msg = str(e)[:50]
            status = f"✗ {err_msg}"
        
        print(f"  M={M:5d}: {status}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
