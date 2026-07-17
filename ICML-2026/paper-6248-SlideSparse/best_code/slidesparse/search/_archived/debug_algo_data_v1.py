#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试脚本：验证 cuBLASLt algo_data 参数传递

这个脚本用于诊断离线搜索的 algo_data 在在线推理时失败的问题。

测试步骤：
1. 加载离线搜索结果的 algo_data
2. 在同一进程中用该 algo_data 执行 GEMM
3. 对比：直接使用启发式 vs 使用保存的 algo_data

问题假设：
- algo_data (cublasLtMatmulAlgo_t) 可能依赖创建它时的上下文
- algo_data 可能不是完全自包含的，可能引用了外部状态
"""

import base64
import ctypes
import json
import sys
from pathlib import Path

import torch

# 添加搜索目录到路径
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
        ctypes.c_void_p,   # W_ptr
        ctypes.c_void_p,   # A_ptr
        ctypes.c_void_p,   # C_ptr
        ctypes.c_int64,    # N
        ctypes.c_int64,    # K
        ctypes.c_int64,    # M
        ctypes.c_char_p,   # dtype
        ctypes.c_char_p,   # outdtype
        ctypes.c_int,      # warmup
        ctypes.c_int,      # repeat
        ctypes.c_int,      # topk
        ctypes.POINTER(ctypes.c_int),        # out_alg_ids
        ctypes.POINTER(ctypes.c_float),      # out_lat_us
        ctypes.POINTER(ctypes.c_float),      # out_tops
        ctypes.POINTER(ctypes.c_int64),      # out_workspace
        ctypes.POINTER(ctypes.c_float),      # out_waves_count
        ctypes.POINTER(ctypes.c_uint8),      # out_algo_data
        ctypes.POINTER(ctypes.c_uint8),      # out_valid
        ctypes.POINTER(ctypes.c_int),        # out_num_valid
        ctypes.POINTER(ctypes.c_int),        # out_alg_count
        ctypes.c_void_p,   # stream
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
    
    # FP8 GEMM 签名
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
        W_q.data_ptr(),
        A_q.data_ptr(),
        R_out.data_ptr(),
        N, K, M,
        dtype.encode(),
        outdtype.encode(),
        5,   # warmup
        20,  # repeat
        topk,
        out_alg_ids,
        out_lat_us,
        out_tops,
        out_workspace,
        out_waves_count,
        out_algo_data,
        out_valid,
        ctypes.byref(out_num_valid),
        ctypes.byref(out_alg_count),
        None,
    )
    
    if ret != 0:
        error = search_lib.cublaslt_alg_search_get_last_error()
        raise RuntimeError(f"搜索失败: {error.decode() if error else 'unknown error'}")
    
    # 返回最好的算法
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
    
    # 处理 algo_data
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
    print("cuBLASLt algo_data 参数传递调试")
    print("=" * 70)
    print(f"GPU: {hw_info.gpu_full_name} ({hw_info.cc_tag})")
    print()
    
    # === 配置 ===
    dtype = "fp8e4m3"
    outdtype = "bf16"
    N, K, M = 3072, 2048, 1024
    
    print(f"测试配置: dtype={dtype}, outdtype={outdtype}")
    print(f"矩阵维度: N={N}, K={K}, M={M}")
    print()
    
    # === 编译/加载搜索扩展 ===
    print("[1] 加载搜索扩展...")
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
    print(f"    搜索扩展: {search_so}")
    
    # === 加载 GEMM 扩展 ===
    print("[2] 加载 GEMM 扩展...")
    gemm_so = SCRIPT_DIR.parent / "csrc" / "cublaslt_gemm" / "build" / "cublaslt_gemm.cpython-312-x86_64-linux-gnu.so"
    if not gemm_so.exists():
        # 尝试找到 .so 文件
        gemm_dir = SCRIPT_DIR.parent / "csrc" / "cublaslt_gemm" / "build"
        if gemm_dir.exists():
            so_files = list(gemm_dir.glob("cublaslt_gemm*.so"))
            if so_files:
                gemm_so = so_files[0]
    
    if not gemm_so.exists():
        print(f"    [ERROR] GEMM 库不存在: {gemm_so}")
        print("    请先运行推理测试以编译 GEMM 库")
        return
    
    gemm_lib = ctypes.CDLL(str(gemm_so))
    setup_lib_signatures_gemm(gemm_lib)
    print(f"    GEMM 扩展: {gemm_so}")
    
    # === 生成测试数据 ===
    print("[3] 生成测试数据...")
    W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    W_q = quantize_tensor(W, dtype)
    A_q = quantize_tensor(A, dtype)
    print(f"    W_q: {W_q.shape}, {W_q.dtype}")
    print(f"    A_q: {A_q.shape}, {A_q.dtype}")
    
    # === 测试1: 使用启发式（默认算法） ===
    print()
    print("[4] 测试默认算法（启发式）...")
    try:
        out_default = call_gemm_with_algo(
            gemm_lib, W_q, A_q, N, K, M, "bf16", None, 0
        )
        print(f"    ✓ 成功! 输出形状: {out_default.shape}")
    except Exception as e:
        print(f"    ✗ 失败: {e}")
        out_default = None
    
    # === 测试2: 在同一进程中搜索并使用 ===
    print()
    print("[5] 在同一进程中搜索并使用 algo_data...")
    try:
        algo_info = search_and_get_algo(
            search_lib, W_q, A_q, N, K, M, dtype, outdtype
        )
        if algo_info:
            print(f"    搜索到算法: alg_id={algo_info['alg_id']}, workspace={algo_info['workspace']}")
            print(f"    algo_data (前16字节): {algo_info['algo_data'][:16].hex()}")
            
            # 立即用这个 algo_data 执行 GEMM
            out_searched = call_gemm_with_algo(
                gemm_lib, W_q, A_q, N, K, M, "bf16",
                algo_info['algo_data'], algo_info['workspace']
            )
            print(f"    ✓ 成功! 输出形状: {out_searched.shape}")
        else:
            print("    ✗ 搜索未返回有效算法")
    except Exception as e:
        print(f"    ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
    
    # === 测试3: 从 JSON 加载并使用 ===
    print()
    print("[6] 从 JSON 文件加载 algo_data 并使用...")
    
    # 查找 JSON 文件
    from slidesparse.utils import build_hw_dir_name
    hw_folder = build_hw_dir_name()
    json_path = (SCRIPT_DIR / "cuBLASLt_AlgSearch" / "alg_search_results" / 
                 hw_folder / "alg_search_Llama3.2-1B-FP8_out-BF16.json")
    
    if json_path.exists():
        with open(json_path) as f:
            config = json.load(f)
        
        nk_key = f"({N},{K})"
        m_key = str(M)
        
        nk_entry = config["nk_entries"].get(nk_key)
        if nk_entry:
            algo_cfg = nk_entry["alg_by_m"].get(m_key)
            if algo_cfg:
                algo_b64 = algo_cfg["algo_data"]
                algo_data_loaded = base64.b64decode(algo_b64)
                workspace_loaded = algo_cfg.get("workspace", 0)
                
                print(f"    从 JSON 加载: {nk_key} -> M={m_key}")
                print(f"    algo_data (前16字节): {algo_data_loaded[:16].hex()}")
                print(f"    workspace: {workspace_loaded}")
                
                try:
                    out_loaded = call_gemm_with_algo(
                        gemm_lib, W_q, A_q, N, K, M, "bf16",
                        algo_data_loaded, workspace_loaded
                    )
                    print(f"    ✓ 成功! 输出形状: {out_loaded.shape}")
                except Exception as e:
                    print(f"    ✗ 失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"    找不到 M={m_key} 的配置")
        else:
            print(f"    找不到 NK={nk_key} 的配置")
    else:
        print(f"    JSON 文件不存在: {json_path}")
    
    # === 测试4: 对比搜索得到的 algo_data 和 JSON 中的是否一致 ===
    print()
    print("[7] 对比算法数据...")
    if 'algo_info' in dir() and algo_info and 'algo_data_loaded' in dir():
        if algo_info['algo_data'] == algo_data_loaded:
            print("    ✓ 搜索得到的 algo_data 与 JSON 中的一致")
        else:
            print("    ✗ algo_data 不一致!")
            print(f"    搜索: {algo_info['algo_data'].hex()}")
            print(f"    JSON:  {algo_data_loaded.hex()}")
    
    print()
    print("=" * 70)
    print("调试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
