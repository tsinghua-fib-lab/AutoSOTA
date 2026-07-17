#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线场景测试：模拟完全脱离上下文后的复用

这个脚本分两个阶段运行：
1. 阶段 1（--phase=search）：搜索并保存配置到文件
2. 阶段 2（--phase=exec）：加载配置并执行（新进程，新 CUDA context）

用法：
  # 先搜索
  python test_offline_context.py --phase=search
  
  # 再执行（建议重启 Python 或开新终端）
  python test_offline_context.py --phase=exec

这样可以确保：
- 搜索时的 CUDA context 已被销毁
- 执行时是全新的 CUDA context
- 验证 algo_data 是否真正可以跨 context 复用
"""

import argparse
import base64
import ctypes
import json
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

# 保存配置的文件
CONFIG_FILE = SCRIPT_DIR / "offline_test_config.json"


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


def search_algo(search_lib, W_q, A_q, N, K, M, dtype, outdtype):
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
        "algo_data_b64": base64.b64encode(algo_data).decode(),
        "workspace": out_workspace[0],
        "split_k": u32[3],
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


def phase_search():
    """阶段 1：搜索并保存配置"""
    print("=" * 80)
    print("阶段 1：搜索配置")
    print("=" * 80)
    print(f"GPU: {hw_info.gpu_full_name} ({hw_info.cc_tag})")
    print(f"PID: {__import__('os').getpid()}")
    print()
    
    dtype = "fp8e4m3"
    outdtype = "bf16"
    
    # 加载搜索扩展
    search_src = SCRIPT_DIR / "cuBLASLt_AlgSearch" / "alg_search_cublaslt.cu"
    search_build = SCRIPT_DIR / "cuBLASLt_AlgSearch" / "build"
    search_so = build_search_extension(
        name="alg_search_cublaslt", source_file=search_src,
        build_dir=search_build, backend="cublaslt", force=False,
    )
    search_lib = load_search_extension(search_so, backend="cublaslt", setup_func=setup_lib_signatures_search)
    
    # 测试配置
    nk_configs = [
        (3072, 2048),
        (2048, 8192),
    ]
    M_list = [16, 128, 1024, 4096, 8192, 16384]
    
    results = {
        "gpu": hw_info.gpu_full_name,
        "pid": __import__('os').getpid(),
        "configs": {}
    }
    
    for N, K in nk_configs:
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        W_q = quantize_tensor(W, dtype)
        
        nk_key = f"({N},{K})"
        results["configs"][nk_key] = {}
        
        print(f"\n搜索 {nk_key}:")
        
        for M in M_list:
            A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            A_q = quantize_tensor(A, dtype)
            
            cfg = search_algo(search_lib, W_q, A_q, N, K, M, dtype, outdtype)
            
            if cfg:
                results["configs"][nk_key][str(M)] = cfg
                print(f"  M={M:5d}: alg_id={cfg['alg_id']}, split_k={cfg['split_k']}, ws={cfg['workspace']}")
    
    # 保存配置
    with open(CONFIG_FILE, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n配置已保存到: {CONFIG_FILE}")
    print("\n请运行以下命令进行阶段 2 测试（新进程）：")
    print(f"  python {__file__} --phase=exec")


def phase_exec():
    """阶段 2：加载配置并执行"""
    print("=" * 80)
    print("阶段 2：加载配置并执行（新 CUDA context）")
    print("=" * 80)
    print(f"GPU: {hw_info.gpu_full_name} ({hw_info.cc_tag})")
    print(f"PID: {__import__('os').getpid()}")
    print()
    
    # 加载配置
    if not CONFIG_FILE.exists():
        print(f"错误：配置文件不存在: {CONFIG_FILE}")
        print("请先运行: python {__file__} --phase=search")
        return
    
    with open(CONFIG_FILE) as f:
        saved = json.load(f)
    
    print(f"搜索时的 GPU: {saved['gpu']}")
    print(f"搜索时的 PID: {saved['pid']}")
    print(f"当前 PID: {__import__('os').getpid()}")
    print()
    
    if saved['pid'] == __import__('os').getpid():
        print("⚠ 警告：PID 相同，建议开新进程运行以确保 CUDA context 重建")
    
    dtype = "fp8e4m3"
    
    # 加载 GEMM 扩展
    gemm_so = SCRIPT_DIR.parent / "csrc" / "cublaslt_gemm" / "build"
    so_files = list(gemm_so.glob("cublaslt_gemm*.so"))
    gemm_lib = ctypes.CDLL(str(so_files[0]))
    setup_lib_signatures_gemm(gemm_lib)
    
    # 测试 M 列表（包含搜索过的和未搜索的）
    test_M_list = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 6000, 8192, 12000, 16384]
    
    for nk_key, m_configs in saved["configs"].items():
        N, K = eval(nk_key)  # "(N,K)" -> (N, K)
        
        print("=" * 80)
        print(f"测试 {nk_key}")
        print("=" * 80)
        
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        W_q = quantize_tensor(W, dtype)
        
        # 找到最大 M 的配置（应该是 ws=0）
        max_M = max(int(m) for m in m_configs.keys())
        max_cfg = m_configs[str(max_M)]
        max_algo_data = base64.b64decode(max_cfg["algo_data_b64"])
        
        print(f"\n使用 M={max_M} 的配置 (split_k={max_cfg['split_k']}, ws={max_cfg['workspace']})")
        print()
        
        # 测试所有 M
        print("[测试 1] 使用最大M配置执行各种M:")
        for M_exec in test_M_list:
            A = torch.randn(M_exec, K, device="cuda", dtype=torch.bfloat16)
            A_q = quantize_tensor(A, dtype)
            
            try:
                out = call_gemm(gemm_lib, W_q, A_q, N, K, M_exec,
                               max_algo_data, max_cfg['workspace'])
                searched = "已搜索" if str(M_exec) in m_configs else "未搜索"
                print(f"  M={M_exec:5d}: ✓ ({searched})")
            except Exception as e:
                print(f"  M={M_exec:5d}: ✗ ({str(e)[:40]})")
        
        # 测试 ws>0 的配置（如果有）
        ws_positive_configs = {m: c for m, c in m_configs.items() if c['workspace'] > 0}
        
        if ws_positive_configs:
            print(f"\n[测试 2] ws>0 配置的跨M复用:")
            
            # 取最小的 ws>0 配置
            min_ws_M = min(int(m) for m in ws_positive_configs.keys())
            min_ws_cfg = ws_positive_configs[str(min_ws_M)]
            min_ws_algo = base64.b64decode(min_ws_cfg["algo_data_b64"])
            
            print(f"使用 M={min_ws_M} 的配置 (split_k={min_ws_cfg['split_k']}, ws={min_ws_cfg['workspace']})")
            
            for M_exec in test_M_list:
                A = torch.randn(M_exec, K, device="cuda", dtype=torch.bfloat16)
                A_q = quantize_tensor(A, dtype)
                
                try:
                    out = call_gemm(gemm_lib, W_q, A_q, N, K, M_exec,
                                   min_ws_algo, min_ws_cfg['workspace'])
                    status = "✓"
                except:
                    status = "✗"
                
                relation = "<=" if M_exec <= min_ws_M else ">"
                print(f"  M={M_exec:5d} {relation} {min_ws_M}: {status}")
    
    print()
    print("=" * 80)
    print("离线测试完成")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="离线场景测试")
    parser.add_argument("--phase", choices=["search", "exec", "both"], default="both",
                       help="运行阶段: search=搜索, exec=执行, both=两者都运行")
    args = parser.parse_args()
    
    if args.phase == "search":
        phase_search()
    elif args.phase == "exec":
        phase_exec()
    else:
        print("运行两个阶段（同一进程，仅供快速验证）...")
        print("注意：完整测试建议分两个进程运行\n")
        phase_search()
        print("\n" + "=" * 80 + "\n")
        phase_exec()


if __name__ == "__main__":
    main()
