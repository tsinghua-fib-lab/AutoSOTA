#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试脚本 v5：深入分析 cublasLtMatmulAlgo_t 的复用边界

核心假设验证：
1. algo_data 是否绑定到特定的 M 值？
2. algo_data 是否绑定到特定的 N/K 值？
3. alg_id（算法索引）是否可以跨 M 复用？
4. algo_data 内部结构是什么？哪些字段是固定的？

测试策略：
- 固定 N, K，变化 M：测试 M 维度的兼容性边界
- 使用 alg_id 而不是完整 algo_data：验证 ID 是否可复用
- 打印 algo_data 内部结构对比
"""

import base64
import ctypes
import json
import struct
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

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
    """设置搜索扩展的函数签名"""
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
    
    lib.cublaslt_alg_search_is_available.argtypes = []
    lib.cublaslt_alg_search_is_available.restype = ctypes.c_int
    
    lib.cublaslt_alg_search_get_last_error.argtypes = []
    lib.cublaslt_alg_search_get_last_error.restype = ctypes.c_char_p


def setup_lib_signatures_gemm(lib: ctypes.CDLL) -> None:
    """设置 GEMM 扩展的函数签名"""
    lib.cublaslt_gemm_get_last_error.argtypes = []
    lib.cublaslt_gemm_get_last_error.restype = ctypes.c_char_p
    
    gemm_sig = ([ctypes.c_void_p] * 3 + 
               [ctypes.c_int64] * 3 + 
               [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p])
    lib.cublaslt_fp8_mm.argtypes = gemm_sig
    lib.cublaslt_fp8_mm.restype = ctypes.c_int
    
    lib.cublaslt_is_available.argtypes = []
    lib.cublaslt_is_available.restype = ctypes.c_int


def search_and_get_algo(search_lib, W_q, A_q, N, K, M, dtype, outdtype) -> Optional[Dict]:
    """执行搜索并返回 algo_data"""
    R_torch_dtype = get_output_torch_dtype(outdtype)
    R_out = torch.zeros(M, N, dtype=R_torch_dtype, device=W_q.device)
    
    topk = 10  # 获取更多算法以便分析
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
        W_q.data_ptr(), A_q.data_ptr(), R_out.data_ptr(),
        N, K, M,
        dtype.encode(), outdtype.encode(),
        3, 10, topk,  # warmup, repeat, topk
        out_alg_ids, out_lat_us, out_tops, out_workspace,
        out_waves_count, out_algo_data, out_valid,
        ctypes.byref(out_num_valid), ctypes.byref(out_alg_count),
        None,
    )
    
    if ret != 0:
        error = search_lib.cublaslt_alg_search_get_last_error()
        raise RuntimeError(f"搜索失败: {error.decode() if error else 'unknown error'}")
    
    # 收集所有有效算法
    results = []
    for i in range(topk):
        if out_valid[i]:
            algo_bytes = bytes(out_algo_data[i*64:(i+1)*64])
            results.append({
                "alg_id": out_alg_ids[i],
                "algo_data": algo_bytes,
                "workspace": out_workspace[i],
                "lat_us": out_lat_us[i],
            })
    
    if results:
        return {
            "best": results[0],
            "all": results,
            "total_count": out_alg_count.value,
        }
    return None


def call_gemm_with_algo(gemm_lib, W_q, A_q, N, K, M, inner_dtype, algo_data, workspace):
    """使用指定的 algo_data 执行 GEMM"""
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


def analyze_algo_data(algo_data: bytes, label: str = "") -> Dict:
    """分析 algo_data 的内部结构"""
    # cublasLtMatmulAlgo_t 是一个 64 字节的不透明结构
    # 根据 cuBLAS 文档，前几个字节可能包含：
    # - alg_id (算法索引)
    # - tile 配置
    # - split-K 信息
    # - reduction scheme
    
    # 尝试解析为多种格式
    result = {
        "label": label,
        "raw_hex": algo_data.hex(),
        "raw_hex_short": algo_data[:32].hex() + "...",
    }
    
    # 尝试解析前 16 个 int32
    ints = struct.unpack("<16I", algo_data)
    result["as_uint32"] = ints[:8]  # 只显示前 8 个
    
    # 尝试解析前 8 个 int64
    longs = struct.unpack("<8Q", algo_data)
    result["as_uint64"] = longs[:4]  # 只显示前 4 个
    
    return result


def print_algo_comparison(algo1: Dict, algo2: Dict, label1: str, label2: str):
    """对比两个 algo_data"""
    print(f"\n{'='*60}")
    print(f"对比: {label1} vs {label2}")
    print(f"{'='*60}")
    
    data1 = algo1["algo_data"]
    data2 = algo2["algo_data"]
    
    # 逐字节对比
    diff_count = 0
    diff_positions = []
    for i in range(64):
        if data1[i] != data2[i]:
            diff_count += 1
            diff_positions.append(i)
    
    print(f"alg_id: {algo1['alg_id']} vs {algo2['alg_id']}")
    print(f"workspace: {algo1['workspace']} vs {algo2['workspace']}")
    print(f"差异字节数: {diff_count}/64")
    
    if diff_count > 0 and diff_count <= 20:
        print(f"差异位置: {diff_positions}")
        for pos in diff_positions[:10]:
            print(f"  字节 {pos}: 0x{data1[pos]:02x} -> 0x{data2[pos]:02x}")
    
    # 解析并对比
    a1 = analyze_algo_data(data1, label1)
    a2 = analyze_algo_data(data2, label2)
    
    print(f"\n{label1} uint32[0:8]: {a1['as_uint32']}")
    print(f"{label2} uint32[0:8]: {a2['as_uint32']}")


def main():
    print("=" * 70)
    print("cuBLASLt algo_data 深入分析 v5")
    print("=" * 70)
    print(f"GPU: {hw_info.gpu_full_name} ({hw_info.cc_tag})")
    print()
    
    dtype = "fp8e4m3"
    outdtype = "bf16"
    
    # === 加载扩展 ===
    print("[1] 加载扩展...")
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
    
    gemm_so = SCRIPT_DIR.parent / "csrc" / "cublaslt_gemm" / "build"
    so_files = list(gemm_so.glob("cublaslt_gemm*.so"))
    if not so_files:
        print("    [ERROR] GEMM 库不存在")
        return
    gemm_lib = ctypes.CDLL(str(so_files[0]))
    setup_lib_signatures_gemm(gemm_lib)
    print("    ✓ 扩展加载完成")
    
    # === 准备测试数据 ===
    print("\n[2] 准备测试数据...")
    max_M = 16384
    N, K = 3072, 2048  # 固定 NK
    
    W_full = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    A_full = torch.randn(max_M, K, device="cuda", dtype=torch.bfloat16)
    W_q = quantize_tensor(W_full, dtype)
    A_q_full = quantize_tensor(A_full, dtype)
    
    print(f"    W_q: {W_q.shape}, N={N}, K={K}")
    print(f"    A_q: {A_q_full.shape}")
    
    # === 实验 1：不同 M 值的 algo_data 对比 ===
    print("\n" + "=" * 70)
    print("[实验 1] 不同 M 值搜索的 algo_data 内部结构对比")
    print("=" * 70)
    
    test_ms = [16, 128, 1024, 4096, 8192, 16384]
    algo_by_m = {}
    
    for M in test_ms:
        A_q = A_q_full[:M, :].contiguous()
        result = search_and_get_algo(search_lib, W_q, A_q, N, K, M, dtype, outdtype)
        if result:
            algo_by_m[M] = result["best"]
            print(f"  M={M:5d}: alg_id={result['best']['alg_id']}, workspace={result['best']['workspace']}, "
                  f"总算法数={result['total_count']}, 有效数={len(result['all'])}")
    
    # 对比不同 M 的 algo_data
    if len(algo_by_m) >= 2:
        ms = sorted(algo_by_m.keys())
        print_algo_comparison(algo_by_m[ms[0]], algo_by_m[ms[-1]], f"M={ms[0]}", f"M={ms[-1]}")
        
        if len(ms) >= 3:
            print_algo_comparison(algo_by_m[ms[1]], algo_by_m[ms[2]], f"M={ms[1]}", f"M={ms[2]}")
    
    # === 实验 2：用 M=X 搜索的 algo_data 执行 M=Y ===
    print("\n" + "=" * 70)
    print("[实验 2] 交叉执行测试：用 M=X 的 algo_data 执行 M=Y")
    print("=" * 70)
    
    # 使用 M=1024 搜索的配置
    M_search = 1024
    if M_search in algo_by_m:
        algo_1024 = algo_by_m[M_search]
        print(f"使用 M={M_search} 搜索的配置 (alg_id={algo_1024['alg_id']})")
        
        results = []
        for M_exec in [16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
            A_q_exec = A_q_full[:M_exec, :].contiguous()
            
            try:
                out = call_gemm_with_algo(
                    gemm_lib, W_q, A_q_exec, N, K, M_exec, "bf16",
                    algo_1024['algo_data'], algo_1024['workspace']
                )
                status = "✓"
                results.append((M_exec, True))
            except Exception as e:
                status = f"✗"
                results.append((M_exec, False))
            
            print(f"  执行 M={M_exec:5d}: {status}")
        
        # 分析失败模式
        success = [m for m, ok in results if ok]
        failed = [m for m, ok in results if not ok]
        print(f"\n  成功: {success}")
        print(f"  失败: {failed}")
    
    # === 实验 3：同 alg_id 不同 M 的 algo_data 差异 ===
    print("\n" + "=" * 70)
    print("[实验 3] 相同 alg_id 在不同 M 下的 algo_data 差异")
    print("=" * 70)
    
    # 找到 alg_id 相同的情况
    alg_id_groups = {}
    for M, algo in algo_by_m.items():
        aid = algo["alg_id"]
        if aid not in alg_id_groups:
            alg_id_groups[aid] = []
        alg_id_groups[aid].append((M, algo))
    
    for aid, group in alg_id_groups.items():
        if len(group) >= 2:
            print(f"\nalg_id={aid} 出现在 M={[m for m, _ in group]}")
            m1, algo1 = group[0]
            m2, algo2 = group[-1]
            print_algo_comparison(algo1, algo2, f"M={m1}", f"M={m2}")
    
    # === 实验 4：只传 alg_id 而不是完整 algo_data ===
    print("\n" + "=" * 70)
    print("[实验 4] 验证假设：如果搜索时不保存 algo_data，只保存 alg_id，在线重新搜索")
    print("=" * 70)
    
    # 模拟：保存 alg_id，在线时用实际 M 重新搜索，找到相同 alg_id 的 algo
    M_search = 1024
    M_exec_target = 8192
    
    if M_search in algo_by_m:
        saved_alg_id = algo_by_m[M_search]["alg_id"]
        print(f"离线保存: M={M_search} 的 alg_id={saved_alg_id}")
        
        # 在线：用实际 M 搜索，找到相同 alg_id 的 algo
        A_q_exec = A_q_full[:M_exec_target, :].contiguous()
        online_result = search_and_get_algo(search_lib, W_q, A_q_exec, N, K, M_exec_target, dtype, outdtype)
        
        if online_result:
            # 找 alg_id 匹配的
            matched = None
            for algo in online_result["all"]:
                if algo["alg_id"] == saved_alg_id:
                    matched = algo
                    break
            
            if matched:
                print(f"在线找到匹配 alg_id={saved_alg_id} 的 algo")
                
                # 用这个新的 algo_data 执行
                try:
                    out = call_gemm_with_algo(
                        gemm_lib, W_q, A_q_exec, N, K, M_exec_target, "bf16",
                        matched['algo_data'], matched['workspace']
                    )
                    print(f"  ✓ 使用在线搜索的 algo_data 执行 M={M_exec_target} 成功!")
                except Exception as e:
                    print(f"  ✗ 执行失败: {e}")
                
                # 对比离线和在线的 algo_data
                print_algo_comparison(algo_by_m[M_search], matched, 
                                     f"离线 M={M_search}", f"在线 M={M_exec_target}")
            else:
                print(f"  ⚠ 在线搜索未找到 alg_id={saved_alg_id}")
                print(f"  可用的 alg_ids: {[a['alg_id'] for a in online_result['all']]}")
    
    # === 实验 5：不传 algo_data（使用默认启发式）===
    print("\n" + "=" * 70)
    print("[实验 5] 默认启发式 vs 离线配置")
    print("=" * 70)
    
    for M in [1024, 4096, 8192]:
        A_q = A_q_full[:M, :].contiguous()
        
        # 使用默认启发式
        try:
            out_default = call_gemm_with_algo(
                gemm_lib, W_q, A_q, N, K, M, "bf16", None, 0
            )
            default_ok = True
        except Exception as e:
            default_ok = False
        
        # 使用离线配置（如果有）
        if M in algo_by_m:
            try:
                out_offline = call_gemm_with_algo(
                    gemm_lib, W_q, A_q, N, K, M, "bf16",
                    algo_by_m[M]['algo_data'], algo_by_m[M]['workspace']
                )
                offline_ok = True
            except Exception as e:
                offline_ok = False
        else:
            offline_ok = "N/A"
        
        print(f"  M={M:5d}: 默认启发式={'✓' if default_ok else '✗'}, "
              f"离线配置={('✓' if offline_ok == True else '✗') if offline_ok != 'N/A' else 'N/A'}")
    
    print("\n" + "=" * 70)
    print("分析完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
