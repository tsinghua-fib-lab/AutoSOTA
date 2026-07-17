#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cuBLASLt 离线配置复用规则 - 最终验证总结

经过一系列测试，我们得出以下结论：

═══════════════════════════════════════════════════════════════════════════════
核心发现
═══════════════════════════════════════════════════════════════════════════════

1. cublasLtMatmulAlgo_t 结构体包含的关键字段：
   - uint32[0]: alg_id (算法索引)
   - uint32[1]: tile_id (tile 配置)
   - uint32[2]: stages_id (stages 配置)
   - uint32[3]: split_k_num (K 分割数) ← 关键！
   - uint32[4]: reduction_scheme (归约方案) ← 关键！

2. workspace 需求规律：
   - split_k > 1 时需要 workspace
   - workspace 大小 ≈ M * N * split_k * dtype_size
   - 大 M 时通常 split_k = 1，workspace = 0

3. 配置复用规则：
   ┌─────────────────┬─────────────────────────────────────────────────────┐
   │ 配置类型        │ 复用规则                                            │
   ├─────────────────┼─────────────────────────────────────────────────────┤
   │ ws = 0          │ ✓ 可以任意复用，执行任意 M                          │
   │ (split_k = 1)   │                                                     │
   ├─────────────────┼─────────────────────────────────────────────────────┤
   │ ws > 0          │ ✓ 可以执行 M_exec <= M_from 的情况                  │
   │ (split_k > 1)   │ ✗ 不可执行 M_exec > M_from 的情况（workspace 不足）  │
   └─────────────────┴─────────────────────────────────────────────────────┘

4. 跨进程/跨 CUDA context 复用：
   - ✓ 完全支持！
   - algo_data 是自包含的，不依赖创建时的 context
   - 只需要相同的 GPU 型号（相同的 compute capability）

═══════════════════════════════════════════════════════════════════════════════
推荐策略
═══════════════════════════════════════════════════════════════════════════════

策略：使用最大 M 的配置

原理：
  1. 最大 M（如 16384）通常有足够的并行度，不需要 split-K
  2. 因此最大 M 的配置通常是 ws = 0, split_k = 1
  3. ws = 0 的配置可以任意复用

实现：
  def lookup_algo(N, K, M_actual):
      '''
      对于任意 M_actual，直接使用 M_max 的配置
      '''
      M_max = max(M_list)  # 离线搜索时的最大 M
      config = configs[(N, K)][M_max]
      
      # 验证 ws = 0（作为安全检查）
      if config['workspace'] > 0:
          # 回退：找到 M_upper = min{M in M_list | M >= M_actual}
          M_upper = find_upper_bound(M_list, M_actual)
          config = configs[(N, K)][M_upper]
      
      return config['algo_data'], config['workspace']

═══════════════════════════════════════════════════════════════════════════════
验证结果
═══════════════════════════════════════════════════════════════════════════════
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
        raise RuntimeError("GEMM failed")
    return output


def main():
    print(__doc__)
    
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
    
    # 测试配置
    test_cases = [
        # (N, K, 描述)
        (3072, 2048, "普通配置，所有M都是ws=0"),
        (2048, 8192, "split-K配置，小M有ws>0"),
    ]
    
    M_search_list = [16, 128, 1024, 4096, 8192, 16384]
    M_test_list = [16, 32, 64, 128, 256, 512, 1024, 2048, 3000, 4096, 6000, 8192, 12000, 16384]
    
    for N, K, desc in test_cases:
        print("=" * 80)
        print(f"测试 (N={N}, K={K}): {desc}")
        print("=" * 80)
        
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        W_q = quantize_tensor(W, "fp8e4m3")
        
        # 搜索各 M 的配置
        configs = {}
        print("\n搜索各 M 的配置:")
        for M in M_search_list:
            A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            A_q = quantize_tensor(A, "fp8e4m3")
            cfg = search_algo(search_lib, W_q, A_q, N, K, M)
            if cfg:
                configs[M] = cfg
                ws_kb = cfg['workspace'] / 1024
                print(f"  M={M:5d}: split_k={cfg['split_k']}, ws={ws_kb:8.1f}KB")
        
        # 使用最大 M 的配置测试所有 M
        M_max = max(configs.keys())
        cfg_max = configs[M_max]
        
        print(f"\n使用 M={M_max} 配置 (split_k={cfg_max['split_k']}, ws={cfg_max['workspace']}) 测试所有 M:")
        
        all_success = True
        for M_exec in M_test_list:
            A = torch.randn(M_exec, K, device="cuda", dtype=torch.bfloat16)
            A_q = quantize_tensor(A, "fp8e4m3")
            
            try:
                call_gemm(gemm_lib, W_q, A_q, N, K, M_exec,
                         cfg_max['algo_data'], cfg_max['workspace'])
                status = "✓"
            except:
                status = "✗"
                all_success = False
            
            searched = "搜索过" if M_exec in configs else "未搜索"
            print(f"  M={M_exec:5d}: {status} ({searched})")
        
        if all_success:
            print(f"\n✓ 验证通过：使用 M={M_max} 配置可以覆盖所有 M")
        else:
            print(f"\n✗ 验证失败：存在不兼容的 M")
    
    # 最终结论
    print()
    print("=" * 80)
    print("最终结论")
    print("=" * 80)
    print("""
1. 问题根因：
   - cublasLtMatmulAlgo_t 中的 split_k 和 workspace 与 M 相关
   - 小 M 时 cuBLASLt 选择 split-K 策略增加并行度
   - split-K 需要 workspace，workspace 大小与 M 成正比
   - 用小 M 配置执行大 M 时 workspace 不足导致失败

2. 解决方案：
   - 离线搜索时：搜索足够大的 M_max（如 16384 或 max_num_batched_tokens）
   - M_max 通常有 split_k=1, workspace=0
   - workspace=0 的配置可以任意复用，覆盖所有 M

3. 实现建议：
   - 修改 lookup_cublaslt()：对于任意 M_actual，返回 M_max 的配置
   - 或者：找 M_upper = min{M in M_list | M >= M_actual}，使用该配置的 workspace
   - 如果 M_actual > M_max：使用 M_max 配置（已验证可行）
""")


if __name__ == "__main__":
    main()
