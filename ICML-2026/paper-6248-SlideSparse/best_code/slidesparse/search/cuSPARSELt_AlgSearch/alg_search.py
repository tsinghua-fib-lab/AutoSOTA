#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
cuSPARSELt 算法离线搜索

架构说明：
=========
- Python 端：负责外层 NK 循环、参数解析、GPU 检测、数据生成、结果落盘
- C++ 端：负责内层 M 循环、算法枚举、cuSPARSELt API 调用、精确计时

2:4 稀疏矩阵乘法 (SpMM):
- 权重 W 进行 2:4 剪枝后压缩
- 固定 Layout: T/N + Col/Col + Col

搜索策略:
- 自适应 Split-K 倍增策略 (1, 2, 4, 8, ...)
- Segment-K 测试 (SM90+ 支持 split_k=-1)
- 官方 API 搜索对比 (cusparseLtMatmulSearch)
- 每个 alg_id 独立压缩权重

运行示例:
    python3 alg_search.py --dtype int8 --outdtype bf16 --model Qwen2.5-0.5B-INT8
    python3 alg_search.py --dtype fp8e4m3 --outdtype bf16 --model Qwen2.5-0.5B-FP8
"""

import argparse
import ctypes
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch

# 添加 search 目录到路径
SCRIPT_DIR = Path(__file__).parent.absolute()
SEARCH_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SEARCH_DIR))

from utils import (
    hw_info,
    # 编译与加载
    build_search_extension,
    load_search_extension,
    # 模型 NK 工具
    get_nk_list_for_search,
    # 数据准备
    quantize_int8,
    to_fp8_e4m3,
    get_output_torch_dtype,
    # 结果保存
    save_alg_search_results,
    # 验证
    verify_gemm_result,
    # 常量和验证
    SUPPORTED_DTYPES,
    SUPPORTED_OUTDTYPES,
    validate_dtype_outdtype_combination,
    get_default_outdtype,
    default_m_list,
)


# =============================================================================
# CUDA 扩展加载
# =============================================================================

def setup_lib_signatures(lib: ctypes.CDLL) -> None:
    """设置 CUDA 扩展的函数签名"""
    lib.cusparselt_search_single_m.argtypes = [
        ctypes.c_void_p,   # W_pruned_ptr
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
        ctypes.c_int,      # test_segment_k
        ctypes.c_int,      # do_api_search
        # 输出
        ctypes.POINTER(ctypes.c_int),        # out_alg_ids
        ctypes.POINTER(ctypes.c_int),        # out_split_k
        ctypes.POINTER(ctypes.c_float),      # out_lat_us
        ctypes.POINTER(ctypes.c_float),      # out_tops
        ctypes.POINTER(ctypes.c_int64),      # out_workspace
        ctypes.POINTER(ctypes.c_uint8),      # out_valid
        ctypes.POINTER(ctypes.c_int),        # out_num_valid
        ctypes.POINTER(ctypes.c_int),        # out_alg_count
        ctypes.POINTER(ctypes.c_int),        # out_config_count
        ctypes.POINTER(ctypes.c_int),        # out_api_alg_id
        ctypes.POINTER(ctypes.c_int),        # out_api_split_k
        ctypes.POINTER(ctypes.c_float),      # out_api_lat_us
        ctypes.POINTER(ctypes.c_int),        # out_api_rank
        ctypes.c_void_p,   # stream
    ]
    lib.cusparselt_search_single_m.restype = ctypes.c_int
    
    lib.cusparselt_prune_24.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_char_p,
        ctypes.c_void_p,
    ]
    lib.cusparselt_prune_24.restype = ctypes.c_int
    
    lib.cusparselt_compress.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_char_p,
        ctypes.c_void_p,
    ]
    lib.cusparselt_compress.restype = ctypes.c_int64
    
    lib.cusparselt_get_compressed_size.argtypes = [
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_char_p,
    ]
    lib.cusparselt_get_compressed_size.restype = ctypes.c_int64
    
    lib.cusparselt_supports_segment_k.argtypes = []
    lib.cusparselt_supports_segment_k.restype = ctypes.c_int
    
    lib.cusparselt_alg_search_is_available.argtypes = []
    lib.cusparselt_alg_search_is_available.restype = ctypes.c_int
    
    lib.cusparselt_alg_search_get_last_error.argtypes = []
    lib.cusparselt_alg_search_get_last_error.restype = ctypes.c_char_p


# =============================================================================
# 数据准备 (cuSPARSELt 特定的压缩流程)
# =============================================================================

def prepare_and_prune_weight(
    lib: ctypes.CDLL,
    W_bf16: torch.Tensor,
    dtype: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    准备并剪枝权重矩阵。
    
    返回:
        (W_pruned, W_compressed, W_q): 
            W_pruned: 剪枝后的矩阵 [K, N]
            W_compressed: 压缩后的矩阵
            W_q: 原始量化权重 [N, K]（用于 verify）
    """
    N, K = W_bf16.shape
    
    # 量化
    if dtype == "int8":
        W_q, _ = quantize_int8(W_bf16)
    elif dtype == "fp8e4m3":
        W_q = to_fp8_e4m3(W_bf16)
    else:
        raise ValueError(f"不支持的数据类型: {dtype}")
    
    # 转置为 K x N (列主序存储)。保持转置视图的列主序 stride，避免 contiguous 破坏布局
    W_t = W_q.t()
    
    # Prune 2:4
    W_pruned = torch.empty_like(W_t)
    ret = lib.cusparselt_prune_24(
        W_t.data_ptr(),
        W_pruned.data_ptr(),
        K, N,
        dtype.encode(),
        None,
    )
    if ret != 0:
        error = lib.cusparselt_alg_search_get_last_error()
        raise RuntimeError(f"Prune 失败: {error.decode() if error else 'unknown'}")
    
    torch.cuda.synchronize()
    
    # 获取压缩大小
    compressed_size = lib.cusparselt_get_compressed_size(K, N, dtype.encode())
    if compressed_size < 0:
        raise RuntimeError("获取压缩大小失败")
    
    # 压缩
    W_compressed = torch.empty(compressed_size, dtype=torch.uint8, device=W_t.device)
    ret = lib.cusparselt_compress(
        W_pruned.data_ptr(),
        W_compressed.data_ptr(),
        K, N,
        dtype.encode(),
        None,
    )
    if ret < 0:
        error = lib.cusparselt_alg_search_get_last_error()
        raise RuntimeError(f"Compress 失败: {error.decode() if error else 'unknown'}")
    
    torch.cuda.synchronize()
    
    # 返回剪枝后权重 [K, N] 的转置 [N, K] 作为 W_q 用于 verify
    # 注意：剪枝会改变权重值（将某些元素置零），verify 需要使用剪枝后的权重
    W_q_pruned = W_pruned.t().contiguous()  # [N, K]
    
    return W_pruned, W_compressed, W_q_pruned


def prepare_activation(
    A_bf16: torch.Tensor,
    dtype: str,
    memory_efficient: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    准备激活矩阵。
    
    返回:
        (A_transposed, A_q):
            A_transposed: 转置后的激活 [K, M] (用于 CUDA)
            A_q: 原始量化激活 [M, K] (用于 verify)
    
    Args:
        A_bf16: 输入激活矩阵 [M, K]
        dtype: 目标数据类型
        memory_efficient: 启用显存优化模式（会修改输入张量）
    """
    if dtype == "int8":
        # 使用 inplace 模式减少显存占用
        A_q, _ = quantize_int8(A_bf16, inplace=memory_efficient)
    elif dtype == "fp8e4m3":
        A_q = to_fp8_e4m3(A_bf16)
    else:
        raise ValueError(f"不支持的数据类型: {dtype}")
    
    # 转置为 K x M (列主序)
    A_transposed = A_q.t()
    
    return A_transposed, A_q


# =============================================================================
# 搜索核心
# =============================================================================

def smart_score(lat_us: float, workspace: int, split_k: int) -> float:
    """
    计算算法的综合评分 (Score)，越低越好。
    引入了三段式显存惩罚和查表式 Split-K 惩罚。
    
    Score = Latency * (1 + WS_Penalty + SK_Penalty)
    
    设计理念：
    - Workspace: 小显存无感，中显存敏感，大显存拒绝
    - Split-K: 离散风险定价，SK=2/4 低风险，SK>=16 高风险
    """
    # Segment-K (-1) 视为 Split-K=1 (无额外调度风险)
    effective_sk = 1 if split_k == -1 else split_k
    ws_mb = workspace / (1024 * 1024)

    # === Workspace 惩罚模型 (三段式) ===
    ws_penalty = 0.0
    SAFE_LIMIT = 16.0    # 16MB 以内：安全区 (L2 Cache 级别)
    HARD_LIMIT = 256.0   # 256MB 以上：高危区 (严重挤占 KV Cache)
    
    if ws_mb <= SAFE_LIMIT:
        # [安全区]：完全无惩罚
        ws_penalty = 0.0
    elif ws_mb <= HARD_LIMIT:
        # [敏感区]：线性增长，每增加 10MB 性能要求提升 0.5%
        ws_penalty = (ws_mb - SAFE_LIMIT) * 0.0005  # 0.05% per MB
    else:
        # [高危区]：指数爆炸，几乎只有性能翻倍才能抵消
        base_penalty = (HARD_LIMIT - SAFE_LIMIT) * 0.0005
        excess = ws_mb - HARD_LIMIT
        ws_penalty = base_penalty + (excess * 0.005) + (excess ** 2) * 0.00001

    # === Split-K 惩罚模型 (风险定价表) ===
    # 手动定义每个档位的风险溢价
    SK_RISK_TABLE = {
        1:  0.00,   # 基准
        2:  0.01,   # 1%：几乎无风险
        4:  0.03,   # 3%：需要有可见提升
        8:  0.08,   # 8%：调度风险开始显著
        16: 0.20,   # 20%：高风险，除非性能提升巨大 (1.2x)
        32: 0.50,   # 50%：极高风险
        64: 1.00    # 100%：除非性能翻倍
    }
    sk_penalty = SK_RISK_TABLE.get(effective_sk, 0.5)

    # === 综合打分 ===
    return lat_us * (1.0 + ws_penalty + sk_penalty)


def rerank_candidates(
    m_results: Dict[int, Dict[str, Any]],
    m_list: List[int],
    final_topk: int = 3,
    max_latency_tolerance: float = 0.025,
) -> Dict[int, Dict[str, Any]]:
    """
    对搜索结果进行重排序，综合考虑 latency、workspace、split_k。
    
    采用 "Filter then Sort" 策略：
    1. 对每个 M，找到最优延时，过滤掉超过容忍度的候选（性能护栏）
    2. 在通过护栏的候选中，用 smart_score 排序选最稳的
    3. 只保留 final_topk 个结果
    
    Args:
        m_results: 每个 M 的搜索结果 {M: {"results": [...], ...}}
        m_list: M 列表
        final_topk: 最终保留的候选数，默认 3
        max_latency_tolerance: 最大延时容忍度，默认 2.5%
            即不接受比最优解慢 2.5% 以上的配置
    
    Returns:
        重排序后的结果字典，格式与输入相同
    """
    new_m_results = {}
    
    for M in m_list:
        if M not in m_results:
            continue
        
        m_res = m_results[M]
        candidates = m_res.get("results", [])
        
        if not candidates:
            new_m_results[M] = m_res
            continue
        
        # 1. 找到最优延时
        best_raw_lat = min(c["lat_us"] for c in candidates)
        
        # 2. 【性能护栏】过滤掉延时超过容忍度的候选
        limit_lat = best_raw_lat * (1.0 + max_latency_tolerance)
        filtered = [c for c in candidates if c["lat_us"] <= limit_lat]
        
        # 3. 用 smart_score 排序
        filtered.sort(key=lambda c: smart_score(c["lat_us"], c["workspace"], c["split_k"]))
        
        # 只保留 final_topk 个
        top_candidates = filtered[:final_topk]
        
        new_m_results[M] = {
            "results": top_candidates,
            "num_valid": len(top_candidates),
            "alg_count": m_res.get("alg_count", 0),
        }
    
    return new_m_results


def search_single_nk(
    lib: ctypes.CDLL,
    N: int, K: int, M: int,
    W_pruned: torch.Tensor,
    A_transposed: torch.Tensor,
    dtype: str,
    outdtype: str,
    warmup: int,
    repeat: int,
    topk: int = 3,
    test_segment_k: bool = True,
    do_api_search: bool = True,
    verify: bool = False,
    W_q_for_verify: Optional[torch.Tensor] = None,
    A_q_for_verify: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """搜索单个 (N, K, M) 组合的最佳算法"""
    # 分配输出缓冲
    R_torch_dtype = get_output_torch_dtype(outdtype)
    # Column Major [N, M]：直接用列主序 stride 分配，避免行主序写入错位
    R_out = torch.empty_strided((N, M), (1, N), dtype=R_torch_dtype, device=A_transposed.device)
    R_out.zero_()
    
    # 分配输出数组
    out_alg_ids = (ctypes.c_int * topk)()
    out_split_k = (ctypes.c_int * topk)()
    out_lat_us = (ctypes.c_float * topk)()
    out_tops = (ctypes.c_float * topk)()
    out_workspace = (ctypes.c_int64 * topk)()
    out_valid = (ctypes.c_uint8 * topk)()
    out_num_valid = ctypes.c_int(0)
    out_alg_count = ctypes.c_int(0)
    out_config_count = ctypes.c_int(0)
    out_api_alg_id = ctypes.c_int(-1)
    out_api_split_k = ctypes.c_int(1)
    out_api_lat_us = ctypes.c_float(0.0)
    out_api_rank = ctypes.c_int(-1)
    
    # 调用 C 函数 (传入剪枝后的 W_pruned, CUDA 内部会对每个 alg_id 重新压缩)
    ret = lib.cusparselt_search_single_m(
        W_pruned.data_ptr(),
        A_transposed.data_ptr(),
        R_out.data_ptr(),
        N, K, M,
        dtype.encode(),
        outdtype.encode(),
        warmup,
        repeat,
        topk,
        1 if test_segment_k else 0,
        1 if do_api_search else 0,
        out_alg_ids,
        out_split_k,
        out_lat_us,
        out_tops,
        out_workspace,
        out_valid,
        ctypes.byref(out_num_valid),
        ctypes.byref(out_alg_count),
        ctypes.byref(out_config_count),
        ctypes.byref(out_api_alg_id),
        ctypes.byref(out_api_split_k),
        ctypes.byref(out_api_lat_us),
        ctypes.byref(out_api_rank),
        None,
    )
    
    if ret != 0:
        # 处理不支持的数据类型组合（如 INT8+FP32）
        # 返回空结果而不是抛异常
        return {
            "results": [],
            "num_valid": 0,
            "alg_count": 0,
            "config_count": 0,
            "api_result": None,
            "verify_result": None,
            "error": lib.cusparselt_alg_search_get_last_error().decode() if lib.cusparselt_alg_search_get_last_error() else "unknown error",
        }
    
    # 转换结果
    results = []
    for i in range(topk):
        if out_valid[i]:
            results.append({
                "alg_id": out_alg_ids[i],
                "split_k": out_split_k[i],
                "lat_us": out_lat_us[i],
                "tops": out_tops[i],
                "workspace": out_workspace[i],
            })
    
    # 验证正确性
    verify_result = None
    if verify and W_q_for_verify is not None and A_q_for_verify is not None:
        verify_result = verify_gemm_result(
            W_q=W_q_for_verify,
            A_q=A_q_for_verify,
            R_out=R_out,
            M=M,
            # R_out 已按列主序 stride 分配为 [N, M]，直接与参考对齐
            is_col_major=False,
        )
        if verify_result["critical"]:
            print(f"    [CRITICAL] M={M}: {verify_result['message']}")
        elif not verify_result["passed"]:
            print(f"    [WARN] M={M}: {verify_result['message']}")
    
    return {
        "results": results,
        "num_valid": out_num_valid.value,
        "alg_count": out_alg_count.value,
        "config_count": out_config_count.value,
        "api_result": {
            "alg_id": out_api_alg_id.value,
            "split_k": out_api_split_k.value,
            "lat_us": out_api_lat_us.value,
            "rank": out_api_rank.value,
        } if out_api_alg_id.value >= 0 else None,
        "verify_result": verify_result,
    }


def estimate_memory_mb(M: int, K: int, dtype: str) -> float:
    """估算单个 NK 搜索所需的峰值显存 (MB)"""
    # A 矩阵 (bf16): M × K × 2 bytes
    a_bf16_mb = (M * K * 2) / (1024 * 1024)
    # A 量化后: M × K × (1 或 1 bytes for int8/fp8)
    a_q_mb = (M * K * 1) / (1024 * 1024)
    # quantize_int8 中间结果 (如果非 inplace): M × K × 4 bytes (float32 temp)
    temp_mb = (M * K * 4) / (1024 * 1024) if dtype == "int8" else 0
    
    # 总估算 (带安全余量 1.5x)
    return (a_bf16_mb + a_q_mb + temp_mb) * 1.5


def get_adaptive_max_m(K: int, m_list: List[int], dtype: str, 
                       gpu_memory_limit_mb: Optional[float] = None) -> int:
    """
    根据 K 值和 GPU 可用显存，自适应计算最大可用的 M 值。
    
    Args:
        K: 当前 NK 的 K 维度
        m_list: M 值列表
        dtype: 数据类型
        gpu_memory_limit_mb: GPU 显存限制 (MB)，None 则自动检测
    
    Returns:
        安全的最大 M 值
    """
    if gpu_memory_limit_mb is None:
        # 获取当前可用显存，保留 2GB 余量
        free_mem = torch.cuda.mem_get_info()[0] / (1024 * 1024)
        gpu_memory_limit_mb = max(free_mem - 2048, 1024)  # 至少保留 1GB 可用
    
    # 从大到小尝试 M 值
    sorted_m = sorted(m_list, reverse=True)
    for m in sorted_m:
        if estimate_memory_mb(m, K, dtype) < gpu_memory_limit_mb:
            return m
    
    # 如果都不行，返回最小的 M
    return min(m_list)


def run_search(
    lib: ctypes.CDLL,
    dtype: str,
    outdtype: str,
    nk_list: List,
    m_list: List[int],
    warmup: int,
    repeat: int,
    topk: int = 3,
    test_segment_k: bool = True,
    do_api_search: bool = True,
    verify: bool = False,
    verbose: bool = True,
    memory_safe: bool = True,
) -> Dict:
    """
    运行完整的算法搜索。
    
    搜索策略：
    1. 收集更多候选 (search_topk=16) 用于重排序
    2. 使用 smart_score 综合考虑 latency/workspace/split_k
    3. 应用 rerank_candidates 进行重排序，保留 final_topk=3
    
    Args:
        memory_safe: 启用内存安全模式，自动限制大 K 时的 M 值
    """
    results = []
    max_M = max(m_list)
    total_nk = len(nk_list)
    
    max_alg_count = 0
    max_config_count = 0
    supports_segment_k = bool(lib.cusparselt_supports_segment_k())
    
    # 收集更多候选用于 rerank
    search_topk = 16
    
    # verify 统计
    verify_stats = {"total": 0, "passed": 0, "warned": 0, "critical": 0}
    
    for nk_id, nk in enumerate(nk_list):
        N, K = nk[0], nk[1]
        if verbose:
            print(f"    NK {nk_id+1}/{total_nk}: ({N}, {K})", flush=True)
        
        # 内存安全模式：自适应调整当前 NK 可用的最大 M
        if memory_safe:
            current_max_M = get_adaptive_max_m(K, m_list, dtype)
            if current_max_M < max_M and verbose:
                print(f"      [内存安全] K={K} 较大，限制 M_max: {max_M} -> {current_max_M}")
            effective_m_list = [m for m in m_list if m <= current_max_M]
        else:
            current_max_M = max_M
            effective_m_list = m_list
        
        # 生成随机数据
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        A = torch.randn(current_max_M, K, device="cuda", dtype=torch.bfloat16)
        
        # 剪枝权重 (不需要预压缩，CUDA 内部会对每个 alg_id 重新压缩)
        # W_q_pruned [N, K] 是剪枝后的权重，用于 verify
        W_pruned, _, W_q_pruned = prepare_and_prune_weight(lib, W, dtype)
        
        # 准备激活
        # A_transposed [K, M] 用于 CUDA, A_q [M, K] 用于 verify
        A_transposed, A_q = prepare_activation(A, dtype)
        
        nk_m_results = {}
        
        for M in effective_m_list:  # 使用自适应的 M 列表
            # 切片 (A_transposed 是 K x M_max, 切片得到 K x M)
            A_slice = A_transposed[:, :M]
            # verify 用的 A_q 切片 (M, K) -> (M_slice, K)
            A_q_slice = A_q[:M, :].contiguous() if verify else None
            
            # 搜索更多候选用于 rerank
            out = search_single_nk(
                lib, N, K, M,
                W_pruned, A_slice,
                dtype, outdtype,
                warmup, repeat, search_topk,
                test_segment_k,
                do_api_search,
                verify=verify,
                W_q_for_verify=W_q_pruned if verify else None,
                A_q_for_verify=A_q_slice,
            )
            
            nk_m_results[M] = out
            
            if out["alg_count"] > max_alg_count:
                max_alg_count = out["alg_count"]
            if out.get("config_count", 0) > max_config_count:
                max_config_count = out["config_count"]
            
            # 更新 verify 统计
            if verify and out.get("verify_result"):
                vr = out["verify_result"]
                verify_stats["total"] += 1
                if vr["critical"]:
                    verify_stats["critical"] += 1
                elif vr["passed"]:
                    verify_stats["passed"] += 1
                else:
                    verify_stats["warned"] += 1
        
        # 重排序：综合考虑 latency/workspace/split_k，保留 top3
        reranked_results = rerank_candidates(nk_m_results, effective_m_list, final_topk=topk)
        
        nk_results = {
            "nk_id": nk_id,
            "N": N,
            "K": K,
            "m_results": reranked_results,
            "effective_m_list": effective_m_list,  # 记录实际使用的 M 列表
        }
        
        if verbose:
            first_m = effective_m_list[0] if effective_m_list else m_list[0]
            first_result = reranked_results.get(first_m, {})
            alg_count = first_result.get("alg_count", 0)
            num_valid = first_result.get("num_valid", 0)
            print(f"      → 算法数: {alg_count}, 有效: {num_valid} (rerank 后)")
        
        results.append(nk_results)
        
        # 及时清理显存，避免累积
        del W, A, W_pruned, A_transposed, A_q
        if verify:
            del W_q_pruned
        torch.cuda.empty_cache()  # 每个 NK 后立即清理
    
    torch.cuda.empty_cache()
    
    # 打印 verify 汇总
    if verify and verbose:
        print()
        print(f"    验证统计: 总计={verify_stats['total']}, "
              f"通过={verify_stats['passed']}, "
              f"警告={verify_stats['warned']}, "
              f"严重错误={verify_stats['critical']}")
    
    return {
        "dtype": dtype,
        "outdtype": outdtype,
        "results": results,
        "M_list": m_list,
        "NK_list": nk_list,
        "max_alg_count": max_alg_count,
        "max_config_count": max_config_count,
        "supports_segment_k": supports_segment_k,
        "test_segment_k": test_segment_k,
        "do_api_search": do_api_search,
    }


# =============================================================================
# 主流程
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="cuSPARSELt 算法离线搜索")
    p.add_argument("--dtype", default="int8", choices=SUPPORTED_DTYPES, help="输入数据类型")
    p.add_argument("--outdtype", default="bf16", choices=SUPPORTED_OUTDTYPES, help="输出数据类型")
    p.add_argument("--model", default=None, help="模型名称（如 Qwen2.5-0.5B-INT8）或路径，必须与 checkpoints/ 目录下的文件夹名匹配。不指定则使用 BitNet-2B-BF16 默认配置")
    p.add_argument("--Lmax", type=int, default=None, help="最大 L 值（slide sparse），会为 L=4,6,...,Lmax 生成所有 NK")
    p.add_argument("--M-quick", action="store_true", dest="m_quick", help="M-quick 模式: 使用固定 M 列表 [16, 128, 1024, 4096, 16384]")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--repeat", type=int, default=100)
    p.add_argument("--verify", action="store_true", help="开启正确性校验")
    p.add_argument("--compile", action="store_true", help="强制重新编译 CUDA 扩展")
    p.add_argument("--no_segment_k", action="store_true", help="禁用 Segment-K 测试")
    p.add_argument("--no_api_search", action="store_true", help="禁用官方 API 搜索对比")
    p.add_argument("--out_dir", default=None, help="输出目录")
    p.add_argument("--m_list", type=str, default=None, help="M 列表，逗号分隔")
    return p.parse_args()


def main():
    args = parse_args()
    
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 环境")
    
    # 检查 FP8 硬件支持（CC >= 8.9，Ada/Hopper+）
    if args.dtype in ("fp8e4m3", "fp8") and not hw_info.supports_fp8:
        raise RuntimeError(
            f"GPU {hw_info.gpu_name} ({hw_info.cc_tag}) 不支持原生 FP8 GEMM。\n"
            f"FP8 需要 CC >= 8.9 (Ada Lovelace / Hopper / Blackwell)，"
            f"当前 GPU 为 {hw_info.arch_name} (CC {hw_info.cc_major}.{hw_info.cc_minor})。\n"
            f"请使用 --dtype int8 或在支持 FP8 的 GPU 上运行。"
        )
    
    # 验证并获取实际使用的 outdtype
    # cuSPARSELt INT8 支持 bf16 或 int32，不支持 fp32
    actual_outdtype = validate_dtype_outdtype_combination(
        args.dtype, args.outdtype, backend="cusparselt"
    )
    
    # 获取 NK 列表和模型名称（统一使用 get_nk_list_for_search）
    nk_list, model_name = get_nk_list_for_search(args.model, args.Lmax)
    
    test_segment_k = not args.no_segment_k
    do_api_search = not args.no_api_search
    
    print("=" * 60)
    print("cuSPARSELt 算法离线搜索 (2:4 稀疏)")
    print("=" * 60)
    print(f"GPU: {hw_info.gpu_full_name} ({hw_info.cc_tag}, {hw_info.arch_name})")
    print(f"模型: {model_name}")
    print(f"参数: dtype={args.dtype}, outdtype={actual_outdtype}")
    print(f"Segment-K 测试: {'开启' if test_segment_k else '关闭'}")
    print(f"API 搜索对比: {'开启' if do_api_search else '关闭'}")
    print()
    
    # 输出目录 (脚本所在目录下)
    out_dir = Path(args.out_dir) if args.out_dir else SCRIPT_DIR / "alg_search_results"
    
    print("[1/4] 编译 CUDA 扩展...")
    src_path = SCRIPT_DIR / "alg_search_cusparselt.cu"
    build_dir = SCRIPT_DIR / "build"
    so_path = build_search_extension(
        name="alg_search_cusparselt",
        source_file=src_path,
        build_dir=build_dir,
        backend="cusparselt",
        force=args.compile,
    )
    
    print("[2/4] 加载 CUDA 扩展...")
    lib = load_search_extension(so_path, backend="cusparselt", setup_func=setup_lib_signatures)
    
    if not lib.cusparselt_alg_search_is_available():
        raise RuntimeError("cuSPARSELt 不可用")
    print("✓ cuSPARSELt 可用")
    
    supports_segment_k = bool(lib.cusparselt_supports_segment_k())
    print(f"✓ Segment-K 支持: {'是' if supports_segment_k else '否'}")
    
    if args.Lmax:
        print(f"Lmax: {args.Lmax} (slide sparse L=4,6,...,{args.Lmax})")
    
    # 获取 M 列表
    if args.m_quick:
        m_list = [16, 128, 1024, 4096, 16384]
    elif args.m_list:
        m_list = [int(x.strip()) for x in args.m_list.split(",")]
    else:
        m_list = default_m_list()
    
    print()
    print(f"[3/4] 开始算法搜索...")
    print(f"      NK 组合: {len(nk_list)} 个, M 列表: {m_list}")
    print()
    
    ret = run_search(
        lib,
        args.dtype,
        actual_outdtype,  # 使用实际的 outdtype
        nk_list,
        m_list,
        args.warmup,
        args.repeat,
        topk=3,
        test_segment_k=test_segment_k,
        do_api_search=do_api_search,
        verify=args.verify,
        verbose=True,
    )
    
    saved_dir = save_alg_search_results(
        out_dir,
        model_name,
        args.dtype,
        actual_outdtype,  # 使用实际的 outdtype
        ret,
        args.warmup,
        args.repeat,
        args.verify,
        layout="TNCCcol_sparse24",
        is_sparse=True,
        has_split_k=True,
    )
    
    print()
    print(f"[4/4] 完成! 结果已保存到:")
    print(f"      - {saved_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
