#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cuSPARSELt 算法离线搜索

架构说明：
=========
- Python 端：负责外层 NK 循环、参数解析、GPU 检测、数据生成、结果落盘
- C++ 端：负责内层 M 循环、算法枚举、cuSPARSELt API 调用、精确计时

1. Python 控制外层 NK 循环，方便做进度条、异常捕获、断点续跑
2. C++ 控制内层 M 循环和算法枚举，避免跨进程通信开销
3. JSON 在 Python 端生成
4. torch.utils.cpp_extension.load 自动检测 GPU 架构，无需手动指定 -arch

固定 Layout:
- T/N + Col/Col + Col (权重W在左，稀疏矩阵)
- W[N,K]^T_col * A[K,M]_col = C[N,M]_col

运行示例:
CUDA_VISIBLE_DEVICES=0 python3 alg_search.py --dtype int8 --outdtype bf16 --model BitNet-2B4T --verify --compile
CUDA_VISIBLE_DEVICES=0 python3 alg_search.py --dtype fp8e4m3 --outdtype bf16 --model BitNet-2B4T --verify

CUDA_VISIBLE_DEVICES=0 python3 alg_search.py --dtype int8 --outdtype fp32 --model BitNet-2B4T --verify
CUDA_VISIBLE_DEVICES=0 python3 alg_search.py --dtype fp8e4m3 --outdtype fp32 --model BitNet-2B4T --verify

"""

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch

# 添加 test 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import (
    get_hw_info,
    get_normalize_dtype,
    load_cuda_extension,
    build_output_dir_name,
    build_result_filename,
    supports_segment_k,
    check_dtype_support,
    build_search_meta,
    build_csv_header,
    SUPPORTED_DTYPES,
    SUPPORTED_OUTDTYPES,
)

# 从 slidesparse.utils 导入查表函数（供运行时使用）
from slidesparse.utils import lookup_best_cusparselt_alg


# 注意: SUPPORTED_DTYPES 和 SUPPORTED_OUTDTYPES 从 utils 导入


# === 默认 NK/M 列表 ===

def default_nk_list() -> List[Tuple[int, int]]:
    return [
        (3840, 2560),  # Wqkv
        (2560, 2560),  # Wo
        (13824, 2560), # W13
        (2560, 6912),  # W2
    ]


def default_m_list() -> List[int]:
    # pow2 序列从 16 到 16384，覆盖 decode 和 prefill 的各种 batch size
    return [16, 64, 128, 512, 2048, 8192, 16384]


# 注意: lookup_best_cusparselt_alg 从 slidesparse.utils 导入


# === 重排序函数：综合考虑 latency/workspace/split_k ===

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


def rerank_candidates(raw_out: Dict, final_topk: int = 3, 
                      max_latency_tolerance: float = 0.025) -> Dict:
    """
    对搜索结果进行重排序，综合考虑 latency、workspace、split_k。
    
    采用 "Filter then Sort" 策略：
    1. 对每个 M，找到最优延时，过滤掉超过容忍度的候选（性能护栏）
    2. 在通过护栏的候选中，用 smart_score 排序选最稳的
    3. 只保留 final_topk 个结果
    
    Args:
        raw_out: search_topk 返回的原始结果
        final_topk: 最终保留的候选数，默认 3
        max_latency_tolerance: 最大延时容忍度，默认 2.5%
            即不接受比最优解慢 2.5% 以上的配置
    
    Returns:
        格式与输入相同，但 topk 维度变为 final_topk
    """
    # 获取原始数据
    topk_alg_id = raw_out["topk_alg_id"].cpu()
    topk_split_k = raw_out["topk_split_k"].cpu()
    topk_lat_us = raw_out["topk_lat_us"].cpu()
    topk_tops = raw_out["topk_tops"].cpu()
    topk_workspace = raw_out["topk_workspace"].cpu()
    valid_mask = raw_out["valid_mask"].cpu()
    
    numM, orig_topk = topk_alg_id.shape
    
    # 创建新的输出张量
    new_alg_id = torch.full((numM, final_topk), -1, dtype=torch.int32)
    new_split_k = torch.full((numM, final_topk), 1, dtype=torch.int32)
    new_lat_us = torch.zeros((numM, final_topk), dtype=torch.float32)
    new_tops = torch.zeros((numM, final_topk), dtype=torch.float32)
    new_workspace = torch.zeros((numM, final_topk), dtype=torch.int64)
    new_valid = torch.zeros((numM, final_topk), dtype=torch.uint8)
    new_num_valid = torch.zeros((numM,), dtype=torch.int32)
    
    for m_idx in range(numM):
        # 1. 收集该 M 下的所有有效候选，同时记录最优延时
        candidates = []
        best_raw_lat = float('inf')
        
        for k in range(orig_topk):
            if valid_mask[m_idx, k]:
                lat = float(topk_lat_us[m_idx, k].item())
                if lat < best_raw_lat:
                    best_raw_lat = lat
                    
                candidates.append({
                    'alg_id': int(topk_alg_id[m_idx, k].item()),
                    'split_k': int(topk_split_k[m_idx, k].item()),
                    'lat_us': lat,
                    'tops': float(topk_tops[m_idx, k].item()),
                    'workspace': int(topk_workspace[m_idx, k].item()),
                })
        
        if not candidates:
            continue
        
        # 2. 【性能护栏】过滤掉延时超过容忍度的候选
        # 只保留性能在 [最优, 最优 * (1 + tolerance)] 范围内的候选者
        limit_lat = best_raw_lat * (1.0 + max_latency_tolerance)
        candidates = [c for c in candidates if c['lat_us'] <= limit_lat]
        
        # 3. 在通过护栏的候选中，用 smart_score 选最"稳"的
        # 此时即便有惩罚，也只会在容忍度范围内权衡
        candidates.sort(key=lambda c: smart_score(c['lat_us'], c['workspace'], c['split_k']))
        
        # 只保留 final_topk 个
        top_candidates = candidates[:final_topk]
        
        # 填充结果
        for i, c in enumerate(top_candidates):
            new_alg_id[m_idx, i] = c['alg_id']
            new_split_k[m_idx, i] = c['split_k']
            new_lat_us[m_idx, i] = c['lat_us']
            new_tops[m_idx, i] = c['tops']
            new_workspace[m_idx, i] = c['workspace']
            new_valid[m_idx, i] = 1
        
        new_num_valid[m_idx] = len(top_candidates)
    
    # 构造新的输出字典
    new_out = {
        "M_list": raw_out["M_list"],
        "NK": raw_out["NK"],
        "topk_alg_id": new_alg_id,
        "topk_split_k": new_split_k,
        "topk_lat_us": new_lat_us,
        "topk_tops": new_tops,
        "topk_workspace": new_workspace,
        "valid_mask": new_valid,
        "alg_count": raw_out["alg_count"],        # 保留算法数量统计
        "config_count": raw_out["config_count"],  # 保留配置数量统计
        "num_valid_algs_per_M": new_num_valid,
        # 保留官方 API 搜索结果
        "api_alg_id": raw_out.get("api_alg_id"),
        "api_split_k": raw_out.get("api_split_k"),
        "api_lat_us": raw_out.get("api_lat_us"),
        "api_lat_rank": raw_out.get("api_lat_rank"),
        "api_total_configs": raw_out.get("api_total_configs"),
    }
    
    # 保留 verify 相关数据（如果有）
    if "verify_max_abs_err" in raw_out:
        new_out["verify_max_abs_err"] = raw_out["verify_max_abs_err"]
    if "verify_failed_algs" in raw_out:
        new_out["verify_failed_algs"] = raw_out["verify_failed_algs"]
    
    return new_out


# === 运行一次 layout 的搜索 ===

def run_search(ext, dtype: str, outdtype: str, nk_list: List[Tuple[int, int]], m_list: List[int],
               warmup: int, repeat: int, verify: bool, verbose: bool = True) -> Dict:
    """
    运行算法搜索。
    
    固定布局: T/N + Col/Col + Col (权重W在左，稀疏矩阵，Column Major 输出)
    每个 NK 组合生成新的随机数据
    """
    layout = "TNCCcol"  # 固定布局: TN+CC+Col
    results = []
    max_M = max(m_list)
    blacklist = []  # 不再需要屏蔽任何算法
    total_nk = len(nk_list)
    
    # 检测是否支持 Segment-K
    test_segment_k, segment_k_reason = supports_segment_k()
    if verbose:
        if test_segment_k:
            print(f"    [Segment-K] 当前架构支持 Segment-K，将测试 split_k=-1")
        else:
            print(f"    [Segment-K] {segment_k_reason}")

    for nk_id, (N, K) in enumerate(nk_list):
        if verbose:
            print(f"    NK {nk_id+1}/{total_nk}: ({N}, {K})", flush=True)
        
        # 生成随机数据（每个 NK 新分配，简洁明了）
        max_M = max(m_list)
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        A = torch.randn(max_M, K, device="cuda", dtype=torch.bfloat16)

        # 先做 2:4 剪枝
        W_pruned = ext.prune_24(W, layout)

        # 调用搜索（收集更多候选用于重排序）
        raw_out = ext.search_topk(
            W_pruned,
            A,
            m_list,
            layout,
            dtype,
            outdtype,
            warmup,
            repeat,
            verify,
            blacklist,
            16,  # 收集 16 个候选用于 rerank
            test_segment_k,  # 是否测试 Segment-K (split_k=-1)
        )
        
        # 获取算法统计信息
        alg_count = raw_out["alg_count"]  # 有效算法数量 (ID 范围 [0, alg_count))
        config_count = raw_out["config_count"]  # 实际测试的配置数 (alg_id × split_k 组合)
        
        # === 输出官方 API 搜索结果的延迟排名 ===
        # 在 rerank 之前显示，便于对比官方 API 与我们手动搜索的差异
        if verbose:
            api_lat_rank = raw_out["api_lat_rank"].cpu()
            api_alg_id = raw_out["api_alg_id"].cpu()
            api_split_k = raw_out["api_split_k"].cpu()
            api_total = raw_out["api_total_configs"].cpu()
            
            api_rank_strs = []
            for i, m in enumerate(m_list):
                rank = int(api_lat_rank[i].item())
                total = int(api_total[i].item())
                if rank > 0:
                    alg_id_val = int(api_alg_id[i].item())
                    split_k_val = int(api_split_k[i].item())
                    api_rank_strs.append(f"M={m}:{rank}/{total}")
                else:
                    api_rank_strs.append(f"M={m}:N/A")
            print(f"      [MatmulSearch] rank: {', '.join(api_rank_strs)}")
        
        # 重排序：综合考虑 latency/workspace/split_k，保留 top3
        out = rerank_candidates(raw_out, final_topk=3)
        
        if verbose:
            print(f"      → 算法数: {alg_count}，配置数: {config_count}")
        
        results.append({
            "nk_id": nk_id,
            "N": N,
            "K": K,
            "raw": out,
        })
        
        # 释放当前 NK 的张量
        del W, A, W_pruned
    
    torch.cuda.empty_cache()
    
    return {
        "dtype": dtype,
        "outdtype": outdtype,
        "results": results,
        "M_list": m_list,
        "NK_list": nk_list,
    }

# === 落盘工具 ===

def save_outputs(out_dir: Path, model_name: str, dtype: str, outdtype: str,
                 search_ret: Dict, warmup: int, repeat: int, verify: bool) -> Path:
    """
    保存搜索结果到 CSV 和 JSON 文件。
    
    文件夹命名: {GPU}_{cc}_{dtype}_{outdtype}_{model_name}
    文件命名: alg_search_cusparselt_{model_name}.csv, alg_search_cusparselt_{model_name}.json
    
    CSV 排序规则：先按 M 升序，M 相同时按 nk_list 传入顺序排序。
    
    JSON 格式设计用于两步查询：
    1. 先按 (N, K) 查找对应的 nk_entry
    2. 在 nk_entry 的 m_thresholds 中找到 <= 目标 M 的最大阈值，使用其 best_alg_id
    
    Returns:
        保存结果的子目录路径
    """
    layout = "TNCCcol"  # 固定布局，仅用于元数据记录
    hw = get_hw_info()
    
    # 子目录命名: {GPU}_{cc}_{dtype}_{outdtype}_{model_name}
    subdir_name = build_output_dir_name(model_name, dtype, outdtype)
    subdir = out_dir / subdir_name
    subdir.mkdir(parents=True, exist_ok=True)
    
    # 文件命名: {prefix}_{model_name}.{ext}
    csv_path = subdir / build_result_filename("alg_search_bench", model_name, "csv")
    json_path = subdir / build_result_filename("alg_search_LUT", model_name, "json")
    
    # 获取 alg_count 和 config_count
    # alg_count: 有效算法数量 (通过 CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID 获取)
    # config_count: 实际测试的配置数 (alg_id × split_k 组合)
    first_raw = search_ret["results"][0]["raw"] if search_ret["results"] else {}
    alg_count = first_raw.get("alg_count", 0)
    config_count = first_raw.get("config_count", 0)
    
    meta = {
        "gpu_name": hw.gpu_full_name,
        "gpu_short_name": hw.gpu_name,
        "compute_capability": hw.cc_tag,
        "arch_name": hw.arch_name,
        "model_name": model_name,
        "layout": layout,
        "dtype": dtype,
        "outdtype": outdtype,
        "alg_count": alg_count,       # 有效算法数量 (ID 范围 [0, alg_count))
        "config_count": config_count, # 实际测试的配置数 (alg_id × split_k 组合)
        "warmup": warmup,
        "repeat": repeat,
        "verify": verify,
        "torch_version": torch.__version__,
        "cuda_version_driver": hw.cuda_driver_version,
        "cuda_version_runtime": hw.cuda_runtime_version,
        "time": datetime.datetime.now().isoformat(),
        "M_list": search_ret["M_list"],
        "NK_list": search_ret["NK_list"],
    }

    # === CSV 生成（按 M 升序，M 相同时按 nk_list 顺序）===
    lines = []
    header_info = [
        f"# GPU: {hw.gpu_full_name}",
        f"# CC: {hw.cc_tag}",
        f"# Model: {model_name}",
        f"# alg_count: {alg_count}, config_count: {config_count}",
        f"# torch: {torch.__version__}",
        f"# CUDA driver: {hw.cuda_driver_version}, runtime: {hw.cuda_runtime_version}",
        f"# layout: {layout}, dtype: {dtype}, outdtype: {outdtype}, warmup={warmup}, repeat={repeat}, verify={verify}",
        f"# M_list: {search_ret['M_list']}",
        f"# NK_list: {search_ret['NK_list']}",
    ]
    lines.extend(header_info)
    # CSV列顺序: M,N,K,alg_count,config_count, 然后每个算法: tops, lat_us, id, ws, split_k
    lines.append("M,N,K,alg_count,config_count,tops1,lat_us1,id1,ws1,split_k1,tops2,lat_us2,id2,ws2,split_k2,tops3,lat_us3,id3,ws3,split_k3")

    # 收集所有数据行，用于排序
    csv_rows = []  # [(M, nk_idx, csv_line_str), ...]
    
    for nk_idx, res in enumerate(search_ret["results"]):
        raw = res["raw"]
        topk_id = raw["topk_alg_id"].cpu()
        topk_split_k = raw["topk_split_k"].cpu()
        topk_lat = raw["topk_lat_us"].cpu()
        topk_tops = raw["topk_tops"].cpu()
        topk_workspace = raw["topk_workspace"].cpu()
        valid = raw["valid_mask"].cpu()
        # 每个 NK 可能有不同的 alg_count/config_count（理论上相同，但保险起见取各自值）
        nk_alg_count = raw.get("alg_count", alg_count)
        nk_config_count = raw.get("config_count", config_count)

        for m_i, M in enumerate(search_ret["M_list"]):
            algs = topk_id[m_i]
            split_ks = topk_split_k[m_i]
            lats = topk_lat[m_i]
            tops = topk_tops[m_i]
            wss = topk_workspace[m_i]
            vmask = valid[m_i]

            # 列顺序: M,N,K,alg_count,config_count, 然后每个算法: tops, lat_us, id, ws, split_k
            csv_values = [str(M), str(res["N"]), str(res["K"]), str(nk_alg_count), str(nk_config_count)]
            for k in range(3):
                if vmask[k]:
                    csv_values.extend([
                        f"{float(tops[k].item()):.6f}",
                        f"{float(lats[k].item()):.3f}",
                        str(int(algs[k].item())),
                        str(int(wss[k].item())),
                        str(int(split_ks[k].item())),
                    ])
                else:
                    csv_values.extend(["", "", "", "", ""])  # 5 个空字段
            csv_rows.append((M, nk_idx, ",".join(csv_values)))

    # 排序：先按 M 升序，M 相同时按 nk_idx（即 nk_list 顺序）
    csv_rows.sort(key=lambda x: (x[0], x[1]))
    for _, _, line in csv_rows:
        lines.append(line)

    csv_path.write_text("\n".join(lines))

    # === JSON 生成（完整版：保存 alg_id、split_k、workspace 配置）===
    # 格式设计：
    # {
    #   "meta": {...},
    #   "nk_entries": {
    #     "(N,K)": {
    #       "m_thresholds": [m1, m2, ...],  # 升序排列的 M 值
    #       "alg_by_m": {
    #         "m1": [
    #           {"alg_id": id1, "split_k": sk1, "workspace": ws1},  # best
    #           {"alg_id": id2, "split_k": sk2, "workspace": ws2},  # 2nd
    #           {"alg_id": id3, "split_k": sk3, "workspace": ws3},  # 3rd
    #         ],
    #         "m2": [...],
    #         ...
    #       }
    #     }
    #   }
    # }
    # 
    # 查询逻辑：
    # 1. 用 (N, K) 找到 nk_entry
    # 2. 在 m_thresholds 中找到 <= query_M 的最大值 m_key
    # 3. 返回 alg_by_m[m_key][0] 作为最佳配置（包含 alg_id、split_k、workspace）
    
    nk_entries = {}
    
    for nk_idx, res in enumerate(search_ret["results"]):
        N, K = res["N"], res["K"]
        nk_key = f"({N},{K})"
        
        raw = res["raw"]
        topk_id = raw["topk_alg_id"].cpu()
        topk_split_k = raw["topk_split_k"].cpu()
        topk_workspace = raw["topk_workspace"].cpu()
        valid = raw["valid_mask"].cpu()

        m_thresholds = []
        alg_by_m = {}
        
        for m_i, M in enumerate(search_ret["M_list"]):
            algs = topk_id[m_i]
            split_ks = topk_split_k[m_i]
            wss = topk_workspace[m_i]
            vmask = valid[m_i]

            # 只有当有有效结果时才记录
            if vmask[0]:
                m_thresholds.append(M)
                # 完整格式：记录 top3 的完整配置 {alg_id, split_k, workspace}
                top3_configs = []
                for k in range(3):
                    if vmask[k]:
                        top3_configs.append({
                            "alg_id": int(algs[k].item()),
                            "split_k": int(split_ks[k].item()),
                            "workspace": int(wss[k].item()),
                        })
                alg_by_m[str(M)] = top3_configs
        
        nk_entries[nk_key] = {
            "m_thresholds": m_thresholds,
            "alg_by_m": alg_by_m,
        }

    json_payload = {
        "meta": meta,
        "nk_entries": nk_entries,
    }
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False))

    print(f"已生成: {csv_path}")
    print(f"已生成: {json_path}")
    
    return subdir


# === 主流程 ===

def parse_args():
    p = argparse.ArgumentParser(description="cuSPARSELt 算法离线搜索 v1.0")
    p.add_argument("--dtype", default="int8", choices=SUPPORTED_DTYPES, help="数据类型")
    p.add_argument("--outdtype", default="bf16", choices=SUPPORTED_OUTDTYPES, help="输出类型")
    p.add_argument("--model", default="BitNet-2B4T", help="模型名称，用于输出文件命名")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--repeat", type=int, default=100)
    p.add_argument("--verify", action="store_true", help="开启正确性校验")
    p.add_argument("--compile", action="store_true", help="强制重新编译当前架构的 CUDA 扩展")
    p.add_argument("--out_dir", default=None, help="输出目录，默认 ./alg_search_results/<timestamp>")
    return p.parse_args()


def main():
    args = parse_args()
    
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 环境")

    # 获取硬件信息
    hw = get_hw_info()
    
    # 根据 dtype 调整模型名称后缀
    dtype_suffix = get_normalize_dtype(args.dtype)
    model_name = f"{args.model}-{dtype_suffix}"

    # === 显示配置信息 ===
    print("="*60)
    print("cuSPARSELt 算法离线搜索 v1.0")
    print("="*60)
    
    print(f"GPU: {hw.gpu_full_name} ({hw.cc_tag}, {hw.arch_name})")
    print(f"参数: dtype={args.dtype}, outdtype={args.outdtype}, model={args.model}, warmup={args.warmup}, repeat={args.repeat}")
    if args.verify:
        print("注意: 已开启 verify 模式，会降低搜索速度")
    print()

    # 输出根目录（默认为 ./alg_search_results）
    out_dir = Path(args.out_dir) if args.out_dir else Path("./alg_search_results")
    
    # 获取简化的 GPU 名称
    print(f"GPU 简称: {hw.gpu_name}")
    print()

    # 构建目录和源文件路径
    src_path = Path(__file__).parent / "alg_search_cusparselt.cu"
    build_dir = Path(__file__).parent / "build"
    ext = load_cuda_extension("alg_search", "cusparselt", src_path, build_dir, verbose=True, force_compile=args.compile)

    # === 预测试 dtype 兼容性（通过实际调用 cuSPARSELt）===
    try:
        check_dtype_support(ext, args.dtype, args.outdtype, hw.arch_name,
                           backend="cusparselt", script_type="alg_search", verbose=True)
    except ValueError as e:
        print(f"\n❌ 错误: {e}")
        print("")
        return
    
    print()

    nk_list = default_nk_list()
    m_list = default_m_list()
    
    print(f"[3/4] 开始算法搜索...")
    print(f"      NK 组合: {len(nk_list)} 个, M 列表: {m_list}")
    print()

    ret = run_search(
        ext,
        args.dtype,
        args.outdtype,
        nk_list,
        m_list,
        args.warmup,
        args.repeat,
        args.verify,
        verbose=True,
    )
    saved_dir = save_outputs(
        out_dir,
        model_name,
        args.dtype,
        args.outdtype,
        ret,
        args.warmup,
        args.repeat,
        args.verify,
    )
    
    print(f"[4/4] 完成! 结果已保存到:")
    print(f"      - {saved_dir}")
    print("="*60)
if __name__ == "__main__":
    main()
