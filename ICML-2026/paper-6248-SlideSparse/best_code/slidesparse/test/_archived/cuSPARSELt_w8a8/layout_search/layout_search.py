#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cuSPARSELt Layout 性能测试

测试不同 layout 组合的 SpMM 性能，支持 INT8 和 FP8。

Layout 组合说明：
================
测试 4 种主要 layout 配置 × 2 种输出顺序 = 8 种组合：
1. T/N + Col/Col + (Col/Row)  - opW=T, opA=N, orderW=Col, orderA=Col
2. N/T + Row/Row + (Col/Row)  - opW=N, opA=T, orderW=Row, orderA=Row
3. N/N + Row/Col + (Col/Row)  - opW=N, opA=N, orderW=Row, orderA=Col
4. T/T + Col/Row + (Col/Row)  - opW=T, opA=T, orderW=Col, orderA=Row

运行示例:
CUDA_VISIBLE_DEVICES=0 python3 layout_search.py --dtype int8 --outdtype bf16 --model BitNet-2B4T --compile
CUDA_VISIBLE_DEVICES=0 python3 layout_search.py --dtype fp8e4m3 --outdtype bf16 --model BitNet-2B4T

CUDA_VISIBLE_DEVICES=0 python3 layout_search.py --dtype int8 --outdtype fp32 --model BitNet-2B4T
CUDA_VISIBLE_DEVICES=0 python3 layout_search.py --dtype fp8e4m3 --outdtype fp32 --model BitNet-2B4T

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
    SUPPORTED_DTYPES,
    SUPPORTED_OUTDTYPES,
)


# === Layout 配置 ===

# 4 种主要 layout 配置
LAYOUT_CONFIGS = [
    {"name": "TN+CC", "opW": "T", "opA": "N", "orderW": "Col", "orderA": "Col"},
    {"name": "NT+RR", "opW": "N", "opA": "T", "orderW": "Row", "orderA": "Row"},
    {"name": "NN+RC", "opW": "N", "opA": "N", "orderW": "Row", "orderA": "Col"},
    {"name": "TT+CR", "opW": "T", "opA": "T", "orderW": "Col", "orderA": "Row"},
]

# 输出矩阵顺序
OUTPUT_ORDERS = ["Col", "Row"]


# === 默认配置 ===

def default_nk_list() -> List[Tuple[int, int, str]]:
    """返回 (N, K, name) 列表"""
    return [
        (3840, 2560, "Wqkv"),
        (2560, 2560, "Wo"),
        (13824, 2560, "W13"),
        (2560, 6912, "W2"),
    ]


def default_m_list() -> List[int]:
    return [16, 256, 2048, 16384]


# 注意: SUPPORTED_DTYPES 和 SUPPORTED_OUTDTYPES 从 utils 导入


# === 测试运行 ===

def run_layout_benchmark(ext, dtype: str, outdtype: str, nk_list: List[Tuple[int, int, str]], 
                         m_list: List[int], warmup: int, repeat: int,
                         verbose: bool = True) -> Dict:
    """
    运行所有 layout 组合的性能测试。
    
    返回结构：
    {
        "dtype": str,
        "nk_list": [...],
        "m_list": [...],
        "results": {
            "(N,K)": {
                "M": {
                    "layout_name+orderR": {
                        "supported": bool,
                        "alg_count": int,
                        "config_count": int,
                        "best_tops": float,
                        "best_lat_us": float,
                        "best_id": int,
                        "best_ws": int,
                        "best_split_k": int
                    }
                }
            }
        }
    }
    """
    results = {}
    total_nk = len(nk_list)
    
    # 检测是否支持 Segment-K
    test_segment_k, segment_k_reason = supports_segment_k()
    if verbose:
        if test_segment_k:
            print(f"    [Segment-K] 当前架构支持 Segment-K，将测试 split_k=-1")
        else:
            print(f"    [Segment-K] {segment_k_reason}")
    
    for nk_idx, (N, K, nk_name) in enumerate(nk_list):
        if verbose:
            print(f"  NK {nk_idx+1}/{total_nk}: ({N}, {K}) - {nk_name}")
        
        nk_key = f"({N},{K})"
        results[nk_key] = {"name": nk_name}
        
        for M in m_list:
            if verbose:
                print(f"    M={M}:", end=" ", flush=True)
            
            m_results = {}
            
            for layout_cfg in LAYOUT_CONFIGS:
                for orderR in OUTPUT_ORDERS:
                    config_name = f"{layout_cfg['name']}+{orderR}"
                    
                    try:
                        # 调用 C++ 扩展进行测试
                        out = ext.test_layout(
                            N, K, M,
                            layout_cfg["opW"],
                            layout_cfg["opA"],
                            layout_cfg["orderW"],
                            layout_cfg["orderA"],
                            orderR,
                            dtype,
                            outdtype,
                            warmup,
                            repeat,
                            test_segment_k,  # 是否测试 Segment-K (split_k=-1)
                        )
                        
                        m_results[config_name] = {
                            "supported": out["supported"],
                            "alg_count": out.get("alg_count", 0),
                            "config_count": out.get("config_count", 0),
                            "best_tops": out.get("best_tops", 0.0),
                            "best_lat_us": out.get("best_lat_us", 0.0),
                            "best_id": out.get("best_id", -1),
                            "best_ws": out.get("best_ws", 0),
                            "best_split_k": out.get("best_split_k", 1),
                        }
                        
                    except Exception as e:
                        m_results[config_name] = {
                            "supported": False,
                            "error": str(e),
                            "alg_count": 0,
                            "config_count": 0,
                            "best_tops": 0.0,
                            "best_lat_us": 0.0,
                            "best_id": -1,
                            "best_ws": 0,
                            "best_split_k": 1,
                        }
            
            results[nk_key][str(M)] = m_results
            
            # 统计支持的配置数
            supported_count = sum(1 for v in m_results.values() if v.get("supported", False))
            if verbose:
                print(f"{supported_count}/8 layouts ✓")
    
    return {
        "dtype": dtype,
        "outdtype": outdtype,
        "nk_list": [(n, k, name) for n, k, name in nk_list],
        "m_list": m_list,
        "results": results,
    }


# === 结果保存 ===

def save_outputs(out_dir: Path, model_name: str, dtype: str, outdtype: str,
                 benchmark_ret: Dict, warmup: int, repeat: int) -> Path:
    """保存测试结果到 CSV 和 JSON 文件"""
    hw = get_hw_info()
    
    subdir_name = build_output_dir_name(model_name, dtype, outdtype)
    subdir = out_dir / subdir_name
    subdir.mkdir(parents=True, exist_ok=True)
    
    csv_path = subdir / build_result_filename("layout_search_bench", model_name, "csv")
    json_path = subdir / build_result_filename("layout_search_INFO", model_name, "json")
    
    meta = {
        "gpu_name": hw.gpu_name,
        "compute_capability": hw.cc_tag,
        "arch_name": hw.arch_name,
        "dtype": dtype,
        "outdtype": outdtype,
        "warmup": warmup,
        "repeat": repeat,
        "torch_version": torch.__version__,
        "cuda_version_driver": hw.cuda_driver_version,
        "cuda_version_runtime": hw.cuda_runtime_version,
        "time": datetime.datetime.now().isoformat(),
        "m_list": benchmark_ret["m_list"],
        "nk_list": benchmark_ret["nk_list"],
        "layout_configs": [cfg["name"] for cfg in LAYOUT_CONFIGS],
        "output_orders": OUTPUT_ORDERS,
    }

    # === CSV 生成 ===
    lines = []
    lines.append(f"# GPU: {hw.gpu_name}")
    lines.append(f"# CC: {hw.cc_tag}")
    lines.append(f"# dtype: {dtype}, outdtype: {outdtype}, warmup={warmup}, repeat={repeat}")
    lines.append(f"# Layouts: {[cfg['name'] for cfg in LAYOUT_CONFIGS]}")
    # CSV列顺序: tops, lat, id, ws, split_k (best结果)
    lines.append("M,N,K,layout,orderR,supported,alg_count,config_count,best_tops,best_lat_us,best_id,best_ws,best_split_k")

    for nk_key, nk_data in benchmark_ret["results"].items():
        # 解析 nk_key
        N, K = map(int, nk_key.strip("()").split(","))
        
        for m_key, m_data in nk_data.items():
            if m_key == "name":
                continue
            M = int(m_key)
            
            for config_name, result in m_data.items():
                # 解析 config_name: "TN+CC+Col"
                parts = config_name.rsplit("+", 1)
                layout_name = parts[0]
                orderR = parts[1] if len(parts) > 1 else "Col"
                
                supported = 1 if result.get("supported", False) else 0
                alg_count = result.get("alg_count", 0)
                config_count = result.get("config_count", 0)
                
                best_tops = result.get("best_tops", 0.0)
                best_lat = result.get("best_lat_us", 0.0)
                best_id = result.get("best_id", -1)
                best_ws = result.get("best_ws", 0)
                best_split_k = result.get("best_split_k", 1)
                
                lines.append(f"{M},{N},{K},{layout_name},{orderR},{supported},{alg_count},{config_count},{best_tops:.6f},{best_lat:.3f},{best_id},{best_ws},{best_split_k}")
    
    csv_path.write_text("\n".join(lines))

    # === JSON 生成 ===
    json_payload = {
        "meta": meta,
        "results": benchmark_ret["results"],
    }
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False))

    print(f"已生成: {csv_path}")
    print(f"已生成: {json_path}")
    
    return subdir


# === 主流程 ===

def parse_args():
    p = argparse.ArgumentParser(description="cuSPARSELt Layout 性能测试")
    p.add_argument("--dtype", default="int8", choices=SUPPORTED_DTYPES, help="输入数据类型")
    p.add_argument("--outdtype", default="bf16", choices=SUPPORTED_OUTDTYPES, help="输出数据类型")
    p.add_argument("--model", default="BitNet-2B4T", help="模型名称，用于输出文件命名")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=50)
    p.add_argument("--compile", action="store_true", help="强制重新编译 CUDA 扩展")
    p.add_argument("--out_dir", default=None, help="输出目录")
    return p.parse_args()


def main():
    args = parse_args()
    
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 环境")

    print("="*60)
    print("cuSPARSELt Layout 性能测试")
    print("="*60)
    
    hw = get_hw_info()
    print(f"GPU: {hw.gpu_name} (CC {hw.cc_tag}, {hw.arch_name})")
    print(f"参数: dtype={args.dtype}, outdtype={args.outdtype}, model={args.model}, warmup={args.warmup}, repeat={args.repeat}")
    print()

    out_dir = Path(args.out_dir) if args.out_dir else Path("./layout_search_results")
    
    print(f"GPU 简称: {hw.gpu_name}")
    print()

    # 构建目录和源文件路径
    src_path = Path(__file__).parent / "layout_search_cusparselt.cu"
    build_dir = Path(__file__).parent / "build"
    ext = load_cuda_extension("layout_search", "cusparselt", src_path, build_dir, verbose=True, force_compile=args.compile)

    try:
        check_dtype_support(ext, args.dtype, args.outdtype, hw.arch_name,
                           backend="cusparselt", script_type="layout_search", verbose=True)
    except ValueError as e:
        print(f"\n❌ 错误: {e}")
        return
    
    print()

    nk_list = default_nk_list()
    m_list = default_m_list()
    
    print(f"[3/4] 开始 Layout 性能测试...")
    print(f"      NK 组合: {len(nk_list)} 个, M 列表: {m_list}")
    print(f"      Layout 配置: {len(LAYOUT_CONFIGS)} × {len(OUTPUT_ORDERS)} = {len(LAYOUT_CONFIGS) * len(OUTPUT_ORDERS)} 种")
    print()

    ret = run_layout_benchmark(
        ext,
        args.dtype,
        args.outdtype,
        nk_list,
        m_list,
        args.warmup,
        args.repeat,
        verbose=True,
    )
    
    saved_dir = save_outputs(
        out_dir,
        args.model,
        args.dtype,
        args.outdtype,
        ret,
        args.warmup,
        args.repeat,
    )
    
    print(f"[4/4] 完成! 结果已保存到:")
    print(f"      - {saved_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
