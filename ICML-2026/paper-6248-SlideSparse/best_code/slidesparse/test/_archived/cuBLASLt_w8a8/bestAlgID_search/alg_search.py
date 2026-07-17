#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cuBLASLt 算法离线搜索

架构说明：
=========
- Python 端：负责外层 NK 循环、参数解析、GPU 检测、数据生成、结果落盘
- C++ 端：负责内层 M 循环、算法枚举、cuBLASLt API 调用、精确计时

1. Python 控制外层 NK 循环，方便做进度条、异常捕获、断点续跑
2. C++ 控制内层 M 循环和算法枚举，避免跨进程通信开销
3. JSON 在 Python 端生成
4. torch.utils.cpp_extension.load 自动检测 GPU 架构，无需手动指定 -arch

固定 Layout:
- T/N + Col/Col + Col (权重W在左)
- W[N,K]^T_col * A[K,M]_col = C[N,M]_col

运行示例:
CUDA_VISIBLE_DEVICES=0 python3 alg_search.py --dtype int8 --outdtype bf16 --model BitNet-2B4T --verify --compile
CUDA_VISIBLE_DEVICES=0 python3 alg_search.py --dtype fp8e4m3 --outdtype bf16 --model BitNet-2B4T --verify

CUDA_VISIBLE_DEVICES=0 python3 alg_search.py --dtype int8 --outdtype fp32 --model BitNet-2B4T --verify
CUDA_VISIBLE_DEVICES=0 python3 alg_search.py --dtype fp8e4m3 --outdtype fp32 --model BitNet-2B4T --verify

"""

import argparse
import base64
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
    check_dtype_support,
    build_search_meta,
    build_csv_header,
    SUPPORTED_DTYPES,
    SUPPORTED_OUTDTYPES,
)

# 从 slidesparse.utils 导入查表函数（供运行时使用）
from slidesparse.utils import lookup_best_cublaslt_alg, decode_cublaslt_algo_data


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


# 注意: lookup_best_cublaslt_alg 和 decode_cublaslt_algo_data 从 slidesparse.utils 导入


# === 运行一次 layout 的搜索 ===

def run_search(ext, dtype: str, outdtype: str, nk_list: List[Tuple[int, int]], m_list: List[int],
               warmup: int, repeat: int, verify: bool, verbose: bool = True) -> Dict:
    """
    运行算法搜索。
    
    固定布局: T/N + Col/Col + Col (权重W在左，稠密矩阵，Column Major 输出)
    每个 NK 组合生成新的随机数据
    """
    layout = "TNCCcol"  # 固定布局: TN+CC+Col
    results = []
    max_M = max(m_list)
    blacklist = []  # 不再需要屏蔽任何算法
    total_nk = len(nk_list)

    for nk_id, (N, K) in enumerate(nk_list):
        if verbose:
            print(f"    NK {nk_id+1}/{total_nk}: ({N}, {K})", flush=True)
        
        # 生成随机数据（每个 NK 新分配，简洁明了）
        max_M = max(m_list)
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        A = torch.randn(max_M, K, device="cuda", dtype=torch.bfloat16)

        # 调用搜索（cuBLASLt 不需要剪枝）
        out = ext.search_topk(
            W,
            A,
            m_list,
            layout,
            dtype,
            outdtype,
            warmup,
            repeat,
            verify,
            blacklist,
            3,
        )
        
        # 显示返回的算法数和每个 M 的有效算法数
        alg_count = out["alg_count"]
        config_count = out["config_count"]
        num_valid_per_m = out["num_valid_algs_per_M"].cpu().tolist()
        
        if verbose:
            # 显示每个 M 的有效算法数（取第一个作为代表）
            first_valid = num_valid_per_m[0] if num_valid_per_m else 0
            print(f"      → 启发式搜索返回: {alg_count} 个算法，有效算法数: {first_valid}")
        
        results.append({
            "nk_id": nk_id,
            "N": N,
            "K": K,
            "raw": out,
        })
        
        # 释放当前 NK 的张量
        del W, A
    
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
    
    文件夹命名: {GPU}_{CC}_{dtype}_{outdtype}_{model_name}
    文件命名: alg_id_benchmark_results_{model_name}.csv, alg_id_LUT_{model_name}.json
    
    CSV 排序规则：先按 M 升序，M 相同时按 nk_list 传入顺序排序。
    
    JSON 格式设计用于两步查询：
    1. 先按 (N, K) 查找对应的 nk_entry
    2. 在 nk_entry 的 m_thresholds 中找到 <= 目标 M 的最大阈值，使用其 best_alg_id
    
    Returns:
        保存结果的子目录路径
    """
    layout = "TNCCcol"  # 固定布局，仅用于元数据记录
    hw = get_hw_info()
    
    # 子目录命名: {GPU}_{CC}_{dtype}_{outdtype}_{model_name}
    subdir_name = build_output_dir_name(model_name, dtype, outdtype)
    subdir = out_dir / subdir_name
    subdir.mkdir(parents=True, exist_ok=True)
    
    # 文件命名: {prefix}_{model_name}.{ext}
    csv_path = subdir / build_result_filename("alg_search_bench", model_name, "csv")
    json_path = subdir / build_result_filename("alg_search_LUT", model_name, "json")
    
    # 获取 alg_count 和 config_count（cuBLASLt 启发式搜索返回的数量）
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
        "alg_count": alg_count,
        "config_count": config_count,
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
    # CSV列顺序: M,N,K,alg_count,config_count, 然后每个算法: tops, lat_us, id, ws, waves_count
    lines.append("M,N,K,alg_count,config_count,tops1,lat_us1,id1,ws1,waves1,tops2,lat_us2,id2,ws2,waves2,tops3,lat_us3,id3,ws3,waves3")

    # 收集所有数据行，用于排序
    csv_rows = []  # [(M, nk_idx, csv_line_str), ...]
    
    for nk_idx, res in enumerate(search_ret["results"]):
        raw = res["raw"]
        topk_id = raw["topk_alg_id"].cpu()
        topk_lat = raw["topk_lat_us"].cpu()
        topk_tops = raw["topk_tops"].cpu()
        topk_workspace = raw["topk_workspace"].cpu()
        topk_waves = raw["topk_waves_count"].cpu()
        valid = raw["valid_mask"].cpu()
        # 每个 NK 可能有不同的 alg_count/config_count（理论上相同，但保险起见取各自值）
        nk_alg_count = raw.get("alg_count", alg_count)
        nk_config_count = raw.get("config_count", config_count)

        for m_i, M in enumerate(search_ret["M_list"]):
            algs = topk_id[m_i]
            lats = topk_lat[m_i]
            tops = topk_tops[m_i]
            wss = topk_workspace[m_i]
            waves = topk_waves[m_i]
            vmask = valid[m_i]

            # 列顺序: M,N,K,alg_count,config_count, 然后每个算法: tops, lat_us, id, ws, waves_count
            csv_values = [str(M), str(res["N"]), str(res["K"]), str(nk_alg_count), str(nk_config_count)]
            for k in range(3):
                if vmask[k]:
                    csv_values.extend([
                        f"{float(tops[k].item()):.6f}",
                        f"{float(lats[k].item()):.3f}",
                        str(int(algs[k].item())),
                        str(int(wss[k].item())),
                        f"{float(waves[k].item()):.4f}",
                    ])
                else:
                    csv_values.extend(["", "", "", "", ""])  # 5 个空字段
            csv_rows.append((M, nk_idx, ",".join(csv_values)))

    # 排序：先按 M 升序，M 相同时按 nk_idx（即 nk_list 顺序）
    csv_rows.sort(key=lambda x: (x[0], x[1]))
    for _, _, line in csv_rows:
        lines.append(line)

    csv_path.write_text("\n".join(lines))

    # === JSON 生成（简化版：只保留 top3 的 64B algo_data）===
    nk_entries = {}
    
    for nk_idx, res in enumerate(search_ret["results"]):
        N, K = res["N"], res["K"]
        nk_key = f"({N},{K})"
        
        raw = res["raw"]
        topk_algo_data = raw["topk_algo_data"].cpu()  # [num_M, topk, 64]
        valid = raw["valid_mask"].cpu()

        m_thresholds = []
        alg_by_m = {}
        
        for m_i, M in enumerate(search_ret["M_list"]):
            vmask = valid[m_i]

            # 只有当有有效结果时才记录
            if vmask[0]:
                m_thresholds.append(M)
                # 简化格式：只记录 top3 的 algo_data (base64)
                top3_b64 = []
                for k in range(3):
                    if vmask[k]:
                        # 提取 64B 原始数据并 base64 编码
                        algo_bytes = bytes(topk_algo_data[m_i, k].numpy().tolist())
                        algo_b64 = base64.b64encode(algo_bytes).decode('ascii')
                        top3_b64.append(algo_b64)
                alg_by_m[str(M)] = top3_b64
        
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
    p = argparse.ArgumentParser(description="cuBLASLt 算法离线搜索 v2.0")
    p.add_argument("--dtype", default="int8", choices=SUPPORTED_DTYPES, help="输入数据类型")
    p.add_argument("--outdtype", default="bf16", choices=SUPPORTED_OUTDTYPES, help="输出数据类型")
    p.add_argument("--model", default="BitNet-2B4T", help="模型名称（用于文件命名）")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--repeat", type=int, default=100)
    p.add_argument("--verify", action="store_true", help="开启正确性校验")
    p.add_argument("--compile", action="store_true", help="强制重新编译当前架构的 CUDA 扩展")
    p.add_argument("--out_dir", default=None, help="输出目录，默认 ./alg_search_results")
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
    print("cuBLASLt 算法离线搜索 v2.0")
    print("="*60)
    
    print(f"GPU: {hw.gpu_full_name} ({hw.cc_tag}, {hw.arch_name})")
    print(f"模型: {model_name}")
    print(f"参数: dtype={args.dtype}, outdtype={args.outdtype}, warmup={args.warmup}, repeat={args.repeat}")
    if args.verify:
        print("注意: 已开启 verify 模式，会降低搜索速度")
    print()

    # 输出根目录
    out_dir = Path(args.out_dir) if args.out_dir else Path("./alg_search_results")

    # 加载 CUDA 扩展
    src_path = Path(__file__).parent / "alg_search_cublaslt.cu"
    build_dir = Path(__file__).parent / "build"
    ext = load_cuda_extension("alg_search", "cublaslt", src_path, build_dir, verbose=True, force_compile=args.compile)

    # === 预测试 dtype 兼容性（通过实际调用 cuBLASLt）===
    try:
        check_dtype_support(ext, args.dtype, args.outdtype, hw.arch_name,
                           backend="cublaslt", script_type="alg_search", verbose=True)
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
