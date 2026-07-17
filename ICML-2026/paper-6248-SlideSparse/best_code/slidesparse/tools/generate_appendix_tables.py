#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse Appendix 表格生成脚本

生成两套表格:
1. cuBLASLt baseline: 以 Dense GEMM 为基线，展示各稀疏度的加速比
2. cuSPARSELt baseline: 以 2:4 稀疏为基线，展示各稀疏度的算法效率

输出目录结构:
    all_results_table/
    ├── cuBLASLt_baseline_csv/
    │   ├── appendix_a_square_*.csv
    │   ├── appendix_b_model_kernel_*.csv
    │   ├── appendix_c_prefill_*.csv
    │   └── appendix_d_decode_*.csv
    └── cuSPARSELt_baseline_csv/
        └── ... (同上，但列为效率指标)

Usage:
    python3 generate_appendix_tables.py
"""

import os
import sys
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# =============================================================================
# 路径设置
# =============================================================================

_SCRIPT_DIR = Path(__file__).parent
_SLIDESPARSE_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _SLIDESPARSE_ROOT.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from slidesparse.tools.utils import (
    Colors,
    print_header,
    print_subheader,
    print_info,
    print_success,
    print_warning,
    print_error,
)

# =============================================================================
# 配置常量
# =============================================================================

# GPU 顺序 (按架构)
GPU_ORDER = ["A100", "RTX4090", "H100", "B200", "RTX5080", "GB10"]

# GPU 目录名映射
GPU_KERNEL_DIR_MAP = {
    "A100": "A100_cc80",
    "RTX4090": "RTX4090_cc89",
    "H100": "H100_cc90",
    "B200": "B200_cc100",
    "RTX5080": "RTX5080_cc120",
    "GB10": "GB10_cc121",
}

# GEMM 精度列表
DTYPES = ["BF16", "FP16", "INT8", "FP8", "FP4"]

# dtype 目录名映射
DTYPE_DIR_MAP = {
    "BF16": "BF16",
    "FP16": "FP16",
    "INT8": "INT8",
    "FP8": "FP8",
    "FP4": "FP4",
}

# 稀疏度列表 (Kernel)
KERNEL_SPARSITY_LIST = ["2:4", "2:6", "2:8", "2:10", "2:12", "2:14", "2:16", "2:∞"]
KERNEL_SPARSITY_MAP = {
    "2:4": "2_4", "2:6": "2_6", "2:8": "2_8", "2:10": "2_10",
    "2:12": "2_12", "2:14": "2_14", "2:16": "2_16", "2:∞": "2_inf"
}

# 稀疏度列表 (E2E)
E2E_SPARSITY_LIST = ["2:4", "2:6", "2:8", "2:10"]
E2E_SPARSITY_MAP = {"2:4": "2_4", "2:6": "2_6", "2:8": "2_8", "2:10": "2_10"}

# 理论加速比映射 (相对于 2:4)
# 计算公式: Density(2:4) / Density(2:L) = 0.5 / ((L-2)/L)
# 2:6: 0.5 / (4/6) = 0.75
# 2:8: 0.5 / (6/8) = 0.667
# ...
THEORETICAL_RATIOS = {
    "2_4":  1.0,              # Baseline
    "2_6":  0.75,             # 0.5 / (4/6)
    "2_8":  0.666666667,      # 0.5 / (6/8)
    "2_10": 0.625,            # 0.5 / (8/10)
    "2_12": 0.6,              # 0.5 / (10/12)
    "2_14": 0.583333333,      # 0.5 / (12/14)
    "2_16": 0.571428571,      # 0.5 / (14/16)
    "2_inf": 0.5              # 0.5 / 1.0 (Dense)
}

# 模型列表 - 按参数量排序
MODELS_SIMPLE = ["Llama3.2-1B", "BitNet-2B", "Llama3.2-3B", "Qwen2.5-7B", "Qwen2.5-14B"]

MODELS_E2E_INT8 = [
    "Llama3.2-1B-INT8", "BitNet-2B-INT8", "Llama3.2-3B-INT8",
    "Qwen2.5-7B-INT8", "Qwen2.5-14B-INT8",
]
MODELS_E2E_FP8 = [
    "Llama3.2-1B-FP8", "BitNet-2B-FP8", "Llama3.2-3B-FP8",
    "Qwen2.5-7B-FP8", "Qwen2.5-14B-FP8",
]

# M 值上限
M_MAX_APPENDIX_A = 16384
M_MAX_APPENDIX_B = 16384
M_MAX_APPENDIX_C = 32768

# 数据源目录
KERNEL_RESULTS_DIR = _SLIDESPARSE_ROOT / "benchmark_kernel" / "kernel_speedup_results"
E2E_RESULTS_DIR = _SCRIPT_DIR / "end2end_speedup_results"

# 输出目录
OUTPUT_BASE_DIR = _SCRIPT_DIR / "all_results_table"
OUTPUT_CUBLAS_CSV = OUTPUT_BASE_DIR / "cuBLASLt_baseline_csv"
OUTPUT_CUSPARSE_CSV = OUTPUT_BASE_DIR / "cuSPARSELt_baseline_csv"


# =============================================================================
# 工具函数
# =============================================================================

def find_kernel_hw_dir(gpu_name: str) -> Optional[Path]:
    """查找 Kernel 结果中对应 GPU 的目录"""
    prefix = GPU_KERNEL_DIR_MAP.get(gpu_name)
    if not prefix:
        return None
    
    for d in KERNEL_RESULTS_DIR.iterdir():
        if d.is_dir() and d.name.startswith(prefix):
            return d
    return None


def read_csv_to_dict(csv_path: Path) -> List[Dict]:
    """读取 CSV 文件为字典列表"""
    if not csv_path.exists():
        return []
    
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def format_scientific(val: str) -> str:
    """格式化为科学计数法"""
    if not val or val.strip() == '':
        return ''
    try:
        v = float(val)
        if v == 0:
            return '0'
        return f"{v:.2e}"
    except:
        return ''


def format_speedup(val: str) -> str:
    """格式化 speedup 值"""
    if not val or val.strip() == '':
        return ''
    try:
        return f"{float(val):.2f}"
    except:
        return ''


def format_efficiency(val: float) -> str:
    """格式化效率值为百分比"""
    if val is None:
        return ''
    return f"{val * 100:.1f}%"


def get_model_simple_name(model_full: str) -> str:
    """从完整模型名获取简化名"""
    if model_full.endswith("-INT8"):
        return model_full[:-5]
    elif model_full.endswith("-FP8"):
        return model_full[:-4]
    return model_full


def calculate_efficiency(speedup_target: float, speedup_24: float, 
                         sparsity_key: str) -> Optional[float]:
    """
    计算算法效率
    
    Args:
        speedup_target: 目标稀疏度相对于 Dense 的加速比
        speedup_24: 2:4 稀疏度相对于 Dense 的加速比
        sparsity_key: 稀疏度键 (如 "2_6")
    
    Returns:
        效率值 (如 0.98 表示 98%)
    """
    if speedup_24 is None or speedup_24 <= 0:
        return None
    if speedup_target is None or speedup_target <= 0:
        return None
    
    theoretical_ratio = THEORETICAL_RATIOS.get(sparsity_key)
    if theoretical_ratio is None or theoretical_ratio <= 0:
        return None
    
    # 实测比率 = 目标加速比 / 2:4加速比
    actual_ratio = speedup_target / speedup_24
    
    # 效率 = 实测比率 / 理论比率
    efficiency = actual_ratio / theoretical_ratio
    
    return efficiency


# =============================================================================
# Appendix A: Square Kernel Performance
# =============================================================================

def generate_appendix_a():
    """生成 Appendix A: Square Kernel 大表 (两套)"""
    print_header("生成 Appendix A: Square Kernel Performance")
    
    for dtype in DTYPES:
        print_subheader(f"处理 {dtype}")
        
        rows_cublas = []
        rows_cusparse = []
        
        for gpu in GPU_ORDER:
            hw_dir = find_kernel_hw_dir(gpu)
            if not hw_dir:
                continue
            
            dtype_dir = DTYPE_DIR_MAP.get(dtype, dtype)
            latency_csv = hw_dir / "latency" / dtype_dir / "latency_SQUARE.csv"
            speedup_csv = hw_dir / "speedup" / dtype_dir / "speedup_SQUARE.csv"
            
            if not latency_csv.exists():
                print_warning(f"  {gpu}: {dtype} 无数据")
                continue
            
            latency_data = read_csv_to_dict(latency_csv)
            speedup_data = read_csv_to_dict(speedup_csv)
            
            # 建立索引
            speedup_index = {row.get('M', ''): row for row in speedup_data if row.get('M')}
            latency_index = {row.get('M', ''): row for row in latency_data if row.get('M')}
            
            row_count = 0
            for lat_row in latency_data:
                m = lat_row.get('M', '')
                if not m:
                    continue
                
                try:
                    if int(m) > M_MAX_APPENDIX_A:
                        continue
                except:
                    continue
                
                sp_row = speedup_index.get(m, {})
                
                # ===== cuBLASLt baseline =====
                out_cublas = {
                    'GPU': gpu,
                    'M': m,
                    'cuBLASLt Latency (μs)': format_scientific(lat_row.get('cuBLAS', '')),
                }
                for sp_new in KERNEL_SPARSITY_LIST:
                    sp_old = KERNEL_SPARSITY_MAP[sp_new]
                    out_cublas[sp_new] = format_speedup(sp_row.get(f'cuSPARSE_{sp_old}', ''))
                rows_cublas.append(out_cublas)
                
                # ===== cuSPARSELt baseline =====
                # 获取 2:4 的绝对延时和加速比
                lat_24 = lat_row.get('cuSPARSE_2_4', '')
                speedup_24_str = sp_row.get('cuSPARSE_2_4', '')
                
                try:
                    speedup_24 = float(speedup_24_str) if speedup_24_str else None
                except:
                    speedup_24 = None
                
                out_cusparse = {
                    'GPU': gpu,
                    'M': m,
                    'cuSPARSELt 2:4 Latency (μs)': format_scientific(lat_24),
                }
                
                # 计算各稀疏度的效率 (跳过 2:4 本身)
                for sp_new in KERNEL_SPARSITY_LIST:
                    if sp_new == "2:4":
                        continue
                    sp_old = KERNEL_SPARSITY_MAP[sp_new]
                    speedup_target_str = sp_row.get(f'cuSPARSE_{sp_old}', '')
                    try:
                        speedup_target = float(speedup_target_str) if speedup_target_str else None
                    except:
                        speedup_target = None
                    
                    eff = calculate_efficiency(speedup_target, speedup_24, sp_old)
                    out_cusparse[f'Eff_{sp_new}'] = format_efficiency(eff) if eff else ''
                
                rows_cusparse.append(out_cusparse)
                row_count += 1
            
            if row_count > 0:
                print_success(f"  {gpu}: {row_count} 行")
        
        # 写入 cuBLASLt baseline CSV
        if rows_cublas:
            output_path = OUTPUT_CUBLAS_CSV / f"appendix_a_square_{dtype}.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = ['GPU', 'M', 'cuBLASLt Latency (μs)'] + KERNEL_SPARSITY_LIST
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows_cublas)
            print_success(f"  cuBLASLt baseline: {output_path.name} ({len(rows_cublas)} 行)")
        
        # 写入 cuSPARSELt baseline CSV
        if rows_cusparse:
            output_path = OUTPUT_CUSPARSE_CSV / f"appendix_a_square_{dtype}.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            eff_cols = [f'Eff_{sp}' for sp in KERNEL_SPARSITY_LIST if sp != "2:4"]
            fieldnames = ['GPU', 'M', 'cuSPARSELt 2:4 Latency (μs)'] + eff_cols
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows_cusparse)
            print_success(f"  cuSPARSELt baseline: {output_path.name} ({len(rows_cusparse)} 行)")


# =============================================================================
# Appendix B: Model-Aware Kernel Performance
# =============================================================================

def generate_appendix_b():
    """生成 Appendix B: Model Kernel 大表 (两套)"""
    print_header("生成 Appendix B: Model-Aware Kernel Performance")
    
    for dtype in DTYPES:
        print_subheader(f"处理 {dtype}")
        
        rows_cublas = []
        rows_cusparse = []
        
        for gpu in GPU_ORDER:
            hw_dir = find_kernel_hw_dir(gpu)
            if not hw_dir:
                continue
            
            dtype_dir = DTYPE_DIR_MAP.get(dtype, dtype)
            latency_dir = hw_dir / "latency" / dtype_dir
            speedup_dir = hw_dir / "speedup" / dtype_dir
            
            if not latency_dir.exists():
                print_warning(f"  {gpu}: {dtype} 无数据")
                continue
            
            for model_simple in MODELS_SIMPLE:
                # 查找文件
                model_file = None
                suffix_found = None
                for suffix in ["-INT8", "-FP8"]:
                    candidate = latency_dir / f"total_latency_{model_simple}{suffix}.csv"
                    if candidate.exists():
                        model_file = candidate
                        suffix_found = suffix
                        break
                
                if not model_file:
                    continue
                
                speedup_file = speedup_dir / f"total_speedup_{model_simple}{suffix_found}.csv"
                
                latency_data = read_csv_to_dict(model_file)
                speedup_data = read_csv_to_dict(speedup_file) if speedup_file.exists() else []
                
                speedup_index = {row.get('M', ''): row for row in speedup_data if row.get('M')}
                
                for lat_row in latency_data:
                    m = lat_row.get('M', '')
                    if not m:
                        continue
                    
                    try:
                        if int(m) > M_MAX_APPENDIX_B:
                            continue
                    except:
                        continue
                    
                    sp_row = speedup_index.get(m, {})
                    
                    # ===== cuBLASLt baseline =====
                    out_cublas = {
                        'GPU': gpu,
                        'Model': model_simple,
                        'M': m,
                        'cuBLASLt Latency (μs)': format_scientific(lat_row.get('cuBLAS', '')),
                    }
                    for sp_new in KERNEL_SPARSITY_LIST:
                        sp_old = KERNEL_SPARSITY_MAP[sp_new]
                        out_cublas[sp_new] = format_speedup(sp_row.get(f'cuSPARSE_{sp_old}', ''))
                    rows_cublas.append(out_cublas)
                    
                    # ===== cuSPARSELt baseline =====
                    lat_24 = lat_row.get('cuSPARSE_2_4', '')
                    speedup_24_str = sp_row.get('cuSPARSE_2_4', '')
                    try:
                        speedup_24 = float(speedup_24_str) if speedup_24_str else None
                    except:
                        speedup_24 = None
                    
                    out_cusparse = {
                        'GPU': gpu,
                        'Model': model_simple,
                        'M': m,
                        'cuSPARSELt 2:4 Latency (μs)': format_scientific(lat_24),
                    }
                    
                    for sp_new in KERNEL_SPARSITY_LIST:
                        if sp_new == "2:4":
                            continue
                        sp_old = KERNEL_SPARSITY_MAP[sp_new]
                        speedup_target_str = sp_row.get(f'cuSPARSE_{sp_old}', '')
                        try:
                            speedup_target = float(speedup_target_str) if speedup_target_str else None
                        except:
                            speedup_target = None
                        
                        eff = calculate_efficiency(speedup_target, speedup_24, sp_old)
                        out_cusparse[f'Eff_{sp_new}'] = format_efficiency(eff) if eff else ''
                    
                    rows_cusparse.append(out_cusparse)
            
            print_info(f"  {gpu}: 处理完成")
        
        # 写入 CSV
        if rows_cublas:
            output_path = OUTPUT_CUBLAS_CSV / f"appendix_b_model_kernel_{dtype}.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = ['GPU', 'Model', 'M', 'cuBLASLt Latency (μs)'] + KERNEL_SPARSITY_LIST
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows_cublas)
            print_success(f"  cuBLASLt baseline: {output_path.name} ({len(rows_cublas)} 行)")
        
        if rows_cusparse:
            output_path = OUTPUT_CUSPARSE_CSV / f"appendix_b_model_kernel_{dtype}.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            eff_cols = [f'Eff_{sp}' for sp in KERNEL_SPARSITY_LIST if sp != "2:4"]
            fieldnames = ['GPU', 'Model', 'M', 'cuSPARSELt 2:4 Latency (μs)'] + eff_cols
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows_cusparse)
            print_success(f"  cuSPARSELt baseline: {output_path.name} ({len(rows_cusparse)} 行)")


# =============================================================================
# Appendix C: End-to-End Prefill Performance
# =============================================================================

def generate_appendix_c():
    """生成 Appendix C: E2E Prefill 大表 (两套)"""
    print_header("生成 Appendix C: End-to-End Prefill Performance")
    
    for precision, models_list in [("INT8", MODELS_E2E_INT8), ("FP8", MODELS_E2E_FP8)]:
        print_subheader(f"处理 {precision}")
        
        rows_cublas = []
        rows_cusparse = []
        
        for gpu in GPU_ORDER:
            hw_dir = E2E_RESULTS_DIR / gpu / "prefill"
            
            if not hw_dir.exists():
                continue
            
            for model_full in models_list:
                model_simple = get_model_simple_name(model_full)
                
                abs_csv = hw_dir / f"absolute_throughput_{model_full}.csv"
                speedup_csv = hw_dir / f"speedup_{model_full}.csv"
                
                if not abs_csv.exists():
                    continue
                
                abs_data = read_csv_to_dict(abs_csv)
                speedup_data = read_csv_to_dict(speedup_csv) if speedup_csv.exists() else []
                
                speedup_index = {row.get('M', ''): row for row in speedup_data if row.get('M')}
                
                for abs_row in abs_data:
                    m = abs_row.get('M', '')
                    if not m:
                        continue
                    
                    try:
                        if int(m) > M_MAX_APPENDIX_C:
                            continue
                    except:
                        continue
                    
                    sp_row = speedup_index.get(m, {})
                    
                    # ===== cuBLASLt baseline =====
                    out_cublas = {
                        'GPU': gpu,
                        'Model': model_simple,
                        'M': m,
                        'cuBLASLt Throughput (token/s)': format_scientific(abs_row.get('cuBLAS', '')),
                    }
                    for sp_new in E2E_SPARSITY_LIST:
                        sp_old = E2E_SPARSITY_MAP[sp_new]
                        out_cublas[sp_new] = format_speedup(sp_row.get(f'cusparse_{sp_old}', ''))
                    rows_cublas.append(out_cublas)
                    
                    # ===== cuSPARSELt baseline =====
                    # E2E 用吞吐量，2:4 的绝对吞吐
                    throughput_24 = abs_row.get('cusparse_2_4', '')
                    speedup_24_str = sp_row.get('cusparse_2_4', '')
                    try:
                        speedup_24 = float(speedup_24_str) if speedup_24_str else None
                    except:
                        speedup_24 = None
                    
                    out_cusparse = {
                        'GPU': gpu,
                        'Model': model_simple,
                        'M': m,
                        'cuSPARSELt 2:4 Throughput (token/s)': format_scientific(throughput_24),
                    }
                    
                    for sp_new in E2E_SPARSITY_LIST:
                        if sp_new == "2:4":
                            continue
                        sp_old = E2E_SPARSITY_MAP[sp_new]
                        speedup_target_str = sp_row.get(f'cusparse_{sp_old}', '')
                        try:
                            speedup_target = float(speedup_target_str) if speedup_target_str else None
                        except:
                            speedup_target = None
                        
                        eff = calculate_efficiency(speedup_target, speedup_24, sp_old)
                        out_cusparse[f'Eff_{sp_new}'] = format_efficiency(eff) if eff else ''
                    
                    rows_cusparse.append(out_cusparse)
            
            print_info(f"  {gpu}: 处理完成")
        
        # 写入 CSV
        if rows_cublas:
            output_path = OUTPUT_CUBLAS_CSV / f"appendix_c_prefill_{precision}.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = ['GPU', 'Model', 'M', 'cuBLASLt Throughput (token/s)'] + E2E_SPARSITY_LIST
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows_cublas)
            print_success(f"  cuBLASLt baseline: {output_path.name} ({len(rows_cublas)} 行)")
        
        if rows_cusparse:
            output_path = OUTPUT_CUSPARSE_CSV / f"appendix_c_prefill_{precision}.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            eff_cols = [f'Eff_{sp}' for sp in E2E_SPARSITY_LIST if sp != "2:4"]
            fieldnames = ['GPU', 'Model', 'M', 'cuSPARSELt 2:4 Throughput (token/s)'] + eff_cols
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows_cusparse)
            print_success(f"  cuSPARSELt baseline: {output_path.name} ({len(rows_cusparse)} 行)")


# =============================================================================
# Appendix D: End-to-End Decode Performance
# =============================================================================

def generate_appendix_d():
    """生成 Appendix D: E2E Decode 大表 (两套)"""
    print_header("生成 Appendix D: End-to-End Decode Performance")
    
    for precision, models_list in [("INT8", MODELS_E2E_INT8), ("FP8", MODELS_E2E_FP8)]:
        print_subheader(f"处理 {precision}")
        
        rows_cublas = []
        rows_cusparse = []
        
        for gpu in GPU_ORDER:
            hw_dir = E2E_RESULTS_DIR / gpu / "decode"
            
            if not hw_dir.exists():
                continue
            
            for model_full in models_list:
                model_simple = get_model_simple_name(model_full)
                
                abs_csv = hw_dir / f"absolute_throughput_{model_full}.csv"
                speedup_csv = hw_dir / f"speedup_{model_full}.csv"
                
                if not abs_csv.exists():
                    continue
                
                abs_data = read_csv_to_dict(abs_csv)
                speedup_data = read_csv_to_dict(speedup_csv) if speedup_csv.exists() else []
                
                speedup_index = {row.get('M', ''): row for row in speedup_data if row.get('M')}
                
                for abs_row in abs_data:
                    m = abs_row.get('M', '')
                    if not m:
                        continue
                    
                    sp_row = speedup_index.get(m, {})
                    
                    # ===== cuBLASLt baseline =====
                    out_cublas = {
                        'GPU': gpu,
                        'Model': model_simple,
                        'M': m,
                        'cuBLASLt Throughput (token/s)': format_scientific(abs_row.get('cuBLAS', '')),
                    }
                    for sp_new in E2E_SPARSITY_LIST:
                        sp_old = E2E_SPARSITY_MAP[sp_new]
                        out_cublas[sp_new] = format_speedup(sp_row.get(f'cusparse_{sp_old}', ''))
                    rows_cublas.append(out_cublas)
                    
                    # ===== cuSPARSELt baseline =====
                    throughput_24 = abs_row.get('cusparse_2_4', '')
                    speedup_24_str = sp_row.get('cusparse_2_4', '')
                    try:
                        speedup_24 = float(speedup_24_str) if speedup_24_str else None
                    except:
                        speedup_24 = None
                    
                    out_cusparse = {
                        'GPU': gpu,
                        'Model': model_simple,
                        'M': m,
                        'cuSPARSELt 2:4 Throughput (token/s)': format_scientific(throughput_24),
                    }
                    
                    for sp_new in E2E_SPARSITY_LIST:
                        if sp_new == "2:4":
                            continue
                        sp_old = E2E_SPARSITY_MAP[sp_new]
                        speedup_target_str = sp_row.get(f'cusparse_{sp_old}', '')
                        try:
                            speedup_target = float(speedup_target_str) if speedup_target_str else None
                        except:
                            speedup_target = None
                        
                        eff = calculate_efficiency(speedup_target, speedup_24, sp_old)
                        out_cusparse[f'Eff_{sp_new}'] = format_efficiency(eff) if eff else ''
                    
                    rows_cusparse.append(out_cusparse)
            
            print_info(f"  {gpu}: 处理完成")
        
        # 写入 CSV
        if rows_cublas:
            output_path = OUTPUT_CUBLAS_CSV / f"appendix_d_decode_{precision}.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = ['GPU', 'Model', 'M', 'cuBLASLt Throughput (token/s)'] + E2E_SPARSITY_LIST
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows_cublas)
            print_success(f"  cuBLASLt baseline: {output_path.name} ({len(rows_cublas)} 行)")
        
        if rows_cusparse:
            output_path = OUTPUT_CUSPARSE_CSV / f"appendix_d_decode_{precision}.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            eff_cols = [f'Eff_{sp}' for sp in E2E_SPARSITY_LIST if sp != "2:4"]
            fieldnames = ['GPU', 'Model', 'M', 'cuSPARSELt 2:4 Throughput (token/s)'] + eff_cols
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows_cusparse)
            print_success(f"  cuSPARSELt baseline: {output_path.name} ({len(rows_cusparse)} 行)")


# =============================================================================
# 主函数
# =============================================================================

def main():
    print_header("=" * 60)
    print_header("SlideSparse Appendix 表格生成 (双基线版)")
    print_header("=" * 60)
    
    print_info(f"Kernel 数据源: {KERNEL_RESULTS_DIR}")
    print_info(f"E2E 数据源: {E2E_RESULTS_DIR}")
    print_info(f"输出目录: {OUTPUT_BASE_DIR}")
    print()
    
    generate_appendix_a()
    print()
    
    generate_appendix_b()
    print()
    
    generate_appendix_c()
    print()
    
    generate_appendix_d()
    print()
    
    print_header("=" * 60)
    print_success("所有 Appendix 表格生成完成!")
    print_header("=" * 60)
    
    # 列出输出文件
    for folder_name, folder_path in [("cuBLASLt baseline", OUTPUT_CUBLAS_CSV), 
                                      ("cuSPARSELt baseline", OUTPUT_CUSPARSE_CSV)]:
        print_info(f"\n{folder_name} 文件列表:")
        if folder_path.exists():
            for f in sorted(folder_path.iterdir()):
                if f.suffix == '.csv':
                    size = f.stat().st_size
                    print_info(f"  {f.name} ({size} bytes)")


if __name__ == "__main__":
    main()
