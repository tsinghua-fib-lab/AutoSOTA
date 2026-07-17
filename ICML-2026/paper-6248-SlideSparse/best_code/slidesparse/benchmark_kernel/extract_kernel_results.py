#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse Kernel Benchmark Result Extraction Script

Extract TOPS and Latency data from cuBLASLt and cuSPARSELt alg_search_results,
calculate speedup (cuBLAS / cuSPARSE), generate summary CSV.

Output directory structure:
    kernel_speedup_results/
    ├── {hw_folder}/
    │   ├── tops/{dtype}/
    │   │   ├── tops_Llama3.2-1B-INT8.csv
    │   │   ├── tops_SQUARE.csv
    │   │   └── ...
    │   ├── latency/{dtype}/
    │   │   ├── latency_Llama3.2-1B-INT8.csv        # All MNK
    │   │   ├── total_latency_Llama3.2-1B-INT8.csv  # Summarized by M (model only)
    │   │   ├── latency_SQUARE.csv
    │   │   └── ...
    │   └── speedup/{dtype}/
    │       ├── speedup_Llama3.2-1B-INT8.csv        # All MNK
    │       ├── total_speedup_Llama3.2-1B-INT8.csv  # Summarized by M (model only)
    │       ├── speedup_SQUARE.csv
    │       └── ...
    └── ...

Usage:
    # Extract all results
    python3 extract_kernel_results.py
    
    # Extract specific dtype only
    python3 extract_kernel_results.py --dtype fp8e4m3
    
    # Extract specific model only
    python3 extract_kernel_results.py --model Llama3.2-1B-INT8
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

# =============================================================================
# Path Setup
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

from slidesparse.benchmark_kernel.utils import (
    build_hw_folder_name,
    build_dtype_folder_name,
    build_result_filename,
    calculate_k_slide,
    get_k_expansion_factor,
    hw_info,
)


# =============================================================================
# Config Constants
# =============================================================================

# Model list (including BitNet)
MODELS = [
    "Llama3.2-1B-INT8", "Llama3.2-1B-FP8",
    "Llama3.2-3B-INT8", "Llama3.2-3B-FP8",
    "Qwen2.5-7B-INT8", "Qwen2.5-7B-FP8",
    "Qwen2.5-14B-INT8", "Qwen2.5-14B-FP8",
    "BitNet-2B-INT8", "BitNet-2B-FP8",
]

# 5 precision types
DTYPES = ["fp16", "bf16", "int8", "fp8e4m3", "fp4e2m1"]

# Sparsity list (high + low sparsity)
SPARSITY_LIST = ["2_4", "2_6", "2_8", "2_10", "2_12", "2_14", "2_16", "2_inf"]

# Source data directories
CUBLASLT_DIR = _SCRIPT_DIR / "cuBLASLt" / "alg_search_results"
CUSPARSELT_DIR = _SCRIPT_DIR / "cuSPARSELt" / "alg_search_results"

# Output directory
OUTPUT_DIR = _SCRIPT_DIR / "kernel_speedup_results"


# =============================================================================
# Utility Functions
# =============================================================================

def load_json_results(json_path: Optional[Path]) -> Optional[Dict]:
    """Load JSON result file"""
    if json_path is None or not json_path.exists():
        return None
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def get_nk_list_from_json(data: Dict) -> List[Tuple[int, int]]:
    """
    Get complete NK list from JSON meta
    
    Returns:
        [(N, K), ...] list
    """
    if not data:
        return []
    
    meta = data.get("meta", {})
    nk_list = meta.get("NK_list", [])
    
    # NK_list 格式: [[N1, K1], [N2, K2], ...]
    return [(int(nk[0]), int(nk[1])) for nk in nk_list]


def build_index_from_json(data: Dict) -> Dict[Tuple[int, int, int], Dict]:
    """
    Build index from JSON data
    
    Returns:
        {(M, N, K): result_dict}
    """
    index = {}
    if not data:
        return index
    
    for nk_key, nk_data in data.get("nk_entries", {}).items():
        try:
            nk_str = nk_key.strip("()")
            N, K = map(int, nk_str.split(","))
        except:
            continue
        
        for m_str, m_data in nk_data.get("alg_by_m", {}).items():
            M = int(m_str)
            if m_data:
                index[(M, N, K)] = m_data
    
    return index


def find_cublaslt_json(hw_folder: str, model_name: str, dtype: str) -> Optional[Path]:
    """Find cuBLASLt JSON file"""
    dtype_folder = build_dtype_folder_name(dtype)
    json_filename = build_result_filename("alg_search", model_name, "json")
    
    json_path = CUBLASLT_DIR / hw_folder / dtype_folder / json_filename
    return json_path if json_path.exists() else None


def find_cusparselt_json(hw_folder: str, model_name: str, dtype: str, sparsity: str) -> Optional[Path]:
    """Find cuSPARSELt JSON file"""
    dtype_folder = build_dtype_folder_name(dtype)
    json_filename = build_result_filename("alg_search", model_name, "json", sparsity)
    
    json_path = CUSPARSELT_DIR / hw_folder / dtype_folder / json_filename
    return json_path if json_path.exists() else None


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ExtractedRow:
    """One extracted row (for one MNK combination)"""
    m_value: int
    n_value: int
    k_value: int
    cublas_tops: Optional[float]
    cublas_lat_us: Optional[float]
    cusparse_tops: Dict[str, Optional[float]]  # sparsity -> tops
    cusparse_lat_us: Dict[str, Optional[float]]  # sparsity -> lat_us


@dataclass
class TotalRow:
    """Summarized data by M (sum of all NK)"""
    m_value: int
    cublas_total_lat: Optional[float]  # Sum of cuBLAS latency for all NK
    cusparse_total_lat: Dict[str, Optional[float]]  # sparsity -> sum of latency for all NK


# =============================================================================
# Data Extraction
# =============================================================================

def extract_model_data(
    hw_folder: str,
    model_name: str,
    dtype: str,
    sparsity_list: List[str],
) -> Tuple[List[ExtractedRow], List[Tuple[int, int]]]:
    """
    Extract all data for a single model
    
    Returns:
        (rows, nk_list): Extracted data rows and complete NK list
    """
    # Load cuBLASLt data
    cublas_json_path = find_cublaslt_json(hw_folder, model_name, dtype)
    cublas_data = load_json_results(cublas_json_path)
    cublas_index = build_index_from_json(cublas_data)
    
    # Get complete NK list (from cuBLASLt JSON meta)
    nk_list = get_nk_list_from_json(cublas_data)
    
    # Load cuSPARSELt data (for each sparsity)
    cusparse_data_map = {}  # sparsity -> json_data
    cusparse_indices = {}   # sparsity -> index
    
    for sp in sparsity_list:
        sp_json_path = find_cusparselt_json(hw_folder, model_name, dtype, sp)
        sp_data = load_json_results(sp_json_path)
        if sp_data:
            cusparse_data_map[sp] = sp_data
            cusparse_indices[sp] = build_index_from_json(sp_data)
            # If cuBLASLt has no NK list, try to get from cuSPARSELt
            if not nk_list:
                nk_list = get_nk_list_from_json(sp_data)
    
    if not cublas_index and not cusparse_indices:
        return [], []
    
    # Collect all unique (M, N, K) combinations (from cuBLASLt)
    rows = []
    for (M, N, K), cublas_result in sorted(cublas_index.items()):
        cublas_tops = cublas_result.get("tops")
        cublas_lat = cublas_result.get("lat_us")
        
        # Collect results for each sparsity
        cusparse_tops = {}
        cusparse_lat = {}
        
        for sp in sparsity_list:
            # Calculate K_slide
            K_slide = calculate_k_slide(K, sp, dtype=dtype)
            
            sp_index = cusparse_indices.get(sp, {})
            sp_key = (M, N, K_slide)
            
            if sp_key in sp_index:
                sp_result = sp_index[sp_key]
                cusparse_tops[sp] = sp_result.get("tops")
                cusparse_lat[sp] = sp_result.get("lat_us")
            else:
                cusparse_tops[sp] = None
                cusparse_lat[sp] = None
        
        row = ExtractedRow(
            m_value=M,
            n_value=N,
            k_value=K,
            cublas_tops=cublas_tops,
            cublas_lat_us=cublas_lat,
            cusparse_tops=cusparse_tops,
            cusparse_lat_us=cusparse_lat,
        )
        rows.append(row)
    
    return rows, nk_list


def compute_total_rows(
    rows: List[ExtractedRow],
    nk_list: List[Tuple[int, int]],
    sparsity_list: List[str],
) -> List[TotalRow]:
    """
    Compute summarized data by M (sum latency of all NK)
    
    Rules:
    - If any NK pair's cuBLAS is missing for a given M, leave cuBLAS total empty
    - If any NK pair's cuSPARSE for a sparsity is missing, leave that sparsity total empty
    
    Args:
        rows: Extracted data rows
        nk_list: Complete NK list (defines how many NK pairs the model should have)
        sparsity_list: Sparsity list
    
    Returns:
        TotalRow list (sorted by M)
    """
    if not nk_list or len(nk_list) == 0:
        return []
    
    # Group by M
    m_to_rows: Dict[int, List[ExtractedRow]] = defaultdict(list)
    for row in rows:
        m_to_rows[row.m_value].append(row)
    
    # Build NK set for completeness check
    nk_set = set(nk_list)
    expected_nk_count = len(nk_list)
    
    total_rows = []
    for m in sorted(m_to_rows.keys()):
        m_rows = m_to_rows[m]
        
        # Check if this M has complete NK list
        actual_nk_set = {(row.n_value, row.k_value) for row in m_rows}
        
        # cuBLAS total: all NK must have data
        cublas_total = None
        if actual_nk_set == nk_set:
            # Check if all rows have cuBLAS latency
            all_cublas_valid = all(
                row.cublas_lat_us is not None and row.cublas_lat_us > 0
                for row in m_rows
            )
            if all_cublas_valid:
                cublas_total = sum(row.cublas_lat_us for row in m_rows)
        
        # cuSPARSE total: check each sparsity independently
        cusparse_totals = {}
        for sp in sparsity_list:
            if actual_nk_set == nk_set:
                # Check if all rows have this sparsity's cuSPARSE latency
                all_sp_valid = all(
                    row.cusparse_lat_us.get(sp) is not None and row.cusparse_lat_us.get(sp) > 0
                    for row in m_rows
                )
                if all_sp_valid:
                    cusparse_totals[sp] = sum(row.cusparse_lat_us.get(sp) for row in m_rows)
                else:
                    cusparse_totals[sp] = None
            else:
                cusparse_totals[sp] = None
        
        total_rows.append(TotalRow(
            m_value=m,
            cublas_total_lat=cublas_total,
            cusparse_total_lat=cusparse_totals,
        ))
    
    return total_rows


# =============================================================================
# CSV Write Functions
# =============================================================================

def write_tops_csv(rows: List[ExtractedRow], output_path: Path, sparsity_list: List[str]) -> int:
    """
    Write TOPS CSV
    
    Returns:
        Valid data row count
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    valid_count = 0
    with open(output_path, "w") as f:
        # Header
        header = ["M", "N", "K", "cuBLAS"]
        for sp in sparsity_list:
            header.append(f"cuSPARSE_{sp}")
        f.write(",".join(header) + "\n")
        
        for row in rows:
            values = [
                str(row.m_value),
                str(row.n_value),
                str(row.k_value),
                f"{row.cublas_tops:.2f}" if row.cublas_tops is not None else "",
            ]
            
            for sp in sparsity_list:
                sp_tops = row.cusparse_tops.get(sp)
                values.append(f"{sp_tops:.2f}" if sp_tops is not None else "")
            
            f.write(",".join(values) + "\n")
            
            if row.cublas_tops is not None:
                valid_count += 1
    
    return valid_count


def write_latency_csv(rows: List[ExtractedRow], output_path: Path, sparsity_list: List[str]) -> int:
    """
    Write Latency CSV (unit: microseconds)
    
    Returns:
        Valid data row count
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    valid_count = 0
    with open(output_path, "w") as f:
        # Header
        header = ["M", "N", "K", "cuBLAS"]
        for sp in sparsity_list:
            header.append(f"cuSPARSE_{sp}")
        f.write(",".join(header) + "\n")
        
        for row in rows:
            values = [
                str(row.m_value),
                str(row.n_value),
                str(row.k_value),
                f"{row.cublas_lat_us:.3f}" if row.cublas_lat_us is not None else "",
            ]
            
            for sp in sparsity_list:
                sp_lat = row.cusparse_lat_us.get(sp)
                values.append(f"{sp_lat:.3f}" if sp_lat is not None else "")
            
            f.write(",".join(values) + "\n")
            
            if row.cublas_lat_us is not None:
                valid_count += 1
    
    return valid_count


def write_total_latency_csv(total_rows: List[TotalRow], output_path: Path, sparsity_list: List[str]) -> int:
    """
    Write Total Latency CSV (summarized by M, unit: microseconds)
    
    Returns:
        Valid data row count
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    valid_count = 0
    with open(output_path, "w") as f:
        # Header (no N, K columns)
        header = ["M", "cuBLAS"]
        for sp in sparsity_list:
            header.append(f"cuSPARSE_{sp}")
        f.write(",".join(header) + "\n")
        
        for row in total_rows:
            values = [
                str(row.m_value),
                f"{row.cublas_total_lat:.3f}" if row.cublas_total_lat is not None else "",
            ]
            
            for sp in sparsity_list:
                sp_lat = row.cusparse_total_lat.get(sp)
                values.append(f"{sp_lat:.3f}" if sp_lat is not None else "")
            
            f.write(",".join(values) + "\n")
            
            if row.cublas_total_lat is not None:
                valid_count += 1
    
    return valid_count


def write_speedup_csv(rows: List[ExtractedRow], output_path: Path, sparsity_list: List[str]) -> int:
    """
    Write Speedup CSV
    
    Speedup = cuBLAS_latency / cuSPARSE_latency
    (lower latency is better, so cuBLAS/cuSPARSE, >1 means cuSPARSE is faster)
    
    Returns:
        Valid data row count
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    valid_count = 0
    with open(output_path, "w") as f:
        # Header
        header = ["M", "N", "K", "cuBLAS"]
        for sp in sparsity_list:
            header.append(f"cuSPARSE_{sp}")
        f.write(",".join(header) + "\n")
        
        for row in rows:
            # If cuBLAS latency missing, leave entire speedup row empty (only output MNK)
            if row.cublas_lat_us is None or row.cublas_lat_us <= 0:
                values = [
                    str(row.m_value),
                    str(row.n_value),
                    str(row.k_value),
                    "",  # cuBLAS column empty
                ]
                for sp in sparsity_list:
                    values.append("")  # All cuSPARSE columns empty
                f.write(",".join(values) + "\n")
                continue
            
            values = [
                str(row.m_value),
                str(row.n_value),
                str(row.k_value),
                "1.00",  # cuBLAS as baseline = 1.00
            ]
            
            for sp in sparsity_list:
                sp_lat = row.cusparse_lat_us.get(sp)
                if sp_lat is not None and sp_lat > 0:
                    # Speedup = cuBLAS_lat / cuSPARSE_lat
                    speedup = row.cublas_lat_us / sp_lat
                    values.append(f"{speedup:.2f}")
                else:
                    values.append("")
            
            f.write(",".join(values) + "\n")
            valid_count += 1
    
    return valid_count


def write_total_speedup_csv(total_rows: List[TotalRow], output_path: Path, sparsity_list: List[str]) -> int:
    """
    Write Total Speedup CSV (summarized by M)
    
    Speedup = cuBLAS_total_lat / cuSPARSE_total_lat
    
    Returns:
        Valid data row count
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    valid_count = 0
    with open(output_path, "w") as f:
        # Header (no N, K columns)
        header = ["M", "cuBLAS"]
        for sp in sparsity_list:
            header.append(f"cuSPARSE_{sp}")
        f.write(",".join(header) + "\n")
        
        for row in total_rows:
            # If cuBLAS total missing, leave entire row empty (only output M)
            if row.cublas_total_lat is None or row.cublas_total_lat <= 0:
                values = [str(row.m_value), ""]
                for sp in sparsity_list:
                    values.append("")
                f.write(",".join(values) + "\n")
                continue
            
            values = [
                str(row.m_value),
                "1.00",  # cuBLAS as baseline = 1.00
            ]
            
            for sp in sparsity_list:
                sp_lat = row.cusparse_total_lat.get(sp)
                if sp_lat is not None and sp_lat > 0:
                    speedup = row.cublas_total_lat / sp_lat
                    values.append(f"{speedup:.2f}")
                else:
                    values.append("")
            
            f.write(",".join(values) + "\n")
            valid_count += 1
    
    return valid_count


# =============================================================================
# Process Single Model
# =============================================================================

def process_model(
    hw_folder: str,
    model_name: str,
    dtype: str,
    sparsity_list: List[str],
    output_base: Path,
    is_square: bool = False,
) -> bool:
    """
    Process all data for a single model
    
    Returns:
        Whether valid data exists
    """
    # Extract data
    rows, nk_list = extract_model_data(hw_folder, model_name, dtype, sparsity_list)
    
    if not rows:
        return False
    
    dtype_folder = build_dtype_folder_name(dtype)
    
    # 1. Write TOPS
    tops_dir = output_base / hw_folder / "tops" / dtype_folder
    tops_csv = tops_dir / f"tops_{model_name}.csv"
    write_tops_csv(rows, tops_csv, sparsity_list)
    
    # 2. Write Latency
    latency_dir = output_base / hw_folder / "latency" / dtype_folder
    latency_csv = latency_dir / f"latency_{model_name}.csv"
    write_latency_csv(rows, latency_csv, sparsity_list)
    
    # 3. Write Speedup
    speedup_dir = output_base / hw_folder / "speedup" / dtype_folder
    speedup_csv = speedup_dir / f"speedup_{model_name}.csv"
    write_speedup_csv(rows, speedup_csv, sparsity_list)
    
    # 4. For non-SQUARE models, compute and write Total
    if not is_square and nk_list and len(nk_list) > 1:
        total_rows = compute_total_rows(rows, nk_list, sparsity_list)
        
        if total_rows:
            # Total Latency
            total_latency_csv = latency_dir / f"total_latency_{model_name}.csv"
            write_total_latency_csv(total_rows, total_latency_csv, sparsity_list)
            
            # Total Speedup
            total_speedup_csv = speedup_dir / f"total_speedup_{model_name}.csv"
            write_total_speedup_csv(total_rows, total_speedup_csv, sparsity_list)
    
    return True


# =============================================================================
# Find Available Data
# =============================================================================

def find_available_hw_folders() -> List[str]:
    """Find available hardware folders"""
    hw_folders = set()
    
    # Search from cuBLASLt directory
    if CUBLASLT_DIR.exists():
        for d in CUBLASLT_DIR.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                hw_folders.add(d.name)
    
    # Search from cuSPARSELt directory
    if CUSPARSELT_DIR.exists():
        for d in CUSPARSELT_DIR.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                hw_folders.add(d.name)
    
    return sorted(hw_folders)


def find_available_dtypes(hw_folder: str) -> List[str]:
    """Find available dtypes for a hardware"""
    dtypes = set()
    
    cublas_hw_dir = CUBLASLT_DIR / hw_folder
    if cublas_hw_dir.exists():
        for d in cublas_hw_dir.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                dtypes.add(d.name.lower())
    
    cusparse_hw_dir = CUSPARSELT_DIR / hw_folder
    if cusparse_hw_dir.exists():
        for d in cusparse_hw_dir.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                dtypes.add(d.name.lower())
    
    # Normalize dtype names
    dtype_map = {
        "fp16": "fp16",
        "bf16": "bf16",
        "int8": "int8",
        "fp8": "fp8e4m3",
        "fp8e4m3": "fp8e4m3",
        "fp4": "fp4e2m1",
        "fp4e2m1": "fp4e2m1",
    }
    
    normalized = set()
    for dt in dtypes:
        if dt in dtype_map:
            normalized.add(dtype_map[dt])
    
    return sorted(normalized)


def find_available_models(hw_folder: str, dtype: str) -> List[str]:
    """Find available models for a hardware and dtype"""
    models = set()
    dtype_folder = build_dtype_folder_name(dtype)
    
    # Search from cuBLASLt
    cublas_dir = CUBLASLT_DIR / hw_folder / dtype_folder
    if cublas_dir.exists():
        for f in cublas_dir.iterdir():
            if f.is_file() and f.suffix == ".json" and f.name.startswith("alg_search_"):
                # alg_search_Llama3.2-1B-INT8.json -> Llama3.2-1B-INT8
                model_name = f.stem.replace("alg_search_", "")
                models.add(model_name)
    
    return sorted(models)


# =============================================================================
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract TOPS, Latency, Speedup from kernel benchmark results"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help=f"Specify dtype, default auto-detect"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specify model name (including SQUARE), default auto-detect"
    )
    parser.add_argument(
        "--sparsity",
        type=str,
        default=None,
        help=f"Specify sparsity list (comma-separated), default: {','.join(SPARSITY_LIST)}"
    )
    
    args = parser.parse_args()
    
    # 解析稀疏度列表
    sparsity_list = args.sparsity.split(",") if args.sparsity else SPARSITY_LIST
    
    # Find available hardware folders
    hw_folders = find_available_hw_folders()
    
    if not hw_folders:
        print_error("No result directory found")
        print_info(f"cuBLASLt dir: {CUBLASLT_DIR}")
        print_info(f"cuSPARSELt dir: {CUSPARSELT_DIR}")
        return 1
    
    print()
    print("=" * 70)
    print("SlideSparse Kernel Benchmark Result Extraction")
    print("=" * 70)
    print()
    print_info(f"Hardware folders: {hw_folders}")
    print_info(f"Sparsity: {sparsity_list}")
    print_info(f"Output dir: {OUTPUT_DIR}")
    print()
    
    total_success = 0
    total_skip = 0
    
    for hw_folder in hw_folders:
        print_header(f"Processing hardware: {hw_folder}")
        
        # Determine dtype list
        if args.dtype:
            dtype_list = [args.dtype]
        else:
            dtype_list = find_available_dtypes(hw_folder)
        
        if not dtype_list:
            print_warning(f"  No dtype data found")
            continue
        
        for dtype in dtype_list:
            print_subheader(f"dtype: {dtype}")
            
            # Determine model list
            if args.model:
                model_list = [args.model]
            else:
                model_list = find_available_models(hw_folder, dtype)
            
            if not model_list:
                print_warning(f"    No model data found")
                continue
            
            for model in model_list:
                is_square = (model.upper() == "SQUARE")
                
                has_data = process_model(
                    hw_folder=hw_folder,
                    model_name=model,
                    dtype=dtype,
                    sparsity_list=sparsity_list,
                    output_base=OUTPUT_DIR,
                    is_square=is_square,
                )
                
                if has_data:
                    total_success += 1
                    if is_square:
                        print_success(f"    {model}: tops + latency + speedup")
                    else:
                        print_success(f"    {model}: tops + latency + speedup + total_latency + total_speedup")
                else:
                    total_skip += 1
                    print_warning(f"    {model}: no data")
    
    print()
    print("=" * 70)
    print("Extraction Complete!")
    print("=" * 70)
    print()
    print_success(f"Success: {total_success} models")
    if total_skip > 0:
        print_warning(f"Skipped: {total_skip} models (no data)")
    print()
    print_info(f"Results saved in: {OUTPUT_DIR}")
    print()
    print("Output structure:")
    print(f"  {OUTPUT_DIR}/{{hw}}/tops/{{dtype}}/tops_*.csv")
    print(f"  {OUTPUT_DIR}/{{hw}}/latency/{{dtype}}/latency_*.csv")
    print(f"  {OUTPUT_DIR}/{{hw}}/latency/{{dtype}}/total_latency_*.csv  (model only)")
    print(f"  {OUTPUT_DIR}/{{hw}}/speedup/{{dtype}}/speedup_*.csv")
    print(f"  {OUTPUT_DIR}/{{hw}}/speedup/{{dtype}}/total_speedup_*.csv  (model only)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
