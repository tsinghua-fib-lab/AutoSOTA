#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse End-to-End Results Extraction Script

Extract tokens/s data from throughput_benchmark_results,
generate summary CSV files by hardware/stage/model.

Output directory structure:
    end2end_speedup_results/
    ├── A100/
    │   ├── prefill/
    │   │   ├── absolute_throughput_Llama3.2-1B-INT8.csv
    │   │   ├── speedup_Llama3.2-1B-INT8.csv
    │   │   └── ...
    │   └── decode/
    │       └── ...
    ├── B200/
    ├── H100/
    ├── RTX4090/
    └── RTX5080/

Usage:
    # Extract results for all hardware
    python3 extract_end2end_results.py
    
    # Extract for specific hardware only
    python3 extract_end2end_results.py --hardware B200
    python3 extract_end2end_results.py --hardware A100,H100
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

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


# =============================================================================
# Configuration Constants
# =============================================================================

# 6 hardware platforms
HARDWARE_LIST = ["A100", "B200", "H100", "RTX4090", "RTX5080", "GB10"]

# 10 models (5 base x 2 quant types)
MODELS = [
    "Llama3.2-1B-INT8", "Llama3.2-1B-FP8",
    "Llama3.2-3B-INT8", "Llama3.2-3B-FP8",
    "Qwen2.5-7B-INT8", "Qwen2.5-7B-FP8",
    "Qwen2.5-14B-INT8", "Qwen2.5-14B-FP8",
    "BitNet-2B-INT8", "BitNet-2B-FP8",
]

# Test stages
STAGES = ["prefill", "decode"]

# Sparsity list (cusparselt)
SPARSITY_LIST = ["2_4", "2_6", "2_8", "2_10"]

# M value lists (from prepare_for_vllm_bench.py)
M_LIST_PREFILL = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
M_LIST_DECODE = [64, 128, 256, 512]

# Source data directory
SOURCE_DIR = _SCRIPT_DIR / "throughput_benchmark_results"

# Output directory
OUTPUT_DIR = _SCRIPT_DIR / "end2end_speedup_results"


# =============================================================================
# Utility Functions
# =============================================================================

def get_quant_from_model(model_name: str) -> str:
    """Extract quantization type from model name"""
    if "INT8" in model_name:
        return "INT8"
    elif "FP8" in model_name:
        return "FP8"
    return "UNKNOWN"


def find_hw_dirs(hw_name: str, stage: str) -> List[Path]:
    """
    Find all result directories for specified hardware in given stage
    
    Args:
        hw_name: Hardware name (e.g., "B200")
        stage: "prefill" or "decode"
    
    Returns:
        List of matching directories (may have both INT8 and FP8)
    """
    stage_dir = SOURCE_DIR / stage
    if not stage_dir.exists():
        return []
    
    matched = []
    for d in stage_dir.iterdir():
        if d.is_dir() and d.name.startswith(hw_name + "_"):
            matched.append(d)
    
    return sorted(matched)


def get_m_list(stage: str) -> List[int]:
    """Get M value list for the corresponding stage"""
    return M_LIST_PREFILL if stage == "prefill" else M_LIST_DECODE


def read_tokens_per_second(json_path: Path) -> Optional[float]:
    """
    Read tokens_per_second from JSON file
    
    Returns:
        tokens_per_second value, None if file doesn't exist or invalid
    """
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        tps = data.get("tokens_per_second", 0)
        if tps > 0:
            return tps
        return None
    except Exception:
        return None


def find_json_for_model_m(
    hw_dir: Path,
    backend: str,
    model_name: str,
    m_value: int,
    sparsity: Optional[str] = None,
) -> Path:
    """
    Build JSON file path
    
    Args:
        hw_dir: Hardware directory (e.g., B200_cc100_INT8_py312_cu129_x86_64)
        backend: "cublaslt" or "cusparselt"
        model_name: Model name (e.g., "Llama3.2-1B-INT8")
        m_value: M value
        sparsity: Sparsity (only for cusparselt)
    
    Returns:
        JSON file path
    """
    if backend == "cublaslt":
        return hw_dir / "cublaslt" / "json" / f"{model_name}_M{m_value}.json"
    elif backend == "cusparselt" and sparsity:
        return hw_dir / "cusparselt" / sparsity / "json" / f"{model_name}_M{m_value}.json"
    else:
        raise ValueError(f"Invalid backend/sparsity: {backend}/{sparsity}")


def find_hw_dir_for_model(hw_dirs: List[Path], model_name: str) -> Optional[Path]:
    """
    Find matching hardware directory based on model's quantization type
    
    Args:
        hw_dirs: List of all directories for this hardware
        model_name: Model name
    
    Returns:
        Matching hardware directory, None if not found
    """
    quant = get_quant_from_model(model_name)
    
    for hw_dir in hw_dirs:
        dir_name = hw_dir.name
        # INT8 model matches INT8 directory
        if quant == "INT8" and "_INT8_" in dir_name:
            return hw_dir
        # FP8 model matches FP8E4M3 directory
        if quant == "FP8" and "_FP8" in dir_name:
            return hw_dir
    
    return None


@dataclass
class ExtractedRow:
    """A row of extracted data"""
    m_value: int
    cublas: Optional[float]
    cusparse_2_4: Optional[float]
    cusparse_2_6: Optional[float]
    cusparse_2_8: Optional[float]
    cusparse_2_10: Optional[float]


def extract_model_data(
    hw_dirs: List[Path],
    model_name: str,
    stage: str,
) -> List[ExtractedRow]:
    """
    Extract all M value data for a single model
    
    Returns:
        List of ExtractedRow
    """
    hw_dir = find_hw_dir_for_model(hw_dirs, model_name)
    if hw_dir is None:
        # Hardware doesn't support this model's quant type
        return []
    
    m_list = get_m_list(stage)
    rows = []
    
    for m_value in m_list:
        # cuBLAS
        cublas_json = find_json_for_model_m(hw_dir, "cublaslt", model_name, m_value)
        cublas_tps = read_tokens_per_second(cublas_json)
        
        # cuSPARSE for each sparsity
        cusparse_values = {}
        for sp in SPARSITY_LIST:
            sp_json = find_json_for_model_m(hw_dir, "cusparselt", model_name, m_value, sp)
            sp_tps = read_tokens_per_second(sp_json)
            cusparse_values[sp] = sp_tps
        
        row = ExtractedRow(
            m_value=m_value,
            cublas=cublas_tps,
            cusparse_2_4=cusparse_values.get("2_4"),
            cusparse_2_6=cusparse_values.get("2_6"),
            cusparse_2_8=cusparse_values.get("2_8"),
            cusparse_2_10=cusparse_values.get("2_10"),
        )
        rows.append(row)
    
    return rows


def write_absolute_csv(rows: List[ExtractedRow], output_path: Path) -> int:
    """
    Write absolute throughput CSV
    
    Returns:
        Number of valid data rows
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    valid_count = 0
    with open(output_path, "w") as f:
        f.write("M,cuBLAS,cusparse_2_4,cusparse_2_6,cusparse_2_8,cusparse_2_10\n")
        
        for row in rows:
            values = [
                str(row.m_value),
                f"{row.cublas:.3f}" if row.cublas is not None else "",
                f"{row.cusparse_2_4:.3f}" if row.cusparse_2_4 is not None else "",
                f"{row.cusparse_2_6:.3f}" if row.cusparse_2_6 is not None else "",
                f"{row.cusparse_2_8:.3f}" if row.cusparse_2_8 is not None else "",
                f"{row.cusparse_2_10:.3f}" if row.cusparse_2_10 is not None else "",
            ]
            f.write(",".join(values) + "\n")
            
            # Count valid rows (at least has cuBLAS data)
            if row.cublas is not None:
                valid_count += 1
    
    return valid_count


def write_speedup_csv(rows: List[ExtractedRow], output_path: Path) -> int:
    """
    Write speedup CSV (relative to cuBLAS)
    
    Returns:
        Number of valid data rows
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    valid_count = 0
    with open(output_path, "w") as f:
        f.write("M,cuBLAS,cusparse_2_4,cusparse_2_6,cusparse_2_8,cusparse_2_10\n")
        
        for row in rows:
            # cuBLAS as baseline, speedup is always 1.00
            if row.cublas is None or row.cublas <= 0:
                # No baseline, leave all empty
                f.write(f"{row.m_value},,,,,\n")
                continue
            
            valid_count += 1
            
            def calc_speedup(val: Optional[float]) -> str:
                if val is None:
                    return ""
                return f"{val / row.cublas:.2f}"
            
            values = [
                str(row.m_value),
                "1.00",  # cuBLAS baseline
                calc_speedup(row.cusparse_2_4),
                calc_speedup(row.cusparse_2_6),
                calc_speedup(row.cusparse_2_8),
                calc_speedup(row.cusparse_2_10),
            ]
            f.write(",".join(values) + "\n")
    
    return valid_count


def process_hardware(hw_name: str) -> Tuple[int, int]:
    """
    Process all data for a single hardware
    
    Returns:
        (success_count, fail/skip_count)
    """
    print_header(f"Processing hardware: {hw_name}")
    
    success_count = 0
    skip_count = 0
    
    for stage in STAGES:
        print_subheader(f"Stage: {stage}")
        
        hw_dirs = find_hw_dirs(hw_name, stage)
        if not hw_dirs:
            print_warning(f"  No {stage} result directory found for {hw_name}")
            skip_count += len(MODELS)
            continue
        
        print_info(f"  Found directories: {[d.name for d in hw_dirs]}")
        
        for model_name in MODELS:
            rows = extract_model_data(hw_dirs, model_name, stage)
            
            if not rows:
                print_warning(f"  {model_name}: No data (quant type may not be supported)")
                skip_count += 1
                continue
            
            # Output directory
            output_dir = OUTPUT_DIR / hw_name / stage
            
            # Write absolute throughput CSV
            abs_csv = output_dir / f"absolute_throughput_{model_name}.csv"
            valid_abs = write_absolute_csv(rows, abs_csv)
            
            # Write speedup CSV
            speedup_csv = output_dir / f"speedup_{model_name}.csv"
            valid_speedup = write_speedup_csv(rows, speedup_csv)
            
            if valid_abs > 0:
                print_success(f"  {model_name}: {valid_abs}/{len(rows)} rows valid")
                success_count += 1
            else:
                print_warning(f"  {model_name}: All data invalid")
                skip_count += 1
    
    return success_count, skip_count


def main():
    parser = argparse.ArgumentParser(
        description="Extract end-to-end results from throughput_benchmark_results"
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default=None,
        help=f"Specify hardware (comma-separated), default all: {','.join(HARDWARE_LIST)}"
    )
    
    args = parser.parse_args()
    
    # Parse hardware list
    if args.hardware:
        hw_list = [h.strip() for h in args.hardware.split(",")]
        # Validate
        for hw in hw_list:
            if hw not in HARDWARE_LIST:
                print_error(f"Unknown hardware: {hw}")
                print_info(f"Supported hardware: {HARDWARE_LIST}")
                sys.exit(1)
    else:
        hw_list = HARDWARE_LIST
    
    print()
    print("=" * 70)
    print("SlideSparse End-to-End Results Extraction")
    print("=" * 70)
    print()
    print_info(f"Hardware list: {hw_list}")
    print_info(f"Model list: {MODELS}")
    print_info(f"Stages: {STAGES}")
    print_info(f"Sparsity: {SPARSITY_LIST}")
    print_info(f"Source dir: {SOURCE_DIR}")
    print_info(f"Output dir: {OUTPUT_DIR}")
    print()
    
    total_success = 0
    total_skip = 0
    
    for hw_name in hw_list:
        success, skip = process_hardware(hw_name)
        total_success += success
        total_skip += skip
    
    print()
    print("=" * 70)
    print("Extraction completed!")
    print("=" * 70)
    print()
    print_success(f"Success: {total_success} model x stage combinations")
    if total_skip > 0:
        print_warning(f"Skipped: {total_skip} model x stage combinations")
    print()
    print_info(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
