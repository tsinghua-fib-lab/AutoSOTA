#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse Kernel Benchmark Entry Point

Test performance difference: cuBLASLt (Dense) vs cuSPARSELt (Sparse)

Two running modes:
==================
1. Model-based mode: Specify --model <model_name>, extract real NK dimensions from checkpoint
   - Supports --Lmax: Auto-generate L=4,6,...,Lmax slided NK values
2. Square mode: No --model or --model "square", M=N=K=[64, 128, ..., 16384]

Core features:
==============
- Algorithm search for cuBLASLt dense GEMM and cuSPARSELt sparse GEMM
- Multiple sparsity configurations:
  - --Lmax: Auto-generate 2_4, 2_6, ..., 2_Lmax and 2_inf (recommended)
  - --sparsity: Manually specify sparsity list (legacy)
- Calculate and output speedup (Sparse / Dense)

Supported data types:
=====================
- FP16:    FP16 input, FP32 compute, FP16 output
- BF16:    BF16 input, FP32 compute, BF16 output
- INT8:    INT8 input, INT32 compute, INT8 output
- FP8E4M3: FP8 input, FP32 compute, FP8 output
- FP4E2M1: FP4 input, FP32 compute, FP4 output
- all:     Test all supported types

Sparsity parameter:
===================
Format: {Z}_{L} means keep Z non-zeros per L elements
- 2_4:  K_factor = 1.00, standard 2:4 sparsity (50% non-zero)
- 2_6:  K_factor = 1.33 (33% non-zero)
- 2_8:  K_factor = 1.50 (25% non-zero)
- 2_10: K_factor = 1.60 (20% non-zero)
- 2_inf: K_factor = 2.00 (theoretical max, full sparse)

Usage examples:
===============
# Model-based mode (recommend --Lmax)
python benchmark_entry.py --model Qwen2.5-0.5B --dtype fp8e4m3 --Lmax 8
# Equivalent to --sparsity 2_4,2_6,2_8,2_inf

# Square mode (no --model)
python benchmark_entry.py --dtype all -- --Lmax 10

# Test all dtype (default sparsity=2_4)
python benchmark_entry.py --model square --dtype all

# Manual sparsity (legacy)
python benchmark_entry.py --model Llama3.2-1B --dtype fp8e4m3 --sparsity 2_4,2_8

# Test cuBLASLt only (no sparsity needed)
python benchmark_entry.py --model Llama3.2-1B --dtype all --backend cublaslt


python benchmark_entry.py --model Llama3.2-1B --M-quick --dtype all --backend cusparselt --sparsity 2_4,2_6

"""

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add path
SCRIPT_DIR = Path(__file__).parent.absolute()
SLIDESPARSE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SLIDESPARSE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from slidesparse.benchmark_kernel.utils import (
    DTYPE_CONFIG,
    SUPPORTED_DTYPES,
    DEFAULT_SPARSITY_LIST,
    ALIGNMENT,
    DEFAULT_M_LIST,
    M_QUICK_LIST,
    SQUARE_M_LIST,
    hw_info,
    check_dtype_support,
    get_supported_dtypes_for_gpu,
    check_cusparselt_support,
    calculate_k_slide,
    get_k_expansion_factor,
    pad_to_alignment,
    get_sparsity_list_for_benchmark,
    get_nk_list_for_benchmark,
    build_hw_folder_name,
    build_dtype_folder_name,
    build_output_dir,
    build_result_filename,
    compute_speedup,
    merge_benchmark_results,
)


# =============================================================================
# Sub-script Paths
# =============================================================================

CUBLASLT_SCRIPT = SCRIPT_DIR / "cuBLASLt" / "alg_search.py"
CUSPARSELT_SCRIPT = SCRIPT_DIR / "cuSPARSELt" / "alg_search.py"


# =============================================================================
# Utility Functions
# =============================================================================

def run_cublaslt_search(
    dtype: str,
    model: str,  # Always pass model name (can be "square")
    m_list: List[int],
    warmup: int,
    repeat: int,
    compile_flag: bool,
    out_dir: Path,
) -> Tuple[int, Optional[Path]]:
    """
    Run cuBLASLt search
    
    Returns:
        (return_code, json_path): Return code and JSON result file path
    """
    cmd = [
        sys.executable, str(CUBLASLT_SCRIPT),
        "--dtype", dtype,
        "--model", model,
        "--warmup", str(warmup),
        "--repeat", str(repeat),
        "--out_dir", str(out_dir),
        "--m_list", ",".join(str(m) for m in m_list),
    ]
    
    if compile_flag:
        cmd.append("--compile")
    
    print(f"[cuBLASLt] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    # Get result file path (new structure: hw_folder/dtype_folder/file)
    if result.returncode == 0:
        result_dir = build_output_dir(out_dir, dtype)
        json_filename = build_result_filename("alg_search", model.upper(), "json")
        json_path = result_dir / json_filename
        if json_path.exists():
            return result.returncode, json_path
    
    return result.returncode, None


def run_cusparselt_search(
    dtype: str,
    model: str,  # Always pass model name (can be "square")
    sparsity: str,
    m_list: List[int],
    warmup: int,
    repeat: int,
    compile_flag: bool,
    out_dir: Path,
    no_segment_k: bool = False,
    no_api_search: bool = False,
) -> Tuple[int, Optional[Path]]:
    """
    Run cuSPARSELt search
    
    Returns:
        (return_code, json_path): Return code and JSON result file path
    """
    cmd = [
        sys.executable, str(CUSPARSELT_SCRIPT),
        "--dtype", dtype,
        "--model", model,
        "--warmup", str(warmup),
        "--repeat", str(repeat),
        "--out_dir", str(out_dir),
        "--m_list", ",".join(str(m) for m in m_list),
        "--sparsity", sparsity,
    ]
    
    if compile_flag:
        cmd.append("--compile")
    
    if no_segment_k:
        cmd.append("--no_segment_k")
    
    if no_api_search:
        cmd.append("--no_api_search")
    
    print(f"[cuSPARSELt] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    # Get result file path (new structure: hw_folder/dtype_folder/file)
    if result.returncode == 0:
        result_dir = build_output_dir(out_dir, dtype)
        json_filename = build_result_filename("alg_search", model.upper(), "json", sparsity)
        json_path = result_dir / json_filename
        if json_path.exists():
            return result.returncode, json_path
    
    return result.returncode, None


def load_json_results(json_path: Path) -> Optional[Dict]:
    """Load JSON result file"""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load {json_path}: {e}")
        return None


def compute_and_save_speedup(
    cublaslt_json: Path,
    cusparselt_json: Path,
    model_name: str,
    dtype: str,
    sparsity: str,
    output_dir: Path,
) -> Optional[Path]:
    """
    Compute and save speedup results
    
    Returns:
        Saved CSV file path
    """
    cublaslt_data = load_json_results(cublaslt_json)
    cusparselt_data = load_json_results(cusparselt_json)
    
    if not cublaslt_data or not cusparselt_data:
        print("[WARN] Failed to load results for speedup calculation")
        return None
    
    hw_folder = build_hw_folder_name()
    out_dir = output_dir / hw_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_filename = build_result_filename("speedup", model_name, "csv", sparsity)
    csv_path = out_dir / csv_filename
    
    k_factor = get_k_expansion_factor(sparsity)
    
    # Build CSV
    lines = [
        f"# GPU: {hw_info.gpu_full_name}",
        f"# Model: {model_name}",
        f"# dtype: {dtype.upper()}",
        f"# Sparsity: {sparsity} (K_factor={k_factor:.3f})",
        f"# Time: {datetime.datetime.now().isoformat()}",
        "M,N,K,K_slide,cublaslt_tops,cusparselt_tops,speedup,cublaslt_lat_us,cusparselt_lat_us,cublaslt_alg_id,cusparselt_alg_id,cusparselt_split_k",
    ]
    
    # Build index: (M, N, K) -> result
    # JSON structure: { "nk_entries": { "(N,K)": { "alg_by_m": { "M": {...} } } } }
    cublaslt_index = {}
    for nk_key, nk_data in cublaslt_data.get("nk_entries", {}).items():
        # nk_key format: "(N,K)" -> parse N, K
        try:
            nk_str = nk_key.strip("()")
            N, K = map(int, nk_str.split(","))
        except:
            continue
        
        for m_str, m_data in nk_data.get("alg_by_m", {}).items():
            M = int(m_str)
            if m_data:
                cublaslt_index[(M, N, K)] = m_data
    
    cusparselt_index = {}
    for nk_key, nk_data in cusparselt_data.get("nk_entries", {}).items():
        # nk_key format: "(N,K)"
        try:
            nk_str = nk_key.strip("()")
            N, K = map(int, nk_str.split(","))
        except:
            continue
        
        K_slide = calculate_k_slide(K, sparsity, dtype=dtype)
        for m_str, m_data in nk_data.get("alg_by_m", {}).items():
            M = int(m_str)
            if m_data:
                cusparselt_index[(M, N, K_slide)] = {
                    **m_data,
                    "orig_K": K,
                }
    
    # Generate result rows
    result_rows = []
    for (M, N, K), dense in sorted(cublaslt_index.items()):
        K_slide = calculate_k_slide(K, sparsity, dtype=dtype)
        sparse_key = (M, N, K_slide)
        
        if sparse_key in cusparselt_index:
            sparse = cusparselt_index[sparse_key]
            
            dense_tops = dense.get("tops", 0)
            sparse_tops = sparse.get("tops", 0)
            speedup = compute_speedup(dense_tops, sparse_tops)
            
            dense_lat = dense.get("lat_us", 0)
            sparse_lat = sparse.get("lat_us", 0)
            dense_alg_id = dense.get("alg_id", -1)
            sparse_alg_id = sparse.get("alg_id", -1)
            sparse_split_k = sparse.get("split_k", -1)
            
            result_rows.append({
                "M": M, "N": N, "K": K, "K_slide": K_slide,
                "dense_tops": dense_tops, "sparse_tops": sparse_tops,
                "speedup": speedup,
                "dense_lat": dense_lat, "sparse_lat": sparse_lat,
                "dense_alg_id": dense_alg_id, "sparse_alg_id": sparse_alg_id,
                "sparse_split_k": sparse_split_k,
            })
    
    # Sort by M, N, K
    result_rows.sort(key=lambda x: (x["M"], x["N"], x["K"]))
    
    for row in result_rows:
        line = (
            f"{row['M']},{row['N']},{row['K']},{row['K_slide']},"
            f"{row['dense_tops']:.6f},{row['sparse_tops']:.6f},{row['speedup']:.4f},"
            f"{row['dense_lat']:.3f},{row['sparse_lat']:.3f},"
            f"{row['dense_alg_id']},{row['sparse_alg_id']},{row['sparse_split_k']}"
        )
        lines.append(line)
    
    csv_path.write_text("\n".join(lines))
    
    return csv_path


def print_speedup_summary(csv_path: Path):
    """Print speedup summary"""
    try:
        lines = csv_path.read_text().strip().split("\n")
        data_lines = [l for l in lines if not l.startswith("#") and l.strip() and "," in l]
        
        if len(data_lines) < 2:  # Need at least header + 1 data line
            return
        
        header = data_lines[0].split(",")
        speedup_idx = header.index("speedup") if "speedup" in header else -1
        m_idx = header.index("M") if "M" in header else 0
        
        if speedup_idx < 0:
            return
        
        speedups = []
        m_speedups = {}
        
        for line in data_lines[1:]:
            parts = line.split(",")
            if len(parts) > speedup_idx:
                try:
                    sp = float(parts[speedup_idx])
                    m = int(parts[m_idx])
                    speedups.append(sp)
                    if m not in m_speedups:
                        m_speedups[m] = []
                    m_speedups[m].append(sp)
                except ValueError:
                    continue
        
        if speedups:
            avg_speedup = sum(speedups) / len(speedups)
            max_speedup = max(speedups)
            min_speedup = min(speedups)
            
            print()
            print("    ┌────────────────────────────────────────┐")
            print("    │          Speedup Summary               │")
            print("    ├────────────────────────────────────────┤")
            print(f"    │  Average Speedup: {avg_speedup:>7.3f}x             │")
            print(f"    │  Max Speedup:     {max_speedup:>7.3f}x             │")
            print(f"    │  Min Speedup:     {min_speedup:>7.3f}x             │")
            print("    ├────────────────────────────────────────┤")
            print("    │  Speedup by M:                         │")
            
            for m in sorted(m_speedups.keys()):
                m_avg = sum(m_speedups[m]) / len(m_speedups[m])
                print(f"    │    M={m:<6}: {m_avg:>7.3f}x                 │")
            
            print("    └────────────────────────────────────────┘")
    except Exception as e:
        print(f"[WARN] Failed to print speedup summary: {e}")


# =============================================================================
# Main Flow
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="SlideSparse Kernel Benchmark Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Model-based mode (recommend --Lmax)
  python benchmark_entry.py --model Qwen2.5-0.5B --dtype fp8e4m3 --Lmax 8
  # Equivalent to --sparsity 2_4,2_6,2_8,2_inf
  
  # Square mode (no --model)
  python benchmark_entry.py --dtype int8 --Lmax 10
  
  # Test standard 2:4 sparsity only (default)
  python benchmark_entry.py --model Llama3.2-1B --dtype bf16
  
  # Test cuBLASLt only (no sparsity needed)
  python benchmark_entry.py --model Qwen2.5-0.5B --dtype all --backend cublaslt
  
  # Manual sparsity (legacy)
  python benchmark_entry.py --model Llama3.2-1B --dtype fp8e4m3 --sparsity 2_4,2_8
        """
    )
    
    # Mode selection
    p.add_argument("--model", type=str, default=None,
                   help="Model name (no value or 'square' enters Square mode)")
    p.add_argument("--dtype", type=str, required=True,
                   choices=SUPPORTED_DTYPES + ["all"],
                   help="Data type (fp16, bf16, int8, fp8e4m3, fp4e2m1, all)")
    
    # Sparsity config (Lmax and sparsity mutually exclusive)
    sparsity_group = p.add_mutually_exclusive_group()
    sparsity_group.add_argument("--Lmax", type=int, default=None,
                   help="Max L value. Auto-generate 2_4, 2_6, ..., 2_Lmax and 2_inf. "
                        "e.g., --Lmax 8 equals --sparsity 2_4,2_6,2_8,2_inf")
    sparsity_group.add_argument("--sparsity", type=str, default=None,
                   help="Manual sparsity list, comma-separated (2_4, 2_6, 2_8, 2_10, 2_inf)")
    
    p.add_argument("--backend", type=str, default="all",
                   choices=["all", "cublaslt", "cusparselt"],
                   help="Test backend")
    
    # M list control
    p.add_argument("--M-quick", action="store_true", dest="m_quick",
                   help="Use quick M list (Model-based mode only)")
    p.add_argument("--m_list", type=str, default=None,
                   help="Custom M list, comma-separated (Square mode: M=N=K uses this list)")
    
    # Test parameters
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--repeat", type=int, default=100)
    
    # Compilation control
    p.add_argument("--compile", action="store_true", dest="force_compile",
                   help="Force recompile CUDA extensions")
    
    # cuSPARSELt options
    p.add_argument("--no_segment_k", action="store_true",
                   help="Disable Segment-K test")
    p.add_argument("--no_api_search", action="store_true",
                   help="Disable official API search comparison")
    
    # Output control
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands only, don't execute")
    p.add_argument("--no-speedup", action="store_true", dest="no_speedup",
                   help="Don't calculate speedup")
    
    return p.parse_args()


def run_benchmark_for_dtype(args, dtype: str, m_list: List[int], model_name: str, sparsity_list: List[str]):
    """Run benchmark for a single dtype"""
    
    # Check dtype support
    supported, reason = check_dtype_support(dtype)
    if not supported:
        print(f"[SKIP] {dtype}: {reason}")
        return
    
    # Check cuSPARSELt support
    backend = args.backend
    if backend in ("all", "cusparselt"):
        cusparselt_ok, cusparselt_reason = check_cusparselt_support()
        if not cusparselt_ok:
            if backend == "cusparselt":
                print(f"[SKIP] {dtype}: {cusparselt_reason}")
                return
            else:
                print(f"[WARN] {cusparselt_reason}")
                print("[INFO] Will only test cuBLASLt")
                backend = "cublaslt"
    
    # Output directory
    out_dir = SCRIPT_DIR
    cublaslt_json_path = None
    cusparselt_json_paths = {}
    
    # Display config
    print()
    print(f"{'=' * 60}")
    print(f"Benchmark dtype={dtype.upper()}")
    print(f"{'=' * 60}")
    
    # Run cuBLASLt
    if backend in ("all", "cublaslt"):
        print()
        print(f"[{dtype}] Running cuBLASLt Dense GEMM Search")
        
        ret, cublaslt_json_path = run_cublaslt_search(
            dtype=dtype,
            model=model_name,
            m_list=m_list,
            warmup=args.warmup,
            repeat=args.repeat,
            compile_flag=args.force_compile,
            out_dir=out_dir / "cuBLASLt" / "alg_search_results",
        )
        
        if ret != 0:
            print(f"[ERROR] cuBLASLt search failed with code {ret}")
            return
    
    # Run cuSPARSELt
    if backend in ("all", "cusparselt"):
        for sparsity in sparsity_list:
            print()
            print(f"[{dtype}] Running cuSPARSELt Sparse GEMM Search (sparsity={sparsity})")
            
            k_factor = get_k_expansion_factor(sparsity)
            print(f"[INFO] K_factor = {k_factor:.3f}")
            
            ret, json_path = run_cusparselt_search(
                dtype=dtype,
                model=model_name,
                sparsity=sparsity,
                m_list=m_list,
                warmup=args.warmup,
                repeat=args.repeat,
                compile_flag=args.force_compile,
                out_dir=out_dir / "cuSPARSELt" / "alg_search_results",
                no_segment_k=args.no_segment_k,
                no_api_search=args.no_api_search,
            )
            
            if ret != 0:
                print(f"[ERROR] cuSPARSELt search (sparsity={sparsity}) failed")
            else:
                cusparselt_json_paths[sparsity] = json_path
    
    # Compute speedup
    if backend == "all" and not args.no_speedup:
        speedup_dir = out_dir / "speedup_results"
        
        for sparsity, cusparselt_json in cusparselt_json_paths.items():
            if cublaslt_json_path and cusparselt_json:
                print(f"\n  [{dtype}] Computing speedup for sparsity={sparsity}...")
                
                csv_path = compute_and_save_speedup(
                    cublaslt_json=cublaslt_json_path,
                    cusparselt_json=cusparselt_json,
                    model_name=model_name,
                    dtype=dtype,
                    sparsity=sparsity,
                    output_dir=speedup_dir,
                )
                
                if csv_path:
                    print(f"    Saved: {csv_path}")
                    print_speedup_summary(csv_path)


def main():
    args = parse_args()
    
    # Determine dtype list to test
    if args.dtype == "all":
        dtype_list = get_supported_dtypes_for_gpu()
        print(f"[INFO] dtype=all, will test: {dtype_list}")
    else:
        dtype_list = [args.dtype]
    
    # Unified m_list processing (same for both modes)
    if args.m_list:
        m_list = [int(x.strip()) for x in args.m_list.split(",")]
    elif args.m_quick:
        m_list = list(M_QUICK_LIST)
    else:
        m_list = list(M_QUICK_LIST)  # Default: [16, 128, 1024, 4096, 16384]
    
    m_list = [pad_to_alignment(m) for m in m_list]
    
    # Generate sparsity list: --Lmax priority, else --sparsity, default ["2_4"]
    if args.Lmax is not None:
        # Use Lmax to auto-generate sparsity list (same as reference + 2_inf)
        sparsity_list = get_sparsity_list_for_benchmark(args.Lmax)
        print(f"[INFO] Lmax={args.Lmax}, sparsity list: {sparsity_list}")
    elif args.sparsity:
        # Manual sparsity (legacy)
        sparsity_list = [s.strip() for s in args.sparsity.split(",")]
    else:
        # Default: only test standard 2:4 sparsity
        sparsity_list = ["2_4"]
    
    # Use get_nk_list_for_benchmark to determine mode and get model_name
    # This ensures model path lookup logic matches sub-scripts
    is_square_mode = (args.model is None or args.model.lower() == "square")
    
    if is_square_mode:
        # Square mode
        model_name = "SQUARE"
        mode = "square"
    else:
        # Model-based mode: use get_nk_list_for_benchmark to get correct model_name
        # Note: we don't pass L_max here, as L_max only affects sparsity list, not model_name
        _, model_name, mode = get_nk_list_for_benchmark(
            model=args.model,
            m_list=m_list if is_square_mode else None,
        )
    
    # Display config
    print("=" * 60)
    print("SlideSparse Kernel Benchmark")
    print("=" * 60)
    print(f"GPU: {hw_info.gpu_full_name} ({hw_info.cc_tag})")
    print(f"Mode: {mode.upper()}")
    print(f"Model: {model_name}")
    if is_square_mode:
        print(f"M=N=K: {m_list}")
    else:
        print(f"M_list: {m_list}")
    print(f"dtype: {dtype_list}")
    print(f"Backend: {args.backend}")
    if args.backend in ("all", "cusparselt"):
        print(f"Sparsity: {sparsity_list}")
    print(f"warmup={args.warmup}, repeat={args.repeat}")
    print("=" * 60)
    
    if args.dry_run:
        print("[DRY-RUN] Not executing, only showing config")
        return
    
    # Run benchmark for each dtype
    for dtype in dtype_list:
        run_benchmark_for_dtype(args, dtype, m_list, model_name, sparsity_list)
    
    # Output directory
    out_dir = SCRIPT_DIR
    
    print()
    print("=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)
    print()
    print("Results saved to:")
    if args.backend in ("all", "cublaslt"):
        print(f"  - cuBLASLt:   {out_dir / 'cuBLASLt' / 'alg_search_results'}")
    if args.backend in ("all", "cusparselt"):
        print(f"  - cuSPARSELt: {out_dir / 'cuSPARSELt' / 'alg_search_results'}")
    if args.backend == "all" and not args.no_speedup:
        print(f"  - Speedup:    {out_dir / 'speedup_results'}")


if __name__ == "__main__":
    main()
