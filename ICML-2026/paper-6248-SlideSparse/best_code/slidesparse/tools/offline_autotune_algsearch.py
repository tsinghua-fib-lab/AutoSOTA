#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse Unified Offline Tuning and Algorithm Search Script

- CUDA cuBLAS: cuBLASLt GEMM algorithm search
- CUDA cuSPARSE: cuSPARSELt 2:4 sparse algorithm search
- Triton Dequant: Dequantization + Bias fusion kernel
- Triton Quant Slide: Quantization + Slide fusion kernel
- Triton Quant Only: Pure quantization kernel


Model naming convention
============
This script has different model naming convention from other tools scripts:

1. Input format: Accepts **base name** (without quant suffix)
   - Recommended: Qwen2.5-0.5B, Llama3.2-1B
   - Can also have suffix (will be auto-removed): Qwen2.5-0.5B-INT8 -> Qwen2.5-0.5B

2. Tuning type: Determined by --dtype parameter (int8/fp8/all), unrelated to model suffix
   - This is because INT8 and FP8 models have same NK config, only quant method differs

3. Passed to sub-scripts:
   - CUDA kernel: Full checkpoint name (e.g., Qwen2.5-0.5B-INT8)
   - Triton kernel: Any existing checkpoint name

Difference from other scripts:
- model_download.py / throughput_benchmark.py: Use registry key (e.g., qwen2.5-0.5b-fp8)
- weight_convert_entry.py: Use full directory name or registry key
- offline_autotune_algsearch.py (this script): Use base name, auto-expand by --dtype


Parameter description:
=========
--model:       Model name, supports base name or full name with suffix
               e.g., "Qwen2.5-0.5B" or "Qwen2.5-0.5B-INT8"
               Note: INT8/FP8 models have same NK config, suffix will be ignored
--dtype:       Input data type, int8/fp8/all (required)
               Specifies quant types to tune, unrelated to model suffix
--outdtype:    Output data type, bf16 (default) optionally --inner-32 for high-precision accumulation
--Lmax:        Maximum sparsity length
--M-quick:     Quick M mode [16, 128, 1024, 4096, 16384]
--m_list:      Custom M list
--warmup:      Warmup iterations (default 25)
--repeat:      Repeat iterations (default 100)
--kernels:     Kernels to tune, format "1,1,0,1,1"
               Order: cuBLAS, cuSPARSE, Triton Dequant, Triton Quant Slide, Triton Quant Only
--skip-build:  Skip build step


Usage examples:
=========
# Tune all kernels (INT8 + FP8), model name supports base name
python3 offline_autotune_algsearch.py --model Qwen2.5-0.5B --dtype all --M-quick

# Can also have suffix (will be ignored, actually tunes by --dtype)
python3 offline_autotune_algsearch.py --model Qwen2.5-0.5B-INT8 --dtype all --M-quick

# Tune INT8 only
python3 offline_autotune_algsearch.py --model Llama3.2-1B --dtype int8 --M-quick

# Tune CUDA kernels only (cuBLAS + cuSPARSE), with high-precision accumulation
python3 offline_autotune_algsearch.py --model Qwen2.5-0.5B --dtype int8 --inner-32 --kernels 1,1,0,0,0

# Tune Triton kernels only
python3 offline_autotune_algsearch.py --model Llama3.2-1B --dtype fp8 --kernels 0,0,1,1,1

# Multi-model tuning
python3 offline_autotune_algsearch.py --model Qwen2.5-0.5B,Llama3.2-1B --dtype all --M-quick
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import traceback

# 添加项目根目录到 path
_TOOLS_DIR = Path(__file__).parent
_SLIDESPARSE_ROOT = _TOOLS_DIR.parent
_PROJECT_ROOT = _SLIDESPARSE_ROOT.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from slidesparse.utils import (
    hw_info,
    DEFAULT_M_LIST,
    M_QUICK_LIST,
    get_nk_list_for_search,
    model_base_name,
    normalize_model_input,
    find_any_model_checkpoint,
    find_model_checkpoint_for_dtype,
)

from slidesparse.tools.utils import (
    print_header,
    print_subheader,
    print_info,
    print_success,
    print_warning,
    print_error,
    Colors,
)


# =============================================================================
# Constants
# =============================================================================

# Kernel name list (fixed order)
KERNEL_NAMES = [
    "cuBLASLt GEMM",
    "cuSPARSELt GEMM",
    "Triton Dequant + Bias",
    "Triton Quant + Slide",
    "Triton Quant Only",
]

KERNEL_SCRIPTS = {
    "cublaslt": _SLIDESPARSE_ROOT / "search" / "cuBLASLt_AlgSearch" / "alg_search.py",
    "cusparselt": _SLIDESPARSE_ROOT / "search" / "cuSPARSELt_AlgSearch" / "alg_search.py",
    "triton_dequant": _SLIDESPARSE_ROOT / "csrc" / "fused_dequant_bias_triton" / "autotune_autogen_dequant_bias.py",
    "triton_quant_slide": _SLIDESPARSE_ROOT / "csrc" / "fused_quant_slide_triton" / "autotune_autogen_quant_slide.py",
    "triton_quant_only": _SLIDESPARSE_ROOT / "csrc" / "quant_only_triton" / "autotune_autogen_quant_only.py",
}

BUILD_SCRIPTS = {
    "cublaslt": _SLIDESPARSE_ROOT / "csrc" / "cublaslt_gemm" / "build_cublaslt.py",
    "cusparselt": _SLIDESPARSE_ROOT / "csrc" / "cusparselt_gemm" / "build_cusparselt.py",
    "compress": _SLIDESPARSE_ROOT / "weight_convert" / "build_compress.py",
}

# Default model list (using base name)
DEFAULT_MODELS = ["Qwen2.5-0.5B", "Llama3.2-1B"]

# Default warmup/repeat
DEFAULT_WARMUP = 25
DEFAULT_REPEAT = 100


# =============================================================================
# Utility Functions
# =============================================================================

def parse_kernel_mask(mask_str: str) -> List[bool]:
    """
    Parse kernel mask string
    
    Args:
        mask_str: String in format "1,1,0,1,1"
        
    Returns:
        Boolean list indicating which kernels to tune
    """
    parts = mask_str.split(",")
    if len(parts) != 5:
        raise ValueError(f"Kernel mask must have 5 values, got: {mask_str}")
    return [int(p.strip()) == 1 for p in parts]


def get_dtype_for_cuda(dtype: str, inner_32: bool) -> Tuple[str, str]:
    """
    Get dtype and outdtype for CUDA kernel
    
    Args:
        dtype: Input type (int8/fp8)
        inner_32: Use high-precision accumulation
        
    Returns:
        (dtype, outdtype)
    """
    if inner_32:
        if dtype == "int8":
            return "int8", "int32"
        else:  # fp8
            return "fp8e4m3", "fp32"
    else:
        if dtype == "int8":
            # cuBLASLt INT8 default int32, cuSPARSELt INT8 default bf16
            return "int8", "bf16"
        else:  # fp8
            return "fp8e4m3", "bf16"


def detect_oom_error(output: str) -> Tuple[bool, Optional[str]]:
    """
    Detect if output contains CUDA OOM error
    
    Returns:
        (is_oom, suggestion): Whether OOM error, and suggestion
    """
    oom_patterns = [
        "CUDA out of memory",
        "OutOfMemoryError",
        "CUDA error: out of memory",
        "RuntimeError: CUDA",
    ]
    
    for pattern in oom_patterns:
        if pattern in output:
            suggestion = None
            if "Tried to allocate" in output:
                # Extract allocation size
                match = re.search(r"Tried to allocate ([\d.]+) ([GM]iB)", output)
                if match:
                    size = match.group(1)
                    unit = match.group(2)
                    suggestion = (
                        f"Failed to allocate {size} {unit} VRAM.\n"
                        f"Suggestions:\n"
                        f"  1. Reduce M list: --m_list 64,128,256,512,1024,2048,4096,8192,16384\n"
                        f"  2. Set env var: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
                        f"  3. Restart process to clear VRAM fragmentation"
                    )
            return True, suggestion
    return False, None


def run_subprocess(cmd: List[str], name: str) -> Tuple[bool, str]:
    """
    Run subprocess and capture output
    
    Args:
        cmd: Command list
        name: Process name (for logging)
        
    Returns:
        (success, output)
    """
    try:
        print_info(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        if result.returncode != 0:
            # Detect OOM error and provide suggestion
            is_oom, suggestion = detect_oom_error(output)
            if is_oom:
                oom_msg = f"\n{'='*60}\n[OOM Detected] Out of memory error\n{'='*60}\n"
                if suggestion:
                    oom_msg += suggestion + "\n"
                oom_msg += "="*60 + "\n"
                return False, oom_msg + output
            return False, output
        return True, output
    except Exception as e:
        return False, f"[{name}] Exception: {str(e)}\n{traceback.format_exc()}"


# =============================================================================
# Build Step
# =============================================================================

def run_build_step(force: bool = True) -> bool:
    """
    Run build step
    
    Args:
        force: Force rebuild
        
    Returns:
        Success or not
    """
    print_header("Step 0: Building CUDA Extensions")
    
    success_count = 0
    total_count = len(BUILD_SCRIPTS)
    
    for name, script_path in BUILD_SCRIPTS.items():
        print_subheader(f"Building {name}")
        
        if not script_path.exists():
            print_error(f"Script not found: {script_path}")
            continue
        
        cmd = [sys.executable, str(script_path), "build"]
        if force:
            cmd.append("--force")
        
        success, output = run_subprocess(cmd, name)
        if success:
            print_success(f"{name} build succeeded")
            success_count += 1
        else:
            print_error(f"{name} build failed:")
            print(output)
    
    return success_count == total_count


# =============================================================================
# CUDA Kernel Tuning
# =============================================================================

def run_cuda_tune(
    kernel_type: str,  # "cublaslt" or "cusparselt"
    dtype: str,
    outdtype: str,
    model: str,
    Lmax: Optional[int],
    m_quick: bool,
    m_list: Optional[List[int]],
    warmup: int,
    repeat: int,
) -> Tuple[bool, str]:
    """
    Run CUDA kernel tuning
    
    Returns:
        (success, message)
    """
    script_path = KERNEL_SCRIPTS[kernel_type]
    
    if not script_path.exists():
        return False, f"Script not found: {script_path}"
    
    cmd = [
        sys.executable, str(script_path),
        "--dtype", dtype,
        "--outdtype", outdtype,
        "--model", model,
        "--warmup", str(warmup),
        "--repeat", str(repeat),
        "--compile",  # Ensure compiled
    ]
    
    if Lmax:
        cmd.extend(["--Lmax", str(Lmax)])
    
    if m_quick:
        cmd.append("--M-quick")
    elif m_list:
        cmd.extend(["--m_list", ",".join(map(str, m_list))])
    
    return run_subprocess(cmd, kernel_type)


# =============================================================================
# Triton Kernel Tuning
# =============================================================================

def run_triton_tune(
    kernel_type: str,  # "triton_dequant", "triton_quant_slide", "triton_quant_only"
    model: str,
    Lmax: Optional[int],
    m_quick: bool,
    m_list: Optional[List[int]],
    warmup: int,
    repeat: int,
) -> Tuple[bool, str]:
    """
    Run Triton kernel tuning
    
    Returns:
        (success, message)
    """
    script_path = KERNEL_SCRIPTS[kernel_type]
    
    if not script_path.exists():
        return False, f"Script not found: {script_path}"
    
    cmd = [
        sys.executable, str(script_path),
        "--model", model,
        "--warmup", str(warmup),
        "--repeat", str(repeat),
    ]
    
    if Lmax:
        cmd.extend(["--Lmax", str(Lmax)])
    
    if m_quick:
        cmd.append("--M-quick")
    elif m_list:
        cmd.extend(["--m_list", ",".join(map(str, m_list))])
    
    return run_subprocess(cmd, kernel_type)


# =============================================================================
# Main Tuning Flow
# =============================================================================

def run_autotune(
    dtypes: List[str],
    outdtype: str,
    inner_32: bool,
    models: List[str],  # base names
    Lmax: Optional[int],
    m_quick: bool,
    m_list: Optional[List[int]],
    warmup: int,
    repeat: int,
    kernel_mask: List[bool],
    skip_build: bool,
) -> dict:
    """
    Run full tuning flow
    
    Args:
        models: Model base name list (without quant suffix)
    
    Returns:
        Result dict {kernel_name: {model: (success, message)}}
    """
    results = {}
    
    # Step 0: Build (if not skipped)
    if not skip_build:
        if not run_build_step(force=True):
            print_warning("Some builds failed, continuing tuning...")
    else:
        print_info("Skipping build step")
    
    # Step 1-5: Tune by kernel order
    kernel_keys = ["cublaslt", "cusparselt", "triton_dequant", "triton_quant_slide", "triton_quant_only"]
    
    for idx, (kernel_key, kernel_name, enabled) in enumerate(zip(kernel_keys, KERNEL_NAMES, kernel_mask)):
        step_num = idx + 1
        
        if not enabled:
            print_header(f"Step {step_num}: {kernel_name} [Skipped]")
            continue
        
        print_header(f"Step {step_num}: {kernel_name}")
        results[kernel_key] = {}
        
        for base_name in models:
            print_subheader(f"Model: {base_name}")
            
            # Find any existing checkpoint to get NK config
            # (INT8 and FP8 have same NK, only need to find one)
            ckpt_path, ckpt_name = find_any_model_checkpoint(base_name)
            if ckpt_path is None:
                print_error(f"Checkpoint directory not found for model '{base_name}'")
                results[kernel_key][base_name] = (False, f"Checkpoint not found")
                continue
            
            # Get NK config
            try:
                nk_list, _ = get_nk_list_for_search(ckpt_name, Lmax)
                print_info(f"NK combinations: {len(nk_list)} (from {ckpt_name})")
            except ValueError as e:
                print_error(f"Model validation failed: {e}")
                results[kernel_key][base_name] = (False, str(e))
                continue
            
            # CUDA Kernel: tune separately by dtype
            if kernel_key in ["cublaslt", "cusparselt"]:
                for dtype in dtypes:
                    # Check FP8 hardware support (CC >= 8.9)
                    if dtype == "fp8" and not hw_info.supports_fp8:
                        print_warning(
                            f"GPU {hw_info.gpu_name} ({hw_info.cc_tag}) doesn't support native FP8, "
                            f"skipping {kernel_name} FP8 tuning"
                        )
                        key = f"{base_name}_{dtype}"
                        results[kernel_key][key] = (
                            True,  # Mark as success (skip is not failure)
                            f"Skipped: GPU doesn't support FP8 (requires CC >= 8.9)"
                        )
                        continue
                    
                    # Find checkpoint for this dtype (for naming output files)
                    dtype_ckpt = find_model_checkpoint_for_dtype(base_name, dtype)
                    if dtype_ckpt is None:
                        print_warning(f"{base_name} {dtype.upper()} checkpoint not found, skipping")
                        continue
                    model_name_for_tune = dtype_ckpt.name  # 如 "Qwen2.5-0.5B-INT8"
                    
                    actual_dtype, actual_outdtype = get_dtype_for_cuda(dtype, inner_32)
                    
                    # cuBLASLt INT8 forced to int32
                    if kernel_key == "cublaslt" and dtype == "int8":
                        actual_outdtype = "int32"
                    
                    print_info(f"dtype={actual_dtype}, outdtype={actual_outdtype}")
                    
                    success, output = run_cuda_tune(
                        kernel_key,
                        actual_dtype,
                        actual_outdtype,
                        model_name_for_tune,  # Pass full model name
                        Lmax,
                        m_quick,
                        m_list,
                        warmup,
                        repeat,
                    )
                    
                    key = f"{base_name}_{dtype}"
                    results[kernel_key][key] = (success, output)
                    
                    if success:
                        print_success(f"{kernel_name} ({dtype}) completed")
                    else:
                        print_error(f"{kernel_name} ({dtype}) failed:")
                        print(output[-2000:] if len(output) > 2000 else output)
            
            # Triton Kernel: use any existing checkpoint name
            else:
                # Triton kernel is dtype-insensitive, use any found checkpoint name
                success, output = run_triton_tune(
                    kernel_key,
                    ckpt_name,  # Use found checkpoint name
                    Lmax,
                    m_quick,
                    m_list,
                    warmup,
                    repeat,
                )
                
                results[kernel_key][base_name] = (success, output)
                
                if success:
                    print_success(f"{kernel_name} completed")
                else:
                    print_error(f"{kernel_name} failed:")
                    print(output[-2000:] if len(output) > 2000 else output)
    
    return results


def print_summary(results: dict, kernel_mask: List[bool]) -> None:
    """Print tuning summary"""
    print_header("Tuning Summary")
    
    kernel_keys = ["cublaslt", "cusparselt", "triton_dequant", "triton_quant_slide", "triton_quant_only"]
    
    success_total = 0
    fail_total = 0
    skip_total = 0
    
    for idx, (kernel_key, kernel_name, enabled) in enumerate(zip(kernel_keys, KERNEL_NAMES, kernel_mask)):
        if not enabled:
            print(f"  {kernel_name}: {Colors.YELLOW}[Skipped]{Colors.NC}")
            skip_total += 1
            continue
        
        if kernel_key not in results:
            print(f"  {kernel_name}: {Colors.RED}[Not Executed]{Colors.NC}")
            fail_total += 1
            continue
        
        kernel_results = results[kernel_key]
        success_count = sum(1 for s, _ in kernel_results.values() if s)
        fail_count = len(kernel_results) - success_count
        
        success_total += success_count
        fail_total += fail_count
        
        if fail_count == 0:
            status = f"{Colors.GREEN}[All Succeeded]{Colors.NC} ({success_count}/{len(kernel_results)})"
        elif success_count == 0:
            status = f"{Colors.RED}[All Failed]{Colors.NC} ({fail_count}/{len(kernel_results)})"
        else:
            status = f"{Colors.YELLOW}[Partial Success]{Colors.NC} ({success_count}/{len(kernel_results)})"
        
        print(f"  {kernel_name}: {status}")
        
        # Check for OOM errors, list separately
        oom_failures = []
        for key, (success, output) in kernel_results.items():
            if not success and "[OOM Detected]" in output:
                oom_failures.append(key)
        
        if oom_failures:
            print(f"    {Colors.RED}⚠ OOM Errors:{Colors.NC} {', '.join(oom_failures)}")
    
    print()
    print(f"Total: Success {success_total}, Failed {fail_total}, Skipped {skip_total}")
    
    # If OOM failures exist, print unified suggestions
    all_oom = False
    for kernel_key in results:
        for key, (success, output) in results[kernel_key].items():
            if not success and "[OOM Detected]" in output:
                all_oom = True
                break
        if all_oom:
            break
    
    if all_oom:
        print()
        print(f"{Colors.YELLOW}{'='*60}{Colors.NC}")
        print(f"{Colors.YELLOW}Note: Out of Memory (OOM) errors detected{Colors.NC}")
        print(f"{Colors.YELLOW}{'='*60}{Colors.NC}")
        print("Possible solutions:")
        print("  1. Reduce M list: --m_list 64,128,256,512,1024,2048,4096,8192,16384")
        print("  2. Set environment variable: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        print("  3. Run failed models separately (avoid memory fragmentation)")
        print("  4. Restart process to clear GPU memory")
        print(f"{Colors.YELLOW}{'='*60}{Colors.NC}")


# =============================================================================
# Main Function
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="SlideSparse unified offline tuning and algorithm search script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Required args
    parser.add_argument(
        "--dtype", required=True, choices=["int8", "fp8", "all"],
        help="Input data type: int8, fp8, or all (tune both)"
    )
    
    # CUDA specific args
    parser.add_argument(
        "--outdtype", default="bf16", choices=["bf16", "fp32", "int32"],
        help="Output data type (default bf16, cuBLAS+INT8 auto fallback to int32)"
    )
    parser.add_argument(
        "--inner-32", action="store_true", dest="inner_32",
        help="Use high precision accumulation: FP8→FP32, INT8→INT32 (CUDA Kernel only)"
    )
    
    # Common args
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model base name (e.g. Qwen2.5-0.5B) or with suffix (suffix ignored). Comma-separated for multiple models."
    )
    parser.add_argument(
        "--Lmax", type=int, default=None,
        help="Max sparse length L (generates all NK for L=4,6,...,Lmax)"
    )
    parser.add_argument(
        "--M-quick", action="store_true", dest="m_quick",
        help="M-quick mode: use fixed M list [16, 128, 1024, 4096, 16384]"
    )
    parser.add_argument(
        "--m_list", type=str, default=None,
        help="Custom M list, comma-separated (e.g. 16,128,512,2048,16384)"
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP,
        help=f"Warmup iterations (default {DEFAULT_WARMUP})"
    )
    parser.add_argument(
        "--repeat", type=int, default=DEFAULT_REPEAT,
        help=f"Repeat iterations (default {DEFAULT_REPEAT})"
    )
    
    # Kernel selection
    parser.add_argument(
        "--kernels", type=str, default="1,1,1,1,1",
        help='Kernels to tune, format "1,1,0,1,1" (order: cuBLAS,cuSPARSE,Dequant,QuantSlide,QuantOnly)'
    )
    
    # Other options
    parser.add_argument(
        "--skip-build", action="store_true", dest="skip_build",
        help="Skip build step (assume already built)"
    )
    parser.add_argument(
        "--info", action="store_true",
        help="Show config info only, do not run tuning"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 解析参数
    if args.dtype == "all":
        dtypes = ["int8", "fp8"]
    else:
        dtypes = [args.dtype]
    
    # 解析模型列表，并标准化为 base name
    if args.model:
        raw_models = [m.strip() for m in args.model.split(",")]
    else:
        raw_models = DEFAULT_MODELS
        print_warning(f"Model not specified, using default: {raw_models}")
    
    # Normalize model names: extract base name, validate existence
    models = []
    model_hints = {}  # Record user-specified quant type (if any)
    for raw in raw_models:
        try:
            base, quant_hint = normalize_model_input(raw)
            if base not in models:  # Deduplicate
                models.append(base)
                if quant_hint:
                    model_hints[base] = quant_hint
            # If user input name with suffix, notify it will be ignored
            if raw != base:
                print_info(f"Model '{raw}' → base name '{base}' (suffix ignored, tuning by --dtype)")
        except ValueError as e:
            print_error(str(e))
            return 1
    
    # Parse M list
    m_list = None
    if args.m_list:
        m_list = [int(x.strip()) for x in args.m_list.split(",")]
    
    # Parse Kernel mask
    try:
        kernel_mask = parse_kernel_mask(args.kernels)
    except ValueError as e:
        print_error(str(e))
        return 1
    
    # Display config info
    print_header("SlideSparse Unified Offline Tuning")
    print(f"  GPU:           {hw_info.gpu_full_name} ({hw_info.cc_tag})")
    print(f"  Python:        {hw_info.python_tag}")
    print(f"  CUDA:          {hw_info.cuda_tag}")
    print(f"  Arch:          {hw_info.arch_tag}")
    print()
    print(f"  Data types:    {dtypes}")
    print(f"  Output dtype:  {args.outdtype}")
    print(f"  High precision:{' Yes' if args.inner_32 else ' No'}")
    print(f"  Models (base): {models}")
    print(f"  Lmax:          {args.Lmax or 'Not specified'}")
    print(f"  M-quick:       {' Yes' if args.m_quick else ' No'}")
    print(f"  M list:        {m_list or ('M_QUICK_LIST' if args.m_quick else 'DEFAULT_M_LIST')}")
    print(f"  Warmup/Repeat: {args.warmup}/{args.repeat}")
    print()
    print("  Kernel tuning:")
    for name, enabled in zip(KERNEL_NAMES, kernel_mask):
        status = f"{Colors.GREEN}✓{Colors.NC}" if enabled else f"{Colors.RED}✗{Colors.NC}"
        print(f"    {status} {name}")
    
    if args.info:
        return 0
    
    # Run tuning
    results = run_autotune(
        dtypes=dtypes,
        outdtype=args.outdtype,
        inner_32=args.inner_32,
        models=models,
        Lmax=args.Lmax,
        m_quick=args.m_quick,
        m_list=m_list,
        warmup=args.warmup,
        repeat=args.repeat,
        kernel_mask=kernel_mask,
        skip_build=args.skip_build,
    )
    
    # Print summary
    print_summary(results, kernel_mask)
    
    # Check for failures
    has_failure = any(
        not success
        for kernel_results in results.values()
        for success, _ in kernel_results.values()
    )
    
    return 1 if has_failure else 0


if __name__ == "__main__":
    sys.exit(main())
