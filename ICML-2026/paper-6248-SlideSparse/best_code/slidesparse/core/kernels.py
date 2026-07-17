# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse Triton Kernel Loading Module

Contains:
- Dequant + Bias kernel (FP8/INT8 shared)
- FP8 Quant kernels (quant_only, quant_slide)
- INT8 Quant kernels (quant_only, quant_slide)

torch.compile compatibility strategy (approach 1+4):
===================================================
1. Preload at module import: scan build dir, preload all compiled tuned kernels
2. Compile-time guard: if cache miss during torch.compile tracing, raise clear error

This ensures:
- Subprocesses auto-complete preload when importing module
- Hot path only has dict reads, no filesystem ops
- Edge cases (e.g., new model not precompiled) get friendly error

Directory structure:
    build/{hw_dir_name}/{kernel_name}_tuned_{model_name}.py
    
Example:
    build/RTX5080_cc120_py312_cu129_x86_64/dequant_bias_tuned_Llama3.2-1B-FP8.py
"""

from pathlib import Path
from typing import Callable, Dict, Optional
import os

import torch
from torch.library import Library
from vllm.logger import init_logger
from vllm.platforms import current_platform

from slidesparse.utils import load_tuned_module, build_hw_dir_name

logger = init_logger(__name__)

# Custom Op Library (shared with gemm_wrapper)
# Note: using FRAGMENT mode to append to existing slidesparse library
_triton_lib = Library("slidesparse", "FRAGMENT")


# ============================================================================
# Directory Configuration
# ============================================================================

_CSRC_DIR = Path(__file__).parent.parent / "csrc"


# ============================================================================
# Kernel Search Configuration
# ============================================================================

# Basic kernel filename mapping
_BASIC_KERNEL_FILES = {
    "dequant_bias": "basic_dequant_bias_triton.py",
    "quant_only":   "basic_quant_only_triton.py",
    "quant_slide":  "basic_quant_slide_triton.py",
}


def _load_basic_kernel(kernel_dir: Path, kernel_type: str) -> object:
    """Load basic kernel module (no model-specific tuning)"""
    if kernel_type not in _BASIC_KERNEL_FILES:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    basic_file = kernel_dir / _BASIC_KERNEL_FILES[kernel_type]
    if not basic_file.exists():
        raise FileNotFoundError(f"Basic kernel not found: {basic_file}")
    
    import importlib.util
    spec = importlib.util.spec_from_file_location(kernel_type, basic_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    logger.info_once(f"Using basic kernel: {basic_file.name}")
    return module


def _search_kernel(
    kernel_dir: Path,
    kernel_type: str,
    tuned_prefix: str,
    model_name: str,
) -> object:
    """
    Search and load kernel module
    
    Search order:
    1. First try tuned kernel: build/{hw_dir}/{tuned_prefix}_{model_name}.py
    2. If not found, fallback to basic kernel: basic_{kernel_type}_triton.py
    
    Args:
        kernel_dir: kernel directory (e.g., csrc/fused_dequant_bias_triton)
        kernel_type: kernel type (dequant_bias, quant_only, quant_slide)
        tuned_prefix: tuned file prefix (e.g., dequant_bias_tuned)
        model_name: model name (should be base name without -SlideSparse- suffix)
    
    Returns:
        Loaded module object
    """
    build_dir = kernel_dir / "build"
    
    # 1. Try to load tuned kernel
    try:
        module = load_tuned_module(tuned_prefix, model_name, build_dir)
        logger.info_once(f"Loaded tuned kernel for model: {model_name}")
        return module
    except FileNotFoundError:
        pass
    
    # 2. Fallback to basic kernel
    logger.warning(f"Tuned kernel not found for {model_name}, using basic kernel")
    return _load_basic_kernel(kernel_dir, kernel_type)


# ============================================================================
# Dequant + Bias Kernel (FP8/INT8 shared)
# ============================================================================

# Cache: model_name -> kernel function
_dequant_bias_cache: Dict[str, Callable] = {}


def _load_dequant_bias_kernel(model_name: str) -> Callable:
    """Load Triton dequant+bias kernel (lazy load per model)"""
    if model_name in _dequant_bias_cache:
        return _dequant_bias_cache[model_name]
    
    # Compile-time guard: if cache miss during torch.compile tracing, raise clear error
    if torch.compiler.is_compiling():
        raise RuntimeError(
            f"Triton kernel 'dequant_bias' for model '{model_name}' was not preloaded!\n"
            f"This error occurs during torch.compile tracing.\n"
            f"Possible causes:\n"
            f"  1. Tuned kernel not found in build directory\n"
            f"  2. init_slidesparse('{model_name}') was not called before model loading\n"
            f"Fix: Run kernel autotuning for this model, or check SLIDESPARSE_MODEL_NAME env var."
        )
    
    kernel_dir = _CSRC_DIR / "fused_dequant_bias_triton"
    module = _search_kernel(
        kernel_dir,
        kernel_type="dequant_bias",
        tuned_prefix="dequant_bias_tuned",
        model_name=model_name,
    )
    
    fn = module.dequant_bias_triton
    # Fix module name to avoid Dynamo trying to import invalid names (e.g., '.' in Llama3.2-1B-FP8 is parsed as package separator)
    fn.__module__ = "slidesparse.core.kernels"
    _dequant_bias_cache[model_name] = fn
    logger.info_once(f"Dequant+bias kernel loaded for model: {model_name}")
    return fn


def dequant_bias_kernel(
    gemm_output: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    model_name: str,
) -> torch.Tensor:
    """
    Dequant + Bias: output = gemm_output * scale_a * scale_b + bias
    
    Supports BF16, FP32, INT32 input (auto-detect)
    Called via torch.library custom op for torch.compile compatibility.
    
    Args:
        gemm_output: GEMM output
        scale_a: activation scale
        scale_b: weight scale  
        bias: optional bias
        out_dtype: output type
        model_name: model name (for loading corresponding tuned kernel)
    """
    # Convert torch.dtype to string (required by custom op schema)
    out_dtype_str = str(out_dtype).replace("torch.", "")
    return torch.ops.slidesparse.dequant_bias(
        gemm_output, scale_a, scale_b, bias, out_dtype_str, model_name
    )


# ============================================================================
# FP8 Quant Only Kernel - cuBLASLt specific
# ============================================================================

# Cache: model_name -> kernel function
_quant_only_fp8_cache: Dict[str, Callable] = {}


def _load_quant_only_fp8_kernel(model_name: str) -> Callable:
    """Load Triton FP8 quant kernel (lazy load per model)"""
    if model_name in _quant_only_fp8_cache:
        return _quant_only_fp8_cache[model_name]
    
    # Compile-time guard
    if torch.compiler.is_compiling():
        raise RuntimeError(
            f"Triton kernel 'quant_only_fp8' for model '{model_name}' was not preloaded!\n"
            f"This error occurs during torch.compile tracing.\n"
            f"Possible causes:\n"
            f"  1. Tuned kernel not found in build directory\n"
            f"  2. init_slidesparse('{model_name}') was not called before model loading\n"
            f"Fix: Run kernel autotuning for this model, or check SLIDESPARSE_MODEL_NAME env var."
        )
    
    kernel_dir = _CSRC_DIR / "quant_only_triton"
    module = _search_kernel(
        kernel_dir,
        kernel_type="quant_only",
        tuned_prefix="quant_only_tuned",
        model_name=model_name,
    )
    
    fn = module.quant_only_fp8_triton
    # Fix module name to avoid Dynamo trying to import invalid names
    fn.__module__ = "slidesparse.core.kernels"
    _quant_only_fp8_cache[model_name] = fn
    logger.info_once(f"FP8 quant kernel loaded for model: {model_name}")
    return fn


def quant_only_fp8_kernel(
    input: torch.Tensor,
    model_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    FP8 Per-token Quantization
    
    Called via torch.library custom op for torch.compile compatibility.
    
    Args:
        input: [M, K] BF16
        model_name: model name (for loading corresponding tuned kernel)
        
    Returns:
        qinput: [M_pad, K_pad] FP8, M_pad=ceil16(M), K_pad=ceil32(K)
        scale_a: [M_pad] FP32, padding region is 1.0
    """
    return torch.ops.slidesparse.quant_only_fp8(input, model_name)


# ============================================================================
# FP8 Quant + Slide Kernel - cuSPARSELt specific
# ============================================================================

# Cache: model_name -> kernel function
_quant_slide_fp8_cache: Dict[str, Callable] = {}


def _load_quant_slide_fp8_kernel(model_name: str) -> Callable:
    """Load Triton FP8 quant+slide kernel (lazy load per model)"""
    if model_name in _quant_slide_fp8_cache:
        return _quant_slide_fp8_cache[model_name]
    
    # Compile-time guard
    if torch.compiler.is_compiling():
        raise RuntimeError(
            f"Triton kernel 'quant_slide_fp8' for model '{model_name}' was not preloaded!\n"
            f"This error occurs during torch.compile tracing.\n"
            f"Possible causes:\n"
            f"  1. Tuned kernel not found in build directory\n"
            f"  2. init_slidesparse('{model_name}') was not called before model loading\n"
            f"Fix: Run kernel autotuning for this model, or check SLIDESPARSE_MODEL_NAME env var."
        )
    
    kernel_dir = _CSRC_DIR / "fused_quant_slide_triton"
    module = _search_kernel(
        kernel_dir,
        kernel_type="quant_slide",
        tuned_prefix="quant_slide_tuned",
        model_name=model_name,
    )
    
    fn = module.quant_slide_fp8_triton
    # Fix module name to avoid Dynamo trying to import invalid names
    fn.__module__ = "slidesparse.core.kernels"
    _quant_slide_fp8_cache[model_name] = fn
    logger.info_once(f"FP8 quant+slide kernel loaded for model: {model_name}")
    return fn


def quant_slide_fp8_kernel(
    input: torch.Tensor,
    model_name: str,
    L: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    FP8 Per-token Quantization + SlideSparse Slide
    
    Called via torch.library custom op for torch.compile compatibility.
    
    Args:
        input: [M, K] BF16
        model_name: model name (for loading corresponding tuned kernel)
        L: sparsity group size (default 8)
        
    Returns:
        qinput: [M_pad, K_slide_pad] FP8
                M_pad=ceil16(M), K_slide_pad=ceil32(K_slide)
                K_slide = K * (L - 2) / (L / 2) = K * 2 * (L - 2) / L
        scale_a: [M_pad] FP32, padding region is 1.0
    """
    return torch.ops.slidesparse.quant_slide_fp8(input, model_name, L)


# ============================================================================
# INT8 Quant Only Kernel - cuBLASLt specific
# ============================================================================

# Cache: model_name -> kernel function
_quant_only_int8_cache: Dict[str, Callable] = {}


def _load_quant_only_int8_kernel(model_name: str) -> Callable:
    """Load Triton INT8 quant kernel (lazy load per model)"""
    if model_name in _quant_only_int8_cache:
        return _quant_only_int8_cache[model_name]
    
    # Compile-time guard
    if torch.compiler.is_compiling():
        raise RuntimeError(
            f"Triton kernel 'quant_only_int8' for model '{model_name}' was not preloaded!\n"
            f"This error occurs during torch.compile tracing.\n"
            f"Possible causes:\n"
            f"  1. Tuned kernel not found in build directory\n"
            f"  2. init_slidesparse('{model_name}') was not called before model loading\n"
            f"Fix: Run kernel autotuning for this model, or check SLIDESPARSE_MODEL_NAME env var."
        )
    
    kernel_dir = _CSRC_DIR / "quant_only_triton"
    module = _search_kernel(
        kernel_dir,
        kernel_type="quant_only",
        tuned_prefix="quant_only_tuned",
        model_name=model_name,
    )
    
    fn = module.quant_only_int8_triton
    # Fix module name to avoid Dynamo trying to import invalid names
    fn.__module__ = "slidesparse.core.kernels"
    _quant_only_int8_cache[model_name] = fn
    logger.info_once(f"INT8 quant kernel loaded for model: {model_name}")
    return fn


def quant_only_int8_kernel(
    input: torch.Tensor,
    model_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    INT8 Per-token Quantization (Symmetric)
    
    Called via torch.library custom op for torch.compile compatibility.
    
    Args:
        input: [M, K] BF16
        model_name: model name (for loading corresponding tuned kernel)
        
    Returns:
        qinput: [M_pad, K_pad] INT8, M_pad=ceil16(M), K_pad=ceil32(K)
        scale_a: [M_pad] FP32, padding region is 1.0
    """
    return torch.ops.slidesparse.quant_only_int8(input, model_name)


# ============================================================================
# INT8 Quant + Slide Kernel - cuSPARSELt specific
# ============================================================================

# Cache: model_name -> kernel function
_quant_slide_int8_cache: Dict[str, Callable] = {}


def _load_quant_slide_int8_kernel(model_name: str) -> Callable:
    """Load Triton INT8 quant+slide kernel (lazy load per model)"""
    if model_name in _quant_slide_int8_cache:
        return _quant_slide_int8_cache[model_name]
    
    # Compile-time guard
    if torch.compiler.is_compiling():
        raise RuntimeError(
            f"Triton kernel 'quant_slide_int8' for model '{model_name}' was not preloaded!\n"
            f"This error occurs during torch.compile tracing.\n"
            f"Possible causes:\n"
            f"  1. Tuned kernel not found in build directory\n"
            f"  2. init_slidesparse('{model_name}') was not called before model loading\n"
            f"Fix: Run kernel autotuning for this model, or check SLIDESPARSE_MODEL_NAME env var."
        )
    
    kernel_dir = _CSRC_DIR / "fused_quant_slide_triton"
    module = _search_kernel(
        kernel_dir,
        kernel_type="quant_slide",
        tuned_prefix="quant_slide_tuned",
        model_name=model_name,
    )
    
    fn = module.quant_slide_int8_triton
    # Fix module name to avoid Dynamo trying to import invalid names
    fn.__module__ = "slidesparse.core.kernels"
    _quant_slide_int8_cache[model_name] = fn
    logger.info_once(f"INT8 quant+slide kernel loaded for model: {model_name}")
    return fn


def quant_slide_int8_kernel(
    input: torch.Tensor,
    model_name: str,
    L: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    INT8 Per-token Quantization + SlideSparse Slide (Symmetric)
    
    Called via torch.library custom op for torch.compile compatibility.
    
    Args:
        input: [M, K] BF16
        model_name: model name (for loading corresponding tuned kernel)
        L: sparsity group size (default 8)
        
    Returns:
        qinput: [M_pad, K_slide_pad] INT8
                M_pad=ceil16(M), K_slide_pad=ceil32(K_slide)
        scale_a: [M_pad] FP32, padding region is 1.0
    """
    return torch.ops.slidesparse.quant_slide_int8(input, model_name, L)


# ============================================================================
# Custom Op Registration (torch.compile compatible)
# ============================================================================
#
# Register torch.library custom ops for Triton kernels:
# - Real impl: calls preloaded Triton kernel
# - Fake impl: returns empty tensor with correct shape, for Dynamo tracing
# ============================================================================

def _ceil16(x: int) -> int:
    """Round up to multiple of 16"""
    return (x + 15) // 16 * 16


def _ceil32(x: int) -> int:
    """Round up to multiple of 32"""
    return (x + 31) // 32 * 32


def _register_triton_custom_ops():
    """Register all Triton kernel custom ops"""
    
    # ========== dequant_bias ==========
    _triton_lib.define(
        "dequant_bias(Tensor gemm_output, Tensor scale_a, Tensor scale_b, "
        "Tensor? bias, str out_dtype_str, str model_name) -> Tensor"
    )
    
    # dtype string mapping
    _dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float": torch.float32,
        "int32": torch.int32,
        "int": torch.int32,
    }
    
    def _dequant_bias_impl(gemm_output, scale_a, scale_b, bias, out_dtype_str, model_name):
        if bias is None:
            # Create dummy bias inside Graph to ensure CUDAGraph manages its lifetime
            bias = torch.zeros(
                gemm_output.shape[1],
                dtype=gemm_output.dtype,
                device=gemm_output.device
            )
        fn = _load_dequant_bias_kernel(model_name)
        out_dtype = _dtype_map.get(out_dtype_str, torch.bfloat16)
        return fn(gemm_output, scale_a, scale_b, bias, out_dtype)
    
    def _dequant_bias_fake(gemm_output, scale_a, scale_b, bias, out_dtype_str, model_name):
        M, N = gemm_output.shape
        out_dtype = _dtype_map.get(out_dtype_str, torch.bfloat16)
        return torch.empty((M, N), dtype=out_dtype, device=gemm_output.device)
    
    _triton_lib.impl("dequant_bias", _dequant_bias_impl, "CUDA")
    _triton_lib._register_fake("dequant_bias", _dequant_bias_fake)
    
    # ========== quant_only_fp8 ==========
    _triton_lib.define(
        "quant_only_fp8(Tensor input, str model_name) -> (Tensor, Tensor)"
    )
    
    def _quant_only_fp8_impl(input, model_name):
        fn = _load_quant_only_fp8_kernel(model_name)
        return fn(input)
    
    def _quant_only_fp8_fake(input, model_name):
        M, K = input.shape
        M_pad = _ceil16(M)
        K_pad = _ceil32(K)
        fp8_dtype = current_platform.fp8_dtype()
        qinput = torch.empty((M_pad, K_pad), dtype=fp8_dtype, device=input.device)
        scale_a = torch.empty((M_pad,), dtype=torch.float32, device=input.device)
        return qinput, scale_a
    
    _triton_lib.impl("quant_only_fp8", _quant_only_fp8_impl, "CUDA")
    _triton_lib._register_fake("quant_only_fp8", _quant_only_fp8_fake)
    
    # ========== quant_slide_fp8 ==========
    _triton_lib.define(
        "quant_slide_fp8(Tensor input, str model_name, int L) -> (Tensor, Tensor)"
    )
    
    def _quant_slide_fp8_impl(input, model_name, L):
        fn = _load_quant_slide_fp8_kernel(model_name)
        return fn(input, L)
    
    def _quant_slide_fp8_fake(input, model_name, L):
        M, K = input.shape
        M_pad = _ceil16(M)
        # K_slide = K * 2 * (L - 2) / L
        K_slide = K * 2 * (L - 2) // L
        K_slide_pad = _ceil32(K_slide)
        fp8_dtype = current_platform.fp8_dtype()
        qinput = torch.empty((M_pad, K_slide_pad), dtype=fp8_dtype, device=input.device)
        scale_a = torch.empty((M_pad,), dtype=torch.float32, device=input.device)
        return qinput, scale_a
    
    _triton_lib.impl("quant_slide_fp8", _quant_slide_fp8_impl, "CUDA")
    _triton_lib._register_fake("quant_slide_fp8", _quant_slide_fp8_fake)
    
    # ========== quant_only_int8 ==========
    _triton_lib.define(
        "quant_only_int8(Tensor input, str model_name) -> (Tensor, Tensor)"
    )
    
    def _quant_only_int8_impl(input, model_name):
        fn = _load_quant_only_int8_kernel(model_name)
        return fn(input)
    
    def _quant_only_int8_fake(input, model_name):
        M, K = input.shape
        M_pad = _ceil16(M)
        K_pad = _ceil32(K)
        qinput = torch.empty((M_pad, K_pad), dtype=torch.int8, device=input.device)
        scale_a = torch.empty((M_pad,), dtype=torch.float32, device=input.device)
        return qinput, scale_a
    
    _triton_lib.impl("quant_only_int8", _quant_only_int8_impl, "CUDA")
    _triton_lib._register_fake("quant_only_int8", _quant_only_int8_fake)
    
    # ========== quant_slide_int8 ==========
    _triton_lib.define(
        "quant_slide_int8(Tensor input, str model_name, int L) -> (Tensor, Tensor)"
    )
    
    def _quant_slide_int8_impl(input, model_name, L):
        fn = _load_quant_slide_int8_kernel(model_name)
        return fn(input, L)
    
    def _quant_slide_int8_fake(input, model_name, L):
        M, K = input.shape
        M_pad = _ceil16(M)
        K_slide = K * 2 * (L - 2) // L
        K_slide_pad = _ceil32(K_slide)
        qinput = torch.empty((M_pad, K_slide_pad), dtype=torch.int8, device=input.device)
        scale_a = torch.empty((M_pad,), dtype=torch.float32, device=input.device)
        return qinput, scale_a
    
    _triton_lib.impl("quant_slide_int8", _quant_slide_int8_impl, "CUDA")
    _triton_lib._register_fake("quant_slide_int8", _quant_slide_int8_fake)
    
    logger.info_once("Triton kernel custom ops registered")


# Register custom ops at module load
_register_triton_custom_ops()


# ============================================================================
# Module Preload
# ============================================================================

def _preload_all_kernels() -> None:
    """
    Scan build directory and preload all compiled tuned kernels
    
    Called at module import to ensure:
    1. Subprocesses auto-complete preload when importing module
    2. Hot path only has dict reads, no filesystem ops
    
    Optimization:
    If env var SLIDESPARSE_MODEL_NAME exists, only load kernels for that model.
    This significantly reduces startup time and memory usage. If specified model
    not found, prints warning and falls back to full scan to avoid "config error
    but silently runs inefficient path" issue.
    """
    try:
        hw_dir = build_hw_dir_name()
    except Exception as e:
        # Skip preload if cannot determine hardware directory (e.g., GPU unavailable)
        logger.debug(f"Skipping kernel preload: {e}")
        return
    
    # Check for target model optimization
    target_model = os.environ.get("SLIDESPARSE_MODEL_NAME")
    if target_model:
        _preload_target_model_kernels(target_model)
        return

    # Fallback to full scan (Legacy behavior)
    _preload_scan_all_kernels(hw_dir)


def _preload_target_model_kernels(target_model_name: str) -> None:
    """Targeted preload for specific model
    
    Args:
        target_model_name: base model name (SLIDESPARSE_MODEL_NAME now strictly without slide suffix)
    """
    # target_model_name is now base name, no need to extract
    model_name = target_model_name
    logger.info(f"Optimization enabled: Only preloading kernels for model '{model_name}'")
    
    # Try to load various kernels
    
    # 1. Dequant Bias
    try:
        _load_dequant_bias_kernel(model_name)
    except Exception as e:
        # Don't fallback, warn explicitly
        logger.warning(f"Target dequant_bias kernel for '{model_name}' not found: {e}")

    # 2. Quant Only (FP8 & INT8)
    try:
        _load_quant_only_fp8_kernel(model_name)
    except Exception as e:
        logger.warning(f"Target quant_only_fp8 kernel for '{model_name}' not found: {e}")
        
    try:
        _load_quant_only_int8_kernel(model_name)
    except Exception as e:
        logger.warning(f"Target quant_only_int8 kernel for '{model_name}' not found: {e}")

    # 3. Quant Slide (FP8 & INT8)
    try:
        _load_quant_slide_fp8_kernel(model_name)
    except Exception as e:
        logger.warning(f"Target quant_slide_fp8 kernel for '{model_name}' not found: {e}")

    try:
        _load_quant_slide_int8_kernel(model_name)
    except Exception as e:
        logger.warning(f"Target quant_slide_int8 kernel for '{model_name}' not found: {e}")


def _preload_scan_all_kernels(hw_dir: str) -> None:
    """Full scan of build directory to load all kernels (original logic)"""
    preloaded_count = 0
    
    # 1. Scan dequant_bias kernels
    dequant_build_dir = _CSRC_DIR / "fused_dequant_bias_triton" / "build" / hw_dir
    if dequant_build_dir.exists():
        for kernel_file in dequant_build_dir.glob("dequant_bias_tuned_*.py"):
            # Extract model_name from filename: dequant_bias_tuned_{model_name}.py
            stem = kernel_file.stem  # e.g., "dequant_bias_tuned_Llama3.2-1B-FP8"
            prefix = "dequant_bias_tuned_"
            if stem.startswith(prefix):
                model_name = stem[len(prefix):]
                try:
                    _load_dequant_bias_kernel(model_name)
                    preloaded_count += 1
                except Exception as e:
                    logger.debug(f"Failed to preload dequant_bias for {model_name}: {e}")
    
    # 2. Scan quant_only kernels (load both FP8 and INT8)
    quant_only_build_dir = _CSRC_DIR / "quant_only_triton" / "build" / hw_dir
    if quant_only_build_dir.exists():
        for kernel_file in quant_only_build_dir.glob("quant_only_tuned_*.py"):
            stem = kernel_file.stem
            prefix = "quant_only_tuned_"
            if stem.startswith(prefix):
                model_name = stem[len(prefix):]
                try:
                    _load_quant_only_fp8_kernel(model_name)
                    preloaded_count += 1
                except Exception as e:
                    logger.debug(f"Failed to preload quant_only_fp8 for {model_name}: {e}")
                try:
                    _load_quant_only_int8_kernel(model_name)
                    preloaded_count += 1
                except Exception as e:
                    logger.debug(f"Failed to preload quant_only_int8 for {model_name}: {e}")
    
    # 3. Scan quant_slide kernels (load both FP8 and INT8)
    quant_slide_build_dir = _CSRC_DIR / "fused_quant_slide_triton" / "build" / hw_dir
    if quant_slide_build_dir.exists():
        for kernel_file in quant_slide_build_dir.glob("quant_slide_tuned_*.py"):
            stem = kernel_file.stem
            prefix = "quant_slide_tuned_"
            if stem.startswith(prefix):
                model_name = stem[len(prefix):]
                try:
                    _load_quant_slide_fp8_kernel(model_name)
                    preloaded_count += 1
                except Exception as e:
                    logger.debug(f"Failed to preload quant_slide_fp8 for {model_name}: {e}")
                try:
                    _load_quant_slide_int8_kernel(model_name)
                    preloaded_count += 1
                except Exception as e:
                    logger.debug(f"Failed to preload quant_slide_int8 for {model_name}: {e}")
    
    if preloaded_count > 0:
        logger.info(f"Preloaded {preloaded_count} Triton kernels from {hw_dir}")


# Execute preload at module import
_preload_all_kernels()


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Search helper
    "_search_kernel",
    "_load_basic_kernel",
    "_BASIC_KERNEL_FILES",
    "_CSRC_DIR",
    
    # Preload
    "_preload_all_kernels",
    
    # Dequant kernel (shared)
    "_load_dequant_bias_kernel",
    "dequant_bias_kernel",
    
    # FP8 Quant kernels
    "_load_quant_only_fp8_kernel",
    "quant_only_fp8_kernel",
    "_load_quant_slide_fp8_kernel",
    "quant_slide_fp8_kernel",
    
    # INT8 Quant kernels
    "_load_quant_only_int8_kernel",
    "quant_only_int8_kernel",
    "_load_quant_slide_int8_kernel",
    "quant_slide_int8_kernel",
]
