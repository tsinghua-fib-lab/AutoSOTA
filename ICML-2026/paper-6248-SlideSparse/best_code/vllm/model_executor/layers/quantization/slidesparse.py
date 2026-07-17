# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SlideSparse Integration Entry - vLLM Forwarding Stub

This file serves as a bridge between SlideSparse and vLLM.
It imports functionality from the plugin module (slidesparse/) into vLLM's quantization framework.

Design Principles:
==================
1. Minimal invasive modification: This is the only new file in vLLM source code
2. All logic resides in the plugin module (slidesparse/)
3. This file only handles imports and forwarding

Usage:
======
SlideSparse is enabled by default, controlled via environment variables:
    vllm serve model_path --quantization compressed-tensors

No need to modify --quantization parameter, keep using compressed-tensors.
SlideSparse transparently hooks into CompressedTensorsW8A8Fp8/Int8.

Environment Variables:
======================
- DISABLE_SLIDESPARSE=1: Fully disable SlideSparse, use vLLM native path
- USE_CUBLASLT=1: Enable cuBLASLt kernel
- USE_CUSPARSELT=1: Enable cuSPARSELt kernel
- INNER_DTYPE_32=1: GEMM uses high-precision accumulation (FP8->FP32, INT8->INT32)

Architecture:
=============
FP8 Path:
- cuBLASLt_FP8_linear: cuBLASLt FP8 GEMM + Triton dequant
- cuSPARSELt_FP8_linear: cuSPARSELt 2:4 sparse FP8 GEMM + Triton dequant
- cutlass_FP8_linear: vLLM CUTLASS kernel fallback

INT8 Path:
- cuBLASLt_INT8_linear: cuBLASLt INT8 GEMM (INT32 output) + Triton dequant
- cuSPARSELt_INT8_linear: cuSPARSELt 2:4 sparse INT8 GEMM + Triton dequant
- cutlass_INT8_linear: vLLM CUTLASS kernel fallback (supports asymmetric quantization)
"""

import os
import sys

# Add slidesparse module to Python path
# Repo root calculation: vllm/model_executor/layers/quantization/slidesparse.py
#                       -> go back 4 levels to repo root
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_CURRENT_DIR))))
)
_SLIDESPARSE_PATH = os.path.join(_REPO_ROOT, "slidesparse")

if _SLIDESPARSE_PATH not in sys.path:
    sys.path.insert(0, _SLIDESPARSE_PATH)

# Import from plugin module
try:
    # Config
    from slidesparse.core.config import (
        is_slidesparse_enabled,
        is_cublaslt_enabled,
        is_cusparselt_enabled,
        is_inner_dtype_32,
        get_slidesparse_status,
        get_sparsity_config,
        get_sparsity_str,
        clear_sparsity_cache,
    )
    
    # FP8
    from slidesparse.core.SlideSparseLinearMethod_FP8 import (
        SlideSparseFp8LinearMethod,
        SlideSparseFp8LinearOp,
        cuBLASLt_FP8_linear,
        cuSPARSELt_FP8_linear,
        cutlass_FP8_linear,
        wrap_scheme_fp8,
    )
    
    # INT8
    from slidesparse.core.SlideSparseLinearMethod_INT8 import (
        SlideSparseInt8LinearMethod,
        SlideSparseInt8LinearOp,
        cuBLASLt_INT8_linear,
        cuSPARSELt_INT8_linear,
        cutlass_INT8_linear,
        wrap_scheme_int8,
    )
    
    # Shared components
    from slidesparse.core.gemm_wrapper import _get_gemm_extension
    from slidesparse.core.profiler import print_profile_stats, reset_profile_stats
    
    _IMPORT_SUCCESS = True
except ImportError as e:
    import warnings
    warnings.warn(
        f"Failed to import slidesparse modules: {e}. "
        "SlideSparse features will be disabled."
    )
    _IMPORT_SUCCESS = False
    
    # Fallback stub functions - Config
    def is_slidesparse_enabled():
        return False
    
    def is_cublaslt_enabled():
        return False
    
    def is_cusparselt_enabled():
        return False
    
    def is_inner_dtype_32():
        return False
    
    def get_slidesparse_status():
        return "SlideSparse backend UNAVAILABLE (import failed)"
    
    def get_sparsity_config():
        return (2, 8, 8/6)
    
    def get_sparsity_str():
        return "2_8"
    
    def clear_sparsity_cache():
        pass
    
    # Fallback stub functions - FP8
    def wrap_scheme_fp8(scheme):
        return scheme
    
    class SlideSparseFp8LinearMethod:
        pass
    
    class SlideSparseFp8LinearOp:
        pass
    
    def cuBLASLt_FP8_linear(*args, **kwargs):
        raise NotImplementedError("SlideSparse import failed")
    
    def cuSPARSELt_FP8_linear(*args, **kwargs):
        raise NotImplementedError("SlideSparse import failed")
    
    def cutlass_FP8_linear(*args, **kwargs):
        raise NotImplementedError("SlideSparse import failed")
    
    # Fallback stub functions - INT8
    def wrap_scheme_int8(scheme):
        return scheme
    
    class SlideSparseInt8LinearMethod:
        pass
    
    class SlideSparseInt8LinearOp:
        pass
    
    def cuBLASLt_INT8_linear(*args, **kwargs):
        raise NotImplementedError("SlideSparse import failed")
    
    def cuSPARSELt_INT8_linear(*args, **kwargs):
        raise NotImplementedError("SlideSparse import failed")
    
    def cutlass_INT8_linear(*args, **kwargs):
        raise NotImplementedError("SlideSparse import failed")
    
    # Fallback stub functions - Shared
    def _get_gemm_extension(backend):
        raise NotImplementedError("SlideSparse import failed")
    
    def print_profile_stats():
        pass
    
    def reset_profile_stats():
        pass

__all__ = [
    # Config
    "is_slidesparse_enabled",
    "is_cublaslt_enabled",
    "is_cusparselt_enabled",
    "is_inner_dtype_32",
    "get_slidesparse_status",
    "get_sparsity_config",
    "get_sparsity_str",
    "clear_sparsity_cache",
    
    # FP8 linear layers
    "SlideSparseFp8LinearMethod",
    "SlideSparseFp8LinearOp",
    "cuBLASLt_FP8_linear",
    "cuSPARSELt_FP8_linear",
    "cutlass_FP8_linear",
    "wrap_scheme_fp8",
    
    # INT8 linear layers
    "SlideSparseInt8LinearMethod",
    "SlideSparseInt8LinearOp",
    "cuBLASLt_INT8_linear",
    "cuSPARSELt_INT8_linear",
    "cutlass_INT8_linear",
    "wrap_scheme_int8",
    
    # Shared components
    "_get_gemm_extension",
    "print_profile_stats",
    "reset_profile_stats",
]
