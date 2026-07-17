# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse Configuration Module

Environment Variable Control:
=============================

1. DISABLE_SLIDESPARSE=1
   - Effect: Fully disable SlideSparse hook, use vLLM native path
   - Default: 0 (SlideSparse enabled)

2. USE_CUBLASLT=1
   - Effect: Switch from external CUTLASS path to cuBLASLt path
   - Default: 0 (use external CUTLASS)
   - Prerequisite: Effective when SlideSparse is enabled
   - Note: Mutually exclusive with USE_CUSPARSELT

3. USE_CUSPARSELT=1
   - Effect: Switch from external CUTLASS path to cuSPARSELt path
   - Default: 0 (use external CUTLASS)
   - Prerequisite: Effective when SlideSparse is enabled
   - Note: Mutually exclusive with USE_CUBLASLT

4. INNER_DTYPE_32=1
   - Effect: GEMM uses high-precision accumulation (FP8->FP32, INT8->INT32)
   - Default: 0 (use BF16)
   - Prerequisite: Effective when USE_CUBLASLT=1 or USE_CUSPARSELT=1

5. SPARSITY=2_8
   - Effect: Specify sparsity format Z:L (e.g., 2:8, 2:6, 2:10)
   - Default: 2_8
   - Prerequisite: Effective when USE_CUSPARSELT=1 (online compression needs sparsity format)
   - Format: Z_L (e.g., "2_8" means 2:8 sparsity)

Dispatch Logic:
===============
1. DISABLE_SLIDESPARSE=1 -> vLLM native path (unrelated to SlideSparse)
2. DISABLE_SLIDESPARSE=0 (default) -> SlideSparse hook path
   2-1. USE_CUBLASLT=0 and USE_CUSPARSELT=0 (default) -> external CUTLASS kernel (fallback)
   2-2. USE_CUBLASLT=1 -> cuBLASLt kernel
   2-3. USE_CUSPARSELT=1 -> cuSPARSELt kernel (requires SPARSITY config)
   2-4. Both set to 1 -> Error (mutually exclusive)
"""

import os

from vllm.logger import init_logger

# Import sparsity config parsing functions from top-level utils
from slidesparse.utils import (
    get_sparsity_config_cached,
    get_sparsity_str,
    clear_sparsity_cache,
)

logger = init_logger(__name__)

# Module-level config validation flag (avoid repeated warnings)
_config_validated = False


# ============================================================================
# Config Validation
# ============================================================================

def _validate_config():
    """Validate environment variable config legality (executed once)"""
    global _config_validated
    if _config_validated:
        return
    _config_validated = True
    
    use_cublaslt = os.environ.get("USE_CUBLASLT", "0") == "1"
    use_cusparselt = os.environ.get("USE_CUSPARSELT", "0") == "1"
    
    if use_cublaslt and use_cusparselt:
        raise ValueError(
            "USE_CUBLASLT=1 and USE_CUSPARSELT=1 are mutually exclusive. "
            "Please set only one of them."
        )


# ============================================================================
# Main Switch: SlideSparse Enable/Disable
# ============================================================================

def is_slidesparse_enabled() -> bool:
    """Check if SlideSparse is enabled
    
    SlideSparse is enabled by default, set DISABLE_SLIDESPARSE=1 to disable.
    When disabled, vLLM native path will be used.
    """
    return os.environ.get("DISABLE_SLIDESPARSE", "0") != "1"


# ============================================================================
# Kernel Backend Selection
# ============================================================================

def is_cublaslt_enabled() -> bool:
    """Check if cuBLASLt kernel is enabled
    
    Set USE_CUBLASLT=1 to enable.
    Default is 0, using external CUTLASS as fallback.
    Only meaningful when SlideSparse is enabled.
    
    Note: Mutually exclusive with USE_CUSPARSELT
    """
    _validate_config()
    return os.environ.get("USE_CUBLASLT", "0") == "1"


def is_cusparselt_enabled() -> bool:
    """Check if cuSPARSELt kernel is enabled
    
    Set USE_CUSPARSELT=1 to enable.
    Mutually exclusive with USE_CUBLASLT.
    """
    _validate_config()
    return os.environ.get("USE_CUSPARSELT", "0") == "1"


# ============================================================================
# GEMM Output Precision Control
# ============================================================================

def is_inner_dtype_32() -> bool:
    """Check if GEMM uses high-precision accumulation
    
    Set INNER_DTYPE_32=1 to enable.
    FP8 input uses FP32, INT8 input uses INT32.
    Only effective when USE_CUBLASLT=1 or USE_CUSPARSELT=1.
    Note: cuBLASLt + INT8 always uses INT32 output (BF16 not supported).
    """
    return os.environ.get("INNER_DTYPE_32", "0") == "1"


# ============================================================================
# Sparsity Format Config (delegated to top-level utils)
# ============================================================================

def get_sparsity_config() -> tuple:
    """
    Get sparsity format configuration
    
    Delegates to slidesparse.utils.get_sparsity_config_cached()
    
    Returns:
        (Z, L, expand_ratio) tuple
    """
    return get_sparsity_config_cached()


# ============================================================================
# Status Information
# ============================================================================

def get_slidesparse_status() -> str:
    """Get SlideSparse overall status information"""
    if not is_slidesparse_enabled():
        return "SlideSparse DISABLED (set DISABLE_SLIDESPARSE=0 to enable)"
    
    # SlideSparse enabled, check specific kernel backend
    if is_cublaslt_enabled():
        inner = "FP32/INT32" if is_inner_dtype_32() else "BF16"
        return f"SlideSparse ENABLED, cuBLASLt kernel (inner_dtype={inner})"
    elif is_cusparselt_enabled():
        inner = "FP32/INT32" if is_inner_dtype_32() else "BF16"
        Z, L, expand_ratio = get_sparsity_config()
        return (f"SlideSparse ENABLED, cuSPARSELt kernel "
                f"(inner_dtype={inner}, sparsity={Z}:{L}, expand_ratio={expand_ratio:.3f})")
    else:
        return "SlideSparse ENABLED, CUTLASS kernel (fallback)"
