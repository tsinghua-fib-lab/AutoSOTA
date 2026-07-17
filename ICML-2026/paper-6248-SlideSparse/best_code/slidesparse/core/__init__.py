# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse Core Logic Module

Contents:
- config: SlideSparse configuration functions
- profiler: Timing diagnostics module
- kernels: Triton kernel loading
- gemm_wrapper: GEMM wrapper and Custom Op
- SlideSparseLinearMethod_FP8: FP8 linear layer method
- SlideSparseLinearMethod_INT8: INT8 linear layer method

Environment Variables:
======================
1. DISABLE_SLIDESPARSE=1  -> Fully disable SlideSparse, use vLLM native path
2. USE_CUBLASLT=1         -> Use cuBLASLt kernel
3. USE_CUSPARSELT=1       -> Use cuSPARSELt kernel
4. INNER_DTYPE_32=1       -> GEMM uses high-precision accumulation (FP8->FP32, INT8->INT32)
5. SPARSITY=2_8           -> Sparsity format (only effective with cuSPARSELt, default 2_8)
6. SLIDESPARSE_PROFILE=1  -> Enable timing diagnostics

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
from slidesparse.core.gemm_wrapper import (
    _get_gemm_extension,
    get_algo_config_manager,
    AlgorithmConfigManager,
)
from slidesparse.core.profiler import print_profile_stats, reset_profile_stats


# ============================================================================
# Initialization Functions
# ============================================================================

def init_slidesparse(model_name_with_slide: str) -> None:
    """
    Initialize SlideSparse system
    
    Must be called before using any SlideSparse GEMM kernel (except CUTLASS fallback).
    Typically called during model loading or before test starts.
    
    This function will:
    1. Set current model name (for loading model-specific tuned kernels)
    2. Preload GEMM algorithm config (if offline search results exist)
    3. Preload Triton kernels (to avoid loading during torch.compile tracing)
    
    torch.compile compatibility:
    - All kernels are preloaded to cache in this function
    - Forward path only reads from cache, no filesystem operations
    
    Naming convention:
    - model_name: Base model name (e.g., "Qwen2.5-0.5B-FP8"), for kernel/config lookup
    - model_name_with_slide: Full checkpoint name (may contain -SlideSparse-2_L suffix)
    
    Args:
        model_name_with_slide: Model name, e.g., "Qwen2.5-0.5B-FP8" or 
                               "Qwen2.5-0.5B-FP8-SlideSparse-2_8"
                               Should match checkpoints/checkpoints_slidesparse directory name
    """
    from slidesparse.core.kernels import (
        _load_dequant_bias_kernel,
        _load_quant_only_fp8_kernel,
        _load_quant_slide_fp8_kernel,
        _load_quant_only_int8_kernel,
        _load_quant_slide_int8_kernel,
    )
    from vllm.logger import init_logger
    logger = init_logger(__name__)
    
    # 1. Set model name and load GEMM algorithm config
    # set_model auto-extracts base model name and sets env vars
    manager = get_algo_config_manager()
    manager.set_model(model_name_with_slide)
    # load_all_configs already called in AlgorithmConfigManager.__init__
    
    # Get base model name for kernel loading
    model_name = manager.get_model_name()
    
    # 2. Preload Triton kernels (based on enabled backend)
    # This ensures no kernel loading (filesystem operations) on forward path
    use_cublaslt = is_cublaslt_enabled()
    use_cusparselt = is_cusparselt_enabled()
    
    if use_cublaslt or use_cusparselt:
        # dequant_bias kernel is shared by cuBLASLt and cuSPARSELt
        _load_dequant_bias_kernel(model_name)
        
    if use_cublaslt:
        # cuBLASLt FP8/INT8 uses quant_only kernel
        _load_quant_only_fp8_kernel(model_name)
        _load_quant_only_int8_kernel(model_name)
        logger.info(f"Preloaded Triton kernels for cuBLASLt: {model_name}")
        
    if use_cusparselt:
        # cuSPARSELt FP8/INT8 uses quant_slide kernel
        _load_quant_slide_fp8_kernel(model_name)
        _load_quant_slide_int8_kernel(model_name)
        logger.info(f"Preloaded Triton kernels for cuSPARSELt: {model_name}")


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
    
    # Initialization
    "init_slidesparse",
    "get_algo_config_manager",
    "AlgorithmConfigManager",
]