# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse CSRC Package

Provides CUDA/Triton extension compilation tools and kernel modules.

Submodules
==========
- utils: Build tools (NVCC flags, build_cuda_extension, etc.)
- cublaslt_gemm: cuBLASLt FP8 GEMM extension
- fused_dequant_bias_triton: Triton fused dequant+bias kernel
- fused_quant_slide_triton: Triton fused quant+slide kernel
- quant_only_triton: Triton quant-only kernel (no slide)
"""

from .utils import (
    get_nvcc_arch_flags,
    build_cuda_extension,
    clean_build_artifacts,
    get_gemm_ldflags,
    get_dequant_autotune_configs,
    CUBLASLT_LDFLAGS,
    CUSPARSELT_LDFLAGS,
)

__all__ = [
    "get_nvcc_arch_flags",
    "build_cuda_extension",
    "clean_build_artifacts",
    "get_gemm_ldflags",
    "get_dequant_autotune_configs",
    "CUBLASLT_LDFLAGS",
    "CUSPARSELT_LDFLAGS",
]
