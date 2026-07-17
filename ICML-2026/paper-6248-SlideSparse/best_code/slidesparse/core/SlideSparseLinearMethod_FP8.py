# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse FP8 Linear Method

This module is the core of SlideSparse FP8, replacing vLLM's FP8 Linear compute path via plugin.

Architecture
============
SlideSparse wraps vLLM's original CompressedTensorsW8A8Fp8 scheme:
- create_weights: delegates to original scheme
- process_weights_after_loading: delegates to original scheme + cuSPARSELt online compression
- apply_weights: replaces with SlideSparse kernel path

Three Kernel Paths (selected via env vars)
==========================================
1. CUTLASS (default fallback)
   - Directly calls vLLM's cutlass_scaled_mm, fused GEMM + dequant + bias
   - Weight shape: [K, N] (transposed by vLLM)
   
2. cuBLASLt (USE_CUBLASLT=1)
   - GEMM: cuBLASLt FP8 matmul (no scale/bias fusion)
   - Dequant+Bias: plugin Triton kernel
   - Weight shape: [N, K] (skip vLLM transpose, keep original row-major)
   
3. cuSPARSELt (USE_CUSPARSELT=1)
   - GEMM: cuSPARSELt 2:4 sparse FP8 matmul (no scale/bias fusion)
   - Dequant+Bias: plugin Triton kernel
   - Weight shape: weight_compressed [compressed_size] uint8 1D (after online compression)
   - Requires SPARSITY env var (default 2_8)

Dimension Naming Convention
===========================
GEMM: output[M, N] = input[M, K] @ weight[K, N]

cuBLASLt path:
- M, K, N: algorithm dimensions (GEMM semantic dimensions)
- M_pad: M aligned to 16
- K_pad: K aligned to 32

cuSPARSELt path:
- M, K, N: original algorithm dimensions
- K_slide: K dimension after slide expansion (K_slide = K * expand_ratio)
- M_pad: M aligned to 16
- K_slide_pad: K_slide aligned to 32
- weight_compressed: cuSPARSELt compressed 1D uint8 tensor

Padding Strategy:
- M_pad: Quant kernel handles 16-alignment internally, outputs [M_pad, K_pad] or [M_pad, K_slide_pad]
- K_pad/K_slide_pad: Quant kernel handles 32-alignment internally
- GEMM computes on padded dimensions, truncates back to original M before Dequant

Environment Variables
=====================
- DISABLE_SLIDESPARSE=1   : Completely disable SlideSparse, use vLLM native path
- USE_CUBLASLT=1          : Use cuBLASLt kernel
- USE_CUSPARSELT=1        : Use cuSPARSELt kernel (mutually exclusive with USE_CUBLASLT)
- INNER_DTYPE_32=1        : GEMM uses high-precision accumulation (FP8->FP32)
- SPARSITY=2_L            : Sparsity format (only for cuSPARSELt, L=4,6,8,10,... default 2_8)
- SLIDESPARSE_PROFILE=1   : Enable SlideSparse profiling
"""

from typing import Optional

import torch
from torch.nn import Module

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform

from slidesparse.utils import SlideSparseConfig, compute_output_k

from .config import (
    is_slidesparse_enabled,
    is_cublaslt_enabled,
    is_cusparselt_enabled,
    is_inner_dtype_32,
    get_sparsity_config,
)
from .profiler import ProfileTimer, profile_step
from .gemm_wrapper import (
    cublaslt_fp8_mm_op,
    cusparselt_fp8_mm_op,
    _get_gemm_extension,
    cuBLASLtGemmWrapper,
    cuSPARSELtGemmWrapper,
    get_algo_config_manager,
)
from .kernels import (
    dequant_bias_kernel,
    _load_dequant_bias_kernel,
    quant_only_fp8_kernel,
    _load_quant_only_fp8_kernel,
    quant_slide_fp8_kernel,
    _load_quant_slide_fp8_kernel,
)


logger = init_logger(__name__)


# ============================================================================
# Helper Function: Get Current Model Name
# ============================================================================

def _get_current_model_name() -> str:
    """
    Get current base model name from AlgorithmConfigManager (without -SlideSparse- suffix)
    
    Returned name directly maps to:
    - Triton kernel filename suffix
    - model_name in GEMM config JSON
    
    If not set, raises clear error
    """
    manager = get_algo_config_manager()
    model_name = manager.get_model_name()
    if model_name is None:
        raise ValueError(
            "Model name not set. Call slidesparse.init_slidesparse(model_name) first.\n"
            "Example: from slidesparse import init_slidesparse; init_slidesparse('Llama3.2-1B-FP8')"
        )
    return model_name


# ============================================================================
# FP8 Linear Functions (Three Kernel Paths)
# ============================================================================
#
# Three functions complete quant + GEMM + dequant internally:
#   - cuBLASLt_FP8_linear:   quant_only + cuBLASLt dense GEMM + Triton dequant
#   - cuSPARSELt_FP8_linear: quant_slide + cuSPARSELt 2:4 sparse GEMM + Triton dequant
#   - cutlass_FP8_linear:    vLLM QuantFP8 + cutlass_scaled_mm (fused dequant)
#
# cuBLASLt and cuSPARSELt compute flow:
#   1. Quant:   qinput[M,K], scale_a = quant_only/quant_slide(input)
#   2. GEMM:    inner[M,N] = weight @ qinput
#   3. Dequant: out[M,N] = inner * scale_a * scale_b + bias
#
#   4. Padding handling:
#   - Quant kernel outputs padded dimensions
#   - GEMM computes on padded dimensions
#   - Truncate back to original M before Dequant (slice has no data copy)
# ============================================================================

def cuBLASLt_FP8_linear(
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    output_shape: list,
    inner_dtype_str: str,
    model_name: str,
    input_scale: Optional[torch.Tensor] = None,
    input_scale_ub: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    cuBLASLt FP8 GEMM + Triton Dequant
    
    Data flow:
        input[M, K] BF16
            | quant_only_fp8_kernel
        qinput[M_pad, K_pad] FP8, scale_a[M_pad]
            | cublaslt_fp8_mm
        gemm_out[M_pad, N] BF16/FP32
            | truncate [:M, :]
        gemm_out[M, N], scale_a[M]
            | dequant_bias_kernel
        output[M, N] out_dtype
    """
    M = input.shape[0]
    
    # Quant: [M, K] -> [M_pad, K_pad]
    # cuBLASLt path always uses Triton quant kernel (requires padding)
    if input.dtype != current_platform.fp8_dtype():
        with ProfileTimer("cuBLASLt.quant"):
            qinput, scale_a_pad = quant_only_fp8_kernel(input, model_name)
    else:
        # Static quantization: input is already FP8, but no padding
        # cuBLASLt GEMM wrapper expects padded dimensions, so static quant not supported
        raise NotImplementedError(
            "cuBLASLt with static quantization is not supported. "
            "Use CUTLASS path or dynamic quantization."
        )
    
    # GEMM: [M_pad, K_pad] @ [N, K_pad].T -> [M_pad, N]
    with ProfileTimer("cuBLASLt.gemm"):
        gemm_out_pad = cublaslt_fp8_mm_op(weight, qinput, inner_dtype_str)
    
    # Truncate padding
    gemm_out = gemm_out_pad[:M, :]
    scale_a = scale_a_pad[:M]
    
    # Dequant + Bias
    with ProfileTimer("cuBLASLt.dequant"):
        output = dequant_bias_kernel(gemm_out, scale_a, weight_scale, bias, out_dtype, model_name)
    
    with ProfileTimer("cuBLASLt.view"):
        result = output.view(*output_shape)
    
    profile_step()
    return result


def cuSPARSELt_FP8_linear(
    *,
    input: torch.Tensor,
    slide_weight_compressed: torch.Tensor,
    out_dtype: torch.dtype,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    output_shape: list,
    inner_dtype_str: str,
    model_name: str,
    input_scale: Optional[torch.Tensor] = None,
    input_scale_ub: Optional[torch.Tensor] = None,
    slide_weight_N: Optional[int] = None,
    slide_weight_K: Optional[int] = None,
    L: int = 8,
) -> torch.Tensor:
    """
    cuSPARSELt 2:4 Sparse FP8 GEMM + Triton Dequant
    
    Data flow:
        input[M, K] BF16
            | quant_slide_fp8_kernel
        qinput[M_pad, K_slide_pad] FP8, scale_a[M_pad]
            | cusparselt_fp8_mm
        gemm_out[M_pad, N] BF16/FP32
            | truncate [:M, :]
        gemm_out[M, N], scale_a[M]
            | dequant_bias_kernel
        output[M, N] out_dtype
    
    Args:
        slide_weight_compressed: [compressed_size] uint8 1D
        slide_weight_N: weight N dimension
        slide_weight_K: weight K_slide dimension (after slide expansion, 32-aligned)
        L: sparsity group size (default 8)
    """
    
    if slide_weight_N is None or slide_weight_K is None:
        raise ValueError(
            "cuSPARSELt requires slide_weight_N and slide_weight_K."
        )
    
    M = input.shape[0]
    
    # Quant + Slide: [M, K] -> [M_pad, K_slide_pad]
    if input.dtype != current_platform.fp8_dtype():
        with ProfileTimer("cuSPARSELt.quant_slide"):
            qinput, scale_a_pad = quant_slide_fp8_kernel(input, model_name, L)
    else:
        raise NotImplementedError(
            "cuSPARSELt with static quantization is not supported yet."
        )
    
    # Verify dimension consistency: qinput K dim should match weight K dim
    K_slide_pad = qinput.shape[1]
    if K_slide_pad != slide_weight_K:
        raise ValueError(
            f"K dimension mismatch: qinput.shape[1]={K_slide_pad}, "
            f"slide_weight_K={slide_weight_K}. "
            "This may indicate L parameter mismatch between weight and activation."
        )
    
    # GEMM: [M_pad, K_slide_pad] @ compressed_weight -> [M_pad, N]
    with ProfileTimer("cuSPARSELt.gemm"):
        gemm_out_pad = cusparselt_fp8_mm_op(
            slide_weight_compressed,
            qinput,
            slide_weight_N,
            K_slide_pad,
            inner_dtype_str
        )
    
    # Truncate padding
    gemm_out = gemm_out_pad[:M, :]
    scale_a = scale_a_pad[:M]
    
    # Dequant + Bias
    with ProfileTimer("cuSPARSELt.dequant"):
        output = dequant_bias_kernel(gemm_out, scale_a, weight_scale, bias, out_dtype, model_name)
    
    with ProfileTimer("cuSPARSELt.view"):
        result = output.view(*output_shape)
    
    profile_step()
    return result


def cutlass_FP8_linear(
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    output_shape: list,
    quant_fn: QuantFP8,
    input_scale: Optional[torch.Tensor] = None,
    input_scale_ub: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    vLLM CUTLASS (fused GEMM + Dequant + Bias)
    
    Data flow:
        input[M, K] BF16
            | QuantFP8
        qinput[M, K] FP8, scale_a
            | cutlass_scaled_mm (fused dequant + bias)
        output[M, N] out_dtype
    """
    # Quant (using vLLM native QuantFP8)
    if input.dtype != current_platform.fp8_dtype():
        with ProfileTimer("CUTLASS.quant"):
            qinput, scale_a = quant_fn(input, input_scale, input_scale_ub)
    else:
        qinput, scale_a = input, input_scale
    
    # CUTLASS 融合 GEMM + Dequant + Bias
    with ProfileTimer("CUTLASS.scaled_mm"):
        output = ops.cutlass_scaled_mm(
            qinput, weight, out_dtype=out_dtype,
            scale_a=scale_a, scale_b=weight_scale, bias=bias
        )
    
    with ProfileTimer("CUTLASS.view"):
        result = output.view(*output_shape)
    
    profile_step()
    return result


# ============================================================================
# SlideSparse FP8 Linear Op
# ============================================================================

class SlideSparseFp8LinearOp:
    """
    SlideSparse FP8 Linear Operation
    
    Selects kernel path based on env vars:
    - USE_CUBLASLT=1: cuBLASLt_FP8_linear
    - USE_CUSPARSELT=1: cuSPARSELt_FP8_linear
    - Default: cutlass_FP8_linear
    
    Note: Triton kernels use lazy loading, loaded on first apply call
    because kernels are model-specific, need current model name.
    """
    
    def __init__(
        self,
        act_quant_static: bool = False,
        act_quant_group_shape: GroupShape = GroupShape.PER_TOKEN,
    ):
        self.act_quant_static = act_quant_static
        self.act_quant_group_shape = act_quant_group_shape
        
        # Create QuantFP8 instance (used by CUTLASS path)
        self.quant_fp8 = QuantFP8(
            static=act_quant_static,
            group_shape=act_quant_group_shape,
            num_token_padding=None,
        )
        
        # Determine kernel path (cache env var check result)
        self._use_cublaslt = is_cublaslt_enabled()
        self._use_cusparselt = is_cusparselt_enabled()
        
        # Cache inner_dtype (env vars don't change within process lifetime)
        self._inner_dtype_str = "fp32" if is_inner_dtype_32() else "bf16"
        
        # Only preload GEMM extension (model-agnostic)
        # Triton kernels lazy-loaded (need model_name)
        if self._use_cublaslt:
            self._kernel_name = "cuBLASLt"
            self._linear_fn = cuBLASLt_FP8_linear
            _get_gemm_extension("cublaslt")
        elif self._use_cusparselt:
            self._kernel_name = "cuSPARSELt"
            self._linear_fn = cuSPARSELt_FP8_linear
            _get_gemm_extension("cusparselt")
        else:
            self._kernel_name = "CUTLASS"
            self._linear_fn = cutlass_FP8_linear
        
        logger.info_once(
            f"SlideSparseFp8LinearOp initialized (kernel={self._kernel_name})"
        )
    
    def apply(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype | None = None,
        input_scale: torch.Tensor | None = None,
        input_scale_ub: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        slide_weight_N: int | None = None,
        slide_weight_K: int | None = None,
        L: int = 8,
    ) -> torch.Tensor:
        """
        Execute FP8 Linear operation
        
        Args:
            input: [..., K] BF16
            weight: weight (shape depends on kernel path)
            weight_scale: [N] FP32
            out_dtype: output type
            input_scale: static quantization scale
            input_scale_ub: static quantization scale upper bound
            bias: [N]
            slide_weight_N: cuSPARSELt specific, N dimension
            slide_weight_K: cuSPARSELt specific, K_slide dimension
            L: cuSPARSELt specific, sparsity group size
        """
        # Get input shape info
        input_shape = input.shape
        input_ndim = input.dim()
        
        # Flatten to 2D (use directly if already 2D, avoid unnecessary view call)
        if input_ndim == 2:
            input_2d = input
            M = input_shape[0]
        else:
            input_2d = input.view(-1, input_shape[-1])
            M = input_2d.shape[0]
        
        # Infer output N dimension (based on cached kernel path, avoid weight.dim() call)
        # - cuSPARSELt: N provided by slide_weight_N param (weight is compressed 1D)
        # - cuBLASLt:   weight [N, K], N at dim=0
        # - CUTLASS:    weight [K, N], N at dim=1
        if self._use_cusparselt:
            output_N = slide_weight_N  # Caller guarantees slide_weight_N is valid
        elif self._use_cublaslt:
            output_N = weight.shape[0]
        else:
            output_N = weight.shape[1]
        
        # Build output_shape (optimized for common 2D input, avoid list unpacking)
        if input_ndim == 2:
            output_shape = [M, output_N]
        else:
            output_shape = [*input_shape[:-1], output_N]
        
        if out_dtype is None:
            out_dtype = input.dtype
        
        # Common args
        common_args = dict(
            input=input_2d,
            out_dtype=out_dtype,
            weight_scale=weight_scale,
            bias=bias,
            output_shape=output_shape,
            input_scale=input_scale,
            input_scale_ub=input_scale_ub,
        )
        
        # Call selected kernel path
        if self._use_cusparselt:
            # cuSPARSELt needs model_name to load Triton kernels
            model_name = _get_current_model_name()
            return self._linear_fn(
                **common_args,
                slide_weight_compressed=weight,
                inner_dtype_str=self._inner_dtype_str,
                model_name=model_name,
                slide_weight_N=slide_weight_N,
                slide_weight_K=slide_weight_K,
                L=L,
            )
        elif self._use_cublaslt:
            # cuBLASLt needs model_name to load Triton kernels
            model_name = _get_current_model_name()
            return self._linear_fn(
                **common_args,
                weight=weight,
                inner_dtype_str=self._inner_dtype_str,
                model_name=model_name,
            )
        else:
            # CUTLASS path doesn't need model_name (uses vLLM native kernel)
            return self._linear_fn(
                **common_args,
                weight=weight,
                quant_fn=self.quant_fp8,
            )


# ============================================================================
# SlideSparse FP8 Linear Method
# ============================================================================

class SlideSparseFp8LinearMethod:
    """
    SlideSparse FP8 Linear Method
    
    Wraps vLLM's original CompressedTensorsW8A8Fp8 scheme:
    - create_weights: delegates to original scheme
    - process_weights_after_loading: delegates + cuBLASLt/cuSPARSELt post-processing
    - apply_weights: uses SlideSparseFp8LinearOp
    
    Weight shape changes:
        Original checkpoint: [N, K] or [N, K_slide] (slidesparse checkpoint)
        After vLLM load: [N, K] or [N, K_slide]
        CUTLASS path: weight.t() -> [K, N]
        cuBLASLt path: keeps [N, K]
        cuSPARSELt path: [N, K_slide] -> compress -> [compressed_size] uint8 1D
    """
    
    def __init__(self, original_scheme):
        self.original_scheme = original_scheme
        self.out_dtype = original_scheme.out_dtype
        self.is_static_input_scheme = original_scheme.is_static_input_scheme
        self.act_q_group_shape = original_scheme.act_q_group_shape
        self.strategy = original_scheme.strategy
        
        self._use_cublaslt = is_cublaslt_enabled()
        self._use_cusparselt = is_cusparselt_enabled()
        
        # Create SlideSparse Op
        self.slidesparse_fp8_linear = SlideSparseFp8LinearOp(
            act_quant_static=self.is_static_input_scheme,
            act_quant_group_shape=self.act_q_group_shape,
        )
        
        # cuSPARSELt sparsity config
        if self._use_cusparselt:
            Z, L, self._expand_ratio = get_sparsity_config()
            self._sparsity_config = SlideSparseConfig(Z=Z, L=L)
            logger.info_once(
                f"SlideSparseFp8LinearMethod: cuSPARSELt "
                f"sparsity={Z}:{L}, expand_ratio={self._expand_ratio:.3f}"
            )
        
        # Preload Triton kernels (torch.compile compatible)
        import os
        model_name = os.environ.get("SLIDESPARSE_MODEL_NAME")
        if model_name and (self._use_cublaslt or self._use_cusparselt):
            # dequant_bias kernel is shared by cuBLASLt and cuSPARSELt
            _load_dequant_bias_kernel(model_name)
            
            if self._use_cublaslt:
                _load_quant_only_fp8_kernel(model_name)
            elif self._use_cusparselt:
                _load_quant_slide_fp8_kernel(model_name)
            
            logger.info_once(
                f"Preloaded FP8 Triton kernels for model: {model_name}"
            )
        
        logger.info_once(
            f"SlideSparseFp8LinearMethod initialized, "
            f"kernel={self.slidesparse_fp8_linear._kernel_name}"
        )
    
    def create_weights(
        self,
        layer: Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        weight_loader,
        **kwargs,
    ):
        """
        Create weight parameters
        
        cuSPARSELt path: expand input_size to match slide checkpoint weight
        Other paths: directly delegate to original scheme
        """
        if self._use_cusparselt:
            # cuSPARSELt: expand input_size to match K dimension after slide
            # Checkpoint weight is already [N, K_slide], need to match
            _, input_size_per_partition_slide = compute_output_k(
                input_size_per_partition, self._sparsity_config
            )
            _, input_size_slide = compute_output_k(
                input_size, self._sparsity_config
            )
            
            return self.original_scheme.create_weights(
                layer=layer,
                input_size_per_partition=input_size_per_partition_slide,
                output_partition_sizes=output_partition_sizes,
                input_size=input_size_slide,
                output_size=output_size,
                params_dtype=params_dtype,
                weight_loader=weight_loader,
                **kwargs,
            )
        else:
            # CUTLASS / cuBLASLt: directly delegate
            return self.original_scheme.create_weights(
                layer=layer,
                input_size_per_partition=input_size_per_partition,
                output_partition_sizes=output_partition_sizes,
                input_size=input_size,
                output_size=output_size,
                params_dtype=params_dtype,
                weight_loader=weight_loader,
                **kwargs,
            )
    
    def process_weights_after_loading(self, layer: Module) -> None:
        """
        Post-load weight processing
        
        Processing logic:
        - CUTLASS: delegates to original scheme (executes weight.t())
        - cuBLASLt: original scheme + transpose back to [N, K]
        - cuSPARSELt: original scheme + transpose back to [N, K_slide] + online compression
        """
        # All paths first call original scheme
        self.original_scheme.process_weights_after_loading(layer)
        
        if not self._use_cublaslt and not self._use_cusparselt:
            # CUTLASS path: return directly
            return
        
        # cuBLASLt / cuSPARSELt: transpose back to [N, K] or [N, K_slide]
        from torch.nn import Parameter
        weight_transposed = layer.weight.data.t()
        layer.weight = Parameter(weight_transposed, requires_grad=False)
        
        if self._use_cusparselt:
            self._compress_weight_online(layer)
    
    def _compress_weight_online(self, layer: Module) -> None:
        """
        cuSPARSELt online compression
        
        Input: layer.weight [N, K_slide] FP8
        Output: layer.weight [compressed_size] uint8 1D
              layer.slide_weight_N: N
              layer.slide_weight_K: K_slide
        """
        from torch.nn import Parameter
        
        try:
            from slidesparse.weight_convert.compress import compress_tensor_online
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import compress_tensor_online: {e}"
            ) from e
        
        slide_weight = layer.weight.data
        N, K_slide = slide_weight.shape
        
        logger.info_once(
            f"cuSPARSELt compression: [{N}, {K_slide}] -> 1D uint8"
        )
        
        weight_compressed = compress_tensor_online(slide_weight, verbose=False)
        
        layer.weight = Parameter(weight_compressed, requires_grad=False)
        layer.slide_weight_N = N
        layer.slide_weight_K = K_slide
        
        logger.info_once(
            f"cuSPARSELt compression done: {weight_compressed.numel()} bytes"
        )
    
    def apply_weights(
        self,
        layer: Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply weights (linear transformation)"""
        input_scale = getattr(layer, "input_scale", None)
        input_scale_ub = getattr(layer, "input_scale_ub", None)
        
        if self._use_cusparselt:
            return self.slidesparse_fp8_linear.apply(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                out_dtype=self.out_dtype,
                input_scale=input_scale,
                input_scale_ub=input_scale_ub,
                bias=bias,
                slide_weight_N=layer.slide_weight_N,
                slide_weight_K=layer.slide_weight_K,
                L=self._sparsity_config.L,
            )
        else:
            return self.slidesparse_fp8_linear.apply(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                out_dtype=self.out_dtype,
                input_scale=input_scale,
                input_scale_ub=input_scale_ub,
                bias=bias,
            )


# ============================================================================
# Factory Function
# ============================================================================

def wrap_scheme_fp8(original_scheme):
    """
    FP8 scheme wrapper entry point
    
    Only wraps W8A8Fp8 scheme, returns others as-is.
    SlideSparseFp8LinearOp internally selects kernel path based on env vars.
    """
    scheme_name = type(original_scheme).__name__
    if "W8A8Fp8" not in scheme_name:
        logger.warning_once(
            f"SlideSparse not supported for {scheme_name}, using original"
        )
        return original_scheme
    
    if is_cublaslt_enabled():
        backend = "cuBLASLt"
    elif is_cusparselt_enabled():
        backend = "cuSPARSELt"
    else:
        backend = "CUTLASS"
    
    logger.info_once(
        f"Wrapping {scheme_name} with SlideSparse ({backend})"
    )
    return SlideSparseFp8LinearMethod(original_scheme)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Linear functions
    "cuBLASLt_FP8_linear",
    "cuSPARSELt_FP8_linear",
    "cutlass_FP8_linear",
    
    # Op and Method classes
    "SlideSparseFp8LinearOp",
    "SlideSparseFp8LinearMethod",
    
    # Factory function
    "wrap_scheme_fp8",
]
