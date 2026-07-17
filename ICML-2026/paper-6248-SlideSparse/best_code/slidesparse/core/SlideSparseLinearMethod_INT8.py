# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse INT8 Linear Method

This module is the core of SlideSparse INT8, replacing vLLM's INT8 Linear compute path via plugin.

Architecture
============
SlideSparse wraps vLLM's original CompressedTensorsW8A8Int8 scheme:
- create_weights: delegates to original scheme
- process_weights_after_loading: delegates to original scheme + cuSPARSELt online compression
- apply_weights: replaces with SlideSparse kernel path

Three Kernel Paths (selected via env vars)
==========================================
1. CUTLASS (default fallback)
   - Directly calls vLLM's cutlass_scaled_mm / cutlass_scaled_mm_azp
   - Fused GEMM + dequant + bias
   - Supports symmetric and asymmetric quantization
   - Weight shape: [K, N] (transposed by vLLM)
   
2. cuBLASLt (USE_CUBLASLT=1)
   - GEMM: cuBLASLt INT8 matmul (output fixed INT32)
   - Dequant+Bias: plugin Triton kernel
   - Weight shape: [N, K] (skip vLLM transpose, keep original row-major)
   - Note: cuBLASLt INT8 only supports symmetric quantization
   
3. cuSPARSELt (USE_CUSPARSELT=1)
   - GEMM: cuSPARSELt 2:4 sparse INT8 matmul
   - Dequant+Bias: plugin Triton kernel
   - Weight shape: weight_compressed [compressed_size] uint8 1D (after online compression)
   - Note: cuSPARSELt INT8 only supports symmetric quantization, output can be BF16 or INT32 (not FP32)

Dimension Naming Convention
===========================
GEMM: output[M, N] = input[M, K] @ weight[K, N]

cuBLASLt path:
- M, K, N: algorithm dimensions (GEMM semantic dimensions)
- M_pad: M aligned to 16
- K_pad: K aligned to 32
- Output fixed INT32

cuSPARSELt path:
- M, K, N: original algorithm dimensions
- K_slide: K dimension after slide expansion
- M_pad: M aligned to 16
- K_slide_pad: K_slide aligned to 32
- Output can be BF16 or INT32

INT8 vs FP8 Key Differences
===========================
1. cuBLASLt INT8 output fixed INT32 (FP8 can choose BF16/FP32)
2. cuSPARSELt INT8 does not support FP32 output (FP8 does)
3. INT8 supports asymmetric quantization (needs azp), but cuBLASLt/cuSPARSELt paths don't
4. vLLM uses ops.scaled_int8_quant instead of QuantFP8

Environment Variables
=====================
- DISABLE_SLIDESPARSE=1   : Completely disable SlideSparse, use vLLM native path
- USE_CUBLASLT=1          : Use cuBLASLt kernel
- USE_CUSPARSELT=1        : Use cuSPARSELt kernel (mutually exclusive with USE_CUBLASLT)
- INNER_DTYPE_32=1        : GEMM uses high-precision accumulation (INT8->INT32)
- SPARSITY=2_L            : Sparsity format (only for cuSPARSELt)
- SLIDESPARSE_PROFILE=1   : Enable SlideSparse profiling
"""

from typing import Optional

import torch
from torch.nn import Module

from vllm import _custom_ops as ops
from vllm.logger import init_logger

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
    cublaslt_int8_mm_op,
    cusparselt_int8_mm_op,
    _get_gemm_extension,
    get_algo_config_manager,
)
from .kernels import (
    dequant_bias_kernel,
    _load_dequant_bias_kernel,
    quant_only_int8_kernel,
    _load_quant_only_int8_kernel,
    quant_slide_int8_kernel,
    _load_quant_slide_int8_kernel,
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
# INT8 Linear Functions (Three Kernel Paths)
# ============================================================================
#
# Three functions complete quant + GEMM + dequant internally:
#   - cuBLASLt_INT8_linear:   quant_only + cuBLASLt dense GEMM + Triton dequant
#   - cuSPARSELt_INT8_linear: quant_slide + cuSPARSELt 2:4 sparse GEMM + Triton dequant
#   - cutlass_INT8_linear:    vLLM ops.scaled_int8_quant + cutlass_scaled_mm
#
# cuBLASLt and cuSPARSELt compute flow:
#   1. Quant:   qinput[M,K], scale_a = quant_only/quant_slide(input)
#   2. GEMM:    inner[M,N] = weight @ qinput
#   3. Dequant: out[M,N] = inner * scale_a * scale_b + bias
#
# CUTLASS compute flow:
#   1. Quant:   qinput, scale_a, azp = ops.scaled_int8_quant(input, ...)
#   2. GEMM:    output = cutlass_scaled_mm[_azp](qinput, weight, scale_a, scale_b, ...)
# ============================================================================

def cuBLASLt_INT8_linear(
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
    input_zero_point: Optional[torch.Tensor] = None,
    azp_adj: Optional[torch.Tensor] = None,
    input_symmetric: bool = True,
) -> torch.Tensor:
    """
    cuBLASLt INT8 GEMM + Triton Dequant
    
    Data flow:
        input[M, K] BF16
            | quant_only_int8_kernel (symmetric quantization)
        qinput[M_pad, K_pad] INT8, scale_a[M_pad]
            | cublaslt_int8_mm (output fixed INT32)
        gemm_out[M_pad, N] INT32
            | truncate [:M, :]
        gemm_out[M, N], scale_a[M]
            | dequant_bias_kernel
        output[M, N] out_dtype
    
    Note:
        cuBLASLt INT8 only supports symmetric quantization.
        Use CUTLASS path for asymmetric quantization.
    """
    # cuBLASLt only supports symmetric quantization
    if not input_symmetric:
        raise NotImplementedError(
            "cuBLASLt INT8 does not support asymmetric quantization. "
            "Use CUTLASS path for asymmetric quantization."
        )
    
    M = input.shape[0]
    
    # Quant: [M, K] -> [M_pad, K_pad]
    # cuBLASLt path uses Triton INT8 quant kernel (symmetric quantization)
    if input.dtype != torch.int8:
        with ProfileTimer("cuBLASLt.quant"):
            qinput, scale_a_pad = quant_only_int8_kernel(input, model_name)
    else:
        # Static quantization: input is already INT8, but no padding
        raise NotImplementedError(
            "cuBLASLt with static quantization is not supported. "
            "Use CUTLASS path or dynamic quantization."
        )
    
    # GEMM: [M_pad, K_pad] @ [N, K_pad].T -> [M_pad, N] INT32
    # Note: cuBLASLt INT8 output fixed INT32, inner_dtype_str is ignored
    with ProfileTimer("cuBLASLt.gemm"):
        gemm_out_pad = cublaslt_int8_mm_op(weight, qinput, "int32")
    
    # Truncate padding
    gemm_out = gemm_out_pad[:M, :]
    scale_a = scale_a_pad[:M]
    
    # Dequant + Bias: INT32 -> out_dtype
    with ProfileTimer("cuBLASLt.dequant"):
        output = dequant_bias_kernel(gemm_out, scale_a, weight_scale, bias, out_dtype, model_name)
    
    with ProfileTimer("cuBLASLt.view"):
        result = output.view(*output_shape)
    
    profile_step()
    return result


def cuSPARSELt_INT8_linear(
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
    input_zero_point: Optional[torch.Tensor] = None,
    azp_adj: Optional[torch.Tensor] = None,
    input_symmetric: bool = True,
    slide_weight_N: Optional[int] = None,
    slide_weight_K: Optional[int] = None,
    L: int = 8,
) -> torch.Tensor:
    """
    cuSPARSELt 2:4 Sparse INT8 GEMM + Triton Dequant
    
    数据流:
        input[M, K] BF16
            ↓ quant_slide_int8_kernel (对称量化)
        qinput[M_pad, K_slide_pad] INT8, scale_a[M_pad]
            ↓ cusparselt_int8_mm
        gemm_out[M_pad, N] BF16/INT32
            ↓ 截断 [:M, :]
        gemm_out[M, N], scale_a[M]
            | dequant_bias_kernel
        output[M, N] out_dtype
    
    Args:
        slide_weight_compressed: [compressed_size] uint8 1D
        slide_weight_N: weight N dimension
        slide_weight_K: weight K_slide dimension
        L: sparsity group size (default 8)
    
    Note:
        cuSPARSELt INT8 only supports symmetric quantization.
        cuSPARSELt INT8 does not support FP32 output, only BF16 or INT32.
    """
    # cuSPARSELt only supports symmetric quantization
    if not input_symmetric:
        raise NotImplementedError(
            "cuSPARSELt INT8 does not support asymmetric quantization. "
            "Use CUTLASS path for asymmetric quantization."
        )
    
    if slide_weight_N is None or slide_weight_K is None:
        raise ValueError(
            "cuSPARSELt requires slide_weight_N and slide_weight_K."
        )
    
    # cuSPARSELt INT8 does not support FP32 output
    if inner_dtype_str == "fp32":
        raise ValueError(
            "cuSPARSELt INT8 does not support FP32 output. "
            "Use 'bf16' (default) or 'int32'. "
            "Set INNER_DTYPE_32=1 to use INT32 output."
        )
    
    M = input.shape[0]
    
    # Quant + Slide: [M, K] -> [M_pad, K_slide_pad]
    if input.dtype != torch.int8:
        with ProfileTimer("cuSPARSELt.quant_slide"):
            qinput, scale_a_pad = quant_slide_int8_kernel(input, model_name, L)
    else:
        raise NotImplementedError(
            "cuSPARSELt with static quantization is not supported yet."
        )
    
    # Verify dimension consistency
    K_slide_pad = qinput.shape[1]
    if K_slide_pad != slide_weight_K:
        raise ValueError(
            f"K dimension mismatch: qinput.shape[1]={K_slide_pad}, "
            f"slide_weight_K={slide_weight_K}. "
            "This may indicate L parameter mismatch between weight and activation."
        )
    
    # GEMM: [M_pad, K_slide_pad] @ compressed_weight -> [M_pad, N]
    with ProfileTimer("cuSPARSELt.gemm"):
        gemm_out_pad = cusparselt_int8_mm_op(
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


def cutlass_INT8_linear(
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    output_shape: list,
    input_scale: Optional[torch.Tensor] = None,
    input_zero_point: Optional[torch.Tensor] = None,
    azp_adj: Optional[torch.Tensor] = None,
    input_symmetric: bool = True,
) -> torch.Tensor:
    """
    vLLM CUTLASS INT8 (fused GEMM + Dequant + Bias)
    
    Data flow:
        input[M, K] BF16
            | ops.scaled_int8_quant
        qinput[M, K] INT8, scale_a, x_zp
            | cutlass_scaled_mm / cutlass_scaled_mm_azp
        output[M, N] out_dtype
    
    Supports symmetric and asymmetric quantization.
    """
    # Quant (using vLLM native ops.scaled_int8_quant)
    if input.dtype != torch.int8:
        with ProfileTimer("CUTLASS.quant"):
            x_q, x_s, x_zp = ops.scaled_int8_quant(
                input.contiguous(),
                input_scale,
                input_zero_point,
                symmetric=input_symmetric
            )
    else:
        # Static quantization: input is already INT8
        x_q, x_s, x_zp = input, input_scale, input_zero_point
    
    # CUTLASS fused GEMM + Dequant + Bias
    with ProfileTimer("CUTLASS.scaled_mm"):
        if x_zp is not None:
            # Asymmetric quantization
            # Currently, static is always per-tensor and dynamic is per-token
            static = input_zero_point is not None
            azp = None if static else x_zp
            output = ops.cutlass_scaled_mm_azp(
                x_q,
                weight,
                scale_a=x_s,
                scale_b=weight_scale,
                out_dtype=out_dtype,
                azp_adj=azp_adj,
                azp=azp,
                bias=bias,
            )
        else:
            # Symmetric quantization
            output = ops.cutlass_scaled_mm(
                x_q, weight, out_dtype=out_dtype,
                scale_a=x_s, scale_b=weight_scale, bias=bias
            )
    
    with ProfileTimer("CUTLASS.view"):
        result = output.view(*output_shape)
    
    profile_step()
    return result


# ============================================================================
# SlideSparse INT8 Linear Op
# ============================================================================

class SlideSparseInt8LinearOp:
    """
    SlideSparse INT8 Linear Operation
    
    Selects kernel path based on env vars:
    - USE_CUBLASLT=1: cuBLASLt_INT8_linear
    - USE_CUSPARSELT=1: cuSPARSELt_INT8_linear
    - Default: cutlass_INT8_linear
    
    Note: cuBLASLt and cuSPARSELt paths only support symmetric quantization.
    """
    
    def __init__(
        self,
        act_quant_static: bool = False,
        input_symmetric: bool = True,
    ):
        self.act_quant_static = act_quant_static
        self.input_symmetric = input_symmetric
        
        # Determine kernel path (cache env var check result)
        self._use_cublaslt = is_cublaslt_enabled()
        self._use_cusparselt = is_cusparselt_enabled()
        
        # Cache inner_dtype
        # Note: cuBLASLt INT8 output fixed INT32, this setting only for cuSPARSELt
        self._inner_dtype_str = "int32" if is_inner_dtype_32() else "bf16"
        
        # Force CUTLASS for asymmetric quantization
        if not input_symmetric and (self._use_cublaslt or self._use_cusparselt):
            logger.warning_once(
                "Asymmetric INT8 quantization detected. "
                "cuBLASLt/cuSPARSELt do not support asymmetric quantization. "
                "Falling back to CUTLASS path."
            )
            self._use_cublaslt = False
            self._use_cusparselt = False
        
        if self._use_cublaslt:
            self._kernel_name = "cuBLASLt"
            self._linear_fn = cuBLASLt_INT8_linear
            _get_gemm_extension("cublaslt")
        elif self._use_cusparselt:
            self._kernel_name = "cuSPARSELt"
            self._linear_fn = cuSPARSELt_INT8_linear
            _get_gemm_extension("cusparselt")
        else:
            self._kernel_name = "CUTLASS"
            self._linear_fn = cutlass_INT8_linear
        
        logger.info_once(
            f"SlideSparseInt8LinearOp initialized (kernel={self._kernel_name})"
        )
    
    def apply(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype | None = None,
        input_scale: torch.Tensor | None = None,
        input_zero_point: torch.Tensor | None = None,
        azp_adj: torch.Tensor | None = None,
        input_symmetric: bool = True,
        bias: torch.Tensor | None = None,
        slide_weight_N: int | None = None,
        slide_weight_K: int | None = None,
        L: int = 8,
    ) -> torch.Tensor:
        """
        Execute INT8 Linear operation
        
        Args:
            input: [..., K] BF16
            weight: weight (shape depends on kernel path)
            weight_scale: [N] FP32
            out_dtype: output type
            input_scale: static quantization scale
            input_zero_point: asymmetric quantization zero point
            azp_adj: AZP adjustment term
            input_symmetric: whether symmetric quantization
            bias: [N]
            slide_weight_N: cuSPARSELt specific, N dimension
            slide_weight_K: cuSPARSELt specific, K_slide dimension
            L: cuSPARSELt specific, sparsity group size
        """
        # Get input shape info
        input_shape = input.shape
        input_ndim = input.dim()
        
        # Flatten to 2D
        if input_ndim == 2:
            input_2d = input
            M = input_shape[0]
        else:
            input_2d = input.view(-1, input_shape[-1])
            M = input_2d.shape[0]
        
        # Infer output N dimension
        if self._use_cusparselt:
            output_N = slide_weight_N
        elif self._use_cublaslt:
            output_N = weight.shape[0]
        else:
            output_N = weight.shape[1]
        
        # Build output_shape
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
            input_zero_point=input_zero_point,
            azp_adj=azp_adj,
            input_symmetric=input_symmetric,
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
            )


# ============================================================================
# SlideSparse INT8 Linear Method
# ============================================================================

class SlideSparseInt8LinearMethod:
    """
    SlideSparse INT8 Linear Method
    
    Wraps vLLM's original CompressedTensorsW8A8Int8 scheme:
    - create_weights: delegates to original scheme
    - process_weights_after_loading: delegates + cuBLASLt/cuSPARSELt post-processing
    - apply_weights: uses SlideSparseInt8LinearOp
    
    Weight shape changes:
        Original checkpoint: [N, K] or [N, K_slide] (slidesparse checkpoint)
        After vLLM load: [N, K] or [N, K_slide]
        CUTLASS path: weight.t() -> [K, N]
        cuBLASLt path: keeps [N, K]
        cuSPARSELt path: [N, K_slide] -> compress -> [compressed_size] uint8 1D
    """
    
    def __init__(self, original_scheme):
        self.original_scheme = original_scheme
        self.strategy = original_scheme.strategy
        self.is_static_input_scheme = original_scheme.is_static_input_scheme
        self.input_symmetric = original_scheme.input_symmetric
        
        self._use_cublaslt = is_cublaslt_enabled()
        self._use_cusparselt = is_cusparselt_enabled()
        
        # Create SlideSparse Op
        self.slidesparse_int8_linear = SlideSparseInt8LinearOp(
            act_quant_static=self.is_static_input_scheme,
            input_symmetric=self.input_symmetric,
        )
        
        # Update kernel path (Op may fallback due to asymmetric quantization)
        self._use_cublaslt = self.slidesparse_int8_linear._use_cublaslt
        self._use_cusparselt = self.slidesparse_int8_linear._use_cusparselt
        
        # cuSPARSELt sparsity config
        if self._use_cusparselt:
            Z, L, self._expand_ratio = get_sparsity_config()
            self._sparsity_config = SlideSparseConfig(Z=Z, L=L)
            logger.info_once(
                f"SlideSparseInt8LinearMethod: cuSPARSELt "
                f"sparsity={Z}:{L}, expand_ratio={self._expand_ratio:.3f}"
            )
        
        # Preload Triton kernels (torch.compile compatible)
        import os
        model_name = os.environ.get("SLIDESPARSE_MODEL_NAME")
        if model_name and (self._use_cublaslt or self._use_cusparselt):
            # dequant_bias kernel is shared by cuBLASLt and cuSPARSELt
            _load_dequant_bias_kernel(model_name)
            
            if self._use_cublaslt:
                _load_quant_only_int8_kernel(model_name)
            elif self._use_cusparselt:
                _load_quant_slide_int8_kernel(model_name)
            
            logger.info_once(
                f"Preloaded INT8 Triton kernels for model: {model_name}"
            )
        
        logger.info_once(
            f"SlideSparseInt8LinearMethod initialized, "
            f"kernel={self.slidesparse_int8_linear._kernel_name}, "
            f"symmetric={self.input_symmetric}"
        )
    
    def create_weights(
        self,
        layer: Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader,
        **kwargs,
    ):
        """
        Create weight parameters
        
        cuSPARSELt path: expand input_size to match slide checkpoint weight
        Other paths: directly delegate to original scheme
        
        Note:
            INT8 scheme create_weights signature differs slightly from FP8.
        """
        if self._use_cusparselt:
            # cuSPARSELt: expand input_size to match K dimension after slide
            _, input_size_per_partition_slide = compute_output_k(
                input_size_per_partition, self._sparsity_config
            )
            
            return self.original_scheme.create_weights(
                layer=layer,
                output_partition_sizes=output_partition_sizes,
                input_size_per_partition=input_size_per_partition_slide,
                params_dtype=params_dtype,
                weight_loader=weight_loader,
                **kwargs,
            )
        else:
            # CUTLASS / cuBLASLt: directly delegate
            return self.original_scheme.create_weights(
                layer=layer,
                output_partition_sizes=output_partition_sizes,
                input_size_per_partition=input_size_per_partition,
                params_dtype=params_dtype,
                weight_loader=weight_loader,
                **kwargs,
            )
    
    def process_weights_after_loading(self, layer: Module) -> None:
        """
        Post-load weight processing
        
        Processing logic:
        - CUTLASS: delegates to original scheme (executes weight.t() and azp_adj calculation)
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
        
        Input: layer.weight [N, K_slide] INT8
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
            f"cuSPARSELt INT8 compression: [{N}, {K_slide}] -> 1D uint8"
        )
        
        weight_compressed = compress_tensor_online(slide_weight, verbose=False)
        
        layer.weight = Parameter(weight_compressed, requires_grad=False)
        layer.slide_weight_N = N
        layer.slide_weight_K = K_slide
        
        logger.info_once(
            f"cuSPARSELt INT8 compression done: {weight_compressed.numel()} bytes"
        )
    
    def apply_weights(
        self,
        layer: Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply weights (linear transformation)"""
        input_scale = getattr(layer, "input_scale", None)
        input_zero_point = getattr(layer, "input_zero_point", None)
        azp_adj = getattr(layer, "azp_adj", None)
        
        if self._use_cusparselt:
            return self.slidesparse_int8_linear.apply(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                out_dtype=x.dtype,
                input_scale=input_scale,
                input_zero_point=input_zero_point,
                azp_adj=azp_adj,
                input_symmetric=self.input_symmetric,
                bias=bias,
                slide_weight_N=layer.slide_weight_N,
                slide_weight_K=layer.slide_weight_K,
                L=self._sparsity_config.L,
            )
        else:
            return self.slidesparse_int8_linear.apply(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                out_dtype=x.dtype,
                input_scale=input_scale,
                input_zero_point=input_zero_point,
                azp_adj=azp_adj,
                input_symmetric=self.input_symmetric,
                bias=bias,
            )


# ============================================================================
# Factory Function
# ============================================================================

def wrap_scheme_int8(original_scheme):
    """
    INT8 scheme wrapper entry point
    
    Only wraps W8A8Int8 scheme, returns others as-is.
    SlideSparseInt8LinearOp internally selects kernel path based on env vars.
    """
    scheme_name = type(original_scheme).__name__
    if "W8A8Int8" not in scheme_name:
        logger.warning_once(
            f"SlideSparse INT8 not supported for {scheme_name}, using original"
        )
        return original_scheme
    
    # Determine actual backend used
    use_cublaslt = is_cublaslt_enabled()
    use_cusparselt = is_cusparselt_enabled()
    
    # Force CUTLASS for asymmetric quantization
    if not original_scheme.input_symmetric and (use_cublaslt or use_cusparselt):
        backend = "CUTLASS (asymmetric fallback)"
    elif use_cublaslt:
        backend = "cuBLASLt"
    elif use_cusparselt:
        backend = "cuSPARSELt"
    else:
        backend = "CUTLASS"
    
    logger.info_once(
        f"Wrapping {scheme_name} with SlideSparse ({backend})"
    )
    return SlideSparseInt8LinearMethod(original_scheme)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Linear functions
    "cuBLASLt_INT8_linear",
    "cuSPARSELt_INT8_linear",
    "cutlass_INT8_linear",
    
    # Op and Method classes
    "SlideSparseInt8LinearOp",
    "SlideSparseInt8LinearMethod",
    
    # Factory function
    "wrap_scheme_int8",
]
