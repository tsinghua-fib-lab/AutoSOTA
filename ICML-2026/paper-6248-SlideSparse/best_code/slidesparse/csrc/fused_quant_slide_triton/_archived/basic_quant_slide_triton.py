"""
Triton Fused Quant + SlideSparse Slide Kernel

功能：
- 输入：[M, K] BF16/FP16/FP32（行主序）
- 输出：[M, K_out] FP8E4M3 或 INT8（slide 扩展后）
- scale：[M] FP32 per-token scale

SlideSparse Slide 逻辑：
- 输入被分成 num_groups 组，每组 L 个元素
- 每组产生 NUM_WINDOWS = (L/2 - 1) 个滑动窗口
- 每个窗口包含 4 个连续元素（位置 [2*w, 2*w+1, 2*w+2, 2*w+3]）
- 输出形状：K_out = num_groups * NUM_WINDOWS * 4

例如 L=8 (2:8 sparsity):
- 每组 8 个元素 -> 3 个窗口 -> 12 个输出元素
- 窗口 0: 元素 [0, 1, 2, 3]
- 窗口 1: 元素 [2, 3, 4, 5]
- 窗口 2: 元素 [4, 5, 6, 7]
- expand_ratio = 12/8 = 1.5x

计算流程：
1. Pass 1: 计算每行 absmax
2. Pass 2: 量化 + Slide 输出

设计特点：
- L 作为 constexpr 参数，支持任意偶数 L（6, 8, 10, 12, ...）
- Triton 自动为每个 L 值编译并缓存 kernel
- 只需 2 个 kernel 函数（FP8 和 INT8）
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


# =============================================================================
# Constants
# =============================================================================

FP8_E4M3_MAX = 448.0
FP8_MIN_SCALING_FACTOR = 1.0 / (FP8_E4M3_MAX * 512.0)

INT8_MAX = 127
INT8_MIN_SCALING_FACTOR = 1.0 / (INT8_MAX * 512.0)


# =============================================================================
# Triton Kernel - FP8 Quant + Slide (Unified for all L values)
# =============================================================================

@triton.jit
def _quant_slide_fp8_kernel(
    x_ptr, out_ptr, scale_ptr,
    M, K_in_orig, K_in_padded, K_out, num_groups,
    stride_x, stride_out,
    L: tl.constexpr,
    NUM_WINDOWS: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused FP8 Quantization + SlideSparse Slide
    
    L and NUM_WINDOWS are constexpr - Triton compiles specialized kernel for each L.
    """
    row = tl.program_id(0)
    
    FP8_MAX: tl.constexpr = 448.0
    MIN_SCALE: tl.constexpr = 1.0 / (448.0 * 512.0)
    
    x_row = x_ptr + row * stride_x
    out_row32 = out_ptr.to(tl.pointer_type(tl.int32)) + row * (stride_out // 4)
    
    # ===== Pass 1: Compute absmax =====
    absmax = tl.zeros((), dtype=tl.float32)
    for k in range(0, K_in_padded, BLOCK_K):
        offs = k + tl.arange(0, BLOCK_K)
        mask = offs < K_in_orig
        xb = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        absmax = tl.maximum(absmax, tl.max(tl.abs(xb)))
    
    absmax = tl.maximum(absmax, 1e-12)
    scale = tl.maximum(absmax / FP8_MAX, MIN_SCALE)
    inv_scale = FP8_MAX / absmax
    tl.store(scale_ptr + row, scale)
    
    # ===== Pass 2: Quant + Slide =====
    for g_start in range(0, num_groups, BLOCK_GROUPS):
        offs_g = tl.arange(0, BLOCK_GROUPS)
        gid = g_start + offs_g
        mask_g = gid < num_groups
        base_in = gid * L
        base_out = gid * NUM_WINDOWS
        
        # Process each window (loop unrolled at compile time since NUM_WINDOWS is constexpr)
        for w in tl.static_range(NUM_WINDOWS):
            win_start = 2 * w
            
            # Load 4 elements for this window
            x0 = tl.load(x_row + base_in + win_start + 0, 
                        mask=mask_g & ((base_in + win_start + 0) < K_in_orig), other=0.0).to(tl.float32)
            x1 = tl.load(x_row + base_in + win_start + 1,
                        mask=mask_g & ((base_in + win_start + 1) < K_in_orig), other=0.0).to(tl.float32)
            x2 = tl.load(x_row + base_in + win_start + 2,
                        mask=mask_g & ((base_in + win_start + 2) < K_in_orig), other=0.0).to(tl.float32)
            x3 = tl.load(x_row + base_in + win_start + 3,
                        mask=mask_g & ((base_in + win_start + 3) < K_in_orig), other=0.0).to(tl.float32)
            
            # Quantize to FP8 (clamp to ensure numerical safety)
            q0 = tl.clamp(x0 * inv_scale, -FP8_MAX, FP8_MAX).to(tl.float8e4nv)
            q1 = tl.clamp(x1 * inv_scale, -FP8_MAX, FP8_MAX).to(tl.float8e4nv)
            q2 = tl.clamp(x2 * inv_scale, -FP8_MAX, FP8_MAX).to(tl.float8e4nv)
            q3 = tl.clamp(x3 * inv_scale, -FP8_MAX, FP8_MAX).to(tl.float8e4nv)
            
            # Pack 4 FP8 values into int32 (little-endian)
            b0 = q0.to(tl.int8, bitcast=True).to(tl.int32) & 0xFF
            b1 = q1.to(tl.int8, bitcast=True).to(tl.int32) & 0xFF
            b2 = q2.to(tl.int8, bitcast=True).to(tl.int32) & 0xFF
            b3 = q3.to(tl.int8, bitcast=True).to(tl.int32) & 0xFF
            
            packed = (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)).to(tl.int32)
            tl.store(out_row32 + base_out + w, packed, mask=mask_g)


# =============================================================================
# Triton Kernel - INT8 Quant + Slide (Unified for all L values)
# =============================================================================

@triton.jit
def _quant_slide_int8_kernel(
    x_ptr, out_ptr, scale_ptr,
    M, K_in_orig, K_in_padded, K_out, num_groups,
    stride_x, stride_out,
    L: tl.constexpr,
    NUM_WINDOWS: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused INT8 Quantization + SlideSparse Slide
    
    L and NUM_WINDOWS are constexpr - Triton compiles specialized kernel for each L.
    """
    row = tl.program_id(0)
    
    INT8_MAX: tl.constexpr = 127.0
    MIN_SCALE: tl.constexpr = 1.0 / (127.0 * 512.0)
    
    x_row = x_ptr + row * stride_x
    out_row32 = out_ptr.to(tl.pointer_type(tl.int32)) + row * (stride_out // 4)
    
    # ===== Pass 1: Compute absmax =====
    absmax = tl.zeros((), dtype=tl.float32)
    for k in range(0, K_in_padded, BLOCK_K):
        offs = k + tl.arange(0, BLOCK_K)
        mask = offs < K_in_orig
        xb = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        absmax = tl.maximum(absmax, tl.max(tl.abs(xb)))
    
    absmax = tl.maximum(absmax, 1e-12)
    scale = tl.maximum(absmax / INT8_MAX, MIN_SCALE)
    inv_scale = INT8_MAX / absmax
    tl.store(scale_ptr + row, scale)
    
    # ===== Pass 2: Quant + Slide =====
    for g_start in range(0, num_groups, BLOCK_GROUPS):
        offs_g = tl.arange(0, BLOCK_GROUPS)
        gid = g_start + offs_g
        mask_g = gid < num_groups
        base_in = gid * L
        base_out = gid * NUM_WINDOWS
        
        # Process each window
        for w in tl.static_range(NUM_WINDOWS):
            win_start = 2 * w
            
            # Load 4 elements for this window
            x0 = tl.load(x_row + base_in + win_start + 0, 
                        mask=mask_g & ((base_in + win_start + 0) < K_in_orig), other=0.0).to(tl.float32)
            x1 = tl.load(x_row + base_in + win_start + 1,
                        mask=mask_g & ((base_in + win_start + 1) < K_in_orig), other=0.0).to(tl.float32)
            x2 = tl.load(x_row + base_in + win_start + 2,
                        mask=mask_g & ((base_in + win_start + 2) < K_in_orig), other=0.0).to(tl.float32)
            x3 = tl.load(x_row + base_in + win_start + 3,
                        mask=mask_g & ((base_in + win_start + 3) < K_in_orig), other=0.0).to(tl.float32)
            
            # Quantize to INT8 with proper rounding and clamp
            q0 = tl.clamp(tl.extra.cuda.libdevice.rint(x0 * inv_scale), -128.0, 127.0).to(tl.int32) & 0xFF
            q1 = tl.clamp(tl.extra.cuda.libdevice.rint(x1 * inv_scale), -128.0, 127.0).to(tl.int32) & 0xFF
            q2 = tl.clamp(tl.extra.cuda.libdevice.rint(x2 * inv_scale), -128.0, 127.0).to(tl.int32) & 0xFF
            q3 = tl.clamp(tl.extra.cuda.libdevice.rint(x3 * inv_scale), -128.0, 127.0).to(tl.int32) & 0xFF
            
            # Pack 4 INT8 values into int32
            packed = (q0 | (q1 << 8) | (q2 << 16) | (q3 << 24)).to(tl.int32)
            tl.store(out_row32 + base_out + w, packed, mask=mask_g)


# =============================================================================
# Helper Functions
# =============================================================================




def _get_num_windows(L: int) -> int:
    """Calculate number of windows: L/2 - 1"""
    assert L >= 4 and L % 2 == 0, f"L must be even and >= 4, got {L}"
    return L // 2 - 1  # L=4 -> 1 window (no slide), L=8 -> 3 windows, etc.


def _compute_output_k(K_in: int, L: int) -> Tuple[int, int, int]:
    """
    Compute output dimensions for slide operation.
    
    Args:
        K_in: Original input K dimension
        L: Group size (must be even, >= 4)
        
    Returns:
        (K_in_padded, K_out, num_groups)
    """
    K_in_padded = ((K_in + L - 1) // L) * L
    num_groups = K_in_padded // L
    num_windows = _get_num_windows(L)
    K_out = num_groups * num_windows * 4
    return K_in_padded, K_out, num_groups


def _get_block_k(K: int) -> int:
    """Get BLOCK_K for Pass 1 (must be power of 2)"""
    if K <= 2048:
        return 2048
    elif K <= 4096:
        return 4096
    else:
        return 4096


def _get_config(M: int, K: int) -> Tuple[int, int, int]:
    """
    Get configuration for quant+slide kernel.
    
    Returns:
        (BLOCK_GROUPS, num_warps, num_stages)
    """
    # BLOCK_GROUPS heuristic
    if K <= 2048:
        block_groups = 128
    elif K <= 4096:
        block_groups = 256
    else:
        block_groups = 256
    
    # num_warps based on M
    if M <= 16:
        num_warps = 4
    elif M <= 512:
        num_warps = 4
    else:
        num_warps = 8
    
    num_stages = 2
    
    return block_groups, num_warps, num_stages


# =============================================================================
# Main Interface Functions
# =============================================================================

def quant_slide_fp8_triton(
    x: torch.Tensor,
    L: int = 8,
    block_groups: int = None,
    num_warps: int = None,
    num_stages: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Triton Fused FP8 Quantization + SlideSparse Slide
    
    Args:
        x: Input tensor [M, K], BF16/FP16/FP32, must be contiguous
        L: Group size for slide (must be even, >= 4), default 8
        block_groups: Number of groups per block (optional, auto)
        num_warps: Number of warps (optional, auto)
        num_stages: Number of stages (optional, auto)
        
    Returns:
        out: Quantized and slid tensor [M_padded, K_out_padded], FP8E4M3
        scale: Per-token scale [M_padded], FP32
        
    Note:
        - K_out = num_groups * NUM_WINDOWS * 4
        - NUM_WINDOWS = L/2 - 1
        - Triton automatically compiles and caches kernel for each L value
    """
    assert x.is_cuda and x.is_contiguous(), "Input must be CUDA contiguous tensor"
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    assert L >= 4 and L % 2 == 0, f"L must be even and >= 4, got {L}"
    
    M, K_in_orig = x.shape
    K_in_padded, K_out, num_groups = _compute_output_k(K_in_orig, L)
    num_windows = _get_num_windows(L)
    
    # Padding: K_out -> 32 aligned (cuSPARSELt), M -> 16 aligned (算法要求)
    K_out_padded = ((K_out + 31) // 32) * 32
    M_padded = ((M + 15) // 16) * 16
    
    # 使用 zeros 分配，padding 区域天然为 0（torch.compile 友好）
    out = torch.zeros(M_padded, K_out_padded, dtype=torch.float8_e4m3fn, device=x.device)
    # scale padding 为 1.0，避免 dequant 时除以 0
    scale = torch.ones(M_padded, dtype=torch.float32, device=x.device)
    
    # Auto config
    if block_groups is None or num_warps is None or num_stages is None:
        auto_bg, auto_nw, auto_ns = _get_config(M, K_in_orig)
        block_groups = block_groups or auto_bg
        num_warps = num_warps or auto_nw
        num_stages = num_stages or auto_ns
    
    block_k = _get_block_k(K_in_orig)
    
    # Launch kernel - L and num_windows as constexpr (只处理有效的 M 行)
    _quant_slide_fp8_kernel[(M,)](
        x, out, scale,
        M, K_in_orig, K_in_padded, K_out, num_groups,
        x.stride(0), K_out_padded,  # output stride 使用 K_out_padded
        L=L,
        NUM_WINDOWS=num_windows,
        BLOCK_GROUPS=block_groups,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    
    return out, scale


def quant_slide_int8_triton(
    x: torch.Tensor,
    L: int = 8,
    block_groups: int = None,
    num_warps: int = None,
    num_stages: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Triton Fused INT8 Quantization + SlideSparse Slide
    
    Args:
        x: Input tensor [M, K], BF16/FP16/FP32, must be contiguous
        L: Group size for slide (must be even, >= 4), default 8
        block_groups: Number of groups per block (optional, auto)
        num_warps: Number of warps (optional, auto)
        num_stages: Number of stages (optional, auto)
        
    Returns:
        out: Quantized and slid tensor [M_padded, K_out_padded], INT8
        scale: Per-token scale [M_padded], FP32
    """
    assert x.is_cuda and x.is_contiguous(), "Input must be CUDA contiguous tensor"
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    assert L >= 4 and L % 2 == 0, f"L must be even and >= 4, got {L}"
    
    M, K_in_orig = x.shape
    K_in_padded, K_out, num_groups = _compute_output_k(K_in_orig, L)
    num_windows = _get_num_windows(L)
    
    # Padding: K -> 32 aligned, M -> 16 aligned
    K_out_padded = ((K_out + 31) // 32) * 32
    M_padded = ((M + 15) // 16) * 16
    
    # 使用 zeros 分配，padding 区域天然为 0（torch.compile 友好）
    out = torch.zeros(M_padded, K_out_padded, dtype=torch.int8, device=x.device)
    # scale padding 为 1.0，避免 dequant 时除以 0
    scale = torch.ones(M_padded, dtype=torch.float32, device=x.device)
    
    # Auto config
    if block_groups is None or num_warps is None or num_stages is None:
        auto_bg, auto_nw, auto_ns = _get_config(M, K_in_orig)
        block_groups = block_groups or auto_bg
        num_warps = num_warps or auto_nw
        num_stages = num_stages or auto_ns
    
    block_k = _get_block_k(K_in_orig)
    
    # Launch kernel (只处理有效的 M 行)
    _quant_slide_int8_kernel[(M,)](
        x, out, scale,
        M, K_in_orig, K_in_padded, K_out, num_groups,
        x.stride(0), K_out_padded,  # output stride 使用 K_out_padded
        L=L,
        NUM_WINDOWS=num_windows,
        BLOCK_GROUPS=block_groups,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    
    return out, scale


# =============================================================================
# PyTorch Reference Implementation (for correctness verification)
# =============================================================================

def quant_slide_fp8_pytorch(
    x: torch.Tensor,
    L: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch reference implementation - FP8 Quant + Slide
    """
    assert x.dim() == 2
    assert L >= 4 and L % 2 == 0
    
    FP8_MAX = 448.0
    MIN_SCALE = 1.0 / (FP8_MAX * 512.0)
    num_windows = _get_num_windows(L)
    
    M, K_in_orig = x.shape
    K_in_padded, K_out, num_groups = _compute_output_k(K_in_orig, L)
    K_out_padded = ((K_out + 15) // 16) * 16
    
    x_float = x.float()
    
    # Compute scale per row
    absmax = x_float.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-12)
    scale = (absmax / FP8_MAX).clamp(min=MIN_SCALE)
    
    # Quantize
    inv_scale = FP8_MAX / absmax
    x_quant = (x_float * inv_scale).clamp(-FP8_MAX, FP8_MAX)
    
    # Pad input to multiple of L
    if K_in_orig < K_in_padded:
        x_quant = torch.nn.functional.pad(x_quant, (0, K_in_padded - K_in_orig))
    
    # Reshape to groups: [M, num_groups, L]
    x_groups = x_quant.view(M, num_groups, L)
    
    # Generate slide windows
    windows = []
    for w in range(num_windows):
        start = 2 * w
        windows.append(x_groups[:, :, start:start+4])
    
    # Stack and reshape: [M, num_groups, num_windows, 4] -> [M, K_out]
    out = torch.stack(windows, dim=2).reshape(M, num_groups * num_windows * 4)
    
    # Convert to FP8
    out = out.to(torch.float8_e4m3fn)
    
    # Pad output
    if K_out < K_out_padded:
        out = torch.nn.functional.pad(
            out.view(torch.int8), (0, K_out_padded - K_out)
        ).view(torch.float8_e4m3fn)
    
    return out, scale.squeeze(-1)


def quant_slide_int8_pytorch(
    x: torch.Tensor,
    L: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch reference implementation - INT8 Quant + Slide
    """
    assert x.dim() == 2
    assert L >= 4 and L % 2 == 0
    
    INT8_MAX = 127.0
    MIN_SCALE = 1.0 / (INT8_MAX * 512.0)
    num_windows = _get_num_windows(L)
    
    M, K_in_orig = x.shape
    K_in_padded, K_out, num_groups = _compute_output_k(K_in_orig, L)
    K_out_padded = ((K_out + 15) // 16) * 16
    
    x_float = x.float()
    
    # Compute scale per row
    absmax = x_float.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-12)
    scale = (absmax / INT8_MAX).clamp(min=MIN_SCALE)
    
    # Quantize
    inv_scale = INT8_MAX / absmax
    x_quant = torch.round(x_float * inv_scale).clamp(-128, 127)
    
    # Pad input to multiple of L
    if K_in_orig < K_in_padded:
        x_quant = torch.nn.functional.pad(x_quant, (0, K_in_padded - K_in_orig))
    
    # Reshape to groups
    x_groups = x_quant.view(M, num_groups, L)
    
    # Generate slide windows
    windows = []
    for w in range(num_windows):
        start = 2 * w
        windows.append(x_groups[:, :, start:start+4])
    
    # Stack and reshape
    out = torch.stack(windows, dim=2).reshape(M, num_groups * num_windows * 4)
    
    # Convert to INT8
    out = out.to(torch.int8)
    
    # Pad output
    if K_out < K_out_padded:
        out = torch.nn.functional.pad(out, (0, K_out_padded - K_out))
    
    return out, scale.squeeze(-1)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Config helpers
    '_get_config',
    '_compute_output_k',
    '_get_num_windows',
    # Main API (only 2 functions!)
    'quant_slide_fp8_triton',
    'quant_slide_int8_triton',
    # PyTorch reference
    'quant_slide_fp8_pytorch',
    'quant_slide_int8_pytorch',
]
