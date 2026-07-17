"""
Triton优化的 FP8/INT8 Per-Token Quantization Kernel

功能：
- 输入：[M, K] BF16/FP16/FP32（行主序）
- 输出：[M, K] FP8E4M3 或 INT8
- scale：[M] FP32 per-token scale

计算流程：
1. 对每行计算 absmax
2. 计算 scale = absmax / QMAX
3. 量化: qout = clamp(round(x / scale), QMIN, QMAX)

数据类型：
- FP8E4M3: QMAX=448.0, QMIN=-448.0
- INT8:    QMAX=127, QMIN=-128

核心设计（优化版）：
- 每行一个 program（grid = M）
- 使用大 BLOCK_K 减少循环次数
- 利用 coalesced memory access
"""

import torch
import triton
import triton.language as tl


# =============================================================================
# Constants
# =============================================================================

FP8_E4M3_MAX = 448.0
FP8_E4M3_MIN = -448.0
FP8_MIN_SCALING_FACTOR = 1.0 / (FP8_E4M3_MAX * 512.0)

INT8_MAX = 127
INT8_MIN = -128
INT8_MIN_SCALING_FACTOR = 1.0 / (INT8_MAX * 512.0)


# =============================================================================
# Triton Kernel - Per-Row Dynamic Quantization
# =============================================================================
#
# 设计思路：
# 1. 每行一个 program（类似 vLLM CUDA 的 per-token block 设计）
# 2. BLOCK_K 作为主要调优参数，控制每次迭代处理的元素数
# 3. M 虽然不在 kernel 参数中，但影响最佳 num_warps/num_stages
# 4. 两遍扫描：Pass 1 计算 absmax，Pass 2 量化
# =============================================================================

@triton.jit
def _quant_only_fp8_kernel(
    x_ptr, out_ptr, scale_ptr,
    M, K: tl.constexpr,
    stride_xm, stride_om,
    BLOCK_K: tl.constexpr,
):
    """
    FP8E4M3 per-token quantization kernel - 每行一个 program
    
    Args:
        x_ptr: 输入指针 [M, K]
        out_ptr: 输出指针 [M, K]
        scale_ptr: scale 指针 [M]
        M: 行数
        K: 列数 (constexpr for loop unrolling)
        stride_xm: 输入行步长
        stride_om: 输出行步长
        BLOCK_K: K 方向块大小
    """
    row = tl.program_id(0)
    
    FP8_MAX: tl.constexpr = 448.0
    MIN_SCALE: tl.constexpr = 1.0 / (448.0 * 512.0)
    
    x_row_ptr = x_ptr + row * stride_xm
    out_row_ptr = out_ptr + row * stride_om
    
    # Pass 1: 计算 absmax
    absmax = tl.zeros((), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x_val = tl.load(x_row_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        absmax = tl.maximum(absmax, tl.max(tl.abs(x_val)))
    
    # 计算 scale
    absmax = tl.maximum(absmax, 1e-12)
    scale = tl.maximum(absmax / FP8_MAX, MIN_SCALE)
    inv_scale = FP8_MAX / absmax
    
    tl.store(scale_ptr + row, scale)
    
    # Pass 2: 量化
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x_val = tl.load(x_row_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        y_val = tl.clamp(x_val * inv_scale, -FP8_MAX, FP8_MAX)
        tl.store(out_row_ptr + offs_k, y_val.to(tl.float8e4nv), mask=mask_k)


@triton.jit
def _quant_only_int8_kernel(
    x_ptr, out_ptr, scale_ptr,
    M, K: tl.constexpr,
    stride_xm, stride_om,
    BLOCK_K: tl.constexpr,
):
    """
    INT8 per-token quantization kernel (symmetric) - 每行一个 program
    """
    row = tl.program_id(0)
    
    INT8_MAX: tl.constexpr = 127.0
    MIN_SCALE: tl.constexpr = 1.0 / (127.0 * 512.0)
    
    x_row_ptr = x_ptr + row * stride_xm
    out_row_ptr = out_ptr + row * stride_om
    
    # Pass 1: 计算 absmax
    absmax = tl.zeros((), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x_val = tl.load(x_row_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        absmax = tl.maximum(absmax, tl.max(tl.abs(x_val)))
    
    # 计算 scale
    absmax = tl.maximum(absmax, 1e-12)
    scale = tl.maximum(absmax / INT8_MAX, MIN_SCALE)
    inv_scale = INT8_MAX / absmax
    
    tl.store(scale_ptr + row, scale)
    
    # Pass 2: 量化
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x_val = tl.load(x_row_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        y_val = tl.clamp(tl.extra.cuda.libdevice.rint(x_val * inv_scale), -128.0, 127.0)
        tl.store(out_row_ptr + offs_k, y_val.to(tl.int8), mask=mask_k)


# =============================================================================
# 配置选择
# =============================================================================




def _get_config(M: int, K: int) -> tuple[int, int, int]:
    """
    根据 M, K 选择最优配置
    
    Args:
        M: 行数 (batch size / token 数)
        K: 列数 (hidden size)
        
    Returns:
        (BLOCK_K, num_warps, num_stages)
        
    设计原则：
    1. BLOCK_K 必须是 2 的幂次
    2. 尽量减少 K / BLOCK_K 的循环次数
    3. M 影响 num_warps：小 M 用少 warps，大 M 用多 warps
    4. num_stages 根据内存带宽和计算强度调整
    """
    # BLOCK_K 选择：尽量减少循环次数
    if K <= 1024:
        block_k = 1024
    elif K <= 2048:
        block_k = 2048
    elif K <= 4096:
        block_k = 2048  # 2 次循环
    else:
        block_k = 4096  # 对于 6912，只需要 2 次循环
    
    # num_warps 根据 M 调整
    if M <= 16:
        num_warps = 4
    elif M <= 256:
        num_warps = 4
    elif M <= 4096:
        num_warps = 8
    else:
        num_warps = 8
    
    # num_stages
    num_stages = 2
    
    return block_k, num_warps, num_stages


# =============================================================================
# 主接口函数
# =============================================================================

def quant_only_fp8_triton(
    x: torch.Tensor,
    block_k: int = None,
    num_warps: int = None,
    num_stages: int = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triton FP8E4M3 per-token quantization
    
    Args:
        x: 输入张量 [M, K]，BF16/FP16/FP32，必须 contiguous
        block_k: K方向块大小（可选，自动选择）
        num_warps: warp 数量（可选，自动选择）
        num_stages: pipeline stages（可选，自动选择）
        
    Returns:
        out: 量化输出 [M_padded, K_padded]，FP8E4M3
        scale: per-token scale [M_padded]，FP32
    """
    assert x.is_cuda and x.is_contiguous(), "Input must be CUDA contiguous tensor"
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    
    M, K = x.shape
    
    # Padding: K -> 32 aligned, M -> 16 aligned
    K_padded = ((K + 31) // 32) * 32
    M_padded = ((M + 15) // 16) * 16
    
    # 使用 zeros 分配，padding 区域天然为 0（torch.compile 友好）
    out = torch.zeros(M_padded, K_padded, dtype=torch.float8_e4m3fn, device=x.device)
    # scale padding 为 1.0，避免 dequant 时除以 0
    scale = torch.ones(M_padded, dtype=torch.float32, device=x.device)
    
    # 自动选择配置
    if block_k is None or num_warps is None or num_stages is None:
        auto_block_k, auto_num_warps, auto_num_stages = _get_config(M, K)
        block_k = block_k or auto_block_k
        num_warps = num_warps or auto_num_warps
        num_stages = num_stages or auto_num_stages
    
    # 每行一个 program (只处理有效的 M 行)
    _quant_only_fp8_kernel[(M,)](
        x, out, scale,
        M, K,
        x.stride(0), K_padded,  # output stride 使用 K_padded
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    
    return out, scale


def quant_only_int8_triton(
    x: torch.Tensor,
    block_k: int = None,
    num_warps: int = None,
    num_stages: int = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triton INT8 per-token quantization (symmetric)
    
    Args:
        x: 输入张量 [M, K]，BF16/FP16/FP32，必须 contiguous
        block_k: K方向块大小（可选，自动选择）
        num_warps: warp 数量（可选，自动选择）
        num_stages: pipeline stages（可选，自动选择）
        
    Returns:
        out: 量化输出 [M_padded, K_padded]，INT8，padding 区域为 0
        scale: per-token scale [M_padded]，FP32，padding 区域为 1.0
    """
    assert x.is_cuda and x.is_contiguous(), "Input must be CUDA contiguous tensor"
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    
    M, K = x.shape
    
    # Padding: K -> 32 aligned, M -> 16 aligned
    K_padded = ((K + 31) // 32) * 32
    M_padded = ((M + 15) // 16) * 16
    
    # 使用 zeros 分配，padding 区域天然为 0（torch.compile 友好）
    out = torch.zeros(M_padded, K_padded, dtype=torch.int8, device=x.device)
    # scale padding 为 1.0，避免 dequant 时除以 0
    scale = torch.ones(M_padded, dtype=torch.float32, device=x.device)
    
    # 自动选择配置
    if block_k is None or num_warps is None or num_stages is None:
        auto_block_k, auto_num_warps, auto_num_stages = _get_config(M, K)
        block_k = block_k or auto_block_k
        num_warps = num_warps or auto_num_warps
        num_stages = num_stages or auto_num_stages
    
    # 每行一个 program (只处理有效的 M 行)
    _quant_only_int8_kernel[(M,)](
        x, out, scale,
        M, K,
        x.stride(0), K_padded,  # output stride 使用 K_padded
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    
    return out, scale



# =============================================================================
# PyTorch参考实现 (用于正确性验证)
# =============================================================================

def quant_only_fp8_pytorch(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch参考实现 - FP8E4M3 per-token quantization
    
    计算:
        absmax = max(abs(x), dim=-1)
        scale = max(absmax / FP8_MAX, MIN_SCALE)
        out = clamp(x * inv_scale, -FP8_MAX, FP8_MAX).to(fp8)
    """
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    
    FP8_MAX = 448.0
    MIN_SCALE = 1.0 / (FP8_MAX * 512.0)
    
    x_float = x.float()
    absmax = x_float.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-12)
    scale = (absmax / FP8_MAX).clamp(min=MIN_SCALE)
    inv_scale = FP8_MAX / absmax
    out = (x_float * inv_scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    
    return out, scale.squeeze(-1)


def quant_only_int8_pytorch(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch参考实现 - INT8 per-token quantization (symmetric)
    
    计算:
        absmax = max(abs(x), dim=-1)
        scale = max(absmax / 127, MIN_SCALE)
        out = round(x * inv_scale).clamp(-128, 127).to(int8)
    """
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    
    INT8_MAX = 127.0
    MIN_SCALE = 1.0 / (INT8_MAX * 512.0)
    
    x_float = x.float()
    absmax = x_float.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-12)
    scale = (absmax / INT8_MAX).clamp(min=MIN_SCALE)
    inv_scale = INT8_MAX / absmax
    out = torch.round(x_float * inv_scale).clamp(-128, 127).to(torch.int8)
    
    return out, scale.squeeze(-1)


# =============================================================================
# 导出接口
# =============================================================================

__all__ = [
    # Config
    '_get_config',
    # Main API
    'quant_only_fp8_triton',
    'quant_only_int8_triton',
    # PyTorch reference
    'quant_only_fp8_pytorch',
    'quant_only_int8_pytorch',
]
