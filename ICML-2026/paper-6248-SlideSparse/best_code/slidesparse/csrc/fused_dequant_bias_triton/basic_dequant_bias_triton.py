"""
Triton优化的Dequant + Bias融合Kernel

功能：
- 输入：[M,N] BF16或FP32 GEMM输出（行主序）
- per-token scale: [M,1] FP32
- per-channel scale: [1,N] FP32  
- per-channel bias: [1,N] BF16
- 输出：[M,N] BF16

计算流程：
1. 读取GEMM输出（如果是BF16则转FP32，如果已是FP32则直接使用）
2. 与per-token scale [M,1] 和 per-channel scale [1,N] 做外积（逐点乘法）
3. 加上per-channel bias（BF16转FP32）
4. 转换回BF16输出
"""

import torch
import triton
import triton.language as tl
from typing import Optional


# =============================================================================
# Triton Kernel - 支持 BF16, FP32 和 INT32 输入
# =============================================================================

@triton.jit
def _dequant_bias_kernel(
    # 输入指针
    gemm_output_ptr,      # [M, N] BF16, FP32 或 INT32
    scale_a_ptr,          # [M] FP32 per-token scale
    scale_b_ptr,          # [N] FP32 per-channel scale
    bias_ptr,             # [N] BF16 bias
    output_ptr,           # [M, N] BF16 输出
    # 形状参数
    M,
    N,
    # 步长 (element count, not bytes)
    stride_gm,
    stride_gn,
    stride_om,
    stride_on,
    # 块大小
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # 输入dtype标志
    INPUT_FP32: tl.constexpr,   # 输入是否为 FP32
    INPUT_INT32: tl.constexpr,  # 输入是否为 INT32 (来自 cuBLASLt INT8 GEMM)
):
    """
    2D分块的Triton kernel用于dequant+bias融合操作
    默认有bias，支持BF16, FP32和INT32输入
    """
    # 获取当前program的2D索引
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 计算当前块的起始行列
    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N
    
    # 生成块内的偏移
    row_offs = row_start + tl.arange(0, BLOCK_M)
    col_offs = col_start + tl.arange(0, BLOCK_N)
    
    # 边界掩码
    row_mask = row_offs < M
    col_mask = col_offs < N
    mask_2d = row_mask[:, None] & col_mask[None, :]
    
    # 加载 scale_a [BLOCK_M] - per-token scale (FP32)
    scale_a = tl.load(scale_a_ptr + row_offs, mask=row_mask, other=1.0)
    
    # 加载 scale_b [BLOCK_N] - per-channel scale (FP32)
    scale_b = tl.load(scale_b_ptr + col_offs, mask=col_mask, other=1.0)
    
    # 加载 bias [BLOCK_N] (BF16 -> FP32)
    bias = tl.load(bias_ptr + col_offs, mask=col_mask, other=0.0)
    bias = bias.to(tl.float32)
    
    # 计算2D索引用于加载gemm_output
    gemm_offs = row_offs[:, None] * stride_gm + col_offs[None, :] * stride_gn
    
    # 加载 gemm_output [BLOCK_M, BLOCK_N]
    gemm_val = tl.load(gemm_output_ptr + gemm_offs, mask=mask_2d, other=0.0)
    
    # 转换为 FP32:
    # - INT32: 来自 cuBLASLt INT8 GEMM, 需要转 FP32
    # - BF16: 来自 cuSPARSELt FP8/INT8 GEMM, 需要转 FP32
    # - FP32: 直接使用
    if INPUT_INT32:
        gemm_val = gemm_val.to(tl.float32)
    elif not INPUT_FP32:
        gemm_val = gemm_val.to(tl.float32)
    
    # 计算外积: output = gemm_output * scale_a[M,1] * scale_b[1,N]
    output_val = gemm_val * scale_a[:, None] * scale_b[None, :]
    
    # 加 bias
    output_val = output_val + bias[None, :]
    
    # 转换为 BF16 并存储
    output_val = output_val.to(tl.bfloat16)
    
    # 计算输出偏移
    output_offs = row_offs[:, None] * stride_om + col_offs[None, :] * stride_on
    
    # 存储结果
    tl.store(output_ptr + output_offs, output_val, mask=mask_2d)



def _get_best_config(M: int, N: int) -> tuple:
    """
    根据矩阵大小选择最优配置
    返回: (BLOCK_M, BLOCK_N, num_warps)
    """
    # 小 M (batch size 小)
    if M <= 128:
        if N <= 4096:
            return 32, 64, 4
        else:
            return 32, 128, 4
    # 中等 M
    elif M <= 2048:
        if N <= 4096:
            return 64, 64, 4
        else:
            return 64, 128, 8
    # 大 M
    else:
        if N <= 4096:
            return 128, 64, 8
        else:
            return 128, 128, 8


def _prepare_scale(scale: torch.Tensor, size: int) -> torch.Tensor:
    """Prepare scale tensor: view as 1D, ensure float32, contiguous."""
    if scale.numel() == 1:
        scale = scale.view(1).expand(size)
    else:
        scale = scale.view(-1)
    # Only convert if not already float32 and contiguous
    if scale.dtype != torch.float32:
        return scale.contiguous().float()
    return scale.contiguous() if not scale.is_contiguous() else scale


# =============================================================================
# 主接口函数
# =============================================================================

def dequant_bias_triton(
    gemm_output: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Triton实现的 Dequant + Bias 操作
    
    计算: output = gemm_output * scale_a[M,1] * scale_b[1,N] + bias[1,N]
    
    自动检测输入dtype：
    - 如果 gemm_output 是 FP32，直接用 FP32 计算
    - 如果 gemm_output 是 BF16，先转为 FP32 再计算
    - 如果 gemm_output 是 INT32 (来自 cuBLASLt INT8 GEMM)，先转为 FP32 再计算
    
    根据矩阵大小自动选择最优 (BLOCK_M, BLOCK_N) 配置
    
    Args:
        gemm_output: GEMM 输出 [M, N]，BF16, FP32 或 INT32（行主序）
        scale_a: per-token scale [M, 1] 或 [M] FP32
        scale_b: per-channel scale [1, N] 或 [N] FP32
        bias: per-channel 偏置 [N] 或 [1, N] BF16
        out_dtype: 输出数据类型（默认BF16）
        
    Returns:
        dequant 后的输出 [M, N]，out_dtype
    """
    assert gemm_output.is_cuda, "gemm_output must be on CUDA"
    assert gemm_output.is_contiguous(), "gemm_output must be contiguous"
    assert gemm_output.dtype in [torch.bfloat16, torch.float32, torch.int32], \
        f"gemm_output must be BF16, FP32 or INT32, got {gemm_output.dtype}"
    
    M, N = gemm_output.shape
    input_fp32 = gemm_output.dtype == torch.float32
    input_int32 = gemm_output.dtype == torch.int32
    
    # 准备 scale_a: 确保是 [M] 的连续FP32张量 (使用优化后的函数)
    scale_a = _prepare_scale(scale_a, M)
    assert scale_a.shape[0] == M, f"scale_a shape mismatch: {scale_a.shape[0]} vs {M}"
    
    # 准备 scale_b: 确保是 [N] 的连续FP32张量 (使用优化后的函数)
    scale_b = _prepare_scale(scale_b, N)
    assert scale_b.shape[0] == N, f"scale_b shape mismatch: {scale_b.shape[0]} vs {N}"
    
    # 准备 bias: 确保是 [N] 的连续BF16张量 (只在需要时转换)
    bias = bias.view(-1)
    if bias.dtype != torch.bfloat16:
        bias = bias.to(torch.bfloat16)
    bias = bias.contiguous() if not bias.is_contiguous() else bias
    assert bias.shape[0] == N, f"bias shape mismatch: {bias.shape[0]} vs {N}"
    
    # 分配输出
    output = torch.empty((M, N), dtype=torch.bfloat16, device=gemm_output.device)
    
    # 根据矩阵大小选择最优配置
    block_m, block_n, num_warps = _get_best_config(M, N)
    
    # 计算grid大小
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    
    # 获取步长 (element count)
    stride_gm, stride_gn = gemm_output.stride()
    stride_om, stride_on = output.stride()
    
    # 启动kernel
    _dequant_bias_kernel[grid](
        gemm_output,
        scale_a,
        scale_b,
        bias,
        output,
        M, N,
        stride_gm, stride_gn,
        stride_om, stride_on,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        INPUT_FP32=input_fp32,
        INPUT_INT32=input_int32,
        num_warps=num_warps,
    )
    
    # 转换为目标dtype
    if out_dtype != torch.bfloat16:
        output = output.to(out_dtype)
    
    return output


# =============================================================================
# PyTorch参考实现 (用于正确性验证)
# =============================================================================

def dequant_bias_pytorch(
    gemm_output: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    PyTorch参考实现（用于正确性验证）
    
    计算: output = gemm_output * scale_a[M,1] * scale_b[1,N] + bias[1,N]
    
    自动检测输入dtype并处理:
    - FP32: 直接计算
    - BF16: 转为 FP32 计算
    - INT32: 转为 FP32 计算
    """
    M, N = gemm_output.shape
    
    if scale_a.numel() == 1:
        scale_a_broadcast = scale_a.view(1, 1).float()
    else:
        scale_a_broadcast = scale_a.view(-1, 1).float()
    
    if scale_b.numel() == 1:
        scale_b_broadcast = scale_b.view(1, 1).float()
    else:
        scale_b_broadcast = scale_b.view(1, -1).float()
    
    # gemm_output 统一转为 FP32 计算
    # - FP32: 直接用
    # - BF16: 转 FP32
    # - INT32: 转 FP32
    output = gemm_output.float() * scale_a_broadcast * scale_b_broadcast
    
    # 加 bias
    output = output + bias.float().view(1, -1)
    
    return output.to(out_dtype)


# =============================================================================
# 导出接口
# =============================================================================

__all__ = [
    'dequant_bias_triton',
    'dequant_bias_pytorch',
]
