#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse CSRC 工具库

本模块提供 Triton Autotune 配置等 CSRC 专用功能。

编译相关功能请使用顶层 slidesparse.utils 模块：
    from slidesparse.utils import (
        build_cuda_extension,
        build_cuda_extension_direct,
        get_nvcc_arch_flags,
        CUBLASLT_LDFLAGS,
        CUSPARSELT_LDFLAGS,
    )

主要功能
========
1. Triton Autotune 配置（dequant, quant kernels）

向后兼容
========
为保持向后兼容，以下符号从顶层 utils 重新导出：
- get_nvcc_arch_flags, get_current_arch_flag
- SUPPORTED_ARCHITECTURES
- DEFAULT_CFLAGS, DEFAULT_CUDA_CFLAGS
- should_rebuild, clean_build_artifacts, build_cuda_extension
- CUBLASLT_LDFLAGS, CUSPARSELT_LDFLAGS, get_gemm_ldflags

使用示例
========
>>> from slidesparse.csrc.utils import get_dequant_autotune_configs
>>> configs = get_dequant_autotune_configs()
"""

# =============================================================================
# 向后兼容：从顶层 utils 重新导出
# =============================================================================

from slidesparse.utils import (
    # NVCC 架构标志
    SUPPORTED_ARCHITECTURES,
    get_nvcc_arch_flags,
    get_current_arch_flag,
    # 编译选项
    DEFAULT_CFLAGS,
    DEFAULT_CUDA_CFLAGS,
    # 编译工具
    should_rebuild,
    clean_build_artifacts,
    build_cuda_extension,
    # GEMM 链接库
    CUBLASLT_LDFLAGS,
    CUSPARSELT_LDFLAGS,
    get_backend_ldflags as get_gemm_ldflags,  # 别名兼容
)


# =============================================================================
# Triton Autotune 配置
# =============================================================================

def get_dequant_autotune_configs():
    """
    获取 dequant+bias kernel 的 Triton autotune 配置
    
    覆盖: SM80(A100), SM89(4090), SM90(H100/H200), SM100(B200), SM120(5080)
    
    Kernel特性:
    - 2D tile kernel，grid = (M/BLOCK_M, N/BLOCK_N)
    - Memory-bound: 读取 gemm[M,N], scale_a[M], scale_b[N], bias[N]，写 out[M,N]
    - 每个 block 处理 BLOCK_M × BLOCK_N 个元素
    
    优化策略:
    - 保留 Proven Winners（实测验证）
    - 保留 Basic Heuristics（必须覆盖）
    - 删除低效配置：num_stages=1（pipeline不足）、极端warp配置
    - 合并相似配置：每个block size只保留2个stages变体
    
    Returns:
        triton.Config 对象列表
    """
    import triton
    
    return [
        # =====================================================================
        # Tier 1: Proven Winners (A100/H100 validated)
        # =====================================================================
        # Small M (M=1~128): 32x32 - 小block减少边界处理开销
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        # Medium M (M=256~8192): 64x32 - 平衡M和N方向
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        # Large M (M=12288+): 128x64 - 高吞吐
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),

        # =====================================================================
        # Tier 2: Basic Heuristics - 基础配置
        # =====================================================================
        # 32x64: Small M 通用配置
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        # 32x128: Small M + Large N
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        # 64x64: Medium M 平衡配置
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        # 64x128: Medium M + Large N
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        # 128x128: Large M + Large N
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),

        # =====================================================================
        # Tier 3: warp 变体探索 (针对不同GPU架构)
        # =====================================================================
        # 64x64 高warp: H100/B200 可能更优
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        # 64x128 低warp: A100/4090 可能更优
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        # 128x64 低warp: 减少warp调度开销
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        # 128x32 高warp: 高M场景
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=8, num_stages=3),

        # =====================================================================
        # Tier 4: H100/H200/B200/5080 高性能配置 (SM90+)
        # =====================================================================
        # 128x128 高warp: 充分利用SM资源
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=16, num_stages=3),
        # 256x64: 超大M场景
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        # 64x256: 超大N场景
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256}, num_warps=8, num_stages=3),

        # =====================================================================
        # Tier 5: 小M特殊优化 (M=1~32)
        # =====================================================================
        # 16x64: 极小M，2 warps避免资源浪费
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        # 16x128: 极小M + 中等N
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        # 32x64 低warp
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=2, num_stages=2),
    ]


def get_quant_autotune_configs():
    """
    获取 quant (per-row quantization) kernel 的 Triton autotune 配置
    
    覆盖: SM80(A100), SM89(4090), SM90(H100/H200), SM100(B200), SM120(5080)
    
    Kernel特性:
    - Per-row kernel，grid = (M,)
    - 两次遍历: Pass1 计算 absmax，Pass2 量化写入
    - Memory-bound: 每行读 2×K 元素，写 K 元素
    - BLOCK_K 控制每次循环处理的元素数
    
    优化策略:
    - BLOCK_K 应接近或覆盖常见 K 值（减少循环次数）
    - 小 M 用低 warps（1-4），大 M 用高 warps（8-16）
    - 删除 num_stages=1（pipeline 不足）
    - 删除 num_warps=32（per-row 设计用不满）
    - 精简为核心有效配置
    
    常见 K 值: 896, 1536, 2048, 2560, 3584, 4096, 5120, 6912, 8192
    
    Returns:
        triton.Config 对象列表
    """
    import triton
    
    return [
        # =====================================================================
        # Tier 1: BLOCK_K=1024 (K <= 2048 的默认选择)
        # =====================================================================
        triton.Config({'BLOCK_K': 1024}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_K': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_K': 1024}, num_warps=8, num_stages=2),
        
        # =====================================================================
        # Tier 2: BLOCK_K=2048 (K=2048~4096 的主力配置)
        # =====================================================================
        triton.Config({'BLOCK_K': 2048}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_K': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 2048}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_K': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 2048}, num_warps=8, num_stages=3),
        
        # =====================================================================
        # Tier 3: BLOCK_K=4096 (K=4096~8192 的主力配置)
        # =====================================================================
        triton.Config({'BLOCK_K': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 4096}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_K': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 4096}, num_warps=16, num_stages=2),
        
        # =====================================================================
        # Tier 4: BLOCK_K=8192 (K > 6000，如 K=6912, 8192)
        # =====================================================================
        triton.Config({'BLOCK_K': 8192}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 8192}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 8192}, num_warps=16, num_stages=2),
        
        # =====================================================================
        # Tier 5: 小 M 特殊优化 (M=1~64) - 低 warp
        # =====================================================================
        triton.Config({'BLOCK_K': 2048}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK_K': 4096}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK_K': 4096}, num_warps=2, num_stages=2),
    ]


def get_quant_slide_autotune_configs():
    """
    获取 quant_slide (per-row quantization + slide) kernel 的 Triton autotune 配置
    
    覆盖: SM80(A100), SM89(4090), SM90(H100/H200), SM100(B200), SM120(5080)
    
    Kernel特性（最重要的kernel，保守优化）:
    - Per-row kernel，grid = (M,)
    - Pass 1: 用 BLOCK_K 遍历整行计算 absmax
    - Pass 2: 用 BLOCK_OUT 遍历输出位置，做 quant + slide（含div/mod索引计算）
    - Memory-bound + 复杂索引计算
    
    调优参数:
    - BLOCK_OUT: 每次处理的输出 int32 数量（每个存4个量化值）
    - BLOCK_K: Pass 1 的块大小
    - num_warps: warp 数量
    - num_stages: pipeline stages
    
    优化策略（保守）:
    - 完整保留 Basic Heuristics（必须覆盖的默认配置）
    - 保留已验证有效的配置
    - 保留足够的高性能GPU探索空间
    - quant_slide 是最重要的 kernel，配置数量适当多一些
    
    Returns:
        triton.Config 对象列表
    """
    import triton
    
    return [
        # =====================================================================
        # BASIC HEURISTICS - 必须保留（default fallback 配置）
        # =====================================================================
        # M <= 64: BLOCK_OUT=128
        triton.Config({'BLOCK_OUT': 128, 'BLOCK_K': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OUT': 128, 'BLOCK_K': 4096}, num_warps=4, num_stages=2),
        # M <= 1024: BLOCK_OUT=256
        triton.Config({'BLOCK_OUT': 256, 'BLOCK_K': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OUT': 256, 'BLOCK_K': 4096}, num_warps=8, num_stages=2),
        # M > 1024: BLOCK_OUT=512
        triton.Config({'BLOCK_OUT': 512, 'BLOCK_K': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OUT': 512, 'BLOCK_K': 4096}, num_warps=8, num_stages=2),
        
        # =====================================================================
        # Tier 1: 小 M (1-64) 优化 - BLOCK_OUT=128
        # =====================================================================
        triton.Config({'BLOCK_OUT': 128, 'BLOCK_K': 2048}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_OUT': 128, 'BLOCK_K': 2048}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OUT': 128, 'BLOCK_K': 4096}, num_warps=4, num_stages=3),
        
        # =====================================================================
        # Tier 2: 中等 M (64-1024) 优化 - BLOCK_OUT=256
        # =====================================================================
        triton.Config({'BLOCK_OUT': 256, 'BLOCK_K': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OUT': 256, 'BLOCK_K': 2048}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OUT': 256, 'BLOCK_K': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OUT': 256, 'BLOCK_K': 4096}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OUT': 256, 'BLOCK_K': 4096}, num_warps=8, num_stages=3),
        
        # =====================================================================
        # Tier 3: 大 M (1024-8192) 优化 - BLOCK_OUT=512
        # =====================================================================
        triton.Config({'BLOCK_OUT': 512, 'BLOCK_K': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OUT': 512, 'BLOCK_K': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OUT': 512, 'BLOCK_K': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_OUT': 512, 'BLOCK_K': 4096}, num_warps=16, num_stages=3),
        
        # =====================================================================
        # Tier 4: 超大 M (8192+) 优化 - BLOCK_OUT=1024
        # =====================================================================
        triton.Config({'BLOCK_OUT': 1024, 'BLOCK_K': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OUT': 1024, 'BLOCK_K': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_OUT': 1024, 'BLOCK_K': 4096}, num_warps=16, num_stages=3),
        
        # =====================================================================
        # Tier 5: 大 K (K > 8192) 特殊优化 - BLOCK_K=8192
        # =====================================================================
        triton.Config({'BLOCK_OUT': 256, 'BLOCK_K': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OUT': 512, 'BLOCK_K': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OUT': 512, 'BLOCK_K': 8192}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_OUT': 1024, 'BLOCK_K': 8192}, num_warps=16, num_stages=2),
        
        # =====================================================================
        # Tier 6: H100/H200/B200/5080 高性能配置 (SM90+)
        # =====================================================================
        # 高warp大BLOCK_OUT: 充分利用SM资源
        triton.Config({'BLOCK_OUT': 512, 'BLOCK_K': 4096}, num_warps=32, num_stages=2),
        triton.Config({'BLOCK_OUT': 1024, 'BLOCK_K': 4096}, num_warps=32, num_stages=2),
        # 超大BLOCK_OUT探索（B200/5080可能受益）
        triton.Config({'BLOCK_OUT': 2048, 'BLOCK_K': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_OUT': 2048, 'BLOCK_K': 4096}, num_warps=16, num_stages=3),
    ]


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    # =========================================================================
    # 向后兼容（从顶层 utils 重新导出）
    # =========================================================================
    # NVCC 架构标志
    'SUPPORTED_ARCHITECTURES',
    'get_nvcc_arch_flags',
    'get_current_arch_flag',
    # 编译选项
    'DEFAULT_CFLAGS',
    'DEFAULT_CUDA_CFLAGS',
    # 编译工具
    'should_rebuild',
    'clean_build_artifacts',
    'build_cuda_extension',
    # GEMM 链接库
    'CUBLASLT_LDFLAGS',
    'CUSPARSELT_LDFLAGS',
    'get_gemm_ldflags',
    
    # =========================================================================
    # 本模块特有功能
    # =========================================================================
    # Triton Autotune 配置
    'get_dequant_autotune_configs',
    'get_quant_autotune_configs',
    'get_quant_slide_autotune_configs',
]
