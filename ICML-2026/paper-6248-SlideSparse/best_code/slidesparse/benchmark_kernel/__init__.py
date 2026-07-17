#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse Kernel Benchmark 模块

本模块用于测试 cuBLASLt (Dense GEMM) 和 cuSPARSELt (Sparse GEMM) 的性能差异。

模块结构:
=========
- utils.py: 通用工具函数
- benchmark_entry.py: 统一入口脚本
- cuBLASLt/: Dense GEMM 算法搜索
  - cublaslt_gemm.cu: CUDA 实现
  - alg_search.py: Python 搜索脚本
  - build_cublaslt.py: 编译脚本
- cuSPARSELt/: Sparse GEMM 算法搜索
  - cusparselt_gemm.cu: CUDA 实现
  - alg_search.py: Python 搜索脚本
  - build_cusparselt.py: 编译脚本

用法:
=====
从 benchmark_kernel 目录运行:

    python benchmark_entry.py --model Qwen2.5-0.5B --dtype fp8e4m3 --sparsity 2_8

或者直接调用子模块:

    from slidesparse.benchmark_kernel.utils import DTYPE_CONFIG
    from slidesparse.benchmark_kernel.cuBLASLt.alg_search import search_single_nk
"""

from .utils import (
    # 常量
    DTYPE_CONFIG,
    SUPPORTED_DTYPES,
    DEFAULT_SPARSITY_LIST,
    ALIGNMENT,
    DEFAULT_M_LIST,
    M_QUICK_LIST,
    # 硬件检测
    hw_info,
    check_dtype_support,
    get_supported_dtypes_for_gpu,
    check_cusparselt_support,
    # Sparsity 计算
    calculate_k_slide,
    get_k_expansion_factor,
    pad_to_alignment,
    # NK 列表
    get_nk_list_for_benchmark,
    # 文件命名
    build_hw_folder_name,
    build_result_filename,
    # 结果整合
    merge_benchmark_results,
)

__all__ = [
    # 常量
    "DTYPE_CONFIG",
    "SUPPORTED_DTYPES",
    "DEFAULT_SPARSITY_LIST",
    "ALIGNMENT",
    "DEFAULT_M_LIST",
    "M_QUICK_LIST",
    # 硬件检测
    "hw_info",
    "check_dtype_support",
    "get_supported_dtypes_for_gpu",
    "check_cusparselt_support",
    # Sparsity 计算
    "calculate_k_slide",
    "get_k_expansion_factor",
    "pad_to_alignment",
    # NK 列表
    "get_nk_list_for_benchmark",
    # 文件命名
    "build_hw_folder_name",
    "build_result_filename",
    # 结果整合
    "merge_benchmark_results",
]
