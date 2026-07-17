#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试脚本 v4：添加详细日志追踪 vLLM 调用路径

直接 import vLLM 组件并追踪实际的调用参数
"""

import os
import sys
from pathlib import Path

# 配置环境变量
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
os.environ["USE_CUBLASLT"] = "1"
os.environ["DISABLE_SLIDESPARSE"] = "0"
os.environ["SLIDESPARSE_MODEL_NAME"] = "Llama3.2-1B-FP8"

import torch

# 需要先设置模型名再 import
from slidesparse.core import init_slidesparse
init_slidesparse("Llama3.2-1B-FP8")

from slidesparse.core.gemm_wrapper import (
    _get_gemm_extension,
    get_algo_config_manager,
)

def main():
    print("=" * 70)
    print("追踪 vLLM 调用路径")
    print("=" * 70)
    
    # 1. 获取算法配置管理器
    print("\n[1] 获取算法配置管理器...")
    mgr = get_algo_config_manager()
    print(f"    当前模型: {mgr.get_model()}")
    print(f"    基础模型: {mgr.get_base_model()}")
    print(f"    cuBLASLt 配置模型数: {len(mgr._cublaslt_configs)}")
    
    # 打印已加载的模型
    print(f"    已加载的 cuBLASLt 模型: {list(mgr._cublaslt_configs.keys())}")
    
    # 2. 获取 GEMM 扩展
    print("\n[2] 获取 GEMM 扩展...")
    ext = _get_gemm_extension("cublaslt")
    print(f"    扩展类型: {type(ext)}")
    
    # 3. 测试查表
    print("\n[3] 测试查表...")
    test_cases = [
        (3072, 2048, 16),
        (3072, 2048, 128),
        (3072, 2048, 1024),
        (2048, 2048, 16),
        (2048, 8192, 16),
    ]
    
    for N, K, M in test_cases:
        algo_data, workspace = mgr.lookup_cublaslt(N, K, M)
        if algo_data:
            print(f"    ({N}, {K}, M={M}): algo_data={algo_data[:8].hex()}..., workspace={workspace}")
        else:
            print(f"    ({N}, {K}, M={M}): 无配置")
    
    # 4. 实际执行 GEMM
    print("\n[4] 实际执行 GEMM...")
    
    # 生成测试数据 (FP8)
    N, K, M = 3072, 2048, 16
    
    W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    
    # 量化为 FP8
    W_q = W.to(torch.float8_e4m3fn)
    A_q = A.to(torch.float8_e4m3fn)
    
    print(f"    W_q: {W_q.shape}, {W_q.dtype}")
    print(f"    A_q: {A_q.shape}, {A_q.dtype}")
    
    # 查表
    algo_data, workspace = mgr.lookup_cublaslt(N, K, M)
    print(f"    查表结果: algo_data={'有' if algo_data else '无'}, workspace={workspace}")
    
    # 调用 GEMM
    print("\n    调用 cublaslt_fp8_mm...")
    try:
        output = ext.cublaslt_fp8_mm(W_q, A_q, "bf16", algo_data, workspace)
        print(f"    ✓ 成功! 输出形状: {output.shape}, dtype: {output.dtype}")
    except Exception as e:
        print(f"    ✗ 失败: {e}")
    
    # 5. 测试不带 algo_data
    print("\n[5] 测试不带 algo_data (使用默认启发式)...")
    try:
        output = ext.cublaslt_fp8_mm(W_q, A_q, "bf16", None, 0)
        print(f"    ✓ 成功! 输出形状: {output.shape}, dtype: {output.dtype}")
    except Exception as e:
        print(f"    ✗ 失败: {e}")
    
    # 6. 测试 Custom Op
    print("\n[6] 测试 Custom Op (torch.ops.slidesparse.cublaslt_fp8_mm)...")
    try:
        output = torch.ops.slidesparse.cublaslt_fp8_mm(W_q, A_q, "bf16")
        print(f"    ✓ 成功! 输出形状: {output.shape}, dtype: {output.dtype}")
    except Exception as e:
        print(f"    ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("调试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
