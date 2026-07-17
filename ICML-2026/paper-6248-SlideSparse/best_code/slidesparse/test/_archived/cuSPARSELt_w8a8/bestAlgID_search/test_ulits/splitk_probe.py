#!/usr/bin/env python3
"""
splitk_probe.py - 探测 Split-K 行为

运行: CUDA_VISIBLE_DEVICES=0 python3 splitk_probe.py
"""

import os
import sys
import ctypes
import ctypes.util

# 1. 加载 cuSPARSELt
print("[0] Loading cuSPARSELt...")
for path in ["/usr/lib/x86_64-linux-gnu/libcusparseLt.so.0",
             "/usr/local/cuda/lib64/libcusparseLt.so.0",
             ctypes.util.find_library("cusparseLt")]:
    if not path: continue
    try:
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        print(f"    Loaded: {path}")
        break
    except: pass
else:
    print("    ERROR: Cannot find libcusparseLt.so")
    sys.exit(1)
print()

# 2. 编译扩展
import torch
from torch.utils.cpp_extension import load
from pathlib import Path

prop = torch.cuda.get_device_properties(0)
print(f"GPU: {prop.name} (CC {prop.major}.{prop.minor})")
print()

print("[1] Compiling probe extension...")
src = Path(__file__).parent / "splitk_probe.cu"
build = Path(__file__).parent / "build_probe"
build.mkdir(exist_ok=True)

ext = load(
    name="splitk_probe",
    sources=[str(src)],
    extra_cuda_cflags=["-O2", f"-arch=sm_{prop.major}{prop.minor}"],
    extra_ldflags=["-lcusparseLt"],
    verbose=False,
    build_directory=str(build),
    with_cuda=True,
)
print("    OK")
print()

# 3. 运行探测（使用不同的 MNK 配置）
print("[2] Running probe...")
print()

# 测试几个典型尺寸
test_configs = [
    (64, 64, 32),      # 最小对齐尺寸
    (2560, 2560, 128), # 中等尺寸
]

for N, K, M in test_configs:
    print(f"\n{'='*50}")
    print(f"Testing N={N}, K={K}, M={M}")
    print(f"{'='*50}\n")
    ext.probe_splitk(N, K, M)
