#!/usr/bin/env python3
"""
splitk_deep_probe.py - 深度探测 Split-K 行为

运行: CUDA_VISIBLE_DEVICES=0 python3 splitk_deep_probe.py
"""

import os, sys, ctypes, ctypes.util

# 加载 cuSPARSELt
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
    print("    ERROR: Cannot find libcusparseLt.so"); sys.exit(1)

# 检查 cuSPARSELt 版本
try:
    import subprocess
    result = subprocess.run(["dpkg", "-l", "libcusparselt0"], capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if 'libcusparselt' in line:
            print(f"    Package: {line.strip()}")
except: pass
print()

import torch
from torch.utils.cpp_extension import load
from pathlib import Path

prop = torch.cuda.get_device_properties(0)
print(f"GPU: {prop.name} (CC {prop.major}.{prop.minor})")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print()

print("[1] Compiling deep probe extension...")
src = Path(__file__).parent / "splitk_deep_probe.cu"
build = Path(__file__).parent / "build_deep_probe"
build.mkdir(exist_ok=True)

ext = load(
    name="splitk_deep_probe",
    sources=[str(src)],
    extra_cuda_cflags=["-O2", f"-arch=sm_{prop.major}{prop.minor}"],
    extra_ldflags=["-lcusparseLt"],
    verbose=False,
    build_directory=str(build),
    with_cuda=True,
)
print("    OK\n")

# 测试两个尺寸
print("[2] Running deep probe...\n")

# 小尺寸
ext.deep_probe(64, 64, 32)

print("\n" + "="*64 + "\n")

# 较大尺寸
ext.deep_probe(2560, 2560, 128)
