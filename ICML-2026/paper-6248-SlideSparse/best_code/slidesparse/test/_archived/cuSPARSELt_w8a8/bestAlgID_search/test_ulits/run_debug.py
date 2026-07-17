#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_debug.py - 运行调试脚本

使用方法:
  CUDA_VISIBLE_DEVICES=0 python3 run_debug.py
"""

import os
import sys
import ctypes
import ctypes.util

# === 1. 先加载 cuSPARSELt ===
print("[Step 0] Loading cuSPARSELt library...")
preferred_paths = [
    os.environ.get("CUSPARSELT_PATH"),
    "/usr/lib/x86_64-linux-gnu/libcusparseLt.so.0",
    "/usr/local/cuda/lib64/libcusparseLt.so.0",
    ctypes.util.find_library("cusparseLt"),
]
loaded = False
for path in preferred_paths:
    if not path:
        continue
    try:
        lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        getattr(lib, "cusparseLtMatmulAlgSelectionDestroy")
        print(f"  Loaded: {path}")
        loaded = True
        break
    except (OSError, AttributeError) as e:
        continue

if not loaded:
    print("  ERROR: Cannot find libcusparseLt.so")
    sys.exit(1)
print()

# === 2. 加载 PyTorch ===
import torch
if not torch.cuda.is_available():
    print("ERROR: CUDA not available")
    sys.exit(1)

prop = torch.cuda.get_device_properties(0)
print(f"GPU: {prop.name} (CC {prop.major}.{prop.minor})")
print()

# === 3. 编译并加载调试扩展 ===
print("[Step 1] Compiling debug extension...")
from torch.utils.cpp_extension import load
from pathlib import Path

src_path = Path(__file__).parent / "debug_verbose.cu"
build_dir = Path(__file__).parent / "build_debug_verbose"
build_dir.mkdir(parents=True, exist_ok=True)

sm_code = f"sm_{prop.major}{prop.minor}"

try:
    ext = load(
        name="debug_verbose",
        sources=[str(src_path)],
        extra_cuda_cflags=["-O0", "-g", f"-arch={sm_code}"],
        extra_ldflags=["-lcusparseLt"],
        verbose=False,
        build_directory=str(build_dir),
        with_cuda=True,
    )
    print("  OK")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)
print()

# === 4. 运行调试测试 ===
print("[Step 2] Running verbose debug test...")
print()
ext.test_search()
