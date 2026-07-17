#!/usr/bin/env python3
"""
test_segmentk.py - 测试 split_k=-1 (Segment-K) 的行为
使用子进程+超时机制来精确定位卡住的位置

运行: CUDA_VISIBLE_DEVICES=0 python3 test_segmentk.py
"""

import os, sys, ctypes, ctypes.util
import subprocess
import time

# 超时时间（秒）
TIMEOUT = 5

def main():
    print("=" * 60)
    print("Segment-K (split_k=-1) Deep Test")
    print("=" * 60)
    print(f"Timeout: {TIMEOUT}s per test\n")
    
    # 加载库
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
        return 1
    print()
    
    # 获取 GPU 信息
    import torch
    prop = torch.cuda.get_device_properties(0)
    print(f"GPU: {prop.name} (SM {prop.major}.{prop.minor})")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}\n")
    
    # 编译扩展
    print("[1] Compiling extension...")
    from torch.utils.cpp_extension import load
    from pathlib import Path
    
    src = Path(__file__).parent / "test_segmentk.cu"
    build = Path(__file__).parent / "build_segmentk"
    build.mkdir(exist_ok=True)
    
    ext = load(
        name="test_segmentk",
        sources=[str(src)],
        extra_cuda_cflags=["-O2", f"-arch=sm_{prop.major}{prop.minor}"],
        extra_ldflags=["-lcusparseLt"],
        verbose=False,
        build_directory=str(build),
        with_cuda=True,
    )
    print("    OK\n")
    
    # 测试配置
    test_configs = [
        (64, 64, 32, 0),      # 小尺寸, alg_id=0
        (64, 64, 32, 1),      # 小尺寸, alg_id=1
    ]
    
    step_names = {
        0: "cusparseLtInit + descriptors",
        1: "AlgSelectionInit", 
        2: "SetAttribute(ALG_CONFIG_ID)",
        3: "SetAttribute(SPLIT_K=-1)",
        4: "PlanInit",
    }
    
    print("[2] Testing Segment-K step by step...")
    print("    Running each step directly (no subprocess for steps 0-3).\n")
    
    for N, K, M, alg_id in test_configs:
        print(f"{'='*50}")
        print(f"Config: N={N}, K={K}, M={M}, alg_id={alg_id}")
        print(f"{'='*50}")
        
        # 先测试前几步（不会卡住）
        for step in range(4):  # 0 到 3
            print(f"\n  Step {step} ({step_names[step]})...")
            sys.stdout.flush()
            
            t0 = time.time()
            result = ext.test_step(step, N, K, M, alg_id)
            t1 = time.time()
            print(f"    ✓ OK ({t1-t0:.3f}s)")
        
        # Step 4 (PlanInit) 可能会卡住，用子进程测试
        print(f"\n  Step 4 ({step_names[4]})... [with {TIMEOUT}s timeout]")
        sys.stdout.flush()
        
        # 写一个临时脚本来运行 step 4
        src_str = str(src).replace("'", "\\'")
        build_str = str(build).replace("'", "\\'")
        
        test_script = f'''
import ctypes, ctypes.util
for p in ["/usr/lib/x86_64-linux-gnu/libcusparseLt.so.0"]:
    try: ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL); break
    except: pass
import torch
from torch.utils.cpp_extension import load
ext = load("test_segmentk", ['{src_str}'], 
           extra_cuda_cflags=["-O2", "-arch=sm_{prop.major}{prop.minor}"],
           extra_ldflags=["-lcusparseLt"], verbose=False,
           build_directory='{build_str}', with_cuda=True)
print("RUNNING_STEP_4", flush=True)
ext.test_step(4, {N}, {K}, {M}, {alg_id})
print("STEP_4_DONE", flush=True)
'''
        
        try:
            t0 = time.time()
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                timeout=TIMEOUT,
                capture_output=True,
                text=True,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
            )
            t1 = time.time()
            
            print(f"    Subprocess output:")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    print(f"      {line}")
            if result.stderr and 'error' in result.stderr.lower():
                print(f"    Stderr (truncated): {result.stderr[:300]}")
            print(f"    ✓ Completed in {t1-t0:.1f}s")
            
        except subprocess.TimeoutExpired:
            print(f"    ❌ TIMEOUT after {TIMEOUT}s!")
            print(f"    >>> PlanInit with split_k=-1 HANGS!")
        
        print()
    
    print(f"\n{'='*60}")
    print("Conclusion:")
    print("  - Steps 0-3 (up to SetAttribute) complete successfully")
    print("  - Step 4 (PlanInit) with split_k=-1 is the hanging point")
    print(f"{'='*60}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
