#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_search.py - 调试 alg_search 卡住问题

逐步执行搜索流程，找出卡住的具体位置。
"""

import sys
import os
import time
import signal
import ctypes
import ctypes.util
from pathlib import Path

import torch

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入统一的库加载函数
from slidesparse.utils import ensure_cusparselt_loaded as _ensure_cusparselt_loaded

# === 超时信号处理 ===
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("操作超时!")

# === 加载 cuSPARSELt ===
def ensure_cusparselt_loaded():
    """封装顶层函数，返回 True/False 而不是 raise"""
    try:
        _ensure_cusparselt_loaded()
        print("  Loaded cuSPARSELt successfully")
        return True
    except OSError as e:
        print(f"  ERROR: {e}")
        return False

# === 加载扩展 ===
def load_ext():
    from torch.utils.cpp_extension import load
    
    src_path = Path(__file__).parent / "alg_search_cusparselt.cu"
    build_dir = Path(__file__).parent / "build_debug"
    build_dir.mkdir(parents=True, exist_ok=True)
    
    prop = torch.cuda.get_device_properties(0)
    sm_code = f"sm_{prop.major}{prop.minor}"
    
    ext = load(
        name="debug_ext",
        sources=[str(src_path)],
        extra_cuda_cflags=["-O0", "-g", f"-arch={sm_code}"],  # Debug build
        extra_ldflags=["-lcusparseLt", "-lnvrtc", "-ldl"],
        verbose=True,
        build_directory=str(build_dir),
        with_cuda=True,
    )
    return ext

def main():
    print("=" * 60)
    print("cuSPARSELt Search Debug Tool")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return 1
    
    prop = torch.cuda.get_device_properties(0)
    print(f"GPU: {prop.name} (CC {prop.major}.{prop.minor})")
    print()
    
    # Step 1: Load cuSPARSELt
    print("[Step 1] Loading cuSPARSELt library...")
    if not ensure_cusparselt_loaded():
        return 1
    print()
    
    # Step 2: Load extension
    print("[Step 2] Loading CUDA extension (this may take a while)...")
    try:
        ext = load_ext()
        print("  OK")
    except Exception as e:
        print(f"  FAILED: {e}")
        return 1
    print()
    
    # Step 3: Create test data
    print("[Step 3] Creating test data...")
    # 使用最小尺寸（满足对齐要求）
    N, K = 64, 64
    M_list = [32]
    
    W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    A = torch.randn(M_list[0], K, device="cuda", dtype=torch.bfloat16)
    print(f"  W: {W.shape}, A: {A.shape}")
    print()
    
    # Step 4: Test prune_24
    print("[Step 4] Testing prune_24...")
    sys.stdout.flush()
    
    # 设置超时
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # 30 秒超时
    
    try:
        t0 = time.time()
        W_pruned = ext.prune_24(W, "TNCCcol")
        t1 = time.time()
        print(f"  OK ({t1-t0:.2f}s)")
    except TimeoutError:
        print("  TIMEOUT (>30s)!")
        return 1
    except Exception as e:
        print(f"  FAILED: {e}")
        return 1
    finally:
        signal.alarm(0)  # 取消超时
    print()
    
    # Step 5: Test search_topk with minimal parameters
    print("[Step 5] Testing search_topk (minimal)...")
    print("  Parameters: warmup=1, repeat=1, verify=False")
    sys.stdout.flush()
    
    signal.alarm(60)  # 60 秒超时
    
    try:
        t0 = time.time()
        print(f"  Calling search_topk at {time.strftime('%H:%M:%S')}...")
        sys.stdout.flush()
        
        out = ext.search_topk(
            W_pruned,
            A,
            M_list,
            "TNCCcol",
            "int8",      # dtype
            1,           # warmup (最小)
            1,           # repeat (最小)
            False,       # verify
            [],          # blacklist
            1,           # topk
        )
        
        t1 = time.time()
        print(f"  OK ({t1-t0:.2f}s)")
        print(f"  Result keys: {list(out.keys())}")
        
        # 打印关键结果
        print(f"  compress_alg_id: {out['compress_alg_id']}")
        print(f"  topk_alg_id: {out['topk_alg_id']}")
        print(f"  topk_split_k: {out['topk_split_k']}")
        print(f"  valid_mask: {out['valid_mask']}")
        
    except TimeoutError:
        print(f"  TIMEOUT (>60s) at {time.strftime('%H:%M:%S')}!")
        print("  >>> 问题定位：search_topk 内部卡住了")
        return 1
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        signal.alarm(0)
    print()
    
    # Step 6: Test with fp8e4m3
    print("[Step 6] Testing search_topk with fp8e4m3...")
    sys.stdout.flush()
    
    signal.alarm(60)
    
    try:
        t0 = time.time()
        print(f"  Calling search_topk at {time.strftime('%H:%M:%S')}...")
        sys.stdout.flush()
        
        out = ext.search_topk(
            W_pruned,
            A,
            M_list,
            "TNCCcol",
            "fp8e4m3",   # dtype
            1,           # warmup
            1,           # repeat
            False,       # verify
            [],          # blacklist
            1,           # topk
        )
        
        t1 = time.time()
        print(f"  OK ({t1-t0:.2f}s)")
        print(f"  compress_alg_id: {out['compress_alg_id']}")
        print(f"  topk_alg_id: {out['topk_alg_id']}")
        print(f"  topk_split_k: {out['topk_split_k']}")
        
    except TimeoutError:
        print(f"  TIMEOUT (>60s)!")
        return 1
    except Exception as e:
        print(f"  FAILED: {e}")
        # FP8 可能不被支持，这是预期的
        print("  (FP8 may not be supported on this GPU)")
    finally:
        signal.alarm(0)
    print()
    
    print("=" * 60)
    print("Debug completed successfully!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
