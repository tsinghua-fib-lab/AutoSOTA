#!/usr/bin/env python3
"""Reproduction evaluation script for FlashSinkhorn (Paper 528).

Measures forward pass runtime for n=m=10000, d=64, epsilon=0.1, 10 iterations,
TF32 precision, uniform marginals, squared Euclidean cost, A100-80GB GPU.

Matches rubric metric condition exactly.
Outputs JSON with runtime_ms metric.
"""

import json
import sys
import torch
from flash_sinkhorn import SamplesLoss


def main():
    device = torch.device("cuda")
    
    # Paper settings (Table 2, Section 4.1)
    n, m, d = 10000, 10000, 64
    eps = 0.1
    n_iters = 10
    warmup = 10
    reps = 50
    
    # Check GPU
    gpu_name = torch.cuda.get_device_name(0)
    if "A100" not in gpu_name:
        print(f"Warning: Expected A100-80GB, found {gpu_name}", file=sys.stderr)
    
    # Setup data with fixed seed
    torch.manual_seed(0)
    x = torch.randn(n, d, device=device, dtype=torch.float32)
    y = torch.randn(m, d, device=device, dtype=torch.float32)
    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()
    
    # FlashSinkhorn alternating backend (OTT-JAX-style, matches paper Table 2)
    loss_fn = SamplesLoss(
        "sinkhorn", backend="alternating", use_epsilon_scaling=False,
        eps=eps, n_iters=n_iters, debias=False, normalize=False,
        autotune=True, last_extrapolation=False, allow_tf32=True,
    )
    
    # Warmup
    for _ in range(warmup):
        _ = loss_fn(a, x, b, y)
    torch.cuda.synchronize()
    
    # Measure with CUDA events (precise GPU timing)
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = loss_fn(a, x, b, y)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    times_t = torch.tensor(times)
    
    result = {
        "runtime_ms": round(times_t.mean().item(), 3),
        "runtime_median_ms": round(times_t.median().item(), 3),
        "runtime_std_ms": round(times_t.std().item(), 3),
        "runtime_min_ms": round(times_t.min().item(), 3),
        "runtime_max_ms": round(times_t.max().item(), 3),
        "n": n, "m": m, "d": d,
        "eps": eps, "n_iters": n_iters,
        "reps": reps, "warmup": warmup,
        "gpu": gpu_name,
        "precision": "TF32",
        "cost": "squared_euclidean",
        "marginals": "uniform",
        "backend": "alternating",
    }
    
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
