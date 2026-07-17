#!/usr/bin/env python3
"""
Simple CUTLASS FP8 GEMM Benchmark

This script benchmarks CUTLASS scaled_mm directly through vLLM's API.
No dependencies on our cuBLASLt wrapper - just pure CUTLASS performance.

Usage:
    python benchmark_cutlass_only.py
"""

import os
import sys
import time
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def benchmark_cutlass_fp8(m: int, n: int, k: int, 
                          warmup: int = 10, 
                          iterations: int = 100,
                          output_dtype: torch.dtype = torch.bfloat16) -> dict:
    """
    Benchmark CUTLASS FP8 GEMM via vLLM's cutlass_scaled_mm
    
    Args:
        m, n, k: Matrix dimensions (A: m×k, B: k×n, C: m×n)
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations
        output_dtype: Output data type
        
    Returns:
        Dictionary with timing and performance metrics
    """
    from vllm._custom_ops import cutlass_scaled_mm
    
    device = torch.device('cuda')
    
    # CUTLASS scaled_mm expects:
    # - A: (M, K) row-major, contiguous
    # - B: (K, N) column-major (i.e., b.stride(0) == 1, b.stride(1) == K)
    # 
    # To create column-major B:
    # 1. Create (N, K) tensor
    # 2. Transpose to get (K, N) view with stride(0)=1, stride(1)=K
    
    # A: (M, K) row-major
    a_fp16 = torch.randn(m, k, dtype=torch.float16, device=device)
    a_fp8 = a_fp16.to(torch.float8_e4m3fn)
    
    # B: Create (N, K) then transpose to get column-major (K, N)
    # This gives stride(0)=1 (column-major)
    b_fp16 = torch.randn(n, k, dtype=torch.float16, device=device)
    b_fp8 = b_fp16.to(torch.float8_e4m3fn).t()  # Now (K, N) column-major
    
    # Verify layout
    assert a_fp8.shape == (m, k), f"A shape mismatch: {a_fp8.shape}"
    assert b_fp8.shape == (k, n), f"B shape mismatch: {b_fp8.shape}"
    assert b_fp8.stride(0) == 1, f"B must be column-major, got stride {b_fp8.stride()}"
    
    # Scale factors
    scale_a = torch.ones(1, dtype=torch.float32, device=device)
    scale_b = torch.ones(1, dtype=torch.float32, device=device)
    
    # Warmup
    for _ in range(warmup):
        _ = cutlass_scaled_mm(a_fp8, b_fp8, scale_a, scale_b, output_dtype)
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = cutlass_scaled_mm(a_fp8, b_fp8, scale_a, scale_b, output_dtype)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    # Calculate metrics
    time_ms = sum(times) / len(times)
    time_std = (sum((t - time_ms) ** 2 for t in times) / len(times)) ** 0.5
    
    flops = 2 * m * n * k  # multiply-add = 2 FLOPs
    tflops = flops / (time_ms * 1e9)
    
    # Memory bandwidth (FP8 = 1 byte each element)
    bytes_moved = (m * k + n * k + m * n) * 1  # A + B + C
    memory_gb_s = bytes_moved / (time_ms * 1e6)
    
    return {
        'm': m, 'n': n, 'k': k,
        'time_ms': time_ms,
        'time_std_ms': time_std,
        'tflops': tflops,
        'memory_gb_s': memory_gb_s,
        'flops': flops,
        'output_dtype': str(output_dtype),
    }


def main():
    print("=" * 80)
    print("CUTLASS FP8 GEMM Benchmark (vLLM's cutlass_scaled_mm)")
    print("=" * 80)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_props = torch.cuda.get_device_properties(0)
    
    print(f"GPU: {gpu_name}")
    print(f"Compute Capability: {gpu_props.major}.{gpu_props.minor}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"SM Count: {gpu_props.multi_processor_count}")
    
    # Theoretical peak (estimate for FP8 Tensor Cores)
    # RTX 5080 FP8: ~900 TFLOPS (estimated)
    # H100 FP8: ~3958 TFLOPS
    print()
    
    # Test configurations
    # Grouped by use case
    test_configs = [
        # === Small Language Models (Qwen2.5-0.5B) ===
        # hidden_size=896, intermediate_size=4864, num_heads=14
        ("Qwen-0.5B single token", 1, 4864, 896),
        ("Qwen-0.5B batch=32", 32, 4864, 896),
        ("Qwen-0.5B batch=128", 128, 4864, 896),
        ("Qwen-0.5B batch=512", 512, 4864, 896),
        
        # === Medium Language Models (Llama3.2-1B) ===
        # hidden_size=2048, intermediate_size=8192, num_heads=32
        ("Llama-1B single token", 1, 8192, 2048),
        ("Llama-1B batch=32", 32, 8192, 2048),
        ("Llama-1B batch=128", 128, 8192, 2048),
        ("Llama-1B batch=512", 512, 8192, 2048),
        
        # === Square matrices (standard GEMM benchmark) ===
        ("Square 1K", 1024, 1024, 1024),
        ("Square 2K", 2048, 2048, 2048),
        ("Square 4K", 4096, 4096, 4096),
        ("Square 8K", 8192, 8192, 8192),
    ]
    
    # Run benchmarks
    results = []
    
    print("\n" + "-" * 100)
    print(f"{'Name':<25} {'M':>8} {'N':>8} {'K':>8} {'Time(ms)':>12} {'TFLOPS':>10} {'GB/s':>10}")
    print("-" * 100)
    
    for name, m, n, k in test_configs:
        try:
            result = benchmark_cutlass_fp8(m, n, k, warmup=20, iterations=100)
            results.append((name, result))
            
            print(f"{name:<25} {m:>8} {n:>8} {k:>8} "
                  f"{result['time_ms']:>12.4f} {result['tflops']:>10.2f} {result['memory_gb_s']:>10.2f}")
        except Exception as e:
            print(f"{name:<25} {m:>8} {n:>8} {k:>8} ERROR: {e}")
    
    print("-" * 100)
    
    # Summary statistics
    if results:
        print("\n" + "=" * 80)
        print("Summary Statistics")
        print("=" * 80)
        
        # Find peak TFLOPS
        peak_result = max(results, key=lambda x: x[1]['tflops'])
        print(f"Peak TFLOPS: {peak_result[1]['tflops']:.2f} ({peak_result[0]})")
        
        # Average TFLOPS for LLM workloads
        llm_results = [r for name, r in results if 'Qwen' in name or 'Llama' in name]
        if llm_results:
            avg_tflops = sum(r['tflops'] for r in llm_results) / len(llm_results)
            print(f"Average TFLOPS (LLM shapes): {avg_tflops:.2f}")
        
        # Square matrix results
        square_results = [r for name, r in results if 'Square' in name]
        if square_results:
            avg_tflops = sum(r['tflops'] for r in square_results) / len(square_results)
            print(f"Average TFLOPS (Square matrices): {avg_tflops:.2f}")
    
    print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    main()
