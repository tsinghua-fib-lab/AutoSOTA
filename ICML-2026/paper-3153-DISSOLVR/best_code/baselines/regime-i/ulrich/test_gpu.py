#!/usr/bin/env python3
"""
Quick GPU test script to verify TensorFlow can use GPU
Run this before ulrich_5seeds.py to diagnose GPU issues
"""

import tensorflow as tf
import numpy as np
import time

print("=" * 60)
print("TensorFlow GPU Test")
print("=" * 60)

# Check TensorFlow version
print(f"\nTensorFlow version: {tf.__version__}")

# Check if built with CUDA
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

# List all devices
print("\nAll devices:")
for device in tf.config.list_physical_devices():
    print(f"  - {device.device_type}: {device.name}")

# Check GPU specifically
gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPU devices found: {len(gpus)}")
if gpus:
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
        try:
            details = tf.config.experimental.get_device_details(gpu)
            print(f"    Details: {details}")
        except:
            print(f"    (Details not available)")
else:
    print("  ✗ NO GPU FOUND!")
    print("\nPossible issues:")
    print("  1. CUDA/cuDNN not installed")
    print("  2. Wrong TensorFlow version (need tensorflow-gpu or tensorflow>=2.0)")
    print("  3. CUDA_VISIBLE_DEVICES not set correctly")
    print("\nTo fix:")
    print("  pip install tensorflow[and-cuda]  # For TF 2.15+")
    print("  or")
    print("  pip install tensorflow-gpu  # For older TF versions")
    exit(1)

# Try to run a computation on GPU
print("\n" + "=" * 60)
print("Testing GPU computation...")
print("=" * 60)

# Create test tensors
size = 5000
print(f"\nCreating {size}x{size} matrices...")

with tf.device('/GPU:0'):
    a = tf.random.normal([size, size])
    b = tf.random.normal([size, size])

    print("Running matrix multiplication on GPU...")
    start = time.time()
    c = tf.matmul(a, b)
    gpu_time = time.time() - start
    print(f"  GPU time: {gpu_time:.4f} seconds")

with tf.device('/CPU:0'):
    a_cpu = tf.random.normal([size, size])
    b_cpu = tf.random.normal([size, size])

    print("Running matrix multiplication on CPU...")
    start = time.time()
    c_cpu = tf.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - start
    print(f"  CPU time: {cpu_time:.4f} seconds")

speedup = cpu_time / gpu_time
print(f"\nGPU speedup: {speedup:.2f}x faster than CPU")

if speedup > 1.5:
    print("✓ GPU is working correctly!")
else:
    print("⚠ GPU might not be working properly (speedup < 1.5x)")

print("\n" + "=" * 60)
print("GPU test complete!")
print("=" * 60)
