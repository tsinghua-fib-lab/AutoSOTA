import sys
sys.path.insert(0, '/repo')
import torch
import numpy as np
import time
import argparse

# Create args matching the 720 horizon experiment
class Args:
    seq_len = 720
    pred_len = 720
    period_len = 24
    basis_num = 6
    enc_in = 321
    use_period_norm = 1
    use_orthogonal = 1
    features = 'M'
    individual = 0

args = Args()

from models.TimeBase import Model
model = Model(args).float()
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params} ({total_params/1000:.3f}K)")

# MACs
from ptflops import get_model_complexity_info
with torch.cuda.device(0):
    macs, params = get_model_complexity_info(model.cuda(), (720, 321), as_strings=True, print_per_layer_stat=False)
    print(f"MACs: {macs}")
    print(f"Params: {params}")

# Inference memory on GPU
model_gpu = model.cuda()
model_gpu.eval()
torch.cuda.reset_peak_memory_stats()
x = torch.randn(1, 720, 321).float().cuda()
with torch.no_grad():
    for _ in range(10):
        out, _ = model_gpu(x)
max_mem = torch.cuda.max_memory_allocated() / 1024 ** 2
print(f"Max Memory (MB) at inference: {max_mem:.4f}")

# CPU inference time
model_cpu = model.cpu()
model_cpu.eval()
x_cpu = torch.randn(1, 720, 321).float()
# warmup
with torch.no_grad():
    for _ in range(10):
        out, _ = model_cpu(x_cpu)

# Timing
times = []
with torch.no_grad():
    for _ in range(100):
        t0 = time.perf_counter()
        out, _ = model_cpu(x_cpu)
        t1 = time.perf_counter()
        times.append((t1-t0)*1000)  # ms
print(f"Infer Time CPU (ms): {np.mean(times):.4f} (std={np.std(times):.4f})")
print(f"Infer Time CPU median (ms): {np.median(times):.4f}")

