import sys
sys.path.insert(0, '/repo')
import torch
import numpy as np
import time

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
model = Model(args).float().cuda()
model.eval()

# Reset stats
torch.cuda.reset_peak_memory_stats(0)

# Single sample inference
x = torch.randn(1, 720, 321).float().cuda()
with torch.no_grad():
    out, _ = model(x)

peak_alloc = torch.cuda.max_memory_allocated(0) / 1024**2
peak_reserved = torch.cuda.max_memory_reserved(0) / 1024**2
print(f"batch_size=1: max_allocated={peak_alloc:.4f} MB, max_reserved={peak_reserved:.4f} MB")

# Also check batch_size=128
torch.cuda.reset_peak_memory_stats(0)
x128 = torch.randn(128, 720, 321).float().cuda()
with torch.no_grad():
    out128, _ = model(x128)
peak_alloc128 = torch.cuda.max_memory_allocated(0) / 1024**2
peak_reserved128 = torch.cuda.max_memory_reserved(0) / 1024**2
print(f"batch_size=128: max_allocated={peak_alloc128:.4f} MB, max_reserved={peak_reserved128:.4f} MB")

# CPU timing with batch=1
model_cpu = model.cpu()
model_cpu.eval()
x_cpu = torch.randn(1, 720, 321).float()
with torch.no_grad():
    for _ in range(10):
        _, _ = model_cpu(x_cpu)
times = []
with torch.no_grad():
    for _ in range(200):
        t0 = time.perf_counter()
        _, _ = model_cpu(x_cpu)
        t1 = time.perf_counter()
        times.append((t1-t0)*1000)
print(f"CPU infer time: mean={np.mean(times):.4f} ms, median={np.median(times):.4f} ms")
