import sys
sys.path.insert(0, '/repo')
import torch
import numpy as np
import time
import os

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

# Measure GPU process memory via pynvml
try:
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    # Baseline before loading model
    base_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"Baseline GPU memory: {base_info.used / 1024**2:.2f} MB")
    
    model = Model(args).float().cuda()
    model.eval()
    
    # After model load
    model_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"After model load GPU memory: {model_info.used / 1024**2:.2f} MB")
    
    x = torch.randn(1, 720, 321).float().cuda()
    with torch.no_grad():
        out, _ = model(x)
    torch.cuda.synchronize()
    
    inf_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"After batch_size=1 inference: {inf_info.used / 1024**2:.2f} MB")
    
except ImportError:
    # Fallback: measure via /proc
    pid = os.getpid()
    model = Model(args).float().cuda()
    model.eval()
    x = torch.randn(1, 720, 321).float().cuda()
    with torch.no_grad():
        out, _ = model(x)
    print("pynvml not available, using /proc approach")
    # Read smaps
    with open(f'/proc/{pid}/smaps_rollup') as f:
        for line in f:
            if 'Rss' in line or 'Pss' in line:
                print(line.strip())
    
    # print allocated
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"Allocated: {alloc:.2f} MB, Reserved: {reserved:.2f} MB")

