import os, sys, time
import torch
import numpy as np
import json

sys.path.insert(0, '/repo')
from infer import load_ckpt, compute_ksmi_mean

DIM = 100
T_SEQ = 200
SLICE_DIM = 5
BATCH_SIZE = 100
DELTA_T = 2
N_RUNS = 5

ckpt_path = '/models/infoatlas_maxdim5_step432500.ckpt'
module, model, cfg, device = load_ckpt(ckpt_path, device='cuda', verbose=False)

torch.manual_seed(42)
np.random.seed(42)
st = torch.randn(T_SEQ, DIM, dtype=torch.float32)
noise = torch.randn(T_SEQ, DIM, dtype=torch.float32) * 0.3
st_delta = 0.85 * torch.roll(st, shifts=DELTA_T, dims=0) + noise
st_delta[:DELTA_T] = st[:DELTA_T]

results = {}
for slices in [10, 15, 20, 25]:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    # Warmup
    for _ in range(3):
        _ = compute_ksmi_mean(
            st.numpy(), st_delta.numpy(),
            projection_dim=SLICE_DIM, model=model, proj_num=slices,
            batchsize=min(BATCH_SIZE, slices), max_dim=cfg.input_dim_x,
            softrank_reg=cfg.softrank_reg, normalize_input=True, device=device,
        )
    
    times = []
    mi_vals = []
    for run in range(N_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        mi_est = compute_ksmi_mean(
            st.numpy(), st_delta.numpy(),
            projection_dim=SLICE_DIM, model=model, proj_num=slices,
            batchsize=min(BATCH_SIZE, slices), max_dim=cfg.input_dim_x,
            softrank_reg=cfg.softrank_reg, normalize_input=True, device=device,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        mi_vals.append(mi_est)
    
    results[slices] = {
        'time_avg': float(np.mean(times)),
        'time_std': float(np.std(times)),
        'time_min': float(min(times)),
        'mi_avg': float(np.mean(mi_vals)),
        'mi_std': float(np.std(mi_vals)),
    }
    print(f'slices={slices:2d}: Time={results[slices]["time_avg"]:.4f}s +/- {results[slices]["time_std"]:.4f}s, MI={results[slices]["mi_avg"]:.4f} +/- {results[slices]["mi_std"]:.4f}')

with open('/repo/outputs/slice_sweep.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSlice sweep saved to /repo/outputs/slice_sweep.json')
