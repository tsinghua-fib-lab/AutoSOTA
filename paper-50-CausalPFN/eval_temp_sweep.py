"""
Temperature sweep experiment for CausalPFN IHDP.
Tries multiple prediction_temperature values to find optimal.
"""
import os
import sys
import warnings
import importlib.util

PKGDIR = "/home/dataset-assist-0/AUTOSOTA/sota-5/auto-pipeline-ab/optimizer/papers/paper-1630/runs/run_20260325_014145/results/pypackages"
if os.path.exists(PKGDIR) and PKGDIR not in sys.path:
    sys.path.insert(0, PKGDIR)

warnings.filterwarnings('ignore')
sys.path.insert(0, '/repo')

import numpy as np
import torch
import random

N_TABLES = int(os.environ.get('N_TABLES', 2))
SEED = int(os.environ.get('SEED', 42))
HF_CACHE = "/home/dataset-assist-0/AUTOSOTA/sota-5/auto-pipeline-ab/optimizer/papers/paper-1630/runs/run_20260325_014145/results/hf_cache"
MODEL_PATH = os.environ.get('MODEL_PATH', 'vdblm/causalpfn')

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_module_direct(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

base_mod = load_module_direct("benchmarks.base", "/repo/benchmarks/base.py")
ihdp_mod = load_module_direct("benchmarks.ihdp", "/repo/benchmarks/ihdp.py")
IHDPDataset = ihdp_mod.IHDPDataset

from causalpfn import CATEEstimator
from causalpfn.evaluation import calculate_pehe

temperatures = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]

dataset = IHDPDataset(n_tables=N_TABLES, seed=SEED)

print(f"\n=== Temperature Sweep ===")
print(f"{'Temp':>6} | {'PEHE_1':>8} | {'PEHE_2':>8} | {'Mean_PEHE':>10}")
print("-" * 40)

# Pre-load datasets
realizations = [dataset[i] for i in range(len(dataset))]

best_temp = 1.0
best_pehe = float('inf')

for T in temperatures:
    pehe_values = []
    for realization_idx, (cate_dset, ate_dset) in enumerate(realizations):
        causalpfn_cate = CATEEstimator(device=str(device), model_path=MODEL_PATH, cache_dir=HF_CACHE)
        causalpfn_cate.prediction_temperature = T
        causalpfn_cate.fit(cate_dset.X_train, cate_dset.t_train, cate_dset.y_train)
        causalpfn_cate_hat = causalpfn_cate.estimate_cate(cate_dset.X_test)
        pehe = calculate_pehe(cate_dset.true_cate, causalpfn_cate_hat)
        pehe_values.append(pehe)

    mean_pehe = np.mean(pehe_values)
    pehe_str = " | ".join(f"{p:.4f}" for p in pehe_values)
    print(f"T={T:>4.1f} | {pehe_str} | {mean_pehe:.4f}")

    if mean_pehe < best_pehe:
        best_pehe = mean_pehe
        best_temp = T

print(f"\nBest temperature: {best_temp} → PEHE={best_pehe:.4f}")
print(f"ihdp_pehe: {best_pehe:.4f}")
