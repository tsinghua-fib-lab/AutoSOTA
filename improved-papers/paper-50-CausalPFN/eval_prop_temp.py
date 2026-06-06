"""
Temperature sweep with propensity score augmentation.
"""
import os, sys, warnings, importlib.util

PKGDIR = "/home/dataset-assist-0/AUTOSOTA/sota-5/auto-pipeline-ab/optimizer/papers/paper-1630/runs/run_20260325_014145/results/pypackages"
if os.path.exists(PKGDIR) and PKGDIR not in sys.path:
    sys.path.insert(0, PKGDIR)

warnings.filterwarnings('ignore')
sys.path.insert(0, '/repo')

import numpy as np
import torch
import random

SEED = 42
N_TABLES = 2
HF_CACHE = "/home/dataset-assist-0/AUTOSOTA/sota-5/auto-pipeline-ab/optimizer/papers/paper-1630/runs/run_20260325_014145/results/hf_cache"
MODEL_PATH = 'vdblm/causalpfn'

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def add_propensity_feature(X_train, t_train, X_test, C=1.0):
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=500, random_state=42, C=C)
    clf.fit(X_tr_sc, t_train)
    ps_train = clf.predict_proba(X_tr_sc)[:, 1].reshape(-1, 1)
    ps_test = clf.predict_proba(X_te_sc)[:, 1].reshape(-1, 1)
    return np.hstack([X_train, ps_train]), np.hstack([X_test, ps_test])

dataset = IHDPDataset(n_tables=N_TABLES, seed=SEED)
realizations = [dataset[i] for i in range(len(dataset))]

# Pre-compute augmented features
aug_data = []
for cate_dset, ate_dset in realizations:
    X_tr_aug, X_te_aug = add_propensity_feature(cate_dset.X_train, cate_dset.t_train, cate_dset.X_test)
    aug_data.append((cate_dset, X_tr_aug, X_te_aug))

temperatures = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]

print(f"\n=== Temperature Sweep (with propensity augmentation) ===")
print(f"{'Temp':>5} | {'PEHE_1':>8} | {'PEHE_2':>8} | {'Mean':>8}")

best_pehe = float('inf')
best_T = 1.0

for T in temperatures:
    pehe_values = []
    for (cate_dset, X_tr_aug, X_te_aug) in aug_data:
        causalpfn_cate = CATEEstimator(device=str(device), model_path=MODEL_PATH, cache_dir=HF_CACHE)
        causalpfn_cate.prediction_temperature = T
        causalpfn_cate.fit(X_tr_aug, cate_dset.t_train, cate_dset.y_train)
        pehe = calculate_pehe(cate_dset.true_cate, causalpfn_cate.estimate_cate(X_te_aug))
        pehe_values.append(pehe)

    mean_pehe = np.mean(pehe_values)
    print(f"T={T:.1f} | {pehe_values[0]:.4f} | {pehe_values[1]:.4f} | {mean_pehe:.4f}")
    if mean_pehe < best_pehe:
        best_pehe = mean_pehe
        best_T = T

print(f"\nBest T={best_T}: PEHE={best_pehe:.4f}")
