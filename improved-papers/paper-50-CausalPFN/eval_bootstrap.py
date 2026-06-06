"""
Bootstrap ensemble experiment: use random 80% subsets of training context,
average CATE predictions. Combined with propensity feature + T=0.8.
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
PRED_TEMP = 0.8

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

print("=== Bootstrap ensemble experiment ===")
print(f"{'N_boot':>6} | {'frac':>5} | {'R1_PEHE':>8} | {'R2_PEHE':>8} | {'Mean':>8}")

for n_boot, frac in [(1, 1.0), (3, 1.0), (5, 1.0), (10, 1.0), (5, 0.8), (5, 0.9), (10, 0.8)]:
    pehe_values = []
    for realization_idx in range(N_TABLES):
        cate_dset, _ = dataset[realization_idx]
        X_tr_aug, X_te_aug = add_propensity_feature(cate_dset.X_train, cate_dset.t_train, cate_dset.X_test)

        rng = np.random.RandomState(SEED + realization_idx + 100)
        all_preds = []

        for b in range(n_boot):
            if frac < 1.0:
                # Bootstrap subsample
                n = len(X_tr_aug)
                idx = rng.choice(n, size=int(n * frac), replace=False)
                X_b = X_tr_aug[idx]
                t_b = cate_dset.t_train[idx]
                y_b = cate_dset.y_train[idx]
            else:
                X_b = X_tr_aug
                t_b = cate_dset.t_train
                y_b = cate_dset.y_train

            causalpfn_cate = CATEEstimator(device=str(device), model_path=MODEL_PATH, cache_dir=HF_CACHE)
            causalpfn_cate.prediction_temperature = PRED_TEMP
            causalpfn_cate.fit(X_b, t_b, y_b)
            pred = causalpfn_cate.estimate_cate(X_te_aug)
            all_preds.append(pred)

        cate_hat = np.mean(all_preds, axis=0)
        pehe = calculate_pehe(cate_dset.true_cate, cate_hat)
        pehe_values.append(pehe)

    mean_pehe = np.mean(pehe_values)
    print(f"{n_boot:>6} | {frac:>5.1f} | {pehe_values[0]:.4f} | {pehe_values[1]:.4f} | {mean_pehe:.4f}")
