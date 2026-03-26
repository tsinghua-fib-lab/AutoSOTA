"""Bootstrap ensemble finer grid around frac=0.9."""
import os, sys, warnings, importlib.util

PKGDIR = "/home/dataset-assist-0/AUTOSOTA/sota-5/auto-pipeline-ab/optimizer/papers/paper-1630/runs/run_20260325_014145/results/pypackages"
if os.path.exists(PKGDIR) and PKGDIR not in sys.path:
    sys.path.insert(0, PKGDIR)

warnings.filterwarnings('ignore')
sys.path.insert(0, '/repo')

import numpy as np
import torch
import random

SEED = 42; N_TABLES = 2
HF_CACHE = "/home/dataset-assist-0/AUTOSOTA/sota-5/auto-pipeline-ab/optimizer/papers/paper-1630/runs/run_20260325_014145/results/hf_cache"
MODEL_PATH = 'vdblm/causalpfn'; PRED_TEMP = 0.8

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

def add_propensity_feature(X_train, t_train, X_test):
    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=500, random_state=42, C=1.0)
    clf.fit(scaler.fit_transform(X_train), t_train)
    ps_tr = clf.predict_proba(scaler.transform(X_train))[:, 1:2]
    ps_te = clf.predict_proba(scaler.transform(X_test))[:, 1:2]
    return np.hstack([X_train, ps_tr]), np.hstack([X_test, ps_te])

def run_bootstrap(dataset, n_boot, frac, stratified=False):
    pehe_values = []
    for realization_idx in range(N_TABLES):
        cate_dset, _ = dataset[realization_idx]
        X_tr_aug, X_te_aug = add_propensity_feature(cate_dset.X_train, cate_dset.t_train, cate_dset.X_test)
        rng = np.random.RandomState(SEED + realization_idx + 100)
        all_preds = []
        n = len(X_tr_aug)

        for b in range(n_boot):
            if stratified:
                # Stratified sampling: maintain treatment/control ratio
                treat_idx = np.where(cate_dset.t_train == 1)[0]
                ctrl_idx = np.where(cate_dset.t_train == 0)[0]
                n_treat = int(len(treat_idx) * frac)
                n_ctrl = int(len(ctrl_idx) * frac)
                sel_treat = rng.choice(treat_idx, n_treat, replace=False)
                sel_ctrl = rng.choice(ctrl_idx, n_ctrl, replace=False)
                idx = np.concatenate([sel_treat, sel_ctrl])
            else:
                idx = rng.choice(n, size=int(n * frac), replace=False)

            causalpfn_cate = CATEEstimator(device=str(device), model_path=MODEL_PATH, cache_dir=HF_CACHE)
            causalpfn_cate.prediction_temperature = PRED_TEMP
            causalpfn_cate.fit(X_tr_aug[idx], cate_dset.t_train[idx], cate_dset.y_train[idx])
            all_preds.append(causalpfn_cate.estimate_cate(X_te_aug))

        pehe_values.append(calculate_pehe(cate_dset.true_cate, np.mean(all_preds, axis=0)))
    return np.mean(pehe_values), pehe_values

dataset = IHDPDataset(n_tables=N_TABLES, seed=SEED)

print(f"{'Config':>20} | {'R1':>8} | {'R2':>8} | {'Mean':>8}")
configs = [
    (7, 0.9, False),
    (10, 0.9, False),
    (15, 0.9, False),
    (5, 0.85, False),
    (5, 0.95, False),
    (7, 0.85, False),
    (7, 0.95, False),
    (5, 0.9, True),   # stratified
    (7, 0.9, True),   # stratified
]

for n_boot, frac, strat in configs:
    mean_p, pv = run_bootstrap(dataset, n_boot, frac, strat)
    label = f"n={n_boot},f={frac:.2f}" + (",str" if strat else "")
    print(f"{label:>20} | {pv[0]:.4f} | {pv[1]:.4f} | {mean_p:.4f}")
