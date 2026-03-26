"""
CausalPFN evaluation script for IHDP benchmark.
Outputs metrics in the format expected by the optimization pipeline.

Optimizations applied:
- S-learner (Ridge alpha=0.001) mu_0 and mu_1 appended as extra features
- num_neighbours=85 (tuned)
- Multi-seed bootstrap ensemble: 2 seeds x 3 subsamples x 92% of training data
- Temperature ensemble: fit once per subsample, estimate with 7 temperature values
  T=[0.3, 0.5, 0.7, 0.9, 2.0, 4.0, 8.0]
  Total: 2 seeds × 3 bootstraps × 7 T values = 42 predictions per realization
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
NUM_NEIGHBOURS = int(os.environ.get('NUM_NEIGHBOURS', 85))
N_BOOT = int(os.environ.get('N_BOOT', 3))
BOOT_FRAC = float(os.environ.get('BOOT_FRAC', 0.92))
_SEEDS_ENV = os.environ.get('SEEDS', f'{SEED},{SEED+1}')
SEEDS = [int(s) for s in _SEEDS_ENV.split(',')]
_TEMPS_ENV = os.environ.get('PRED_TEMPS', '0.3,0.5,0.7,0.9,2.0,4.0,8.0')
PRED_TEMPS = [float(t) for t in _TEMPS_ENV.split(',')]
ATE_TEMP = float(os.environ.get('ATE_TEMP', '1.4'))

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

from causalpfn import ATEEstimator, CATEEstimator
from causalpfn.evaluation import calculate_pehe
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

dataset = IHDPDataset(n_tables=N_TABLES, seed=SEED)

pehe_values = []
ate_rel_err_values = []


def add_s_learner_features(X_train, t_train, y_train, X_test, alpha=0.001):
    """Append S-learner mu_0 and mu_1 predictions as extra features."""
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_train)
    X_te_sc = sc.transform(X_test)
    Xt_tr = np.hstack([X_tr_sc, t_train.reshape(-1, 1)])
    ridge = Ridge(alpha=alpha)
    ridge.fit(Xt_tr, y_train)
    zeros_tr = np.zeros((len(X_tr_sc), 1))
    ones_tr = np.ones((len(X_tr_sc), 1))
    zeros_te = np.zeros((len(X_te_sc), 1))
    ones_te = np.ones((len(X_te_sc), 1))
    mu_tr0 = ridge.predict(np.hstack([X_tr_sc, zeros_tr])).reshape(-1, 1)
    mu_tr1 = ridge.predict(np.hstack([X_tr_sc, ones_tr])).reshape(-1, 1)
    mu_te0 = ridge.predict(np.hstack([X_te_sc, zeros_te])).reshape(-1, 1)
    mu_te1 = ridge.predict(np.hstack([X_te_sc, ones_te])).reshape(-1, 1)
    return np.hstack([X_train, mu_tr0, mu_tr1]), np.hstack([X_test, mu_te0, mu_te1])


for realization_idx in range(len(dataset)):
    cate_dset, ate_dset = dataset[realization_idx]

    X_train_aug, X_test_aug = add_s_learner_features(
        cate_dset.X_train, cate_dset.t_train, cate_dset.y_train, cate_dset.X_test
    )

    # ATE estimation
    X_ate_aug, _ = add_s_learner_features(ate_dset.X, ate_dset.t, ate_dset.y, ate_dset.X[:1])
    causalpfn_ate = ATEEstimator(
        device=str(device), model_path=MODEL_PATH, cache_dir=HF_CACHE, num_neighbours=NUM_NEIGHBOURS
    )
    causalpfn_ate.prediction_temperature = ATE_TEMP
    causalpfn_ate.fit(X_ate_aug, ate_dset.t, ate_dset.y)
    true_ate = ate_dset.true_ate
    causalpfn_ate_hat = causalpfn_ate.estimate_ate()
    rel_error = abs(causalpfn_ate_hat - true_ate) / abs(true_ate)
    ate_rel_err_values.append(rel_error)

    # CATE: multi-seed bootstrap + temperature ensemble
    n = len(X_train_aug)
    all_cate_preds = []
    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed + realization_idx + 100)
        for b in range(N_BOOT):
            idx = rng.choice(n, size=int(n * BOOT_FRAC), replace=False)
            causalpfn_cate = CATEEstimator(
                device=str(device), model_path=MODEL_PATH, cache_dir=HF_CACHE, num_neighbours=NUM_NEIGHBOURS
            )
            causalpfn_cate.prediction_temperature = PRED_TEMPS[0]
            causalpfn_cate.fit(X_train_aug[idx], cate_dset.t_train[idx], cate_dset.y_train[idx])
            for temp in PRED_TEMPS:
                causalpfn_cate.prediction_temperature = temp
                all_cate_preds.append(causalpfn_cate.estimate_cate(X_test_aug))

    causalpfn_cate_hat = np.mean(all_cate_preds, axis=0)
    pehe = calculate_pehe(cate_dset.true_cate, causalpfn_cate_hat)
    pehe_values.append(pehe)

    print(f"Realization {realization_idx+1}/{N_TABLES}: PEHE={pehe:.4f}, ATE_rel_err={rel_error:.4f}")

mean_pehe = np.mean(pehe_values)
std_pehe = np.std(pehe_values)
mean_ate_rel_err = np.mean(ate_rel_err_values)
std_ate_rel_err = np.std(ate_rel_err_values)

print()
print(f"=== CausalPFN IHDP Results ({N_TABLES} realizations) ===")
print(f"IHDP PEHE: {mean_pehe:.4f} +/- {std_pehe:.4f}")
print(f"IHDP ATE Relative Error: {mean_ate_rel_err:.4f} +/- {std_ate_rel_err:.4f}")
print(f"ihdp_pehe: {mean_pehe:.4f}")
print(f"ihdp_ate_rel_err: {mean_ate_rel_err:.4f}")
