"""
Try multiple propensity models as features. Quick experiment.
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def get_propensity_features(X_train, t_train, X_test):
    """Get multiple propensity scores from different models."""
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    features_train, features_test = [X_train], [X_test]

    # Logistic regression
    lr = LogisticRegression(max_iter=500, random_state=42, C=1.0)
    lr.fit(X_tr_sc, t_train)
    features_train.append(lr.predict_proba(X_tr_sc)[:, 1:2])
    features_test.append(lr.predict_proba(X_te_sc)[:, 1:2])

    # GBM propensity
    gbm = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    gbm.fit(X_tr_sc, t_train)
    features_train.append(gbm.predict_proba(X_tr_sc)[:, 1:2])
    features_test.append(gbm.predict_proba(X_te_sc)[:, 1:2])

    return np.hstack(features_train), np.hstack(features_test)


dataset = IHDPDataset(n_tables=N_TABLES, seed=SEED)

# Test different feature sets
configs = [
    ("lr_only", lambda X_tr, t, X_te: (
        np.hstack([X_tr, LogisticRegression(max_iter=500, random_state=42).fit(
            StandardScaler().fit_transform(X_tr), t
        ).predict_proba(StandardScaler().fit(X_tr).transform(X_tr))[:, 1:2]]),
        None  # will handle test separately below
    )),
]

# Simple test: just try LR + GBM propensity with T=0.8
print("Testing: LR + GBM propensity features")
pehe_values = []
for realization_idx in range(N_TABLES):
    cate_dset, _ = dataset[realization_idx]
    X_tr_aug, X_te_aug = get_propensity_features(cate_dset.X_train, cate_dset.t_train, cate_dset.X_test)

    causalpfn_cate = CATEEstimator(device=str(device), model_path=MODEL_PATH, cache_dir=HF_CACHE)
    causalpfn_cate.prediction_temperature = PRED_TEMP
    causalpfn_cate.fit(X_tr_aug, cate_dset.t_train, cate_dset.y_train)
    cate_hat = causalpfn_cate.estimate_cate(X_te_aug)
    pehe = calculate_pehe(cate_dset.true_cate, cate_hat)
    pehe_values.append(pehe)
    print(f"  R{realization_idx+1}: PEHE={pehe:.4f}")

print(f"Mean PEHE: {np.mean(pehe_values):.4f}")
