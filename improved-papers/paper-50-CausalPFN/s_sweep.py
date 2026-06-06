import os, sys, warnings, importlib.util
PKGDIR = "/home/dataset-assist-0/AUTOSOTA/sota-5/auto-pipeline-ab/optimizer/papers/paper-1630/runs/run_20260325_014145/results/pypackages"
sys.path.insert(0, PKGDIR)
HF_CACHE = "/home/dataset-assist-0/AUTOSOTA/sota-5/auto-pipeline-ab/optimizer/papers/paper-1630/runs/run_20260325_014145/results/hf_cache"
warnings.filterwarnings('ignore')
sys.path.insert(0, '/repo')

import numpy as np, torch, random
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
random.seed(42); np.random.seed(42); torch.manual_seed(42)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_module_direct(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod; spec.loader.exec_module(mod); return mod

base_mod = load_module_direct("benchmarks.base", "/repo/benchmarks/base.py")
ihdp_mod = load_module_direct("benchmarks.ihdp", "/repo/benchmarks/ihdp.py")
IHDPDataset = ihdp_mod.IHDPDataset

from causalpfn import CATEEstimator
from causalpfn.evaluation import calculate_pehe

def add_s_features(X_tr, t_tr, y_tr, X_te, alpha=1.0):
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)
    t_col = t_tr.reshape(-1, 1)
    Xt_tr = np.hstack([X_tr_sc, t_col])
    ridge = Ridge(alpha=alpha)
    ridge.fit(Xt_tr, y_tr)
    zeros_tr = np.zeros((len(X_tr_sc), 1))
    ones_tr = np.ones((len(X_tr_sc), 1))
    zeros_te = np.zeros((len(X_te_sc), 1))
    ones_te = np.ones((len(X_te_sc), 1))
    mu_tr0 = ridge.predict(np.hstack([X_tr_sc, zeros_tr])).reshape(-1, 1)
    mu_tr1 = ridge.predict(np.hstack([X_tr_sc, ones_tr])).reshape(-1, 1)
    mu_te0 = ridge.predict(np.hstack([X_te_sc, zeros_te])).reshape(-1, 1)
    mu_te1 = ridge.predict(np.hstack([X_te_sc, ones_te])).reshape(-1, 1)
    return np.hstack([X_tr, mu_tr0, mu_tr1]), np.hstack([X_te, mu_te0, mu_te1])

dataset = IHDPDataset(n_tables=2, seed=42)
R = [dataset[i] for i in range(2)]

print("Ridge alpha sweep for S-learner features:")
for alpha in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]:
    pvs = []
    for (cate_dset, _) in R:
        Xtr, Xte = add_s_features(cate_dset.X_train, cate_dset.t_train, cate_dset.y_train, cate_dset.X_test, alpha)
        est = CATEEstimator(device=str(device), model_path='vdblm/causalpfn', cache_dir=HF_CACHE)
        est.prediction_temperature = 0.8
        est.fit(Xtr, cate_dset.t_train, cate_dset.y_train)
        pvs.append(calculate_pehe(cate_dset.true_cate, est.estimate_cate(Xte)))
    print(f"alpha={alpha:>7}: R1={pvs[0]:.4f} R2={pvs[1]:.4f} Mean={np.mean(pvs):.4f}")

print("\nTemperature sweep with S-learner (alpha=1.0):")
for T in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    pvs = []
    for (cate_dset, _) in R:
        Xtr, Xte = add_s_features(cate_dset.X_train, cate_dset.t_train, cate_dset.y_train, cate_dset.X_test, 1.0)
        est = CATEEstimator(device=str(device), model_path='vdblm/causalpfn', cache_dir=HF_CACHE)
        est.prediction_temperature = T
        est.fit(Xtr, cate_dset.t_train, cate_dset.y_train)
        pvs.append(calculate_pehe(cate_dset.true_cate, est.estimate_cate(Xte)))
    print(f"T={T}: R1={pvs[0]:.4f} R2={pvs[1]:.4f} Mean={np.mean(pvs):.4f}")
