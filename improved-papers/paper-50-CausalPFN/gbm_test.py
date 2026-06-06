import os, sys, warnings, importlib.util
PKGDIR = "/home/dataset-assist-0/AUTOSOTA/sota-5/auto-pipeline-ab/optimizer/papers/paper-1630/runs/run_20260325_014145/results/pypackages"
HF_CACHE = "/home/dataset-assist-0/AUTOSOTA/sota-5/auto-pipeline-ab/optimizer/papers/paper-1630/runs/run_20260325_014145/results/hf_cache"
sys.path.insert(0, PKGDIR)
warnings.filterwarnings('ignore')
sys.path.insert(0, '/repo')

import numpy as np, torch, random
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
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

def add_s_ridge(X_tr, t, y, X_te, alpha=0.001):
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)
    Xt_tr = np.hstack([X_tr_sc, t.reshape(-1,1)])
    ridge = Ridge(alpha=alpha); ridge.fit(Xt_tr, y)
    def pred(X_sc, t_val):
        T = np.full((len(X_sc),1), t_val)
        return ridge.predict(np.hstack([X_sc, T])).reshape(-1,1)
    return (np.hstack([X_tr, pred(X_tr_sc,0), pred(X_tr_sc,1)]),
            np.hstack([X_te, pred(X_te_sc,0), pred(X_te_sc,1)]))

def add_s_gbm(X_tr, t, y, X_te, n_est=50, max_d=3):
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)
    Xt_tr = np.hstack([X_tr_sc, t.reshape(-1,1)])
    gbm = GradientBoostingRegressor(n_estimators=n_est, max_depth=max_d, random_state=42)
    gbm.fit(Xt_tr, y)
    def pred(X_sc, t_val):
        T = np.full((len(X_sc),1), t_val)
        return gbm.predict(np.hstack([X_sc, T])).reshape(-1,1)
    return (np.hstack([X_tr, pred(X_tr_sc,0), pred(X_tr_sc,1)]),
            np.hstack([X_te, pred(X_te_sc,0), pred(X_te_sc,1)]))

def add_s_combined(X_tr, t, y, X_te):
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)
    Xt_tr = np.hstack([X_tr_sc, t.reshape(-1,1)])
    # Ridge (minimal regularization)
    ridge = Ridge(alpha=0.001); ridge.fit(Xt_tr, y)
    # GBM
    gbm = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42); gbm.fit(Xt_tr, y)
    extras_tr, extras_te = [], []
    for model in [ridge, gbm]:
        for t_val in [0, 1]:
            T_tr = np.full((len(X_tr_sc),1), t_val)
            T_te = np.full((len(X_te_sc),1), t_val)
            extras_tr.append(model.predict(np.hstack([X_tr_sc, T_tr])).reshape(-1,1))
            extras_te.append(model.predict(np.hstack([X_te_sc, T_te])).reshape(-1,1))
    return np.hstack([X_tr]+extras_tr), np.hstack([X_te]+extras_te)

dataset = IHDPDataset(n_tables=2, seed=42)
R = [dataset[i] for i in range(2)]

# Test different feature augmentations
configs = [
    ("ridge(0.001)", lambda d: add_s_ridge(d.X_train, d.t_train, d.y_train, d.X_test, 0.001)),
    ("gbm(50,3)", lambda d: add_s_gbm(d.X_train, d.t_train, d.y_train, d.X_test, 50, 3)),
    ("gbm(100,3)", lambda d: add_s_gbm(d.X_train, d.t_train, d.y_train, d.X_test, 100, 3)),
    ("gbm(100,4)", lambda d: add_s_gbm(d.X_train, d.t_train, d.y_train, d.X_test, 100, 4)),
    ("ridge+gbm", lambda d: add_s_combined(d.X_train, d.t_train, d.y_train, d.X_test)),
]

for label, feat_fn in configs:
    for T in [0.9, 1.0]:
        pvs = []
        for (cate_dset, _) in R:
            Xtr, Xte = feat_fn(cate_dset)
            est = CATEEstimator(device=str(device), model_path='vdblm/causalpfn', cache_dir=HF_CACHE)
            est.prediction_temperature = T
            est.fit(Xtr, cate_dset.t_train, cate_dset.y_train)
            pvs.append(calculate_pehe(cate_dset.true_cate, est.estimate_cate(Xte)))
        print(f"{label} T={T}: R1={pvs[0]:.4f} R2={pvs[1]:.4f} Mean={np.mean(pvs):.4f}")
