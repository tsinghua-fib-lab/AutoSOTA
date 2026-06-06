import os, sys, warnings, importlib.util
PKGDIR = "/home/dataset-assist-0/AUTOSOTA/sota-5/auto-pipeline-ab/optimizer/papers/paper-1630/runs/run_20260325_014145/results/pypackages"
HF_CACHE = "/home/dataset-assist-0/AUTOSOTA/sota-5/auto-pipeline-ab/optimizer/papers/paper-1630/runs/run_20260325_014145/results/hf_cache"
sys.path.insert(0, PKGDIR)
warnings.filterwarnings('ignore')
sys.path.insert(0, '/repo')

import numpy as np, torch, random
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
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

def s_ridge_insample(X_tr, t, y, X_te, alpha=0.001):
    sc = StandardScaler(); X_tr_sc = sc.fit_transform(X_tr); X_te_sc = sc.transform(X_te)
    ridge = Ridge(alpha=alpha); ridge.fit(np.hstack([X_tr_sc, t.reshape(-1,1)]), y)
    def pred(X_sc, t_val):
        T = np.full((len(X_sc),1), t_val)
        return ridge.predict(np.hstack([X_sc, T])).reshape(-1,1)
    return (np.hstack([X_tr, pred(X_tr_sc,0), pred(X_tr_sc,1)]),
            np.hstack([X_te, pred(X_te_sc,0), pred(X_te_sc,1)]))

def s_ridge_with_cate_only(X_tr, t, y, X_te, alpha=0.001):
    """Add only estimated CATE (mu_1 - mu_0) as single extra feature."""
    sc = StandardScaler(); X_tr_sc = sc.fit_transform(X_tr); X_te_sc = sc.transform(X_te)
    ridge = Ridge(alpha=alpha); ridge.fit(np.hstack([X_tr_sc, t.reshape(-1,1)]), y)
    def dpred(X_sc):
        zeros = np.zeros((len(X_sc),1)); ones = np.ones((len(X_sc),1))
        return (ridge.predict(np.hstack([X_sc,ones])) - ridge.predict(np.hstack([X_sc,zeros]))).reshape(-1,1)
    return np.hstack([X_tr, dpred(X_tr_sc)]), np.hstack([X_te, dpred(X_te_sc)])

def s_ridge_poly2(X_tr, t, y, X_te, alpha=0.001):
    """Use polynomial features (degree 2) for outcome model."""
    sc = StandardScaler(); X_tr_sc = sc.fit_transform(X_tr); X_te_sc = sc.transform(X_te)
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    X_tr_poly = poly.fit_transform(X_tr_sc); X_te_poly = poly.transform(X_te_sc)
    Xt_tr = np.hstack([X_tr_poly, t.reshape(-1,1)])
    ridge = Ridge(alpha=alpha); ridge.fit(Xt_tr, y)
    def pred(X_poly, t_val):
        T = np.full((len(X_poly),1), t_val)
        return ridge.predict(np.hstack([X_poly, T])).reshape(-1,1)
    return (np.hstack([X_tr, pred(X_tr_poly,0), pred(X_tr_poly,1)]),
            np.hstack([X_te, pred(X_te_poly,0), pred(X_te_poly,1)]))

def add_tx_interactions(X_tr, t, y, X_te, alpha=0.001):
    """Add S-learner features + treatment-feature interactions."""
    sc = StandardScaler(); X_tr_sc = sc.fit_transform(X_tr); X_te_sc = sc.transform(X_te)
    ridge = Ridge(alpha=alpha); ridge.fit(np.hstack([X_tr_sc, t.reshape(-1,1)]), y)
    def pred(X_sc, t_val):
        T = np.full((len(X_sc),1), t_val)
        return ridge.predict(np.hstack([X_sc, T])).reshape(-1,1)
    # Add T*X interactions
    t_tr_c = t.reshape(-1,1); t_te0_c = np.zeros((len(X_te),1)); t_te1_c = np.ones((len(X_te),1))
    return (np.hstack([X_tr, pred(X_tr_sc,0), pred(X_tr_sc,1), X_tr_sc * t_tr_c]),
            np.hstack([X_te, pred(X_te_sc,0), pred(X_te_sc,1), X_te_sc * 0.5]))  # avg of t=0,1

dataset = IHDPDataset(n_tables=2, seed=42)
R = [dataset[i] for i in range(2)]

configs = [
    ("s_ridge_mu0+mu1", s_ridge_insample),
    ("s_ridge_cate_only", s_ridge_with_cate_only),
    ("s_ridge_poly2", s_ridge_poly2),
    ("s_ridge+TX", add_tx_interactions),
]

for label, feat_fn in configs:
    for T_val in [0.9, 1.0]:
        pvs = []
        for (cate_dset, _) in R:
            Xtr, Xte = feat_fn(cate_dset.X_train, cate_dset.t_train, cate_dset.y_train, cate_dset.X_test)
            est = CATEEstimator(device=str(device), model_path='vdblm/causalpfn', cache_dir=HF_CACHE)
            est.prediction_temperature = T_val
            est.fit(Xtr, cate_dset.t_train, cate_dset.y_train)
            pvs.append(calculate_pehe(cate_dset.true_cate, est.estimate_cate(Xte)))
        print(f"{label} T={T_val}: R1={pvs[0]:.4f} R2={pvs[1]:.4f} Mean={np.mean(pvs):.4f}")
