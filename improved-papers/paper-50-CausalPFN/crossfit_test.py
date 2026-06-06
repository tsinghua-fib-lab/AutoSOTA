import os, sys, warnings, importlib.util
PKGDIR = "/home/dataset-assist-0/AUTOSOTA/sota-5/auto-pipeline-ab/optimizer/papers/paper-1630/runs/run_20260325_014145/results/pypackages"
HF_CACHE = "/home/dataset-assist-0/AUTOSOTA/sota-5/auto-pipeline-ab/optimizer/papers/paper-1630/runs/run_20260325_014145/results/hf_cache"
sys.path.insert(0, PKGDIR)
warnings.filterwarnings('ignore')
sys.path.insert(0, '/repo')

import numpy as np, torch, random
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
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

def add_crossfit_features(X_tr, t, y, X_te, alpha=0.001, n_folds=5):
    """Cross-fit S-learner: out-of-fold predictions for training data."""
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)
    
    mu_tr0 = np.zeros(len(X_tr))
    mu_tr1 = np.zeros(len(X_tr))
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf.split(X_tr):
        X_k = X_tr_sc[tr_idx]; t_k = t[tr_idx]; y_k = y[tr_idx]
        X_v = X_tr_sc[val_idx]
        Xt_k = np.hstack([X_k, t_k.reshape(-1,1)])
        ridge = Ridge(alpha=alpha); ridge.fit(Xt_k, y_k)
        zeros_v = np.zeros((len(X_v),1)); ones_v = np.ones((len(X_v),1))
        mu_tr0[val_idx] = ridge.predict(np.hstack([X_v, zeros_v]))
        mu_tr1[val_idx] = ridge.predict(np.hstack([X_v, ones_v]))
    
    # For test: train on all training data
    Xt_full = np.hstack([X_tr_sc, t.reshape(-1,1)])
    ridge_full = Ridge(alpha=alpha); ridge_full.fit(Xt_full, y)
    zeros_te = np.zeros((len(X_te_sc),1)); ones_te = np.ones((len(X_te_sc),1))
    mu_te0 = ridge_full.predict(np.hstack([X_te_sc, zeros_te]))
    mu_te1 = ridge_full.predict(np.hstack([X_te_sc, ones_te]))
    
    return (np.hstack([X_tr, mu_tr0.reshape(-1,1), mu_tr1.reshape(-1,1)]),
            np.hstack([X_te, mu_te0.reshape(-1,1), mu_te1.reshape(-1,1)]))

def add_s_insample(X_tr, t, y, X_te, alpha=0.001):
    sc = StandardScaler(); X_tr_sc = sc.fit_transform(X_tr); X_te_sc = sc.transform(X_te)
    ridge = Ridge(alpha=alpha); ridge.fit(np.hstack([X_tr_sc, t.reshape(-1,1)]), y)
    def pred(X_sc, t_val):
        T = np.full((len(X_sc),1), t_val)
        return ridge.predict(np.hstack([X_sc, T])).reshape(-1,1)
    return np.hstack([X_tr, pred(X_tr_sc,0), pred(X_tr_sc,1)]), np.hstack([X_te, pred(X_te_sc,0), pred(X_te_sc,1)])

dataset = IHDPDataset(n_tables=2, seed=42)
R = [dataset[i] for i in range(2)]

print("Cross-fit vs in-sample S-learner comparison:")
for T_val in [0.9, 1.0]:
    for n_folds in [2, 3, 5, 10]:
        pvs = []
        for (cate_dset, _) in R:
            Xtr, Xte = add_crossfit_features(
                cate_dset.X_train, cate_dset.t_train, cate_dset.y_train, cate_dset.X_test, 
                alpha=0.001, n_folds=n_folds)
            est = CATEEstimator(device=str(device), model_path='vdblm/causalpfn', cache_dir=HF_CACHE)
            est.prediction_temperature = T_val
            est.fit(Xtr, cate_dset.t_train, cate_dset.y_train)
            pvs.append(calculate_pehe(cate_dset.true_cate, est.estimate_cate(Xte)))
        print(f"crossfit-{n_folds}fold T={T_val}: R1={pvs[0]:.4f} R2={pvs[1]:.4f} Mean={np.mean(pvs):.4f}")

print()
# Also test: weighted Ridge using propensity
print("IPW-weighted Ridge S-learner:")
from sklearn.linear_model import LogisticRegression
def add_ipw_s_features(X_tr, t, y, X_te, alpha=0.001):
    sc = StandardScaler(); X_tr_sc = sc.fit_transform(X_tr); X_te_sc = sc.transform(X_te)
    # Get propensity weights
    ps_clf = LogisticRegression(max_iter=1000, random_state=42, C=0.01); ps_clf.fit(X_tr_sc, t)
    ps = ps_clf.predict_proba(X_tr_sc)[:,1]
    ps = np.clip(ps, 0.05, 0.95)  # clip for stability
    ipw = np.where(t == 1, 1.0/ps, 1.0/(1-ps))
    Xt_tr = np.hstack([X_tr_sc, t.reshape(-1,1)])
    ridge = Ridge(alpha=alpha); ridge.fit(Xt_tr, y, sample_weight=ipw)
    def pred(X_sc, t_val):
        T = np.full((len(X_sc),1), t_val)
        return ridge.predict(np.hstack([X_sc, T])).reshape(-1,1)
    return np.hstack([X_tr, pred(X_tr_sc,0), pred(X_tr_sc,1)]), np.hstack([X_te, pred(X_te_sc,0), pred(X_te_sc,1)])

for T_val in [0.9, 1.0]:
    pvs = []
    for (cate_dset, _) in R:
        Xtr, Xte = add_ipw_s_features(cate_dset.X_train, cate_dset.t_train, cate_dset.y_train, cate_dset.X_test)
        est = CATEEstimator(device=str(device), model_path='vdblm/causalpfn', cache_dir=HF_CACHE)
        est.prediction_temperature = T_val
        est.fit(Xtr, cate_dset.t_train, cate_dset.y_train)
        pvs.append(calculate_pehe(cate_dset.true_cate, est.estimate_cate(Xte)))
    print(f"IPW-ridge T={T_val}: R1={pvs[0]:.4f} R2={pvs[1]:.4f} Mean={np.mean(pvs):.4f}")
