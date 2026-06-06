import os, sys, warnings, importlib.util
PKGDIR = "/home/dataset-assist-0/AUTOSOTA/sota-5/auto-pipeline-ab/optimizer/papers/paper-1630/runs/run_20260325_014145/results/pypackages"
HF_CACHE = "/home/dataset-assist-0/AUTOSOTA/sota-5/auto-pipeline-ab/optimizer/papers/paper-1630/runs/run_20260325_014145/results/hf_cache"
sys.path.insert(0, PKGDIR)
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

def augment(X_tr, t_tr, y_tr, X_te, use_prop=True, use_s=True, prop_C=0.01, ridge_alpha=0.001, T=1.0):
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)
    extras_tr, extras_te = [], []
    
    if use_prop:
        clf = LogisticRegression(max_iter=1000, random_state=42, C=prop_C)
        clf.fit(X_tr_sc, t_tr)
        extras_tr.append(clf.predict_proba(X_tr_sc)[:,1:2])
        extras_te.append(clf.predict_proba(X_te_sc)[:,1:2])
    
    if use_s:
        ridge = Ridge(alpha=ridge_alpha)
        ridge.fit(np.hstack([X_tr_sc, t_tr.reshape(-1,1)]), y_tr)
        zeros_tr = np.zeros((len(X_tr_sc), 1)); ones_tr = np.ones((len(X_tr_sc), 1))
        zeros_te = np.zeros((len(X_te_sc), 1)); ones_te = np.ones((len(X_te_sc), 1))
        extras_tr.extend([ridge.predict(np.hstack([X_tr_sc, zeros_tr])).reshape(-1,1),
                          ridge.predict(np.hstack([X_tr_sc, ones_tr])).reshape(-1,1)])
        extras_te.extend([ridge.predict(np.hstack([X_te_sc, zeros_te])).reshape(-1,1),
                          ridge.predict(np.hstack([X_te_sc, ones_te])).reshape(-1,1)])
    
    Xtr_aug = np.hstack([X_tr] + extras_tr) if extras_tr else X_tr
    Xte_aug = np.hstack([X_te] + extras_te) if extras_te else X_te
    return Xtr_aug, Xte_aug, T

dataset = IHDPDataset(n_tables=2, seed=42)
R = [dataset[i] for i in range(2)]

configs = [
    # (prop, s, prop_C, ridge_alpha, T, label)
    (True,  False, 0.01,  1.0,   0.8, "prop(0.01)+T=0.8"),
    (False, True,  None,  0.001, 1.0, "s(0.001)+T=1.0"),
    (False, True,  None,  0.001, 0.9, "s(0.001)+T=0.9"),
    (True,  True,  0.01,  0.001, 1.0, "prop+s+T=1.0"),
    (True,  True,  0.01,  0.001, 0.9, "prop+s+T=0.9"),
    (True,  True,  0.01,  0.001, 0.8, "prop+s+T=0.8"),
    (True,  True,  0.1,   0.001, 1.0, "prop(0.1)+s+T=1.0"),
    (True,  True,  0.001, 0.001, 1.0, "prop(0.001)+s+T=1.0"),
    # Also try s with mu_0 only (since model predicts CATE = mu_1 - mu_0)
]

for (use_prop, use_s, prop_C, ridge_alpha, T, label) in configs:
    pvs = []
    for (cate_dset, _) in R:
        Xtr, Xte, _ = augment(cate_dset.X_train, cate_dset.t_train, cate_dset.y_train, 
                                cate_dset.X_test, use_prop, use_s, prop_C, ridge_alpha, T)
        est = CATEEstimator(device=str(device), model_path='vdblm/causalpfn', cache_dir=HF_CACHE)
        est.prediction_temperature = T
        est.fit(Xtr, cate_dset.t_train, cate_dset.y_train)
        pvs.append(calculate_pehe(cate_dset.true_cate, est.estimate_cate(Xte)))
    print(f"{label}: R1={pvs[0]:.4f} R2={pvs[1]:.4f} Mean={np.mean(pvs):.4f}")
