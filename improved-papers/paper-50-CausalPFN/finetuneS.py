import os, sys, warnings, importlib.util
PKGDIR = "/home/dataset-assist-0/AUTOSOTA/sota-5/auto-pipeline-ab/optimizer/papers/paper-1630/runs/run_20260325_014145/results/pypackages"
HF_CACHE = "/home/dataset-assist-0/AUTOSOTA/sota-5/auto-pipeline-ab/optimizer/papers/paper-1630/runs/run_20260325_014145/results/hf_cache"
sys.path.insert(0, PKGDIR)
warnings.filterwarnings('ignore')
sys.path.insert(0, '/repo')

import numpy as np, torch, random
from sklearn.linear_model import Ridge
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

def s_feats(X_tr, t, y, X_te, alpha=0.001, feats='mu01'):
    sc = StandardScaler(); X_tr_sc = sc.fit_transform(X_tr); X_te_sc = sc.transform(X_te)
    ridge = Ridge(alpha=alpha); ridge.fit(np.hstack([X_tr_sc, t.reshape(-1,1)]), y)
    def pred(X_sc, t_val):
        T = np.full((len(X_sc),1), t_val)
        return ridge.predict(np.hstack([X_sc, T])).reshape(-1,1)
    mu_tr0, mu_tr1 = pred(X_tr_sc,0), pred(X_tr_sc,1)
    mu_te0, mu_te1 = pred(X_te_sc,0), pred(X_te_sc,1)
    if feats == 'mu01':
        return np.hstack([X_tr, mu_tr0, mu_tr1]), np.hstack([X_te, mu_te0, mu_te1])
    elif feats == 'mu0':
        return np.hstack([X_tr, mu_tr0]), np.hstack([X_te, mu_te0])
    elif feats == 'mu1':
        return np.hstack([X_tr, mu_tr1]), np.hstack([X_te, mu_te1])
    elif feats == 'mu01_y':  # also add observed Y as feature? No that doesn't make sense for test
        return np.hstack([X_tr, mu_tr0, mu_tr1]), np.hstack([X_te, mu_te0, mu_te1])
    elif feats == 'mu01_raw':  # use non-standardized X for ridge
        ridge2 = Ridge(alpha=alpha); ridge2.fit(np.hstack([X_tr, t.reshape(-1,1)]), y)
        def pred2(X, t_val):
            T = np.full((len(X),1), t_val)
            return ridge2.predict(np.hstack([X, T])).reshape(-1,1)
        return np.hstack([X_tr, pred2(X_tr,0), pred2(X_tr,1)]), np.hstack([X_te, pred2(X_te,0), pred2(X_te,1)])

dataset = IHDPDataset(n_tables=2, seed=42)
R = [dataset[i] for i in range(2)]

print("Fine-tuning S-learner features:")
for alpha in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
    for feats in ['mu01']:
        pvs = []
        for (cate_dset, _) in R:
            Xtr, Xte = s_feats(cate_dset.X_train, cate_dset.t_train, cate_dset.y_train, cate_dset.X_test, alpha, feats)
            est = CATEEstimator(device=str(device), model_path='vdblm/causalpfn', cache_dir=HF_CACHE)
            est.prediction_temperature = 1.0
            est.fit(Xtr, cate_dset.t_train, cate_dset.y_train)
            pvs.append(calculate_pehe(cate_dset.true_cate, est.estimate_cate(Xte)))
        print(f"a={alpha:.0e} {feats}: R1={pvs[0]:.4f} R2={pvs[1]:.4f} Mean={np.mean(pvs):.4f}")

print()
print("mu0 only vs mu1 only vs mu01:")
for feats in ['mu0', 'mu1', 'mu01']:
    for T_val in [0.9, 1.0]:
        pvs = []
        for (cate_dset, _) in R:
            Xtr, Xte = s_feats(cate_dset.X_train, cate_dset.t_train, cate_dset.y_train, cate_dset.X_test, 0.001, feats)
            est = CATEEstimator(device=str(device), model_path='vdblm/causalpfn', cache_dir=HF_CACHE)
            est.prediction_temperature = T_val
            est.fit(Xtr, cate_dset.t_train, cate_dset.y_train)
            pvs.append(calculate_pehe(cate_dset.true_cate, est.estimate_cate(Xte)))
        print(f"{feats} T={T_val}: R1={pvs[0]:.4f} R2={pvs[1]:.4f} Mean={np.mean(pvs):.4f}")

print()
print("Non-standardized X for ridge:")
pvs = []
for (cate_dset, _) in R:
    Xtr, Xte = s_feats(cate_dset.X_train, cate_dset.t_train, cate_dset.y_train, cate_dset.X_test, 0.001, 'mu01_raw')
    est = CATEEstimator(device=str(device), model_path='vdblm/causalpfn', cache_dir=HF_CACHE)
    est.prediction_temperature = 1.0
    est.fit(Xtr, cate_dset.t_train, cate_dset.y_train)
    pvs.append(calculate_pehe(cate_dset.true_cate, est.estimate_cate(Xte)))
print(f"mu01_raw T=1.0: R1={pvs[0]:.4f} R2={pvs[1]:.4f} Mean={np.mean(pvs):.4f}")
