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

def add_s_feats(X_tr, t, y, X_te, alpha=0.001):
    sc = StandardScaler(); X_tr_sc = sc.fit_transform(X_tr); X_te_sc = sc.transform(X_te)
    ridge = Ridge(alpha=alpha); ridge.fit(np.hstack([X_tr_sc, t.reshape(-1,1)]), y)
    def pred(X_sc, t_val):
        T = np.full((len(X_sc),1), t_val)
        return ridge.predict(np.hstack([X_sc, T])).reshape(-1,1)
    return (np.hstack([X_tr, pred(X_tr_sc,0), pred(X_tr_sc,1)]),
            np.hstack([X_te, pred(X_te_sc,0), pred(X_te_sc,1)]))

dataset = IHDPDataset(n_tables=2, seed=42)
R = [dataset[i] for i in range(2)]

print("Bootstrap with S-learner features (T=1.0):")
print(f"{'n_boot':>6} | {'frac':>5} | {'R1':>8} | {'R2':>8} | {'Mean':>8}")
for n_boot, frac in [(3, 0.9), (5, 0.9), (7, 0.9), (10, 0.9), (5, 0.85), (5, 0.95), (7, 0.85)]:
    pehe_values = []
    for realization_idx in range(2):
        cate_dset, _ = R[realization_idx]
        X_tr_aug, X_te_aug = add_s_feats(cate_dset.X_train, cate_dset.t_train, cate_dset.y_train, cate_dset.X_test)
        rng = np.random.RandomState(42 + realization_idx + 100)
        n = len(X_tr_aug)
        all_preds = []
        for b in range(n_boot):
            idx = rng.choice(n, size=int(n * frac), replace=False)
            est = CATEEstimator(device=str(device), model_path='vdblm/causalpfn', cache_dir=HF_CACHE)
            est.prediction_temperature = 1.0
            est.fit(X_tr_aug[idx], cate_dset.t_train[idx], cate_dset.y_train[idx])
            all_preds.append(est.estimate_cate(X_te_aug))
        pehe_values.append(calculate_pehe(cate_dset.true_cate, np.mean(all_preds, axis=0)))
    mean_pehe = np.mean(pehe_values)
    print(f"{n_boot:>6} | {frac:>5.2f} | {pehe_values[0]:.4f} | {pehe_values[1]:.4f} | {mean_pehe:.4f}")
