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

print("num_neighbours sweep with S-learner features (T=1.0):")
for nn in [50, 100, 200, 400, 600, 800, 1024]:
    pvs = []
    for (cate_dset, _) in R:
        Xtr, Xte = add_s_feats(cate_dset.X_train, cate_dset.t_train, cate_dset.y_train, cate_dset.X_test)
        est = CATEEstimator(device=str(device), model_path='vdblm/causalpfn', cache_dir=HF_CACHE, num_neighbours=nn)
        est.prediction_temperature = 1.0
        est.fit(Xtr, cate_dset.t_train, cate_dset.y_train)
        pvs.append(calculate_pehe(cate_dset.true_cate, est.estimate_cate(Xte)))
    print(f"nn={nn:>5}: R1={pvs[0]:.4f} R2={pvs[1]:.4f} Mean={np.mean(pvs):.4f}")
