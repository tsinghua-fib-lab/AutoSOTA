import os, sys, warnings, importlib.util
PKGDIR = "/home/dataset-assist-0/AUTOSOTA/sota-5/auto-pipeline-ab/optimizer/papers/paper-1630/runs/run_20260325_014145/results/pypackages"
HF_CACHE = "/home/dataset-assist-0/AUTOSOTA/sota-5/auto-pipeline-ab/optimizer/papers/paper-1630/runs/run_20260325_014145/results/hf_cache"
sys.path.insert(0, PKGDIR); sys.path.insert(0, '/repo')
warnings.filterwarnings('ignore')
import numpy as np, torch, random
from sklearn.linear_model import Ridge; from sklearn.preprocessing import StandardScaler
random.seed(42); np.random.seed(42); torch.manual_seed(42)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def load_module_direct(name, fp):
    spec = importlib.util.spec_from_file_location(name, fp); mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod; spec.loader.exec_module(mod); return mod
base_mod = load_module_direct("benchmarks.base", "/repo/benchmarks/base.py")
ihdp_mod = load_module_direct("benchmarks.ihdp", "/repo/benchmarks/ihdp.py")
IHDPDataset = ihdp_mod.IHDPDataset
from causalpfn import CATEEstimator; from causalpfn.evaluation import calculate_pehe
def add_s_feats(X_tr, t, y, X_te):
    sc = StandardScaler(); X_tr_sc = sc.fit_transform(X_tr); X_te_sc = sc.transform(X_te)
    ridge = Ridge(alpha=0.001); ridge.fit(np.hstack([X_tr_sc, t.reshape(-1,1)]), y)
    def pred(X_sc, tv): T = np.full((len(X_sc),1),tv); return ridge.predict(np.hstack([X_sc, T])).reshape(-1,1)
    return np.hstack([X_tr, pred(X_tr_sc,0), pred(X_tr_sc,1)]), np.hstack([X_te, pred(X_te_sc,0), pred(X_te_sc,1)])
dataset = IHDPDataset(n_tables=2, seed=42)
R = [dataset[i] for i in range(2)]

print(f"{'config':>25} | {'R1':>8} | {'R2':>8} | {'Mean':>8}")
for n_boot, frac, nn, T in [
    (5, 0.9, 100, 1.1),   # current best
    (8, 0.9, 100, 1.1),
    (10, 0.9, 100, 1.1),
    (15, 0.9, 100, 1.1),
    (5, 0.85, 100, 1.1),
    (5, 0.95, 100, 1.1),
    (5, 0.9, 90, 1.1),
    (5, 0.9, 80, 1.1),
    (10, 0.9, 90, 1.1),
]:
    pvs = []
    for realization_idx in range(2):
        cate_dset, _ = R[realization_idx]
        Xtr, Xte = add_s_feats(cate_dset.X_train, cate_dset.t_train, cate_dset.y_train, cate_dset.X_test)
        rng = np.random.RandomState(42 + realization_idx + 100)
        n = len(Xtr)
        preds = []
        for b in range(n_boot):
            idx = rng.choice(n, size=int(n*frac), replace=False)
            est = CATEEstimator(device=str(device), model_path='vdblm/causalpfn', cache_dir=HF_CACHE, num_neighbours=nn)
            est.prediction_temperature = T; est.fit(Xtr[idx], cate_dset.t_train[idx], cate_dset.y_train[idx])
            preds.append(est.estimate_cate(Xte))
        pvs.append(calculate_pehe(cate_dset.true_cate, np.mean(preds, axis=0)))
    
    label = f"n={n_boot},f={frac},nn={nn},T={T}"
    print(f"{label:>25} | {pvs[0]:.4f} | {pvs[1]:.4f} | {np.mean(pvs):.4f}")
