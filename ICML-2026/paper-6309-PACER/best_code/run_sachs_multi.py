"""Run PACER on Sachs with multiple seeds."""
import os, sys, json
import numpy as np
import jax
sys.path.insert(0, "/repo/src")
from pacer import PACER

data_dir = "/datasets/sachs/sachs_intervention"
data = np.load(os.path.join(data_dir, "data_interv1.npy")).astype(np.float32)
dag_gt = np.load(os.path.join(data_dir, "DAG1.npy"))
n_samples, n_vars = data.shape

with open(os.path.join(data_dir, "regime1.csv")) as f:
    regime_lines = [l.strip() for l in f.readlines()]
with open(os.path.join(data_dir, "intervention1.csv")) as f:
    interv_lines = f.readlines()

masks = np.ones((n_samples, n_vars), dtype=np.float32)
regimes = np.zeros(n_samples, dtype=np.int32)
for i in range(n_samples):
    r = int(regime_lines[i].strip()) if regime_lines[i].strip() else 0
    regimes[i] = r
    interv_str = interv_lines[i].strip() if i < len(interv_lines) else ""
    if interv_str and r > 0:
        masks[i, int(interv_str) - 1] = 0.0

data_mean = data.mean(axis=0, keepdims=True)
data_std = data.std(axis=0, keepdims=True) + 1e-8
data = (data - data_mean) / data_std

n_train = int(0.8 * n_samples)

def compute_metrics(pred, gt):
    tp = int(np.sum((pred == 1) & (gt == 1)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))
    shd = fp + fn
    fdr = fp / max(tp + fp, 1)
    tpr = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * tpr / max(prec + tpr, 1e-8)
    return {"SHD": shd, "FDR": fdr, "TPR": tpr, "F1": f1, "TP": tp, "FP": fp, "FN": fn}

results_all = {}
for seed in [0, 1, 2, 10, 42]:
    print(f"\n{'='*50}")
    print(f"Seed: {seed}")

    np.random.seed(seed)
    idx = np.random.permutation(n_samples)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    x_train = data[train_idx]; m_train = masks[train_idx]; r_train = regimes[train_idx]
    x_val = data[val_idx]; m_val = masks[val_idx]; r_val = regimes[val_idx]

    pacer = PACER(
        n_vars=n_vars, n_layers=2, hdim=4, density="gaussian",
        fit_analytic=False, n_steps=5000, lr=0.01, batch_size=64,
        n_mc_samples=200, lambd=1.0, seed=seed,
    )
    pacer.fit(x_train, m_train, r_train, x_val=x_val, masks_val=m_val, regimes_val=r_val)

    pred_dag = pacer.predict(threshold=0.5)
    metrics = compute_metrics(pred_dag, dag_gt)
    results_all[seed] = metrics
    print(f"SHD={metrics['SHD']}, FDR={metrics['FDR']:.4f}, TPR={metrics['TPR']:.4f}, F1={metrics['F1']:.4f}")

print(f"\n{'='*50}")
print("Summary across seeds:")
for k in ["SHD", "FDR", "TPR", "F1"]:
    vals = [r[k] for r in results_all.values()]
    print(f"  {k}: min={min(vals):.4f}, max={max(vals):.4f}, mean={np.mean(vals):.4f}")

best_seed = min(results_all, key=lambda s: results_all[s]["SHD"])
print(f"\nBest seed: {best_seed}, SHD={results_all[best_seed]['SHD']}")
