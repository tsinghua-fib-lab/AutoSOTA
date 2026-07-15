"""Compute SID for best PACER seed on Sachs."""
import os, sys
import numpy as np
sys.path.insert(0, "/repo/src")
from pacer import PACER

def compute_sid(pred, gt):
    d = pred.shape[0]
    pred_adj = (pred != 0).astype(int)
    gt_adj = (gt != 0).astype(int)
    np.fill_diagonal(pred_adj, 0)
    np.fill_diagonal(gt_adj, 0)

    def descendants(adj, node):
        desc = set()
        visited = set()
        def dfs(v):
            if v in visited:
                return
            visited.add(v)
            children = np.where(adj[v, :] == 1)[0]
            for c in children:
                if c not in visited:
                    desc.add(c)
                    dfs(c)
        dfs(node)
        return desc

    pred_desc = [descendants(pred_adj, v) for v in range(d)]
    gt_desc = [descendants(gt_adj, v) for v in range(d)]

    sid = 0
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            pa1 = set(np.where(pred_adj[:, j] == 1)[0])
            pa2 = set(np.where(gt_adj[:, j] == 1)[0])
            if pa1 != pa2:
                sid += 1
            elif (pred_desc[i] & pa1) != (gt_desc[i] & pa2):
                sid += 1
    return sid


# Load data
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

# Best seeds from multi-seed run
for best_seed in [0, 1, 10]:
    print(f"\nSeed {best_seed}:")
    np.random.seed(best_seed)
    idx = np.random.permutation(n_samples)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    x_train = data[train_idx]; m_train = masks[train_idx]; r_train = regimes[train_idx]
    x_val = data[val_idx]; m_val = masks[val_idx]; r_val = regimes[val_idx]

    pacer = PACER(
        n_vars=n_vars, n_layers=2, hdim=4, density="gaussian",
        fit_analytic=False, n_steps=5000, lr=0.01, batch_size=64,
        n_mc_samples=200, lambd=1.0, seed=best_seed,
    )
    pacer.fit(x_train, m_train, r_train, x_val=x_val, masks_val=m_val, regimes_val=r_val)

    pred_dag = pacer.predict(threshold=0.5)

    tp = int(np.sum((pred_dag == 1) & (dag_gt == 1)))
    fp = int(np.sum((pred_dag == 1) & (dag_gt == 0)))
    fn = int(np.sum((pred_dag == 0) & (dag_gt == 1)))
    shd = fp + fn
    fdr = fp / max(tp + fp, 1)
    tpr = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * tpr / max(prec + tpr, 1e-8)
    sid = compute_sid(pred_dag, dag_gt)

    print(f"  SHD={shd}, SID={sid}, FDR={fdr:.4f}, TPR={tpr:.4f}, F1={f1:.4f}")
    np.save(f"/repo/pred_dag_seed{best_seed}.npy", pred_dag.astype(int))
