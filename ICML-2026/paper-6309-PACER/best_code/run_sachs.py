"""Run PACER on the Sachs interventional benchmark."""
import os
import sys
import json
import numpy as np
import jax
import jax.numpy as jnp
sys.path.insert(0, "/repo/src")
from pacer import PACER

# ── Load data ───────────────────────────────────────────────
data_dir = "/datasets/sachs/sachs_intervention"
data = np.load(os.path.join(data_dir, "data_interv1.npy")).astype(np.float32)
dag_gt = np.load(os.path.join(data_dir, "DAG1.npy"))
n_samples, n_vars = data.shape
print(f"Data: {data.shape}, DAG: {dag_gt.shape}, edges: {int(dag_gt.sum())}")

# Read regime and intervention files
with open(os.path.join(data_dir, "regime1.csv")) as f:
    regime_lines = [l.strip() for l in f.readlines()]
regimes_raw = np.array([int(l) if l.strip() else 0 for l in regime_lines])

with open(os.path.join(data_dir, "intervention1.csv")) as f:
    interv_lines = f.readlines()

# Build masks and regimes
masks = np.ones((n_samples, n_vars), dtype=np.float32)
regimes = np.zeros(n_samples, dtype=np.int32)

for i in range(n_samples):
    r = int(regime_lines[i].strip()) if regime_lines[i].strip() else 0
    regimes[i] = r
    interv_str = interv_lines[i].strip() if i < len(interv_lines) else ""
    if interv_str and r > 0:
        target_var = int(interv_str) - 1
        masks[i, target_var] = 0.0

print(f"Regime counts: {dict(zip(*np.unique(regimes, return_counts=True)))}")
print(f"Intervened samples: {int(np.sum(masks.min(axis=1) < 1.0))}")
print(f"Obs samples: {int(np.sum(masks.min(axis=1) >= 1.0))}")

# Standardize data
data_mean = data.mean(axis=0, keepdims=True)
data_std = data.std(axis=0, keepdims=True) + 1e-8
data = (data - data_mean) / data_std

# Train/val split
np.random.seed(42)
n_train = int(0.8 * n_samples)
idx = np.random.permutation(n_samples)
train_idx, val_idx = idx[:n_train], idx[n_train:]

x_train = data[train_idx]
masks_train = masks[train_idx]
regimes_train = regimes[train_idx]
x_val = data[val_idx]
masks_val = masks[val_idx]
regimes_val = regimes[val_idx]

print(f"Train: {x_train.shape}, Val: {x_val.shape}")

# ── Fit PACER ────────────────────────────────────────────────
print("\n=== Fitting PACER ===")
pacer = PACER(
    n_vars=n_vars,
    n_layers=2,
    hdim=4,
    density="gaussian",
    fit_analytic=False,
    n_steps=5000,
    lr=0.01,
    batch_size=64,
    n_mc_samples=200,
    lambd=1.0,
    seed=0,  # Paper seed
)
pacer.fit(x_train, masks_train, regimes_train,
          x_val=x_val, masks_val=masks_val, regimes_val=regimes_val)

# ── Predict ─────────────────────────────────────────────────
edge_probs = pacer.predict_proba()
pred_dag = pacer.predict(threshold=0.5)

print(f"\nEdge probs range: [{edge_probs.min():.4f}, {edge_probs.max():.4f}]")
print(f"Predicted edges: {pred_dag.sum()}")

# ── Evaluate ────────────────────────────────────────────────
def compute_metrics(pred, gt):
    tp = int(np.sum((pred == 1) & (gt == 1)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))
    tn = int(np.sum((pred == 0) & (gt == 0)))
    shd = int(np.sum(pred != gt))
    fdr = fp / max(tp + fp, 1)
    tpr = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tpr
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"SHD": shd, "FDR": fdr, "TPR": tpr, "F1": f1,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}

def compute_sid(pred, gt):
    """Compute SID between two DAG adjacency matrices."""
    d = pred.shape[0]
    pred_adj = pred.astype(int).copy()
    gt_adj = gt.astype(int).copy()
    np.fill_diagonal(pred_adj, 0)
    np.fill_diagonal(gt_adj, 0)

    sid = 0

    def reachable(adj, start, targets):
        """BFS to check if start can reach any node in targets."""
        if start in targets:
            return True
        n = adj.shape[0]
        visited = np.zeros(n, dtype=bool)
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            children = np.where(adj[:, node] == 1)[0]
            for child in children:
                if visited[child]:
                    continue
                visited[child] = True
                if child in targets:
                    return True
                queue.append(child)
        return False

    for j in range(d):
        pa_pred = set(np.where(pred_adj[:, j] == 1)[0])
        pa_gt = set(np.where(gt_adj[:, j] == 1)[0])

        for i in range(d):
            if i == j:
                continue
            # Case 1: i is parent of j in at least one graph
            if (i in pa_pred) != (i in pa_gt):
                sid += 1
            elif i in pa_pred and i in pa_gt and pa_pred != pa_gt:
                sid += 1
            elif i not in pa_pred and i not in pa_gt:
                # Case 2: i not a parent of j in either graph
                r_pred = reachable(pred_adj, i, pa_pred)
                r_gt = reachable(gt_adj, i, pa_gt)
                if r_pred != r_gt:
                    sid += 1

    return sid

metrics = compute_metrics(pred_dag, dag_gt)
sid_val = compute_sid(pred_dag, dag_gt)

print(f"\n=== Key Metrics ===")
print(f"SHD: {metrics['SHD']}")
print(f"SID: {sid_val}")
print(f"FDR: {metrics['FDR']:.4f}")
print(f"TPR: {metrics['TPR']:.4f}")
print(f"F1: {metrics['F1']:.4f}")
print(f"TP: {metrics['TP']}, FP: {metrics['FP']}, FN: {metrics['FN']}")

# Save results
results = {
    "SHD": metrics["SHD"],
    "SID": sid_val,
    "FDR": round(metrics["FDR"], 4),
    "TPR": round(metrics["TPR"], 4),
    "F1": round(metrics["F1"], 4),
    "TP": metrics["TP"],
    "FP": metrics["FP"],
    "FN": metrics["FN"],
    "TN": metrics["TN"],
    "predicted_edges": int(pred_dag.sum()),
    "ground_truth_edges": int(dag_gt.sum()),
}
with open("/repo/sachs_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Save predicted adjacency
np.save("/repo/pred_dag.npy", pred_dag.astype(int))
np.save("/repo/edge_probs.npy", edge_probs)

print("\nResults saved to /repo/sachs_results.json")
print("Predicted DAG saved to /repo/pred_dag.npy")
