"""Reproduce PACER on Sachs interventional benchmark.

This script runs PACER with the paper's default hyperparameters on the
Sachs interventional dataset (11 variables, 5846 measurements, 6 regimes)
and computes SHD, SID, FDR, TPR, and F1 metrics.

Paper reference:
  PACER: Acyclic Causal Discovery from Large-scale Interventional Data
  Vinas Torne et al., ICML 2026
  Table 1 - interventional setting
"""
import os
import sys
import json
import numpy as np
sys.path.insert(0, "/repo/src")
import jax
import jax.numpy as jnp
from pacer import PACER

# ── Configuration ───────────────────────────────────────────
SEED = 10
THRESHOLD = 0.5
DATA_DIR = "/datasets/sachs/sachs_intervention"

# ── Load data ───────────────────────────────────────────────
data = np.load(os.path.join(DATA_DIR, "data_interv1.npy")).astype(np.float32)
dag_gt = np.load(os.path.join(DATA_DIR, "DAG1.npy"))
n_samples, n_vars = data.shape

with open(os.path.join(DATA_DIR, "regime1.csv")) as f:
    regime_lines = [l.strip() for l in f.readlines()]
with open(os.path.join(DATA_DIR, "intervention1.csv")) as f:
    interv_lines = f.readlines()

masks = np.ones((n_samples, n_vars), dtype=np.float32)
regimes = np.zeros(n_samples, dtype=np.int32)
for i in range(n_samples):
    r = int(regime_lines[i].strip()) if regime_lines[i].strip() else 0
    regimes[i] = r
    interv_str = interv_lines[i].strip() if i < len(interv_lines) else ""
    if interv_str and r > 0:
        masks[i, int(interv_str) - 1] = 0.0

# Standardize
data_mean = data.mean(axis=0, keepdims=True)
data_std = data.std(axis=0, keepdims=True) + 1e-8
data = (data - data_mean) / data_std

# Train/val split (80/20)
np.random.seed(SEED)
n_train = int(0.8 * n_samples)
idx = np.random.permutation(n_samples)
x_train, m_train, r_train = data[idx[:n_train]], masks[idx[:n_train]], regimes[idx[:n_train]]
x_val, m_val, r_val = data[idx[n_train:]], masks[idx[n_train:]], regimes[idx[n_train:]]

# ── Stage 1: Train with high sparsity ────────────────────────
print("Stage 1: High-sparsity training (lambda=2.0)")
pacer_stage1 = PACER(
    n_vars=n_vars, n_layers=2, hdim=4, density="gaussian",
    fit_analytic=False, n_steps=2000, lr=0.01, batch_size=64,
    n_mc_samples=500, lambd=2.0, seed=SEED,
)
out1 = pacer_stage1.fit(x_train, m_train, r_train,
                         x_val=x_val, masks_val=m_val, regimes_val=r_val)

# Extract DAG from Stage 1
from pacer.estimators import fine_tune
from pacer.objectives import objective_value_and_grad_fn
import jax

stage1_probs = pacer_stage1.predict_proba()
stage1_dag = (stage1_probs >= THRESHOLD).astype(int)

# ── Stage 2: Fine-tune MLP with fixed DAG ────────────────────
print("Stage 2: Fine-tuning MLP with fixed DAG")
key = jax.random.key(SEED + 100)
params_s1 = out1["best_params"]

objective_args_train = (x_train, m_train, r_train)
objective_args_val = (x_val, m_val, r_val) if x_val is not None else None

out2 = fine_tune(
    key,
    objective_value_and_grad_fn("gaussian"),
    objective_args_train,
    params_s1,
    objective_args_val=objective_args_val,
    n_steps=2000,
    learning_rate=0.01,
    batch_size=64,
    sample_batch_fn=PACER.sample_batch_fn,
    optimize_params=["weight_premask", "bias_premask", "layer_weights", "layer_biases"],
    threshold=THRESHOLD,
    density="gaussian",
)

# Use fine-tuned params for prediction
class PseudoPACER:
    def __init__(self, out):
        self.out = out
    def predict_proba(self):
        out = self.out
        bernoulli_probs = jax.nn.sigmoid(out["best_params"]["bernoulli_logits"])
        logits = jnp.exp(out["best_params"]["logits"])
        n = len(logits)
        prob_edge_direction = logits[None, :] / (logits[:, None] + logits[None, :])
        mask = 1.0 - jnp.eye(n, dtype=prob_edge_direction.dtype)
        prob_edge_direction = prob_edge_direction * mask
        edge_probs = np.array(prob_edge_direction * bernoulli_probs).T
        return edge_probs
    def predict(self, threshold=0.5):
        edge_probs = self.predict_proba()
        return (edge_probs >= threshold).astype(int)

pacer = PseudoPACER(out2)

# ── Predict ─────────────────────────────────────────────────
pred_dag = pacer.predict(threshold=THRESHOLD)

# ── Compute metrics ─────────────────────────────────────────
tp = int(np.sum((pred_dag == 1) & (dag_gt == 1)))
fp = int(np.sum((pred_dag == 1) & (dag_gt == 0)))
fn = int(np.sum((pred_dag == 0) & (dag_gt == 1)))
shd = fp + fn
fdr = fp / max(tp + fp, 1)
tpr = tp / max(tp + fn, 1)
prec = tp / max(tp + fp, 1)
f1 = 2 * prec * tpr / max(prec + tpr, 1e-8)

# SID computation
def compute_sid(pred, gt):
    d = pred.shape[0]
    p = (pred != 0).astype(int); g = (gt != 0).astype(int)
    np.fill_diagonal(p, 0); np.fill_diagonal(g, 0)

    def descendants(adj, node):
        desc = set(); visited = set()
        def dfs(v):
            if v in visited: return
            visited.add(v)
            for c in np.where(adj[v, :] == 1)[0]:
                if c not in visited: desc.add(c); dfs(c)
        dfs(node)
        return desc

    pd = [descendants(p, v) for v in range(d)]
    gd = [descendants(g, v) for v in range(d)]

    sid = 0
    for i in range(d):
        for j in range(d):
            if i == j: continue
            pa_p = set(np.where(p[:, j] == 1)[0])
            pa_g = set(np.where(g[:, j] == 1)[0])
            if pa_p != pa_g: sid += 1
            elif (pd[i] & pa_p) != (gd[i] & pa_g): sid += 1
    return sid

sid = compute_sid(pred_dag, dag_gt)

# ── Report ──────────────────────────────────────────────────
results = {
    "SHD": shd, "SID": sid,
    "FDR": round(fdr, 4), "TPR": round(tpr, 4), "F1": round(f1, 4),
    "TP": tp, "FP": fp, "FN": fn,
    "predicted_edges": int(pred_dag.sum()),
    "ground_truth_edges": int(dag_gt.sum()),
}

print(json.dumps(results, indent=2))

with open("/repo/sachs_results.json", "w") as f:
    json.dump(results, f, indent=2)
np.save("/repo/pred_dag_eval.npy", pred_dag.astype(int))
