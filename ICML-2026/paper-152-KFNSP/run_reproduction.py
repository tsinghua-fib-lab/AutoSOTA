import sys, os
sys.path.insert(0, "src")
os.chdir("/repo")

import torch
import numpy as np
import json
from datetime import datetime
from time import perf_counter as pc

from data.load_data import load_dataset
from models.decomposition_models import RegularizedSVR
from utils.hgr import hgr, kde
from utils.gdp import gdp
from utils.pairwise_fairness import pairwise_fairness
from sklearn.model_selection import KFold

SEED = 7341293
FOLDS = 5

print("=== SVR-FKD Reproduction on Crimes Dataset ===")
print("Seed: {}, Folds: {}".format(SEED, FOLDS))
print()

# Load dataset
X, y, p = load_dataset("Crime")
print("Dataset: X={}, y={}, p={}".format(X.shape, y.shape, p.shape))

# Fixed hyperparams from paper Table A.1 for SVR on Crime
EPSILON = 0.01
GAMMA = 0.05
C_reg = 0.75
ALPHA_PRIME = 0.05
NYSTROM_COMP = 0.25  # Use 25% landmarks for Nystroem approximation

# Extended m values for full Pareto front exploration
m_values = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200]

def evaluate(y_hat, y_true, prot_attr):
    y_hat_t = torch.tensor(y_hat.astype(np.float32))
    y_true_t = torch.tensor(y_true.astype(np.float32))
    p_t = torch.tensor(prot_attr.astype(np.float32))

    MAE = torch.nn.functional.l1_loss(y_true_t, y_hat_t).item()
    HGR = hgr(y_hat_t, p_t, density=kde).item()
    GDP_val = gdp(y_hat_t, p_t)
    PF_EO = pairwise_fairness(y_true_t, y_hat_t, p_t, use_label=True)

    if hasattr(PF_EO, "item"):
        PF_EO = PF_EO.item()

    return np.array([MAE, HGR, GDP_val, PF_EO])

X_t = torch.tensor(X.astype(np.float32))
y_t = torch.tensor(y.astype(np.float32))
p_t = torch.tensor(p.astype(np.float32))

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

# results[fold_idx][m_idx][metric_idx]
all_results = {}
for m in m_values:
    all_results[m] = []

t0_total = pc()

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_t, y_t)):
    print("\n--- Fold {}/{} ---".format(fold_idx + 1, FOLDS))

    X_train, y_train, p_train = X_t[train_idx], y_t[train_idx], p_t[train_idx]
    X_test, y_test, p_test = X_t[test_idx], y_t[test_idx], p_t[test_idx]

    for m in m_values:
        t0 = pc()
        model = RegularizedSVR(
            alpha_prime=ALPHA_PRIME,
            gamma=GAMMA,
            C=C_reg,
            eps=EPSILON,
            nystrom_comp=NYSTROM_COMP,
        )
        model.train(X_train, y_train, p_train, iterations=m)
        y_pred = model.predict(X_test).flatten()

        metrics = evaluate(y_pred, y_test.numpy(), p_test.numpy())
        elapsed = pc() - t0

        print("  m={:3d}: MAE={:.6f}, HGR={:.6f}, GDP={:.6f}, PF={:.6f}  [{:.1f}s]".format(
            m, metrics[0], metrics[1], metrics[2], metrics[3], elapsed))

        all_results[m].append(metrics)

t_total = pc() - t0_total
print("\n=== Total time: {:.1f}s ===".format(t_total))

# Compute mean and std across folds
print("\n=== Final Results (mean +/- std across {} folds) ===".format(FOLDS))
print("{:>5s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format(
    "m", "MAE", "HGR [DP]", "GDP [DP]", "PF [EO]"))
print("-" * 65)

final = {}
for m in m_values:
    arr = np.array(all_results[m])
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    final[m] = {"mean": mean.tolist(), "std": std.tolist()}
    print("{:5d}  {:10.6f}  {:10.6f}  {:10.6f}  {:10.6f}".format(
        m, mean[0], mean[1], mean[2], mean[3]))
    print("{:5s}  +/-{:.6f}  +/-{:.6f}  +/-{:.6f}  +/-{:.6f}".format(
        "", std[0], std[1], std[2], std[3]))

# Save results
output = {
    "paper": "Extending Fair Null-Space Projections for Continuous Attributes to Kernel Methods",
    "method": "SVR-FKD",
    "dataset": "Crime (Communities & Crimes)",
    "hyperparams": {
        "epsilon": EPSILON,
        "gamma": GAMMA,
        "C": C_reg,
        "alpha_prime": ALPHA_PRIME,
    },
    "folds": FOLDS,
    "seed": SEED,
    "results": final,
    "timestamp": str(datetime.now()),
}

os.makedirs("/repo/results", exist_ok=True)
fname = "/repo/results/reproduction_SVR-FKD_Crime_{}.json".format(
    datetime.now().strftime("%Y%m%d_%H%M%S"))
with open(fname, "w") as f:
    json.dump(output, f, indent=4)
print("\nResults saved to: {}".format(fname))
