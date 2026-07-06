import os
import torch
import numpy as np
from src.penalized_ot import PenalizedOT
from src.datagen import get_gaussian_mixture
import time

print("Testing full pipeline with 1 penalty value (n_jobs=1)...")
(X, Y), (S_X, S_Y) = get_gaussian_mixture(
    d=2, n_x=250, n_y=25, scale=0.2, p_x0=0.5, p_y0=0.5,
    centers_X=[np.array([0, 0]), np.array([2.0, 0.0])],
    centers_Y=[np.array([1.0, 1.0]), np.array([2.5, 0.5])],
    rng=42,
)
eps = 1.0
F_target = torch.tensor([[0.2, 0.3], [0.28, 0.22]])

penalty_grid = [100.0]  # Just 1 value
penalized_ot = PenalizedOT(
    penalty_grid=penalty_grid,
    entropic_grid=[eps],
    fairness_loss="quota_loss",
)

print("Calling solve with n_jobs=1...")
start = time.time()
results = penalized_ot.solve(X=X, Y=Y, S_X=S_X, S_Y=S_Y, F=F_target, n_jobs=1, use_cache=False)
elapsed = time.time() - start
print(f"Done in {elapsed:.1f}s!")
print(results[["penalty", "fair_cost", "true_cost", "cost_diff", "fairness_loss_value"]])
