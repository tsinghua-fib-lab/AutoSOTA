import os
import torch
import numpy as np

from src.penalized_ot import PenalizedOT
from src.datagen import get_gaussian_mixture

(X, Y), (S_X, S_Y) = get_gaussian_mixture(
    d=2,
    n_x=250,
    n_y=25,
    scale=0.2,
    p_x0=0.5,
    p_y0=0.5,
    centers_X=[np.array([0, 0]), np.array([2.0, 0.0])],
    centers_Y=[np.array([1.0, 1.0]), np.array([2.5, 0.5])],
    rng=42,
)

eps = 1.0
F_target = torch.tensor([[0.2, 0.3], [0.28, 0.22]])

# Penalized OT with quota loss
penalty_grid = np.logspace(0, 3, 30)
penalized_ot = PenalizedOT(
    penalty_grid=penalty_grid,
    entropic_grid=[eps],
    fairness_loss="quota_loss",
)
# Solving the penalized OT problem (sequential due to CUDA pickle constraints)
results = penalized_ot.solve(X=X, Y=Y, S_X=S_X, S_Y=S_Y, F=F_target, n_jobs=1, use_cache=False)
os.makedirs("results/exp_gaussian/", exist_ok=True)
results.to_pickle("results/exp_gaussian/results_penalized.pkl")

print("Experiment finished and results saved at results/exp_gaussian/")
print("Results summary:")
print(results[["penalty", "fair_cost", "true_cost", "cost_diff", "fairness_loss_value"]].head(10))
