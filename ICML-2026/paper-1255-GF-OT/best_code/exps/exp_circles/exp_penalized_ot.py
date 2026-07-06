import os
import torch
import numpy as np

from src.penalized_ot import PenalizedOT
from src.datagen import get_nested_circles

(X, Y), (S_X, S_Y) = get_nested_circles(
    n_x=250,
    n_y=25,
    p_X0=0.5,
    p_Y0=0.5,
    noise_0=0.15,
    noise_1=0.2,
    diameter=4.0,
    rng=42,
    n_outliers_x=4,
    n_outliers_y=2,
)

eps = 1.0
cost_matrix = torch.sum((X[:, None, :] - Y[None, :, :]) ** 2, dim=2)
F_target = torch.tensor([[0.2, 0.3], [0.28, 0.22]])


# Penalized OT with quota loss
penalty_grid = np.logspace(0, 3, 80)
penalized_ot = PenalizedOT(
    penalty_grid=penalty_grid,
    entropic_grid=[eps],
    fairness_loss="quota_loss",
)
# Solving the penalized OT problem
results = penalized_ot.solve(X=X, Y=Y, S_X=S_X, S_Y=S_Y, F=F_target, n_jobs=-1)
os.makedirs("results/exp_circles/", exist_ok=True)
results.to_pickle("results/exp_circles/results_penalized.pkl")

print("Experiment finished and results saved at results/exp_circles/")
