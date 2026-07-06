import os
import torch
import numpy as np
import pandas as pd
from ot import sinkhorn

from src.loss_funcs import quota_loss
from src.solvers import fair_sinkhorn_knopp
from src.datagen import get_gaussian_mixture

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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

F_target = torch.tensor([[0.2, 0.3], [0.28, 0.22]])

eps = 1.0

a = torch.ones((X.shape[0],)) / X.shape[0]
b = torch.ones((Y.shape[0],)) / Y.shape[0]

cost = torch.sum((X[:, None, :] - Y[None, :, :]) ** 2, dim=2)

# Running FairSinkhorn
fair_sinkhorn_results = fair_sinkhorn_knopp(
    a=a, b=b, M=cost, F=F_target, S_X=S_X, S_Y=S_Y, reg=eps, log=True
)

# Running Sinkhorn
unfair_sinkhorn_results = sinkhorn(a, b, cost, reg=eps, log=True)


# Subsample only in the low fairness loss regime to reduce overcrowding

results_penalized = pd.read_pickle(
    "results/exp_gaussian/results_penalized.pkl"
)
low_fairness_mask = results_penalized["fairness_loss_value"] < 0.007
low_fairness = results_penalized[low_fairness_mask].iloc[::6]
high_fairness = results_penalized[~low_fairness_mask]
results_penalized = pd.concat([low_fairness, high_fairness])

cost_fair_sinkhorn = cost * fair_sinkhorn_results[0]
cost_unfair_sinkhorn = cost * unfair_sinkhorn_results[0]

f, ax = plt.subplots(1, 1, figsize=(5, 4), sharey=True, sharex=True)

# Plotting penalized OT results
scatter = ax.scatter(
    results_penalized["fairness_loss_value"],
    results_penalized["cost_diff"],
    c=results_penalized["penalty"],
    cmap="viridis",
    norm=LogNorm(),
    s=100,
    edgecolor="black",
)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Fairness penalty", rotation=270, labelpad=15, fontsize=14)
cbar.ax.tick_params(labelsize=14)

# Add a dummy scatter for legend
ax.scatter(
    [], [], s=100, edgecolor="black", facecolor="gray", label="Penalized OT"
)
# Add stars for FairSinkhorn and Sinkhorn
ax.scatter(
    quota_loss(fair_sinkhorn_results[0], S_X, S_Y, F_target),
    cost_fair_sinkhorn.sum().item() - cost_unfair_sinkhorn.sum().item(),
    marker="*",
    s=150,
    color="red",
    edgecolor="black",
    label="FairSinkhorn",
    zorder=10,
)
ax.scatter(
    quota_loss(unfair_sinkhorn_results[0], S_X, S_Y, F_target),
    0.0,
    marker="*",
    s=150,
    color="yellow",
    edgecolor="black",
    label="Sinkhorn",
    zorder=10,
)
plt.ylabel("Cost difference", fontsize=14)
plt.xlabel("Fairness loss", fontsize=14)
ax.tick_params(axis="both", labelsize=14)

plt.legend(loc="upper right", fontsize=14)

plt.tight_layout()

os.makedirs("figures/exp_gaussian/", exist_ok=True)
plt.savefig("figures/exp_gaussian/exp_fairsinkhorn.pdf", bbox_inches="tight")
