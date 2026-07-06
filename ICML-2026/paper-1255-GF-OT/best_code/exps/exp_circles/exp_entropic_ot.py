import os
import torch
import numpy as np
import pandas as pd
from ot import sinkhorn

from src.loss_funcs import quota_loss
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

eps_grid = np.logspace(0, 2, 40)
cost_matrix = torch.sum((X[:, None, :] - Y[None, :, :]) ** 2, dim=2)

ot_plan_list = [
    sinkhorn(
        a=torch.ones(X.shape[0]) / X.shape[0],
        b=torch.ones(Y.shape[0]) / Y.shape[0],
        M=cost_matrix,
        reg=eps,
    )
    for eps in eps_grid
]

F_target = torch.tensor([[0.2, 0.3], [0.28, 0.22]])
transport_costs = [
    (cost_matrix * ot_plan).sum().item()
    - (cost_matrix * ot_plan_list[0]).sum().item()
    for ot_plan in ot_plan_list
]
fairness_values = [
    quota_loss(ot_plan, S_X, S_Y, F_target) for ot_plan in ot_plan_list
]

df = pd.DataFrame(
    {
        "fairness_loss_value": [fv.item() for fv in fairness_values],
        "cost_diff": transport_costs,
        "penalty": eps_grid,
        "fair_ot_plan": ot_plan_list,
    }
)
os.makedirs("results/exp_circles/", exist_ok=True)
df.to_pickle("results/exp_circles/results_entropic_ot.pkl")
