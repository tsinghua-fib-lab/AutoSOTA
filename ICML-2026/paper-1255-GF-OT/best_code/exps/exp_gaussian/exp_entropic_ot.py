import os
import torch
import numpy as np
import pandas as pd
from ot import sinkhorn


from src.loss_funcs import quota_loss
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

eps_grid = np.logspace(0, 2, 20)
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
os.makedirs("results/exp_gaussian/", exist_ok=True)
df.to_pickle("results/exp_gaussian/results_entropic_ot.pkl")
