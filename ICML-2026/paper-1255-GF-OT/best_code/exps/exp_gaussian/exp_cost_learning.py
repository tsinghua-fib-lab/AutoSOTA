import os
import torch
import numpy as np

from src.cost_learning import TorchFairCostOT
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

# Cost learning with Mahalanobis cost
penalty_grid_mahalanobis = np.logspace(0, 4, 80)
lr_grid_mahalanobis = [1e-1]
cost_learning_mahalanobis = TorchFairCostOT(
    penalty_grid=penalty_grid_mahalanobis,
    entropic_grid=[eps],
    lr_grid=lr_grid_mahalanobis,
    fairness_loss="quota_loss",
    verbose=True,
    optimizer="Adam",
    cost_model_name="mahalanobis",
)
results_mahalanobis = cost_learning_mahalanobis.solve(
    X=X,
    Y=Y,
    S_X=S_X,
    S_Y=S_Y,
    F=F_target,
    n_iter=2000,
    tol=1e-6,
    n_jobs=-1,
    auto_stop=False,
)

results_mahalanobis["final_fairness"] = results_mahalanobis[
    "fairness_loss_value"
].apply(lambda x: x[-1])
results_mahalanobis["final_loss"] = results_mahalanobis["loss"].apply(
    lambda x: x[-1]
)

# Pretraining of the MLP model with \lambda=0 if not already done
if not os.path.exists("exps/exp_gaussian/pretrained_mlp.pt"):
    cost_learning_mlp = TorchFairCostOT(
        penalty_grid=[0.0],
        entropic_grid=[eps],
        lr_grid=[1e-2],
        fairness_loss="quota_loss",
        verbose=True,
        optimizer="Adam",
        cost_model_name="mlp",
        d_hidden=4,
        d_out=2,
        n_layers=2,
    )

    results_mlp = cost_learning_mlp.solve(
        X=X,
        Y=Y,
        S_X=S_X,
        S_Y=S_Y,
        F=F_target,
        n_iter=2000,
        tol=1e-12,
        n_jobs=-1,
    )

    torch.save(
        results_mlp["model"].iloc[0].state_dict(),
        "exps/exp_gaussian/pretrained_mlp.pt",
    )

# Cost learning with MLP cost
penalty_grid_mlp = np.logspace(0, 4, 80)
lr_grid_mlp = [5e-2]
cost_learning_mlp = TorchFairCostOT(
    penalty_grid=penalty_grid_mlp,
    entropic_grid=[eps],
    lr_grid=lr_grid_mlp,
    fairness_loss="quota_loss",
    verbose=True,
    optimizer="Adam",
    cost_model_name="mlp",
    d_hidden=4,
    d_out=2,
    n_layers=2,
)

results_mlp = cost_learning_mlp.solve(
    X=X,
    Y=Y,
    S_X=S_X,
    S_Y=S_Y,
    F=F_target,
    n_iter=2000,
    tol=1e-12,
    pretrained_weights="exps/exp_gaussian/pretrained_mlp.pt",
    n_jobs=-1,
)

results_mlp["final_fairness"] = results_mlp["fairness_loss_value"].apply(
    lambda x: x[-1]
)
results_mlp["final_loss"] = results_mlp["loss"].apply(lambda x: x[-1])

os.makedirs("results/exp_gaussian/", exist_ok=True)
results_mahalanobis.to_pickle("results/exp_gaussian/results_mahalanobis.pkl")
results_mlp.to_pickle("results/exp_gaussian/results_mlp.pkl")

print("Experiment finished and results saved at results/exp_gaussian/")
