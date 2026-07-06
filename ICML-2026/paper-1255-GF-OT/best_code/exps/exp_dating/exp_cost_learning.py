import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from src.cost_learning import TorchFairCostOT
from custom_distance import pre_compute_distance_matrix, table_distance

torch.manual_seed(0)
rng = np.random.RandomState(0)

data_path = Path("data/processed_dating_data.csv")
match_matrix_path = Path("data/possible_individual_matches.npy")

if not data_path.exists():
    raise FileNotFoundError(
        "Data file not found. The preprocessing step has not been run."
    )

if not match_matrix_path.exists():
    raise FileNotFoundError("The matching matrix has not been generated")

df = pd.read_csv("data/processed_dating_data.csv", index_col=0)
match_matrix = np.load("data/possible_individual_matches.npy")

n_individuals = len(df)

idx_X = rng.choice(n_individuals, size=n_individuals // 2, replace=False)
idx_Y = np.setdiff1d(np.arange(n_individuals), idx_X)

df_X = df.iloc[idx_X]
df_Y = df.iloc[idx_Y]

distance_matrix = pre_compute_distance_matrix(
    df_X, df_Y, match_matrix, idx_x=idx_X, idx_y=idx_Y
)
cost_matrix = distance_matrix.sum(dim=-1)

income_levels = sorted(df["income_bracket_encoded"].unique())
income_values = pd.Index(df["income_bracket"].unique())
n_income = len(income_levels)

# S_X[i, k] = 1 if individual i in X has education level income_levels k
S_X = torch.zeros(len(idx_X), n_income)
S_Y = torch.zeros(len(idx_Y), n_income)
for k, e in enumerate(income_levels):
    S_X[:, k] = torch.tensor(
        (df_X["income_bracket_encoded"].values == e).astype(float)
    )

    S_Y[:, k] = torch.tensor(
        (df_Y["income_bracket_encoded"].values == e).astype(float)
    )

eps = 0.1
F_target_incomes = torch.from_numpy(
    np.outer(S_X.mean(axis=0), S_Y.mean(axis=0))
)

# Cost learning with weighted cost
penalty_grid = np.logspace(0, 3, 17)

cost_learning = TorchFairCostOT(
    penalty_grid=penalty_grid,
    entropic_grid=[eps],
    lr_grid=[1e-3],
    fairness_loss="quota_loss",
    verbose=True,
    optimizer="SGD",
    cost_model_name="weighted_distance",
    n_features=9,
    table_distance=table_distance,
    pre_computed_matrix=distance_matrix,
    match_matrix=match_matrix,
)

# Solving the penalized OT problem
results = cost_learning.solve(
    X=df_X,
    Y=df_Y,
    S_X=S_X.argmax(axis=1),
    S_Y=S_Y.argmax(axis=1),
    F=F_target_incomes,
    cost_matrix=cost_matrix,
    n_jobs=1,
    id_X=idx_X,
    id_Y=idx_Y,
)
os.makedirs("results/exp_dating/", exist_ok=True)
results.drop(columns=["model"]).to_pickle(
    "results/exp_dating/results_cost_learning.pkl"
)
print("Experiment finished and results saved at results/exp_dating/")
