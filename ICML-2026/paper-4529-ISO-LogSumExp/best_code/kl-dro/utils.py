import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import os


def load_and_preprocess_data():
    raw = fetch_california_housing()
    X = raw.data
    y = raw.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tX = torch.from_numpy(X_scaled.astype(np.float32))
    ty = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)
    dataset = TensorDataset(tX, ty)

    return dataset, X_scaled, y


def train_linreg(X, y, filename='linreg_weights.pt'):
    model = LinearRegression(fit_intercept=True).fit(X, y)
    state_dict = {
        'weights': torch.tensor(model.coef_, dtype=torch.float32),
        'bias': torch.tensor(model.intercept_, dtype=torch.float32)
    }
    torch.save(state_dict, filename)
    return state_dict


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim: int, with_alpha: bool = False, init_w: str = 'ridge'):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

        state_dict = torch.load('linreg_weights.pt')
        with torch.no_grad():
            self.linear.weight.copy_(state_dict['weights'].unsqueeze(0))
            self.linear.bias.copy_(state_dict['bias'])

        if with_alpha:
            self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def compute_final_objective_stats(data_tuples_lr, n_seeds=10, data_dir='trajectories'):
    data_tuples_obj = []

    for rho, lam, batch_sz, lr in data_tuples_lr:
        final_objectives = []

        for seed in range(n_seeds):
            if rho is None:
                fname = f'batch_logsumexp_lam{lam}_batch{batch_sz}_lr{lr}_rho{rho}_seed{seed}.pickle'
            else:
                fname = f'softplus_approx_lam{lam}_batch{batch_sz}_lr{lr}_rho{rho}_seed{seed}.pickle'

            filepath = os.path.join(data_dir, fname)

            if os.path.exists(filepath):
                try:
                    with open(filepath, 'rb') as f:
                        epochs_passed, logsumexp_vals = pickle.load(f)

                    if len(logsumexp_vals) > 0:
                        final_obj = logsumexp_vals[-1]
                        final_objectives.append(final_obj)
                    else:
                        print(f"Warning: Empty trajectory in {fname}")

                except Exception as e:
                    print(f"Warning: Could not load {fname}: {e}")
            else:
                print(f"Warning: File {fname} not found")

        if len(final_objectives) >= 2:
            mean_final_obj = np.mean(final_objectives)
            std_final_obj = np.std(final_objectives)
            data_tuples_obj.append((rho, lam, batch_sz, mean_final_obj, std_final_obj))
            print(f"Computed: rho={rho}, lam={lam}, B={batch_sz}, lr={lr}: "
                  f"{len(final_objectives)}/{n_seeds} seeds, "
                  f"mean={mean_final_obj:.4f}, std={std_final_obj:.4f}")
        else:
            print(f"Error: Insufficient data for rho={rho}, lam={lam}, B={batch_sz}, lr={lr}: "
                  f"only {len(final_objectives)} seeds available")

    return data_tuples_obj
