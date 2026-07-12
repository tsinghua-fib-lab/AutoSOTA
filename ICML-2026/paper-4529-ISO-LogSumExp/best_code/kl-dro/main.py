import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pickle
import multiprocessing as mp
from torch.utils.data import DataLoader
import os

from utils import load_and_preprocess_data, LinearRegressionModel, train_linreg, compute_final_objective_stats


def batch_logsumexp(preds: torch.Tensor, targets: torch.Tensor, lam=1.):
    errors = (preds - targets) ** 2
    L = errors.squeeze(1)
    with torch.no_grad():
        p = torch.softmax(L/lam, dim=0)
    loss = torch.sum(p * L)
    return loss


def softplus_approx(preds: torch.Tensor, targets: torch.Tensor, model: LinearRegressionModel, rho: float, lam=1.):
    errors = (preds - targets) ** 2
    L = errors.squeeze(1)
    exponent = (L - model.alpha) / lam + math.log(rho)
    loss = (lam / rho) * F.softplus(exponent).mean() + model.alpha
    return loss


def logsumexp_over_dataset(model: nn.Module, X, y, lam=1.):
    with torch.no_grad():
        preds = model(X)
        errors = (preds - y) ** 2
        L = errors.squeeze(1)
    lse = torch.logsumexp(L/lam, dim=0).item() - math.log(len(y))
    return lam * lse


def train(dataset, batch_sz, lr, rho, seed, lam):
    print(f"Running with lam={lam}, batch_sz={batch_sz}, lr={lr}, rho={rho}, seed={seed}")
    torch.manual_seed(seed)
    train_loader = DataLoader(dataset, batch_size=batch_sz, shuffle=True)
    X, y = dataset.tensors

    model = LinearRegressionModel(input_dim=X.shape[1], with_alpha=rho is not None)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)

    logsumexp_vals = [logsumexp_over_dataset(model, X, y, lam=lam)]
    epochs_passed = [0]
    n_epochs = 50

    for epoch in range(1, n_epochs + 1):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = batch_logsumexp(preds, y_batch, lam=lam) if rho is None \
                else softplus_approx(preds, y_batch, model, rho, lam=lam)
            loss.backward()
            optimizer.step()

        if epoch in [10, 20, 30, 40, 50]:
            obj_val = logsumexp_over_dataset(model, X, y, lam=lam)
            if torch.isfinite(torch.tensor(obj_val)):
                epochs_passed.append(epoch)
                logsumexp_vals.append(obj_val)
            else:
                print(f'inf or nan, batch{batch_sz}_lr{lr}_rho{rho}_seed{seed}')
                return 0

    fname = 'batch_logsumexp' if rho is None else 'softplus_approx'
    fname += f'_lam{lam}_batch{batch_sz}_lr{lr}_rho{rho}_seed{seed}.pickle'
    with open("trajectories/" + fname, "wb") as f:
        pickle.dump((epochs_passed, logsumexp_vals), f)
    print(f"Finished run with lam={lam}, batch_sz={batch_sz}, lr={lr}, rho={rho}, seed={seed}")


def main():
    os.makedirs('trajectories', exist_ok=True)
    dataset, X_np, y_np = load_and_preprocess_data()
    train_linreg(X_np, y_np)  # Store least squares solution for initialization

    params = [  # (rho, lam, batch_sz, lr)
        ### lam = 0.2 ###
        (None, 0.2, 10, 1e-6),
        (1e-1, 0.2, 10, 1e-6),
        (1e-3, 0.2, 10, 1e-7),
        (1e-5, 0.2, 10, 1e-9),

        (None, 0.2, 100, 1e-5),
        (1e-1, 0.2, 100, 1e-5),
        (1e-3, 0.2, 100, 1e-6),
        (1e-5, 0.2, 100, 1e-8),

        (None, 0.2, 1000, 1e-5),
        (1e-1, 0.2, 1000, 1e-5),
        (1e-3, 0.2, 1000, 1e-6),
        (1e-5, 0.2, 1000, 1e-8),

        ### lam = 1 ###
        (None, 1., 10, 1e-6),
        (1e-1, 1., 10, 1e-6),
        (1e-3, 1., 10, 1e-7),
        (1e-5, 1., 10, 1e-9),

        (None, 1., 100, 1e-5),
        (1e-1, 1., 100, 1e-5),
        (1e-3, 1., 100, 1e-6),
        (1e-5, 1., 100, 1e-8),

        (None, 1., 1000, 1e-5),
        (1e-1, 1., 1000, 1e-4),
        (1e-3, 1., 1000, 1e-5),
        (1e-5, 1., 1000, 1e-7),

        ### lam = 5 ###
        (None, 5., 10, 1e-6),
        (1e-1, 5., 10, 1e-6),
        (1e-3, 5., 10, 1e-6),
        (1e-5, 5., 10, 1e-9),

        (None, 5., 100, 1e-6),
        (1e-1, 5., 100, 1e-5),
        (1e-3, 5., 100, 1e-6),
        (1e-5, 5., 100, 1e-8),

        (None, 5., 1000, 1e-5),
        (1e-1, 5., 1000, 1e-4),
        (1e-3, 5., 1000, 1e-5),
        (1e-5, 5., 1000, 1e-7),
    ]

    seeds = list(range(10))
    tasks = [(dataset, batch_sz, lr, rho, seed, lam)
             for rho, lam, batch_sz, lr in params
             for seed in seeds]

    with mp.Pool(processes=3) as pool:
        pool.starmap(train, tasks)

    compute_final_objective_stats(params)


if __name__ == "__main__":
    main()
