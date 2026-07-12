import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import os
import pickle

from main import batch_logsumexp, softplus_approx


class LinRegModel(nn.Module):
    def __init__(self, input_dim: int, with_alpha: bool = False):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)

        if with_alpha:
            self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def memory_efficient_wls(X, y, p, ridge=1e-9):
    XtX = torch.mm(X.T, p.unsqueeze(1) * X)
    Xty = torch.mm(X.T, (p * y).unsqueeze(1))
    eye = torch.eye(XtX.size(0), device=X.device, dtype=X.dtype)
    XtX.add_(eye, alpha=ridge)

    theta_star = torch.linalg.solve(XtX, Xty).squeeze()
    return theta_star


def compute_duality_gap(model, X, y, lam, ridge=1e-9):
    model.eval()
    with torch.no_grad():
        preds = model(X)
        losses = (preds - y).squeeze() ** 2  # shape (n,)

        opt_gap = lam * torch.logsumexp(losses / lam, dim=0)
        p = F.softmax(losses / lam, dim=0)

        theta_star = memory_efficient_wls(X, y.squeeze(), p, ridge=ridge)
        pred_star = torch.mm(X, theta_star.unsqueeze(1))
        losses_star = (pred_star - y).squeeze() ** 2  # shape (n,)
        gap = torch.dot(p, losses - losses_star)

    return gap.item(), opt_gap.item() - math.log(X.shape[0])


def train(train_dataset, lam, batch_sz, rho, lr, seed, n_epochs=30, gap_every=3):
    torch.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True)
    X, y = train_dataset.tensors

    model = LinRegModel(input_dim=X.shape[1], with_alpha=rho is not None)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    fname = f'lam{lam}_batch{batch_sz}_rho{rho}_lr{lr}_seed{seed}'
    losses = []

    gap, opt_gap = compute_duality_gap(model, X, y, lam, ridge=1e-9)
    print(f'{fname} | epoch 0, duality gap: {gap:.3f}, opt gap: {opt_gap:.6f}')
    gaps = [gap]
    times = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.
        start = time.time()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)

            loss = batch_logsumexp(preds, y_batch, lam=lam) if rho is None \
                else softplus_approx(preds, y_batch, model, rho, lam=lam)

            if not torch.isfinite(loss):
                print(f'inf or nan, lam{lam}_batch{batch_sz}_rho{rho}_lr{lr:.0e}_seed{seed}')
                return None

            epoch_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.)
            optimizer.step()

        times.append(time.time() - start)
        losses.append(epoch_loss / (X.shape[0] // batch_sz))  # 512116
        print(f'{fname} | epoch {epoch}, loss {losses[-1]:.3f}')

        if epoch % gap_every == 0 or epoch == n_epochs:
            gap, opt_gap = compute_duality_gap(model, X, y, lam, ridge=1e-12)
            print(f'{fname} | epoch {epoch}, duality gap: {gap:.6f}, opt gap: {opt_gap:.6f}')
            gaps.append(gap)

    return gaps


def generate_data(n, d, outlier_fraction=0.01, outlier_scale=1.):
    torch.manual_seed(0)
    X = torch.randn(n, d)
    true_theta = torch.randn(d, 1)
    y = (X @ true_theta) + 0.1 * torch.randn(n, 1)

    num_outliers = int(n * outlier_fraction)
    outlier_indices = torch.randperm(n)[:num_outliers]
    y[outlier_indices] += outlier_scale * torch.randn(num_outliers, 1)

    return TensorDataset(X, y)


def main():
    n, d = 1000, 50
    train_dataset = generate_data(n, d)
    lam, batch_sz = 1., 10
    lr = 1e-3
    n_epochs = 60
    gap_every = n_epochs // 15
    epochs_x = np.arange(0, n_epochs + 1, gap_every)
    results_file = "rho_results.pkl"

    if not os.path.exists(results_file):
        print("Running experiments...")
        rho_results = {}
        for rho in [1e-3, 1e-5]:
            all_seed_gaps = []
            for seed in range(5):
                gaps = train(train_dataset, lam, batch_sz, rho, lr, seed, n_epochs=n_epochs, gap_every=gap_every)
                all_seed_gaps.append(gaps)

            all_seed_gaps = np.array(all_seed_gaps)
            rho_results[rho] = {
                'mean': np.mean(all_seed_gaps, axis=0),
                'min': np.min(all_seed_gaps, axis=0),
                'max': np.max(all_seed_gaps, axis=0)
            }

        with open(results_file, 'wb') as f:
            pickle.dump(rho_results, f)
    else:
        print("Loading results from file...")
        with open(results_file, 'rb') as f:
            rho_results = pickle.load(f)

    plt.figure(figsize=(8, 5))
    for rho, data in rho_results.items():
        exponent = int(np.log10(rho))
        label = fr'$\rho = 10^{{{exponent}}}$'
        line = plt.plot(epochs_x, data['mean'], label=label)
        plt.fill_between(
            epochs_x,
            data['min'],
            data['max'],
            color=line[0].get_color(),
            alpha=0.2
        )

    plt.xlabel('Epoch')
    plt.ylabel('Duality Gap')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.8)
    plt.savefig('duality_gap_shaded.pdf', bbox_inches='tight')


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    main()
