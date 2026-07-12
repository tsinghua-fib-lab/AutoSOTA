"""Final reproduction script for paper 4529 — KL-DRO on California Housing.
Reproduces Table 1, lambda=5, |D|=10, rho=1e-3 configuration.

Paper reports: 0.76 +/- 0.02 (proposed), 0.87 +/- 0.01 (baseline)
"""
import sys, os, pickle, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------
# 1. Load data (from OpenML, same as sklearn fetch_california_housing)
# ---------------------------------------------------------------------------
def load_data():
    data = np.load('/datasets/california_housing_data.npy')
    target = np.load('/datasets/california_housing_target.npy')
    return data, target

def prep_data():
    X, y = load_data()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    tX = torch.from_numpy(Xs.astype(np.float32))
    ty = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)
    return TensorDataset(tX, ty), Xs, y

# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------
class LinRegModel(nn.Module):
    def __init__(self, input_dim, with_alpha=False):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        sd = torch.load('/repo/kl-dro/linreg_weights.pt')
        with torch.no_grad():
            self.linear.weight.copy_(sd['weights'].unsqueeze(0))
            self.linear.bias.copy_(sd['bias'])
        if with_alpha:
            self.alpha = nn.Parameter(torch.zeros(()))
    def forward(self, x):
        return self.linear(x)

# ---------------------------------------------------------------------------
# 3. Objective computation
# ---------------------------------------------------------------------------
def compute_objective(model, X, y, lam):
    with torch.no_grad():
        preds = model(X)
        errors = (preds - y) ** 2
        L = errors.squeeze(1)
    lse = torch.logsumexp(L / lam, dim=0).item() - math.log(len(y))
    return lam * lse

# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------
def train_one_run(dataset, X, y, lam, rho, lr, seed, n_epochs=50):
    torch.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)

    model = LinRegModel(X.shape[1], with_alpha=(rho is not None))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)

    for epoch in range(1, n_epochs + 1):
        for Xb, yb in loader:
            optimizer.zero_grad()
            preds = model(Xb)
            errors = (preds - yb) ** 2
            L = errors.squeeze(1)

            if rho is None:  # baseline: batch logsumexp
                with torch.no_grad():
                    p = torch.softmax(L / lam, dim=0)
                loss = torch.sum(p * L)
            else:  # proposed: softplus approx
                exponent = (L - model.alpha) / lam + math.log(rho)
                loss = (lam / rho) * torch.nn.functional.softplus(exponent).mean() + model.alpha

            loss.backward()
            optimizer.step()

    return compute_objective(model, X, y, lam)

# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    dataset, Xs, y = prep_data()

    # Fit and save least squares init
    lr_model = LinearRegression(fit_intercept=True).fit(Xs, y)
    w = torch.tensor(lr_model.coef_, dtype=torch.float32)
    b = torch.tensor(lr_model.intercept_, dtype=torch.float32)
    torch.save({'weights': w, 'bias': b}, '/repo/kl-dro/linreg_weights.pt')

    X, Y = dataset.tensors
    lam = 5.0
    n_seeds = 10

    # --- Proposed method (rho=1e-3) ---
    # Paper uses grid search over {1e-9,...,1e-4}; we test candidate LRs
    print("=" * 70)
    print("PROPOSED METHOD: rho=1e-3, lambda=5, |D|=10")
    print("=" * 70)

    rho = 1e-3
    best_lr = None
    best_obj = float('inf')

    for lr in [1e-7, 3e-7, 5e-7, 7e-7, 1e-6]:
        results = []
        failed = 0
        for seed in range(n_seeds):
            try:
                obj = train_one_run(dataset, X, Y, lam, rho, lr, seed)
                if np.isfinite(obj):
                    results.append(obj)
                else:
                    failed += 1
            except Exception:
                failed += 1

        if results:
            mean_obj = np.mean(results)
            std_obj = np.std(results)
            print(f"  lr={lr:.0e}: mean={mean_obj:.4f}, std={std_obj:.4f}, "
                  f"n_success={len(results)}/{n_seeds}, n_failed={failed}")
            if len(results) == n_seeds and mean_obj < best_obj:
                best_obj = mean_obj
                best_lr = lr

    print(f"\n  Best LR: {best_lr}, Best mean: {best_obj:.4f}")

    # Re-run with best LR to get final values
    print(f"\n  Final run with lr={best_lr}:")
    final_results_proposed = []
    for seed in range(n_seeds):
        obj = train_one_run(dataset, X, Y, lam, rho, best_lr, seed)
        final_results_proposed.append(obj)
        print(f"    Seed {seed}: {obj:.4f}")

    proposed_mean = np.mean(final_results_proposed)
    proposed_std = np.std(final_results_proposed)
    print(f"  Proposed: {proposed_mean:.4f} +/- {proposed_std:.4f}")

    # --- Baseline ---
    print("\n" + "=" * 70)
    print("BASELINE: lambda=5, |D|=10")
    print("=" * 70)
    baseline_results = []
    for seed in range(n_seeds):
        obj = train_one_run(dataset, X, Y, lam, None, 1e-6, seed)
        baseline_results.append(obj)
        print(f"  Seed {seed}: {obj:.4f}")

    baseline_mean = np.mean(baseline_results)
    baseline_std = np.std(baseline_results)
    print(f"  Baseline: {baseline_mean:.4f} +/- {baseline_std:.4f}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Proposed (rho=1e-3): {proposed_mean:.4f} +/- {proposed_std:.4f}  [paper: 0.76 +/- 0.02]")
    print(f"  Baseline:            {baseline_mean:.4f} +/- {baseline_std:.4f}  [paper: 0.87 +/- 0.01]")
    print(f"  Within CI bounds [0.74, 0.78]: {0.74 <= proposed_mean <= 0.78}")

if __name__ == "__main__":
    main()
