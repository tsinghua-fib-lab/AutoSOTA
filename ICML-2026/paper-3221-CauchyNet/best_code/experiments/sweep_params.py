"""
Parameter sweep for CauchyNet across 1D and multi-dim tasks.
Tests: lambda_std, epsilon, learning_rate, hidden_size
Goal: find patterns for what works in 1D vs multi-dim.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_diabetes, fetch_california_housing, fetch_openml
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from itertools import product as iproduct

# ── CauchyNet with configurable init ──────────────────────────────

class ReciprocalActivation(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return 1.0 / (x + self.eps)

class CauchyNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, eps=1e-8, lambda_std=0.1):
        super().__init__()
        self.lambda_ = nn.Parameter(
            torch.normal(mean=0.0, std=lambda_std, size=(hidden_size, output_size), dtype=torch.cfloat)
        )
        angles = 2 * np.pi * torch.rand(hidden_size, input_size)
        real_part = 2 * np.pi * torch.cos(angles)
        imaginary_part = torch.sin(angles)
        self.xi = nn.Parameter(torch.complex(real_part, imaginary_part))
        self.activation = ReciprocalActivation(eps=eps)

    def forward(self, x):
        xc = torch.complex(x, torch.zeros_like(x)).unsqueeze(1)
        reciprocals = self.activation(self.xi - xc)
        activated = reciprocals.prod(dim=-1)
        y = torch.matmul(activated, self.lambda_)
        return y.real.squeeze(-1).unsqueeze(-1), y.imag.squeeze(-1).unsqueeze(-1)

# ── 1D target functions ───────────────────────────────────────────

def target_paper(x):
    return (1.0/((x+0.6)**2+0.005) - 40*np.exp(-2*(x+0.4)**2)
            + 50*np.sign(x)*np.abs(np.sin(3*x)+0.8)**1.5*np.sin(10*x))

def target_near_singular(x):
    return 4.0/((x-0.5)**2 + 0.01) + 2.0/((x+0.3)**2 + 0.02)

# ── Training helper ───────────────────────────────────────────────

def train_1d(target_fn, hidden_size, eps, lambda_std, lr, epochs=200, n_points=50, n_seeds=5):
    """Train CauchyNet on 1D target, return mean MSE across seeds."""
    x_np = np.linspace(-1, 1, n_points).astype(np.float32)
    y_np = target_fn(x_np).astype(np.float32)

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y_np.reshape(-1,1)).flatten().astype(np.float32)

    indices = np.arange(n_points)
    rng = np.random.RandomState(0)
    rng.shuffle(indices)
    n_train = int(0.5 * n_points)
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    X_tr = torch.tensor(x_np[train_idx]).unsqueeze(-1)
    Y_tr = torch.tensor(y_scaled[train_idx])
    X_te = torch.tensor(x_np[test_idx]).unsqueeze(-1)
    Y_te_raw = y_np[test_idx]

    mses = []
    for seed in range(42, 42 + n_seeds):
        torch.manual_seed(seed)
        model = CauchyNet(1, hidden_size, 1, eps=eps, lambda_std=lambda_std)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        criterion = nn.MSELoss()

        best_loss, best_state = float('inf'), None
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            y_real, y_imag = model(X_tr)
            loss = criterion(y_real, Y_tr.unsqueeze(-1)) + 0.1 * criterion(y_imag, torch.zeros_like(y_imag))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if best_state: model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            preds_scaled = model(X_te)[0].numpy().flatten()
        preds = scaler_y.inverse_transform(preds_scaled.reshape(-1,1)).flatten()
        mses.append(float(mean_squared_error(Y_te_raw, preds)))

    return np.mean(mses), np.std(mses)


def train_tabular(X, y, hidden_size, eps, lambda_std, lr, epochs=200, n_folds=3):
    """Train CauchyNet on tabular data, return mean MSE across folds."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    mses = []
    input_size = X.shape[1]

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_train = scaler_X.fit_transform(X[train_idx]).astype(np.float32)
        X_test = scaler_X.transform(X[test_idx]).astype(np.float32)
        y_train = scaler_y.fit_transform(y[train_idx].reshape(-1,1)).flatten().astype(np.float32)
        y_test_raw = y[test_idx]

        X_tr = torch.tensor(X_train)
        Y_tr = torch.tensor(y_train)
        X_te = torch.tensor(X_test)

        torch.manual_seed(42 + fold)
        model = CauchyNet(input_size, hidden_size, 1, eps=eps, lambda_std=lambda_std)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        criterion = nn.MSELoss()

        train_ds = torch.utils.data.TensorDataset(X_tr, Y_tr)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)

        best_loss, best_state = float('inf'), None
        for epoch in range(epochs):
            model.train()
            epoch_loss, n = 0.0, 0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                y_real, y_imag = model(xb)
                loss = criterion(y_real, yb.unsqueeze(-1)) + 0.1 * criterion(y_imag, torch.zeros_like(y_imag))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item(); n += 1
            scheduler.step()
            avg = epoch_loss / n
            if avg < best_loss:
                best_loss = avg
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if best_state: model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            preds_scaled = model(X_te)[0].numpy().flatten()
        preds = scaler_y.inverse_transform(preds_scaled.reshape(-1,1)).flatten()
        mses.append(float(mean_squared_error(y_test_raw, preds)))

    return np.mean(mses), np.std(mses)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    # Parameter grid
    lambda_stds = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    epsilons = [1e-8, 1e-4, 1e-2, 1e-1, 1.0]
    lrs = [0.001, 0.005, 0.01, 0.05]
    hidden_sizes = [32, 64, 128]

    if mode in ("1d", "all"):
        print("=" * 80)
        print("1D SWEEP: Paper target function (d=1)")
        print("=" * 80)

        # Sweep lambda_std x eps (fix lr=0.01, h=64)
        print("\n--- lambda_std x eps (lr=0.01, h=64) ---")
        print(f"{'lambda_std':>12} {'eps':>10} {'MSE_mean':>12} {'MSE_std':>12}")
        best_1d = (float('inf'), None)
        for lstd, ep in iproduct(lambda_stds, epsilons):
            m, s = train_1d(target_paper, 64, ep, lstd, 0.01)
            tag = " *" if m < best_1d[0] else ""
            print(f"{lstd:>12.3f} {ep:>10.0e} {m:>12.6f} {s:>12.6f}{tag}")
            if m < best_1d[0]: best_1d = (m, (lstd, ep))
        print(f"Best: lambda_std={best_1d[1][0]}, eps={best_1d[1][1]}, MSE={best_1d[0]:.6f}")

        # Sweep lr (fix best lambda_std, eps, h=64)
        best_lstd, best_ep = best_1d[1]
        print(f"\n--- lr sweep (lambda_std={best_lstd}, eps={best_ep}, h=64) ---")
        print(f"{'lr':>10} {'MSE_mean':>12} {'MSE_std':>12}")
        best_lr_1d = (float('inf'), None)
        for lr in lrs:
            m, s = train_1d(target_paper, 64, best_ep, best_lstd, lr)
            tag = " *" if m < best_lr_1d[0] else ""
            print(f"{lr:>10.4f} {m:>12.6f} {s:>12.6f}{tag}")
            if m < best_lr_1d[0]: best_lr_1d = (m, lr)
        print(f"Best lr: {best_lr_1d[1]}, MSE={best_lr_1d[0]:.6f}")

        # Sweep h (fix best params)
        best_lr = best_lr_1d[1]
        print(f"\n--- h sweep (lambda_std={best_lstd}, eps={best_ep}, lr={best_lr}) ---")
        print(f"{'h':>6} {'MSE_mean':>12} {'MSE_std':>12}")
        for h in hidden_sizes:
            m, s = train_1d(target_paper, h, best_ep, best_lstd, best_lr)
            print(f"{h:>6d} {m:>12.6f} {s:>12.6f}")

        # Also test near-singular with best 1D params
        print(f"\n--- Near-singular target (lambda_std={best_lstd}, eps={best_ep}, lr={best_lr}, h=64) ---")
        m, s = train_1d(target_near_singular, 64, best_ep, best_lstd, best_lr)
        print(f"MSE={m:.6f} +/- {s:.6f}")

    if mode in ("tabular", "all"):
        print("\n" + "=" * 80)
        print("TABULAR SWEEP: UCI datasets (d=8,10,11)")
        print("=" * 80)

        # Load datasets, subsample to n=50
        datasets = {}
        data = load_diabetes()
        datasets['Diabetes(d=10)'] = (data.data.astype(np.float32), data.target.astype(np.float32))
        data = fetch_california_housing()
        datasets['California(d=8)'] = (data.data.astype(np.float32), data.target.astype(np.float32))
        try:
            data = fetch_openml(name='wine-quality-red', version=1, as_frame=False, parser='auto')
            datasets['Wine(d=11)'] = (data.data.astype(np.float32), data.target.astype(np.float32))
        except:
            pass

        for ds_name, (X_full, y_full) in datasets.items():
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_full), 50, replace=False)
            X, y = X_full[idx], y_full[idx]
            d = X.shape[1]

            print(f"\n{'='*60}")
            print(f"Dataset: {ds_name} (n={len(X)})")
            print(f"{'='*60}")

            # Sweep lambda_std x eps (fix lr=0.01, h=64)
            print(f"\n--- lambda_std x eps (lr=0.01, h=64) ---")
            print(f"{'lambda_std':>12} {'eps':>10} {'MSE_mean':>12} {'MSE_std':>12}")
            best_tab = (float('inf'), None)
            for lstd, ep in iproduct(lambda_stds, epsilons):
                m, s = train_tabular(X, y, 64, ep, lstd, 0.01)
                tag = " *" if m < best_tab[0] else ""
                print(f"{lstd:>12.3f} {ep:>10.0e} {m:>12.4f} {s:>12.4f}{tag}")
                if m < best_tab[0]: best_tab = (m, (lstd, ep))
            print(f"Best: lambda_std={best_tab[1][0]}, eps={best_tab[1][1]}, MSE={best_tab[0]:.4f}")

            # Sweep lr
            best_lstd, best_ep = best_tab[1]
            print(f"\n--- lr sweep (lambda_std={best_lstd}, eps={best_ep}, h=64) ---")
            print(f"{'lr':>10} {'MSE_mean':>12} {'MSE_std':>12}")
            best_lr_tab = (float('inf'), None)
            for lr in lrs:
                m, s = train_tabular(X, y, 64, ep, best_lstd, lr)
                tag = " *" if m < best_lr_tab[0] else ""
                print(f"{lr:>10.4f} {m:>12.4f} {s:>12.4f}{tag}")
                if m < best_lr_tab[0]: best_lr_tab = (m, lr)
            print(f"Best lr: {best_lr_tab[1]}, MSE={best_lr_tab[0]:.4f}")

            # Sweep h
            best_lr = best_lr_tab[1]
            print(f"\n--- h sweep (lambda_std={best_lstd}, eps={best_ep}, lr={best_lr}) ---")
            print(f"{'h':>6} {'MSE_mean':>12} {'MSE_std':>12}")
            for h in hidden_sizes:
                m, s = train_tabular(X, y, h, best_ep, best_lstd, best_lr)
                print(f"{h:>6d} {m:>12.4f} {s:>12.4f}")

    print("\n\nDONE.")

if __name__ == '__main__':
    main()
