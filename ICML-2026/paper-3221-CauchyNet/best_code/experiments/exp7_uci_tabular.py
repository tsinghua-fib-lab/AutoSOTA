"""
Experiment 7: UCI Tabular Regression Datasets
=============================================
UCI tabular regression benchmarks.

Datasets:
  - Diabetes (sklearn, d=10)
  - California Housing (sklearn, d=8)
  - Wine Quality (UCI via sklearn fetch_openml, d=11)

CauchyNet, FNN (ReLU), and Hybrid FNN(4)->Cauchy(32).
5-fold cross-validation, n=100 per dataset.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_diabetes, fetch_california_housing, fetch_openml
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from shared import (
    CauchyNet, FNN, HybridFNNCauchy, count_real_params,
    cauchy_real_params, matched_fnn_hidden,
    BATCH_SIZE, LR, WEIGHT_DECAY, EPOCHS, IMAG_PENALTY
)


# ──────────────────────────────────────────────────────────────────
# Dataset loaders
# ──────────────────────────────────────────────────────────────────

def load_datasets():
    datasets = {}

    # Diabetes (d=10, n=442)
    data = load_diabetes()
    datasets['Diabetes'] = (data.data.astype(np.float32), data.target.astype(np.float32))

    # California Housing (d=8, n=20640)
    data = fetch_california_housing()
    datasets['California'] = (data.data.astype(np.float32), data.target.astype(np.float32))

    # Wine Quality (d=11, n=~6500)
    try:
        data = fetch_openml(name='wine-quality-red', version=1, as_frame=False, parser='auto')
        datasets['Wine'] = (data.data.astype(np.float32), data.target.astype(np.float32))
    except Exception as e:
        print(f"Warning: Could not load Wine Quality dataset: {e}")
        print("Using synthetic wine-like data as fallback.")
        np.random.seed(123)
        n = 1599
        X_wine = np.random.randn(n, 11).astype(np.float32)
        y_wine = (
            3.0 * np.sin(X_wine[:, 0]) + 2.0 * X_wine[:, 1] ** 2 +
            1.5 * X_wine[:, 2] * X_wine[:, 3] + 0.5 * np.exp(-X_wine[:, 4] ** 2) +
            np.random.randn(n).astype(np.float32) * 0.3 + 5.0
        )
        datasets['Wine(synth)'] = (X_wine, y_wine)

    return datasets


# ──────────────────────────────────────────────────────────────────
# K-fold CV training
# ──────────────────────────────────────────────────────────────────

def train_eval_kfold(model_cls, X, y, hidden_size=64, n_folds=5,
                     epochs=EPOCHS, lr=LR, device='cpu', cauchy_eps=1e-8):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    mse_scores = []
    mae_scores = []

    input_size = X.shape[1]
    num_params = None

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        # Scale features and targets to [0,1]
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_train = scaler_X.fit_transform(X[train_idx]).astype(np.float32)
        X_test = scaler_X.transform(X[test_idx]).astype(np.float32)
        y_train = scaler_y.fit_transform(y[train_idx].reshape(-1, 1)).flatten().astype(np.float32)
        y_test_raw = y[test_idx]

        X_tr = torch.tensor(X_train, dtype=torch.float32)
        Y_tr = torch.tensor(y_train, dtype=torch.float32)
        X_te = torch.tensor(X_test, dtype=torch.float32)

        torch.manual_seed(42 + fold)
        if model_cls == CauchyNet:
            model = model_cls(input_size, hidden_size, 1, eps=cauchy_eps).to(device)
        else:
            model = model_cls(input_size, hidden_size, 1).to(device)

        if num_params is None:
            num_params = count_real_params(model)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        train_ds = torch.utils.data.TensorDataset(X_tr, Y_tr)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True
        )

        best_loss = float('inf')
        best_state = None
        patience = 30
        wait = 0

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            n = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)

                if isinstance(out, tuple):
                    y_real, y_imag = out
                    y_exp = yb.unsqueeze(-1)
                    loss = criterion(y_real, y_exp) + IMAG_PENALTY * criterion(
                        y_imag, torch.zeros_like(y_imag)
                    )
                else:
                    loss = criterion(out, yb.unsqueeze(-1))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n += 1
            avg_loss = epoch_loss / n
            scheduler.step()
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            out = model(X_te.to(device))
            if isinstance(out, tuple):
                preds_scaled = out[0].cpu().numpy().flatten()
            else:
                preds_scaled = out.cpu().numpy().flatten()
        # Inverse transform predictions
        preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

        mse_scores.append(float(mean_squared_error(y_test_raw, preds)))
        mae_scores.append(float(mean_absolute_error(y_test_raw, preds)))

    return {
        'mse_mean': float(np.mean(mse_scores)),
        'mse_std': float(np.std(mse_scores)),
        'mae_mean': float(np.mean(mae_scores)),
        'mae_std': float(np.std(mae_scores)),
        'num_params': num_params,
    }


def train_eval_kfold_hybrid(X, y, input_size, fnn_h, cauchy_h,
                            n_folds=5, epochs=EPOCHS, lr=LR, device='cpu', cauchy_eps=1e-8):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    mse_scores = []
    mae_scores = []
    num_params = None

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_train = scaler_X.fit_transform(X[train_idx]).astype(np.float32)
        X_test = scaler_X.transform(X[test_idx]).astype(np.float32)
        y_train = scaler_y.fit_transform(y[train_idx].reshape(-1, 1)).flatten().astype(np.float32)
        y_test_raw = y[test_idx]

        X_tr = torch.tensor(X_train, dtype=torch.float32)
        Y_tr = torch.tensor(y_train, dtype=torch.float32)
        X_te = torch.tensor(X_test, dtype=torch.float32)

        torch.manual_seed(42 + fold)
        model = HybridFNNCauchy(input_size, fnn_h, cauchy_h, 1, eps=cauchy_eps).to(device)

        if num_params is None:
            num_params = count_real_params(model)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        train_ds = torch.utils.data.TensorDataset(X_tr, Y_tr)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True
        )

        best_loss = float('inf')
        best_state = None
        patience = 30
        wait = 0

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            n = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)

                if isinstance(out, tuple):
                    y_real, y_imag = out
                    y_exp = yb.unsqueeze(-1)
                    loss = criterion(y_real, y_exp) + IMAG_PENALTY * criterion(
                        y_imag, torch.zeros_like(y_imag)
                    )
                else:
                    loss = criterion(out, yb.unsqueeze(-1))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n += 1
            avg_loss = epoch_loss / n
            scheduler.step()
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            out = model(X_te.to(device))
            if isinstance(out, tuple):
                preds_scaled = out[0].cpu().numpy().flatten()
            else:
                preds_scaled = out.cpu().numpy().flatten()
        preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

        mse_scores.append(float(mean_squared_error(y_test_raw, preds)))
        mae_scores.append(float(mean_absolute_error(y_test_raw, preds)))

    return {
        'mse_mean': float(np.mean(mse_scores)),
        'mse_std': float(np.std(mse_scores)),
        'mae_mean': float(np.mean(mae_scores)),
        'mae_std': float(np.std(mae_scores)),
        'num_params': num_params,
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    datasets = load_datasets()
    cauchy_h = 64
    max_train = 100  # Moderate training set; enough to learn but still data-limited

    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}

    for ds_name, (X, y) in datasets.items():
        # Subsample to small training set
        if len(X) > max_train:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X), max_train, replace=False)
            X, y = X[idx], y[idx]

        input_size = X.shape[1]
        target_params = cauchy_real_params(cauchy_h, input_size)

        fnn_h = matched_fnn_hidden(target_params, input_size)

        print(f"\n{'='*50}")
        print(f"Dataset: {ds_name} (n={len(X)}, d={input_size})")
        print(f"CauchyNet h={cauchy_h} ({target_params}p) -> FNN h={fnn_h}")
        print('=' * 50)
        all_results[ds_name] = {}

        # Use tuned CauchyNet hyperparams for multi-dim
        cauchy_eps = 1.0 if input_size >= 8 else 1e-8
        model_lr = 0.05 if input_size >= 8 else LR

        # CauchyNet
        print(f"  CauchyNet (h={cauchy_h}, lr={model_lr})...", end=" ", flush=True)
        r = train_eval_kfold(CauchyNet, X, y, hidden_size=cauchy_h, lr=model_lr,
                             device=device, cauchy_eps=cauchy_eps)
        all_results[ds_name]['CauchyNet'] = r
        print(f"MSE={r['mse_mean']:.4f}+/-{r['mse_std']:.4f}  "
              f"MAE={r['mae_mean']:.4f}+/-{r['mae_std']:.4f}  "
              f"Params={r['num_params']}")

        # FNN (parameter-matched)
        print(f"  FNN (h={fnn_h})...", end=" ", flush=True)
        r = train_eval_kfold(FNN, X, y, hidden_size=fnn_h,
                             device=device)
        all_results[ds_name]['FNN'] = r
        print(f"MSE={r['mse_mean']:.4f}+/-{r['mse_std']:.4f}  "
              f"MAE={r['mae_mean']:.4f}+/-{r['mae_std']:.4f}  "
              f"Params={r['num_params']}")

        # Hybrid FNN(4)->CauchyNet(32)
        hybrid_fnn_h, hybrid_cauchy_h = 4, 32
        print(f"  Hybrid FNN({hybrid_fnn_h})->Cauchy({hybrid_cauchy_h}, lr={model_lr})...", end=" ", flush=True)
        r = train_eval_kfold_hybrid(X, y, input_size, hybrid_fnn_h, hybrid_cauchy_h,
                                    lr=model_lr, device=device, cauchy_eps=cauchy_eps)
        all_results[ds_name]['Hybrid'] = r
        print(f"MSE={r['mse_mean']:.4f}+/-{r['mse_std']:.4f}  "
              f"MAE={r['mae_mean']:.4f}+/-{r['mae_std']:.4f}  "
              f"Params={r['num_params']}")

    # ──────────────────────────────────────────────────────────────
    # LaTeX table
    # ──────────────────────────────────────────────────────────────
    tex = r"""\begin{table}[ht]\centering
\caption{UCI/tabular regression benchmarks. 5-fold CV, $n=100$, $h=64$.}
\label{tab:uci}
\begin{tabular}{llcccc}
\toprule
Dataset & Model & \# Params & Test MSE & Test MAE \\
\midrule
"""
    model_names = ['CauchyNet', 'FNN', 'Hybrid']
    for ds_name in datasets:
        for j, model_name in enumerate(model_names):
            r = all_results[ds_name][model_name]
            prefix = ds_name if j == 0 else ""
            tex += (f"{prefix} & {model_name} & {r['num_params']} & "
                    f"${r['mse_mean']:.4f} \\pm {r['mse_std']:.4f}$ & "
                    f"${r['mae_mean']:.4f} \\pm {r['mae_std']:.4f}$ \\\\\n")
        tex += r"\midrule" + "\n"

    tex = tex.rstrip(r"\midrule" + "\n") + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}"""
    print("\n" + tex)

    with open(os.path.join(out_dir, 'exp7_uci_table.tex'), 'w') as f:
        f.write(tex)

    # JSON
    with open(os.path.join(out_dir, 'exp7_uci_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {out_dir}/")


if __name__ == '__main__':
    main()
