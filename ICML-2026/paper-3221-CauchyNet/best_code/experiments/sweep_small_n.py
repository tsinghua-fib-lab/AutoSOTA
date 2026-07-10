"""
Sweep small training set sizes for 1D tasks where CauchyNet currently loses.
Tests n = {10, 15, 20, 25, 30, 40, 50} to find the crossover point.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from shared import (
    CauchyNet1D, FNN, SIREN, count_real_params,
    cauchy_real_params, matched_fnn_hidden,
    HIDDEN_SIZE, BATCH_SIZE, WEIGHT_DECAY, EPOCHS, IMAG_PENALTY,
    target_function,
)


def train_eval_1d(model_cls, x_train, y_train_scaled, x_test, y_test_raw, scaler_y,
                  hidden_size, lr, n_seeds=5, epochs=200, is_cauchy=False):
    """Train and eval, return mean MSE across seeds."""
    mses = []
    for seed in range(42, 42 + n_seeds):
        torch.manual_seed(seed)
        if is_cauchy:
            model = CauchyNet1D(hidden_size, 1)
        else:
            model = model_cls(1, hidden_size, 1)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        criterion = nn.MSELoss()

        X_tr = torch.tensor(x_train).unsqueeze(-1)
        Y_tr = torch.tensor(y_train_scaled)
        X_te = torch.tensor(x_test).unsqueeze(-1)

        best_loss, best_state = float('inf'), None
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(X_tr)
            if isinstance(out, tuple):
                y_real, y_imag = out
                loss = criterion(y_real, Y_tr.unsqueeze(-1)) + IMAG_PENALTY * criterion(y_imag, torch.zeros_like(y_imag))
            else:
                loss = criterion(out, Y_tr.unsqueeze(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            out = model(X_te)
            if isinstance(out, tuple):
                preds_scaled = out[0].numpy().flatten()
            else:
                preds_scaled = out.numpy().flatten()
        preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        mses.append(float(mean_squared_error(y_test_raw, preds)))
    return np.mean(mses), np.std(mses)


# ── Target functions ──────────────────────────────────────────────

def abs_fn(x):
    return np.abs(x)

def relu_fn(x):
    return np.maximum(x, 0)

def sign_fn(x):
    return np.sign(x)


def sweep_n(target_fn, target_name, n_values, cauchy_h=64):
    """Sweep training set sizes for one target function."""
    target_params = cauchy_real_params(cauchy_h)
    fnn_h = matched_fnn_hidden(target_params)

    print(f"\n{'='*70}")
    print(f"Function: {target_name} | CauchyNet h={cauchy_h} ({target_params}p), FNN/SIREN h={fnn_h}")
    print(f"{'='*70}")
    print(f"{'n':>5} | {'Cauchy MSE':>14} {'FNN MSE':>14} {'SIREN MSE':>14} | {'Winner':>10}")
    print("-" * 70)

    for n in n_values:
        # Generate data
        x_all = np.linspace(-1, 1, 200).astype(np.float32)
        y_all = target_fn(x_all).astype(np.float32)

        # Use fixed test set (last 50 points evenly spaced)
        rng = np.random.RandomState(42)
        indices = np.arange(200)
        rng.shuffle(indices)
        train_idx = indices[:n]
        test_idx = indices[n:n+50]

        x_train = x_all[train_idx]
        x_test = x_all[test_idx]
        y_test_raw = y_all[test_idx]

        scaler_y = MinMaxScaler()
        y_train_scaled = scaler_y.fit_transform(y_all[train_idx].reshape(-1, 1)).flatten().astype(np.float32)

        cauchy_m, cauchy_s = train_eval_1d(None, x_train, y_train_scaled, x_test, y_test_raw,
                                            scaler_y, cauchy_h, lr=0.01, is_cauchy=True)
        fnn_m, fnn_s = train_eval_1d(FNN, x_train, y_train_scaled, x_test, y_test_raw,
                                      scaler_y, fnn_h, lr=0.01)
        siren_m, siren_s = train_eval_1d(SIREN, x_train, y_train_scaled, x_test, y_test_raw,
                                          scaler_y, fnn_h, lr=0.01)

        best = min(cauchy_m, fnn_m, siren_m)
        winner = "CAUCHY" if cauchy_m == best else ("FNN" if fnn_m == best else "SIREN")
        marker = " ***" if winner == "CAUCHY" else ""
        print(f"{n:>5} | {cauchy_m:>14.6f} {fnn_m:>14.6f} {siren_m:>14.6f} | {winner:>10}{marker}")


def main():
    n_values = [10, 15, 20, 25, 30, 40, 50]

    # Paper 1D target
    sweep_n(target_function, "Paper 1D target", n_values)

    # Piecewise functions
    sweep_n(abs_fn, "|x|", n_values)
    sweep_n(relu_fn, "ReLU(x)", n_values)
    sweep_n(sign_fn, "sign(x)", n_values)

    print("\nDONE.")


if __name__ == '__main__':
    main()
