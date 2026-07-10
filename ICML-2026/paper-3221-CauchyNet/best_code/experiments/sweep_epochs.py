"""
Plot training and test curves per epoch for CauchyNet, FNN, SIREN.
Tests paper 1D target and |x| with n=20 to find optimal early stopping.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from shared import (
    CauchyNet1D, FNN, SIREN, count_real_params,
    cauchy_real_params, matched_fnn_hidden,
    HIDDEN_SIZE, WEIGHT_DECAY, IMAG_PENALTY,
    target_function,
)

def train_with_curves(model, X_train, Y_train, X_test, Y_test_scaled,
                      lr=0.01, epochs=500, is_cauchy=False):
    """Train and record train/test MSE at each epoch."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    criterion = nn.MSELoss()

    train_mses, test_mses = [], []

    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        out = model(X_train)
        if isinstance(out, tuple):
            y_real, y_imag = out
            loss = criterion(y_real, Y_train.unsqueeze(-1)) + IMAG_PENALTY * criterion(y_imag, torch.zeros_like(y_imag))
        else:
            loss = criterion(out, Y_train.unsqueeze(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Eval
        model.eval()
        with torch.no_grad():
            out_tr = model(X_train)
            out_te = model(X_test)
            if isinstance(out_tr, tuple):
                pred_tr = out_tr[0].numpy().flatten()
                pred_te = out_te[0].numpy().flatten()
            else:
                pred_tr = out_tr.numpy().flatten()
                pred_te = out_te.numpy().flatten()

        train_mses.append(float(mean_squared_error(Y_train.numpy(), pred_tr)))
        test_mses.append(float(mean_squared_error(Y_test_scaled.numpy(), pred_te)))

    return train_mses, test_mses


def run_target(target_fn, target_name, n_train=20, epochs=500, n_seeds=5):
    cauchy_h = HIDDEN_SIZE
    target_params = cauchy_real_params(cauchy_h)
    fnn_h = matched_fnn_hidden(target_params)

    # Generate data
    x_all = np.linspace(-1, 1, 200).astype(np.float32)
    y_all = target_fn(x_all).astype(np.float32)

    rng = np.random.RandomState(42)
    indices = np.arange(200)
    rng.shuffle(indices)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:n_train+50]

    scaler_y = MinMaxScaler()
    y_all_scaled = scaler_y.fit_transform(y_all.reshape(-1,1)).flatten().astype(np.float32)

    X_tr = torch.tensor(x_all[train_idx]).unsqueeze(-1)
    Y_tr = torch.tensor(y_all_scaled[train_idx])
    X_te = torch.tensor(x_all[test_idx]).unsqueeze(-1)
    Y_te = torch.tensor(y_all_scaled[test_idx])

    models_cfg = {
        'CauchyNet': lambda: CauchyNet1D(cauchy_h),
        'FNN': lambda: FNN(1, fnn_h, 1),
        'SIREN': lambda: SIREN(1, fnn_h, 1),
    }
    colors = {'CauchyNet': 'red', 'FNN': 'blue', 'SIREN': 'green'}

    # Collect curves across seeds
    all_test_curves = {name: [] for name in models_cfg}

    for seed in range(42, 42 + n_seeds):
        for name, factory in models_cfg.items():
            torch.manual_seed(seed)
            model = factory()
            _, test_curve = train_with_curves(model, X_tr, Y_tr, X_te, Y_te,
                                              lr=0.01, epochs=epochs)
            all_test_curves[name].append(test_curve)

    # Average across seeds
    avg_curves = {}
    for name in models_cfg:
        arr = np.array(all_test_curves[name])  # (n_seeds, epochs)
        avg_curves[name] = arr.mean(axis=0)

    # Find best epoch for each model
    print(f"\n{'='*60}")
    print(f"{target_name} (n={n_train})")
    print(f"{'='*60}")
    for name in models_cfg:
        curve = avg_curves[name]
        best_ep = np.argmin(curve)
        print(f"  {name}: best epoch={best_ep+1}, test MSE={curve[best_ep]:.6f}")
        # Also report MSE at various epoch counts
        for ep in [50, 100, 150, 200, 300, 500]:
            if ep <= epochs:
                print(f"    epoch {ep}: test MSE={curve[ep-1]:.6f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in models_cfg:
        ax.plot(range(1, epochs+1), avg_curves[name], color=colors[name],
                label=name, linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Test MSE (scaled)', fontsize=13)
    ax.set_title(f'{target_name} — Test MSE vs Epoch (n={n_train}, avg of {n_seeds} seeds)',
                 fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/training_curves_{target_name.replace(" ", "_").replace("|","abs")}.png', dpi=150)
    plt.savefig(f'results/training_curves_{target_name.replace(" ", "_").replace("|","abs")}.pdf', dpi=150)
    print(f"  Saved plot.")
    return avg_curves


def main():
    # Paper 1D target
    run_target(target_function, "Paper 1D target", n_train=20, epochs=500)

    # |x|
    run_target(lambda x: np.abs(x), "|x|", n_train=20, epochs=500)

    # sign(x)
    run_target(lambda x: np.sign(x).astype(np.float32), "sign(x)", n_train=20, epochs=500)

    print("\nDONE.")


if __name__ == '__main__':
    main()
