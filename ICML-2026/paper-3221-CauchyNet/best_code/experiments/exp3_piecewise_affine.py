"""
Experiment 3: Non-Smooth / Piecewise Function Approximation
============================================================
CauchyNet vs FNN (ReLU), as requested by Reviewer SDeh Q1.

Targets chosen to probe different structural properties:
  - Chirp: non-stationary frequency sin(pi*x*(1+8|x|))
  - Piecewise-smooth: 3 smooth pieces joined with corners
  - Gibbs (square wave): Fourier partial sum with Gibbs ringing
  - Step-ramp: combination of step jumps and linear ramps (piecewise-affine)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from shared import (
    CauchyNet1D, FNN, count_real_params, train_and_eval,
    cauchy_real_params, matched_fnn_hidden,
    HIDDEN_SIZE, BATCH_SIZE, LR, EPOCHS
)


def chirp(x):
    """Chirp signal: frequency increases with |x|."""
    return np.sin(np.pi * x * (1 + 8 * np.abs(x)))


def piecewise_smooth(x):
    """Three smooth pieces joined with corners at x=-0.3 and x=0.4."""
    y = np.zeros_like(x)
    m1 = x < -0.3
    m2 = (x >= -0.3) & (x < 0.4)
    m3 = x >= 0.4
    y[m1] = np.sin(5 * x[m1])
    y[m2] = 0.5 * x[m2] ** 2 - 0.2
    y[m3] = np.exp(-3 * (x[m3] - 0.4)) * np.cos(10 * x[m3])
    return y


def gibbs_square(x):
    """Fourier partial sum (10 odd harmonics) of square wave — Gibbs ringing."""
    result = np.zeros_like(x)
    for k in range(1, 21, 2):
        result += np.sin(k * np.pi * x) / k
    return (4 / np.pi) * result


def step_ramp(x):
    """Combination of step jumps and linear ramps (piecewise-affine)."""
    y = np.zeros_like(x)
    y += 0.3 * (x > -0.7).astype(float)
    y += 0.5 * np.clip(x + 0.2, 0, 0.4)
    y += 0.4 * (x > 0.3).astype(float)
    y -= 0.3 * np.clip(x - 0.6, 0, 0.3)
    return y


FUNCTIONS = {
    'Chirp': chirp,
    'Piecewise-smooth': piecewise_smooth,
    'Gibbs (sq wave)': gibbs_square,
    'Step-ramp': step_ramp,
}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    N_train, N_test = 20, 500
    n_seeds = 5
    seeds = list(range(42, 42 + n_seeds))

    # Parameter-matched single-layer models
    cauchy_h = HIDDEN_SIZE
    target_params = cauchy_real_params(cauchy_h)
    fnn_h = matched_fnn_hidden(target_params)
    print(f"CauchyNet h={cauchy_h} ({target_params}p) -> FNN h={fnn_h}")

    model_classes = {
        'CauchyNet': lambda: CauchyNet1D(cauchy_h),
        'FNN': lambda: FNN(1, fnn_h, 1),
    }

    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}

    for fn_name, fn in FUNCTIONS.items():
        print(f"\n{'='*50}\nFunction: {fn_name}\n{'='*50}")

        x_train = np.linspace(-1, 1, N_train).astype(np.float32)
        y_train_raw = fn(x_train).astype(np.float32)
        x_test = np.linspace(-1, 1, N_test).astype(np.float32)
        y_test_raw = fn(x_test).astype(np.float32)

        # MinMaxScaler on targets
        scaler = MinMaxScaler()
        y_train = scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten().astype(np.float32)
        y_test = scaler.transform(y_test_raw.reshape(-1, 1)).flatten().astype(np.float32)

        X_train = torch.tensor(x_train).unsqueeze(-1)
        Y_train = torch.tensor(y_train)
        X_test = torch.tensor(x_test).unsqueeze(-1)
        Y_test = torch.tensor(y_test)

        all_results[fn_name] = {}

        for model_name, factory in model_classes.items():
            mse_list, mae_list = [], []
            last_preds = None

            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = factory()
                r = train_and_eval(model, X_train, Y_train, X_test, Y_test,
                                  epochs=300, device=device, patience=30)
                mse_list.append(r['mse'])
                mae_list.append(r['mae'])
                # Get predictions for plotting
                model.eval()
                with torch.no_grad():
                    out = model(X_test.to(device))
                    last_preds = out[0].cpu().numpy().flatten() if isinstance(out, tuple) else out.cpu().numpy().flatten()

            all_results[fn_name][model_name] = {
                'mse_mean': float(np.mean(mse_list)),
                'mse_std': float(np.std(mse_list)),
                'mae_mean': float(np.mean(mae_list)),
                'mae_std': float(np.std(mae_list)),
                'num_params': r['num_params'],
                'preds': last_preds.tolist(),
            }
            print(f"  {model_name}: MSE={np.mean(mse_list):.6f}+/-{np.std(mse_list):.6f}  "
                  f"Params={r['num_params']}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    colors = {'CauchyNet': 'red', 'FNN': 'blue'}

    for i, (fn_name, fn) in enumerate(FUNCTIONS.items()):
        ax = axes[i]
        x_plot = np.linspace(-1, 1, N_test)
        y_true_raw = fn(x_plot)
        scaler = MinMaxScaler()
        y_true = scaler.fit_transform(y_true_raw.reshape(-1, 1)).flatten()
        ax.plot(x_plot, y_true, 'k--', linewidth=2, label='True')
        for model_name in model_classes:
            preds = all_results[fn_name][model_name]['preds']
            ax.plot(x_plot, preds, color=colors[model_name], linewidth=1.5,
                    label=model_name, alpha=0.8)
        ax.set_title(fn_name, fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle('Piecewise-Affine Function Approximation (scaled)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'exp3_piecewise_affine.pdf'), dpi=150)
    plt.savefig(os.path.join(out_dir, 'exp3_piecewise_affine.png'), dpi=150)

    # LaTeX
    tex = r"""\begin{table}[ht]\centering
\caption{Piecewise-affine/non-smooth functions. 5 seeds, $n=20$ training points, $h=64$.}
\label{tab:piecewise}
\begin{tabular}{llccc}
\toprule
Function & Model & \# Params & Test MSE & Test MAE \\
\midrule
"""
    for fn_name in FUNCTIONS:
        for j, model_name in enumerate(model_classes):
            r = all_results[fn_name][model_name]
            prefix = fn_name if j == 0 else ""
            tex += (f"{prefix} & {model_name} & {r['num_params']} & "
                    f"${r['mse_mean']:.6f} \\pm {r['mse_std']:.6f}$ & "
                    f"${r['mae_mean']:.6f} \\pm {r['mae_std']:.6f}$ \\\\\n")
        tex += r"\midrule" + "\n"
    tex = tex.rstrip(r"\midrule" + "\n") + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}"""
    print("\n" + tex)

    with open(os.path.join(out_dir, 'exp3_piecewise_table.tex'), 'w') as f:
        f.write(tex)

    save_data = {}
    for fn_name in all_results:
        save_data[fn_name] = {}
        for m in all_results[fn_name]:
            d = {k: v for k, v in all_results[fn_name][m].items() if k != 'preds'}
            save_data[fn_name][m] = d
    with open(os.path.join(out_dir, 'exp3_piecewise_results.json'), 'w') as f:
        json.dump(save_data, f, indent=2)


if __name__ == '__main__':
    main()
