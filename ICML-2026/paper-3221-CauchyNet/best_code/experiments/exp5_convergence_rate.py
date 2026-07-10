"""
Experiment 5: Convergence Rate vs Hidden Width h
=================================================
Sweeps h in {8,16,32,64,128,256}. Analytic, near-singular, C^2 targets.
CauchyNet vs parameter-matched FNN at each width.

Uses 50 train / 25 val points, 500 test points.
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
from sklearn.metrics import mean_squared_error

from shared import (
    CauchyNet1D, FNN, count_real_params, train_and_eval,
    cauchy_real_params, matched_fnn_hidden,
    LR, BATCH_SIZE, WEIGHT_DECAY
)

FUNCTIONS = {
    'Analytic': lambda x: np.sin(5 * x) * np.exp(-x ** 2),
    'Near-singular': lambda x: 1.0 / ((x + 0.6) ** 2 + 0.005),
    'C^2': lambda x: np.abs(x) ** 2.5,
}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    hidden_sizes = [32, 64, 128, 256]
    n_seeds = 5
    seeds = list(range(42, 42 + n_seeds))
    N_total = 75     # 50 train + 25 val
    N_test = 500
    epochs = 500

    x_all = np.linspace(-1, 1, N_total).astype(np.float32)
    x_test = np.linspace(-1, 1, N_test).astype(np.float32)

    # Fixed 50/25 train/val split
    rng = np.random.RandomState(0)
    idx = rng.permutation(N_total)
    train_idx = idx[:50]
    val_idx = idx[50:]

    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}

    for fn_name, fn in FUNCTIONS.items():
        print(f"\n{'='*50}\nFunction: {fn_name}\n{'='*50}")

        y_all_raw = fn(x_all).astype(np.float32)
        y_test_raw = fn(x_test).astype(np.float32)

        scaler = MinMaxScaler()
        y_all = scaler.fit_transform(y_all_raw.reshape(-1, 1)).flatten().astype(np.float32)
        y_test = scaler.transform(y_test_raw.reshape(-1, 1)).flatten().astype(np.float32)

        X_train = torch.tensor(x_all[train_idx]).unsqueeze(-1)
        Y_train = torch.tensor(y_all[train_idx])
        X_val = torch.tensor(x_all[val_idx]).unsqueeze(-1)
        Y_val = torch.tensor(y_all[val_idx])
        X_test = torch.tensor(x_test).unsqueeze(-1)
        Y_test = torch.tensor(y_test)

        all_results[fn_name] = {
            'CauchyNet': {'h': hidden_sizes, 'mse_mean': [], 'mse_std': [], 'params': []},
            'FNN': {'h': [], 'mse_mean': [], 'mse_std': [], 'params': []},
        }

        for h in hidden_sizes:
            # CauchyNet
            cauchy_mse_list = []
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = CauchyNet1D(h)
                r = train_and_eval(model, X_train, Y_train, X_test, Y_test,
                                   epochs=epochs, device=device, patience=50,
                                   X_val=X_val, Y_val=Y_val)
                cauchy_mse_list.append(r['mse'])

            cauchy_params = count_real_params(CauchyNet1D(h))
            cauchy_mean = float(np.mean(cauchy_mse_list))
            cauchy_std = float(np.std(cauchy_mse_list))
            all_results[fn_name]['CauchyNet']['mse_mean'].append(cauchy_mean)
            all_results[fn_name]['CauchyNet']['mse_std'].append(cauchy_std)
            all_results[fn_name]['CauchyNet']['params'].append(cauchy_params)

            # FNN (parameter-matched)
            target_params = cauchy_real_params(h)
            fnn_h = matched_fnn_hidden(target_params)
            fnn_mse_list = []
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = FNN(1, fnn_h, 1)
                r = train_and_eval(model, X_train, Y_train, X_test, Y_test,
                                   epochs=epochs, device=device, patience=50,
                                   X_val=X_val, Y_val=Y_val)
                fnn_mse_list.append(r['mse'])

            fnn_params = count_real_params(FNN(1, fnn_h, 1))
            fnn_mean = float(np.mean(fnn_mse_list))
            fnn_std = float(np.std(fnn_mse_list))
            all_results[fn_name]['FNN']['h'].append(fnn_h)
            all_results[fn_name]['FNN']['mse_mean'].append(fnn_mean)
            all_results[fn_name]['FNN']['mse_std'].append(fnn_std)
            all_results[fn_name]['FNN']['params'].append(fnn_params)

            ratio = fnn_mean / cauchy_mean if cauchy_mean > 0 else 0
            print(f"  h={h:4d}: CauchyNet({cauchy_params}p)={cauchy_mean:.8f}  "
                  f"FNN({fnn_params}p)={fnn_mean:.8f}  ratio={ratio:.1f}x")

    # Plot: CauchyNet vs FNN convergence
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {'CauchyNet': 'red', 'FNN': 'blue'}
    markers = {'CauchyNet': 'o', 'FNN': 's'}

    for i, (fn_name, data) in enumerate(all_results.items()):
        ax = axes[i]
        for model_name in ['CauchyNet', 'FNN']:
            h = np.array(hidden_sizes)
            mse_mean = np.array(data[model_name]['mse_mean'])
            mse_std = np.array(data[model_name]['mse_std'])
            ax.errorbar(h, mse_mean, yerr=mse_std,
                        fmt=f'{markers[model_name]}-',
                        color=colors[model_name], capsize=4,
                        linewidth=2, markersize=8, label=model_name)

        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('CauchyNet Hidden Width h', fontsize=13)
        ax.set_ylabel('Test MSE (scaled)', fontsize=13)
        ax.set_title(fn_name, fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        # Convergence slopes for CauchyNet
        log_h = np.log(h)
        log_mse = np.log(np.array(data['CauchyNet']['mse_mean']) + 1e-15)
        coeffs = np.polyfit(log_h, log_mse, 1)
        ax.text(0.05, 0.05, f"CauchyNet slope: {coeffs[0]:.2f}",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('Convergence Rate vs Hidden Width (n=50 train, val-based early stopping)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'exp5_convergence_rate.pdf'), dpi=150)
    plt.savefig(os.path.join(out_dir, 'exp5_convergence_rate.png'), dpi=150)

    # LaTeX table
    tex = r"""\begin{table}[ht]\centering
\caption{Convergence rate: CauchyNet vs FNN at matched parameter counts. 5 seeds, $n=50$ train, $n=25$ val.}
\label{tab:convergence}
\begin{tabular}{lccccc}
\toprule
Target & $h$ & CauchyNet MSE & FNN MSE & Ratio (FNN/Cauchy) \\
\midrule
"""
    for fn_name in FUNCTIONS:
        first = True
        for j, h in enumerate(hidden_sizes):
            c = all_results[fn_name]['CauchyNet']['mse_mean'][j]
            f = all_results[fn_name]['FNN']['mse_mean'][j]
            ratio = f / c if c > 0 else float('inf')
            prefix = fn_name if first else ""
            first = False
            tex += f"{prefix} & {h} & ${c:.6f}$ & ${f:.6f}$ & {ratio:.1f}x \\\\\n"
        tex += r"\midrule" + "\n"
    tex = tex.rstrip(r"\midrule" + "\n") + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}"""
    print("\n" + tex)

    with open(os.path.join(out_dir, 'exp5_convergence_table.tex'), 'w') as f:
        f.write(tex)

    with open(os.path.join(out_dir, 'exp5_convergence_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\nDone.")


if __name__ == '__main__':
    main()
