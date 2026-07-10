"""
Experiment 10: Sample Efficiency Sweep
=======================================
Varies training set size n on near-singular and mixed targets.
CauchyNet's inductive bias (rational function structure) constrains
the interpolant at small n.
"""

import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from shared import (
    CauchyNet1D, FNN, count_real_params, train_and_eval,
    cauchy_real_params, matched_fnn_hidden,
    HIDDEN_SIZE, BATCH_SIZE, LR, EPOCHS
)


def near_singular(x):
    """Single sharp peak: 1/((x+0.6)^2 + 0.005)."""
    return 1.0 / ((x + 0.6) ** 2 + 0.005)


def mixed_target(x):
    """Near-singular peak + oscillation."""
    return 1.0 / ((x + 0.6) ** 2 + 0.005) + 5.0 * np.sin(8 * x) * np.exp(-2 * x ** 2)


FUNCTIONS = {
    'Near-singular': near_singular,
    'Mixed (peak + oscillation)': mixed_target,
}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    sample_sizes = [8, 12, 16, 20, 30, 50, 80, 120]
    N_test = 500
    n_seeds = 10
    seeds = list(range(42, 42 + n_seeds))

    cauchy_h = HIDDEN_SIZE
    target_params = cauchy_real_params(cauchy_h)
    fnn_h = matched_fnn_hidden(target_params)
    print(f"CauchyNet h={cauchy_h} ({target_params}p) -> FNN h={fnn_h}")

    model_factories = {
        'CauchyNet': lambda: CauchyNet1D(cauchy_h),
        'FNN': lambda: FNN(1, fnn_h, 1),
    }

    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}

    for fn_name, fn in FUNCTIONS.items():
        print(f"\n{'='*50}")
        print(f"Function: {fn_name}")
        print('=' * 50)
        all_results[fn_name] = {}

        x_test = np.linspace(-1, 1, N_test).astype(np.float32)
        y_test_raw = fn(x_test).astype(np.float32)

        for n in sample_sizes:
            key = f"n={n}"
            all_results[fn_name][key] = {}

            x_train = np.linspace(-1, 1, n).astype(np.float32)
            y_train_raw = fn(x_train).astype(np.float32)

            scaler = MinMaxScaler()
            y_train = scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten().astype(np.float32)
            y_test = scaler.transform(y_test_raw.reshape(-1, 1)).flatten().astype(np.float32)

            X_train = torch.tensor(x_train).unsqueeze(-1)
            Y_train = torch.tensor(y_train)
            X_test = torch.tensor(x_test).unsqueeze(-1)
            Y_test = torch.tensor(y_test)

            for model_name, factory in model_factories.items():
                mse_list = []
                for seed in seeds:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    model = factory()
                    r = train_and_eval(model, X_train, Y_train, X_test, Y_test,
                                       epochs=300, device=device, patience=30)
                    mse_list.append(r['mse'])

                all_results[fn_name][key][model_name] = {
                    'mse_mean': float(np.mean(mse_list)),
                    'mse_std': float(np.std(mse_list)),
                }

            c = all_results[fn_name][key]['CauchyNet']['mse_mean']
            f = all_results[fn_name][key]['FNN']['mse_mean']
            ratio = f / c if c > 0 else 0
            winner = "CauchyNet" if c < f else "FNN"
            print(f"  n={n:3d}: CauchyNet={c:.5f}  FNN={f:.5f}  ratio={ratio:.2f}x  [{winner}]")

    # Plot: MSE vs n for each function
    fig, axes = plt.subplots(1, len(FUNCTIONS), figsize=(6 * len(FUNCTIONS), 5))
    if len(FUNCTIONS) == 1:
        axes = [axes]
    colors = {'CauchyNet': 'red', 'FNN': 'blue'}

    for i, (fn_name, fn) in enumerate(FUNCTIONS.items()):
        ax = axes[i]
        for model_name in model_factories:
            means = [all_results[fn_name][f"n={n}"][model_name]['mse_mean'] for n in sample_sizes]
            stds = [all_results[fn_name][f"n={n}"][model_name]['mse_std'] for n in sample_sizes]
            ax.errorbar(sample_sizes, means, yerr=stds, marker='o',
                        label=model_name, color=colors[model_name], linewidth=2, capsize=4)

        ax.set_xlabel('Training samples (n)', fontsize=13)
        ax.set_ylabel('Test MSE', fontsize=13)
        ax.set_title(fn_name, fontsize=13, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

    fig.suptitle('Sample Efficiency: MSE vs Training Set Size', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'exp10_sample_efficiency.pdf'), dpi=150)
    plt.savefig(os.path.join(out_dir, 'exp10_sample_efficiency.png'), dpi=150)

    with open(os.path.join(out_dir, 'exp10_sample_efficiency_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {out_dir}/")


if __name__ == '__main__':
    main()
