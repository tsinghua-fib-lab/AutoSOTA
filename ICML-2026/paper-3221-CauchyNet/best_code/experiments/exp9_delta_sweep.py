"""
Experiment 9: Delta-Sweep on Near-Singular Targets
====================================================
Varies pole distance delta in f(x) = 1/((x-a)^2 + delta^2).
As delta shrinks, CauchyNet maintains accuracy (O(1) units per pole)
while FNN degrades (needs O(1/delta) width).
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


def near_singular_target(x, delta):
    """Near-singular peak + smooth background:
    f(x) = 1/((x - 0.3)^2 + delta^2) + sin(4x)
    The peak sharpens as delta -> 0; the sine ensures the model
    must fit structure away from the peak too.
    """
    return 1.0 / ((x - 0.3) ** 2 + delta ** 2) + np.sin(4 * x)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    N_train = 50
    N_test = 500
    n_seeds = 10  # paper claims 10 seeds per cell
    seeds = list(range(42, 42 + n_seeds))
    deltas = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]

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

    for delta in deltas:
        key = f"delta={delta}"
        print(f"\n{'='*50}")
        print(f"delta = {delta}")
        print('=' * 50)

        x_train = np.linspace(-1, 1, N_train).astype(np.float32)
        x_test = np.linspace(-1, 1, N_test).astype(np.float32)
        y_train_raw = near_singular_target(x_train, delta).astype(np.float32)
        y_test_raw = near_singular_target(x_test, delta).astype(np.float32)

        scaler = MinMaxScaler()
        y_train = scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten().astype(np.float32)
        y_test = scaler.transform(y_test_raw.reshape(-1, 1)).flatten().astype(np.float32)

        X_train = torch.tensor(x_train).unsqueeze(-1)
        Y_train = torch.tensor(y_train)
        X_test = torch.tensor(x_test).unsqueeze(-1)
        Y_test = torch.tensor(y_test)

        all_results[key] = {}

        for model_name, factory in model_factories.items():
            mse_list = []
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = factory()
                r = train_and_eval(model, X_train, Y_train, X_test, Y_test,
                                   epochs=300, device=device, patience=30)
                mse_list.append(r['mse'])

            all_results[key][model_name] = {
                'mse_mean': float(np.mean(mse_list)),
                'mse_std': float(np.std(mse_list)),
                'num_params': count_real_params(factory()),
            }
            print(f"  {model_name}: MSE={np.mean(mse_list):.6f}+/-{np.std(mse_list):.6f}")

    # Plot: MSE vs delta
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'CauchyNet': 'red', 'FNN': 'blue'}
    for model_name in model_factories:
        means = [all_results[f"delta={d}"][model_name]['mse_mean'] for d in deltas]
        stds = [all_results[f"delta={d}"][model_name]['mse_std'] for d in deltas]
        ax.errorbar(deltas, means, yerr=stds, marker='o', label=model_name,
                    color=colors[model_name], linewidth=2, capsize=4)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Pole distance (delta)', fontsize=13)
    ax.set_ylabel('Test MSE', fontsize=13)
    ax.set_title('Near-Singular Approximation: MSE vs Pole Distance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'exp9_delta_sweep.pdf'), dpi=150)
    plt.savefig(os.path.join(out_dir, 'exp9_delta_sweep.png'), dpi=150)

    # LaTeX table
    tex = r"""\begin{table}[ht]\centering
\caption{Near-singular targets with varying pole distance $\delta$. 5 seeds, $n=50$, $h=64$.}
\label{tab:delta_sweep}
\begin{tabular}{lcccc}
\toprule
$\delta$ & CauchyNet MSE & FNN MSE & Ratio (FNN/Cauchy) \\
\midrule
"""
    for delta in deltas:
        key = f"delta={delta}"
        c = all_results[key]['CauchyNet']['mse_mean']
        f = all_results[key]['FNN']['mse_mean']
        ratio = f / c if c > 0 else float('inf')
        tex += f"{delta} & ${c:.6f}$ & ${f:.6f}$ & {ratio:.1f}x \\\\\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}"""
    print("\n" + tex)

    with open(os.path.join(out_dir, 'exp9_delta_sweep_table.tex'), 'w') as f:
        f.write(tex)

    with open(os.path.join(out_dir, 'exp9_delta_sweep_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {out_dir}/")


if __name__ == '__main__':
    main()
