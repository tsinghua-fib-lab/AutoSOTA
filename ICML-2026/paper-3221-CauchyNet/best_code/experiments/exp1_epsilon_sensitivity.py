"""
Experiment 1: Epsilon Sensitivity Sweep
========================================
Sweeps epsilon in {1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0} + learnable epsilon.
10 random seeds per setting. Reports MSE, MAE, gradient norms.
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from shared import (
    CauchyNet1D, ReciprocalActivation, count_real_params,
    prepare_1d_data, train_and_eval,
    HIDDEN_SIZE, BATCH_SIZE, LR, EPOCHS, IMAG_PENALTY
)


class CauchyNet1D_LearnableEps(nn.Module):
    """CauchyNet with learnable epsilon."""
    def __init__(self, hidden_size, output_size=1, eps_init=1e-4):
        super().__init__()
        self.lambda_ = nn.Parameter(
            torch.normal(mean=0.0, std=0.1, size=(hidden_size, output_size), dtype=torch.cfloat)
        )
        angles = 2 * np.pi * torch.rand(hidden_size)
        real_part = 2 * np.pi * torch.cos(angles)
        imaginary_part = torch.sin(angles)
        self.xi = nn.Parameter(torch.complex(real_part, imaginary_part))
        # Learnable log(eps) so eps stays positive
        self.log_eps = nn.Parameter(torch.tensor(np.log(eps_init)))

    def forward(self, x):
        eps = torch.exp(self.log_eps)
        xc = torch.complex(x, torch.zeros_like(x)).unsqueeze(-1)
        xc = xc.view(x.size(0), 1, -1)
        activated = 1.0 / (self.xi - xc + eps)
        activated_flat = activated.view(x.size(0), -1)
        y = torch.matmul(activated_flat, self.lambda_)
        return y.real.squeeze(-1).unsqueeze(-1), y.imag.squeeze(-1).unsqueeze(-1)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    X_train, Y_train, X_val, Y_val, X_test, Y_test, Y_test_raw, scaler_y = prepare_1d_data()

    epsilons = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0]
    n_seeds = 10
    seeds = list(range(42, 42 + n_seeds))
    patience = 30

    results = {}

    # Fixed epsilon sweep
    for eps in epsilons:
        eps_key = f"{eps:.0e}"
        results[eps_key] = {'mse': [], 'mae': [], 'grad_norms': [], 'train_losses': [], 'best_epochs': []}
        print(f"\n--- epsilon = {eps_key} ---")

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = CauchyNet1D(HIDDEN_SIZE, eps=eps)
            r = train_and_eval(model, X_train, Y_train, X_test, Y_test,
                              device=device, return_grad_norms=True,
                              X_val=X_val, Y_val=Y_val, patience=patience)
            results[eps_key]['mse'].append(r['mse'])
            results[eps_key]['mae'].append(r['mae'])
            results[eps_key]['grad_norms'].append(r['grad_norms'])
            results[eps_key]['train_losses'].append(r['train_losses'])
            results[eps_key]['best_epochs'].append(r['best_epoch'])
            print(f"  seed {seed}: MSE={r['mse']:.6f}  MAE={r['mae']:.6f}  stopped@{r['total_epochs']}/{EPOCHS} best@{r['best_epoch']}")

    # Learnable epsilon
    eps_key = "learnable"
    results[eps_key] = {'mse': [], 'mae': [], 'grad_norms': [], 'train_losses': [], 'final_eps': [], 'best_epochs': []}
    print(f"\n--- epsilon = learnable ---")
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = CauchyNet1D_LearnableEps(HIDDEN_SIZE, eps_init=1e-4)
        r = train_and_eval(model, X_train, Y_train, X_test, Y_test,
                          device=device, return_grad_norms=True,
                          X_val=X_val, Y_val=Y_val, patience=patience)
        final_eps = float(torch.exp(model.log_eps).item())
        results[eps_key]['mse'].append(r['mse'])
        results[eps_key]['mae'].append(r['mae'])
        results[eps_key]['grad_norms'].append(r['grad_norms'])
        results[eps_key]['train_losses'].append(r['train_losses'])
        results[eps_key]['final_eps'].append(final_eps)
        results[eps_key]['best_epochs'].append(r['best_epoch'])
        print(f"  seed {seed}: MSE={r['mse']:.6f}  MAE={r['mae']:.6f}  final_eps={final_eps:.6e}  stopped@{r['total_epochs']} best@{r['best_epoch']}")

    # ── Save & plot ───────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)

    save_data = {}
    for k, v in results.items():
        save_data[k] = {'mse': v['mse'], 'mae': v['mae']}
        if 'final_eps' in v:
            save_data[k]['final_eps'] = v['final_eps']
    with open(os.path.join(out_dir, 'exp1_epsilon_results.json'), 'w') as f:
        json.dump(save_data, f, indent=2)

    # Boxplot
    eps_labels = [f"{e:.0e}" for e in epsilons] + ["learnable"]
    mse_data = [results[k]['mse'] for k in eps_labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(mse_data, tick_labels=eps_labels, patch_artist=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(eps_labels)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel(r'$\varepsilon$', fontsize=14)
    ax.set_ylabel('Test MSE (scaled targets)', fontsize=14)
    ax.set_title(r'CauchyNet Test MSE vs $\varepsilon$ (10 seeds)', fontsize=15, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'exp1_mse_vs_epsilon.pdf'), dpi=150)
    plt.savefig(os.path.join(out_dir, 'exp1_mse_vs_epsilon.png'), dpi=150)

    # Gradient norms — pad to same length for averaging (early stopping causes different lengths)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, eps_key in enumerate(eps_labels):
        if eps_key == "learnable":
            continue
        gn_list = results[eps_key]['grad_norms']
        max_len = max(len(g) for g in gn_list)
        # Pad shorter runs with their last value
        gn_padded = np.array([g + [g[-1]] * (max_len - len(g)) for g in gn_list])
        mean_gn = gn_padded.mean(axis=0)
        std_gn = gn_padded.std(axis=0)
        epochs_range = np.arange(1, max_len + 1)
        ax.plot(epochs_range, mean_gn, label=fr'$\varepsilon$={eps_key}', color=colors[i])
        ax.fill_between(epochs_range, mean_gn - std_gn, mean_gn + std_gn,
                        alpha=0.15, color=colors[i])
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Gradient Norm (before clipping)', fontsize=14)
    ax.set_title('Gradient Norms During Training', fontsize=15, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'exp1_gradient_norms.pdf'), dpi=150)
    plt.savefig(os.path.join(out_dir, 'exp1_gradient_norms.png'), dpi=150)

    # LaTeX table
    print("\n" + "=" * 60)
    print("LaTeX Table: Epsilon Sensitivity")
    print("=" * 60)
    header = r"""\begin{table}[ht]\centering
\caption{Sensitivity of CauchyNet to $\varepsilon$ in the reciprocal activation $1/(z+\varepsilon)$.
Results on MinMaxScaler-normalized targets, 10 random seeds.}
\label{tab:epsilon}
\begin{tabular}{lccc}
\toprule
$\varepsilon$ & Test MSE & Test MAE & Grad Norm (final) \\
\midrule"""
    print(header)
    for eps_key in eps_labels:
        mse_arr = np.array(results[eps_key]['mse'])
        mae_arr = np.array(results[eps_key]['mae'])
        final_gn = np.array([gn[-1] for gn in results[eps_key]['grad_norms']])
        row = (f"{eps_key} & "
               f"${mse_arr.mean():.6f} \\pm {mse_arr.std():.6f}$ & "
               f"${mae_arr.mean():.6f} \\pm {mae_arr.std():.6f}$ & "
               f"${final_gn.mean():.4f} \\pm {final_gn.std():.4f}$ \\\\")
        print(row)
    print(r"""\bottomrule
\end{tabular}
\end{table}""")

    with open(os.path.join(out_dir, 'exp1_epsilon_table.tex'), 'w') as f:
        f.write(header + '\n')
        for eps_key in eps_labels:
            mse_arr = np.array(results[eps_key]['mse'])
            mae_arr = np.array(results[eps_key]['mae'])
            final_gn = np.array([gn[-1] for gn in results[eps_key]['grad_norms']])
            f.write(f"{eps_key} & "
                    f"${mse_arr.mean():.6f} \\pm {mse_arr.std():.6f}$ & "
                    f"${mae_arr.mean():.6f} \\pm {mae_arr.std():.6f}$ & "
                    f"${final_gn.mean():.4f} \\pm {final_gn.std():.4f}$ \\\\\n")
        f.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n")

    print(f"\nAll saved to {out_dir}/")


if __name__ == '__main__':
    main()
