"""
Experiment 6: Multi-Layer CauchyNet
=====================================
Tests whether stacking CauchyNet layers improves performance, addressing
the reviewer question about depth vs single-layer architecture.

Variants:
  - CauchyNet-1L: standard single-layer (baseline)
  - CauchyNet-2L: two Cauchy layers with intermediate real projection
  - CauchyNet-Res: single Cauchy layer + residual real-valued MLP branch
  - CauchyNet-Skip: two Cauchy layers with skip connection

All tested on the paper's 1D target function.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared import (
    CauchyNet1D, ReciprocalActivation, count_real_params,
    prepare_1d_data, train_and_eval,
    HIDDEN_SIZE, BATCH_SIZE, LR, EPOCHS, IMAG_PENALTY
)

# ──────────────────────────────────────────────────────────────────
# Multi-layer variants
# ──────────────────────────────────────────────────────────────────

class CauchyNet_2L(nn.Module):
    """Two Cauchy layers: first maps R->C->R (intermediate), then R->C->R.
    Returns (real, imag) tuple for imaginary penalty."""
    def __init__(self, hidden_size, intermediate_dim=16):
        super().__init__()
        h1 = hidden_size
        h2 = hidden_size // 2

        # Layer 1: 1D input -> h1 Cauchy units -> intermediate_dim
        self.lambda1 = nn.Parameter(torch.normal(mean=0.0, std=0.1, size=(h1, intermediate_dim), dtype=torch.cfloat))
        angles1 = 2 * np.pi * torch.rand(h1)
        self.xi1 = nn.Parameter(torch.complex(
            2 * np.pi * torch.cos(angles1), torch.sin(angles1)
        ))

        # Layer 2: intermediate_dim -> h2 Cauchy units -> output
        self.lambda2 = nn.Parameter(torch.normal(mean=0.0, std=0.1, size=(h2 * intermediate_dim, 1), dtype=torch.cfloat))
        angles2 = 2 * np.pi * torch.rand(h2, intermediate_dim)
        self.xi2 = nn.Parameter(torch.complex(
            2 * np.pi * torch.cos(angles2), torch.sin(angles2)
        ))

        self.intermediate_dim = intermediate_dim
        self.act = ReciprocalActivation()

    def forward(self, x):
        # Layer 1: 1D input
        xc = torch.complex(x, torch.zeros_like(x)).unsqueeze(-1)  # (batch, 1, 1)
        xc = xc.view(x.size(0), 1, -1)  # (batch, 1, 1)
        a1 = self.act(self.xi1 - xc)  # (batch, h1, 1)
        a1 = a1.view(x.size(0), -1)  # (batch, h1)
        h1_out = torch.matmul(a1, self.lambda1).real  # (batch, intermediate_dim)

        # Layer 2: treat h1_out as multi-dim input
        h1c = torch.complex(h1_out, torch.zeros_like(h1_out))  # (batch, intermediate_dim)
        h1c = h1c.unsqueeze(1)  # (batch, 1, intermediate_dim)
        # xi2: (h2, intermediate_dim) broadcasts to (batch, h2, intermediate_dim)
        a2 = self.act(self.xi2 - h1c)  # (batch, h2, intermediate_dim)
        a2 = a2.view(h1_out.size(0), -1)  # (batch, h2*intermediate_dim)
        y = torch.matmul(a2, self.lambda2)  # (batch, 1) complex
        return y.real.squeeze(-1).unsqueeze(-1), y.imag.squeeze(-1).unsqueeze(-1)


class CauchyNet_Res(nn.Module):
    """CauchyNet with a parallel real-valued residual branch.
    Returns (real, imag) tuple."""
    def __init__(self, hidden_size):
        super().__init__()
        # Cauchy branch
        self.lambda_ = nn.Parameter(torch.normal(mean=0.0, std=0.1, size=(hidden_size, 1), dtype=torch.cfloat))
        angles = 2 * np.pi * torch.rand(hidden_size)
        self.xi = nn.Parameter(torch.complex(
            2 * np.pi * torch.cos(angles), torch.sin(angles)
        ))
        self.act = ReciprocalActivation()

        # Residual branch: small MLP
        res_h = 16
        self.res_fc1 = nn.Linear(1, res_h)
        self.res_fc2 = nn.Linear(res_h, 1)

        # Mixing weight
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # Cauchy branch
        xc = torch.complex(x, torch.zeros_like(x)).unsqueeze(-1)
        xc = xc.view(x.size(0), 1, -1)
        activated = self.act(self.xi - xc)
        activated = activated.view(x.size(0), -1)
        y_cauchy = torch.matmul(activated, self.lambda_)  # complex

        # Residual branch (real only)
        y_res = self.res_fc2(torch.relu(self.res_fc1(x)))

        # Mix real parts, pass through imag
        y_real = self.alpha * y_cauchy.real + (1 - self.alpha) * y_res
        y_imag = self.alpha * y_cauchy.imag
        return y_real, y_imag


class CauchyNet_Skip(nn.Module):
    """Two Cauchy layers with skip connection from input.
    Returns (real, imag) tuple."""
    def __init__(self, hidden_size, intermediate=8):
        super().__init__()
        h1 = hidden_size
        self.intermediate = intermediate

        # Layer 1: 1D -> h1 -> intermediate
        self.lambda1 = nn.Parameter(torch.normal(mean=0.0, std=0.1, size=(h1, intermediate), dtype=torch.cfloat))
        angles1 = 2 * np.pi * torch.rand(h1)
        self.xi1 = nn.Parameter(torch.complex(
            2 * np.pi * torch.cos(angles1), torch.sin(angles1)
        ))

        # Layer 2: takes intermediate + 1 (skip) concatenated
        h2 = hidden_size // 2
        skip_dim = intermediate + 1
        self.lambda2 = nn.Parameter(torch.normal(
            mean=0.0, std=0.1, size=(h2 * skip_dim, 1), dtype=torch.cfloat
        ))
        angles2 = 2 * np.pi * torch.rand(h2, skip_dim)
        self.xi2 = nn.Parameter(torch.complex(
            2 * np.pi * torch.cos(angles2), torch.sin(angles2)
        ))

        self.act = ReciprocalActivation()

    def forward(self, x):
        # Layer 1
        xc = torch.complex(x, torch.zeros_like(x)).unsqueeze(-1)
        xc = xc.view(x.size(0), 1, -1)  # (batch, 1, 1)
        a1 = self.act(self.xi1 - xc)  # (batch, h1, 1)
        a1 = a1.view(x.size(0), -1)  # (batch, h1)
        h1 = torch.matmul(a1, self.lambda1).real  # (batch, intermediate)

        # Skip: concat h1 with original input
        h1_skip = torch.cat([h1, x], dim=-1)  # (batch, intermediate + 1)

        # Layer 2
        h1c = torch.complex(h1_skip, torch.zeros_like(h1_skip))
        h1c = h1c.unsqueeze(1)  # (batch, 1, skip_dim)
        # xi2: (h2, skip_dim) broadcasts to (batch, h2, skip_dim)
        a2 = self.act(self.xi2 - h1c)  # (batch, h2, skip_dim)
        a2 = a2.view(h1_skip.size(0), -1)  # (batch, h2*skip_dim)
        y = torch.matmul(a2, self.lambda2)  # (batch, 1) complex
        return y.real.squeeze(-1).unsqueeze(-1), y.imag.squeeze(-1).unsqueeze(-1)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    X_train, Y_train, X_val, Y_val, X_test, Y_test, _, _ = prepare_1d_data()

    n_seeds = 10
    seeds = list(range(42, 42 + n_seeds))

    model_factories = {
        'CauchyNet-1L': lambda: CauchyNet1D(HIDDEN_SIZE),
        'CauchyNet-2L': lambda: CauchyNet_2L(HIDDEN_SIZE),
        'CauchyNet-Res': lambda: CauchyNet_Res(HIDDEN_SIZE),
        'CauchyNet-Skip': lambda: CauchyNet_Skip(HIDDEN_SIZE),
    }

    # Print param counts
    print("\nParameter counts (real):")
    for name, factory in model_factories.items():
        m = factory()
        print(f"  {name}: {count_real_params(m)}")

    all_results = {}
    for name, factory in model_factories.items():
        all_results[name] = {'mse': [], 'mae': [], 'train_losses': [], 'num_params': None}
        print(f"\n--- {name} ---")
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = factory()
            if all_results[name]['num_params'] is None:
                all_results[name]['num_params'] = count_real_params(model)
            r = train_and_eval(model, X_train, Y_train, X_test, Y_test,
                              epochs=300, device=device,
                              X_val=X_val, Y_val=Y_val, patience=30)
            all_results[name]['mse'].append(r['mse'])
            all_results[name]['mae'].append(r['mae'])
            all_results[name]['train_losses'].append(r['train_losses'])
            print(f"  seed {seed}: MSE={r['mse']:.6f}  stopped@{r['total_epochs']} best@{r['best_epoch']}")

    # ──────────────────────────────────────────────────────────────
    # Output
    # ──────────────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)

    # Boxplot
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(all_results.keys())
    data = [all_results[n]['mse'] for n in names]
    labels = [f"{n}\n({all_results[n]['num_params']}p)" for n in names]

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    colors_list = ['red', 'blue', 'green', 'orange']
    for patch, c in zip(bp['boxes'], colors_list):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)

    ax.set_ylabel('Test MSE (scaled)', fontsize=14)
    ax.set_title('Multi-Layer CauchyNet Variants (10 seeds)', fontsize=15, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'exp6_multilayer.pdf'), dpi=150)
    plt.savefig(os.path.join(out_dir, 'exp6_multilayer.png'), dpi=150)

    # Training curves — pad to same length for averaging
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(names):
        tl_list = all_results[name]['train_losses']
        max_len = max(len(t) for t in tl_list)
        tl = np.array([t + [t[-1]] * (max_len - len(t)) for t in tl_list])
        mean_tl = tl.mean(axis=0)
        std_tl = tl.std(axis=0)
        epochs_range = np.arange(1, len(mean_tl) + 1)
        ax2.plot(epochs_range, mean_tl, color=colors_list[i], linewidth=2, label=name)
        ax2.fill_between(epochs_range, mean_tl - std_tl, mean_tl + std_tl,
                        alpha=0.15, color=colors_list[i])
    ax2.set_yscale('log')
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Training Loss', fontsize=14)
    ax2.set_title('Training Curves: Multi-Layer Variants', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'exp6_multilayer_training.pdf'), dpi=150)
    plt.savefig(os.path.join(out_dir, 'exp6_multilayer_training.png'), dpi=150)

    # LaTeX table
    tex = r"""\begin{table}[ht]\centering
\caption{Multi-layer CauchyNet variants on the 1D target function. 10 random seeds, MinMaxScaler.}
\label{tab:multilayer}
\begin{tabular}{lccc}
\toprule
Variant & \# Params & Test MSE & Test MAE \\
\midrule
"""
    for name in names:
        r = all_results[name]
        mse = np.array(r['mse'])
        mae = np.array(r['mae'])
        tex += (f"{name} & {r['num_params']} & "
                f"${mse.mean():.6f} \\pm {mse.std():.6f}$ & "
                f"${mae.mean():.6f} \\pm {mae.std():.6f}$ \\\\\n")
    tex += r"""\bottomrule
\end{tabular}
\end{table}"""
    print("\n" + tex)

    with open(os.path.join(out_dir, 'exp6_multilayer_table.tex'), 'w') as f:
        f.write(tex)

    # JSON
    save_data = {}
    for name in names:
        save_data[name] = {
            'num_params': all_results[name]['num_params'],
            'mse': all_results[name]['mse'],
            'mae': all_results[name]['mae'],
        }
    with open(os.path.join(out_dir, 'exp6_multilayer_results.json'), 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nAll results saved to {out_dir}/")


if __name__ == '__main__':
    main()
