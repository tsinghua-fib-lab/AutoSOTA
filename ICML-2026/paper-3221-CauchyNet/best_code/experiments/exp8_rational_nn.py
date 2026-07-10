"""
Experiment 8: Rational Neural Networks Comparison
==================================================
Compares CauchyNet against Rational Neural Networks (Boulle et al. 2020)
as requested by reviewers.

Rational NN uses Pade-type rational activation functions:
  sigma(x) = P(x)/Q(x) where P, Q are learnable polynomials.

We implement a simple version with degree (3,2) Pade approximant.
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

from shared import (
    CauchyNet1D, count_real_params, train_and_eval,
    cauchy_real_params,
    HIDDEN_SIZE, BATCH_SIZE, LR, EPOCHS
)

# ──────────────────────────────────────────────────────────────────
# Rational NN models
# ──────────────────────────────────────────────────────────────────

class RationalActivation(nn.Module):
    """Learnable rational activation: P(x)/Q(x) with Pade (m,n) = (3,2).

    Follows Boulle et al. (2020) "Rational neural networks".
    P(x) = a0 + a1*x + a2*x^2 + a3*x^3
    Q(x) = 1 + |b1|*x^2 + |b2|*x^4   (ensure no real poles via even powers + abs)
    """
    def __init__(self):
        super().__init__()
        # Initialize to approximate ReLU
        self.a = nn.Parameter(torch.tensor([0.0, 1.0, 0.1, 0.01]))  # [a0, a1, a2, a3]
        self.b = nn.Parameter(torch.tensor([0.1, 0.01]))  # [b1, b2]

    def forward(self, x):
        P = self.a[0] + self.a[1] * x + self.a[2] * x**2 + self.a[3] * x**3
        Q = 1.0 + torch.abs(self.b[0]) * x**2 + torch.abs(self.b[1]) * x**4
        return P / Q


class RationalNN(nn.Module):
    """Neural network with rational activation functions (single hidden layer)."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act1 = RationalActivation()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        return self.fc2(x)


class RationalNN_Deep(nn.Module):
    """Two-hidden-layer Rational NN."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act1 = RationalActivation()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.act2 = RationalActivation()
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)


# ──────────────────────────────────────────────────────────────────
# Target functions
# ──────────────────────────────────────────────────────────────────

def target_1d(x):
    """Paper's 1D target function."""
    return (
        1.0 / ((x + 0.6) ** 2 + 0.005)
        - 40.0 * np.exp(-2.0 * (x + 0.4) ** 2)
        + 50.0 * np.sign(x) * np.abs(np.sin(3 * x) + 0.8) ** 1.5 * np.sin(10 * x)
    )

def near_singular(x):
    """Near-singular function where rational methods should excel."""
    return 1.0 / ((x - 0.3) ** 2 + 0.01) + 1.0 / ((x + 0.5) ** 2 + 0.02)

def smooth_oscillatory(x):
    """Smooth oscillatory function."""
    return np.sin(8 * x) * np.exp(-2 * x ** 2)


FUNCTIONS = {
    'Paper 1D target': target_1d,
    'Near-singular': near_singular,
    'Smooth oscillatory': smooth_oscillatory,
}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    N_train = 50
    N_test = 500
    n_seeds = 10
    seeds = list(range(42, 42 + n_seeds))

    # Parameter-matched single-layer models
    cauchy_h = HIDDEN_SIZE
    target_params = cauchy_real_params(cauchy_h)
    # RationalNN: same as FNN but activation adds 6 params (a[4]+b[2])
    # params = (input+1)*h + (h+1)*output + 6 = 3h+7 for 1D
    rat_h = (target_params - 7) // 3
    print(f"CauchyNet h={cauchy_h} ({target_params}p) -> RatNN h={rat_h}")

    model_factories = {
        'CauchyNet': lambda: CauchyNet1D(cauchy_h),
        'RationalNN': lambda: RationalNN(1, rat_h, 1),
    }

    # Print param counts
    print("Parameter counts (real):")
    for name, factory in model_factories.items():
        m = factory()
        print(f"  {name}: {count_real_params(m)}")

    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}

    for fn_name, fn in FUNCTIONS.items():
        print(f"\n{'='*50}")
        print(f"Function: {fn_name}")
        print('=' * 50)

        x_train = np.linspace(-1, 1, N_train).astype(np.float32)
        x_test = np.linspace(-1, 1, N_test).astype(np.float32)
        y_train_raw = fn(x_train).astype(np.float32)
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

        for model_name, factory in model_factories.items():
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
                'num_params': count_real_params(factory()),
                'preds': last_preds.tolist(),
            }
            print(f"  {model_name}: MSE={np.mean(mse_list):.6f}+/-{np.std(mse_list):.6f}  "
                  f"Params={count_real_params(factory())}")

    # ──────────────────────────────────────────────────────────────
    # Plots
    # ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {
        'CauchyNet': 'red', 'RationalNN': 'purple'
    }
    x_plot = np.linspace(-1, 1, N_test)

    for i, (fn_name, fn) in enumerate(FUNCTIONS.items()):
        ax = axes[i]
        y_true_raw = fn(x_plot)
        scaler_plot = MinMaxScaler()
        y_true = scaler_plot.fit_transform(y_true_raw.reshape(-1, 1)).flatten()
        ax.plot(x_plot, y_true, 'k--', linewidth=2, label='True')

        for model_name in model_factories:
            preds = all_results[fn_name][model_name]['preds']
            ax.plot(x_plot, preds, color=colors[model_name], linewidth=1.5,
                    label=model_name, alpha=0.8)

        ax.set_title(fn_name, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle('CauchyNet vs Rational Neural Networks (scaled)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'exp8_rational_nn.pdf'), dpi=150)
    plt.savefig(os.path.join(out_dir, 'exp8_rational_nn.png'), dpi=150)

    # ──────────────────────────────────────────────────────────────
    # LaTeX table
    # ──────────────────────────────────────────────────────────────
    tex = r"""\begin{table}[ht]\centering
\caption{CauchyNet vs Rational Neural Networks (Boull\'e et al. 2020).
$h=64$, 10 random seeds, 50 training points, MinMaxScaler.}
\label{tab:rational}
\begin{tabular}{llccc}
\toprule
Function & Model & \# Params & Test MSE & Test MAE \\
\midrule
"""
    for fn_name in FUNCTIONS:
        for j, model_name in enumerate(model_factories):
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

    with open(os.path.join(out_dir, 'exp8_rational_table.tex'), 'w') as f:
        f.write(tex)

    # JSON (without large preds arrays)
    save_data = {}
    for fn_name in all_results:
        save_data[fn_name] = {}
        for m in all_results[fn_name]:
            d = {k: v for k, v in all_results[fn_name][m].items() if k != 'preds'}
            save_data[fn_name][m] = d
    with open(os.path.join(out_dir, 'exp8_rational_results.json'), 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nAll results saved to {out_dir}/")


if __name__ == '__main__':
    main()
