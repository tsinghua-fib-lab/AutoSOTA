"""
Experiment 4: Parameter-Matched Baselines
==========================================
All models at ~256 real params. 10 seeds.
Tests on TWO targets: paper's 1D target and a near-singular target.
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
    CauchyNet1D, FNN, count_real_params,
    prepare_1d_data, train_and_eval, cauchy_real_params, matched_fnn_hidden,
    BATCH_SIZE, LR, EPOCHS, HIDDEN_SIZE
)


def near_singular(x):
    """Near-singular target: peak + smooth background."""
    return 1.0 / ((x - 0.3) ** 2 + 0.01) + np.sin(4 * x)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    cauchy_h = HIDDEN_SIZE
    target_params = cauchy_real_params(cauchy_h)
    fnn_h = matched_fnn_hidden(target_params)
    print(f"CauchyNet h={cauchy_h} ({target_params}p) -> FNN h={fnn_h}")

    model_configs = {
        'CauchyNet': lambda: CauchyNet1D(cauchy_h),
        'FNN': lambda: FNN(1, fnn_h, 1),
    }

    print("Parameter counts (real):")
    for name, factory in model_configs.items():
        print(f"  {name}: {count_real_params(factory())}")

    n_seeds = 10
    seeds = list(range(42, 42 + n_seeds))
    N_train = 20

    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)

    # ── Target 1: Paper's 1D function ──
    print(f"\n{'='*50}")
    print("Target: Paper 1D function")
    print('=' * 50)

    X_train, Y_train, X_val, Y_val, X_test, Y_test, _, _ = prepare_1d_data(n_points=N_train)

    results_paper = {}
    for name, factory in model_configs.items():
        results_paper[name] = {'mse': [], 'mae': [], 'num_params': None}
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            r = train_and_eval(factory(), X_train, Y_train, X_test, Y_test,
                               device=device, X_val=X_val, Y_val=Y_val, patience=30)
            results_paper[name]['mse'].append(r['mse'])
            results_paper[name]['mae'].append(r['mae'])
            results_paper[name]['num_params'] = r['num_params']
        mse = np.array(results_paper[name]['mse'])
        print(f"  {name}: MSE={mse.mean():.6f}+/-{mse.std():.6f}")

    # ── Target 2: Near-singular ──
    print(f"\n{'='*50}")
    print("Target: Near-singular")
    print('=' * 50)

    x_train = np.linspace(-1, 1, N_train).astype(np.float32)
    x_test = np.linspace(-1, 1, 500).astype(np.float32)
    y_train_raw = near_singular(x_train).astype(np.float32)
    y_test_raw = near_singular(x_test).astype(np.float32)

    scaler = MinMaxScaler()
    y_train = scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten().astype(np.float32)
    y_test = scaler.transform(y_test_raw.reshape(-1, 1)).flatten().astype(np.float32)

    X_tr = torch.tensor(x_train).unsqueeze(-1)
    Y_tr = torch.tensor(y_train)
    X_te = torch.tensor(x_test).unsqueeze(-1)
    Y_te = torch.tensor(y_test)

    results_singular = {}
    for name, factory in model_configs.items():
        results_singular[name] = {'mse': [], 'mae': [], 'num_params': None}
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            r = train_and_eval(factory(), X_tr, Y_tr, X_te, Y_te,
                               device=device, patience=30)
            results_singular[name]['mse'].append(r['mse'])
            results_singular[name]['mae'].append(r['mae'])
            results_singular[name]['num_params'] = r['num_params']
        mse = np.array(results_singular[name]['mse'])
        print(f"  {name}: MSE={mse.mean():.6f}+/-{mse.std():.6f}")

    # ── Boxplot (side by side) ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {'CauchyNet': 'red', 'FNN': 'blue'}

    for ax, (title, results) in zip(axes, [
        ("Paper's 1D target", results_paper),
        ("Near-singular target", results_singular),
    ]):
        names = list(results.keys())
        data = [results[n]['mse'] for n in names]
        labels = [f"{n}\n({results[n]['num_params']}p)" for n in names]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        for patch, n in zip(bp['boxes'], names):
            patch.set_facecolor(colors[n])
            patch.set_alpha(0.5)
        ax.set_ylabel('Test MSE (scaled)')
        ax.set_title(title, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(alpha=0.3)

    fig.suptitle('Parameter-Matched Comparison (n=20, 10 seeds)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'exp4_param_matched.pdf'), dpi=150)
    plt.savefig(os.path.join(out_dir, 'exp4_param_matched.png'), dpi=150)

    # ── LaTeX ──
    tex = r"""\begin{table}[ht]\centering
\caption{Parameter-matched comparison. $\approx 256$ real params. $n=20$, 10 seeds.}
\label{tab:param_matched}
\begin{tabular}{llccc}
\toprule
Target & Model & \# Params & Test MSE & Test MAE \\
\midrule
"""
    for title, results in [("Paper 1D", results_paper), ("Near-singular", results_singular)]:
        for j, name in enumerate(results):
            r = results[name]
            mse = np.array(r['mse'])
            mae = np.array(r['mae'])
            prefix = title if j == 0 else ""
            tex += (f"{prefix} & {name} & {r['num_params']} & "
                    f"${mse.mean():.6f} \\pm {mse.std():.6f}$ & "
                    f"${mae.mean():.6f} \\pm {mae.std():.6f}$ \\\\\n")
        tex += r"\midrule" + "\n"
    tex = tex.rstrip(r"\midrule" + "\n") + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}"""
    print("\n" + tex)

    with open(os.path.join(out_dir, 'exp4_param_matched_table.tex'), 'w') as f:
        f.write(tex)

    save_data = {
        'Paper 1D': {n: {k: v for k, v in results_paper[n].items()} for n in results_paper},
        'Near-singular': {n: {k: v for k, v in results_singular[n].items()} for n in results_singular},
    }
    with open(os.path.join(out_dir, 'exp4_param_matched_results.json'), 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nSaved to {out_dir}/")


if __name__ == '__main__':
    main()
