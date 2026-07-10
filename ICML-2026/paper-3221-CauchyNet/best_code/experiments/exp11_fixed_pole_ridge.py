"""
Experiment 11: fixed-pole CauchyNet with closed-form coefficient fitting.

This experiment targets the paper's mixed 1D function directly.  Pole
locations are fixed on an ellipse and only the complex output coefficients are
learned.  Because the model is linear in those coefficients, we fit the real
and imaginary coefficient parts with ridge least squares instead of Adam.

The purpose is to separate the fixed Cauchy dictionary from optimization noise
in the older trainable-pole run.  The ReLU FNN baseline is parameter matched to
the trainable-pole h=64 CauchyNet setting and is trained with a longer budget
than the default appendix scripts.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from shared import FNN, cauchy_real_params, matched_fnn_hidden, target_function, train_and_eval


HIDDEN_SIZE = 64
R_RE = 1.5
R_IM = 0.15
N_TRAIN_VALUES = [20, 25, 30, 40, 50]
N_SEEDS = 10
N_TEST = 1000
ALPHAS = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
FNN_EPOCHS = 500
FNN_LR = 1e-2
FNN_PATIENCE = 50


def fixed_pole_features(x, hidden_size=HIDDEN_SIZE, r_re=R_RE, r_im=R_IM):
    """Real design matrix for Re(sum_k theta_k / (xi_k - x)).

    If phi_k = 1 / (xi_k - x) = a_k + i b_k and theta_k = u_k + i v_k,
    then Re(phi_k theta_k) = a_k u_k - b_k v_k.  The real ridge design
    therefore has columns [a_1,...,a_h,-b_1,...,-b_h].
    """
    x = np.asarray(x, dtype=np.float64)
    angles = 2.0 * np.pi * np.arange(hidden_size, dtype=np.float64) / hidden_size
    xi = r_re * np.cos(angles) + 1j * r_im * np.sin(angles)
    phi = 1.0 / (xi[None, :] - x[:, None])
    return np.concatenate([phi.real, -phi.imag], axis=1)


def make_split(n_train, seed, n_test=N_TEST):
    """Create train/validation data and a dense midpoint test grid.

    Targets are scaled using training targets only.  The midpoint test grid
    avoids exact overlap with the train/validation grid.
    """
    n_val = max(8, n_train // 2)
    x_pool = np.linspace(-1.0, 1.0, n_train + n_val, dtype=np.float32)
    y_pool_raw = target_function(x_pool).astype(np.float32)

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(x_pool))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    scaler = MinMaxScaler()
    y_train = scaler.fit_transform(y_pool_raw[train_idx].reshape(-1, 1)).ravel().astype(np.float32)
    y_val = scaler.transform(y_pool_raw[val_idx].reshape(-1, 1)).ravel().astype(np.float32)

    step = 2.0 / n_test
    x_test = (-1.0 + 0.5 * step + step * np.arange(n_test)).astype(np.float32)
    y_test_raw = target_function(x_test).astype(np.float32)
    y_test = scaler.transform(y_test_raw.reshape(-1, 1)).ravel().astype(np.float32)

    return {
        "x_train": x_pool[train_idx],
        "y_train": y_train,
        "x_val": x_pool[val_idx],
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }


def fit_ridge(split):
    """Fit fixed-pole coefficients with validation-selected ridge strength."""
    x_train, y_train = split["x_train"], split["y_train"].astype(np.float64)
    x_val, y_val = split["x_val"], split["y_val"].astype(np.float64)
    x_test, y_test = split["x_test"], split["y_test"].astype(np.float64)

    xtr = fixed_pole_features(x_train)
    xval = fixed_pole_features(x_val)
    xte = fixed_pole_features(x_test)

    best = None
    eye = np.eye(xtr.shape[1], dtype=np.float64)
    xtx = xtr.T @ xtr
    xty = xtr.T @ y_train

    for alpha in ALPHAS:
        try:
            if alpha == 0.0:
                weights = np.linalg.lstsq(xtr, y_train, rcond=1e-10)[0]
            else:
                weights = np.linalg.solve(xtx + alpha * eye, xty)
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(xtx + alpha * eye, xty, rcond=1e-10)[0]

        val_pred = xval @ weights
        val_mse = mean_squared_error(y_val, val_pred)
        if best is None or val_mse < best["val_mse"]:
            best = {"val_mse": float(val_mse), "alpha": float(alpha), "weights": weights}

    test_pred = xte @ best["weights"]
    return {
        "mse": float(mean_squared_error(y_test, test_pred)),
        "mae": float(mean_absolute_error(y_test, test_pred)),
        "alpha": best["alpha"],
        "val_mse": best["val_mse"],
    }


def fit_fnn(split, seed, device):
    """Train the parameter-matched ReLU FNN baseline."""
    target_params = cauchy_real_params(HIDDEN_SIZE)
    fnn_h = matched_fnn_hidden(target_params)

    torch.manual_seed(seed)
    np.random.seed(seed)
    result = train_and_eval(
        FNN(1, fnn_h, 1),
        torch.tensor(split["x_train"]).unsqueeze(-1),
        torch.tensor(split["y_train"]),
        torch.tensor(split["x_test"]).unsqueeze(-1),
        torch.tensor(split["y_test"]),
        epochs=FNN_EPOCHS,
        lr=FNN_LR,
        device=device,
        X_val=torch.tensor(split["x_val"]).unsqueeze(-1),
        Y_val=torch.tensor(split["y_val"]),
        patience=FNN_PATIENCE,
    )
    return {
        "mse": result["mse"],
        "mae": result["mae"],
        "best_epoch": result["best_epoch"],
        "total_epochs": result["total_epochs"],
    }


def summarize(per_seed):
    mses = np.array([r["mse"] for r in per_seed], dtype=np.float64)
    maes = np.array([r["mae"] for r in per_seed], dtype=np.float64)
    return {
        "mse_mean": float(mses.mean()),
        "mse_std": float(mses.std()),
        "mae_mean": float(maes.mean()),
        "mae_std": float(maes.std()),
    }


def fmt_number(x):
    if x < 1e-3:
        return f"{x:.6f}".rstrip("0").rstrip(".")
    return f"{x:.4f}".rstrip("0").rstrip(".")


def write_table(results, out_path):
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{cccc}",
        r"\toprule",
        r"$n_{\mathrm{train}}$ & Fixed-$\xi$ CN ridge (128 p) & FNN (256 p) & ratio \\",
        r"\midrule",
    ]
    for n_train in N_TRAIN_VALUES:
        cell = results[str(n_train)]
        cn = cell["FixedPoleCauchyRidge"]
        fnn = cell["FNN"]
        ratio = fnn["mse_mean"] / cn["mse_mean"]
        lines.append(
            rf"{n_train} & \textbf{{${fmt_number(cn['mse_mean'])} \pm {fmt_number(cn['mse_std'])}$}} "
            rf"& ${fmt_number(fnn['mse_mean'])} \pm {fmt_number(fnn['mse_std'])}$ & ${ratio:.1f}\times$ \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Closed-form fixed-pole CauchyNet on the Paper 1D mixed target.}",
            r"\label{tab:exp11_fixed_pole_ridge}",
            r"\end{table}",
            "",
        ]
    )
    out_path.write_text("\n".join(lines))


def plot_results(results, out_dir):
    x = np.array(N_TRAIN_VALUES)
    cn_mean = np.array([results[str(n)]["FixedPoleCauchyRidge"]["mse_mean"] for n in x])
    cn_sem = np.array([results[str(n)]["FixedPoleCauchyRidge"]["mse_std"] for n in x]) / np.sqrt(N_SEEDS)
    fnn_mean = np.array([results[str(n)]["FNN"]["mse_mean"] for n in x])
    fnn_sem = np.array([results[str(n)]["FNN"]["mse_std"] for n in x]) / np.sqrt(N_SEEDS)

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.errorbar(x, cn_mean, yerr=cn_sem, marker="o", color="#C2410C", linewidth=2.0,
                capsize=3, label="Fixed-pole Cauchy ridge")
    ax.errorbar(x, fnn_mean, yerr=fnn_sem, marker="s", color="#2563EB", linewidth=2.0,
                capsize=3, label="FNN")
    ax.set_yscale("log")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Dense-grid test MSE")
    ax.set_title("Mixed 1D target")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "exp11_fixed_pole_ridge.pdf", dpi=200)
    fig.savefig(out_dir / "exp11_fixed_pole_ridge.png", dpi=200)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)

    print(f"Device: {device}")
    print(
        "Fixed-pole Cauchy ridge: "
        f"h={HIDDEN_SIZE}, params={2 * HIDDEN_SIZE}, ellipse=({R_RE}, {R_IM})"
    )
    print(
        "FNN baseline: "
        f"h={matched_fnn_hidden(cauchy_real_params(HIDDEN_SIZE))}, "
        f"params={cauchy_real_params(HIDDEN_SIZE)}, epochs={FNN_EPOCHS}"
    )

    results = {
        "target": "Paper 1D mixed target",
        "protocol": {
            "hidden_size": HIDDEN_SIZE,
            "r_re": R_RE,
            "r_im": R_IM,
            "fixed_pole_params": 2 * HIDDEN_SIZE,
            "fnn_params": cauchy_real_params(HIDDEN_SIZE),
            "n_test": N_TEST,
            "n_seeds": N_SEEDS,
            "target_scaling": "MinMaxScaler fitted on training targets only",
            "ridge_alphas": ALPHAS,
            "fnn_epochs": FNN_EPOCHS,
            "fnn_lr": FNN_LR,
            "fnn_patience": FNN_PATIENCE,
        },
        "results": {},
    }

    for n_train in N_TRAIN_VALUES:
        print(f"\n== n_train={n_train} ==")
        cn_runs = []
        fnn_runs = []
        for seed in range(N_SEEDS):
            split = make_split(n_train=n_train, seed=seed)
            cn_runs.append(fit_ridge(split))
            fnn_runs.append(fit_fnn(split, seed=seed, device=device))

        cn_summary = summarize(cn_runs)
        fnn_summary = summarize(fnn_runs)
        ratio = fnn_summary["mse_mean"] / cn_summary["mse_mean"]
        print(
            f"  Fixed-pole ridge: MSE={cn_summary['mse_mean']:.6g} "
            f"+/- {cn_summary['mse_std']:.6g}, MAE={cn_summary['mae_mean']:.6g}"
        )
        print(
            f"  FNN:              MSE={fnn_summary['mse_mean']:.6g} "
            f"+/- {fnn_summary['mse_std']:.6g}, MAE={fnn_summary['mae_mean']:.6g}"
        )
        print(f"  Ratio FNN/CN:     {ratio:.1f}x")

        results["results"][str(n_train)] = {
            "FixedPoleCauchyRidge": {
                **cn_summary,
                "num_params": 2 * HIDDEN_SIZE,
                "per_seed": cn_runs,
            },
            "FNN": {
                **fnn_summary,
                "num_params": cauchy_real_params(HIDDEN_SIZE),
                "per_seed": fnn_runs,
            },
            "ratio_fnn_over_cauchy": float(ratio),
        }

    json_path = out_dir / "exp11_fixed_pole_ridge_results.json"
    table_path = out_dir / "exp11_fixed_pole_ridge_table.tex"
    json_path.write_text(json.dumps(results, indent=2))
    write_table(results["results"], table_path)
    plot_results(results["results"], out_dir)
    print(f"\nSaved {json_path}")
    print(f"Saved {table_path}")


if __name__ == "__main__":
    main()
