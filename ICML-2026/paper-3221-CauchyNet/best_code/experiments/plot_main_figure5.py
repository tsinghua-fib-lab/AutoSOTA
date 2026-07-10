"""Build the camera-ready main empirical figure.

The figure is intentionally compact:
  A. validation loss over the final 10-seed gap-filling run,
  B. per-seed test MAE from the authoritative JSON summary,
  C. one deterministic reconstruction from the same final CauchyNet config.

Inputs:
  results/best_config_gap_filling.json
  results/best_config_gap_filling_curves.npz

Outputs:
  ../../Fig/figure_5_final.pdf
  ../../Fig/figure_5_final.png
"""

import json
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from best_config_gap_filling import CauchyNet, build_data, f


HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(HERE, "results")
FIG_DIR = os.path.join(ROOT, "Fig")

SUMMARY_PATH = os.path.join(RESULTS, "best_config_gap_filling.json")
CURVES_PATH = os.path.join(RESULTS, "best_config_gap_filling_curves.npz")
RECON_PATH = os.path.join(RESULTS, "main_figure5_reconstruction.npz")
PDF_PATH = os.path.join(FIG_DIR, "figure_5_final.pdf")
PNG_PATH = os.path.join(FIG_DIR, "figure_5_final.png")


MODEL_ORDER = ["CauchyNet", "SIREN", "FNN_ReLU", "NBeats"]
MODEL_LABELS = {
    "CauchyNet": "CauchyNet",
    "SIREN": "SIREN",
    "FNN_ReLU": "FNN",
    "NBeats": "N-BEATS",
}
COLORS = {
    "CauchyNet": "#E69F00",  # Okabe-Ito orange
    "SIREN": "#0072B2",      # Okabe-Ito blue
    "FNN_ReLU": "#009E73",   # Okabe-Ito green
    "NBeats": "#CC79A7",     # Okabe-Ito reddish purple
}
LINESTYLES = {
    "CauchyNet": "-",
    "SIREN": "--",
    "FNN_ReLU": "-.",
    "NBeats": ":",
}


def load_summary():
    with open(SUMMARY_PATH, "r") as fp:
        return json.load(fp)


def train_representative_reconstruction(seed=0):
    """Train one final-config CauchyNet for the visual reconstruction panel."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_x, train_y, val_x, val_y, test_x, test_y = build_data(seed=10)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=32, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y), batch_size=32, shuffle=False
    )

    model = CauchyNet(hidden_size=64, r_re=2.5, r_im=0.4, seed=seed)
    opt = optim.Adam(model.parameters(), lr=5e-2)
    crit = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    for _ in range(3000):
        model.train(True)
        for xb, yb in train_loader:
            opt.zero_grad()
            yr, _ = model(xb)
            loss = crit(yr, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.train(False)
        with torch.no_grad():
            val_losses = []
            for xb, yb in val_loader:
                yr, _ = model(xb)
                val_losses.append(crit(yr, yb).item())
            val_loss = float(np.mean(val_losses))
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    x_grid = torch.linspace(-2, 2, 800).unsqueeze(-1)
    model.train(False)
    with torch.no_grad():
        pred_grid, _ = model(x_grid)
        true_grid = f(x_grid.squeeze()).unsqueeze(-1)

    np.savez_compressed(
        RECON_PATH,
        x_grid=x_grid.numpy().ravel(),
        y_true=true_grid.numpy().ravel(),
        y_pred=pred_grid.numpy().ravel(),
        train_x=train_x.numpy().ravel(),
        train_y=train_y.numpy().ravel(),
        val_x=val_x.numpy().ravel(),
        val_y=val_y.numpy().ravel(),
        test_x=test_x.numpy().ravel(),
        test_y=test_y.numpy().ravel(),
        seed=np.array([seed]),
        best_val=np.array([best_val]),
    )


def ensure_reconstruction():
    if "--retrain" in sys.argv or not os.path.exists(RECON_PATH):
        train_representative_reconstruction(seed=0)
    return np.load(RECON_PATH)


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    summary = load_summary()
    curves = np.load(CURVES_PATH)
    recon = ensure_reconstruction()

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        1, 3, figsize=(7.05, 2.3), gridspec_kw={"width_ratios": [1.25, 1.0, 1.15]}
    )

    # A. Validation convergence.
    ax = axes[0]
    for model in MODEL_ORDER:
        arr = curves[f"{model}_val"]
        ep = np.arange(1, arr.shape[1] + 1)
        m = arr.mean(axis=0)
        ax.plot(
            ep,
            m,
            color=COLORS[model],
            linestyle=LINESTYLES[model],
            linewidth=1.1,
            label=MODEL_LABELS[model],
        )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE")
    ax.set_title("Validation loss")
    ax.grid(True, which="major", alpha=0.25, linewidth=0.45)
    ax.set_xlim(0, 3750)
    ax.set_xticks([0, 1000, 2000, 3000])
    end_label_offsets = {
        "CauchyNet": 0.78,
        "SIREN": 0.72,
        "FNN_ReLU": 1.0,
        "NBeats": 1.5,
    }
    for model in MODEL_ORDER:
        arr = curves[f"{model}_val"]
        y = max(float(arr.mean(axis=0)[-1]) * end_label_offsets[model], 3e-6)
        ax.text(
            3075,
            y,
            MODEL_LABELS[model],
            color=COLORS[model],
            fontsize=6.8,
            va="center",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.4},
        )
    despine(ax)

    # B. Per-seed test MAE.
    ax = axes[1]
    per_seed_mae = [
        [row["mae_mean"] for row in summary[model]["per_seed"]]
        for model in MODEL_ORDER
    ]
    box = ax.boxplot(
        per_seed_mae,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 0.8},
        whiskerprops={"linewidth": 0.7},
        capprops={"linewidth": 0.7},
        boxprops={"linewidth": 0.7},
    )
    for patch, model in zip(box["boxes"], MODEL_ORDER):
        patch.set_facecolor(COLORS[model])
        patch.set_alpha(0.35)
    rng = np.random.default_rng(3)
    for i, (model, values) in enumerate(zip(MODEL_ORDER, per_seed_mae), start=1):
        jitter = rng.normal(0, 0.035, size=len(values))
        ax.scatter(
            i + jitter,
            values,
            s=9,
            color=COLORS[model],
            edgecolor="white",
            linewidth=0.25,
            zorder=3,
        )
        ax.scatter(
            i,
            summary[model]["mae_mean"],
            marker="D",
            s=16,
            color=COLORS[model],
            edgecolor="black",
            linewidth=0.35,
            zorder=4,
        )
    ax.set_yscale("log")
    ax.set_ylabel("Test MAE")
    ax.set_title("Held-out gap error")
    ax.set_xticks(range(1, len(MODEL_ORDER) + 1))
    ax.set_xticklabels(
        [
            f"{MODEL_LABELS[m]}\n{summary[m]['params']}p"
            for m in MODEL_ORDER
        ],
        rotation=0,
    )
    ax.grid(True, which="major", axis="y", alpha=0.25, linewidth=0.45)
    despine(ax)

    # C. Representative reconstruction.
    ax = axes[2]
    ax.plot(recon["x_grid"], recon["y_true"], color="black", linewidth=1.1, linestyle="--", label="True")
    ax.plot(
        recon["x_grid"],
        recon["y_pred"],
        color=COLORS["CauchyNet"],
        linewidth=1.35,
        label="CauchyNet",
    )
    ax.scatter(
        recon["train_x"],
        recon["train_y"],
        s=4,
        color="#6B7280",
        alpha=0.35,
        linewidth=0,
        label="Train",
    )
    ax.scatter(
        recon["test_x"],
        recon["test_y"],
        s=5,
        color="#D55E00",
        alpha=0.65,
        linewidth=0,
        label="Held out",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("g(x)")
    ax.set_title("Gap reconstruction")
    ax.grid(True, alpha=0.2, linewidth=0.45)
    ax.legend(loc="upper left", frameon=False, ncol=2, handlelength=1.4, columnspacing=0.8)
    despine(ax)

    for label, ax in zip(["A", "B", "C"], axes):
        ax.text(
            -0.13,
            1.07,
            label,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            va="top",
            ha="left",
        )

    fig.tight_layout(w_pad=1.0)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(PDF_PATH, bbox_inches="tight")
    fig.savefig(PNG_PATH, dpi=450, bbox_inches="tight")
    print(PDF_PATH)
    print(PNG_PATH)


if __name__ == "__main__":
    main()
