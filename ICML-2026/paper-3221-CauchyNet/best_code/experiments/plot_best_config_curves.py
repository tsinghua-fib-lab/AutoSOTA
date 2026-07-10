"""
Train all five models at the best CauchyNet config (from sweep_cauchynet.py)
and plot their training/validation loss curves. Produces a publication-quality
PDF for the paper supplement.

Best config (sweep_cauchynet.py result):
    h        = 64
    lr       = 5e-2
    n_train  = 200
    n_val    = 50
    epochs   = 3000
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from best_config_gap_filling import (
    CauchyNet, SIREN, FNN_ReLU, _make_nbeats, train_score_one,
)
from sweep_cauchynet import build_dataset

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    H, LR, N_TRAIN, N_VAL, EPOCHS, N_SEEDS = 64, 5e-2, 200, 50, 3000, 10

    tX, tY, vX, vY, teX, teY = build_dataset(N_TRAIN, N_VAL, n_test=100, seed=10)
    print(f"Train {len(tX)}, Val {len(vX)}, Test {len(teX)}")
    train_loader = DataLoader(TensorDataset(tX, tY), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(vX, vY), batch_size=32, shuffle=False)
    test_data = (teX, teY)

    # FNN params: 2H + (H+1). For H=64: 193.
    # N-BEATS at h_nb: 3 * (h_nb^2 + 5 h_nb + 2). Pick h_nb to match FNN.
    fnn_p = 2 * H + (H + 1)
    h_nb = max(2, int(round((-5 + (25 + 4 * (fnn_p / 3 - 2)) ** 0.5) / 2)))

    factories = {
        "CauchyNet": lambda s: CauchyNet(hidden_size=H, r_re=2.5, r_im=0.4, seed=s),
        "SIREN":     lambda s: SIREN(hidden_size=H),
        "FNN_ReLU":  lambda s: FNN_ReLU(h=H),
        "NBeats":    lambda s: _make_nbeats(h=h_nb, num_blocks=3),
    }
    imag_pens = {}  # all default to 0.0

    curves = {}
    for name, factory in factories.items():
        print(f"\n--- {name} ({N_SEEDS} seeds, {EPOCHS} epochs, lr={LR}) ---")
        tr, vl, maes = [], [], []
        for s in range(N_SEEDS):
            torch.manual_seed(s); np.random.seed(s)
            model = factory(s)
            res = train_score_one(
                f"{name} seed{s}", model, train_loader, val_loader, test_data,
                epochs=EPOCHS, lr=LR, imag_pen=imag_pens.get(name, 0.0), log=False,
            )
            tr.append(res["train_curve"]); vl.append(res["val_curve"])
            maes.append(res["mae_mean"])
        curves[name] = dict(
            train=np.array(tr), val=np.array(vl),
            mae_mean=float(np.mean(maes)), mae_std=float(np.std(maes)),
            params=res["params"],
        )
        print(f"  MAE = {curves[name]['mae_mean']:.4f} ± {curves[name]['mae_std']:.4f}  "
              f"params = {curves[name]['params']}")

    # Save raw curves
    npz_path = os.path.join(OUT_DIR, "best_config_curves.npz")
    np.savez_compressed(
        npz_path,
        **{f"{k}_train": v["train"] for k, v in curves.items()},
        **{f"{k}_val": v["val"] for k, v in curves.items()},
    )
    print(f"\nSaved {npz_path}")

    # ── Plot ─────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
        "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    colors = {
        "CauchyNet": "#e66101",  # orange
        "SIREN":     "#1f78b4",  # blue
        "FNN_ReLU":  "#33a02c",  # green
        "NBeats":    "#6a3d9a",  # purple
    }
    labels = {
        "CauchyNet": "CauchyNet",
        "SIREN":     "SIREN",
        "FNN_ReLU":  "FNN (ReLU)",
        "NBeats":    "N-BEATS",
    }
    order = ["CauchyNet", "NBeats", "FNN_ReLU", "SIREN"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for k in order:
        v = curves[k]
        ep = np.arange(1, v["train"].shape[1] + 1)
        for ax, arr in zip(axes, (v["train"], v["val"])):
            m = arr.mean(0); s = arr.std(0)
            ax.plot(ep, m, label=labels[k], color=colors[k], linewidth=1.6)
            ax.fill_between(ep, np.maximum(m - s, 1e-9), m + s,
                            color=colors[k], alpha=0.15, linewidth=0)
    for ax, title in zip(axes, ("Training loss (MSE)", "Validation loss (MSE)")):
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="upper right", frameon=True, framealpha=0.9)
    fig.suptitle(
        f"Gap-filling loss curves at best config "
        f"($h{{=}}{H}$, lr$={LR}$, $n_\\mathrm{{train}}{{=}}{N_TRAIN}$, "
        f"{EPOCHS} epochs, {N_SEEDS} seeds, $\\pm 1\\sigma$)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    pdf_path = os.path.join(OUT_DIR, "best_config_curves.pdf")
    png_path = os.path.join(OUT_DIR, "best_config_curves.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
