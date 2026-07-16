import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy.stats import gaussian_kde, spearmanr
from tueplots import bundles

bundles.icml2024()

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data"
RESULTS_ROOT = BASE_DIR / "results"

parser = argparse.ArgumentParser()
parser.add_argument("--loss-kind", type=str, default="beta", choices=["beta", "binary"])
parser.add_argument("--plot-kind", type=str, default="kde", choices=["scatter", "kde"])
args = parser.parse_args()

bench_x = "piqa"
bench_y = "mmlu"
x_name = f"ability_{bench_x}"
y_name = f"ability_{bench_y}"
file_prefix = "prob" if args.loss_kind == "beta" else "binary"
results_dir = RESULTS_ROOT / "transfer_benches_theta_corr"
results_dir.mkdir(parents=True, exist_ok=True)

for irt_model in ("1pl", "2pl"):
    irt_suffix = "_2pl" if irt_model == "2pl" else ""
    input_path = DATA_ROOT / f"5_{file_prefix}_matrix_cated{irt_suffix}.parquet"
    out_path = results_dir / f"{file_prefix}_{irt_model}_{bench_x}_vs_{bench_y}.png"

    df = pd.read_parquet(input_path)
    x = df.index.get_level_values(x_name).to_numpy(dtype=np.float32)
    y = df.index.get_level_values(y_name).to_numpy(dtype=np.float32)

    rho, _ = spearmanr(x, y)

    if args.plot_kind == "kde":
        max_kde_points = 50000
        if x.size > max_kde_points:
            rng = np.random.default_rng(0)
            sample_idx = rng.choice(x.size, size=max_kde_points, replace=False)
            x_kde = x[sample_idx]
            y_kde = y[sample_idx]
        else:
            x_kde = x
            y_kde = y

        kde = gaussian_kde(np.vstack([x_kde, y_kde]))
        x_margin = 0.05 * max(x.max() - x.min(), 1e-6)
        y_margin = 0.05 * max(y.max() - y.min(), 1e-6)
        x_grid = np.linspace(x.min() - x_margin, x.max() + x_margin, 120)
        y_grid = np.linspace(y.min() - y_margin, y.max() + y_margin, 120)
        xx, yy = np.meshgrid(x_grid, y_grid)
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig = plt.figure(figsize=(6, 6))
        gs = gridspec.GridSpec(
            5,
            5,
            figure=fig,
            wspace=0.05,
            hspace=0.05,
            width_ratios=[1, 4, 4, 4, 4],
            height_ratios=[4, 4, 4, 4, 1],
        )
        ax_main = fig.add_subplot(gs[0:4, 1:5])
        ax_left = fig.add_subplot(gs[0:4, 0], sharey=ax_main)
        ax_bottom = fig.add_subplot(gs[4, 1:5], sharex=ax_main)

        if args.plot_kind == "kde":
            ax_main.contourf(xx, yy, zz, levels=30, cmap="Blues")
        else:
            ax_main.scatter(x, y, s=12)
        ax_main.plot(
            [x.min(), x.max()],
            [y.min(), y.max()],
            linestyle="--",
            linewidth=1,
            color="black",
        )
        ax_main.set_xlabel(rf"{x_name} $\theta$", fontsize=18)
        ax_main.set_ylabel(rf"{y_name} $\theta$", fontsize=18)
        ax_main.tick_params(axis="both", labelsize=14)

        ax_left.hist(y, bins=30, orientation="horizontal", color="0.7", alpha=1.0)
        ax_left.invert_xaxis()
        ax_left.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
        ax_left.set_xticks([])
        for spine in ("top", "right", "bottom", "left"):
            ax_left.spines[spine].set_visible(False)

        ax_bottom.hist(x, bins=30, color="0.7", alpha=1.0)
        ax_bottom.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
        ax_bottom.set_yticks([])
        for spine in ("top", "right", "bottom", "left"):
            ax_bottom.spines[spine].set_visible(False)

        fig.suptitle(
            rf"{bench_x} $\theta$ vs {bench_y} $\theta$ ($\rho$ = {rho:.2f})",
            fontsize=18,
        )
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
