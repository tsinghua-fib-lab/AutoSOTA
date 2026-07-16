import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from tueplots import bundles

bundles.icml2024()

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data"
RESULTS_ROOT = BASE_DIR / "results" / "transfer_benches_theta_corr_scaleup"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--loss-kind", type=str, default="beta", choices=["beta", "binary"])
args = parser.parse_args()

file_prefix = "prob" if args.loss_kind == "beta" else "binary"

for irt_model in ("1pl", "2pl"):
    irt_suffix = "_2pl" if irt_model == "2pl" else ""
    input_path = DATA_ROOT / f"5_{file_prefix}_matrix_cated{irt_suffix}.parquet"
    out_path = RESULTS_ROOT / f"{file_prefix}_{irt_model}_theta_corr_heatmap.png"

    df = pd.read_parquet(input_path)
    bench_names = sorted(
        name.replace("ability_", "")
        for name in df.index.names
        if isinstance(name, str) and name.startswith("ability_")
    )
    n_benches = len(bench_names)
    print(f"n benches: {n_benches}")
    heat_vals = np.full((n_benches, n_benches), np.nan, dtype=np.float32)

    for i, bench_i in enumerate(bench_names):
        x = df.index.get_level_values(f"ability_{bench_i}").to_numpy(dtype=np.float32)
        for j, bench_j in enumerate(bench_names):
            y = df.index.get_level_values(f"ability_{bench_j}").to_numpy(dtype=np.float32)
            rho, _ = spearmanr(x, y)
            heat_vals[i, j] = rho

    fig_w = max(10, 0.9 * len(bench_names) + 4)
    fig_h = max(7, 0.6 * len(bench_names) + 2)
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(heat_vals, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(bench_names)))
        ax.set_xticklabels(bench_names, rotation=45, ha="right", fontsize=20)
        ax.set_yticks(np.arange(len(bench_names)))
        ax.set_yticklabels(bench_names, fontsize=20)
        for i in range(heat_vals.shape[0]):
            for j in range(heat_vals.shape[1]):
                ax.text(j, i, f"{heat_vals[i, j]:.2f}", ha="center", va="center", fontsize=16)
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Correlation", fontsize=20)
        cbar.ax.tick_params(labelsize=16)
        ax.set_xlabel("Benchmark", fontsize=20)
        ax.set_ylabel("Benchmark", fontsize=20)
        ax.set_title(rf"$\theta$ Correlation", fontsize=22)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
