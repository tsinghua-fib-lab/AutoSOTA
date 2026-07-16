from pathlib import Path
import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tueplots import bundles

bundles.icml2024()
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "transfer_benches_theta_corr_scaleup"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

input_hf_repo = "irsl_testtime_resmat2"

for irt_model in ("1pl", "2pl"):
    irt_suffix = "_2pl" if irt_model == "2pl" else ""
    calibrate_path = DATA_DIR / f"1_calibreated_{input_hf_repo}{irt_suffix}.pt"
    cat_path = DATA_DIR / f"2_cated_{input_hf_repo}{irt_suffix}.pt"
    out_path = RESULTS_DIR / f"{input_hf_repo}{irt_suffix}_theta_corr_heatmap.png"

    calibrate_payload = torch.load(calibrate_path, map_location="cpu", weights_only=False)
    cat_payload = torch.load(cat_path, map_location="cpu", weights_only=False)

    bench_names = sorted(set(calibrate_payload["train_thetas"]) | set(cat_payload["test_thetas"]))
    n_benches = len(bench_names)
    print(f"n_benches: {n_benches}")
    heat_vals = np.full((n_benches, n_benches), np.nan, dtype=np.float32)

    theta_by_bench = {}
    for bench in bench_names:
        train_pairs = calibrate_payload["train_thetas"].get(bench, [])
        test_pairs = cat_payload["test_thetas"].get(bench, [])
        theta_map = {
            **{model_name: float(theta) for model_name, theta in train_pairs},
            **{model_name: float(theta) for model_name, theta in test_pairs},
        }
        theta_by_bench[bench] = theta_map

    for i, bench_i in enumerate(bench_names):
        for j, bench_j in enumerate(bench_names):
            common_models = sorted(set(theta_by_bench[bench_i]) & set(theta_by_bench[bench_j]))
            x = np.array([theta_by_bench[bench_i][model_name] for model_name in common_models], dtype=np.float32)
            y = np.array([theta_by_bench[bench_j][model_name] for model_name in common_models], dtype=np.float32)
            rho, _ = spearmanr(x, y)
            heat_vals[i, j] = rho

    fig_w = max(7, 0.55 * len(bench_names) + 2.5)
    fig_h = max(5.5, 0.5 * len(bench_names) + 1.5)
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(heat_vals, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(bench_names)))
        ax.set_xticklabels(bench_names, rotation=45, ha="right", fontsize=18)
        ax.set_yticks(np.arange(len(bench_names)))
        ax.set_yticklabels(bench_names, fontsize=18)
        for i in range(heat_vals.shape[0]):
            for j in range(heat_vals.shape[1]):
                ax.text(j, i, f"{heat_vals[i, j]:.2f}", ha="center", va="center", fontsize=18)
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Correlation", fontsize=24)
        cbar.ax.tick_params(labelsize=20)
        ax.set_xlabel("Benchmark", fontsize=24)
        ax.set_ylabel("Benchmark", fontsize=24)
        ax.set_title(rf"$\theta$ Correlation", fontsize=26)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
