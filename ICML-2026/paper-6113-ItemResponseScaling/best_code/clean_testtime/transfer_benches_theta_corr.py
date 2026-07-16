import warnings
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from tueplots import bundles

bundles.icml2024()
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "transfer_benches_theta_corr"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

input_hf_repo = "irsl_testtime_resmat2"
bench_x = "aime2024"
bench_y = "aime2025"

for irt_model in ("1pl", "2pl"):
    irt_suffix = "_2pl" if irt_model == "2pl" else ""
    calibrate_path = DATA_DIR / f"1_calibreated_{input_hf_repo}{irt_suffix}.pt"
    input_path = DATA_DIR / f"2_cated_{input_hf_repo}{irt_suffix}.pt"
    out_path = RESULTS_DIR / f"{input_hf_repo}{irt_suffix}_{bench_x}_vs_{bench_y}.png"

    calibrate_payload = torch.load(calibrate_path, map_location="cpu", weights_only=False)
    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    train_pairs_x = calibrate_payload["train_thetas"][bench_x]
    train_pairs_y = calibrate_payload["train_thetas"][bench_y]
    test_pairs_x = payload["test_thetas"][bench_x]
    test_pairs_y = payload["test_thetas"][bench_y]
    theta_map_x = {
        **{model_name: float(theta) for model_name, theta in train_pairs_x},
        **{model_name: float(theta) for model_name, theta in test_pairs_x},
    }
    theta_map_y = {
        **{model_name: float(theta) for model_name, theta in train_pairs_y},
        **{model_name: float(theta) for model_name, theta in test_pairs_y},
    }
    all_models = sorted(set(theta_map_x) & set(theta_map_y))

    x = np.array([theta_map_x[model_name] for model_name in all_models], dtype=np.float32)
    y = np.array([theta_map_y[model_name] for model_name in all_models], dtype=np.float32)

    rho, _ = spearmanr(x, y)

    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(x, y, s=24)
        ax.plot(
            [x.min(), x.max()],
            [y.min(), y.max()],
            linestyle="--",
            linewidth=1,
            color="black",
        )
        ax.set_xlabel(r"aime2024 $\theta$", fontsize=18)
        ax.set_ylabel(r"aime2025 $\theta$", fontsize=18)
        ax.tick_params(axis="both", labelsize=14)
        fig.suptitle(rf"aime2024 $\theta$ vs aime2025 $\theta$ ($\rho$ = {rho:.2f})", fontsize=18)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
