import argparse
from pathlib import Path
import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tueplots import bundles
import sys
import pickle
bundles.icml2024()
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent))
from utils import compute_pass_datk_gts, compute_pass_datk_irt
from scipy.special import expit

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "5_generalization_law_curve"
PROB_THRESHOLD = 0.005
HEATMAP_DATA_PATH = DATA_DIR / "5_hard_mae_heatmap.pkl"

parser = argparse.ArgumentParser()
parser.add_argument("--input-hf-repo", type=str, default="irsl_testtime_resmat2", choices=["irsl_testtime_resmat1", "irsl_testtime_resmat2"])
args = parser.parse_args()

est_path = DATA_DIR / f"4_generalization_estimation_{args.input_hf_repo}.pt"
est_payload = torch.load(est_path, map_location="cpu", weights_only=False)

dataset_names = sorted(est_payload.keys())
model_names = list(est_payload[dataset_names[0]]["model_names"])
heat_vals = np.full((len(dataset_names), len(model_names)), np.nan, dtype=np.float32)

for dataset_idx, dataset in enumerate(tqdm(dataset_names, desc="datasets")):
    bench_info = est_payload[dataset]
    test_models = list(bench_info["model_names"])
    theta_from_easy = np.array(bench_info["theta_from_easy"], dtype=np.float32)
    easy_zs = np.array(bench_info["easy_zs"], dtype=np.float32)
    hard_zs = np.array(bench_info["hard_zs"], dtype=np.float32)
    easy_tensor = np.array(bench_info["easy_tensor"], dtype=np.float32)
    hard_tensor = np.array(bench_info["hard_tensor"], dtype=np.float32)

    output_dir = RESULTS_DIR / args.input_hf_repo / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    n_samples = easy_tensor.shape[-1]
    sample_arange = np.arange(1, n_samples + 1)
    
    for model_idx, model_name in enumerate(test_models):
        easy_model_tensor = easy_tensor[model_idx]
        hard_model_tensor = hard_tensor[model_idx]

        easy_mask = np.nanmean(easy_model_tensor, axis=-1) >= PROB_THRESHOLD
        hard_mask = np.nanmean(hard_model_tensor, axis=-1) >= PROB_THRESHOLD
        easy_model_tensor = easy_model_tensor[easy_mask]
        hard_model_tensor = hard_model_tensor[hard_mask]
        easy_zs_masked = easy_zs[easy_mask]
        hard_zs_masked = hard_zs[hard_mask]

        pass_easy_gt = compute_pass_datk_gts(easy_model_tensor)
        pass_hard_gt = compute_pass_datk_gts(hard_model_tensor)

        theta = float(theta_from_easy[model_idx])
        easy_probs = expit(theta + easy_zs_masked)
        hard_probs = expit(theta + hard_zs_masked)
        pass_easy_est = compute_pass_datk_irt(easy_probs, n_samples)
        pass_hard_est = compute_pass_datk_irt(hard_probs, n_samples)
        heat_vals[dataset_idx, model_idx] = float(np.mean(np.abs(pass_hard_est - pass_hard_gt)))

        with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.loglog(sample_arange, -np.log(pass_easy_gt), label="Easy GT", color="blue")
            ax.loglog(sample_arange, -np.log(pass_hard_gt), label="Hard GT", color="red")
            ax.loglog(sample_arange, -np.log(pass_easy_est), label="Easy Est", color="blue", linestyle="--")
            ax.loglog(sample_arange, -np.log(pass_hard_est), label="Hard Est", color="red", linestyle="--")
            mae_str = f"{np.mean(np.abs(pass_hard_gt - pass_hard_est)):.1e}".replace("e-0", "e-").replace("e+0", "e+")
            ax.set_title(
                f"{model_name}, {dataset}\nMAE = Abs(Hard GT - Hard Est) = {mae_str}",
                fontsize=16,
            )
            ax.set_xlabel(r"Number of Samples $k$", fontsize=16)
            ax.set_ylabel(r"$-\log(\mathrm{Pass@k})$", fontsize=16)
            ax.legend(fontsize=12)
            ax.tick_params(axis="both", labelsize=12)
            fig.tight_layout()
            fig.savefig(
                output_dir / f"hardeasy_{args.input_hf_repo.replace('4_generalization_estimation_irsl_testtime_', '')}_{model_name}_{dataset}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

# heatmap: rows=bench, cols=model
abs_max = np.nanmax(np.abs(heat_vals))
vmin, vmax = 0.0, abs_max
fig_w = max(7, 0.6 * len(model_names) + 2)
fig_h = max(6, 0.5 * len(dataset_names) + 2)
with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(heat_vals, aspect="auto", cmap="Blues", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=12)
    ax.set_yticks(np.arange(len(dataset_names)))
    ax.set_yticklabels(dataset_names, fontsize=12)
    for i in range(heat_vals.shape[0]):
        for j in range(heat_vals.shape[1]):
            ax.text(j, i, f"{heat_vals[i, j]:.2e}", ha="center", va="center", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("MAE = Abs(Hard Est - Hard GT)", fontsize=14)
    ax.set_xlabel("Model", fontsize=14)
    ax.set_ylabel("Benchmark", fontsize=14)
    heatmap_path = RESULTS_DIR / args.input_hf_repo / "hard_mae_heatmap.png"
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

with open(HEATMAP_DATA_PATH, "wb") as f:
    pickle.dump(
        {
            "bench_names": dataset_names,
            "model_names": model_names,
            "heat_vals": heat_vals,
        },
        f,
    )
