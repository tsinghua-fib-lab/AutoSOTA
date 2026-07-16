from pathlib import Path
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from tueplots import bundles
from scipy.special import expit
import pickle
import sys
bundles.icml2024()
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent))
from utils import calibrate_1pl_theta, compute_pass_datk_gts, compute_pass_datk_irt

DEVICE = "cuda:7"
PROB_THRESHOLD = 0.005

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "6_generalization_across_aime"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_DATA_PATH = DATA_DIR / "5_hard_mae_heatmap.pkl"

input_path = DATA_DIR / "1_calibreated_irsl_testtime_resmat2.pt"
payload = torch.load(input_path, map_location="cpu", weights_only=False)

data_tensor = np.array(payload["data_tensor"], dtype=np.float32)
model_names = list(payload["models"])
test_models = list(payload["test_models"])
test_model_indices = [i for i, m in enumerate(model_names) if m in set(test_models)]
datasets = list(payload["datasets"])
zs = np.array(payload["zs"], dtype=np.float32)

aime24_indices = [i for i, d in enumerate(datasets) if d == "aime2024"]
aime25_indices = [i for i, d in enumerate(datasets) if d == "aime2025"]
bench_tensor_24 = data_tensor[test_model_indices][:, aime24_indices, :]
bench_tensor_25 = data_tensor[test_model_indices][:, aime25_indices, :]

probmat_24 = np.nanmean(bench_tensor_24, axis=-1)
zs_24 = torch.tensor(zs[aime24_indices])
thetas = calibrate_1pl_theta(torch.tensor(probmat_24), DEVICE, zs_24)

n_samples = data_tensor.shape[-1]
sample_arange = np.arange(1, n_samples + 1)
aime_mae = {}
for model_idx, model_name in enumerate(test_models):
    model_tensor_24 = bench_tensor_24[model_idx]
    model_tensor_25 = bench_tensor_25[model_idx]

    mask_24 = np.nanmean(model_tensor_24, axis=-1) >= PROB_THRESHOLD
    mask_25 = np.nanmean(model_tensor_25, axis=-1) >= PROB_THRESHOLD
    model_tensor_24 = model_tensor_24[mask_24]
    model_tensor_25 = model_tensor_25[mask_25]
    zs_24_masked = zs[aime24_indices][mask_24]
    zs_25_masked = zs[aime25_indices][mask_25]

    pass_24_gt = compute_pass_datk_gts(model_tensor_24)
    pass_25_gt = compute_pass_datk_gts(model_tensor_25)

    theta = float(thetas[model_idx])
    probs_24 = expit(theta + zs_24_masked)
    probs_25 = expit(theta + zs_25_masked)
    pass_24_est = compute_pass_datk_irt(probs_24, n_samples)
    pass_25_est = compute_pass_datk_irt(probs_25, n_samples)
    aime_mae[model_name] = float(np.mean(np.abs(pass_25_est - pass_25_gt)))

    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.loglog(sample_arange, -np.log(pass_24_gt), label="AIME24 GT", color="blue")
        ax.loglog(sample_arange, -np.log(pass_24_est), label="AIME24 Est", color="blue", linestyle="--")
        ax.loglog(sample_arange, -np.log(pass_25_gt), label="AIME25 GT", color="red")
        ax.loglog(sample_arange, -np.log(pass_25_est), label="AIME25 Est", color="red", linestyle="--")
        mae_str = f"{np.mean(np.abs(pass_25_est - pass_25_gt)):.1e}".replace("e-0", "e-").replace("e+0", "e+")
        ax.set_title(f"{model_name}, aime_24_to_25\nMAE = Abs(AIME25 GT - AIME25 Est) = {mae_str}", fontsize=16)
        ax.set_xlabel(r"Number of Samples $k$", fontsize=16)
        ax.set_ylabel(r"$-\log(\mathrm{Pass@k})$", fontsize=16)
        ax.legend(fontsize=12)
        ax.tick_params(axis="both", labelsize=12)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"generalization_across_aime_{model_name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

# heatmap
with open(HEATMAP_DATA_PATH, "rb") as f:
    heat_data = pickle.load(f)

bench_names = heat_data["bench_names"]
model_names = heat_data["model_names"]
heat_vals = np.array(heat_data["heat_vals"], dtype=np.float32)

aime_row = np.full((1, len(model_names)), np.nan, dtype=np.float32)
for j, m in enumerate(model_names):
    aime_row[0, j] = aime_mae[m]

heat_vals = np.vstack([heat_vals, aime_row])
bench_names = bench_names + ["aime_24_to_25"]

abs_max = np.nanmax(np.abs(heat_vals))
vmin, vmax = 0.0, abs_max
fig_w = max(4.6, 0.4 * len(model_names) + 0.8)
fig_h = max(3.0, 0.22 * len(bench_names) + 0.6)
with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(heat_vals, aspect="auto", cmap="Blues", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(bench_names)))
    ax.set_yticklabels(bench_names, fontsize=10)
    for i in range(heat_vals.shape[0]):
        for j in range(heat_vals.shape[1]):
            label = f"{heat_vals[i, j]:.1e}".replace("e-0", "e-").replace("e+0", "e+")
            ax.text(j, i, label, ha="center", va="center", fontsize=10)
    ax.axhline(y=len(bench_names) - 1.5, color="black", linewidth=2.0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("MAE = Abs(Hard GT - Hard Est)", fontsize=10)
    cbar.ax.tick_params(labelsize=10)
    ax.set_xlabel("LLM", fontsize=10)
    ax.set_ylabel("Benchmark", fontsize=10)
    heatmap_path = RESULTS_DIR / "hard_mae_heatmap_with_aime.png"
    fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# density distribution of MAE = Abs(Hard GT - Hard Est)
within_vals = heat_vals[:-1].ravel()
cross_vals = heat_vals[-1].ravel()
within_vals = within_vals[np.isfinite(within_vals)]
cross_vals = cross_vals[np.isfinite(cross_vals)]
with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
    fig, ax = plt.subplots(figsize=(6, 4))
    total_count = len(within_vals) + len(cross_vals)
    within_w = np.ones_like(within_vals) / total_count if total_count > 0 else None
    cross_w = np.ones_like(cross_vals) / total_count if total_count > 0 else None
    ax.hist(within_vals, bins=30, density=False, weights=within_w, alpha=0.6, color="lightskyblue", label="Within-Benchmark Transfer")
    ax.hist(cross_vals, bins=30, density=False, weights=cross_w, alpha=0.6, color="lightcoral", label="Cross-Benchmark Transfer")
    ax.set_title("MAE Density: Abs(Hard GT - Hard Est)", fontsize=16)
    ax.set_xlabel("MAE", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "hard_mae_density.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
