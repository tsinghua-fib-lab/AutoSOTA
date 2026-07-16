from pathlib import Path
import warnings
import numpy as np
import pandas as pd
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
from utils import calibrate_1pl_theta, calculate_flops

DEVICE = "cuda:3"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "11_generalization_across_arc"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_DATA_PATH = DATA_DIR / "10_hard_mae_heatmap.pkl"

INPUT_PATH = DATA_DIR / "4_prob_matrix_calibrated.parquet"

resmat_df = pd.read_parquet(INPUT_PATH)
test_df = resmat_df[resmat_df.index.get_level_values("model_split") == "test"].copy()

test_ys = test_df.to_numpy(dtype=np.float32)
bench_names = test_df.columns.get_level_values("bench_name")
zs = test_df.columns.get_level_values("difficulty").to_numpy(dtype=np.float32)

arc_easy_mask = bench_names == "arc_easy"
arc_challenge_mask = bench_names == "arc_challenge"

arc_easy_ys = test_ys[:, arc_easy_mask]
arc_challenge_ys = test_ys[:, arc_challenge_mask]
arc_easy_zs = zs[arc_easy_mask]
arc_challenge_zs = zs[arc_challenge_mask]

thetas = calibrate_1pl_theta(torch.tensor(arc_easy_ys), DEVICE, torch.tensor(arc_easy_zs))

index_df = test_df.index.to_frame(index=False)
model_data_mix = np.array(index_df["model_data_mix"])
model_size = index_df["model_size"].tolist()
model_step = index_df["model_step"].astype(int).tolist()

max_step = (
    index_df.groupby(["model_data_mix", "model_size"], as_index=False)["model_step"]
    .max()
    .rename(columns={"model_step": "max_model_step"})
)
index_with_max = index_df.merge(max_step, on=["model_data_mix", "model_size"], how="left")
is_final = (index_with_max["model_step"] == index_with_max["max_model_step"]).to_numpy()
flops = [
    calculate_flops(size, step) if is_final[i] else np.nan
    for i, (size, step) in enumerate(zip(model_size, model_step))
]
flops = np.array(flops, dtype=np.float64)

arc_easy_gt = np.nanmean(arc_easy_ys, axis=1)
arc_challenge_gt = np.nanmean(arc_challenge_ys, axis=1)
arc_easy_est = expit(thetas[:, None] + arc_easy_zs[None, :]).mean(axis=1)
arc_challenge_est = expit(thetas[:, None] + arc_challenge_zs[None, :]).mean(axis=1)

for mix in tqdm(sorted(set(model_data_mix)), desc="data_mix"):
    mix_mask = (model_data_mix == mix) & np.isfinite(flops)
    order = np.argsort(flops[mix_mask])
    flops_sorted = flops[mix_mask][order]

    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(flops_sorted, arc_easy_gt[mix_mask][order], label="ARC Easy GT", color="blue", linestyle="-")
        ax.plot(flops_sorted, arc_challenge_gt[mix_mask][order], label="ARC Challenge GT", color="red", linestyle="-")
        ax.plot(flops_sorted, arc_easy_est[mix_mask][order], label="ARC Easy Est", color="blue", linestyle="--")
        ax.plot(flops_sorted, arc_challenge_est[mix_mask][order], label="ARC Challenge Est", color="red", linestyle="--")
        mae_str = f"{np.mean(np.abs(arc_challenge_gt[mix_mask] - arc_challenge_est[mix_mask])):.1e}".replace("e-0", "e-").replace("e+0", "e+")
        ax.set_title(
            f"{mix}, arc_easy_to_challenge\nMAE = Abs(ARC Challenge GT - ARC Challenge Est) = {mae_str}",
            fontsize=16,
        )
        ax.set_xlabel("FLOP", fontsize=16)
        ax.set_ylabel(r"$\mathrm{p_{Correct Choice}}$", fontsize=16)
        ax.set_xscale("log")
        ax.legend(fontsize=12)
        ax.tick_params(axis="both", labelsize=12)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"generalization_across_arc_{mix}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

# heatmap
with open(HEATMAP_DATA_PATH, "rb") as f:
    heat_data = pickle.load(f)

bench_names = heat_data["bench_names"]
mix_names = heat_data["mix_names"]
heat_vals = np.array(heat_data["heat_vals"], dtype=np.float32)

arc_row = np.full((1, len(mix_names)), np.nan, dtype=np.float32)
for mix_idx, mix in enumerate(mix_names):
    mix_mask = (model_data_mix == mix) & np.isfinite(flops)
    mae = np.nanmean(np.abs(arc_challenge_est[mix_mask] - arc_challenge_gt[mix_mask]))
    arc_row[0, mix_idx] = mae

heat_vals = np.vstack([heat_vals, arc_row])
bench_names = bench_names + ["arc_easy_to_challenge"]

abs_max = np.nanmax(np.abs(heat_vals))
vmin, vmax = 0.0, abs_max
fig_w = max(10, 0.9 * len(mix_names) + 4)
fig_h = max(7, 0.6 * len(bench_names) + 2)
split_idx = len(bench_names) - 1
with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(heat_vals, aspect="auto", cmap="Blues", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(mix_names)))
    ax.set_xticklabels(mix_names, rotation=45, ha="right", fontsize=20)
    ax.set_yticks(np.arange(len(bench_names)))
    ax.set_yticklabels(bench_names, fontsize=20)
    for i in range(heat_vals.shape[0]):
        for j in range(heat_vals.shape[1]):
            label = f"{heat_vals[i, j]:.1e}".replace("e-0", "e-").replace("e+0", "e+")
            ax.text(j, i, label, ha="center", va="center", fontsize=18)
    ax.axhline(y=split_idx - 0.5, color="black", linewidth=2.0, xmin=0.0, xmax=1.0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("MAE = Abs(Hard GT - Hard Est)", fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    ax.set_xlabel("LLM Data Mixture", fontsize=20)
    ax.set_ylabel("Benchmark", fontsize=20)
    heatmap_path = RESULTS_DIR / "hard_mae_heatmap_with_arc.png"
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
