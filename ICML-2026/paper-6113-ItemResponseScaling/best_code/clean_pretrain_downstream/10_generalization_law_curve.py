from pathlib import Path
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tueplots import bundles
from scipy.special import expit
import pickle

bundles.icml2024()
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "10_generalization_law_curve"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_DATA_PATH = DATA_DIR / "10_hard_mae_heatmap.pkl"

INPUT_PATH = DATA_DIR / "9_generalization_estimation.pkl"

with open(INPUT_PATH, "rb") as f:
    est_payload = pickle.load(f)

bench_names = sorted(est_payload.keys())
mix_names = sorted({m for b in est_payload.values() for m in set(b["model_data_mix"])})
heat_vals = np.full((len(bench_names), len(mix_names)), np.nan, dtype=np.float32)

for bench_idx, bench in enumerate(tqdm(bench_names, desc="benches")):
    bench_info = est_payload[bench]
    thetas_from_easy = np.array(bench_info["thetas_from_easy"], dtype=np.float32)
    easy_ys = np.array(bench_info["easy_ys"], dtype=np.float32)
    hard_ys = np.array(bench_info["hard_ys"], dtype=np.float32)
    easy_zs = np.array(bench_info["easy_zs"], dtype=np.float32)
    hard_zs = np.array(bench_info["hard_zs"], dtype=np.float32)
    flops = np.array(bench_info["flops"], dtype=np.float64)
    model_data_mix = np.array(bench_info["model_data_mix"])

    easy_gt = np.nanmean(easy_ys, axis=1)
    hard_gt = np.nanmean(hard_ys, axis=1)
    easy_est = expit(thetas_from_easy[:, None] + easy_zs[None, :]).mean(axis=1)
    hard_est = expit(thetas_from_easy[:, None] + hard_zs[None, :]).mean(axis=1)

    output_dir = RESULTS_DIR / bench
    output_dir.mkdir(parents=True, exist_ok=True)

    for mix in sorted(set(model_data_mix)):
        mix_mask = (model_data_mix == mix) & np.isfinite(flops)
        order = np.argsort(flops[mix_mask])
        flops_sorted = flops[mix_mask][order]

        with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(flops_sorted, easy_gt[mix_mask][order], label="Easy GT", color="blue", linestyle="-")
            ax.plot(flops_sorted, hard_gt[mix_mask][order], label="Hard GT", color="red", linestyle="-")
            ax.plot(flops_sorted, easy_est[mix_mask][order], label="Easy Est", color="blue", linestyle="--")
            ax.plot(flops_sorted, hard_est[mix_mask][order], label="Hard Est", color="red", linestyle="--")
            mae_str = f"{np.mean(np.abs(hard_gt[mix_mask] - hard_est[mix_mask])):.1e}".replace("e-0", "e-").replace("e+0", "e+")
            ax.set_title(
                f"{mix}, {bench}\nMAE = Abs(Hard GT - Hard Est) = {mae_str}",
                fontsize=16,
            )
            ax.set_xlabel("FLOP", fontsize=16)
            ax.set_ylabel(r"$\mathrm{p_{Correct Choice}}$", fontsize=16)
            ax.set_xscale("log")
            ax.legend(fontsize=12)
            ax.tick_params(axis="both", labelsize=12)
            fig.tight_layout()
            fig.savefig(output_dir / f"generalization_law_curve_{bench}_{mix}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

    for mix_idx, mix in enumerate(mix_names):
        mask = (model_data_mix == mix) & np.isfinite(flops)
        mae = np.nanmean(np.abs(hard_est[mask] - hard_gt[mask]))
        heat_vals[bench_idx, mix_idx] = mae

# heatmap
abs_max = np.nanmax(np.abs(heat_vals))
vmin, vmax = 0.0, abs_max
fig_w = max(7, 0.6 * len(mix_names) + 2)
fig_h = max(6, 0.5 * len(bench_names) + 2)
with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(heat_vals, aspect="auto", cmap="Blues", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(mix_names)))
    ax.set_xticklabels(mix_names, rotation=45, ha="right", fontsize=12)
    ax.set_yticks(np.arange(len(bench_names)))
    ax.set_yticklabels(bench_names, fontsize=12)
    for i in range(heat_vals.shape[0]):
        for j in range(heat_vals.shape[1]):
            ax.text(j, i, f"{heat_vals[i, j]:.2e}", ha="center", va="center", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("MAE = Abs(Hard GT - Hard Est)", fontsize=14)
    ax.set_xlabel("Data Mix", fontsize=14)
    ax.set_ylabel("Benchmark", fontsize=14)
    heatmap_path = RESULTS_DIR / "hard_mae_heatmap.png"
    fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

with open(HEATMAP_DATA_PATH, "wb") as f:
    pickle.dump(
        {
            "bench_names": bench_names,
            "mix_names": mix_names,
            "heat_vals": heat_vals,
        },
        f,
    )
