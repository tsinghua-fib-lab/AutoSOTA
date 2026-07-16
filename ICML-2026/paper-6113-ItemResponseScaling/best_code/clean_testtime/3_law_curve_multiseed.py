import argparse
from pathlib import Path
import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from tueplots import bundles

import sys

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent))
from utils import compute_pass_datk_gts, compute_pass_datk_irt

bundles.icml2024()
warnings.filterwarnings("ignore")

PROB_THRESHOLD = 0.005
OUTPUT_ROOT = BASE_DIR / "results" / "3_law_curve_multiseed"


def compute_heatmap_vals(payload):
    data_tensor = np.array(payload["data_tensor"], dtype=np.float32)
    model_names = list(payload["models"])
    test_models = list(payload["test_models"])
    test_model_indices = [i for i, m in enumerate(model_names) if m in set(test_models)]
    datasets = list(payload["datasets"])
    zs = np.array(payload["zs"], dtype=np.float32)
    alphas = None if payload.get("alphas") is None else np.array(payload.get("alphas"), dtype=np.float32)
    test_thetas_by_bench = payload["test_thetas"]
    sample_budget = int(payload["sample_budget"])
    item_budget = int(payload["item_budget"])

    unique_benches = sorted(set(datasets))
    vals = np.full((len(unique_benches), len(test_models)), np.nan, dtype=np.float32)

    for bench_idx, bench in enumerate(unique_benches):
        idxs = [j for j, b in enumerate(datasets) if b == bench]
        bench_tensor = data_tensor[test_model_indices][:, idxs, :]
        theta_pairs = test_thetas_by_bench[bench]
        theta_map = {m: float(t) for m, t in theta_pairs}
        test_thetas = np.array([theta_map[m] for m in test_models], dtype=np.float32)

        if alphas is None:
            irt_probs_all = torch.sigmoid(
                torch.tensor(test_thetas[:, None] + zs[idxs][None, :], dtype=torch.float32)
            ).cpu().numpy()
        else:
            irt_probs_all = torch.sigmoid(
                torch.tensor(alphas[idxs][None, :] * (test_thetas[:, None] + zs[idxs][None, :]), dtype=torch.float32)
            ).cpu().numpy()

        for model_idx, _model in enumerate(test_models):
            model_tensor = bench_tensor[model_idx]
            passat1s_fullset = np.nanmean(model_tensor, axis=-1)
            passat1s_subset_subsample = np.nanmean(model_tensor[:item_budget, :sample_budget], axis=-1)
            irt_probs_beta = irt_probs_all[model_idx]

            mask = passat1s_fullset >= PROB_THRESHOLD
            model_tensor = model_tensor[mask]
            passat1s_subset_subsample = passat1s_subset_subsample[mask]
            irt_probs_beta = irt_probs_beta[mask]

            pass_datk_gts = compute_pass_datk_gts(model_tensor)
            pass_datk_subset_subsample_passat1 = compute_pass_datk_irt(
                passat1s_subset_subsample,
                model_tensor.shape[-1],
            )
            pass_datk_irts_beta = compute_pass_datk_irt(irt_probs_beta, model_tensor.shape[-1])
            mae_irt_beta = np.mean(np.abs(pass_datk_gts - pass_datk_irts_beta))
            mae_sub_passat1 = np.mean(np.abs(pass_datk_gts - pass_datk_subset_subsample_passat1))
            vals[bench_idx, model_idx] = mae_sub_passat1 - mae_irt_beta

    bench_avg_vals = np.nanmean(vals, axis=1, keepdims=True)
    return unique_benches, bench_avg_vals


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiseed-root", type=Path, default=BASE_DIR / "data_multiseed")
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    seed_dirs = sorted(path for path in args.multiseed_root.iterdir() if path.is_dir())

    stem_to_seed_vals = {}
#     stem_to_seed_vals = {
#     "irsl_testtime_resmat2": [
#         vals_seed0,   # shape: [n_benches, 4]
#         vals_seed1,
#         vals_seed2,
#         ...
#     ],
#     "irsl_testtime_resmat2_2pl": [
#         ...
#     ],  ...
# }
    stem_to_benches = {}
#     stem_to_benches = {
#     "irsl_testtime_resmat2": ["bench_a", "bench_b", ...],
#     "irsl_testtime_resmat2_2pl": ["bench_a", "bench_b", ...],
# }

    for seed_dir in tqdm(seed_dirs, desc="seeds"):
        pt_files = sorted(seed_dir.glob("2_cated_*.pt"))

        for pt_path in pt_files:
            stem = pt_path.stem.replace("2_cated_", "")
            payload = torch.load(pt_path, map_location="cpu", weights_only=False)
            bench_names, vals = compute_heatmap_vals(payload)
            if stem not in stem_to_seed_vals:
                stem_to_seed_vals[stem] = []
                stem_to_benches[stem] = bench_names
            else:
                old_benches = stem_to_benches[stem]
                assert old_benches == bench_names
            stem_to_seed_vals[stem].append(vals)

    for stem, seed_vals in stem_to_seed_vals.items():
        stacked = np.stack(seed_vals, axis=0)
        mean_vals = np.nanmean(stacked, axis=0)
        std_vals = np.nanstd(stacked, axis=0)
        bench_names = stem_to_benches[stem]
        col_labels = ["Avg Test LLM"]

        output_dir = OUTPUT_ROOT / stem
        output_dir.mkdir(parents=True, exist_ok=True)

        abs_max = np.nanmax(np.abs(mean_vals))
        vmin, vmax = (-abs_max, abs_max) if np.isfinite(abs_max) else (-1.0, 1.0)
        fig_w = max(4.6, 0.4 * len(col_labels) + 0.8)
        fig_h = max(3.0, 0.22 * len(bench_names) + 0.6)
        with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            im = ax.imshow(mean_vals, aspect="auto", cmap="bwr_r", vmin=vmin, vmax=vmax)
            ax.set_xticks(np.arange(len(col_labels)))
            ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=14)
            ax.set_yticks(np.arange(len(bench_names)))
            ax.set_yticklabels(bench_names, fontsize=14)
            for i in range(mean_vals.shape[0]):
                for j in range(mean_vals.shape[1]):
                    mean_label = f"{mean_vals[i, j]:.1e}".replace("e-0", "e-").replace("e+0", "e+")
                    std_label = f"{std_vals[i, j]:.1e}".replace("e-0", "e-").replace("e+0", "e+")
                    ax.text(j, i, f"{mean_label}\n±{std_label}", ha="center", va="center", fontsize=12)
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label("Traditional MAE - IRSL MAE", fontsize=14)
            cbar.ax.tick_params(labelsize=14)
            ax.set_xlabel("LLM", fontsize=14)
            ax.set_ylabel("Benchmark", fontsize=14)
            ax.set_title("MAE Difference", fontsize=14)
            plt.savefig(output_dir / f"{stem}_heatmap.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

        n_benches = len(bench_names)
        ncols = 2
        nrows = 2
        with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
            fig, axes = plt.subplots(nrows, ncols, figsize=(10, 7), sharex=True, sharey=True)
            axes = np.atleast_1d(axes).ravel()
            for bench_idx, bench_name in enumerate(bench_names):
                ax = axes[bench_idx]
                bench_seed_vals = stacked[:, bench_idx, 0]
                bench_seed_vals = bench_seed_vals[np.isfinite(bench_seed_vals)]
                bench_mean = float(np.mean(bench_seed_vals))
                bench_std = float(np.std(bench_seed_vals))
                mean_label = f"{bench_mean:.1e}".replace("e-0", "e-").replace("e+0", "e+")
                std_label = f"{bench_std:.1e}".replace("e-0", "e-").replace("e+0", "e+")
                ax.hist(bench_seed_vals, bins=20, density=True, color="steelblue", alpha=0.75)
                mean_kwargs = {"color": "darkblue", "linestyle": "--", "linewidth": 1.8}
                zero_kwargs = {"color": "red", "linestyle": "-", "linewidth": 1.5}
                if bench_idx == n_benches - 1:
                    ax.axvline(bench_mean, label="Mean", **mean_kwargs)
                    ax.axvline(0.0, label="Zero", **zero_kwargs)
                    ax.legend(fontsize=14, loc="upper right")
                else:
                    ax.axvline(bench_mean, **mean_kwargs)
                    ax.axvline(0.0, **zero_kwargs)
                ax.set_title(f"{bench_name}\nmean={mean_label}, std={std_label}", fontsize=16)
                if bench_idx >= ncols:
                    ax.set_xlabel("Traditional MAE - IRSL MAE", fontsize=15)
                else:
                    ax.set_xlabel("")
                if bench_idx % ncols == 0:
                    ax.set_ylabel("Density", fontsize=15)
                else:
                    ax.set_ylabel("")
                ax.tick_params(axis="both", labelsize=14)
            for ax in axes[n_benches:]:
                ax.axis("off")
            fig.tight_layout()
            plt.savefig(output_dir / f"{stem}_bench_distributions.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
