import warnings
from pathlib import Path
import os
import multiprocessing as mp
import pickle

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

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "3_law_curve"
PROB_THRESHOLD = 0.005

def plot_law_curve(output_dir, tag, filter_status, model, bench, sample_arange, pass_datk_gts, pass_datk_subset_subsample_passat1, pass_datk_irts_beta, mae_irt_beta, mae_sub_passat1):
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        # ax_left, ax_right = axes
        # ax_left.plot(sample_arange, pass_datk_gts, label="full unbiased (GT)", linewidth=2, color="blue")
        # ax_left.plot(sample_arange, pass_datk_subset_subsample_passat1, label="sub pass@1", linestyle="--", linewidth=2, color="blue")
        # ax_left.plot(sample_arange, pass_datk_irts_beta, label="sub beta-IRT", linewidth=2, color="red")
        # ax_left.set_xlabel("Number of Samples", fontsize=16)
        # ax_left.set_ylabel("Pass@k", fontsize=16)
        # ax_left.set_ylim(0, 1)
        # ax_left.legend(fontsize=14)
        # ax_left.tick_params(axis="both", labelsize=14)
        # ax_right.loglog(sample_arange, -np.log(pass_datk_gts), label="full unbiased (GT)", linewidth=2, color="blue")
        # ax_right.loglog(sample_arange, -np.log(pass_datk_subset_subsample_passat1), label="sub pass@1", linestyle="--", color="blue")
        # ax_right.loglog(sample_arange, -np.log(pass_datk_irts_beta), label="sub beta-IRT", linewidth=2, color="red")
        # ax_right.set_xlabel("Number of Samples", fontsize=16)
        # ax_right.set_ylabel(r"$-\log(\mathrm{Pass@k})$", fontsize=16)
        # ax_right.legend(fontsize=14)
        # ax_right.tick_params(axis="both", labelsize=14)
        
        fig, ax_right = plt.subplots(1, 1, figsize=(6, 6))
        ax_right.loglog(sample_arange, -np.log(pass_datk_gts), label="Ground Truth", linewidth=2, color="black")
        ax_right.loglog(sample_arange, -np.log(pass_datk_subset_subsample_passat1), label="Traditional", linestyle="--", color="blue")
        ax_right.loglog(sample_arange, -np.log(pass_datk_irts_beta), label="IRSL", linewidth=2, linestyle="--", color="red")
        ax_right.set_xlabel(r"Number of Samples $k$", fontsize=16)
        ax_right.set_ylabel(r"$-\log(\mathrm{Pass@k})$", fontsize=16)
        ax_right.legend(fontsize=14)
        ax_right.tick_params(axis="both", labelsize=14)
        beta_str = f"{mae_irt_beta:.1e}".replace("e-0", "e-").replace("e+0", "e+")
        classic_str = f"{mae_sub_passat1:.1e}".replace("e-0", "e-").replace("e+0", "e+")
        diff_str = f"{(mae_sub_passat1 - mae_irt_beta):.1e}".replace("e-0", "e-").replace("e+0", "e+")
        fig.suptitle(
            f"{model}, {bench}\n" #, {filter_status}\n"
            f"Traditional MAE={classic_str}, IRSL MAE={beta_str}\n"
            f"Traditional MAE - IRSL MAE= {diff_str}",
            fontsize=16,
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"law_curve_{tag}_{model}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

def process_config(args):
    payload, stem, bench, model_idx = args
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

    output_dir = RESULTS_DIR / stem / bench
    output_dir.mkdir(parents=True, exist_ok=True)

    idxs = [j for j, b in enumerate(datasets) if b == bench]
    bench_tensor = data_tensor[test_model_indices][:, idxs, :]
    n_test_takers, n_items, n_samples = bench_tensor.shape

    theta_pairs = test_thetas_by_bench[bench]
    theta_map = {m: float(t) for m, t in theta_pairs}
    test_thetas = np.array([theta_map[m] for m in test_models], dtype=np.float32)

    model = test_models[model_idx]
    model_tensor = bench_tensor[model_idx]

    if alphas is None:
        irt_probs_beta = torch.sigmoid(torch.tensor(test_thetas[model_idx] + zs[idxs])).cpu().numpy()
    else:
        irt_probs_beta = torch.sigmoid(torch.tensor(alphas[idxs] * (test_thetas[model_idx] + zs[idxs]))).cpu().numpy()

    # before filter
    passat1s_fullset = np.nanmean(model_tensor, axis=-1)
    passat1s_subset_subsample = np.nanmean(model_tensor[:item_budget, :sample_budget], axis=-1)
    pass_datk_gts = compute_pass_datk_gts(model_tensor)
    pass_datk_subset_subsample_passat1 = compute_pass_datk_irt(passat1s_subset_subsample, n_samples)
    pass_datk_irts_beta = compute_pass_datk_irt(irt_probs_beta, n_samples)
    mae_irt_beta = np.mean(np.abs(pass_datk_gts - pass_datk_irts_beta))
    mae_sub_passat1 = np.mean(np.abs(pass_datk_gts - pass_datk_subset_subsample_passat1))

    sample_arange = np.arange(1, n_samples + 1)
    plot_law_curve(
        output_dir=output_dir,
        tag="before_filter",
        filter_status="Before Filter",
        model=model,
        bench=bench,
        sample_arange=sample_arange,
        pass_datk_gts=pass_datk_gts,
        pass_datk_subset_subsample_passat1=pass_datk_subset_subsample_passat1,
        pass_datk_irts_beta=pass_datk_irts_beta,
        mae_irt_beta=mae_irt_beta,
        mae_sub_passat1=mae_sub_passat1,
    )

    # after filter
    mask = (passat1s_fullset >= PROB_THRESHOLD)
    model_tensor = model_tensor[mask]
    passat1s_subset_subsample = np.nanmean(model_tensor[:item_budget, :sample_budget], axis=-1)
    irt_probs_beta = irt_probs_beta[mask]
    pass_datk_gts = compute_pass_datk_gts(model_tensor)
    pass_datk_subset_subsample_passat1 = compute_pass_datk_irt(passat1s_subset_subsample, n_samples)
    pass_datk_irts_beta = compute_pass_datk_irt(irt_probs_beta, n_samples)
    mae_irt_beta_after = np.mean(np.abs(pass_datk_gts - pass_datk_irts_beta))
    mae_sub_passat1_after = np.mean(np.abs(pass_datk_gts - pass_datk_subset_subsample_passat1))

    plot_law_curve(
        output_dir=output_dir,
        tag="after_filter",
        filter_status="After Filter",
        model=model,
        bench=bench,
        sample_arange=sample_arange,
        pass_datk_gts=pass_datk_gts,
        pass_datk_subset_subsample_passat1=pass_datk_subset_subsample_passat1,
        pass_datk_irts_beta=pass_datk_irts_beta,
        mae_irt_beta=mae_irt_beta_after,
        mae_sub_passat1=mae_sub_passat1_after,
    )

    return bench, model, {
        "mae_irt_beta_before_filter": mae_irt_beta,
        "mae_sub_passat1_before_filter": mae_sub_passat1,
        "mae_irt_beta_after_filter": mae_irt_beta_after,
        "mae_sub_passat1_after_filter": mae_sub_passat1_after,
    }

if __name__ == "__main__":
    pt_files = sorted(DATA_DIR.glob("2_cated_*.pt"))
    for pt_path in tqdm(pt_files, desc="plot"):
        stem = pt_path.stem.replace("2_cated_irsl_testtime_", "")
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        datasets = list(payload["datasets"])
        test_models = list(payload["test_models"])
        unique_benches = sorted(set(datasets))

        tasks = []
        for bench in unique_benches:
            for model_idx in range(len(test_models)):
                tasks.append((payload, stem, bench, model_idx))

        n_cpus = int(os.cpu_count() * 0.8)
        with mp.Pool(processes=n_cpus) as pool:
            results = list(tqdm(pool.imap(process_config, tasks), total=len(tasks), desc=f"{stem} tasks", leave=False))

        results_dict = {}
        for bench, model, metrics in results:
            results_dict.setdefault(bench, {})
            results_dict[bench][model] = metrics

        # heatmap
        diffs = {}
        for bench in unique_benches:
            row = {}
            bench_dict = results_dict[bench]
            for model in test_models:
                mae_beta = bench_dict[model]["mae_irt_beta_after_filter"]
                mae_pass1 = bench_dict[model]["mae_sub_passat1_after_filter"]
                row[model] = mae_pass1 - mae_beta
            diffs[bench] = row
        vals = np.array([[diffs[b][m] for m in test_models] for b in unique_benches], dtype=np.float32)
        abs_max = np.max(np.abs(vals))
        vmin, vmax = -abs_max, abs_max
        fig_w = max(4.6, 0.4 * len(test_models) + 0.8)
        fig_h = max(3.0, 0.22 * len(unique_benches) + 0.6)
        with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            im = ax.imshow(vals, aspect="auto", cmap="bwr_r", vmin=vmin, vmax=vmax)
            ax.set_xticks(np.arange(len(test_models)))
            ax.set_xticklabels(test_models, rotation=45, ha="right", fontsize=10)
            ax.set_yticks(np.arange(len(unique_benches)))
            ax.set_yticklabels(unique_benches, fontsize=10)
            for i in range(vals.shape[0]):
                for j in range(vals.shape[1]):
                    label = f"{vals[i, j]:.1e}".replace("e-0", "e-").replace("e+0", "e+")
                    ax.text(j, i, label, ha="center", va="center", fontsize=10)
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label("Traditional MAE - IRSL MAE", fontsize=10)
            cbar.ax.tick_params(labelsize=10)
            ax.set_xlabel("LLM", fontsize=10)
            ax.set_ylabel("Benchmark", fontsize=10)
            ax.set_title("MAE Difference", fontsize=10) # After Filter", fontsize=10)
            heatmap_path = RESULTS_DIR / stem / f"{stem}_heatmap.png"
            plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

        # density distribution of Traditional MAE - IRSL MAE
        mae_vals = []
        for bench in unique_benches:
            for model in test_models:
                mae_beta = results_dict[bench][model]["mae_irt_beta_after_filter"]
                mae_pass1 = results_dict[bench][model]["mae_sub_passat1_after_filter"]
                mae_vals.append(mae_pass1 - mae_beta)
        mae_vals = np.array(mae_vals, dtype=np.float32)
        mae_vals = mae_vals[np.isfinite(mae_vals)]
        with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(mae_vals, bins=30, density=True, alpha=0.6, color="steelblue")
            ax.set_title("MAE Density: Traditional MAE - IRSL MAE", fontsize=16)
            ax.set_xlabel("Traditional MAE - IRSL MAE", fontsize=16)
            ax.set_ylabel("Density", fontsize=16)
            ax.tick_params(axis="both", labelsize=12)
            fig.tight_layout()
            density_path = RESULTS_DIR / stem / f"{stem}_hard_mae_density.png"
            plt.savefig(density_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
