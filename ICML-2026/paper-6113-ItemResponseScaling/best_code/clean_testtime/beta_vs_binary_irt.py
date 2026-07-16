import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from tqdm import tqdm
from tueplots import bundles

bundles.icml2024()
torch.manual_seed(0)
torch.set_num_threads(1)

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent))
from utils import (
    calibrate_1pl_z,
    cat_beta_1pl,
    cat_binary_1pl,
    compute_pass_datk_gts,
    compute_pass_datk_irt,
)


HF_CACHE_ROOT = BASE_DIR / ".hf_cache"
os.environ["XDG_CACHE_HOME"] = str(HF_CACHE_ROOT)
os.environ["HF_HOME"] = str(HF_CACHE_ROOT / "huggingface")
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_ROOT / "huggingface" / "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ["HF_HUB_CACHE"]
os.environ["HF_XET_CACHE"] = str(HF_CACHE_ROOT / "huggingface" / "xet")
os.environ["HF_ASSETS_CACHE"] = str(HF_CACHE_ROOT / "huggingface" / "assets")
os.environ["HF_HUB_DISABLE_XET"] = "1"

for path in (
    HF_CACHE_ROOT,
    Path(os.environ["HF_HOME"]),
    Path(os.environ["HF_HUB_CACHE"]),
    Path(os.environ["HF_XET_CACHE"]),
    Path(os.environ["HF_ASSETS_CACHE"]),
):
    path.mkdir(parents=True, exist_ok=True)


INPUT_HF_REPO = "irsl_testtime_resmat2"
SPLIT_SEEDS = list(range(100))
CALIBRATE_DEVICE = "cuda:7"
DATA_ROOT = BASE_DIR / "data"
RESULTS_ROOT = BASE_DIR / "results" / "beta_vs_binary_irt"
N_MODELS_FOR_TEST = 4
SAMPLE_BUDGET = 50
ITEM_BUDGET = 30
PROB_THRESHOLD = 0.005


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def estimate_thetas_for_bench(bench_tensor: np.ndarray, bench_zs: np.ndarray, loss_kind: str) -> np.ndarray:
    if loss_kind == "beta":
        obs = np.nanmean(bench_tensor[:, :, :SAMPLE_BUDGET], axis=-1)
        cat_fn = cat_beta_1pl
    elif loss_kind == "binary":
        obs = bench_tensor[:, :, 0]
        cat_fn = cat_binary_1pl
    else:
        raise ValueError(f"Unknown loss_kind: {loss_kind}")

    obs_t = torch.tensor(obs, dtype=torch.float32, device="cpu")
    zs_t = torch.tensor(bench_zs, dtype=torch.float32, device="cpu")

    def _run_one(i: int) -> float:
        theta_path = cat_fn(obs_t[i], zs_t, "cpu", budget=ITEM_BUDGET)
        return float(theta_path[-1])

    theta_list = Parallel(n_jobs=max(1, int((os.cpu_count() or 1) * 0.8)))(
        delayed(_run_one)(i) for i in range(obs_t.shape[0])
    )
    return np.asarray(theta_list, dtype=np.float32)


def compute_maes_for_bench(
    bench_tensor: np.ndarray,
    z_beta: np.ndarray,
    z_binary: np.ndarray,
    theta_beta: np.ndarray,
    theta_binary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_models, _, n_samples = bench_tensor.shape
    beta_maes = np.full(n_models, np.nan, dtype=np.float32)
    binary_maes = np.full(n_models, np.nan, dtype=np.float32)

    for model_idx in range(n_models):
        model_tensor = bench_tensor[model_idx]
        passat1_full = np.nanmean(model_tensor, axis=-1)
        mask = passat1_full >= PROB_THRESHOLD

        filtered_tensor = model_tensor[mask]
        gt_curve = compute_pass_datk_gts(filtered_tensor)
        beta_curve = compute_pass_datk_irt(sigmoid(theta_beta[model_idx] + z_beta[mask]), n_samples)
        binary_curve = compute_pass_datk_irt(sigmoid(theta_binary[model_idx] + z_binary[mask]), n_samples)
        beta_maes[model_idx] = float(np.mean(np.abs(gt_curve - beta_curve)))
        binary_maes[model_idx] = float(np.mean(np.abs(gt_curve - binary_curve)))

    return beta_maes, binary_maes


if __name__ == "__main__":
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    cache_dir = snapshot_download(
        repo_id=f"stair-lab/{INPUT_HF_REPO}",
        repo_type="dataset",
        cache_dir=os.environ["HF_HUB_CACHE"],
    )
    payload = torch.load(f"{cache_dir}/resmat.pt", map_location="cpu", weights_only=False)
    data_tensor = np.asarray(payload["data_tensor"], dtype=np.float32)
    model_names = list(payload["models"])
    datasets = list(payload["datasets"])
    unique_benches = sorted(set(datasets))
    dataset_to_indices = {
        bench: np.asarray([i for i, d in enumerate(datasets) if d == bench], dtype=np.int64)
        for bench in unique_benches
    }

    seed_vals = []

    for split_seed in tqdm(SPLIT_SEEDS, desc="split seeds"):
        rng = np.random.default_rng(split_seed)
        perm = rng.permutation(len(model_names))
        permuted_tensor = data_tensor[perm]
        n_train = permuted_tensor.shape[0] - N_MODELS_FOR_TEST
        train_tensor = permuted_tensor[:n_train]
        test_tensor = permuted_tensor[n_train:]

        train_beta = np.nanmean(train_tensor, axis=-1)
        train_binary = train_tensor[:, :, 0]

        z_beta = calibrate_1pl_z(
            torch.tensor(train_beta, dtype=torch.float32, device=CALIBRATE_DEVICE),
            device=CALIBRATE_DEVICE,
            loss_kind="beta",
        )
        z_binary = calibrate_1pl_z(
            torch.tensor(train_binary, dtype=torch.float32, device=CALIBRATE_DEVICE),
            device=CALIBRATE_DEVICE,
            loss_kind="binary",
        )

        vals = np.full((len(unique_benches), N_MODELS_FOR_TEST), np.nan, dtype=np.float32)

        for bench_idx, bench in enumerate(tqdm(unique_benches, desc=f"seed={split_seed} benches", leave=False)):
            bench_indices = dataset_to_indices[bench]
            bench_test_tensor = test_tensor[:, bench_indices, :]
            bench_z_beta = z_beta[bench_indices]
            bench_z_binary = z_binary[bench_indices]

            theta_beta = estimate_thetas_for_bench(bench_test_tensor, bench_z_beta, loss_kind="beta")
            theta_binary = estimate_thetas_for_bench(bench_test_tensor, bench_z_binary, loss_kind="binary")
            beta_maes, binary_maes = compute_maes_for_bench(
                bench_tensor=bench_test_tensor,
                z_beta=bench_z_beta,
                z_binary=bench_z_binary,
                theta_beta=theta_beta,
                theta_binary=theta_binary,
            )

            vals[bench_idx] = binary_maes - beta_maes

        seed_vals.append(np.nanmean(vals, axis=1, keepdims=True))

    stacked_vals = np.stack(seed_vals, axis=0)
    mean_vals = np.nanmean(stacked_vals, axis=0)
    std_vals = np.nanstd(stacked_vals, axis=0)
    col_labels = ["Avg Test LLM"]

    summary = {
        "split_seeds": SPLIT_SEEDS,
        "bench_names": unique_benches,
        "col_labels": col_labels,
        "stacked_vals": stacked_vals,
        "mean_vals": mean_vals,
        "std_vals": std_vals,
    }

    with open(RESULTS_ROOT / "beta_vs_binary_irt.pkl", "wb") as f:
        pickle.dump(summary, f)

    abs_max = np.nanmax(np.abs(mean_vals))
    vmin, vmax = -abs_max, abs_max
    fig_w = max(4.6, 0.4 * len(col_labels) + 0.8)
    fig_h = max(3.5, 0.34 * len(unique_benches) + 1.0)
    heatmap_path = RESULTS_ROOT / "beta_vs_binary_irt_heatmap.png"

    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(mean_vals, aspect="auto", cmap="bwr_r", vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=10)
        ax.set_yticks(np.arange(len(unique_benches)))
        ax.set_yticklabels(unique_benches, fontsize=10)
        for i in range(mean_vals.shape[0]):
            for j in range(mean_vals.shape[1]):
                mean_label = f"{mean_vals[i, j]:.1e}".replace("e-0", "e-").replace("e+0", "e+")
                std_label = f"{std_vals[i, j]:.1e}".replace("e-0", "e-").replace("e+0", "e+")
                ax.text(j, i, f"{mean_label}\n±{std_label}", ha="center", va="center", fontsize=8)
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Binary-IRT MAE - Beta-IRT MAE", fontsize=10)
        cbar.ax.tick_params(labelsize=10)
        ax.set_xlabel("LLM", fontsize=10)
        ax.set_ylabel("Benchmark", fontsize=10)
        plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    dist_path = RESULTS_ROOT / "beta_vs_binary_irt_bench_distributions.png"
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()
        for bench_idx, bench_name in enumerate(unique_benches):
            ax = axes[bench_idx]
            bench_seed_vals = stacked_vals[:, bench_idx, 0]
            bench_seed_vals = bench_seed_vals[np.isfinite(bench_seed_vals)]
            bench_mean = float(np.mean(bench_seed_vals))
            bench_std = float(np.std(bench_seed_vals))
            mean_label = f"{bench_mean:.1e}".replace("e-0", "e-").replace("e+0", "e+")
            std_label = f"{bench_std:.1e}".replace("e-0", "e-").replace("e+0", "e+")
            ax.hist(bench_seed_vals, bins=20, density=True, color="steelblue", alpha=0.75)
            mean_kwargs = {"color": "darkblue", "linestyle": "--", "linewidth": 1.8}
            zero_kwargs = {"color": "red", "linestyle": "-", "linewidth": 1.5}
            if bench_idx == len(unique_benches) - 1:
                ax.axvline(bench_mean, label="Mean", **mean_kwargs)
                ax.axvline(0.0, label="Zero", **zero_kwargs)
                ax.legend(fontsize=11, loc="upper right")
            else:
                ax.axvline(bench_mean, **mean_kwargs)
                ax.axvline(0.0, **zero_kwargs)
            ax.set_title(f"{bench_name}\nmean={mean_label}, std={std_label}", fontsize=13)
            if bench_idx >= 2:
                ax.set_xlabel("Binary-IRT MAE - Beta-IRT MAE", fontsize=12)
            else:
                ax.set_xlabel("")
            if bench_idx % 2 == 0:
                ax.set_ylabel("Density", fontsize=12)
            else:
                ax.set_ylabel("")
            ax.tick_params(axis="both", labelsize=11)
        for ax in axes[len(unique_benches):]:
            ax.axis("off")
        fig.tight_layout()
        plt.savefig(dist_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    finite_vals = mean_vals[np.isfinite(mean_vals)]
    beta_wins = int(np.sum(finite_vals > 0))
    binary_wins = int(np.sum(finite_vals < 0))
    ties = int(np.sum(finite_vals == 0))
