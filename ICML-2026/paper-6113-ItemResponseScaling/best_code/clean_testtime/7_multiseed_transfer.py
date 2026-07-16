import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.special import expit
from tqdm import tqdm
from tueplots import bundles
import sys

torch.manual_seed(0)
torch.set_num_threads(1)
bundles.icml2024()
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent))
from utils import calibrate_1pl_z, calibrate_1pl_theta, compute_pass_datk_gts, compute_pass_datk_irt

DEVICE = "cuda:7"
N_SEEDS = 100
N_MODELS_FOR_TEST = 4
PROB_THRESHOLD = 0.005
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "7_multiseed_transfer"

parser = argparse.ArgumentParser()
parser.add_argument("--input-hf-repo", type=str, default="irsl_testtime_resmat2", choices=["irsl_testtime_resmat1", "irsl_testtime_resmat2"])
args = parser.parse_args()

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

input_path = DATA_DIR / f"1_calibreated_{args.input_hf_repo}.pt"
payload = torch.load(input_path, map_location="cpu", weights_only=False)

response_tensor = np.array(payload["data_tensor"], dtype=np.float32)  # (n_models, n_items, n_samples)
model_names = list(payload["models"])
datasets = list(payload["datasets"])
n_models = len(model_names)
n_samples = response_tensor.shape[-1]
unique_benches = sorted(set(datasets))
datasets_arr = np.array(datasets)
bench_item_indices = {bench: np.where(datasets_arr == bench)[0] for bench in unique_benches}

cache_path = DATA_DIR / f"7_cache_{args.input_hf_repo}.pkl"

if cache_path.exists():
    print(f"Loading cached results from {cache_path}")
    with open(cache_path, "rb") as f:
        cached = pickle.load(f)
    within_maes = cached["within_maes"]
    cross_maes = cached["cross_maes"]
else:
    within_maes = []
    cross_maes = []

    for seed in tqdm(range(N_SEEDS), desc="seeds"):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_models)
        n_train = n_models - N_MODELS_FOR_TEST
        train_model_indices = perm[:n_train]
        test_model_indices = perm[n_train:]

        train_probmat = np.nanmean(response_tensor[train_model_indices], axis=-1)  # (n_train, n_items)
        test_response_tensor = response_tensor[test_model_indices]                 # (N_TEST, n_items, n_samples)

        item_difficulties = calibrate_1pl_z(
            torch.tensor(train_probmat, dtype=torch.float32, device=DEVICE),
            device=DEVICE,
        )

        # Within-benchmark transfer
        for bench in unique_benches:
            bench_items = bench_item_indices[bench]
            bench_response_tensor = test_response_tensor[:, bench_items, :]  # (N_TEST, n_bench_items, n_samples)
            bench_item_difficulties = item_difficulties[bench_items]

            item_avg_passat1 = np.nanmean(np.nanmean(bench_response_tensor, axis=-1), axis=0)
            difficulty_order = np.argsort(item_avg_passat1)  # ascending: hardest first
            half = len(difficulty_order) // 2
            easy_item_positions = difficulty_order[half:]
            hard_item_positions = difficulty_order[:half]

            easy_probmat = np.nanmean(bench_response_tensor[:, easy_item_positions, :], axis=-1)
            hard_response_tensor = bench_response_tensor[:, hard_item_positions, :]
            hard_item_difficulties = bench_item_difficulties[hard_item_positions]

            test_thetas = calibrate_1pl_theta(
                torch.tensor(easy_probmat, dtype=torch.float32),
                DEVICE,
                torch.tensor(bench_item_difficulties[easy_item_positions], dtype=torch.float32),
            )

            for model_idx in range(N_MODELS_FOR_TEST):
                theta = float(test_thetas[model_idx])
                hard_responses = hard_response_tensor[model_idx]
                above_threshold = np.nanmean(hard_responses, axis=-1) >= PROB_THRESHOLD
                if above_threshold.sum() == 0:
                    continue
                irt_probs = expit(theta + hard_item_difficulties[above_threshold])
                pass_datk_gt = compute_pass_datk_gts(hard_responses[above_threshold])
                pass_datk_est = compute_pass_datk_irt(irt_probs, n_samples)
                within_maes.append(float(np.mean(np.abs(pass_datk_gt - pass_datk_est))))

        # Cross-benchmark transfer
        aime2024_items = bench_item_indices["aime2024"]
        aime2025_items = bench_item_indices["aime2025"]
        aime2024_probmat = np.nanmean(test_response_tensor[:, aime2024_items, :], axis=-1)

        test_thetas = calibrate_1pl_theta(
            torch.tensor(aime2024_probmat, dtype=torch.float32),
            DEVICE,
            torch.tensor(item_difficulties[aime2024_items], dtype=torch.float32),
        )

        for model_idx in range(N_MODELS_FOR_TEST):
            theta = float(test_thetas[model_idx])
            aime2025_responses = test_response_tensor[model_idx, aime2025_items, :]
            above_threshold = np.nanmean(aime2025_responses, axis=-1) >= PROB_THRESHOLD
            if above_threshold.sum() == 0:
                continue
            irt_probs = expit(theta + item_difficulties[aime2025_items][above_threshold])
            pass_datk_gt = compute_pass_datk_gts(aime2025_responses[above_threshold])
            pass_datk_est = compute_pass_datk_irt(irt_probs, n_samples)
            cross_maes.append(float(np.mean(np.abs(pass_datk_gt - pass_datk_est))))

    within_maes = np.array(within_maes, dtype=np.float32)
    cross_maes = np.array(cross_maes, dtype=np.float32)

    with open(cache_path, "wb") as f:
        pickle.dump({"within_maes": within_maes, "cross_maes": cross_maes}, f)

total = len(within_maes) + len(cross_maes)
within_weights = np.ones_like(within_maes) / total
cross_weights = np.ones_like(cross_maes) / total

out_path = RESULTS_DIR / f"{args.input_hf_repo}_transfer.png"
with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(within_maes, bins=30, density=False, weights=within_weights, alpha=0.6,
            color="lightskyblue", label="Within-Benchmark Transfer")
    ax.hist(cross_maes, bins=30, density=False, weights=cross_weights, alpha=0.6,
            color="lightcoral", label="Cross-Benchmark Transfer")
    ax.set_title("MAE Density: Abs(Hard GT - Hard Est)", fontsize=16)
    ax.set_xlabel("MAE", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
