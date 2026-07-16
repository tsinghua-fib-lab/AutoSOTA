import argparse
import os
from pathlib import Path
import numpy as np
import torch
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from tqdm import tqdm
from tueplots import bundles
import sys
torch.manual_seed(0)
torch.set_num_threads(1)
bundles.icml2024()

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent))
from utils import cat_beta_1pl, cat_beta_2pl

parser = argparse.ArgumentParser()
parser.add_argument("--input-hf-repo", type=str, default="irsl_testtime_resmat2", choices=["irsl_testtime_resmat1", "irsl_testtime_resmat2"])
parser.add_argument("--irt-model", type=str, default="1pl", choices=["1pl", "2pl"])
parser.add_argument("--data-root", type=Path, default=BASE_DIR / "data")
parser.add_argument("--results-root", type=Path, default=BASE_DIR / "results")
args = parser.parse_args()

DEVICE = "cpu"
SAMPLE_BUDGET = 50
ITEM_BUDGET = 50 if args.input_hf_repo == "irsl_testtime_resmat1" else 30
data_dir = args.data_root
results_dir = args.results_root / "2_cat"
data_dir.mkdir(parents=True, exist_ok=True)

irt_suffix = "_2pl" if args.irt_model == "2pl" else ""
input_path = data_dir / f"1_calibreated_{args.input_hf_repo}{irt_suffix}.pt"
payload = torch.load(input_path, map_location="cpu", weights_only=False)

data_tensor = np.array(payload["data_tensor"], dtype=np.float32)
model_names = list(payload["models"])
test_models = list(payload["test_models"])
test_model_indices = [i for i, m in enumerate(model_names) if m in set(test_models)]
datasets = list(payload["datasets"])
zs = np.array(payload["zs"], dtype=np.float32)
alphas = np.array(payload["alphas"], dtype=np.float32) if args.irt_model == "2pl" else None

output_dict = {
    "sample_budget": SAMPLE_BUDGET,
    "item_budget": ITEM_BUDGET,
    "models": model_names,
    "test_models": test_models,
    "datasets": datasets,
    "zs": zs,
    "alphas": alphas,
    "data_tensor": data_tensor,
    "test_thetas": {},
}

unique_datasets = sorted(set(datasets))
for dataset in tqdm(unique_datasets, desc="datasets"):
    item_indices = [i for i, d in enumerate(datasets) if d == dataset]
    bench_tensor = data_tensor[test_model_indices][:, item_indices, :]
    bench_probmat = torch.nanmean(
        torch.tensor(bench_tensor[:, :, :SAMPLE_BUDGET], dtype=torch.float, device=DEVICE),
        dim=-1,
    )
    bench_zs = torch.tensor(zs[item_indices], dtype=torch.float, device=DEVICE)    
    bench_alphas = torch.tensor(
        alphas[item_indices], dtype=torch.float, device=DEVICE
    ) if alphas is not None else None

    if args.irt_model == "2pl":
        def _run_one(i):
            return cat_beta_2pl(bench_probmat[i], bench_alphas, bench_zs, DEVICE, budget=ITEM_BUDGET)
    else:
        def _run_one(i):
            return cat_beta_1pl(bench_probmat[i], bench_zs, DEVICE, budget=ITEM_BUDGET)

    n_jobs = int((os.cpu_count()) * 0.8)
    thetass = Parallel(n_jobs=n_jobs)(
        delayed(_run_one)(i) for i in tqdm(range(bench_probmat.shape[0]), desc=f"{dataset} cat", leave=False)
    )
    thetass = np.array(thetass, dtype=np.float32)

    output_dir = results_dir / f"{args.input_hf_repo}{irt_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    n_test_takers = thetass.shape[0]
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig, axes = plt.subplots(nrows=n_test_takers, ncols=1, figsize=(6, 2 * n_test_takers), sharex=True)
        for i, ax in enumerate(axes):
            ax.plot(np.arange(thetass.shape[-1]), thetass[i], label=test_models[i])
            ax.set_ylabel(r"$\theta$", fontsize=16)
            ax.legend(fontsize=14)
            ax.tick_params(axis="both", labelsize=14)
        axes[-1].set_xlabel("Budget", fontsize=16)
        plt.tight_layout()
        plt.savefig(output_dir / f"theta_convergence_{dataset}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    output_dict["test_thetas"][dataset] = list(zip(test_models, thetass[:, -1]))

output_path = data_dir / f"2_cated_{args.input_hf_repo}{irt_suffix}.pt"
torch.save(output_dict, output_path)
