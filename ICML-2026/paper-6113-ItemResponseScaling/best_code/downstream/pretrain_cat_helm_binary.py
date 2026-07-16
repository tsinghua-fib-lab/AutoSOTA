import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
torch.manual_seed(0)
torch.set_num_threads(1)
from tqdm import tqdm
from collections import defaultdict
from joblib import Parallel, delayed
import sys
sys.path.append("..")
from utils import visualize_response_matrix, cat_binary_1pl
from tueplots import bundles
bundles.icml2024()
from huggingface_hub import snapshot_download
from torch.distributions import Bernoulli
import logging

REPO_IDS = [
        "EleutherAI/pythia-12b",
        # "EleutherAI/pythia-6.9b",
        # "EleutherAI/pythia-2.8b",
        # "EleutherAI/pythia-1.4b",
        # "EleutherAI/pythia-1b",
        # "EleutherAI/pythia-410m",
        # "EleutherAI/pythia-160m",
        # "EleutherAI/pythia-70m",
        # "EleutherAI/pythia-14m",
        "LLM360/Amber",
        # "HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints",
        # "HuggingFaceTB/SmolLM2-360M-intermediate-checkpoints",
        # "HuggingFaceTB/SmolLM2-135M-intermediate-checkpoints",
    ]
MODELS = [repo_id.split("/")[1] for repo_id in REPO_IDS]

SCENARIOS = ['babi_qa', 'raft']

STEP2FLOP = {
    "pythia-12b": 300 * 1e9 / 143000 * 12 * 1e9,
    "Amber": 1.26 * 1e12 / 358 * 6.7 * 1e9,
    "SmolLM2-1.7B": 250 * 1e9 / 125000 * 1.7 * 1e9,
    "SmolLM2-360M": 250 * 1e9 / 160000 * 360 * 1e6,
    "SmolLM2-135M": 250 * 1e9 / 240000 * 135 * 1e6,
}

# SCENARIOS = ['babi_qa', 'civil_comments', 'commonsense',
#     'dyck_language_np=3', 'entity_data_imputation', 'entity_matching',
#     'gsm', 'legal_support', 'legalbench', 'mmlu', 'raft',
#     'synthetic_reasoning', 'wikifact'] # 'med_qa', 'boolq', 'imdb'

if __name__ == "__main__":
    # Auto-detect device (prefer CUDA if available, otherwise CPU)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    budget = 100
    max_workers = 256
    results_dict = defaultdict(lambda: defaultdict(dict))
    
    os.makedirs("../result/pretrain_binarycat", exist_ok=True)
    logging.basicConfig(
        filename="../result/pretrain_binarycat/run.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    cache_dir = snapshot_download(repo_id="stair-lab/irsl_downstream_resmat", repo_type="dataset")
    for model in tqdm(MODELS):
        with open(f"{cache_dir}/results_{model}.pkl", "rb") as f:
            resmat_full = pickle.load(f)
        
        for scenario in tqdm(SCENARIOS):
            output_dir = f"../result/pretrain_binarycat/{scenario}_{model}"
            os.makedirs(output_dir, exist_ok=True)
            resmat = resmat_full.loc[:, ~resmat_full.columns.get_level_values("z").isna()]
            resmat = resmat.loc[:, resmat.columns.get_level_values("scenario") == scenario]
            resmat = resmat[~resmat.isna().all(axis=1)]
            visualize_response_matrix(resmat, resmat, f"{output_dir}/response_matrix.png")
            
            steps = np.array([float(name.split("-")[-1]) for name in resmat.index])
            ys = torch.tensor(resmat.values, dtype=torch.float, device=device)
            n_test_takers, n_items = ys.shape
            nan_pct = torch.isnan(ys).float().mean().item() * 100
            logging.info(f"model={model} scenario={scenario} shape={ys.shape} nan_pct={nan_pct:.2f}%")
            zs = torch.tensor(resmat.columns.get_level_values("z").astype(float), dtype=torch.float, device=device)
            
            # z distribution
            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                plt.figsize=(6, 6)
                plt.hist(zs.cpu().numpy(), bins=30)
                plt.xlabel("z values", fontsize=10)
                plt.ylabel("Frequency", fontsize=10)
                plt.tick_params(axis="both", labelsize=10)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/zs_distribution.png", dpi=300, bbox_inches="tight")
                plt.close()

            # irt theta on subset questions
            def _run_one(i):
                return cat_binary_1pl(ys[i], zs, device, budget)
            thetass = Parallel(n_jobs=max_workers)(delayed(_run_one)(i) for i in tqdm(range(ys.shape[0])))
            thetass = torch.tensor(thetass, dtype=float).squeeze(-1)

            # theta convergence
            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                fig, axes = plt.subplots(nrows=ys.shape[0], ncols=1, figsize=(6, 2*ys.shape[0]), sharex=True)
                for i, ax in enumerate(axes):
                    ax.plot(np.arange(thetass.shape[-1]), thetass[i].cpu().numpy(), label=int(steps[i]))
                    ax.set_ylabel("Theta", fontsize=16)
                    ax.legend(fontsize=16)
                    ax.tick_params(axis="both", labelsize=16)
                axes[-1].set_xlabel("Budget", fontsize=16)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/theta_convergence.png", dpi=100, bbox_inches="tight")
                plt.close()
            
            # law curve
            final_thetas = thetass[:, -1].cpu().numpy()
            # step_pcts = steps.astype(int) / steps.astype(int).max() * 100.0
            flops = steps * STEP2FLOP[model]
            means_all = torch.nanmean(ys, dim=1).cpu().numpy() # mean score on all questions
            means_sub = torch.nanmean(ys[:, torch.randperm(ys.shape[1])[:budget]], dim=1).cpu().numpy() # mean score on random subset
            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                fig, ax1 = plt.subplots(figsize=(6, 4))
                # ax1.plot(step_pcts, means_sub, color="tab:blue", linewidth=1.5, label="random subset mean")
                # ax1.plot(step_pcts, means_all, color='tab:blue', linewidth=1.5, label="full set mean", linestyle="--")
                ax1.plot(flops, means_sub, color="skyblue", linewidth=1.5, label="random subset mean", alpha=0.4)
                ax1.plot(flops, means_all, color='tab:blue', linewidth=1.5, label="full set mean", alpha=0.4)
                ax1.set_xlabel("FLOP", fontsize=16)
                ax1.set_ylabel("Acc", color='tab:blue', fontsize=16)
                ax1.set_title(f"{model}, {scenario}", fontsize=16)
                ax1.tick_params(axis="x", labelsize=16)
                ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=16)
                ax2 = ax1.twinx()
                # ax2.plot(step_pcts, final_thetas, color='tab:red', linewidth=1.5, label=r"CAT $\theta$")
                ax2.plot(flops, final_thetas, color='tab:red', linewidth=1.5, label=r"CAT $\theta$", alpha=0.4)
                ax2.set_ylabel(r"CAT $\theta$", color='tab:red', fontsize=16)
                ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=16)
                # ax2.set_ylim(bottom = -7.5)
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=16)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/law_curve_{model}_{scenario}.png", dpi=300, bbox_inches="tight")
                plt.close()
            
            results_dict[scenario][model] = {
                "steps": steps,
                "ys": ys,
                "thetass": thetass,
            }
        
    final_results_dict = {k: dict(v) for k, v in results_dict.items()}
    with open(f"../result/pretrain_binarycat/result.pkl", "wb") as f:
        pickle.dump(final_results_dict, f)