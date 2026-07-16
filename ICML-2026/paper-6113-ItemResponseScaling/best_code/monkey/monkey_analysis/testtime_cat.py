import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
torch.manual_seed(0)
torch.set_num_threads(1)
from collections import defaultdict
from matplotlib import gridspec
from scipy.stats import spearmanr
from tqdm import tqdm
from joblib import Parallel, delayed
import sys
sys.path.append("../..")
from utils import cat_beta_1pl, compute_pass_datk_gts, compute_pass_datk_irt, cat_binary_1pl
from tueplots import bundles
bundles.icml2024()
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    device = "cpu"
    # FILE_NAME = "irsl_testtime_resmat1"
    FILE_NAME = "irsl_testtime_resmat2"
    sample_budget = 50
    item_budget = 50 if FILE_NAME == "irsl_testtime_resmat1" else 30
    
    # Note: weights_only=False is required for PyTorch 2.6+
    testtime_resmat = torch.load(f"{FILE_NAME}_withz.pt", map_location="cpu", weights_only=False)
    data_tensor = testtime_resmat["data_tensor"]
    model_names = testtime_resmat["models"]
    datasets  = testtime_resmat["datasets"]
    zs = testtime_resmat["zs"]
    print(testtime_resmat["test_models"])
    if FILE_NAME == "irsl_testtime_resmat1":
        helm_zs = testtime_resmat["helm_zs"]
    print(data_tensor.shape)
    
    results_dict = defaultdict(lambda: defaultdict(dict))
    for scen in tqdm(sorted(set(datasets))):
        output_dir = f"../../result/{FILE_NAME}/{scen}"
        os.makedirs(output_dir, exist_ok=True)

        idxs = [j for j, s in enumerate(datasets) if s == scen]
        scen_tensor = data_tensor[:, idxs, :]
        print(f"{scen}: shape = {scen_tensor.shape}")
        n_test_takers, n_items, n_samples = scen_tensor.shape
        scen_probmat = torch.nanmean(
            torch.tensor(scen_tensor[:, :, :sample_budget], dtype=torch.float, device=device),
            dim = -1,
        )
        scen_binarymat = torch.tensor(scen_tensor[:, :, 0], dtype=torch.float, device=device)
        scen_zs = torch.tensor(zs[idxs], dtype=torch.float, device=device)
        
        # z distribution plot
        with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
            plt.figsize=(6, 6)
            plt.hist(scen_zs.numpy(), bins=30)
            plt.xlabel("z values", fontsize=10)
            plt.ylabel("Frequency", fontsize=10)
            plt.tick_params(axis="both", labelsize=10)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/zs_distribution.png", dpi=300, bbox_inches="tight")
            plt.close()

        # zs VS helm_zs corr plot
        if FILE_NAME == "irsl_testtime_resmat1":
            scen_helm_zs = helm_zs[idxs]
            spearman_corr, _ = spearmanr(scen_zs, scen_helm_zs)
            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                plt.figure(figsize=(6, 6))
                plt.scatter(scen_zs.numpy(), scen_helm_zs, s=10)
                plt.xlabel(r"Our $z$", fontsize=16)
                plt.ylabel(r"$z$ from REEval", fontsize=16)
                plt.title(rf"{scen}: $\rho$={spearman_corr:.2f}", fontsize=20)
                plt.tick_params(axis="both", labelsize=14)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/zs_vs_helmzs.png", dpi=300, bbox_inches="tight")
                plt.close()
                
        # binary-irt cat on binarymat
        def _run_one_binary(i):
            return cat_binary_1pl(scen_binarymat[i], scen_zs, device, budget=item_budget)
        thetass_binary = Parallel(n_jobs=-1)(delayed(_run_one_binary)(i) for i in tqdm(range(scen_binarymat.shape[0])))
        thetass_binary = torch.tensor(thetass_binary, dtype=torch.float) # (n_models, budget)
        final_thetas_binary = thetass_binary[:, -1]
        # theta convergence
        with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
            fig, axes = plt.subplots(nrows=n_test_takers, ncols=1, figsize=(6, 2*n_test_takers), sharex=True)
            for i, ax in enumerate(axes):
                ax.plot(np.arange(thetass_binary.shape[-1]), thetass_binary[i].cpu().numpy(), label=model_names[i])
                ax.set_ylabel(r"$\theta$", fontsize=16)
                ax.legend(fontsize=16)
                ax.tick_params(axis="both", labelsize=16)
            axes[-1].set_xlabel("Budget", fontsize=16)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/binaryirt_theta_convergence.png", dpi=100, bbox_inches="tight")
            plt.close()

        # beta-irt cat on probmat
        # cat_beta_1pl(scen_probmat[-1], scen_zs, device)
        def _run_one_beta(i):
            return cat_beta_1pl(scen_probmat[i], scen_zs, device, budget=item_budget)
        thetass_beta = Parallel(n_jobs=-1)(delayed(_run_one_beta)(i) for i in tqdm(range(scen_probmat.shape[0])))
        thetass_beta = torch.tensor(thetass_beta, dtype=torch.float) # (n_models, budget)
        final_thetas_beta = thetass_beta[:, -1]
        # theta convergence
        with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
            fig, axes = plt.subplots(nrows=n_test_takers, ncols=1, figsize=(6, 2*n_test_takers), sharex=True)
            for i, ax in enumerate(axes):
                ax.plot(np.arange(thetass_beta.shape[-1]), thetass_beta[i].cpu().numpy(), label=model_names[i])
                ax.set_ylabel(r"$\theta$", fontsize=16)
                ax.legend(fontsize=16)
                ax.tick_params(axis="both", labelsize=16)
            axes[-1].set_xlabel("Budget", fontsize=16)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/betairt_theta_convergence.png", dpi=100, bbox_inches="tight")
            plt.close()
        
        for i, model in enumerate(model_names):
            model_tensor = scen_tensor[i]
            
            # plot irt corr passat1s
            irt_probs_beta = torch.sigmoid(float(final_thetas_beta[i]) + scen_zs).cpu().numpy()
            irt_probs_binary = torch.sigmoid(float(final_thetas_binary[i]) + scen_zs).cpu().numpy()
            passat1s_fullset = np.nanmean(model_tensor, axis=-1)
            rho, _ = spearmanr(irt_probs_beta, passat1s_fullset, nan_policy="omit")
            results_dict[scen][model]["irtprob_corr_passat1"] = rho
            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                fig = plt.figure(figsize=(6, 6))
                gs = gridspec.GridSpec(5, 5, figure=fig, wspace=0.05, hspace=0.05)
                ax_scatter = fig.add_subplot(gs[0:4, 1:5])
                ax_left    = fig.add_subplot(gs[0:4, 0], sharey=ax_scatter)
                ax_bottom  = fig.add_subplot(gs[4,   1:5], sharex=ax_scatter)
                # ---- scatter ----
                ax_scatter.scatter(irt_probs_beta, passat1s_fullset, s=10)
                ax_scatter.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="black")
                ax_scatter.set_xlim(0, 1)
                ax_scatter.set_ylim(0, 1)
                ax_scatter.set_xlabel("IRT Probability", fontsize=18)
                ax_scatter.set_ylabel("pass@1", fontsize=18)
                ax_scatter.tick_params(axis="both", labelsize=14)
                # ---- left marginal (y) ----
                ax_left.hist(passat1s_fullset, bins=30, orientation="horizontal")
                ax_left.set_ylim(0, 1)
                ax_left.invert_xaxis()
                ax_left.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
                ax_left.set_xticks([])
                for s in ("top", "right", "bottom", "left"):
                    ax_left.spines[s].set_visible(False)
                # ---- bottom marginal (x) ----
                ax_bottom.hist(irt_probs_beta, bins=30)
                ax_bottom.set_xlim(0, 1)
                ax_bottom.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
                ax_bottom.set_yticks([])
                for s in ("top", "right", "bottom", "left"):
                    ax_bottom.spines[s].set_visible(False)
                # ---- final ----
                fig.suptitle(rf"{model}, {scen} ($\rho$ = {rho:.2f})", fontsize=16)
                plt.subplots_adjust(wspace=0.05, hspace=0.05)
                fig.savefig(f"{output_dir}/irt_corr_passat1_{model}.png", dpi=300, bbox_inches="tight")
                plt.close(fig)
            
            # law curve before filter
            passat1s_subset_subsample = np.nanmean(model_tensor[:item_budget, :sample_budget], axis=-1)
            pass_datk_gts = compute_pass_datk_gts(model_tensor)
            # pass_datk_gts = compute_pass_datk_irt(passat1s, n_samples)
            pass_datk_subset_subsample_passat1 = compute_pass_datk_irt(passat1s_subset_subsample, n_samples)
            pass_datk_irts_beta = compute_pass_datk_irt(irt_probs_beta, n_samples)
            pass_datk_irts_binary = compute_pass_datk_irt(irt_probs_binary, n_samples)
            
            mae_irt_beta = np.mean(np.abs(pass_datk_gts - pass_datk_irts_beta))
            mae_irt_binary = np.mean(np.abs(pass_datk_gts - pass_datk_irts_binary))
            mae_sub_passat1 = np.mean(np.abs(pass_datk_gts - pass_datk_subset_subsample_passat1))
            results_dict[scen][model]["mae_irt_beta_before_filter"] = mae_irt_beta
            results_dict[scen][model]["mae_irt_binary_before_filter"] = mae_irt_binary
            results_dict[scen][model]["mae_sub_passat1_before_filter"] = mae_sub_passat1
            
            sample_arange = np.arange(1, n_samples + 1)
            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                ax_left, ax_right = axes
                # Left: Pass@k vs k
                ax_left.plot(sample_arange, pass_datk_gts, label="full unbiased (GT)", linewidth=2, color="blue")
                ax_left.plot(sample_arange, pass_datk_subset_subsample_passat1, label="sub pass@1", linestyle="--", linewidth=2, color="blue")
                ax_left.plot(sample_arange, pass_datk_irts_beta, label="sub beta-IRT", linewidth=2, color="red")
                # ax_left.plot(sample_arange, pass_datk_irts_binary, label="sub binary-IRT", linewidth=2, linestyle="--", color="red")
                ax_left.set_xlabel("Number of Samples", fontsize=16)
                ax_left.set_ylabel("Pass@k", fontsize=16)
                ax_left.set_ylim(0, 1)
                ax_left.legend(fontsize=14)
                ax_left.tick_params(axis="both", labelsize=14)
                # Right: log–log of -log(Pass@k) vs k
                ax_right.loglog(sample_arange, -np.log(pass_datk_gts), label="full unbiased (GT)", linewidth=2, color="blue")
                ax_right.loglog(sample_arange, -np.log(pass_datk_subset_subsample_passat1), label="sub pass@1", linewidth=2, linestyle="--", color="blue")
                ax_right.loglog(sample_arange, -np.log(pass_datk_irts_beta), label="sub beta-IRT", linewidth=2, color="red")
                # ax_right.loglog(sample_arange, -np.log(pass_datk_irts_binary), label="sub binary-IRT", linewidth=2, linestyle="--", color="red")
                ax_right.set_xlabel("Number of Samples", fontsize=16)
                ax_right.set_ylabel(r"$-\log(\mathrm{Pass@k})$", fontsize=16)
                ax_right.legend(fontsize=14)
                ax_right.tick_params(axis="both", labelsize=14)
                # final
                fig.suptitle(
                    # rf"{model}, {scen} (MAE: beta-IRT={mae_irt_beta:.2e}, binary-IRT={mae_irt_binary:.2e}, pass@1={mae_sub_passat1:.2e})",
                    rf"{model}, {scen} (MAE: beta-IRT={mae_irt_beta:.2e}, pass@1={mae_sub_passat1:.2e})",
                    fontsize=16
                )
                fig.tight_layout()
                fig.savefig(f"{output_dir}/law_curve_before_filter_{model}.png", dpi=300, bbox_inches="tight")
                plt.close(fig)
                
            # law curve after filter
            mask = (passat1s_fullset >= 0.01)
            model_tensor = model_tensor[mask]
            passat1s_subset_subsample = np.nanmean(model_tensor[:item_budget, :sample_budget], axis=-1)
            irt_probs_beta = irt_probs_beta[mask]
            irt_probs_binary = irt_probs_binary[mask]

            pass_datk_gts = compute_pass_datk_gts(model_tensor)
            # pass_datk_gts = compute_pass_datk_irt(passat1s, n_samples)
            pass_datk_subset_subsample_passat1 = compute_pass_datk_irt(passat1s_subset_subsample, n_samples)
            pass_datk_irts_beta = compute_pass_datk_irt(irt_probs_beta, n_samples)
            pass_datk_irts_binary = compute_pass_datk_irt(irt_probs_binary, n_samples)
            
            mae_irt_beta = np.mean(np.abs(pass_datk_gts - pass_datk_irts_beta))
            mae_irt_binary = np.mean(np.abs(pass_datk_gts - pass_datk_irts_binary))
            mae_sub_passat1 = np.mean(np.abs(pass_datk_gts - pass_datk_subset_subsample_passat1))
            results_dict[scen][model]["mae_irt_beta_after_filter"] = mae_irt_beta
            results_dict[scen][model]["mae_irt_binary_after_filter"] = mae_irt_binary
            results_dict[scen][model]["mae_sub_passat1_after_filter"] = mae_sub_passat1
            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                ax_left, ax_right = axes
                # Left: Pass@k vs k
                ax_left.plot(sample_arange, pass_datk_gts, label="full unbiased (GT)", linewidth=2, color="blue")
                ax_left.plot(sample_arange, pass_datk_subset_subsample_passat1, label="sub pass@1", linestyle="--", linewidth=2, color="blue")
                ax_left.plot(sample_arange, pass_datk_irts_beta, label="sub beta-IRT", linewidth=2, color="red")
                # ax_left.plot(sample_arange, pass_datk_irts_binary, label="sub binary-IRT", linewidth=2, linestyle="--", color="red")
                ax_left.set_xlabel("Number of Samples", fontsize=16)
                ax_left.set_ylabel("Pass@k", fontsize=16)
                ax_left.set_ylim(0, 1)
                ax_left.legend(fontsize=14)
                ax_left.tick_params(axis="both", labelsize=14)
                # Right: log–log of -log(Pass@k) vs k
                ax_right.loglog(sample_arange, -np.log(pass_datk_gts), label="full unbiased (GT)", linewidth=2, color="blue")
                ax_right.loglog(sample_arange, -np.log(pass_datk_subset_subsample_passat1), label="sub pass@1", linewidth=2, linestyle="--", color="blue")
                ax_right.loglog(sample_arange, -np.log(pass_datk_irts_beta), label="sub beta-IRT", linewidth=2, color="red")
                # ax_right.loglog(sample_arange, -np.log(pass_datk_irts_binary), label="sub binary-IRT", linewidth=2, linestyle="--", color="red")
                ax_right.set_xlabel("Number of Samples", fontsize=16)
                ax_right.set_ylabel(r"$-\log(\mathrm{Pass@k})$", fontsize=16)
                ax_right.legend(fontsize=14)
                ax_right.tick_params(axis="both", labelsize=14)
                # final
                fig.suptitle(
                    # rf"{model}, {scen} (MAE: beta-IRT={mae_irt_beta:.2e}, binary-IRT={mae_irt_binary:.2e}, pass@1={mae_sub_passat1:.2e})",
                    rf"{model}, {scen} (MAE: beta-IRT={mae_irt_beta:.2e}, pass@1={mae_sub_passat1:.2e})",
                    fontsize=16
                )
                fig.tight_layout()
                fig.savefig(f"{output_dir}/law_curve_after_filter_{model}.png", dpi=300, bbox_inches="tight")
                plt.close(fig)
                
    final_results_dict = {k: dict(v) for k, v in results_dict.items()}
    with open(f"{FILE_NAME}_result.pkl", "wb") as f:
        pickle.dump(final_results_dict, f)