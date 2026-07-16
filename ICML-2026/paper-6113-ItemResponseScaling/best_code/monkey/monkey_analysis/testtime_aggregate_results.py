import os
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tueplots import bundles
bundles.icml2024()

if __name__ == "__main__":
    FILE_NAME = "irsl_testtime_resmat1"
    # FILE_NAME = "irsl_testtime_resmat2"

    # Note: weights_only=False is required for PyTorch 2.6+
    testtime_resmat = torch.load(f"{FILE_NAME}_withz.pt", map_location="cpu", weights_only=False)
    model_names = list(testtime_resmat["models"])          # all models
    test_model_names = list(testtime_resmat["test_models"])# test models
    train_model_names = [m for m in model_names if m not in test_model_names]
    datasets = sorted(set(testtime_resmat["datasets"]))

    with open(f"{FILE_NAME}_result.pkl", "rb") as f:
        results_dict = pickle.load(f)

    # average rho
    train_corrs = []
    test_corrs  = []
    for scen in datasets:
        scen_dict = results_dict[scen]
        for model in model_names:
            rho = results_dict[scen][model]["irtprob_corr_passat1"]
            if model in train_model_names:
                train_corrs.append(rho)
            else:
                test_corrs.append(rho)
    train_avg = np.mean(train_corrs)
    test_avg  = np.mean(test_corrs)
    print(f"[irtprob_corr_passat1] Train avg: {train_avg:.4f}")
    print(f"[irtprob_corr_passat1] Test  avg: {test_avg:.4f}")
    
    # heatmap
    diffs = {}
    for scen in datasets:
        row = {}
        scen_dict = results_dict[scen]
        for model in model_names:
            mae_beta  = scen_dict[model]["mae_irt_beta_after_filter"]
            mae_pass1 = scen_dict[model]["mae_sub_passat1_after_filter"]
            row[model] = mae_pass1 - mae_beta
        diffs[scen] = row
    columns_order = train_model_names + test_model_names
    diff_df = pd.DataFrame.from_dict(diffs, orient="index")
    diff_df = diff_df.reindex(index=datasets, columns=columns_order)
    vals = diff_df.values
    abs_max = np.max(np.abs(vals))
    vmin, vmax = -abs_max, abs_max
    fig_w = max(10, 0.8*len(columns_order) + 4)
    fig_h = max(8, 0.6*len(datasets) + 4)
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(vals, aspect="auto", cmap="bwr", vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(columns_order)))
        ax.set_xticklabels(columns_order, rotation=45, ha="right", fontsize=10)
        ax.set_yticks(np.arange(len(datasets)))
        ax.set_yticklabels(datasets, fontsize=10)
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                ax.text(j, i, f"{vals[i, j]:.2e}", ha="center", va="center", fontsize=10)
        split_idx = len(train_model_names)
        ax.axvline(x=split_idx - 0.5, color="black", linewidth=1.5)
        ax.text((split_idx - 1) / 2, -1, "Train", ha="center", va="center", fontsize=14, fontweight="bold", transform=ax.transData)
        ax.text(split_idx + (len(columns_order) - split_idx - 1) / 2, -1, "Test", ha="center", va="center", fontsize=14, fontweight="bold", transform=ax.transData)
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("MAE(pass@1) − MAE(beta-IRT)", fontsize=12)
        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Dataset", fontsize=12)
        ax.set_title("MAE Difference After Filter (pass@1 − beta-IRT) — Train/Test Split", fontsize=14)
        # plt.tight_layout()
        png_path = f"{FILE_NAME}_mae_diff_heatmap.png"
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
