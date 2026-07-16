import os
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error                
from scipy.stats import spearmanr
from tueplots import bundles
bundles.icml2024()

def backtest_krr(x, y, A):
    x = x.reshape(-1, 1)
    n = len(y)
    split = int(A * n)
    x_tr, y_tr_gt = x[:split], y[:split]
    x_te, y_te_gt = x[split:], y[split:]
    kr = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1)
    kr.fit(x_tr, y_tr_gt)
    y_tr_pred = kr.predict(x_tr)
    y_te_pred = kr.predict(x_te)
    train_mae = mean_absolute_error(y_tr_pred, y_tr_gt) / (y.max() - y.min())
    test_mae = mean_absolute_error(y_te_pred, y_te_gt) / (y.max() - y.min())
    return (train_mae, test_mae)

def total_variance(a):
    n = len(a)
    num = np.abs(np.diff(a)).sum()
    denom = abs(a[-1] - a[0])
    return (n / (n - 1.0)) * (num / denom)

if __name__ == "__main__":
    DROP_FIRST_FRAC = 0.10           # drop first 10% steps
    A_LIST = [0.2, 0.4, 0.6, 0.8]    # multiple A values for back testing
    
    with open(f"withtheta_resmat2.pkl", "rb") as f:
        results_dict = pickle.load(f)
    results_dict = defaultdict(lambda: defaultdict(dict), results_dict)
        
    for dataset, fam_dict in results_dict.items():
        for model_family, value_dict in fam_dict.items():
            output_dir = f"../result/pretrain_mainfigures_resmat2/{dataset}_{model_family}"
            os.makedirs(output_dir, exist_ok=True)
            
            flops = value_dict["flops"]
            thetass_pvocab = value_dict["thetass_pvocab"]
            thetass_pchoices = value_dict["thetass_pchoices"]
            thetass_acc = value_dict["thetass_acc"]
            
            # theta convergence
            for thetass, name in zip(
                [thetass_acc, thetass_pvocab, thetass_pchoices],
                ["acc", "pvocab", "p_choices"]
            ):
                with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                    fig, axes = plt.subplots(nrows=thetass.shape[0], ncols=1, figsize=(6, 2*thetass.shape[0]), sharex=True)
                    for i, ax in enumerate(axes):
                        ax.plot(np.arange(thetass.shape[-1]), thetass[i], label=f"{flops[i]/1e21:.2f} * 1e21")
                        ax.set_ylabel(r"$\theta$", fontsize=16)
                        ax.legend(fontsize=16)
                        ax.tick_params(axis="both", labelsize=16)
                    axes[-1].set_xlabel("Budget", fontsize=16)
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/theta_convergence_{name}.png", dpi=100, bbox_inches="tight")
                    plt.close()

            # law curve
            final_thetas_pvocab = thetass_pvocab[:, -1]
            final_thetas_pchoices = thetass_pchoices[:, -1]
            final_thetas_acc = thetass_acc[:, -1]
            y_sub_pvocab = value_dict["y_sub_prob_vocab_correct"]
            y_sub_pchoices = value_dict["y_sub_prob_choices_correct"]
            # y_sub_acc = value_dict["y_sub_acc"]
            
            subset_size = 50
            resmat = value_dict["resmat_acc"]
            rng = np.random.default_rng(0)
            y_sub_rows = []
            for _, row in resmat.iterrows():
                vals = row.to_numpy()
                k = subset_size
                chosen = rng.choice(len(vals), size=k, replace=False)
                y_sub_rows.append(float(np.nanmean(vals[chosen])))
            y_sub_acc = np.array(y_sub_rows)
            assert y_sub_acc.shape == y_sub_pchoices.shape
                
            y_full_pvocab = value_dict["y_full_prob_vocab_correct"]
            y_full_pchoices = value_dict["y_full_prob_choices_correct"]
            
            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                fig, ax1 = plt.subplots(figsize=(8, 4))
                if dataset.split("_")[0] == "mmlu":
                    ax1.plot(flops, y_full_pvocab, color="tab:blue", linewidth=1.5, label="fullset p_vocab (GT)")
                    ax1.plot(flops, y_sub_pvocab, color="tab:blue", linewidth=1.5, label="subset p_vocab", linestyle="--")
                else:
                    ax1.plot(flops, y_full_pchoices, color="tab:blue", linewidth=1.5, label="fullset p_choices (GT)")
                ax1.plot(flops, y_sub_pchoices, color='tab:blue', linewidth=1.5, label="subset p_choices", linestyle=":")
                # ax1.plot(flops, y_sub_acc, color='tab:blue', linewidth=1.5, label="subset acc", linestyle="-.")
                ax1.set_xlabel("FLOP", fontsize=16)
                ax1.set_xscale("log")
                ax1.set_ylabel("Mean score", color='tab:blue', fontsize=16)
                ax1.set_title(f"{model_family}, {dataset}", fontsize=18)
                ax1.tick_params(axis="x", labelsize=16)
                ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=16)
                ax2 = ax1.twinx()
                if dataset.split("_")[0] == "mmlu":
                    ax2.plot(flops, final_thetas_pvocab, color='tab:red', linewidth=1.5, label="subset beta-IRT p_vocab", linestyle="--")
                ax2.plot(flops, final_thetas_pchoices, color='tab:red', linewidth=1.5, label="subset beta-IRT p_choices", linestyle=":")
                # ax2.plot(flops, final_thetas_acc, color='tab:red', linewidth=1.5, label="subset binary-IRT acc", linestyle="-.")
                ax2.set_ylabel(r"IRT $\theta$", color='tab:red', fontsize=16)
                # ax2.set_ylim(-5, None)
                ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=16)
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="center left", bbox_to_anchor=(1.25, 0.5))
                plt.tight_layout()
                plt.savefig(f"{output_dir}/law_curve.png", dpi=300, bbox_inches="tight")
                plt.close()

            # prob corr
            discris = torch.tensor(value_dict["discris"], dtype=torch.float)   # (n_items,)
            zs      = torch.tensor(value_dict["zs"],      dtype=torch.float)   # (n_items,)
            metric_specs = [("p_choices", torch.tensor(final_thetas_pchoices, dtype=torch.float), value_dict["resmat_prob_choices_correct"].values)]
            if dataset.split("_")[0] == "mmlu":
                metric_specs += [("p_vocab", torch.tensor(final_thetas_pvocab, dtype=torch.float), value_dict["resmat_prob_vocab_correct"].values)]
            for shortname, final_thetas_vec, resmat in metric_specs:
                irt_probs = torch.sigmoid(discris[None, :] * (final_thetas_vec[:, None] - zs[None, :])).flatten().numpy()
                resmat = resmat.flatten()
                rho, pval = spearmanr(irt_probs, resmat, nan_policy="omit")
                mask = (~np.isnan(irt_probs)) & (~np.isnan(resmat)) & np.isfinite(irt_probs) & np.isfinite(resmat)
                irt_probs = irt_probs[mask]
                resmat = resmat[mask]

                with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                    fig = plt.figure(figsize=(6, 6))
                    gs = gridspec.GridSpec(
                        5, 5, figure=fig,
                        wspace=0.05, hspace=0.05
                    )
                    ax_scatter = fig.add_subplot(gs[0:4, 1:5])   # big square area
                    ax_left    = fig.add_subplot(gs[0:4, 0], sharey=ax_scatter)  # LEFT marginal
                    ax_bottom  = fig.add_subplot(gs[4,   1:5], sharex=ax_scatter)  # BOTTOM marginal

                    # ---- scatter ----
                    ax_scatter.scatter(irt_probs, resmat, s=10)
                    ax_scatter.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="black")
                    ax_scatter.set_xlim(0, 1)
                    ax_scatter.set_ylim(0, 1)
                    ax_scatter.set_xlabel("IRT Prob", fontsize=18)
                    ax_scatter.set_ylabel(shortname, fontsize=18)
                    ax_scatter.tick_params(axis="both", labelsize=14)
                    
                    # ---- left marginal (y) ----
                    ax_left.hist(resmat, bins=30, orientation="horizontal")
                    ax_left.set_ylim(0, 1)
                    ax_left.invert_xaxis()
                    ax_left.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
                    ax_left.set_xticks([])
                    for s in ("top", "right", "bottom", "left"):
                        ax_left.spines[s].set_visible(False)

                    # ---- bottom marginal (x) ----
                    ax_bottom.hist(irt_probs, bins=30)
                    ax_bottom.set_xlim(0, 1)
                    ax_bottom.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
                    ax_bottom.set_yticks([])
                    for s in ("top", "right", "bottom", "left"):
                        ax_bottom.spines[s].set_visible(False)

                    fig.suptitle(rf"{model_family}, {dataset} ($\rho$ = {rho:.2f})", fontsize=16)
                    plt.subplots_adjust(wspace=0.05, hspace=0.05)
                    fig.savefig(f"{output_dir}/irt_corr_{shortname}.png", dpi=300, bbox_inches="tight")
                    plt.close(fig)
