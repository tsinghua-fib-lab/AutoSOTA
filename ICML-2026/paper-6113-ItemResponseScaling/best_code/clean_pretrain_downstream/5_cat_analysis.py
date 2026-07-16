import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy.special import expit
from scipy.stats import spearmanr, gaussian_kde
from tueplots import bundles
from tqdm import tqdm
bundles.icml2024()
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--loss-kind", type=str, default="beta", choices=["beta", "binary"])
parser.add_argument("--irt-model", type=str, default="1pl", choices=["1pl", "2pl"])
args = parser.parse_args()

SAMPLE_QUESTIONS = 5

BASE_DIR = Path(__file__).resolve().parent
irt_suffix = "_2pl" if args.irt_model == "2pl" else ""
input_path = (
    BASE_DIR / "data" / f"5_prob_matrix_cated{irt_suffix}.parquet"
    if args.loss_kind == "beta"
    else BASE_DIR / "data" / f"5_binary_matrix_cated{irt_suffix}.parquet"
)
scatter_root = BASE_DIR / "results" / f"5_cat_analysis{irt_suffix}" / "corr_scatter"
scatter_root.mkdir(parents=True, exist_ok=True)
curve_root = BASE_DIR / "results" / f"5_cat_analysis{irt_suffix}" / "irt_curve"
curve_root.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(input_path)
test_df = df[df.index.get_level_values("model_split") == "test"].copy()
ys = test_df.to_numpy(dtype=np.float32)
bench_names = test_df.columns.get_level_values("bench_name").map(
    lambda b: "mmlu" if b.startswith("mmlu") else b
)
unique_bench_names = sorted(bench_names.unique())
zs = test_df.columns.get_level_values("difficulty").to_numpy(dtype=np.float32)
if args.irt_model == "2pl":
    alphas = test_df.columns.get_level_values("discrimination").to_numpy(dtype=np.float32)

for bench in tqdm(unique_bench_names, desc="benches"):
    bench_thetas = test_df.index.get_level_values(f"ability_{bench}").to_numpy(dtype=np.float32)
    
    bench_mask = bench_names == bench
    bench_ys = ys[:, bench_mask]
    bench_zs = zs[bench_mask]
    if args.irt_model == "2pl":
        bench_alphas = alphas[bench_mask]

    # Scatter plot + correlation
    if args.loss_kind == "beta":
        if args.irt_model == "2pl":
            p_pred_full = expit(bench_alphas[None, :] * (bench_thetas[:, None] + bench_zs[None, :]))
        else:
            p_pred_full = expit(bench_thetas[:, None] + bench_zs[None, :])
        rho, _ = spearmanr(p_pred_full.reshape(-1), bench_ys.reshape(-1))
        out_path = scatter_root / f"prob_corr_{bench}.png"
        # with plt.rc_context(bundles.icml2024(usetex=True, family='serif')):
        #     fig = plt.figure(figsize=(6, 6))
        #     gs = gridspec.GridSpec(5, 5, figure=fig, wspace=0.05, hspace=0.05)
        #     ax_scatter = fig.add_subplot(gs[0:4, 1:5])
        #     ax_left = fig.add_subplot(gs[0:4, 0], sharey=ax_scatter)
        #     ax_bottom = fig.add_subplot(gs[4, 1:5], sharex=ax_scatter)

        #     ax_scatter.scatter(p_pred_full.reshape(-1), bench_ys.reshape(-1), s=10)
        #     ax_scatter.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="black")
        #     ax_scatter.set_xlim(0, 1)
        #     ax_scatter.set_ylim(0, 1)
        #     ax_scatter.set_xlabel(r"Beta-IRT Predicted $\mathrm{p_{Correct Choice}}$", fontsize=18)
        #     ax_scatter.set_ylabel(r"Empirical $\mathrm{p_{Correct Choice}}$", fontsize=18)
        #     ax_scatter.tick_params(axis="both", labelsize=14)

        #     ax_left.hist(bench_ys.reshape(-1), bins=30, orientation="horizontal")
        #     ax_left.set_ylim(0, 1)
        #     ax_left.invert_xaxis()
        #     ax_left.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
        #     ax_left.set_xticks([])
        #     for spine in ("top", "right", "bottom", "left"):
        #         ax_left.spines[spine].set_visible(False)

        #     ax_bottom.hist(p_pred_full.reshape(-1), bins=30)
        #     ax_bottom.set_xlim(0, 1)
        #     ax_bottom.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
        #     ax_bottom.set_yticks([])
        #     for spine in ("top", "right", "bottom", "left"):
        #         ax_bottom.spines[spine].set_visible(False)

        #     fig.suptitle(rf"{bench} ($\rho$ = {rho:.2f})", fontsize=18)
        #     plt.subplots_adjust(wspace=0.05, hspace=0.05)
        #     fig.savefig(out_path, dpi=300, bbox_inches="tight")
        #     plt.close(fig)
        with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
            fig = plt.figure(figsize=(6, 6))
            gs = gridspec.GridSpec(
                5,
                5,
                figure=fig,
                wspace=0.05,
                hspace=0.05,
                width_ratios=[1, 4, 4, 4, 4],
                height_ratios=[4, 4, 4, 4, 1],
            )
            ax_main = fig.add_subplot(gs[0:4, 1:5])
            ax_left = fig.add_subplot(gs[0:4, 0], sharey=ax_main)
            ax_bottom = fig.add_subplot(gs[4, 1:5], sharex=ax_main)

            x = p_pred_full.reshape(-1)
            y = bench_ys.reshape(-1)
            max_kde_points = 50000
            if x.size > max_kde_points:
                rng = np.random.default_rng(0)
                sample_idx = rng.choice(x.size, size=max_kde_points, replace=False)
                x_kde = x[sample_idx]
                y_kde = y[sample_idx]
            else:
                x_kde = x
                y_kde = y
            xy = np.vstack([x, y])
            kde = gaussian_kde(np.vstack([x_kde, y_kde]))
            grid = np.linspace(0, 1, 120)
            xx, yy = np.meshgrid(grid, grid)
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

            ax_main.contourf(xx, yy, zz, levels=30, cmap="Blues")
            ax_main.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="black")
            ax_main.set_xlim(0, 1)
            ax_main.set_ylim(0, 1)
            ax_main.set_xlabel(r"Beta-IRT Predicted $\mathrm{p_{Correct Choice}}$", fontsize=18)
            ax_main.set_ylabel(r"Empirical $\mathrm{p_{Correct Choice}}$", fontsize=18)
            ax_main.tick_params(axis="both", labelsize=14)

            ax_left.hist(y, bins=30, orientation="horizontal", color="0.7", alpha=1.0)
            ax_left.set_ylim(0, 1)
            ax_left.invert_xaxis()
            ax_left.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
            ax_left.set_xticks([])
            for spine in ("top", "right", "bottom", "left"):
                ax_left.spines[spine].set_visible(False)

            ax_bottom.hist(x, bins=30, color="0.7", alpha=1.0)
            ax_bottom.set_xlim(0, 1)
            ax_bottom.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
            ax_bottom.set_yticks([])
            for spine in ("top", "right", "bottom", "left"):
                ax_bottom.spines[spine].set_visible(False)

            fig.suptitle(rf"{bench} ($\rho$ = {rho:.2f})", fontsize=18)
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

    # IRT curves per question (sampled)
    sample_idxs = np.random.default_rng(0).choice(bench_ys.shape[1], size=SAMPLE_QUESTIONS, replace=False)
    theta_arange = np.linspace(bench_thetas.min() - 1, bench_thetas.max() + 1, 200)

    for idx in sample_idxs:
        z_j = bench_zs[idx]
        if args.irt_model == "2pl":
            alpha_j = bench_alphas[idx]
        y_j = bench_ys[:, idx]
        if args.irt_model == "2pl":
            curve = expit(alpha_j * (theta_arange + z_j))
        else:
            curve = expit(theta_arange + z_j)

        with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
            plt.figure(figsize=(6, 4))
            plt.scatter(bench_thetas, y_j, s=10, label=r"Empirical $\mathrm{p_{Correct Choice}}$")
            plt.plot(theta_arange, curve, color="red", label="Beta-IRT Curve")
            plt.xlabel(r"$\theta$", fontsize=14)
            plt.ylabel(r"$\mathrm{p_{Correct Choice}}$", fontsize=14)
            plt.ylim(0, 1)
            plt.legend(fontsize=10)
            plt.title(f"{bench}, Question {idx}", fontsize=14)
            plt.tight_layout()
            out_path = curve_root / f"{args.loss_kind}_{bench}_{idx}.png"
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
