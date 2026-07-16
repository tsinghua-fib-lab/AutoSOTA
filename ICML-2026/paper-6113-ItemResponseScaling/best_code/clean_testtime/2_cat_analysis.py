import warnings
from pathlib import Path
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy.special import expit
from scipy.stats import spearmanr, gaussian_kde
from tqdm import tqdm
from tueplots import bundles
bundles.icml2024()
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "2_cat_analysis"

RNG = np.random.default_rng(0)
N_SAMPLE_QUESTIONS = 5

pt_files = sorted(DATA_DIR.glob("2_cated_*.pt"))
for pt_path in tqdm(pt_files, desc="cat analysis"):
    stem = pt_path.stem.replace("2_cated_irsl_testtime_", "")
    stem_root = RESULTS_DIR / stem
    CORR_SCATTER_DIR = stem_root / "corr_scatter"
    IRT_CURVE_DIR = stem_root / "irt_curve"
    CORR_SCATTER_DIR.mkdir(parents=True, exist_ok=True)
    IRT_CURVE_DIR.mkdir(parents=True, exist_ok=True)

    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    data_tensor = np.array(payload["data_tensor"], dtype=np.float32)
    model_names = list(payload["models"])
    test_models = list(payload["test_models"])
    test_model_indices = [i for i, m in enumerate(model_names) if m in set(test_models)]
    datasets = list(payload["datasets"])
    zs = np.array(payload["zs"], dtype=np.float32)
    alphas = None if payload.get("alphas") is None else np.array(payload.get("alphas"), dtype=np.float32)
    test_thetas_by_bench = payload["test_thetas"]

    test_probmat = np.nanmean(data_tensor[test_model_indices, :, :], axis=-1)

    # 1. scatter plot + correlation
    for dataset in tqdm(sorted(set(datasets)), desc=f"{stem} corr_scatter", leave=False):
        item_indices = [i for i, d in enumerate(datasets) if d == dataset]
        bench_ys = test_probmat[:, item_indices]
        theta_pairs = test_thetas_by_bench[dataset]
        theta_map = {m: float(t) for m, t in theta_pairs}
        test_thetas = np.array([theta_map[m] for m in test_models], dtype=np.float32)

        if alphas is None:
            p_pred_full = expit(test_thetas[:, None] + zs[item_indices][None, :])
        else:
            p_pred_full = expit(alphas[item_indices][None, :] * (test_thetas[:, None] + zs[item_indices][None, :]))
        rho, _ = spearmanr(p_pred_full.reshape(-1), bench_ys.reshape(-1))
        out_path = CORR_SCATTER_DIR / f"{stem}_{dataset}_prob_corr.png"

        # with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        #     fig = plt.figure(figsize=(6, 6))
        #     gs = gridspec.GridSpec(5, 5, figure=fig, wspace=0.05, hspace=0.05)
        #     ax_scatter = fig.add_subplot(gs[0:4, 1:5])
        #     ax_left = fig.add_subplot(gs[0:4, 0], sharey=ax_scatter)
        #     ax_bottom = fig.add_subplot(gs[4, 1:5], sharex=ax_scatter)

        #     ax_scatter.scatter(p_pred_full.reshape(-1), bench_ys.reshape(-1), s=10)
        #     ax_scatter.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="black")
        #     ax_scatter.set_xlim(0, 1)
        #     ax_scatter.set_ylim(0, 1)
        #     ax_scatter.set_xlabel(r"Beta-IRT Predicted $\mathrm{Pass@1}$", fontsize=18)
        #     ax_scatter.set_ylabel(r"Empirical $\mathrm{Pass@1}$", fontsize=18)
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

        #     fig.suptitle(rf"{dataset} ($\rho$ = {rho:.2f})", fontsize=18)
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
            kde = gaussian_kde(np.vstack([x, y]))
            grid = np.linspace(0, 1, 120)
            xx, yy = np.meshgrid(grid, grid)
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

            ax_main.contourf(xx, yy, zz, levels=30, cmap="Blues")
            ax_main.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="black")
            ax_main.set_xlim(0, 1)
            ax_main.set_ylim(0, 1)
            ax_main.set_xlabel(r"Beta-IRT Predicted $\mathrm{Pass@1}$", fontsize=18)
            ax_main.set_ylabel(r"Empirical $\mathrm{Pass@1}$", fontsize=18)
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

            fig.suptitle(rf"{dataset} ($\rho$ = {rho:.2f})", fontsize=18)
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

    # 2. irt curve
    for dataset in tqdm(sorted(set(datasets)), desc=f"{stem} irt_curve", leave=False):
        item_indices = [i for i, d in enumerate(datasets) if d == dataset]
        theta_pairs = test_thetas_by_bench[dataset]
        theta_map = {m: float(t) for m, t in theta_pairs}
        test_thetas = np.array([theta_map[m] for m in test_models], dtype=np.float32)
        sampled = RNG.choice(item_indices, size=min(N_SAMPLE_QUESTIONS, len(item_indices)), replace=False)
        theta_range = np.linspace(test_thetas.min() - 1.0, test_thetas.max() + 1.0, 200)

        for item_idx in sampled:
            responses = test_probmat[:, item_idx]
            z_val = float(zs[item_idx])
            alpha_val = 1.0 if alphas is None else float(alphas[item_idx])
            curve = expit(alpha_val * (theta_range + z_val))
            out_path = IRT_CURVE_DIR / f"{stem}_{dataset}_{item_idx}.png"

            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                plt.figure(figsize=(6, 4))
                plt.scatter(test_thetas, responses, s=10, label=r"Empirical $\mathrm{Pass@1}$")
                plt.plot(theta_range, curve, color="red", label="Beta-IRT Curve")
                plt.xlabel(r"$\theta$", fontsize=14)
                plt.ylabel(r"$\mathrm{Pass@1}$", fontsize=14)
                plt.ylim(0, 1)
                plt.tick_params(axis="both", labelsize=12)
                plt.legend(fontsize=12)
                plt.title(f"{dataset}, Question {item_idx}", fontsize=14)
                plt.tight_layout()
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                plt.close()
