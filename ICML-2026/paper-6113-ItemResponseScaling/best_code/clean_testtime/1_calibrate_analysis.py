import warnings
from pathlib import Path
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy.special import expit
from scipy.stats import spearmanr
from tqdm import tqdm
from tueplots import bundles
bundles.icml2024()
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "1_calibrate_analysis"

RNG = np.random.default_rng(0)
N_SAMPLE_QUESTIONS = 5

pt_files = sorted(DATA_DIR.glob("1_calibreated_*.pt"))
for pt_path in tqdm(pt_files, desc="calibrate analysis"):
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    
    stem = pt_path.stem.replace("1_calibreated_irsl_testtime_", "")
    stem_root = RESULTS_DIR / stem
    ITEM_PARA_DIR = stem_root / "item_para_distri"
    ZS_HELM_DIR = stem_root / "zs_vs_helmzs"
    IRT_CURVE_DIR = stem_root / "irt_curve"
    CORR_SCATTER_DIR = stem_root / "corr_scatter"
    ITEM_PARA_DIR.mkdir(parents=True, exist_ok=True)
    ZS_HELM_DIR.mkdir(parents=True, exist_ok=True)
    IRT_CURVE_DIR.mkdir(parents=True, exist_ok=True)
    CORR_SCATTER_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. zs & alphas distribution plot
    zs = np.array(payload["zs"], dtype=np.float32)
    alphas = None if payload.get("alphas") is None else np.array(payload.get("alphas"), dtype=np.float32)

    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        plt.figure(figsize=(6, 4))
        plt.hist(zs, bins=40, density=True)
        plt.title(f"{stem}: z distribution", fontsize=14)
        plt.xlabel("z", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.tight_layout()
        plt.savefig(ITEM_PARA_DIR / f"{stem}_z.png", dpi=300, bbox_inches="tight")
        plt.close()

    if alphas is not None:
        with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
            plt.figure(figsize=(6, 4))
            plt.hist(alphas, bins=40, density=True)
            plt.title(f"{stem}: alpha distribution", fontsize=14)
            plt.xlabel("alpha", fontsize=12)
            plt.ylabel("Density", fontsize=12)
            plt.tight_layout()
            plt.savefig(ITEM_PARA_DIR / f"{stem}_alpha.png", dpi=300, bbox_inches="tight")
            plt.close()

    # 2. zs corr with helm_zs
    datasets = list(payload["datasets"])
    if "helm_zs" in payload:
        helm_zs = np.array(payload["helm_zs"], dtype=np.float32)
        for dataset in tqdm(sorted(set(datasets)), desc=f"{stem} zs_vs_helmzs", leave=False):
            item_indices = [i for i, d in enumerate(datasets) if d == dataset]
            zs_subset = zs[item_indices]
            helm_subset = helm_zs[item_indices]
            rho, _ = spearmanr(zs_subset, helm_subset)
            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                plt.figure(figsize=(6, 6))
                plt.scatter(zs_subset, helm_subset, s=12)
                min_val = float(min(zs_subset.min(), helm_subset.min()))
                max_val = float(max(zs_subset.max(), helm_subset.max()))
                plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1)
                plt.title(f"{stem}: {dataset} (rho={rho:.2f})", fontsize=18)
                plt.xlabel(r"Our $z$", fontsize=18)
                plt.ylabel(r"$z$ from HELM", fontsize=18)
                plt.tick_params(axis="both", labelsize=14)
                plt.tight_layout()
                plt.savefig(
                    ZS_HELM_DIR / f"{stem}_{dataset}_zsvshelmzs.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

    # 3. scatter plot + correlation
    data_tensor = np.array(payload["data_tensor"], dtype=np.float32)
    model_names = list(payload["models"])
    test_models = set(payload["test_models"])
    train_model_names = [m for m in model_names if m not in test_models]
    train_indices = [i for i, m in enumerate(model_names) if m not in test_models]
    train_probmat = np.nanmean(data_tensor[train_indices, :, :], axis=-1)
    train_thetas_by_bench = payload["train_thetas"]

    for dataset in tqdm(sorted(set(datasets)), desc=f"{stem} corr_scatter", leave=False):
        item_indices = [i for i, d in enumerate(datasets) if d == dataset]
        bench_ys = train_probmat[:, item_indices]
        theta_pairs = train_thetas_by_bench[dataset]
        theta_map = {m: float(t) for m, t in theta_pairs}
        train_thetas = np.array([theta_map[m] for m in train_model_names], dtype=np.float32)
        if alphas is None:
            p_pred_full = expit(train_thetas[:, None] + zs[item_indices][None, :])
        else:
            p_pred_full = expit(alphas[item_indices][None, :] * (train_thetas[:, None] + zs[item_indices][None, :]))
        rho, _ = spearmanr(p_pred_full.reshape(-1), bench_ys.reshape(-1))
        out_path = CORR_SCATTER_DIR / f"{stem}_{dataset}_prob_corr.png"
        with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
            fig = plt.figure(figsize=(6, 6))
            gs = gridspec.GridSpec(5, 5, figure=fig, wspace=0.05, hspace=0.05)
            ax_scatter = fig.add_subplot(gs[0:4, 1:5])
            ax_left = fig.add_subplot(gs[0:4, 0], sharey=ax_scatter)
            ax_bottom = fig.add_subplot(gs[4, 1:5], sharex=ax_scatter)

            ax_scatter.scatter(p_pred_full.reshape(-1), bench_ys.reshape(-1), s=10)
            ax_scatter.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="black")
            ax_scatter.set_xlim(0, 1)
            ax_scatter.set_ylim(0, 1)
            ax_scatter.set_xlabel("IRT Probability", fontsize=18)
            ax_scatter.set_ylabel("Empirical Probability", fontsize=18)
            ax_scatter.tick_params(axis="both", labelsize=14)

            ax_left.hist(bench_ys.reshape(-1), bins=30, orientation="horizontal")
            ax_left.set_ylim(0, 1)
            ax_left.invert_xaxis()
            ax_left.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
            ax_left.set_xticks([])
            for spine in ("top", "right", "bottom", "left"):
                ax_left.spines[spine].set_visible(False)

            ax_bottom.hist(p_pred_full.reshape(-1), bins=30)
            ax_bottom.set_xlim(0, 1)
            ax_bottom.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
            ax_bottom.set_yticks([])
            for spine in ("top", "right", "bottom", "left"):
                ax_bottom.spines[spine].set_visible(False)

            fig.suptitle(rf"{dataset} ($\rho$ = {rho:.2f})", fontsize=16)
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

    # 4. irt curve
    for dataset in tqdm(sorted(set(datasets)), desc=f"{stem} irt_curve", leave=False):
        item_indices = [i for i, d in enumerate(datasets) if d == dataset]
        theta_pairs = train_thetas_by_bench[dataset]
        theta_map = {m: float(t) for m, t in theta_pairs}
        train_thetas = np.array([theta_map[m] for m in train_model_names], dtype=np.float32)
        sampled = RNG.choice(item_indices, size=N_SAMPLE_QUESTIONS, replace=False)
        for item_idx in sampled:
            responses = train_probmat[:, item_idx]
            z_val = float(zs[item_idx])
            alpha_val = 1.0 if alphas is None else float(alphas[item_idx])
            theta_range = np.linspace(train_thetas.min() - 1.0, train_thetas.max() + 1.0, 200)
            curve = expit(alpha_val * (theta_range + z_val))
            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                plt.figure(figsize=(6, 4))
                plt.scatter(train_thetas, responses, s=10, label="Responses")
                plt.plot(theta_range, curve, color="red", label="IRT Prob")
                plt.xlabel(r"$\theta$", fontsize=12)
                plt.ylabel("IRT Prob / Responses", fontsize=12)
                plt.ylim(0, 1)
                plt.legend(fontsize=10)
                plt.title(f"{dataset}, item {item_idx}", fontsize=12)
                plt.tight_layout()
                plt.savefig(IRT_CURVE_DIR / f"{stem}_{dataset}_{item_idx}.png", dpi=300, bbox_inches="tight")
                plt.close()
