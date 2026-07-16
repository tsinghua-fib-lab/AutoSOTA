import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy.special import expit
from scipy.stats import spearmanr
from tueplots import bundles
from tqdm import tqdm

bundles.icml2024()

parser = argparse.ArgumentParser()
parser.add_argument("--loss-kind", type=str, default="beta", choices=["beta", "binary"])
parser.add_argument("--sample-questions", type=int, default=5, help="Questions to plot per bench for IRT curves.")
args = parser.parse_args()

base_dir = Path(__file__).resolve().parent
input_path = (
    base_dir / "5_prob_matrix_with_theta.parquet"
    if args.loss_kind == "beta"
    else base_dir / "5_binary_matrix_with_theta.parquet"
)
scatter_root = base_dir / "results" / "5_cat_prob_correlation"
scatter_root.mkdir(parents=True, exist_ok=True)
curve_root = base_dir / "results" / "5_IRT_curve_test"
curve_root.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(input_path)
tuple_cols = [ast.literal_eval(col) if isinstance(col, str) else col for col in df.columns]
df.columns = pd.MultiIndex.from_tuples(tuple_cols, names=["question_id", "difficulty"])
question_ids = df.columns.get_level_values("question_id")
bench_names = question_ids.map(lambda q: q.rsplit("_", 1)[0])
bench_names = bench_names.map(lambda b: "mmlu" if b.startswith("mmlu") else b).unique()

df_reset = df.reset_index()
for bench in tqdm(bench_names, desc="benches"):
    theta_col = f"model_theta_{bench}"
    bench_mask = question_ids.str.startswith(bench)
    bench_cols = df.columns[bench_mask]
    difficulties = bench_cols.get_level_values("difficulty").to_numpy(dtype=np.float64)

    theta_vals = df_reset[theta_col].to_numpy(dtype=np.float64)
    probs_true = df_reset[bench_cols].to_numpy(dtype=np.float64)

    # Scatter plot + correlation
    p_pred_full = expit(theta_vals[:, None] + difficulties[None, :])
    rho, _ = spearmanr(p_pred_full.reshape(-1), probs_true.reshape(-1))
    
    out_path = scatter_root / f"{args.loss_kind}_{bench}_correlation.png"
    with plt.rc_context(bundles.icml2024(usetex=True, family='serif')):
        fig = plt.figure(figsize=(6, 6))
        gs = gridspec.GridSpec(5, 5, figure=fig, wspace=0.05, hspace=0.05)
        ax_scatter = fig.add_subplot(gs[0:4, 1:5])
        ax_left = fig.add_subplot(gs[0:4, 0], sharey=ax_scatter)
        ax_bottom = fig.add_subplot(gs[4, 1:5], sharex=ax_scatter)

        ax_scatter.scatter(p_pred_full.reshape(-1), probs_true.reshape(-1), s=10)
        ax_scatter.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="black")
        ax_scatter.set_xlim(0, 1)
        ax_scatter.set_ylim(0, 1)
        ax_scatter.set_xlabel("IRT Probability", fontsize=18)
        ax_scatter.set_ylabel("Empirical Probability", fontsize=18)
        ax_scatter.tick_params(axis="both", labelsize=14)

        ax_left.hist(probs_true.reshape(-1), bins=30, orientation="horizontal")
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

        fig.suptitle(rf"{bench} ($\rho$ = {rho:.2f})", fontsize=16)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # IRT curves per question (sampled)
    n_items = probs_true.shape[1]
    sample_idxs = np.random.default_rng(0).choice(n_items, size=args.sample_questions, replace=False)
    theta_grid = np.linspace(theta_vals.min() - 1, theta_vals.max() + 1, 200)

    for idx in sample_idxs:
        qid = bench_cols.get_level_values("question_id")[idx]
        z_j = difficulties[idx]
        resp = probs_true[:, idx]
        mask = np.isfinite(resp)
        curve = expit(theta_grid + z_j)

        with plt.rc_context(bundles.icml2024(usetex=True, family='serif')):
            plt.figure(figsize=(6, 4))
            plt.scatter(theta_vals[mask], resp[mask], s=10, label="responses")
            plt.plot(theta_grid, curve, color="red", label="IRT curve")
            plt.xlabel(r"$\theta$", fontsize=14)
            plt.ylabel("Probability", fontsize=14)
            plt.ylim(0, 1)
            plt.legend(fontsize=12)
            plt.title(qid, fontsize=14)
            plt.tight_layout()
            plt.savefig(curve_root / f"{args.loss_kind}_{qid}.png", dpi=200, bbox_inches="tight")
            plt.close()
