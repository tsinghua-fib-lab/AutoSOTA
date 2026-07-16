from pathlib import Path
import pickle
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy.special import expit
from scipy.stats import spearmanr
from tueplots import bundles
bundles.icml2024()
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent))
from utils import (
    recursive_defaultdict,
    fn_step1_classic,
    fn_step2_classic,
    fn_step1_irt,
)

LONG_INPUT_PATH = BASE_DIR / "data" / "6_long.parquet"
DIFF_INPUT_PATH = BASE_DIR / "data" / "6_difficulty.pkl"
DICT_INPUT_PATH = BASE_DIR / "data" / "7_filt_laws.pkl"
PROB_MATRIX_1PL_PATH = BASE_DIR / "data" / "5_prob_matrix_cated.parquet"
PROB_MATRIX_2PL_PATH = BASE_DIR / "data" / "5_prob_matrix_cated_2pl.parquet"
DECISION_ACC_OUTPUT_DIR = BASE_DIR / "results" / "8_plot" / "decision_accuracy"
CURVE_FIT_OUTPUT_DIR = BASE_DIR / "results" / "8_plot" / "curve_fit"
DIFFICULTY_PLOT_OUTPUT_DIR = BASE_DIR / "results" / "8_plot" / "prob_corr"

def calculate_decisionacc(gt_rank, pred_rank):
    gt_pos = {name: i for i, name in enumerate(gt_rank)}
    pred_pos = {name: i for i, name in enumerate(pred_rank)}
    assert set(gt_pos) == set(pred_pos)
    total = 0
    match = 0
    for i in range(len(gt_rank)):
        for j in range(i + 1, len(gt_rank)):
            total += 1
            a, b = gt_rank[i], gt_rank[j]
            if (gt_pos[a] - gt_pos[b]) * (pred_pos[a] - pred_pos[b]) > 0:
                match += 1
    return match / total

def rank_mix(df, col_name):
    return df.set_index("model_data_mix")[col_name].sort_values(ascending=False).index.tolist()

if __name__ == "__main__":
    # 1. decision accuracy
    DECISION_ACC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df_input = pd.read_parquet(LONG_INPUT_PATH)
    target_flop = df_input["FLOP"].max()
    target_df = df_input[df_input["FLOP"] == target_flop]
    assert len(target_df) == 20
    with open(DICT_INPUT_PATH, "rb") as f:
        input_dict = pickle.load(f)

    for bench, bench_dict in tqdm(input_dict.items(), desc="decision accuracy"):
        gt_rank = rank_mix(target_df, f"acc_full_{bench}")

        bench_rows = []
        for mix, mix_dict in bench_dict.items():
            for max_size, size_dict in mix_dict.items():
                bench_rows.append(
                    {
                        "flop_ratio": size_dict["max_size_flop"] / target_flop,
                        "max_model_size": max_size,
                        "model_data_mix": mix,
                        "classic_acc": size_dict["classic"]["pred_1B"]["acc"],
                        "classic_prob": size_dict["classic"]["pred_1B"]["prob"],
                        "irt_binary_1pl": size_dict["irt"]["pred_1B"]["binary_1pl"],
                        "irt_beta_1pl": size_dict["irt"]["pred_1B"]["beta_1pl"],
                        "irt_binary_2pl": size_dict["irt"]["pred_1B"]["binary_2pl"],
                        "irt_beta_2pl": size_dict["irt"]["pred_1B"]["beta_2pl"],
                    }
                )
        bench_df = pd.DataFrame(bench_rows)
        
        plot_rows = []
        for max_size in tqdm(bench_df["max_model_size"].unique(), desc="max_model_size", leave=False):
            flop_df = bench_df[bench_df["max_model_size"] == max_size]
            flop_ratio = flop_df["flop_ratio"].mean()
            classic_acc_rank = rank_mix(flop_df, "classic_acc")
            classic_acc_decisionacc = calculate_decisionacc(gt_rank, classic_acc_rank)
            classic_prob_rank = rank_mix(flop_df, "classic_prob")
            classic_prob_decisionacc = calculate_decisionacc(gt_rank, classic_prob_rank)
            irt_binary_1pl_rank = rank_mix(flop_df, "irt_binary_1pl")
            irt_binary_1pl_decisionacc = calculate_decisionacc(gt_rank, irt_binary_1pl_rank)
            irt_beta_1pl_rank = rank_mix(flop_df, "irt_beta_1pl")
            irt_beta_1pl_decisionacc = calculate_decisionacc(gt_rank, irt_beta_1pl_rank)
            irt_binary_2pl_rank = rank_mix(flop_df, "irt_binary_2pl")
            irt_binary_2pl_decisionacc = calculate_decisionacc(gt_rank, irt_binary_2pl_rank)
            irt_beta_2pl_rank = rank_mix(flop_df, "irt_beta_2pl")
            irt_beta_2pl_decisionacc = calculate_decisionacc(gt_rank, irt_beta_2pl_rank)
            plot_rows.append(
                {
                    "flop_ratio": flop_ratio,
                    "classic_acc_decisionacc": classic_acc_decisionacc,
                    "classic_prob_decisionacc": classic_prob_decisionacc,
                    "irt_binary_1pl_decisionacc": irt_binary_1pl_decisionacc,
                    "irt_beta_1pl_decisionacc": irt_beta_1pl_decisionacc,
                    "irt_binary_2pl_decisionacc": irt_binary_2pl_decisionacc,
                    "irt_beta_2pl_decisionacc": irt_beta_2pl_decisionacc,
                }
            )
        plot_df = pd.DataFrame(plot_rows)

        with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(
                plot_df["flop_ratio"],
                plot_df["classic_acc_decisionacc"],
                marker="o",
                markersize=4,
                alpha=0.5,
                color="black",
                linestyle="--",
                label="Traditional Acc",
            )
            ax.plot(
                plot_df["flop_ratio"],
                plot_df["classic_prob_decisionacc"],
                marker="o",
                markersize=4,
                alpha=0.5,
                color="black",
                linestyle="-",
                label=r"Traditional $\mathrm{p_{Correct Choice}}$",
            )
            ax.plot(
                plot_df["flop_ratio"],
                plot_df["irt_binary_1pl_decisionacc"],
                marker="o",
                markersize=4,
                alpha=0.5,
                color="tab:blue",
                linestyle="--",
                label="Binary-IRT 1PL",
            )
            ax.plot(
                plot_df["flop_ratio"],
                plot_df["irt_beta_1pl_decisionacc"],
                marker="o",
                markersize=4,
                alpha=0.5,
                color="tab:blue",
                linestyle="-",
                label="Beta-IRT 1PL",
            )
            ax.plot(
                plot_df["flop_ratio"],
                plot_df["irt_binary_2pl_decisionacc"],
                marker="o",
                markersize=4,
                alpha=0.5,
                color="tab:red",
                linestyle="--",
                label="Binary-IRT 2PL",
            )
            ax.plot(
                plot_df["flop_ratio"],
                plot_df["irt_beta_2pl_decisionacc"],
                marker="o",
                markersize=4,
                alpha=0.5,
                color="tab:red",
                linestyle="-",
                label="Beta-IRT 2PL",
            )
            ax.set_xlabel("Max FLOP for Predicting / Target FLOP", fontsize=16)
            ax.set_ylabel("Decision Accuracy", fontsize=16)
            ax.set_title(f"{bench}", fontsize=18)
            ax.set_xscale("log")
            ax.set_ylim(0.0, 1.0)
            ax.tick_params(axis="both", labelsize=14)
            ax.legend(fontsize=12)
            fig.tight_layout()
            fig.savefig(
                DECISION_ACC_OUTPUT_DIR / f"decision_accuracy_{bench}.png",
                dpi=300, bbox_inches="tight"
            )
            plt.close(fig)
                
    # 2. curve fit
    for bench, bench_dict in tqdm(input_dict.items(), desc="curve fit"):
        for mix, mix_dict in tqdm(bench_dict.items(), desc="mix", leave=False):
            for max_size, size_dict in tqdm(mix_dict.items(), desc="max_size", leave=False):
                output_dir = CURVE_FIT_OUTPUT_DIR / bench / mix / f"max_size_{max_size}"
                output_dir.mkdir(parents=True, exist_ok=True)

                # classic step1
                classic_step1_data = size_dict["classic"]["data"]["step1"]
                flops = np.array([row[0] for row in classic_step1_data], dtype=float)
                bpbs = np.array([row[1] for row in classic_step1_data], dtype=float)
                f1_paras = size_dict["classic"]["paras"]["step1"]
                x_curve = np.logspace(np.log10(flops.min()), np.log10(flops.max()), 200)
                y_curve = fn_step1_classic(x_curve, f1_paras)

                with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.scatter(flops, bpbs, color="tab:blue", alpha=0.7)
                    ax.plot(x_curve, y_curve, color="tab:blue")
                    ax.set_xlabel("FLOP", fontsize=16)
                    ax.set_ylabel(r"Benchmark-Specific Loss $L$", fontsize=16)
                    # ax.set_title(f"Traditional Step1: {bench}, {mix}, Max Model Size={max_size}", fontsize=18)
                    ax.set_title(f"Traditional Step1: {bench}, {mix}", fontsize=18)
                    ax.set_xscale("log")
                    ax.tick_params(axis="both", labelsize=14)
                    fig.tight_layout()
                    fig.savefig(output_dir / "classic_step1.png", dpi=300, bbox_inches="tight")
                    plt.close(fig)

                # classic step2
                classic_step2_data = size_dict["classic"]["data"]["step2"]
                bpbs = np.array([row[0] for row in classic_step2_data], dtype=float)
                accs = np.array([row[1] for row in classic_step2_data], dtype=float)
                probs = np.array([row[2] for row in classic_step2_data], dtype=float)
                f21_paras = size_dict["classic"]["paras"]["step2_acc"]
                f22_paras = size_dict["classic"]["paras"]["step2_prob"]
                x_curve = np.linspace(bpbs.min(), bpbs.max(), 200)
                y_acc_curve = fn_step2_classic(x_curve, f21_paras)
                y_prob_curve = fn_step2_classic(x_curve, f22_paras)

                with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.scatter(bpbs, accs, color="tab:blue", alpha=0.7, label="Empirical Acc")
                    ax.scatter(bpbs, probs, color="tab:red", alpha=0.7, label=r"Empirical $\mathrm{p_{Correct Choice}}$")
                    ax.plot(x_curve, y_acc_curve, color="tab:blue", label="Acc Curve")
                    ax.plot(x_curve, y_prob_curve, color="tab:red", label=r"$\mathrm{p_{Correct Choice}}$ Curve")
                    ax.set_xlabel(r"Benchmark-Specific Loss $L$", fontsize=16)
                    ax.set_ylabel(r"Acc / $\mathrm{p_{Correct Choice}}$", fontsize=16)
                    # ax.set_title(f"Classic Step2: {bench}, {mix}, Max Model Size={max_size}", fontsize=18)
                    ax.set_title(f"Traditional Step2: {bench}, {mix}", fontsize=18)
                    ax.tick_params(axis="both", labelsize=14)
                    ax.legend(fontsize=12)
                    fig.tight_layout()
                    fig.savefig(output_dir / "classic_step2.png", dpi=300, bbox_inches="tight")
                    plt.close(fig)

                # irt step1
                irt_step1 = size_dict["irt"]["data"]["step1"]
                flops = np.array([row[0] for row in irt_step1], dtype=float)
                theta_binary_1pl = np.array([row[1] for row in irt_step1], dtype=float)
                theta_beta_1pl = np.array([row[2] for row in irt_step1], dtype=float)
                theta_binary_2pl = np.array([row[3] for row in irt_step1], dtype=float)
                theta_beta_2pl = np.array([row[4] for row in irt_step1], dtype=float)
                g11_paras = size_dict["irt"]["paras"]["step1_binary_1pl"]
                g12_paras = size_dict["irt"]["paras"]["step1_beta_1pl"]
                g13_paras = size_dict["irt"]["paras"]["step1_binary_2pl"]
                g14_paras = size_dict["irt"]["paras"]["step1_beta_2pl"]
                x_curve = np.logspace(np.log10(flops.min()), np.log10(flops.max()), 200)
                y_binary_1pl_curve = fn_step1_irt(x_curve, g11_paras)
                y_beta_1pl_curve = fn_step1_irt(x_curve, g12_paras)
                y_binary_2pl_curve = fn_step1_irt(x_curve, g13_paras)
                y_beta_2pl_curve = fn_step1_irt(x_curve, g14_paras)

                with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.scatter(flops, theta_binary_1pl, color="tab:blue", alpha=0.7, label=r"Empirical Binary-IRT 1PL $\theta$")
                    ax.scatter(flops, theta_beta_1pl, color="tab:red", alpha=0.7, label=r"Empirical Beta-IRT 1PL $\theta$")
                    ax.scatter(flops, theta_binary_2pl, color="tab:green", alpha=0.7, label=r"Empirical Binary-IRT 2PL $\theta$")
                    ax.scatter(flops, theta_beta_2pl, color="tab:purple", alpha=0.7, label=r"Empirical Beta-IRT 2PL $\theta$")
                    ax.plot(x_curve, y_binary_1pl_curve, color="tab:blue", label="Binary-IRT 1PL Curve")
                    ax.plot(x_curve, y_beta_1pl_curve, color="tab:red", label="Beta-IRT 1PL Curve")
                    ax.plot(x_curve, y_binary_2pl_curve, color="tab:green", label="Binary-IRT 2PL Curve")
                    ax.plot(x_curve, y_beta_2pl_curve, color="tab:purple", label="Beta-IRT 2PL Curve")
                    ax.set_xlabel("FLOP", fontsize=16)
                    ax.set_ylabel(r"$\theta$", fontsize=16)
                    # ax.set_title(f"IRT Step1: {bench}, {mix}, Max Model Size={max_size}", fontsize=18)
                    ax.set_title(f"IRSL Step1: {bench}, {mix}", fontsize=18)
                    ax.set_xscale("log")
                    ax.tick_params(axis="both", labelsize=14)
                    ax.legend(fontsize=12, framealpha=0.5)
                    fig.tight_layout()
                    fig.savefig(output_dir / "irt_step1.png", dpi=300, bbox_inches="tight")
                    plt.close(fig)

    # 3. prob correlation plot
    with open(DIFF_INPUT_PATH, "rb") as f:
        difficulty_dict = pickle.load(f)

    prob_1pl_df = pd.read_parquet(PROB_MATRIX_1PL_PATH)
    prob_1pl_target_df = prob_1pl_df[
        (prob_1pl_df.index.get_level_values("model_split") == "test")
        & (prob_1pl_df.index.get_level_values("model_size") == "1B")
        & (
            prob_1pl_df.index.get_level_values("model_step").astype(int)
            == prob_1pl_df.index.get_level_values("model_step").astype(int).max()
        )
    ].copy()
    assert len(prob_1pl_target_df) == 20
    prob_2pl_df = pd.read_parquet(PROB_MATRIX_2PL_PATH)
    prob_2pl_target_df = prob_2pl_df[
        (prob_2pl_df.index.get_level_values("model_split") == "test")
        & (prob_2pl_df.index.get_level_values("model_size") == "1B")
        & (
            prob_2pl_df.index.get_level_values("model_step").astype(int)
            == prob_2pl_df.index.get_level_values("model_step").astype(int).max()
        )
    ].copy()
    assert len(prob_2pl_target_df) == 20
    bench_names = prob_1pl_target_df.columns.get_level_values("bench_name").map(
        lambda b: "mmlu" if b.startswith("mmlu") else b
    )
    zs_1pl = prob_1pl_target_df.columns.get_level_values("difficulty").to_numpy(dtype=np.float32)
    zs_2pl = prob_2pl_target_df.columns.get_level_values("difficulty").to_numpy(dtype=np.float32)
    ds_2pl = prob_2pl_target_df.columns.get_level_values("discrimination").to_numpy(dtype=np.float32)

    for bench, bench_dict in tqdm(input_dict.items(), desc="difficulty plot"):
        bench_mask = bench_names == bench
        bench_z_1pl = zs_1pl[bench_mask]
        bench_z_2pl = zs_2pl[bench_mask]
        bench_d_2pl = ds_2pl[bench_mask]

        for mix, mix_dict in tqdm(bench_dict.items(), desc="mix", leave=False):
            mix_prob_1pl = prob_1pl_target_df.xs(mix, level="model_data_mix").iloc[0].to_numpy(dtype=float)
            prob_gt_1pl = mix_prob_1pl[bench_mask]
            mix_prob_2pl = prob_2pl_target_df.xs(mix, level="model_data_mix").iloc[0].to_numpy(dtype=float)
            prob_gt_2pl = mix_prob_2pl[bench_mask]

            for max_size, size_dict in tqdm(mix_dict.items(), desc="max_size", leave=False):
                output_dir = DIFFICULTY_PLOT_OUTPUT_DIR / bench / mix / f"max_size_{max_size}"
                output_dir.mkdir(parents=True, exist_ok=True)

                theta_pred_1pl = fn_step1_irt(
                    flop=target_flop,
                    paras=size_dict["irt"]["paras"]["step1_beta_1pl"],
                )
                prob_pred_1pl = expit(theta_pred_1pl + bench_z_1pl)
                rho_beta_1pl, _ = spearmanr(prob_pred_1pl, prob_gt_1pl)

                with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                    fig = plt.figure(figsize=(6, 6))
                    gs = gridspec.GridSpec(5, 5, figure=fig, wspace=0.05, hspace=0.05)
                    ax_scatter = fig.add_subplot(gs[0:4, 1:5])
                    ax_left = fig.add_subplot(gs[0:4, 0], sharey=ax_scatter)
                    ax_bottom = fig.add_subplot(gs[4, 1:5], sharex=ax_scatter)

                    ax_scatter.scatter(prob_pred_1pl, prob_gt_1pl, s=10)
                    ax_scatter.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="black")
                    ax_scatter.set_xlim(0, 1)
                    ax_scatter.set_ylim(0, 1)
                    ax_scatter.set_xlabel("IRT Probability", fontsize=16)
                    ax_scatter.set_ylabel("Empirical Probability", fontsize=16)
                    ax_scatter.tick_params(axis="both", labelsize=12)

                    ax_left.hist(prob_gt_1pl, bins=30, orientation="horizontal")
                    ax_left.set_ylim(0, 1)
                    ax_left.invert_xaxis()
                    ax_left.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
                    ax_left.set_xticks([])
                    for spine in ("top", "right", "bottom", "left"):
                        ax_left.spines[spine].set_visible(False)

                    ax_bottom.hist(prob_pred_1pl, bins=30)
                    ax_bottom.set_xlim(0, 1)
                    ax_bottom.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
                    ax_bottom.set_yticks([])
                    for spine in ("top", "right", "bottom", "left"):
                        ax_bottom.spines[spine].set_visible(False)

                    fig.suptitle(
                        rf"{bench}, {mix}, {max_size} (beta 1PL, $\rho$ = {rho_beta_1pl:.2f})",
                        fontsize=14,
                    )
                    plt.subplots_adjust(wspace=0.05, hspace=0.05)
                    fig.savefig(output_dir / f"scatter_corr_beta_1pl_{mix}_{max_size}_{bench}.png", dpi=300, bbox_inches="tight")
                    plt.close(fig)

                theta_pred_2pl = fn_step1_irt(
                    flop=target_flop,
                    paras=size_dict["irt"]["paras"]["step1_beta_2pl"],
                )
                prob_pred_2pl = expit(bench_d_2pl * (theta_pred_2pl + bench_z_2pl))
                rho_beta_2pl, _ = spearmanr(prob_pred_2pl, prob_gt_2pl)

                with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                    fig = plt.figure(figsize=(6, 6))
                    gs = gridspec.GridSpec(5, 5, figure=fig, wspace=0.05, hspace=0.05)
                    ax_scatter = fig.add_subplot(gs[0:4, 1:5])
                    ax_left = fig.add_subplot(gs[0:4, 0], sharey=ax_scatter)
                    ax_bottom = fig.add_subplot(gs[4, 1:5], sharex=ax_scatter)

                    ax_scatter.scatter(prob_pred_2pl, prob_gt_2pl, s=10)
                    ax_scatter.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="black")
                    ax_scatter.set_xlim(0, 1)
                    ax_scatter.set_ylim(0, 1)
                    ax_scatter.set_xlabel("IRT Probability", fontsize=16)
                    ax_scatter.set_ylabel("Empirical Probability", fontsize=16)
                    ax_scatter.tick_params(axis="both", labelsize=12)

                    ax_left.hist(prob_gt_2pl, bins=30, orientation="horizontal")
                    ax_left.set_ylim(0, 1)
                    ax_left.invert_xaxis()
                    ax_left.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
                    ax_left.set_xticks([])
                    for spine in ("top", "right", "bottom", "left"):
                        ax_left.spines[spine].set_visible(False)

                    ax_bottom.hist(prob_pred_2pl, bins=30)
                    ax_bottom.set_xlim(0, 1)
                    ax_bottom.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
                    ax_bottom.set_yticks([])
                    for spine in ("top", "right", "bottom", "left"):
                        ax_bottom.spines[spine].set_visible(False)

                    fig.suptitle(
                        rf"{bench}, {mix}, {max_size} (beta 2PL, $\rho$ = {rho_beta_2pl:.2f})",
                        fontsize=14,
                    )
                    plt.subplots_adjust(wspace=0.05, hspace=0.05)
                    fig.savefig(output_dir / f"scatter_corr_beta_2pl_{mix}_{max_size}_{bench}.png", dpi=300, bbox_inches="tight")
                    plt.close(fig)
