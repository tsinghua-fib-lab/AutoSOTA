import argparse
from pathlib import Path
import pickle
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tueplots import bundles

bundles.icml2024()

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent))
OUTPUT_ROOT = BASE_DIR / "results" / "8_plot_multiseed" / "decision_accuracy"

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

def build_seed_plot_df(seed_dir: Path):
    long_path = seed_dir / "6_long.parquet"
    laws_path = seed_dir / "7_filt_laws.pkl"

    df_input = pd.read_parquet(long_path)
    target_flop = df_input["FLOP"].max()
    target_df = df_input[df_input["FLOP"] == target_flop]

    with open(laws_path, "rb") as f:
        input_dict = pickle.load(f)

    bench_to_plot_df = {}
    for bench, bench_dict in input_dict.items():
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
        for max_size in sorted(bench_df["max_model_size"].unique()):
            flop_df = bench_df[bench_df["max_model_size"] == max_size]
            flop_ratio = flop_df["flop_ratio"].mean()
            plot_rows.append(
                {
                    "flop_ratio": flop_ratio,
                    "classic_acc_decisionacc": calculate_decisionacc(gt_rank, rank_mix(flop_df, "classic_acc")),
                    "classic_prob_decisionacc": calculate_decisionacc(gt_rank, rank_mix(flop_df, "classic_prob")),
                    "irt_binary_1pl_decisionacc": calculate_decisionacc(gt_rank, rank_mix(flop_df, "irt_binary_1pl")),
                    "irt_beta_1pl_decisionacc": calculate_decisionacc(gt_rank, rank_mix(flop_df, "irt_beta_1pl")),
                    "irt_binary_2pl_decisionacc": calculate_decisionacc(gt_rank, rank_mix(flop_df, "irt_binary_2pl")),
                    "irt_beta_2pl_decisionacc": calculate_decisionacc(gt_rank, rank_mix(flop_df, "irt_beta_2pl")),
                }
            )
        bench_to_plot_df[bench] = pd.DataFrame(plot_rows).sort_values("flop_ratio").reset_index(drop=True)

    return bench_to_plot_df


def ensure_matching_flop_grid(per_seed_frames, bench):
    flop_grids = [frame["flop_ratio"].to_numpy(dtype=float) for frame in per_seed_frames]
    expected_shape = flop_grids[0].shape
    for idx, current in enumerate(flop_grids[1:], start=1):
        if expected_shape != current.shape:
            raise ValueError(
                f"Inconsistent flop_ratio grid for bench={bench} between seed 0 and seed index {idx}"
            )
    return np.mean(np.stack(flop_grids, axis=0), axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiseed-root", type=Path, default=BASE_DIR / "data_multiseed")
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    multiseed_dirs = sorted(path for path in args.multiseed_root.iterdir() if path.is_dir())[:4]
    seed_sources = [("0", BASE_DIR / "data")] + [
        (str(idx), seed_dir) for idx, seed_dir in enumerate(multiseed_dirs, start=1)
    ]

    seed_payloads = {}
#     seed_payloads = {
#     "1": {
#         "arc_easy": plot_df_for_arc_easy,
#         "arc_challenge": plot_df_for_arc_challenge,
#         ...
#     },
#     "2": {
#         ...
#     },
#     ...
# }
    # plot_df = pd.DataFrame([
    #     {
    #         "flop_ratio": ...,
    #         "classic_acc_decisionacc": ...,
    #         "classic_prob_decisionacc": ...,
    #         "irt_binary_1pl_decisionacc": ...,
    #         "irt_beta_1pl_decisionacc": ...,
    #         "irt_binary_2pl_decisionacc": ...,
    #         "irt_beta_2pl_decisionacc": ...,
    #     },
    #     ...
    # ])

    for seed_name, seed_dir in seed_sources:
        seed_payloads[seed_name] = build_seed_plot_df(seed_dir)

    seed_names = sorted(seed_payloads.keys())
    benches = sorted(seed_payloads[seed_names[0]].keys())
    for seed_name in seed_names[1:]:
        current_benches = set(seed_payloads[seed_name].keys())
        assert current_benches == set(benches), f"Bench set mismatch for seed={seed_name}"

    plot_specs = [
        ("classic_acc_decisionacc", "Traditional Acc", "black", "--"),
        ("classic_prob_decisionacc", r"Traditional $\mathrm{p_{Correct Choice}}$", "black", "-"),
        ("irt_binary_1pl_decisionacc", "Binary-IRT 1PL", "tab:blue", "--"),
        ("irt_beta_1pl_decisionacc", "Beta-IRT 1PL", "tab:blue", "-"),
        ("irt_binary_2pl_decisionacc", "Binary-IRT 2PL", "tab:red", "--"),
        ("irt_beta_2pl_decisionacc", "Beta-IRT 2PL", "tab:red", "-"),
    ]

    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig, axes = plt.subplots(2, 5, figsize=(24, 9), sharex=False, sharey=False)
        axes = axes.flatten()
        legend_handles = None
        legend_labels = None

        for bench_idx, bench in enumerate(benches):
            ax = axes[bench_idx]
            per_seed_frames = [seed_payloads[seed_name][bench] for seed_name in seed_names]
            flop_ratio = ensure_matching_flop_grid(per_seed_frames, bench)
            stacked = np.stack(
                [frame[[spec[0] for spec in plot_specs]].to_numpy(dtype=float) for frame in per_seed_frames],
                axis=0,
            )
            means = stacked.mean(axis=0)
            stds = stacked.std(axis=0)

            for idx, (col_name, label, color, linestyle) in enumerate(plot_specs):
                # lower = np.clip(means[:, idx] - stds[:, idx], 0.0, 1.0)
                # upper = np.clip(means[:, idx] + stds[:, idx], 0.0, 1.0)
                # ax.fill_between(
                #     flop_ratio,
                #     lower,
                #     upper,
                #     color=color,
                #     alpha=0.12,
                #     linewidth=0,
                # )
                ax.plot(
                    flop_ratio,
                    means[:, idx],
                    marker="o",
                    markersize=3.5,
                    alpha=0.9,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                    label=label,
                )

            row_idx, col_idx = divmod(bench_idx, 5)
            ax.set_title(f"{bench}", fontsize=26)
            ax.set_xscale("log")
            ax.tick_params(axis="x", labelsize=20)
            ax.tick_params(axis="y", labelsize=20)
            if col_idx == 0:
                ax.set_ylabel("Decision Accuracy", fontsize=24)
            if row_idx == 1:
                ax.set_xlabel("Max FLOP for Predicting / Target FLOP", fontsize=18)
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=6,
            fontsize=22,
            frameon=True,
            bbox_to_anchor=(0.5, 1.02),
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(
            OUTPUT_ROOT / "decision_accuracy_all_benches.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
