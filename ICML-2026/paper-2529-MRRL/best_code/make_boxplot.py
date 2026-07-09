import argparse
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_varying_part(strings):
    """
    Identifies the unique part of each string by removing the common
    prefix and common suffix among a list of strings.
    """
    if not strings:
        return []
    if len(strings) == 1:
        return strings

    # Find common prefix
    prefix = os.path.commonprefix(strings)

    # Find common suffix
    reversed_strings = [s[::-1] for s in strings]
    suffix = os.path.commonprefix(reversed_strings)[::-1]

    # Strip them out
    # If the strings are identical except for the middle, this returns the middle.
    varying = [s[len(prefix) : len(s) - len(suffix)] for s in strings]
    return varying, prefix + suffix  # prefix.rstrip("_-")


def generate_flexible_boxplot(exp_map, pop_num, output_path, base_title):
    all_data = []

    for label, filepath in exp_map.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df["group_var"] = label  # e.g., 'meanvar', 'poly2', 'poly3'
            all_data.append(df)
        else:
            print(f"Warning: File not found: {filepath}")

    if not all_data:
        print("No data available to plot.")
        return

    df_total = pd.concat(all_data, ignore_index=True)
    df_plot = df_total[df_total["pop_num"] == pop_num].copy()

    if df_plot.empty:
        print(f"No data for pop_num={pop_num}")
        return

    target_order = [
        "Z",
        "hW",
        "hWchV",
        "tslsCondPop",
        "limlCondPop",
        "DoubleML",
        "DFIV",
    ]
    df_plot = df_plot[df_plot["instrument"].isin(target_order)]

    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")

    sns.boxplot(
        data=df_plot,
        x="instrument",
        y="estimate",
        hue="group_var",
        order=target_order,
        palette="viridis",
        showfliers=False,
    )

    plt.axhline(
        y=1.0, color="red", linestyle="--", linewidth=1.5, label="True Effect"
    )

    pop_label = "Combined" if pop_num == -1 else f"Population {pop_num}"
    plt.title(f"Causal Estimates: {base_title} ({pop_label})", fontsize=14)
    plt.ylabel("Estimate Value")
    plt.legend(title="Setup", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Successfully saved boxplot to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_ids", nargs="+", required=True)
    parser.add_argument("--exp_grp", type=str, required=True)
    parser.add_argument("--pop_num", type=int, default=-1)
    parser.add_argument("--use_test_data", action="store_true")
    parser.add_argument("--exclude_sim_ids", type=int, nargs="*", default=[])
    parser.add_argument(
        "--ckpt_strategy",
        type=str,
        default="best",
        choices=["best", "last"],
        help="Whether to load the best-*.ckpt or last.ckpt for evaluation",
    )
    parser.add_argument("--metric_key", type=str, default="val_tot_loss")
    args = parser.parse_args()

    # 1. Automatically find the labels by looking at what varies
    labels, base_name = get_varying_part(args.exp_ids)
    suffix = "outsample" if args.use_test_data else "insample"
    if args.exclude_sim_ids:
        suffix += "_no" + "".join(map(str, args.exclude_sim_ids))

    exp_results_dir = os.path.join("results", args.exp_grp)
    if not os.path.exists(exp_results_dir):
        print(f"exp_dir not found: {exp_results_dir}")
    exp_map = {}
    for label, exp_id in zip(labels, args.exp_ids):
        # Clean up labels like '_meanvar_' or '-poly2-'
        clean_label = label.strip("_-")
        metric_key = args.metric_key.replace("/", "_")
        filepath = os.path.join(
            exp_results_dir,
            f"summary_{exp_id}_{args.ckpt_strategy}_{metric_key}_{suffix}.csv",
        )
        exp_map[clean_label] = filepath

    # 2. Construct out_path
    out_filename = f"{args.ckpt_strategy}_{metric_key}_{suffix}_{base_name}_pop{args.pop_num}.png"
    out_path = os.path.join(exp_results_dir, out_filename)

    generate_flexible_boxplot(exp_map, args.pop_num, out_path, base_name)
