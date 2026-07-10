import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    mpl_config = Path("/tmp/yw676_split_regression_mplconfig")
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ======= Paths =======
RESULTS_CSV_20 = Path("results/final.csv")
RESULTS_CSV_5 = Path("results/final_5.csv")
OUTPUT_PATH = Path("results/images/completion_rate_plot_full.pdf")


# ======= Color palette =======
colors = {
    "claritree": "#2E2585",
    "streed": "#D55E00",
}

labels_map = {
    "claritree": "CLARITree",
    "streed": "STreeD",
}

linestyles = {
    20: "-",
    5: "--",
}


# ======= Global style =======
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 20,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "legend.fontsize": 17,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "axes.linewidth": 1.2,
})


# ======= Time limit =======
TIME_LIMIT = 590
DISPLAY_TIME_LIMIT = 600


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--5",
        dest="include_final_5",
        action="store_true",
        help="Also include results/final_5.csv with number of thresholds = 5.",
    )
    return parser.parse_args()


def load_one_result(csv_path, expected_n_thresholds):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cannot find {csv_path}. Run this script from the project root."
        )

    df = pd.read_csv(csv_path)
    df = df[df["outer"].eq("mean")].copy()

    if "depth" in df.columns:
        df = df[df["depth"] == 4]

    if "n_thresholds" in df.columns:
        df = df[df["n_thresholds"] == expected_n_thresholds]

    df["threshold_setting"] = expected_n_thresholds
    df["source_file"] = csv_path.name
    return df


def load_all_results(include_final_5=False):
    dfs = []

    df20 = load_one_result(RESULTS_CSV_20, expected_n_thresholds=20)
    dfs.append(df20)

    if include_final_5:
        df5 = load_one_result(RESULTS_CSV_5, expected_n_thresholds=5)
        dfs.append(df5)

    return pd.concat(dfs, ignore_index=True)


def extract_training_times(df, method, threshold_setting):
    sub = df[
        (df["method"] == method)
        & (df["threshold_setting"] == threshold_setting)
    ].copy()

    times = sub["train_time_s"].values
    times = times[~np.isnan(times)]
    times = times[times > 0]
    return times


def calculate_completion_rate(times, time_points, time_limit=TIME_LIMIT):
    completion_rates = []
    total_tasks = len(times)

    completed_tasks = times[times <= time_limit]

    for t in time_points:
        if t > time_limit:
            completed = len(completed_tasks)
        else:
            completed = np.sum(completed_tasks <= t)

        rate = (completed / total_tasks) * 100 if total_tasks > 0 else 0
        completion_rates.append(rate)

    return np.array(completion_rates)


def plot_completion_rate(include_final_5=False):
    print("Loading data...")
    df = load_all_results(include_final_5=include_final_5)
    print(f"Loaded {len(df)} total records")

    methods = ["claritree", "streed"]
    threshold_settings = [20]
    if include_final_5:
        threshold_settings.append(5)

    method_times = {}

    for threshold_setting in threshold_settings:
        print(f"\n===== n_thresholds = {threshold_setting} =====")
        for method in methods:
            times = extract_training_times(df, method, threshold_setting)
            method_times[(method, threshold_setting)] = times

            print(f"{method}: {len(times)} training time records")
            if len(times) > 0:
                print(
                    f"  Min: {np.min(times):.4f}s, "
                    f"Max: {np.max(times):.4f}s, "
                    f"Mean: {np.mean(times):.4f}s"
                )

    all_times = np.concatenate([
        times for times in method_times.values() if len(times) > 0
    ])

    min_time = np.min(all_times)
    max_time = min(np.max(all_times), TIME_LIMIT * 1.1)

    log_time_points = np.logspace(
        np.log10(max(min_time, 1e-4)),
        np.log10(max_time),
        1000,
    )

    print(f"\nTime limit: {TIME_LIMIT}s")

    for threshold_setting in threshold_settings:
        for method in methods:
            times = method_times[(method, threshold_setting)]
            if len(times) == 0:
                continue

            completed = np.sum(times <= TIME_LIMIT)
            total = len(times)
            incomplete = total - completed

            print(
                f"{method}, n_thresholds={threshold_setting}: "
                f"{completed}/{total} completed "
                f"({completed / total * 100:.1f}%), "
                f"{incomplete} incomplete (> {TIME_LIMIT}s)"
            )

    fig, ax = plt.subplots(figsize=(10, 7))

    zorder_map = {
        ("claritree", 20): 4,
        ("claritree", 5): 3,
        ("streed", 20): 2,
        ("streed", 5): 1,
    }

    for threshold_setting in threshold_settings:
        for method in methods:
            times = method_times[(method, threshold_setting)]
            if len(times) == 0:
                continue

            completion_rates = calculate_completion_rate(
                times, log_time_points, TIME_LIMIT
            )

            label = (
                f"{labels_map[method]} "
                f"(number of thresholds = {threshold_setting})"
            )

            ax.plot(
                log_time_points,
                completion_rates,
                color=colors[method],
                linestyle=linestyles[threshold_setting],
                linewidth=2.7,
                label=label,
                alpha=1.0,
                zorder=zorder_map.get((method, threshold_setting), 0),
            )

    ax.set_xscale("log")
    ax.set_xlabel("Training Time (s)", fontsize=20, labelpad=8)
    ax.set_ylabel("Completion Rate (%)", fontsize=20, labelpad=8)

    ax.axvline(
        x=TIME_LIMIT,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        alpha=0.8,
        zorder=0,
        label=f"Time Limit ({DISPLAY_TIME_LIMIT}s)",
    )

    ax.set_ylim([0, 105])
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    ax.grid(True, alpha=0.25, linestyle=":", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    ax.legend(loc="upper left", fontsize=16, frameon=False)

    ax.set_title(
        "Training Completion Rate Across All Datasets",
        fontsize=22,
        fontweight="bold",
        pad=12,
    )

    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, bbox_inches="tight", pad_inches=0.1)
    print(f"\n[saved] Figure saved as: {OUTPUT_PATH}")


if __name__ == "__main__":
    args = parse_args()
    plot_completion_rate(include_final_5=args.include_final_5)