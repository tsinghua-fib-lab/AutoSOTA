import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ======= Paths =======
RESULTS_CSV = "results/final.csv"
OUTPUT_PATH = "results/images/teaser_new.pdf"

# ======= Dataset files =======
datasets = [
    "airfoil",
    "california_housing",
]
SMALL_MEDIUM_DATASETS = [
    "airfoil",
    "auction",
    "auto_mpg",
    "energe_c",
    "energe_h",
    "insurance",
    "optical_net",
    "real_estate",
    "servo",
    "synch",
    "yacht",
]
LARGE_DATASETS = [
    "california_housing",
    "seoul_bike",
    "temperature_max",
    "temperature_min",
    "walmart",
]
LARGE_FULL_TIMEOUT_DATASETS = {
    "california_housing",
    "temperature_max",
    "temperature_min",
}
SUMMARY_DATASETS = SMALL_MEDIUM_DATASETS + LARGE_DATASETS
SUMMARY_LABELS = {
    "airfoil": "Airfoil",
    "auction": "Auction",
    "auto_mpg": "Auto MPG",
    "energe_c": "Energy-C",
    "energe_h": "Energy-H",
    "insurance": "Insurance",
    "optical_net": "Optical Net",
    "real_estate": "Real Estate",
    "servo": "Servo",
    "synch": "Synch",
    "yacht": "Yacht",
    "california_housing": "California",
    "seoul_bike": "Seoul Bike",
    "temperature_max": "Temp Max",
    "temperature_min": "Temp Min",
    "walmart": "Walmart",
}

# ======= final.csv layout (same source as plot_linear.py) =======
KEY_COLS = [
    "dataset",
    "method",
    "depth",
    "lambda",
    "kappa",
    "cv_folds",
    "n_thresholds",
    "thresholds_strategy",
    "min_leaf_node_size",
    "cost_complexity",
    "pilot_grid_index",
    "ridge_penalty",
    "lasso_penalty",
]
METRIC_COLS = [
    "val_r2",
    "train_r2",
    "test_r2",
    "train_time_s",
    "n_leaves",
]
PARAM_COL = {
    "claritree": "lambda",
    "greedy": "lambda",
    "streed": "cost_complexity",
    "streed_s": "cost_complexity",
    "guide": "cost_complexity",
    "pilot": "cost_complexity",
    "m5": "cost_complexity",
}
METHOD_RENAME = {
    "claritree": "cholickety",
    "greedy": "cholesky",
    "streed_s": "streed_sl",
}
COLUMN_RENAME = {
    "train_r2_mean": "r2_train_mean",
    "train_r2_std": "r2_train_std",
    "test_r2_mean": "r2_test_mean",
    "test_r2_std": "r2_test_std",
    "n_leaves_mean": "leaves_mean",
    "n_leaves_std": "leaves_std",
}

# ======= Color palette (same as plot_linear.py) =======
colors = {
    "cholickety": "#2E2585",  # deep indigo-blue (primary, dominant)
    "streed":     "#D55E00",  # vermillion (Okabe–Ito, colorblind-safe)
    "streed_sl":  "#E69F00",  # muted orange/yellow (lighter + less saturated)
    "cholesky":   "#7F7F7F",  # medium gray (visible but clearly weak)
    "pilot":      "#56B4E9",  # sky blue (Okabe-Ito, colorblind-safe)
    "guide":      "#CC79A7",  # purple/magenta (Okabe–Ito)
    "m5":         "#009E73",  # teal-green (M5, colorblind-safe)
}
gap_colors = {
    "positive": "#D55E00",  # StreeD higher R²
    "negative": "#2E2585",  # CLARITree higher R²
}
speedup_color = "#E69F00"

# ======= Z-order for overlap control =======
zorder_map = {
    "cholickety": 5,  # Top (ours)
    "streed": 4,      # Second
    "streed_sl": 3,   # Third
    "guide": 2,       # Fourth
    "cholesky": 1,    # Bottom (greedy)
    "pilot": 0.5,     # Between greedy and m5
    "m5": 0,          # Lowest
}

# ======= Method name mapping =======
labels_map = {
    "cholickety": "CLARITree (Ours)",
    "cholesky": "CholeskyTree",
    "streed": "STreeD (Optimal)",
    "streed_sl": "STreeD-S",
    "pilot": "PILOT",
    "guide": "GUIDE",
    "m5": "M5",
}

# ======= Global style =======
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "legend.fontsize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.linewidth": 1.2,
})

# ======= Time limit =======
TIME_LIMIT = 590
EPS = 1e-12
depth_value = 4  # Assuming depth=4 for all datasets
n_thresholds_value = 20
PANEL_TITLE_SIZE = 17
PANEL_XLABEL_SIZE = 15
PANEL_YLABEL_SIZE = 16
PANEL_TICK_SIZE = 14
SUMMARY_DATASET_TICK_SIZE = 12
SUMMARY_VALUE_SIZE = 9.5


def large_full_streed_root():
    return os.path.join(
        "results",
        "baseline",
        "streed",
        f"linear_regression_tree_depth{depth_value}_threshold_{n_thresholds_value}",
        "large_full",
    )


def summarize_large_full_rows(rows):
    if "outer" in rows.columns:
        rows = rows[~rows["outer"].isin(["mean", "std", "best_by_mean_val_r2"])].copy()

    keys = [col for col in KEY_COLS if col in rows.columns]
    metrics = [col for col in METRIC_COLS if col in rows.columns]
    if not keys or not metrics or rows.empty:
        return pd.DataFrame()

    mean_df = rows.groupby(keys, as_index=False, dropna=False)[metrics].mean()
    mean_df["outer"] = "mean"

    std_df = rows.groupby(keys, as_index=False, dropna=False)[metrics].std()
    std_df["outer"] = "std"

    if "val_r2" not in mean_df.columns:
        return pd.concat([mean_df, std_df], ignore_index=True, sort=False)

    best_df = (
        mean_df.sort_values("val_r2", ascending=False)
        .groupby(["dataset", "method"], as_index=False, dropna=False)
        .head(1)
        .copy()
    )
    best_df["outer"] = "best_by_mean_val_r2"
    return pd.concat([mean_df, std_df, best_df], ignore_index=True, sort=False)


def load_large_full_streed_summary_rows():
    parts = []
    root = large_full_streed_root()

    for dataset in LARGE_DATASETS:
        dataset_dir = os.path.join(root, dataset)
        merged_path = os.path.join(dataset_dir, f"{dataset}_outer0-4_d{depth_value}.csv")

        if os.path.exists(merged_path):
            rows = pd.read_csv(merged_path)
        else:
            raw_paths = [
                os.path.join(dataset_dir, f"{dataset}_outer{outer}_d{depth_value}.csv")
                for outer in range(5)
            ]
            missing = [path for path in raw_paths if not os.path.exists(path)]
            if missing:
                print(f"[warn] Missing large_full StreeD rows for {dataset}; using {RESULTS_CSV}")
                continue
            rows = pd.concat([pd.read_csv(path) for path in raw_paths], ignore_index=True, sort=False)

        if "method" in rows.columns:
            rows = rows[rows["method"] == "streed"].copy()

        if "outer" in rows.columns:
            summary_rows = rows[rows["outer"].isin(["std", "best_by_mean_val_r2"])].copy()
        else:
            summary_rows = pd.DataFrame()
        needs_summary = summary_rows.empty or summary_rows["outer"].nunique() < 2
        if needs_summary:
            summary_rows = summarize_large_full_rows(rows)
            if "outer" in summary_rows.columns:
                summary_rows = summary_rows[summary_rows["outer"].isin(["std", "best_by_mean_val_r2"])].copy()

        if summary_rows.empty:
            print(f"[warn] Could not summarize large_full StreeD rows for {dataset}; using {RESULTS_CSV}")
            continue
        parts.append(summary_rows)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True, sort=False)


def replace_large_summary_streed_rows(df):
    large_full = load_large_full_streed_summary_rows()
    if large_full.empty:
        return df

    datasets = sorted(large_full["dataset"].dropna().unique())
    mask = df["dataset"].isin(datasets) & df["method"].eq("streed")
    print(f"[info] Using large_full StreeD summary rows for: {', '.join(datasets)}")
    return pd.concat([df.loc[~mask], large_full], ignore_index=True, sort=False)


def load_table_best_rows():
    """Load final rows selected by mean validation R², matching table_linear.py."""
    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(f"Cannot find {RESULTS_CSV}. Run this script from the project root.")

    df = pd.read_csv(RESULTS_CSV)
    df = replace_large_summary_streed_rows(df)
    df = df[df["dataset"].isin(SUMMARY_DATASETS)].copy()
    if "depth" in df.columns:
        df = df[df["depth"] == depth_value]
    if "n_thresholds" in df.columns:
        df = df[df["n_thresholds"] == n_thresholds_value]

    keys = [col for col in KEY_COLS if col in df.columns]
    metrics = [col for col in METRIC_COLS if col in df.columns]
    best = df[df["outer"] == "best_by_mean_val_r2"][keys + metrics].copy()
    std = df[df["outer"] == "std"][keys + metrics].copy()
    best = best.rename(columns={col: f"{col}_mean" for col in metrics})
    std = std.rename(columns={col: f"{col}_std" for col in metrics})
    rows = best.merge(std, on=keys, how="left")

    numeric_cols = [
        "test_r2_mean",
        "test_r2_std",
        "train_time_s_mean",
        "train_time_s_std",
        "val_r2_mean",
        "val_r2_std",
    ]
    for col in numeric_cols:
        if col in rows.columns:
            rows[col] = pd.to_numeric(rows[col], errors="coerce")
    return rows


def build_summary(rows):
    """CLARITree summary against StreeD using table-selected rows."""
    summary = []
    for dataset in SUMMARY_DATASETS:
        dataset_rows = rows[rows["dataset"] == dataset]
        ours = dataset_rows[dataset_rows["method"] == "claritree"]
        streed = dataset_rows[dataset_rows["method"] == "streed"]
        if ours.empty or streed.empty:
            continue
        ours = ours.iloc[0]
        streed = streed.iloc[0]
        summary.append({
            "dataset": dataset,
            "label": SUMMARY_LABELS.get(dataset, dataset.replace("_", " ").title()),
            "accuracy_gap": streed["test_r2_mean"] - ours["test_r2_mean"],
            "compute_gain": streed["train_time_s_mean"] / max(ours["train_time_s_mean"], EPS),
            "streed_timeout": streed["train_time_s_mean"] > TIME_LIMIT,
        })
    return pd.DataFrame(summary)


def style_summary_axis(ax):
    ax.set_facecolor("white")
    ax.grid(True, axis="x", alpha=0.25, linestyle=":", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d1d5db")
    ax.spines["bottom"].set_color("#d1d5db")
    ax.tick_params(axis="both", labelsize=PANEL_TICK_SIZE)


def format_accuracy_gap(value):
    rounded = round(float(value), 2)
    if rounded == 0:
        return "0.00"
    return f"{rounded:+.2f}"


def plot_accuracy_gap(ax, summary, dataset_group, title, gap_limit):
    rows = summary.set_index("dataset").loc[dataset_group].reset_index()
    y = np.arange(len(rows))
    values = rows["accuracy_gap"].to_numpy(dtype=float)
    bar_colors = [gap_colors["positive"] if value >= 0 else gap_colors["negative"] for value in values]
    ax.barh(y, values, color=bar_colors, alpha=0.90, zorder=2)
    ax.axvline(0, color="#333333", linewidth=1.0, alpha=0.65, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(rows["label"], fontsize=SUMMARY_DATASET_TICK_SIZE)
    ax.invert_yaxis()
    ax.set_xlim(-gap_limit, gap_limit)
    ax.set_xlabel("Optimal R² − CLARITree R²", fontsize=PANEL_XLABEL_SIZE, labelpad=6)
    ax.set_title(title, fontsize=PANEL_TITLE_SIZE, fontweight="bold", pad=8)
    style_summary_axis(ax)
    offset = gap_limit * 0.04
    for yy, value in zip(y, values):
        if not np.isfinite(value):
            continue
        if abs(value) > gap_limit * 0.18:
            ha = "center"
            x = value / 2
            text_color = "white"
        else:
            ha = "left" if value >= 0 else "right"
            x = value + offset if value >= 0 else value - offset
            text_color = "#2b2b2b"
        ax.text(x, yy, format_accuracy_gap(value), va="center", ha=ha, fontsize=SUMMARY_VALUE_SIZE,
                color=text_color, fontweight="bold")


def plot_compute_gain(
    ax,
    summary,
    dataset_group,
    title,
    gain_limit,
    lower_bound=False,
    lower_bound_datasets=None,
):
    lower_bound_datasets = set() if lower_bound_datasets is None else set(lower_bound_datasets)
    rows = summary.set_index("dataset").loc[dataset_group].reset_index()
    y = np.arange(len(rows))
    values = rows["compute_gain"].to_numpy(dtype=float)
    ax.barh(y, values, color=speedup_color, alpha=0.92, zorder=2)
    ax.axvline(1, color="#333333", linewidth=1.0, alpha=0.55, zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(1, gain_limit)
    ax.set_yticks(y)
    ax.set_yticklabels(rows["label"], fontsize=SUMMARY_DATASET_TICK_SIZE)
    ax.invert_yaxis()
    xlabel = "Optimal / CLARITree time"
    if lower_bound:
        xlabel += " (timeout)"
    ax.set_xlabel(xlabel, fontsize=PANEL_XLABEL_SIZE, labelpad=6)
    ax.set_title(title, fontsize=PANEL_TITLE_SIZE, fontweight="bold", pad=8)
    style_summary_axis(ax)
    for yy, value, dataset in zip(y, values, rows["dataset"]):
        if not np.isfinite(value):
            continue
        is_lower_bound = lower_bound or dataset in lower_bound_datasets
        label = f">{value:.0f}×" if is_lower_bound else f"{value:.0f}×"
        ax.text(value * 1.10, yy, label, va="center", ha="left",
                fontsize=SUMMARY_VALUE_SIZE, color="#1f1f1f", fontweight="bold")

# ======= Load and process data for each dataset =======
def load_and_process_dataset(dataset_name):
    """Load CSV and merge mean/std data."""
    csv_path = RESULTS_CSV
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found, skipping...")
        return None

    df = pd.read_csv(csv_path)
    df = df[df["dataset"] == dataset_name].copy()
    if "depth" in df.columns:
        df = df[df["depth"] == depth_value]
    if "n_thresholds" in df.columns:
        df = df[df["n_thresholds"] == n_thresholds_value]
    if df.empty:
        print(f"Warning: no rows found for {dataset_name}, skipping...")
        return None

    keys = [col for col in KEY_COLS if col in df.columns]
    metrics = [col for col in METRIC_COLS if col in df.columns]
    mean_df = df[df['outer'] == 'mean'][keys + metrics].copy()
    std_df = df[df['outer'] == 'std'][keys + metrics].copy()
    if mean_df.empty or std_df.empty:
        print(f"Warning: mean/std rows missing for {dataset_name}, skipping...")
        return None

    merged = mean_df.merge(std_df, on=keys, suffixes=('_mean','_std'))
    merged["complexity"] = pd.NA
    for method, col in PARAM_COL.items():
        if col in merged.columns:
            mask = merged["method"] == method
            merged.loc[mask, "complexity"] = merged.loc[mask, col]

    selected = []
    for _, group in merged.dropna(subset=["complexity"]).groupby(
        ["dataset", "method", "depth", "n_thresholds", "complexity"], dropna=False
    ):
        group = group[pd.to_numeric(group["val_r2_mean"], errors="coerce").notna()]
        if group.empty:
            continue
        order = group.assign(
            _std=pd.to_numeric(group["val_r2_std"], errors="coerce").fillna(float("inf"))
        )
        selected.append(order.sort_values(["val_r2_mean", "_std"], ascending=[False, True]).iloc[0])
    if not selected:
        print(f"Warning: no valid complexity-selected rows for {dataset_name}, skipping...")
        return None

    merged = pd.DataFrame(selected).drop(columns=["_std"], errors="ignore")
    merged["method"] = merged["method"].replace(METHOD_RENAME)
    merged = merged.rename(columns=COLUMN_RENAME)
    merged = merged[merged["method"].isin(labels_map)]

    return merged

# ======= Create 2x4 subplot layout =======
# Left two columns keep the original dataset panels; right two columns summarize all datasets.
fig = plt.figure(figsize=(20, 8.5))
gs = fig.add_gridspec(2, 4, hspace=0.30, wspace=0.30,
                      height_ratios=[1, 1], width_ratios=[1.1, 1.1, 0.95, 0.95])

# Subtle background colors for dataset separation
dataset_bg_colors = ['#fafbfc', '#f8f9fa', '#fafbfc', '#f8f9fa']
dataset_border_colors = ['#e1e4e8', '#d1d5db', '#e1e4e8', '#d1d5db']

# Store axes for each dataset to enable y-axis sharing
dataset_axes = {}

# ======= Process and plot each dataset =======
for idx, dataset_name in enumerate(datasets):
    merged = load_and_process_dataset(dataset_name)
    if merged is None:
        continue

    # Check time-limited methods for this dataset
    time_limited_methods = set(
        merged.loc[merged["train_time_s_mean"] > TIME_LIMIT, "method"].unique()
    )

    # Dataset name formatting
    dataset_display = dataset_name.replace("_", " ").title()
    # Fix typo: arifoil -> airfoil
    if dataset_display == "Arifoil":
        dataset_display = "Airfoil"

    # Top subplot: Test R² vs Time (column idx, row 0)
    ax_top = fig.add_subplot(gs[0, idx])
    ax_top.set_facecolor(dataset_bg_colors[idx])

    # Plot with elegant style - no lines for time plot, just markers with error bars
    for method, group in merged.groupby('method'):
        group_plot = group[group['r2_test_mean'] >= 0].sort_values('train_time_s_mean')
        if len(group_plot) == 0:
            continue

        # For time-limited methods, use dashed line; for others, no line (just markers)
        # Our method (cholickety) should be fully opaque (alpha=1.0)
        alpha_val = 1.0 if method == "cholickety" else (0.85 if method in time_limited_methods else 0.9)
        line_alpha = 1.0 if method == "cholickety" else 0.4

        if method in time_limited_methods:
            # Time-limited: show with dashed line to indicate timeout, hollow markers
            ax_top.plot(group_plot['train_time_s_mean'], group_plot['r2_test_mean'],
                       linestyle='--', color=colors[method], alpha=line_alpha, linewidth=1.5,
                       zorder=zorder_map[method]-0.5)
            ax_top.errorbar(group_plot['train_time_s_mean'], group_plot['r2_test_mean'],
                          xerr=group_plot['train_time_s_std'], yerr=group_plot['r2_test_std'],
                          fmt='o', color=colors[method], linestyle='None',
                          markersize=7 if method=="cholickety" else 5.5,
                          markerfacecolor='none', markeredgewidth=1.5,
                          capsize=2.5, alpha=alpha_val, capthick=1.5,
                          elinewidth=1.5, label=labels_map[method] if idx == 0 else "",
                          zorder=zorder_map[method])
        else:
            # Normal methods: just markers, no connecting lines (more elegant), solid markers
            ax_top.errorbar(group_plot['train_time_s_mean'], group_plot['r2_test_mean'],
                          xerr=group_plot['train_time_s_std'], yerr=group_plot['r2_test_std'],
                          fmt='o', color=colors[method], linestyle='None',
                          markersize=8 if method=="cholickety" else 6,
                          capsize=3, alpha=alpha_val, capthick=1.5,
                          elinewidth=1.8, label=labels_map[method] if idx == 0 else "",
                          zorder=zorder_map[method])

    ax_top.set_xscale('log')
    # Add vertical reference line at time limit for visual emphasis
    ax_top.axvline(x=TIME_LIMIT, color='gray', linestyle=':', linewidth=1.5,
                   alpha=0.8, zorder=0, label='Time Limit (600s)' if idx == 0 else '')
    ax_top.set_xlabel("Training Time (s)", fontsize=15, labelpad=6)
    ax_top.set_ylabel("Test R²", fontsize=16, labelpad=8, fontweight='medium')

    # Only show dataset label and (a), (b), (c), (d) on top subplot
    ax_top.set_title(f"({chr(97+idx)}) {dataset_display}", fontsize=17, fontweight="bold", pad=8)
    ax_top.grid(True, alpha=0.25, linestyle=':', linewidth=0.8, zorder=0)
    # Remove top and right spines for cleaner look
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)

    # Elegant border (only left and bottom)
    ax_top.spines['left'].set_linewidth(1.2)
    ax_top.spines['left'].set_color(dataset_border_colors[idx])
    ax_top.spines['bottom'].set_linewidth(1.2)
    ax_top.spines['bottom'].set_color(dataset_border_colors[idx])

    # Bottom subplot: Test R² vs Leaves (column idx, row 1) - share y-axis with top
    ax_bottom = fig.add_subplot(gs[1, idx], sharey=ax_top)
    ax_bottom.set_facecolor(dataset_bg_colors[idx])

    for method, group in merged.groupby('method'):
        group_plot = group[group['r2_test_mean'] >= 0].sort_values('leaves_mean')
        if len(group_plot) == 0:
            continue

        # For leaves plot, use lines to show trend (more meaningful)
        # Our method (cholickety) should be fully opaque (alpha=1.0)
        linestyle = '--' if method in time_limited_methods else '-'
        line_alpha = 1.0 if method == "cholickety" else (0.5 if method in time_limited_methods else 0.6)
        marker_alpha = 1.0 if method == "cholickety" else 0.9

        ax_bottom.plot(group_plot['leaves_mean'], group_plot['r2_test_mean'],
                      linestyle=linestyle, color=colors[method],
                      alpha=line_alpha,
                      linewidth=2.2 if method=="cholickety" else 1.8,
                      zorder=zorder_map[method]-0.3)
        # Use hollow markers only for time-limited methods
        if method in time_limited_methods:
            ax_bottom.errorbar(group_plot['leaves_mean'], group_plot['r2_test_mean'],
                              xerr=group_plot['leaves_std'], yerr=group_plot['r2_test_std'],
                              fmt='o', color=colors[method], linestyle='None',
                              markersize=8 if method=="cholickety" else 6,
                              markerfacecolor='none', markeredgewidth=1.8,
                              capsize=3, alpha=marker_alpha, capthick=1.5,
                              elinewidth=1.8, zorder=zorder_map[method])
        else:
            ax_bottom.errorbar(group_plot['leaves_mean'], group_plot['r2_test_mean'],
                              xerr=group_plot['leaves_std'], yerr=group_plot['r2_test_std'],
                              fmt='o', color=colors[method], linestyle='None',
                              markersize=8 if method=="cholickety" else 6,
                              capsize=3, alpha=marker_alpha, capthick=1.5,
                              elinewidth=1.8, zorder=zorder_map[method])

    ax_bottom.set_xlabel("Number of Leaves", fontsize=15, labelpad=6)
    ax_bottom.set_xticks([4, 8, 12, 16])
    ax_bottom.set_ylabel("Test R²", fontsize=16, labelpad=8, fontweight='medium')

    ax_bottom.grid(True, alpha=0.25, linestyle=':', linewidth=0.8, zorder=0)
    # Remove top and right spines for cleaner look
    ax_bottom.spines['top'].set_visible(False)
    ax_bottom.spines['right'].set_visible(False)

    # Elegant border (only left and bottom)
    ax_bottom.spines['left'].set_linewidth(1.2)
    ax_bottom.spines['left'].set_color(dataset_border_colors[idx])
    ax_bottom.spines['bottom'].set_linewidth(1.2)
    ax_bottom.spines['bottom'].set_color(dataset_border_colors[idx])

    # Set ylim for California_housing number of leaves plot
    if dataset_name == "california_housing":
        ax_bottom.set_ylim(0.5, 0.8)

    # Store for potential further customization
    dataset_axes[idx] = (ax_top, ax_bottom)

# ======= Summary panels: table-selected CLARITree vs StreeD =======
summary = build_summary(load_table_best_rows())
gap_limit = max(0.015, float(summary["accuracy_gap"].abs().max()) * 1.18)
gain_limit = max(10.0, float(summary["compute_gain"].max()) * 2.5)

ax_gap_small = fig.add_subplot(gs[0, 2])
plot_accuracy_gap(
    ax_gap_small,
    summary,
    SMALL_MEDIUM_DATASETS,
    "(c) Accuracy Gap: Small/Medium",
    gap_limit,
)

ax_gap_large = fig.add_subplot(gs[1, 2], sharex=ax_gap_small)
plot_accuracy_gap(
    ax_gap_large,
    summary,
    LARGE_DATASETS,
    "(e) Accuracy Gap: Large",
    gap_limit,
)

ax_gain_small = fig.add_subplot(gs[0, 3])
plot_compute_gain(
    ax_gain_small,
    summary,
    SMALL_MEDIUM_DATASETS,
    "(d) Speedup: Small/Medium",
    gain_limit,
)

ax_gain_large = fig.add_subplot(gs[1, 3], sharex=ax_gain_small)
plot_compute_gain(
    ax_gain_large,
    summary,
    LARGE_DATASETS,
    "(f) Speedup: Large",
    gain_limit,
    lower_bound_datasets=LARGE_FULL_TIMEOUT_DATASETS,
)

# ======= Shared legend =======
# Get handles from first subplot
handles, labels = fig.axes[0].get_legend_handles_labels()

# Define desired legend order (matching z-order)
legend_order = ["cholickety", "streed", "guide", "streed_sl", "cholesky", "pilot", "m5"]

# Create mapping from label to method name
label_to_method = {v: k for k, v in labels_map.items()}

# Reorder handles and labels according to desired order
ordered_handles = []
ordered_labels = []
for method in legend_order:
    method_label = labels_map[method]
    if method_label in labels:
        idx = labels.index(method_label)
        ordered_handles.append(handles[idx])
        ordered_labels.append(labels[idx])

legend = fig.legend(
    ordered_handles, ordered_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.975),
    ncol=7, fontsize=15, frameon=False,
    columnspacing=1.5, handletextpad=0.6, handlelength=1.2
)
for text in legend.get_texts():
    if text.get_text() == "CLARITree (Ours)":
        text.set_fontweight("bold")

# ======= Main title =======
reg_type = "Linear Regression Trees"
title_text = f"Test R² Performance — {reg_type} (Depth = {depth_value}, Thresholds = 20)"
fig.suptitle(title_text, fontsize=22, fontweight="bold", y=0.995)

# ======= Layout adjustment =======
plt.subplots_adjust(top=0.88, bottom=0.10, left=0.06, right=0.98, hspace=0.30, wspace=0.30)

# ======= Save as PDF =======
output_path = OUTPUT_PATH
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
print(f"[saved] Figure saved as: {output_path}")

plt.show()
