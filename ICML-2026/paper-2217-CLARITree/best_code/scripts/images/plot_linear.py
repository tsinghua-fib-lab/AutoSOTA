from __future__ import annotations

import argparse
import os
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    mpl_config = Path("/tmp/yw676_split_regression_mplconfig")
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FINAL_CSV = Path("results/final.csv")
OPENML_FINAL_CSV = Path("results/final_openml.csv")
OUT_ROOT = Path("results/images")
OPENML_OUT_ROOT = Path("results/images_openml")
TIME_LIMIT = 590.0
TIME_MARK = 600.0

METHODS = ["claritree", "streed", "streed_s", "guide", "greedy", "pilot", "m5"]
LABELS = {
    "claritree": "CLARITree (Ours)",
    "streed": "STreeD",
    "streed_s": "STreeD-S",
    "guide": "GUIDE",
    "greedy": "Greedy",
    "pilot": "PILOT",
    "m5": "M5",
}

# Same palette as the reference plotting script, mapped to the current method names.
COLORS = {
    "claritree": "#2E2585",
    "streed": "#D55E00",
    "streed_s": "#E69F00",
    "guide": "#CC79A7",
    "greedy": "#7F7F7F",
    "pilot": "#56B4E9",
    "m5": "#009E73",
}
ALPHA = {
    "claritree": 1.0,
    "streed": 0.7,
    "streed_s": 0.7,
    "guide": 0.6,
    "greedy": 0.8,
    "pilot": 0.7,
    "m5": 0.65,
}
ZORDER = {
    "claritree": 5,
    "streed": 4,
    "streed_s": 3,
    "guide": 2,
    "greedy": 1,
    "pilot": 0.5,
    "m5": 0,
}
PARAM_COL = {
    "claritree": "lambda",
    "greedy": "lambda",
    "streed": "cost_complexity",
    "streed_s": "cost_complexity",
    "guide": "cost_complexity",
    "pilot": "cost_complexity",
    "m5": "cost_complexity",
}

KEY_COLS = [
    "dataset",
    "method",
    "depth",
    "lambda",
    "kappa",
    "cv_folds",
    "n_thresholds",
    "threshold_label",
    "thresholds_strategy",
    "min_leaf_node_size",
    "cost_complexity",
    "pilot_grid_index",
    "ridge_penalty",
    "lasso_penalty",
]
METRICS = [
    "val_r2",
    "train_r2",
    "test_r2",
    "train_time_s",
    "n_leaves",
]


def default_final_csv(openml: bool, threshold_label: str | None = None) -> Path:
    if threshold_label is None:
        return OPENML_FINAL_CSV if openml else FINAL_CSV
    base = "final_openml" if openml else "final"
    return Path("results") / f"{base}_{threshold_label}.csv"


def threshold_group_column(rows: pd.DataFrame) -> str:
    if "threshold_label" in rows.columns and rows["threshold_label"].notna().any():
        return "threshold_label"
    return "n_thresholds"


def threshold_display(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value)
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return text
    if np.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return text


def threshold_path_token(value: object) -> str:
    label = threshold_display(value)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label)


def display_dataset(name: str) -> str:
    name = Path(name).name
    special = {
        "energe_c": "Energy (Cooling)",
        "energe_h": "Energy (Heating)",
        "temperature_min": "Temperature (Min)",
        "temperature_max": "Temperature (Max)",
    }
    return special.get(name, name.replace("_", " ").title())


def load_mean_std(path: Path, threshold_label: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if threshold_label is not None:
        if "threshold_label" not in df.columns:
            df["threshold_label"] = threshold_label
        else:
            df["threshold_label"] = df["threshold_label"].fillna(threshold_label)
    keys = [c for c in KEY_COLS if c in df.columns]
    metrics = [c for c in METRICS if c in df.columns]

    mean = df[df["outer"].eq("mean")][keys + metrics].copy()
    std = df[df["outer"].eq("std")][keys + metrics].copy()
    mean = mean.rename(columns={c: f"{c}_mean" for c in metrics})
    std = std.rename(columns={c: f"{c}_std" for c in metrics})

    merged = mean.merge(std, on=keys, how="left")
    merged["complexity"] = np.nan
    for method, col in PARAM_COL.items():
        if col in merged.columns:
            mask = merged["method"].eq(method)
            merged.loc[mask, "complexity"] = merged.loc[mask, col]
    return merged


def best_per_complexity(rows: pd.DataFrame) -> pd.DataFrame:
    selected = []
    threshold_col = threshold_group_column(rows)
    for _, group in rows.dropna(subset=["complexity"]).groupby(
        ["dataset", "method", "depth", threshold_col, "complexity"], dropna=False
    ):
        group = group[np.isfinite(group["val_r2_mean"])]
        if group.empty:
            continue
        order = group.assign(_std=group["val_r2_std"].fillna(np.inf))
        selected.append(order.sort_values(["val_r2_mean", "_std"], ascending=[False, True]).iloc[0])
    return pd.DataFrame(selected)


def err_values(frame: pd.DataFrame, col: str) -> np.ndarray | None:
    if col not in frame or not frame[col].notna().any():
        return None
    return pd.to_numeric(frame[col], errors="coerce").fillna(0).clip(lower=0).to_numpy()


def plot_points(
    ax: plt.Axes,
    rows: pd.DataFrame,
    method: str,
    x_col: str,
    y_col: str,
    *,
    connect: bool,
    show_label: bool,
) -> None:
    yerr_col = y_col.replace("_mean", "_std")
    xerr_col = x_col.replace("_mean", "_std")

    rows = rows.copy()
    for col in [x_col, y_col, yerr_col, xerr_col, "train_time_s_mean"]:
        if col in rows:
            rows[col] = pd.to_numeric(rows[col], errors="coerce")
    rows = rows.dropna(subset=[x_col, y_col])
    rows = rows[np.isfinite(rows[x_col]) & np.isfinite(rows[y_col])]
    if x_col == "train_time_s_mean":
        rows = rows[rows[x_col] > 0]
    rows = rows[(rows[y_col] >= 0) & (rows[y_col] <= 1)]
    if rows.empty:
        return

    rows = rows.sort_values(x_col)
    color = COLORS[method]
    line_alpha = 0.95 if method == "claritree" else 0.55

    if connect and len(rows) > 1:
        ax.plot(
            rows[x_col],
            rows[y_col],
            color=color,
            linewidth=2.0 if method == "claritree" else 1.6,
            alpha=line_alpha,
            zorder=ZORDER[method],
        )

    normal = rows[rows["train_time_s_mean"] <= TIME_LIMIT]
    timeout = rows[rows["train_time_s_mean"] > TIME_LIMIT]

    if not normal.empty:
        ax.errorbar(
            normal[x_col],
            normal[y_col],
            xerr=err_values(normal, xerr_col),
            yerr=err_values(normal, yerr_col),
            fmt="o",
            color=color,
            linestyle="None",
            markersize=8 if method == "claritree" else 6,
            capsize=3,
            alpha=ALPHA[method],
            label=LABELS[method] if show_label else None,
            zorder=ZORDER[method],
        )

    if not timeout.empty:
        ax.errorbar(
            timeout[x_col],
            timeout[y_col],
            xerr=err_values(timeout, xerr_col),
            yerr=err_values(timeout, yerr_col),
            fmt="o",
            color=color,
            markerfacecolor="none",
            markeredgewidth=1.8,
            linestyle="None",
            markersize=8 if method == "claritree" else 6,
            capsize=3,
            alpha=ALPHA[method],
            label=LABELS[method] if show_label and normal.empty else None,
            zorder=ZORDER[method],
        )


def make_dataset_plot(rows: pd.DataFrame, out_dir: Path, openml: bool = False) -> None:
    dataset = rows["dataset"].iloc[0]
    depth = int(rows["depth"].iloc[0])
    threshold_col = threshold_group_column(rows)
    threshold = threshold_display(rows[threshold_col].iloc[0])

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 15,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    panels = [
        (axes[0, 0], "train_time_s_mean", "train_r2_mean", "Train R^2 vs Training Time", False, True),
        (axes[0, 1], "train_time_s_mean", "test_r2_mean", "Test R^2 vs Training Time", False, False),
        (axes[1, 0], "n_leaves_mean", "train_r2_mean", "Train R^2 vs Leaves", True, False),
        (axes[1, 1], "n_leaves_mean", "test_r2_mean", "Test R^2 vs Leaves", True, False),
    ]

    for ax, x_col, y_col, title, connect, show_label in panels:
        y_name = "Train R^2" if y_col == "train_r2_mean" else "Test R^2"
        x_name = "Training Time (s, log scale)" if x_col == "train_time_s_mean" else "Number of leaves"
        for method in METHODS:
            method_rows = rows[rows["method"].eq(method)]
            if not method_rows.empty:
                plot_points(
                    ax,
                    method_rows,
                    method,
                    x_col,
                    y_col,
                    connect=connect,
                    show_label=show_label,
                )
        if x_col == "train_time_s_mean":
            ax.set_xscale("log")
            ax.axvline(TIME_MARK, color="gray", linestyle=":", linewidth=1.5, alpha=0.8, zorder=0)
        ax.set_title(title)
        ax.set_xlabel(x_name)
        ax.set_ylabel(f"{y_name} (mean +/- std)")
        ax.set_ylim(-0.02, 1.04)
        ax.grid(True, alpha=0.25, linestyle=":", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        legend = fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),
            ncol=4,
            frameon=False,
        )
        for text in legend.get_texts():
            if text.get_text() == "CLARITree (Ours)":
                text.set_fontweight("bold")
    fig.suptitle(
        f"{display_dataset(dataset)}: linear regression tree depth {depth}, threshold {threshold}",
        fontsize=30,
        fontweight="bold",
        y=0.995,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.82])
    image_name = "image_openml.png" if openml else "image.png"
    fig.savefig(out_dir / image_name, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--openml", action="store_true", help="Read results/final_openml.csv and write OpenML figures.")
    parser.add_argument("--5", dest="threshold_5", action="store_true", help="Read/write threshold=5 figures.")
    parser.add_argument("--full", action="store_true", help="Read/write threshold=full figures.")
    args = parser.parse_args()

    if args.full and args.threshold_5:
        parser.error("--full and --5 cannot be used together")
    threshold_label = "full" if args.full else "5" if args.threshold_5 else None
    final_csv = args.final or default_final_csv(args.openml, threshold_label)
    out_root = args.out_root or (OPENML_OUT_ROOT if args.openml else OUT_ROOT)
    rows = best_per_complexity(load_mean_std(final_csv, threshold_label))
    threshold_col = threshold_group_column(rows)
    count = 0
    for (dataset, depth, threshold), group in rows.groupby(["dataset", "depth", threshold_col], dropna=False):
        out_dir = (
            out_root
            / str(dataset)
            / f"linear_regression_tree_depth{int(depth)}_threshold_{threshold_path_token(threshold)}"
        )
        make_dataset_plot(group, out_dir, args.openml)
        count += 1
    print(f"Generated {count} figures under {out_root}")


if __name__ == "__main__":
    main()
