from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CSV_DIR = SCRIPT_DIR / "csv"
OUTPUT_DIR = SCRIPT_DIR / "plots_output"


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "figure.figsize": (5, 4),
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    }
)


def load_kernel_results():
    files = sorted(CSV_DIR.glob("results_*.csv"))
    if not files:
        raise RuntimeError(f"No kernel result files found in {CSV_DIR}")

    frames = [pd.read_csv(path) for path in files]
    data = pd.concat(frames, ignore_index=True)

    keep = {
        "tree_kernel_1": "Tree",
        # "tree_kernel_2": "Tree",
        # "tree_kernel_3": "Tree",
        "wl_kernel_1": "WL",
        # "wl_kernel_2": "WL",
        # "wl_kernel_3": "WL",
        "damg_kernel_1": "DAMG",
        # "damg_kernel_2": "DAMG",
        # "damg_kernel_3": "DAMG",
    }
    data = data[data["kernel"].isin(keep)].copy()
    data["kernel_pretty"] = data["kernel"].map(keep)
    data["space_type"] = np.where(data["target"].str.contains("refined"), "refined", "global")
    data["target_len"] = data["target"].str.extract(r"target_len_(\d+)").astype(int)
    return data


def load_bo_results():
    pattern = re.compile(
        r"^(?P<kind>bo_trace|random_trace)"
        r"_target_len_(?P<target_len>\d+)"
        r"(?:_(?P<space>global|refinement_\d+))?"
        r"_run_(?P<run>\d+)\.csv$"
    )

    frames = []
    for path in sorted(CSV_DIR.glob("*.csv")):
        match = pattern.match(path.name)
        if match is None:
            continue

        df = pd.read_csv(path)
        df["kind"] = match.group("kind")
        df["target_len"] = int(match.group("target_len"))
        df["space"] = match.group("space") or "global"
        df["run"] = int(match.group("run"))
        frames.append(df)

    if not frames:
        raise RuntimeError(f"No BO trace files found in {CSV_DIR}")

    data = pd.concat(frames, ignore_index=True)
    data["evaluation"] = pd.to_numeric(data["evaluation"], errors="coerce")
    data["best_objective_value"] = pd.to_numeric(data["best_objective_value"], errors="coerce")
    data["method_label"] = data["kind"].map({"bo_trace": "BO", "random_trace": "Random"})

    # Average over full experiments, not over reused run ids.
    data["experiment_key"] = (
        data["method_label"].astype(str)
        + "|"
        + data["target_name"].astype(str)
        + "|"
        + data["kernel_name"].astype(str)
        + "|"
        + data["seed"].astype(str)
    )

    data = data.dropna(subset=["evaluation", "best_objective_value", "experiment_key", "method_label"])
    return data


def plot_alignment_vs_surrogate():
    data = load_kernel_results()

    fig, ax = plt.subplots(figsize=(5, 4))
    colors = {"Tree": "#1f77b4", "WL": "#ff7f0e", "DAMG": "#2ca02c"}
    markers = {"global": "o", "refined": "s"}
    sizes = {3: 35, 4: 85, 5: 135}

    for kernel in ["Tree", "WL", "DAMG"]:
        for space_type in ["global", "refined"]:
            subset = data[
                (data["kernel_pretty"] == kernel) & (data["space_type"] == space_type)
            ]
            if subset.empty:
                continue

            ax.scatter(
                subset["kernel_objective_spearman_correlation"],
                subset["surrogate_r2_mean"],
                label=f"{kernel} ({space_type})",
                color=colors[kernel],
                marker=markers[space_type],
                s=subset["target_len"].map(sizes).to_numpy(dtype=float),
                alpha=0.45,
                edgecolors="none",
            )

    ax.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("Kernel-Objective Alignment")
    ax.set_ylabel("Surrogate Performance (R²)")
    ax.set_title("Kernel Alignment and Surrogate Performance")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_alignment_vs_surrogate.pdf")
    plt.close(fig)


def plot_surrogate_r2_violin():
    data = load_kernel_results()

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    kernels = ["Tree", "WL", "DAMG"]
    colors = {"Tree": "#1f77b4", "WL": "#ff7f0e", "DAMG": "#2ca02c"}
    space_types = ["global", "refined"]

    values = []
    positions = []
    body_colors = []
    tick_positions = []

    for i, kernel in enumerate(kernels):
        base = i * 3
        tick_positions.append(base + 1.5)
        for offset, space_type in zip([1, 2], space_types, strict=False):
            subset = data[
                (data["kernel_pretty"] == kernel) & (data["space_type"] == space_type)
            ]
            values.append(subset["surrogate_r2_mean"].dropna().to_numpy(dtype=float))
            positions.append(base + offset)
            body_colors.append(colors[kernel])

    violin = ax.violinplot(values, positions=positions, showmeans=True, showmedians=False)
    for body, color in zip(violin["bodies"], body_colors, strict=False):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.3)

    for key in ["cbars", "cmins", "cmaxes", "cmeans"]:
        if key in violin:
            violin[key].set_color("black")
            violin[key].set_linewidth(1.0)

    ax.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(kernels)
    # ax.set_xlabel("Kernel")
    ax.set_ylabel("Surrogate Performance (R²)")
    ax.set_title("Surrogate Performance by Kernel")

    for pos, label in zip(positions, ["G", "R", "G", "R", "G", "R"], strict=False):
        ax.text(pos, ax.get_ylim()[0], label, ha="center", va="bottom", fontsize=9, alpha=0.8)

    y_min, y_max = ax.get_ylim()
    y_pad = 0.02 * (y_max - y_min)
    for pos, vals in zip(positions, values, strict=False):
        if len(vals) == 0:
            continue
        vmin = float(np.min(vals))
        vmean = float(np.mean(vals))
        vmax = float(np.max(vals))
        ax.text(pos + 0.08, vmin - y_pad, f"{vmin:.2f}", fontsize=8, alpha=0.9)
        ax.text(pos + 0.08, vmean, f"{vmean:.2f}", fontsize=8, alpha=0.9)
        ax.text(pos + 0.08, vmax + y_pad, f"{vmax:.2f}", fontsize=8, alpha=0.9)

    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + 0.02 * (y_max - y_min))

    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.16, top=0.9)
    fig.savefig(OUTPUT_DIR / "fig_surrogate_r2_violin.pdf")
    plt.close(fig)


def print_grouped_latex_table():
    data = load_kernel_results()

    summary = (
        data.groupby(["kernel_pretty", "space_type"])
        .agg(
            nmse_mean=("surrogate_normalized_mse_mean", "mean"),
            nmse_max=("surrogate_normalized_mse_mean", "max"),
            r2_mean=("surrogate_r2_mean", "mean"),
            r2_max=("surrogate_r2_mean", "max"),
        )
        .reset_index()
    )

    kernel_order = ["Tree", "WL", "DAMG"]
    space_order = ["global", "refined"]
    summary["kernel_pretty"] = pd.Categorical(summary["kernel_pretty"], categories=kernel_order, ordered=True)
    summary["space_type"] = pd.Categorical(summary["space_type"], categories=space_order, ordered=True)
    summary = summary.sort_values(["kernel_pretty", "space_type"])

    best_nmse_mean = float(summary["nmse_mean"].min())
    best_nmse_max = float(summary["nmse_max"].min())
    best_r2_mean = float(summary["r2_mean"].max())
    best_r2_max = float(summary["r2_max"].max())

    def fmt(value, *, bold=False, underline=False):
        text = f"{value:.3f}"
        if bold:
            text = rf"\textbf{{{text}}}"
        if underline:
            text = rf"\underline{{{text}}}"
        return text

    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Kernel & Space & \multicolumn{2}{c}{NMSE $\downarrow$} & \multicolumn{2}{c}{$R^2$ $\uparrow$} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r" &  & Mean & Max & Mean & Max \\",
        r"\midrule",
    ]

    for kernel in kernel_order:
        rows = summary[summary["kernel_pretty"] == kernel]
        for idx, row in enumerate(rows.itertuples(index=False)):
            kernel_label = kernel if idx == 0 else ""
            space_label = "Global" if str(row.space_type) == "global" else "Refined"
            lines.append(
                f"{kernel_label} & {space_label} & "
                f"{fmt(row.nmse_mean, bold=np.isclose(row.nmse_mean, best_nmse_mean))} & "
                f"{fmt(row.nmse_max, underline=np.isclose(row.nmse_max, best_nmse_max))} & "
                f"{fmt(row.r2_mean, bold=np.isclose(row.r2_mean, best_r2_mean))} & "
                f"{fmt(row.r2_max, underline=np.isclose(row.r2_max, best_r2_max))} \\\\"
            )
        if kernel != kernel_order[-1]:
            lines.append(r"\addlinespace")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    print("\n".join(lines))


def plot_bo_vs_random_progress():
    data = load_bo_results()

    data = data.copy()
    bo_mask = data["method_label"] == "BO"
    data.loc[bo_mask, "evaluation"] = data.loc[bo_mask, "evaluation"] + 10
    data = data[data["evaluation"] <= 100]

    summary = (
        data.groupby(["method_label", "evaluation", "experiment_key"])["best_objective_value"]
        .last()
        .reset_index()
        .groupby(["method_label", "evaluation"])["best_objective_value"]
        .agg(["mean", "sem"])
        .reset_index()
    )
    summary["sem"] = summary["sem"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    colors = {"BO": "#1f77b4", "Random": "#ff7f0e"}

    for method in ["BO", "Random"]:
        subset = summary[summary["method_label"] == method].sort_values("evaluation")
        x = subset["evaluation"].to_numpy(dtype=float)
        y = subset["mean"].to_numpy(dtype=float)
        sem = subset["sem"].to_numpy(dtype=float)

        ax.plot(x, y, color=colors[method], linewidth=2.2, label=method)
        ax.fill_between(x, y - sem, y + sem, color=colors[method], alpha=0.18, linewidth=0)

    ax.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Best objective value")
    ax.set_title("Optimization progress")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "bo_vs_random_progress.pdf")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_alignment_vs_surrogate()
    plot_surrogate_r2_violin()
    plot_bo_vs_random_progress()
    print_grouped_latex_table()
    print(f"Saved figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
