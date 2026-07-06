from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import polars as pl

import globals as g
from analysis.figures_src.ablation.plotting import _order_methods as _order_methods_ablation
from analysis.figures_src.camera_ready.plotting import _audit_plot
from analysis.figures_src.camera_ready.utils import maybe_copy_pdf_to_paper
from analysis.figures_src.fig_grass_camera_ready.config import CONFIG
from analysis.figures_src.method_colors import get_method_color_map
from analysis.figures_src.method_labels import format_method_labels
from analysis.figures_src.plot_styles import BLOCK_PERM_METHODS
from analysis.figures_src.run_selection import find_latest_results
from io_utils import ensure_dir, read_json, write_json
from logging_utils import get_logger
from provenance import get_git_state, get_tree_hashes


_LOGGER = get_logger("fig_grass_camera_ready")
TREE_HASH_PATHS = ("bench", "sketches", "kernels", "data", "external/GraSS")

def _ordered_methods(grouped):
    """Return methods in ablation order, filtered to the plot data."""
    methods = [m for m in CONFIG.methods if m in grouped["method"].unique().to_list()]
    return _order_methods_ablation(methods)


def _display_labels(methods):
    """Return camera-ready labels, appending bold (Ours) where appropriate."""
    labels = format_method_labels(methods)
    label_map = dict(zip(methods, labels))
    for method in methods:
        display = label_map[method]
        if method in BLOCK_PERM_METHODS or method == g.METHOD_FLASH_BLOCK_ROW:
            if "(Ours)" in display:
                label_map[method] = display
                continue
            display = f"{display} $\\mathbf{{(Ours)}}$"
        label_map[method] = display
    return label_map


def _resolve_run_paths(required_columns, required_datasets, required_tasks):
    """Resolve run paths based on config settings."""
    if CONFIG.run_paths:
        return [g.REPO_ROOT / path for path in CONFIG.run_paths]
    if CONFIG.use_latest_run:
        return [
            find_latest_results(
                required_columns,
                required_datasets,
                required_tasks,
                tree_hash_paths=TREE_HASH_PATHS,
            )
        ]
    raise ValueError("No run paths specified and use_latest_run is False.")


def _load_data(run_paths):
    """Load and concatenate results parquet files."""
    frames = []
    for path in run_paths:
        frames.append(pl.read_parquet(path).with_columns(pl.lit(str(path)).alias("run_path")))
    return pl.concat(frames) if frames else pl.DataFrame()


def _filter_data(df):
    """Filter to the configured dataset/model/task/methods."""
    df = df.filter(
        (pl.col("task") == CONFIG.task)
        & (pl.col("dataset") == CONFIG.dataset)
        & (pl.col("dataset_name") == CONFIG.dataset_name)
        & (pl.col("model") == CONFIG.model)
        & (pl.col("method").is_in(CONFIG.methods))
    )
    if CONFIG.dataset_group is not None and "dataset_group" in df.columns:
        df = df.filter(pl.col("dataset_group") == CONFIG.dataset_group)
    if "mlp_activation" in df.columns:
        df = df.filter(pl.col("mlp_activation") == CONFIG.activation)
    if "mlp_dropout_rate" in df.columns:
        df = df.filter(pl.col("mlp_dropout_rate") == CONFIG.dropout_rate)
    if df.is_empty():
        raise ValueError("No matching GraSS data found for the configured filters.")
    return df


def _aggregate(df):
    """Aggregate mean/std for projection time and metric."""
    df = df.filter(
        pl.col(CONFIG.proj_time_col).is_not_null()
        & pl.col(CONFIG.metric_col).is_not_null()
    )
    aggregations = [
        pl.col(CONFIG.proj_time_col).mean().alias("time_mean"),
        pl.col(CONFIG.metric_col).mean().alias("metric_mean"),
        pl.len().alias("count"),
    ]

    if hasattr(CONFIG, "proj_time_std_col") and CONFIG.proj_time_std_col in df.columns:
        aggregations.append(pl.col(CONFIG.proj_time_std_col).mean().alias("time_std"))
    else:
        aggregations.append(pl.col(CONFIG.proj_time_col).std().alias("time_std"))

    if hasattr(CONFIG, "metric_std_col") and CONFIG.metric_std_col in df.columns:
        aggregations.append(pl.col(CONFIG.metric_std_col).mean().alias("metric_std"))
    else:
        aggregations.append(pl.col(CONFIG.metric_col).std().alias("metric_std"))

    return df.group_by(["method", "k"]).agg(aggregations)


def _marker_map(k_values):
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8", "p"]
    if len(k_values) > len(markers):
        raise ValueError("Too many k values for available markers.")
    return {k: markers[idx] for idx, k in enumerate(sorted(k_values))}


def _plot_pareto(grouped, color_map):
    """Plot proj-only time vs LDS with ellipses."""
    k_values = sorted(grouped["k"].unique().to_list())
    marker_map = _marker_map(k_values)

    fig, ax = plt.subplots(figsize=(CONFIG.pareto_width_in, CONFIG.pareto_height_in))
    fig.subplots_adjust(right=0.72)

    methods = _ordered_methods(grouped)
    label_map = _display_labels(methods)
    for method in methods:
        method_df = grouped.filter(pl.col("method") == method).sort("k")
        if method_df.is_empty():
            continue
        xs = []
        ys = []
        for row in method_df.to_dicts():
            x = row["time_mean"]
            y = row["metric_mean"]
            xs.append(x)
            ys.append(y)
            x_std = row.get("time_std")
            x_std = float(x_std) if x_std is not None and np.isfinite(x_std) else 0.0
            if x_std > 0:
                ax.errorbar(
                    x,
                    y,
                    xerr=CONFIG.std_multiplier * x_std,
                    fmt="none",
                    ecolor=color_map[method],
                    elinewidth=1.0,
                    capsize=2.5,
                    capthick=1.0,
                    alpha=0.9,
                    zorder=1,
                )
            ax.scatter(
                x,
                y,
                marker=marker_map[row["k"]],
                s=40,
                color=color_map[method],
                edgecolor="white",
                linewidth=0.4,
                zorder=2,
            )
        if xs and ys:
            ax.plot(xs, ys, color=color_map[method], linewidth=1.6)

    ax.set_xlabel(
        "Projection time per sample (ms)", fontsize=CONFIG.min_font_size + 1
    )
    ax.set_ylabel("LDS (higher is better)", fontsize=CONFIG.min_font_size + 1)
    ax.set_title(CONFIG.title, fontsize=CONFIG.min_font_size + 1)
    ax.tick_params(axis="both", labelsize=CONFIG.min_font_size)
    ax.grid(alpha=0.25)

    xs = grouped["time_mean"].to_list()
    ys = grouped["metric_mean"].to_list()
    if xs and ys:
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
        x_pad = (x_max - x_min) * 0.05 if x_max > x_min else max(x_min * 0.05, 1.0)
        y_pad = (y_max - y_min) * 0.05 if y_max > y_min else max(y_min * 0.05, 1e-3)
        ax.set_xlim(left=0, right=x_max + x_pad)
        ax.set_ylim(top=y_max + y_pad)

    method_handles = [
        Line2D(
            [0],
            [0],
            color=color_map[method],
            marker=None,
            linestyle="-",
            linewidth=1.6,
            label=label_map.get(method, method),
        )
        for method in methods
        if method in color_map
    ]
    k_handles = [
        Line2D([0], [0], color="black", marker=marker_map[k], linestyle="none", label=f"{k}")
        for k in k_values
    ]
    legend1 = ax.legend(
        handles=method_handles,
        fontsize=CONFIG.min_font_size - 1,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    ax.add_artist(legend1)
    legend2 = ax.legend(
        handles=k_handles,
        title="Sketch dimension (k)",
        fontsize=CONFIG.min_font_size - 1,
        title_fontsize=CONFIG.min_font_size - 1,
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        borderaxespad=0.0,
    )

    audit = _audit_plot(
        fig, ax, list(label_map.values()), "LDS (higher is better)", CONFIG
    )
    output_base = g.FIGURES_DIR() / f"{CONFIG.output_prefix}_pareto"
    ensure_dir(output_base.parent)
    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight", bbox_extra_artists=(legend1, legend2))
    fig.savefig(png_path, bbox_inches="tight", bbox_extra_artists=(legend1, legend2), dpi=300)
    plt.close(fig)

    maybe_copy_pdf_to_paper(pdf_path)
    return [pdf_path, png_path], audit


def _plot_bar(grouped, color_map):
    """Plot proj-only time bars with error bars and speedup annotations."""
    dims = sorted(grouped["k"].unique().to_list())
    methods = _ordered_methods(grouped)
    label_map = _display_labels(methods)
    group_spacing = 1.8
    x = np.arange(len(dims)) * group_spacing
    n_methods = len(methods)
    width = 0.3 if n_methods <= 5 else 0.2 if n_methods <= 6 else 0.12
    bar_spacing = width
    offset = (n_methods - 1) / 2.0

    baseline = (
        grouped.filter(pl.col("method") == CONFIG.baseline_method)
        .select(["k", "time_mean"])
        .to_dicts()
    )
    baseline_by_k = {row["k"]: row["time_mean"] for row in baseline}
    if not baseline_by_k:
        raise ValueError("Missing baseline timings for speedup annotations.")

    fig, ax = plt.subplots(figsize=(CONFIG.bar_width_in, CONFIG.bar_height_in))
    fig.subplots_adjust(left=0.12, right=0.98)

    for idx, method in enumerate(methods):
        method_df = grouped.filter(pl.col("method") == method).sort("k")
        if method_df.is_empty():
            continue
        means = []
        stds = []
        for dim in dims:
            row = method_df.filter(pl.col("k") == dim)
            if row.is_empty():
                means.append(np.nan)
                stds.append(0.0)
            else:
                means.append(float(row["time_mean"][0]))
                std_val = row["time_std"][0]
                stds.append(float(std_val) if std_val is not None and np.isfinite(std_val) else 0.0)
        positions = x + (idx - offset) * bar_spacing
        ax.bar(
            positions,
            means,
            width=width,
            color=color_map[method],
            yerr=stds,
            capsize=3,
            edgecolor="black",
            linewidth=0.3,
        )
        for pos, dim, mean, std in zip(positions, dims, means, stds):
            if not np.isfinite(mean) or mean <= 0:
                continue
            baseline_mean = baseline_by_k.get(dim)
            if baseline_mean is None or baseline_mean <= 0:
                continue
            speedup = baseline_mean / mean
            label = f"{speedup:.2f}x"
            std_val = float(std) if std is not None and np.isfinite(std) else 0.0
            pad = max(0.2, mean * 0.0025)
            ax.text(
                pos,
                mean + std_val + pad,
                label,
                ha="center",
                va="bottom",
                fontsize=CONFIG.min_font_size - 1,
                rotation=0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(dim) for dim in dims])
    ax.set_xlabel("Sketch dimension (k)", fontsize=CONFIG.min_font_size + 1)
    ax.set_ylabel(
        "Projection time per sample (ms)", fontsize=CONFIG.min_font_size + 1
    )
    ax.set_title("Speedup relative to GraSS baseline", fontsize=CONFIG.min_font_size + 1)
    ax.tick_params(axis="both", labelsize=CONFIG.min_font_size)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    y_max = grouped["time_mean"].max()
    if y_max is not None:
        ax.set_ylim(bottom=0, top=float(y_max) * 1.15)

    audit = _audit_plot(
        fig,
        ax,
        list(label_map.values()),
        "Projection time per sample (ms)",
        CONFIG,
    )
    output_base = g.FIGURES_DIR() / f"{CONFIG.output_prefix}_proj_only_bar"
    ensure_dir(output_base.parent)
    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    maybe_copy_pdf_to_paper(pdf_path)
    return [pdf_path, png_path], audit


def _update_manifest(entries, run_paths, git_state, tree_hashes):
    """Update the figures manifest with provenance information."""
    manifest_path = g.FIGURE_MANIFEST_PATH()
    manifest = read_json(manifest_path) if manifest_path.exists() else {}

    for entry in entries:
        output_name = entry["output_name"]
        manifest[output_name] = {
            "script": str(Path(__file__).relative_to(g.REPO_ROOT)),
            "commit": git_state["commit"],
            "dirty": git_state["dirty"],
            "tree_hashes": tree_hashes,
            "inputs": [str(Path(path).relative_to(g.REPO_ROOT)) for path in run_paths],
            "dataset": CONFIG.dataset,
            "dataset_name": CONFIG.dataset_name,
            "model": CONFIG.model,
            "task": CONFIG.task,
            "time_col": CONFIG.proj_time_col,
            "metric": CONFIG.metric_col,
            "aggregation": "mean/std",
            "sample_count": entry["sample_count"],
            "audit": entry["audit"],
            "make_target": "fig.grass.camera_ready",
        }

    ensure_dir(manifest_path.parent)
    write_json(manifest_path, manifest)


def main():
    """Generate camera-ready GraSS plots."""
    required_columns = {
        "task",
        "dataset",
        "dataset_name",
        "model",
        "method",
        "k",
        CONFIG.proj_time_col,
        CONFIG.metric_col,
    }
    if hasattr(CONFIG, "proj_time_std_col"):
        required_columns.add(CONFIG.proj_time_std_col)
    if hasattr(CONFIG, "metric_std_col"):
        required_columns.add(CONFIG.metric_std_col)
    if CONFIG.dataset_group is not None:
        required_columns.add("dataset_group")
    if CONFIG.activation:
        required_columns.add("mlp_activation")
    if CONFIG.dropout_rate is not None:
        required_columns.add("mlp_dropout_rate")

    required_datasets = [(CONFIG.dataset, CONFIG.dataset_name, CONFIG.dataset_group)]
    required_tasks = {CONFIG.task}
    run_paths = _resolve_run_paths(required_columns, required_datasets, required_tasks)

    df = _load_data(run_paths)
    df = _filter_data(df)
    grouped = _aggregate(df)

    color_map = get_method_color_map([m for m in CONFIG.methods if m in grouped["method"].unique().to_list()])

    git_state = get_git_state()
    tree_hashes = get_tree_hashes(TREE_HASH_PATHS)

    entries = []
    pareto_paths, pareto_audit = _plot_pareto(grouped, color_map)
    for path in pareto_paths:
        entries.append({"output_name": path.name, "sample_count": int(df.height), "audit": pareto_audit})
        _LOGGER.info("Generated %s", path)

    bar_paths, bar_audit = _plot_bar(grouped, color_map)
    for path in bar_paths:
        entries.append({"output_name": path.name, "sample_count": int(df.height), "audit": bar_audit})
        _LOGGER.info("Generated %s", path)

    _update_manifest(entries, run_paths, git_state, tree_hashes)


if __name__ == "__main__":
    main()
