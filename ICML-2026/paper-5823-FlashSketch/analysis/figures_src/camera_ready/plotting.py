from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import polars as pl

import globals as g
from analysis.figures_src.method_colors import get_method_color_map
from analysis.figures_src.method_labels import format_method_labels
from analysis.figures_src.plot_styles import BLOCK_PERM_METHODS, build_style_map
from analysis.figures_src.run_selection import find_latest_results
from analysis.figures_src.camera_ready.utils import get_gpu_info, maybe_copy_pdf_to_paper
from io_utils import ensure_dir, read_json, write_json
from logging_utils import get_logger
from provenance import get_git_state, get_tree_hashes


TREE_HASH_PATHS = (
    "bench/camera_ready",
    "bench/e2e/tasks",
    "sketches",
    "kernels",
    "data",
)


def _find_latest_results(required_columns, required_datasets, required_tasks):
    """Return the latest E2E results.parquet under file_storage/runs."""
    return find_latest_results(
        required_columns,
        required_datasets,
        required_tasks,
        tree_hash_paths=TREE_HASH_PATHS,
    )


def _resolve_run_paths(config, required_columns, required_datasets, required_tasks):
    """Resolve run paths based on config settings."""
    if config.run_paths:
        return [g.REPO_ROOT / path for path in config.run_paths]
    if config.use_latest_run:
        return [_find_latest_results(required_columns, required_datasets, required_tasks)]
    raise ValueError("No run paths specified and use_latest_run is False.")


def _load_data(run_paths):
    """Load and concatenate results parquet files."""
    frames = []
    for path in run_paths:
        frames.append(pl.read_parquet(path).with_columns(pl.lit(str(path)).alias("run_path")))
    return pl.concat(frames) if frames else pl.DataFrame()


def _add_method_label(df, config):
    """Add a method_label column using the configured label fields."""
    parts = [pl.col("method")]
    bool_labels = config.method_label_bool_labels
    value_only = set(config.method_label_value_only_fields)

    for field in config.method_label_fields:
        if field not in df.columns:
            continue
        if field in bool_labels:
            true_label, false_label = bool_labels[field]
            part = (
                pl.when(pl.col(field).is_not_null())
                .then(
                    pl.when(pl.col(field))
                    .then(pl.lit(true_label))
                    .otherwise(pl.lit(false_label))
                )
                .otherwise(pl.lit(""))
            )
        elif field in value_only:
            part = (
                pl.when(pl.col(field).is_not_null())
                .then(pl.col(field).cast(pl.Utf8))
                .otherwise(pl.lit(""))
            )
        else:
            part = (
                pl.when(pl.col(field).is_not_null())
                .then(
                    pl.concat_str(
                        [pl.lit(f"{field}="), pl.col(field).cast(pl.Utf8)],
                        separator="",
                    )
                )
                .otherwise(pl.lit(""))
            )
        parts.append(part)

    df = df.with_columns(pl.concat_str(parts, separator=" ").alias("method_label"))
    return df.with_columns(
        pl.col("method_label")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("method_label")
    )


def _format_method_labels(methods):
    """Return camera-ready display labels."""
    return format_method_labels(methods)


def _make_color_map(methods):
    """Return a stable color map for sketch method names."""
    return get_method_color_map(methods)


def _make_marker_map(k_values):
    """Return a consistent marker map for sketch dimensions."""
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8", "p"]
    if len(k_values) > len(markers):
        raise ValueError("Too many k values for available markers.")
    return {k: markers[idx] for idx, k in enumerate(sorted(k_values))}


def _aggregate_metric(df, metric, time_col, config):
    """Aggregate metric vs time by method and k using mean with std dev ellipses."""
    filtered = df.filter(pl.col(metric).is_not_null() & pl.col(time_col).is_not_null())
    if filtered.is_empty():
        return pl.DataFrame(), filtered

    if "method_label" not in filtered.columns:
        filtered = _add_method_label(filtered, config)

    if "k" not in filtered.columns:
        raise ValueError("E2E data must include k to plot error vs time.")

    if config.filter_outliers:
        filtered = _filter_outliers(filtered, metric, time_col, config)
        if filtered.is_empty():
            return pl.DataFrame(), filtered

    group_cols = ["method_label", "method", "k"]
    if "s" in filtered.columns:
        group_cols.append("s")
    if "kappa" in filtered.columns:
        group_cols.append("kappa")

    grouped = filtered.group_by(group_cols).agg(
        [
            pl.col(time_col).mean().alias("time_mean"),
            pl.col(time_col).std().alias("time_std"),
            pl.col(metric).mean().alias("metric_mean"),
            pl.col(metric).std().alias("metric_std"),
            pl.len().alias("count"),
        ]
    )
    return grouped, filtered


def _filter_outliers(df, metric, time_col, config):
    """Filter per-method outliers using IQR thresholds on time and metric."""
    grouped = df.group_by(["method_label", "k"]).agg(
        [
            pl.col(time_col).quantile(0.25).alias("time_p25"),
            pl.col(time_col).quantile(0.75).alias("time_p75"),
            pl.col(metric).quantile(0.25).alias("metric_p25"),
            pl.col(metric).quantile(0.75).alias("metric_p75"),
        ]
    )
    joined = df.join(grouped, on=["method_label", "k"], how="left").with_columns(
        [
            (pl.col("time_p75") - pl.col("time_p25")).alias("time_iqr"),
            (pl.col("metric_p75") - pl.col("metric_p25")).alias("metric_iqr"),
        ]
    )

    mult = config.outlier_iqr_multiplier
    time_low = pl.col("time_p25") - mult * pl.col("time_iqr")
    time_high = pl.col("time_p75") + mult * pl.col("time_iqr")
    metric_low = pl.col("metric_p25") - mult * pl.col("metric_iqr")
    metric_high = pl.col("metric_p75") + mult * pl.col("metric_iqr")

    time_ok = pl.when(pl.col("time_iqr") <= 0).then(True).otherwise(
        (pl.col(time_col) >= time_low) & (pl.col(time_col) <= time_high)
    )
    metric_ok = pl.when(pl.col("metric_iqr") <= 0).then(True).otherwise(
        (pl.col(metric) >= metric_low) & (pl.col(metric) <= metric_high)
    )

    return joined.filter(time_ok & metric_ok).select(df.columns)


def _time_col_for_task(task):
    """Return the time column and label for a task (wall clock only)."""
    if task in {g.TASK_GRAM_APPROX, g.TASK_OSE_ERROR}:
        return ("sketch_time_ms", "Wall-clock sketch time (ms)")
    return ("total_time_ms", "Wall-clock total time (ms)")


def _audit_plot(fig, ax, methods, metric_label, config):
    """Audit plot layout and raise if requirements are not met."""
    width_in, height_in = fig.get_size_inches()
    issues = []

    if width_in < 3.3 or height_in < 2.0:
        issues.append(f"figure size too small ({width_in:.2f}x{height_in:.2f} in)")

    if not ax.get_xlabel() or not ax.get_ylabel():
        issues.append("missing axis labels")

    tick_sizes = [label.get_fontsize() for label in ax.get_xticklabels() + ax.get_yticklabels()]
    min_tick = min(tick_sizes) if tick_sizes else 0
    if min_tick < config.min_font_size:
        issues.append(f"tick label size below minimum ({min_tick})")

    if methods and max(len(method) for method in methods) > 25 and width_in < 4.5:
        issues.append("long method labels may be cramped")

    audit = {
        "metric": metric_label,
        "width_in": float(width_in),
        "height_in": float(height_in),
        "min_tick_size": float(min_tick),
        "methods": len(methods),
        "issues": issues,
    }

    if issues:
        raise ValueError(f"Plot audit failed: {issues}")

    return audit


def _filter_dataset(df, plot_spec):
    """Filter the dataframe to the requested dataset."""
    filtered = df.filter(pl.col("dataset") == plot_spec.dataset)
    if plot_spec.dataset_name:
        filtered = filtered.filter(pl.col("dataset_name") == plot_spec.dataset_name)
    if plot_spec.dataset_group and "dataset_group" in df.columns:
        filtered = filtered.filter(pl.col("dataset_group") == plot_spec.dataset_group)
    return filtered


def _set_axis_limits(ax, df, metric, time_col, config, log_scale=False):
    """Set axis limits, trimming only when outliers are filtered."""
    if df.is_empty():
        return

    if config.filter_outliers:
        time_lo = df.select(pl.col(time_col).quantile(config.axis_quantile_lo)).item()
        time_hi = df.select(pl.col(time_col).quantile(config.axis_quantile_hi)).item()
        metric_lo = df.select(pl.col(metric).quantile(config.axis_quantile_lo)).item()
        metric_hi = df.select(pl.col(metric).quantile(config.axis_quantile_hi)).item()
    else:
        time_lo = df.select(pl.col(time_col).min()).item()
        time_hi = df.select(pl.col(time_col).max()).item()
        metric_lo = df.select(pl.col(metric).min()).item()
        metric_hi = df.select(pl.col(metric).max()).item()

    def _finite(value):
        return value is not None and math.isfinite(value)

    if not (_finite(time_lo) and _finite(time_hi)):
        return
    if not (_finite(metric_lo) and _finite(metric_hi)):
        return

    if time_hi == time_lo:
        time_hi = time_lo * 1.05 if time_lo != 0 else 1.0
    if metric_hi == metric_lo:
        metric_hi = metric_lo * 1.05 if metric_lo != 0 else 1.0

    if log_scale:
        time_min = df.select(pl.col(time_col).min()).item()
        metric_min = df.select(pl.col(metric).min()).item()
        if not (_finite(time_min) and _finite(metric_min)):
            raise ValueError("Log-log plot requires finite time and metric values.")
        if time_min <= 0 or metric_min <= 0:
            raise ValueError("Log-log plot requires strictly positive time and metric values.")
        pad = max(0.0, float(config.axis_pad_frac))
        ax.set_xlim(left=time_min * (1.0 - pad), right=time_hi * (1.0 + pad))
        ax.set_ylim(bottom=metric_min * (1.0 - pad), top=metric_hi * (1.0 + pad))
        return

    if time_lo is not None and time_hi is not None:
        if time_lo >= 0:
            ax.set_xlim(left=0, right=time_hi * 1.05)
        else:
            ax.set_xlim(left=time_lo, right=time_hi * 1.05)

    if metric_lo is not None and metric_hi is not None:
        if metric_lo >= 0:
            ax.set_ylim(bottom=0, top=metric_hi * 1.05)
        else:
            ax.set_ylim(bottom=metric_lo, top=metric_hi * 1.05)


def _maybe_copy_pdf_to_paper(pdf_path):
    """Optionally copy PDFs to paper/figures when env var is set."""
    maybe_copy_pdf_to_paper(pdf_path)


def _plot_metric_task(df, plot_spec, output_path, config):
    """Plot error vs time for a metric and task."""
    task_df = df.filter(pl.col("task") == plot_spec.task)
    task_df = _filter_dataset(task_df, plot_spec)
    if task_df.is_empty():
        raise ValueError(f"No data for plot {plot_spec.name} in E2E results.")

    time_col, time_label = _time_col_for_task(plot_spec.task)
    grouped, filtered = _aggregate_metric(task_df, plot_spec.metric.name, time_col, config)
    if grouped.is_empty():
        raise ValueError(
            f"No data for metric {plot_spec.metric.name} with time {time_col} in plot {plot_spec.name}."
        )

    removed_non_positive = 0
    positive_mask = (pl.col(plot_spec.metric.name) > 0) & (pl.col(time_col) > 0)
    if not filtered.is_empty():
        removed_non_positive = filtered.filter(~positive_mask).height
        if removed_non_positive > 0:
            filtered = filtered.filter(positive_mask)
            grouped, filtered = _aggregate_metric(filtered, plot_spec.metric.name, time_col, config)
            if grouped.is_empty():
                raise ValueError("No positive values available for log-log plotting.")

    sample_count = int(filtered.height)
    methods = sorted(grouped["method_label"].unique().to_list())
    method_names = sorted(grouped["method"].unique().to_list())
    k_values = sorted(grouped["k"].unique().to_list())
    display_methods = _format_method_labels(methods)
    label_to_display = dict(zip(methods, display_methods))
    label_to_method = {
        row["method_label"]: row["method"]
        for row in grouped.select(["method_label", "method"]).unique().to_dicts()
    }
    method_label_map = {}
    for method_label in methods:
        display = label_to_display[method_label]
        method = label_to_method.get(method_label)
        if method in BLOCK_PERM_METHODS or method == g.METHOD_FLASH_BLOCK_ROW:
            if "(Ours)" not in display:
                display = f"{display} $\\mathbf{{(Ours)}}$"
        method_label_map[method_label] = display

    def _method_sort_key(label):
        method = label_to_method.get(label, "")
        if method == g.METHOD_FLASH_SKETCH:
            return (0, "")
        return (1, method_label_map.get(label, label))

    methods = sorted(methods, key=_method_sort_key)

    color_map = _make_color_map(method_names)
    style_map = build_style_map(grouped, color_map)
    marker_map = _make_marker_map(k_values)

    fig_width = config.base_width_in
    fig_height = config.base_height_in
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=0.28, right=0.72)

    for method in methods:
        method_df = grouped.filter(pl.col("method_label") == method).sort("k")
        style = style_map[method]
        xs = []
        ys = []
        for row in method_df.to_dicts():
            x = row["time_mean"]
            y = row["metric_mean"]
            xs.append(x)
            ys.append(y)
            x_std = row.get("time_std")
            y_std = row.get("metric_std")
            x_std = float(x_std) if x_std is not None and math.isfinite(x_std) else 0.0
            y_std = float(y_std) if y_std is not None and math.isfinite(y_std) else 0.0
            width = max(2.0 * config.std_multiplier * x_std, 1e-9)
            height = max(2.0 * config.std_multiplier * y_std, 1e-9)
            if x_std > 0 or y_std > 0:
                ellipse = Ellipse(
                    (x, y),
                    width=width,
                    height=height,
                    facecolor=style["color"],
                    edgecolor=style["color"],
                    alpha=config.ellipse_alpha * style["alpha"],
                    linewidth=0.8,
                    zorder=1,
                )
                ax.add_patch(ellipse)
            ax.scatter(
                x,
                y,
                marker=marker_map[row["k"]],
                s=40,
                color=style["color"],
                alpha=style["alpha"],
                edgecolor="white",
                linewidth=0.4,
                zorder=2,
            )
        if xs and ys:
            ax.plot(
                xs,
                ys,
                color=style["color"],
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
                alpha=style["alpha"],
            )

    if config.annotate_speedup:
        target_k = config.speedup_k if config.speedup_k is not None else max(k_values)
        speedup_rows = grouped.filter(pl.col("k") == target_k)
        if speedup_rows.height >= 2:
            candidates = speedup_rows.select(["method_label", "time_mean", "metric_mean"]).to_dicts()
            candidates = [row for row in candidates if row["time_mean"] is not None]
            if len(candidates) >= 2:
                candidates.sort(key=lambda row: row["time_mean"])
                fastest = candidates[0]
                runner_up = candidates[1]
                x_fast = float(fastest["time_mean"])
                y_fast = float(fastest["metric_mean"])
                x_second = float(runner_up["time_mean"])
                y_second = float(runner_up["metric_mean"])
                if x_fast > 0:
                    speedup = x_second / x_fast
                    label = f"{speedup:.2f}x speedup"
                    ax.annotate(
                        "",
                        xy=(x_fast, y_fast),
                        xytext=(x_second, y_second),
                        textcoords="data",
                        arrowprops=dict(arrowstyle="->", lw=1.2, color="black"),
                    )
                    mid_x = math.sqrt(x_fast * x_second)
                    mid_y = math.sqrt(y_fast * y_second)
                    if config.speedup_text_offset:
                        mid_y = mid_y * (1.0 + config.speedup_text_offset)
                    ax.text(
                        mid_x,
                        mid_y,
                        label,
                        ha="center",
                        va=config.speedup_text_va,
                        fontsize=config.min_font_size,
                    )

    ax.set_xlabel(time_label, fontsize=config.min_font_size + 1)
    ax.set_ylabel(plot_spec.metric.label, fontsize=config.min_font_size + 1)
    task_label = config.task_label_map.get(plot_spec.task, plot_spec.task)
    ax.set_title(f"{plot_spec.title}\n{task_label}", fontsize=config.min_font_size + 1)
    ax.tick_params(axis="both", labelsize=config.min_font_size)
    ax.set_axisbelow(True)
    ax.grid(alpha=0.25)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if not config.use_auto_limits:
        _set_axis_limits(ax, filtered, plot_spec.metric.name, time_col, config, log_scale=True)

    method_handles = [
        Line2D(
            [0],
            [0],
            color=style_map[method]["color"],
            alpha=style_map[method]["alpha"],
            marker=None,
            markersize=0,
            linestyle=style_map[method]["linestyle"],
            linewidth=style_map[method]["linewidth"],
            label=method_label_map[method],
        )
        for method in methods
    ]
    k_handles = [
        Line2D([0], [0], color="black", marker=marker_map[k], linestyle="none", label=f"{k}")
        for k in k_values
    ]
    legend1 = ax.legend(
        handles=method_handles,
        fontsize=config.min_font_size - 1,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        ncol=1,
        columnspacing=0.8,
        handletextpad=0.6,
    )
    ax.add_artist(legend1)
    legend2 = ax.legend(
        handles=k_handles,
        title="Sketch dimension (k)",
        fontsize=config.min_font_size - 1,
        title_fontsize=config.min_font_size - 1,
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        borderaxespad=0.0,
    )

    audit = _audit_plot(fig, ax, display_methods, plot_spec.metric.label, config)

    ensure_dir(Path(output_path).parent)
    output_path = Path(output_path)
    fig.savefig(output_path, bbox_inches="tight", bbox_extra_artists=(legend1, legend2))
    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight", bbox_extra_artists=(legend1, legend2), dpi=300)
    plt.close(fig)

    _maybe_copy_pdf_to_paper(output_path)

    return audit, [output_path, png_path], sample_count


def _update_manifest(entries, run_paths, git_state, tree_hashes, config, make_target, script_path):
    """Update the paper figures manifest with provenance information."""
    manifest_path = g.FIGURE_MANIFEST_PATH()
    if manifest_path.exists():
        manifest = read_json(manifest_path)
    else:
        manifest = {}

    output_names = {entry["output_name"] for entry in entries}
    manifest = {
        name: payload
        for name, payload in manifest.items()
        if not (payload.get("make_target") == make_target and name not in output_names)
    }

    for entry in entries:
        output_name = entry["output_name"]
        manifest[output_name] = {
            "script": script_path,
            "commit": git_state["commit"],
            "dirty": git_state["dirty"],
            "tree_hashes": tree_hashes,
            "inputs": [str(Path(path).relative_to(g.REPO_ROOT)) for path in run_paths],
            "metric": entry["metric"],
            "task": entry["task"],
            "time_axis": entry["time_axis"],
            "time_col": entry["time_col"],
            "aggregation": "mean",
            "error_bars": "std_dev_ellipse",
            "outlier_filter": config.filter_outliers,
            "sample_count": entry["sample_count"],
            "audit": entry["audit"],
            "make_target": make_target,
        }

    ensure_dir(manifest_path.parent)
    write_json(manifest_path, manifest)


def generate_figures(config, make_target, script_path=None):
    """Generate camera-ready figures for the supplied config."""
    logger = get_logger(make_target)
    required_columns = {"task", "method", "k", "dataset", "dataset_name", "dataset_group"}
    for field in config.method_label_fields:
        required_columns.add(field)
    for plot_spec in config.plots:
        required_columns |= set(plot_spec.metric.requires or [])
    tasks = {plot_spec.task for plot_spec in config.plots}
    if tasks.intersection({g.TASK_GRAM_APPROX, g.TASK_OSE_ERROR}):
        required_columns.add("sketch_time_ms")
    if any(task not in {g.TASK_GRAM_APPROX, g.TASK_OSE_ERROR} for task in tasks):
        required_columns.add("total_time_ms")
    git_state = get_git_state()
    tree_hashes = get_tree_hashes(TREE_HASH_PATHS)
    required_datasets = [
        (plot.dataset, plot.dataset_name, plot.dataset_group)
        for plot in config.plots
    ]
    required_tasks = {plot.task for plot in config.plots}
    run_paths = _resolve_run_paths(config, sorted(required_columns), required_datasets, required_tasks)
    df = _load_data(run_paths)

    if df.is_empty():
        raise ValueError("No E2E data available to plot.")

    gpu_info = get_gpu_info()
    gpu_slug = gpu_info["slug"]
    entries = []
    for plot_spec in config.plots:
        output_path = g.FIGURES_DIR() / f"{plot_spec.output_name}_{gpu_slug}.pdf"
        audit, output_paths, sample_count = _plot_metric_task(
            df, plot_spec, output_path, config
        )

        for out_path in output_paths:
            logger.info("Generated %s", out_path)
            entries.append(
                {
                    "output_name": out_path.name,
                    "metric": plot_spec.metric.name,
                    "task": plot_spec.task,
                    "time_axis": "wall",
                    "time_col": _time_col_for_task(plot_spec.task)[0],
                    "audit": audit,
                    "sample_count": sample_count,
                }
            )

    if script_path is None:
        script_path = str(Path(__file__).relative_to(g.REPO_ROOT))
    _update_manifest(entries, run_paths, git_state, tree_hashes, config, make_target, script_path)
