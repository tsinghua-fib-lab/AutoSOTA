from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from pathlib import Path
import math
from dataclasses import replace
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import polars as pl

import globals as g
from analysis.figures_src.ablation.common import FigureConfig
from analysis.figures_src.method_colors import get_method_color_map
from analysis.figures_src.method_labels import format_method_labels
from analysis.figures_src.run_selection import find_latest_results
from analysis.figures_src.camera_ready.utils import get_gpu_info, maybe_copy_pdf_to_paper
from bench.ablation.methods import ABLATION_K_VALUES
from io_utils import ensure_dir, read_json, write_json
from logging_utils import get_logger
from provenance import get_git_state, get_tree_hashes


TREE_HASH_PATHS = (
    "bench/ablation",
    "bench/e2e/tasks",
    "sketches",
    "kernels",
    "data",
)
TITLE_WRAP_WIDTH = 32

FLASH_METHODS = {g.METHOD_FLASH_SKETCH, g.METHOD_FLASH_BLOCK_ROW}
SJLT_METHODS = {g.METHOD_SJLT_GRASS_KERNEL, g.METHOD_SJLT_CUSPARSE}

FLASH_LINESTYLES = [
    "-",
    "--",
    "-.",
    ":",
    (0, (3, 1, 1, 1)),
    (0, (5, 2)),
    (0, (1, 1)),
    (0, (5, 1, 1, 1)),
    (0, (3, 2, 1, 2)),
]
SJLT_LINESTYLES = ["-", "--", "-."]
DEFAULT_LINESTYLES = ["-"]

DEFAULT_LINEWIDTH = 2.0
DEFAULT_ALPHA = 1.0
MARKER_SIZE = 40

GLOBAL_K_VALUES = tuple(ABLATION_K_VALUES)

METHOD_ORDER = [
    g.METHOD_FLASH_SKETCH,
    g.METHOD_FLASH_BLOCK_ROW,
    g.METHOD_SJLT_GRASS_KERNEL,
    g.METHOD_SJLT_CUSPARSE,
    g.METHOD_SRHT_FWHT,
    g.METHOD_GAUSSIAN_DENSE_CUBLAS,
]


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


def _wrap_title(text):
    """Wrap title text to a consistent width to keep figure sizes stable."""
    return textwrap.fill(
        text,
        width=TITLE_WRAP_WIDTH,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _make_color_map(methods):
    """Return a stable color map for sketch method names."""
    return get_method_color_map(methods)


def _make_marker_map(k_values):
    """Return a consistent marker map for sketch dimensions."""
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8", "p"]
    if len(k_values) > len(markers):
        raise ValueError("Too many k values for available markers.")
    return {k: markers[idx] for idx, k in enumerate(sorted(k_values))}


def _variant_fields(grouped):
    """Return variant field names present in grouped data."""
    fields = []
    for field in ("kappa", "s"):
        if field in grouped.columns:
            fields.append(field)
    return fields


def _variant_key(row, fields):
    """Return the variant key tuple for a row."""
    return tuple(row.get(field) for field in fields)


def _variant_filter_expr(variant, fields):
    """Return a Polars filter expression matching a variant tuple."""
    expr = pl.lit(True)
    for field, value in zip(fields, variant):
        if value is None:
            expr &= pl.col(field).is_null()
        else:
            expr &= pl.col(field) == value
    return expr


def _sort_variants(method, variants, fields):
    """Return variants sorted deterministically for a method."""
    if not variants:
        return []
    if method in FLASH_METHODS and fields == ["kappa", "s"]:
        return sorted(variants, key=lambda item: (item[0] or 0, item[1] or 0))
    if method in SJLT_METHODS:
        return sorted(variants, key=lambda item: item[-1] or 0)
    return sorted(variants, key=lambda item: str(item))


def _line_styles_for_method(method, variant_count):
    """Return a list of line styles for a method's variants."""
    if method in FLASH_METHODS:
        styles = FLASH_LINESTYLES
    elif method in SJLT_METHODS:
        styles = SJLT_LINESTYLES
    else:
        styles = DEFAULT_LINESTYLES
    if variant_count > len(styles):
        raise ValueError(f"Not enough line styles for {method} variants ({variant_count}).")
    return styles


def _build_style_map(grouped, color_map):
    """Return style map keyed by (method, variant) tuples."""
    if "method" not in grouped.columns:
        raise ValueError("Style mapping requires a method column.")

    fields = _variant_fields(grouped)
    rows = grouped.select(["method"] + fields).unique().to_dicts()

    variants_by_method = {}
    for row in rows:
        method = row["method"]
        key = _variant_key(row, fields)
        variants_by_method.setdefault(method, set()).add(key)

    style_map = {}
    for method, variants in variants_by_method.items():
        ordered = _sort_variants(method, variants, fields)
        styles = _line_styles_for_method(method, len(ordered))
        for idx, variant in enumerate(ordered):
            style_map[(method, variant)] = {
                "color": color_map[method],
                "alpha": DEFAULT_ALPHA,
                "linewidth": DEFAULT_LINEWIDTH,
                "linestyle": styles[idx],
            }
    return style_map, variants_by_method, fields


def _variant_label(method, variant, fields):
    """Return a human-readable label for a variant tuple."""
    payload = {field: value for field, value in zip(fields, variant)}
    if method in FLASH_METHODS:
        kappa = payload.get("kappa")
        s = payload.get("s")
        if kappa is None or s is None:
            return ""
        return f"$\\kappa$={kappa}, s={s}"
    if method in SJLT_METHODS:
        s = payload.get("s")
        return f"s={s}" if s is not None else ""
    return ""


def _time_col_for_task(task):
    """Return the time column and label for a task (wall clock only)."""
    if task in {g.TASK_GRAM_APPROX, g.TASK_OSE_ERROR}:
        return ("sketch_time_ms", "Wall-clock sketch time (ms)")
    return ("total_time_ms", "Wall-clock total time (ms)")


def _aggregate_metric(df, metric, time_col, config):
    """Aggregate metric vs time by method and k using mean with std dev ellipses."""
    filtered = df.filter(
        pl.col(metric).is_not_null()
        & pl.col(time_col).is_not_null()
        & pl.col(metric).is_finite()
        & pl.col(time_col).is_finite()
    )
    if filtered.is_empty():
        return pl.DataFrame(), filtered

    if config.filter_outliers:
        filtered = _filter_outliers(filtered, metric, time_col, config)
        if filtered.is_empty():
            return pl.DataFrame(), filtered

    group_cols = ["method", "k"]
    for field in ("s", "kappa"):
        if field in filtered.columns:
            group_cols.append(field)

    log_time = pl.when(pl.col(time_col) > 0).then(pl.col(time_col).log10())
    log_metric = pl.when(pl.col(metric) > 0).then(pl.col(metric).log10())

    grouped = filtered.group_by(group_cols).agg(
        [
            pl.col(time_col).mean().alias("time_mean"),
            pl.col(time_col).std().alias("time_std"),
            pl.col(metric).mean().alias("metric_mean"),
            pl.col(metric).std().alias("metric_std"),
            log_time.std().alias("time_log_std"),
            log_metric.std().alias("metric_log_std"),
            pl.len().alias("count"),
        ]
    )
    return grouped, filtered


def _log_ellipse_points(center_x, center_y, width_log, height_log, num=72):
    """Return ellipse points computed in log space and mapped back to linear."""
    if center_x <= 0 or center_y <= 0:
        return None
    center_log_x = math.log10(center_x)
    center_log_y = math.log10(center_y)
    theta = np.linspace(0.0, 2.0 * math.pi, num=num, endpoint=False)
    x_log = center_log_x + 0.5 * width_log * np.cos(theta)
    y_log = center_log_y + 0.5 * height_log * np.sin(theta)
    return np.column_stack((10**x_log, 10**y_log))


def _filter_outliers(df, metric, time_col, config):
    """Filter per-method outliers using IQR thresholds on time and metric."""
    grouped = df.group_by(["method", "k"]).agg(
        [
            pl.col(time_col).quantile(0.25).alias("time_p25"),
            pl.col(time_col).quantile(0.75).alias("time_p75"),
            pl.col(metric).quantile(0.25).alias("metric_p25"),
            pl.col(metric).quantile(0.75).alias("metric_p75"),
        ]
    )
    joined = df.join(grouped, on=["method", "k"], how="left").with_columns(
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


def _set_axis_limits(ax, df, metric, time_col, config, log_scale=False):
    """Apply axis limits using quantiles to avoid extreme whitespace."""
    x_vals = df.select(pl.col(time_col).drop_nulls()).to_series().to_list()
    y_vals = df.select(pl.col(metric).drop_nulls()).to_series().to_list()
    x_vals = [value for value in x_vals if math.isfinite(value)]
    y_vals = [value for value in y_vals if math.isfinite(value)]
    if not x_vals or not y_vals:
        return

    x_series = pl.Series(x_vals)
    y_series = pl.Series(y_vals)
    x_lo = x_series.quantile(config.axis_quantile_lo)
    x_hi = x_series.quantile(config.axis_quantile_hi)
    y_lo = y_series.quantile(config.axis_quantile_lo)
    y_hi = y_series.quantile(config.axis_quantile_hi)

    if log_scale:
        x_lo = max(x_lo, min(x_vals))
        y_lo = max(y_lo, min(y_vals))

    ax.set_xlim(float(x_lo), float(x_hi))
    ax.set_ylim(float(y_lo), float(y_hi))


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
    if plot_spec.d is not None and "d" in df.columns:
        filtered = filtered.filter(pl.col("d") == plot_spec.d)
    if plot_spec.n is not None and "n" in df.columns:
        filtered = filtered.filter(pl.col("n") == plot_spec.n)
    return filtered


def _order_methods(methods):
    """Return methods ordered with FlashSketch first."""
    ordered = []
    for method in METHOD_ORDER:
        if method in methods:
            ordered.append(method)
    remaining = sorted([method for method in methods if method not in ordered])
    return ordered + remaining


def _save_legend(handles, labels, output_path, title, font_size, ncol=1):
    """Save a legend-only figure."""
    fig = plt.figure()
    legend = fig.legend(
        handles=handles,
        labels=labels,
        title=title,
        loc="center",
        frameon=False,
        fontsize=font_size,
        title_fontsize=font_size,
        ncol=ncol,
    )
    fig.canvas.draw()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


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
    method_names = sorted(grouped["method"].unique().to_list())
    k_values = sorted(GLOBAL_K_VALUES)
    method_names = _order_methods(method_names)

    display_methods = format_method_labels(method_names)
    method_label_map = dict(zip(method_names, display_methods))

    color_map = _make_color_map(method_names)
    style_map, variants_by_method, variant_fields = _build_style_map(grouped, color_map)
    marker_map = _make_marker_map(k_values)

    fig_width = config.base_width_in
    fig_height = config.base_height_in
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=0.24, right=0.70)

    for method in method_names:
        variants = _sort_variants(method, variants_by_method.get(method, []), variant_fields)
        for variant in variants:
            variant_mask = _variant_filter_expr(variant, variant_fields)
            method_df = grouped.filter((pl.col("method") == method) & variant_mask).sort("k")
            if method_df.is_empty():
                continue
            style = style_map[(method, variant)]
            xs = []
            ys = []
            for row in method_df.to_dicts():
                x = row["time_mean"]
                y = row["metric_mean"]
                xs.append(x)
                ys.append(y)
                x_log_std = row.get("time_log_std")
                y_log_std = row.get("metric_log_std")
                x_log_std = (
                    float(x_log_std) if x_log_std is not None and math.isfinite(x_log_std) else 0.0
                )
                y_log_std = (
                    float(y_log_std) if y_log_std is not None and math.isfinite(y_log_std) else 0.0
                )
                width_log = 2.0 * config.std_multiplier * x_log_std
                height_log = 2.0 * config.std_multiplier * y_log_std
                if width_log > 0 or height_log > 0:
                    points = _log_ellipse_points(x, y, width_log, height_log)
                    if points is not None:
                        ellipse = Polygon(
                            points,
                            closed=True,
                            facecolor=style["color"],
                            edgecolor=style["color"],
                            alpha=config.ellipse_alpha * style["alpha"],
                            linewidth=0.8,
                            zorder=1,
                        )
                        ax.add_patch(ellipse)
                if row["k"] not in marker_map:
                    raise ValueError(f"Encountered k={row['k']} not in global k sweep.")
                ax.scatter(
                    x,
                    y,
                    marker=marker_map[row["k"]],
                    s=MARKER_SIZE,
                    color=style["color"],
                    alpha=style["alpha"],
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=2,
                )
            ax.plot(
                xs,
                ys,
                color=style["color"],
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
                alpha=style["alpha"],
            )

    ax.set_xlabel(time_label, fontsize=config.min_font_size + 1)
    ax.set_ylabel(plot_spec.metric.label, fontsize=config.min_font_size + 1)
    task_label = config.task_label_map.get(plot_spec.task, plot_spec.task)
    title = _wrap_title(plot_spec.title)
    ax.set_title(f"{title}\n{task_label}", fontsize=config.min_font_size + 1)
    ax.tick_params(axis="both", labelsize=config.min_font_size)
    ax.set_axisbelow(True)
    ax.grid(alpha=0.25)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _set_axis_limits(ax, filtered, plot_spec.metric.name, time_col, config, log_scale=True)
    if plot_spec.task == g.TASK_SKETCH_SOLVE_LS:
        y_min, y_max = ax.get_ylim()
        if y_min < 3.0:
            ax.set_ylim(y_min, min(y_max, 3.0))

    legend_handles = []
    legend_labels = []
    for method in method_names:
        variants = _sort_variants(method, variants_by_method.get(method, []), variant_fields)
        if not variants:
            continue
        display = method_label_map[method]
        if len(variants) == 1:
            variant = variants[0]
            style = style_map[(method, variant)]
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                )
            )
            legend_labels.append(display)
            continue

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color_map[method],
                linestyle="-",
                linewidth=DEFAULT_LINEWIDTH,
            )
        )
        legend_labels.append(display)

        for variant in variants:
            label = _variant_label(method, variant, variant_fields)
            if not label:
                continue
            style = style_map[(method, variant)]
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                )
            )
            legend_labels.append(f"  {label}")

    k_handles = [
        Line2D([0], [0], color="black", marker=marker_map[k], linestyle="none", label=f"{k}")
        for k in k_values
    ]

    legend1 = None
    legend2 = None
    if config.include_legends:
        legend1 = ax.legend(
            handles=legend_handles,
            labels=legend_labels,
            fontsize=config.min_font_size - 1,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            ncol=1,
            columnspacing=0.8,
            handletextpad=0.6,
            handlelength=2.6,
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

    audit = _audit_plot(fig, ax, method_names, plot_spec.metric.label, config)

    ensure_dir(Path(output_path).parent)
    output_path = Path(output_path)
    extra_artists = ()
    if config.include_legends and legend1 is not None and legend2 is not None:
        extra_artists = (legend1, legend2)
    output_paths = []
    save_main = not config.export_legend_only
    if save_main:
        save_kwargs = {}
        save_kwargs["bbox_inches"] = "tight"
        if config.include_legends and extra_artists:
            save_kwargs["bbox_extra_artists"] = extra_artists
        fig.savefig(output_path, **save_kwargs)
        png_path = output_path.with_suffix(".png")
        fig.savefig(png_path, dpi=300, **save_kwargs)
        output_paths.extend([output_path, png_path])
    plt.close(fig)

    if config.export_legend_only:
        legend_methods_path = output_path.with_name(f"{output_path.stem}_legend_methods.pdf")
        legend_k_path = output_path.with_name(f"{output_path.stem}_legend_k.pdf")
        _save_legend(
            legend_handles,
            legend_labels,
            legend_methods_path,
            None,
            config.min_font_size - 1,
            ncol=2,
        )
        _save_legend(
            k_handles,
            [f"{k}" for k in k_values],
            legend_k_path,
            "Sketch dimension (k)",
            config.min_font_size - 1,
        )
        legend_methods_png = legend_methods_path.with_suffix(".png")
        legend_k_png = legend_k_path.with_suffix(".png")
        _save_legend(
            legend_handles,
            legend_labels,
            legend_methods_png,
            None,
            config.min_font_size - 1,
            ncol=2,
        )
        _save_legend(
            k_handles,
            [f"{k}" for k in k_values],
            legend_k_png,
            "Sketch dimension (k)",
            config.min_font_size - 1,
        )
        output_paths.extend([legend_methods_path, legend_methods_png, legend_k_path, legend_k_png])

    return audit, output_paths, sample_count


def _update_manifest(
    entries, run_paths, git_state, tree_hashes, gpu_info, config, make_target, script_path
):
    """Update the figures manifest with provenance information."""
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
            "gpu": gpu_info["name"],
            "gpu_slug": gpu_info["slug"],
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
    """Generate ablation figures for the supplied config."""
    logger = get_logger(make_target)
    required_columns = {"task", "method", "k", "dataset", "dataset_name", "dataset_group", "d", "n"}
    required_columns.update(config.method_label_fields)

    required_tasks = {plot.task for plot in config.plots}
    required_datasets = [
        (plot.dataset, plot.dataset_name, plot.dataset_group) for plot in config.plots
    ]

    run_paths = _resolve_run_paths(config, required_columns, required_datasets, required_tasks)
    df = _load_data(run_paths)
    if df.is_empty():
        raise ValueError("No data loaded for ablation plotting.")

    entries = []
    output_paths = []
    git_state = get_git_state()
    tree_hashes = get_tree_hashes(TREE_HASH_PATHS)
    gpu_info = get_gpu_info()

    for plot_spec in config.plots:
        output_name = f"{plot_spec.output_name}_{gpu_info['slug']}"
        output_path = g.FIGURES_DIR() / f"{output_name}.pdf"
        try:
            audit, outputs, sample_count = _plot_metric_task(df, plot_spec, output_path, config)
        except ValueError as exc:
            logger.warning("Skipping plot %s: %s", plot_spec.name, exc)
            continue
        time_col, time_label = _time_col_for_task(plot_spec.task)
        entries.append(
            {
                "output_name": output_name,
                "metric": plot_spec.metric.name,
                "task": plot_spec.task,
                "time_axis": time_label,
                "time_col": time_col,
                "sample_count": sample_count,
                "audit": audit,
            }
        )
        output_paths.extend(outputs)
        for path in outputs:
            logger.info("Generated %s", path)
            path_obj = Path(path)
            if path_obj.suffix.lower() == ".pdf":
                maybe_copy_pdf_to_paper(path_obj, subdir="appendix")

    script_path = script_path or ""
    _update_manifest(
        entries, run_paths, git_state, tree_hashes, gpu_info, config, make_target, script_path
    )

    return output_paths


def generate_legends(config, make_target, script_path=None):
    """Generate legend-only ablation figures for the supplied config."""
    legend_config = replace(config, include_legends=False, export_legend_only=True)
    return generate_figures(legend_config, make_target, script_path)
