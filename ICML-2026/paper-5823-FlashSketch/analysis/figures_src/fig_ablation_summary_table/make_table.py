from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

import globals as g
from analysis.figures_src.ablation.plotting import FLASH_METHODS, SJLT_METHODS, TREE_HASH_PATHS
from analysis.figures_src.camera_ready.utils import get_gpu_info
from analysis.figures_src.fig_ablation_summary_table.config import (
    BASE_REQUIRED_COLUMNS,
    DATASETS,
    SPEEDUP_LABEL,
    SPEEDUP_OUTPUT_NAME,
    SPEEDUP_TASKS,
    TABLES,
    TASK_LABELS,
)
from analysis.figures_src.method_labels import format_method_labels
from analysis.figures_src.run_selection import find_latest_results
from io_utils import ensure_dir, read_json, write_json
from logging_utils import get_logger
from provenance import get_git_state, get_tree_hashes


_LOGGER = get_logger("fig.ablation.summary-table")

BASELINE_METHODS = [
    g.METHOD_SJLT_CUSPARSE,
    g.METHOD_SJLT_GRASS_KERNEL,
    g.METHOD_SRHT_FWHT,
    g.METHOD_GAUSSIAN_DENSE_CUBLAS,
]
BASELINE_HEADER_MAP = {
    g.METHOD_SJLT_CUSPARSE: r"\shortstack{SJLT\\(cuSPARSE)}",
    g.METHOD_SJLT_GRASS_KERNEL: r"\shortstack{SJLT\\(GraSS Kernel)}",
    g.METHOD_SRHT_FWHT: r"\shortstack{Subsampled\\FHT}",
    g.METHOD_GAUSSIAN_DENSE_CUBLAS: r"\shortstack{Dense Gaussian\\(cuBLAS)}",
}
TASK_TABLE_LABELS = {
    g.TASK_SKETCH_SOLVE_LS: r"\shortstack{Sketch+\\Solve}",
    g.TASK_RIDGE_REGRESSION: r"\shortstack{Sketch-and-\\ridge regression}",
    g.TASK_OSE_ERROR: "OSE",
    g.TASK_GRAM_APPROX: r"\shortstack{Gram matrix\\approximation}",
}


def _dataset_label_expr():
    """Return a Polars expression for dataset labels."""
    base = (
        pl.when(pl.col("dataset_group") != "")
        .then(pl.format("{}/{}", pl.col("dataset_group"), pl.col("dataset_name")))
        .otherwise(pl.col("dataset_name"))
    )
    return (
        pl.when(pl.col("d").is_not_null() & pl.col("n").is_not_null())
        .then(pl.format("{} ({}x{})", base, pl.col("d"), pl.col("n")))
        .otherwise(base)
        .alias("dataset_label")
    )


def _variant_label_expr():
    """Return a Polars expression for variant label strings."""
    flash = pl.col("method").is_in(list(FLASH_METHODS))
    sjlt = pl.col("method").is_in(list(SJLT_METHODS))
    return (
        pl.when(flash)
        .then(pl.format("kappa={}, s={}", pl.col("kappa"), pl.col("s")))
        .when(sjlt)
        .then(pl.format("s={}", pl.col("s")))
        .otherwise(pl.lit(""))
        .alias("variant_label")
    )


def _format_method_labels(df):
    """Add a camera-ready method label column to df."""
    df = df.with_columns(_variant_label_expr())
    df = df.with_columns(
        pl.concat_str([pl.col("method"), pl.col("variant_label")], separator=" ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("method_label_raw")
    )
    labels = df.select("method_label_raw").unique().to_series().to_list()
    formatted = format_method_labels(labels)
    label_map = dict(zip(labels, formatted))
    return df.with_columns(pl.col("method_label_raw").replace(label_map).alias("method_label"))


def _render_table(df, title, output_path):
    """Render a summary table into PDF/PNG outputs."""
    if df.is_empty():
        raise ValueError("No rows to render in ablation summary table.")

    rows = df.to_dicts()
    col_labels = ["dataset", "method", "time_ms", "residual", "count"]
    cell_text = [
        [
            row["dataset_label"],
            row["method_label"],
            f"{row['time_ms']:.4f}",
            f"{row['metric']:.4g}",
            f"{int(row['count'])}",
        ]
        for row in rows
    ]

    fig_height = max(2.5, 0.22 * len(rows))
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)
    ax.set_title(title, fontsize=10, pad=12)

    ensure_dir(Path(output_path).parent)
    output_path = Path(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return [output_path, png_path]


def _update_manifest(entries, git_state, tree_hashes, gpu_info):
    """Update the figures manifest with ablation summary-table entries."""
    manifest_path = g.FIGURE_MANIFEST_PATH()
    if manifest_path.exists():
        manifest = read_json(manifest_path)
    else:
        manifest = {}

    for entry in entries:
        run_paths = entry["run_path"]
        if isinstance(run_paths, (list, tuple)):
            inputs = [str(Path(path).relative_to(g.REPO_ROOT)) for path in run_paths]
        else:
            inputs = [str(Path(run_paths).relative_to(g.REPO_ROOT))]
        manifest[entry["output_name"]] = {
            "script": str(Path(__file__).relative_to(g.REPO_ROOT)),
            "commit": git_state["commit"],
            "dirty": git_state["dirty"],
            "tree_hashes": tree_hashes,
            "gpu": gpu_info["name"],
            "gpu_slug": gpu_info["slug"],
            "inputs": inputs,
            "sample_count": entry["sample_count"],
            "task": entry["task"],
            "make_target": "fig.ablation.summary-table",
        }

    ensure_dir(manifest_path.parent)
    write_json(manifest_path, manifest)


def _compute_speedups(df, table_specs):
    """Return a dataframe of FlashSketch speedups over each baseline per task."""
    if df.is_empty():
        raise ValueError("No rows available for speedup summary table.")

    speedup_rows = []
    sample_counts = {}
    for spec in table_specs:
        task_df = df.filter(pl.col("task") == spec.task)
        task_df = task_df.filter(pl.col(spec.time_col).is_not_null())
        task_df = task_df.filter(pl.col(spec.metric).is_not_null())
        if task_df.is_empty():
            continue

        key_cols = [
            "dataset",
            "dataset_name",
            "dataset_group",
            "d",
            "n",
            "k",
        ]
        method_cols = key_cols + ["method"]
        variant_cols = [field for field in ("kappa", "s") if field in task_df.columns]
        group_cols = method_cols + variant_cols

        grouped = (
            task_df.group_by(group_cols)
            .agg(
                [
                    pl.col(spec.time_col).mean().alias("time_mean"),
                    pl.col(spec.metric).mean().alias("metric_mean"),
                ]
            )
        )

        required_methods = [g.METHOD_FLASH_SKETCH, *BASELINE_METHODS]
        available = (
            grouped.filter(pl.col("method").is_in(required_methods))
            .group_by(key_cols)
            .agg(pl.n_unique("method").alias("method_count"))
            .filter(pl.col("method_count") == len(required_methods))
            .select(key_cols)
        )
        if available.is_empty():
            continue
        grouped = grouped.join(available, on=key_cols, how="inner")

        min_times = (
            grouped.group_by(method_cols)
            .agg(pl.col("time_mean").min().alias("time_min"))
        )
        best = grouped.join(min_times, on=method_cols, how="inner")
        best = best.filter(pl.col("time_mean") == pl.col("time_min"))
        best = best.group_by(method_cols).agg(
            [
                pl.first("time_mean").alias("time_mean"),
                pl.first("metric_mean").alias("metric_mean"),
            ]
        )

        flash = best.filter(pl.col("method") == g.METHOD_FLASH_SKETCH).rename(
            {"time_mean": "flash_time"}
        )
        if flash.is_empty():
            continue
        flash_samples = task_df.filter(
            pl.col("method") == g.METHOD_FLASH_SKETCH
        ).join(available, on=key_cols, how="inner").height
        sample_counts[spec.task] = flash_samples

        for baseline in BASELINE_METHODS:
            base_df = best.filter(pl.col("method") == baseline).rename(
                {"time_mean": "baseline_time"}
            )
            if base_df.is_empty():
                continue
            joined = flash.join(
                base_df.select(
                    [
                        "dataset",
                        "dataset_name",
                        "dataset_group",
                        "d",
                        "n",
                        "k",
                        "baseline_time",
                    ]
                ),
                on=key_cols,
                how="inner",
            )
            joined = joined.with_columns(
                (pl.col("baseline_time") / pl.col("flash_time")).alias("speedup")
            )
            joined = joined.filter(
                pl.col("speedup").is_finite() & (pl.col("speedup") > 0)
            )
            if joined.is_empty():
                continue
            speedup_rows.append(
                joined.select(
                    key_cols + ["speedup"]
                ).with_columns(
                    [
                        pl.lit(spec.task).alias("task"),
                        pl.lit(baseline).alias("baseline"),
                    ]
                )
            )

    if not speedup_rows:
        return pl.DataFrame(), {}

    return pl.concat(speedup_rows, how="vertical"), sample_counts


def _render_speedup_table(df, output_path, sample_count):
    """Write the speedup summary LaTeX table."""
    if df.is_empty():
        raise ValueError("No data available for speedup summary table.")

    key_cols = [
        "dataset",
        "dataset_name",
        "dataset_group",
        "d",
        "n",
        "k",
        "task",
    ]
    next_best = (
        df.group_by(key_cols)
        .agg(pl.col("speedup").min().alias("speedup_best"))
        .filter(pl.col("speedup_best").is_finite() & (pl.col("speedup_best") > 0))
    )
    if next_best.is_empty():
        raise ValueError("No data available for next-best baseline speedups.")
    next_best_geo = float(next_best.select(pl.col("speedup_best").log().mean().exp()).item())

    agg = (
        df.group_by(["task", "baseline"])
        .agg(
            [
                pl.col("speedup").log().mean().exp().alias("speedup_geo"),
                pl.len().alias("count"),
            ]
        )
        .sort(["task", "baseline"])
    )

    baseline_map = {base: BASELINE_HEADER_MAP.get(base, base) for base in BASELINE_METHODS}
    baselines = BASELINE_METHODS

    col_spec = "c" * (len(baselines) + 1)
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        (
            "  \\caption{Geomean speedups of \\method vs baselines "
            "aggregated over shapes, datasets and configs. "
            f"\\textbf{{Global geomean vs next best baseline: {next_best_geo:.2f}x.}}}}"
        ),
        f"  \\label{{{SPEEDUP_LABEL}}}",
        "  \\small",
        "  \\setlength{\\tabcolsep}{4pt}",
        "  \\resizebox{\\linewidth}{!}{%",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        "    Task & "
        + " & ".join(baseline_map[base] for base in baselines)
        + " \\\\",
        "    \\midrule",
    ]

    tasks = agg.select("task").unique().to_series().to_list()
    total_cols = len(baselines) + 1
    rule = f"    \\cmidrule(lr){{1-{total_cols}}}"
    for idx, task in enumerate(tasks):
        task_label = TASK_TABLE_LABELS.get(task, TASK_LABELS.get(task, task))
        row = agg.filter(pl.col("task") == task)
        values = []
        for baseline in baselines:
            entry = row.filter(pl.col("baseline") == baseline)
            if entry.is_empty():
                values.append("--")
            else:
                values.append(f"{entry.item(0, 'speedup_geo'):.2f}")
        lines.append(f"    {task_label} & " + " & ".join(values) + " \\\\")
        if idx < len(tasks) - 1:
            lines.extend(["    \\addlinespace[0.2em]", rule])

    lines.extend(["    \\bottomrule", "  \\end{tabular}}", "\\end{table}"])

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main():
    """Generate ablation summary tables for sketch-and-solve and ridge regression."""
    gpu_info = get_gpu_info()
    tree_hashes = get_tree_hashes(TREE_HASH_PATHS)
    git_state = get_git_state()

    required_datasets = []
    for entry in DATASETS:
        cfg = entry[0] if isinstance(entry, tuple) else entry
        required_datasets.append((cfg.dataset, cfg.name, getattr(cfg, "group", None)))
    entries = []

    for table in TABLES:
        required_cols = set(BASE_REQUIRED_COLUMNS)
        required_cols.update({table.metric, table.time_col})
        run_path = find_latest_results(
            required_cols,
            required_datasets=required_datasets,
            required_tasks={table.task},
            tree_hash_paths=TREE_HASH_PATHS,
        )
        df = pl.read_parquet(run_path).filter(pl.col("task") == table.task)
        df = df.with_columns(_dataset_label_expr())
        df = _format_method_labels(df)

        agg = (
            df.group_by(["dataset_label", "method_label"])
            .agg(
                [
                    pl.col(table.time_col).mean().alias("time_ms"),
                    pl.col(table.metric).mean().alias("metric"),
                    pl.len().alias("count"),
                ]
            )
            .sort(["dataset_label", "time_ms"])
        )
        title = f"Ablation summary — {TASK_LABELS[table.task]}"
        output_name = f"{table.output_name}_{gpu_info['slug']}"
        output_path = g.FIGURES_DIR() / f"{output_name}.pdf"
        outputs = _render_table(agg, title, output_path)
        for path in outputs:
            _LOGGER.info("Generated %s", path)

        entries.append(
            {
                "output_name": output_name,
                "run_path": str(run_path),
                "sample_count": int(df.height),
                "task": table.task,
            }
        )

    speedup_frames = []
    speedup_run_paths = []
    speedup_counts = {}
    for spec in SPEEDUP_TASKS:
        speedup_required_cols = set(BASE_REQUIRED_COLUMNS)
        speedup_required_cols.update({spec.metric, spec.time_col})
        speedup_run_path = find_latest_results(
            speedup_required_cols,
            required_datasets=required_datasets,
            required_tasks={spec.task},
            tree_hash_paths=TREE_HASH_PATHS,
        )
        speedup_run_paths.append(str(speedup_run_path))
        df_task = pl.read_parquet(speedup_run_path)
        speedups_task, counts = _compute_speedups(df_task, [spec])
        speedup_frames.append(speedups_task)
        speedup_counts.update(counts)
    speedups = (
        pl.concat(speedup_frames, how="vertical") if speedup_frames else pl.DataFrame()
    )
    if speedup_counts:
        unique_counts = sorted({count for count in speedup_counts.values()})
        if len(unique_counts) > 1:
            raise ValueError(
                f"Speedup sample counts differ across tasks: {speedup_counts}"
            )
        sample_count = unique_counts[0]
    else:
        sample_count = 0
    latex_path = g.PAPER_TABLES_DIR() / f"{SPEEDUP_OUTPUT_NAME}.tex"
    speedup_out = _render_speedup_table(speedups, latex_path, sample_count)
    entries.append(
        {
            "output_name": SPEEDUP_OUTPUT_NAME,
            "run_path": speedup_run_paths,
            "sample_count": int(speedups.height),
            "task": "randnla_speedup_summary",
        }
    )

    _LOGGER.info("Generated %s", speedup_out)
    _update_manifest(entries, git_state, tree_hashes, gpu_info)


if __name__ == "__main__":
    main()
