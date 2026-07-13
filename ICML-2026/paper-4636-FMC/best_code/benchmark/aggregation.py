"""Aggregation module for benchmark experiments (Phase 4).

This module handles aggregating results from all methods and variants
into unified tables and comparison summaries.

Usage:
    aggregator = BenchmarkAggregator.from_config(cfg)

    # Aggregate results for a task
    results = aggregator.aggregate_task("pendulum")

    # Generate comparison tables
    aggregator.generate_comparison_table(tasks=["pendulum", "gaussian"])

    # Export to LaTeX
    aggregator.export_latex_table(tasks=["pendulum"])
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from core.checkpointing import CheckpointManager, ResultsPath
from utils.data_split import DEFAULT_CALIBRATION_SIZES


@dataclass
class AggregationConfig:
    """Configuration for aggregation."""

    # Default metrics (joint) for tasks without reference posterior
    metrics: List[str] = field(default_factory=lambda: [
        "joint_c2st", "joint_mmd", "joint_wasserstein"
    ])

    # Conditional metrics for tasks with sample_reference_posterior
    conditional_metrics: List[str] = field(default_factory=lambda: [
        "cond_c2st", "cond_mmd"
    ])

    # Tasks that should use conditional metrics
    conditional_tasks: List[str] = field(default_factory=list)

    # Aggregation parameters
    calibration_sizes: List[int] = field(default_factory=lambda: DEFAULT_CALIBRATION_SIZES.copy())
    seeds: List[int] = field(default_factory=lambda: [33, 43, 53])

    # Statistics to compute
    compute_mean: bool = True
    compute_std: bool = True
    compute_median: bool = False
    compute_iqr: bool = False

    # Paths
    results_dir: Path = Path("results")
    experiment_name: str = "comparison_benchmark"

    def get_metrics_for_task(self, task: str) -> List[str]:
        """Get the appropriate metrics for a task."""
        if task in self.conditional_tasks:
            return self.conditional_metrics
        return self.metrics

    @classmethod
    def from_config(cls, cfg) -> "AggregationConfig":
        """Create from Hydra config."""
        eval_cfg = cfg.get("evaluation", {})

        return cls(
            metrics=list(eval_cfg.get("metrics", ["joint_c2st", "joint_mmd", "joint_wasserstein"])),
            conditional_metrics=list(eval_cfg.get("conditional_metrics", ["cond_c2st", "cond_mmd"])),
            conditional_tasks=list(eval_cfg.get("conditional_tasks", [])),
            calibration_sizes=list(cfg.get("data", {}).get(
                "calibration_sizes", DEFAULT_CALIBRATION_SIZES
            )),
            seeds=list(cfg.get("seeds", [33, 43, 53])),
            results_dir=Path(cfg.get("results_dir", "results")),
            experiment_name=cfg.get("experiment", {}).get("name", "comparison_benchmark"),
        )


class BenchmarkAggregator:
    """Handles Phase 4: Aggregation of benchmark results."""

    def __init__(self, config: AggregationConfig):
        """Initialize aggregator.

        Args:
            config: Aggregation configuration
        """
        self.config = config

        # Setup paths and checkpoint manager
        self.paths = ResultsPath(config.results_dir, config.experiment_name)
        self.checkpoint_manager = CheckpointManager(self.paths)

    @classmethod
    def from_config(cls, cfg) -> "BenchmarkAggregator":
        """Create aggregator from Hydra config."""
        return cls(AggregationConfig.from_config(cfg))

    def aggregate_task(
        self,
        task: str,
        include_baselines: bool = True,
        include_simulation_models: bool = True,
        variants: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Aggregate all results for a task.

        Args:
            task: Task name
            include_baselines: Include baseline results
            include_simulation_models: Include simulation model results
            variants: List of variants to include (defaults to all)

        Returns:
            Aggregated results organized by method
        """
        print(f"\nAggregating results for {task}...")

        results = {
            "task": task,
            "methods": {},
            "aggregated_at": datetime.now().isoformat(),
        }

        # Collect simulation model results
        if include_simulation_models:
            for model_type in ["npe", "fmpe"]:
                if self.checkpoint_manager.shared_simulation_model_exists(task, model_type):
                    sim_path = self.paths.shared_simulation_model_path(task, model_type)
                    metrics_path = sim_path / "metrics.json"
                    if metrics_path.exists():
                        with open(metrics_path) as f:
                            metrics = json.load(f)
                        results["methods"][model_type] = {
                            "type": "simulation",
                            "metrics": metrics.get("metrics", {}),
                        }

        # Collect baseline results
        if include_baselines:
            baselines = self.checkpoint_manager.list_shared_baselines(
                task, self.config.seeds[0] if self.config.seeds else 0
            )

            for baseline in baselines:
                baseline_metrics = self._aggregate_method_metrics(
                    task, baseline, is_baseline=True
                )
                if baseline_metrics:
                    results["methods"][baseline] = {
                        "type": "baseline",
                        **baseline_metrics,
                    }

        # Collect variant results
        if variants is None:
            variants = self.checkpoint_manager.list_variants()

        for variant_name in variants:
            variant_cfg = self.checkpoint_manager.load_variant_config(variant_name)
            if variant_cfg is None:
                continue

            method = variant_cfg.get("method", "fm_post_transform")
            variant_metrics = self._aggregate_variant_metrics(task, variant_name, method)

            if variant_metrics:
                results["methods"][variant_name] = {
                    "type": "variant",
                    "method": method,
                    "base_dist": variant_cfg.get("base_dist", "unknown"),
                    **variant_metrics,
                }

        # Save aggregated results
        self.checkpoint_manager.save_benchmark_aggregated_metrics(
            task, results, "all_methods_metrics.json"
        )

        return results

    def _aggregate_method_metrics(
        self,
        task: str,
        method: str,
        is_baseline: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Aggregate metrics across seeds and ncals for a method."""
        all_metrics = {}

        # Get task-specific metrics
        task_metrics = self.config.get_metrics_for_task(task)

        for ncal in self.config.calibration_sizes:
            ncal_metrics = {}

            seed_values = {metric: [] for metric in task_metrics}

            for seed in self.config.seeds:
                # Load metrics
                if not is_baseline:
                    raise ValueError(
                        "Non-baseline aggregation not supported here. "
                        "Use _aggregate_variant_metrics for variants."
                    )
                metrics = self.checkpoint_manager.load_shared_baseline_metrics(
                    task, seed, method, ncal
                )

                if metrics is None:
                    continue

                metric_data = metrics.get("metrics", {})
                for metric_name in task_metrics:
                    if metric_name in metric_data:
                        seed_values[metric_name].append(metric_data[metric_name])

            # Compute statistics
            for metric_name, values in seed_values.items():
                if not values:
                    continue

                values = np.array(values)
                stats = {}

                if self.config.compute_mean:
                    stats["mean"] = float(np.mean(values))
                if self.config.compute_std:
                    stats["std"] = float(np.std(values))
                if self.config.compute_median:
                    stats["median"] = float(np.median(values))
                if self.config.compute_iqr:
                    stats["iqr"] = float(np.percentile(values, 75) - np.percentile(values, 25))

                stats["n_seeds"] = len(values)
                ncal_metrics[metric_name] = stats

            if ncal_metrics:
                all_metrics[f"ncal_{ncal}"] = ncal_metrics

        return {"by_ncal": all_metrics} if all_metrics else None

    def _aggregate_variant_metrics(
        self,
        task: str,
        variant_name: str,
        method: str,
    ) -> Optional[Dict[str, Any]]:
        """Aggregate metrics for a variant."""
        all_metrics = {}

        # Get task-specific metrics
        task_metrics = self.config.get_metrics_for_task(task)

        for ncal in self.config.calibration_sizes:
            ncal_metrics = {}
            seed_values = {metric: [] for metric in task_metrics}

            for seed in self.config.seeds:
                metrics = self.checkpoint_manager.load_variant_metrics(
                    variant_name, task, seed, method, ncal
                )

                if metrics is None:
                    continue

                metric_data = metrics.get("metrics", {})
                for metric_name in task_metrics:
                    if metric_name in metric_data:
                        seed_values[metric_name].append(metric_data[metric_name])

            # Compute statistics
            for metric_name, values in seed_values.items():
                if not values:
                    continue

                values = np.array(values)
                stats = {}

                if self.config.compute_mean:
                    stats["mean"] = float(np.mean(values))
                if self.config.compute_std:
                    stats["std"] = float(np.std(values))
                if self.config.compute_median:
                    stats["median"] = float(np.median(values))
                if self.config.compute_iqr:
                    stats["iqr"] = float(np.percentile(values, 75) - np.percentile(values, 25))

                stats["n_seeds"] = len(values)
                ncal_metrics[metric_name] = stats

            if ncal_metrics:
                all_metrics[f"ncal_{ncal}"] = ncal_metrics

        return {"by_ncal": all_metrics} if all_metrics else None

    def generate_comparison_table(
        self,
        tasks: List[str],
        metric: str = "joint_c2st",
        ncal: int = 200,
        output_format: str = "csv",
    ) -> List[Dict[str, Any]]:
        """Generate comparison table across methods and tasks.

        Args:
            tasks: List of tasks
            metric: Metric to compare
            ncal: Calibration size to use
            output_format: Output format ('csv', 'dict')

        Returns:
            Table data as list of dicts
        """
        print(f"\nGenerating comparison table (metric={metric}, ncal={ncal})...")

        # First aggregate all tasks
        aggregated = {}
        for task in tasks:
            aggregated[task] = self.aggregate_task(task)

        # Build table rows
        rows = []

        # Collect all methods across tasks
        all_methods = set()
        for task in tasks:
            all_methods.update(aggregated[task]["methods"].keys())

        # Sort methods by type
        baselines = []
        variants = []
        sim_models = []

        for method in all_methods:
            # Check method type in first task that has it
            for task in tasks:
                if method in aggregated[task]["methods"]:
                    method_data = aggregated[task]["methods"][method]
                    if method_data.get("type") == "baseline":
                        baselines.append(method)
                    elif method_data.get("type") == "variant":
                        variants.append(method)
                    elif method_data.get("type") == "simulation":
                        sim_models.append(method)
                    break

        # Order: simulation models, baselines, variants
        ordered_methods = sorted(sim_models) + sorted(baselines) + sorted(variants)

        for method in ordered_methods:
            row = {"method": method}

            for task in tasks:
                if method not in aggregated[task]["methods"]:
                    row[task] = None
                    row[f"{task}_std"] = None
                    continue

                method_data = aggregated[task]["methods"][method]

                # Handle simulation models (direct metrics, no ncal dependency)
                if method_data.get("type") == "simulation":
                    metrics = method_data.get("metrics", {})
                    if metric in metrics:
                        row[task] = metrics[metric]
                        row[f"{task}_std"] = 0.0  # No std for simulation models
                    else:
                        row[task] = None
                        row[f"{task}_std"] = None
                else:
                    # Handle baselines/variants (by_ncal structure)
                    by_ncal = method_data.get("by_ncal", {})
                    ncal_key = f"ncal_{ncal}"

                    if ncal_key in by_ncal and metric in by_ncal[ncal_key]:
                        stats = by_ncal[ncal_key][metric]
                        row[task] = stats.get("mean")
                        row[f"{task}_std"] = stats.get("std")
                    else:
                        row[task] = None
                        row[f"{task}_std"] = None

            rows.append(row)

        # Save table
        if output_format == "csv":
            self._save_csv_table(rows, tasks, metric, ncal)

        return rows

    def _save_csv_table(
        self,
        rows: List[Dict],
        tasks: List[str],
        metric: str,
        ncal: int,
    ):
        """Save table as CSV."""
        import csv

        output_path = self.paths.benchmark_aggregated_path() / f"comparison_{metric}_ncal{ncal}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build fieldnames
        fieldnames = ["method"]
        for task in tasks:
            fieldnames.extend([task, f"{task}_std"])

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Saved comparison table to {output_path}")

    def export_latex_table(
        self,
        tasks: List[str],
        metric: str = "joint_c2st",
        ncal: int = 200,
        caption: Optional[str] = None,
        label: Optional[str] = None,
    ) -> str:
        """Export comparison table as LaTeX.

        Args:
            tasks: List of tasks
            metric: Metric to compare
            ncal: Calibration size
            caption: Optional table caption
            label: Optional table label

        Returns:
            LaTeX table string
        """
        rows = self.generate_comparison_table(tasks, metric, ncal, output_format="dict")

        # Build LaTeX
        latex_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
        ]

        if caption:
            latex_lines.append(f"\\caption{{{caption}}}")
        if label:
            latex_lines.append(f"\\label{{{label}}}")

        # Table header
        cols = "l" + "c" * len(tasks)
        latex_lines.append(f"\\begin{{tabular}}{{{cols}}}")
        latex_lines.append("\\toprule")
        latex_lines.append("Method & " + " & ".join(tasks) + " \\\\")
        latex_lines.append("\\midrule")

        # Find best values per task for bolding
        best_per_task = {}
        for task in tasks:
            values = [r[task] for r in rows if r[task] is not None]
            if values:
                # For C2ST, closer to 0.5 is better
                if "c2st" in metric.lower():
                    best_per_task[task] = min(values, key=lambda x: abs(x - 0.5))
                else:
                    best_per_task[task] = min(values)

        # Table rows
        for row in rows:
            cells = [row["method"].replace("_", "\\_")]

            for task in tasks:
                value = row[task]
                std = row.get(f"{task}_std")

                if value is None:
                    cells.append("--")
                else:
                    # Format value
                    if "c2st" in metric.lower():
                        is_best = abs(value - 0.5) == abs(best_per_task.get(task, value) - 0.5)
                    else:
                        is_best = value == best_per_task.get(task, float("inf"))

                    val_str = f"{value:.3f}"
                    if std is not None:
                        val_str += f" $\\pm$ {std:.3f}"

                    if is_best:
                        val_str = f"\\textbf{{{val_str}}}"

                    cells.append(val_str)

            latex_lines.append(" & ".join(cells) + " \\\\")

        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        latex_str = "\n".join(latex_lines)

        # Save
        output_path = self.paths.benchmark_aggregated_path() / f"table_{metric}_ncal{ncal}.tex"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(latex_str)

        print(f"Saved LaTeX table to {output_path}")

        return latex_str

    def generate_summary_report(
        self,
        tasks: List[str],
        output_path: Optional[Path] = None,
    ) -> str:
        """Generate a summary report of all benchmark results.

        Args:
            tasks: List of tasks
            output_path: Optional path to save report

        Returns:
            Report string
        """
        lines = [
            "=" * 60,
            "BENCHMARK SUMMARY REPORT",
            f"Generated: {datetime.now().isoformat()}",
            "=" * 60,
            "",
        ]

        for task in tasks:
            lines.extend([
                f"\n{'='*40}",
                f"Task: {task}",
                f"{'='*40}",
            ])

            results = self.aggregate_task(task)

            for method, data in results["methods"].items():
                lines.append(f"\n{method} ({data.get('type', 'unknown')}):")

                # Handle simulation models (direct metrics, no ncal)
                if data.get("type") == "simulation":
                    metrics = data.get("metrics", {})
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            lines.append(f"  {metric_name}: {value:.4f}")
                else:
                    # Handle baselines/variants (by_ncal structure)
                    by_ncal = data.get("by_ncal", {})
                    for ncal_key, metrics in sorted(by_ncal.items()):
                        lines.append(f"  {ncal_key}:")
                        for metric_name, stats in metrics.items():
                            mean = stats.get("mean", 0)
                            std = stats.get("std", 0)
                            lines.append(f"    {metric_name}: {mean:.4f} +/- {std:.4f}")

        report = "\n".join(lines)

        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)
            print(f"Saved report to {output_path}")
        else:
            default_path = self.paths.benchmark_aggregated_path() / "summary_report.txt"
            default_path.parent.mkdir(parents=True, exist_ok=True)
            with open(default_path, "w") as f:
                f.write(report)
            print(f"Saved report to {default_path}")

        return report


# Convenience functions


def aggregate_results(
    cfg,
    tasks: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Aggregate results for all tasks.

    Args:
        cfg: Hydra configuration
        tasks: List of tasks (defaults to cfg.tasks)

    Returns:
        Aggregated results
    """
    aggregator = BenchmarkAggregator.from_config(cfg)
    tasks = tasks or list(cfg.get("tasks", []))

    results = {}
    for task in tasks:
        results[task] = aggregator.aggregate_task(task)

    return results


def generate_tables(
    cfg,
    tasks: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    ncals: Optional[List[int]] = None,
) -> None:
    """Generate comparison tables for all metrics and ncals.

    Args:
        cfg: Hydra configuration
        tasks: List of tasks
        metrics: List of metrics (if not specified, uses task-specific metrics)
        ncals: List of ncal values
    """
    aggregator = BenchmarkAggregator.from_config(cfg)
    tasks = tasks or list(cfg.get("tasks", []))
    ncals = ncals or aggregator.config.calibration_sizes

    # Collect all relevant metrics across tasks
    if metrics is None:
        all_metrics = set()
        for task in tasks:
            task_metrics = aggregator.config.get_metrics_for_task(task)
            all_metrics.update(task_metrics)
        metrics = list(all_metrics)

    for metric in metrics:
        for ncal in ncals:
            aggregator.generate_comparison_table(tasks, metric=metric, ncal=ncal)
            aggregator.export_latex_table(tasks, metric=metric, ncal=ncal)


def generate_report(
    cfg,
    tasks: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
) -> str:
    """Generate summary report.

    Args:
        cfg: Hydra configuration
        tasks: List of tasks
        output_path: Optional output path

    Returns:
        Report string
    """
    aggregator = BenchmarkAggregator.from_config(cfg)
    tasks = tasks or list(cfg.get("tasks", []))
    return aggregator.generate_summary_report(tasks, output_path)
