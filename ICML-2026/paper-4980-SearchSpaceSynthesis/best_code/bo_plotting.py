# =============================================================================
# EXPERIMENT PLOTTING
# =============================================================================
#
# This module generates ICML-quality plots for the Bayesian Optimization
# experiments. The actual matplotlib code is intentionally NOT implemented in
# this PR — the bo_runner refactor only had to GUARANTEE that all data needed
# for the planned plots is persisted to CSV. Each plot function below raises
# NotImplementedError until the camera-ready plotting style is finalised.
#
#
# =============================================================================
# INPUT DATA (post-aggregation, written by bo_cli aggregate)
# =============================================================================
#
# `aggregated_trace.csv`   columns
#     experiment_id, method, target_name, kernel_name, seed, evaluation,
#     objective_value, best_objective_value,
#     bo_overhead_time, objective_eval_time, cumulative_wallclock_time,
#     distance_to_best, distance_to_topk
#
# `aggregated_ranking.csv` columns
#     experiment_id, method, target_name, kernel_name, seed, evaluation,
#     kendall_tau, spearman_rho
#
#
# =============================================================================
# PLOTS
# =============================================================================
#
# Plot 1    Optimization performance  : evaluation              vs best_objective_value
# Plot 2    Runtime efficiency        : cumulative_wallclock_time vs best_objective_value
# Plot 3    Kendall tau               : evaluation              vs kendall_tau
# Plot 4    Spearman correlation      : evaluation              vs spearman_rho
# Plot 5    Distance to best program  : evaluation              vs distance_to_best
# Plot 6    Distance to top-k programs: evaluation              vs distance_to_topk
#
#
# =============================================================================
# STYLE
# =============================================================================
#
# Use matplotlib with gridlines. Save figures as PDF for ICML submission.
#
# =============================================================================
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_aggregated_results(results_root: str | Path) -> dict[str, pd.DataFrame]:
    """Load the most recent aggregated_trace.csv / aggregated_ranking.csv.

    Searches `results_root/aggregated__*` subdirectories (most recent timestamp
    wins) for the canonical aggregated CSV files written by
    ``bo_cli.aggregate_command``. Returns a dict with keys ``trace`` and
    ``ranking``; missing files yield empty DataFrames so downstream plotting
    code can branch cheaply.
    """
    base = Path(results_root)
    if not base.exists():
        return {"trace": pd.DataFrame(), "ranking": pd.DataFrame()}

    aggregated_dirs = sorted(base.glob("aggregated__*"), reverse=True)
    if not aggregated_dirs:
        return {"trace": pd.DataFrame(), "ranking": pd.DataFrame()}

    latest = aggregated_dirs[0]

    def _safe_read(path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    return {
        "trace": _safe_read(latest / "aggregated_trace.csv"),
        "ranking": _safe_read(latest / "aggregated_ranking.csv"),
    }


def plot_optimization_curve(trace_df: pd.DataFrame, output_dir: str | Path) -> None:
    """Plot 1: best_objective_value vs evaluation, grouped by method × target × kernel.

    Reads columns: evaluation, best_objective_value, method, target_name, kernel_name, seed.
    """
    raise NotImplementedError(
        "plot_optimization_curve: implement when the ICML plotting style is finalised."
    )


def plot_runtime_curve(trace_df: pd.DataFrame, output_dir: str | Path) -> None:
    """Plot 2: best_objective_value vs cumulative_wallclock_time, grouped by method × target × kernel.

    Reads columns: cumulative_wallclock_time, best_objective_value, method, target_name, kernel_name, seed.
    """
    raise NotImplementedError(
        "plot_runtime_curve: implement when the ICML plotting style is finalised."
    )


def plot_ranking_metrics(ranking_df: pd.DataFrame, output_dir: str | Path) -> None:
    """Plots 3 + 4: Kendall τ and Spearman ρ vs evaluation, grouped by method × target × kernel.

    Reads columns: evaluation, kendall_tau, spearman_rho, method, target_name, kernel_name, seed.
    """
    raise NotImplementedError(
        "plot_ranking_metrics: implement when the ICML plotting style is finalised."
    )


def plot_distance_curves(trace_df: pd.DataFrame, output_dir: str | Path) -> None:
    """Plots 5 + 6: distance_to_best and distance_to_topk vs evaluation.

    Reads columns: evaluation, distance_to_best, distance_to_topk, method, target_name, kernel_name, seed.
    """
    raise NotImplementedError(
        "plot_distance_curves: implement when the ICML plotting style is finalised."
    )
