from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from dataclasses import dataclass

import globals as g
from bench.ablation.config_ridge_regression import CONFIG as RIDGE_CONFIG
from bench.ablation.config_gram import CONFIG as GRAM_CONFIG
from bench.ablation.config_sketch_solve import CONFIG as SKETCH_SOLVE_CONFIG
from bench.ablation.config_ose_error import CONFIG as OSE_CONFIG


@dataclass(frozen=True)
class TableSpec:
    """Specification for an ablation summary table."""

    task: str
    metric: str
    time_col: str
    output_name: str


TABLES = [
    TableSpec(
        task=g.TASK_SKETCH_SOLVE_LS,
        metric="residual",
        time_col="total_time_ms",
        output_name="fig_ablation_summary_table_sketch_and_solve",
    ),
    TableSpec(
        task=g.TASK_RIDGE_REGRESSION,
        metric="residual",
        time_col="total_time_ms",
        output_name="fig_ablation_summary_table_ridge_regression",
    ),
]


DATASETS = RIDGE_CONFIG.datasets
TASK_LABELS = {
    g.TASK_SKETCH_SOLVE_LS: "Sketch+Solve",
    g.TASK_RIDGE_REGRESSION: "Sketch-and-ridge-regression",
    g.TASK_OSE_ERROR: "OSE",
    g.TASK_GRAM_APPROX: "Gram matrix approximation",
}

SPEEDUP_TASKS = [
    TableSpec(
        task=g.TASK_SKETCH_SOLVE_LS,
        metric="residual",
        time_col="total_time_ms",
        output_name="randnla_speedup_summary",
    ),
    TableSpec(
        task=g.TASK_RIDGE_REGRESSION,
        metric="residual",
        time_col="total_time_ms",
        output_name="randnla_speedup_summary",
    ),
    TableSpec(
        task=g.TASK_OSE_ERROR,
        metric="ose_spec_err",
        time_col="sketch_time_ms",
        output_name="randnla_speedup_summary",
    ),
    TableSpec(
        task=g.TASK_GRAM_APPROX,
        metric="gram_fro_rel",
        time_col="sketch_time_ms",
        output_name="randnla_speedup_summary",
    ),
]

SPEEDUP_OUTPUT_NAME = "randnla_speedup_summary"
SPEEDUP_LABEL = "tab:randnla_speedup_summary"

BASE_REQUIRED_COLUMNS = {
    "task",
    "method",
    "k",
    "dataset",
    "dataset_name",
    "dataset_group",
    "d",
    "n",
    "kappa",
    "s",
    "seed",
}
