"""Structured metrics for CPC-LLM pipeline logging.

Pydantic models define the experiment data schema. JSON serialization is the
source of truth; wandb consumes the nested dict (via ``model_dump()``) as a lightweight
secondary sink, logging keys like ``cpc/beta_t``, ``ar_sampling/n_accepted``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Pydantic models (serialization targets)
# ---------------------------------------------------------------------------


class SampleQualityMetrics(BaseModel):
    """Aggregate quality stats for a set of samples (accepted or rejected).

    Computed from the ``score`` column of proposal DataFrames. Parsability
    requires a non-NaN score; feasibility additionally excludes infinite scores.
    """

    model_config = ConfigDict(ser_json_inf_nan="constants")

    n_samples: int
    n_parsable: int
    n_feasible: int
    frac_feasible: float
    mean_score: float
    min_score: float
    max_score: float


class ARSamplingMetrics(BaseModel):
    """Metrics from acceptance-rejection / independent MH sampling."""

    model_config = ConfigDict(ser_json_inf_nan="constants")

    n_accepted: int
    n_calls: int
    n_proposals_total: int
    acceptance_rate: float
    proposal_type: str  # "unconstrained", "safe", or "mixture"
    ar_to_imh_switch: bool
    imh_switch_call_idx: int | None
    safe_prop_mix_weight: float | None
    env_const: float
    accepted_quality: SampleQualityMetrics
    rejected_quality: SampleQualityMetrics


class CPCSearchMetrics(BaseModel):
    """Metrics from the CPC beta grid search."""

    model_config = ConfigDict(ser_json_inf_nan="constants")

    beta_t: float
    psi_hat_t: float
    grid_size: int
    grid_position_selected: int
    risk_margin: float  # adjusted_alpha - w_infeasible_normalized at selection
    w_test: float
    proposal_selected: str
    switch_to_mixture: bool
    switch_to_optimized: bool
    psi_hat_intersection_safe: float
    psi_hat_intersection_unconstrained: float
    envelope_const: float


class DatasetSizes(BaseModel):
    """Sample counts at calibration/training split."""

    n_cal: int
    n_train: int
    n_combined: int


class StageTiming(BaseModel):
    """Wall-clock seconds for each pipeline stage within a round."""

    training_s: float
    seed_selection_s: float
    likelihood_computation_s: float
    cpc_search_s: float
    ar_sampling_s: float
    dataset_split_s: float
    total_round_s: float


class RoundSummary(BaseModel):
    """One-per-round structured summary, written as ``round_summary.json``."""

    model_config = ConfigDict(ser_json_inf_nan="constants")

    round_idx: int
    round_type: str  # "sft", "dpo", or "marge"
    cpc: CPCSearchMetrics
    ar_sampling: ARSamplingMetrics | None = None
    dataset_sizes: DatasetSizes
    temperatures: list[float]
    timing: StageTiming


# ---------------------------------------------------------------------------
# Dataclass return types (internal, not serialized)
# ---------------------------------------------------------------------------


@dataclass
class CPCSearchResult:
    """Structured return value from ``cpc_beta_search()``.

    Replaces the previous 10-tuple return with named fields.
    """

    beta_t: float
    psi_hat_t: float
    constrained_liks_df: pd.DataFrame
    constrained_liks_fp: str
    unconstrained_df: pd.DataFrame
    unconstrained_liks_fp: str
    proposal: str
    psi_hat_intersection_safe: float
    psi_hat_intersection_unconstrained: float
    envelope_const: float
    search_metrics: CPCSearchMetrics


@dataclass
class ARSamplingResult:
    """Structured return value from ``accept_reject_sample_and_get_likelihoods()``.

    Replaces the previous 4-tuple return with named fields.
    """

    unconstrained_df: pd.DataFrame | None
    unconstrained_fp: str | None
    constrained_df: pd.DataFrame | None
    constrained_fp: str | None
    all_proposals_fp: str | None
    sampling_metrics: ARSamplingMetrics | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_sample_quality(
    df: pd.DataFrame, score_col: str = "score"
) -> SampleQualityMetrics:
    """Compute aggregate quality metrics from a DataFrame with a score column.

    Args:
        df: DataFrame containing at least a ``score_col`` column.
        score_col: Name of the column holding objective values.

    Returns:
        Quality metrics for the provided samples.
    """
    import numpy as np

    n_samples = len(df)
    if n_samples == 0:
        return SampleQualityMetrics(
            n_samples=0,
            n_parsable=0,
            n_feasible=0,
            frac_feasible=0.0,
            mean_score=float("nan"),
            min_score=float("nan"),
            max_score=float("nan"),
        )

    scores = df[score_col].to_numpy(dtype=float)
    finite_mask = np.isfinite(scores)
    n_parsable = int(np.sum(~np.isnan(scores)))
    n_feasible = int(np.sum(finite_mask))
    finite_scores = scores[finite_mask]

    return SampleQualityMetrics(
        n_samples=n_samples,
        n_parsable=n_parsable,
        n_feasible=n_feasible,
        frac_feasible=n_feasible / n_samples,
        mean_score=float(np.mean(finite_scores)) if n_feasible > 0 else float("nan"),
        min_score=float(np.min(finite_scores)) if n_feasible > 0 else float("nan"),
        max_score=float(np.max(finite_scores)) if n_feasible > 0 else float("nan"),
    )
