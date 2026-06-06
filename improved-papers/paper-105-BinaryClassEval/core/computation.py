"""Computation layer for net benefit analysis."""
from __future__ import annotations
from typing import Sequence

import numpy as np

from stats.isotonic import get_calibration_model
from stats.bootstrap import bootstrap_net_benefit_ci
from stats.prevalence import log_odds_grid, default_prevalence_grid

from core.net_benefit import net_benefit_for_prevalences
from core.proto import ArrayLike, Number
from proto.config import ComputationConfig, SubgroupComputedData
from proto.subgroup import SubgroupResults


def compute_subgroup_metrics(
    subgroup: SubgroupResults,
    config: ComputationConfig
) -> SubgroupComputedData:
    """Compute all metrics for a single subgroup.

    This function handles all computation logic including:
    - Computing the log-odds grid
    - Computing the net benefit curve
    - Computing confidence intervals if requested
    - Computing calibrated curves if requested
    - Computing training and shifted points

    Parameters
    ----------
    subgroup : SubgroupResults
        Subgroup data to compute metrics for.
    config : ComputationConfig
        Configuration for computation.

    Returns
    -------
    SubgroupComputedData
        Computed metrics for the subgroup.
    """
    # 1. Compute log-odds grid
    logodds_grid = log_odds_grid(config.prevalence_grid)

    # 2. Compute net benefit curve
    nb_curve = net_benefit_for_prevalences(
        subgroup.y_true,
        subgroup.y_pred_proba,
        prevalence_grid=config.prevalence_grid,
        cost_ratio=config.cost_ratio,
        train_prevalence=subgroup.prevalence if not config.train_prevalence_override else None,
        normalize=config.normalize
    )

    # 3. Compute training point
    train_nb = net_benefit_for_prevalences(
        subgroup.y_true,
        subgroup.y_pred_proba,
        prevalence_grid=np.array([subgroup.prevalence]),
        cost_ratio=config.cost_ratio,
        train_prevalence=subgroup.prevalence if not config.train_prevalence_override else None,
        normalize=config.normalize
    )[0]
    training_point = (subgroup.log_odds, train_nb)

    # 4. Compute shifted point if requested
    shifted_point = None
    if config.diamond_shift_amount is not None:
        shifted_logodds = subgroup.log_odds + config.diamond_shift_amount
        shifted_prevalence = np.exp(shifted_logodds) / (1 + np.exp(shifted_logodds))

        shifted_nb = net_benefit_for_prevalences(
            subgroup.y_true,
            subgroup.y_pred_proba,
            prevalence_grid=np.array([shifted_prevalence]),
            cost_ratio=config.cost_ratio,
            train_prevalence=subgroup.prevalence if not config.train_prevalence_override else None,
            normalize=config.normalize
        )[0]

        shifted_point = (shifted_logodds, shifted_nb)

    # 5. Compute calibrated curve if requested
    calibrated_nb_curve = None
    if config.compute_calibrated:
        # Get calibration model
        calibration_model = get_calibration_model(subgroup.y_true, subgroup.y_pred_proba)

        # Apply calibration
        calibrated_scores = calibration_model.transform(subgroup.y_pred_proba)

        # Compute calibrated net benefit curve
        calibrated_nb_curve = net_benefit_for_prevalences(
            subgroup.y_true,
            calibrated_scores,
            prevalence_grid=config.prevalence_grid,
            cost_ratio=config.cost_ratio,
            train_prevalence=subgroup.prevalence if not config.train_prevalence_override else None,
            normalize=config.normalize
        )

    # 6. Compute confidence intervals if requested
    nb_ci_lower, nb_ci_upper = None, None
    if config.compute_ci:
        # Call bootstrap function
        lower, _, upper = bootstrap_net_benefit_ci(
            subgroup.y_true,
            subgroup.y_pred_proba,
            config.prevalence_grid,
            config.cost_ratio,
            n_bootstrap=config.n_bootstrap,
            train_prevalence=subgroup.prevalence if not config.train_prevalence_override else None,
            normalize=config.normalize,
            random_seed=config.random_seed,
            calibrate=config.compute_calibrated
        )
        nb_ci_lower, nb_ci_upper = lower, upper

    return SubgroupComputedData(
        name=subgroup.name,
        display_label=subgroup.display_label,
        prevalence=subgroup.prevalence,
        log_odds_grid=logodds_grid,
        nb_curve=nb_curve,
        calibrated_nb_curve=calibrated_nb_curve,
        nb_ci_lower=nb_ci_lower,
        nb_ci_upper=nb_ci_upper,
        training_point=training_point,
        shifted_point=shifted_point,
        auc_roc=subgroup.auc_roc
    )
    
def prepare_data_for_visualization(
    subgroups: Sequence[SubgroupResults],
    config: ComputationConfig
) -> list[SubgroupComputedData]:
    """Prepare data for visualization by computing metrics for all subgroups.
    
    Parameters
    ----------
    subgroups : Sequence[SubgroupResults]
        List of subgroup data to compute metrics for.
    config : ComputationConfig
        Configuration for computation.
        
    Returns
    -------
    list[SubgroupComputedData]
        List of computed data for each subgroup.
    """
    return [compute_subgroup_metrics(subgroup, config) for subgroup in subgroups]
