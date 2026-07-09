import numpy as np
import pandas as pd
from omegaconf import DictConfig

from ..data_contracts import CON_LIK_PREFIX, LIK_PREFIX


def constrain_likelihoods(
    cfg: DictConfig,
    likelihoods_mat: np.ndarray,
    betas: np.ndarray | list[float],
    psis: np.ndarray | list[float],
) -> np.ndarray:
    """Process matrix of unconstrained likelihoods into constrained likelihoods"""
    n_prop, n_models = np.shape(likelihoods_mat)

    if n_models > 2 and cfg.conformal_policy_control.constrain_against == "init":
        ## If constraining against initial safe policy, only want first model and current model
        raise ValueError(
            "Modified to only constrain likelihoods relative to original safe policy"
        )

    constrained_likelihoods_mat = np.zeros((n_prop, n_models))

    ## First col of likelihoods_mat should already be safe/constrained
    constrained_likelihoods_mat[:, 0] = likelihoods_mat[:, 0]

    ## Compute constrained likelihoods for each subsequent policy and bound
    if cfg.conformal_policy_control.constrain_against == "init":
        constrained_likelihoods_mat[:, 1] = np.where(
            likelihoods_mat[:, 1] / constrained_likelihoods_mat[:, 0] < betas[1],
            likelihoods_mat[:, 1] / psis[1],
            constrained_likelihoods_mat[:, 0] * (betas[1] / psis[1]),
        )
    else:
        for i in range(1, n_models):
            constrained_likelihoods_mat[:, i] = np.where(
                likelihoods_mat[:, i] / constrained_likelihoods_mat[:, i - 1]
                < betas[i],
                likelihoods_mat[:, i] / psis[i],
                constrained_likelihoods_mat[:, i - 1] * (betas[i] / psis[i]),
            )

    return constrained_likelihoods_mat


def mixture_pdf_from_densities_mat(
    constrained_densities_all_steps: np.ndarray, mixture_weights: np.ndarray
) -> np.ndarray:
    """Compute a mixture PDF from per-model constrained densities.

    Args:
        constrained_densities_all_steps: Constrained density values of shape
            (n_samples, n_models), where columns correspond to models t=0, ..., T-1.
        mixture_weights: Relative weight for each model's distribution, of shape
            (T,). Typically the calibration dataset size for each model
            (mixture_weights[i] = n_cal_model_i). Normalized to sum to 1 internally.

    Returns:
        Mixture PDF values of shape (n_samples,).
    """
    mixture_weights_normed = mixture_weights / np.sum(mixture_weights)

    mixture_pdfs = constrained_densities_all_steps @ mixture_weights_normed

    return mixture_pdfs


def check_col_names(df: pd.DataFrame) -> None:
    """Validate that likelihood column indices are monotonically increasing.

    Args:
        df: DataFrame with likelihood columns named ``lik_r{i}`` or
            ``constrained_lik_r{i}``.

    Raises:
        ValueError: If the numeric indices extracted from column names are not
            consecutive (incrementing by 1).
    """
    lik_cols = []
    for c in df.columns:
        if c.startswith(LIK_PREFIX) or c.startswith(CON_LIK_PREFIX):
            lik_cols.append(c)

    col_indices = [int(c.split("_")[-1][1:]) for c in lik_cols]
    for i in range(len(col_indices)):
        if i > 0 and col_indices[i] - col_indices[i - 1] != 1:
            raise ValueError(f"col indices not increasing {df.columns}")
