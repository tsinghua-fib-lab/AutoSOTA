"""
Utils for transformations
"""
# Author: Alicia Curth
from typing import Any, Optional

import numpy as np

PW_TRANSFORMATION = "PW"
DR_TRANSFORMATION = "DR"
RA_TRANSFORMATION = "RA"

ALL_TRANSFORMATIONS = [PW_TRANSFORMATION, DR_TRANSFORMATION, RA_TRANSFORMATION]


def aipw_te_transformation(
    y: np.ndarray,
    w: np.ndarray,
    p: Optional[np.ndarray],
    mu_0: np.ndarray,
    mu_1: np.ndarray,
) -> np.ndarray:
    """
    Transforms data to efficient influence function pseudo-outcome for CATE estimation

    Parameters
    ----------
    y : array-like of shape (n_samples,) or (n_samples, )
        The observed outcome variable
    w: array-like of shape (n_samples,)
        The observed treatment indicator
    p: array-like of shape (n_samples,)
        The treatment propensity, estimated or known. Can be None, then p=0.5 is assumed
    mu_0: array-like of shape (n_samples,)
        Estimated or known potential outcome mean of the control group
    mu_1: array-like of shape (n_samples,)
        Estimated or known potential outcome mean of the treatment group

    Returns
    -------
    d_hat:
        EIF transformation for CATE
    """
    if p is None:
        # assume equal
        p = np.full(len(y), 0.5)

    w_1 = w / p
    w_0 = (1 - w) / (1 - p)
    return (w_1 - w_0) * y + ((1 - w_1) * mu_1 - (1 - w_0) * mu_0)


def ht_te_transformation(
    y: np.ndarray,
    w: np.ndarray,
    p: Optional[np.ndarray] = None,
    mu_0: Optional[np.ndarray] = None,
    mu_1: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Transform data to Horvitz-Thompson transformation for CATE

    Parameters
    ----------
    y : array-like of shape (n_samples,) or (n_samples, )
        The observed outcome variable
    w: array-like of shape (n_samples,)
        The observed treatment indicator
    p: array-like of shape (n_samples,)
        The treatment propensity, estimated or known. Can be None, then p=0.5 is assumed
    mu_0: array-like of shape (n_samples,)
        Placeholder, not used. Estimated or known potential outcome mean of the control group
    mu_1: array-like of shape (n_samples,)
        Placerholder, not used. Estimated or known potential outcome mean of the treatment group

    Returns
    -------
    res: array-like of shape (n_samples,)
        Horvitz-Thompson transformed data
    """
    if p is None:
        # assume equal propensities
        p = np.full(len(y), 0.5)
    return (w / p - (1 - w) / (1 - p)) * y


def ra_te_transformation(
    y: np.ndarray,
    w: np.ndarray,
    p: Optional[np.ndarray],
    mu_0: np.ndarray,
    mu_1: np.ndarray,
) -> np.ndarray:
    """
    Transform data to regression adjustment for CATE

    Parameters
    ----------
    y : array-like of shape (n_samples,) or (n_samples, )
        The observed outcome variable
    w: array-like of shape (n_samples,)
        The observed treatment indicator
    p: array-like of shape (n_samples,)
        Placeholder, not used. The treatment propensity, estimated or known.
    mu_0: array-like of shape (n_samples,)
         Estimated or known potential outcome mean of the control group
    mu_1: array-like of shape (n_samples,)
        Estimated or known potential outcome mean of the treatment group

    Returns
    -------
    res: array-like of shape (n_samples,)
        Regression adjusted transformation
    """
    return w * (y - mu_0) + (1 - w) * (mu_1 - y)


TRANSFORMATION_DICT = {
    PW_TRANSFORMATION: ht_te_transformation,
    RA_TRANSFORMATION: ra_te_transformation,
    DR_TRANSFORMATION: aipw_te_transformation,
}


def _get_transformation_function(transformation_name: str) -> Any:
    """
    Get transformation function associated with a name
    """
    if transformation_name not in ALL_TRANSFORMATIONS:
        raise ValueError(
            "Parameter first stage should be in "
            "catenets.models.transformations.ALL_TRANSFORMATIONS."
            " You passed {}".format(transformation_name)
        )
    return TRANSFORMATION_DICT[transformation_name]
