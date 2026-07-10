"""
Unbiased Transformations for CATE
"""
# Author: Alicia Curth
from typing import Optional

import torch


def dr_transformation_cate(
    y: torch.Tensor,
    w: torch.Tensor,
    p: torch.Tensor,
    mu_0: torch.Tensor,
    mu_1: torch.Tensor,
) -> torch.Tensor:
    """
    Transforms data to efficient influence function/aipw pseudo-outcome for CATE estimation

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
        p = torch.full(y.shape, 0.5)

    EPS = 1e-7
    w_1 = w / (p + EPS)
    w_0 = (1 - w) / (EPS + 1 - p)
    return (w_1 - w_0) * y + ((1 - w_1) * mu_1 - (1 - w_0) * mu_0)


def pw_transformation_cate(
    y: torch.Tensor,
    w: torch.Tensor,
    p: Optional[torch.Tensor] = None,
    mu_0: Optional[torch.Tensor] = None,
    mu_1: Optional[torch.Tensor] = None,
) -> torch.Tensor:
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
         Estimated or known potential outcome mean of the control group. Placeholder, not used.
    mu_1: array-like of shape (n_samples,)
        Estimated or known potential outcome mean of the treatment group. Placeholder, not used.
    Returns
    -------
    res: array-like of shape (n_samples,)
        Horvitz-Thompson transformed data
    """
    if p is None:
        # assume equal propensities
        p = torch.full(y.shape, 0.5)
    return (w / p - (1 - w) / (1 - p)) * y


def ra_transformation_cate(
    y: torch.Tensor,
    w: torch.Tensor,
    p: torch.Tensor,
    mu_0: torch.Tensor,
    mu_1: torch.Tensor,
) -> torch.Tensor:
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


def u_transformation_cate(
    y: torch.Tensor, w: torch.Tensor, p: torch.Tensor, mu: torch.Tensor
) -> torch.Tensor:
    """
    Transform data to U-transformation (described in Kuenzel et al, 2019, Nie & Wager, 2017)
    which underlies both R-learner and U-learner

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
    if p is None:
        # assume equal propensities
        p = torch.full(y.shape, 0.5)
    return (y - mu) / (w - p)
