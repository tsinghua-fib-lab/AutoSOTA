"""
Implement different reweighting/balancing strategies as in Li et al (2018)
"""
# Author: Alicia Curth
from typing import Optional

import torch

IPW_NAME = "ipw"
TRUNC_IPW_NAME = "truncipw"
OVERLAP_NAME = "overlap"
MATCHING_NAME = "match"
PROP = "prop"
ONE_MINUS_PROP = "1-prop"

ALL_WEIGHTING_STRATEGIES = [
    IPW_NAME,
    TRUNC_IPW_NAME,
    OVERLAP_NAME,
    MATCHING_NAME,
    PROP,
    ONE_MINUS_PROP,
]


def compute_importance_weights(
    propensity: torch.Tensor,
    w: torch.Tensor,
    weighting_strategy: str,
    weight_args: Optional[dict] = None,
) -> torch.Tensor:
    if weighting_strategy not in ALL_WEIGHTING_STRATEGIES:
        raise ValueError(
            f"weighting_strategy should be in {ALL_WEIGHTING_STRATEGIES}"
            f"You passed {weighting_strategy}"
        )
    if weight_args is None:
        weight_args = {}

    if weighting_strategy == PROP:
        return propensity
    elif weighting_strategy == ONE_MINUS_PROP:
        return 1 - propensity
    elif weighting_strategy == IPW_NAME:
        return compute_ipw(propensity, w)
    elif weighting_strategy == TRUNC_IPW_NAME:
        return compute_trunc_ipw(propensity, w, **weight_args)
    elif weighting_strategy == OVERLAP_NAME:
        return compute_overlap_weights(propensity, w)
    elif weighting_strategy == MATCHING_NAME:
        return compute_matching_weights(propensity, w)


def compute_ipw(propensity: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    p_hat = torch.mean(w)
    return w * p_hat / propensity + (1 - w) * (1 - p_hat) / (1 - propensity)


def compute_trunc_ipw(
    propensity: torch.Tensor, w: torch.Tensor, cutoff: float = 0.05
) -> torch.Tensor:
    ipw = compute_ipw(propensity, w)
    return torch.where((propensity > cutoff) & (propensity < 1 - cutoff), ipw, 0)


# TODO check normalizing these weights
def compute_matching_weights(propensity: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    ipw = compute_ipw(propensity, w)
    return torch.minimum(ipw, 1 - ipw) * ipw


def compute_overlap_weights(propensity: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    ipw = compute_ipw(propensity, w)
    return propensity * (1 - propensity) * ipw
