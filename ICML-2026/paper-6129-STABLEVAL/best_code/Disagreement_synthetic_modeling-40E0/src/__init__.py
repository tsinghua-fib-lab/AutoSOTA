"""Synthetic Disagreement Study Package."""

from .config import Config
from .agents import generate_true_labels, compute_ground_truth_scores
from .annotators import AnnotatorPool
from .labeling import generate_observed_labels
from .aggregation import majority_vote, dawid_skene_em, posterior_expected_credit
from .metrics import compute_mse, compute_kendall_tau, compute_stability

__all__ = [
    "Config",
    "generate_true_labels",
    "compute_ground_truth_scores",
    "AnnotatorPool",
    "generate_observed_labels",
    "majority_vote",
    "dawid_skene_em",
    "posterior_expected_credit",
    "compute_mse",
    "compute_kendall_tau",
    "compute_stability",
]