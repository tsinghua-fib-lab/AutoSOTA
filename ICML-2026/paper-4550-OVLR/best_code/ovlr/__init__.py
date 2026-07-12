"""
OVLR: Output-Level Variance-Reduced Likelihood Ratio Gradient Estimation.

Paper: OVLR: Efficient, Scalable, and Robust Training via
       Output-Level Variance-Reduced Likelihood Ratio
       ICML 2026
"""

__version__ = "1.0.0"

from .noise import (
    # Symmetric antithetic noise
    SymmetricGaussianNoise,
    SymmetricLaplaceNoise,
    SymmetricRademacherNoise,
    SymmetricStudentTNoise,
    # Asymmetric noise (for ablation)
    AsymmetricGaussianNoise,
    # Score-function noise with neg_score method
    GaussianScoreNoise,
    StudentTScoreNoise,
    LaplaceScoreNoise,
    RademacherDirectionNoise,
    # Factory function
    get_noise_fn,
)

from .estimator import (
    OVLRGradientEstimator,
    ScoreFunctionOVLRGradientEstimator,
    TwoPointSPSAOVLRGradientEstimator,
)

__all__ = [
    # Estimators
    "OVLRGradientEstimator",
    "ScoreFunctionOVLRGradientEstimator",
    "TwoPointSPSAOVLRGradientEstimator",
    # Symmetric noise
    "SymmetricGaussianNoise",
    "SymmetricLaplaceNoise",
    "SymmetricRademacherNoise",
    "SymmetricStudentTNoise",
    # Asymmetric noise
    "AsymmetricGaussianNoise",
    # Score-function noise
    "GaussianScoreNoise",
    "StudentTScoreNoise",
    "LaplaceScoreNoise",
    "RademacherDirectionNoise",
    # Factory
    "get_noise_fn",
]
