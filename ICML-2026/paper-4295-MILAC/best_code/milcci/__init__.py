# -*- coding: utf-8 -*-
"""
MILCCI: Multi-axis Interpretable Latent Component and Condition Inference.

Decomposes a 3D tensor Y (neurons x time x trials) into condition-varying
spatial maps A and temporal traces Phi, with similarity regularization
along multiple label axes.
"""
from .core import fit, reconstruct
from .evaluation import per_trial_r2, global_r2, reconstruction_correlation
from .synthetic import generate_synthetic_data
from . import plotting

__version__ = '0.1.0'
__all__ = [
    'fit', 'reconstruct',
    'per_trial_r2', 'global_r2', 'reconstruction_correlation',
    'generate_synthetic_data',
    'plotting',
]
