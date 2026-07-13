"""Trainer implementations.

This module contains all trainer implementations organized by category:
- simulation: NPE, FMPE (base posterior estimators)
- calibration: RoPE, fm_post_transform, etc. (calibration methods)
- baselines: DPE, MF-NPE, NPE-RS (comparison baselines)

Trainers are automatically registered when imported.
"""

from training.trainers import baselines, calibration, simulation

__all__ = ["baselines", "calibration", "simulation"]
