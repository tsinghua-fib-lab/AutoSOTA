# -*- coding: utf-8 -*-
"""
Spiking neuron models.

UltraLIF models (paper):
    UltraLIF:    Temporal, 2-term LSE, fixed tau.
    UltraPLIF:   Temporal, 2-term LSE, learnable tau.
    UltraDLIF:   Spatial, 3-term LSE, fixed tau.
    UltraDPLIF:  Spatial, 3-term LSE, learnable tau.

Baseline models:
    LIF:         Standard LIF with surrogate gradient.
    PLIF:        Parametric LIF (learnable tau).
    AdaLIF:      Adaptive-threshold LIF.
    FullPLIF:    Learnable tau + threshold.
    DSpike:      Li et al. NeurIPS 2021.
    DSpikePlus:  DSpike with learnable tau.
    SigmaLIF:    Sigmoid-only ablation control.
"""

from .ultra import UltraLIF, UltraPLIF, UltraLIF_DS, UltraPLIF_DS
from .ultradlif import UltraDLIF, UltraDPLIF
from .lif import LIF, PLIF, AdaLIF, FullPLIF
from .baselines import DSpike, DSpikePlus, SigmaLIF

__all__ = [
    # Paper models
    "UltraLIF",
    "UltraPLIF",
    "UltraDLIF",
    "UltraDPLIF",
    # Ablation variants
    "UltraLIF_DS",
    "UltraPLIF_DS",
    # Baselines
    "LIF",
    "PLIF",
    "AdaLIF",
    "FullPLIF",
    "DSpike",
    "DSpikePlus",
    "SigmaLIF",
]
