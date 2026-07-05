# -*- coding: utf-8 -*-
"""
UltraLIF — Fully Differentiable Spiking Neural Networks via Ultradiscretization.

Paper: "UltraLIF: Fully Differentiable Spiking Neural Networks via
Ultradiscretization" (ICML 2026). arXiv:2602.11206

Quick start:
    >>> from ultralif import UltraLIF, SNN, get_dataset, train_model, set_seed
    >>> set_seed(42)
    >>> train_loader, test_loader, in_dim, n_cls = get_dataset('mnist')
    >>> neuron = UltraLIF(dim=256)
    >>> model = SNN(neuron, in_dim=in_dim, hid_dim=256, out_dim=n_cls)
    >>> best_acc, history, _ = train_model(model, train_loader, test_loader,
    ...                                     epochs=100, lr=1e-3, device='cuda')

Model name mapping (code name -> paper name):
    UltraLIF   -> UltraLIF  (temporal, 2-term LSE, fixed tau)
    UltraPLIF  -> UltraPLIF (temporal, 2-term LSE, learnable tau)
    UltraDLIF  -> UltraDLIF (spatial,  3-term LSE, fixed tau)
    UltraDPLIF -> UltraDPLIF(spatial,  3-term LSE, learnable tau)
"""

from .neurons.ultra import UltraLIF, UltraPLIF, UltraLIF_DS, UltraPLIF_DS
from .neurons.ultradlif import UltraDLIF, UltraDPLIF
from .neurons.lif import LIF, PLIF, AdaLIF, FullPLIF
from .neurons.baselines import DSpike, DSpikePlus, SigmaLIF
from .networks.fc import SNN, DeepSNN, TripleSNN
from .networks.conv import ConvSNN, DeepConvSNN
from .networks.resnet import SpikingResNet18
from .datasets.loader import get_dataset
from .datasets.encoding import rate_encode
from .datasets.utils import set_seed, get_device
from .training.trainer import train_model
from .training.metrics import count_spikes_epoch, compute_energy_proxy

__version__ = "1.0.0"

__all__ = [
    # UltraLIF neurons (temporal)
    "UltraLIF",
    "UltraPLIF",
    "UltraLIF_DS",
    "UltraPLIF_DS",
    # UltraDLIF neurons (spatial)
    "UltraDLIF",
    "UltraDPLIF",
    # Baseline neurons
    "LIF",
    "PLIF",
    "AdaLIF",
    "FullPLIF",
    "DSpike",
    "DSpikePlus",
    "SigmaLIF",
    # Networks
    "SNN",
    "DeepSNN",
    "TripleSNN",
    "ConvSNN",
    "DeepConvSNN",
    "SpikingResNet18",
    # Data
    "get_dataset",
    "rate_encode",
    "set_seed",
    "get_device",
    # Training
    "train_model",
    "count_spikes_epoch",
    "compute_energy_proxy",
]
