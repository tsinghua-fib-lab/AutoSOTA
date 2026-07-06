# -*- coding: utf-8 -*-
"""
Spiking neural network architectures.

Fully-connected:
    SNN:            Single hidden layer.
    DeepSNN:        Two hidden layers (+BN, +residual).
    TripleSNN:      Three hidden layers (+BN, +residual).

Convolutional:
    ConvSNN:        2-layer Conv-SNN.
    DeepConvSNN:    4-layer Conv-SNN.

Fully spiking ResNet:
    SpikeBasicBlock:  BasicBlock with spiking neurons.
    SpikingResNet18:  Full 17-layer spiking ResNet-18.
"""

from .fc import SNN, DeepSNN, TripleSNN
from .conv import ConvSNN, DeepConvSNN
from .resnet import SpikeBasicBlock, SpikingResNet18

__all__ = [
    "SNN",
    "DeepSNN",
    "TripleSNN",
    "ConvSNN",
    "DeepConvSNN",
    "SpikeBasicBlock",
    "SpikingResNet18",
]
