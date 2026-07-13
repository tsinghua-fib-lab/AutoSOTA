"""
Models Module

This module contains neural network models for representing finitely convex/concave
functions and computing infimal convolutions with implicit differentiation.

Classes:
    - FiniteModel: Unified finite representation for convex/concave functions
    - FiniteSeparableModel: Separable finite representation (efficient for product kernels)
    - InfConvolution: Differentiable infimal convolution operation
"""

from .finite_model import FiniteModel, FiniteSeparableModel
from .inf_convolution import InfConvolution
from .helpers import ZeroMean

__all__ = ["FiniteModel", "FiniteSeparableModel", "InfConvolution", "ZeroMean"]