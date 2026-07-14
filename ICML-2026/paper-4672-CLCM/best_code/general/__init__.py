"""
General-architecture Equilibrium Fisher Control (EFC).

Extends EFC from fully-connected layers to arbitrary nn.Module building blocks
(convolutions, residual blocks, etc.) using autograd for all Jacobian and
gradient computations.
"""
