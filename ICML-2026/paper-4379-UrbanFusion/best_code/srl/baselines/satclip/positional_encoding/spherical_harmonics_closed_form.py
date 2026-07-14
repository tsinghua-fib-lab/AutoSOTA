"""
Original code from: https://github.com/microsoft/satclip

Original source:
Klemmer, Konstantin; Rolf, Esther; Robinson, Caleb; Mackey, Lester; Rußwurm, Marc.
"SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery."
arXiv preprint published November 28, 2023. Later published as conference paper at AAAI 2025.
"""

import math

import torch


####################### Spherical Harmonics utilities ########################
# Code copied from https://github.com/BachiLi/redner/blob/master/pyredner/utils.py
# Code adapted from "Spherical Harmonic Lighting: The Gritty Details", Robin Green
# http://silviojemma.com/public/papers/lighting/spherical-harmonic-lighting.pdf
def associated_legendre_polynomial(l, m, x):
    pmm = torch.ones_like(x)
    if m > 0:
        somx2 = torch.sqrt((1 - x) * (1 + x))
        fact = 1.0
        for i in range(1, m + 1):
            pmm = pmm * (-fact) * somx2
            fact += 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmmp1
    pll = torch.zeros_like(x)
    for ll in range(m + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    return pll


def SH_renormalization(l, m):
    return math.sqrt(
        (2.0 * l + 1.0)
        * math.factorial(l - m)
        / (4 * math.pi * math.factorial(l + m))
    )


def SH(m, l, phi, theta):
    if m == 0:
        return SH_renormalization(l, m) * associated_legendre_polynomial(
            l, m, torch.cos(theta)
        )
    elif m > 0:
        return (
            math.sqrt(2.0)
            * SH_renormalization(l, m)
            * torch.cos(m * phi)
            * associated_legendre_polynomial(l, m, torch.cos(theta))
        )
    else:
        return (
            math.sqrt(2.0)
            * SH_renormalization(l, -m)
            * torch.sin(-m * phi)
            * associated_legendre_polynomial(l, -m, torch.cos(theta))
        )
