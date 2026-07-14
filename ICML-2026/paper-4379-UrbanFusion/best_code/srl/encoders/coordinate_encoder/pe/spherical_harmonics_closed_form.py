#!/usr/bin/env python3
"""
Description: Implementation of the spherical harmonics functions for the
analytical computation of the spherical harmonics. Implementation by the paper
"GEOGRAPHIC LOCATION ENCODING WITH SPHERICAL SHARMONICS AND SINUSOIDAL
REPRESENTATION NETWORKS"

####################### Spherical Harmonics utilities ########################
# Code from https://github.com/BachiLi/redner/blob/master/pyredner/utils.py
# Code  from "Spherical Harmonic Lighting: The Gritty Details", Robin Green
# http://silviojemma.com/public/papers/lighting/spherical-harmonic-lighting.pdf
"""

import math

import torch


def associated_legendre_polynomial(
    l_index: int, m_index: int, x: torch.Tensor
) -> torch.Tensor:
    """
    Compute the associated Legendre polynomial P_{l_index}^{m_index}(x).

    Parameters
    ----------
    l_index : int
        The degree of the polynomial.
    m_index : int
        The order of the polynomial.
    x : torch.Tensor
        The input value.

    Returns
    -------
    torch.Tensor
        The value of the polynomial at x
    """
    pmm = torch.ones_like(x)
    if m_index > 0:
        somx2 = torch.sqrt((1 - x) * (1 + x))
        fact = 1.0
        for i in range(1, m_index + 1):
            pmm = pmm * (-fact) * somx2
            fact += 2.0
    if l_index == m_index:
        return pmm
    pmmp1 = x * (2.0 * m_index + 1.0) * pmm
    if l_index == m_index + 1:
        return pmmp1
    pll = torch.zeros_like(x)
    for ll in range(m_index + 2, l_index + 1):
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m_index - 1.0) * pmm) / (
            ll - m_index
        )
        pmm = pmmp1
        pmmp1 = pll
    return pll


def SH_renormalization(l_index: int, m_index: int) -> float:
    """
    Compute the renormalization factor for the spherical harmonics.
    "
    Parameters
    ----------
    l_index : int
        The degree of the spherical harmonics.
    m_index : int
        The order of the spherical harmonics.

    Returns
    -------
    float
        The renormalization factor.
    """
    return math.sqrt(
        (2.0 * l_index + 1.0)
        * math.factorial(l_index - m_index)
        / (4 * math.pi * math.factorial(l_index + m_index))
    )


def SH(m_index, l_index, phi, theta):
    if m_index == 0:
        return SH_renormalization(
            l_index, m_index
        ) * associated_legendre_polynomial(l_index, m_index, torch.cos(theta))
    elif m_index > 0:
        return (
            math.sqrt(2.0)
            * SH_renormalization(l_index, m_index)
            * torch.cos(m_index * phi)
            * associated_legendre_polynomial(
                l_index, m_index, torch.cos(theta)
            )
        )
    else:
        return (
            math.sqrt(2.0)
            * SH_renormalization(l_index, -m_index)
            * torch.sin(-m_index * phi)
            * associated_legendre_polynomial(
                l_index, -m_index, torch.cos(theta)
            )
        )
