#!/usr/bin/env python3
"""
Description: Implementation by the paper "GEOGRAPHIC
LOCATION ENCODING WITH SPHERICAL SHARMONICS AND SINUSOIDAL REPRESENTATION
NETWORKS". This function prints the source code for
spherical_harmonics_ylms.py to console.

spherical_harmonics pre-computes the
analytical solutions to each real spherical harmonic with sympy
the script contains different functions for different degrees l and orders m

Marc Russwurm
"""

import sys
from datetime import datetime

from sympy import Abs, Symbol, assoc_legendre, cos, factorial, pi, sin, sqrt

theta = Symbol("theta")
phi = Symbol("phi")


def calc_ylm(l_index: int, m_index: int) -> Symbol:
    """
    Calculate the spherical harmonic Ylm(theta, phi) for given l and m.

    See last equation of:
    https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form

    Parameters
    ----------
    l_index : int
        degree of the spherical harmonic
    m_index : int
        order of the spherical harmonic

    Returns
    -------
    Symbol
        the spherical harmonic Ylm(theta, phi)
    """

    if m_index < 0:
        Plm = assoc_legendre(l_index, Abs(m_index), cos(theta))
        Plm_bar = (
            sqrt(
                ((2 * l_index + 1) / (4 * pi))
                * (
                    factorial(l_index - Abs(m_index))
                    / factorial(l_index + Abs(m_index))
                )
            )
            * Plm
        )

        Ylm = (-1) ** m_index * sqrt(2) * Plm_bar * sin(Abs(m_index) * phi)
    elif m_index == 0:
        Ylm = sqrt((2 * l_index + 1) / (4 * pi)) * assoc_legendre(
            l_index, m_index, cos(theta)
        )
    else:  # m > 0
        Plm = assoc_legendre(l_index, m_index, cos(theta))
        Plm_bar = (
            sqrt(
                ((2 * l_index + 1) / (4 * pi))
                * (factorial(l_index - m_index) / factorial(l_index + m_index))
            )
            * Plm
        )

        Ylm = (-1) ** m_index * sqrt(2) * Plm_bar * cos(m_index * phi)
    return Ylm


def print_function(l_index, m_index):
    fname = f"Yl{l_index}_m{m_index}".replace("-", "_minus_")
    print()
    print("@torch.jit.script")
    print(f"def {fname}(theta, phi):")
    print("    return " + str(calc_ylm(l_index, m_index).evalf()))


# max number of Legendre Polynomials
L = 101

head = (
    """\"\"\"
analytic expressions of spherical harmonics generated with sympy file
Marc Russwurm generated """
    + str(datetime.date(datetime.now()))
    + """

run
python """
    + sys.argv[0]
    + """ > spherical_harmonics_ylm.py

to generate the source code
\"\"\"

import torch
from torch import cos, sin

def get_SH(m,l):
  fname = f"Yl{l}_m{m}".replace("-","_minus_")
  return globals()[fname]

def SH(m, l, phi, theta):
  Ylm = get_SH(m,l)
  return Ylm(theta, phi)
"""
)
print(head)
print()

for l_index in range(L):
    for m_index in range(-l_index, l_index + 1):
        print_function(l_index, m_index)
