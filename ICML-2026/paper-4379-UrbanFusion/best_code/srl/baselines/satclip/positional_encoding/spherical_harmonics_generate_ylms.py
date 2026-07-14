"""
Original code from: https://github.com/microsoft/satclip

Original source:
Klemmer, Konstantin; Rolf, Esther; Robinson, Caleb; Mackey, Lester; Rußwurm, Marc.
"SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery."
arXiv preprint published November 28, 2023. Later published as conference paper at AAAI 2025.

This function prints the source code for spherical_harmonics_ylms.py to console

spherical_harmonics pre-computes the analytical solutions to each real spherical harmonic with sympy
the script contains different functions for different degrees l and orders m

Marc Russwurm
"""

import sys
from datetime import datetime

from sympy import Abs, Symbol, assoc_legendre, cos, factorial, pi, sin, sqrt

theta = Symbol("theta")
phi = Symbol("phi")


def calc_ylm(l, m):
    """
    see last equation of https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    """
    if m < 0:
        Plm = assoc_legendre(l, Abs(m), cos(theta))
        Plm_bar = (
            sqrt(
                ((2 * l + 1) / (4 * pi))
                * (factorial(l - Abs(m)) / factorial(l + Abs(m)))
            )
            * Plm
        )

        Ylm = (-1) ** m * sqrt(2) * Plm_bar * sin(Abs(m) * phi)
    elif m == 0:
        Ylm = sqrt((2 * l + 1) / 4 * pi) * assoc_legendre(l, m, cos(theta))
    else:  # m > 0
        Plm = assoc_legendre(l, m, cos(theta))
        Plm_bar = (
            sqrt(
                ((2 * l + 1) / (4 * pi))
                * (factorial(l - m) / factorial(l + m))
            )
            * Plm
        )

        Ylm = (-1) ** m * sqrt(2) * Plm_bar * cos(m * phi)
    return Ylm


def print_function(l, m):
    fname = f"Yl{l}_m{m}".replace("-", "_minus_")
    print()
    print("@torch.jit.script")
    print(f"def {fname}(theta, phi):")
    print("    return " + str(calc_ylm(l, m).evalf()))


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

for l in range(L):
    for m in range(-l, l + 1):
        print_function(l, m)
