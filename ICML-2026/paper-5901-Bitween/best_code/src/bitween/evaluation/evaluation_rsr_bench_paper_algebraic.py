"""
Algebraic RSR experiments for ICML 2026 rebuttal (Reviewer E7bc).

Tests RSR discovery on algebraic structures beyond scalar real-valued functions:
  - 2x2 matrix determinant (multiplicative RSR)
  - 2x2 matrix trace (additive RSR)
  - Quaternion norm (multiplicative RSR)
  - Characteristic polynomial (trace of A^2)
  - Symmetric polynomials (elementary symmetric e2, e3)
  - Bilinear map: integer multiplication as a proxy for matrix-style RSRs

These functions are encoded as multi-variable scalar functions so they fit
directly into Bitween's existing framework without any code changes.

For example, a 2x2 matrix [[a,b],[c,d]] is represented as f(a,b,c,d) = ad - bc.

Usage:
    python -m bitween.evaluation.evaluation_rsr_bench_algebraic \
        --res_dir results/algebraic --method MULTIPLE_REGRESSION
"""

import os
from time import time

import numpy as np

from bitween.config import Config, Method, MILPSolver
from bitween.evaluation.evaluation_rsr_bench_paper import evaluate, get_parser
from bitween.miscs import getLogger
from bitween.sampler import Distribution, Domain

config = Config()
log = getLogger(__name__, config.logger_level, empty_format=True)


# ============================================================================
# 2x2 Matrix Determinant: det([[a,b],[c,d]]) = a*d - b*c
#
# Known RSR: det(A) = det(A*R) / det(R)  for random invertible R
# Encoded as f(a,b,c,d) over 4 scalar variables.
#
# Additive RSR: det(A+R) involves det(A), det(R), and cross terms.
# For 2x2: det(A+R) = det(A) + det(R) + (a1*d2 + a2*d1 - b1*c2 - b2*c1)
# where A=[[a1,b1],[c1,d1]], R=[[a2,b2],[c2,d2]]
# ============================================================================


def test_det2x2_additive(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    """2x2 determinant with additive queries: det(A+R), det(A-R), det(A), det(R)."""

    def f(a, b, c, d):
        return a * d - b * c

    def _sp_f(a, b, c, d):
        return a * d - b * c

    evaluate(
        domain=Domain.Integer,
        distribution=Distribution(np.random.randint, low=-5, high=5),
        exprs=[
            "f(a1+a2, b1+b2, c1+c2, d1+d2)",  # det(A+R)
            "f(a1-a2, b1-b2, c1-c2, d1-d2)",  # det(A-R)
            "f(a1, b1, c1, d1)",  # det(A)
            "f(a2, b2, c2, d2)",  # det(R)
        ],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="A01_det2x2_additive",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=30,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


def test_det2x2_multiplicative(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    """2x2 determinant with multiplicative queries.
    det(A*R) = det(A)*det(R), so f(A*R)/f(R) = f(A).
    A*R for 2x2: [[a1,b1],[c1,d1]] * [[a2,b2],[c2,d2]]
      = [[a1*a2+b1*c2, a1*b2+b1*d2], [c1*a2+d1*c2, c1*b2+d1*d2]]
    We encode det(A*R) as a function of all 8 variables.
    """

    def f(a, b, c, d):
        return a * d - b * c

    def _sp_f(a, b, c, d):
        return a * d - b * c

    # det(A*R) where A=[[a1,b1],[c1,d1]], R=[[a2,b2],[c2,d2]]
    # A*R = [[a1*a2+b1*c2, a1*b2+b1*d2], [c1*a2+d1*c2, c1*b2+d1*d2]]
    def det_product(a1, b1, c1, d1, a2, b2, c2, d2):
        return (a1 * a2 + b1 * c2) * (c1 * b2 + d1 * d2) - (a1 * b2 + b1 * d2) * (
            c1 * a2 + d1 * c2
        )

    def _sp_det_product(a1, b1, c1, d1, a2, b2, c2, d2):
        return (a1 * a2 + b1 * c2) * (c1 * b2 + d1 * d2) - (a1 * b2 + b1 * d2) * (
            c1 * a2 + d1 * c2
        )

    evaluate(
        domain=Domain.Integer,
        distribution=Distribution(np.random.randint, low=-3, high=3),
        exprs=[
            # det(A*R) = product of matrix entries
            "det_product(a1, b1, c1, d1, a2, b2, c2, d2)",
            "f(a1, b1, c1, d1)",  # det(A)
            "f(a2, b2, c2, d2)",  # det(R)
        ],
        infer_funcs=[det_product, f],
        sympy_funcs=[_sp_det_product, _sp_f],
        test_id="A02_det2x2_multiplicative",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=30,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


# ============================================================================
# 2x2 Matrix Trace: tr([[a,b],[c,d]]) = a + d
#
# Known RSR: tr(A) = tr(A+R) - tr(R) (additive, trivially linear)
# Also: tr(A) = tr(R^{-1}AR) (conjugation invariance)
# ============================================================================


def test_trace2x2(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    """2x2 trace with additive queries."""

    def f(a, b, c, d):
        return a + d

    def _sp_f(a, b, c, d):
        return a + d

    evaluate(
        domain=Domain.Integer,
        distribution=Distribution(np.random.randint, low=-5, high=5),
        exprs=[
            "f(a1+a2, b1+b2, c1+c2, d1+d2)",  # tr(A+R)
            "f(a1-a2, b1-b2, c1-c2, d1-d2)",  # tr(A-R)
            "f(a1, b1, c1, d1)",  # tr(A)
            "f(a2, b2, c2, d2)",  # tr(R)
        ],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="A03_trace2x2",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=1,
        n=15,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


# ============================================================================
# Quaternion Norm: |q| = sqrt(a^2 + b^2 + c^2 + d^2)
#
# Known RSR: |q| = |q*r| / |r| since the norm is multiplicative.
# Encoding: f(a,b,c,d) = sqrt(a^2 + b^2 + c^2 + d^2)
#
# We test the squared norm (avoids sqrt issues with polynomial recovery):
# |q|^2 = a^2 + b^2 + c^2 + d^2
# |q|^2 = |q+r|^2 - 2*(a1*a2+b1*b2+c1*c2+d1*d2) - |r|^2
# This is a degree-2 additive RSR.
# ============================================================================


def test_quaternion_norm_sq(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    """Squared quaternion norm with additive queries."""

    def f(a, b, c, d):
        return a**2 + b**2 + c**2 + d**2

    def _sp_f(a, b, c, d):
        return a**2 + b**2 + c**2 + d**2

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=[
            "f(a1+a2, b1+b2, c1+c2, d1+d2)",  # |q+r|^2
            "f(a1-a2, b1-b2, c1-c2, d1-d2)",  # |q-r|^2
            "f(a1, b1, c1, d1)",  # |q|^2
            "f(a2, b2, c2, d2)",  # |r|^2
        ],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="A04_quaternion_norm_sq",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=30,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


def test_quaternion_norm_sq_multiplicative(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    """Squared quaternion norm with multiplicative queries.
    Quaternion product: (a1,b1,c1,d1)*(a2,b2,c2,d2) =
      (a1a2-b1b2-c1c2-d1d2, a1b2+b1a2+c1d2-d1c2,
       a1c2-b1d2+c1a2+d1b2, a1d2+b1c2-c1b2+d1a2)
    |q1*q2|^2 = |q1|^2 * |q2|^2
    """

    def norm_sq(a, b, c, d):
        return a**2 + b**2 + c**2 + d**2

    def _sp_norm_sq(a, b, c, d):
        return a**2 + b**2 + c**2 + d**2

    # Quaternion product norm squared (8 inputs)
    def qprod_norm_sq(a1, b1, c1, d1, a2, b2, c2, d2):
        # quaternion product components
        pa = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
        pb = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
        pc = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
        pd = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2
        return pa**2 + pb**2 + pc**2 + pd**2

    def _sp_qprod_norm_sq(a1, b1, c1, d1, a2, b2, c2, d2):
        pa = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
        pb = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
        pc = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
        pd = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2
        return pa**2 + pb**2 + pc**2 + pd**2

    evaluate(
        domain=Domain.Integer,
        distribution=Distribution(np.random.randint, low=-3, high=3),
        exprs=[
            "qprod_norm_sq(a1, b1, c1, d1, a2, b2, c2, d2)",  # |q1*q2|^2
            "norm_sq(a1, b1, c1, d1)",  # |q1|^2
            "norm_sq(a2, b2, c2, d2)",  # |q2|^2
        ],
        infer_funcs=[qprod_norm_sq, norm_sq],
        sympy_funcs=[_sp_qprod_norm_sq, _sp_norm_sq],
        test_id="A05_quaternion_norm_sq_mult",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=30,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


# ============================================================================
# Trace of A^2: tr(A^2) for 2x2 matrices
#
# tr(A^2) = a^2 + d^2 + 2bc  for A = [[a,b],[c,d]]
# This is a degree-2 polynomial in the entries, so it should be RSR.
# RSR: tr(A^2) = tr((A+R)^2) - 2*tr(AR) - tr(R^2)
# ============================================================================


def test_trace_squared_2x2(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    """tr(A^2) for 2x2 matrices — a degree-2 polynomial in matrix entries."""

    def f(a, b, c, d):
        # tr(A^2) where A = [[a,b],[c,d]]
        # A^2 = [[a^2+bc, ab+bd], [ca+dc, cb+d^2]]
        # tr(A^2) = a^2 + 2bc + d^2
        return a**2 + 2 * b * c + d**2

    def _sp_f(a, b, c, d):
        return a**2 + 2 * b * c + d**2

    evaluate(
        domain=Domain.Integer,
        distribution=Distribution(np.random.randint, low=-5, high=5),
        exprs=[
            "f(a1+a2, b1+b2, c1+c2, d1+d2)",  # tr((A+R)^2)
            "f(a1-a2, b1-b2, c1-c2, d1-d2)",  # tr((A-R)^2)
            "f(a1, b1, c1, d1)",  # tr(A^2)
            "f(a2, b2, c2, d2)",  # tr(R^2)
        ],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="A06_trace_squared_2x2",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=30,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


# ============================================================================
# Elementary Symmetric Polynomials
#
# e2(x,y,z) = xy + xz + yz
# e3(x,y,z) = xyz
# These are degree 2 and 3 polynomials, hence RSR by Fact 4.2.
# ============================================================================


def test_elem_sym_e2(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    """Elementary symmetric polynomial e2(x,y,z) = xy + xz + yz."""

    def f(x, y, z):
        return x * y + x * z + y * z

    def _sp_f(x, y, z):
        return x * y + x * z + y * z

    evaluate(
        domain=Domain.Integer,
        distribution=Distribution(np.random.randint, low=-5, high=5),
        exprs=[
            "f(x1+x2, y1+y2, z1+z2)",  # e2(x+r, y+s, z+t)
            "f(x1-x2, y1-y2, z1-z2)",  # e2(x-r, y-s, z-t)
            "f(x1, y1, z1)",  # e2(x, y, z)
            "f(x2, y2, z2)",  # e2(r, s, t)
        ],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="A07_elem_sym_e2",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=30,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


def test_elem_sym_e3(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    """Elementary symmetric polynomial e3(x,y,z) = xyz."""

    def f(x, y, z):
        return x * y * z

    def _sp_f(x, y, z):
        return x * y * z

    evaluate(
        domain=Domain.Integer,
        distribution=Distribution(np.random.randint, low=-5, high=5),
        exprs=[
            "f(x1+x2, y1+y2, z1+z2)",
            "f(x1-x2, y1-y2, z1-z2)",
            "f(x1, y1, z1)",
            "f(x2, y2, z2)",
        ],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="A08_elem_sym_e3",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=3,
        n=50,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


# ============================================================================
# Power sum p2(x,y,z) = x^2 + y^2 + z^2
# Relates to e2 via Newton's identity: p2 = e1^2 - 2*e2
# Degree-2 symmetric polynomial — should be easily RSR.
# ============================================================================


def test_power_sum_p2(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    """Power sum p2(x,y,z) = x^2 + y^2 + z^2."""

    def f(x, y, z):
        return x**2 + y**2 + z**2

    def _sp_f(x, y, z):
        return x**2 + y**2 + z**2

    evaluate(
        domain=Domain.Integer,
        distribution=Distribution(np.random.randint, low=-5, high=5),
        exprs=[
            "f(x1+x2, y1+y2, z1+z2)",
            "f(x1-x2, y1-y2, z1-z2)",
            "f(x1, y1, z1)",
            "f(x2, y2, z2)",
        ],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="A09_power_sum_p2",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=30,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


# ============================================================================
# Bilinear dot product: f(x1,x2,y1,y2) = x1*y1 + x2*y2
# A proxy for inner products in higher dimensions.
# RSR: f(x+r) = f(x) + <x,r> + <r,x> + f(r)  (expanding the bilinear form)
# ============================================================================


def test_dot_product_2d(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    """2D dot product: f(x1,x2) = x1^2 + x2^2 (self-inner-product / squared norm)."""

    def f(x, y):
        return x**2 + y**2

    def _sp_f(x, y):
        return x**2 + y**2

    evaluate(
        domain=Domain.Integer,
        distribution=Distribution(np.random.randint, low=-5, high=5),
        exprs=[
            "f(x1+x2, y1+y2)",  # |v+r|^2
            "f(x1-x2, y1-y2)",  # |v-r|^2
            "f(x1, y1)",  # |v|^2
            "f(x2, y2)",  # |r|^2
        ],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="A10_dot_product_2d",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=20,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


# ============================================================================
# Cross product magnitude (3D): |a × b| for fixed b
# f(x,y,z) = sqrt((y*bz - z*by)^2 + (z*bx - x*bz)^2 + (x*by - y*bx)^2)
# We use the squared version to keep polynomial recovery.
# ============================================================================


def test_cross_product_sq(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    """Squared cross product magnitude |a × b|^2 for 3D vectors."""

    def f(x1, y1, z1, x2, y2, z2):
        cx = y1 * z2 - z1 * y2
        cy = z1 * x2 - x1 * z2
        cz = x1 * y2 - y1 * x2
        return cx**2 + cy**2 + cz**2

    def _sp_f(x1, y1, z1, x2, y2, z2):
        cx = y1 * z2 - z1 * y2
        cy = z1 * x2 - x1 * z2
        cz = x1 * y2 - y1 * x2
        return cx**2 + cy**2 + cz**2

    evaluate(
        domain=Domain.Integer,
        distribution=Distribution(np.random.randint, low=-3, high=3),
        exprs=[
            "f(x1+a1, y1+a2, z1+a3, x2, y2, z2)",  # |(a+r) × b|^2
            "f(x1-a1, y1-a2, z1-a3, x2, y2, z2)",  # |(a-r) × b|^2
            "f(x1, y1, z1, x2, y2, z2)",  # |a × b|^2
            "f(a1, a2, a3, x2, y2, z2)",  # |r × b|^2
        ],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="A11_cross_product_sq",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=50,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    res_dir = args.res_dir
    os.makedirs(res_dir, exist_ok=True)

    method = args.method
    milp = args.milp
    timeout_sec = args.timeout_sec

    test_args = (res_dir, method, milp, timeout_sec)

    st = time()

    # Matrix functions
    test_det2x2_additive(*test_args)  # A01
    test_det2x2_multiplicative(*test_args)  # A02
    test_trace2x2(*test_args)  # A03

    # Quaternion norm
    test_quaternion_norm_sq(*test_args)  # A04
    test_quaternion_norm_sq_multiplicative(*test_args)  # A05

    # Matrix polynomial: tr(A^2)
    test_trace_squared_2x2(*test_args)  # A06

    # Symmetric polynomials
    test_elem_sym_e2(*test_args)  # A07
    test_elem_sym_e3(*test_args)  # A08
    test_power_sum_p2(*test_args)  # A09

    # Vector operations
    test_dot_product_2d(*test_args)  # A10
    test_cross_product_sq(*test_args)  # A11

    log.info(f"\nTotal Time: {time() - st:.2f}s")
