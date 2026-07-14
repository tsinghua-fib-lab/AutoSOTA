"""
Extended Algebraic RSR experiments for ICML 2026 camera-ready (Reviewer E7bc).

V-Bitween tests on advanced algebraic structures:
  - Octonion norm squared (additive and multiplicative)
  - Clifford algebra Cl(3,0) conjugation norm
  - Clifford algebra Cl(2,0) determinant (multiplicative)
  - Lie bracket tr([A,B]^2) for gl(2)
  - sl(2) Killing form

Usage:
    python -m bitween.evaluation.evaluation_rsr_bench_paper_algebraic_extended \
        --res_dir results/algebraic_extended --method MULTIPLE_REGRESSION
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
# Helper: Quaternion multiplication (used by octonion Cayley-Dickson product)
# ============================================================================


def _qmul(a1, b1, c1, d1, a2, b2, c2, d2):
    """Quaternion product (a1+b1i+c1j+d1k)(a2+b2i+c2j+d2k)."""
    return (
        a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
        a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
        a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
        a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
    )


# ============================================================================
# Helper: Cl(2,0) geometric product
# ============================================================================


def _cl20_product(s1, a1, b1, c1, s2, a2, b2, c2):
    """Geometric product in Cl(2,0). Returns (scalar, e1, e2, e12) components."""
    ps = s1 * s2 + a1 * a2 + b1 * b2 - c1 * c2
    pa = s1 * a2 + a1 * s2 - b1 * c2 + c1 * b2
    pb = s1 * b2 + b1 * s2 + a1 * c2 - c1 * a2
    pc = s1 * c2 + a1 * b2 - b1 * a2 + c1 * s2
    return (ps, pa, pb, pc)


# ============================================================================
# A12: Octonion Norm Squared (additive queries)
# ============================================================================


def test_octonion_norm_sq(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    """Squared octonion norm with additive queries: 8 variables, degree 2."""

    def f(a, b, c, d, p, q, r, s):
        return a**2 + b**2 + c**2 + d**2 + p**2 + q**2 + r**2 + s**2

    def _sp_f(a, b, c, d, p, q, r, s):
        return a**2 + b**2 + c**2 + d**2 + p**2 + q**2 + r**2 + s**2

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=[
            "f(a1+a2, b1+b2, c1+c2, d1+d2, p1+p2, q1+q2, r1+r2, s1+s2)",
            "f(a1-a2, b1-b2, c1-c2, d1-d2, p1-p2, q1-q2, r1-r2, s1-s2)",
            "f(a1, b1, c1, d1, p1, q1, r1, s1)",
            "f(a2, b2, c2, d2, p2, q2, r2, s2)",
        ],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="A12_octonion_norm_sq",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=30,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


# ============================================================================
# A13: Octonion Product Norm Squared (multiplicative / Moufang loop)
# ============================================================================


def test_octonion_norm_sq_multiplicative(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    """Squared octonion norm multiplicativity: |o1*o2|^2 = |o1|^2 * |o2|^2."""

    def oct_norm_sq(a, b, c, d, p, q, r, s):
        return a**2 + b**2 + c**2 + d**2 + p**2 + q**2 + r**2 + s**2

    def _sp_oct_norm_sq(a, b, c, d, p, q, r, s):
        return a**2 + b**2 + c**2 + d**2 + p**2 + q**2 + r**2 + s**2

    def oct_prod_norm_sq(
        a1, b1, c1, d1, p1, q1, r1, s1, a2, b2, c2, d2, p2, q2, r2, s2
    ):
        """Norm squared of octonion product via Cayley-Dickson."""
        pp = _qmul(a1, b1, c1, d1, a2, b2, c2, d2)
        cq2_q1 = _qmul(p2, -q2, -r2, -s2, p1, q1, r1, s1)
        rp = tuple(pp[i] - cq2_q1[i] for i in range(4))

        q2p1 = _qmul(p2, q2, r2, s2, a1, b1, c1, d1)
        q1cp2 = _qmul(p1, q1, r1, s1, a2, -b2, -c2, -d2)
        rq = tuple(q2p1[i] + q1cp2[i] for i in range(4))

        return sum(x**2 for x in rp) + sum(x**2 for x in rq)

    def _sp_oct_prod_norm_sq(
        a1, b1, c1, d1, p1, q1, r1, s1, a2, b2, c2, d2, p2, q2, r2, s2
    ):
        pp = _qmul(a1, b1, c1, d1, a2, b2, c2, d2)
        cq2_q1 = _qmul(p2, -q2, -r2, -s2, p1, q1, r1, s1)
        rp = tuple(pp[i] - cq2_q1[i] for i in range(4))

        q2p1 = _qmul(p2, q2, r2, s2, a1, b1, c1, d1)
        q1cp2 = _qmul(p1, q1, r1, s1, a2, -b2, -c2, -d2)
        rq = tuple(q2p1[i] + q1cp2[i] for i in range(4))

        return sum(x**2 for x in rp) + sum(x**2 for x in rq)

    evaluate(
        domain=Domain.Integer,
        distribution=Distribution(np.random.randint, low=-3, high=3),
        exprs=[
            "oct_prod_norm_sq(a1, b1, c1, d1, p1, q1, r1, s1,"
            " a2, b2, c2, d2, p2, q2, r2, s2)",
            "oct_norm_sq(a1, b1, c1, d1, p1, q1, r1, s1)",
            "oct_norm_sq(a2, b2, c2, d2, p2, q2, r2, s2)",
        ],
        infer_funcs=[oct_prod_norm_sq, oct_norm_sq],
        sympy_funcs=[_sp_oct_prod_norm_sq, _sp_oct_norm_sq],
        test_id="A13_octonion_norm_sq_mult",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=30,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


# ============================================================================
# A14: Clifford Cl(3,0) Conjugation Norm (additive queries)
# Indefinite quadratic form with signature (4,4).
# ============================================================================


def test_cl3_conj_norm(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    """Cl(3,0) conjugation norm: signature (4,4) indefinite quadratic form."""

    def f(s, v1, v2, v3, b1, b2, b3, t):
        return s**2 - v1**2 - v2**2 - v3**2 + b1**2 + b2**2 + b3**2 - t**2

    def _sp_f(s, v1, v2, v3, b1, b2, b3, t):
        return s**2 - v1**2 - v2**2 - v3**2 + b1**2 + b2**2 + b3**2 - t**2

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=[
            "f(s1+s2, v11+v12, v21+v22, v31+v32,"
            " b11+b12, b21+b22, b31+b32, t1+t2)",
            "f(s1-s2, v11-v12, v21-v22, v31-v32,"
            " b11-b12, b21-b22, b31-b32, t1-t2)",
            "f(s1, v11, v21, v31, b11, b21, b31, t1)",
            "f(s2, v12, v22, v32, b12, b22, b32, t2)",
        ],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="A14_cl3_conj_norm",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=30,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


# ============================================================================
# A15: Clifford Cl(2,0) Determinant (multiplicative)
# det(x) = s^2 - a^2 - b^2 + c^2, signature (2,2).
# det(xy) = det(x)*det(y) via Cl(2,0) ~ M_2(R).
# ============================================================================


def test_cl2_det_multiplicative(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    """Cl(2,0) determinant multiplicativity: det(xy) = det(x)*det(y)."""

    def cl20_det(s, a, b, c):
        return s**2 - a**2 - b**2 + c**2

    def _sp_cl20_det(s, a, b, c):
        return s**2 - a**2 - b**2 + c**2

    def cl20_prod_det(s1, a1, b1, c1, s2, a2, b2, c2):
        ps, pa, pb, pc = _cl20_product(s1, a1, b1, c1, s2, a2, b2, c2)
        return ps**2 - pa**2 - pb**2 + pc**2

    def _sp_cl20_prod_det(s1, a1, b1, c1, s2, a2, b2, c2):
        ps, pa, pb, pc = _cl20_product(s1, a1, b1, c1, s2, a2, b2, c2)
        return ps**2 - pa**2 - pb**2 + pc**2

    evaluate(
        domain=Domain.Integer,
        distribution=Distribution(np.random.randint, low=-3, high=3),
        exprs=[
            "cl20_prod_det(s1, a1, b1, c1, s2, a2, b2, c2)",
            "cl20_det(s1, a1, b1, c1)",
            "cl20_det(s2, a2, b2, c2)",
        ],
        infer_funcs=[cl20_prod_det, cl20_det],
        sympy_funcs=[_sp_cl20_prod_det, _sp_cl20_det],
        test_id="A15_cl2_det_mult",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=30,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


# ============================================================================
# A16: Lie Bracket tr([A,B]^2) for gl(2) — degree-4 polynomial
# ============================================================================


def test_lie_bracket_trace_sq(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    """tr([A,B]^2) for 2x2 matrices: degree-4 Lie algebra invariant."""

    def f(a1, b1, c1, d1, a2, b2, c2, d2):
        m11 = b1 * c2 - b2 * c1
        m12 = a1 * b2 - a2 * b1 + b1 * d2 - b2 * d1
        m21 = c1 * a2 - c2 * a1 + d1 * c2 - d2 * c1
        return 2 * m11**2 + 2 * m12 * m21

    def _sp_f(a1, b1, c1, d1, a2, b2, c2, d2):
        m11 = b1 * c2 - b2 * c1
        m12 = a1 * b2 - a2 * b1 + b1 * d2 - b2 * d1
        m21 = c1 * a2 - c2 * a1 + d1 * c2 - d2 * c1
        return 2 * m11**2 + 2 * m12 * m21

    evaluate(
        domain=Domain.Integer,
        distribution=Distribution(np.random.randint, low=-3, high=3),
        exprs=[
            "f(a1+p1, b1+p2, c1+p3, d1+p4, a2+q1, b2+q2, c2+q3, d2+q4)",
            "f(a1-p1, b1-p2, c1-p3, d1-p4, a2-q1, b2-q2, c2-q3, d2-q4)",
            "f(a1, b1, c1, d1, a2, b2, c2, d2)",
            "f(p1, p2, p3, p4, q1, q2, q3, q4)",
        ],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="A16_lie_bracket_trace_sq",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=4,
        n=100,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


# ============================================================================
# A17: sl(2) Killing Form — degree-2 polynomial
# f(a,b,c) = a^2 + bc = -det([[a,b],[c,-a]])
# ============================================================================


def test_sl2_killing(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    """sl(2) Killing form: a^2 + bc for traceless X = [[a,b],[c,-a]]."""

    def f(a, b, c):
        return a**2 + b * c

    def _sp_f(a, b, c):
        return a**2 + b * c

    evaluate(
        domain=Domain.Integer,
        distribution=Distribution(np.random.randint, low=-5, high=5),
        exprs=[
            "f(a1+a2, b1+b2, c1+c2)",
            "f(a1-a2, b1-b2, c1-c2)",
            "f(a1, b1, c1)",
            "f(a2, b2, c2)",
        ],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="A17_sl2_killing",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=30,
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

    # Octonions (Moufang loop)
    test_octonion_norm_sq(*test_args)  # A12
    test_octonion_norm_sq_multiplicative(*test_args)  # A13

    # Clifford algebras
    test_cl3_conj_norm(*test_args)  # A14
    test_cl2_det_multiplicative(*test_args)  # A15

    # Lie algebras
    test_lie_bracket_trace_sq(*test_args)  # A16
    test_sl2_killing(*test_args)  # A17

    log.info(f"\nTotal Time: {time() - st:.2f}s")
