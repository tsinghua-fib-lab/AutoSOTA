"""
Extended Algebraic RSR experiments for ICML 2026 camera-ready (Reviewer E7bc).

Tests A-Bitween on advanced algebraic structures requested by Reviewer E7bc:
  - Octonion norm squared (additive and multiplicative)
  - Clifford algebra Cl(3,0) conjugation norm
  - Clifford algebra Cl(2,0) determinant (multiplicative)
  - Lie bracket tr([A,B]^2) for gl(2)
  - sl(2) Killing form

Octonion multiplication uses the Cayley-Dickson construction from quaternions.
The unit octonions form a Moufang loop, so the multiplicative norm test (A13)
demonstrates the Moufang loop norm property |o1*o2|^2 = |o1|^2 * |o2|^2.

Clifford algebras Cl(n,0) generalize quaternions (Cl(0,2) ~ H).
Cl(2,0) ~ M_2(R) (2x2 real matrices) and Cl(3,0) ~ M_2(C).

Usage:
    python -m bitween.evaluation.evaluation_rsr_bench_agentic_paper_algebraic_extended \
        --agent_type bedrock \
        --model_id us.anthropic.claude-opus-4-1-20250805-v1:0 \
        --region_name us-west-2 \
        --res_dir ./results/agentic_algebraic_extended \
        --enable_thinking
"""

import os
from time import time

import numpy as np

from bitween.agent import BaseAgent
from bitween.config import Config
from bitween.evaluation.evaluation_rsr_bench_agentic_paper import (
    evaluate,
    get_agent_from_args,
    get_parser,
)
from bitween.miscs import getLogger
from bitween.sampler import Distribution, Domain

config = Config()
log = getLogger(__name__, config.logger_level, empty_format=True)


# ============================================================================
# Helper: Quaternion multiplication (used by octonion Cayley-Dickson product)
# (a,b,c,d) represents quaternion a + bi + cj + dk
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
# A12: Octonion Norm Squared (additive queries)
#
# Octonion o = (a,b,c,d,p,q,r,s) has |o|^2 = a^2+b^2+c^2+d^2+p^2+q^2+r^2+s^2
# This is the sum-of-squares norm on R^8. Degree-2 additive RSR via
# polarization: f(o+t) + f(o-t) - 2f(o) - 2f(t) = 0.
# ============================================================================


def test_octonion_norm_sq(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    """Squared octonion norm with additive queries."""

    def f(a, b, c, d, p, q, r, s):
        return a**2 + b**2 + c**2 + d**2 + p**2 + q**2 + r**2 + s**2

    def _sp_f(a, b, c, d, p, q, r, s):
        return a**2 + b**2 + c**2 + d**2 + p**2 + q**2 + r**2 + s**2

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="A12_octonion_norm_sq",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


# ============================================================================
# A13: Octonion Product Norm Squared (multiplicative / Moufang loop)
#
# Octonions are a normed division algebra (Cayley-Dickson doubling of
# quaternions). Despite being non-associative, the norm is multiplicative:
#   |o1 * o2|^2 = |o1|^2 * |o2|^2   (Hurwitz's theorem)
#
# The unit octonions form a Moufang loop — a non-associative algebraic
# structure satisfying the Moufang identities.
#
# Cayley-Dickson product: o = (p, q) where p, q are quaternions.
#   (p1,q1) * (p2,q2) = (p1*p2 - conj(q2)*q1, q2*p1 + q1*conj(p2))
# ============================================================================


def test_octonion_norm_sq_multiplicative(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    """Squared octonion norm with multiplicative queries (Moufang loop)."""

    def oct_norm_sq(a, b, c, d, p, q, r, s):
        return a**2 + b**2 + c**2 + d**2 + p**2 + q**2 + r**2 + s**2

    def _sp_oct_norm_sq(a, b, c, d, p, q, r, s):
        return a**2 + b**2 + c**2 + d**2 + p**2 + q**2 + r**2 + s**2

    def oct_prod_norm_sq(
        a1, b1, c1, d1, p1, q1, r1, s1, a2, b2, c2, d2, p2, q2, r2, s2
    ):
        """Norm squared of octonion product via Cayley-Dickson.

        o1 = (quat_P1, quat_Q1) where P1=(a1,b1,c1,d1), Q1=(p1,q1,r1,s1)
        o2 = (quat_P2, quat_Q2) where P2=(a2,b2,c2,d2), Q2=(p2,q2,r2,s2)

        Product: (P1*P2 - conj(Q2)*Q1, Q2*P1 + Q1*conj(P2))
        """
        # result_part1 = P1*P2 - conj(Q2)*Q1
        pp = _qmul(a1, b1, c1, d1, a2, b2, c2, d2)
        cq2_q1 = _qmul(p2, -q2, -r2, -s2, p1, q1, r1, s1)
        rp = tuple(pp[i] - cq2_q1[i] for i in range(4))

        # result_part2 = Q2*P1 + Q1*conj(P2)
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
        infer_funcs=[oct_prod_norm_sq, oct_norm_sq],
        sympy_funcs=[_sp_oct_prod_norm_sq, _sp_oct_norm_sq],
        test_id="A13_octonion_norm_sq_mult",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


# ============================================================================
# A14: Clifford Algebra Cl(3,0) Conjugation Norm (additive queries)
#
# Cl(3,0) has basis {1, e1, e2, e3, e12, e13, e23, e123} with e_i^2 = +1.
# A multivector x = s + v1*e1 + v2*e2 + v3*e3 + b1*e12 + b2*e13 + b3*e23 + t*e123
#
# The Clifford conjugate x_bar = alpha(x~) (grade involution composed with
# reversal) negates grades 1 and 2, preserves grades 0 and 3:
#   x_bar = s - v1*e1 - v2*e2 - v3*e3 - b1*e12 - b2*e13 - b3*e23 + t*e123
#
# The scalar part of x * x_bar is:
#   <x * x_bar>_0 = s^2 - v1^2 - v2^2 - v3^2 + b1^2 + b2^2 + b3^2 - t^2
#
# This is an INDEFINITE quadratic form with signature (4,4), contrasting
# with the positive-definite norms in the quaternion/octonion benchmarks.
# ============================================================================


def test_cl3_conj_norm(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    """Cl(3,0) conjugation norm: indefinite quadratic form, signature (4,4)."""

    def f(s, v1, v2, v3, b1, b2, b3, t):
        # Scalar part of x * x_bar in Cl(3,0)
        # Signs: grade 0 (+), grade 1 (-), grade 2 (+), grade 3 (-)
        # because e_i^2 = +1 but conjugation negates grades 1,2
        # and e_{ij}^2 = -1, e_{123}^2 = -1
        return s**2 - v1**2 - v2**2 - v3**2 + b1**2 + b2**2 + b3**2 - t**2

    def _sp_f(s, v1, v2, v3, b1, b2, b3, t):
        return s**2 - v1**2 - v2**2 - v3**2 + b1**2 + b2**2 + b3**2 - t**2

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="A14_cl3_conj_norm",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


# ============================================================================
# A15: Clifford Algebra Cl(2,0) Determinant (multiplicative)
#
# Cl(2,0) has basis {1, e1, e2, e12} with e1^2 = e2^2 = +1, e12^2 = -1.
# Cl(2,0) is isomorphic to M_2(R) via:
#   x = s + a*e1 + b*e2 + c*e12  -->  [[s+a, b+c], [b-c, s-a]]
#
# The determinant is: det(x) = s^2 - a^2 - b^2 + c^2
# This is an indefinite quadratic form with signature (2,2).
#
# Since Cl(2,0) ~ M_2(R) and the geometric product corresponds to matrix
# multiplication, the determinant is multiplicative:
#   det(x * y) = det(x) * det(y)
# ============================================================================


def _cl20_product(s1, a1, b1, c1, s2, a2, b2, c2):
    """Geometric product in Cl(2,0). Returns (scalar, e1, e2, e12) components.

    Multiplication rules: e1^2=1, e2^2=1, e1*e2=e12, e2*e1=-e12,
    e1*e12=e2, e12*e1=-e2, e2*e12=-e1, e12*e2=e1, e12^2=-1.
    """
    ps = s1 * s2 + a1 * a2 + b1 * b2 - c1 * c2
    pa = s1 * a2 + a1 * s2 - b1 * c2 + c1 * b2
    pb = s1 * b2 + b1 * s2 + a1 * c2 - c1 * a2
    pc = s1 * c2 + a1 * b2 - b1 * a2 + c1 * s2
    return (ps, pa, pb, pc)


def test_cl2_det_multiplicative(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    """Cl(2,0) determinant multiplicativity: det(xy) = det(x)*det(y)."""

    def cl20_det(s, a, b, c):
        return s**2 - a**2 - b**2 + c**2

    def _sp_cl20_det(s, a, b, c):
        return s**2 - a**2 - b**2 + c**2

    def cl20_prod_det(s1, a1, b1, c1, s2, a2, b2, c2):
        """Determinant of geometric product in Cl(2,0)."""
        ps, pa, pb, pc = _cl20_product(s1, a1, b1, c1, s2, a2, b2, c2)
        return ps**2 - pa**2 - pb**2 + pc**2

    def _sp_cl20_prod_det(s1, a1, b1, c1, s2, a2, b2, c2):
        ps, pa, pb, pc = _cl20_product(s1, a1, b1, c1, s2, a2, b2, c2)
        return ps**2 - pa**2 - pb**2 + pc**2

    evaluate(
        domain=Domain.Integer,
        distribution=Distribution(np.random.randint, low=-3, high=3),
        infer_funcs=[cl20_prod_det, cl20_det],
        sympy_funcs=[_sp_cl20_prod_det, _sp_cl20_det],
        test_id="A15_cl2_det_mult",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


# ============================================================================
# A16: Lie Bracket Trace Squared for gl(2)
#
# For 2x2 matrices A, B, the Lie bracket is [A,B] = AB - BA.
# The commutator is always traceless: tr([A,B]) = 0.
# By Cayley-Hamilton for 2x2: tr([A,B]^2) = -2*det([A,B]).
#
# Explicitly: tr([A,B]^2) = 2*(b1*c2 - b2*c1)^2
#   + 2*(a1*b2 - a2*b1 + b1*d2 - b2*d1)*(c1*a2 - c2*a1 + d1*c2 - d2*c1)
#
# This is a degree-4 polynomial in 8 variables — the highest degree we test,
# probing whether A-Bitween can handle non-trivial polynomial RSRs.
# ============================================================================


def test_lie_bracket_trace_sq(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    """tr([A,B]^2) for 2x2 matrices: degree-4 Lie algebra invariant."""

    def f(a1, b1, c1, d1, a2, b2, c2, d2):
        # [A,B] = AB - BA components
        m11 = b1 * c2 - b2 * c1
        m12 = a1 * b2 - a2 * b1 + b1 * d2 - b2 * d1
        m21 = c1 * a2 - c2 * a1 + d1 * c2 - d2 * c1
        # m22 = -m11 (traceless)
        # tr([A,B]^2) = 2*m11^2 + 2*m12*m21
        return 2 * m11**2 + 2 * m12 * m21

    def _sp_f(a1, b1, c1, d1, a2, b2, c2, d2):
        m11 = b1 * c2 - b2 * c1
        m12 = a1 * b2 - a2 * b1 + b1 * d2 - b2 * d1
        m21 = c1 * a2 - c2 * a1 + d1 * c2 - d2 * c1
        return 2 * m11**2 + 2 * m12 * m21

    evaluate(
        domain=Domain.Integer,
        distribution=Distribution(np.random.randint, low=-3, high=3),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="A16_lie_bracket_trace_sq",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


# ============================================================================
# A17: sl(2) Killing Form
#
# For the Lie algebra sl(2,R) of traceless 2x2 matrices:
#   X = [[a, b], [c, -a]]
#
# The Killing form is B(X,Y) = 4*tr(XY). The quadratic form is:
#   B(X,X) = 4*tr(X^2) = 4*(2a^2 + 2bc) = 8a^2 + 8bc
#
# We test f(a,b,c) = a^2 + bc (proportional to the Killing form, and
# equal to -det(X)). This is a degree-2 polynomial in 3 variables.
#
# The Killing form determines the structure of a semisimple Lie algebra
# (Cartan's criterion). For sl(2), it has signature (2,1).
# ============================================================================


def test_sl2_killing(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    """sl(2) Killing form: B(X,X)/8 = a^2 + bc for X = [[a,b],[c,-a]]."""

    def f(a, b, c):
        return a**2 + b * c

    def _sp_f(a, b, c):
        return a**2 + b * c

    evaluate(
        domain=Domain.Integer,
        distribution=Distribution(np.random.randint, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="A17_sl2_killing",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    agent = get_agent_from_args(args)

    res_dir = args.res_dir
    os.makedirs(res_dir, exist_ok=True)

    custom_tools = args.custom_tools
    mcp_tools = args.mcp_tools
    timeout_sec = args.timeout_sec

    test_args = (agent, res_dir, custom_tools, mcp_tools, timeout_sec)

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
