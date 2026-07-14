"""
Agentic Algebraic RSR experiments for ICML 2026 rebuttal (Reviewer E7bc).

Tests A-Bitween on algebraic structures beyond scalar real-valued functions:
  - 2x2 matrix determinant (additive and multiplicative RSR)
  - 2x2 matrix trace (additive RSR)
  - Quaternion norm squared (additive and multiplicative RSR)
  - Trace of A^2 (degree-2 polynomial)
  - Symmetric polynomials (e2, e3)
  - Power sum p2
  - Dot product (2D squared norm)
  - Cross product magnitude squared

These functions are encoded as multi-variable scalar functions so they fit
directly into Bitween's existing framework without any code changes.

Usage:
    python -m bitween.evaluation.evaluation_rsr_bench_agentic_algebraic \
        --agent_type bedrock \
        --model_id us.anthropic.claude-sonnet-4-20250514-v1:0 \
        --region_name us-west-2 \
        --res_dir ./results/agentic_algebraic \
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
# A01: 2x2 Matrix Determinant (additive queries)
# ============================================================================


def test_det2x2_additive(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
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
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="A01_det2x2_additive",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


# ============================================================================
# A02: 2x2 Matrix Determinant (multiplicative queries)
# ============================================================================


def test_det2x2_multiplicative(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    """2x2 determinant with multiplicative queries.
    det(A*R) = det(A)*det(R).
    """

    def f(a, b, c, d):
        return a * d - b * c

    def _sp_f(a, b, c, d):
        return a * d - b * c

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
        infer_funcs=[det_product, f],
        sympy_funcs=[_sp_det_product, _sp_f],
        test_id="A02_det2x2_multiplicative",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


# ============================================================================
# A03: 2x2 Matrix Trace (additive queries)
# ============================================================================


def test_trace2x2(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
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
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="A03_trace2x2",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


# ============================================================================
# A04: Quaternion Norm Squared (additive queries)
# ============================================================================


def test_quaternion_norm_sq(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
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
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="A04_quaternion_norm_sq",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


# ============================================================================
# A05: Quaternion Norm Squared (multiplicative queries)
# ============================================================================


def test_quaternion_norm_sq_multiplicative(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    """Squared quaternion norm with multiplicative queries.
    |q1*q2|^2 = |q1|^2 * |q2|^2
    """

    def norm_sq(a, b, c, d):
        return a**2 + b**2 + c**2 + d**2

    def _sp_norm_sq(a, b, c, d):
        return a**2 + b**2 + c**2 + d**2

    def qprod_norm_sq(a1, b1, c1, d1, a2, b2, c2, d2):
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
        infer_funcs=[qprod_norm_sq, norm_sq],
        sympy_funcs=[_sp_qprod_norm_sq, _sp_norm_sq],
        test_id="A05_quaternion_norm_sq_mult",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


# ============================================================================
# A06: Trace of A^2 for 2x2 matrices
# ============================================================================


def test_trace_squared_2x2(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    """tr(A^2) for 2x2 matrices: a^2 + 2bc + d^2."""

    def f(a, b, c, d):
        return a**2 + 2 * b * c + d**2

    def _sp_f(a, b, c, d):
        return a**2 + 2 * b * c + d**2

    evaluate(
        domain=Domain.Integer,
        distribution=Distribution(np.random.randint, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="A06_trace_squared_2x2",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


# ============================================================================
# A07: Elementary Symmetric Polynomial e2(x,y,z) = xy + xz + yz
# ============================================================================


def test_elem_sym_e2(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
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
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="A07_elem_sym_e2",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


# ============================================================================
# A08: Elementary Symmetric Polynomial e3(x,y,z) = xyz
# ============================================================================


def test_elem_sym_e3(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
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
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="A08_elem_sym_e3",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


# ============================================================================
# A09: Power Sum p2(x,y,z) = x^2 + y^2 + z^2
# ============================================================================


def test_power_sum_p2(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
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
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="A09_power_sum_p2",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


# ============================================================================
# A10: 2D Dot Product (squared norm)
# ============================================================================


def test_dot_product_2d(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    """2D dot product: f(x,y) = x^2 + y^2 (squared norm)."""

    def f(x, y):
        return x**2 + y**2

    def _sp_f(x, y):
        return x**2 + y**2

    evaluate(
        domain=Domain.Integer,
        distribution=Distribution(np.random.randint, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="A10_dot_product_2d",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


# ============================================================================
# A11: Cross Product Magnitude Squared
# ============================================================================


def test_cross_product_sq(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    """Squared cross product magnitude |a x b|^2 for 3D vectors."""

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
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="A11_cross_product_sq",
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
