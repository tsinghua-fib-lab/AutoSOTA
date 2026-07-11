"""Lightweight SIGS smoke test.

This script is intentionally small and data-free. It verifies that the package can
be imported, the grammar exposes valid production masks, expressions can be
classified, and a simple closed-form heat-equation solution has zero symbolic
residual. It is meant for quick installation checks and CI, not for reproducing
paper-scale experiments.
"""

from __future__ import annotations

import sympy as sp

from sigs.grammar import GCFG, S, T, get_mask
from sigs.utils import ExpressionUtils, MathClass


def check_grammar() -> None:
    productions = GCFG.productions()
    assert len(productions) > 0, "grammar has no productions"

    s_mask = get_mask(S, GCFG)
    t_mask = get_mask(T, GCFG)

    assert len(s_mask) == len(productions), "S mask length does not match grammar"
    assert len(t_mask) == len(productions), "T mask length does not match grammar"
    assert any(s_mask), "S has no admissible productions"
    assert any(t_mask), "T has no admissible productions"


def check_expression_classification() -> None:
    flags = ExpressionUtils.parse_expression("x*t")
    assert not flags.parse_error, "failed to parse a simple expression"
    assert flags.math_class == MathClass.SPATIOTEMPORAL_2D, flags.math_class

    flags = ExpressionUtils.parse_expression("x*y")
    assert not flags.parse_error, "failed to parse a two-dimensional spatial expression"
    assert flags.math_class == MathClass.SPATIAL_2D, flags.math_class


def check_symbolic_residual() -> None:
    x, t = sp.symbols("x t")
    u = sp.sin(sp.pi * x) * sp.exp(-(sp.pi**2) * t)

    # Heat equation u_t = u_xx, written as u_t - u_xx = 0.
    residual = sp.simplify(sp.diff(u, t) - sp.diff(u, x, 2))
    assert residual == 0, f"unexpected symbolic residual: {residual}"


def main() -> None:
    check_grammar()
    check_expression_classification()
    check_symbolic_residual()
    print("SIGS smoke test passed.")


if __name__ == "__main__":
    main()
