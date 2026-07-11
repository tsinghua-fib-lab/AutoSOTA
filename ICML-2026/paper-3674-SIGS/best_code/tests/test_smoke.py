"""Data-free tests for the public SIGS package.

These tests mirror the lightweight smoke test used in CI. They are intentionally
small so that contributors and reviewers can verify the installation without
model checkpoints, cluster databases, or paper-scale experiment artifacts.
"""

import sympy as sp

from sigs.grammar import GCFG, S, T, get_mask
from sigs.utils import ExpressionUtils, MathClass


def test_grammar_masks_are_nonempty_and_aligned():
    productions = GCFG.productions()
    assert len(productions) > 0

    s_mask = get_mask(S, GCFG)
    t_mask = get_mask(T, GCFG)

    assert len(s_mask) == len(productions)
    assert len(t_mask) == len(productions)
    assert any(s_mask)
    assert any(t_mask)


def test_expression_classification_for_basic_variable_signatures():
    spatiotemporal = ExpressionUtils.parse_expression("x*t")
    assert not spatiotemporal.parse_error
    assert spatiotemporal.math_class == MathClass.SPATIOTEMPORAL_2D

    spatial_2d = ExpressionUtils.parse_expression("x*y")
    assert not spatial_2d.parse_error
    assert spatial_2d.math_class == MathClass.SPATIAL_2D


def test_symbolic_heat_equation_residual_is_zero():
    x, t = sp.symbols("x t")
    u = sp.sin(sp.pi * x) * sp.exp(-(sp.pi**2) * t)

    residual = sp.simplify(sp.diff(u, t) - sp.diff(u, x, 2))
    assert residual == 0
