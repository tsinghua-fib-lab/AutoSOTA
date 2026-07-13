"""Sanity checks for the scalar root-finding helpers."""

from __future__ import annotations

import math

import pytest

from dropout_mft.numerics import (
    find_positive_root,
    solve_bracketed,
    solve_with_expanding_upper,
)


def test_solve_bracketed_finds_known_root():
    # x**2 - 2 has a root at sqrt(2) on [0, 2].
    root = solve_bracketed(lambda x: x * x - 2.0, (0.0, 2.0))
    assert root == pytest.approx(math.sqrt(2.0))


def test_solve_bracketed_returns_endpoint_root():
    assert solve_bracketed(lambda x: x - 1.0, (1.0, 3.0)) == pytest.approx(1.0)


def test_find_positive_root_from_guess():
    root = find_positive_root(lambda x: x * x - 0.25, guess=0.5)
    assert root == pytest.approx(0.5, abs=1e-6)


def test_find_positive_root_falls_back_to_guess():
    # No sign change anywhere positive; fallback returns the guess.
    root = find_positive_root(lambda x: x * x + 1.0, guess=0.3)
    assert root == pytest.approx(0.3)


def test_solve_with_expanding_upper_extends_bracket():
    # Root at x = 5, but the initial upper bound is too small.
    root = solve_with_expanding_upper(lambda x: x - 5.0, lower=0.0, upper=1.0)
    assert root == pytest.approx(5.0)
