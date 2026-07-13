"""Numerical helpers that wrap standard SciPy solvers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import root_scalar


def solve_bracketed(
    fn: Callable[[float], float],
    bracket: tuple[float, float],
    *,
    xtol: float = 1e-14,
    rtol: float = 1e-12,
    maxiter: int = 500,
) -> float:
    """Solve a scalar root with SciPy's brentq implementation."""

    lo, hi = bracket
    f_lo, f_hi = fn(lo), fn(hi)
    if f_lo == 0:
        return float(lo)
    if f_hi == 0:
        return float(hi)
    result = root_scalar(fn, bracket=(lo, hi), method="brentq", xtol=xtol, rtol=rtol, maxiter=maxiter)
    if not result.converged:
        raise RuntimeError(f"root_scalar did not converge on bracket {bracket}")
    return float(result.root)


def find_positive_root(
    fn: Callable[[float], float],
    guess: float,
    *,
    lower: float = 1e-12,
    upper: float = 1.9,
    initial_factor: float = 100.0,
    expansion_factor: float = 2.0,
    max_expansions: int = 80,
    grid_lower: float = 1e-12,
    grid_upper: float = 1.0,
    grid_points: int = 800,
    fallback_to_guess: bool = True,
) -> float:
    """Find a positive sign-changing bracket, then solve it with SciPy."""

    lo = max(lower, guess / initial_factor)
    hi = min(upper, guess * initial_factor)
    f_lo, f_hi = fn(lo), fn(hi)
    if f_lo == 0:
        return float(lo)
    if f_hi == 0:
        return float(hi)

    for _ in range(max_expansions):
        if f_lo * f_hi < 0:
            break
        lo = max(lower, lo / expansion_factor)
        hi = min(upper, hi * expansion_factor)
        f_lo, f_hi = fn(lo), fn(hi)
        if f_lo == 0:
            return float(lo)
        if f_hi == 0:
            return float(hi)
        if lo <= lower and hi >= upper:
            break

    if f_lo * f_hi > 0:
        grid = np.exp(np.linspace(np.log(grid_lower), np.log(grid_upper), grid_points))
        values = np.array([fn(point) for point in grid])
        idx = np.where(values[:-1] * values[1:] < 0)[0]
        if len(idx) == 0:
            if fallback_to_guess:
                return float(guess)
            raise ValueError("Could not find a positive sign-changing bracket")
        lo, hi = grid[idx[0]], grid[idx[0] + 1]

    return solve_bracketed(fn, (lo, hi))


def solve_with_expanding_upper(
    fn: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    expansion_factor: float = 2.0,
    max_expansions: int = 100,
) -> float:
    """Expand the upper endpoint until SciPy can solve a bracketed root."""

    lo, hi = lower, upper
    f_lo, f_hi = fn(lo), fn(hi)
    if f_lo == 0:
        return float(lo)
    if f_hi == 0:
        return float(hi)
    for _ in range(max_expansions):
        if f_lo * f_hi < 0:
            return solve_bracketed(fn, (lo, hi))
        hi *= expansion_factor
        f_hi = fn(hi)
        if f_hi == 0:
            return float(hi)
    raise ValueError("Could not find a sign-changing bracket")
