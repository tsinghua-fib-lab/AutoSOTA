"""FFD-based Shapley value approximators.

Python implementation of the RD (Algorithm 1) and RD-Corrected (Algorithm 2)
estimators from:

  Zhou, Z., Mee, R., Hamers, H., & Zheng, W. (2025). Fast approximation of
  Shapley values through fractional factorial designs. Journal of the American
  Statistical Association, 1-20. https://doi.org/10.1080/01621459.2025.2529027
  Code: https://github.com/ZhengZhou0613/FFD-based-SV-approximation

Value functions are pseudoboolean: f(x), x in {0,1}^d -> float.

These methods have a *fixed* evaluation budget determined by drec(n), not a
flexible budget parameter.  Each instance runs its full FFD procedure exactly
once (on the first call to approximate()) and caches the result.  Every
subsequent call — regardless of the requested budget — returns the cached
result immediately.  On a budget-vs-error plot the FFD methods therefore
appear as a single horizontal dot at their natural evaluation count.

Functions / classes
-------------------
shapley_rd(f, d)
    Algorithm 1: RD estimator.
shapley_rd_corrected(f, d, t1, t2)
    Algorithm 2: RD with bias detection and correction.
FFD_RD
    shapiq Approximator wrapper around shapley_rd.
FFD_RD_Corrected
    shapiq Approximator wrapper around shapley_rd_corrected.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import scipy.interpolate as interp
import scipy.stats as stats

from shapiq.approximator.base import Approximator
from shapiq.interaction_values import InteractionValues


# ---------------------------------------------------------------------------
# Design generation
# ---------------------------------------------------------------------------

def _recur_design(p: int) -> np.ndarray:
    """Recursive Hadamard design with p = 2^h + 1 factors.

    Returns a matrix of shape (4*(p-1), p) with +-1 entries.
    Column 0 is the focal column used by the RD estimator.
    The design satisfies the pairing property: for every row v, -v is present.
    """
    h = int(round(np.log2(p - 1)))
    H = np.array([[1.0]])
    for _ in range(h):
        H = np.block([[H, H], [H, -H]])
    col0_top = np.ones((2 * H.shape[0], 1))
    col0_bot = -np.ones((2 * H.shape[0], 1))
    rest_top = np.vstack([H, -H])
    rest_bot = np.vstack([-H, H])
    return np.vstack([
        np.hstack([col0_top, rest_top]),
        np.hstack([col0_bot, rest_bot]),
    ])


def _drec(d: int) -> int:
    """Smallest p = 2^i + 1 >= d, for i in 2..15."""
    for i in range(2, 16):
        if d <= 2**i + 1:
            return 2**i + 1
    raise ValueError(f"d={d} exceeds maximum supported size (2^15 + 1 = 32769)")


def _y_combine(d: int) -> np.ndarray:
    """Estimator weights for RD (Theorem 3). Returns array of length 4*(d-1)."""
    n = 4 * (d - 1)
    l = np.full(d - 1, 2.0 / (3 * n))
    l[0] = (d + 1) / (3 * n)
    l = np.concatenate([l, l, -l, -l])
    return 2.0 * l


def _rlhd(d: int) -> np.ndarray:
    """Random Latin Hypercube Design: integer matrix (d, d), rows are 1-indexed permutations."""
    base = np.arange(1, d + 1)
    rows = [np.roll(base, -i) for i in range(d)]
    design = np.array(rows)
    design = design[:, np.random.permutation(d)]
    return design[np.random.permutation(d)]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _eval_pm1_row(f, row: np.ndarray) -> float:
    """Evaluate f on the {0,1} coalition encoded by a +-1 row (+ means in)."""
    return f((row > 0).astype(float))


def _eval_prefix(f, prefix, d: int) -> float:
    """Evaluate f on the coalition given by a prefix of a 1-indexed permutation."""
    x = np.zeros(d)
    x[np.asarray(prefix, dtype=int) - 1] = 1.0
    return f(x)


# ---------------------------------------------------------------------------
# Algorithm 1: RD estimator
# ---------------------------------------------------------------------------

def shapley_rd(f, d: int) -> tuple[np.ndarray, int]:
    """Estimate Shapley values via the Recursive Design (RD) method.

    Uses a recursive Hadamard fractional factorial design with implicit paired
    sampling (for every coalition S in the design, its complement N\\S is also
    present).  Exact for pseudoboolean functions whose odd-degree Fourier
    components have degree <= 3.

    Args:
        f: Callable f(x) where x is a {0,1} numpy array of length d.
        d: Number of players.

    Returns:
        phi:    Shapley value estimates, shape (d,).
        n_eval: Number of unique game evaluations used.
    """
    drecur = _drec(d)
    rows_pp = 4 * (drecur - 1)

    perm_rest = np.random.permutation(np.arange(1, drecur))
    col_perm = np.concatenate([[0], perm_rest])
    design0 = _recur_design(drecur)[:, col_perm][:, :d]  # (rows_pp, d)

    design = np.tile(design0, (d, 1))  # (d * rows_pp, d)
    for j in range(1, d):
        s, e = j * rows_pp, (j + 1) * rows_pp
        a = design[s:e, 0].copy()
        design[s:e, :j] = design[s:e, 1:j + 1]
        design[s:e, j] = a

    # Deduplicate rows
    unique_map: dict[bytes, int] = {}
    reprowtag = np.empty(len(design), dtype=int)
    unique_rows = []
    for idx in range(len(design)):
        key = design[idx].tobytes()
        if key not in unique_map:
            unique_map[key] = len(unique_rows)
            unique_rows.append(design[idx])
        reprowtag[idx] = unique_map[key]

    unique_design = np.array(unique_rows)
    f_vals = np.array([_eval_pm1_row(f, row) for row in unique_design])

    weights = _y_combine(drecur)
    phi = np.zeros(d)
    for i in range(d):
        indices = reprowtag[i * rows_pp:(i + 1) * rows_pp]
        phi[i] = np.dot(weights, f_vals[indices])

    return phi, len(unique_rows)


# ---------------------------------------------------------------------------
# Algorithm 2: RD with bias correction
# ---------------------------------------------------------------------------

def _ls_on_design(f, d: int, key_players, design: np.ndarray) -> tuple[dict, int]:
    """Estimate Shapley values for key_players using a pre-generated LHD design."""
    key_set = set(int(k) for k in key_players)
    accum = np.zeros(d)
    n_eval = 0

    for row in design:
        prev = 0.0
        prev_was_key = False
        for j in range(d):
            p0 = int(row[j]) - 1
            if p0 in key_set:
                curr = _eval_prefix(f, row[:j + 1], d)
                n_eval += 1
                if j == 0 or prev_was_key:
                    marginal = curr - prev
                else:
                    prev_val = _eval_prefix(f, row[:j], d)
                    n_eval += 1
                    marginal = curr - prev_val
                accum[p0] += marginal
                prev = curr
                prev_was_key = True
            else:
                prev_was_key = False

    n_perms = len(design)
    return {k: accum[k] / n_perms for k in key_players}, n_eval


def _linreg_test(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """OLS y = alpha + beta*x. Returns (p_intercept, p_slope, r_squared)."""
    n = len(x)
    if n < 3:
        return 1.0, 1.0, 0.0
    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (slope * x + intercept)
    sse = float(np.sum(residuals ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r_sq = 1.0 - sse / sst if sst > 0 else 0.0

    se_sq = sse / (n - 2) if n > 2 else np.inf
    x_mean = np.mean(x)
    sxx = float(np.sum((x - x_mean) ** 2))
    if sxx <= 0:
        return 1.0, 1.0, r_sq

    t_slope = slope / np.sqrt(se_sq / sxx)
    t_intercept = intercept / np.sqrt(se_sq * (1.0 / n + x_mean ** 2 / sxx))
    p_slope = 2.0 * float(stats.t.sf(abs(t_slope), df=n - 2))
    p_intercept = 2.0 * float(stats.t.sf(abs(t_intercept), df=n - 2))
    return p_intercept, p_slope, r_sq


def shapley_rd_corrected(
    f, d: int, t1: int = 6, t2: int = 21
) -> tuple[np.ndarray, int, bool]:
    """RD with bias detection and correction.

    Runs Algorithm 1 (RD) then applies a two-path correction:

    - Shrinkage path (linearity test passes): rescales phi_rd so that
      the efficiency axiom sum(phi) = f(grand) - f(empty) holds exactly.
    - Spline path (linearity test fails): fits a smoothing spline from
      phi_rd[key_players] to phi_ls[key_players] and applies it to all d players.

    Args:
        f:  Game callable f(x), x in {0,1}^d.
        d:  Number of players.
        t1: Key players for the initial linearity test (default 6).
        t2: Key players for the spline correction if test fails (default 21).

    Returns:
        phi_star: Bias-corrected Shapley estimates, shape (d,).
        n_eval:   Total game evaluations used.
        passed:   True if shrinkage path was taken, False if spline path.
    """
    phi_rd, n_total = shapley_rd(f, d)

    lhd_design = np.vstack([_rlhd(d) for _ in range(6)])  # (6*d, d)

    order = np.argsort(phi_rd)
    key1 = order[np.round(np.linspace(0, d - 1, t1)).astype(int)]
    phi_ls1, n_ls1 = _ls_on_design(f, d, key1, lhd_design)
    n_total += n_ls1

    phi_rd_k1 = phi_rd[key1]
    phi_ls_k1 = np.array([phi_ls1[k] for k in key1])

    p_int, p_slope, r_sq = _linreg_test(phi_ls_k1, phi_rd_k1)
    passed = (p_int > 0.05) and (p_slope < 0.05) and (r_sq > 0.75)

    if passed:
        target = f(np.ones(d)) - f(np.zeros(d))
        n_total += 2
        s = float(np.sum(phi_rd))
        phi_star = phi_rd if abs(s) < 1e-12 else phi_rd - (s - target) / s * phi_rd
        return phi_star, n_total, True

    key2 = order[np.round(np.linspace(0, d - 1, t2)).astype(int)]
    add_key = np.setdiff1d(key2, key1)
    phi_ls2, n_ls2 = _ls_on_design(f, d, add_key, lhd_design)
    n_total += n_ls2

    phi_ls_k2 = np.array([phi_ls1.get(k, phi_ls2.get(k, float("nan"))) for k in key2])
    phi_rd_k2 = phi_rd[key2]

    sort_idx = np.argsort(phi_rd_k2)
    x_fit, y_fit = phi_rd_k2[sort_idx], phi_ls_k2[sort_idx]

    valid = np.isfinite(x_fit) & np.isfinite(y_fit)
    x_fit, y_fit = x_fit[valid], y_fit[valid]

    x_unique, inv = np.unique(x_fit, return_inverse=True)
    y_unique = np.array([y_fit[inv == k].mean() for k in range(len(x_unique))])
    x_fit, y_fit = x_unique, y_unique

    try:
        if len(x_fit) >= 4:
            spline = interp.make_smoothing_spline(x_fit, y_fit)
            phi_star = spline(np.clip(phi_rd, x_fit[0], x_fit[-1]))
        else:
            raise ValueError("too few unique points for spline")
    except Exception:
        phi_star = np.interp(phi_rd, x_fit, y_fit)

    return phi_star, n_total, False


# ---------------------------------------------------------------------------
# shapiq Approximator wrappers
# ---------------------------------------------------------------------------

def _game_to_f(game: Callable[[np.ndarray], np.ndarray], n: int):
    """Wrap a shapiq-style game as the scalar f(x) expected by FFD."""
    def f(x: np.ndarray) -> float:
        return float(game(x.reshape(1, n))[0])
    return f


def _build_interaction_values(
    phi: np.ndarray,
    n: int,
    baseline_value: float,
    budget: int,
    estimated: bool,
) -> InteractionValues:
    """Pack a Shapley value array into an InteractionValues object."""
    values = np.empty(n + 1)
    interaction_lookup: dict[tuple[int, ...], int] = {}
    values[0] = baseline_value
    interaction_lookup[()] = 0
    for i in range(n):
        values[i + 1] = phi[i]
        interaction_lookup[(i,)] = i + 1
    return InteractionValues(
        values=values,
        index="SV",
        interaction_lookup=interaction_lookup,
        baseline_value=baseline_value,
        min_order=0,
        max_order=1,
        n_players=n,
        estimated=estimated,
        estimation_budget=budget,
    )


class FFD_RD(Approximator):
    """Shapley estimator via Recursive Fractional Factorial Design (Algorithm 1).

    Runs once on the first call to approximate() and caches the result.
    """

    def __init__(self, n: int, random_state: int | None = None):
        super().__init__(
            n, min_order=0, max_order=1, index="SV",
            top_order=False, random_state=random_state,
        )
        self.random_state = random_state
        self.runtime_last_approximate_run: dict[str, float] = {}
        self._cached_result: InteractionValues | None = None

    def approximate(
        self, budget: int, game: Callable[[np.ndarray], np.ndarray],
    ) -> InteractionValues:
        if self._cached_result is not None:
            self.runtime_last_approximate_run = {"evaluations": 0.0, "total": 0.0}
            return self._cached_result

        start = time.time()
        if self.random_state is not None:
            np.random.seed(self.random_state)

        f = _game_to_f(game, self.n)
        baseline_value = float(game(np.zeros((1, self.n)))[0])

        eval_start = time.time()
        phi, n_evals = shapley_rd(f, self.n)
        eval_end = time.time()

        self.runtime_last_approximate_run = {
            "evaluations": eval_end - eval_start,
            "total": time.time() - start,
        }
        self._cached_result = _build_interaction_values(
            phi, self.n, baseline_value, budget=n_evals, estimated=True,
        )
        return self._cached_result


class FFD_RD_Corrected(Approximator):
    """Shapley estimator via RD + bias correction (Algorithm 2).

    Runs once on the first call to approximate() and caches the result.
    """

    def __init__(self, n: int, random_state: int | None = None):
        super().__init__(
            n, min_order=0, max_order=1, index="SV",
            top_order=False, random_state=random_state,
        )
        self.random_state = random_state
        self.runtime_last_approximate_run: dict[str, float] = {}
        self._cached_result: InteractionValues | None = None

    def approximate(
        self, budget: int, game: Callable[[np.ndarray], np.ndarray],
    ) -> InteractionValues:
        if self._cached_result is not None:
            self.runtime_last_approximate_run = {"evaluations": 0.0, "total": 0.0}
            return self._cached_result

        start = time.time()
        if self.random_state is not None:
            np.random.seed(self.random_state)

        f = _game_to_f(game, self.n)
        baseline_value = float(game(np.zeros((1, self.n)))[0])

        eval_start = time.time()
        phi, n_evals, _ = shapley_rd_corrected(f, self.n)
        eval_end = time.time()

        self.runtime_last_approximate_run = {
            "evaluations": eval_end - eval_start,
            "total": time.time() - start,
        }
        self._cached_result = _build_interaction_values(
            phi, self.n, baseline_value, budget=n_evals, estimated=True,
        )
        return self._cached_result
