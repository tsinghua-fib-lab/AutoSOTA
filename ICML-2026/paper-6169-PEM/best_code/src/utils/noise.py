from __future__ import annotations

import math

import numpy as np


class NoisyProblem:
    """
    Wrap a deterministic COCO problem and return noisy evaluations to the optimizer,
    while leaving the underlying COCO bookkeeping as the true function values.
    """

    def __init__(self, problem, *, noise_model: str, noise_sigma: float, seed: int):
        self._problem = problem
        self.noise_model = str(noise_model)
        self.noise_sigma = float(noise_sigma)
        self._rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)

        x0 = getattr(problem, "initial_solution", None)
        if x0 is None:
            dim = int(getattr(problem, "dimension", 0))
            x0 = np.zeros(dim, dtype=float)
        self._x0 = np.asarray(x0, dtype=float)

        lower = getattr(problem, "lower_bounds", None)
        upper = getattr(problem, "upper_bounds", None)
        if lower is None or upper is None:
            lower = -5.0 * np.ones_like(self._x0)
            upper = 5.0 * np.ones_like(self._x0)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        self._span = np.maximum(upper - lower, 1e-12)

    def _radial_factor(self, x: np.ndarray) -> float:
        """RMS normalized distance from the initial solution (dimensionless)."""
        dx = (np.asarray(x, dtype=float) - self._x0) / self._span
        if int(dx.size) <= 0:
            return 0.0
        return float(np.linalg.norm(dx) / math.sqrt(float(dx.size)))

    def __call__(self, x):
        true_val = float(self._problem(x))
        s = float(self.noise_sigma)
        if s <= 0.0:
            return true_val

        model = str(self.noise_model)
        if model == "additive":
            return true_val + s * float(self._rng.randn())
        if model == "additive_rel":
            return true_val + s * (1.0 + abs(true_val)) * float(self._rng.randn())
        if model == "radial_additive":
            s_eff = s * self._radial_factor(np.asarray(x, dtype=float))
            return true_val + s_eff * float(self._rng.randn())
        if model == "radial_additive_rel":
            s_eff = s * self._radial_factor(np.asarray(x, dtype=float))
            return true_val + s_eff * (1.0 + abs(true_val)) * float(self._rng.randn())
        if model == "additive_rel_t":
            df = 3.0
            z = float(self._rng.standard_t(df))
            z = z / math.sqrt(df / max(1e-12, (df - 2.0)))
            return true_val + s * (1.0 + abs(true_val)) * z
        if model == "lognormal_mult":
            z = float(self._rng.randn())
            factor = float(np.exp(s * z - 0.5 * (s * s)))
            return true_val * factor
        if model == "radial_lognormal_mult":
            s_eff = s * self._radial_factor(np.asarray(x, dtype=float))
            z = float(self._rng.randn())
            factor = float(np.exp(s_eff * z - 0.5 * (s_eff * s_eff)))
            return true_val * factor
        raise ValueError(f"Unknown noise_model: {model}")

    def __getattr__(self, name: str):
        return getattr(self._problem, name)

