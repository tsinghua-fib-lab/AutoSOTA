import math
import numpy as np


class BudgetExceeded(Exception):
    """Raised when the BBOB evaluation budget is exhausted."""


class ExpScoreTransform:
    """Monotone transform that maps objectives to positive scores."""

    def __init__(self, alpha, max_exp=60.0):
        self.alpha = float(alpha)
        self.max_exp = float(max_exp)

    def __call__(self, value):
        exp_arg = -self.alpha * float(value)
        if exp_arg > self.max_exp:
            exp_arg = self.max_exp
        elif exp_arg < -self.max_exp:
            exp_arg = -self.max_exp
        return math.exp(exp_arg)


def make_exp_score_transform(initial_value, power=1.0, max_exp=60.0):
    """Create a stable exponential score transform based on the initial value."""
    scale = max(1.0, abs(float(initial_value)))
    alpha = float(power) / scale
    return ExpScoreTransform(alpha, max_exp=max_exp)


class BBOBEvaluator:
    """Wraps a COCO/BBOB problem and returns transformed scores."""

    def __init__(self, problem, max_evals, score_transform):
        self.problem = problem
        self.max_evals = int(max_evals)
        self.score_transform = score_transform
        self.lower = np.asarray(problem.lower_bounds, dtype=float)
        self.upper = np.asarray(problem.upper_bounds, dtype=float)
        self.dim = int(problem.dimension)
        self.best_f = math.inf
        self.best_x = None

    def _check_budget(self):
        if self.problem.evaluations >= self.max_evals or self.problem.final_target_hit:
            raise BudgetExceeded()

    def evaluate(self, x):
        self._check_budget()
        x_clip = np.clip(x, self.lower, self.upper)
        value = float(self.problem(x_clip))
        if value < self.best_f:
            self.best_f = value
            self.best_x = x_clip.copy()
        return self.score_transform(value)

    def remaining(self):
        return max(0, self.max_evals - int(self.problem.evaluations))
