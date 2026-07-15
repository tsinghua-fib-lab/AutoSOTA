import math
import numpy as np

try:
    import cma  # type: ignore
except Exception:  # pragma: no cover
    cma = None

from utils.cma_lock import cma_locked


class SubspaceCMAES:
    """Random-orthonormal-subspace CMA-ES baseline for low effective dimension problems."""

    def __init__(self, problem, max_evals, *, seed=0, subspace_dim=10):
        self.problem = problem
        self.max_evals = int(max_evals)
        self.rng = np.random.RandomState(int(seed))

        self.dim = int(problem.dimension)
        self.k = int(max(1, min(int(subspace_dim), self.dim)))

        self.lower = np.asarray(problem.lower_bounds, dtype=float)
        self.upper = np.asarray(problem.upper_bounds, dtype=float)
        self.span_min = float(np.min(self.upper - self.lower))

        self.x0 = np.clip(np.asarray(problem.initial_solution, dtype=float), self.lower, self.upper)

        g = self.rng.randn(self.dim, self.k)
        q, _r = np.linalg.qr(g)
        self.basis = q[:, : self.k]  # (dim,k)

    def _map(self, u):
        x = self.x0 + self.basis @ np.asarray(u, dtype=float)
        return np.clip(x, self.lower, self.upper)

    def run(self):
        if cma is None or self.max_evals <= 0:
            return

        # Thread-safety: prevent pycma global-RNG cross-talk in threaded sweeps.
        with cma_locked(seed=42):
            u0 = np.zeros(self.k, dtype=float)
            sigma0 = 0.3 * self.span_min

            opts = {
                "maxfevals": int(self.max_evals),
                "seed": 42,
                "verbose": -9,
                "verb_log": 0,
                "verb_time": 0,
                "popsize": max(4, 4 + int(3 * math.log(self.k))),
                "tolfun": 0.0,
                "tolfunhist": 0.0,
                "tolx": 0.0,
                "tolstagnation": int(1e9),
                "tolxstagnation": False,
                "tolflatfitness": int(1e9),
            }

            es = cma.CMAEvolutionStrategy(u0, sigma0, opts)

            while self.problem.evaluations < self.max_evals and not self.problem.final_target_hit:
                solutions = es.ask()
                remaining = int(self.max_evals - self.problem.evaluations)
                if remaining <= 0:
                    break
                if remaining < es.popsize:
                    for u in solutions[:remaining]:
                        if self.problem.final_target_hit:
                            break
                        self.problem(self._map(u))
                    break

                values = []
                for u in solutions:
                    if self.problem.final_target_hit:
                        break
                    values.append(self.problem(self._map(u)))
                if len(values) < es.popsize:
                    break
                es.tell(solutions[: len(values)], values)


def my_optimizer_rs10(problem, max_evals):
    """Entry point: k=min(10,d)."""
    k = min(10, int(problem.dimension))
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 99991
    ) & 0xFFFFFFFF
    SubspaceCMAES(problem, max_evals, seed=seed, subspace_dim=k).run()


def my_optimizer_rs5(problem, max_evals):
    """Entry point: k=min(5,d)."""
    k = min(5, int(problem.dimension))
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 55555
    ) & 0xFFFFFFFF
    SubspaceCMAES(problem, max_evals, seed=seed, subspace_dim=k).run()
