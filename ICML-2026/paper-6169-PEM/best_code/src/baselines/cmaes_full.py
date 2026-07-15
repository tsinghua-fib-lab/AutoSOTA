import math
import numpy as np

try:
    import cma  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cma = None

from utils.cma_lock import cma_locked


def _run_cma_package(problem, max_evals, seed):
    # Thread-safety: prevent pycma global-RNG cross-talk in threaded sweeps.
    with cma_locked(seed=int(seed)):
        dim = int(problem.dimension)
        lower = np.asarray(problem.lower_bounds, dtype=float)
        upper = np.asarray(problem.upper_bounds, dtype=float)

        x0 = np.clip(problem.initial_solution, lower, upper)
        sigma0 = 0.3 * float(np.min(upper - lower))

        opts = {
            "bounds": [lower, upper],
            "maxfevals": int(max_evals),
            "seed": int(seed),
            "verbose": -9,
            "verb_log": 0,
            "verb_time": 0,
            "popsize": max(4, 4 + int(3 * math.log(dim))),
        }

        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

        while (
            not es.stop()
            and problem.evaluations < max_evals
            and not problem.final_target_hit
        ):
            solutions = es.ask()
            remaining = int(max_evals - problem.evaluations)
            if remaining <= 0:
                break
            if remaining < es.popsize:
                for x in solutions[:remaining]:
                    if problem.final_target_hit:
                        break
                    x_clip = np.clip(x, lower, upper)
                    problem(x_clip)
                break

            values = []
            for x in solutions:
                if problem.final_target_hit:
                    break
                x_clip = np.clip(x, lower, upper)
                values.append(problem(x_clip))
            if len(values) < es.popsize:
                break
            es.tell(solutions[: len(values)], values)


class CMAESFallback:
    """Minimal CMA-ES implementation when pycma is unavailable."""

    def __init__(self, problem, max_evals, seed=None):
        self.problem = problem
        self.max_evals = int(max_evals)
        self.rng = np.random.RandomState(seed)

        self.dim = int(problem.dimension)
        self.lower = np.asarray(problem.lower_bounds, dtype=float)
        self.upper = np.asarray(problem.upper_bounds, dtype=float)

        self.lambda_ = 4 + int(3 * math.log(self.dim))
        self.mu = self.lambda_ // 2

        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        weights = weights / np.sum(weights)
        self.weights = weights
        self.mueff = 1.0 / np.sum(weights ** 2)

        self.sigma = 0.3 * float(np.min(self.upper - self.lower))
        self.mean = np.clip(problem.initial_solution, self.lower, self.upper)

        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1 - self.c1,
            2 * (self.mueff - 2 + 1 / self.mueff) / ((self.dim + 2) ** 2 + self.mueff),
        )
        self.damps = 1 + 2 * max(0.0, math.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs

        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)

        self.diagonal_mode = self.dim > 100
        if self.diagonal_mode:
            self.C = np.ones(self.dim)
        else:
            self.C = np.eye(self.dim)
            self.B = np.eye(self.dim)
            self.D = np.ones(self.dim)
            self.invsqrtC = np.eye(self.dim)
            self.eigeneval = 0
            self.chiN = math.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))

        if self.diagonal_mode:
            self.chiN = math.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))

    def run(self):
        generation = 0
        while (
            self.problem.evaluations < self.max_evals
            and not self.problem.final_target_hit
        ):
            offspring = []
            for _ in range(self.lambda_):
                if self.problem.evaluations >= self.max_evals or self.problem.final_target_hit:
                    break
                z = self.rng.randn(self.dim)
                if self.diagonal_mode:
                    y = np.sqrt(self.C) * z
                else:
                    y = np.dot(self.B, self.D * z)
                x = self.mean + self.sigma * y
                x_eval = np.clip(x, self.lower, self.upper)
                f = float(self.problem(x_eval))
                offspring.append((y, f))

            if not offspring:
                break

            offspring.sort(key=lambda item: item[1])
            top = offspring[: self.mu]

            y_w = np.zeros(self.dim)
            for i in range(self.mu):
                y_w += self.weights[i] * top[i][0]

            self.mean = self.mean + self.sigma * y_w

            if self.diagonal_mode:
                invsqrtC_y_w = y_w / np.sqrt(self.C)
            else:
                invsqrtC_y_w = np.dot(self.invsqrtC, y_w)

            self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * invsqrtC_y_w
            norm_ps = np.linalg.norm(self.ps)
            self.sigma *= math.exp((self.cs / self.damps) * (norm_ps / self.chiN - 1))

            hsig_cond = norm_ps / math.sqrt(1 - (1 - self.cs) ** (2 * (generation + 1))) / self.chiN
            hsig = 1.0 if hsig_cond < (1.4 + 2 / (self.dim + 1)) else 0.0

            self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_w

            if self.diagonal_mode:
                rank_mu = np.zeros(self.dim)
                for i in range(self.mu):
                    rank_mu += self.weights[i] * (top[i][0] ** 2)
                self.C = (1 - self.c1 - self.cmu) * self.C + \
                    self.c1 * (self.pc ** 2 + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + \
                    self.cmu * rank_mu
                self.C = np.maximum(self.C, 1e-30)
            else:
                rank_mu = np.zeros((self.dim, self.dim))
                for i in range(self.mu):
                    yi = top[i][0]
                    rank_mu += self.weights[i] * np.outer(yi, yi)
                self.C = (1 - self.c1 - self.cmu) * self.C + \
                    self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + \
                    self.cmu * rank_mu

                if (self.problem.evaluations - self.eigeneval) > self.lambda_ / (self.c1 + self.cmu) / self.dim / 10:
                    self.eigeneval = self.problem.evaluations
                    self.C = np.triu(self.C) + np.triu(self.C, 1).T
                    evals, evecs = np.linalg.eigh(self.C)
                    self.D = np.sqrt(np.maximum(evals, 1e-30))
                    self.B = evecs
                    self.invsqrtC = np.dot(evecs * (1.0 / self.D), evecs.T)

            generation += 1


def my_optimizer(problem, max_evals):
    """BBOB entry point."""
    if cma is not None:
        seed = (
            int(getattr(problem, "id_function", 0)) * 1000003
            + int(getattr(problem, "id_instance", 0)) * 1009
            + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
            + 2424
        ) & 0xFFFFFFFF
        _run_cma_package(problem, max_evals, seed)
        return
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 2424
    ) & 0xFFFFFFFF
    fallback = CMAESFallback(problem, max_evals, seed=seed)
    fallback.run()
