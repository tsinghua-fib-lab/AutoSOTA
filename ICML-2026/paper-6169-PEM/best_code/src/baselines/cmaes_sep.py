import math
import numpy as np

try:
    import cma  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cma = None

from utils.cma_lock import cma_locked


def my_optimizer(problem, max_evals):
    """COCO/BBOB entry point: diagonal CMA-ES (sep-CMA-ES style)."""
    if cma is None:
        return

    # Thread-safety: prevent pycma global-RNG cross-talk in threaded sweeps.
    dim = int(problem.dimension)
    lower = np.asarray(problem.lower_bounds, dtype=float)
    upper = np.asarray(problem.upper_bounds, dtype=float)

    x0 = np.clip(problem.initial_solution, lower, upper)
    sigma0 = 0.3 * float(np.min(upper - lower))

    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 42424
    ) & 0xFFFFFFFF

    with cma_locked(seed=int(seed)):
        opts = {
            "bounds": [lower, upper],
            "maxfevals": int(max_evals),
            "seed": int(seed),
            "verbose": -9,
            "verb_log": 0,
            "verb_time": 0,
            "CMA_diagonal": True,
            "popsize": max(4, 4 + int(3 * math.log(dim))),
            "tolfun": 0.0,
            "tolfunhist": 0.0,
            "tolx": 0.0,
            "tolstagnation": int(1e9),
            "tolxstagnation": False,
            "tolflatfitness": int(1e9),
        }

        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

        while problem.evaluations < max_evals and not problem.final_target_hit:
            solutions = es.ask()
            remaining = int(max_evals - problem.evaluations)
            if remaining <= 0:
                break
            if remaining < es.popsize:
                for x in solutions[:remaining]:
                    if problem.final_target_hit:
                        break
                    problem(np.clip(x, lower, upper))
                break

            values = []
            for x in solutions:
                if problem.final_target_hit:
                    break
                values.append(problem(np.clip(x, lower, upper)))
            if len(values) < es.popsize:
                break
            es.tell(solutions[: len(values)], values)
