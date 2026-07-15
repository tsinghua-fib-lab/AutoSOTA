"""LRA-CMA-ES baseline (Nomura, Akimoto & Ono, ACM TELO 2025).

Uses the ``cmaes`` package (``pip install cmaes``) with ``lr_adapt=True``.
The ask-tell API differs from pycma: ask() returns one solution at a time,
tell() takes a list of (solution, value) tuples.
"""

import math

import numpy as np

try:
    from cmaes import CMA as _CMA_LRA  # Nomura's cmaes package
except Exception:  # pragma: no cover - optional dependency
    _CMA_LRA = None


def my_optimizer(problem, max_evals):
    """COCO/BBOB entry point: LRA-CMA-ES (learning-rate adaptation)."""
    if _CMA_LRA is None:
        return

    dim = int(problem.dimension)
    lower = np.asarray(problem.lower_bounds, dtype=float)
    upper = np.asarray(problem.upper_bounds, dtype=float)

    x0 = np.clip(problem.initial_solution, lower, upper).astype(float)
    sigma0 = 0.3 * float(np.min(upper - lower))

    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 88888
    ) & 0xFFFFFFFF

    popsize = max(4, 4 + int(3 * math.log(dim)))

    # cmaes package expects bounds as (dim, 2) array: [[lo_0, hi_0], ...]
    bounds = np.column_stack([lower, upper])

    optimizer = _CMA_LRA(
        mean=x0.copy(),
        sigma=sigma0,
        bounds=bounds,
        seed=int(seed),
        population_size=popsize,
        lr_adapt=True,
    )

    while problem.evaluations < max_evals and not problem.final_target_hit:
        remaining = int(max_evals - problem.evaluations)
        if remaining <= 0:
            break

        # Ask: collect full population
        candidates = [optimizer.ask() for _ in range(optimizer.population_size)]

        if remaining < optimizer.population_size:
            for x in candidates[:remaining]:
                if problem.final_target_hit:
                    break
                problem(np.clip(x, lower, upper))
            break

        # Evaluate all candidates
        values = []
        for x in candidates:
            if problem.final_target_hit:
                break
            values.append(problem(np.clip(x, lower, upper)))

        if len(values) < optimizer.population_size:
            break

        # Tell: cmaes package takes list of (solution, value) tuples
        optimizer.tell([(candidates[i], values[i]) for i in range(len(values))])
