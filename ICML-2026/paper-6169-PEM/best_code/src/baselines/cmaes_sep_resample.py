import math
import numpy as np

try:
    import cma  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cma = None

from utils.cma_lock import cma_locked


def _cmaes_sep_resample(problem, max_evals: int, *, reps: int) -> None:
    """
    Diagonal CMA-ES with fixed-k resampling per candidate (mean aggregation).

    This is a "strawman-but-strong" baseline for noisy optimization under a fixed
    evaluation budget: reduce misranking by spending more evaluations.
    """

    if cma is None:
        return

    reps = int(max(1, reps))
    max_evals = int(max_evals)

    dim = int(problem.dimension)
    lower = np.asarray(problem.lower_bounds, dtype=float)
    upper = np.asarray(problem.upper_bounds, dtype=float)

    x0 = np.clip(problem.initial_solution, lower, upper)
    sigma0 = 0.3 * float(np.min(upper - lower))

    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 64000
        + 97 * int(reps)
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

        while int(getattr(problem, "evaluations", 0)) < int(max_evals) and not bool(getattr(problem, "final_target_hit", False)):
            solutions = es.ask()

            remaining = int(max_evals) - int(getattr(problem, "evaluations", 0))
            if remaining <= 0:
                break

            popsize = int(es.popsize)
            if remaining < popsize:
                # Not enough budget to evaluate one full generation even once; follow the same
                # "spend remaining evals then stop" convention as other baselines.
                for x in solutions[:remaining]:
                    if bool(getattr(problem, "final_target_hit", False)):
                        break
                    problem(np.clip(x, lower, upper))
                break

            # Evaluate each candidate at least once; then spend any remaining evaluations on
            # additional resamples up to `reps` per candidate (last generation may use <reps).
            sums = np.zeros(popsize, dtype=float)
            counts = np.zeros(popsize, dtype=int)

            # Round 1: one eval per candidate.
            for i, x in enumerate(solutions):
                if bool(getattr(problem, "final_target_hit", False)) or int(getattr(problem, "evaluations", 0)) >= int(max_evals):
                    break
                xc = np.clip(x, lower, upper)
                sums[i] += float(problem(xc))
                counts[i] += 1

            if int(np.min(counts)) < 1:
                break

            # Additional rounds: distribute remaining evaluations uniformly across candidates.
            # This uses the entire budget while preserving the spirit of fixed-k resampling.
            for _round in range(2, int(reps) + 1):
                remaining = int(max_evals) - int(getattr(problem, "evaluations", 0))
                if remaining <= 0 or bool(getattr(problem, "final_target_hit", False)):
                    break
                n = int(min(popsize, remaining))
                for i in range(n):
                    if bool(getattr(problem, "final_target_hit", False)) or int(getattr(problem, "evaluations", 0)) >= int(max_evals):
                        break
                    xc = np.clip(solutions[i], lower, upper)
                    sums[i] += float(problem(xc))
                    counts[i] += 1

            values = (sums / np.maximum(1, counts)).tolist()
            es.tell(solutions, values)


def my_optimizer_resample2(problem, max_evals):
    """Entry point: sep-CMA-ES with 2x resampling (mean)."""
    _cmaes_sep_resample(problem, max_evals, reps=2)


def my_optimizer_resample3(problem, max_evals):
    """Entry point: sep-CMA-ES with 3x resampling (mean)."""
    _cmaes_sep_resample(problem, max_evals, reps=3)


def my_optimizer_resample5(problem, max_evals):
    """Entry point: sep-CMA-ES with 5x resampling (mean)."""
    _cmaes_sep_resample(problem, max_evals, reps=5)


def my_optimizer_resample10(problem, max_evals):
    """Entry point: sep-CMA-ES with 10x resampling (mean)."""
    _cmaes_sep_resample(problem, max_evals, reps=10)
