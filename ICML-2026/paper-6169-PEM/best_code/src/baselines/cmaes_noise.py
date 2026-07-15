import math
import numpy as np

try:
    import cma  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cma = None

from utils.cma_lock import cma_locked


def _make_noise_handler(*, dim: int, maxevals: int, epsilon: float):
    if cma is None:
        return None
    # pycma NoiseHandler (Hansen et al. 2009-based).
    # Use explicit settings so that "UH-CMA-ES" is not accidentally a no-resampling config.
    return cma.optimization_tools.NoiseHandler(
        dim,
        maxevals=[1, 1, int(maxevals)],
        aggregate=np.median,
        reevals=None,
        epsilon=float(epsilon),
        parallel=False,
    )


def _run_cmaes_noisehandler(problem, max_evals: int, *, maxevals: int, epsilon: float) -> None:
    if cma is None:
        return

    dim = int(problem.dimension)
    lower = np.asarray(problem.lower_bounds, dtype=float)
    upper = np.asarray(problem.upper_bounds, dtype=float)

    x0 = np.clip(problem.initial_solution, lower, upper)
    sigma0 = 0.3 * float(np.min(upper - lower))

    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 4242
    ) & 0xFFFFFFFF

    with cma_locked(seed=int(seed)):
        opts = {
            "bounds": [lower, upper],
            "maxfevals": int(max_evals),
            "seed": int(seed),
            "verbose": -9,
            "verb_log": 0,
            "verb_time": 0,
            "popsize": max(4, 4 + int(3 * math.log(dim))),
            "tolfun": 0.0,
            "tolfunhist": 0.0,
            "tolx": 0.0,
            "tolstagnation": int(1e9),
            "tolxstagnation": False,
            "tolflatfitness": int(1e9),
            "tolconditioncov": 1e30,
            "tolupsigma": 1e30,
        }

        def objective(x):
            if problem.final_target_hit or problem.evaluations >= max_evals:
                return float(problem.best_observed_fvalue1)
            x_clip = np.clip(x, lower, upper)
            return float(problem(x_clip))

        noisehandler = _make_noise_handler(dim=dim, maxevals=int(maxevals), epsilon=float(epsilon))
        if noisehandler is None:
            return

        try:
            cma.fmin(objective, x0, sigma0, opts, noise_handler=noisehandler)
        except Exception:
            return


def my_optimizer(problem, max_evals):
    """
    COCO/BBOB entry point: CMA-ES with pycma NoiseHandler (Hansen et al. 2009-based).

    Intended for 'bbob-noisy' where function indices are 101–130.
    """
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
        + 4242
    ) & 0xFFFFFFFF

    # Backward-compatible baseline: keep the original behavior.
    # NOTE: this uses the default NoiseHandler settings, which include maxevals=[1,1,1].
    _run_cmaes_noisehandler(problem, int(max_evals), maxevals=1, epsilon=1e-7)


def my_optimizer_uh_maxevals10(problem, max_evals):
    """UH-CMA-ES variant: allow up to 10 evaluations per fitness, reevaluate exact points (epsilon=0)."""
    _run_cmaes_noisehandler(problem, int(max_evals), maxevals=10, epsilon=0.0)


def my_optimizer_uh_maxevals30(problem, max_evals):
    """UH-CMA-ES variant: allow up to 30 evaluations per fitness, reevaluate exact points (epsilon=0)."""
    _run_cmaes_noisehandler(problem, int(max_evals), maxevals=30, epsilon=0.0)
