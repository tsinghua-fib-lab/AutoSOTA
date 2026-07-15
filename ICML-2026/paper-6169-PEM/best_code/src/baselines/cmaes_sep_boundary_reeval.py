import math
import numpy as np

try:
    import cma  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cma = None

from utils.cma_lock import cma_locked


def _choose_boundary_band(order: np.ndarray, mu: int, *, band: int, count: int) -> np.ndarray:
    lam = int(order.size)
    if lam <= 0 or count <= 0:
        return np.array([], dtype=int)
    mu0 = int(max(1, min(int(mu), lam)))
    center = max(0, min(lam - 1, mu0 - 1))
    start = max(0, center - int(band))
    end = min(lam, center + int(band) + 1)
    band_idx = order[start:end]
    if int(band_idx.size) >= int(count):
        return np.asarray(band_idx[:count], dtype=int)
    rest = order[:count]
    idx = np.unique(np.concatenate([band_idx, rest]))
    return np.asarray(idx[:count], dtype=int)


def _rank_disagreement(f_a: np.ndarray, f_b: np.ndarray) -> float:
    lam = int(f_a.size)
    if lam <= 1:
        return 0.0
    order_a = np.argsort(f_a)
    order_b = np.argsort(f_b)
    ranks_a = np.empty(lam, dtype=int)
    ranks_b = np.empty(lam, dtype=int)
    ranks_a[order_a] = np.arange(lam)
    ranks_b[order_b] = np.arange(lam)
    return float(np.mean(np.abs(ranks_a - ranks_b)) / float(lam))


def my_optimizer(problem, max_evals):
    """
    COCO/BBOB entry point: CMA-ES-sep with misranking-adaptive boundary reevaluation.

    Mechanism:
    - Evaluate each candidate once.
    - Periodically reevaluate a few candidates around the top-μ boundary and replace their
      fitness with the median across repeats.
    - Use the induced top-μ flip-rate as a noise proxy to scale how many boundary points
      are reevaluated next generations.
    """

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
        + 525252
    ) & 0xFFFFFFFF

    # Thread-safety: prevent pycma global-RNG cross-talk in threaded sweeps.
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

        noise_ema = 0.0
        noise_ema_decay = 0.2

        reeval_band = 2
        reeval_max_frac = 0.3
        reeval_extra_per_point = 1

        probe_count = 2

        gen = 0
        while problem.evaluations < max_evals and not problem.final_target_hit:
            solutions = es.ask()
            lam = int(es.popsize)
            remaining = int(max_evals - problem.evaluations)
            if remaining <= 0:
                break
            if remaining < lam:
                for x in solutions[:remaining]:
                    if problem.final_target_hit:
                        break
                    problem(np.clip(x, lower, upper))
                break

            f_raw = np.empty(lam, dtype=float)
            for i, x in enumerate(solutions):
                if problem.final_target_hit or problem.evaluations >= max_evals:
                    f_raw = f_raw[:i]
                    solutions = solutions[:i]
                    break
                f_raw[i] = float(problem(np.clip(x, lower, upper)))

            if len(f_raw) < lam:
                break

        mu = int(getattr(es.sp, "mu", lam // 2))
        order = np.argsort(f_raw)

        max_count = int(math.floor(float(reeval_max_frac) * float(lam)))
        max_count = max(0, min(max_count, lam))
        if gen == 0:
            count = int(min(max_count, probe_count))
        else:
            count = int(min(max_count, int(round(float(noise_ema) * float(lam)))))

        idx_reeval = _choose_boundary_band(order, mu, band=reeval_band, count=count)
        f_used = np.asarray(f_raw, dtype=float).copy()
        f_a = np.asarray(f_raw, dtype=float).copy()
        f_b = np.asarray(f_raw, dtype=float).copy()

        for idx in idx_reeval.tolist():
            if problem.final_target_hit or problem.evaluations >= max_evals:
                break
            vals = [float(f_raw[idx])]
            for _ in range(int(reeval_extra_per_point)):
                if problem.final_target_hit or problem.evaluations >= max_evals:
                    break
                vals.append(float(problem(np.clip(solutions[idx], lower, upper))))
            f_used[idx] = float(np.median(np.asarray(vals, dtype=float)))
            vals_arr = np.asarray(vals, dtype=float)
            vals_aa = vals_arr[::2]
            vals_bb = vals_arr[1::2]
            if vals_bb.size == 0:
                vals_bb = vals_aa
            if vals_aa.size == 0:
                vals_aa = vals_bb
            f_a[idx] = float(np.median(vals_aa))
            f_b[idx] = float(np.median(vals_bb))

        noise_level = _rank_disagreement(f_a, f_b)
        a = float(noise_ema_decay)
        noise_ema = (1.0 - a) * float(noise_ema) + a * float(noise_level)

        es.tell(solutions, f_used.tolist())
        gen += 1
