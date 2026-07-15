import math
import numpy as np

try:
    import cma  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cma = None

from utils.cma_lock import cma_locked


def _bootstrap_rank_order(
    problem,
    xs: np.ndarray,
    fvals: np.ndarray,
    *,
    max_evals: int,
    rng: np.random.RandomState,
    mu: int,
    bootstrap_samples: int,
    reeval_band: int,
    reeval_extra_per_point: int,
    z_clip: float,
) -> tuple[np.ndarray, float]:
    """
    Compute a robust ordering of candidates by estimating top-μ membership probabilities
    via parametric bootstrap.

    The ordering is by descending p_i = P(rank_i < μ), tie-broken by aggregated f.
    """

    lam = int(fvals.size)
    mu = int(max(1, min(mu, lam)))

    # Select a small band around the μ-boundary for reevaluation.
    order_det = np.argsort(fvals)
    center = max(0, min(lam - 1, mu - 1))
    start = max(0, center - int(reeval_band))
    end = min(lam, center + int(reeval_band) + 1)
    idx_reeval = order_det[start:end]

    samples = [np.asarray([float(fvals[i])], dtype=float) for i in range(lam)]
    f_used = fvals.astype(float, copy=True)

    if int(reeval_extra_per_point) > 0 and idx_reeval.size > 0:
        for idx in idx_reeval.tolist():
            if int(getattr(problem, "evaluations", 0)) >= int(max_evals) or bool(getattr(problem, "final_target_hit", False)):
                break
            vals = [float(f_used[idx])]
            for _ in range(int(reeval_extra_per_point)):
                if int(getattr(problem, "evaluations", 0)) >= int(max_evals) or bool(getattr(problem, "final_target_hit", False)):
                    break
                vals.append(float(problem(xs[idx])))
            arr = np.asarray(vals, dtype=float)
            samples[int(idx)] = arr
            f_used[int(idx)] = float(np.median(arr))

    # Fit a lightweight heteroscedastic scale model: |noise| ≈ s0 + s1 * |f|.
    xs_fit = []
    rs_fit = []
    for arr in samples:
        if int(getattr(arr, "size", 0)) <= 1:
            continue
        m = float(np.median(arr))
        r = np.asarray(arr, dtype=float) - m
        xs_fit.extend([abs(m)] * int(r.size))
        rs_fit.extend(r.tolist())

    z_pool = np.array([], dtype=float)
    s0 = 0.0
    s1 = 0.0
    if len(rs_fit) >= 6:
        x = np.asarray(xs_fit, dtype=float)
        r = np.asarray(rs_fit, dtype=float)
        y = np.abs(r)
        X = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        s0 = float(max(0.0, coef[0]))
        s1 = float(max(0.0, coef[1]))
        scale = np.maximum(s0 + s1 * x, 1e-12)
        z = r / scale
        z = np.clip(z, -float(z_clip), float(z_clip))
        z_pool = z.astype(float, copy=False)

    B = int(max(1, bootstrap_samples))
    counts = np.zeros(lam, dtype=float)
    f_boot = np.empty(lam, dtype=float)

    for _ in range(B):
        for i in range(lam):
            arr = samples[i]
            if int(getattr(arr, "size", 0)) > 1:
                f_boot[i] = float(arr[int(rng.randint(0, int(arr.size)))])
            else:
                base = float(f_used[i])
                if z_pool.size <= 0:
                    f_boot[i] = base
                else:
                    z = float(z_pool[int(rng.randint(0, int(z_pool.size)))])
                    scale = float(max(1e-12, s0 + s1 * abs(base)))
                    f_boot[i] = base + z * scale

        order = np.argsort(f_boot)
        counts[order[:mu]] += 1.0

    p = counts / float(B)
    overlap = float(np.sum(p * p) / float(mu)) if mu > 0 else 1.0
    overlap_min = float(mu) / float(lam)
    denom = float(max(1e-12, 1.0 - overlap_min))
    stability = float((overlap - overlap_min) / denom)
    stability = float(np.clip(stability, 0.0, 1.0))
    # Order by high top-μ membership probability, tie-break by aggregated fitness.
    order = np.lexsort((f_used, -p))
    return order.astype(int, copy=False), stability


def my_optimizer(problem, max_evals):
    """
    COCO/BBOB entry point: diagonal CMA-ES with probabilistic rank ordering under noise.

    This baseline keeps CMA-ES mechanics (pycma) but replaces the per-generation
    ordering of candidates by a bootstrap-estimated top-μ membership probability.
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
        + 42425
    ) & 0xFFFFFFFF
    rng = np.random.RandomState(int(seed) ^ 0xA5A5A5A5)

    with cma_locked(seed=int(seed)):
        popsize = max(4, 4 + int(3 * math.log(dim)))
        opts = {
            "bounds": [lower, upper],
            "maxfevals": int(max_evals),
            "seed": int(seed),
            "verbose": -9,
            "verb_log": 0,
            "verb_time": 0,
            "CMA_diagonal": True,
            "popsize": int(popsize),
            "tolfun": 0.0,
            "tolfunhist": 0.0,
            "tolx": 0.0,
            "tolstagnation": int(1e9),
            "tolxstagnation": False,
            "tolflatfitness": int(1e9),
        }
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

        # Small-budget defaults for rapid iteration.
        bootstrap_samples = 32
        reeval_band = 1
        reeval_extra_per_point = 1
        z_clip = 10.0
        stability_threshold = 0.95

        while int(getattr(problem, "evaluations", 0)) < int(max_evals) and not bool(getattr(problem, "final_target_hit", False)):
            solutions = es.ask()
            remaining = int(max_evals) - int(getattr(problem, "evaluations", 0))
            if remaining <= 0:
                break

            if remaining < int(es.popsize):
                for x in solutions[:remaining]:
                    if bool(getattr(problem, "final_target_hit", False)):
                        break
                    problem(np.clip(x, lower, upper))
                break

            xs = np.empty((int(es.popsize), dim), dtype=float)
            fvals = np.empty(int(es.popsize), dtype=float)
            for i, x in enumerate(solutions):
                if bool(getattr(problem, "final_target_hit", False)) or int(getattr(problem, "evaluations", 0)) >= int(max_evals):
                    xs = xs[:i]
                    fvals = fvals[:i]
                    solutions = solutions[:i]
                    break
                x_clip = np.clip(x, lower, upper)
                xs[i] = x_clip
                fvals[i] = float(problem(x_clip))

            if int(fvals.size) < int(es.popsize):
                break

            mu = int(max(1, int(es.popsize) // 2))
            order, stability = _bootstrap_rank_order(
                problem,
                xs,
                fvals,
                max_evals=int(max_evals),
                rng=rng,
                mu=mu,
                bootstrap_samples=int(bootstrap_samples),
                reeval_band=int(reeval_band),
                reeval_extra_per_point=int(reeval_extra_per_point),
                z_clip=float(z_clip),
            )

            if float(stability) >= float(stability_threshold):
                es.tell(solutions, fvals.tolist())
            else:
                pseudo = np.empty(int(es.popsize), dtype=float)
                pseudo[order] = np.arange(int(es.popsize), dtype=float)
                es.tell(solutions, pseudo.tolist())
