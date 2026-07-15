import math
import numpy as np

try:
    import cma  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cma = None

from algorithms.berw_es import (
    my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero as berw_hetero,
)
from algorithms.berw_es import (
    my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero_robust as berw_hetero_robust,
)
from algorithms.berw_es import (
    my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero_logw as berw_hetero_logw,
)
from baselines.cmaes_sep import my_optimizer as cmaes_sep

from utils.cma_lock import CMA_GLOBAL_LOCK, cma_locked


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


def _topmu_disagreement(f_a: np.ndarray, f_b: np.ndarray, *, mu: int) -> float:
    """1 - overlap(Top-μ sets) under two noisy draws."""
    lam = int(f_a.size)
    if lam <= 1:
        return 0.0
    mu = int(max(1, min(int(mu), lam)))
    top_a = set(np.argsort(f_a)[:mu].tolist())
    top_b = set(np.argsort(f_b)[:mu].tolist())
    overlap = float(len(top_a.intersection(top_b))) / float(mu)
    return float(1.0 - overlap)


def _misranking_probe(problem, max_evals: int, *, lam_override: int | None = None) -> float | None:
    """
    Spend a tiny budget to estimate misranking severity at the *initial* working point.

    Protocol (fast, no extra deps beyond pycma when available):
    - sample one CMA-style population around initial_solution,
    - evaluate each candidate twice under the native noisy objective,
    - measure normalized mean |Δrank| between the two noisy draws.
    """

    if int(max_evals) <= 0:
        return None

    dim = int(problem.dimension)
    lower = np.asarray(problem.lower_bounds, dtype=float)
    upper = np.asarray(problem.upper_bounds, dtype=float)
    width = np.maximum(upper - lower, 1e-12)

    x0 = np.clip(np.asarray(problem.initial_solution, dtype=float), lower, upper)
    sigma0 = 0.3 * float(np.min(width))

    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 515151
    ) & 0xFFFFFFFF
    rng = np.random.RandomState(int(seed))

    lam = max(4, 4 + int(3 * math.log(dim)))
    if lam_override is not None:
        lam = int(max(2, int(lam_override)))

    if int(getattr(problem, "evaluations", 0)) + 2 * int(lam) > int(max_evals):
        return None

    if cma is not None:
        with cma_locked(seed=int(seed)):
            opts = {
                "bounds": [lower, upper],
                "seed": int(seed),
                "verbose": -9,
                "verb_log": 0,
                "verb_time": 0,
                "CMA_diagonal": True,
                "popsize": int(lam),
                "tolfun": 0.0,
                "tolfunhist": 0.0,
                "tolx": 0.0,
                "tolstagnation": int(1e9),
                "tolxstagnation": False,
                "tolflatfitness": int(1e9),
            }
            es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
            xs = np.asarray(es.ask(), dtype=float)
            if xs.ndim != 2 or xs.shape[1] != dim:
                xs = np.asarray(xs, dtype=float).reshape((lam, dim))
    else:
        xs = x0[None, :] + rng.randn(int(lam), dim) * float(sigma0)

    xs = np.clip(xs, lower, upper)

    f1 = np.empty(int(lam), dtype=float)
    f2 = np.empty(int(lam), dtype=float)
    for i in range(int(lam)):
        if bool(getattr(problem, "final_target_hit", False)) or int(getattr(problem, "evaluations", 0)) >= int(max_evals):
            return None
        f1[i] = float(problem(xs[i]))
    for i in range(int(lam)):
        if bool(getattr(problem, "final_target_hit", False)) or int(getattr(problem, "evaluations", 0)) >= int(max_evals):
            return None
        f2[i] = float(problem(xs[i]))

    return _rank_disagreement(f1, f2)


def _elite_flip_probe(
    problem, max_evals: int, *, mu_frac: float = 0.5, lam_override: int | None = None
) -> float | None:
    """
    Spend a tiny budget to estimate *elite-set instability* at the initial working point.

    Returns: 1 - |Top-μ(f1) ∩ Top-μ(f2)| / μ, where μ = floor(mu_frac * λ).

    This is closer to the mechanism of ES updates: misranking far from the μ-boundary
    matters less than flips in top-μ membership.
    """

    if int(max_evals) <= 0:
        return None

    dim = int(problem.dimension)
    lower = np.asarray(problem.lower_bounds, dtype=float)
    upper = np.asarray(problem.upper_bounds, dtype=float)
    width = np.maximum(upper - lower, 1e-12)

    x0 = np.clip(np.asarray(problem.initial_solution, dtype=float), lower, upper)
    sigma0 = 0.3 * float(np.min(width))

    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 717171
    ) & 0xFFFFFFFF
    rng = np.random.RandomState(int(seed))

    lam = max(4, 4 + int(3 * math.log(dim)))
    if lam_override is not None:
        lam = int(max(2, int(lam_override)))
    mu = int(max(1, min(lam, int(math.floor(float(mu_frac) * float(lam))))))

    if int(getattr(problem, "evaluations", 0)) + 2 * int(lam) > int(max_evals):
        return None

    if cma is not None:
        with cma_locked(seed=int(seed)):
            opts = {
                "bounds": [lower, upper],
                "seed": int(seed),
                "verbose": -9,
                "verb_log": 0,
                "verb_time": 0,
                "CMA_diagonal": True,
                "popsize": int(lam),
                "tolfun": 0.0,
                "tolfunhist": 0.0,
                "tolx": 0.0,
                "tolstagnation": int(1e9),
                "tolxstagnation": False,
                "tolflatfitness": int(1e9),
            }
            es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
            xs = np.asarray(es.ask(), dtype=float)
            if xs.ndim != 2 or xs.shape[1] != dim:
                xs = np.asarray(xs, dtype=float).reshape((lam, dim))
    else:
        xs = x0[None, :] + rng.randn(int(lam), dim) * float(sigma0)

    xs = np.clip(xs, lower, upper)

    f1 = np.empty(int(lam), dtype=float)
    f2 = np.empty(int(lam), dtype=float)
    for i in range(int(lam)):
        if bool(getattr(problem, "final_target_hit", False)) or int(getattr(problem, "evaluations", 0)) >= int(max_evals):
            return None
        f1[i] = float(problem(xs[i]))
    for i in range(int(lam)):
        if bool(getattr(problem, "final_target_hit", False)) or int(getattr(problem, "evaluations", 0)) >= int(max_evals):
            return None
        f2[i] = float(problem(xs[i]))

    return _topmu_disagreement(f1, f2, mu=mu)


def _variance_probe(problem, max_evals: int, *, reps: int) -> float | None:
    """
    Baseline probe: repeated evaluations of x0 to estimate relative noise scale.

    Returns relative std: std(f(x0)) / (|mean(f(x0))| + eps).
    """

    if int(max_evals) <= 0:
        return None
    reps = int(max(2, reps))
    if int(getattr(problem, "evaluations", 0)) + reps > int(max_evals):
        return None

    x0 = np.asarray(problem.initial_solution, dtype=float)
    vals = np.empty(reps, dtype=float)
    for i in range(reps):
        if bool(getattr(problem, "final_target_hit", False)) or int(getattr(problem, "evaluations", 0)) >= int(max_evals):
            return None
        vals[i] = float(problem(x0))
    mu = float(np.mean(vals))
    sd = float(np.std(vals))
    return float(sd / (abs(mu) + 1e-12))


def _tail_ratio_probe(
    problem, max_evals: int, *, reps: int = 2, lam_override: int | None = None
) -> tuple[float | None, float | None]:
    """
    Return (rank_disagreement, tail_ratio) from a small probing budget.

    tail_ratio is computed on per-candidate absolute differences |f_a - f_b|:
        q90(|Δ|) / (median(|Δ|) + eps)

    This is a lightweight proxy for heavy-tailed noise (large outliers).
    """

    if int(max_evals) <= 0:
        return None, None

    reps = int(max(2, reps))

    dim = int(problem.dimension)
    lower = np.asarray(problem.lower_bounds, dtype=float)
    upper = np.asarray(problem.upper_bounds, dtype=float)
    width = np.maximum(upper - lower, 1e-12)

    x0 = np.clip(np.asarray(problem.initial_solution, dtype=float), lower, upper)
    sigma0 = 0.3 * float(np.min(width))

    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 616161
    ) & 0xFFFFFFFF
    rng = np.random.RandomState(int(seed))

    lam = max(4, 4 + int(3 * math.log(dim)))
    if lam_override is not None:
        lam = int(max(2, int(lam_override)))

    needed = int(reps) * int(lam)
    if int(getattr(problem, "evaluations", 0)) + needed > int(max_evals):
        return None, None

    if cma is not None:
        with cma_locked(seed=int(seed)):
            opts = {
                "bounds": [lower, upper],
                "seed": int(seed),
                "verbose": -9,
                "verb_log": 0,
                "verb_time": 0,
                "CMA_diagonal": True,
                "popsize": int(lam),
                "tolfun": 0.0,
                "tolfunhist": 0.0,
                "tolx": 0.0,
                "tolstagnation": int(1e9),
                "tolxstagnation": False,
                "tolflatfitness": int(1e9),
            }
            es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
            xs = np.asarray(es.ask(), dtype=float)
            if xs.ndim != 2 or xs.shape[1] != dim:
                xs = np.asarray(xs, dtype=float).reshape((lam, dim))
    else:
        xs = x0[None, :] + rng.randn(int(lam), dim) * float(sigma0)

    xs = np.clip(xs, lower, upper)

    vals = np.empty((int(reps), int(lam)), dtype=float)
    for r in range(int(reps)):
        for i in range(int(lam)):
            if bool(getattr(problem, "final_target_hit", False)) or int(getattr(problem, "evaluations", 0)) >= int(max_evals):
                return None, None
            vals[r, i] = float(problem(xs[i]))

    f1 = vals[0, :]
    f2 = vals[1, :]
    rd = _rank_disagreement(f1, f2)

    diffs = np.abs(f1 - f2)
    med = float(np.median(diffs))
    q90 = float(np.percentile(diffs, 90))
    tail_ratio = q90 / float(med + 1e-12)

    return float(rd), float(tail_ratio)


def _cma_setup(problem, max_evals: int, *, seed_offset: int = 0):
    """
    Create a sep-CMA-ES (diagonal) instance consistent with `baselines.cmaes_sep`.

    Returns (es, lower, upper) or (None, None, None) if pycma is unavailable.
    """

    if cma is None:
        return None, None, None

    dim = int(problem.dimension)
    lower = np.asarray(problem.lower_bounds, dtype=float)
    upper = np.asarray(problem.upper_bounds, dtype=float)
    width = np.maximum(upper - lower, 1e-12)

    x0 = np.clip(np.asarray(problem.initial_solution, dtype=float), lower, upper)
    sigma0 = 0.3 * float(np.min(width))

    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 42424
        + int(seed_offset)
    ) & 0xFFFFFFFF

    # pycma may touch NumPy's global RNG; reset it for determinism.
    np.random.seed(int(seed))

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
    return es, lower, upper


def _arg_topk_by_distance(xs: np.ndarray, x0: np.ndarray, *, k: int) -> np.ndarray:
    xs = np.asarray(xs, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    k = int(max(1, k))
    if xs.shape[0] <= k:
        return np.arange(xs.shape[0], dtype=int)
    d2 = np.sum((xs - x0[None, :]) ** 2, axis=1)
    return np.argsort(-d2)[:k].astype(int, copy=False)


def _allclose_noise_free(diffs: np.ndarray, base: np.ndarray, *, atol: float, rtol: float) -> bool:
    diffs = np.asarray(diffs, dtype=float)
    base = np.asarray(base, dtype=float)
    tol = float(atol) + float(rtol) * np.maximum(1.0, np.abs(base))
    return bool(np.all(diffs <= tol))


def _misranking_probe_switch_warmstart(problem, max_evals, *, threshold: float) -> None:
    """
    Cost-aware ProbeSwitch variant (warmstart):
    - Reuses the first CMA generation as the "candidate set" (no extra sampling cost).
    - Early-exits as "deterministic" if a few farthest candidates have no reeval noise.

    Intended to address a practical risk: probe overhead can measurably hurt deterministic regimes
    under fixed evaluation budgets (observed on external nonconvex MLP).
    """

    # Use the same CMA seed as `CMA-ES-sep` for fair overhead comparisons.
    # Thread-safety: prevent pycma global-RNG cross-talk in threaded sweeps.
    with CMA_GLOBAL_LOCK:
        es, lower, upper = _cma_setup(problem, int(max_evals), seed_offset=0)
        if es is None:
            # Fallback to the non-warmstart switch.
            my_optimizer_misranking_probe_switch_t012(problem, max_evals)
            return

        # --- First generation (shared) ---
        if bool(getattr(problem, "final_target_hit", False)) or int(getattr(problem, "evaluations", 0)) >= int(max_evals):
            return

        solutions = es.ask()
        remaining = int(max_evals) - int(getattr(problem, "evaluations", 0))
        if remaining <= 0:
            return
        if remaining < int(es.popsize):
            # Not enough budget to finish one full CMA generation; just spend remaining evals.
            for x in solutions[:remaining]:
                if bool(getattr(problem, "final_target_hit", False)):
                    break
                problem(np.clip(x, lower, upper))
            return

        xs = np.empty((int(es.popsize), int(problem.dimension)), dtype=float)
        f1 = np.empty(int(es.popsize), dtype=float)
        x0 = np.clip(np.asarray(problem.initial_solution, dtype=float), lower, upper)
        for i, x in enumerate(solutions):
            if bool(getattr(problem, "final_target_hit", False)) or int(getattr(problem, "evaluations", 0)) >= int(max_evals):
                return
            xc = np.clip(x, lower, upper)
            xs[i] = xc
            f1[i] = float(problem(xc))

        # --- Early deterministic check (cheap) ---
        k_check = 4
        idx_check = _arg_topk_by_distance(xs, x0, k=k_check)
        f_check = np.empty(idx_check.size, dtype=float)
        for j, idx in enumerate(idx_check.tolist()):
            if bool(getattr(problem, "final_target_hit", False)) or int(getattr(problem, "evaluations", 0)) >= int(max_evals):
                return
            f_check[j] = float(problem(xs[int(idx)]))
        diffs = np.abs(f_check - f1[idx_check])
        if _allclose_noise_free(diffs, f1[idx_check], atol=0.0, rtol=0.0):
            # Deterministic regime: continue CMA, no full misranking probe.
            es.tell(solutions, f1.tolist())
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
                values = []
                for x in solutions:
                    if bool(getattr(problem, "final_target_hit", False)):
                        break
                    values.append(float(problem(np.clip(x, lower, upper))))
                if len(values) < int(es.popsize):
                    break
                es.tell(solutions, values)
            return

        # --- Full misranking probe on the shared candidate set ---
        # Reuse `f_check` as part of the 2nd draw to avoid extra overhead in noisy regimes.
        f2 = np.empty(int(es.popsize), dtype=float)
        f2[idx_check] = f_check
        mask = np.ones(int(es.popsize), dtype=bool)
        mask[idx_check] = False
        for i in np.where(mask)[0].tolist():
            if bool(getattr(problem, "final_target_hit", False)) or int(getattr(problem, "evaluations", 0)) >= int(max_evals):
                return
            f2[int(i)] = float(problem(xs[int(i)]))
        rd = _rank_disagreement(f1, f2)

        if float(rd) >= float(threshold):
            # In high-misranking regimes, a noisy "best-of-one-draw" warmstart can be
            # actively harmful (noise-lucky outliers). Prefer a clean BERW start.
            berw_hetero(problem, max_evals)
            return

        # Otherwise continue CMA.
        es.tell(solutions, f1.tolist())
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
            values = []
            for x in solutions:
                if bool(getattr(problem, "final_target_hit", False)):
                    break
                values.append(float(problem(np.clip(x, lower, upper))))
            if len(values) < int(es.popsize):
                break
            es.tell(solutions, values)


def my_optimizer_misranking_probe_switch_warmstart_t012(problem, max_evals):
    """Warmstart variant: threshold 0.12 (default)."""
    _misranking_probe_switch_warmstart(problem, max_evals, threshold=0.12)


def my_optimizer_misranking_probe_switch_warmstart_t022(problem, max_evals):
    """Warmstart variant: threshold 0.22 (conservative / safer default for transfer)."""
    _misranking_probe_switch_warmstart(problem, max_evals, threshold=0.22)


def my_optimizer_noise_probe_switch_warmstart(problem, max_evals):
    """
    Cost-aware 3-way NoiseProbe switch (warmstart):
    - Uses the first CMA generation as the shared candidate set.
    - Early-exits as deterministic if a few farthest points show zero reeval noise.

    Actions: (CMA-ES-sep, BERW-Hetero, BERW-HeteroRobust).
    """

    # Use the same CMA seed as `CMA-ES-sep` for fair overhead comparisons.
    # Thread-safety: prevent pycma global-RNG cross-talk in threaded sweeps.
    with CMA_GLOBAL_LOCK:
        es, lower, upper = _cma_setup(problem, int(max_evals), seed_offset=0)
        if es is None:
            my_optimizer_noise_probe_switch(problem, max_evals)
            return

        if bool(getattr(problem, "final_target_hit", False)) or int(getattr(problem, "evaluations", 0)) >= int(max_evals):
            return

        solutions = es.ask()
        remaining = int(max_evals) - int(getattr(problem, "evaluations", 0))
        if remaining <= 0:
            return
        if remaining < int(es.popsize):
            for x in solutions[:remaining]:
                if bool(getattr(problem, "final_target_hit", False)):
                    break
                problem(np.clip(x, lower, upper))
            return

        xs = np.empty((int(es.popsize), int(problem.dimension)), dtype=float)
        f1 = np.empty(int(es.popsize), dtype=float)
        x0 = np.clip(np.asarray(problem.initial_solution, dtype=float), lower, upper)
        for i, x in enumerate(solutions):
            if bool(getattr(problem, "final_target_hit", False)) or int(getattr(problem, "evaluations", 0)) >= int(max_evals):
                return
            xc = np.clip(x, lower, upper)
            xs[i] = xc
            f1[i] = float(problem(xc))

        k_check = 4
        idx_check = _arg_topk_by_distance(xs, x0, k=k_check)
        f_check = np.empty(idx_check.size, dtype=float)
        for j, idx in enumerate(idx_check.tolist()):
            if bool(getattr(problem, "final_target_hit", False)) or int(getattr(problem, "evaluations", 0)) >= int(max_evals):
                return
            f_check[j] = float(problem(xs[int(idx)]))
        diffs = np.abs(f_check - f1[idx_check])
        if _allclose_noise_free(diffs, f1[idx_check], atol=0.0, rtol=0.0):
            # Deterministic: continue CMA.
            es.tell(solutions, f1.tolist())
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
                values = []
                for x in solutions:
                    if bool(getattr(problem, "final_target_hit", False)):
                        break
                    values.append(float(problem(np.clip(x, lower, upper))))
                if len(values) < int(es.popsize):
                    break
                es.tell(solutions, values)
            return

        # Reuse `f_check` as part of the 2nd draw to avoid extra overhead in noisy regimes.
        f2 = np.empty(int(es.popsize), dtype=float)
        f2[idx_check] = f_check
        mask = np.ones(int(es.popsize), dtype=bool)
        mask[idx_check] = False
        for i in np.where(mask)[0].tolist():
            if bool(getattr(problem, "final_target_hit", False)) or int(getattr(problem, "evaluations", 0)) >= int(max_evals):
                return
            f2[int(i)] = float(problem(xs[int(i)]))

        rd = _rank_disagreement(f1, f2)
        diffs = np.abs(f1 - f2)
        med = float(np.median(diffs))
        q90 = float(np.percentile(diffs, 90))
        tail_ratio = q90 / float(med + 1e-12)

        rd_threshold = 0.12
        tail_threshold = 4.0

        if float(rd) < float(rd_threshold):
            # Continue CMA.
            es.tell(solutions, f1.tolist())
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
                values = []
                for x in solutions:
                    if bool(getattr(problem, "final_target_hit", False)):
                        break
                    values.append(float(problem(np.clip(x, lower, upper))))
                if len(values) < int(es.popsize):
                    break
                es.tell(solutions, values)
            return

        if float(tail_ratio) >= float(tail_threshold):
            berw_hetero_robust(problem, max_evals)
        else:
            berw_hetero(problem, max_evals)

def my_optimizer_misranking_probe_switch(problem, max_evals):
    """
    COCO/BBOB entry point: misranking-probe switch between CMA-ES-sep and BERW-Hetero.

    Rationale:
    - Low misranking regime: deterministic ordering is reliable; robust probabilistic selection can hurt.
    - High misranking regime: robust probabilistic selection helps; standard CMA can mis-update.
    """

    rd = _misranking_probe(problem, int(max_evals))

    # Conservative default thresholds (chosen to separate near-deterministic vs highly-misranked regimes).
    # The bbob-noisy D=40 `--sampling es` stats show a strong bimodality (≈0 vs ≈0.3).
    threshold = 0.12

    if rd is not None and float(rd) >= float(threshold):
        berw_hetero(problem, max_evals)
    else:
        cmaes_sep(problem, max_evals)


def my_optimizer_elite_flip_probe_switch(problem, max_evals):
    """
    Switch using elite-set instability (top-μ flip rate) as the probe.
    """

    flip = _elite_flip_probe(problem, int(max_evals), mu_frac=0.5)
    threshold = 0.25
    if flip is not None and float(flip) >= float(threshold):
        berw_hetero(problem, max_evals)
    else:
        cmaes_sep(problem, max_evals)


def _misranking_probe_switch(problem, max_evals, *, threshold: float) -> None:
    rd = _misranking_probe(problem, int(max_evals))
    if rd is not None and float(rd) >= float(threshold):
        berw_hetero(problem, max_evals)
    else:
        cmaes_sep(problem, max_evals)


def _misranking_probe_switch_to(problem, max_evals, *, threshold: float, berw_optimizer) -> None:
    """
    Generalized misranking-probe switch:
    - If rd >= threshold -> run the provided BERW optimizer
    - Else -> run CMA-ES-sep

    Useful for external tasks where the preferred "robust" branch differs from
    the default COCO branch (BERW-Hetero).
    """

    rd = _misranking_probe(problem, int(max_evals))
    if rd is not None and float(rd) >= float(threshold):
        berw_optimizer(problem, max_evals)
    else:
        cmaes_sep(problem, max_evals)


def my_optimizer_misranking_probe_switch_t008(problem, max_evals):
    """Threshold ablation: 0.08."""
    _misranking_probe_switch(problem, max_evals, threshold=0.08)


def my_optimizer_misranking_probe_switch_t010(problem, max_evals):
    """Threshold ablation: 0.10."""
    _misranking_probe_switch(problem, max_evals, threshold=0.10)


def my_optimizer_misranking_probe_switch_t012(problem, max_evals):
    """Threshold ablation: 0.12 (default)."""
    _misranking_probe_switch(problem, max_evals, threshold=0.12)


def my_optimizer_misranking_probe_switch_t014(problem, max_evals):
    """Threshold ablation: 0.14."""
    _misranking_probe_switch(problem, max_evals, threshold=0.14)


def my_optimizer_misranking_probe_switch_t016(problem, max_evals):
    """Threshold ablation: 0.16."""
    _misranking_probe_switch(problem, max_evals, threshold=0.16)


def my_optimizer_misranking_probe_switch_t022(problem, max_evals):
    """Threshold ablation: 0.22 (conservative / safer default for transfer)."""
    _misranking_probe_switch(problem, max_evals, threshold=0.22)


def my_optimizer_misranking_probe_switch_t019(problem, max_evals):
    """Threshold: 0.19 (optimal for LQR)."""
    _misranking_probe_switch(problem, max_evals, threshold=0.19)


def my_optimizer_misranking_probe_switch_t021(problem, max_evals):
    """Threshold: 0.21 (optimal for CartPole-HT)."""
    _misranking_probe_switch(problem, max_evals, threshold=0.21)


def my_optimizer_misranking_probe_switch_t026(problem, max_evals):
    """Threshold: 0.26 (optimal for HPO Digits)."""
    _misranking_probe_switch(problem, max_evals, threshold=0.26)


def my_optimizer_misranking_probe_switch_t038(problem, max_evals):
    """Threshold: 0.38 (optimal for COCO d=20)."""
    _misranking_probe_switch(problem, max_evals, threshold=0.38)


def my_optimizer_misranking_probe_switch_t046(problem, max_evals):
    """Threshold: 0.46 (optimal for COCO d=10)."""
    _misranking_probe_switch(problem, max_evals, threshold=0.46)


def my_optimizer_misranking_probe_switch_robust_t012(problem, max_evals):
    """Robust branch (BERW-HeteroRobust): threshold 0.12 (default)."""
    _misranking_probe_switch_to(problem, max_evals, threshold=0.12, berw_optimizer=berw_hetero_robust)


def my_optimizer_misranking_probe_switch_robust_t022(problem, max_evals):
    """Robust branch (BERW-HeteroRobust): threshold 0.22 (conservative / safer default for transfer)."""
    _misranking_probe_switch_to(problem, max_evals, threshold=0.22, berw_optimizer=berw_hetero_robust)


def my_optimizer_misranking_probe_switch_robust_t019(problem, max_evals):
    """Robust branch: threshold 0.19 (optimal for LQR)."""
    _misranking_probe_switch_to(problem, max_evals, threshold=0.19, berw_optimizer=berw_hetero_robust)


def my_optimizer_misranking_probe_switch_robust_t021(problem, max_evals):
    """Robust branch: threshold 0.21 (optimal for CartPole-HT)."""
    _misranking_probe_switch_to(problem, max_evals, threshold=0.21, berw_optimizer=berw_hetero_robust)


def my_optimizer_misranking_probe_switch_robust_t026(problem, max_evals):
    """Robust branch: threshold 0.26 (optimal for HPO Digits)."""
    _misranking_probe_switch_to(problem, max_evals, threshold=0.26, berw_optimizer=berw_hetero_robust)


# --- Log-weight variants (weight-function ablation) ---

def my_optimizer_misranking_probe_switch_logw_t012(problem, max_evals):
    """ProbeSwitch with BERW-Hetero-LogW branch, threshold 0.12."""
    _misranking_probe_switch_to(problem, max_evals, threshold=0.12, berw_optimizer=berw_hetero_logw)


def my_optimizer_misranking_probe_switch_logw_t022(problem, max_evals):
    """ProbeSwitch with BERW-Hetero-LogW branch, threshold 0.22."""
    _misranking_probe_switch_to(problem, max_evals, threshold=0.22, berw_optimizer=berw_hetero_logw)


def my_optimizer_random_switch(problem, max_evals):
    """Ablation baseline: random choice between CMA-ES-sep and BERW-Hetero."""

    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 919191
    ) & 0xFFFFFFFF
    rng = np.random.RandomState(int(seed))
    if float(rng.rand()) < 0.5:
        cmaes_sep(problem, max_evals)
    else:
        berw_hetero(problem, max_evals)


def my_optimizer_variance_probe_switch(problem, max_evals):
    """
    Ablation baseline: switch using a variance-like proxy at x0 (relative std).
    """

    rel_sd = _variance_probe(problem, int(max_evals), reps=10)
    threshold = 0.05
    if rel_sd is not None and float(rel_sd) >= float(threshold):
        berw_hetero(problem, max_evals)
    else:
        cmaes_sep(problem, max_evals)


def my_optimizer_noise_probe_switch(problem, max_evals):
    """
    COCO/BBOB entry point: small probe -> choose among (CMA-sep, BERW-Hetero, BERW-HeteroRobust).

    Heuristic:
    - If misranking is low: run CMA-ES-sep.
    - Else: if noise appears heavy-tailed: run a more robust BERW variant.
    - Else: run BERW-Hetero.
    """

    rd, tail_ratio = _tail_ratio_probe(problem, int(max_evals), reps=2)

    rd_threshold = 0.12
    tail_threshold = 4.0

    if rd is not None and float(rd) < float(rd_threshold):
        cmaes_sep(problem, max_evals)
        return

    if tail_ratio is not None and float(tail_ratio) >= float(tail_threshold):
        berw_hetero_robust(problem, max_evals)
    else:
        berw_hetero(problem, max_evals)


def my_optimizer_combined_probe_switch(problem, max_evals):
    """
    Ablation/robustness: switch using BOTH probes (OR rule).

    Motivation:
    - On bbob-noisy, a variance-at-x0 proxy can be a strong predictor.
    - On state-dependent noise (e.g., radial noise wrappers), variance at x0 can be ~0
      while misranking on a candidate set is high.
    - Using an OR rule aims to cover both regimes with minimal extra budget.
    """

    rd = _misranking_probe(problem, int(max_evals))
    rel_sd = _variance_probe(problem, int(max_evals), reps=10)

    rd_threshold = 0.12
    var_threshold = 0.05

    if (rd is not None and float(rd) >= float(rd_threshold)) or (rel_sd is not None and float(rel_sd) >= float(var_threshold)):
        berw_hetero(problem, max_evals)
    else:
        cmaes_sep(problem, max_evals)


def my_optimizer_cascaded_probe_switch(problem, max_evals):
    """
    Cascaded switch (variance -> misranking):

    - If variance at x0 is large, switch to BERW-Hetero (cheap, strong baseline on bbob-noisy).
    - Otherwise, fall back to misranking-probe on a candidate set to catch x-dependent noise
      regimes where Var[f(x0)] can be ~0 but misranking is high.
    """

    rel_sd = _variance_probe(problem, int(max_evals), reps=10)
    var_threshold = 0.05
    if rel_sd is not None and float(rel_sd) >= float(var_threshold):
        berw_hetero(problem, max_evals)
        return

    rd = _misranking_probe(problem, int(max_evals))
    rd_threshold = 0.12
    if rd is not None and float(rd) >= float(rd_threshold):
        berw_hetero(problem, max_evals)
    else:
        cmaes_sep(problem, max_evals)
