import numpy as np
from typing import Callable, Tuple, Optional, Literal, Dict, Any

PBAAlgo = Literal["pso", "de", "fss"]


# ============================================================
# Utils
# ============================================================
def _clip(x: np.ndarray, lb: float, ub: Optional[float]) -> np.ndarray:
    if ub is None:
        return np.maximum(x, lb)
    return np.clip(x, lb, ub)


def _default_ub_from_data(A: np.ndarray) -> float:
    # A very simple, stable upper-bound heuristic.
    return float(np.maximum(A.max(), 1.0))


def _iters_from_budget(eval_budget: int, pop: int, min_iters: int = 20) -> int:
    # Rough mapping: each iteration ~ pop objective evaluations.
    return max(min_iters, eval_budget // max(1, pop))


# ============================================================
# PSO (Particle Swarm Optimization)
# ============================================================
def pso_minimize(
    f: Callable[[np.ndarray], float],
    dim: int,
    pop: int = 40,
    iters: int = 200,
    lb: float = 0.0,
    ub: Optional[float] = 1.0,
    seed: int = 0,
    w_inertia: float = 0.72,
    c1: float = 1.49,
    c2: float = 1.49,
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    if ub is None:
        ub = 1.0

    X = rng.uniform(lb, ub, size=(pop, dim))
    V = rng.uniform(-(ub - lb), (ub - lb), size=(pop, dim)) * 0.1

    pbest = X.copy()
    pbest_f = np.array([f(x) for x in X], dtype=float)
    g_idx = int(np.argmin(pbest_f))
    gbest = pbest[g_idx].copy()
    gbest_f = float(pbest_f[g_idx])

    for _ in range(iters):
        r1 = rng.random(size=(pop, dim))
        r2 = rng.random(size=(pop, dim))

        V = w_inertia * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest[None, :] - X)
        X = X + V
        X = _clip(X, lb=lb, ub=ub)

        fx = np.array([f(x) for x in X], dtype=float)
        improve = fx < pbest_f
        pbest[improve] = X[improve]
        pbest_f[improve] = fx[improve]

        g_idx = int(np.argmin(pbest_f))
        if pbest_f[g_idx] < gbest_f:
            gbest_f = float(pbest_f[g_idx])
            gbest = pbest[g_idx].copy()

    return gbest, gbest_f


# ============================================================
# DE (Differential Evolution) - DE/rand/1/bin
# ============================================================
def de_minimize(
    f: Callable[[np.ndarray], float],
    dim: int,
    pop: int = 40,
    iters: int = 200,
    lb: float = 0.0,
    ub: Optional[float] = 1.0,
    seed: int = 0,
    F: float = 0.6,
    CR: float = 0.5,
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    if ub is None:
        ub = 1.0

    X = rng.uniform(lb, ub, size=(pop, dim))
    FX = np.array([f(x) for x in X], dtype=float)

    for _ in range(iters):
        for i in range(pop):
            idxs = [j for j in range(pop) if j != i]
            a, b, c = rng.choice(idxs, size=3, replace=False)
            mutant = X[a] + F * (X[b] - X[c])

            cross = rng.random(dim) < CR
            cross[rng.integers(0, dim)] = True
            trial = np.where(cross, mutant, X[i])
            trial = _clip(trial, lb=lb, ub=ub)

            f_trial = float(f(trial))
            if f_trial < FX[i]:
                X[i] = trial
                FX[i] = f_trial

    best_i = int(np.argmin(FX))
    return X[best_i].copy(), float(FX[best_i])


# ============================================================
# FSS (Fish School Search)
# ============================================================
def fss_minimize(
    f: Callable[[np.ndarray], float],
    dim: int,
    pop: int = 30,
    iters: int = 150,
    lb: float = 0.0,
    ub: Optional[float] = 1.0,
    seed: int = 0,
    step_ind_init: float = 0.1,
    step_ind_final: float = 0.001,
    step_vol_init: float = 0.01,
    step_vol_final: float = 0.0001,
    w_min: float = 1.0,
    w_max: float = 5.0,
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    if ub is None:
        ub = 1.0

    X = rng.uniform(lb, ub, size=(pop, dim))
    FX = np.array([f(x) for x in X], dtype=float)
    W = np.full(pop, (w_min + w_max) / 2.0, dtype=float)

    def lerp(a, b, t):
        return a + (b - a) * t

    def barycenter(x: np.ndarray, w: np.ndarray) -> np.ndarray:
        sw = float(np.sum(w)) + 1e-12
        return (w[:, None] * x).sum(axis=0) / sw

    best_i = int(np.argmin(FX))
    gbest = X[best_i].copy()
    gbest_f = float(FX[best_i])

    total_weight_prev = float(np.sum(W))

    for t in range(iters):
        alpha = t / max(1, iters - 1)
        step_ind = lerp(step_ind_init, step_ind_final, alpha)
        step_vol = lerp(step_vol_init, step_vol_final, alpha)

        # 1) Individual movement (greedy accept)
        delta_X = np.zeros_like(X)
        delta_f = np.zeros(pop, dtype=float)

        for i in range(pop):
            r = rng.uniform(-1.0, 1.0, size=dim)
            cand = X[i] + step_ind * r * (ub - lb)
            cand = _clip(cand, lb=lb, ub=ub)

            f_cand = float(f(cand))
            improvement = FX[i] - f_cand
            if improvement > 0:
                delta_X[i] = cand - X[i]
                delta_f[i] = improvement
                X[i] = cand
                FX[i] = f_cand

        # 2) Feeding
        max_delta = float(np.max(delta_f)) if np.max(delta_f) > 0 else 0.0
        if max_delta > 0:
            W = W + (delta_f / (max_delta + 1e-12))
        W = np.clip(W, w_min, w_max)

        # 3) Collective-instinctive movement
        sum_df = float(np.sum(delta_f))
        if sum_df > 0:
            I = (delta_X * delta_f[:, None]).sum(axis=0) / (sum_df + 1e-12)
            X = _clip(X + I[None, :], lb=lb, ub=ub)
            FX = np.array([f(x) for x in X], dtype=float)

        # 4) Collective-volitive movement
        bc = barycenter(X, W)
        total_weight = float(np.sum(W))

        # Contract if improved (total weight increased), else expand
        direction = -1.0 if total_weight > total_weight_prev else 1.0
        rand_scale = rng.random(size=(pop, 1))
        X = X + direction * step_vol * rand_scale * (X - bc[None, :])
        X = _clip(X, lb=lb, ub=ub)
        FX = np.array([f(x) for x in X], dtype=float)

        total_weight_prev = total_weight

        best_i = int(np.argmin(FX))
        if FX[best_i] < gbest_f:
            gbest_f = float(FX[best_i])
            gbest = X[best_i].copy()

    return gbest, gbest_f


# ============================================================
# Unified PBA interface
# ============================================================
def pba_minimize(
    algo: PBAAlgo,
    f: Callable[[np.ndarray], float],
    dim: int,
    pop: int,
    iters: int,
    lb: float,
    ub: Optional[float],
    seed: int,
    algo_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, float]:
    algo_kwargs = algo_kwargs or {}
    if algo == "pso":
        return pso_minimize(f, dim=dim, pop=pop, iters=iters, lb=lb, ub=ub, seed=seed, **algo_kwargs)
    if algo == "de":
        return de_minimize(f, dim=dim, pop=pop, iters=iters, lb=lb, ub=ub, seed=seed, **algo_kwargs)
    if algo == "fss":
        return fss_minimize(f, dim=dim, pop=pop, iters=iters, lb=lb, ub=ub, seed=seed, **algo_kwargs)
    raise ValueError(f"Unknown algo={algo}. Use one of: 'pso', 'de', 'fss'.")


# ============================================================
# Unified PBA-based NMF initialization (row-wise W, col-wise H)
# ============================================================
def nmf_init_pba(
    A: np.ndarray,
    k: int,
    algo: PBAAlgo = "pso",
    pop: int = 40,
    eval_budget: int = 2500,
    lb: float = 0.0,
    ub: Optional[float] = None,
    seed: int = 0,
    h0_scale: float = 0.1,
    algo_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    PBA-based NMF initialization:
      1) random nonnegative H0
      2) for each row i:  minimize ||a_i - w_i H0||_2^2  => W[i,:]
      3) for each col j:  minimize ||a_:j - W h_j||_2^2  => H[:,j]
    """
    A = np.asarray(A, dtype=float)
    if np.any(A < 0):
        raise ValueError("A must be nonnegative.")
    m, n = A.shape
    if not (1 <= k <= min(m, n)):
        raise ValueError("k must be in [1, min(m,n)].")

    if ub is None:
        ub = _default_ub_from_data(A)

    rng = np.random.default_rng(seed)
    H0 = rng.random((k, n)) * (ub * h0_scale)
    H0 = _clip(H0, lb=lb, ub=ub)

    iters = _iters_from_budget(eval_budget, pop)

    # --- Optimize W row-wise ---
    W = np.zeros((m, k), dtype=float)
    for i in range(m):
        a_i = A[i, :]

        def f_row(w_i: np.ndarray) -> float:
            r = a_i - (w_i @ H0)
            return float(np.dot(r, r))

        w_best, _ = pba_minimize(
            algo=algo, f=f_row, dim=k, pop=pop, iters=iters,
            lb=lb, ub=ub, seed=seed * 1000003 + i, algo_kwargs=algo_kwargs
        )
        W[i, :] = w_best

    # --- Optimize H column-wise ---
    H = np.zeros((k, n), dtype=float)
    for j in range(n):
        a_j = A[:, j]

        def f_col(h_j: np.ndarray) -> float:
            r = a_j - (W @ h_j)
            return float(np.dot(r, r))

        h_best, _ = pba_minimize(
            algo=algo, f=f_col, dim=k, pop=pop, iters=iters,
            lb=lb, ub=ub, seed=seed * 1000003 + (m + j), algo_kwargs=algo_kwargs
        )
        H[:, j] = h_best

    return W, H

