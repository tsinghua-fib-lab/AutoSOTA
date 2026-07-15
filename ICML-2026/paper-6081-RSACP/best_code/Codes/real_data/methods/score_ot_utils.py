import math
from functools import lru_cache

import numpy as np


def ecdf_vals(x, grid):
    """Empirical CDF F_x(t), right-continuous."""
    x = np.sort(np.asarray(x, dtype=float).reshape(-1))
    grid = np.asarray(grid, dtype=float)
    if x.size == 0:
        raise ValueError("ecdf_vals requires a nonempty sample.")
    return np.searchsorted(x, grid, side="right") / float(x.size)


def quantile_step(x, u):
    """Discrete empirical quantile."""
    x = np.sort(np.asarray(x, dtype=float).reshape(-1))
    u = np.asarray(u, dtype=float)
    if x.size == 0:
        raise ValueError("quantile_step requires a nonempty sample.")
    idx = np.ceil(np.clip(u, 0.0, 1.0) * x.size).astype(int) - 1
    return x[np.clip(idx, 0, x.size - 1)]


def quantile_interp(x, u):
    """Interpolated empirical quantile, kept for non-RSA baselines."""
    x = np.sort(np.asarray(x, dtype=float).reshape(-1))
    u = np.asarray(u, dtype=float)
    if x.size == 0:
        raise ValueError("quantile_interp requires a nonempty sample.")
    if x.size == 1:
        return np.full_like(u, x[0], dtype=float)
    grid = (np.arange(x.size) + 0.5) / x.size
    return np.interp(np.clip(u, 0.0, 1.0), grid, x, left=x[0], right=x[-1])


def sample_like(scores_target, n_synth, seed=0, method="interp"):
    """Sample synthetic scores from an empirical score distribution."""
    rng = np.random.default_rng(int(seed))
    u = rng.uniform(0.0, 1.0, size=int(n_synth))
    if method == "step":
        return quantile_step(scores_target, u)
    if method == "interp":
        return quantile_interp(scores_target, u)
    raise ValueError(f"Unknown empirical sampling method: {method!r}")


def ot_map_scores(source_scores, target_scores, scores_to_map, method="interp"):
    """
    Legacy 1D monotone map Q_target(F_source(.)).

    This helper remains available for older baselines. It is not the RSA-CP
    method used by SplitConformalRealPlusOTScore.
    """
    source_scores = np.asarray(source_scores, dtype=float).reshape(-1)
    target_scores = np.asarray(target_scores, dtype=float).reshape(-1)
    scores_to_map = np.asarray(scores_to_map, dtype=float)
    if source_scores.size == 0 or target_scores.size == 0:
        raise ValueError("ot_map_scores requires nonempty source and target scores.")
    u = ecdf_vals(source_scores, scores_to_map)
    if method == "step":
        return quantile_step(target_scores, u)
    if method == "interp":
        return quantile_interp(target_scores, u)
    raise ValueError(f"Unknown OT mapping method: {method!r}")


def _barycentric_ot_support(S_source, S_target):
    """
    Fit the one-dimensional barycentric OT map from source to target samples.

    The map on each sorted source atom i is
        T(S_(i)) = sum_j P_ij S_ref_(j) / sum_j P_ij,
    with P the empirical OT coupling between uniform source and target atoms.
    """
    src = np.sort(np.asarray(S_source, dtype=float).reshape(-1))
    tgt = np.sort(np.asarray(S_target, dtype=float).reshape(-1))
    m = len(src)
    N = len(tgt)
    if m == 0 or N == 0:
        raise ValueError("barycentric OT requires nonempty source and target scores.")
    if m == 1:
        return np.array([src[0]], dtype=float), np.array([float(np.mean(tgt))], dtype=float)

    num = np.zeros(m, dtype=float)
    den = np.zeros(m, dtype=float)
    i = 0
    j = 0
    rem_src = 1.0 / m
    rem_tgt = 1.0 / N
    tol = 1e-14

    while i < m and j < N:
        delta = min(rem_src, rem_tgt)
        num[i] += delta * tgt[j]
        den[i] += delta
        rem_src -= delta
        rem_tgt -= delta
        if rem_src <= tol:
            i += 1
            if i < m:
                rem_src = 1.0 / m
        if rem_tgt <= tol:
            j += 1
            if j < N:
                rem_tgt = 1.0 / N

    mapped_src = num / np.maximum(den, 1e-300)
    mapped_src = np.maximum.accumulate(mapped_src)

    unique_src, inv = np.unique(src, return_inverse=True)
    unique_mapped = np.zeros_like(unique_src, dtype=float)
    counts = np.zeros_like(unique_src, dtype=float)
    for idx, val in enumerate(mapped_src):
        unique_mapped[inv[idx]] += val
        counts[inv[idx]] += 1.0
    unique_mapped /= np.maximum(counts, 1.0)
    unique_mapped = np.maximum.accumulate(unique_mapped)
    return unique_src, unique_mapped


def rsa_ot_map(S_new, S_source, S_target):
    """
    RSA-CP barycentric OT map from real/source scores to reference scores.
    """
    S_new = np.asarray(S_new, dtype=float).reshape(-1)
    src, mapped = _barycentric_ot_support(S_source, S_target)
    if src.size == 1:
        return np.full_like(S_new, mapped[0], dtype=float)
    return np.interp(S_new, src, mapped, left=mapped[0], right=mapped[-1])


@lru_cache(maxsize=None)
def _q_betabin_cached(p_rounded, N, a, b):
    p = float(p_rounded)
    N = int(N)
    a = float(a)
    b = float(b)
    if N < 0:
        raise ValueError("N must be nonnegative.")
    if p <= 0:
        return 0
    if p >= 1:
        return N
    x = np.arange(N + 1, dtype=float)
    lgamma = np.vectorize(math.lgamma)
    logpmf = (
        lgamma(N + 1)
        - lgamma(x + 1)
        - lgamma(N - x + 1)
        + lgamma(x + a)
        + lgamma(N - x + b)
        - lgamma(N + a + b)
        - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    )
    logpmf -= np.max(logpmf)
    pmf = np.exp(logpmf)
    pmf /= pmf.sum()
    return int(np.searchsorted(np.cumsum(pmf), p, side="left"))


def q_betabin(p, N, a, b):
    """Beta-Binomial(N, a, b) quantile."""
    return _q_betabin_cached(round(float(p), 12), int(N), float(a), float(b))


def A_value(A_sorted, idx):
    """One-based safe accessor for sorted augmented scores."""
    idx = int(idx)
    if idx <= 0:
        return -np.inf
    if idx > len(A_sorted):
        return np.inf
    return float(A_sorted[idx - 1])


def prepare_rsacp_state(S_real, S_ref, alpha, beta, use_ot=True, prior_scale=1.0, real_weight=1.0):
    """Precompute reusable state for vectorized RSA-CP candidate decisions."""
    S_real = np.asarray(S_real, dtype=float).reshape(-1)
    S_ref = np.asarray(S_ref, dtype=float).reshape(-1)
    if S_real.size == 0 or S_ref.size == 0:
        raise ValueError("RSA-CP requires nonempty real and reference scores.")
    if use_ot:
        map_x, map_y = _barycentric_ot_support(S_real, S_ref)
        Z_real = np.interp(S_real, map_x, map_y, left=map_y[0], right=map_y[-1])
    else:
        map_x, map_y = None, None
        Z_real = S_real.copy()

    m = int(S_real.size)
    N = int(S_ref.size)
    A = np.sort(np.concatenate([Z_real, S_ref]))
    J = int(np.ceil((1.0 - float(alpha)) * (float(real_weight) * m + N + 1)))
    J = min(J, m + N)
    J = max(J, 1)

    q_by_k = np.zeros(m + 2, dtype=float)
    ps = float(prior_scale)
    for k in range(1, m + 2):
        b_minus = q_betabin(float(beta) / 2.0, N, k * ps, (m + 2 - k) * ps)
        b_plus = q_betabin(1.0 - float(beta) / 2.0, N, k * ps, (m + 2 - k) * ps)
        q_by_k[k] = max(
            min(A_value(A, k + b_plus), A_value(A, J)),
            A_value(A, k + b_minus),
        )

    return {
        "S_real": S_real,
        "S_ref": S_ref,
        "Z_real": Z_real,
        "Z_real_sorted": np.sort(Z_real),
        "alpha": float(alpha),
        "beta": float(beta),
        "use_ot": bool(use_ot),
        "map_x": map_x,
        "map_y": map_y,
        "q_by_k": q_by_k,
        "A": A,
        "J": J,
    }


def rsacp_decision_from_state(S_new, state):
    """Vectorized candidate-specific RSA-CP decision using precomputed state."""
    S_new = np.asarray(S_new, dtype=float).reshape(-1)
    if state["use_ot"]:
        Z_new = np.interp(
            S_new,
            state["map_x"],
            state["map_y"],
            left=state["map_y"][0],
            right=state["map_y"][-1],
        )
    else:
        Z_new = S_new.copy()
    k = 1 + np.searchsorted(state["Z_real_sorted"], Z_new, side="right")
    k = np.clip(k, 1, len(state["q_by_k"]) - 1)
    q_vals = state["q_by_k"][k]
    include = Z_new <= q_vals
    return {
        "score": S_new,
        "mapped_score": Z_new,
        "rank_k": k.astype(int),
        "q_rsa_mapped": q_vals,
        "include": include,
    }


def get_rsacp_decision(S_new, S_real, S_ref, alpha, beta, use_ot=True):
    """
    Candidate-specific RSA-CP decision for one or more candidate scores.
    """
    state = prepare_rsacp_state(S_real, S_ref, alpha=alpha, beta=beta, use_ot=use_ot)
    return rsacp_decision_from_state(S_new, state)


def get_rsacp_quantile(S_real, S_ref, alpha, beta, use_ot=True, grid_size=5000, max_expand=6):
    """
    Dense-grid scalar RSA-CP half-width for regression score thresholds.
    """
    S_real = np.asarray(S_real, dtype=float).reshape(-1)
    S_ref = np.asarray(S_ref, dtype=float).reshape(-1)
    state = prepare_rsacp_state(S_real, S_ref, alpha=alpha, beta=beta, use_ot=use_ot)
    upper = np.nanmax(np.concatenate([S_real, S_ref]))
    upper = max(float(upper) * 1.5, float(upper) + 1e-8, 1e-8)
    for _ in range(int(max_expand)):
        grid = np.linspace(0.0, upper, int(grid_size))
        include = rsacp_decision_from_state(grid, state)["include"]
        if not np.any(include):
            return 0.0
        if not np.all(include):
            return float(np.max(grid[include]))
        upper *= 2.0
    return np.inf


def standard_conformal_quantile(scores, alpha, is_aps=False):
    """Finite-sample split conformal quantile."""
    scores = np.sort(np.asarray(scores, dtype=float).reshape(-1))
    n = len(scores)
    if n == 0:
        raise ValueError("No calibration scores supplied.")
    level = (1.0 - float(alpha)) * (1.0 + 1.0 / float(n))
    if level > 1.0:
        return 1.0 if is_aps else np.inf
    return float(scores[int(np.ceil(level * n)) - 1])


@lru_cache(maxsize=None)
def _hg_window_indices(N, m, beta):
    """Hypergeometric SPI rank windows, one-based synthetic rank bounds."""
    N = int(N)
    m = int(m)
    beta = float(beta)
    lower = np.zeros(m, dtype=int)
    upper = np.zeros(m, dtype=int)
    ks = np.arange(1, N + 1, dtype=float)
    lgamma = np.vectorize(math.lgamma)
    log_denom = math.lgamma(N + m + 1) - math.lgamma(m + 1) - math.lgamma(N + 1)
    for i in range(1, m + 1):
        logpmf = (
            lgamma(ks + i - 1)
            - math.lgamma(i)
            - lgamma(ks)
            + lgamma(N + m - ks - i + 2)
            - math.lgamma(m - i + 1)
            - lgamma(N - ks + 2)
            - log_denom
        )
        logpmf -= np.max(logpmf)
        pmf = np.exp(logpmf)
        pmf /= pmf.sum()
        cdf = np.cumsum(pmf)
        lower[i - 1] = int(np.searchsorted(cdf, beta / 2.0, side="left") + 1)
        upper[i - 1] = int(np.searchsorted(cdf, 1.0 - beta / 2.0, side="left") + 1)
    return tuple(lower.tolist()), tuple(upper.tolist())


def spi_quantile_scores(S_real, S_syn, alpha, beta, is_aps=False):
    """
    Score-level SPI fast-form threshold used as a benchmark.
    """
    S_real = np.sort(np.asarray(S_real, dtype=float).reshape(-1))
    S_syn = np.sort(np.asarray(S_syn, dtype=float).reshape(-1))
    m_real = len(S_real)
    N = len(S_syn)
    if m_real == 0 or N == 0:
        raise ValueError("SPI requires nonempty real and synthetic/reference scores.")

    m_with_candidate = m_real + 1
    r_minus, r_plus = _hg_window_indices(N, m_with_candidate, float(beta))
    level = (1.0 - float(alpha)) * (1.0 + 1.0 / float(N))
    threshold = int(np.ceil(N * level))

    r_plus_rank = 0
    r_minus_rank = 0
    for idx, val in enumerate(r_plus, start=1):
        if val <= threshold:
            r_plus_rank = idx
    for idx, val in enumerate(r_minus, start=1):
        if val <= threshold:
            r_minus_rank = idx

    level_plus = (1.0 - float(alpha)) * (
        1.0 + 1.0 / float(N) + 1.0 / float(N * (1.0 - float(alpha)))
    )
    if level_plus > 1.0:
        q_syn_prime = 1.0 if is_aps else np.inf
    else:
        q_syn_prime = float(S_syn[int(np.ceil(level_plus * N)) - 1])

    def real_order_value(rank):
        if rank <= 0:
            return -np.inf
        if rank > m_real:
            return 1.0 if is_aps else np.inf
        return float(S_real[rank - 1])

    q_spi = max(
        min(q_syn_prime, real_order_value(r_minus_rank)),
        real_order_value(r_plus_rank),
    )
    return float(min(q_spi, 1.0) if is_aps else q_spi)
