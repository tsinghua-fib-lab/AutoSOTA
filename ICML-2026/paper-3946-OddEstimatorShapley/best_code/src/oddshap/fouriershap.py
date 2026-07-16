"""FourierSHAP: sparse Fourier extraction and Shapley value computation.

Numpy-only implementation based on:

  Gorji, A., Amrollahi, A., & Krause, A. (2025). SHAP values via sparse
  Fourier representation. In The Thirty-ninth Annual Conference on Neural
  Information Processing Systems. https://openreview.net/forum?id=wab4BEAUt6

Original source code https://openreview.net/forum?id=jh7JQkgWGe:
  - bbfourier_Oct_1-3/algorithms/fourier_explainer.py
  - bbfourier_Oct_1-3/fourier_extractor/jax_fourier_extractor.py
  - bbfourier_Oct_1-3/swht_jax/swht_jax.py

Public API
----------
budget_to_b(n, budget, T)
    Choose the largest b such that T*(n+1)*2^b <= budget.
extract_fourier(game_fn, n, b, T, batch_size, seed)
    Run SWHT and return (freqs, amps) matrix representation.
freqs_amps_to_dict(freqs, amps)
    Convert (freqs, amps) to frozenset -> amplitude dict.
fourier_shap_analytical(f_t, n)
    Exact dataset-free Shapley values: phi_i = sum_{T ni i} (-2/|T|) * A_T.
fourier_explainer_matrix_batch(f_t, data, queries)
    Dataset-dependent vectorised Shapley computation.
fourier_explainer_matrix_cat_batch(f_t, data, queries, feature_map)
    Variant for binarised categorical features.
FourierSHAP_Approximator
    shapiq Approximator wrapper around extract_fourier + fourier_shap_analytical.
"""

from __future__ import annotations

import itertools
import math
import time
import warnings
import numpy as np


# ---------------------------------------------------------------------------
# WHT (Fast Walsh-Hadamard Transform, unnormalized)
# ---------------------------------------------------------------------------

def _fwht(a: np.ndarray) -> np.ndarray:
    """Unnormalized Fast Walsh-Hadamard Transform."""
    a = a.copy().astype(float)
    N = len(a)
    h = 1
    while h < N:
        a = a.reshape(-1, 2 * h)
        top = a[:, :h] + a[:, h:]
        bot = a[:, :h] - a[:, h:]
        a = np.hstack([top, bot])
        h *= 2
    return a.ravel()


# ---------------------------------------------------------------------------
# Time-sample generation
# ---------------------------------------------------------------------------

def _get_hashing_matrix(n: int, b: int, seed: int) -> np.ndarray:
    """Deterministic (n, b) binary hashing matrix for a given round seed."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 2, size=(n, b))


def _get_time_samples(n: int, b: int, T: int, base_seed: int = 0) -> np.ndarray:
    """Return time-sample coalitions of shape (T, n, 2^b)."""
    B = 2 ** b
    t_list = np.array(list(itertools.product([0, 1], repeat=b)), dtype=int).T  # (b, B)
    time_samples = []
    for i in range(T):
        H = _get_hashing_matrix(n, b, seed=base_seed + i)
        time_samples.append((H @ t_list) % 2)  # (n, B)
    return np.array(time_samples)  # (T, n, B)


# ---------------------------------------------------------------------------
# Batched game evaluation
# ---------------------------------------------------------------------------

def _eval_batched(game_fn, coalitions: np.ndarray, batch_size: int) -> np.ndarray:
    """Evaluate game_fn on (m, n) coalitions in batches, return (m,) values."""
    m = coalitions.shape[0]
    out = np.empty(m)
    for start in range(0, m, batch_size):
        end = min(start + batch_size, m)
        out[start:end] = game_fn(coalitions[start:end])
    return out


# ---------------------------------------------------------------------------
# SWHT peeling
# ---------------------------------------------------------------------------

def _hash_to_index(hashed_freqs: np.ndarray) -> np.ndarray:
    """Convert (b, B) binary matrix to (B,) integer bucket indices."""
    b = hashed_freqs.shape[0]
    powers = (2 ** np.arange(b - 1, -1, -1)).reshape(-1, 1)
    return (powers * hashed_freqs).sum(axis=0).astype(int)


def _peel_round(
    ref_wht: np.ndarray,
    shifted_wht: np.ndarray,
    freqs: np.ndarray,
    amps: np.ndarray,
    n: int,
    b: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One SWHT peeling round; returns (recovered_freqs, residual_amps)."""
    H = _get_hashing_matrix(n, b, seed)
    hashed_freqs = (H.T @ freqs) % 2        # (b, B)
    index = _hash_to_index(hashed_freqs)    # (B,)

    ref_peeled = ref_wht.copy()
    np.add.at(ref_peeled, index, -amps)

    signed_amps = np.where(freqs == 0, 1, -1) * amps  # (n, B)
    shift_peeled = shifted_wht.copy()
    for k in range(n):
        np.add.at(shift_peeled[k], index, -signed_amps[k])

    recovered_freqs = np.where(
        np.sign(ref_peeled) == np.sign(shift_peeled), 0, 1
    ).astype(int)
    return recovered_freqs, ref_peeled


def _sparse_wht(
    ref_whts: np.ndarray,
    shifted_whts: np.ndarray,
    n: int,
    b: int,
    T: int,
    base_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """SWHT peeling across T rounds; returns (freqs, amps) shape (n,B), (B,)."""
    B = 2 ** b
    freqs = np.zeros((n, B), dtype=int)
    amps = np.zeros(B)

    for i in range(T):
        rec_freqs, rec_amps = _peel_round(
            ref_whts[i], shifted_whts[i], freqs, amps, n, b, seed=base_seed + i
        )

        all_freqs = np.hstack([freqs, rec_freqs])    # (n, 2B)
        all_amps = np.concatenate([amps, rec_amps])  # (2B,)

        unique_freqs, inverse = np.unique(all_freqs, axis=1, return_inverse=True)
        n_unique = unique_freqs.shape[1]
        unique_amps = np.zeros(n_unique)
        np.add.at(unique_amps, inverse, all_amps)

        if n_unique <= B:
            pad = B - n_unique
            freqs = np.hstack([unique_freqs, np.zeros((n, pad), dtype=int)])
            amps = np.concatenate([unique_amps, np.zeros(pad)])
        else:
            top_idx = np.argsort(np.abs(unique_amps))[-B:]
            freqs = unique_freqs[:, top_idx]
            amps = unique_amps[top_idx]

    return freqs, amps


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def budget_to_b(n: int, budget: int, T: int = 5) -> int:
    """Choose the largest b such that T * (n+1) * 2^b <= budget.

    The SWHT uses exactly T * (n+1) * 2^b game evaluations per call to
    extract_fourier.  Given a fixed evaluation budget, this returns the b
    that spends as much of it as possible (maximising B = 2^b recoverable
    Fourier terms).

    Args:
        n      : Number of players.
        budget : Maximum number of game evaluations allowed.
        T      : Number of peeling rounds (default 5).

    Returns:
        b : Largest integer such that T*(n+1)*2^b <= budget.  Minimum 1.

    Raises:
        ValueError : If the budget is too small even for b=1.

    Examples:
        budget_to_b(n=14, budget=5000, T=5)  ->  6   (uses 4800 evals)
        budget_to_b(n=16, budget=10000, T=5) ->  6   (uses 5440 evals)
    """
    evals_per_b = T * (n + 1)
    b = int(math.floor(math.log2(budget / evals_per_b)))
    if b < 1:
        raise ValueError(
            f"Budget {budget} is too small: even b=1 needs {evals_per_b * 2} evals."
        )
    return b


def extract_fourier(
    game_fn,
    n: int,
    b: int,
    T: int = 5,
    batch_size: int = 2048,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover the K = 2^b strongest Fourier coefficients of a black-box game.

    Uses the Sparse Walsh-Hadamard Transform (SWHT) peeling algorithm.
    Total game evaluations: T * (n + 1) * 2^b.

    Args:
        game_fn    : Callable (coalitions: np.ndarray shape (m, n)) -> (m,).
                     Coalitions are binary {0,1}, 1 = player present.
        n          : Number of players.
        b          : Log2 of the sparsity budget (B = 2^b coefficients kept).
        T          : Number of peeling rounds (default 5).
        batch_size : Evaluations per batch passed to game_fn (default 2048).
        seed       : Base random seed for reproducibility (default 0).

    Returns:
        freqs : np.ndarray shape (n, B) binary — each column is a frequency
                vector (support of one Fourier term).
        amps  : np.ndarray shape (B,) — amplitudes A_S such that
                f(x) approx sum_k amps[k] * (-1)^(x . freqs[:,k]).
    """
    B = 2 ** b
    sqrt_B = np.sqrt(B)

    time_samples = _get_time_samples(n, b, T, base_seed=seed)  # (T, n, B)
    ref_whts = np.zeros((T, B))
    shifted_whts = np.zeros((T, n, B))

    for i in range(T):
        ts = time_samples[i]  # (n, B)

        ref_vals = _eval_batched(game_fn, ts.T.astype(float), batch_size)
        ref_whts[i] = _fwht(ref_vals) / sqrt_B

        for k in range(n):
            ts_shifted = ts.copy()
            ts_shifted[k] = 1 - ts_shifted[k]
            shift_vals = _eval_batched(game_fn, ts_shifted.T.astype(float), batch_size)
            shifted_whts[i, k] = _fwht(shift_vals) / sqrt_B

    freqs, amps = _sparse_wht(ref_whts, shifted_whts, n, b, T, base_seed=seed)
    # SWHT accumulates amplitudes in units of sqrt(B)*A_S; normalise back.
    amps = amps / sqrt_B
    return freqs, amps


def freqs_amps_to_dict(freqs: np.ndarray, amps: np.ndarray) -> dict:
    """Convert (freqs, amps) matrix format to frozenset -> amplitude dict.

    Aggregates duplicate frequency vectors by summing amplitudes.
    Empty-support and zero-amplitude entries are dropped.

    Args:
        freqs : shape (n, B) binary array.
        amps  : shape (B,) float array.

    Returns:
        f_t : dict mapping frozenset of player indices to float amplitude.
    """
    f_t: dict = {}
    freqs_np = np.asarray(freqs)
    amps_np = np.asarray(amps)
    for k in range(freqs_np.shape[1]):
        support = frozenset(int(i) for i in np.where(freqs_np[:, k])[0])
        if len(support) == 0:
            continue
        f_t[support] = f_t.get(support, 0.0) + float(amps_np[k])
    return {s: a for s, a in f_t.items() if abs(a) > 0.0}


def fourier_shap_analytical(f_t: dict, n: int) -> np.ndarray:
    """Exact dataset-free Shapley values from a Fourier representation.

    Closed-form relationship between Fourier coefficients and Shapley values:

        phi_i = sum_{T containing i} (-2 / |T|) * A_T

    Only odd-degree terms contribute; even-degree terms cancel exactly.

    Args:
        f_t : dict mapping frozenset of player indices to amplitude A_T.
        n   : Number of players.

    Returns:
        phi : Shapley values, shape (n,).
    """
    phi = np.zeros(n)
    for T, A in f_t.items():
        k = len(T)
        if k == 0 or k % 2 == 0:
            continue
        weight = -2.0 / k
        for i in T:
            phi[i] += weight * A
    return phi


def fourier_explainer_matrix_batch(
    f_t: dict,
    data: np.ndarray,
    queries: np.ndarray,
) -> np.ndarray:
    """Dataset-dependent Shapley values from a sparse Fourier representation.

    Args:
        f_t     : dict mapping frozenset of feature indices to amplitude.
        data    : Background dataset, shape (n_data, n_features), binary {0,1}.
        queries : Instances to explain, shape (n_queries, n_features), binary.

    Returns:
        shap_values : shape (n_queries, n_features).
    """
    data_set_size = data.shape[0]
    feature_size = data.shape[1]
    query_count = queries.shape[0]

    data = np.array(data, dtype=float)
    queries = np.array(queries, dtype=float)

    if len(f_t) == 0:
        warnings.warn("WARNING: Fourier transform is empty.")

    diff = (data != np.expand_dims(queries, axis=1))
    sub = data - np.expand_dims(queries, axis=1)

    shap_values = np.zeros([query_count, feature_size])
    for freq, A in f_t.items():
        f = list(freq)
        if len(f) == 0:
            continue

        mat_diff = sub[:, :, f]
        mat_sign = np.sum(data[:, f], axis=1, keepdims=True) - data[:, f]
        mat_sign = np.where((mat_sign % 2) == 1, -1, 1)
        mat_sign = np.expand_dims(mat_sign, axis=0)

        mat_coeff = np.sum(diff[:, :, f], axis=2, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            mat_coeff = np.where((mat_coeff % 2) == 1, 1 / mat_coeff, 0)

        shap_values[:, f] += A * np.sum((mat_coeff * mat_sign) * mat_diff, axis=1)
    shap_values *= (2 / data_set_size)
    return shap_values


def fourier_explainer_matrix_cat_batch(
    f_t: dict,
    data: np.ndarray,
    queries: np.ndarray,
    feature_map: np.ndarray,
) -> np.ndarray:
    """Dataset-dependent Shapley computation for binarised categorical features.

    Args:
        f_t         : dict mapping frozenset of feature indices to amplitude.
        data        : Binarised dataset, shape (n_data, n_bin_features).
        queries     : Instances to explain, shape (n_queries, n_bin_features).
        feature_map : Array of length n_bin_features mapping each binary
                      feature to its original feature index.

    Returns:
        shap_values : shape (n_queries, n_original_features).
    """
    data_set_size = data.shape[0]
    feature_size = len(set(feature_map))
    query_count = queries.shape[0]

    data = np.array(data, dtype=float)
    queries = np.array(queries, dtype=float)

    if len(f_t) == 0:
        warnings.warn("WARNING: Fourier transform is empty.")

    shap_values = np.zeros([query_count, feature_size])
    for freq, A in f_t.items():
        f = sorted(list(freq))
        if len(f) == 0:
            continue
        mapped_features = list(feature_map[f])
        features = sorted(list(set(mapped_features)))
        feature_starts = [mapped_features.index(o_f) for o_f in features]

        f_x = np.add.reduceat(data[:, f], feature_starts, axis=1) % 2
        f_x_s = np.add.reduceat(queries[:, f], feature_starts, axis=1) % 2

        mat_diff = f_x - np.expand_dims(f_x_s, axis=1)
        mat_sign = np.sum(data[:, f], axis=1, keepdims=True) - f_x
        mat_sign = np.where((mat_sign % 2) == 1, -1, 1)
        mat_sign = np.expand_dims(mat_sign, axis=0)

        diff = (f_x != np.expand_dims(f_x_s, axis=1))
        mat_coeff = np.sum(diff, axis=2, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            mat_coeff = np.where((mat_coeff % 2) == 1, 1 / mat_coeff, 0)

        shap_values[:, features] += A * np.sum((mat_coeff * mat_sign) * mat_diff, axis=1)

    shap_values *= (2 / data_set_size)
    return shap_values


# ---------------------------------------------------------------------------
# shapiq Approximator wrapper
# ---------------------------------------------------------------------------

def _build_interaction_values_fourier(
    phi: np.ndarray,
    n: int,
    baseline_value: float,
    budget: int,
) -> "InteractionValues":
    from shapiq.interaction_values import InteractionValues
    values = np.empty(n + 1)
    interaction_lookup: dict[tuple[int, ...], int] = {}
    values[0] = baseline_value
    interaction_lookup[()] = 0
    for i in range(n):
        values[i + 1] = phi[i]
        interaction_lookup[(i,)] = i + 1
    return InteractionValues(
        values=values,
        index="SV",
        interaction_lookup=interaction_lookup,
        baseline_value=baseline_value,
        min_order=0,
        max_order=1,
        n_players=n,
        estimated=True,
        estimation_budget=budget,
    )


class FourierSHAP_Approximator:
    """Shapley estimator via sparse Fourier extraction (Gorji et al., NeurIPS 2025).

    At each call to approximate(budget, game), chooses the largest b such that
    T*(n+1)*2^b <= budget, runs SWHT to recover the Fourier representation,
    then computes Shapley values analytically from the Fourier coefficients.
    """

    def __init__(self, n: int, T: int = 5, random_state: int | None = None):
        self.n = n
        self.T = T
        self.random_state = random_state if random_state is not None else 0
        self.name = "FourierSHAP"
        self.runtime_last_approximate_run: dict[str, float] = {}

    def approximate(self, budget: int, game) -> "InteractionValues":
        start = time.time()

        b = budget_to_b(self.n, budget, self.T)
        actual_budget = self.T * (self.n + 1) * (2 ** b)

        baseline_value = float(game(np.zeros((1, self.n)))[0])

        eval_start = time.time()
        freqs, amps = extract_fourier(
            game, n=self.n, b=b, T=self.T, seed=self.random_state,
        )
        eval_end = time.time()

        f_t = freqs_amps_to_dict(freqs, amps)
        phi = fourier_shap_analytical(f_t, self.n)

        self.runtime_last_approximate_run = {
            "evaluations": eval_end - eval_start,
            "total": time.time() - start,
        }
        return _build_interaction_values_fourier(
            phi, self.n, baseline_value, budget=actual_budget,
        )
