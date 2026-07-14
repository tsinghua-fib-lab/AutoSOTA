# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bayesian Quadrature active sampler with three GP feature setups.

This module implements BQ active sampling using a Gaussian Process with
three feature setups:

1. **SF (Score Features)** — :class:`BQPriorSampler` uses other models'
   predictions as GP features with a **linear kernel**.
2. **RPF (Raw Prompt Features)** — :func:`_bq_matern_active_sampling` and
   :func:`_bq_matern_random_sampling` use raw text embeddings with a
   standalone **Matérn 2.5 kernel** and neutral prior (0.5).  No encoder
   training required.
3. **TPF (Tuned Prompt Features)** — :class:`BQEncoderSampler` uses a
   pre-trained neural encoder's phi embeddings with a **Matérn 2.5 kernel**
   (via the encoder).

Each setup supports three selection strategies:

- **Active** — variance-reduction acquisition (BQ-SF, BQ-RPF, BQ-TPF)
- **Random** — random point selection (BQ-SF Rand, BQ-RPF Rand, BQ-TPF Rand)
- **Rounded** — active BQ with rounded posterior mean (BQ-SF Rounded, etc.)

Example::

    from proeval.sampler import BQPriorSampler, BQEncoderSampler
    from proeval.sampler.bq import _bq_matern_active_sampling

    # SF (linear kernel, score features)
    sampler = BQPriorSampler(noise_variance=0.3)
    result = sampler.sample(predictions="svamp", target_model="gemini25_flash", budget=20)

    # RPF (Matérn kernel, raw embeddings, no encoder)
    result = _bq_matern_active_sampling(embeddings, test_y, u=0.5*np.ones(n), budget=50)

    # TPF (Matérn kernel, encoder embeddings)
    sampler = BQEncoderSampler(
        encoder_path="encoder.pth", embeddings_path="embeddings.npy",
    )
    result = sampler.sample(predictions="gsm8k", target_model="gemma3_27b", budget=50)

    print(result.estimates[-1])   # final mean estimate
    print(result.mae(true_mean))  # final MAE
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from proeval.sampler.data import (
    extract_model_predictions,
    load_predictions,
    setup_train_test_split,
)


def _resolve_predictions(
    predictions: Union[str, "pd.DataFrame", "Dataset"],  # noqa: F821
    data_dir: Optional[str] = None,
) -> Tuple["pd.DataFrame", Optional[str]]:
    """Normalise the ``predictions`` argument to ``(DataFrame, dataset_name)``.

    Accepts a dataset name (loaded by convention), a pre-loaded DataFrame, or a
    :class:`~proeval.utils.Dataset` (uses its cached/lazily-loaded predictions
    and its ``name``). ``dataset_name`` is ``None`` for a bare DataFrame, which
    disables name-dependent behaviour (GMM selection, DICES label binarisation).
    """
    # Local import keeps the sampler import-light and avoids any cycle.
    from proeval.utils.dataset import Dataset

    if isinstance(predictions, Dataset):
        return predictions.predictions(data_dir=data_dir), predictions.name
    if isinstance(predictions, str):
        return load_predictions(predictions, data_dir=data_dir), predictions
    return predictions, None


# Result container
@dataclass
class SamplingResult:
    """Container for BQ sampling results.

    Attributes:
        estimates: Posterior mean estimate at each step ``(budget,)``.
        rounded_estimates: Rounded posterior mean estimate at each step ``(budget,)``.
        selected_indices: Indices of selected samples in acquisition order.
        posterior_mean: Final posterior mean ``(n_samples,)``.
        posterior_var: Final posterior variance ``(n_samples,)``.
        prior_mean: Prior mean used ``(n_samples,)``.
        integral_variance: BQ integral posterior variance at each step ``(budget,)``.
            This is the variance of the mean estimate, computed as
            ``mean(posterior_diagonal_variance)``.
    """

    estimates: np.ndarray
    selected_indices: List[int]
    posterior_mean: np.ndarray
    posterior_var: np.ndarray
    prior_mean: np.ndarray
    rounded_estimates: Optional[np.ndarray] = None
    integral_variance: Optional[np.ndarray] = None

    def mae(self, true_mean: float) -> float:
        """Return final MAE ``|estimate - true_mean|``."""
        return float(np.abs(self.estimates[-1] - true_mean))

    def mae_curve(self, true_mean: float) -> np.ndarray:
        """Return MAE at every step."""
        return np.abs(self.estimates - true_mean)

    @property
    def integral_std(self) -> Optional[np.ndarray]:
        """Return integral standard deviation at each step."""
        if self.integral_variance is None:
            return None
        return np.sqrt(np.maximum(self.integral_variance, 0.0))

    def should_abstain(self, threshold: float = 0.05) -> bool:
        """Return True if final integral std exceeds *threshold*.

        A high integral std means the BQ estimate is uncertain;
        the method should abstain rather than reporting a point estimate.
        """
        if self.integral_variance is None:
            return False
        final_std = float(np.sqrt(max(self.integral_variance[-1], 0.0)))
        return final_std > threshold



# Internal GP helpers
def _get_posterior(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    noise_variance: float,
    train_x_inds: List[int],
    u: np.ndarray,
    full_cov: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute GP posterior with learned prior mean *u*."""
    k_t_inv = np.linalg.inv(
        np.dot(train_x, train_x.T) / noise_variance + np.eye(train_x.shape[0])
    )
    u_t = u + (1 / noise_variance) * np.dot(
        np.dot(np.dot(test_x.T, k_t_inv), train_x),
        train_y - u[train_x_inds],
    )
    if full_cov:
        s_t = np.dot(np.dot(test_x.T, k_t_inv), test_x)
    else:
        s_t = np.sum(np.dot(test_x.T, k_t_inv) * test_x.T, axis=1)
    return u_t, s_t


def _find_best_i_and_update(
    A: np.ndarray, K: np.ndarray, Q: np.ndarray, e: float
) -> Tuple[int, np.ndarray]:
    """Find best index to sample and update inverse covariance."""
    s = A.sum(axis=0)
    v = s @ K
    numerator_roots = Q @ v
    numerators = e * (numerator_roots ** 2)
    QK = Q @ K
    quad_forms = np.einsum("ij,ij->i", Q, QK)
    denominators = 1 + e * quad_forms
    scores = numerators / denominators
    best_i = int(np.argmax(scores))

    KQ_best = QK[best_i]
    denom_best = denominators[best_i]
    B_best = e * np.outer(KQ_best, KQ_best) / denom_best
    return best_i, K - B_best


def _variance_improvement(
    train_x: np.ndarray,
    k_t_inv: Optional[np.ndarray],
    noise_variance: float,
    unlabeled_indices: List[int],
    test_x: np.ndarray,
) -> Tuple[int, np.ndarray]:
    """Select the next sample that maximally reduces posterior variance."""
    if train_x.shape[1] == 0:
        k_t_inv = np.eye(test_x.shape[0])
    A = test_x.T
    Q = test_x[:, unlabeled_indices].T
    best_local_idx, k_t_inv = _find_best_i_and_update(A, k_t_inv, Q, noise_variance)
    return unlabeled_indices[best_local_idx], k_t_inv


def _variance_improvement_gp(
    posterior_cov: np.ndarray,
    unlabeled_indices: List[int],
    noise_variance: float = 0.0,
) -> int:
    """Select the point with maximum integral variance reduction.

    For BQ, the integral variance is ``1ᵀ K' 1 = sum(K')``.
    Using rank-1 covariance update, the integral variance reduction
    when conditioning on point *i* is::

        Δ_i = (Σ_j K'[j, i])² / (K'[i, i] + σ²)

    Returns the global index of the best candidate.
    """
    cols = posterior_cov[:, unlabeled_indices]          # (M, n_unlabeled)
    col_sums = np.sum(cols, axis=0)                     # (n_unlabeled,)
    diag_vals = np.diag(posterior_cov)[unlabeled_indices]
    denominators = np.maximum(diag_vals + noise_variance, 1e-10)
    variance_reductions = (col_sums ** 2) / denominators
    best_local_idx = int(np.argmax(variance_reductions))
    return unlabeled_indices[best_local_idx]


# Core BQ sampling loop
def _bq_active_sampling(
    test_x: np.ndarray,
    test_y: np.ndarray,
    u: np.ndarray,
    S: np.ndarray,
    budget: int,
    n_init: int = 0,
    noise_variance: float = 0.3,
) -> SamplingResult:
    """Run the core BQ active sampling loop.

    Returns a :class:`SamplingResult` with posterior-mean estimates at each
    step, the acquisition-order indices, and the final posterior.
    """
    n_samples = test_x.shape[1]

    # ALGO-03: Stratified initial sampling by prior mean quantiles
    # Divides samples into n_init quantiles of u and picks one random
    # sample from each, ensuring diverse initial observations.
    labeled_indices: List[int] = []
    if n_init > 0:
        # Sort by prior mean and partition into n_init quantiles
        sorted_by_prior = np.argsort(u)
        quantile_size = max(1, n_samples // n_init)
        for q in range(n_init):
            start = q * quantile_size
            end = min(start + quantile_size, n_samples)
            if start < end:
                candidates = sorted_by_prior[start:end]
                # Prefer "interesting" range (0.2-0.6) when available within quantile
                interesting = [i for i in candidates if 0.15 < u[i] < 0.7]
                pool = interesting if len(interesting) >= 1 else candidates
                labeled_indices.append(int(np.random.choice(pool)))
    unlabeled_indices = [i for i in range(n_samples) if i not in labeled_indices]

    estimates = np.ones(budget) * np.mean(u)
    rounded_estimates = np.ones(budget) * np.mean(np.round(u))
    
    true_prior_integral_var = np.sum(S) / (n_samples * n_samples)
    integral_variance = np.ones(budget) * true_prior_integral_var
    k_t_inv = None
    if n_init > 0 and len(labeled_indices) > 0:
        # Initialize k_t_inv for the first _variance_improvement call
        # (needed when starting with pre-labeled samples at t > 0)
        train_x_init = test_x[:, labeled_indices]
        k_t_inv = np.linalg.inv(
            np.dot(train_x_init, train_x_init.T) / noise_variance
            + np.eye(train_x_init.shape[0])
        )
    mu_x = np.mean(test_x, axis=1, keepdims=True)

    for t in range(n_init, budget):
        train_x_t = test_x[:, labeled_indices]
        train_y_t = test_y[labeled_indices]

        if t == 0:
            u_t = u
            int_var_t = true_prior_integral_var
        else:
            u_t, s_t = _get_posterior(
                train_x_t, train_y_t, test_x, noise_variance, labeled_indices, u
            )
            _, s_mu = _get_posterior(
                train_x_t, train_y_t, mu_x, noise_variance, labeled_indices, u
            )
            int_var_t = s_mu[0]

        rounded_estimates[t] = np.mean(np.round(u_t))
        estimates[t] = np.mean(u_t)
        # Integral variance is variance of the mean
        integral_variance[t] = np.maximum(int_var_t, 0.0)

        if t < budget - 1:
            best_idx, k_t_inv = _variance_improvement(
                train_x_t, k_t_inv, noise_variance, unlabeled_indices, test_x
            )
            labeled_indices.append(best_idx)
            unlabeled_indices.remove(best_idx)

    # Back-fill initial steps
    for t in range(n_init):
        idx_slice = labeled_indices[: t + 1]
        u_t, s_t = _get_posterior(
            test_x[:, idx_slice], test_y[idx_slice], test_x,
            noise_variance, idx_slice, u,
        )
        _, s_mu = _get_posterior(
            test_x[:, idx_slice], test_y[idx_slice], mu_x,
            noise_variance, idx_slice, u,
        )
        rounded_estimates[t] = np.mean(np.round(u_t))
        estimates[t] = np.mean(u_t)
        integral_variance[t] = np.maximum(s_mu[0], 0.0)

    # Final posterior
    if labeled_indices:
        final_u, final_s = _get_posterior(
            test_x[:, labeled_indices], test_y[labeled_indices],
            test_x, noise_variance, labeled_indices, u,
        )
    else:
        final_u, final_s = u, np.diag(S)

    return SamplingResult(
        estimates=estimates,
        rounded_estimates=rounded_estimates,
        selected_indices=labeled_indices,
        posterior_mean=final_u,
        posterior_var=final_s,
        prior_mean=u,
        integral_variance=integral_variance,
    )


def _bq_random_sampling(
    test_x: np.ndarray,
    test_y: np.ndarray,
    u: np.ndarray,
    S: np.ndarray,
    budget: int,
    noise_variance: float = 0.3,
) -> SamplingResult:
    """BQ estimation with **random** point selection.

    Selects points uniformly at random (no active acquisition), but
    uses the GP posterior with learned prior *u* to estimate the mean.
    This baseline isolates the value of BQ's active acquisition from
    its posterior-based estimation.

    Returns a :class:`SamplingResult` identical in structure to
    :func:`_bq_active_sampling`.
    """
    n_samples = test_x.shape[1]

    # Randomly select all indices up front
    all_indices = list(np.random.choice(n_samples, min(budget, n_samples), replace=False))

    estimates = np.ones(budget) * np.mean(u)
    
    true_prior_integral_var = np.sum(S) / (n_samples * n_samples)
    integral_variance = np.ones(budget) * true_prior_integral_var
    mu_x = np.mean(test_x, axis=1, keepdims=True)

    for t in range(budget):
        labeled_indices = all_indices[: t + 1]
        train_x_t = test_x[:, labeled_indices]
        train_y_t = test_y[labeled_indices]

        if t == 0 and len(labeled_indices) == 0:
            u_t, s_t = u, np.diag(S)
            int_var_t = true_prior_integral_var
        else:
            u_t, s_t = _get_posterior(
                train_x_t, train_y_t, test_x, noise_variance, labeled_indices, u
            )
            _, s_mu = _get_posterior(
                train_x_t, train_y_t, mu_x, noise_variance, labeled_indices, u
            )
            int_var_t = s_mu[0]

        estimates[t] = np.mean(u_t)
        integral_variance[t] = np.maximum(int_var_t, 0.0)

    # Final posterior
    labeled_indices = all_indices[: budget]
    if labeled_indices:
        final_u, final_s = _get_posterior(
            test_x[:, labeled_indices], test_y[labeled_indices],
            test_x, noise_variance, labeled_indices, u,
        )
    else:
        final_u, final_s = u, np.diag(S)

    return SamplingResult(
        estimates=estimates,
        selected_indices=all_indices[: budget],
        posterior_mean=final_u,
        posterior_var=final_s,
        prior_mean=u,
        integral_variance=integral_variance,
    )


# Standalone Matérn kernel BQ (for RPF — no encoder needed)
def _compute_matern_kernel_np(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    lengthscale: float = 1.0,
    nu: float = 2.5,
) -> np.ndarray:
    """Compute Matérn kernel matrix from embeddings using numpy.

    Args:
        X: Embeddings ``(n, d)``.
        Y: Optional second set of embeddings ``(m, d)``. If ``None``, computes
            ``K(X, X)``.
        lengthscale: Kernel lengthscale.
        nu: Matérn smoothness (0.5, 1.5, or 2.5).

    Returns:
        Kernel matrix ``(n, m)`` or ``(n, n)`` if ``Y is None``.
    """
    if Y is None:
        Y = X
    # Pairwise distances
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)  # (n, 1)
    Y_sq = np.sum(Y ** 2, axis=1, keepdims=True)  # (m, 1)
    dist_sq = X_sq + Y_sq.T - 2 * X @ Y.T          # (n, m)
    dist_sq = np.maximum(dist_sq, 0.0)
    dist = np.sqrt(dist_sq + 1e-12)
    scaled_dist = dist / lengthscale

    if nu == 0.5:
        return np.exp(-scaled_dist)
    elif nu == 1.5:
        sqrt3 = np.sqrt(3)
        return (1 + sqrt3 * scaled_dist) * np.exp(-sqrt3 * scaled_dist)
    else:  # nu == 2.5 (default)
        sqrt5 = np.sqrt(5)
        return (
            (1 + sqrt5 * scaled_dist + (5 / 3) * scaled_dist ** 2)
            * np.exp(-sqrt5 * scaled_dist)
        )


def _get_posterior_matern(
    train_emb: np.ndarray,
    train_y: np.ndarray,
    test_emb: np.ndarray,
    noise_variance: float,
    train_inds: List[int],
    u: np.ndarray,
    lengthscale: float = 1.0,
    nu: float = 2.5,
    full_cov: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """GP posterior with standalone Matérn kernel (no encoder).

    Args:
        train_emb: Labeled sample embeddings ``(n_labeled, d)``.
        train_y: Labels ``(n_labeled,)``.
        test_emb: All sample embeddings ``(n_samples, d)``.
        noise_variance: Observation noise.
        train_inds: Indices of labeled samples.
        u: Prior mean ``(n_samples,)``.
        lengthscale: Matérn lengthscale.
        nu: Matérn smoothness.
        full_cov: Whether to return full covariance matrix.

    Returns:
        ``(posterior_mean, posterior_var_or_cov)``.
    """
    n_train = train_emb.shape[0]

    K_train = _compute_matern_kernel_np(train_emb, lengthscale=lengthscale, nu=nu)
    K_train_reg = K_train + noise_variance * np.eye(n_train)
    K_test_train = _compute_matern_kernel_np(test_emb, train_emb, lengthscale=lengthscale, nu=nu)

    # Cholesky solve
    try:
        L = np.linalg.cholesky(K_train_reg)
    except np.linalg.LinAlgError:
        K_train_reg += 1e-4 * np.eye(n_train)
        L = np.linalg.cholesky(K_train_reg)

    y_residual = train_y - u[train_inds]
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_residual))

    posterior_mean = u + K_test_train @ alpha

    if full_cov:
        K_test = _compute_matern_kernel_np(test_emb, lengthscale=lengthscale, nu=nu)
        v = np.linalg.solve(L, K_test_train.T)
        posterior_cov = K_test - v.T @ v
        return posterior_mean, posterior_cov
    else:
        v = np.linalg.solve(L, K_test_train.T)
        posterior_var = 1.0 - np.sum(v ** 2, axis=0)  # K_test_diag = 1 for Matérn self-kernel
        posterior_var = np.maximum(posterior_var, 1e-10)
        return posterior_mean, posterior_var


def _bq_matern_active_sampling(
    embeddings: np.ndarray,
    test_y: np.ndarray,
    u: np.ndarray,
    budget: int,
    n_init: int = 0,
    noise_variance: float = 0.3,
    lengthscale: float = 1.0,
    nu: float = 2.5,
) -> SamplingResult:
    """Active BQ sampling with standalone Matérn kernel on raw embeddings.

    Uses the integral variance-reduction acquisition criterion via full
    posterior covariance (same as encoder Matérn path), but without needing
    a PyTorch encoder.

    Args:
        embeddings: Raw text embeddings ``(n_samples, d)``. Will be normalized.
        test_y: True labels ``(n_samples,)``.
        u: Prior mean ``(n_samples,)``.
        budget: Sampling budget.
        n_init: Random initial samples.
        noise_variance: GP noise variance.
        lengthscale: Matérn lengthscale.
        nu: Matérn smoothness.

    Returns:
        :class:`SamplingResult` with estimates, rounded estimates, and
        integral variance.
    """
    # Normalize embeddings
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    n_samples = emb_norm.shape[0]

    # ALGO-03: Stratified initial sampling by prior mean quantiles
    labeled_indices: List[int] = []
    if n_init > 0:
        sorted_by_prior = np.argsort(u)
        quantile_size = max(1, n_samples // n_init)
        for q in range(n_init):
            start = q * quantile_size
            end = min(start + quantile_size, n_samples)
            if start < end:
                candidates = sorted_by_prior[start:end]
                interesting = [i for i in candidates if 0.15 < u[i] < 0.7]
                pool = interesting if len(interesting) >= 1 else candidates
                labeled_indices.append(int(np.random.choice(pool)))
    unlabeled_indices = [i for i in range(n_samples) if i not in labeled_indices]

    estimates = np.ones(budget) * np.mean(u)
    rounded_estimates = np.ones(budget) * np.mean(np.round(np.clip(u, 0, 1)))
    integral_variance = np.ones(budget) * 1.0

    for t in range(n_init, budget):
        if t == 0:
            u_t = u.copy()
            posterior_cov = _compute_matern_kernel_np(emb_norm, lengthscale=lengthscale, nu=nu)
        else:
            u_t, posterior_cov = _get_posterior_matern(
                emb_norm[labeled_indices], test_y[labeled_indices],
                emb_norm, noise_variance, labeled_indices, u,
                lengthscale=lengthscale, nu=nu, full_cov=True,
            )

        estimates[t] = np.mean(u_t)
        rounded_estimates[t] = np.mean(np.round(np.clip(u_t, 0, 1)))
        integral_variance[t] = np.maximum(
            np.sum(posterior_cov) / (n_samples * n_samples), 0.0
        )

        if t < budget - 1:
            best_idx = _variance_improvement_gp(
                posterior_cov, unlabeled_indices, noise_variance,
            )
            labeled_indices.append(best_idx)
            unlabeled_indices.remove(best_idx)

    # Back-fill init steps
    for t in range(n_init):
        idx_slice = labeled_indices[: t + 1]
        u_t, cov = _get_posterior_matern(
            emb_norm[idx_slice], test_y[idx_slice], emb_norm,
            noise_variance, idx_slice, u,
            lengthscale=lengthscale, nu=nu, full_cov=True,
        )
        estimates[t] = np.mean(u_t)
        rounded_estimates[t] = np.mean(np.round(np.clip(u_t, 0, 1)))
        integral_variance[t] = np.maximum(np.sum(cov) / (n_samples * n_samples), 0.0)

    # Final posterior (marginal)
    if labeled_indices:
        final_u, final_s = _get_posterior_matern(
            emb_norm[labeled_indices], test_y[labeled_indices],
            emb_norm, noise_variance, labeled_indices, u,
            lengthscale=lengthscale, nu=nu, full_cov=False,
        )
    else:
        final_u = u
        final_s = np.ones(n_samples)

    return SamplingResult(
        estimates=estimates,
        rounded_estimates=rounded_estimates,
        selected_indices=labeled_indices,
        posterior_mean=final_u,
        posterior_var=final_s,
        prior_mean=u,
        integral_variance=integral_variance,
    )


def _bq_matern_random_sampling(
    embeddings: np.ndarray,
    test_y: np.ndarray,
    u: np.ndarray,
    budget: int,
    noise_variance: float = 0.3,
    lengthscale: float = 1.0,
    nu: float = 2.5,
) -> SamplingResult:
    """BQ estimation with random point selection and Matérn kernel.

    Selects points uniformly at random but uses Matérn kernel GP
    posterior for estimation. Isolates the value of Matérn BQ estimation
    from active acquisition.

    Args:
        embeddings: Raw text embeddings ``(n_samples, d)``. Will be normalized.
        test_y: True labels ``(n_samples,)``.
        u: Prior mean ``(n_samples,)``.
        budget: Sampling budget.
        noise_variance: GP noise variance.
        lengthscale: Matérn lengthscale.
        nu: Matérn smoothness.

    Returns:
        :class:`SamplingResult`.
    """
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    n_samples = emb_norm.shape[0]

    all_indices = list(np.random.choice(n_samples, min(budget, n_samples), replace=False))

    estimates = np.ones(budget) * np.mean(u)
    integral_variance = np.ones(budget) * 1.0

    for t in range(budget):
        labeled_indices = all_indices[: t + 1]

        if t == 0 and len(labeled_indices) == 0:
            int_var_t = 1.0
        else:
            u_t, cov = _get_posterior_matern(
                emb_norm[labeled_indices], test_y[labeled_indices],
                emb_norm, noise_variance, labeled_indices, u,
                lengthscale=lengthscale, nu=nu, full_cov=True,
            )
            estimates[t] = np.mean(u_t)
            int_var_t = np.maximum(np.sum(cov) / (n_samples * n_samples), 0.0)

        integral_variance[t] = int_var_t

    # Final posterior
    labeled_indices = all_indices[: budget]
    if labeled_indices:
        final_u, final_s = _get_posterior_matern(
            emb_norm[labeled_indices], test_y[labeled_indices],
            emb_norm, noise_variance, labeled_indices, u,
            lengthscale=lengthscale, nu=nu, full_cov=False,
        )
    else:
        final_u = u
        final_s = np.ones(n_samples)

    return SamplingResult(
        estimates=estimates,
        selected_indices=all_indices[: budget],
        posterior_mean=final_u,
        posterior_var=final_s,
        prior_mean=u,
        integral_variance=integral_variance,
    )


# Public API
class BQPriorSampler:
    """Bayesian Quadrature active sampler with learned prior.

    Uses other models' predictions as features for a GP with a linear
    kernel to efficiently estimate a target model's accuracy.

    Args:
        noise_variance: GP observation noise variance.
        n_init: Number of random initial samples before active acquisition.

    Example::

        sampler = BQPriorSampler(noise_variance=0.3)
        result = sampler.sample(predictions="svamp", target_model="gemini25_flash", budget=20)
    """

    def __init__(
        self,
        noise_variance: float = 0.3,
        n_init: int = 0,
    ):
        self.noise_variance = noise_variance
        self.n_init = n_init

    def sample(
        self,
        predictions: Union[str, pd.DataFrame, "Dataset"],  # noqa: F821
        target_model: Union[int, str] = "gemini25_flash",
        budget: int = 50,
        data_dir: str = None,
        pretrain_indices: Optional[List[int]] = None,
        pretrain_mode: str = "gmm",
        reference_benchmarks: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> SamplingResult:
        """Run BQ active sampling.

        Args:
            predictions: A dataset name (e.g. ``"svamp"``) loaded from
                ``data_dir``, a pre-loaded DataFrame, or a
                :class:`~proeval.utils.Dataset` (its ``name`` is used for GMM
                selection and label binarisation).
            target_model: Index or name of the model to target for testing.
            budget: Number of samples to acquire.
            data_dir: Directory containing prediction CSVs (default: ``data/``).
            pretrain_indices: Optional explicit list of model indices to use as
                pre-training features.  If ``None``, behaviour depends on
                *pretrain_mode*.
            pretrain_mode: ``"all"`` (default) uses every model except the target.
                ``"gmm"`` auto-selects models via GMM clustering on reference
                benchmarks.  Ignored when *pretrain_indices* is provided.
            reference_benchmarks: Benchmarks for GMM clustering.  Only used
                when ``pretrain_mode="gmm"``.  ``None`` → auto-selected by
                benchmark category.
            seed: Random seed for reproducibility.

        Returns:
            :class:`SamplingResult` with estimates, selected indices, and posterior.
        """
        if seed is not None:
            np.random.seed(seed)

        # Load data (accepts a dataset name, a DataFrame, or a Dataset)
        df, dataset_name = _resolve_predictions(predictions, data_dir)

        pred_matrix, model_names = extract_model_predictions(df, dataset_name)

        # Resolve target model
        if isinstance(target_model, str):
            if target_model not in model_names:
                raise ValueError(
                    f"Model {target_model!r} not found. Available: {model_names}"
                )
            target_idx = model_names.index(target_model)
        else:
            target_idx = int(target_model)

        # Auto-select pretrain indices via GMM if requested
        if pretrain_indices is None and pretrain_mode == "gmm":
            if dataset_name is None:
                raise ValueError(
                    "pretrain_mode='gmm' requires a dataset name string "
                    "for 'predictions', not a DataFrame."
                )
            from proeval.sampler.pretrain_selector import select_pretrain_models_gmm

            target_name = (
                target_model
                if isinstance(target_model, str)
                else model_names[target_idx]
            )
            pretrain_indices, _ = select_pretrain_models_gmm(
                target_benchmark=dataset_name,
                target_model=target_name,
                data_dir=data_dir,
                reference_benchmarks=reference_benchmarks,
            )

        _, test_x, test_y, u, S = setup_train_test_split(
            pred_matrix, target_idx, pretrain_indices
        )

        return _bq_active_sampling(
            test_x, test_y, u, S,
            budget=budget,
            n_init=self.n_init,
            noise_variance=self.noise_variance,
        )

    # Expose internal helpers for advanced users
    get_posterior = staticmethod(_get_posterior)


# Backward compatibility alias
BQSampler = BQPriorSampler


# Encoder-based BQ sampling (Case 2)
def _bq_encoder_sampling(
    phi_embeddings: np.ndarray,
    test_y: np.ndarray,
    u: np.ndarray,
    noise_variance: float,
    encoder,
    budget: int,
    n_init: int = 0,
    device=None,
) -> SamplingResult:
    """Run the core BQ active sampling loop with encoder kernel.

    Case II BQ method.  Uses the encoder's kernel
    function for GP posterior updates and the proper BQ integral
    variance-reduction acquisition criterion.

    For **linear** kernels the fast ``_variance_improvement`` helper
    (rank-1 update on the inverse) is used.  For **Matérn / RBF** the
    full posterior covariance is maintained and
    ``_variance_improvement_gp`` selects the point that maximally
    reduces the integral posterior variance.

    Returns a :class:`SamplingResult` with both raw and rounded
    estimates.
    """
    import torch
    from proeval.encoder import compute_kernel_matrix
    from proeval.generator.core import get_posterior_embedding

    if device is None:
        device = next(encoder.parameters()).device

    kernel_type = getattr(encoder, "kernel_type", "linear")
    n_samples = phi_embeddings.shape[0]

    # ALGO-03: Stratified initial sampling by prior mean quantiles
    labeled_indices: List[int] = []
    if n_init > 0:
        sorted_by_prior = np.argsort(u)
        quantile_size = max(1, n_samples // n_init)
        for q in range(n_init):
            start = q * quantile_size
            end = min(start + quantile_size, n_samples)
            if start < end:
                candidates = sorted_by_prior[start:end]
                interesting = [i for i in candidates if 0.15 < u[i] < 0.7]
                pool = interesting if len(interesting) >= 1 else candidates
                labeled_indices.append(int(np.random.choice(pool)))
    unlabeled_indices = [i for i in range(n_samples) if i not in labeled_indices]

    estimates = np.ones(budget) * np.mean(u)
    rounded_estimates = np.ones(budget) * np.mean(np.round(np.clip(u, 0, 1)))
    integral_variance = np.ones(budget) * 1.0  # fallback prior integral variance

    use_gp_path = (kernel_type != "linear")

    if use_gp_path:
        # Matérn / RBF path: maintain full posterior covariance
        for t in range(n_init, budget):
            if t == 0:
                u_t = u.copy()
                # Prior covariance from encoder kernel
                phi_t = torch.from_numpy(phi_embeddings).float().to(device)
                posterior_cov = compute_kernel_matrix(
                    phi_t, encoder
                ).detach().cpu().numpy()
            else:
                phi_train = phi_embeddings[labeled_indices]
                u_t, posterior_cov = get_posterior_embedding(
                    phi_train, test_y[labeled_indices],
                    phi_embeddings, noise_variance,
                    labeled_indices, u, encoder, device,
                    full_cov=True,
                )

            estimates[t] = np.mean(u_t)
            rounded_estimates[t] = np.mean(np.round(np.clip(u_t, 0, 1)))
            # Integral variance from full posterior covariance
            integral_variance[t] = np.maximum(np.sum(posterior_cov) / (n_samples * n_samples), 0.0)

            if t < budget - 1:
                best_idx = _variance_improvement_gp(
                    posterior_cov, unlabeled_indices, noise_variance,
                )
                labeled_indices.append(best_idx)
                unlabeled_indices.remove(best_idx)

        # Back-fill initial steps
        for t in range(n_init):
            idx_slice = labeled_indices[: t + 1]
            phi_train = phi_embeddings[idx_slice]
            u_t, full_cov = get_posterior_embedding(
                phi_train, test_y[idx_slice], phi_embeddings,
                noise_variance, idx_slice, u, encoder, device,
                full_cov=True,
            )
            estimates[t] = np.mean(u_t)
            rounded_estimates[t] = np.mean(np.round(np.clip(u_t, 0, 1)))
            integral_variance[t] = np.maximum(np.sum(full_cov) / (n_samples * n_samples), 0.0)

        # Final posterior (marginal variance for result)
        if labeled_indices:
            phi_train = phi_embeddings[labeled_indices]
            final_u, final_s = get_posterior_embedding(
                phi_train, test_y[labeled_indices], phi_embeddings,
                noise_variance, labeled_indices, u, encoder, device,
            )
        else:
            final_u = u
            final_s = np.ones(n_samples)

    else:
        # Linear kernel path: use fast rank-1 update
        test_x = phi_embeddings.T  # (d, m)
        k_t_inv = None
        mu_x = np.mean(test_x, axis=1, keepdims=True)

        # compute prior mean_var properly
        prior_cov_sum = np.dot(np.dot(mu_x.T, np.eye(test_x.shape[0])), mu_x)[0, 0]

        for t in range(n_init, budget):
            train_x = test_x[:, labeled_indices] if labeled_indices else test_x[:, []]
            train_y_arr = test_y[labeled_indices] if labeled_indices else np.array([])

            if t == 0:
                u_t = u.copy()
                int_var_t = prior_cov_sum  # prior variance
            else:
                u_t, s_t = _get_posterior(
                    train_x, train_y_arr, test_x,
                    noise_variance, labeled_indices, u,
                )
                _, s_mu = _get_posterior(
                    train_x, train_y_arr, mu_x,
                    noise_variance, labeled_indices, u,
                )
                int_var_t = s_mu[0]

            estimates[t] = np.mean(u_t)
            rounded_estimates[t] = np.mean(np.round(np.clip(u_t, 0, 1)))
            integral_variance[t] = np.maximum(int_var_t, 0.0)

            if t < budget - 1:
                best_idx, k_t_inv = _variance_improvement(
                    train_x, k_t_inv, noise_variance,
                    unlabeled_indices, test_x,
                )
                labeled_indices.append(best_idx)
                unlabeled_indices.remove(best_idx)

        # Back-fill initial steps
        for t in range(n_init):
            idx_slice = labeled_indices[: t + 1]
            u_t, s_t = _get_posterior(
                test_x[:, idx_slice], test_y[idx_slice],
                test_x, noise_variance, idx_slice, u,
            )
            _, s_mu = _get_posterior(
                test_x[:, idx_slice], test_y[idx_slice],
                mu_x, noise_variance, idx_slice, u,
            )
            estimates[t] = np.mean(u_t)
            rounded_estimates[t] = np.mean(np.round(np.clip(u_t, 0, 1)))
            integral_variance[t] = np.maximum(s_mu[0], 0.0)

        # Final posterior
        if labeled_indices:
            final_u, final_s = _get_posterior(
                test_x[:, labeled_indices], test_y[labeled_indices],
                test_x, noise_variance, labeled_indices, u,
            )
        else:
            final_u = u
            final_s = np.ones(n_samples)

    return SamplingResult(
        estimates=estimates,
        rounded_estimates=rounded_estimates,
        selected_indices=labeled_indices,
        posterior_mean=final_u,
        posterior_var=final_s,
        prior_mean=u,
        integral_variance=integral_variance,
    )


class BQEncoderSampler:
    """Bayesian Quadrature active sampler with encoder-based prior (Case 2).

    Uses a pre-trained neural encoder's phi embeddings and kernel for GP
    posterior updates, instead of model predictions.

    Args:
        encoder_path: Path to trained encoder ``.pth`` file.
        embeddings_path: Path to question embeddings ``.npy`` file.
        noise_variance: GP noise variance (default: uses encoder's learned var).
        n_init: Number of random initial samples (default 0).

    Example::

        sampler = BQEncoderSampler(
            encoder_path="encoder.pth",
            embeddings_path="embeddings.npy",
        )
        result = sampler.sample(
            predictions="gsm8k", target_model="gemma3_27b", budget=50,
        )
        print(result.mae(true_mean))
    """

    def __init__(
        self,
        encoder_path: str,
        embeddings_path: str,
        noise_variance: Optional[float] = None,
        n_init: int = 0,
    ):
        self.encoder_path = encoder_path
        self.embeddings_path = embeddings_path
        self._noise_variance_override = noise_variance
        self.n_init = n_init

        # Load encoder and compute prior
        from proeval.generator.core import setup_encoder_prior

        encoder, phi_emb, u, S, var = setup_encoder_prior(
            encoder_path, embeddings_path,
        )
        self.encoder = encoder
        self.phi_embeddings = phi_emb
        self.prior_u = u
        self.prior_S = S
        self.noise_variance = noise_variance if noise_variance is not None else var

    def sample(
        self,
        predictions: Union[str, pd.DataFrame, "Dataset"],  # noqa: F821
        target_model: Union[int, str] = "gemini25_flash",
        budget: int = 50,
        data_dir: str = None,
        seed: Optional[int] = None,
    ) -> SamplingResult:
        """Run BQ active sampling with encoder prior.

        Args:
            predictions: Dataset name, pre-loaded DataFrame, or a
                :class:`~proeval.utils.Dataset`.
            target_model: Index or name of the target model.
            budget: Number of samples to acquire.
            data_dir: Data directory path.
            seed: Random seed.

        Returns:
            :class:`SamplingResult` with estimates, indices, and posterior.
        """
        if seed is not None:
            np.random.seed(seed)

        # Load data to get test_y (accepts a name, DataFrame, or Dataset)
        df, dataset_name = _resolve_predictions(predictions, data_dir)

        pred_matrix, model_names = extract_model_predictions(df, dataset_name)

        # Resolve target model
        if isinstance(target_model, str):
            if target_model not in model_names:
                raise ValueError(
                    f"Model {target_model!r} not found. Available: {model_names}"
                )
            target_idx = model_names.index(target_model)
        else:
            target_idx = int(target_model)

        test_y = pred_matrix[:, target_idx]

        return _bq_encoder_sampling(
            self.phi_embeddings,
            test_y,
            self.prior_u,
            self.noise_variance,
            self.encoder,
            budget=budget,
            n_init=self.n_init,
        )


def _bq_encoder_random_sampling(
    phi_embeddings: np.ndarray,
    test_y: np.ndarray,
    u: np.ndarray,
    noise_variance: float,
    encoder,
    budget: int,
    device=None,
) -> SamplingResult:
    """BQ estimation with random selection and encoder Matérn kernel.

    Selects points uniformly at random but uses the encoder's kernel
    (Matérn/RBF/linear) for GP posterior updates.  This isolates the
    value of the encoder's kernel from active acquisition.

    Returns a :class:`SamplingResult` identical in structure to
    :func:`_bq_encoder_sampling`.
    """
    import torch
    from proeval.encoder import compute_kernel_matrix
    from proeval.generator.core import get_posterior_embedding

    if device is None:
        device = next(encoder.parameters()).device

    kernel_type = getattr(encoder, "kernel_type", "linear")
    n_samples = phi_embeddings.shape[0]

    # Random selection up front
    all_indices = list(np.random.choice(n_samples, min(budget, n_samples), replace=False))

    estimates = np.ones(budget) * np.mean(u)
    integral_variance = np.ones(budget) * 1.0

    use_gp_path = (kernel_type != "linear")

    if use_gp_path:
        for t in range(budget):
            labeled_indices = all_indices[: t + 1]

            if t == 0 and len(labeled_indices) == 0:
                int_var_t = 1.0
                u_t = u.copy()
            else:
                phi_train = phi_embeddings[labeled_indices]
                u_t, posterior_cov = get_posterior_embedding(
                    phi_train, test_y[labeled_indices],
                    phi_embeddings, noise_variance,
                    labeled_indices, u, encoder, device,
                    full_cov=True,
                )
                int_var_t = np.maximum(
                    np.sum(posterior_cov) / (n_samples * n_samples), 0.0
                )

            estimates[t] = np.mean(u_t)
            integral_variance[t] = int_var_t

        # Final posterior (marginal)
        labeled_indices = all_indices[: budget]
        if labeled_indices:
            phi_train = phi_embeddings[labeled_indices]
            final_u, final_s = get_posterior_embedding(
                phi_train, test_y[labeled_indices], phi_embeddings,
                noise_variance, labeled_indices, u, encoder, device,
            )
        else:
            final_u = u
            final_s = np.ones(n_samples)

    else:
        # Linear kernel path
        test_x = phi_embeddings.T
        mu_x = np.mean(test_x, axis=1, keepdims=True)
        prior_cov_sum = np.dot(np.dot(mu_x.T, np.eye(test_x.shape[0])), mu_x)[0, 0]

        for t in range(budget):
            labeled_indices = all_indices[: t + 1]
            train_x = test_x[:, labeled_indices]
            train_y_arr = test_y[labeled_indices]

            if t == 0 and len(labeled_indices) == 0:
                int_var_t = prior_cov_sum
            else:
                u_t, _ = _get_posterior(
                    train_x, train_y_arr, test_x,
                    noise_variance, labeled_indices, u,
                )
                _, s_mu = _get_posterior(
                    train_x, train_y_arr, mu_x,
                    noise_variance, labeled_indices, u,
                )
                estimates[t] = np.mean(u_t)
                int_var_t = s_mu[0]

            integral_variance[t] = np.maximum(int_var_t, 0.0)

        labeled_indices = all_indices[: budget]
        if labeled_indices:
            final_u, final_s = _get_posterior(
                test_x[:, labeled_indices], test_y[labeled_indices],
                test_x, noise_variance, labeled_indices, u,
            )
        else:
            final_u = u
            final_s = np.ones(n_samples)

    return SamplingResult(
        estimates=estimates,
        selected_indices=all_indices[: budget],
        posterior_mean=final_u,
        posterior_var=final_s,
        prior_mean=u,
        integral_variance=integral_variance,
    )
