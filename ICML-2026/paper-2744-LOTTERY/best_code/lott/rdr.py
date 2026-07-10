"""
Reference-Dependent Representations (RDRs) for LoTT.

Each RDR is a function f: X -> R learned exclusively from reference samples X^tr,
such that larger values of f(y) indicate lower compatibility with the reference
distribution P. See Section 4.2 of the paper for details.

RDR families:
  - ME-RDR:          Kernel Mean Embedding distance (consistent, guarantees power -> 1)
  - MultiScale-ME:   Multi-scale KME for better finite-sample power (ALGO-01)
  - Mahalanobis-RDR: Mahalanobis distance for global structure
  - kNN-RDR:         k-nearest-neighbor distance for local scale geometry
  - LOF-RDR:         Local Outlier Factor for local relative density
  - Landmark-RDR:    Subset kernel similarity for localized detection
"""

import numpy as np
import torch
from typing import Optional, List


class ME_RDR:
    """
    Kernel Mean Embedding RDR for local similarity patterns.

    Measures squared distance to the kernel mean embedding of the reference
    distribution. Uses Gaussian RBF kernel (characteristic), guaranteeing
    consistency: power -> 1 as sample sizes grow (Remark 4.2 in paper).

    f(x) = k(x,x) - 2 * mean_z k(x,z) + mean_{z,z'} k(z,z')
    """

    def __init__(self, X_ref: torch.Tensor, bandwidth: float = None):
        self.X_ref = X_ref
        self.bandwidth = bandwidth or self._median_heuristic(X_ref)
        self.device = X_ref.device

        K_xx = self._compute_kernel(X_ref, X_ref)
        self.mean_K_xx = K_xx.mean().item()

    def _median_heuristic(self, X: torch.Tensor) -> float:
        with torch.no_grad():
            dists = torch.cdist(X, X)
            return torch.median(dists[dists > 0]).item()

    def _compute_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(X, Y)
        return torch.exp(-dists**2 / (2 * self.bandwidth**2))

    def __call__(self, Z: torch.Tensor) -> torch.Tensor:
        K_zx = self._compute_kernel(Z, self.X_ref)
        mean_K_zx = K_zx.mean(dim=1)
        return 1 - 2 * mean_K_zx + self.mean_K_xx


class MultiScaleME_RDR:
    """
    Multi-Scale Kernel Mean Embedding RDR (ALGO-01).

    Aggregates KME distances over multiple bandwidth scales instead of a single
    median-heuristic bandwidth. Inspired by MMDAgg (Schrab et al., JMLR 2023),
    which proved minimax-adaptivity over Sobolev balls using multi-bandwidth
    aggregation.

    Constructs M scales centered on the median heuristic:
      bandwidths = [base_bw * 2**i for i in range(-n_scales//2+1, n_scales//2+1)]

    Aggregation methods:
      - 'mean': arithmetic mean of per-scale KME distances
      - 'softmax': softmax-weighted mean with temperature tau

    The multi-scale formulation preserves consistency while improving
    finite-sample power for distribution differences at diverse scales.
    """

    def __init__(self, X_ref: torch.Tensor, bandwidth: float = None,
                 n_scales: int = 5, aggregation: str = 'softmax',
                 tau: float = 1.0, scales: Optional[List[float]] = None):
        self.X_ref = X_ref
        self.device = X_ref.device
        self.aggregation = aggregation
        self.tau = tau

        base_bw = bandwidth or self._median_heuristic(X_ref)

        if scales is not None:
            self.bandwidths = [max(base_bw * s, 0.01) for s in scales]
        else:
            self.bandwidths = [max(base_bw * (2**i), 0.01)
                              for i in range(-(n_scales//2), n_scales//2 + 1)]
        self.n_scales = len(self.bandwidths)

        # Precompute per-scale kernel statistics
        self.mean_K_xx = []
        for bw in self.bandwidths:
            K_xx = self._compute_kernel(X_ref, X_ref, bw)
            self.mean_K_xx.append(K_xx.mean().item())

    def _median_heuristic(self, X: torch.Tensor) -> float:
        with torch.no_grad():
            dists = torch.cdist(X, X)
            d = dists[dists > 0]
            if d.numel() == 0:
                return 1.0
            return torch.median(d).item()

    def _compute_kernel(self, X: torch.Tensor, Y: torch.Tensor, bandwidth: float) -> torch.Tensor:
        dists = torch.cdist(X, Y)
        return torch.exp(-dists**2 / (2 * bandwidth**2))

    def __call__(self, Z: torch.Tensor) -> torch.Tensor:
        per_scale_scores = []
        for i, bw in enumerate(self.bandwidths):
            K_zx = self._compute_kernel(Z, self.X_ref, bw)
            mean_K_zx = K_zx.mean(dim=1)
            score = 1 - 2 * mean_K_zx + self.mean_K_xx[i]
            per_scale_scores.append(score)

        stacked = torch.stack(per_scale_scores, dim=1)  # (m, n_scales)

        if self.aggregation == 'softmax':
            # Softmax weighting: emphasize scales with stronger signal
            mean_per_scale = stacked.mean(dim=0)
            weights = torch.softmax(mean_per_scale / self.tau, dim=0)
            return (stacked * weights).sum(dim=1)
        else:
            # Arithmetic mean
            return stacked.mean(dim=1)


class Mahalanobis_RDR:
    """
    Mahalanobis-RDR for global structure.

    Measures Mahalanobis distance from the reference distribution's center:
      f_Mah(x) = (x - mu)^T Sigma^{-1} (x - mu)
    Effective for detecting global mean and covariance shifts.
    """

    def __init__(self, X_ref: torch.Tensor):
        self.mu = X_ref.mean(dim=0)
        n, d = X_ref.shape
        centered = X_ref - self.mu
        cov = (centered.T @ centered) / (n - 1)
        cov = cov + 1e-6 * torch.eye(d, device=X_ref.device)
        self.cov_inv = torch.linalg.pinv(cov)

    def __call__(self, Z: torch.Tensor) -> torch.Tensor:
        diff = Z - self.mu
        return (diff @ self.cov_inv * diff).sum(dim=1)


class KNN_RDR:
    """
    kNN-RDR for local scale geometry.

    Computes mean distance to k nearest neighbors in X^tr:
      f_kNN(x) = sum_{i=1}^k d_{(i)}(x) / k
    Sensitive to local sparsity/density changes and manifold-type alternatives.
    """

    def __init__(self, X_ref: torch.Tensor, k: int = 10):
        self.X_ref = X_ref
        self.k = min(k, len(X_ref))

    def __call__(self, Z: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(Z, self.X_ref)
        knn_dists, _ = torch.topk(dists, self.k, largest=False, dim=1)
        return knn_dists.mean(dim=1)


class LOF_RDR:
    """
    LOF-RDR for local relative density.

    Based on Local Outlier Factor (Breunig et al., 2000):
      f_LOF(x) = median_{y in N_k(x)} lrd(y) / lrd(x)
    Captures local relative density irregularities, detecting how outlying
    x is with respect to its surrounding region.
    """

    def __init__(self, X_ref: torch.Tensor, k: int = 30):
        self.X_ref = X_ref
        self.k = min(k, len(X_ref) - 1)
        self._compute_reference_lrd()

    def _compute_reference_lrd(self):
        dists = torch.cdist(self.X_ref, self.X_ref)
        knn_dists, knn_idx = torch.topk(dists, self.k + 1, largest=False, dim=1)
        self.k_dist_ref = knn_dists[:, -1]
        reach_dists = torch.maximum(knn_dists[:, 1:], self.k_dist_ref[knn_idx[:, 1:]])
        self.lrd_ref = 1.0 / (reach_dists.median(dim=1).values + 1e-10)

    def __call__(self, Z: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(Z, self.X_ref)
        knn_dists, knn_idx = torch.topk(dists, self.k, largest=False, dim=1)
        reach_dists = torch.maximum(knn_dists, self.k_dist_ref[knn_idx])
        lrd_z = 1.0 / (reach_dists.median(dim=1).values + 1e-10)
        neighbor_lrd = self.lrd_ref[knn_idx]
        return neighbor_lrd.median(dim=1).values / (lrd_z + 1e-10)


class SubsetKernel_RDR:
    """
    Individual member of the ME-RDR family with a specific test location set.

    Implements f_l(x) = mean_{v in S_l} kappa(x, v) for a fixed subset S_l
    of reference points (Remark 4.2 in the paper). Each instance corresponds
    to one test location set, and multiple instances collectively expand the
    ME-RDR family to probe different local regions of the reference distribution.
    """

    def __init__(self, subset: torch.Tensor, bandwidth: float, device: torch.device):
        self.subset = subset
        self.bandwidth = bandwidth
        self.device = device

    def __call__(self, Z: torch.Tensor) -> torch.Tensor:
        if Z.device != self.device:
            Z = Z.to(self.device)
        dists = torch.cdist(Z, self.subset)
        K = torch.exp(-dists**2 / (2.0 * self.bandwidth**2))
        return K.mean(dim=1)


class LandmarkRDR:
    """
    Expands the ME-RDR family with M additional test location sets (Remark 4.2).

    The paper defines ME-RDR as f_l(x) = kappa(x, v_l) for test locations
    {v_l} subset of X^tr. This class creates M random subsets of X^tr as
    test location sets, each producing an independent ME-RDR member. Using
    multiple subsets probes different local regions of P, improving detection
    of localized departures. The uncertainty-guided weighting (Section 5)
    then selects which of these expanded RDRs contribute to the final test.
    """

    def __init__(
        self,
        X_ref: torch.Tensor,
        M: int = 10,
        subset_size: int = 100,
        bandwidth: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        if X_ref.ndim != 2:
            raise ValueError(f"X_ref must be 2D (n,d). Got shape={tuple(X_ref.shape)}")
        n = X_ref.shape[0]
        if not (1 <= subset_size <= n):
            raise ValueError(f"subset_size must be in [1, n] where n={n}. Got {subset_size}")

        self.X_ref = X_ref
        self.device = X_ref.device
        self.M = M
        self.subset_size = subset_size
        self.bandwidth = float(bandwidth) if bandwidth is not None else self._median_heuristic(X_ref)

        self.subsets = self._create_subsets(X_ref, M, self.subset_size, seed)

    def _create_subsets(self, X_ref, M, subset_size, seed):
        n = X_ref.shape[0]
        subsets = []
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        for _ in range(M):
            indices = np.random.choice(n, size=subset_size, replace=False)
            subsets.append(X_ref[indices])
        return subsets

    def _median_heuristic(self, X: torch.Tensor) -> float:
        with torch.no_grad():
            dists = torch.cdist(X, X)
            d = dists[dists > 0]
            if d.numel() == 0:
                return 1.0
            return torch.median(d).item()

    def _compute_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(X, Y)
        return torch.exp(-dists**2 / (2.0 * self.bandwidth**2))

    def __call__(self, Z: torch.Tensor) -> torch.Tensor:
        if Z.ndim != 2:
            raise ValueError(f"Z must be 2D (m,d). Got shape={tuple(Z.shape)}")
        if Z.device != self.device:
            Z = Z.to(self.device)
        all_scores = []
        for subset in self.subsets:
            K = self._compute_kernel(Z, subset)
            scores = K.mean(dim=1)
            all_scores.append(scores)
        return torch.stack(all_scores, dim=1)

    def get_individual_rdrs(self) -> list:
        """Return M individual SubsetKernel_RDR objects for use in LoTTWithSelection."""
        rdrs = []
        for l in range(self.M):
            rdr = SubsetKernel_RDR(
                subset=self.subsets[l],
                bandwidth=self.bandwidth,
                device=self.device
            )
            rdrs.append(rdr)
        return rdrs
