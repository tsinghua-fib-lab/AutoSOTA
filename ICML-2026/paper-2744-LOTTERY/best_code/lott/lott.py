"""
LoTT: Learning from Reference-Only Samples in Two-Sample Testing under Size Asymmetry.

Core implementation of the LoTT framework (Section 4) and uncertainty-guided
selection/weighting mechanism (Section 5).

Two variants:
  - LoTT:              Equal-weight aggregation across a fixed set of RDRs.
  - LoTTWithSelection:  Uncertainty-guided weighting with optional landmark RDRs.
"""

import numpy as np
import torch
from typing import Dict

from .rdr import (
    ME_RDR,
    MultiScaleME_RDR,
    LOF_RDR,
    KNN_RDR,
    Mahalanobis_RDR,
    LandmarkRDR,
)


class LoTT:
    """
    LoTT with equal-weight aggregation over a fixed RDR family.

    The test statistic aggregates standardized RDR scores:
      T(Y) = sum_f T_f(Y),  where T_f(Y) = (mean_j a_f(y_j))^2
    and a_f is the standardized score (Section 4.3).

    Permutation testing pools X^hold and Y for valid Type I error control
    (Section 4.4, Theorem 6.1).
    """

    def __init__(self, alpha: float = 0.05, n_permutations: int = 500):
        self.alpha = alpha
        self.n_permutations = n_permutations

    def fit(self, X_train: torch.Tensor, X_calib: torch.Tensor, X_hold: torch.Tensor,
                 k_knn: int = 5, k_lof: int = 20, use_multiscale_me: bool = False,
                 multiscale_bw_scales=None, **kwargs):
        self.device = X_train.device
        base_bw = torch.cdist(X_train, X_train).median().item()

        if use_multiscale_me:
            me_rdr = MultiScaleME_RDR(
                X_train, bandwidth=base_bw,
                scales=multiscale_bw_scales
            )
        else:
            me_rdr = ME_RDR(X_train, bandwidth=base_bw)

        self.rdrs = [
            me_rdr,
            LOF_RDR(X_train, k=k_lof),
            KNN_RDR(X_train, k=k_knn),
            Mahalanobis_RDR(X_train),
        ]
        self.K = len(self.rdrs)

        calib_scores = self._compute_scores(X_calib)
        self.mu = calib_scores.mean(axis=0)
        self.std = calib_scores.std(axis=0) + 1e-10
        self.signs = np.array([1, 1, 1, 1])

        self.hold_scores = self._compute_scores(X_hold)

    def _compute_scores(self, X: torch.Tensor) -> np.ndarray:
        scores = []
        for rdr in self.rdrs:
            scores.append(rdr(X).detach().cpu().numpy())
        return np.stack(scores, axis=1)

    def _compute_stat(self, scores: np.ndarray) -> float:
        standardized = (scores - self.mu) / self.std * self.signs
        return standardized.sum(axis=1).mean()

    def test(self, Y: torch.Tensor) -> Dict:
        m = len(Y)
        y_scores = self._compute_scores(Y)
        test_stat = self._compute_stat(y_scores)

        pooled = np.vstack([self.hold_scores, y_scores])
        n_total = len(pooled)

        null_stats = []
        for _ in range(self.n_permutations):
            perm = np.random.permutation(n_total)
            pseudo_Y = pooled[perm[-m:]]
            null_stats.append(self._compute_stat(pseudo_Y))

        null_stats = np.array(null_stats)
        p_value = (np.sum(null_stats >= test_stat) + 1) / (self.n_permutations + 1)

        return {
            'reject': p_value < self.alpha,
            'p_value': p_value,
            'statistic': test_stat
        }


class LoTTWithSelection:
    """
    LoTT with uncertainty-guided RDR selection and weighting.

    Extends LoTT by:
      1. Adding M landmark RDRs to the base family (Section 4.2)
      2. Estimating uncertainty (variance) of each RDR via resampling (Section 5)
      3. Weighting RDRs to downweight unstable ones

    Selection methods:
      - 'precision_weight': w_f = 1/variance  (stability only)
      - 'sensitivity_weight': w_f = delta_f / variance  (stability + sensitivity,
        Section 5 combined scheme; guards against degenerate constant RDRs)
      - 'top_n': select n lowest-variance RDRs with equal weight
      - 'threshold': select RDRs below a variance threshold

    Statistic formulations (ALGO-06):
      - 'mean_of_squares': T = sum_f w_f * mean_i(z_if^2)
      - 'square_of_mean': T = sum_f w_f * (mean_i(z_if))^2
      - 'hybrid': 0.5 * mean_of_squares + 0.5 * square_of_mean

    The weighted statistic T_weight(Y) = sum_f w_f * T_f(Y) reduces null
    variability and tightens permutation thresholds (Theorem 6.2).
    """

    RDR_NAMES = ['ME', 'LOF', 'KNN', 'Mahalanobis']

    def __init__(self, alpha: float = 0.05, n_permutations: int = 500,
                 selection_method: str = 'top_n', n_select: int = 2,
                 variance_threshold: float = None,
                 perturbation_scale: float = 0.01,
                 statistic_formulation: str = 'mean_of_squares',
                 verbose: bool = True):
        self.alpha = alpha
        self.n_permutations = n_permutations
        self.selection_method = selection_method
        self.n_select = n_select
        self.variance_threshold = variance_threshold
        self.perturbation_scale = perturbation_scale
        self.statistic_formulation = statistic_formulation
        self.verbose = verbose

    def fit(self, X_train: torch.Tensor, X_calib: torch.Tensor, X_hold: torch.Tensor,
            M: int = 10, subset_size: int = 10, k_knn: int = 5, k_lof: int = 20,
            use_multiscale_me: bool = False, multiscale_bw_scales=None, **kwargs):
        """
        Fit LoTT with RDR selection.

        Steps:
          1. Construct all RDRs (base + M landmark RDRs)
          2. Estimate uncertainty of each RDR via resampling on X^cal
          3. Select/weight RDRs based on stability
          4. Prepare holdout scores for permutation testing
        """
        self.device = X_train.device
        base_bw = torch.cdist(X_train, X_train).median().item()
        self.M = M
        self.subset_size = subset_size

        landmark_rdr = LandmarkRDR(X_train, M=M, subset_size=subset_size, bandwidth=base_bw)
        landmark_rdrs = landmark_rdr.get_individual_rdrs()

        # ALGO-01: Optionally use multi-scale ME-RDR
        if use_multiscale_me:
            me_rdr = MultiScaleME_RDR(
                X_train, bandwidth=base_bw,
                scales=multiscale_bw_scales
            )
            base_names = ['MultiScaleME', 'LOF', 'KNN', 'Mahalanobis']
        else:
            me_rdr = ME_RDR(X_train, bandwidth=base_bw)
            base_names = ['ME', 'LOF', 'KNN', 'Mahalanobis']

        self.all_rdrs = [
            me_rdr,
            LOF_RDR(X_train, k=k_lof),
            KNN_RDR(X_train, k=k_knn),
            Mahalanobis_RDR(X_train),
        ] + landmark_rdrs

        landmark_names = [f'Landmark_{l}' for l in range(M)]
        self.RDR_NAMES = base_names + landmark_names

        self.all_signs = np.ones(len(self.all_rdrs))

        all_calib_scores = self._compute_all_scores(X_calib)
        self.rdr_variances = all_calib_scores.var(axis=0)

        # Compute sensitivity if needed: measures how much each RDR responds
        # to a small perturbation of the calibration set (Section 5).
        # Guards against degenerate (constant) RDRs that have low variance
        # but carry no information about distributional differences.
        self.rdr_sensitivities = None
        if self.selection_method == 'sensitivity_weight':
            self.rdr_sensitivities = self._compute_sensitivity(X_calib)

        self._select_rdrs(all_calib_scores)

        calib_scores = self._compute_scores(X_calib)
        self.mu = calib_scores.mean(axis=0)
        self.std = calib_scores.std(axis=0) + 1e-10

        self.hold_scores = self._compute_scores(X_hold)

        if self.verbose:
            self._print_selection_info()

    def _compute_sensitivity(self, X_calib: torch.Tensor) -> np.ndarray:
        """
        Compute sensitivity of each RDR to a small perturbation (Section 5).

        Constructs X_cal_tilde by adding mild Gaussian noise to X^cal, then:
          delta_f = |mean(a_f(X_cal_tilde)) - mean(a_f(X_cal))| / n_cal

        RDRs with delta_f close to zero are largely insensitive and unlikely
        to contribute power. This prevents degenerate constant-output RDRs
        from dominating under precision weighting alone.
        """
        noise = torch.randn_like(X_calib) * self.perturbation_scale
        X_calib_perturbed = X_calib + noise

        sensitivities = np.zeros(len(self.all_rdrs))
        n_cal = len(X_calib)
        for i, rdr in enumerate(self.all_rdrs):
            scores_clean = rdr(X_calib).detach().cpu().numpy()
            scores_perturbed = rdr(X_calib_perturbed).detach().cpu().numpy()
            sensitivities[i] = np.abs(scores_perturbed.mean() - scores_clean.mean()) / n_cal
        return sensitivities

    def _select_rdrs(self, calib_scores: np.ndarray):
        variances = self.rdr_variances
        n_rdrs = len(self.all_rdrs)

        if self.selection_method == 'top_n':
            n_select = min(self.n_select, n_rdrs)
            self.selected_indices = np.argsort(variances)[:n_select]
            self.weights = np.ones(n_select)

        elif self.selection_method == 'precision_weight':
            self.selected_indices = np.arange(n_rdrs)
            precisions = 1.0 / (variances + 1e-10)
            self.weights = precisions / precisions.sum() * n_rdrs

        elif self.selection_method == 'sensitivity_weight':
            # Combined weighting: w_f = delta_f / sigma_f^2 (Section 5)
            # Favors RDRs that are both stable (small variance) and responsive
            # (large sensitivity). Guards against degenerate constant RDRs.
            self.selected_indices = np.arange(n_rdrs)
            sensitivities = self.rdr_sensitivities
            raw_weights = sensitivities / (variances + 1e-10)
            self.weights = raw_weights / (raw_weights.sum() + 1e-10) * n_rdrs

        elif self.selection_method == 'threshold':
            if self.variance_threshold is None:
                self.variance_threshold = np.median(variances)
            self.selected_indices = np.where(variances <= self.variance_threshold)[0]
            if len(self.selected_indices) == 0:
                self.selected_indices = np.array([np.argmin(variances)])
            self.weights = np.ones(len(self.selected_indices))
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")

        self.rdrs = [self.all_rdrs[i] for i in self.selected_indices]
        self.signs = self.all_signs[self.selected_indices]
        self.K = len(self.rdrs)

    def _print_selection_info(self):
        print("\n" + "="*60)
        print("RDR SELECTION SUMMARY")
        print("="*60)
        print(f"Selection method: {self.selection_method}")
        print("\nAll RDR stats on calibration set:")
        for i, (name, var) in enumerate(zip(self.RDR_NAMES, self.rdr_variances)):
            selected = "*" if i in self.selected_indices else " "
            sens_str = ""
            if self.rdr_sensitivities is not None:
                sens_str = f", sensitivity = {self.rdr_sensitivities[i]:.6f}"
            print(f"  [{selected}] {name}: variance = {var:.6f}{sens_str}")

        show_weights = self.selection_method in ('precision_weight', 'sensitivity_weight')
        print(f"\nSelected {len(self.selected_indices)} RDRs:")
        for idx, w_idx in enumerate(self.selected_indices):
            weight_str = f", weight={self.weights[idx]:.3f}" if show_weights else ""
            print(f"  - {self.RDR_NAMES[w_idx]} (variance={self.rdr_variances[w_idx]:.6f}{weight_str})")
        print("="*60 + "\n")

    def _compute_all_scores(self, X: torch.Tensor) -> np.ndarray:
        """Estimate RDR uncertainty via resampling on calibration set."""
        scores = []
        for rdr in self.all_rdrs:
            resampled = []
            for _ in range(20):
                n = len(X)
                idx = np.random.choice(n, size=n//2, replace=False)
                X_sample = X[idx]
                resampled.append(rdr(X_sample).detach().cpu().numpy().mean(axis=0))
            scores.append(np.array(resampled))
        return np.stack(scores, axis=1)

    def _compute_scores(self, X: torch.Tensor) -> np.ndarray:
        scores = []
        for rdr in self.rdrs:
            scores.append(rdr(X).detach().cpu().numpy())
        return np.stack(scores, axis=1)

    def _compute_stat(self, scores: np.ndarray) -> float:
        """
        Weighted aggregated test statistic (ALGO-06: multiple formulations).

        For each RDR k:
          1. Standardize: z_ik = (score_ik - mu_k) / std_k
          2. Compute per-RDR statistic:
             - mean_of_squares: MSS_k = (1/m) sum_i z_ik^2
             - square_of_mean: SOM_k = ((1/m) sum_i z_ik)^2
             - hybrid: 0.5 * MSS_k + 0.5 * SOM_k
          3. Weighted sum: T = sum_k w_k * stat_k
        """
        standardized = (scores - self.mu) / self.std * self.signs

        if self.statistic_formulation == 'square_of_mean':
            # Square of per-RDR mean: sensitive to consistent directional deviations
            per_rdr_stats = (standardized.mean(axis=0)) ** 2
        elif self.statistic_formulation == 'hybrid':
            # Hybrid: balances both types of deviation
            mean_of_squares = (standardized ** 2).mean(axis=0)
            square_of_mean = (standardized.mean(axis=0)) ** 2
            per_rdr_stats = 0.5 * mean_of_squares + 0.5 * square_of_mean
        else:
            # mean_of_squares (default): sensitive to any deviation magnitude
            per_rdr_stats = (standardized ** 2).mean(axis=0)

        return (self.weights * per_rdr_stats).sum()

    def test(self, Y: torch.Tensor) -> Dict:
        m = len(Y)
        y_scores = self._compute_scores(Y)
        test_stat = self._compute_stat(y_scores)

        pooled = np.vstack([self.hold_scores, y_scores])
        n_total = len(pooled)

        null_stats = []
        for _ in range(self.n_permutations):
            perm = np.random.permutation(n_total)
            pseudo_Y = pooled[perm[-m:]]
            null_stats.append(self._compute_stat(pseudo_Y))

        null_stats = np.array(null_stats)
        p_value = (np.sum(null_stats >= test_stat) + 1) / (self.n_permutations + 1)

        return {
            'reject': p_value < self.alpha,
            'p_value': p_value,
            'statistic': test_stat,
            'selected_rdrs': [self.RDR_NAMES[i] for i in self.selected_indices],
            'rdr_variances': dict(zip(self.RDR_NAMES, self.rdr_variances))
        }

    def get_selection_stats(self) -> Dict:
        stats = {
            'method': self.selection_method,
            'all_variances': dict(zip(self.RDR_NAMES, self.rdr_variances)),
            'selected_indices': self.selected_indices.tolist(),
            'selected_names': [self.RDR_NAMES[i] for i in self.selected_indices],
            'weights': self.weights.tolist(),
            'variance_ranking': [self.RDR_NAMES[i] for i in np.argsort(self.rdr_variances)]
        }
        if self.rdr_sensitivities is not None:
            stats['all_sensitivities'] = dict(zip(self.RDR_NAMES, self.rdr_sensitivities))
        return stats
