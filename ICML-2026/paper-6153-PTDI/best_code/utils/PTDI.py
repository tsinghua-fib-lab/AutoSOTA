
from typing import Any, Dict, List, Optional
import numpy as np


def BH_matrix_multidim_optimized(calibration_scores, test_scores, q_vector, scale_ratio = 1):
    """
    Optimized BH procedure that avoids creating giant intermediate arrays and vectorizes result assignment.

    Parameters:
    calibration_scores: numpy array of shape (m, ncalib)
    test_scores: numpy array of shape (m, ntest)
    q_vector: numpy array of shape (k,) - vector of false discovery rate thresholds
    scale_ratio: scalar or numpy array of shape (m,)
    
    Returns:
    result: numpy array of shape (m, k, ntest)
    """
    m, ncalib = calibration_scores.shape
    _, ntest = test_scores.shape
    k = q_vector.shape[0]

    # --- Step 1: Calculate p-values iteratively to save memory ---
    pvals = np.zeros((m, ntest))
    
    if np.isscalar(scale_ratio):
        scale_factor = scale_ratio
    else:
        scale_factor = np.asarray(scale_ratio).reshape(m, 1)

    for i in range(m):
        calib_row = calibration_scores[i, :].reshape(-1, 1)
        test_row = test_scores[i, :].reshape(1, -1)
        
        less_than = np.sum(calib_row < test_row, axis=0)
        equal_to = np.sum(calib_row == test_row, axis=0)
        
        pvals[i, :] = (less_than + np.random.uniform(size=ntest) * (equal_to + 1)) / (ncalib + 1)
        
    pvals *= scale_factor

    # --- Step 2: Perform the BH procedure in a fully vectorized way ---
    sorted_indices = np.argsort(pvals, axis=1)
    sorted_pvals = pvals[np.arange(m)[:, None], sorted_indices]
    
    j = np.arange(1, ntest + 1)
    thresholds = q_vector[:, None] * (j / ntest)
    
    is_significant = sorted_pvals[:, None, :] <= thresholds[None, :, :]
    cummax = np.maximum.accumulate(is_significant[:, :, ::-1], axis=2)[:, :, ::-1]
    
    result = np.zeros((m, k, ntest), dtype=np.int8)

    # --- Step 3: Use advanced indexing to set significant results without loops ---
    m_indices, q_indices, sorted_j_indices = np.where(cummax)
    original_col_indices = sorted_indices[m_indices, sorted_j_indices]
    result[m_indices, q_indices, original_col_indices] = 1
    
    return result

class SelectivePrediction:
    
    def __init__(self, evaluation_calculator=None):
        """
        Initialize the SelectivePrediction.
        
        Args:
            evaluation_calculator: An instance of the evaluation calculator class to use.
                                  If None, will use the default EvaluationCalculator.
        """
        self.evaluation_calculator = evaluation_calculator or EvaluationCalculator
        
    
    def calculate_average_results_from_matrix(self, mem_p_score_matrix, non_mem_p_score_matrix, cal_p_score_matrix, target_fdrs, scale_ratio=1):
        """
        Calculate average results over multiple random samplings.
        
        Args:
            mem_p_score: Member p-scores
            non_mem_p_score: Non-member p-scores
            cal_p_score: Calibration p-scores
            target_fdrs: List of target false discovery rates
            exp_num: Number of iteration
            sample_number: Number of samples
            
        Returns:
            Dictionary of averaged results
        """

        results = self.calculate_results_vector( 
            mem_p_score_matrix, 
            non_mem_p_score_matrix, 
            cal_p_score_matrix, 
            target_fdrs,
            scale_ratio = scale_ratio
        )
        
        results_mean = {}
        for key, value in results.items():
            results_mean[key] = np.mean(value, axis=0)
            
        return results_mean, results
    
    def calculate_results_vector(self, mem_p_score, non_mem_p_score, cal_p_score, target_fdrs, scale_ratio=1):
        """
        Calculate results using vectorized operations for multiple FDR targets.
        
        Args:
            mem_p_score: Matrix of member p-scores
            non_mem_p_score: Matrix of non-member p-scores
            cal_p_score: Matrix of calibration p-scores
            target_fdrs: List of target false discovery rates
            
        Returns:
            Dictionary of results for each metric
        """
        test_score = np.concatenate([mem_p_score, non_mem_p_score], axis=1)
        y_true_mem = np.ones(mem_p_score.shape[1])
        y_true_non_mem = np.zeros(non_mem_p_score.shape[1])
        y_test_true = np.concatenate([y_true_mem, y_true_non_mem])
        
        predict = BH_matrix_multidim_optimized(
            calibration_scores=cal_p_score,
            test_scores=test_score, 
            q_vector=target_fdrs,
            scale_ratio=scale_ratio
        )
        
        result = self.evaluation_calculator.calculate_selective_metrics_extended(y_test_true, predict)            
        return result
    
    
class EvaluationCalculator:
    evaluations = {}  # Dictionary to store registered evaluations

    @classmethod
    def register_evaluation(cls, func):
        """Decorator to register new evaluation functions."""
        cls.evaluations[func.__name__] = func
        return func

    
    @staticmethod
    def calculate_selective_metrics(label, pred_label):

        label = np.array(label)
        pred_label = np.array(pred_label)
        
        true_positive = np.sum((label == 1) & (pred_label == 1))  # TP
        false_positive = np.sum((label == 0) & (pred_label == 1))  # FP
        true_negative = np.sum((label == 0) & (pred_label == 0))  # TN
        false_negative = np.sum((label == 1) & (pred_label == 0))  # FN
        
        total_discoveries = true_positive + false_positive
        fdp = false_positive / total_discoveries if total_discoveries > 0 else 0
        
        total_positives = true_positive + false_negative
        power = true_positive / total_positives if total_positives > 0 else 0
        
        total_samples = len(label)
        accuracy = (true_positive + true_negative) / total_samples if total_samples > 0 else 0
        
        total_negatives = true_negative + false_positive
        fpr = false_positive / total_negatives if total_negatives > 0 else 0
        
        tpr = power  
        
        return {
            "FDP": fdp,
            "Power": power,
            "FPR": fpr,
            "TPR": tpr,
            "Accuracy": accuracy,
        }
        
    
    
    @staticmethod
    def calculate_selective_metrics_extended(label, pred_label):
        label = np.array(label)  # n-dimensional
        pred_label = np.array(pred_label)  # Assuming shape: (m, k, n)
        
        # Reshape label for broadcasting
        label_expanded = label[np.newaxis, np.newaxis, :]  # (1, 1, n)
        
        # Broadcast to match pred_label dimensions
        label_broadcasted = np.broadcast_to(label_expanded, pred_label.shape)  # (m, k, n)
        
        # Calculate metrics
        true_positive = np.sum((label_broadcasted == 1) & (pred_label == 1), axis=2)  # (m, k)
        false_positive = np.sum((label_broadcasted == 0) & (pred_label == 1), axis=2)  # (m, k)
        true_negative = np.sum((label_broadcasted == 0) & (pred_label == 0), axis=2)  # (m, k)
        false_negative = np.sum((label_broadcasted == 1) & (pred_label == 0), axis=2)  # (m, k)
        
        total_discoveries = true_positive + false_positive
        fdp = np.divide(false_positive, total_discoveries, 
                    out=np.zeros_like(total_discoveries, dtype=float), 
                    where=total_discoveries > 0)
        
        total_positives = true_positive + false_negative
        power = np.divide(true_positive, total_positives, 
                        out=np.zeros_like(total_positives, dtype=float), 
                        where=total_positives > 0)
        
        total_samples = len(label)  # n is the number of samples
        accuracy = (true_positive + true_negative) / total_samples
        
        total_negatives = true_negative + false_positive
        fpr = np.divide(false_positive, total_negatives, 
                    out=np.zeros_like(total_negatives, dtype=float), 
                    where=total_negatives > 0)
        
        tpr = power  # TPR is the same as Power
        
        return {
            "Accuracy": accuracy,   # (m, k)
            "FDP": fdp,           # (k,)
            "Power": power  ,     # (k,)
            "FPR": fpr,       # (k,)
            "TPR": tpr ,      # (k,)
        }


def out_calibrated_sampling(
                            member_score, 
                            non_member_score,
                            num_exp=1000,
                            test_member_ratio =0.88,
                            test_non_member_ratio = 0.06,
                            cal_ratio = 0.94,
                            seed = None
                            ):
    """Self out-calibrated sampling mode"""
    # Sample member matrix
    member_matrix, _ = random_sample_matrix_by_ratio(
        member_score, p=test_member_ratio, m=num_exp, seed=seed
    )
    
    # Sample non_member matrix and generate calibration matrix
    non_member_matrix, cal_matrix = tuple(
        random_sample_matrix_by_ratio_list(non_member_score, p=[test_non_member_ratio, cal_ratio], m=num_exp,
                                           seed=None if seed is None else seed + 10000)
    )
    
    return member_matrix, non_member_matrix, cal_matrix



  

def random_sample_matrix_by_ratio(arr, p=0.8, m=1000, seed=None):
    """
    Returns:
        tuple: (selected_matrix, unselected_matrix)
        - selected_matrix: shape (m, int(p * len(arr))) - 
        - unselected_matrix: shape (m, len(arr) - int(p * len(arr))) 
    """
    arr = np.array(arr)
    arr_len = len(arr)
    
    if not (0 < p <= 1):
        raise ValueError(f"Proportion p ({p}) must be between 0 and 1")
    
    n = int(p * arr_len)
    if n == 0:
        raise ValueError(f"Proportion p ({p}) too small, results in 0 samples")
    if n >= arr_len:
        raise ValueError(f"Proportion p ({p}) too large, not enough elements to leave unselected")

    rng = np.random.default_rng(seed)
    random_matrix = rng.random((m, arr_len))
    sorted_indices = np.argsort(random_matrix, axis=1)
    
    selected_indices = sorted_indices[:, :n]
    unselected_indices = sorted_indices[:, n:]
    
    selected_matrix = arr[selected_indices]
    unselected_matrix = arr[unselected_indices]
    
    return selected_matrix, unselected_matrix


def _sample_single_ratio(arr, p, m, arr_len, seed=None):
    """Helper function for handling single ratio case (fully vectorized)"""
    if not (0 < p <= 1):
        raise ValueError(f"Proportion p ({p}) must be between 0 and 1")
    
    n = int(p * arr_len)
    if n == 0:
        raise ValueError(f"Proportion p ({p}) too small, results in 0 samples")
    if n >= arr_len:
        raise ValueError(f"Proportion p ({p}) too large, not enough elements to leave unselected")

    # Fully vectorized operations
    rng = np.random.default_rng(seed)
    random_matrix = rng.random((m, arr_len))
    sorted_indices = np.argsort(random_matrix, axis=1)
    
    # Vectorized slicing
    selected_indices = sorted_indices[:, :n]
    unselected_indices = sorted_indices[:, n:]
    
    # Vectorized indexing
    selected_matrix = arr[selected_indices]
    unselected_matrix = arr[unselected_indices]
    
    return selected_matrix, unselected_matrix

# Alternative implementation using advanced indexing (even more vectorized)
def random_sample_matrix_by_ratio_list(arr, p, m=800, seed=None):
    """
    Alternative implementation with maximum vectorization using broadcasting and advanced indexing.
    
    Returns:
        If p is a single number:
            tuple: (selected_matrix, unselected_matrix)
        If p is a list of length n:
            list of length n: [matrix1, matrix2, ...] where each matrix has columns corresponding to its ratio
    """
    arr = np.array(arr)
    arr_len = len(arr)
    
    if isinstance(p, (int, float)):
        selected, unselected = _sample_single_ratio(arr, p, m, arr_len)
        return selected, unselected  
    
    if isinstance(p, (list, tuple)):
        p = np.array(p)
        
        # Validation (vectorized)
        if np.sum(p) > 1:
            raise ValueError(f"Sum of proportions ({np.sum(p)}) cannot exceed 1")
        if np.any((p <= 0) | (p > 1)):
            raise ValueError("All proportions must be between 0 and 1")
        
        # Calculate sample sizes (vectorized)
        sample_sizes = (p * arr_len).astype(int)
        if np.any(sample_sizes == 0):
            raise ValueError("Some proportions too small, result in 0 samples")
        if np.sum(sample_sizes) > arr_len:
            raise ValueError("Total samples > array length")
        
        # Generate all random permutations at once
        rng = np.random.default_rng(seed)
        random_matrix = rng.random((m, arr_len))
        sorted_indices = np.argsort(random_matrix, axis=1)
        
        # Calculate split points for each ratio
        split_points = np.concatenate([[0], np.cumsum(sample_sizes)])
        
        # Return list of matrices, each corresponding to one ratio
        # Each matrix has shape (m, sample_size_for_this_ratio)
        results = [
            arr[sorted_indices[:, split_points[i]:split_points[i+1]]]
            for i in range(len(p))
        ]
        
        return results
    
    else:
        raise ValueError("p must be a number or a list/tuple of numbers")


class DualKernelOptimizer:
    """
    A standalone optimizer specifically for estimating the null proportion (pi0)
    using a dual-kernel density estimation approach to eliminate first-order bias.
    """
    def __init__(self, gamma: float = 3.0):
        self.gamma = float(gamma)
        if self.gamma <= 1.0:
            self.gamma = 1.1

        # 1. Calculate weights to eliminate first-order bias
        self.w1 = -1.0 / (self.gamma - 1.0)
        self.w0 = 1.0 - self.w1
        
        # 2. Pre-calculate variance factor Omega(gamma)
        term_self_1 = (self.w0**2) / 2.0
        term_self_2 = (self.w1**2) / (2.0 * self.gamma)
        term_cross  = (2.0 * self.w0 * self.w1) / (self.gamma + 1.0)
        self.omega_gamma = term_self_1 + term_self_2 + term_cross

    def _calculate_bandwidth(self, p_values: np.ndarray) -> np.ndarray:
        """Internal method to calculate optimal bandwidth."""
        n_exp, m = p_values.shape

        # Step A: Estimate local slope c
        sum_log_p = np.sum(np.log(p_values), axis=1)
        c_hat = -m / (sum_log_p - 1e-15)

        # Safety: Clamp c if it's too close to 1.0 (unstable region)
        c_safe = c_hat.copy()
        mask_unstable = (c_safe > 0.95) & (c_safe < 1.05)
        c_safe[mask_unstable] = 1.05
        c_safe = np.maximum(c_safe, 0.1)

        # Step B: Calculate optimal bandwidth b (minimizing MSE)
        numerator = self.omega_gamma
        term_data = 4.0 * m * c_safe * np.power(c_safe - 1.0, 4)
        term_gamma = self.gamma**2
        denominator = term_data * term_gamma + 1e-12

        h_pow_5 = numerator / denominator
        b = np.power(h_pow_5, 0.2)

        # Step C: Dynamic constraints
        max_allowed_b = 0.8 / self.gamma 
        return np.clip(b, 0.001, max_allowed_b)[:, np.newaxis]

    def estimate(self, p_values: np.ndarray) -> np.ndarray:
        """
        Estimate the null proportion (pi0) given a matrix of p-values.
        
        Args:
            p_values (np.ndarray): Shape (n_experiments, n_samples)
            
        Returns:
            np.ndarray: Estimated pi0 values, Shape (n_experiments,)
        """
        # 1. Get optimal bandwidths
        b_vec = self._calculate_bandwidth(p_values)

        # 2. Define Kernel Mean function
        def beta_kernel_mean(bw_vec, multiplier):
            h = bw_vec * multiplier
            h = np.maximum(h, 1e-6) # Prevent division by zero
            alpha = (1.0 / h) + 1.0
            log_p = np.log(p_values)
            log_pdf = np.log(alpha) + (alpha - 1.0) * log_p
            return np.mean(np.exp(log_pdf), axis=1)

        # 3. Compute estimates from both kernels
        est_1 = beta_kernel_mean(b_vec, 1.0)
        est_2 = beta_kernel_mean(b_vec, self.gamma)
        
        # 4. Combine to remove bias
        pi0_hat = self.w0 * est_1 + self.w1 * est_2
        
        return np.clip(pi0_hat, 0.0, 1.0)


class MinStoreyEstimator:
    """
    Min-Storey pi0 estimator for conformal p-values.
    Takes the minimum of Storey estimators over a grid of lambda values,
    with a finite-sample correction factor.
    
    Reference: Gao & Roquain (2026), "On min-Storey estimators for multiple
    testing and conformal novelty detection"
    """
    def __init__(self, lambda_grid=None, correction='finite_sample'):
        if lambda_grid is None:
            # Dense grid of lambda values for Storey's estimator
            self.lambda_grid = np.array([
                0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 
                0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80
            ])
        else:
            self.lambda_grid = np.array(lambda_grid)
        self.correction = correction
    
    def estimate_pi0_and_gamma(self, p_values):
        """
        Estimate pi0 using Min-Storey: min over lambda of Storey's estimator.
        
        Args:
            p_values: numpy array of shape (n_experiments, n_samples)
            
        Returns:
            pi0_estimates: numpy array of shape (n_experiments,)
        """
        n_exp, m = p_values.shape
        
        pi0_estimates = np.zeros(n_exp)
        for i in range(n_exp):
            p_vals = p_values[i]
            storey_ests = []
            for lam in self.lambda_grid:
                n_greater = np.sum(p_vals >= lam)
                # Storey's estimator: (1 + #{p_i >= lambda}) / (m * (1 - lambda) + 1e-10)
                storey = (1.0 + n_greater) / (m * (1.0 - lam) + 1e-10)
                storey = min(storey, 1.0)  # cap at 1
                storey_ests.append(storey)
            
            pi0_raw = min(storey_ests)
            
            if self.correction == 'finite_sample':
                # Finite-sample correction: add 1/m for conservative protection
                pi0 = min(pi0_raw + 1.0 / m, 1.0)
            elif self.correction == 'none':
                pi0 = pi0_raw
            else:
                pi0 = min(pi0_raw + 1.0 / m, 1.0)
            
            pi0_estimates[i] = max(pi0, 0.05)  # floor at 0.05
        
        return np.clip(pi0_estimates, 0.05, 1.0)


class IMSEstimator:
    """
    Interval-Min-Storey (IMS) pi0 estimator for conformal p-values.
    Minimizes over intervals rather than single lambda values,
    designed for exactly uniform null distributions.
    
    Reference: Gao & Roquain (2026), "On min-Storey estimators for 
    multiple testing and conformal novelty detection"
    """
    def __init__(self, lambda_grid=None, kappa_min=0.5, eps=0.1, correction='finite_sample'):
        if lambda_grid is None:
            self.lambda_grid = np.array([
                0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 
                0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80
            ])
        else:
            self.lambda_grid = np.array(lambda_grid)
        self.kappa_min = kappa_min  # lower bound for intervals
        self.eps = eps  # minimum interval length
        self.correction = correction
    
    def estimate_pi0_and_gamma(self, p_values):
        """
        Estimate pi0 using IMS: min over intervals [lambda, mu].
        
        Args:
            p_values: numpy array of shape (n_experiments, n_samples)
            
        Returns:
            pi0_estimates: numpy array of shape (n_experiments,)
        """
        n_exp, m = p_values.shape
        
        pi0_estimates = np.zeros(n_exp)
        for i in range(n_exp):
            p_vals = p_values[i]
            
            # Find lower bound: median of p-values (or larger to avoid degenerate regions)
            kappa = max(self.kappa_min, np.median(p_vals))
            
            # Filter lambda grid to values above kappa
            valid_lambdas = self.lambda_grid[self.lambda_grid >= kappa]
            if len(valid_lambdas) == 0:
                valid_lambdas = np.array([kappa])
            
            best_pi0 = 1.0
            
            # For each lambda (lower bound of interval)
            for lam in valid_lambdas:
                # For each mu > lambda + eps (upper bound of interval)
                for mu in self.lambda_grid:
                    if mu <= lam + self.eps:
                        continue
                    
                    interval_length = mu - lam
                    n_in_interval = np.sum((p_vals >= lam) & (p_vals <= mu))
                    
                    # IMS estimate for this interval
                    pi0_interval = (1.0 + n_in_interval) / (m * interval_length + 1e-10)
                    pi0_interval = min(pi0_interval, 1.0)
                    
                    if pi0_interval < best_pi0:
                        best_pi0 = pi0_interval
            
            if self.correction == 'finite_sample':
                pi0 = min(best_pi0 + 1.0 / m, 1.0)
            else:
                pi0 = best_pi0
            
            pi0_estimates[i] = max(pi0, 0.05)
        
        return np.clip(pi0_estimates, 0.05, 1.0)


class PowerEnhancedStableEstimator:
    """
    Estimator designed to balance Power (lower pi0) and FDR stability.
    Uses an adaptive penalty to prevent pi0 underestimation when signal is strong.
    """
    def __init__(
        self,
        gamma_grid: Optional[List[float]] = None,
        n_check_iters: int = 50,
        sample_ratio: float = 1.0,
        stability_lambda: float = 1,
        rng=None,
    ):  
        default_grid = [
            0.01, 0.05, 0.1, 0.2, 0.5, 0.6, 0.8,
            1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0
        ]
        self.gamma_grid = np.array(gamma_grid if gamma_grid is not None else default_grid)

        self.n_check_iters = n_check_iters
        self.sample_ratio = sample_ratio
        self.stability_lambda = stability_lambda  # Reusing subtraction_quantile
        self.rng = rng if rng is not None else np.random.default_rng()

    
    
    def estimate_pi0_and_gamma(self, p_values: np.ndarray):
        n_exp, n_samples = p_values.shape
        n_sub = max(1, int(n_samples * self.sample_ratio))
        n_gammas = len(self.gamma_grid)

        mean_map = np.zeros((n_exp, n_gammas))
        std_map = np.zeros((n_exp, n_gammas))

        # 1. Run subsampling to assess stability across gamma grid
        for g_idx, g_val in enumerate(self.gamma_grid):
            # Assume DualKernelOptimizer is defined elsewhere
            optimizer = DualKernelOptimizer(gamma=g_val)
            runs = np.zeros((n_exp, self.n_check_iters))

            for k in range(self.n_check_iters):
                idx = self.rng.integers(0, n_samples, size=n_sub)
                runs[:, k] = optimizer.estimate(p_values[:, idx])

            mean_map[:, g_idx] = np.mean(runs, axis=1)
            std_map[:, g_idx] = np.std(runs, axis=1, ddof=1)


        row_avg_p = np.mean(p_values, axis=1, keepdims=True)
        signal_factor = np.clip((0.5 - row_avg_p) * 2, 0.1, 1.0) 

        # 2. Adaptive Stability with signal-aware scaling
        row_variation = np.max(std_map, axis=1, keepdims=True) - np.min(std_map, axis=1, keepdims=True)
        # We use a square root to prevent over-penalizing in noisy datasets
        adaptive_penalty = np.sqrt(std_map / (row_variation + 1e-6))
        
        # 3. Dynamic Lambda
        # In Wikimia (weak signal), effective_lambda will be smaller, favoring mean_pi0 (Power)
        effective_lambda = self.stability_lambda * signal_factor

        # 5. Final Score
        score_map = mean_map + (effective_lambda * adaptive_penalty)
        # boundary_barrier

        best_idx = np.argmin(score_map, axis=1)
        row_idx = np.arange(n_exp)

        final_pi0 = mean_map[row_idx, best_idx]
        selected_gamma = self.gamma_grid[best_idx]

        
        self._last_mean_map = mean_map
        self._last_std_map = std_map
        self._last_row_variation = row_variation
        
        return np.clip(final_pi0, 0.05, 1.0)

    

