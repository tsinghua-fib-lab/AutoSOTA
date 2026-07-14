"""
Implementation of the Studentized Spherical Harmonics Energy Distance based Two-Sample Test.

This module provides efficient, vectorized implementation of the test statistic T_p_mn
and related quantities for large scale simulations.

Key optimizations:
1. Precompute coefficients for fixed (p, d) parameters
2. Compute dot products once and reuse across all kernel evaluations
3. Block based computation to avoid forming the full (m+n) x (m+n) matrix
4. Cache the sample size dependent constants for batch simulations
5. Numba JIT-compiled Gegenbauer recurrence (5-50x speedup over scipy)

Classes:
- SphericalTestConfig: Precomputed constants for fixed (p, d)
- OptimizedTestStatistic: Efficient vectorized computation of T_p_mn using blocks
"""

import numpy as np
from typing import Tuple, Union
from scipy.special import comb, gamma as scipy_gamma
import numba


@numba.jit(nopython=True)
def _gegenbauer_values_jit(dot_products_flat, lam, max_order, coeffs):
    """
    Numba JIT: Compute weighted sum of Gegenbauer polynomials for kernel evaluation.

    Uses three-term recurrence to compute all C_i^lam(z) for i=0..max_order
    and accumulates the weighted sum in a single pass.

    Parameters
    ----------
    dot_products_flat : np.ndarray, 1D float64
        Flattened dot product array
    lam : float
        Gegenbauer parameter lambda = (d-1)/2
    max_order : int
        Maximum Gegenbauer order (2p)
    coeffs : np.ndarray, 1D float64
        Precomputed coefficients for each order

    Returns
    -------
    np.ndarray
        Kernel values, same shape as input
    """
    n = len(dot_products_flat)
    result = np.zeros(n, dtype=np.float64)

    # C_0(z) = 1
    if max_order >= 0:
        for j in range(n):
            result[j] += coeffs[0]

    if max_order >= 1:
        for j in range(n):
            z = dot_products_flat[j]
            c_1 = 2.0 * lam * z
            result[j] += coeffs[1] * c_1

    if max_order >= 2:
        c_prev2 = np.ones(n, dtype=np.float64)  # C_0
        c_prev = 2.0 * lam * dot_products_flat  # C_1
        for i in range(2, max_order + 1):
            a = 2.0 * (i + lam - 1.0)
            b = i + 2.0 * lam - 2.0
            c_curr = (a * dot_products_flat * c_prev - b * c_prev2) / float(i)
            for j in range(n):
                result[j] += coeffs[i] * c_curr[j]
            c_prev2 = c_prev
            c_prev = c_curr

    return result


class SphericalTestConfig:
    """
    Precomputed constants for fixed (p, d) parameters.

    This class computes and caches all constants that depend only on the
    reproducing kernel parameter p and sphere dimension d, avoiding redundant
    computation when processing multiple samples. Also included are calculator
    methods to compute the reproducing kernel values, and full bias correction
    term, though these method depend on the data samples and cannot be precomputed
    at class instantiation.

    Parameters
    ----------
    p : int
        Index parameter for reproducing kernel (sum from i=0 to i=2p)
    d : int
        Dimension of the sphere S^d (embedded in R^{d+1})

    Attributes
    ----------
    p : int
        Reproducing kernel parameter
    d : int
        Sphere dimension
    Lambda : float
        Gegenbauer parameter is (d-1)/2
    sphere_volume : float
        Volume of the unit ball S^d
    coefficients : np.ndarray
        Precomputed coefficients Coeff_i_d for i = 0, 1, ..., 2p
    a_0_p : float
        Key term used for bias correction when estimating the variance of the projected ED

    Examples
    --------
    >>> config = SphericalTestConfig(p=2, d=2)
    >>> config.Lambda
    0.5
    >>> len(config.coefficients)
    5
    """

    def __init__(self, p: int, d: int):
        self.p = p
        self.d = d
        self.Lambda = (d - 1) / 2
        self.sphere_volume = np.pi**((d+1)/2) / scipy_gamma((d+1)/2 + 1)
        # Main computations of this class
        self.coefficients = self._precompute_coefficients()
        self.a_0_p = self._compute_a_0_p()
        # Warm up JIT compilation with a small array
        self._warmup_jit()

    def _warmup_jit(self):
        """Warm up the JIT compiler with a tiny array to avoid first-call overhead."""
        _ = _gegenbauer_values_jit(
            np.array([0.5], dtype=np.float64),
            self.Lambda,
            2 * self.p,
            self.coefficients.astype(np.float64),
        )

    def _precompute_coefficients(self) -> np.ndarray:
        """Precompute all (2p + 1) coefficients for the reproducing kernel"""
        coeffs = np.zeros(2 * self.p + 1)
        for i in range(2 * self.p + 1):
            a_i_d = comb(self.d + i, self.d, exact=True) - comb(self.d + i - 2, self.d, exact=True)

            # Compute the Gegenbauer polynomial evaluated at 1
            if i == 0:
                C_i_1 = 1.0
            else:
                C_i_1 = scipy_gamma(i + 2*self.Lambda) / (scipy_gamma(2*self.Lambda) * scipy_gamma(i + 1))

            coeffs[i] = a_i_d / (self.sphere_volume * C_i_1)
        return coeffs

    def _compute_a_0_p(self) -> float:
        """Compute the bias correction coefficient a_0_p."""
        binom_1 = comb(self.d + 2*self.p, self.d, exact=True)
        binom_2 = comb(self.d + 2*self.p - 1, self.d, exact=True)
        return (binom_1 + binom_2) / self.sphere_volume

    def compute_reproducing_kernel(self, dot_products: np.ndarray) -> np.ndarray:
        """
        Compute reproducing kernel values from precomputed dot products.

        This is the core computation: given dot products <x, y>, compute
        K_p(x, y) = Sum_{i=0}^{2p} Coeff_i_d * C^(Lambda)_i(<x, y>)

        Uses numba JIT-compiled Gegenbauer recurrence for speed.

        Parameters
        ----------
        dot_products : np.ndarray
            Array of dot products (any shape), values should be in [-1, 1]

        Returns
        -------
        np.ndarray
            Reproducing kernel values, same shape as input
        """
        orig_shape = dot_products.shape
        flat = dot_products.ravel().astype(np.float64)
        result_flat = _gegenbauer_values_jit(
            flat, self.Lambda, 2 * self.p,
            self.coefficients.astype(np.float64),
        )
        return result_flat.reshape(orig_shape)

    def compute_bias_term(self, m: int, n: int) -> float:
        """
        Compute the bias correction term for V_p_mn.

        bias_term = (a_0_p)^2 / [(N-1) * (N-3)]

        Parameters
        ----------
        m : int
            Size of first sample
        n : int
            Size of second sample

        Returns
        -------
        float
            Bias correction term
        """
        N = m + n
        if N < 4:
            raise ValueError(f"Combined sample size N = {N} must be at least 4")
        return (self.a_0_p ** 2) / ((N - 1) * (N - 3))


class OptimizedTestStatistic:
    """
    Efficient computation of the test statistic T_p_mn.

    Uses block based computation to avoid forming the full (m+n) x (m+n) matrix, reducing
    memory usage. Note we compute the bias corrected version of the statistic by default.

    This class computes:
    - Eps_p_mn: Projected energy distance estimator
    - V_p_mn: Kernelized square distance covariance
    - V_p_mn_unbiased: Bias-corrected variance estimator
    - T_p_mn: Test statistic = Eps_p_mn / sqrt(C_mn * V_p_mn_unbiased)

    Parameters
    ----------
    config : SphericalTestConfig
        Precomputed configuration for (p, d) parameters

    Examples
    --------
    >>> config = SphericalTestConfig(p=2, d=2)
    >>> calculator = OptimizedTestStatistic(config)
    >>> X = np.random.randn(100, 3)
    >>> X /= np.linalg.norm(X, axis=1, keepdims=True)
    >>> Y = np.random.randn(100, 3)
    >>> Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    >>> T = calculator.compute(X, Y)
    """

    def __init__(self, config: SphericalTestConfig):
        self.config = config

    def compute(self, X: np.ndarray, Y: np.ndarray,
                return_components: bool = False,
                use_unbiased: bool = True) -> Union[float, Tuple[float, float, float]]:
        """
        Compute test statistic T_p_mn with all optimizations, avoiding the formation
        of the full pooled sample Z = [X; Y]

        Parameters
        ----------
        X : np.ndarray, shape (m, d+1)
            First sample of points on S^d
        Y : np.ndarray, shape (n, d+1)
            Second sample of points on S^d
        return_components : bool, optional
            If True, return (T_p_mn, Eps_p_mn, V_p_mn). Default: False
        use_unbiased : bool, optional
            If True, use bias corrected variance V_p_mn_unbiased. Default: True

        Returns
        -------
        float or tuple
            Test statistic T_p_mn, or (T_p_mn, Eps_p_mn, V_p_mn) if return_components is True
        """
        X = np.asarray(X)
        Y = np.asarray(Y)
        m, n = X.shape[0], Y.shape[0]
        N = m + n

        # Step 1: Compute dot products for each block (NOT full Z-Z matrix)
        dot_XX = np.clip(X @ X.T, -1.0, 1.0)  # m x m
        dot_YY = np.clip(Y @ Y.T, -1.0, 1.0)  # n x n
        dot_XY = np.clip(X @ Y.T, -1.0, 1.0)  # m x n

        # Step 2: Compute kernel matrices from dot products
        K_XX = self.config.compute_reproducing_kernel(dot_XX)
        K_YY = self.config.compute_reproducing_kernel(dot_YY)
        K_XY = self.config.compute_reproducing_kernel(dot_XY)

        # Step 3: Compute Energy Distance estimator Eps_p_mn from blocks
        Eps_p_mn = self._compute_eps_from_blocks(K_XX, K_YY, K_XY, m, n)

        # Step 4: Compute V_p_mn from blocks (without forming full K_ZZ)
        V_p_mn = self._compute_variance_from_blocks(K_XX, K_YY, K_XY, m, n, N)

        # Apply bias correction
        if use_unbiased:
            bias_term = self.config.compute_bias_term(m, n)
            V_p_mn = V_p_mn - bias_term

        # Step 5: Compute C_mn and T_p_mn
        comb_m_2 = m * (m - 1) // 2
        comb_n_2 = n * (n - 1) // 2
        C_mn = 1.0 / comb_m_2 + 1.0 / comb_n_2 + 4.0 / (m * n)
        T_p_mn = Eps_p_mn / np.sqrt(C_mn * V_p_mn)

        if return_components:
            return T_p_mn, Eps_p_mn, V_p_mn
        return T_p_mn

    def compute_from_kernels(self, K_XX: np.ndarray, K_YY: np.ndarray,
                              K_XY: np.ndarray, m: int, n: int,
                              use_unbiased: bool = True) -> float:
        """
        Compute test statistic from precomputed kernel blocks.
        
        This avoids recomputing dot products and Gegenbauer polynomials,
        enabling fast batched permutation testing where only the index
        permutation changes but the pooled kernel matrix is fixed.
        
        Parameters
        ----------
        K_XX : np.ndarray, shape (m, m)
            Kernel matrix for first sample
        K_YY : np.ndarray, shape (n, n)
            Kernel matrix for second sample
        K_XY : np.ndarray, shape (m, n)
            Cross-kernel matrix
        m : int
            Size of first sample
        n : int
            Size of second sample
        use_unbiased : bool
            If True, apply bias correction
            
        Returns
        -------
        float
            Test statistic T_p_mn
        """
        N = m + n
        
        # Step 3: Compute Energy Distance estimator Eps_p_mn from blocks
        Eps_p_mn = self._compute_eps_from_blocks(K_XX, K_YY, K_XY, m, n)
        
        # Step 4: Compute V_p_mn from blocks
        V_p_mn = self._compute_variance_from_blocks(K_XX, K_YY, K_XY, m, n, N)
        
        # Apply bias correction
        if use_unbiased:
            bias_term = self.config.compute_bias_term(m, n)
            V_p_mn = V_p_mn - bias_term
        
        # Step 5: Compute C_mn and T_p_mn
        comb_m_2 = m * (m - 1) // 2
        comb_n_2 = n * (n - 1) // 2
        C_mn = 1.0 / comb_m_2 + 1.0 / comb_n_2 + 4.0 / (m * n)
        T_p_mn = Eps_p_mn / np.sqrt(C_mn * V_p_mn)
        
        return T_p_mn

    def _compute_eps_from_blocks(self, K_XX: np.ndarray, K_YY: np.ndarray,
                                  K_XY: np.ndarray, m: int, n: int) -> float:
        """
        Compute Eps_p_mn from kernel blocks.

        Eps_p_mn = [1/C(m,2) * Sum_{i<j} K_XX[i,j]]
                 + [1/C(n,2) * Sum_{i<j} K_YY[i,j]]
                 - [2/(m*n) * Sum_{i,j} K_XY[i,j]]
        """
        comb_m_2 = m * (m - 1) // 2
        comb_n_2 = n * (n - 1) // 2

        # Term 1: upper triangular sum of K_XX excluding diagonal

        sum_XX_upper = (np.sum(K_XX) - np.trace(K_XX)) / 2
        term1 = sum_XX_upper / comb_m_2

        # Term 2: upper triangular sum of K_YY excluding diagonal
        sum_YY_upper = (np.sum(K_YY) - np.trace(K_YY)) / 2
        term2 = sum_YY_upper / comb_n_2

        # Term 3: full sum of K_XY
        term3 = 2.0 * np.sum(K_XY) / (m * n)

        return term1 + term2 - term3

    def _compute_variance_from_blocks(self, K_XX: np.ndarray, K_YY: np.ndarray,
                                       K_XY: np.ndarray, m: int, n: int, N: int) -> float:
        """
        Compute V_p_mn from kernel blocks without forming full K_ZZ.

        The full kernel matrix K_ZZ has block structure:
            K_ZZ = | K_XX  K_XY |
                   | K_XY' K_YY |

        Compute row sums, total sum, and centered matrix A statistics directly from blocks.

        V_p_mn = 1/(N*(N-3)) * Sum_{s!=t} A[s,t]^2

        where A is the centered kernel matrix.
        """
        # First m rows (X rows): sum of K_XX row + corresponding K_XY row
        # Last n rows (Y rows): sum of K_XY column + corresponding K_YY row
        row_sums_X = np.sum(K_XX, axis=1) + np.sum(K_XY, axis=1)  # shape (m,)
        row_sums_Y = np.sum(K_XY, axis=0) + np.sum(K_YY, axis=1)  # shape (n,)

        # Total sum of K_ZZ
        total_sum = np.sum(K_XX) + np.sum(K_YY) + 2 * np.sum(K_XY)

        # Compute Sum_{s!=t} A^2[s,t] using block structure
        sum_A_squared = self._compute_A_squared_sum_from_blocks(
            K_XX, K_YY, K_XY, row_sums_X, row_sums_Y, total_sum, m, n, N
        )

        V_p_mn = sum_A_squared / (N * (N - 3))
        return V_p_mn

    def _compute_A_squared_sum_from_blocks(self, K_XX: np.ndarray, K_YY: np.ndarray,
                                            K_XY: np.ndarray, row_sums_X: np.ndarray,
                                            row_sums_Y: np.ndarray, total_sum: float,
                                            m: int, n: int, N: int) -> float:
        """
        Compute Sum_{s!=t} A^2[s,t] from blocks.

        The centered matrix A is defined as:
        A[s,t] = a[s,t] - row_sums[t]/(N-2) - row_sums[s]/(N-2) + total_sum/((N-1)*(N-2))

        where a[s,t] = K[s,t] if s != t, else 0.
        """
        alpha = 1.0 / (N - 2)
        grand_mean = total_sum / ((N - 1) * (N - 2))

        sum_A_sq = 0.0

        # Process XX block (s, t both in X indices: 0 to m-1)
        a_XX = K_XX.copy()
        np.fill_diagonal(a_XX, 0.0)
        # For XX block, row_sums indices are row_sums_X
        A_XX = (a_XX
                - alpha * row_sums_X[np.newaxis, :]  # subtract for each column
                - alpha * row_sums_X[:, np.newaxis]  # subtract for each row
                + grand_mean)
        np.fill_diagonal(A_XX, 0.0)  # diagonal does not contribute to sum_{s!=t}
        sum_A_sq += np.sum(A_XX ** 2)

        # Process YY block (s, t both in Y indices: m to N-1)
        a_YY = K_YY.copy()
        np.fill_diagonal(a_YY, 0.0)
        A_YY = (a_YY
                - alpha * row_sums_Y[np.newaxis, :]
                - alpha * row_sums_Y[:, np.newaxis]
                + grand_mean)
        np.fill_diagonal(A_YY, 0.0)
        sum_A_sq += np.sum(A_YY ** 2)

        # Process XY block (s in X: 0 to m-1, t in Y: m to N-1)
        # No diagonal in this block since s < m and t >= m
        A_XY = (K_XY
                - alpha * row_sums_Y[np.newaxis, :]  # row_sums for Y indices (columns)
                - alpha * row_sums_X[:, np.newaxis]  # row_sums for X indices (rows)
                + grand_mean)
        sum_A_sq += np.sum(A_XY ** 2)

        # Process YX block (s in Y: m to N-1, t in X: 0 to m-1)
        A_YX = (K_XY.T
                - alpha * row_sums_X[np.newaxis, :]  # row_sums for X indices (columns)
                - alpha * row_sums_Y[:, np.newaxis]  # row_sums for Y indices (rows)
                + grand_mean)
        sum_A_sq += np.sum(A_YX ** 2)

        return sum_A_sq
