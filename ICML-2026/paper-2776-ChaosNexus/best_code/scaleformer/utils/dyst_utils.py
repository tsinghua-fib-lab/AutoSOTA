"""
Additional dynamical systems utils to be merged with dysts repo at a future date
"""

from typing import Any

import dysts.flows as flows  # type: ignore
import numpy as np
from dysts.systems import DynSys  # type: ignore
from scipy.spatial.distance import pdist

from scaleformer.coupling_maps import RandomAdditiveCouplingMap
from scaleformer.skew_system import SkewProduct


def init_skew_system_from_params(
    driver_name: str,
    response_name: str,
    param_dict: dict[str, Any],
    **kwargs,
) -> DynSys:
    """
    Initialize a skew-product dynamical system from saved parameters.
    Assumes RandomAdditiveCouplingMap.
    """
    system_name = f"{driver_name}_{response_name}"
    required_keys = [
        "driver_params",
        "response_params",
        "coupling_map",
    ]
    for key in required_keys:
        if key not in param_dict:
            raise ValueError(f"Key {key} not found in param_dict for {system_name}")

    driver = getattr(flows, driver_name)(parameters=param_dict["driver_params"])
    response = getattr(flows, response_name)(parameters=param_dict["response_params"])

    coupling_map = RandomAdditiveCouplingMap._deserialize(param_dict["coupling_map"])

    sys = SkewProduct(
        driver=driver, response=response, coupling_map=coupling_map, **kwargs
    )

    return sys


### Utils for optimal time lag via Mutual Information ###


def mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 64) -> float:
    """
    Compute the mutual information between two 1D arrays x and y
        - Uses a 2D histogram with the given number of bins.
        - sum_{i,j} p_xy[i,j] * log(p_xy[i,j] / (p_x[i]*p_y[j]))
    """
    pxy, _, _ = np.histogram2d(x, y, bins=bins, density=True)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    mi = 0.0
    for i in range(pxy.shape[0]):
        for j in range(pxy.shape[1]):
            if pxy[i, j] > 0:
                mi += pxy[i, j] * (np.log(pxy[i, j]) - np.log(px[i]) - np.log(py[j]))
    return mi


def optimal_delay(
    x: np.ndarray,
    max_delay: int = 50,
    bins: int = 64,
    conv_window_size: int = 3,
    first_k_minima_to_consider: int = 1,
) -> int:
    """
    Computes the mutual information I(tau) = I( x(t), x(t+tau) ) for tau in {1, 2, ..., max_delay}
    Returns the time lag tau corresponding to the first prominent local minimum.

    Parameters:
        x: 1D array of shape (T,)
        max_delay: maximum time lag to consider
        bins: number of bins for the histogram
        conv_window_size: size of the convolution window for smoothing the MI curve
        first_k_minima_to_consider: number of minima to consider for determining first prominent minimum
    """
    mi_values = []
    assert x.ndim == 1, "x must be a 1D array"
    T = len(x)
    for tau in range(1, max_delay + 1):
        # Use only overlapping segments
        mi_tau = mutual_information(x[: T - tau], x[tau:], bins=bins)
        mi_values.append(mi_tau)
    mi_values = np.array(mi_values)

    # Find a prominent local minimum
    # 1. smooth the MI curve to reduce noise
    smoothed_mi = np.convolve(
        mi_values, np.ones(conv_window_size) / conv_window_size, mode="valid"
    )

    # 2. Calculate the prominence of each minimum
    minima_indices = []
    prominences = []

    # 3. Find all local minima in the smoothed curve
    for i in range(1, len(smoothed_mi) - 1):
        if smoothed_mi[i] < smoothed_mi[i - 1] and smoothed_mi[i] < smoothed_mi[i + 1]:
            minima_indices.append(i)

            # Calculate prominence (height difference to nearby values on smoothed MI curve)
            left_max = np.max(smoothed_mi[: i + 1])
            right_max = np.max(smoothed_mi[i:])
            lower_max = min(left_max, right_max)
            prominence = lower_max - smoothed_mi[i]
            prominences.append(prominence)
            if len(prominences) >= first_k_minima_to_consider:
                break

    # If no minima found, return the global minimum
    if len(minima_indices) == 0:
        first_min = np.argmin(mi_values) + 1
    else:
        # Find the most prominent minimum among the first first_k_minima_to_consider
        num_to_consider = min(first_k_minima_to_consider, len(minima_indices))
        best_idx = np.argmax(prominences[:num_to_consider])
        # Adjust index to account for smoothing window and 1-based tau
        first_min = minima_indices[best_idx] + (conv_window_size // 2) + 1

    return int(first_min)


### Utils for Zero-One Test for Chaos ###


def compute_translation_variables(
    phi: np.ndarray, c: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters:
        phi: 1D array of shape (T,)
        c: constant in (0,pi); if None, a random value in (pi/5, 4pi/5) is chosen to avoid resonances.
    Returns:
        p: 1D array of shape (T,), translation variable p(n) computed as cumulative sum of phi(n) * cos(c * n)
        q: 1D array of shape (T,), translation variable q(n) computed as cumulative sum of phi(n) * sin(c * n)
    """
    T = len(phi)
    n = np.arange(1, T + 1)
    # Compute translation variables
    p = np.cumsum(phi * np.cos(c * n))
    q = np.cumsum(phi * np.sin(c * n))
    return p, q


def compute_mean_square_displacement(
    p: np.ndarray, q: np.ndarray, max_shift_ratio: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the mean square displacement (MSD) over a range of shifts.

    Parameters:
      p, q: translation variables (1D arrays)
      max_shift_ratio: maximum fraction of the length to use for shifts.

    Returns:
      shift_indices: array of shift indices
      MSD: array of mean square displacements corresponding to shift_indices.
    """
    T = len(p)
    max_shift = int(max_shift_ratio * T)
    shift_indices = np.arange(1, max_shift + 1)
    MSD = np.empty_like(shift_indices, dtype=float)

    # For each time shift n, compute the mean squared difference
    for idx, shift_idx in enumerate(shift_indices):
        diff_p = p[shift_idx:] - p[:-shift_idx]
        diff_q = q[shift_idx:] - q[:-shift_idx]
        MSD[idx] = np.mean(diff_p**2 + diff_q**2)
    return shift_indices, MSD


def compute_K_statistic(shift_indices: np.ndarray, MSD: np.ndarray) -> float:
    """
    Computes the correlation coefficient between the shift indices and the MSD.
    A value near 1 indicates linear growth (chaos), while near 0 indicates bounded behavior.

    Parameters:
        shift_indices: 1D array of shape (T,), the shift indices
        MSD: 1D array of shape (T,), the mean square displacement computed as the mean of the squared differences between the translation variables

    Returns:
        K: correlation coefficient
    """
    corr_matrix = np.corrcoef(shift_indices, MSD)  # 2x2 correlation matrix
    K = corr_matrix[0, 1]
    return K


def zero_one_test(phi: np.ndarray, c: float | None = None) -> float:
    """
    Performs the 0–1 test for chaos on a scalar observable.

    Parameters:
      phi: univariate time series from the trajectory (e.g. x, y, or z coordinate, or norm), of length T
      c: constant in (0,pi); if None, a random value in (pi/5, 4pi/5) is chosen to avoid resonances.
            NOTE: this range seems to capture intrinsic diffusive behavior without interference from unwanted resonant effects.

    Returns:
      K: the computed correlation coefficient.
    """
    if c is None:
        # Choosing c in (pi/5, 4*pi/5) can help avoid resonances.
        c = np.random.uniform(np.pi / 10, 1 * np.pi / 5)

    p, q = compute_translation_variables(phi, c)
    shift_indices, MSD = compute_mean_square_displacement(p, q)
    K = compute_K_statistic(shift_indices, MSD)  # correlation coefficient

    return K


# Test using the Euclidean norm as observable.
def run_zero_one_sweep(
    timeseries: np.ndarray,
    c_min: float = np.pi / 5,
    c_max: float = 4 * np.pi / 5,
    n_runs: int = 100,
    k: int = 1,
) -> np.ndarray:
    """
    Runs a sweep of zero_one_test for the given univariate timeseries and c_vals

    Parameters:
        timeseries: univariate observed timeseries (e.g. x, y, z or norm of traj) of length T
        c_min: min value for c parameter, defaults to pi/5
        c_max: max value for c parameter, defaults to 4pi/5
        k: number of minima to consider for determining first prominent minimum
            the subsampling interval is the optimal delay computed using the mutual information
                NOTE: the performance of the 0-1 test for chaos is sensitive to this choice
                Subsampling helps to de-correlate timeseries that are oversampled (very similar consecutive points, excessive correlation),
                    ensuring that the time series better reflects the intrinsic dynamics of the system rather than oversampling artifacts
        n_runs: number of random c values to try
        threshold: threshold on |K| to decide if the system is chaotic

    Returns:
        K_vals: array of |K| values from the runs
    """
    assert timeseries.ndim == 1, "timeseries must be 1D"
    c_vals = np.random.uniform(c_min, c_max, n_runs)
    K_vals = []
    tau_opt = optimal_delay(
        timeseries, max_delay=50, bins=64, first_k_minima_to_consider=k
    )

    timeseries = timeseries[::tau_opt]
    for c_val in c_vals:
        K = zero_one_test(
            timeseries,
            c=c_val,
        )
        K_vals.append(K)

    K_vals = np.array(K_vals)
    return K_vals


### Utils for GP Dimension


def compute_gp_dimension(
    points: np.ndarray,
    num_r: int = 50,
    scaling_range_idx: tuple[int, int] | slice | None = None,
    verbose: bool = False,
) -> float:
    """
    Parameters:
        points (ndarray): Array of points with shape (T, dim).
        num_r (int): Number of r values to consider in the logarithmically spaced range.
        scaling_range_idx (tuple or slice): Indices to select the scaling region for the linear fit.
                                           If None, a default middle 50% region is used.

    NOTE: this is slightly different from the dysts implementation, remains to be seen which one to prefer
    Returns:
        D2 (float): Estimated correlation (GP) dimension.
    """
    # Define the range of r values based on the pairwise distances
    # We use the min and max distance from the data to set our range.
    distances = pdist(points, metric="euclidean")
    N_pairs = len(distances)

    r_min = np.min(distances)
    r_max = np.max(distances)
    if verbose:
        print(f"r_min: {r_min}, r_max: {r_max}")
    r_min = max(r_min, 1e-10)
    # Generate logarithmically spaced r values
    r_vals = np.logspace(np.log10(r_min), np.log10(r_max), num_r)

    # Compute the correlation integral C(r)
    # Compute all pairwise distances (for N points, there are N*(N-1)/2 distances)
    # For each r value, count the number of pairs with distance < r
    C = np.array([np.sum(distances < r) / N_pairs for r in r_vals])

    # Compute logarithms (safely to avoid log(0))
    log_r = np.log10(
        r_vals
    )  # r_vals are already guaranteed to be > 0 from r_min = max(r_min, 1e-10)
    # Ensure C values are positive before taking log
    C_safe = np.maximum(C, 1e-10)  # Add small epsilon to avoid log(0)
    log_C = np.log10(C_safe)

    # Select a scaling region for the linear fit
    # Here, if not provided, we choose the middle 50% of the r-range.
    if scaling_range_idx is None:
        scaling_range = slice(num_r // 4, 3 * num_r // 4)
    else:
        scaling_range = scaling_range_idx

    # Fit a line (using least squares) over the selected scaling region
    slope, _ = np.polyfit(log_r[scaling_range], log_C[scaling_range], 1)
    D2 = slope  # The slope corresponds to the correlation dimension D2

    return D2
