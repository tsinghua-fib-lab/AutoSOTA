"""
Suite of tests to determine if generated trajectories are valid attractors
"""

import functools
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
from dysts.analysis import max_lyapunov_exponent_rosenstein
from scaleformer.utils import run_zero_one_sweep
from scipy.fft import rfft
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist
from statsmodels.tsa.stattools import adfuller, kpss


@dataclass
class AttractorValidator:
    """
    Framework to add tests, which are executed sequentially to determine if generated trajectories are valid attractors.
    Upon first failure, the trajectory sample is added to the failed ensemble.
    To add custom tests, define functions that take a trajectory and return a boolean (True if the trajectory passes the test, False otherwise).
    """

    transient_time_frac: float = 0.05  # should be low, should be on attractor
    tests: Optional[List[Callable]] = None

    multiprocess_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.failed_checks = defaultdict(list)  # Dict[str, List[Tuple[int, str]]]
        self.valid_dyst_counts = defaultdict(int)  # Dict[str, int]
        self.failed_samples = defaultdict(list)  # Dict[str, List[int]]
        self.valid_samples = defaultdict(list)  # Dict[str, List[int]]

    def reset(self):
        """
        Reset all defaultdict attributes to their initial state.
        """
        self.failed_checks.clear()
        self.valid_dyst_counts.clear()
        self.failed_samples.clear()
        self.valid_samples.clear()

    def _execute_test_fn(
        self,
        test_fn: Callable,
        traj_sample: np.ndarray,
    ) -> Tuple[bool, str]:
        """
        Execute a single test for a given trajectory sample of a system.
        Args:
            test_fn: the attractor test function to execute
            dyst_name: name of the dyst
            traj_sample: the trajectory sample to test
            sample_idx: index of the sample

        Returns:
            bool: True if the test passed, False otherwise
        """
        original_func = (
            test_fn.func if isinstance(test_fn, functools.partial) else test_fn
        )
        func_name = original_func.__name__
        status = test_fn(traj_sample)
        return status, func_name

    def _filter_system_worker_fn(
        self,
        dyst_name: str,
        all_traj: np.ndarray,
        first_sample_idx: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, str]], List[int]]:
        """
        Multiprocessed version of self._filter_dyst without any verbose output

        TODO: figure out how to log safely during multiprocessing
        """
        failed_checks_samples = []
        valid_samples = []
        valid_attractor_trajs = []
        failed_attractor_trajs = []
        for i, traj_sample in enumerate(all_traj):
            sample_idx = first_sample_idx + i
            # cut off transient time
            transient_time = int(traj_sample.shape[1] * self.transient_time_frac)
            traj_sample = traj_sample[:, transient_time:]
            # execute all tests in sequence
            status = True
            for test_fn in self.tests or []:
                status, test_name = self._execute_test_fn(test_fn, traj_sample)
                if not status:
                    failed_check = (sample_idx, test_name)
                    failed_checks_samples.append(failed_check)
                    break
            # if traj sample failed a test, move on to next trajectory sample for this dyst
            if not status:
                failed_attractor_trajs.append(traj_sample)
                continue
            valid_attractor_trajs.append(traj_sample)
            valid_samples.append(sample_idx)
        return (
            np.array(valid_attractor_trajs),
            np.array(failed_attractor_trajs),
            failed_checks_samples,
            valid_samples,
        )

    def filter_ensemble(
        self, ensemble: Dict[str, np.ndarray], first_sample_idx: int = 0
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Execute all tests for all trajectory samples in the ensemble, and split the ensemble into valid and failed ensembles.
        Args:
            ensemble: The trajectory ensemble to filter
            first_sample_idx: The index of the first sample for the generated trajectories of the ensemble

        Returns:
            valid_attractor_ensemble: A new ensemble with only the valid trajectories
            failed_attractor_ensemble: A new ensemble with only the failed trajectories
        """
        valid_attractor_ensemble: Dict[str, np.ndarray] = {}
        failed_attractor_ensemble: Dict[str, np.ndarray] = {}
        for dyst_name, all_traj in ensemble.items():
            (
                valid_attractor_trajs,
                failed_attractor_trajs,
                failed_checks,
                valid_samples,
            ) = self._filter_system_worker_fn(dyst_name, all_traj, first_sample_idx)

            self.failed_checks[dyst_name].extend(failed_checks)
            self.failed_samples[dyst_name].extend([ind for ind, _ in failed_checks])
            self.valid_samples[dyst_name].extend(valid_samples)
            self.valid_dyst_counts[dyst_name] += len(valid_samples)

            if len(failed_attractor_trajs) > 0:
                failed_attractor_ensemble[dyst_name] = failed_attractor_trajs

            if len(valid_attractor_trajs) == 0:
                continue

            valid_attractor_ensemble[dyst_name] = valid_attractor_trajs

        return valid_attractor_ensemble, failed_attractor_ensemble

    def multiprocessed_filter_ensemble(
        self, ensemble: Dict[str, np.ndarray], first_sample_idx: int = 0
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Multiprocessed version of self.filter_ensemble
        """
        with Pool(**self.multiprocess_kwargs) as pool:
            results = pool.starmap(
                self._filter_system_worker_fn,
                [
                    (dyst_name, all_traj, first_sample_idx)
                    for dyst_name, all_traj in ensemble.items()
                ],
            )
        valid_trajs, failed_trajs, failed_checks, valid_samples = zip(*results)
        for dyst_name, failed_check_lst in zip(list(ensemble.keys()), failed_checks):
            if len(failed_check_lst) > 0:
                self.failed_checks[dyst_name].append(failed_check_lst)
                self.failed_samples[dyst_name].extend(
                    [index for index, _ in failed_check_lst]
                )
        for dyst_name, valid_samples_lst in zip(list(ensemble.keys()), valid_samples):
            if len(valid_samples_lst) > 0:
                self.valid_samples[dyst_name].extend(valid_samples_lst)
                self.valid_dyst_counts[dyst_name] += len(valid_samples_lst)
        # form the valid and failed ensembles
        # note: relies on python 3.7+ order preservation in dictionaries
        valid_ensemble = {
            k: v for k, v in zip(list(ensemble.keys()), valid_trajs) if v.shape[0] > 0
        }
        failed_ensemble = {
            k: v for k, v in zip(list(ensemble.keys()), failed_trajs) if v.shape[0] > 0
        }
        return valid_ensemble, failed_ensemble


def check_boundedness(
    traj: np.ndarray, threshold: float = 1e4, max_zscore: float = 10, eps: float = 1e-10
) -> bool:
    """
    Check if a multi-dimensional trajectory is bounded (not diverging).

    Args:
        traj: np.ndarray of shape (num_dims, num_timepoints), the trajectory data.
        threshold: Maximum absolute value of the trajectory to consider as diverging.
        max_zscore: Maximum z-score of the trajectory to consider as diverging.
    Returns:
        bool: False if the system is diverging, True otherwise.
    """
    if np.any(np.abs(traj) > threshold):
        return False

    mean = np.nanmean(traj.T, axis=0)
    std = np.nanstd(traj.T, axis=0)
    std = np.where(std < eps, eps, std)

    standardized_traj = (traj.T - mean) / std

    if np.max(np.abs(standardized_traj)) > max_zscore:
        return False

    return True


def check_not_fixed_point(
    traj: np.ndarray, tail_prop: float = 0.05, atol: float = 1e-3
) -> bool:
    """
    Check if the system trajectory converges to a fixed point.
    Actually, this tests the variance decay in the trajectory to detect a fixed point.

    Args:
        traj: np.ndarray of shape (num_dims, num_timepoints), the trajectory data.
        tail_prop: Proportion of the trajectory to consider for variance comparison.
        atol: Absolute tolerance for detecting a fixed point.
    Returns:
        bool: False if the system is approaching a fixed point, True otherwise.
    """
    n = traj.shape[1]
    tail = int(tail_prop * n)
    distances = np.linalg.norm(np.diff(traj[:, -tail:], axis=1), axis=0)

    if np.allclose(distances, 0, atol=atol):
        return False

    return True


def check_not_trajectory_decay(
    traj: np.ndarray, tail_prop: float = 0.5, atol: float = 1e-3
) -> bool:
    """
    Check if a multi-dimensional trajectory is not decaying.
    Args:
        traj: np.ndarray of shape (num_dims, num_timepoints), the trajectory data.
        tail_prop: Proportion of the trajectory to consider for variance comparison.
        atol: Absolute tolerance for detecting a fixed point.
    Returns:
        bool: False if the system is approaching a fixed point, True otherwise.
    """
    # Check if any dimension of the trajectory is a straight line in the last tail_prop of the trajectory
    n = traj.shape[1]
    tail = int(tail_prop * n)
    for dim in range(traj.shape[0]):
        diffs = np.diff(traj[dim, -tail:])
        if np.allclose(diffs, 0, atol=atol):
            return False
    return True


def check_not_limit_cycle(
    traj: np.ndarray,
    tolerance: float = 1e-3,
    min_prop_recurrences: float = 0.0,
    min_counts_per_rtime: int = 100,
    min_block_length: int = 1,
    min_recurrence_time: int = 1,
    enforce_endpoint_recurrence: bool = False,
) -> bool:
    """
    Checks if a multidimensional trajectory is collapsing to a limit cycle.

    Args:
        traj (ndarray): 2D array of shape (num_vars, num_timepoints), where each row is a time series.
        tolerance (float): Tolerance for detecting revisits to the same region in phase space.
        min_prop_recurrences (float): Minimum proportion of the trajectory length that must be recurrences to consider a limit cycle
        min_counts_per_rtime (int): Minimum number of counts per recurrence time to consider a recurrence time as valid
        min_block_length (int): Minimum block length of consecutive recurrence times to consider a recurrence time as valid
        min_recurrence_time (int): Minimum recurrence time to consider a recurrence time as valid
                e.g. Setting min_recurrence_time = 1 means that we can catch when the integration fails (or converges to fixed point)
        enforce_endpoint_recurrence (bool): Whether to enforce that either of the endpoints are recurrences
                e.g. Setting enforce_endpoint_recurrence = True means that we are operating in a stricter regime where we require either
                     the initial or final point to be a recurrence (repeated some time in the trajectory).

    The default args are designed to be lenient, and catch pathological cases beyond purely limit cycles.
        For strict mode, can set e.g. min_prop_recurrences = 0.1, min_block_length=50, min_recurrence_time = 10, enforce_endpoint_recurrence = True,
    Returns:
        bool: True if the trajectory is not collapsing to a limit cycle, False otherwise.
    """
    n = traj.shape[1]

    # Step 1: Calculate the pairwise distance matrix, shape should be (N, N)
    dist_matrix = cdist(traj.T, traj.T, metric="euclidean").astype(np.float16)
    dist_matrix = np.triu(dist_matrix, k=1)

    # Step 2: Get recurrence times from thresholding distance matrix
    recurrence_indices = np.asarray(
        (dist_matrix < tolerance) & (dist_matrix > 0)
    ).nonzero()

    n_recurrences = len(recurrence_indices[0])
    if n_recurrences == 0:
        return True

    if enforce_endpoint_recurrence:
        # check if an eps neighborhood around either n-1 or 0 is in either of the recurrence indices
        eps = 0
        if not any(
            (n - 1) - max(indices) <= eps or min(indices) - 0 <= eps
            for indices in recurrence_indices
        ):
            return True

    # get recurrence times
    recurrence_times = np.abs(recurrence_indices[0] - recurrence_indices[1])
    recurrence_times = recurrence_times[recurrence_times >= min_recurrence_time]

    # Heuristic 1: Check if there are enough recurrences to consider a limit cycle
    n_recurrences = len(recurrence_times)
    if n_recurrences < int(min_prop_recurrences * n):
        return True

    # Heuristic 2: Check if there are enough valid recurrence times
    rtimes_counts = Counter(recurrence_times)
    n_valid_rtimes = sum(
        1 for count in rtimes_counts.values() if count >= min_counts_per_rtime
    )
    if n_valid_rtimes < 1:
        return True

    # Heuristic 3: Check if the valid recurrence times are formed of blocks of consecutive timepoints
    if min_block_length > 1:
        rtimes_dict = defaultdict(list)
        block_length = 1
        prev_rtime = None
        prev_t1 = None
        prev_t2 = None
        rtimes_is_valid = False
        num_blocks = 0
        # assuming recurrence_indices[0] is sorted
        for t1, t2 in zip(*recurrence_indices):
            rtime = abs(t2 - t1)
            if rtime < min_recurrence_time:
                continue
            if (
                rtime == prev_rtime
                and abs(t1 - prev_t1) == 1
                and abs(t2 - prev_t2) == 1
            ):
                block_length += 1
            else:
                if block_length > min_block_length:
                    rtimes_dict[prev_rtime].append(block_length)
                    num_blocks += 1
                block_length = 1
            prev_t1, prev_t2, prev_rtime = t1, t2, rtime
            if block_length > min_block_length * 2:
                rtimes_is_valid = True
                break
            if num_blocks >= 2:  # if valid, save computation and break
                rtimes_is_valid = True
                break
        if not rtimes_is_valid:
            return True

    return False


def check_lyapunov_exponent(traj: np.ndarray, traj_len: int = 100) -> bool:
    """
    Check if the Lyapunov exponent of the trajectory is greater than 1.
    Args:
        traj (ndarray): 2D array of shape (num_vars, num_timepoints), where each row is a time series.
    Returns:
        bool: False if the Lyapunov exponent is less than 1, True otherwise.
    """
    # TODO: debug this, the rosenstein implementation expects univariate time series, not broadcastable
    lyapunov_exponent = max_lyapunov_exponent_rosenstein(
        traj.T, trajectory_len=traj_len
    )
    if lyapunov_exponent < 0:
        return False
    return True


def check_power_spectrum(
    traj: np.ndarray,
    rel_peak_height: float = 1e-4,
    rel_prominence: float = 1e-4,
    min_peaks: int = 3,
) -> bool:
    """Check if a multi-dimensional trajectory has characteristics of chaos via power spectrum.

    Args:
        traj: Array of shape (num_vars, num_timepoints)
        rel_peak_height: Minimum relative peak height threshold
        rel_prominence: Minimum relative peak prominence threshold
        min_peaks: Minimum number of significant peaks for chaos

    Returns:
        True if the system exhibits chaotic characteristics
    """
    power = np.abs(rfft(traj, axis=1)) ** 2  # type: ignore

    power_maxes = power.max(axis=1)
    power_mins = power.min(axis=1)

    peaks_per_dim = [
        find_peaks(
            power[dim],
            height=max(rel_peak_height * power_maxes[dim], power_mins[dim]),
            prominence=max(rel_prominence * power_maxes[dim], power_mins[dim]),
        )[0]
        for dim in range(power.shape[0])
    ]

    return any(len(peaks) >= min_peaks for peaks in peaks_per_dim)


def check_not_linear(
    traj: np.ndarray, r2_threshold: float = 0.98, eps: float = 1e-10
) -> bool:
    """Check if n-dimensional trajectory follows a straight line using PCA.

    Args:
        traj: Array of shape (num_dims, num_timepoints)
        r2_threshold: Variance explained threshold above which trajectory is considered linear
        eps: Small value to prevent division by zero

    Returns:
        bool: False if trajectory is linear, True otherwise
    """
    points = traj.T  # (num_timepoints, num_dims)

    if np.any(~np.isfinite(points)):
        return False

    mean = np.nanmean(points, axis=0)
    std = np.nanstd(points, axis=0)
    std = np.where(std < eps, eps, std)

    standardized_points = (points - mean) / std

    try:
        _, s, _ = np.linalg.svd(standardized_points, full_matrices=False)
        explained_variance_ratio = s**2 / (np.sum(s**2) + eps)
        res = explained_variance_ratio[0] <= r2_threshold
        return res
    except Exception as e:
        print(f"Error in check_not_linear: {e}")
        return True  # fallback if SVD fails


def check_stationarity(traj: np.ndarray, p_value: float = 0.05) -> bool:
    """
    ADF tests for presence of a unit root, with null hypothesis that time_series is non-stationary.
    KPSS tests for stationarity around a constant (or deterministic trend), with null hypothesis that time_series is stationary.

    Args:
        traj (ndarray): 2D array of shape (num_vars, num_timepoints), where each row is a time series.
        p_value: float = 0.05, significance level for stationarity tests

    Returns:
        bool: True if the trajectory is stationary, False otherwise.
    """
    with warnings.catch_warnings():  # kpss test is annoyingly verbose
        warnings.filterwarnings("ignore", "The test statistic is outside of the range")

        for d in range(traj.shape[0]):
            coord = traj[d, :]

            try:
                result_adf = adfuller(coord, autolag="AIC")
                result_kpss = kpss(coord, regression="c")
            except ValueError:  # probably due to constant values
                return False

            status_adf = result_adf[1] < p_value
            status_kpss = result_kpss[1] >= p_value

            if not status_adf and not status_kpss:
                return False

    return True


def check_zero_one_test(
    traj: np.ndarray,
    threshold: float = 0.5,
    strategy: Literal["median", "mean", "score"] = "median",
) -> bool:
    """
    Compute the zero-one test for a specified system.
    If any dimension is chaotic according to the zero-one test, we soft-pass the system as chaotic.

    Parameters:
        trajectories: np.ndarray of shape (n_samples, n_dims, timesteps)
        threshold: float, threshold on the median of the zero-one test to decide if the system is chaotic
    Returns:
        bool, True if the system is chaotic, False otherwise
    """
    # standard_traj = safe_standardize(traj)
    # go dimension by dimension
    agg_fn = np.median if strategy == "median" else np.mean
    if strategy == "score":
        agg_fn = lambda x: np.sum(x >= threshold) / len(x)

    for dim in range(traj.shape[0]):
        timeseries = traj[dim, :].squeeze()
        K_vals = run_zero_one_sweep(
            timeseries, c_min=np.pi / 5, c_max=4 * np.pi / 5, k=1, n_runs=100
        )
        if agg_fn(K_vals) >= threshold:
            return True
    return False


def check_smooth(
    traj: np.ndarray, freq_threshold: float = 0.3, jump_std_factor: float = 3.0
) -> bool:  # type: ignore
    """
    Check if a multi-dimensional trajectory is smooth.
    """
    pass
