import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from utils.functions import sigmoid, EPE
from utils.subgroup import in_region, log_tail_rej_score
from scipy.special import log_softmax
from sksurv.metrics import concordance_index_censored
from config.constants import METRIC_COLS

def box_intersection(B1, B2, eps=1e-10):
    """
    Compute intersection volume of two boxes.

    Handles degenerate dimensions (zero-width) by treating them as constraints.
    If a dimension is degenerate in one box, checks if the other box intersects
    that constraint hyperplane, then computes volume in non-degenerate dimensions.

    Parameters
    ----------
    B1, B2 : array, shape (2, d)
        Box bounds [lower, upper]
    eps : float, default=1e-10
        Threshold for considering a dimension degenerate

    Returns
    -------
    float
        Intersection volume (in non-degenerate dimensions)
    """
    assert len(B1.shape) == 2
    assert len(B2.shape) == 2
    assert B1.shape == B2.shape

    vol = 1.
    n_nondegen = 0

    for i in range(B1.shape[1]):
        width1 = B1[1, i] - B1[0, i]
        width2 = B2[1, i] - B2[0, i]

        # Check if either dimension is degenerate
        degen1 = width1 < eps
        degen2 = width2 < eps

        if degen1 and degen2:
            # Both degenerate: check if they're at the same location
            if abs(B1[0, i] - B2[0, i]) > eps:
                return 0  # No intersection
            # Otherwise, constraint matches, continue
        elif degen1:
            # B1 degenerate: check if B2 contains the constraint value
            constraint_val = B1[0, i]
            if constraint_val < B2[0, i] - eps or constraint_val > B2[1, i] + eps:
                return 0  # B2 doesn't intersect the constraint
            # Otherwise, constraint satisfied, continue
        elif degen2:
            # B2 degenerate: check if B1 contains the constraint value
            constraint_val = B2[0, i]
            if constraint_val < B1[0, i] - eps or constraint_val > B1[1, i] + eps:
                return 0  # B1 doesn't intersect the constraint
            # Otherwise, constraint satisfied, continue
        else:
            # Both non-degenerate: standard intersection
            lower = max(B1[0, i], B2[0, i])
            upper = min(B1[1, i], B2[1, i])
            if lower >= upper:
                return 0
            vol *= (upper - lower)
            n_nondegen += 1

    # If all dimensions were degenerate, return 0 volume
    if n_nondegen == 0:
        return 0.0

    return vol


def precision(R_hat, R):
    return box_intersection(R_hat, R) / box_intersection(R_hat, R_hat)


def recall(R_hat, R):
    return box_intersection(R_hat, R) / box_intersection(R, R)


def f1(R_hat, R):
    if R_hat is None or R is None:
      return  np.nan  # real data does not have R_star
    return 2. / ((1. / recall(R_hat, R)) + (1. / precision(R_hat, R)))


def distributional_precision(X_subgp, R_hat, R_star):
    true_labels_1 = in_region(X_subgp, R_star)
    true_labels_2 = ~true_labels_1
    pred_labels = in_region(X_subgp, R_hat)
    return max(np.sum(true_labels_1 & pred_labels) / np.sum(pred_labels), np.sum(true_labels_2 & pred_labels) / np.sum(pred_labels))


def distributional_recall(X_subgp, R_hat, R_star):
    true_labels_1 = in_region(X_subgp, R_star)
    true_labels_2 = ~true_labels_1
    pred_labels = in_region(X_subgp, R_hat)
    return max(np.sum(true_labels_1 & pred_labels) / np.sum(true_labels_1), np.sum(true_labels_2 & pred_labels) / np.sum(true_labels_2))


def distributional_f1(X_subgp, R_hat, R_star):
    true_labels_1 = in_region(X_subgp, R_star)
    true_labels_2 = ~true_labels_1
    pred_labels = in_region(X_subgp, R_hat)
    return max(2 * np.sum(true_labels_1 & pred_labels) / (np.sum(true_labels_1) + np.sum(pred_labels)),
               2 * np.sum(true_labels_2 & pred_labels) / (np.sum(true_labels_2) + np.sum(pred_labels)))


def distributional_iou(X_subgp, R_hat, R_star):
    true_labels_1 = in_region(X_subgp, R_star)
    true_labels_2 = ~true_labels_1
    pred_labels = in_region(X_subgp, R_hat)
    return max(np.sum(true_labels_1 & pred_labels) / np.sum(true_labels_1 | pred_labels),
               np.sum(true_labels_2 & pred_labels) / np.sum(true_labels_2 | pred_labels))


def box_volume(R, eps=1e-10):
    """
    Compute volume (hyperrectangle measure) of a box.

    Parameters
    ----------
    R : array, shape (2, d) or None
        Box bounds [lower, upper]
    eps : float, default=1e-10
        Threshold for considering a dimension degenerate (zero-width)

    Returns
    -------
    float
        Volume of the box (product of widths along non-degenerate dimensions)

    Notes
    -----
    If a dimension has width < eps, it is considered degenerate and excluded
    from volume calculation. This allows handling lower-dimensional manifolds
    (e.g., hyperplanes) embedded in higher-dimensional space.
    """
    if R is None:
        return 0.0

    vol = 1.0
    n_nondegen = 0
    for i in range(R.shape[1]):
        width = R[1, i] - R[0, i]
        if width < -eps:  # Invalid box
            return 0.0
        elif width > eps:  # Non-degenerate dimension
            vol *= width
            n_nondegen += 1

    # If all dimensions are degenerate, treat as zero volume
    if n_nondegen == 0:
        return 0.0

    return vol


def box_union_volume(R1, R2, eps=1e-10):
    """
    Compute volume of union of two boxes.

    Uses inclusion-exclusion principle:
    |A ∪ B| = |A| + |B| - |A ∩ B|

    Handles degenerate dimensions by projecting both boxes to the same
    dimensional subspace before computing volumes.

    Parameters
    ----------
    R1, R2 : array, shape (2, d)
        Box bounds
    eps : float, default=1e-10
        Threshold for considering a dimension degenerate

    Returns
    -------
    float
        Volume of union (in non-degenerate dimensions)

    Notes
    -----
    When boxes have different degeneracy patterns:
    - Identifies dimensions that are degenerate in EITHER box
    - Checks constraint satisfaction in degenerate dimensions
    - Computes volumes only in dimensions non-degenerate in BOTH boxes
    - This ensures volumes are computed in the same dimensional subspace
    """
    # Check if boxes can be unioned (must intersect in degenerate dimensions)
    intersection_vol = box_intersection(R1, R2, eps=eps)

    # If no intersection, union is not well-defined for degenerate cases
    if intersection_vol == 0:
        # Could be no overlap, or constraints not satisfied
        # In either case, return sum of individual volumes
        vol_r1 = box_volume(R1, eps=eps)
        vol_r2 = box_volume(R2, eps=eps)
        return vol_r1 + vol_r2

    # Identify which dimensions are non-degenerate in BOTH boxes
    # Only count volume in these shared non-degenerate dimensions
    vol_r1 = 1.0
    vol_r2 = 1.0
    n_shared_nondegen = 0

    for i in range(R1.shape[1]):
        width1 = R1[1, i] - R1[0, i]
        width2 = R2[1, i] - R2[0, i]

        # Only count dimension if non-degenerate in BOTH
        if width1 > eps and width2 > eps:
            vol_r1 *= width1
            vol_r2 *= width2
            n_shared_nondegen += 1

    if n_shared_nondegen == 0:
        return 0.0

    return vol_r1 + vol_r2 - intersection_vol


def iou(R1, R2, eps=1e-10):
    """
    Intersection over Union (Jaccard index) for two boxes.

    IoU = |R1 ∩ R2| / |R1 ∪ R2|

    Handles degenerate dimensions (zero-width) by treating them as constraints
    and computing IoU only in non-degenerate dimensions.

    Parameters
    ----------
    R1, R2 : array, shape (2, d) or None
        Box bounds
    eps : float, default=1e-10
        Threshold for considering a dimension degenerate

    Returns
    -------
    float
        IoU score in [0, 1]. Returns 0 if either box is None or union is empty.

    Notes
    -----
    IoU is a standard metric in object detection and region overlap evaluation.
    - IoU = 1: Perfect overlap
    - IoU = 0: No overlap
    - IoU >= 0.5: Typically considered "good" overlap in computer vision
    - IoU >= 0.8: Very strong overlap (our default threshold for ground truth recovery)

    For degenerate dimensions (e.g., hyperplanes where lower=upper):
    - Checks if the non-degenerate box intersects the constraint
    - Computes IoU only in remaining non-degenerate dimensions
    """
    if R1 is None or R2 is None:
        return 0.0

    intersection = box_intersection(R1, R2, eps=eps)
    union = box_union_volume(R1, R2, eps=eps)

    if union == 0:
        return 0.0

    return intersection / union


def find_best_matching_region(R_discovered, gt_regions, metric='iou'):
    """
    Find ground truth region with best overlap with discovered region.

    Parameters
    ----------
    R_discovered : array, shape (2, d)
        Discovered region bounds
    gt_regions : dict
        Dictionary of {name: gt_region_bounds}
    metric : str, default='iou'
        Metric to use for matching. Options: 'iou' or 'f1'

    Returns
    -------
    best_match : str or None
        Name of best matching GT region (None if no regions provided)
    best_score : float
        Score with best match
    all_scores : dict
        Scores with all GT regions {name: score}

    Example
    -------
    >>> R_disc = np.array([[0, 0], [10, 10]])
    >>> gt_regions = {
    ...     'A': np.array([[-5, -5], [5, 5]]),
    ...     'B': np.array([[8, 8], [15, 15]])
    ... }
    >>> best, score, all_scores = find_best_matching_region(R_disc, gt_regions)
    >>> print(f"Best match: {best}, IoU: {score:.2f}")
    """
    best_match = None
    best_score = 0.0
    all_scores = {}

    for name, R_gt in gt_regions.items():
        if metric == 'iou':
            score = iou(R_discovered, R_gt)
        elif metric == 'f1':
            score = f1(R_discovered, R_gt)
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'iou' or 'f1'.")

        all_scores[name] = score
        if score > best_score:
            best_score = score
            best_match = name

    return best_match, best_score, all_scores


def subgp_size(X_adjust, X_subgp, Y, R, beta):
    return sum(in_region(X_subgp, R)) / len(X_subgp)


def c_ind(X_adjust, X_subgp, Y, R, beta):
    ind = in_region(X_subgp, R)
    region_X_adjust = X_adjust[ind]
    region_Y = Y[ind]
    event_indicator = region_Y['failure']
    event_time = region_Y['time']
    estimate = region_X_adjust @ beta
    return concordance_index_censored(event_indicator, event_time, estimate)[0]


def epe(X_adjust, X_subgp, Y, R, beta):
    ind = in_region(X_subgp, R)
    return EPE(X_adjust[ind], Y[ind], beta)


def pll(X_adjust, X_subgp, Y, R, beta):
    """
    The input data X are assumed to be sorted by event time (censoring or failure).
    cens = Y['failure'] is the boolean vector of censoring indicators,
    cens[i] = 1{X[i] failed (wasn't censored)}.
    """
    ind = in_region(X_subgp, R)
    region_X_adjust = X_adjust[ind]
    region_Y = Y[ind]
    failure_indices = np.where(region_Y['failure'])[0]
    return sum([log_softmax(region_X_adjust[i:] @ beta)[0] for i in failure_indices])


def c_ind_p_val(X_adjust, X_subgp, Y, R, beta):
    ind = in_region(X_subgp, R)
    in_X_adjust = X_adjust[ind]
    in_Y = Y[ind]
    n = len(in_X_adjust)
    perm = np.random.permutation(n)
    C = 0 # Using notation from the paper
    mu = 0
    for i in range(n // 2):
        times = (in_Y['time'][perm[2 * i]], in_Y['time'][perm[2 * i + 1]])
        cens = (in_Y['failure'][perm[2 * i]], in_Y['failure'][perm[2 * i + 1]])
        pred = (np.dot(beta, in_X_adjust[perm[2 * i]]), np.dot(beta, in_X_adjust[perm[2 * i + 1]]))
        first = np.argmin(times)
        if cens[first]: # Only use pairs where the earlier event time is uncensored
            p = np.max([sigmoid(pred[0] - pred[1]), sigmoid(pred[1] - pred[0])])
            mu += p
            if pred[first] > pred[1 - first]:
                C += 1
    if mu == 0:
        return 1.
    else:
        delta = C / mu - 1.
        if delta >= 0:
            p_val = np.exp(-(delta ** 2) * mu / (2 + delta))
        else:
            p_val = np.exp(-(delta ** 2) * mu / 2)
    return min(p_val, 1.)


def epe_p_val(X_adjust, X_subgp, Y, R, beta, seed=None):
    ind = in_region(X_subgp, R)
    in_X_adjust = X_adjust[ind]
    in_Y = Y[ind]
    n = len(in_X_adjust)
    if seed is None:
        perm = np.random.permutation(n)
    else:
        perm = np.random.RandomState(seed).permutation(n)
    epe_hat = 0 # Using notation from the paper
    mu = 0
    numerator = 0
    num_valid_pairs = 0
    for i in range(n // 2):
        times = (in_Y['time'][perm[2 * i]], in_Y['time'][perm[2 * i + 1]])
        cens = (in_Y['failure'][perm[2 * i]], in_Y['failure'][perm[2 * i + 1]])
        pred = (np.dot(beta, in_X_adjust[perm[2 * i]]), np.dot(beta, in_X_adjust[perm[2 * i + 1]]))
        first = np.argmin(times)
        if cens[first]:
            num_valid_pairs += 1
            p = sigmoid(pred[first] - pred[1-first])
            epe_hat -= np.log(p)
            mu -= p * np.log(p) + (1-p) * np.log(1-p)
            numerator += p * (1-p) * (np.log(p/(1-p))**2)

    tau = np.abs(epe_hat - mu)
    if num_valid_pairs == 0:
        return 1.
    else:
        return min(numerator / (tau ** 2), 1)
    

def loo_crs_dist(X_adjust, X_subgp, Y, R, beta, score=log_tail_rej_score, n_jobs=-1):
    n = len(X_adjust)
    ind = in_region(X_subgp, R)
    X_adjust_in = X_adjust[ind]
    Y_in = Y[ind]

    def evaluate(i):
        try:
            test_x = X_adjust[i]
            test_y = Y[i]
            s = score(test_x, test_y, X_adjust_in, Y_in, beta)
        except:
            s = np.nan
        return s

    scores = Parallel(n_jobs=n_jobs)(
        delayed(evaluate)(i) for i in range(n)
    )
    return scores


def rej_frac(X_adjust, X_subgp, Y, R, beta, thresholds, score=log_tail_rej_score, n_jobs=-1):
    scores = np.array(loo_crs_dist(X_adjust, X_subgp, Y, R, beta, score, n_jobs))
    return [np.mean(scores < np.log(t)) for t in thresholds]


def run_eval(X_adjust, X_subgp, Y, X_adjust_test, X_subgp_test, Y_test, results, R_star = None, n_jobs=-1):
    """
    Evaluate subgroup discovery results.

    Parameters
    ----------
    X_adjust : array, shape (n_train, n_adjust_features)
        Training features for Cox model adjustment
    X_subgp : array, shape (n_train, n_subgp_features)
        Training features for subgroup definition
    Y : structured array, shape (n_train,)
        Training outcomes
    X_adjust_test : array, shape (n_test, n_adjust_features)
        Test features for Cox model adjustment
    X_subgp_test : array, shape (n_test, n_subgp_features)
        Test features for subgroup definition
    Y_test : structured array, shape (n_test,)
        Test outcomes
    results : list of dict
        Algorithm results containing 'R' and 'beta'
    R_star : array, optional
        True subgroup region (for synthetic data)
    n_jobs : int
        Number of parallel jobs

    Returns
    -------
    results_df : DataFrame
        Evaluation metrics for each result
    """
    def evaluate(r):
        R = r['R']
        beta = r['beta']
        try:
            d = {
                'train_epe': epe(X_adjust, X_subgp, Y, R, beta),
                'train_c_ind': c_ind(X_adjust, X_subgp, Y, R, beta),
                'train_pll': pll(X_adjust, X_subgp, Y, R, beta),
                'train_size': subgp_size(X_adjust, X_subgp, Y, R, beta),
                'train_c_p': c_ind_p_val(X_adjust, X_subgp, Y, R, beta),
                'train_epe_p': epe_p_val(X_adjust, X_subgp, Y, R, beta),
                'train_precision': distributional_precision(X_subgp, R, R_star) if R_star is not None else np.nan,
                'train_recall': distributional_recall(X_subgp, R, R_star) if R_star is not None else np.nan,
                'train_f1': distributional_f1(X_subgp, R, R_star) if R_star is not None else np.nan,
                'train_iou': distributional_iou(X_subgp, R, R_star) if R_star is not None else np.nan,
                'test_epe': epe(X_adjust_test, X_subgp_test, Y_test, R, beta),
                'test_c_ind': c_ind(X_adjust_test, X_subgp_test, Y_test, R, beta),
                'test_pll': pll(X_adjust_test, X_subgp_test, Y_test, R, beta),
                'test_size': subgp_size(X_adjust_test, X_subgp_test, Y_test, R, beta),
                'test_c_p': c_ind_p_val(X_adjust_test, X_subgp_test, Y_test, R, beta),
                'test_epe_p': epe_p_val(X_adjust_test, X_subgp_test, Y_test, R, beta),
                'test_precision': distributional_precision(X_subgp_test, R, R_star) if R_star is not None else np.nan,
                'test_recall': distributional_recall(X_subgp_test, R, R_star) if R_star is not None else np.nan,
                'test_f1': distributional_f1(X_subgp_test, R, R_star) if R_star is not None else np.nan,
                'test_iou': distributional_iou(X_subgp_test, R, R_star) if R_star is not None else np.nan
            }
        except:
            d = {
                'train_epe': np.nan,
                'train_c_ind': np.nan,
                'train_pll': np.nan,
                'train_size': np.nan,
                'train_c_p': np.nan,
                'train_epe_p': np.nan,
                'train_precision': np.nan,
                'train_recall': np.nan,
                'train_f1': np.nan,
                'train_iou': np.nan,
                'test_epe': np.nan,
                'test_c_ind': np.nan,
                'test_pll': np.nan,
                'test_size': np.nan,
                'test_c_p': np.nan,
                'test_epe_p': np.nan,
                'train_precision': np.nan,
                'train_recall': np.nan,
                'train_f1': np.nan,
                'train_iou': np.nan
            }
        return d

    eval_results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate)(r) for r in tqdm(results)
    )
    # results_df = pd.DataFrame(columns=METRIC_COLS, dtype=object)
    results_df = pd.DataFrame(eval_results, columns=METRIC_COLS, dtype=object)
    return results_df