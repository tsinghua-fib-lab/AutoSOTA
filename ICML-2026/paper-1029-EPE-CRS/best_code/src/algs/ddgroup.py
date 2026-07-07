import numpy as np
from sklearn.neighbors import NearestNeighbors
from utils.functions import sigmoid
from joblib import Parallel, delayed
from tqdm import tqdm
from utils.subgroup import *


def core_group(X_adjust, X_subgp, Y, p, metric, n_jobs=-1, **kwargs):
    """
    Finds a core group of k nearest neighbors (in subgroup space) with lowest score.

    Parameters
    ----------
    X_adjust : array
        Features for Cox model
    X_subgp : array
        Features for computing neighborhoods
    Y : array
        Outcomes
    p : float
        Fraction of points to use as neighborhood size
    metric : function
        Metric function that takes (X_adjust, Y) and returns score

    Returns
    -------
    core_ind : list
        Indices of core group points
    """
    n = X_subgp.shape[0]
    k = int(p * n)
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_subgp)
    _, indices = nbrs.kneighbors(X_subgp)
    assert len(indices) == n

    def evaluate(i):
        nbhd = sorted(indices[i])
        X_adjust_nbhd = X_adjust[nbhd]
        Y_nbhd = Y[nbhd]
        score = metric(X_adjust_nbhd, Y_nbhd, **kwargs)
        return score, nbhd

    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate)(i) for i in tqdm(range(n))
    )

    best_score, core_ind = min(results, key=lambda x: x[0])
    return core_ind


def reject_points(X_adjust, Y, core_ind, score, threshold):
    """
    Reject points based on score threshold.

    Returns boolean array where True = rejected.
    """
    X_adjust_core = X_adjust[core_ind]
    Y_core = Y[core_ind]
    beta = fit_cox(X_adjust_core, Y_core)
    scores = np.array([score(x, y, X_adjust_core, Y_core, beta) for x, y in tqdm(zip(X_adjust, Y))])
    abs_threshold = np.quantile(scores, threshold)
    labels = scores < abs_threshold
    return labels


def get_scores_parallel(X_adjust, Y, beta, core_ind, score, n_jobs=-1, **kwargs):
    n = len(X_adjust)
    X_adjust_core = X_adjust[core_ind]
    Y_core = Y[core_ind]

    def evaluate(i):
        test_x = X_adjust[i]
        test_y = Y[i]
        s = score(test_x, test_y, X_adjust_core, Y_core, beta)
        return s

    scores = Parallel(n_jobs=n_jobs)(
        delayed(evaluate)(i) for i in tqdm(range(n))
    )
    return scores


def directed_infty_norm(x, S):
    best = 0
    for j in range(len(x)):
        if S[j] != set():
            best = max(best, max([x[j] * s for s in S[j]]))
    return best


def grow_region(X_subgp, labels, B_subgp, center=None, speeds=None, tol=1e-5, shrinkage=0):
    """
    Grow region in subgroup space.

    Parameters
    ----------
    X_subgp : array, shape (n, n_subgp_features)
        Features in subgroup space
    labels : bool array
        True for points to include in region
    B_subgp : array, shape (2, n_subgp_features)
        Bounding box for subgroup space

    Returns
    -------
    R : array, shape (2, n_subgp_features)
        Grown region
    """
    X2 = X_subgp[labels].copy()
    n, d = X2.shape
    R = B_subgp.copy()

    if center is not None:
        X2 -= center
        R -= center.reshape((1, d))
    if speeds is not None:
        assert len(speeds) == d
        for j in range(d):
            X2[:, j] /= speeds[j]
            R[:, j] /= speeds[j]

    S = [set([-1, 1]) for j in range(d)]
    while X2.any() and S != [set() for _ in range(d)]:
        directed_infty_norms = [directed_infty_norm(x, S) for x in X2]
        i = np.argmin(directed_infty_norms)
        j = list(np.abs(X2[i]) == directed_infty_norms[i]).index(True)
        sign = int(np.sign(X2[i, j]))

        S[j].remove(sign)
        R[int((sign + 1) / 2), j] = X2[i, j]

        X2 = X2[[k for k in range(len(X2)) if sign * X2[k, j] < directed_infty_norms[i] + shrinkage]]

    if speeds is not None:
        for j in range(d):
            R[:, j] *= speeds[j]

    if center is not None:
        R += center.reshape((1, d))

    if np.linalg.norm(R - B_subgp) <= tol:
        return B_subgp

    return R


def ddgroup_job(X_adjust, X_subgp, Y, B_subgp, core_sizes, rejection_thresholds):
    """
    DDGroup algorithm with EPE metric.
    """
    results = []
    for core_size in core_sizes:
        core_ind = core_group(X_adjust, X_subgp, Y, core_size, epe_metric)
        beta_hat = fit_cox(X_adjust[core_ind], Y[core_ind])
        scores = get_scores_parallel(X_adjust, Y, beta_hat, core_ind, log_tail_rej_score)

        for t in tqdm(rejection_thresholds):
            abs_threshold = np.nanquantile(scores, t)
            labels = scores < abs_threshold

            # Grow region in subgroup space only
            R = grow_region(X_subgp, labels, B_subgp, center=np.mean(X_subgp[core_ind], axis=0))

            # Find points in region and fit Cox model
            ind = in_region(X_subgp, R)

            try:
                beta = fit_cox(X_adjust[ind], Y[ind])
                results.append({
                    'subgroup_id': 0,
                    'R': R.copy(),
                    'beta': beta.copy(),
                    'core_size': core_size,
                    'rejection_threshold': t
                })
            except:
                print(f"Failed to fit Cox model for core size {core_size} and rejection threshold {t}.")
                results.append({
                    'subgroup_id': 0,
                    'R': R.copy(),
                    'beta': np.full(X_adjust.shape[1], np.nan),
                    'core_size': core_size,
                    'rejection_threshold': t
                })
    return results


def c_ind_ddgroup_job(X_adjust, X_subgp, Y, B_subgp, core_sizes, rejection_thresholds):
    """
    DDGroup algorithm with C-index metric.
    """
    results = []
    for core_size in core_sizes:
        core_ind = core_group(X_adjust, X_subgp, Y, core_size, c_ind_metric)
        beta_hat = fit_cox(X_adjust[core_ind], Y[core_ind])
        scores = get_scores_parallel(X_adjust, Y, beta_hat, core_ind, c_ind_rej_score)

        for t in tqdm(rejection_thresholds):
            abs_threshold = np.nanquantile(scores, t)
            labels = scores < abs_threshold

            R = grow_region(X_subgp, labels, B_subgp, center=np.mean(X_subgp[core_ind], axis=0))
            ind = in_region(X_subgp, R)

            try:
                beta = fit_cox(X_adjust[ind], Y[ind])
                results.append({
                    'subgroup_id': 0,
                    'R': R.copy(),
                    'beta': beta.copy(),
                    'core_size': core_size,
                    'rejection_threshold': t
                })
            except:
                print(f"Failed to fit Cox model for core size {core_size} and rejection threshold {t} (C-index DDGroup).")
                results.append({
                    'subgroup_id': 0,
                    'R': R.copy(),
                    'beta': np.full(X_adjust.shape[1], np.nan),
                    'core_size': core_size,
                    'rejection_threshold': t
                })
    return results


def pl_ddgroup_job(X_adjust, X_subgp, Y, B_subgp, core_sizes, rejection_thresholds):
    """
    DDGroup algorithm with partial likelihood metric.
    """
    results = []
    for core_size in core_sizes:
        core_ind = core_group(X_adjust, X_subgp, Y, core_size, pl_metric)
        beta_hat = fit_cox(X_adjust[core_ind], Y[core_ind])
        scores = get_scores_parallel(X_adjust, Y, beta_hat, core_ind, pl_rej_score)

        for t in tqdm(rejection_thresholds):
            abs_threshold = np.nanquantile(scores, t)
            labels = scores < abs_threshold

            R = grow_region(X_subgp, labels, B_subgp, center=np.mean(X_subgp[core_ind], axis=0))
            ind = in_region(X_subgp, R)

            try:
                beta = fit_cox(X_adjust[ind], Y[ind])
                results.append({
                    'subgroup_id': 0,
                    'R': R.copy(),
                    'beta': beta.copy(),
                    'core_size': core_size,
                    'rejection_threshold': t
                })
            except:
                print(f"Failed to fit Cox model for core size {core_size} and rejection threshold {t} (PL DDGroup).")
                results.append({
                    'subgroup_id': 0,
                    'R': R.copy(),
                    'beta': np.full(X_adjust.shape[1], np.nan),
                    'core_size': core_size,
                    'rejection_threshold': t
                })
    return results


def no_exp_ddgroup_job(X_adjust, X_subgp, Y, B_subgp, core_size):
    """
    DDGroup without expansion - just use bounding box of core group.
    """
    results = []

    core_ind = core_group(X_adjust, X_subgp, Y, core_size, epe_metric)

    # Bounding box in subgroup space only
    R = bounding_box(X_subgp[core_ind]).reshape(2, -1)
    ind = in_region(X_subgp, R)

    try:
        beta = fit_cox(X_adjust[ind], Y[ind])
        results.append({
            'subgroup_id': 0,
            'R': R.copy(),
            'beta': beta.copy(),
            'core_size': core_size
        })
    except:
        print(f"Failed to fit Cox model for core size {core_size}. (No expansion)")
        results.append({
            'subgroup_id': 0,
            'R': R.copy(),
            'beta': np.full(X_adjust.shape[1], np.nan),
            'core_size': core_size
        })
    return results