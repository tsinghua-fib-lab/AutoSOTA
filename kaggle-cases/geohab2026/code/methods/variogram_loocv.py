"""
Multi-lag indicator variogram fitting and per-class spatial LOOCV screening.

Companion code for writeups/loocv_spatial_cv.md.

Two public functions:
    fit_indicator_variogram  -- fit a variogram model at a given max-lag window
    screen_loocv_folds       -- pre-flight fold-quality check before LOOCV

Dependencies: numpy, scipy
"""

import numpy as np
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Variogram models
# ---------------------------------------------------------------------------

def gaussian_variogram(h, nugget, sill, range_a):
    """Gaussian variogram model."""
    return nugget + (sill - nugget) * (1 - np.exp(-3 * (h / range_a) ** 2))


def spherical_variogram(h, nugget, sill, range_a):
    """Spherical variogram model."""
    h = np.asarray(h, dtype=float)
    return np.where(
        h <= range_a,
        nugget + (sill - nugget) * (1.5 * h / range_a - 0.5 * (h / range_a) ** 3),
        nugget + (sill - nugget),
    )


# ---------------------------------------------------------------------------
# Indicator variogram fitting
# ---------------------------------------------------------------------------

def fit_indicator_variogram(coords, indicator, xlag, xltol, nlag, min_pairs=30):
    """
    Compute an omnidirectional experimental indicator variogram and fit a model.

    Tries Gaussian and spherical models; returns the better fit by RMSE.

    Parameters
    ----------
    coords : (N, 2) array
        Point coordinates in metres (projected CRS).
    indicator : (N,) array of int or bool
        Binary indicator: 1 where point belongs to the target class, 0 elsewhere.
    xlag : float
        Lag bin spacing in metres.
    xltol : float
        Half-width of each lag bin in metres (tolerance).
    nlag : int
        Number of lag bins. Max experimental lag = nlag * xlag.
    min_pairs : int
        Minimum number of pairs required to compute gamma for a bin.
        Bins below this threshold are set to NaN and excluded from fitting.

    Returns
    -------
    dict or None
        Keys: lag_centers, gamma, npairs, nugget, sill, range, model,
              bound_hit, upper_bound.
        Returns None if fewer than 3 valid bins exist after NaN filtering.

    Notes
    -----
    Uses upper-triangle pair enumeration (np.triu_indices) rather than the
    full N×N distance matrix. Suitable for N up to ~10,000; for larger N
    consider cKDTree-based pair enumeration with a distance cutoff.

    The upper fitting bound for the range parameter is set to
    1.25 × max(lag_centers), which is the max experimental lag plus one
    bin-width margin. Fits where the returned range exceeds 0.95 × upper_bound
    are flagged as bound_hit=True and should not be used as LOOCV exclusion
    radii without further investigation.
    """
    indicator = np.asarray(indicator, dtype=np.int8)
    n = len(coords)

    # Pairwise distances -- upper triangle only
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = coords[i_idx, 0] - coords[j_idx, 0]
    dy = coords[i_idx, 1] - coords[j_idx, 1]
    dists = np.sqrt(dx ** 2 + dy ** 2)
    diff_sq = (indicator[i_idx].astype(np.int32) - indicator[j_idx].astype(np.int32)) ** 2

    # Bin into lags
    lag_centers = np.arange(1, nlag + 1) * xlag
    gamma = np.full(nlag, np.nan)
    npairs = np.zeros(nlag, dtype=int)

    for k, center in enumerate(lag_centers):
        mask = np.abs(dists - center) <= xltol
        npairs[k] = int(mask.sum())
        if npairs[k] >= min_pairs:
            gamma[k] = 0.5 * float(diff_sq[mask].mean())

    valid = ~np.isnan(gamma)
    if valid.sum() < 3:
        return None

    h_valid = lag_centers[valid]
    g_valid = gamma[valid]
    upper_bound = float(h_valid[-1] * 1.25)

    var_I = float(indicator.var())
    p0 = [0.2 * var_I, 0.8 * var_I, upper_bound * 0.5]
    bounds = ([0.0, 1e-8, xlag], [var_I + 1e-8, 2 * var_I + 1e-6, upper_bound])

    best_model, best_popt, best_rmse = None, None, np.inf
    for name, fn in [("gaussian", gaussian_variogram), ("spherical", spherical_variogram)]:
        try:
            popt, _ = curve_fit(fn, h_valid, g_valid, p0=p0, bounds=bounds, maxfev=5000)
            rmse = float(np.sqrt(np.mean((fn(h_valid, *popt) - g_valid) ** 2)))
            if rmse < best_rmse:
                best_model, best_popt, best_rmse = name, popt, rmse
        except RuntimeError:
            pass

    if best_popt is None:
        return None

    fitted_range = float(best_popt[2])
    return {
        "lag_centers": h_valid,
        "gamma": g_valid,
        "npairs": npairs[valid],
        "nugget": float(best_popt[0]),
        "sill": float(best_popt[0] + best_popt[1]),
        "range": fitted_range,
        "model": best_model,
        "rmse": best_rmse,
        "bound_hit": fitted_range > 0.95 * upper_bound,
        "upper_bound": upper_bound,
    }


# ---------------------------------------------------------------------------
# Multi-lag stability check
# ---------------------------------------------------------------------------

def assess_radius_stability(range_a, range_b, stable_threshold=0.15, borderline_threshold=0.25):
    """
    Classify range stability between two fitting windows A and B.

    Parameters
    ----------
    range_a : float
        Fitted range from the shorter window (Run A).
    range_b : float
        Fitted range from the longer window (Run B).
    stable_threshold : float
        |r_A - r_B| / r_A <= this → 'stable'.
    borderline_threshold : float
        stable_threshold < ratio <= this → 'borderline'; else 'unstable'.

    Returns
    -------
    dict with keys: ratio, band ('stable' | 'borderline' | 'unstable').
    """
    if range_a is None or range_b is None:
        return {"ratio": None, "band": "unstable"}
    ratio = abs(range_a - range_b) / range_a
    if ratio <= stable_threshold:
        band = "stable"
    elif ratio <= borderline_threshold:
        band = "borderline"
    else:
        band = "unstable"
    return {"ratio": float(ratio), "band": band}


# ---------------------------------------------------------------------------
# Pre-flight LOOCV screening
# ---------------------------------------------------------------------------

def screen_loocv_folds(coords, labels, class_radii,
                       min_fold_absolute=100, min_class_reps=10):
    """
    Pre-flight screening for per-class spatial LOOCV.

    Computes the full N×N pairwise distance matrix once (VanderPlas broadcasting
    pattern, stored as float32) then evaluates eligible training set size and
    per-class representation for every candidate held-out point.

    Parameters
    ----------
    coords : (N, 2) array
        Training point coordinates in metres.
    labels : (N,) array
        Class label for each training point.
    class_radii : dict
        Mapping from class label to exclusion radius in metres.
    min_fold_absolute : int
        Flag a fold if its eligible training set has fewer than this many points.
    min_class_reps : int
        Flag a fold if any class has fewer than this many representatives in the
        eligible training set.

    Returns
    -------
    list of dict
        One entry per training point. Keys:
            idx, held_out_class, radius_used, n_eligible,
            class_counts (dict), flags (list of str).

    Notes
    -----
    Memory: N×N float32 requires ~4 N² bytes (~147 MB for N=6256).
    For N > 8000 consider chunked computation or cKDTree-based alternatives.
    """
    coords = np.asarray(coords, dtype=np.float64)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    n = len(coords)

    # Full pairwise distance matrix: float64 differences → float32 storage
    diff = coords.reshape(n, 1, 2) - coords
    dist_matrix = np.sqrt((diff ** 2).sum(axis=2)).astype(np.float32)

    fold_reports = []
    for i in range(n):
        held_out_class = labels[i]
        radius = class_radii[held_out_class]

        excluded = dist_matrix[i] <= radius
        excluded[i] = True       # always exclude self
        eligible = ~excluded

        n_eligible = int(eligible.sum())
        class_counts = {c: int(((labels == c) & eligible).sum()) for c in classes}

        min_reps = min(class_counts.values())
        worst_class = min(class_counts, key=class_counts.get)

        flags = []
        if n_eligible < min_fold_absolute:
            flags.append(f"MIN_FOLD_ABSOLUTE: {n_eligible} < {min_fold_absolute}")
        if min_reps < min_class_reps:
            flags.append(f"MIN_CLASS_REPS: {worst_class}={min_reps} < {min_class_reps}")

        fold_reports.append({
            "idx": i,
            "held_out_class": held_out_class,
            "radius_used": radius,
            "n_eligible": n_eligible,
            "class_counts": class_counts,
            "flags": flags,
        })

    return fold_reports
