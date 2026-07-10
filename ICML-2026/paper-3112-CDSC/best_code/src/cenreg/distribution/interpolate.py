import numpy as np


def linear_interpolation(kx: np.ndarray, ky: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Linear interpolation.

    Parameters
    ----------
    kx : np.ndarray
        One-dimensional or two-dimensional array containing the x-coordinates
        of the data points.
    ky : np.ndarray
        One-dimensional or two-dimensional array containing the y-coordinates
        of the data points.
    x : np.ndarray
        One-dimensional or two-dimensional array containing the x-coordinates
        where to evaluate the interpolated values.

    Returns
    -------
    ret : np.ndarray
        One-dimensional or two-dimensional array containing the interpolated values.
    """

    if kx.ndim == 1:
        if ky.ndim == 2:
            if kx.shape[0] != ky.shape[1]:
                raise ValueError("kx and ky must have compatible shapes")

    # compute idx and ratio
    if kx.ndim == 1:
        idx = np.searchsorted(kx, x, side="right")
        idx = np.clip(idx, 1, len(kx) - 1)
        lb = kx[idx - 1]
        ub = kx[idx]
    else:
        # @note this part may be improved by using np.apply_along_axis (not verified)
        # idx = np.diag(np.apply_along_axis(np.searchsorted, 1, kx, x.reshape(-1))).reshape(-1,1)
        list_lb = []
        list_ub = []
        list_idx = []
        if x.ndim == 1:
            for i in range(kx.shape[0]):
                idx = np.searchsorted(kx[i], x, side="right")
                idx = np.clip(idx, 1, kx.shape[1] - 1)
                lb = kx[i, idx - 1]
                ub = kx[i, idx]
                list_lb.append(lb.reshape(1, -1))
                list_ub.append(ub.reshape(1, -1))
                list_idx.append(idx.reshape(1, -1))
        else:
            for i in range(kx.shape[0]):
                idx = np.searchsorted(kx[i], x[i], side="right")
                idx = np.clip(idx, 1, kx.shape[1] - 1)
                lb = kx[i, idx - 1]
                ub = kx[i, idx]
                list_lb.append(lb.reshape(1, -1))
                list_ub.append(ub.reshape(1, -1))
                list_idx.append(idx.reshape(1, -1))
        lb = np.concatenate(list_lb, 0)
        ub = np.concatenate(list_ub, 0)
        idx = np.concatenate(list_idx, 0)
    denominator = np.clip(ub - lb, 0.0001, np.inf)
    numerator = np.clip(x - lb, 0.0, ub - lb)
    ratio = numerator / denominator

    # linear interpolation
    if ky.ndim == 1:
        left = ky[idx - 1]
        right = ky[idx]
    elif idx.ndim == 1:
        left = np.take(ky, idx - 1, -1)
        right = np.take(ky, idx, -1)
    else:
        left = np.take_along_axis(ky, idx - 1, -1)
        right = np.take_along_axis(ky, idx, -1)
    return left + ratio * (right - left)
