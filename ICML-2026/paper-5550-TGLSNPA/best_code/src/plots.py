"""
Plotting utilities

"""
__date__ = "June 2023 - July 2025"

import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np


def stats_to_colors(stats, r_max=1.0, power=1.5, mode="angles"):
    cmap = colormaps["hsv"]
    
    if mode == "angles":
        z = np.cos(stats) + 1j * np.sin(stats)
    elif mode == "expanded_complex":
        assert stats.shape[-1] == 2, f"{stats.shape}"
        z = stats[...,0] + 1j * stats[...,1]
    elif mode == "complex":
        z = stats
    else:
        raise NotImplementedError(mode)

    c = cmap((np.angle(z) % (2 * np.pi)) / (2 * np.pi))[..., :-1]
    r = (np.abs(z[...,None]) / r_max).clip(0,1) ** power
    white = np.ones(tuple(1 for _ in z.shape) + (3,))
    out = ((1.0 - r) * white + r * c).clip(0,1)
    return out


def a1_to_cov_and_relation(A2):
    """
    Interpret A2 (shape [F, L, R, 2, R, 2]) as the covariance of the stacked real-imag vector [x; y],
    where z = x + i y is complex normal. The 2-axes index real/imag in that order.

    Let the real block covariance be:
        S = Cov([x; y]) = [[S_xx, S_xy],
                           [S_yx, S_yy]]

    Then the complex covariance C and relation P are:
        C = S_xx + S_yy + i (S_yx - S_xy)
        P = S_xx - S_yy + i (S_xy + S_yx)

    Returns
    -------
    C : np.ndarray, complex64/128
        Shape [F, R, L, R], the Hermitian covariance E[ z z^H ].
    P : np.ndarray, complex64/128
        Shape [F, R, L, R], the (generally nonzero) relation E[ z z^T ].

    """
    assert A2.ndim == 6, f"Expected 6D, got {A2.ndim}D"
    F, L, R, two1, R2, two2 = A2.shape
    assert two1 == 2 and two2 == 2, f"Expected ...2 and ...2, got {A2.shape}"
    assert R == R2, f"Mismatched block sizes: {A2.shape}"

    # Real/imag blocks
    S_xx = A2[:, :, :, 0, :, 0]  # rr
    S_xy = A2[:, :, :, 0, :, 1]  # ri
    S_yx = A2[:, :, :, 1, :, 0]  # ir
    S_yy = A2[:, :, :, 1, :, 1]  # ii

    # Complex covariance and relation
    C = (S_xx + S_yy) + 1j * (S_yx - S_xy)
    P = (S_xx - S_yy) + 1j * (S_xy + S_yx)

    return C, P


def a2_to_cov_and_relation(A2):
    """
    Interpret A2 (shape [F, R, L, 2, R, L, 2]) as the covariance of the stacked real-imag vector [x; y],
    where z = x + i y is complex normal. The 2-axes index real/imag in that order.

    Let the real block covariance be:
        S = Cov([x; y]) = [[S_xx, S_xy],
                           [S_yx, S_yy]]

    Then the complex covariance C and relation P are:
        C = S_xx + S_yy + i (S_yx - S_xy)
        P = S_xx - S_yy + i (S_xy + S_yx)

    Returns
    -------
    C : np.ndarray, complex64/128
        Shape [F, R, L, R, L], the Hermitian covariance E[ z z^H ].
    P : np.ndarray, complex64/128
        Shape [F, R, L, R, L], the (generally nonzero) relation E[ z z^T ].

    """
    assert A2.ndim == 7, f"Expected 7D, got {A2.ndim}D"
    F, R, L, two1, R2, L2, two2 = A2.shape
    assert two1 == 2 and two2 == 2, f"Expected ...2 and ...2, got {A2.shape}"
    assert R == R2 and L == L2, f"Mismatched block sizes: {A2.shape}"

    # Real/imag blocks
    S_xx = A2[:, :, :, 0, :, :, 0]  # rr
    S_xy = A2[:, :, :, 0, :, :, 1]  # ri
    S_yx = A2[:, :, :, 1, :, :, 0]  # ir
    S_yy = A2[:, :, :, 1, :, :, 1]  # ii

    # Complex covariance and relation
    C = (S_xx + S_yy) + 1j * (S_yx - S_xy)
    P = (S_xx - S_yy) + 1j * (S_xy + S_yx)

    return C, P


def plot_stats(stats, title=None, ax=None, r_max=1.0, fn="temp.png"):
    """
    Plot the first and second order circular statistics.

    Parameters
    ----------
    stats : numpy.ndarray
        Shape: [d,d,2]
    """
    assert stats.ndim == 3, f"{stats.shape}"
    assert stats.shape[0] == stats.shape[1], f"{stats.shape}"
    assert stats.shape[2] == 2, f"{stats.shape}"

    arr = stats_to_colors(stats, r_max=r_max, mode="expanded_complex")
    if ax is not None:
        ax.imshow(arr)
        return
    plt.imshow(arr)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.savefig(fn)
    plt.close("all")


###
