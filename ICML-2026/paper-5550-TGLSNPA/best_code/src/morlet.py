import numpy as np


def morlet2(M, s, w=5):
    """
    Drop-in replacement for scipy.signal.morlet2(M, s, w=5).
    Complex Morlet wavelet designed for scipy.signal.cwt-style usage.
    Parameters
    ----------
    M : int
        Length of the wavelet.
    s : float
        Width parameter / scale.
    w : float, optional
        Omega0. Default is 5.
    Returns
    -------
    output : ndarray, shape (M,)
        Complex Morlet wavelet, normalized by sqrt(1/s).
    """
    M = int(M)
    s = float(s)
    w = float(w)
    x = np.arange(0, M) - (M - 1.0) / 2
    x = x / s
    wavelet = np.exp(1j * w * x) * np.exp(-0.5 * x**2) * np.pi**(-0.25)
    output = np.sqrt(1.0 / s) * wavelet
    return output
