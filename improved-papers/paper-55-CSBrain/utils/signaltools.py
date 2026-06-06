"""
signaltools.py (Only a few functions) of Scipy's Signal processing package, implimented for PyTorch
Currently implimeted: resample

"""

import sys
import torch
import torch.fft

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2020, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "0.0.1"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Only x, num and axis of the resample function have been tested"


def _isrealobj(x):
    d = x.dtype
    if d in (torch.complex32, torch.complex64, torch.complex128):
        return False
    else:
        return True


def resample(x, num, t=None, axis=0, window=None, domain='time'):
    """
    Resample `x` to `num` samples using Fourier method along the given axis.

    The resampled signal starts at the same value as `x` but is sampled
    with a spacing of ``len(x) / num * (spacing of x)``.  Because a
    Fourier method is used, the signal is assumed to be periodic.

    Parameters
    ----------
    x : array_like
        The data to be resampled.
    num : int or array_like
        The number of samples in the resampled signal.
        If array_like is supplied, then the resample function will be
        called recursively for each element of num.
    t : array_like, optional
        If `t` is given, it is assumed to be the equally spaced sample
        positions associated with the signal data in `x`.
    axis : (int, optional) or (array_like)
        The axis of `x` that is resampled.  Default is 0.
        If num is array_like, then axis has to be supplied and has to be array_like.
        Each element of axis should have one-on-on mapping wtih num.
        If num is int but axis is array_like, then num will be repeated and will be
        made a list with same number of elements as axis. Then will proceed both as array_like.
    window : array_like, callable, string, float, or tuple, optional
        Specifies the window applied to the signal in the Fourier
        domain.  See below for details.
    domain : string, optional
        A string indicating the domain of the input `x`:
        ``time`` Consider the input `x` as time-domain (Default),
        ``freq`` Consider the input `x` as frequency-domain.

    Returns
    -------
    resampled_x or (resampled_x, resampled_t)
        Either the resampled array, or, if `t` was given, a tuple
        containing the resampled array and the corresponding resampled
        positions.

    See Also
    --------
    decimate : Downsample the signal after applying an FIR or IIR filter.
    resample_poly : Resample using polyphase filtering and an FIR filter.

    Notes
    -----
    The argument `window` controls a Fourier-domain window that tapers
    the Fourier spectrum before zero-padding to alleviate ringing in
    the resampled values for sampled signals you didn't intend to be
    interpreted as band-limited.

    If `window` is a function, then it is called with a vector of inputs
    indicating the frequency bins (i.e. fftfreq(x.shape[axis]) ).

    If `window` is an array of the same length as `x.shape[axis]` it is
    assumed to be the window to be applied directly in the Fourier
    domain (with dc and low-frequency first).

    For any other type of `window`, the function `scipy.signal.get_window`
    is called to generate the window.

    The first sample of the returned vector is the same as the first
    sample of the input vector.  The spacing between samples is changed
    from ``dx`` to ``dx * len(x) / num``.

    If `t` is not None, then it is used solely to calculate the resampled
    positions `resampled_t`

    As noted, `resample` uses FFT transformations, which can be very
    slow if the number of input or output samples is large and prime;
    see `scipy.fft.fft`.

    Examples
    --------
    Note that the end of the resampled data rises to meet the first
    sample of the next cycle:

    >>> from scipy import signal

    >>> x = np.linspace(0, 10, 20, endpoint=False)
    >>> y = np.cos(-x**2/6.0)
    >>> f = signal.resample(y, 100)
    >>> xnew = np.linspace(0, 10, 100, endpoint=False)

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x, y, 'go-', xnew, f, '.-', 10, y[0], 'ro')
    >>> plt.legend(['data', 'resampled'], loc='best')
    >>> plt.show()
    """

    if domain not in ('time', 'freq'):
        raise ValueError("Acceptable domain flags are 'time' or"
                         " 'freq', not domain={}".format(domain))

    if hasattr(axis, "__len__") and not hasattr(num, "__len__"):
        num = [num] * len(axis)

    if hasattr(num, "__len__"):
        if hasattr(axis, "__len__") and len(num) == len(axis):
            _temp = x
            _t_list = []
            for i in range(len(num)):
                _num = num[i]
                _axis = axis[i]
                if t is None:
                    _temp = resample(_temp, _num, t, _axis, window, domain)
                else:
                    _temp, _t = resample(_temp, _num, t, _axis, window, domain)
                    _t_list.append(_t)
            if t is None:
                return _temp
            else:
                return _temp, torch.stack(_t_list)
        else:
            raise ValueError("if num is array like, then axis also has to be array like and of the same length")

    Nx = x.shape[axis]

    # Check if we can use faster real FFT
    real_input = _isrealobj(x)

    if domain == 'time':
        # Forward transform
        if real_input:
            X = torch.fft.rfft(x, dim=axis)
        else:  # Full complex FFT
            X = torch.fft.fft(x, dim=axis)
    else:  # domain == 'freq'
        X = x

    # Apply window to spectrum
    if window is not None:
        if callable(window):
            W = window(torch.fft.fftfreq(Nx))
        elif isinstance(window, torch.Tensor):
            if window.shape != (Nx,):
                raise ValueError('window must have the same length as data')
            W = window
        else:
            sys.exit(
                "Window can only be either a function or Tensor. Window generation with get_window function of scipy.signal hasn't been implimented yet.")
            W = torch.fft.ifftshift(get_window(window, Nx))

        newshape_W = [1] * x.ndim
        newshape_W[axis] = X.shape[axis]
        if real_input:
            # Fold the window back on itself to mimic complex behavior
            W_real = W.clone()
            W_real[1:] += W_real[-1:0:-1]
            W_real[1:] *= 0.5
            X *= W_real[:newshape_W[axis]].reshape(newshape_W)
        else:
            X *= W.reshape(newshape_W)

    # Copy each half of the original spectrum to the output spectrum, either
    # truncating high frequences (downsampling) or zero-padding them
    # (upsampling)

    # Placeholder array for output spectrum
    newshape = list(x.shape)
    if real_input:
        newshape[axis] = num // 2 + 1
    else:
        newshape[axis] = num
    Y = torch.zeros(newshape, dtype=X.dtype, device=x.device)

    # Copy positive frequency components (and Nyquist, if present)
    N = min(num, Nx)
    nyq = N // 2 + 1  # Slice index that includes Nyquist if present
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(0, nyq)
    Y[tuple(sl)] = X[tuple(sl)]
    if not real_input:
        # Copy negative frequency components
        if N > 2:  # (slice expression doesn't collapse to empty array)
            sl[axis] = slice(nyq - N, None)
            Y[tuple(sl)] = X[tuple(sl)]

    # Split/join Nyquist component(s) if present
    # So far we have set Y[+N/2]=X[+N/2]
    if N % 2 == 0:
        if num < Nx:  # downsampling
            if real_input:
                sl[axis] = slice(N // 2, N // 2 + 1)
                Y[tuple(sl)] *= 2.
            else:
                # select the component of Y at frequency +N/2,
                # add the component of X at -N/2
                sl[axis] = slice(-N // 2, -N // 2 + 1)
                Y[tuple(sl)] += X[tuple(sl)]
        elif Nx < num:  # upsampling
            # select the component at frequency +N/2 and halve it
            sl[axis] = slice(N // 2, N // 2 + 1)
            Y[tuple(sl)] *= 0.5
            if not real_input:
                temp = Y[tuple(sl)]
                # set the component at -N/2 equal to the component at +N/2
                sl[axis] = slice(num - N // 2, num - N // 2 + 1)
                Y[tuple(sl)] = temp

    # Inverse transform
    if real_input:
        y = torch.fft.irfft(Y, num, dim=axis)
    else:
        y = torch.fft.ifft(Y, dim=axis, overwrite_x=True)

    y *= (float(num) / float(Nx))

    if t is None:
        return y
    else:
        new_t = torch.arange(0, num) * (t[1] - t[0]) * Nx / float(num) + t[0]
        return y, new_t