import numpy as np

from cenreg.distribution.interpolate import linear_interpolation


class QuantileDist:
    """
    Quantile function.
    """

    def __init__(
        self,
        q: np.ndarray,
        v: np.ndarray,
        interpolate: str = "linear",
    ):
        """
        Initialization.

        Parameters
        -------
        q : np.ndarray
            One-dimensional array containing quantiles.
            q must be strictly increasing, and q[0] == 0.0 and q[-1] == 1.0 must hold.
        v : np.ndarray
            Values corresponding to each quantile.
            v must be one-dimensional or two-dimensional.
        interpolate : str
            Only 'linear' is supported, and linear interpolation is used.
        """

        q = np.array(q)
        v = np.array(v)
        assert len(q.shape) == 1
        assert np.all(np.diff(q) > 0.0), "q must be strictly increasing"
        assert len(v.shape) <= 2
        if len(v.shape) == 1:
            assert np.all(np.diff(v) >= 0.0), "v must be non-decreasing"
            assert v.shape[0] == q.shape[0], "v and q must have the same length"
        else:
            assert np.all(np.diff(v, axis=1) >= 0.0), "v must be non-decreasing"
            assert v.shape[1] == q.shape[0], "v and q must have the same length"
        assert interpolate in ["linear"]

        self.q = q
        self.v = v
        self.interpolate = interpolate

    def cdf(self, y: float | int | np.ndarray):
        """
        Cumulative distribution function (i.e., inverse of quantile function).

        Parameters
        -------
        y : np.ndarray | float
            Values for which the CDF is computed.

        Returns
        -------
        cum_p : np.ndarray
            CDF values for each value in y.
            Array shape is equal to the shape of y.
        """
        if np.isscalar(y):
            y = np.array([y], dtype=float)
        else:
            y = np.asarray(y)

        if self.interpolate == "linear":
            # linear interpolation implementation
            ret = np.zeros_like(y)
            mask_low = y <= self.v[0]
            ret[mask_low] = 0.0
            mask_high = y > self.v[-1]
            ret[mask_high] = 1.0
            mask = np.logical_not(mask_low | mask_high)
            ret[mask] = linear_interpolation(self.v, self.q, y[mask])
            return ret
        else:
            raise NotImplementedError("Only 'linear' interpolation is supported for CDF.")

    def icdf(self, quantiles: float | int | np.ndarray) -> np.ndarray:
        """
        Inverse cumulative distribution function (i.e., quantile function).

        If the input is 0.0, return self.v[0].
        If the input is 1.0, return self.v[-1].
        If the input is between 0.0 and 1.0, return the corresponding bin value.

        Parameters
        -------
        quantiles : float | np.ndarray
            Quantiles for which the inverse CDF is computed.

        Returns
        -------
        icdf_values : np.ndarray
            Compute inverse CDF values for each value in quantiles.
            Array shape is equal to the shape of quantiles.
        """

        if np.isscalar(quantiles):
            quantiles = np.array([quantiles], dtype=float)
        else:
            quantiles = np.asarray(quantiles)
        if np.any(quantiles < 0.0):
            raise ValueError("quantiles must be non-negative.")
        if np.any(quantiles > 1.0):
            raise ValueError("quantiles must be at most 1.0.")

        return linear_interpolation(self.q, self.v, quantiles)

    def survival_function(self, y: float | np.ndarray) -> np.ndarray:
        """
        Survival function (i.e., 1 - CDF).

        Parameters
        -------
        y : np.ndarray | float
            Values for which the survival function is computed.

        Returns
        -------
        survival_values : np.ndarray
            Survival function values for each value in y.
            Array shape is equal to the shape of y.
        """

        cdf_values = self.cdf(y)
        return 1.0 - cdf_values
