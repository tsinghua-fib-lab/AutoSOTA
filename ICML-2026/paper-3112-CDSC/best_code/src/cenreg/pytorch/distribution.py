import torch


def _linear_interpolation(kx, ky, x):
    # compute idx and ratio
    idx = torch.searchsorted(kx, x, right=True)
    if kx.dim() == 1:
        idx = torch.clamp(idx, min=1, max=len(kx) - 1)
        lb = kx[idx - 1]
        ub = kx[idx]
    else:
        idx = torch.clamp(idx, min=1, max=kx.shape[1] - 1)
        lb = torch.gather(kx, 1, idx - 1)
        ub = torch.gather(kx, 1, idx)
    denominator = torch.clamp(ub - lb, min=0.0001)
    ratio = (x - lb) / denominator

    # linear interpolation
    if ky.dim() == 1:
        left = ky[idx - 1]
        right = ky[idx]
    else:
        left = torch.gather(ky, 1, idx - 1)
        right = torch.gather(ky, 1, idx)
    return torch.lerp(left, right, ratio)


class LinearCDF:
    """
    Distribution functions with linear interpolation.

    A distribution function is represented as a discrete cumulative
    distribution function (CDF) at pre-defined quantile levels (boundaries).
    The values between probabilities are computed by using linear interpolation.

    If qk_values are two-dimensional tensor, then each row corresponds to a CDF.
    """

    def __init__(
        self,
        boundaries: torch.Tensor,
        values: torch.Tensor | None = None,
        apply_cumsum: bool = True,
    ):
        """
        Distribution function initialization.

        Parameters
        ----------
        boundaries : Tensor
            One-dimensional tensor containing the boundaries used to represent CDF.

        values : Tensor
            One or two-dimensional tensor containing the values of CDFs or PDFs.
            If cdf_values is two-dimensional tensor, then
                Each row corresponds to a CDF and
                Tensor shape must be [num_CDF, len(boundaries)].
                If apply_cumsum is True, then
                    cdf_values[:,j] stores the value of PDF at boundaries[j].
                If apply_cumsum is False, then
                    cdf_values[:,j] stores the value of CDF at boundaries[j].
        """

        self.boundaries = boundaries
        self.set_knot_values(values, apply_cumsum=apply_cumsum)

    def average_cdf(self, y, mask=None, add_edge=False):
        values = self.cdf(y, mask, add_edge)
        if values.dim() > 1:
            return torch.mean(values, 0)
        else:
            return values

    def cdf(self, y, mask=None, add_edges=False):
        """
        Cumulative distribution function (i.e., inverse of quantile function).

        Parameters
        ----------
        y : Tensor
            CDF values are computed for values y.
            If dimension of y is one, then cdf(y) is computed for all CDFs.
            If dimension of y is two, then cdf(y) is computed for each corresponding CDF.
        mask : Tensor
            Mask to compute CDF for a subset of CDFs.
            Tensor must be one-dimensional and its length must be equal to
            the number of CDFs.
        add_edges : bool
            If True, then the CDF values at the boundaries are added.

        Returns
        -------
        cdf_values : Tensor
            Compute CDF values for each value in y.
            Tensor shape is equal to the shape of y.
        """
        if y.dim() == 1:
            if self.cdf_values.dim() > 1:
                y = torch.tile(y, (self.cdf_values.shape[0], 1))
        if mask is None:
            values = self.cdf_values
        else:
            values = self.cdf_values[mask]
        ret = _linear_interpolation(self.boundaries, values, y)
        if add_edges:
            if ret.dim() == 1:
                zero = torch.zeros(1, device=ret.device)
                one = torch.ones(1, device=ret.device)
                ret = torch.cat([zero, ret, one])
            else:
                zeros = torch.zeros(ret.shape[0], 1, device=ret.device)
                ones = torch.ones(ret.shape[0], 1, device=ret.device)
                ret = torch.cat([zeros, ret, ones], 1)
        return ret

    def icdf(self, alpha, mask=None, add_edges=False):
        """
        Quantile function (i.e., inverse of cumulative distribution function).

        Parameters
        ----------
        alpha : Tensor
            Quantile values are computed for quantile levels alpha.
            If dimension of alpha is one, then icdf(alpha) is computed for all CDFs.
            If dimension of alpha is two, then icdf(alpha) is computed for each corresponding CDF.
        mask : Tensor
            Mask to compute CDF for a subset of CDFs.
            Tensor must be one-dimensional and its length must be equal to
            the number of CDFs.
        add_edges : bool
            If True, then the inverse of the CDF values at the boundaries are added.

        Returns
        -------
        y : Tensor
            Compute y.
            Tensor shape is equal to the shape of alpha.
        """
        if alpha.dim() == 1:
            if self.cdf_values.dim() > 1:
                alpha = torch.tile(alpha, (self.cdf_values.shape[0], 1))
        if mask is None:
            values = self.cdf_values
        else:
            values = self.cdf_values[mask]
        ret = _linear_interpolation(values, self.boundaries, alpha)
        if add_edges:
            if self.cdf_values.dim() == 1:
                first = torch.tile(self.cdf_values[0], (ret.shape[0], 1))
                last = torch.tile(self.cdf_values[-1], (ret.shape[0], 1))
            else:
                first = self.cdf_values[:, 0].view(-1, 1)
                last = self.cdf_values[:, -1].view(-1, 1)
            ret = torch.cat([first, ret, last], 1)
        return ret

    def get_boundary_lengths(self):
        return self.boundaries[1:] - self.boundaries[:-1]

    def set_knot_values(self, values, apply_cumsum=True):
        """
        Set values of CDF values.

        Parameters
        ----------
        values : Tensor
            One or two-dimensional tensor containing the values of CDFs.
            If cdf_values is two-dimensional tensor, then each row corresponds to a CDF and
            cdf_values[:,j] stores the value of CDF at boundries[j].
            Tensor shape must be [num_CDF, len(boundaries)].
        apply_cumsum : bool
            If True, then cdf_values is assumed to be the probablity distribution functions (PDFs) and
            the cumulative sum of cdf_values is computed.
        """

        if values is None:
            return

        # set values
        if apply_cumsum:
            if values.dim() == 1:
                cum_values = torch.cumsum(values, dim=0)
                values = torch.cat([torch.tensor([0.0], device=cum_values.device), cum_values], 0)
            else:
                cum_values = torch.cumsum(values, dim=1)
                zeros = torch.zeros(cum_values.shape[0], 1, device=values.device)
                values = torch.cat([zeros, cum_values], 1)
        self.cdf_values = values

        # verify values
        if values.dim() == 1:
            if values.shape[0] != len(self.boundaries):
                raise ValueError(
                    f"cdf_values.shape[0] ({values.shape[0]}) must be equal to "
                    f"the length of boundaries ({len(self.boundaries)})"
                )
        else:
            if values.shape[1] != len(self.boundaries):
                raise ValueError(
                    f"cdf_values.shape[1] ({values.shape[1]}) must be equal to "
                    f"the length of boundaries ({len(self.boundaries)})"
                )


class LinearQuantileFunction:
    """
    Quantile functions with linear interpolation.

    A quantile function is defined by a set of quantile values (qk_values)
    at pre-defined quantile levels (qk_levels).
    The values between quantile values are computed by using linear interpolation.

    If qk_values are two-dimensional tensor, then each row corresponds
    to a quantile function.
    """

    def __init__(
        self,
        qk_levels: torch.Tensor,
        qk_values: torch.Tensor | None = None,
        apply_cumsum: bool = True,
    ):
        """
        Quantile function initialization.

        Parameters
        ----------
        qk_levels : Tensor
            One-dimensional tensor containing the positions (in quantile levels)
            of quantile knots in increasing order such that
                qk_levels[0] = 0.0
                qk_levels[-1] = 1.0

        qk_values : Tensor
            One or two-dimensional tensor containing the values of quantile knots.
            If qk_values is two-dimensional tensor, then
                each row corresponds to a quantile function and
                qk_values[:,j] stores the value of quantile function at qk_levels[j].
                Tensor shape must be [num_quantile_function, len(qk_levels)].
        """

        self.qk_levels = qk_levels
        self.set_knot_values(qk_values, apply_cumsum=apply_cumsum)

    def average_cdf(self, y, mask=None, add_edge=False):
        values = self.cdf(y, mask, add_edge)
        if values.dim() > 1:
            return torch.mean(values, 0)
        else:
            return values

    def cdf(self, y, mask=None, add_edges=False):
        """
        Cumulative distribution function (i.e., inverse of quantile function).

        Parameters
        ----------
        y : Tensor
            Quantile levels are computed for quantile values y.
            If dimension of y is one, then cdf(y) is computed for all quantile functions.
            If dimension of y is two, then cdf(y) is computed for each corresponding quantile function.
        mask : Tensor
            Mask to compute quantile function for a subset of quantile functions.
            Tensor must be one-dimensional and its length must be equal to
            the number of quantile functions.
        add_edges : bool
            If True, then the CDF values at the boundaries are added.

        Returns
        -------
        q_levels : Tensor
            Compute quantile levels for each value in y.
            Tensor shape is equal to the shape of y.
        """
        if y.dim() == 1:
            if self.qk_values.dim() > 1:
                y = torch.tile(y, (self.qk_values.shape[0], 1))
        if mask is None:
            values = self.qk_values
        else:
            values = self.qk_values[mask]
        ret = _linear_interpolation(values, self.qk_levels, y)
        if add_edges:
            if ret.dim() == 1:
                zero = torch.zeros(1, device=ret.device)
                one = torch.ones(1, device=ret.device)
                ret = torch.cat([zero, ret, one])
            else:
                zeros = torch.zeros(ret.shape[0], 1, device=ret.device)
                ones = torch.ones(ret.shape[0], 1, device=ret.device)
                ret = torch.cat([zeros, ret, ones], 1)
        return ret

    def icdf(self, alpha, mask=None, add_edges=False):
        """
        Quantile function (i.e., inverse of cumulative distribution function).

        Parameters
        ----------
        alpha : Tensor
            Quantile values are computed for quantile levels alpha.
            If dimension of alpha is one, then icdf(alpha) is computed for all quantile functions.
            If dimension of alpha is two, then icdf(alpha) is computed for each corresponding quantile function.
        mask : Tensor
            Mask to compute quantile function for a subset of quantile functions.
            Tensor must be one-dimensional and its length must be equal to
            the number of quantile functions.
        add_edges : bool
            If True, then the inverse of the CDF values at the boundaries are added.

        Returns
        -------
        y : Tensor
            Compute y.
            Tensor shape is equal to the shape of alpha.
        """
        if alpha.dim() == 1:
            if self.qk_values.dim() > 1:
                alpha = torch.tile(alpha, (self.qk_values.shape[0], 1))
        if mask is None:
            values = self.qk_values
        else:
            values = self.qk_values[mask]
        ret = _linear_interpolation(self.qk_levels, values, alpha)
        if add_edges:
            if ret.dim() == 1:
                first = self.qk_values[0].view(-1)
                last = self.qk_values[-1].view(-1)
                ret = torch.cat([first, ret, last])
            else:
                if self.qk_values.dim() == 1:
                    first = torch.tile(self.qk_values[0], (ret.shape[0], 1))
                    last = torch.tile(self.qk_values[-1], (ret.shape[0], 1))
                else:
                    first = self.qk_values[:, 0].view(-1, 1)
                    last = self.qk_values[:, -1].view(-1, 1)
                ret = torch.cat([first, ret, last], 1)
        return ret

    def get_qk_lengths(self):
        return self.qk_levels[1:] - self.qk_levels[:-1]

    def set_knot_values(self, qk_values, apply_cumsum=True):
        """
        Set values of quantile knots.

        Parameters
        ----------
        qk_values : Tensor
            One or two-dimensional tensor containing the values of quantile knots.
            If qk_values is two-dimensional tensor, then
            each row corresponds to a quantile function and
            qk_values[:,j] stores the value of quantile function at qk_levels[j].
            Tensor shape must be [num_quantile_function, len(qk_levels)].
        apply_cumsum : bool
            If True, then qk_values is assumed to be the differences of quantile values and
            the cumulative sum of qk_values is computed.
        """

        if qk_values is None:
            return

        # set values
        if apply_cumsum:
            if qk_values.dim() == 1:
                cum_values = torch.cumsum(qk_values, dim=0)
                qk_values = torch.cat([torch.tensor([0.0], device=cum_values.device), cum_values], 0)
            else:
                cum_values = torch.cumsum(qk_values, dim=1)
                zeros = torch.zeros(cum_values.shape[0], 1, device=qk_values.device)
                qk_values = torch.cat([zeros, cum_values], 1)
        self.qk_values = qk_values

        # check validity of values
        if qk_values.dim() == 1:
            if qk_values.shape[0] != len(self.qk_levels):
                raise ValueError("qk_values.shape[0] != len(qk_levels)")
        else:
            if qk_values.shape[1] != len(self.qk_levels):
                raise ValueError("qk_values.shape[1] != len(qk_levels)")


def _linear_interpolation(kx: torch.Tensor, ky: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Linear interpolation for PyTorch.

    Parameters
    ----------
    kx : torch.Tensor
        One-dimensional or two-dimensional tensor containing the x-coordinates
        of the data points.
    ky : torch.Tensor
        One-dimensional or two-dimensional tensor containing the y-coordinates
        of the data points.
    x : torch.Tensor
        One-dimensional or two-dimensional tensor containing the x-coordinates
        where to evaluate the interpolated values.

    Returns
    -------
    ret : torch.Tensor
        One-dimensional or two-dimensional tensor containing the interpolated values.
    """

    if kx.ndim == 1:
        if ky.ndim == 2:
            if kx.shape[0] != ky.shape[1]:
                raise ValueError("kx.shape[0] != ky.shape[1]")

    # compute idx and ratio
    if kx.ndim == 1:
        idx = torch.searchsorted(kx, x, right=True)
        idx = torch.clamp(idx, 1, len(kx) - 1)
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
                idx = torch.searchsorted(kx[i], x, right=True)
                idx = torch.clamp(idx, 1, kx.shape[1] - 1)
                lb = kx[i, idx - 1]
                ub = kx[i, idx]
                list_lb.append(lb.reshape(1, -1))
                list_ub.append(ub.reshape(1, -1))
                list_idx.append(idx.reshape(1, -1))
        else:
            for i in range(kx.shape[0]):
                idx = torch.searchsorted(kx[i], x[i], right=True)
                idx = torch.clamp(idx, 1, kx.shape[1] - 1)
                lb = kx[i, idx - 1]
                ub = kx[i, idx]
                list_lb.append(lb.reshape(1, -1))
                list_ub.append(ub.reshape(1, -1))
                list_idx.append(idx.reshape(1, -1))
        lb = torch.cat(list_lb, 0)
        ub = torch.cat(list_ub, 0)
        idx = torch.cat(list_idx, 0)
    denominator = torch.clamp(ub - lb, 0.0001, float("inf"))
    numerator = torch.clamp(x - lb, torch.tensor(0.0, device=x.device), ub - lb)
    ratio = numerator / denominator

    # linear interpolation
    if ky.ndim == 1:
        left = ky[idx - 1]
        right = ky[idx]
    elif idx.ndim == 1:
        left = torch.index_select(ky, -1, idx - 1)
        right = torch.index_select(ky, -1, idx)
    else:
        left = torch.take_along_dim(ky, idx - 1, -1)
        right = torch.take_along_dim(ky, idx, -1)
    return left + ratio * (right - left)


class CumulativeDist:
    """
    Cumulative distribution function for PyTorch.
    """

    def __init__(
        self,
        b: torch.Tensor,
        p: torch.Tensor | None = None,
        cum_p: torch.Tensor | None = None,
        interpolate: str = "linear",
    ):
        """
        Initialization.

        Parameters
        -------
        b : torch.Tensor
            Array containing boundaries of bins.
            b must be strictly increasing, and b[0] != -float("inf") and b[-1] != float("inf") must hold.
        p : torch.Tensor | None
            Probability distribution for each bin.
            p must be one-dimensional or two-dimensional.
        cum_p : torch.Tensor | None
            Cumulative probability distribution.
            cum_p must be one-dimensional or two-dimensional.
        interpolate : str
            'linear', 'left', or 'right' indicating the interpolation method.
            If 'linear' is set, linear interpolation is used.
            If 'left' is set, the CDF value at the left edge of each bin is used.
            If 'right' is set, the CDF value at the right edge of each bin is used.
        """

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)
        if b.dim() != 1:
            raise ValueError("b must be a one-dimensional array.")
        if cum_p is not None:
            if p is not None:
                raise ValueError("Only one of p and cum_p can be provided.")
            if not isinstance(cum_p, torch.Tensor):
                cum_p = torch.tensor(cum_p)
            self._validate_cum_p(cum_p, b)
        elif p is not None:
            if not isinstance(p, torch.Tensor):
                p = torch.tensor(p)
            cum_p = self._validate_p(p, b)
        if interpolate not in ["linear", "left", "right"]:
            raise ValueError("interpolate must be 'linear', 'left', or 'right'.")

        self.b = b
        self.cum_p = cum_p
        self.interpolate = interpolate

    def _validate_cum_p(self, cum_p: torch.Tensor, b: torch.Tensor):
        if cum_p.dim() > 2:
            raise ValueError("cum_p must be one-dimensional or two-dimensional.")
        if cum_p.dim() == 1:
            if b.shape[0] - 1 != cum_p.shape[0]:
                raise ValueError("Length of cum_p must be one less than length of b.")
            if torch.any(cum_p < 0.0) or torch.any(cum_p > 1.0):
                raise ValueError("cum_p must be in the range [0.0, 1.0].")
            if torch.all(torch.diff(cum_p) < 0.0):
                raise ValueError("cum_p must be non-decreasing.")
        else:  # cum_p.dim() == 2
            if b.shape[0] - 1 != cum_p.shape[1]:
                raise ValueError(
                    f"Length of cum_p ({cum_p.shape[1]}) must be one less than length of b ({b.shape[0]})."
                )
            if torch.any(torch.diff(cum_p, dim=1) < 0.0):  # allow cum_p[i, j] == cum_p[i, j + 1]
                raise ValueError("cum_p must be non-decreasing.")

    def _validate_p(self, p: torch.Tensor, b: torch.Tensor):
        if p.dim() > 2:
            raise ValueError("p must be one-dimensional or two-dimensional.")
        if torch.any(p < 0.0):
            raise ValueError("p must be non-negative.")
        if p.dim() == 1:
            if b.shape[0] - 1 != p.shape[0]:
                raise ValueError("Length of p must be one less than length of b.")
            cum_p = torch.cumsum(p, dim=0)
        else:  # p.dim() == 2
            if b.shape[0] - 1 != p.shape[1]:
                raise ValueError(f"Length of p ({p.shape[1]}) must be one less than length of b ({b.shape[0]}).")
            cum_p = torch.cumsum(p, dim=1)
        return cum_p

    def _require_cum_p(self) -> torch.Tensor:
        if self.cum_p is None:
            raise ValueError("cum_p must be set before evaluating CDF/ICDF.")
        return self.cum_p

    def _normalize_quantiles(self, quantiles: float | torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(quantiles, torch.Tensor):
            return quantiles.to(dtype=dtype)
        if isinstance(quantiles, int | float):
            return torch.tensor([quantiles], dtype=dtype)
        raise ValueError("quantiles must be a scalar or a torch.Tensor.")

    def cdf(self, y: float | torch.Tensor) -> torch.Tensor:
        """
        Cumulative distribution function (i.e., inverse of quantile function).

        Parameters
        -------
        y : torch.Tensor | float
            Values for which the CDF is computed.
            y can be a scalar or a one-dimensional array or a two-dimensional array.

        Returns
        -------
        probabilities : torch.Tensor
            CDF values for each value in y.
        """

        cum_p = self._require_cum_p()

        # check input y
        if isinstance(y, torch.Tensor):
            if len(y.shape) > 2:
                raise ValueError("y must be a scalar or a one-dimensional array or a two-dimensional array.")
            y = y.to(dtype=cum_p.dtype)
        else:
            if isinstance(y, float):
                y = torch.tensor([y], dtype=cum_p.dtype)
            else:
                raise ValueError("y must be a scalar or a one-dimensional array or a two-dimensional array.")

        if cum_p.ndim == 2 and y.ndim == 1:
            y = y.unsqueeze(0).repeat(cum_p.shape[0], 1)

        if self.interpolate == "linear":
            # linear interpolation implementation
            ret = torch.zeros_like(y)
            mask_low = y <= self.b[0]
            ret[mask_low] = 0.0
            mask_high = y > self.b[-1]
            ret[mask_high] = 1.0
            mask = ~(mask_low | mask_high)
            if cum_p.ndim == 1:
                temp = torch.cat([torch.tensor([0.0], dtype=cum_p.dtype, device=cum_p.device), cum_p])
                ret[mask] = _linear_interpolation(self.b, temp, y[mask])
            else:  # cum_p.ndim == 2
                zeros = torch.zeros((cum_p.shape[0], 1), dtype=cum_p.dtype, device=cum_p.device)
                temp = torch.cat([zeros, cum_p], dim=1)
                temp2 = _linear_interpolation(self.b, temp, y)
                ret[mask] = temp2[mask]
                # ret[mask] = linear_interpolation(boundaries, temp, y)[mask]
            return ret
        else:
            # step function implementation
            idx = torch.searchsorted(self.b, y, side=self.interpolate)
            idx = torch.clamp(idx, 1, len(self.b) - 1)
            if cum_p.dim() == 1:
                ret = cum_p[idx - 1]
                if self.interpolate == "left":
                    ret[y <= self.b[0]] = 0.0
                    ret[y > self.b[-1]] = 1.0
                else:
                    ret[y < self.b[0]] = 0.0
                    ret[y >= self.b[-1]] = 1.0
            else:  # cum_p.dim() == 2
                ret = torch.take_along_dim(cum_p, idx - 1, dim=1)
                if self.interpolate == "left":
                    ret[y <= self.b[0]] = 0.0
                    ret[y > self.b[-1]] = 1.0
                else:
                    ret[y < self.b[0]] = 0.0
                    ret[y >= self.b[-1]] = 1.0
            return ret

    def icdf(self, quantiles: float | torch.Tensor) -> torch.Tensor:
        """
        Inverse cumulative distribution function (i.e., quantile function).

        If the input is 0.0, return self.b[0].
        If the input is 1.0, return self.b[-1].
        If the input is between 0.0 and 1.0, return the corresponding bin value.

        Note:
        For any alpha in [0.0, 1.0], we usually assume that self.cdf(self.icdf(alpha)) == alpha holds.
        However, the inverse CDF of empirical distribution defined here does not always satisfy this property.

        Parameters
        -------
        quantiles : float | torch.Tensor
            Quantiles for which the inverse CDF is computed.

        Returns
        -------
        icdf_values : torch.Tensor
            Compute inverse CDF values for each value in quantiles.
        """

        cum_p = self._require_cum_p()

        quantiles = self._normalize_quantiles(quantiles, cum_p.dtype)
        if torch.any(quantiles < 0.0):
            raise ValueError("quantiles must be non-negative.")
        if torch.any(quantiles > 1.0):
            raise ValueError("quantiles must be at most 1.0.")
        if cum_p.ndim == 2 and quantiles.ndim == 1:
            quantiles = quantiles.unsqueeze(0).repeat(cum_p.shape[0], 1)

        if self.interpolate == "linear":
            # linear interpolation implementation
            ret = torch.zeros_like(quantiles)
            if cum_p.ndim == 1:
                mask_low = quantiles <= 0.0
                mask_high = quantiles >= cum_p[-1]
            else:
                mask_low = quantiles <= 0.0
                mask_high = quantiles >= cum_p[:, -1].reshape(-1, 1)
            ret[mask_low] = self.b[0]
            ret[mask_high] = self.b[-1]
            mask = ~(mask_low | mask_high)
            if cum_p.ndim == 1:
                temp = torch.cat([torch.tensor([0.0], dtype=cum_p.dtype, device=cum_p.device), cum_p])
                ret[mask] = _linear_interpolation(temp, self.b, quantiles[mask])
            else:
                zeros = torch.zeros((cum_p.shape[0], 1), dtype=cum_p.dtype, device=cum_p.device)
                temp = torch.cat([zeros, cum_p], dim=1)
                ret[mask] = _linear_interpolation(temp, self.b, quantiles)[mask]
            return ret
        else:
            # step function implementation
            ret = torch.zeros_like(quantiles)
            idx = torch.searchsorted(cum_p, quantiles, side="left")
            if cum_p[0] > 0.0:
                mask_low = idx <= 0
            else:
                idx = torch.maximum(idx, torch.tensor(1, dtype=idx.dtype, device=idx.device))
                mask_low = torch.full(idx.shape, False, dtype=torch.bool)
            ret[mask_low] = self.b[0]
            if cum_p[-1] < 1.0:
                mask_high = idx >= len(cum_p)
            else:
                idx = torch.minimum(idx, torch.tensor(len(cum_p) - 1, dtype=idx.dtype, device=idx.device))
                mask_high = torch.full(idx.shape, False, dtype=torch.bool)
            ret[mask_high] = self.b[-1]
            mask = ~(mask_low | mask_high)
            ret[mask] = self.b[idx[mask]]
            return ret

    def set_knot_values(
        self,
        p: torch.Tensor | None = None,
        cum_p: torch.Tensor | None = None,
        apply_cumsum: bool = True,
    ):
        """
        Set CDF values.

        Parameters
        ----------
        p : torch.Tensor | None
            One or two-dimensional tensor containing the values of PDFs.
            If p is two-dimensional tensor, then each row corresponds to a PDF and
            p[:,j] stores the value of PDF at boundries[j].
            Tensor shape must be [num_PDF, len(boundaries)].
        cum_p : torch.Tensor | None
            One or two-dimensional tensor containing the values of CDFs.
            If cum_p is two-dimensional tensor, then each row corresponds to a CDF and
            cum_p[:,j] stores the value of CDF at boundries[j].
            Tensor shape must be [num_CDF, len(boundaries)].
        apply_cumsum : bool
            If True, then cdf_values is assumed to be the probablity distribution functions (PDFs) and
            the cumulative sum of cdf_values is computed.
        """

        if cum_p is not None:
            if p is not None:
                raise ValueError("Only one of p and cum_p can be provided.")
            if not isinstance(cum_p, torch.Tensor):
                cum_p = torch.tensor(cum_p)
            self._validate_cum_p(cum_p, self.b)
        elif p is not None:
            if not isinstance(p, torch.Tensor):
                p = torch.tensor(p)
            if apply_cumsum:
                if p.dim() == 1:
                    cum_p = torch.cumsum(p, dim=0)
                    cum_p = torch.cat([torch.tensor([0.0], dtype=p.dtype, device=p.device), cum_p], 0)
                else:
                    cum_p = torch.cumsum(p, dim=1)
            else:
                cum_p = p
            self._validate_cum_p(cum_p, self.b)
        else:
            return
        self.cum_p = cum_p

    def survival_function(self, y: float | torch.Tensor) -> torch.Tensor:
        """
        Survival function (i.e., 1 - CDF).

        Parameters
        -------
        y : torch.Tensor | float
            Values for which the survival function is computed.

        Returns
        -------
        survival_values : torch.Tensor
            Survival function values for each value in y.
            Array shape is equal to the shape of y.
        """

        cdf_values = self.cdf(y)
        return 1.0 - cdf_values
