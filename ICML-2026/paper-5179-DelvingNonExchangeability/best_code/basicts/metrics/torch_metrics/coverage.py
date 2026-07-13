import torch
from torchmetrics import Metric
from torchmetrics.utilities.checks import _check_same_shape

from tsl.metrics.torch import MaskedMetric


def is_covered(target, interval):
    """
    Determines whether interval covers the target prediction.

    Args:
        intervals: shape [2, ...]
        target: ground truth forecast values

    Returns:
        individual and joint coverage rates
    """

    lower, upper = interval
    # [batch, 1, n_outputs, horizon]
    return torch.logical_and(target >= lower, target <= upper).float()


def coverage_loss(y_hat, y):
    """

    :param y_hat: intervals, shape [2, ...]
    :param y: ground truth, shape [...]
    :return: Whether each prediction is covered or not
    """
    y_hat_lower, y_hat_upper = y_hat[0], y_hat[-1]
    _check_same_shape(y_hat_upper, y)
    return torch.logical_and(y >= y_hat_lower, y <= y_hat_upper).float()


def delta_coverage_loss(y_hat, y, alpha):
    cov = coverage_loss(y_hat, y)
    return cov - (1 - alpha)


def pi_width_loss(y_hat, y):
    _check_same_shape(y_hat[0], y)
    return y_hat[-1] - y_hat[0]


############################################################################################

class MaskedDeltaCoverage(MaskedMetric):
    """Coverage metric.

    Args:
        mask_nans (bool, optional): Whether to automatically mask nan values.
        mask_inf (bool, optional): Whether to automatically mask infinite
            values.
        compute_on_step (bool, optional): Whether to compute the metric
            right-away or if accumulate the results. This should be :obj:`True`
            when using the metric to compute a loss function, :obj:`False` if
            the metric is used for logging the aggregate error across different
            mini-batches.
        at (int, optional): Whether to compute the metric only w.r.t. a certain
            time step.
    """

    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = False
    shape_check = False

    def __init__(self,
                 alpha,
                 mask_nans=False,
                 mask_inf=False,
                 at=None,
                 **kwargs):
        super(MaskedDeltaCoverage, self).__init__(metric_fn=delta_coverage_loss,
                                                  mask_nans=mask_nans,
                                                  mask_inf=mask_inf,
                                                  metric_fn_kwargs={"alpha": alpha},
                                                  at=at,
                                                  **kwargs)

    def update(self, y_hat, y, mask=None):
        y_hat = y_hat[:, :, self.at]
        y = y[:, self.at]
        if mask is not None:
            mask = mask[:, self.at]
        if self.is_masked(mask):
            val = self.metric_fn(y_hat, y)
            mask = self._check_mask(mask, val)
            val = torch.where(mask, val, torch.zeros_like(val))
            self.value += val.sum()
            self.numel += mask.sum()
        else:
            val = self.metric_fn(y_hat, y)
            self.value += val.sum()
            self.numel += val.numel()


class MaskedPIWidth(MaskedMetric):
    """Quantile loss.

    Args:
        qs (List): Target quantiles.
        mask_nans (bool, optional): Whether to automatically mask nan values.
        mask_inf (bool, optional): Whether to automatically mask infinite
            values.
        at (int, optional): Whether to compute the metric only w.r.t. a certain
            time step.
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    shape_check = False

    def __init__(
            self, mask_nans=False, mask_inf=False, at=None, **kwargs
    ):
        super(MaskedPIWidth, self).__init__(
            metric_fn=pi_width_loss,
            mask_nans=mask_nans,
            mask_inf=mask_inf,
            at=at,
            **kwargs,
        )

    def update(self, y_hat, y, mask=None):
        y_hat = y_hat[:, :, self.at]
        y = y[:, self.at]
        if mask is not None:
            mask = mask[:, self.at]
        if self.is_masked(mask):
            val = self.metric_fn(y_hat, y)
            mask = self._check_mask(mask, val)
            val = torch.where(mask, val, torch.zeros_like(val))
            self.value += val.sum()
            self.numel += mask.sum()
        else:
            val = self.metric_fn(y_hat, y)
            self.value += val.sum()
            self.numel += val.numel()


class MaskedPIMedianWidth(Metric):
    """Median prediction interval width."""

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(self, mask_nans=False, mask_inf=False, at=None, **kwargs):
        super(MaskedPIMedianWidth, self).__init__(**kwargs)
        self.mask_nans = mask_nans
        self.mask_inf = mask_inf
        self.at = at
        self.add_state("widths", default=[], dist_reduce_fx=None)

    def update(self, y_hat, y, mask=None):
        y_hat = y_hat[:, :, self.at]
        y = y[:, self.at]
        if mask is not None:
            mask = mask[:, self.at]
        width = y_hat[-1] - y_hat[0]
        if mask is not None:
            width = width[mask.bool()]
        else:
            width = width.reshape(-1)
        if self.mask_nans:
            width = width[~torch.isnan(width)]
        if self.mask_inf:
            width = width[~torch.isinf(width)]
        if width.numel() == 0:
            return
        self.widths.append(width.detach())

    def compute(self):
        if not self.widths:
            return torch.tensor(float("nan"))
        width = torch.cat([w.reshape(-1) for w in self.widths])
        if width.numel() == 0:
            return torch.tensor(float("nan"))
        return torch.median(width)

    def reset(self):
        super(MaskedPIMedianWidth, self).reset()
        self.widths = []


class MaskedCoverage(MaskedMetric):
    """Coverage metric.

    Args:
        mask_nans (bool, optional): Whether to automatically mask nan values.
        mask_inf (bool, optional): Whether to automatically mask infinite
            values.
        compute_on_step (bool, optional): Whether to compute the metric
            right-away or if accumulate the results. This should be :obj:`True`
            when using the metric to compute a loss function, :obj:`False` if
            the metric is used for logging the aggregate error across different
            mini-batches.
        at (int, optional): Whether to compute the metric only w.r.t. a certain
            time step.
    """

    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = False
    shape_check = False

    def __init__(self,
                 mask_nans=False,
                 mask_inf=False,
                 at=None,
                 **kwargs):
        super(MaskedCoverage, self).__init__(metric_fn=coverage_loss,
                                             mask_nans=mask_nans,
                                             mask_inf=mask_inf,
                                             at=at,
                                             **kwargs)

    def update(self, y_hat, y, mask=None):
        y_hat = y_hat[:, :, self.at]
        y = y[:, self.at]
        if mask is not None:
            mask = mask[:, self.at]
        if self.is_masked(mask):
            val = self.metric_fn(y_hat, y)
            mask = self._check_mask(mask, val)
            val = torch.where(mask, val, torch.zeros_like(val))
            self.value += val.sum()
            self.numel += mask.sum()
        else:
            val = self.metric_fn(y_hat, y)
            self.value += val.sum()
            self.numel += val.numel()
