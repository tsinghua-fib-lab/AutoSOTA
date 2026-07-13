import torch
from torchmetrics.utilities.checks import _check_same_shape

from tsl.metrics.torch import MaskedMetric


def pinball_loss(y_hat, y, q):
    err = y - y_hat
    return torch.maximum((q - 1) * err, q * err)


def multi_quantile_pinball_loss(y_hat, y, q):
    q = torch.as_tensor(q, dtype=y_hat.dtype, device=y_hat.device)

    if q.ndim == 1:
        assert y_hat.size(0) == len(q)
        _check_same_shape(y_hat[0], y)

        q = q.view(-1, *( (1,) * (y_hat.ndim - 1)))

        return pinball_loss(y_hat, y[None], q).sum(0)
    return pinball_loss(y_hat, y, q)


class MaskedPinballLoss(MaskedMetric):
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
            self, mask_nans=False, mask_inf=False, **kwargs
    ):
        super(MaskedPinballLoss, self).__init__(
            metric_fn=pinball_loss,
            mask_nans=mask_nans,
            mask_inf=mask_inf,
            at=None,
            **kwargs,
        )

    def _compute_masked(self, y_hat, y, q, mask):
        val = self.metric_fn(y_hat, y, q)
        mask = self._check_mask(mask, val)
        val = torch.where(mask, val, torch.zeros_like(val))
        return val.sum(), mask.sum()

    def _compute_std(self, y_hat, y, q):
        val = self.metric_fn(y_hat, y, q)
        return val.sum(), val.numel()

    def update(self, y_hat, y, q, mask=None):
        if self.is_masked(mask):
            val, numel = self._compute_masked(y_hat, y, q, mask)
        else:
            val, numel = self._compute_std(y_hat, y, q)
        self.value += val
        self.numel += numel


class MaskedMultiPinballLoss(MaskedMetric):
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
            self, qs, mask_nans=False, mask_inf=False, at=None, **kwargs
    ):
        super(MaskedMultiPinballLoss, self).__init__(
            metric_fn=multi_quantile_pinball_loss,
            mask_nans=mask_nans,
            mask_inf=mask_inf,
            metric_fn_kwargs={'q': qs},
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
