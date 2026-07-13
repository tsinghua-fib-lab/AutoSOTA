from torchmetrics.utilities.checks import _check_same_shape
import torch

from tsl.metrics.torch import MaskedMetric


def winkler_score(y_hat, y, alpha, lam=2):
    _check_same_shape(y_hat[0], y)
    y_hat_low, y_hat_up = y_hat[0], y_hat[-1]
    width = y_hat[-1] - y_hat[0]
    coeff = lam / alpha
    score = width + \
            coeff * ((y_hat_low - y) * (y < y_hat_low) + \
                           (y - y_hat_up) * (y > y_hat_up))
    return score


class MaskedWinklerScore(MaskedMetric):
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
    higher_is_better: bool = False
    full_state_update: bool = False
    shape_check = False

    def __init__(self,
                 alpha,
                 mask_nans=False,
                 mask_inf=False,
                 at=None,
                 lam=2.,
                 **kwargs):
        super(MaskedWinklerScore, self).__init__(metric_fn=winkler_score,
                                                  mask_nans=mask_nans,
                                                  mask_inf=mask_inf,
                                                  metric_fn_kwargs={"alpha": alpha, "lam": lam},
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
