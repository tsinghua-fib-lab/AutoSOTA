import numpy as np
import torch
import torch.nn.functional as F

from cenreg.pytorch.distribution import CumulativeDist, LinearCDF


def negative_log_likelihood(
    dist,
    lb: torch.Tensor,
    ub: torch.Tensor,
    proportional: bool = True,
    eps: float = 0.0001,
) -> torch.Tensor:
    """
    Compute Negative log-likelihood.

    Parameters
    ----------
    dist: predicted distribution

    lb: Tensor of shape [batch_size]
        lower bound of the interval-censored data

    ub: Tensor of shape [batch_size]
        upper bound of the interval-censored data

    proportional: bool
        whether to distribute the probability mass proportionally for censored data

    eps: float
        small value to avoid numerical issues

    Returns
    -------
    loss : Tensor of shape [batch_size]
    """

    y_bins = dist.b
    idx_lb = torch.searchsorted(y_bins, lb.view(-1, 1), side="right")
    idx_lb = torch.clamp(idx_lb, min=1, max=len(y_bins) - 1)
    idx_ub = torch.searchsorted(y_bins, ub.view(-1, 1), side="left")
    idx_ub = torch.clamp(idx_ub, min=1, max=len(y_bins) - 1)
    b_lb = y_bins[idx_lb - 1]
    b_ub = y_bins[idx_ub]
    F_lb = dist.cdf(b_lb)
    F_ub = dist.cdf(b_ub)

    if proportional:
        interval = (idx_lb + 1 == idx_ub)[:, 0]
        F_lb[interval] = dist.cdf(lb.view(-1, 1))[interval]
        F_ub[interval] = dist.cdf(ub.view(-1, 1))[interval]

    loss = -torch.log(F_ub - F_lb + eps)
    return loss.view(-1)


class NegativeLogLikelihoodSurvival:
    """
    Loss class for negative log-likelihood for right-censored data.
    """

    def __init__(self, y_bins: torch.Tensor, apply_cumsum: bool = True, proportional: bool = True):
        if not isinstance(y_bins, torch.Tensor):
            raise ValueError("y_bins should be a torch.Tensor")

        self.distribution = CumulativeDist(y_bins)
        self.y_bins = y_bins
        self.apply_cumsum = apply_cumsum
        self.proportional = proportional
        self.eps = 0.0001

    def loss(
        self,
        pred: torch.Tensor,
        y: torch.Tensor,
        uncensored: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(pred.shape) != 2:
            raise ValueError("pred must be of shape [batch_size, num_bins]")
        if len(y.shape) != 1:
            raise ValueError("y must be of shape [batch_size]")
        if len(pred) != len(y):
            raise ValueError("pred and y must have the same length")
        if uncensored is None:
            raise ValueError("uncensored must be provided")

        self.distribution.set_knot_values(pred, apply_cumsum=self.apply_cumsum)
        lb = y
        ub = y
        ub[~uncensored] = np.inf
        return negative_log_likelihood(self.distribution, lb, ub, self.proportional, self.eps)


class NegativeLogLikelihoodInterval:
    """
    Loss class for negative log-likelihood for interval-censored data.
    """

    def __init__(
        self,
        y_bins: torch.Tensor,
        apply_cumsum: bool = True,
        proportional: bool = True,
        eps: float = 0.0001,
    ):
        if not isinstance(y_bins, torch.Tensor):
            raise ValueError("y_bins should be a torch.Tensor")
        if len(y_bins.shape) != 1:
            raise ValueError("y_bins should be a 1D tensor")
        diff = torch.diff(y_bins)
        if torch.any(diff <= 0.0):
            raise ValueError("y_bins should be sorted in ascending order")
        if y_bins[0] == -np.inf:
            raise ValueError("y_bins should not contain -inf")
        if y_bins[-1] == np.inf:
            raise ValueError("y_bins should not contain inf")

        self.distribution = CumulativeDist(y_bins)
        self.y_bins = y_bins
        self.apply_cumsum = apply_cumsum
        self.proportional = proportional
        self.eps = eps

    def loss(
        self,
        pred: torch.Tensor,
        lb: torch.Tensor,
        ub: torch.Tensor,
    ) -> torch.Tensor:
        if len(pred.shape) != 2:
            raise ValueError("pred must be of shape [batch_size, num_bins]")
        if len(lb.shape) != 1:
            raise ValueError("lb must be of shape [batch_size]")
        if len(ub.shape) != 1:
            raise ValueError("ub must be of shape [batch_size]")
        if len(lb) != len(ub):
            raise ValueError("lb and ub must have the same length")
        if len(pred) != len(lb):
            raise ValueError("pred and lb must have the same length")

        if self.apply_cumsum:
            self.distribution.set_knot_values(p=pred)
        else:
            self.distribution.set_knot_values(cum_p=pred)
        return negative_log_likelihood(self.distribution, lb, ub, self.proportional, self.eps)


class CNLLCR:
    """
    Censored Negative Log Likelihood for Competing Risks
    """

    def __init__(self, boundaries: torch.Tensor, num_risks: int):
        if not isinstance(boundaries, torch.Tensor):
            raise ValueError("boundaries should be a torch.Tensor")

        self.max_time = boundaries[-1]
        self.list_distribution = []
        for _ in range(num_risks):
            self.list_distribution.append(LinearCDF(boundaries))
        self.boundaries = boundaries
        self.eps = 0.0001

    def loss(self, pred: torch.Tensor, observed_times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        num_risks = len(self.list_distribution)
        idx = torch.searchsorted(self.boundaries, observed_times.view(-1, 1), right=True).view(-1)
        b_lb = self.boundaries[idx - 1]
        b_ub = self.boundaries[idx]

        loss = torch.zeros(observed_times.shape[0], device=observed_times.device)
        for i in range(num_risks):
            dist = self.list_distribution[i]
            dist.set_knot_values(pred[i, :, :])
            F_lb = dist.cdf(b_lb.view(-1, 1))
            F_ub = dist.cdf(b_ub.view(-1, 1))

            uncensored = events == i
            F_lb_uncensored = F_lb[uncensored]
            F_ub_uncensored = F_ub[uncensored]
            pu = torch.clamp(F_ub_uncensored - F_lb_uncensored, min=0.0)
            loss[uncensored] -= torch.log(pu + self.eps).view(-1)

            F_lb_censored = F_lb[~uncensored]
            F_ub_censored = F_ub[~uncensored]
            c = observed_times[~uncensored].view(-1, 1)
            F_c = dist.cdf(c, ~uncensored)
            denominator = torch.clamp(1.0 - F_c, min=self.eps)
            w = torch.clamp((F_ub_censored - F_c) / denominator, min=0.0, max=1.0)
            w = w.detach()
            pc1 = F_ub_censored - F_lb_censored + self.eps
            loss[~uncensored] -= (w * torch.log(pc1)).view(-1)
            pc2 = 1.0 - F_ub_censored + self.eps
            loss[~uncensored] -= ((1.0 - w) * torch.log(pc2)).view(-1)
        return loss


def brier(
    dist,
    y: torch.Tensor,
    y_bins: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the Brier score.

    Parameters
    ----------
    dist: predicted distribution

    y: Tensor of shape [batch_size]

    y_bins: Tensor of shape [num_bin+1]

    Returns
    -------
    loss : Tensor of shape [batch_size]
    """

    if len(y.shape) != 1:
        raise ValueError("y should be a 1D tensor")

    if y_bins is None:
        y_bins = dist.b
    if len(y_bins.shape) != 1:
        raise ValueError("y_bins should be a 1D tensor")

    idx = torch.searchsorted(y_bins, y.view(-1, 1), right=True)
    F_pred = dist.cdf(y_bins)
    y_pred = F_pred[:, 1:] - F_pred[:, :-1]
    onehot = F.one_hot((idx - 1).view(-1), num_classes=len(y_bins) - 1).float()
    loss = F.mse_loss(y_pred, onehot, reduction="none")
    return loss.sum(dim=1)


class Brier:
    """
    Loss class for the Brier score.
    """

    def __init__(self, y_bins: torch.Tensor, apply_cumsum: bool = True):
        if not isinstance(y_bins, torch.Tensor):
            raise ValueError("y_bins should be a torch.Tensor")

        self.distribution = LinearCDF(y_bins)
        self.y_bins = y_bins
        self.apply_cumsum = apply_cumsum

    def loss(
        self,
        pred: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        if len(pred.shape) != 2:
            raise ValueError("pred must be of shape [batch_size, num_bins]")
        if len(y.shape) != 1:
            raise ValueError("y must be of shape [batch_size]")

        self.distribution.set_knot_values(pred, apply_cumsum=self.apply_cumsum)
        return brier(self.distribution, y, self.y_bins)


def ranked_probability_score(
    dist,
    y: torch.Tensor,
    y_bins: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the ranked probability score.

    Parameters
    ----------
    dist: predicted distribution

    y: Tensor of shape [batch_size]

    y_bins: Tensor of shape [num_bins+1]

    Returns
    -------
    loss : Tensor of shape [batch_size]
    """

    if len(y.shape) != 1:
        raise ValueError("y should be a 1D tensor")

    if y_bins is None:
        y_bins = dist.b
    if len(y_bins.shape) != 1:
        raise ValueError("y_bins should be a 1D tensor")

    F_pred = dist.cdf(y_bins[1:-1])
    idx = torch.searchsorted(y_bins, y.view(-1, 1), right=True) - 1
    num_cls = len(y_bins) - 1
    label = torch.triu(torch.ones(num_cls, num_cls, device=y.device))[idx.view(-1)]
    loss = F.mse_loss(F_pred, label[:, :-1], reduction="none")
    return loss.sum(dim=1)


class RankedProbabilityScore:
    """
    Loss class for the ranked probability score.
    """

    def __init__(self, y_bins: torch.Tensor, apply_cumsum: bool = True):
        if not isinstance(y_bins, torch.Tensor):
            raise ValueError("y_bins should be a torch.Tensor")

        self.distribution = LinearCDF(y_bins)
        self.y_bins = y_bins
        self.apply_cumsum = apply_cumsum

    def loss(
        self,
        pred: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        if len(pred.shape) != 2:
            raise ValueError("pred must be of shape [batch_size, num_bins]")
        if len(y.shape) != 1:
            raise ValueError("y must be of shape [batch_size]")

        self.distribution.set_knot_values(pred, apply_cumsum=self.apply_cumsum)
        return ranked_probability_score(self.distribution, y, self.y_bins)
