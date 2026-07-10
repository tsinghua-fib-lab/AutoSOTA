import torch

from cenreg.pytorch.copula_torch import IndependenceCopula, SurvivalCopula


def negative_log_likelihood(
    F_lb: torch.Tensor,
    F_ub: torch.Tensor,
    eps: float = 0.0001,
) -> torch.Tensor:
    """
    Compute Negative log-likelihood.

    Parameters
    ----------
    F_lb: Tensor of shape [batch_size]
        CDF values at the lower bound of the interval-censored data

    F_ub: Tensor of shape [batch_size]
        CDF values at the upper bound of the interval-censored data

    eps: float
        small value to avoid numerical issues

    Returns
    -------
    loss : Tensor of shape [batch_size]
    """

    uncensored = (F_lb == F_ub).detach()
    F_lb = F_lb
    F_ub = F_ub

    if torch.any(uncensored):
        pred_sum = torch.sum(F_lb[uncensored])
        df = torch.autograd.grad(pred_sum, F_lb[uncensored], create_graph=True)[0]

        ret = torch.zeros_like(F_lb)
        ret[uncensored] = -torch.log(df[uncensored] + eps)
        ret[~uncensored] = -torch.log(F_ub[~uncensored] - F_lb[~uncensored] + eps)
    else:
        ret = -torch.log(F_ub - F_lb + eps)
    return ret


class NegativeLogLikelihoodSurvival:
    """
    Negative Log-Likelihood for survival data (right-censored)
    """

    def __init__(self, eps=0.0001):
        self.eps = eps

    def loss(
        self,
        F_t: torch.Tensor,
        events: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if events is None:
            raise ValueError("events must be provided")
        if len(F_t.shape) != 1:
            raise ValueError(f"F_t must be 1D tensor, got {F_t.shape}")
        if F_t.shape[0] != events.shape[0]:
            raise ValueError(
                f"F_t and events must have the same number of samples, got {F_t.shape[0]} and {events.shape[0]}"
            )

        F_ub = torch.full(F_t.shape, 1.0, device=F_t.device)
        uncensored = events > 0
        F_ub[uncensored] = F_t[uncensored]
        return negative_log_likelihood(F_t, F_ub, self.eps)


class NegativeLogLikelihoodInterval:
    """
    Negative Log-Likelihood for interval-censored data
    """

    def __init__(self, eps=0.0001):
        self.eps = eps

    def loss(self, F_lb: torch.Tensor, F_ub: torch.Tensor) -> torch.Tensor:
        if F_lb.shape != F_ub.shape:
            raise ValueError(f"F_lb and F_ub must have the same shape, got {F_lb.shape} and {F_ub.shape}")

        return negative_log_likelihood(F_lb, F_ub, self.eps)


class CopulaNegativeLogLikelihood:
    """
    Negative Log Likelihood with survival copula

    return -log ((dC/dF) (dF/dt))
    """

    def __init__(self, copula=None, survival_copula=None, eps=0.0001):
        self.eps = eps
        if copula is None:
            if survival_copula is None:
                self.survival_copula = IndependenceCopula()
            else:
                self.survival_copula = survival_copula
        else:
            if survival_copula is None:
                self.survival_copula = SurvivalCopula(copula)
            else:
                self.survival_copula = survival_copula
                print("Warning: survival_copula is not None. copula is ignored.")

    def loss(self, F_pred: torch.Tensor, observed_times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        if len(F_pred.shape) != 2:
            raise ValueError("F_pred must be of shape [batch_size, num_risks]")
        if F_pred.shape[0] != observed_times.shape[0]:
            raise ValueError(
                f"F_pred and observed_times must have the same number of samples, "
                f"got {F_pred.shape[0]} and {observed_times.shape[0]}"
            )

        df = torch.zeros_like(observed_times)
        num_risks = F_pred.shape[1]
        for k in range(num_risks):
            mask = (events == k).detach()
            pred_sum = torch.sum(F_pred[:, k])
            temp = torch.autograd.grad(pred_sum, observed_times, create_graph=True)[0]
            df[mask] = temp[mask]
        log_df = torch.log(df + self.eps)

        s = 1.0 - F_pred
        if self.survival_copula is None:
            raise NotImplementedError()
            # TODO compute survival copula using normal copula
        else:
            c = self.survival_copula.cdf(s)
        c_sum = torch.sum(c)
        dc = torch.autograd.grad(c_sum, s, create_graph=True)[0]
        events = events.view(-1, 1).detach()
        log_dc = torch.log(torch.gather(dc, 1, events) + self.eps)
        return -(log_df + log_dc)
