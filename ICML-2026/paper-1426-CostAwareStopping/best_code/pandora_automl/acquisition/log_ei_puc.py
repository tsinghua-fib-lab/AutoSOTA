#!/usr/bin/env python3

# Original code from Raul Astudillo https://github.com/RaulAstudillo06/BudgetedBO
# Adapted to the log variant for numerical robustness by Qian Xie, 2024

from typing import Callable, Optional, Union
import torch
from torch import Tensor
from botorch.acquisition import AnalyticAcquisitionFunction, LogExpectedImprovement
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from botorch.acquisition.analytic import _log_ei_helper


class LogExpectedImprovementWithCost(AnalyticAcquisitionFunction):
    """
    Computes the logarithm of the classic Expected Improvement With Cost acquisition function in a numerically robust manner:

    `LogEIC(x; alpha) = LogEI(x) - alpha * log(c(x)),` 
    
    where LogEI(x) = log(E(max(f(x) - best_f, 0))), alpha is a cost exponent (decay factor) that reduces or increases the emphasis of the cost function c(x).
    """
    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        maximize: bool = True,
        cost: Optional[Callable] = None,
        cost_exponent: Union[float, Tensor] = 1.0,
        unknown_cost: bool = False,
        inverse_cost: bool = True
    ):
        r"""Logarithm of Single-outcome/Two-outcome ExpectedImprovementWithCost (analytic).
        Args:
            model: A fitted single-outcome model or a fitted two-outcome model, 
                where the first output corresponds to the objective 
                and the second one to the log-cost.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            maximize: If True, consider the problem a maximization problem.
            unknown_cost: If True, consider the problem an unknown-cost problem.
            inverse_cost: If True, consider -lnE[1/c(x)] for the unknown-cost scenario; otherwise, consider lnE[c(x)] for the unknown-cost scenario.
        """
        # use AcquisitionFunction constructor to avoid check for objective
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.maximize = maximize
        self.cost = cost
        self.unknown_cost = unknown_cost
        self.inverse_cost = inverse_cost
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("cost_exponent", torch.as_tensor(cost_exponent))
    
    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor, cost_X: Optional[Tensor] = None) -> Tensor:
        if self.unknown_cost:
            # Handling the unknown cost scenario
            posterior = self.model.posterior(X)
            means = posterior.mean  # (b) x 2
            vars = posterior.variance.clamp_min(1e-6)  # (b) x 2
            stds = vars.sqrt()

            mean_obj = means[..., 0]
            std_obj = stds[..., 0]
            u = (mean_obj - self.best_f) / std_obj
            if not self.maximize:
                u = -u
            log_ei = _log_ei_helper(u) + std_obj.log()  # (b) x 1
            if self.inverse_cost:
                log_c = self.cost_exponent * means[..., 1] - 0.5 * (torch.square(self.cost_exponent) * vars[..., 1])  # E[1/c(x)]^{-1} using MGF
            else:
                log_c = self.cost_exponent * means[..., 1] + 0.5 * (torch.square(self.cost_exponent) * vars[..., 1])  # E[c(x)] using MGF
            log_ei_puc = log_ei - log_c  # (b) x 1
            return log_ei_puc.squeeze(dim=-1)
        else:
            # Handling the known cost scenario
            LogEI = LogExpectedImprovement(model=self.model, best_f=self.best_f, maximize=self.maximize)
            log_ei = LogEI(X)
 
            if cost_X is not None:
                cost_vals = cost_X.view(log_ei.shape)
            else:
                cost_vals = self.cost(X).view(log_ei.shape)

            log_eic = log_ei - self.cost_exponent * cost_vals.log()
            return log_eic