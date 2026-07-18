#!/usr/bin/env python3

# Original code from Raul Astudillo https://github.com/RaulAstudillo06/BudgetedBO
# Copyright (c) 2021 Raul Astudillo
# Modifications made by Qian Xie, 2024 to handle known costs

from typing import Callable, Optional, Union
import torch
from torch import Tensor
from botorch.acquisition import AnalyticAcquisitionFunction, ExpectedImprovement
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from torch.distributions import Normal

class ExpectedImprovementWithCost(AnalyticAcquisitionFunction):
    """
    This is the acquisition function EI(x) / c(x) ^ alpha, where alpha is a cost exponent (decay
    factor) that reduces or increases the emphasis of the cost function c(x).
    """
    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        maximize: bool = True,
        cost: Optional[Callable] = None,
        cost_exponent: Union[float, Tensor] = 1.0,
        unknown_cost: bool = False
    ):
        r"""Single-outcome/Two-outcome ExpectedImprovementWithCost (analytic).
        Args:
            model: A fitted single-outcome model or a fitted two-outcome model, 
                where the first output corresponds to the objective 
                and the second one to the log-cost.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            maximize: If True, consider the problem a maximization problem.
        """
        # use AcquisitionFunction constructor to avoid check for objective
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.maximize = maximize
        self.cost = cost
        self.unknown_cost = unknown_cost
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("cost_exponent", torch.as_tensor(cost_exponent))
    
    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
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
            standard_normal = Normal(0, 1)
            pdf_u = torch.exp(standard_normal.log_prob(u))
            cdf_u = standard_normal.cdf(u)
            ei = std_obj * (pdf_u + u * cdf_u)  # (b) x 1
            mgf = torch.exp(-(self.cost_exponent * means[..., 1]) + 0.5 * (torch.square(self.cost_exponent) * vars[..., 1]))
            ei_puc = ei.mul(mgf)  # (b) x 1
            return ei_puc.squeeze(dim=-1)
        else:
            # Handling the known cost scenario
            EI = ExpectedImprovement(model=self.model, best_f=self.best_f, maximize=self.maximize)
            ei_puc = EI(X) / torch.pow(self.cost(X).view(EI(X).shape), self.cost_exponent)
            return ei_puc