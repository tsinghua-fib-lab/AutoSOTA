#!/usr/bin/env python3

# Copyright (c) 2024 Qian Xie

from typing import Union
import torch
from torch import Tensor
from botorch.acquisition import AnalyticAcquisitionFunction, ExpectedImprovement, LogExpectedImprovement
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform


class LogVanillaExpectedImprovement(AnalyticAcquisitionFunction):
    """
    This is the acquisition function log(EI(x)), where EI here is the vanilla Expected Improvement acquisition function.
    """
    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        maximize: bool = True,
    ):
        r"""Logarithm of single-outcome vanilla Expected Improvement (analytic) without treatment for numerical robustness.
        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a b-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        # use AcquisitionFunction constructor to avoid check for objective
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.maximize = maximize
        self.register_buffer("best_f", torch.as_tensor(best_f))
    
    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        EI = ExpectedImprovement(model=self.model, best_f=self.best_f, maximize=self.maximize)
        return (EI(X)+1e-6).log()
    
class StableExpectedImprovement(AnalyticAcquisitionFunction):
    """
    This is the acquisition function exp(LogEI(x)), where LogEI here is the numerically robust logarithm of the Expected Improvement acquisition function.
    """
    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        maximize: bool = True,
    ):
        r"""Numerically-stable single-outcome Expected Improvement (analytic).
        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a b-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        # use AcquisitionFunction constructor to avoid check for objective
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.maximize = maximize
        self.register_buffer("best_f", torch.as_tensor(best_f))
    
    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        LogEI = LogExpectedImprovement(model=self.model, best_f=self.best_f, maximize=self.maximize)
        return torch.exp(LogEI(X))