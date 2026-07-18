#!/usr/bin/env python3

# Original code from Raul Astudillo https://github.com/RaulAstudillo06/BudgetedBO
# Copyright (c) 2021 Raul Astudillo

import torch
from botorch.posteriors import Posterior
from botorch.sampling.normal import NormalMCSampler
from botorch.utils.sampling import manual_seed


class PosteriorMeanSampler(NormalMCSampler):
    r"""
    """

    def _construct_base_samples(self, posterior: Posterior) -> None:
        r"""
        """
        target_shape = self._get_collapsed_shape(posterior=posterior)
        if self.base_samples is None or self.base_samples.shape != target_shape:
            with manual_seed(seed=self.seed):
                base_samples = torch.zeros(target_shape, device=posterior.device, dtype=posterior.dtype)
            self.register_buffer("base_samples", base_samples)
        if self.base_samples.device != posterior.device:
            self.to(device=posterior.device)  # pragma: nocover
        if self.base_samples.dtype != posterior.dtype:
            self.to(dtype=posterior.dtype)