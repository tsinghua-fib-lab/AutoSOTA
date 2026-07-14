#!/usr/bin/env python3
"""
Description: Learning rate schedule with linear warmup and cosine decline.
Initializes the learning rate with a linear warmup from 0 to 1 across a
specified number of warm-up steps during training. Subsequently, it follows
a cosine pattern for the remaining steps (total steps minus warm-up steps).
"""

# Standard library imports
import math

import torch
from torch.optim.lr_scheduler import LambdaLR


class CosineSchedule(LambdaLR):
    def __init__(
        self,
        optimizer: torch.optim,
        steps_warmup: int,
        steps_total: int,
        cycles: float = 0.5,
        last_epoch: int = -1,
    ) -> None:
        """
        Learning rate schedule with linear warmup and cosine decline.
        Initializes the learning rate with a linear warmup from 0 to 1 across
        a specified number of warm-up steps during training. Subsequently, it
        follows a cosine pattern for the remaining steps (total steps minus
        warm-up steps).

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer for which the learning rate is scheduled.
        steps_warmup : int
            Number of warm-up steps.
        steps_total : int
            Total number of training steps.
        cycles : float, optional
            Number of cosine cycles (default is 0.5).
        last_epoch : int, optional
            The index of the last epoch (default is -1).
        """

        # Store parameters
        self.steps_warmup = steps_warmup
        self.steps_total = steps_total
        self.cycles = cycles

        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step: int) -> float:
        """
        Compute the learning rate at a given step.

        Parameters
        ----------
        step : int
            Training step.

        Returns
        -------
        float
            Learning rate at the given step.
        """

        # warmup
        if step < self.steps_warmup:
            return float(step) / float(max(1.0, self.steps_warmup))

        # follow cosine pattern after warmup
        progress = float(step - self.steps_warmup) / float(
            max(1, self.steps_total - self.steps_warmup)
        )
        return max(
            0.0,
            0.5
            * (1.0 + math.cos(math.pi * float(self.cycles) * 2.0 * progress)),
        )
