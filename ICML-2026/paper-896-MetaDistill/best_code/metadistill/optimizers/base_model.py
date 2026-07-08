from __future__ import annotations

from typing import Optional, Union

import torch

from tasks import TaskProblem


class BaseBBOModel:
    """Base interface shared by gradient-free and gradient-trained optimizers."""

    def __init__(self) -> None:
        self.name = "BaseBBOModel"

    def reset(self) -> None:
        """Reset internal state for a new run."""
        return None

    def get_distr_param(self):
        """Optional: return distribution parameters for ES-style methods."""
        return None


class GradTrainedBBO(torch.nn.Module, BaseBBOModel):
    """Optimizer implemented as a torch.nn.Module (supports gradients)."""

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        BaseBBOModel.__init__(self)

        self.name = "GradTrainedBBO"
        self.needs_backward = True

    def forward(self, xp: torch.Tensor, problem: TaskProblem):
        """Generate next population from the current population."""
        raise NotImplementedError


class GradFreeBBO(BaseBBOModel):
    """Gradient-free optimizer interface."""

    def __init__(self) -> None:
        super().__init__()

        self.name = "GradFreeBBO"
        self.needs_backward = False

    def step(self, xp: torch.Tensor, problem: TaskProblem):
        """Generate next population from the current population."""
        raise NotImplementedError

    def __call__(self, xp: torch.Tensor, problem: TaskProblem):
        """Delegate to step() under torch.no_grad()."""
        with torch.no_grad():
            return self.step(xp, problem)


class BBORunner:
    """Thin wrapper to run either GradTrainedBBO or GradFreeBBO."""

    def __init__(
        self,
        algo: Optional[Union[GradTrainedBBO, GradFreeBBO]] = None,
        problem: Optional[TaskProblem] = None,
    ) -> None:
        self.algo = algo
        self.problem = problem

    def reset(self, problem: TaskProblem, problemdim=None) -> None:
        self.problem = problem
        if self.algo.name == "CMAES":
            self.algo.reset(bounds=[problem.fun["xlb"], problem.fun["xub"]])
        else:
            self.algo.reset()

    def _grad_step(self, xp: torch.Tensor):
        return self.algo(xp, self.problem)

    def _grad_free_step(self, xp: torch.Tensor):
        with torch.no_grad():
            return self.algo.step(xp, self.problem)

    def step(self, xp: torch.Tensor):
        if self.algo.needs_backward:
            return self._grad_step(xp)
        return self._grad_free_step(xp)
