import cma
import torch
import numpy
from torch_basic_settings import *
from optimizers.base_model import GradFreeBBO


# wrapper for cma-es
class CMAES(GradFreeBBO):
    def __init__(self, popsize, problemdim, generations, bounds):
        GradFreeBBO.__init__(self)
        self.name = "CMAES"

        self.popsize = popsize
        self.problemdim = problemdim
        self.generations = generations
        self.optimizer = cma.CMAEvolutionStrategy(
            [numpy.random.rand() for _ in range(self.problemdim)],
            numpy.random.rand(),
            {
                "popsize": self.popsize,
                "bounds": [
                    [bounds[0] for _ in range(self.problemdim)],
                    [bounds[1] for _ in range(self.problemdim)],
                ],
                "maxiter": generations,
                "verb_disp": 0,
            },
        )

        self.last_pop = torch.rand((1, self.popsize, self.problemdim))

    def step(self, pop=None, task=None):
        stop = self.optimizer.stop()
        if not stop:
            pop = self.optimizer.ask()
            _pop = (
                torch.from_numpy(numpy.array(pop)).view(1, self.popsize, -1).to(DEVICE)
            )  # (1, popsize, dim)
            _pop, _f_val = task.calfitness(_pop)
            f_val = _f_val.view(
                self.popsize,
            )
            f_val = f_val.tolist()
            self.optimizer.tell(pop, f_val)
            self.last_pop = _pop

            info = {"fitness": _f_val.detach()}
            return _pop, info

        else:
            info = {"fitness": self.last_pop[..., 0:1].detach()}
            return self.last_pop, info  # (b, n, 1)

    def reset(self, bounds=None):
        self.last_pop = torch.rand((1, self.popsize, self.problemdim))

        self.optimizer = None
        self.optimizer = cma.CMAEvolutionStrategy(
            [numpy.random.rand() for _ in range(self.problemdim)],
            numpy.random.rand(),
            {
                "popsize": self.popsize,
                "bounds": [
                    [bounds[0] for _ in range(self.problemdim)],
                    [bounds[1] for _ in range(self.problemdim)],
                ],
                "maxiter": self.generations,
                "verb_disp": 0,
            },
        )

    def get_distr_param(self):
        params = {}

        mean = torch.from_numpy(self.optimizer.mean).view(1, -1).to(DEVICE)  # (1, dim)

        cov_mat = self.optimizer.sm.C
        sigma = self.optimizer.sigma
        cov_mat = (
            torch.from_numpy(sigma * cov_mat).view(self.problemdim, -1).to(DEVICE)
        )  # (dim, dim)

        params["mean"] = mean
        params["cov_mat"] = cov_mat

        return params
