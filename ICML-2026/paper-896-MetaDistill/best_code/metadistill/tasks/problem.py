from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from optimizers.base_model import BBORunner
from tasks.utils import getFitness, genOffset, setOffset
from torch_basic_settings import *


class Problem:
    def __init__(self):
        self.useRepaire = False

    def repaire(self, x):
        raise NotImplementedError

    def calfitness(self, x):
        raise NotImplementedError


class taskProblem(Problem):
    def __init__(self, fun=None, repaire=True, dim=None):
        super().__init__()
        self.fun = fun
        self.useRepaire = repaire
        self.dim = dim

    def repaire(self, x):
        # Robustness: replace NaN/Inf before clamping
        x = torch.nan_to_num(x, nan=0.0, posinf=float(self.fun["xub"]), neginf=float(self.fun["xlb"]))
        xlbmask = torch.zeros_like(x, device=DEVICE)
        xlbmask[x < self.fun["xlb"]] = 1
        normalmask = 1 - xlbmask
        xlbmask = xlbmask * self.fun["xlb"]
        x = normalmask * x + xlbmask

        xubmask = torch.zeros_like(x, device=DEVICE)
        xubmask[x > self.fun["xub"]] = 1
        normalmask = 1 - xubmask
        xubmask = xubmask * self.fun["xub"]
        x = normalmask * x + xubmask
        return x

    def calfitness(self, x):
        """
        Evaluate fitness and return population with fitness.

        Args:
            x: candidate solutions, shape (B, N, D)

        Returns:
            pop: concatenated fitness and solutions, shape (B, N, 1 + D)
            fitness: shape (B, N, 1)
        """

        # x=torch.matmul(x,self.w) #b,n,d

        if self.useRepaire:
            x1 = self.repaire(x)
        else:
            x1 = x

        b, n, d = x.shape
        if self.getfunname() in list(range(1, 25)):
            x1 = x1.view((-1, d))
        r = getFitness(x1, self.fun)  # b,n,1
        r = torch.unsqueeze(r, -1)
        r = r.view((b, n, 1))

        x1 = x1.view((b, n, d))
        pop = torch.cat((r, x1), dim=-1)  # b,n,d
        return pop, r

    def genRandomPop(self, batchShape):
        lb = self.fun["xlb"]
        ub = self.fun["xub"]
        return torch.rand(batchShape, device=DEVICE) * (ub - lb) + lb

    def reoffset(self):
        genOffset(self.dim, self.fun)

    def setOffset(self, offset):
        for key in offset.keys():
            self.fun[key] = offset[key]

    def getfunname(self):
        return self.fun["fid"]

    def setfun(self, fun):
        self.fun = fun


class NeuroEvoTask(taskProblem):
    def __init__(self, fun=None, repaire=True, net=None, step=256):
        super().__init__(fun, repaire, dim=net.get_dim())

        self.net = net
        self.step = step

    def set_net(self, net):
        self.net = net

    def calfitness(self, x):
        if self.useRepaire:
            x1 = self.repaire(x)
        else:
            x1 = x

        pop_with_fitness, fitness = self.fun["fun"](x=x1, net=self.net, steps=self.step)

        return pop_with_fitness, fitness


class MetaTask(Problem):
    def __init__(
        self,
        repaire: bool = True,
        dim: int = 10,
        inner_algo: BBORunner = None,
        inner_generations: int = 100,
        inner_funcs: Iterable = None,
        inner_popshape: Iterable = None,
    ):
        self.inner_algo = inner_algo
        self.inner_funcs = inner_funcs
        self.inner_popshape = inner_popshape
        self.inner_generation = inner_generations
        self.use_repaire = repaire
        self.dim = dim

        if (
            self.inner_algo is None
            or self.inner_funcs is None
            or self.inner_popshape is None
        ):
            raise ValueError(
                "parameters inner_algo, inner_funcs, inner_popshape cannot be None"
            )

    def evaluate_candidate(self, func):
        inner_task = taskProblem(fun=func, repaire=True, dim=self.inner_popshape[-1])
        pop = inner_task.genRandomPop(self.inner_popshape)
        pop, _ = inner_task.calfitness(pop)
        cur_best = pop[..., 0].view(-1).min()
        for _ in range(self.inner_generation):
            pop, _ = self.inner_algo.algo.step(pop, inner_task)
            if pop[..., 0].view(-1).min() < cur_best:
                cur_best = pop[..., 0].view(-1).min().clone()

        return cur_best

    def repaire(self, x):
        ub_mask = torch.zeros_like(x, device=DEVICE, dtype=DTYPE)
        lb_mask = torch.zeros_like(x, device=DEVICE, dtype=DTYPE)
        ub_mask[x > 1.0] = 1.0
        lb_mask[x < -1.0] = 1.0

        x = (1 - ub_mask) * x + ub_mask
        x = (1 - lb_mask) * x - lb_mask

        return x

    def calfitness(self, x):
        """
        :param x: parameter population with shape (B, N, D)
        """
        b, n, d = x.shape
        x = x.view(-1, x.shape[-1])
        meta_fitness = []
        for func in self.inner_funcs:
            genOffset(dim=self.inner_popshape[-1], fun=func)
            task_fitness = []
            for params in x:
                self.inner_algo.algo.reset()
                if self.use_repaire:
                    params = self.repaire(params)
                self.inner_algo.algo.update_params(params)
                param_fitness = self.evaluate_candidate(func=func)
                task_fitness.append(param_fitness)

            meta_fitness.append(task_fitness)
                
        meta_fitness = torch.tensor(meta_fitness, device=DEVICE, dtype=DTYPE)  # (K, N)
        mu = torch.mean(meta_fitness, dim=-1, keepdim=True)
        sigma = torch.std(meta_fitness, dim=-1, keepdim=True) + 1e-8
        meta_fitness = (meta_fitness - mu) / sigma
        meta_fitness = torch.median(meta_fitness, dim=0, keepdim=True).values

        meta_fitness = meta_fitness.unsqueeze(-1)
        x = x.view(b, n, d)
        x = torch.cat([meta_fitness, x], dim=-1)

        return x, meta_fitness

    def init_population(self, popshape):
        return torch.rand(size=popshape)
