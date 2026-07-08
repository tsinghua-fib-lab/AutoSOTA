import torch

from torch_basic_settings import DEVICE, DTYPE
from optimizers.base_model import GradFreeBBO
from optimizers.utils import one2one_selection, sort_by_fitness


class DE(GradFreeBBO):
    """
    PyTorch implementation for vanilla DE/rand/1
    """

    def __init__(self, config: dict):
        GradFreeBBO.__init__(self)
        self.name = "DE"

        self.F = float(config.get("F", 0.5))
        self.CR = float(config.get("CR", 0.5))
        self.popsize = int(config.get("popsize", 100))

        if self.popsize < 4:
            raise ValueError(f"popsize must be >= 4 for DE, got {self.popsize}")
        self.problemdim = int(config["problemdim"])
        self.minimize = bool(config.get("minimize", True))

    def step(self, xp, problem):
        """
        :param xp: Parent population with fitness, torch.Tensor, shape=(1, popsize, 1+problem_dim)
        :param problem: problem (mostly black-box) to be optimized, tasks.TaskProblem
        """
        if not isinstance(xp, torch.Tensor):
            raise TypeError(f"xp must be a torch.Tensor, got {type(xp)}")
        if (
            xp.dim() != 3
            or xp.shape[0] != 1
            or xp.shape[1] != self.popsize
            or xp.shape[-1] != self.problemdim + 1
        ):
            raise ValueError(
                f"Invalid population shape. Expected (1, {self.popsize}, {self.problemdim + 1}), got {tuple(xp.shape)}"
            )

        xp = xp.to(DEVICE).to(DTYPE)
        xp = sort_by_fitness(xp)
        if not self.minimize:
            xp = xp.flip(dims=(1,))

        x = xp[..., 1:]

        # Mutation
        # P(r1[i] != r2[i] != r3[i] != i) == 1, space: O(n^2)
        weight = torch.ones((self.popsize, self.popsize), device=DEVICE, dtype=DTYPE)
        mask = torch.eye(self.popsize, device=DEVICE, dtype=torch.bool)
        weight = weight.masked_fill(mask=mask, value=-torch.inf)
        weight = weight.softmax(dim=-1)

        r = torch.multinomial(weight, 3)
        r1, r2, r3 = r[..., 0], r[..., 1], r[..., 2]
        x_r1, x_r2, x_r3 = x[:, r1, :], x[:, r2, :], x[:, r3, :]

        x_mut = x_r1 + self.F * (x_r2 - x_r3)

        # Crossover
        select_mask = (
            torch.rand((1, self.popsize, self.problemdim), device=DEVICE, dtype=DTYPE)
            < self.CR
        ).to(DTYPE)

        j_rand = torch.randint(
            low=0, high=self.problemdim, size=(1, self.popsize, 1), device=DEVICE
        )
        select_mask = select_mask.scatter(-1, j_rand, 1).to(DTYPE)

        candidates = (1 - select_mask) * x + select_mask * x_mut
        candidates, _ = problem.calfitness(candidates)  # (1, n, 1+d)

        # 1-to-1 Selection
        offspring = one2one_selection(xp, candidates, minimize=self.minimize)

        return offspring, {}
