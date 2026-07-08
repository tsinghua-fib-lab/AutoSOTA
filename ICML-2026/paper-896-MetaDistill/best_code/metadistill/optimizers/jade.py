import torch

from torch_basic_settings import DEVICE, DTYPE
from optimizers.base_model import GradFreeBBO
from optimizers.utils import sort_by_fitness


class JADE(GradFreeBBO):
    """
    PyTorch implementation for "Adaptive Differential Evolution with Optional External Archive" (JADE).
    Original paper: "JADE: Adaptive Differential Evolution with Optional External Archive",
    url: "https://ieeexplore.ieee.org/document/5208221".
    """

    def __init__(self, config: dict):
        GradFreeBBO.__init__(self)
        self.name = "JADE"

        self.c = float(config.get("c", 0.1))  # "learning rate"
        self.p = float(config.get("p", 0.05))  # elites rate
        self.popsize = int(config.get("popsize", 100))

        if self.popsize < 4:
            raise ValueError(f"popsize must be >= 4 for JADE, got {self.popsize}")
        self.problemdim = int(config["problemdim"])
        self.minimize = bool(config.get("minimize", True))

        self.mu_F = float(config.get("mu_F", 0.5))
        self.mu_CR = float(config.get("mu_CR", 0.5))
        self._init_mu_F = self.mu_F
        self._init_mu_CR = self.mu_CR
        self.archive = None

    def step(self, xp, problem):
        """
        :param xp: 种群及适应度, shape=(1, pop_size, 1+problem_dim); xp[..., 0]为适应度
        :param problem: BBO problem
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
        ## Sample F
        F = torch.rand((self.popsize,), device=DEVICE, dtype=DTYPE).clamp(
            1e-6, 1 - 1e-6
        )  # ~U(0, 1)
        F = self.mu_F + 0.1 * torch.tan(torch.pi * (F - 0.5))  # ~Cauthy(mu_F, 0.1)
        F = torch.clamp(F, min=0, max=1.0)

        ## Sample p_best[i]
        k = max(2, int(self.p * self.popsize))
        p_best = torch.randint(low=0, high=k, size=(self.popsize,), device=DEVICE)
        x_p_best = x[:, p_best, :]

        ## Sample r1, r2
        if self.archive is None or self.archive.numel() <= 0:
            union = x
            n_union = x.shape[1]
        else:
            union = torch.cat([x, self.archive], dim=1)
            n_union = union.shape[1]

        ref = torch.arange(self.popsize, device=DEVICE)
        r1 = torch.randint(
            low=0, high=self.popsize - 1, size=(self.popsize,), device=DEVICE
        )
        r1 = r1 + (r1 >= ref).long()

        r2 = torch.randint(low=0, high=n_union - 2, size=(self.popsize,), device=DEVICE)
        min_idx = torch.minimum(r1, ref)
        max_idx = torch.maximum(r1, ref)
        r2 = r2 + (r2 >= min_idx).long()
        r2 = r2 + (r2 >= max_idx).long()
        r2 = r2.clamp(max=n_union - 1)

        x_r1, x_r2 = x[:, r1, :], union[:, r2, :]

        ## Generate mutant population: v_i = x_i + F * (x_pbest_i - x_i) + F * (x_r1 - x_r2)
        x_mut = x + F.view(1, -1, 1) * (x_p_best - x) + F.view(1, -1, 1) * (x_r1 - x_r2)

        # Crossover
        ## Sample CR
        CR = (
            self.mu_CR + 0.1 * torch.randn((self.popsize,), device=DEVICE, dtype=DTYPE)
        ).clamp(0, 1)
        select_mask = (
            torch.rand((1, self.popsize, self.problemdim), device=DEVICE, dtype=DTYPE)
            < CR.view(1, -1, 1)
        ).to(DTYPE)
        j_rand = torch.randint(
            low=0, high=self.problemdim, size=(1, self.popsize, 1), device=DEVICE
        )
        select_mask = select_mask.scatter(-1, j_rand, 1).to(DTYPE)

        ## Generate candidates
        candidates = (1 - select_mask) * x + select_mask * x_mut
        candidates, _ = problem.calfitness(
            candidates
        )  # (1, n, 1+d), 在第一维拼接适应度

        # 1-to-1 Selection
        target_fit = xp[..., 0].squeeze(0)
        trial_fit = candidates[..., 0].squeeze(0)
        if self.minimize:
            replace_mask_1d = trial_fit <= target_fit
            success_mask = trial_fit < target_fit
        else:
            replace_mask_1d = trial_fit >= target_fit
            success_mask = trial_fit > target_fit

        replace_mask = replace_mask_1d.view(1, self.popsize, 1)
        offspring = torch.where(replace_mask, candidates, xp)

        # Adaptation
        ## Update archive (store replaced parents for strict improvements)
        if bool(success_mask.any()):
            a = x[:, success_mask, :]
            if a.numel() > 0:
                if self.archive is not None:
                    self.archive = torch.cat([self.archive, a], dim=1)
                else:
                    self.archive = a.clone()
        n_archive = 0 if self.archive is None else int(self.archive.shape[1])
        if n_archive > self.popsize:
            keep_idx = torch.randperm(n_archive, device=DEVICE)[: self.popsize]
            self.archive = self.archive[:, keep_idx, :]

        # Update distribution of F
        success_F = F[success_mask]
        if success_F.numel() > 0:
            L_mean = success_F.pow(2).sum() / success_F.sum()
            self.mu_F = (1 - self.c) * self.mu_F + self.c * L_mean

        # Update distribution of CR
        success_CR = CR[success_mask]
        if success_CR.numel() > 0:
            mean = success_CR.mean()
            self.mu_CR = (1 - self.c) * self.mu_CR + self.c * mean

        return offspring, {}

    def reset(self):
        self.mu_F = self._init_mu_F
        self.mu_CR = self._init_mu_CR
        self.archive = None
