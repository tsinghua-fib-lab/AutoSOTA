import torch
from torch_basic_settings import DEVICE, DTYPE
from optimizers.base_model import GradFreeBBO
from optimizers.utils import sort_by_fitness


class SHADE(GradFreeBBO):
    """PyTorch implementation of SHADE (Success-History based Adaptive Differential Evolution).

    Interface follows existing DE/JADE in `optimizers/de.py`:
    - input population `xp` includes fitness in first column with shape (1, popsize, 1+problemdim)
    - `problem.calfitness` evaluates candidates and returns the same shape

    Core SHADE components implemented:
    - success-history memory: M_F, M_CR (length H) with cyclic mem_idx
    - parameter sampling:
        F_i ~ Cauchy(M_F[r], 0.1), truncated to (0, 1]
        CR_i ~ Normal(M_CR[r], 0.1), clipped to [0, 1]
    - mutation: current-to-pbest/1 + archive
    - crossover: binomial
    - selection: 1-to-1
    - archive update: store replaced parents
    - memory update: weighted mean for CR, weighted Lehmer mean for F
    """

    def __init__(self, config: dict):
        GradFreeBBO.__init__(self)
        self.name = "SHADE"

        self.popsize = int(config.get("popsize", 100))

        if self.popsize < 4:
            raise ValueError(f"popsize must be >= 4 for SHADE, got {self.popsize}")
        self.problemdim = int(config["problemdim"])
        self.minimize = bool(config.get("minimize", True))

        self.p = float(config.get("p", 0.05))
        self.H = int(config.get("H", config.get("memory_size", 20)))
        self.archive_size = int(config.get("archive_size", self.popsize))

        self.seed = config.get("seed", None)

        self._rng = None
        self._rng_is_cuda = False

        # required internal states
        self.pop = None  # (NP, D)
        self.fit = None  # (NP,)
        self.best_x = None
        self.best_f = None
        self.archive = None  # (1, |A|, D)
        self.M_F = None  # (H,)
        self.M_CR = None  # (H,)
        self.mem_idx = 0
        self.fe_count = 0

        self.reset()
        if self.seed is not None:
            self.set_seed(self.seed)

    def _init_rng(self):
        try:
            self._rng = torch.Generator(device=DEVICE)
            self._rng_is_cuda = DEVICE.type == "cuda"
        except Exception:
            self._rng = torch.Generator()
            self._rng_is_cuda = False

    def set_seed(self, seed: int):
        seed = int(seed)
        self.seed = seed
        if self._rng is None:
            self._init_rng()
        self._rng.manual_seed(seed)

    def _rand(self, shape):
        if self._rng is None:
            self._init_rng()
        if DEVICE.type == "cuda" and not self._rng_is_cuda:
            return torch.rand(shape, device="cpu", dtype=DTYPE, generator=self._rng).to(
                DEVICE
            )
        return torch.rand(shape, device=DEVICE, dtype=DTYPE, generator=self._rng)

    def _randn(self, shape):
        if self._rng is None:
            self._init_rng()
        if DEVICE.type == "cuda" and not self._rng_is_cuda:
            return torch.randn(
                shape, device="cpu", dtype=DTYPE, generator=self._rng
            ).to(DEVICE)
        return torch.randn(shape, device=DEVICE, dtype=DTYPE, generator=self._rng)

    def _randint(self, low: int, high: int, shape):
        if self._rng is None:
            self._init_rng()
        if DEVICE.type == "cuda" and not self._rng_is_cuda:
            return torch.randint(
                low, high, shape, device="cpu", generator=self._rng
            ).to(DEVICE)
        return torch.randint(low, high, shape, device=DEVICE, generator=self._rng)

    def _randperm(self, n: int):
        if self._rng is None:
            self._init_rng()
        if DEVICE.type == "cuda" and not self._rng_is_cuda:
            return torch.randperm(n, device="cpu", generator=self._rng).to(DEVICE)
        return torch.randperm(n, device=DEVICE, generator=self._rng)

    def reset(self):
        self.archive = None

        self.M_F = torch.full((self.H,), 0.5, device=DEVICE, dtype=DTYPE)
        self.M_CR = torch.full((self.H,), 0.5, device=DEVICE, dtype=DTYPE)
        self.mem_idx = 0

        self.pop = None
        self.fit = None
        self.best_x = None
        self.best_f = None
        self.fe_count = 0

        if self._rng is None:
            self._init_rng()

    def _repair_bounds(self, x: torch.Tensor, problem, lb, ub) -> torch.Tensor:
        repaire = getattr(problem, "repaire", None)
        use_repaire = getattr(problem, "useRepaire", None)
        if callable(repaire) and (use_repaire is None or bool(use_repaire)):
            try:
                return repaire(x)
            except Exception:
                pass
        if lb is not None and ub is not None:
            return x.clamp(lb, ub)
        return x

    def _sort_pop(self, xp: torch.Tensor) -> torch.Tensor:
        xp = sort_by_fitness(xp)
        if not self.minimize:
            xp = xp.flip(dims=(1,))
        return xp

    def _sample_F_CR(self):
        mem_r = self._randint(0, self.H, (self.popsize,))
        mu_F = self.M_F[mem_r]
        mu_CR = self.M_CR[mem_r]

        u = self._rand((self.popsize,)).clamp(1e-12, 1.0 - 1e-12)
        F = mu_F + 0.1 * torch.tan(torch.pi * (u - 0.5))
        invalid = F <= 0
        for _ in range(16):
            if not invalid.any():
                break
            u2 = self._rand((int(invalid.sum().item()),)).clamp(1e-12, 1.0 - 1e-12)
            F2 = mu_F[invalid] + 0.1 * torch.tan(torch.pi * (u2 - 0.5))
            F[invalid] = F2
            invalid = F <= 0
        F = F.clamp(min=1e-6, max=1.0)

        CR = (mu_CR + 0.1 * self._randn((self.popsize,))).clamp(0.0, 1.0)
        return F.to(DTYPE), CR.to(DTYPE)

    def step(self, xp, problem):
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

        # Count the initial population evaluation once (xp already contains fitness).
        if self.fe_count == 0:
            self.fe_count = int(self.popsize)
        xp = self._sort_pop(xp)
        x = xp[..., 1:]  # (1, NP, D)

        # Bounds for stable clipping if available
        lb = None
        ub = None
        fun = getattr(problem, "fun", None)
        if isinstance(fun, dict):
            lb = fun.get("xlb", None)
            ub = fun.get("xub", None)

        # === Parameter sampling ===
        F, CR = self._sample_F_CR()  # (NP,), (NP,)

        # === current-to-pbest/1 mutation with archive ===
        k = max(2, int(self.p * self.popsize))
        k = min(k, self.popsize)
        pbest_idx = self._randint(0, k, (self.popsize,))
        x_pbest = x[:, pbest_idx, :]

        ref = torch.arange(self.popsize, device=DEVICE)
        r1 = self._randint(0, self.popsize - 1, (self.popsize,))
        r1 = r1 + (r1 >= ref).long()

        if self.archive is None or self.archive.numel() == 0:
            union = x
        else:
            union = torch.cat([x, self.archive], dim=1)
        n_union = int(union.shape[1])
        if n_union <= 2:
            union = x
            n_union = int(union.shape[1])

        r2 = self._randint(0, n_union - 2, (self.popsize,))
        min_idx = torch.minimum(r1, ref)
        max_idx = torch.maximum(r1, ref)
        r2 = r2 + (r2 >= min_idx).long()
        r2 = r2 + (r2 >= max_idx).long()
        r2 = r2.clamp(max=n_union - 1)

        x_r1 = x[:, r1, :]
        x_r2 = union[:, r2, :]

        F_b = F.view(1, -1, 1)
        v = x + F_b * (x_pbest - x) + F_b * (x_r1 - x_r2)

        # === Binomial crossover ===
        cross_mask = self._rand((1, self.popsize, self.problemdim)) < CR.view(1, -1, 1)
        j_rand = self._randint(0, self.problemdim, (1, self.popsize, 1))
        cross_mask = cross_mask.scatter(-1, j_rand, True)
        u = torch.where(cross_mask, v, x)
        u = self._repair_bounds(u, problem, lb, ub)

        # === Evaluation ===
        candidates, _ = problem.calfitness(u)  # (1, NP, 1+D)
        self.fe_count += int(self.popsize)

        # === Selection + success mask ===
        target_fit = xp[..., 0].squeeze(0)
        trial_fit = candidates[..., 0].squeeze(0)
        if self.minimize:
            replace_mask_1d = trial_fit <= target_fit
            success_mask = trial_fit < target_fit
            delta = (target_fit - trial_fit).clamp(min=0.0)
        else:
            replace_mask_1d = trial_fit >= target_fit
            success_mask = trial_fit > target_fit
            delta = (trial_fit - target_fit).clamp(min=0.0)

        replace_mask = replace_mask_1d.view(1, self.popsize, 1)
        offspring = torch.where(replace_mask, candidates, xp)

        # === Archive update ===
        if bool(success_mask.any()):
            replaced = x[:, success_mask, :]
            if replaced.numel() > 0:
                if self.archive is None:
                    self.archive = replaced.clone()
                else:
                    self.archive = torch.cat([self.archive, replaced], dim=1)
                n_archive = int(self.archive.shape[1])
                if n_archive > self.archive_size:
                    keep = self._randperm(n_archive)[: self.archive_size]
                    self.archive = self.archive[:, keep, :]

        # === Success-history update ===
        if bool(success_mask.any()):
            success_delta = delta[success_mask]
            success_F = F[success_mask]
            success_CR = CR[success_mask]
            if success_F.numel() > 0:
                w_sum = success_delta.sum()
                if (not torch.isfinite(w_sum)) or float(w_sum.item()) <= 0.0:
                    w = torch.full_like(success_F, 1.0 / float(success_F.numel()))
                else:
                    w = success_delta / w_sum

                m_cr = (w * success_CR).sum().clamp(0.0, 1.0)
                denom = (w * success_F).sum()
                if float(denom.item()) > 0.0:
                    m_f = (w * success_F.pow(2)).sum() / denom
                    m_f = m_f.clamp(min=1e-6, max=1.0)

                    self.M_CR[self.mem_idx] = m_cr
                    self.M_F[self.mem_idx] = m_f
                    self.mem_idx = (self.mem_idx + 1) % self.H

        # === Track state ===
        self.pop = offspring[0, :, 1:].detach()
        self.fit = offspring[0, :, 0].detach()

        if self.best_f is None:
            if self.minimize:
                best_idx = torch.argmin(self.fit)
            else:
                best_idx = torch.argmax(self.fit)
            self.best_f = self.fit[best_idx].clone()
            self.best_x = self.pop[best_idx].clone()
        else:
            if self.minimize:
                cur_best_idx = torch.argmin(self.fit)
                cur_best_f = self.fit[cur_best_idx]
                if cur_best_f < self.best_f:
                    self.best_f = cur_best_f.clone()
                    self.best_x = self.pop[cur_best_idx].clone()
            else:
                cur_best_idx = torch.argmax(self.fit)
                cur_best_f = self.fit[cur_best_idx]
                if cur_best_f > self.best_f:
                    self.best_f = cur_best_f.clone()
                    self.best_x = self.pop[cur_best_idx].clone()

        return offspring, {}
