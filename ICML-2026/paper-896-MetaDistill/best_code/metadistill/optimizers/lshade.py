import math

import torch

from optimizers.base_model import GradFreeBBO
from optimizers.utils import sort_by_fitness
from torch_basic_settings import DEVICE, DTYPE


class LSHADE(GradFreeBBO):
    """PyTorch implementation of L-SHADE (SHADE + Linear Population Size Reduction).

    Interface contract matches DE/JADE/SHADE:
    - xp includes fitness in first column, shape (1, NP, 1+problemdim)
    - step() performs one generation step and returns offspring with the same layout

    Defaults (as approved):
    - np_min = 4
    - archive_size = archive_factor * NP_cur, with archive_factor=1.0
    - allow variable population size output: (1, NP_g, 1+D)

    Engineering guardrails:
    - LPSR progress excludes the initial population evaluation
    - pbest pool has a lower bound of 2
    - r1/r2 sampling uses defensive clamping and small-NP safeguards
    """

    def __init__(self, config: dict):
        GradFreeBBO.__init__(self)
        self.name = "LSHADE"
        self.needs_backward = False

        self.NP_init = int(config.get("popsize", 100))
        self.popsize = int(self.NP_init)
        self.problemdim = int(config["problemdim"])
        self.minimize = bool(config.get("minimize", True))

        self.NP_min = max(4, int(config.get("np_min", 4)))

        if self.NP_init < self.NP_min:
            raise ValueError(
                f"NP_init must be >= {self.NP_min} for LSHADE, got {self.NP_init}"
            )
        self.archive_factor = float(config.get("archive_factor", 1.0))

        self.p = float(config.get("p", 0.05))
        self.H = int(config.get("H", config.get("memory_size", 20)))

        self.max_fe = config.get("max_fe", None)
        if self.max_fe is not None:
            self.max_fe = int(self.max_fe)

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
        self.popsize = int(self.NP_init)
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

    def _sort_pop(self, xp: torch.Tensor) -> torch.Tensor:
        xp = sort_by_fitness(xp)
        if not self.minimize:
            xp = xp.flip(dims=(1,))
        return xp

    def _sample_F_CR(self, NP_cur: int):
        mem_r = self._randint(0, self.H, (NP_cur,))
        mu_F = self.M_F[mem_r]
        mu_CR = self.M_CR[mem_r]

        u = self._rand((NP_cur,)).clamp(1e-12, 1.0 - 1e-12)
        F = mu_F + 0.1 * torch.tan(torch.pi * (u - 0.5))

        invalid = F <= 0
        n_tries = 0
        while invalid.any():
            n_tries += 1
            if n_tries > 128:
                F = F.clamp(min=1e-6)
                break
            u2 = self._rand((int(invalid.sum().item()),)).clamp(1e-12, 1.0 - 1e-12)
            F2 = mu_F[invalid] + 0.1 * torch.tan(torch.pi * (u2 - 0.5))
            F[invalid] = F2
            invalid = F <= 0

        F = F.clamp(min=1e-6, max=1.0)
        CR = (mu_CR + 0.1 * self._randn((NP_cur,))).clamp(0.0, 1.0)

        return F.to(DTYPE), CR.to(DTYPE)

    def _compute_np_next(self, NP_cur: int) -> int:
        if self.max_fe is None or self.max_fe <= self.NP_init:
            return NP_cur

        denom = float(self.max_fe - self.NP_init)
        progress = (float(self.fe_count - self.NP_init) / denom) if denom > 0 else 1.0
        progress = max(0.0, min(1.0, progress))

        target = int(round(self.NP_init - progress * (self.NP_init - self.NP_min)))
        target = max(self.NP_min, min(self.NP_init, target))
        return min(NP_cur, target)

    def _trim_archive(self, cap: int):
        if self.archive is None or self.archive.numel() == 0:
            return
        cap = max(0, int(cap))
        n = int(self.archive.shape[1])
        if n <= cap:
            return
        keep = self._randperm(n)[:cap]
        self.archive = self.archive[:, keep, :]

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

    def step(self, xp, problem):
        if xp.dim() != 3 or xp.shape[0] != 1 or xp.shape[-1] != self.problemdim + 1:
            raise ValueError(
                f"Invalid population shape. Expected (1, NP, {self.problemdim + 1}), got {tuple(xp.shape)}"
            )

        xp = xp.to(DEVICE).to(DTYPE)

        NP_cur = int(xp.shape[1])
        if NP_cur < self.NP_min:
            raise ValueError(
                f"NP_cur={NP_cur} < NP_min={self.NP_min}; cannot run LSHADE"
            )
        self.popsize = NP_cur

        # Count the initial population evaluation once (xp already contains fitness).
        if self.fe_count == 0:
            self.fe_count = NP_cur

        xp = self._sort_pop(xp)
        x = xp[..., 1:]  # (1, NP, D)

        # Bounds for stable clipping if available
        lb = None
        ub = None
        fun = getattr(problem, "fun", None)
        if isinstance(fun, dict):
            lb = fun.get("xlb", None)
            ub = fun.get("xub", None)

        # === Parameter sampling (success-history) ===
        F, CR = self._sample_F_CR(NP_cur)

        # === current-to-pbest/1 mutation with archive ===
        pbest_pool = max(2, int(math.ceil(self.p * NP_cur)))
        pbest_pool = min(pbest_pool, NP_cur)
        pbest_idx = self._randint(0, pbest_pool, (NP_cur,))
        x_pbest = x[:, pbest_idx, :]

        ref = torch.arange(NP_cur, device=DEVICE)
        r1 = self._randint(0, NP_cur - 1, (NP_cur,))
        r1 = r1 + (r1 >= ref).long()

        if self.archive is None or self.archive.numel() == 0:
            union = x
        else:
            union = torch.cat([x, self.archive], dim=1)
        n_union = int(union.shape[1])
        if n_union < 3:
            union = x
            n_union = int(union.shape[1])

        # Sample r2 from union excluding i and r1 (JADE-style)
        r2 = self._randint(0, n_union - 2, (NP_cur,))
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
        cross_mask = self._rand((1, NP_cur, self.problemdim)) < CR.view(1, -1, 1)
        j_rand = self._randint(0, self.problemdim, (1, NP_cur, 1))
        cross_mask = cross_mask.scatter(-1, j_rand, True)
        u = torch.where(cross_mask, v, x)
        # === Boundary handling ===
        u = self._repair_bounds(u, problem, lb, ub)

        # === Evaluation ===
        candidates, _ = problem.calfitness(u)  # (1, NP, 1+D)
        self.fe_count += NP_cur

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

        replace_mask = replace_mask_1d.view(1, NP_cur, 1)
        offspring = torch.where(replace_mask, candidates, xp)

        # === Archive update ===
        if bool(success_mask.any()):
            replaced = x[:, success_mask, :]
            if replaced.numel() > 0:
                if self.archive is None:
                    self.archive = replaced.clone()
                else:
                    self.archive = torch.cat([self.archive, replaced], dim=1)

        self._trim_archive(cap=int(round(self.archive_factor * NP_cur)))

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

        # === LPSR (Linear Population Size Reduction) ===
        NP_next = self._compute_np_next(NP_cur)
        if NP_next < NP_cur:
            offspring = self._sort_pop(offspring)
            offspring = offspring[:, :NP_next, :]

        self.popsize = int(offspring.shape[1])
        self._trim_archive(cap=int(round(self.archive_factor * self.popsize)))

        # === Track state ===
        self.pop = offspring[0, :, 1:].detach()
        self.fit = offspring[0, :, 0].detach()

        if self.minimize:
            cur_best_idx = torch.argmin(self.fit)
            cur_best_f = self.fit[cur_best_idx]
            better = self.best_f is None or cur_best_f < self.best_f
        else:
            cur_best_idx = torch.argmax(self.fit)
            cur_best_f = self.fit[cur_best_idx]
            better = self.best_f is None or cur_best_f > self.best_f

        if better:
            self.best_f = cur_best_f.detach().clone()
            self.best_x = self.pop[cur_best_idx].detach().clone()

        return offspring, {}
