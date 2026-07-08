import math
from typing import Optional, Tuple, TYPE_CHECKING

import torch

from optimizers.base_model import GradFreeBBO
from torch_basic_settings import DEVICE, DTYPE

if TYPE_CHECKING:
    from tasks import TaskProblem


class GA(GradFreeBBO):
    """
    Real-coded Genetic Algorithm (GA) with:
    - Tournament selection
    - SBX crossover (Simulated Binary Crossover)
    - Polynomial mutation
    - (mu + lambda) elitist replacement

    Interface follows DE/JADE:
    - xp: torch.Tensor, shape=(1, popsize, 1+problemdim), xp[..., 0] is fitness
    - problem.calfitness(x) returns (pop_with_fitness, fitness)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.name = "GA"
        self.needs_backward = False

        self.popsize = int(config.get("popsize", 100))
        self.problemdim = int(config["problemdim"])
        self.minimize = bool(config.get("minimize", True))

        self.tournament_k = int(config.get("tournament_k", 3))
        self.pc = float(config.get("pc", 0.9))
        self.eta_c = float(config.get("eta_c", 15.0))
        self.pm = float(config.get("pm", 1.0 / max(1, self.problemdim)))
        self.eta_m = float(config.get("eta_m", 20.0))

        self.assume_fitness = bool(config.get("assume_fitness", False))
        self.seed = config.get("seed", None)

        self._rng = torch.Generator(device=DEVICE)
        if self.seed is not None:
            self._rng.manual_seed(int(self.seed))

        self.fe_count = 0
        self.best_x: Optional[torch.Tensor] = None
        self.best_f: Optional[torch.Tensor] = None

        if self.popsize < 2:
            raise ValueError(f"GA requires popsize>=2, got {self.popsize}")
        if self.tournament_k < 1:
            raise ValueError(f"tournament_k must be >=1, got {self.tournament_k}")
        if not (0.0 <= self.pc <= 1.0):
            raise ValueError(f"pc must be in [0, 1], got {self.pc}")
        if not (0.0 <= self.pm <= 1.0):
            raise ValueError(f"pm must be in [0, 1], got {self.pm}")
        if self.eta_c <= 0:
            raise ValueError(f"eta_c must be > 0, got {self.eta_c}")
        if self.eta_m <= 0:
            raise ValueError(f"eta_m must be > 0, got {self.eta_m}")

    def reset(self):
        self.fe_count = 0
        self.best_x = None
        self.best_f = None
        if self.seed is not None:
            self._rng.manual_seed(int(self.seed))

    def _assert_xp_shape(self, xp: torch.Tensor) -> None:
        expected = (1, self.popsize, self.problemdim + 1)
        if xp.ndim != 3 or tuple(xp.shape) != expected:
            raise ValueError(
                f"Invalid population shape. Expected {expected}, got {tuple(xp.shape)}"
            )

    def _rand(self, shape, device, dtype=None) -> torch.Tensor:
        return torch.rand(shape, generator=self._rng, device=device, dtype=dtype)

    def _randint(self, low: int, high: int, shape, device) -> torch.Tensor:
        return torch.randint(
            low=low, high=high, size=shape, generator=self._rng, device=device
        )

    def _try_get_bounds(
        self, problem: "TaskProblem", device, dtype
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        fun = getattr(problem, "fun", None)
        if not isinstance(fun, dict) or "xlb" not in fun or "xub" not in fun:
            return None

        lb = torch.as_tensor(fun["xlb"], device=device, dtype=dtype)
        ub = torch.as_tensor(fun["xub"], device=device, dtype=dtype)

        if lb.ndim == 0:
            lb = lb.view(1, 1, 1).expand(1, 1, self.problemdim)
        else:
            lb = lb.view(1, 1, -1)
        if ub.ndim == 0:
            ub = ub.view(1, 1, 1).expand(1, 1, self.problemdim)
        else:
            ub = ub.view(1, 1, -1)

        if lb.shape[-1] != self.problemdim or ub.shape[-1] != self.problemdim:
            raise ValueError(
                f"Bounds dimension mismatch: got lb={tuple(lb.shape)}, ub={tuple(ub.shape)}, D={self.problemdim}"
            )
        return lb, ub

    def _repaire(self, x: torch.Tensor, problem: "TaskProblem") -> torch.Tensor:
        if (
            hasattr(problem, "useRepaire")
            and bool(getattr(problem, "useRepaire"))
            and hasattr(problem, "repaire")
        ):
            return problem.repaire(x)

        bounds = self._try_get_bounds(problem, device=x.device, dtype=x.dtype)
        if bounds is not None:
            lb, ub = bounds
            return x.clamp(lb, ub)
        return x

    def _tournament_select(self, fit_1d: torch.Tensor, n_select: int) -> torch.Tensor:
        device = fit_1d.device
        cand = self._randint(
            0, self.popsize, (n_select, self.tournament_k), device=device
        )  # (n_select, k)
        fit_cand = fit_1d[cand]  # (n_select, k)
        best_pos = (
            torch.argmin(fit_cand, dim=1)
            if self.minimize
            else torch.argmax(fit_cand, dim=1)
        )
        sel = cand.gather(1, best_pos.view(-1, 1)).squeeze(1)  # (n_select,)
        return sel

    def _sbx(self, p1: torch.Tensor, p2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # p1/p2: (1, n_pairs, D)
        device, dtype = p1.device, p1.dtype
        n_pairs = p1.shape[1]
        do_cross = (self._rand((1, n_pairs, 1), device=device) < self.pc)

        eps = torch.finfo(dtype).eps
        u = self._rand((1, n_pairs, self.problemdim), device=device, dtype=dtype).clamp(
            min=eps, max=1.0 - eps
        )
        eta = float(self.eta_c)
        inv = 1.0 / (eta + 1.0)

        beta = torch.where(
            u <= 0.5,
            (2.0 * u).pow(inv),
            (1.0 / (2.0 * (1.0 - u))).pow(inv),
        )
        c1_cross = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
        c2_cross = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)

        c1 = torch.where(do_cross, c1_cross, p1)
        c2 = torch.where(do_cross, c2_cross, p2)
        return c1, c2

    def _polynomial_mutation(self, x: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
        # x: (1, n, D), lb/ub: (1, 1, D)
        device, dtype = x.device, x.dtype
        n = x.shape[1]
        D = x.shape[2]

        span = ub - lb
        span_ok = span > 0
        span_safe = torch.where(span_ok, span, torch.ones_like(span))

        mut_mask = (self._rand((1, n, D), device=device) < self.pm) & span_ok
        if not bool(mut_mask.any()):
            return x

        x = x.clamp(lb, ub)
        eps = torch.finfo(dtype).eps
        u = self._rand((1, n, D), device=device, dtype=dtype).clamp(min=eps, max=1.0 - eps)

        delta1 = (x - lb) / span_safe
        delta2 = (ub - x) / span_safe

        eta_m = float(self.eta_m)
        mut_pow = 1.0 / (eta_m + 1.0)

        xy1 = 1.0 - delta1
        val1 = 2.0 * u + (1.0 - 2.0 * u) * xy1.pow(eta_m + 1.0)
        deltaq1 = val1.pow(mut_pow) - 1.0

        xy2 = 1.0 - delta2
        val2 = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * xy2.pow(eta_m + 1.0)
        deltaq2 = 1.0 - val2.pow(mut_pow)

        deltaq = torch.where(u <= 0.5, deltaq1, deltaq2)
        x_mut = x + deltaq * span

        x_next = torch.where(mut_mask, x_mut, x)
        return x_next.clamp(lb, ub)

    def _update_best(self, pop: torch.Tensor) -> None:
        fit = pop[..., 0].squeeze(0)  # (n,)
        best_idx = torch.argmin(fit) if self.minimize else torch.argmax(fit)

        best_f = fit[best_idx].detach()
        best_x = pop[:, best_idx : best_idx + 1, 1:].detach()

        if self.best_f is None:
            self.best_f = best_f
            self.best_x = best_x
            return

        improved = best_f < self.best_f if self.minimize else best_f > self.best_f
        if bool(improved):
            self.best_f = best_f
            self.best_x = best_x

    def step(self, xp: torch.Tensor, problem: "TaskProblem"):
        self._assert_xp_shape(xp)
        xp = xp.to(DEVICE).to(DTYPE)

        # Robust mode (default): backfill fitness only if non-finite values exist.
        if not self.assume_fitness:
            fit = xp[..., 0]
            if not bool(torch.isfinite(fit).all()):
                xp, _ = problem.calfitness(xp[..., 1:])
                self.fe_count += int(xp.shape[1])

        x = xp[..., 1:]  # (1, n, D)
        fit_1d = xp[..., 0].squeeze(0)  # (n,)
        bounds = self._try_get_bounds(problem, device=x.device, dtype=x.dtype)

        # Parent selection
        n_pairs = int(math.ceil(self.popsize / 2))
        n_parents = 2 * n_pairs
        parent_idx = self._tournament_select(fit_1d, n_parents)  # (2*n_pairs,)
        parents = x[:, parent_idx, :].view(1, n_pairs, 2, self.problemdim)
        p1, p2 = parents[:, :, 0, :], parents[:, :, 1, :]

        # SBX crossover
        c1, c2 = self._sbx(p1, p2)
        children = torch.stack([c1, c2], dim=2).view(1, n_parents, self.problemdim)
        children = children[:, : self.popsize, :]

        # Repair + polynomial mutation + repair
        children = self._repaire(children, problem)
        if bounds is not None:
            lb, ub = bounds
            children = self._polynomial_mutation(children, lb=lb, ub=ub)
        children = self._repaire(children, problem)

        # Evaluate children
        children_pop, _ = problem.calfitness(children)
        self.fe_count += int(children_pop.shape[1])

        # (mu + lambda) elitist replacement
        combined = torch.cat([xp, children_pop], dim=1)  # (1, 2n, 1+D)
        fit_c = combined[..., 0]
        idx = torch.argsort(fit_c, dim=1, descending=not self.minimize)
        idx = idx[:, : self.popsize]
        next_pop = torch.gather(combined, dim=1, index=idx.unsqueeze(-1).expand_as(xp))

        self._update_best(next_pop)
        return next_pop, {}
