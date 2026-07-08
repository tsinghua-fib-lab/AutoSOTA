"""
Particle Swarm Optimization (PSO) - Standard gbest topology.

Real-coded, torch-only, one forward() call = one generation.
Interface aligned with DE/JADE/GA: forward(xp, problem).
"""

from typing import Optional, Tuple, TYPE_CHECKING

import torch

from optimizers.base_model import GradFreeBBO
from torch_basic_settings import DEVICE, DTYPE

if TYPE_CHECKING:
    from tasks import TaskProblem


class PSO(GradFreeBBO):
    """
    Standard gbest-PSO (global best topology) with:
    - Constriction-style parameters (w=0.7298, c1=c2=1.49618)
    - Velocity clamping with optional zero-on-clip
    - Boundary repair (problem.repaire or clamp)

    Interface follows DE/JADE/GA:
    - xp: torch.Tensor, shape=(1, popsize, 1+problemdim), xp[..., 0] is fitness
    - problem.calfitness(x) returns (pop_with_fitness, fitness)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.name = "PSO"
        self.needs_backward = False

        # Population parameters
        self.popsize = int(config.get("popsize", 100))
        self.problemdim = int(config["problemdim"])
        self.minimize = bool(config.get("minimize", True))

        # PSO-specific parameters (constriction-style defaults)
        self.w = float(config.get("w", 0.7298))  # inertia weight
        self.c1 = float(config.get("c1", 1.49618))  # cognitive coefficient
        self.c2 = float(config.get("c2", 1.49618))  # social coefficient

        # Velocity constraints
        self.v_max_ratio = float(config.get("v_max_ratio", 0.2))
        self.v_init_ratio = float(config.get("v_init_ratio", 0.1))
        self.zero_v_on_clip = bool(config.get("zero_v_on_clip", True))

        # Fitness handling
        self.assume_fitness = bool(config.get("assume_fitness", False))
        self.seed = config.get("seed", None)

        # Local RNG for reproducibility
        self._rng = torch.Generator(device=DEVICE)
        if self.seed is not None:
            self._rng.manual_seed(int(self.seed))

        # Persistent state (initialized on first forward)
        self.v: Optional[torch.Tensor] = None  # velocity (1, NP, D)
        self.pbest_x: Optional[torch.Tensor] = None  # personal best positions
        self.pbest_f: Optional[torch.Tensor] = None  # personal best fitness
        self.gbest_x: Optional[torch.Tensor] = None  # global best position
        self.gbest_f: Optional[torch.Tensor] = None  # global best fitness
        self.fe_count = 0

        # Validation
        if self.popsize < 1:
            raise ValueError(f"PSO requires popsize>=1, got {self.popsize}")
        if self.w < 0:
            raise ValueError(f"w (inertia) must be >= 0, got {self.w}")
        if self.c1 < 0 or self.c2 < 0:
            raise ValueError(f"c1, c2 must be >= 0, got c1={self.c1}, c2={self.c2}")

    def reset(self):
        """Reset all adaptive state for a new optimization run."""
        self.v = None
        self.pbest_x = None
        self.pbest_f = None
        self.gbest_x = None
        self.gbest_f = None
        self.fe_count = 0
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

    def _try_get_bounds(
        self, problem: "TaskProblem", device, dtype
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Attempt to extract bounds from problem.fun.
        Returns None if bounds are not available (graceful degradation).
        """
        fun = getattr(problem, "fun", None)
        if not isinstance(fun, dict) or "xlb" not in fun or "xub" not in fun:
            return None

        lb = torch.as_tensor(fun["xlb"], device=device, dtype=dtype)
        ub = torch.as_tensor(fun["xub"], device=device, dtype=dtype)

        # Reshape to (1, 1, D) for broadcasting
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
                f"Bounds dimension mismatch: got lb={tuple(lb.shape)}, "
                f"ub={tuple(ub.shape)}, D={self.problemdim}"
            )
        return lb, ub

    def _repaire(self, x: torch.Tensor, problem: "TaskProblem") -> torch.Tensor:
        """Apply problem.repaire if available, otherwise clamp to bounds."""
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

    def _ensure_fitness(
        self, xp: torch.Tensor, problem: "TaskProblem"
    ) -> Tuple[torch.Tensor, int]:
        """
        Ensure xp has valid (finite) fitness values.
        Returns (evaluated_xp, fe_used).
        """
        if self.assume_fitness:
            return xp, 0
        fit = xp[..., 0]
        if bool(torch.isfinite(fit).all()):
            return xp, 0
        evaluated, _ = problem.calfitness(xp[..., 1:])
        return evaluated, int(evaluated.shape[1])

    def _init_state(
        self, x: torch.Tensor, fit: torch.Tensor, problem: "TaskProblem"
    ) -> None:
        """Initialize velocity, pbest, and gbest on first forward call."""
        device, dtype = x.device, x.dtype

        # Initialize velocity
        bounds = self._try_get_bounds(problem, device, dtype)
        if bounds is not None:
            lb, ub = bounds
            span = ub - lb
            # v ~ U(-v_init_ratio*span, +v_init_ratio*span)
            self.v = (2 * self._rand(x.shape, device, dtype) - 1) * self.v_init_ratio * span
        else:
            # No bounds: initialize velocity to zero (safest)
            self.v = torch.zeros_like(x)

        # Initialize personal best
        self.pbest_x = x.clone()
        self.pbest_f = fit.clone()  # (1, NP)

        # Initialize global best from pbest
        self._update_gbest_from_pbest()

    def _update_gbest_from_pbest(self) -> None:
        """Update global best from current personal bests."""
        fit_1d = self.pbest_f.squeeze(0)  # (NP,)
        best_idx = torch.argmin(fit_1d) if self.minimize else torch.argmax(fit_1d)

        candidate_f = fit_1d[best_idx].detach()
        candidate_x = self.pbest_x[:, best_idx : best_idx + 1, :].detach()

        if self.gbest_f is None:
            self.gbest_f = candidate_f
            self.gbest_x = candidate_x
            return

        improved = candidate_f < self.gbest_f if self.minimize else candidate_f > self.gbest_f
        if bool(improved):
            self.gbest_f = candidate_f
            self.gbest_x = candidate_x

    def step(self, xp: torch.Tensor, problem: "TaskProblem"):
        """
        Perform one generation of PSO.

        Args:
            xp: Population tensor, shape=(1, popsize, 1+problemdim)
                xp[..., 0] = fitness, xp[..., 1:] = decision variables
            problem: TaskProblem instance with calfitness method

        Returns:
            Next generation population with same shape as input
        """
        self._assert_xp_shape(xp)
        xp = xp.to(DEVICE).to(DTYPE)

        # Ensure fitness is valid (backfill if needed)
        xp, fe_used = self._ensure_fitness(xp, problem)
        self.fe_count += fe_used

        x = xp[..., 1:]  # (1, NP, D)
        fit = xp[..., 0]  # (1, NP)
        device, dtype = x.device, x.dtype

        # Initialize state on first call
        if self.v is None:
            self._init_state(x, fit, problem)

        # Get bounds for velocity clamping
        bounds = self._try_get_bounds(problem, device, dtype)

        # Sample random coefficients r1, r2 ~ U(0, 1)
        r1 = self._rand((1, self.popsize, self.problemdim), device, dtype)
        r2 = self._rand((1, self.popsize, self.problemdim), device, dtype)

        # Update velocity: v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        cognitive = self.c1 * r1 * (self.pbest_x - x)
        social = self.c2 * r2 * (self.gbest_x - x)  # gbest broadcasts to (1, NP, D)
        v_new = self.w * self.v + cognitive + social

        # Apply velocity clamping if bounds available
        if bounds is not None:
            lb, ub = bounds
            vmax = self.v_max_ratio * (ub - lb)  # (1, 1, D)
            v_new = v_new.clamp(-vmax, vmax)

        # Update position: x_new = x + v
        x_new = x + v_new

        # Apply boundary handling
        if bounds is not None:
            lb, ub = bounds
            # Detect which dimensions hit bounds (before clamping)
            out_of_bounds = (x_new < lb) | (x_new > ub)
            # Clamp positions
            x_new = x_new.clamp(lb, ub)
            # Zero velocity on clipped dimensions (if enabled)
            if self.zero_v_on_clip:
                v_new = torch.where(out_of_bounds, torch.zeros_like(v_new), v_new)

        # Apply problem-specific repair (may override clamp)
        x_new = self._repaire(x_new, problem)

        # Update stored velocity
        self.v = v_new

        # Evaluate new positions
        new_pop, _ = problem.calfitness(x_new)
        self.fe_count += int(new_pop.shape[1])
        new_fit = new_pop[..., 0]  # (1, NP)

        # Update personal best (strict improvement only)
        improved = new_fit < self.pbest_f if self.minimize else new_fit > self.pbest_f
        self.pbest_x = torch.where(improved.unsqueeze(-1), x_new, self.pbest_x)
        self.pbest_f = torch.where(improved, new_fit, self.pbest_f)

        # Update global best from personal bests
        self._update_gbest_from_pbest()

        return new_pop, {}
