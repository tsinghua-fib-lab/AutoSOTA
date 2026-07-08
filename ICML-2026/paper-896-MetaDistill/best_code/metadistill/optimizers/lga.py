import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from optimizers.base_model import GradFreeBBO, GradTrainedBBO
from torch_basic_settings import DEVICE, DTYPE


torch.set_default_dtype(DTYPE)
torch.set_default_device(DEVICE)


def _centered_ranks(fitness: torch.Tensor, minimize: bool) -> torch.Tensor:
    """
    Fitness -> centered ranks in [-0.5, 0.5], best = -0.5.
    fitness: (N,)
    """
    if minimize:
        sorted_idx = torch.argsort(fitness, dim=0)
    else:
        sorted_idx = torch.argsort(fitness, dim=0, descending=True)
    rank_idx = torch.argsort(sorted_idx, dim=0).to(DTYPE)
    if fitness.numel() <= 1:
        return torch.zeros_like(rank_idx)
    return rank_idx / (fitness.numel() - 1) - 0.5


def _z_score(x: torch.Tensor) -> torch.Tensor:
    mu = x.mean()
    sigma = x.std().clamp_min(1e-8)
    return (x - mu) / sigma


def _get_bounds_or_fail(problem, device, dtype, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    fun = getattr(problem, "fun", None)
    if isinstance(fun, dict) and "xlb" in fun and "xub" in fun:
        lb = torch.as_tensor(fun["xlb"], device=device, dtype=dtype)
        ub = torch.as_tensor(fun["xub"], device=device, dtype=dtype)
    elif hasattr(problem, "xlb") and hasattr(problem, "xub"):
        lb = torch.as_tensor(problem.xlb, device=device, dtype=dtype)
        ub = torch.as_tensor(problem.xub, device=device, dtype=dtype)
    else:
        raise ValueError(
            "LGA requires explicit bounds. Please provide one of:\n"
            "  1. problem.fun['xlb'] and problem.fun['xub']\n"
            "  2. problem.xlb and problem.xub attributes"
        )

    if lb.ndim == 0:
        lb = lb.view(1, 1, 1).expand(1, 1, dim)
    else:
        lb = lb.view(1, 1, -1)
    if ub.ndim == 0:
        ub = ub.view(1, 1, 1).expand(1, 1, dim)
    else:
        ub = ub.view(1, 1, -1)

    if lb.shape[-1] != dim or ub.shape[-1] != dim:
        raise ValueError(f"Bounds dimension mismatch: lb={tuple(lb.shape)}, ub={tuple(ub.shape)}, expected D={dim}")
    return lb, ub


def _gumbel_softmax_st(
    logits: torch.Tensor,
    tau: float,
    hard: bool,
    rng: Optional[torch.Generator],
    dim: int = -1,
    eps: float = 1e-20,
) -> torch.Tensor:
    """
    Straight-through Gumbel-Softmax with an explicit RNG (so it matches other optimizers' seeded behavior).
    """
    tau = float(tau)
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")

    u = torch.rand(logits.shape, generator=rng, device=logits.device, dtype=logits.dtype).clamp(min=eps, max=1.0 - eps)
    g = -torch.log(-torch.log(u))
    y = torch.softmax((logits + g) / tau, dim=dim)

    if not hard:
        return y

    index = y.max(dim=dim, keepdim=True)[1]
    y_hard = torch.zeros_like(y).scatter_(dim, index, 1.0)
    return (y_hard - y).detach() + y


class LGA(GradFreeBBO):
    """
    Learned Genetic Algorithm (LGA) - GECCO 2023 implementation (core operators).

    Paper: "Discovering Attention-Based Genetic Algorithms via Meta-Black-Box Optimization"

    This implementation follows Sec.4 "Attention-Based Genetic Operators":
    - Selection via cross-attention producing row-stochastic selection matrix M_S (Eq. around lines 433-506 in paper text)
      and categorical sampling per parent slot: S ~ Categorical(M_S).
    - Mutation Rate Adaptation (MRA) via self-attention on sampled parent features:
        delta_sigma = exp(0.5 * A_M W_sigma),  sigma_C = delta_sigma * sigma_P'
      (paper lines 509-532).

    Notes (strict-to-paper within this repo constraints):
    - Sampling is uniform-with-replacement (no learned sampling operator; Appendix A in paper).
    - No crossover operator (paper focuses on selection + MRA in main text; crossover is optional in Appendix A).
    - Population/archive size: E = N = popsize (paper default during meta-training).
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.name = "LGA"
        self.needs_backward = False

        self.popsize = int(config.get("popsize", 16))
        self.problemdim = int(config["problemdim"])
        self.minimize = bool(config.get("minimize", True))

        # Paper hyperparameters: D_K = 16 (Sec.4), <1500 params
        self.dk = int(config.get("dk", 16))
        self.df = 2  # z-score + centered rank (paper bullet (1))
        self.dsigma = 2  # z-score + [-1,1] normalization (paper bullet (5))
        self.fm_dim = int(self.df + 1 + self.dsigma)  # + best-so-far boolean (paper bullet (6))

        self.initial_sigma = float(config.get("sigma_init", 0.1))
        self.sigma_min = float(config.get("sigma_min", 1e-6))
        self.sigma_max = float(config.get("sigma_max", 10.0))

        self.seed = config.get("seed", None)
        self._rng = torch.Generator(device=DEVICE)
        if self.seed is not None:
            self._rng.manual_seed(int(self.seed))

        # ===== Learned parameters theta (paper line 533) =====
        # Selection (cross-attention):
        self.W_QP = torch.zeros((self.df, self.dk), device=DEVICE, dtype=DTYPE)
        self.W_KC = torch.zeros((self.df, self.dk), device=DEVICE, dtype=DTYPE)
        self.W_VC = torch.zeros((self.df, self.dk), device=DEVICE, dtype=DTYPE)
        self.W_QS = torch.zeros((self.dk, self.dk), device=DEVICE, dtype=DTYPE)
        self.W_KS = torch.zeros((self.df, self.dk), device=DEVICE, dtype=DTYPE)

        # MRA (self-attention):
        self.W_QM = torch.zeros((self.fm_dim, self.dk), device=DEVICE, dtype=DTYPE)
        self.W_KM = torch.zeros((self.fm_dim, self.dk), device=DEVICE, dtype=DTYPE)
        self.W_VM = torch.zeros((self.fm_dim, self.dk), device=DEVICE, dtype=DTYPE)
        self.W_sigma = torch.zeros((self.dk, 1), device=DEVICE, dtype=DTYPE)

        self.n_params = (
            self.W_QP.numel()
            + self.W_KC.numel()
            + self.W_VC.numel()
            + self.W_QS.numel()
            + self.W_KS.numel()
            + self.W_QM.numel()
            + self.W_KM.numel()
            + self.W_VM.numel()
            + self.W_sigma.numel()
        )

        # State: parent mutation rates sigma_P in R^E (E=N)
        self._sigma_parents = torch.full((self.popsize,), self.initial_sigma, device=DEVICE, dtype=DTYPE)
        self._best_fitness: Optional[torch.Tensor] = None

        # Small random init for usability before meta-training (MetaBBO will override via update_params)
        init_scale = float(config.get("param_init_scale", 0.02))
        self._init_params(init_scale)

        if self.popsize < 2:
            raise ValueError(f"LGA requires popsize >= 2, got {self.popsize}")

    def _init_params(self, scale: float) -> None:
        scale = float(scale)
        self.W_QP = torch.randn(self.W_QP.shape, generator=self._rng, device=self.W_QP.device, dtype=self.W_QP.dtype) * scale
        self.W_KC = torch.randn(self.W_KC.shape, generator=self._rng, device=self.W_KC.device, dtype=self.W_KC.dtype) * scale
        self.W_VC = torch.randn(self.W_VC.shape, generator=self._rng, device=self.W_VC.device, dtype=self.W_VC.dtype) * scale
        self.W_QS = torch.randn(self.W_QS.shape, generator=self._rng, device=self.W_QS.device, dtype=self.W_QS.dtype) * scale
        self.W_KS = torch.randn(self.W_KS.shape, generator=self._rng, device=self.W_KS.device, dtype=self.W_KS.dtype) * scale
        self.W_QM = torch.randn(self.W_QM.shape, generator=self._rng, device=self.W_QM.device, dtype=self.W_QM.dtype) * scale
        self.W_KM = torch.randn(self.W_KM.shape, generator=self._rng, device=self.W_KM.device, dtype=self.W_KM.dtype) * scale
        self.W_VM = torch.randn(self.W_VM.shape, generator=self._rng, device=self.W_VM.device, dtype=self.W_VM.dtype) * scale
        self.W_sigma = torch.randn(self.W_sigma.shape, generator=self._rng, device=self.W_sigma.device, dtype=self.W_sigma.dtype) * scale

    def reset(self, bounds=None):
        self._sigma_parents = torch.full((self.popsize,), self.initial_sigma, device=DEVICE, dtype=DTYPE)
        self._best_fitness = None
        if self.seed is not None:
            self._rng.manual_seed(int(self.seed))

    # -------------------------------------------------------------------------
    # MetaBBO parameter-vector interface
    # -------------------------------------------------------------------------

    def update_params(self, params: torch.Tensor) -> None:
        params = params.to(device=DEVICE, dtype=DTYPE).view(-1)
        if params.numel() != self.n_params:
            raise ValueError(f"LGA params size mismatch: got {params.numel()}, expected {self.n_params}")

        idx = 0

        def take(num: int) -> torch.Tensor:
            nonlocal idx
            out = params[idx:idx + num]
            idx += num
            return out

        self.W_QP = take(self.W_QP.numel()).view_as(self.W_QP).clone()
        self.W_KC = take(self.W_KC.numel()).view_as(self.W_KC).clone()
        self.W_VC = take(self.W_VC.numel()).view_as(self.W_VC).clone()
        self.W_QS = take(self.W_QS.numel()).view_as(self.W_QS).clone()
        self.W_KS = take(self.W_KS.numel()).view_as(self.W_KS).clone()
        self.W_QM = take(self.W_QM.numel()).view_as(self.W_QM).clone()
        self.W_KM = take(self.W_KM.numel()).view_as(self.W_KM).clone()
        self.W_VM = take(self.W_VM.numel()).view_as(self.W_VM).clone()
        self.W_sigma = take(self.W_sigma.numel()).view_as(self.W_sigma).clone()

        if idx != params.numel():
            raise RuntimeError("LGA param slicing error")

    # -------------------------------------------------------------------------
    # Core operators (paper Sec.4)
    # -------------------------------------------------------------------------

    def _sample_parents_uniform(self, xp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample N parents with replacement uniformly from archive (E=N).
        Returns:
            xP_tilde: (N, D)
            fP_tilde: (N,)
            sigmaP_tilde: (N,)
        """
        xP = xp[0, :, 1:]  # (N, D)
        fP = xp[0, :, 0].view(-1)  # (N,)
        idx = torch.randint(0, self.popsize, (self.popsize,), generator=self._rng, device=DEVICE)
        return xP[idx], fP[idx], self._sigma_parents[idx]

    def _mra_sigma(self, fP_tilde: torch.Tensor, sigmaP_tilde: torch.Tensor) -> torch.Tensor:
        """
        MRA via self-attention (paper lines 509-532):
          A_M = softmax(Q_M K_M^T / sqrt(DK)) V_M
          delta_sigma = exp(0.5 * A_M W_sigma)
          sigma_C = delta_sigma * sigma_P'
        """
        # Fitness transforms (separate on sampled parents): z-score + centered rank
        f_z = _z_score(fP_tilde).clamp(-5.0, 5.0)
        f_r = _centered_ranks(fP_tilde, minimize=self.minimize)

        if self._best_fitness is None:
            best = fP_tilde.min() if self.minimize else fP_tilde.max()
            self._best_fitness = best.detach().clone()
        if self.minimize:
            f_best_flag = (fP_tilde < self._best_fitness).to(DTYPE)
        else:
            f_best_flag = (fP_tilde > self._best_fitness).to(DTYPE)

        # Sigma transforms: z-score + [-1,1] normalization
        s_z = _z_score(sigmaP_tilde).clamp(-5.0, 5.0)
        s_min = sigmaP_tilde.min()
        s_max = sigmaP_tilde.max()
        s_norm = (2.0 * (sigmaP_tilde - s_min) / (s_max - s_min + 1e-8) - 1.0).clamp(-1.0, 1.0)

        FM = torch.stack([f_z, f_r, f_best_flag, s_z, s_norm], dim=-1)  # (N, 5)

        Q = FM @ self.W_QM  # (N, DK)
        K = FM @ self.W_KM  # (N, DK)
        V = FM @ self.W_VM  # (N, DK)

        attn = (Q @ K.transpose(0, 1)) / math.sqrt(self.dk)  # (N, N)
        A = torch.softmax(attn, dim=-1) @ V  # (N, DK)
        delta_sigma = torch.exp(0.5 * (A @ self.W_sigma).squeeze(-1))  # (N,)

        sigmaC = (delta_sigma * sigmaP_tilde).clamp(self.sigma_min, self.sigma_max)
        return sigmaC

    def _selection_matrix(self, fC: torch.Tensor, fP: torch.Tensor) -> torch.Tensor:
        """
        Build selection probability matrix M_S in R^{E x (N+1)} (paper lines 433-453).

        Inputs:
            fC: (N,) children fitness
            fP: (E=N,) parent fitness
        Returns:
            M_S: (E, N+1) row-stochastic
        """
        # Joint fitness transforms F in R^{(N+E) x DF} (paper bullet (1))
        f_all = torch.cat([fC, fP], dim=0)  # (2N,)
        z_all = _z_score(f_all).clamp(-5.0, 5.0)
        r_all = _centered_ranks(f_all, minimize=self.minimize)
        F_all = torch.stack([z_all, r_all], dim=-1)  # (2N, 2)

        FC = F_all[: self.popsize, :]  # (N, DF)
        FP = F_all[self.popsize :, :]  # (E, DF)

        # Cross-attention: A_S = softmax(Q_P K_C^T / sqrt(DK)) V_C
        QP = FP @ self.W_QP  # (E, DK)
        KC = FC @ self.W_KC  # (N, DK)
        VC = FC @ self.W_VC  # (N, DK)
        AS = torch.softmax((QP @ KC.transpose(0, 1)) / math.sqrt(self.dk), dim=-1) @ VC  # (E, DK)

        # Second stage to construct selection matrix
        QS = AS @ self.W_QS  # (E, DK)
        KS = FC @ self.W_KS  # (N, DK)

        logits = (QS @ KS.transpose(0, 1)) / math.sqrt(self.dk)  # (E, N)
        ones_col = torch.ones((self.popsize, 1), device=DEVICE, dtype=DTYPE)  # (E, 1)
        logits_ext = torch.cat([logits, ones_col], dim=1)  # (E, N+1)
        MS = torch.softmax(logits_ext, dim=1)
        return MS

    def _apply_selection(
        self,
        xp: torch.Tensor,
        children: torch.Tensor,
        sigma_children: torch.Tensor,
        MS: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample S ~ Categorical(M_S) row-wise and update parent archive (paper lines 497-506).

        xp: (1, N, 1+D) parents
        children: (1, N, 1+D) children evaluated
        sigma_children: (N,)
        MS: (E=N, N+1)
        """
        # Row-wise categorical sampling
        E = self.popsize
        choices = torch.multinomial(MS, num_samples=1, replacement=True, generator=self._rng).squeeze(-1)  # (E,)

        next_pop = xp.clone()
        sigma_next = self._sigma_parents.clone()

        # Replace if choose child index < N, otherwise keep
        for i in range(E):
            j = int(choices[i].item())
            if j < self.popsize:
                next_pop[0, i, :] = children[0, j, :]
                sigma_next[i] = sigma_children[j]

        self._sigma_parents = sigma_next
        return next_pop

    # -------------------------------------------------------------------------
    # Main step
    # -------------------------------------------------------------------------

    def step(self, xp: torch.Tensor, problem):
        if xp.ndim != 3 or xp.shape[0] != 1 or xp.shape[1] != self.popsize or xp.shape[2] != self.problemdim + 1:
            raise ValueError(
                f"Invalid population shape. Expected (1, {self.popsize}, {self.problemdim + 1}), got {tuple(xp.shape)}"
            )

        xp = xp.to(DEVICE).to(DTYPE)
        device, dtype = xp.device, xp.dtype

        # Bounds (used for repair/clamp)
        lb, ub = _get_bounds_or_fail(problem, device, dtype, self.problemdim)

        # Initialize best fitness tracking
        fitP = xp[0, :, 0].view(-1)
        if self._best_fitness is None:
            self._best_fitness = (fitP.min() if self.minimize else fitP.max()).detach().clone()

        # 1) Sample parents (uniform)
        xP_tilde, fP_tilde, sigmaP_tilde = self._sample_parents_uniform(xp)

        # 2) MRA -> sigmaC
        sigmaC = self._mra_sigma(fP_tilde, sigmaP_tilde)  # (N,)

        # 3) Mutation: X_C = X~_P + sigma_C * epsilon
        eps = torch.randn((self.popsize, self.problemdim), generator=self._rng, device=DEVICE, dtype=DTYPE)
        xC = xP_tilde + sigmaC.view(-1, 1) * eps  # (N, D)

        # Repair/clamp
        if hasattr(problem, "useRepaire") and bool(getattr(problem, "useRepaire")) and hasattr(problem, "repaire"):
            xC = problem.repaire(xC.view(1, self.popsize, self.problemdim))[0]
        else:
            xC = xC.view(1, self.popsize, self.problemdim).clamp(lb, ub)[0]

        # 4) Evaluate children
        child_pop, _ = problem.calfitness(xC.view(1, self.popsize, self.problemdim))
        fC = child_pop[0, :, 0].view(-1)

        # Update best fitness with children observations
        if self.minimize:
            self._best_fitness = torch.minimum(self._best_fitness, fC.min())
        else:
            self._best_fitness = torch.maximum(self._best_fitness, fC.max())

        # 5) Selection via cross-attention selection matrix M_S and categorical sampling
        MS = self._selection_matrix(fC=fC, fP=fitP)
        next_pop = self._apply_selection(xp=xp, children=child_pop, sigma_children=sigmaC, MS=MS)

        info = {
            "best_fitness": self._best_fitness.detach().clone(),
            "sigma_parents_mean": self._sigma_parents.mean().detach().clone(),
            "sigma_parents_std": self._sigma_parents.std().detach().clone(),
        }

        return next_pop, info


class GradBasedLGA(GradTrainedBBO):
    """
    Gradient-based (autodiff) version of LGA, intended for gradient training frameworks.

    Key difference vs. GradFree LGA:
    - Selection uses straight-through Gumbel-Softmax instead of torch.multinomial, enabling gradients to flow
      into the selection operator parameters (cross-attention weights).
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.name = "GradBasedLGA"
        self.needs_backward = True

        self.popsize = int(config.get("popsize", 16))
        self.problemdim = int(config["problemdim"])
        self.minimize = bool(config.get("minimize", True))

        self.dk = int(config.get("dk", 16))
        self.df = 2  # z-score + centered rank
        self.dsigma = 2  # z-score + [-1,1] normalization
        self.fm_dim = int(self.df + 1 + self.dsigma)  # + best-so-far boolean

        self.initial_sigma = float(config.get("sigma_init", 0.1))
        self.sigma_min = float(config.get("sigma_min", 1e-6))
        self.sigma_max = float(config.get("sigma_max", 10.0))

        self.sel_tau = float(config.get("sel_tau", 1.0))
        self.sel_hard = bool(config.get("sel_hard", True))

        # Parent sampling (for mutation):
        # - "uniform": original behavior (torch.randint)
        # - "soft": differentiable approximation via Gumbel-Softmax weights
        self.parent_sampling = str(config.get("parent_sampling", "uniform")).lower()
        self.parent_tau = float(config.get("parent_tau", 1.0))
        self.parent_hard = bool(config.get("parent_hard", False))
        if self.parent_sampling not in {"uniform", "soft"}:
            raise ValueError(f"Invalid parent_sampling: {self.parent_sampling}")

        self.seed = config.get("seed", None)
        self._rng = torch.Generator(device=DEVICE)
        if self.seed is not None:
            self._rng.manual_seed(int(self.seed))

        scale = float(config.get("param_init_scale", 0.02))

        # ===== Learnable parameters theta (same shapes as GradFree LGA) =====
        # Selection (cross-attention):
        self.W_QP = nn.Parameter(torch.randn((self.df, self.dk), generator=self._rng, device=DEVICE, dtype=DTYPE) * scale)
        self.W_KC = nn.Parameter(torch.randn((self.df, self.dk), generator=self._rng, device=DEVICE, dtype=DTYPE) * scale)
        self.W_VC = nn.Parameter(torch.randn((self.df, self.dk), generator=self._rng, device=DEVICE, dtype=DTYPE) * scale)
        self.W_QS = nn.Parameter(torch.randn((self.dk, self.dk), generator=self._rng, device=DEVICE, dtype=DTYPE) * scale)
        self.W_KS = nn.Parameter(torch.randn((self.df, self.dk), generator=self._rng, device=DEVICE, dtype=DTYPE) * scale)

        # MRA (self-attention):
        self.W_QM = nn.Parameter(torch.randn((self.fm_dim, self.dk), generator=self._rng, device=DEVICE, dtype=DTYPE) * scale)
        self.W_KM = nn.Parameter(torch.randn((self.fm_dim, self.dk), generator=self._rng, device=DEVICE, dtype=DTYPE) * scale)
        self.W_VM = nn.Parameter(torch.randn((self.fm_dim, self.dk), generator=self._rng, device=DEVICE, dtype=DTYPE) * scale)
        self.W_sigma = nn.Parameter(torch.randn((self.dk, 1), generator=self._rng, device=DEVICE, dtype=DTYPE) * scale)

        # Episode state (reset each run)
        self._sigma_parents: Optional[torch.Tensor] = None
        self._best_fitness: Optional[torch.Tensor] = None

        self.to(device=DEVICE, dtype=DTYPE)

    def reset(self, bounds=None):
        self._sigma_parents = torch.full((self.popsize,), self.initial_sigma, device=DEVICE, dtype=DTYPE)
        self._best_fitness = None
        if self.seed is not None:
            self._rng.manual_seed(int(self.seed))

    def export_state(self) -> Dict[str, Any]:
        """
        Export *episode state only* (no learnable parameters).

        This is used by meta-trainers that interleave multiple tasks/functions (fids) and thus need
        per-fid optimizer state isolation (sigma/best/rng) while sharing the same learnable weights.
        """
        sigma = None if self._sigma_parents is None else self._sigma_parents.detach().clone()
        best = None if self._best_fitness is None else self._best_fitness.detach().clone()
        return {
            "sigma_parents": sigma,
            "best_fitness": best,
            "rng_state": self._rng.get_state(),
        }

    def import_state(self, state: Optional[Dict[str, Any]]) -> None:
        """
        Import previously exported episode state (see export_state()).
        """
        if state is None:
            self.reset()
            return

        sigma = state.get("sigma_parents", None)
        if sigma is None:
            self._sigma_parents = None
        else:
            sigma = sigma.detach().clone().to(device=DEVICE, dtype=DTYPE).view(-1)
            if sigma.numel() != self.popsize:
                raise ValueError(f"Invalid sigma_parents size: got {sigma.numel()}, expected {self.popsize}")
            self._sigma_parents = sigma

        best = state.get("best_fitness", None)
        self._best_fitness = None if best is None else best.detach().clone().to(device=DEVICE, dtype=DTYPE)

        rng_state = state.get("rng_state", None)
        if rng_state is not None:
            self._rng.set_state(rng_state)

    def _sample_parents_uniform(self, xp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xP = xp[0, :, 1:]  # (N, D)
        fP = xp[0, :, 0].view(-1)  # (N,)
        idx = torch.randint(0, self.popsize, (self.popsize,), generator=self._rng, device=DEVICE)
        assert self._sigma_parents is not None
        sigmaP = self._sigma_parents.to(device=DEVICE, dtype=DTYPE)
        return xP[idx], fP[idx], sigmaP[idx]

    def _sample_parents_soft(self, xp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Differentiable parent "sampling" via soft attention.

        This approximates uniform-with-replacement sampling by drawing a (soft) one-hot weight vector
        per child using Gumbel-Softmax on zero logits, then taking a weighted sum over parents.
        """
        xP = xp[0, :, 1:]  # (N, D)
        fP = xp[0, :, 0].view(-1)  # (N,)
        assert self._sigma_parents is not None
        sigmaP = self._sigma_parents.to(device=DEVICE, dtype=DTYPE)  # (N,)

        logits = torch.zeros((self.popsize, self.popsize), device=DEVICE, dtype=DTYPE)
        weights = _gumbel_softmax_st(
            logits=logits,
            tau=self.parent_tau,
            hard=self.parent_hard,
            rng=self._rng,
            dim=1,
        )  # (N, N)

        xP_tilde = weights @ xP  # (N, D)
        fP_tilde = (weights @ fP.view(-1, 1)).squeeze(-1)  # (N,)
        sigmaP_tilde = (weights @ sigmaP.view(-1, 1)).squeeze(-1)  # (N,)
        return xP_tilde, fP_tilde, sigmaP_tilde

    def _mra_sigma(self, fP_tilde: torch.Tensor, sigmaP_tilde: torch.Tensor) -> torch.Tensor:
        # Robustness: keep features finite to avoid NaNs propagating into attention.
        sigmaP_tilde = torch.nan_to_num(
            sigmaP_tilde,
            nan=float(self.initial_sigma),
            posinf=float(self.sigma_max),
            neginf=float(self.sigma_min),
        ).clamp(self.sigma_min, self.sigma_max)
        fP_tilde = torch.nan_to_num(fP_tilde, nan=0.0, posinf=1e20, neginf=-1e20).clamp(min=-1e20, max=1e20)

        f_z = _z_score(fP_tilde).clamp(-5.0, 5.0)
        f_r = _centered_ranks(fP_tilde, minimize=self.minimize)

        if self._best_fitness is None:
            best = fP_tilde.min() if self.minimize else fP_tilde.max()
            self._best_fitness = best.detach().clone()

        if self.minimize:
            f_best_flag = (fP_tilde < self._best_fitness).to(DTYPE)
        else:
            f_best_flag = (fP_tilde > self._best_fitness).to(DTYPE)

        s_z = _z_score(sigmaP_tilde).clamp(-5.0, 5.0)
        s_min = sigmaP_tilde.min()
        s_max = sigmaP_tilde.max()
        s_norm = (2.0 * (sigmaP_tilde - s_min) / (s_max - s_min + 1e-8) - 1.0).clamp(-1.0, 1.0)

        FM = torch.stack([f_z, f_r, f_best_flag, s_z, s_norm], dim=-1)  # (N, 5)

        Q = FM @ self.W_QM  # (N, DK)
        K = FM @ self.W_KM  # (N, DK)
        V = FM @ self.W_VM  # (N, DK)

        attn = (Q @ K.transpose(0, 1)) / math.sqrt(self.dk)  # (N, N)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=1e4, neginf=-1e4).clamp(min=-1e4, max=1e4)
        A = torch.softmax(attn, dim=-1) @ V  # (N, DK)
        delta_in = 0.5 * (A @ self.W_sigma).squeeze(-1)
        delta_in = torch.nan_to_num(delta_in, nan=0.0, posinf=10.0, neginf=-10.0).clamp(min=-10.0, max=10.0)
        delta_sigma = torch.exp(delta_in)  # (N,)
        return (delta_sigma * sigmaP_tilde).clamp(self.sigma_min, self.sigma_max)

    def _selection_logits(self, fC: torch.Tensor, fP: torch.Tensor) -> torch.Tensor:
        fC = torch.nan_to_num(fC, nan=0.0, posinf=1e20, neginf=-1e20).clamp(min=-1e20, max=1e20)
        fP = torch.nan_to_num(fP, nan=0.0, posinf=1e20, neginf=-1e20).clamp(min=-1e20, max=1e20)
        f_all = torch.cat([fC, fP], dim=0)  # (2N,)
        z_all = _z_score(f_all).clamp(-5.0, 5.0)
        r_all = _centered_ranks(f_all, minimize=self.minimize)
        F_all = torch.stack([z_all, r_all], dim=-1)  # (2N, 2)

        FC = F_all[: self.popsize, :]  # (N, DF)
        FP = F_all[self.popsize :, :]  # (E, DF)

        QP = FP @ self.W_QP  # (E, DK)
        KC = FC @ self.W_KC  # (N, DK)
        VC = FC @ self.W_VC  # (N, DK)
        scores = (QP @ KC.transpose(0, 1)) / math.sqrt(self.dk)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4).clamp(min=-1e4, max=1e4)
        AS = torch.softmax(scores, dim=-1) @ VC  # (E, DK)

        QS = AS @ self.W_QS  # (E, DK)
        KS = FC @ self.W_KS  # (N, DK)

        logits = (QS @ KS.transpose(0, 1)) / math.sqrt(self.dk)  # (E, N)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4).clamp(min=-1e4, max=1e4)
        ones_col = torch.ones((self.popsize, 1), device=DEVICE, dtype=DTYPE)  # (E, 1)
        return torch.cat([logits, ones_col], dim=1)  # (E, N+1)

    def _apply_selection_st(
        self,
        xp: torch.Tensor,
        children: torch.Tensor,
        sigma_children: torch.Tensor,
        logits_ext: torch.Tensor,
    ) -> torch.Tensor:
        E = self.popsize
        weights = _gumbel_softmax_st(
            logits=logits_ext,
            tau=self.sel_tau,
            hard=self.sel_hard,
            rng=self._rng,
            dim=1,
        )  # (E, N+1)

        w_children = weights[:, :E]  # (E, N)
        w_keep = weights[:, E:].view(E, 1)  # (E, 1)

        xp_flat = xp[0, :, :]  # (E, 1+D)
        child_flat = children[0, :, :]  # (N, 1+D)

        next_from_children = w_children @ child_flat  # (E, 1+D)
        next_pop = next_from_children + w_keep * xp_flat  # (E, 1+D)

        assert self._sigma_parents is not None
        sigma_parents = self._sigma_parents.to(device=DEVICE, dtype=DTYPE)  # (E,)
        sigma_next = (w_children @ sigma_children.view(E, 1)).squeeze(-1) + w_keep.squeeze(-1) * sigma_parents

        self._sigma_parents = sigma_next.detach()
        return next_pop.view(1, E, -1)

    def forward(self, xp: torch.Tensor, problem):
        if xp.ndim != 3 or xp.shape[0] != 1 or xp.shape[1] != self.popsize or xp.shape[2] != self.problemdim + 1:
            raise ValueError(
                f"Invalid population shape. Expected (1, {self.popsize}, {self.problemdim + 1}), got {tuple(xp.shape)}"
            )

        xp = xp.to(DEVICE).to(DTYPE)
        device, dtype = xp.device, xp.dtype

        lb, ub = _get_bounds_or_fail(problem, device, dtype, self.problemdim)

        if self._sigma_parents is None:
            self._sigma_parents = torch.full((self.popsize,), self.initial_sigma, device=DEVICE, dtype=DTYPE)

        fitP = xp[0, :, 0].view(-1)
        # Safety: sanitize non-finite fitness to avoid NaNs propagating through attention/softmax.
        bad = 1e20 if self.minimize else -1e20
        fitP_clean = torch.nan_to_num(fitP, nan=bad, posinf=bad, neginf=bad).clamp(min=-1e20, max=1e20)
        if not torch.equal(fitP_clean, fitP):
            xp = xp.clone()
            xp[0, :, 0] = fitP_clean
        fitP = fitP_clean
        if self._best_fitness is None:
            self._best_fitness = (fitP.min() if self.minimize else fitP.max()).detach().clone()

        if self.parent_sampling == "soft":
            xP_tilde, fP_tilde, sigmaP_tilde = self._sample_parents_soft(xp)
        else:
            xP_tilde, fP_tilde, sigmaP_tilde = self._sample_parents_uniform(xp)

        sigmaC = self._mra_sigma(fP_tilde, sigmaP_tilde)  # (N,)

        eps = torch.randn((self.popsize, self.problemdim), generator=self._rng, device=DEVICE, dtype=DTYPE)
        xC = xP_tilde + sigmaC.view(-1, 1) * eps  # (N, D)

        if hasattr(problem, "useRepaire") and bool(getattr(problem, "useRepaire")) and hasattr(problem, "repaire"):
            xC = problem.repaire(xC.view(1, self.popsize, self.problemdim))[0]
        else:
            xC = xC.view(1, self.popsize, self.problemdim).clamp(lb, ub)[0]
        xC = torch.nan_to_num(xC, nan=0.0, posinf=0.0, neginf=0.0).clamp(lb[0, 0], ub[0, 0])

        child_pop, _ = problem.calfitness(xC.view(1, self.popsize, self.problemdim))
        fC = child_pop[0, :, 0].view(-1)
        fC_clean = torch.nan_to_num(fC, nan=bad, posinf=bad, neginf=bad).clamp(min=-1e20, max=1e20)
        if not torch.equal(fC_clean, fC):
            child_pop = child_pop.clone()
            child_pop[0, :, 0] = fC_clean
        fC = fC_clean

        if self.minimize:
            self._best_fitness = torch.minimum(self._best_fitness, fC.min()).detach()
        else:
            self._best_fitness = torch.maximum(self._best_fitness, fC.max()).detach()

        logits_ext = self._selection_logits(fC=fC, fP=fitP)
        next_pop = self._apply_selection_st(xp=xp, children=child_pop, sigma_children=sigmaC, logits_ext=logits_ext)

        info = {
            "best_fitness": self._best_fitness.detach().clone(),
            "sigma_parents_mean": self._sigma_parents.mean().detach().clone() if self._sigma_parents is not None else None,
            "sigma_parents_std": self._sigma_parents.std().detach().clone() if self._sigma_parents is not None else None,
        }

        return next_pop, info
