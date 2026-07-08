import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from optimizers.base_model import GradTrainedBBO
from optimizers.utils import one2one_selection
from torch_basic_settings import DEVICE, DTYPE


torch.set_default_dtype(DTYPE)
torch.set_default_device(DEVICE)


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
            "LDE requires explicit bounds. Please provide one of:\n"
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


def _histogram(values_01: torch.Tensor, n_bins: int) -> torch.Tensor:
    """
    Histogram over [0,1] with n_bins bins. Returns normalized frequencies (sum=1).
    values_01: (N,) in [0,1]
    """
    n_bins = int(n_bins)
    if n_bins <= 0:
        raise ValueError(f"n_bins must be > 0, got {n_bins}")
    v = values_01.clamp(0.0, 1.0)
    idx = torch.clamp((v * n_bins).to(torch.int64), 0, n_bins - 1)
    hist = torch.bincount(idx, minlength=n_bins).to(DTYPE)
    return hist / max(1, v.numel())


def _minmax_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize to [0,1] using per-vector min/max (as in Eq.(8) in the paper).
    """
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + 1e-8)


@dataclass
class LDETrainingInfo:
    log_prob: torch.Tensor  # scalar (sum over 2N action dims)
    reward: torch.Tensor  # scalar


class LDEController(nn.Module):
    """
    LSTM parameter controller (paper Sec.III-B/C).

    Input A_t = [U_t, F_t] where:
      - U_t = [h_t, \bar{h_t}] from Eq.(8) (histogram of normalized fitness and its moving average)
      - F_t is the population fitness vector (we feed normalized fitness for scale invariance)

    Output:
      - mean actions for per-individual (F_i, CR_i), i=1..N
      - policy: Normal(mean, sigma_policy^2) (paper Eq.(9))
    """

    def __init__(self, input_dim: int, popsize: int, hidden_dim: int):
        super().__init__()
        self.popsize = int(popsize)
        self.hidden_dim = int(hidden_dim)

        self.lstm = nn.LSTMCell(input_size=int(input_dim), hidden_size=self.hidden_dim)
        self.head_F = nn.Linear(self.hidden_dim, self.popsize)
        self.head_CR = nn.Linear(self.hidden_dim, self.popsize)

    def forward(self, a_t: torch.Tensor, hc: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        a_t: (1, input_dim)
        hc: (h, c) each (1, hidden_dim)
        Returns:
          mu_F: (1, N) in [0,1]
          mu_CR: (1, N) in [0,1]
          (h_next, c_next)
        """
        h, c = self.lstm(a_t, hc)
        mu_F = torch.sigmoid(self.head_F(h))
        mu_CR = torch.sigmoid(self.head_CR(h))
        return mu_F, mu_CR, (h, c)


class LDE(GradTrainedBBO):
    """
    Learned Differential Evolution (LDE) per IEEE TEVC 2021.

    Paper: "Learning Adaptive Differential Evolution Algorithm From Optimization Experiences by Policy Gradient"

    Implements:
    - DE/current-to-pbest/1 mutation (Eq.(1))
    - Binomial crossover (Eq.(2))
    - Parameter controller: LSTM over [U_t, F_t] (Eq.(8), Sec.III-B/C)
    - Policy: Gaussian pi(A_t|S_t)=N(A_t|LSTM(S_t;W), sigma^2) (Eq.(9))
    - Reward: relative improvement of best fitness (Eq.(10); adapted for min/max)
    - Training uses REINFORCE; see meta_trainers/rl_trainer.py
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.name = "LDE"
        self.needs_backward = True

        self.popsize = int(config.get("popsize", 50))
        self.problemdim = int(config["problemdim"])
        self.minimize = bool(config.get("minimize", True))

        # Controller + state features
        self.n_bins = int(config.get("n_bins", 5))  # b in Eq.(8)
        self.hist_window = int(config.get("hist_window", 5))  # g in Eq.(8)

        self.hidden_dim = int(config.get("hidden_dim", 500))  # paper studies 500..3000
        self.policy_sigma = float(config.get("policy_sigma", 0.1))  # sigma in Eq.(9)

        # How to sample actions from the Gaussian policy.
        # - "sample": Normal.sample (non-differentiable, matches PG/REINFORCE)
        # - "rsample": Normal.rsample (reparameterized, enables SSFT/Distill gradients)
        # - "mean": use mean actions deterministically
        self.action_mode = str(config.get("action_mode", "sample")).lower()
        if self.action_mode not in {"sample", "rsample", "mean"}:
            raise ValueError(f"Invalid action_mode: {self.action_mode} (expected \"sample\"|\"rsample\"|\"mean\")")

        # DE parameters
        self.p_best = float(config.get("p_best", 0.1))  # p in current-to-pbest/1

        self.seed = config.get("seed", None)
        self._rng = torch.Generator(device=DEVICE)
        if self.seed is not None:
            self._rng.manual_seed(int(self.seed))

        input_dim = 2 * self.n_bins + self.popsize
        self.controller = LDEController(input_dim=input_dim, popsize=self.popsize, hidden_dim=self.hidden_dim)

        # Episode state (reset each run)
        self._h: Optional[torch.Tensor] = None
        self._c: Optional[torch.Tensor] = None
        self._histories: List[torch.Tensor] = []
        self._best_prev: Optional[torch.Tensor] = None

        self.to(device=DEVICE, dtype=DTYPE)

        if self.popsize < 4:
            raise ValueError(f"LDE requires popsize >= 4, got {self.popsize}")
        if self.n_bins <= 0 or self.hist_window <= 0:
            raise ValueError("n_bins and hist_window must be > 0")

    def reset(self):
        self._h = torch.zeros((1, self.hidden_dim), device=DEVICE, dtype=DTYPE)
        self._c = torch.zeros((1, self.hidden_dim), device=DEVICE, dtype=DTYPE)
        self._histories = []
        self._best_prev = None
        if self.seed is not None:
            self._rng.manual_seed(int(self.seed))

    def export_state(self) -> Dict:
        """Export internal episode state for per-function-id isolation in meta-trainers."""
        return {
            "h": None if self._h is None else self._h.detach().clone(),
            "c": None if self._c is None else self._c.detach().clone(),
            "histories": [h.detach().clone() for h in self._histories],
            "best_prev": None if self._best_prev is None else self._best_prev.detach().clone(),
            "rng_state": self._rng.get_state(),
        }

    def import_state(self, state: Dict) -> None:
        """Restore internal episode state exported by export_state()."""
        self._h = state.get("h", None)
        self._c = state.get("c", None)
        self._histories = list(state.get("histories", []))
        self._best_prev = state.get("best_prev", None)
        rng_state = state.get("rng_state", None)
        if rng_state is not None:
            self._rng.set_state(rng_state)

    # -------------------------------------------------------------------------
    # State feature construction (Eq.(8))
    # -------------------------------------------------------------------------

    def _make_U_t(self, fitness: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        fitness: (N,) current population fitness values.
        Returns:
          h_t: (b,) histogram of normalized fitness (Eq.(8))
          hbar_t: (b,) moving average of past g histograms (Eq.(8))
        """
        f_norm = _minmax_norm(fitness)  # (N,) in [0,1]
        h_t = _histogram(f_norm, self.n_bins)  # (b,)

        if len(self._histories) == 0:
            hbar = torch.zeros_like(h_t)
        else:
            recent = self._histories[-self.hist_window :]
            hbar = torch.stack(recent, dim=0).mean(dim=0)

        return h_t, hbar

    def _make_A_t(self, fitness: torch.Tensor) -> torch.Tensor:
        """
        A_t = [U_t, F_t] (paper Algorithm 1 / Eq.(8) usage).
        We feed normalized fitness vector for scale invariance.
        Returns: (1, 2*b + N)
        """
        h_t, hbar_t = self._make_U_t(fitness)
        f_norm = _minmax_norm(fitness)  # (N,)
        a = torch.cat([h_t, hbar_t, f_norm.to(DTYPE)], dim=0).view(1, -1)
        return a

    # -------------------------------------------------------------------------
    # DE/current-to-pbest/1 (Eq.(1)) + binomial crossover (Eq.(2))
    # -------------------------------------------------------------------------

    def _generate_r1_r2(self, n: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized distinct indices r1, r2 for each i (excluding i and each other).
        Returns: (r1, r2) each (n,)
        """
        scores = torch.rand((n, n), generator=self._rng, device=device, dtype=DTYPE)
        scores = scores.masked_fill(torch.eye(n, device=device, dtype=torch.bool), float("-inf"))
        perm = torch.argsort(scores, dim=1, descending=True)  # (n, n)
        r1 = perm[:, 0]
        r2 = perm[:, 1]
        return r1, r2

    def _select_pbest(self, fitness: torch.Tensor) -> torch.Tensor:
        """
        Select p-best indices per individual (paper Eq.(1)).
        Returns: (N,) indices.
        """
        n = fitness.numel()
        p_count = max(2, int(math.ceil(n * self.p_best)))
        if self.minimize:
            sorted_idx = torch.argsort(fitness, dim=0)  # best first
        else:
            sorted_idx = torch.argsort(fitness, dim=0, descending=True)
        top = sorted_idx[:p_count]
        choice = torch.randint(0, p_count, (n,), generator=self._rng, device=fitness.device)
        return top[choice]

    def _mutation_current_to_pbest(self, x: torch.Tensor, fitness: torch.Tensor, F_vec: torch.Tensor) -> torch.Tensor:
        """
        x: (N, D)
        fitness: (N,)
        F_vec: (N,) scaling factors
        Returns: v (N, D)
        """
        n, d = x.shape
        pbest_idx = self._select_pbest(fitness)  # (N,)
        r1, r2 = self._generate_r1_r2(n, x.device)

        x_pbest = x[pbest_idx, :]
        x_r1 = x[r1, :]
        x_r2 = x[r2, :]

        F_b = F_vec.view(n, 1)
        v = x + F_b * (x_pbest - x) + F_b * (x_r1 - x_r2)
        return v

    # -------------------------------------------------------------------------
    # Policy sampling (Eq.(9)) + reward (Eq.(10))
    # -------------------------------------------------------------------------

    def _sample_actions(
        self, mu_F: torch.Tensor, mu_CR: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        mu_F/mu_CR: (1, N)
        Returns:
          F: (N,) in [0,1]
          CR: (N,) in [0,1]
          log_prob: scalar (sum over 2N dims)
        """
        sigma = torch.full_like(mu_F, self.policy_sigma)
        dist_F = Normal(mu_F, sigma)
        dist_CR = Normal(mu_CR, sigma)

        if self.action_mode == "mean":
            F_raw = mu_F
            CR_raw = mu_CR
        elif self.action_mode == "rsample":
            F_raw = dist_F.rsample()
            CR_raw = dist_CR.rsample()
        else:
            F_raw = dist_F.sample()
            CR_raw = dist_CR.sample()

        # Detach samples to avoid pathwise gradients through actions when log_prob is used (PG/REINFORCE).
        log_prob = dist_F.log_prob(F_raw.detach()).sum() + dist_CR.log_prob(CR_raw.detach()).sum()

        F = F_raw.clamp(0.0, 1.0).view(-1)
        CR = CR_raw.clamp(0.0, 1.0).view(-1)
        return F, CR, log_prob

    def _compute_reward(self, best_prev: torch.Tensor, best_now: torch.Tensor) -> torch.Tensor:
        """
        Paper Eq.(10) uses relative improvement of best fitness.
        Here we adapt to minimization/maximization.
        """
        denom = best_prev.abs().clamp_min(1e-8)
        if self.minimize:
            return ((best_prev - best_now) / denom).clamp(min=0.0)
        return ((best_now - best_prev) / denom).clamp(min=0.0)

    # -------------------------------------------------------------------------
    # One generation step (used for both training and inference)
    # -------------------------------------------------------------------------

    def forward(self, xp: torch.Tensor, problem) -> Tuple[torch.Tensor, Dict]:
        if xp.ndim != 3 or xp.shape[0] != 1 or xp.shape[1] != self.popsize or xp.shape[2] != self.problemdim + 1:
            raise ValueError(
                f"Invalid population shape. Expected (1, {self.popsize}, {self.problemdim + 1}), got {tuple(xp.shape)}"
            )

        xp = xp.to(DEVICE).to(DTYPE)
        device, dtype = xp.device, xp.dtype

        lb, ub = _get_bounds_or_fail(problem, device, dtype, self.problemdim)

        x = xp[0, :, 1:]  # (N, D)
        fitness = xp[0, :, 0].view(-1)  # (N,)

        # Build LSTM input A_t = [U_t, F_t] (Eq.(8) + Algorithm 1)
        a_t = self._make_A_t(fitness)  # (1, 2*b + N)

        # Controller outputs mean actions
        if self._h is None or self._c is None:
            self.reset()
        mu_F, mu_CR, (self._h, self._c) = self.controller(a_t, (self._h, self._c))

        # Sample actions from Gaussian policy (Eq.(9))
        F_vec, CR_vec, log_prob = self._sample_actions(mu_F, mu_CR)

        # Mutation (Eq.(1))
        v = self._mutation_current_to_pbest(x, fitness, F_vec)  # (N, D)

        # Crossover (Eq.(2))
        rand_mask = torch.rand((self.popsize, self.problemdim), generator=self._rng, device=DEVICE, dtype=DTYPE)
        cross_mask = rand_mask < CR_vec.view(-1, 1)
        j_rand = torch.randint(0, self.problemdim, (self.popsize,), generator=self._rng, device=DEVICE)
        cross_mask[torch.arange(self.popsize, device=DEVICE), j_rand] = True

        u = torch.where(cross_mask, v, x)  # (N, D)

        # Repair/clamp
        u = u.view(1, self.popsize, self.problemdim)
        if hasattr(problem, "useRepaire") and bool(getattr(problem, "useRepaire")) and hasattr(problem, "repaire"):
            u = problem.repaire(u)
        else:
            u = u.clamp(lb, ub)

        # Evaluate
        trial_pop, _ = problem.calfitness(u)

        # Selection
        offspring = one2one_selection(xp, trial_pop, minimize=self.minimize)

        # Reward (Eq.(10), adapted)
        best_now = offspring[0, :, 0].min() if self.minimize else offspring[0, :, 0].max()
        if self._best_prev is None:
            self._best_prev = fitness.min() if self.minimize else fitness.max()
        reward = self._compute_reward(self._best_prev.detach(), best_now.detach())
        self._best_prev = best_now.detach().clone()

        # Update histogram history AFTER using h_t for A_t (Eq.(8) definition)
        h_t, _ = self._make_U_t(fitness)
        self._histories.append(h_t.detach())
        if len(self._histories) > self.hist_window:
            self._histories = self._histories[-self.hist_window :]

        info = {
            "training": LDETrainingInfo(log_prob=log_prob, reward=reward),
            "F_mean": mu_F.detach().clone(),
            "CR_mean": mu_CR.detach().clone(),
        }
        return offspring, info
