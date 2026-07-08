import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from optimizers.base_model import GradFreeBBO, GradTrainedBBO
from torch_basic_settings import DEVICE, DTYPE


torch.set_default_dtype(DTYPE)
torch.set_default_device(DEVICE)


def _tanh_timestamp(gen_counter: int, gamma: torch.Tensor) -> torch.Tensor:
    t = torch.as_tensor(gen_counter, device=gamma.device, dtype=gamma.dtype)
    return torch.tanh(t / gamma - 1.0)


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
        return torch.zeros_like(rank_idx)  # degenerate
    return rank_idx / (fitness.numel() - 1) - 0.5


def _z_score(x: torch.Tensor) -> torch.Tensor:
    mu = x.mean()
    sigma = x.std().clamp_min(1e-8)
    return (x - mu) / sigma


class _LESState:
    def __init__(self, dim: int, timescales: torch.Tensor):
        self.dim = int(dim)
        self.timescales = timescales  # (3,)

        self.mean = torch.zeros((self.dim,), device=DEVICE, dtype=DTYPE)
        self.sigma = torch.ones((self.dim,), device=DEVICE, dtype=DTYPE)

        # Evolution paths (D, 3) per ICLR 2023 appendix pseudocode.
        self.path_c = torch.zeros((self.dim, self.timescales.numel()), device=DEVICE, dtype=DTYPE)
        self.path_sigma = torch.zeros((self.dim, self.timescales.numel()), device=DEVICE, dtype=DTYPE)

        self.best_fitness: Optional[torch.Tensor] = None
        self.gen_counter: int = 0

    def reset(
        self,
        init_min: float,
        init_max: float,
        sigma_init: float,
        t0_min: int,
        t0_max: int,
        minimize: bool,
        rng: torch.Generator,
    ) -> None:
        init_min = float(init_min)
        init_max = float(init_max)
        if init_max < init_min:
            init_min, init_max = init_max, init_min

        self.mean = torch.rand((self.dim,), generator=rng, device=DEVICE, dtype=DTYPE) * (init_max - init_min) + init_min
        self.sigma = torch.full((self.dim,), float(sigma_init), device=DEVICE, dtype=DTYPE)

        self.path_c.zero_()
        self.path_sigma.zero_()

        self.gen_counter = int(torch.randint(int(t0_min), int(t0_max) + 1, (1,), generator=rng, device=DEVICE).item())
        if minimize:
            self.best_fitness = torch.tensor(float("inf"), device=DEVICE, dtype=DTYPE)
        else:
            self.best_fitness = torch.tensor(float("-inf"), device=DEVICE, dtype=DTYPE)


class _LESNetFree:
    """
    Gradient-free (parameter-vector) implementation of LES networks.

    Paper: ICLR 2023 "Discovering Evolution Strategies via Meta-Black-Box Optimization"
    - Recombination weights: self-attention over fitness features (Listing 2, AttentionWeights)
    - Learning-rate modulation: per-dimension MLP over evolution paths + tanh timestamp (Listing 2, EvoPathMLP)
    """

    def __init__(
        self,
        attn_hidden_dims: int = 8,
        mlp_hidden_dims: int = 8,
        gamma: Optional[torch.Tensor] = None,
    ):
        self.attn_hidden_dims = int(attn_hidden_dims)
        self.mlp_hidden_dims = int(mlp_hidden_dims)
        self.gamma = gamma if gamma is not None else torch.tensor(
            [1, 3, 10, 30, 50, 100, 250, 500, 750, 1000, 1250, 1500, 2000],
            device=DEVICE,
            dtype=DTYPE,
        )

        # Recombination weights: Dense(X)->keys/queries/values (with bias)
        self.wq = torch.zeros((3, self.attn_hidden_dims), device=DEVICE, dtype=DTYPE)
        self.bq = torch.zeros((self.attn_hidden_dims,), device=DEVICE, dtype=DTYPE)
        self.wk = torch.zeros((3, self.attn_hidden_dims), device=DEVICE, dtype=DTYPE)
        self.bk = torch.zeros((self.attn_hidden_dims,), device=DEVICE, dtype=DTYPE)
        self.wv = torch.zeros((3, 1), device=DEVICE, dtype=DTYPE)
        self.bv = torch.zeros((1,), device=DEVICE, dtype=DTYPE)

        # Learning-rate modulation MLP (shared across dimensions)
        self.mlp_in_dim = int(3 + 3 + self.gamma.numel())
        self.w1 = torch.zeros((self.mlp_in_dim, self.mlp_hidden_dims), device=DEVICE, dtype=DTYPE)
        self.b1 = torch.zeros((self.mlp_hidden_dims,), device=DEVICE, dtype=DTYPE)
        self.w_mu = torch.zeros((self.mlp_hidden_dims, 1), device=DEVICE, dtype=DTYPE)
        self.b_mu = torch.zeros((1,), device=DEVICE, dtype=DTYPE)
        self.w_sigma = torch.zeros((self.mlp_hidden_dims, 1), device=DEVICE, dtype=DTYPE)
        self.b_sigma = torch.zeros((1,), device=DEVICE, dtype=DTYPE)

        self.n_params = (
            self.wq.numel() + self.bq.numel()
            + self.wk.numel() + self.bk.numel()
            + self.wv.numel() + self.bv.numel()
            + self.w1.numel() + self.b1.numel()
            + self.w_mu.numel() + self.b_mu.numel()
            + self.w_sigma.numel() + self.b_sigma.numel()
        )

    def init_params(self, rng: torch.Generator, scale: float = 0.02) -> None:
        """
        Initialize weights similar to small-random (paper uses learned params via MetaBBO).
        """
        scale = float(scale)
        self.wq = torch.randn(self.wq.shape, generator=rng, device=self.wq.device, dtype=self.wq.dtype) * scale
        self.bq = torch.zeros_like(self.bq)
        self.wk = torch.randn(self.wk.shape, generator=rng, device=self.wk.device, dtype=self.wk.dtype) * scale
        self.bk = torch.zeros_like(self.bk)
        self.wv = torch.randn(self.wv.shape, generator=rng, device=self.wv.device, dtype=self.wv.dtype) * scale
        self.bv = torch.zeros_like(self.bv)

        self.w1 = torch.randn(self.w1.shape, generator=rng, device=self.w1.device, dtype=self.w1.dtype) * scale
        self.b1 = torch.zeros_like(self.b1)
        self.w_mu = torch.randn(self.w_mu.shape, generator=rng, device=self.w_mu.device, dtype=self.w_mu.dtype) * scale
        self.b_mu = torch.zeros_like(self.b_mu)
        self.w_sigma = torch.randn(self.w_sigma.shape, generator=rng, device=self.w_sigma.device, dtype=self.w_sigma.dtype) * scale
        self.b_sigma = torch.zeros_like(self.b_sigma)

    def update_params(self, flat: torch.Tensor) -> None:
        flat = flat.to(device=DEVICE, dtype=DTYPE).view(-1)
        if flat.numel() != self.n_params:
            raise ValueError(f"LES params size mismatch: got {flat.numel()}, expected {self.n_params}")

        idx = 0

        def take(num: int) -> torch.Tensor:
            nonlocal idx
            out = flat[idx:idx + num]
            idx += num
            return out

        self.wq = take(self.wq.numel()).view_as(self.wq).clone()
        self.bq = take(self.bq.numel()).view_as(self.bq).clone()
        self.wk = take(self.wk.numel()).view_as(self.wk).clone()
        self.bk = take(self.bk.numel()).view_as(self.bk).clone()
        self.wv = take(self.wv.numel()).view_as(self.wv).clone()
        self.bv = take(self.bv.numel()).view_as(self.bv).clone()

        self.w1 = take(self.w1.numel()).view_as(self.w1).clone()
        self.b1 = take(self.b1.numel()).view_as(self.b1).clone()
        self.w_mu = take(self.w_mu.numel()).view_as(self.w_mu).clone()
        self.b_mu = take(self.b_mu.numel()).view_as(self.b_mu).clone()
        self.w_sigma = take(self.w_sigma.numel()).view_as(self.w_sigma).clone()
        self.b_sigma = take(self.b_sigma.numel()).view_as(self.b_sigma).clone()

        if idx != flat.numel():
            raise RuntimeError("LES param slicing error")

    def recombination_weights(self, fit_features: torch.Tensor) -> torch.Tensor:
        """
        fit_features: (N, 3) -> weights: (N, 1)

        Matches Listing 2 (AttentionWeights) and Sec.4.1:
          A = softmax(QK^T / sqrt(DK))
          weights = softmax(A V)
        """
        if fit_features.ndim != 2 or fit_features.shape[1] != 3:
            raise ValueError(f"fit_features must be (N, 3), got {tuple(fit_features.shape)}")

        Q = fit_features @ self.wq + self.bq  # (N, DK)
        K = fit_features @ self.wk + self.bk  # (N, DK)
        V = fit_features @ self.wv + self.bv  # (N, 1)

        attn = (Q @ K.transpose(0, 1)) / math.sqrt(self.attn_hidden_dims)  # (N, N)
        A = torch.softmax(attn, dim=-1)
        scores = (A @ V).squeeze(-1)  # (N,)
        w = torch.softmax(scores, dim=0).unsqueeze(-1)  # (N, 1)
        return w

    def lrates(
        self,
        path_c: torch.Tensor,
        path_sigma: torch.Tensor,
        time_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        path_c/path_sigma: (D, 3)
        time_embed: (|gamma|,)
        returns lr_mean/lr_sigma: (D,)
        """
        if path_c.ndim != 2 or path_sigma.ndim != 2:
            raise ValueError("path_c/path_sigma must be 2D tensors")
        if path_c.shape != path_sigma.shape:
            raise ValueError(f"path shapes mismatch: {tuple(path_c.shape)} vs {tuple(path_sigma.shape)}")
        if time_embed.ndim != 1 or time_embed.numel() != self.gamma.numel():
            raise ValueError(f"time_embed must be ({self.gamma.numel()},), got {tuple(time_embed.shape)}")

        D = path_c.shape[0]
        timestamps = time_embed.view(1, -1).expand(D, -1)  # (D, |gamma|)
        X = torch.cat([path_c, path_sigma, timestamps], dim=1)  # (D, 3+3+|gamma|)

        hidden = torch.relu(X @ self.w1 + self.b1)  # (D, H)
        lr_mean = torch.sigmoid(hidden @ self.w_mu + self.b_mu).squeeze(-1)  # (D,)
        lr_sigma = torch.sigmoid(hidden @ self.w_sigma + self.b_sigma).squeeze(-1)  # (D,)
        return lr_mean, lr_sigma


class _LESNetGrad(nn.Module):
    """
    Gradient-based implementation of the same LES architecture (for self-supervised/distill training).
    """

    def __init__(self, attn_hidden_dims: int = 8, mlp_hidden_dims: int = 8, gamma: Optional[torch.Tensor] = None):
        super().__init__()
        self.attn_hidden_dims = int(attn_hidden_dims)
        self.mlp_hidden_dims = int(mlp_hidden_dims)
        self.gamma = gamma if gamma is not None else torch.tensor(
            [1, 3, 10, 30, 50, 100, 250, 500, 750, 1000, 1250, 1500, 2000],
            device=DEVICE,
            dtype=DTYPE,
        )

        self.wq = nn.Linear(3, self.attn_hidden_dims, bias=True)
        self.wk = nn.Linear(3, self.attn_hidden_dims, bias=True)
        self.wv = nn.Linear(3, 1, bias=True)

        self.mlp_in_dim = int(3 + 3 + self.gamma.numel())
        self.fc1 = nn.Linear(self.mlp_in_dim, self.mlp_hidden_dims, bias=True)
        self.mu_head = nn.Linear(self.mlp_hidden_dims, 1, bias=True)
        self.sigma_head = nn.Linear(self.mlp_hidden_dims, 1, bias=True)

    def recombination_weights(self, fit_features: torch.Tensor) -> torch.Tensor:
        Q = self.wq(fit_features)  # (N, DK)
        K = self.wk(fit_features)  # (N, DK)
        V = self.wv(fit_features)  # (N, 1)

        attn = (Q @ K.transpose(0, 1)) / math.sqrt(self.attn_hidden_dims)
        A = torch.softmax(attn, dim=-1)
        scores = (A @ V).squeeze(-1)
        w = torch.softmax(scores, dim=0).unsqueeze(-1)
        return w

    def lrates(self, path_c: torch.Tensor, path_sigma: torch.Tensor, time_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        D = path_c.shape[0]
        timestamps = time_embed.view(1, -1).expand(D, -1)
        X = torch.cat([path_c, path_sigma, timestamps], dim=1)
        hidden = torch.relu(self.fc1(X))
        lr_mean = torch.sigmoid(self.mu_head(hidden)).squeeze(-1)
        lr_sigma = torch.sigmoid(self.sigma_head(hidden)).squeeze(-1)
        return lr_mean, lr_sigma


class _LESCore:
    """
    Shared LES state machine implementing Listing 2 (ask/tell style) for a single population (batch=1).
    """

    def __init__(self, config: Dict):
        self.popsize = int(config.get("popsize", 100))
        self.problemdim = int(config["problemdim"])
        self.minimize = bool(config.get("minimize", True))

        # Paper defaults (Appendix D.1 / Listing 2)
        self.gamma = torch.tensor(
            [1, 3, 10, 30, 50, 100, 250, 500, 750, 1000, 1250, 1500, 2000],
            device=DEVICE,
            dtype=DTYPE,
        )
        self.timescales = torch.tensor([0.1, 0.5, 0.9], device=DEVICE, dtype=DTYPE)

        self.sigma_init = float(config.get("sigma_init", 1.0))
        self.init_min = float(config.get("init_min", -5.0))
        self.init_max = float(config.get("init_max", 5.0))
        self.t0_min = int(config.get("t0_min", 0))
        self.t0_max = int(config.get("t0_max", 2000))

        # For clipping candidates (Listing 2 uses params.clip_min/clip_max)
        self.clip_min = config.get("clip_min", None)
        self.clip_max = config.get("clip_max", None)

        self.seed = config.get("seed", None)
        self._rng = torch.Generator(device=DEVICE)
        if self.seed is not None:
            self._rng.manual_seed(int(self.seed))

        self.state = _LESState(dim=self.problemdim, timescales=self.timescales)

    def reset(self):
        self.state.reset(
            init_min=self.init_min,
            init_max=self.init_max,
            sigma_init=self.sigma_init,
            t0_min=self.t0_min,
            t0_max=self.t0_max,
            minimize=self.minimize,
            rng=self._rng,
        )

    def _get_clip_bounds(self, problem) -> Tuple[Optional[float], Optional[float]]:
        if self.clip_min is not None and self.clip_max is not None:
            return float(self.clip_min), float(self.clip_max)

        fun = getattr(problem, "fun", None)
        if isinstance(fun, dict) and "xlb" in fun and "xub" in fun:
            return float(fun["xlb"]), float(fun["xub"])
        return None, None

    def _fitness_features(self, fitness: torch.Tensor) -> torch.Tensor:
        """
        fitness: (N,)
        returns (N, 3): [zscore(f), centered_rank, better_than_best_so_far]
        """
        z = _z_score(fitness).clamp(-5.0, 5.0)
        r = _centered_ranks(fitness, minimize=self.minimize)

        if self.state.best_fitness is None:
            best = torch.tensor(float("inf") if self.minimize else float("-inf"), device=DEVICE, dtype=DTYPE)
        else:
            best = self.state.best_fitness

        if self.minimize:
            is_best = (fitness < best).to(DTYPE)
        else:
            is_best = (fitness > best).to(DTYPE)

        return torch.stack([z, r, is_best], dim=-1)  # (N, 3)

    def tell_and_ask(self, xp: torch.Tensor, problem, net) -> Tuple[torch.Tensor, Dict]:
        """
        xp: evaluated population (1, N, 1+D)
        net: object providing recombination_weights() and lrates()
        """
        if xp.ndim != 3 or xp.shape[0] != 1 or xp.shape[1] != self.popsize or xp.shape[2] != self.problemdim + 1:
            raise ValueError(
                f"Invalid population shape. Expected (1, {self.popsize}, {self.problemdim + 1}), got {tuple(xp.shape)}"
            )

        xp = xp.to(DEVICE).to(DTYPE)
        x = xp[0, :, 1:]  # (N, D)
        fit = xp[0, :, 0].view(-1)  # (N,)

        # Build fitness tokens Ft (Sec.4.1 / Listing 2)
        Ft = self._fitness_features(fit)  # (N, 3)

        # Recombination weights via self-attention (Sec.4.1 / Listing 2)
        weights = net.recombination_weights(Ft)  # (N, 1), sums to 1

        # Update best fitness tracking (used for next generation feature)
        if self.minimize:
            best_now = fit.min()
            best_ever = (
                torch.tensor(float("inf"), device=DEVICE, dtype=DTYPE)
                if self.state.best_fitness is None
                else self.state.best_fitness
            )
            self.state.best_fitness = torch.minimum(best_ever, best_now)
        else:
            best_now = fit.max()
            best_ever = (
                torch.tensor(float("-inf"), device=DEVICE, dtype=DTYPE)
                if self.state.best_fitness is None
                else self.state.best_fitness
            )
            self.state.best_fitness = torch.maximum(best_ever, best_now)

        # Weight-diff / weight-noise (Listing 2)
        mean = self.state.mean
        sigma = self.state.sigma.clamp_min(1e-12)
        weight_diff = (weights * (x - mean.view(1, -1))).sum(dim=0)  # (D,)
        weight_noise = (weights * (x - mean.view(1, -1)) / sigma.view(1, -1)).sum(dim=0)  # (D,)

        # Evolution path updates: EMA at three timescales (Appendix text + Listing 2)
        #
        # NOTE: Avoid in-place slice assignment here. The optimizer state is a *buffer* (not a Parameter),
        # and slice assignment would silently break the autograd graph (the assignment op is not differentiable).
        # Using a functional update keeps gradients flowing to the network through `weight_diff/weight_noise`.
        a = self.timescales.view(1, -1)  # (1, T)
        self.state.path_c = (1.0 - a) * self.state.path_c + a * weight_diff.view(-1, 1)
        self.state.path_sigma = (1.0 - a) * self.state.path_sigma + a * weight_noise.view(-1, 1)

        # Learning-rate modulation (Appendix / Listing 2)
        time_embed = _tanh_timestamp(self.state.gen_counter, self.gamma)  # (13,)
        lr_mean, lr_sigma = net.lrates(self.state.path_c, self.state.path_sigma, time_embed)  # (D,), (D,)

        # Weighted updates of mean/std (Eq.(2)(3) / Listing 2)
        weighted_mean = (weights * x).sum(dim=0)  # (D,)
        weighted_sigma = torch.sqrt((weights * (x - mean.view(1, -1)) ** 2).sum(dim=0) + 1e-10)  # (D,)

        mean = mean + lr_mean * (weighted_mean - mean)
        sigma = sigma + lr_sigma * (weighted_sigma - sigma)

        # Clip mean/sigma (Listing 2)
        clip_min, clip_max = self._get_clip_bounds(problem)
        if clip_min is not None and clip_max is not None:
            mean = mean.clamp(clip_min, clip_max)
            sigma = sigma.clamp(0.0, clip_max)
        else:
            sigma = sigma.clamp_min(0.0)

        self.state.mean = mean
        self.state.sigma = sigma
        self.state.gen_counter += 1

        # Ask: sample new candidates
        noise = torch.randn((1, self.popsize, self.problemdim), generator=self._rng, device=DEVICE, dtype=DTYPE)
        cand = self.state.mean.view(1, 1, -1) + noise * self.state.sigma.view(1, 1, -1)

        if clip_min is not None and clip_max is not None:
            cand = cand.clamp(clip_min, clip_max)

        cand, _ = problem.calfitness(cand)
        info = {
            "mean": self.state.mean.detach().clone(),
            "sigma": self.state.sigma.detach().clone(),
            "best_fitness": self.state.best_fitness.detach().clone(),
        }
        return cand, info


class GradFreeLES(GradFreeBBO):
    """
    Gradient-free LES used for MetaBBO (outer CMA-ES).

    Strictly follows ICLR 2023 Listing 2 / Sec.4.1:
    - Fitness tokens Ft = [zscore, centered_rank, best_so_far_bool]
    - Self-attention over Ft to obtain recombination weights w
    - Evo-path MLP to modulate per-dimension learning rates (mean/std)
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.name = "GradFreeLES"
        self.needs_backward = False

        self.core = _LESCore(config)
        self.net = _LESNetFree(
            attn_hidden_dims=int(config.get("attn_hdim", 8)),
            mlp_hidden_dims=int(config.get("mlp_hdim", 8)),
            gamma=self.core.gamma,
        )
        self.net.init_params(self.core._rng, scale=float(config.get("param_init_scale", 0.02)))

        self.n_params = int(self.net.n_params)

    def reset(self, bounds=None):
        self.core.reset()

    def update_params(self, params: torch.Tensor):
        self.net.update_params(params)

    def step(self, xp: torch.Tensor, problem):
        return self.core.tell_and_ask(xp, problem, self.net)

    def get_distr_param(self):
        mean = self.core.state.mean.view(1, -1)
        cov = torch.diag(self.core.state.sigma ** 2)
        return mean, cov


class GradBasedLES(GradTrainedBBO):
    """
    Gradient-based LES with the same architecture (for self-supervised/distillation training).
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.name = "GradBasedLES"
        self.needs_backward = True

        self.core = _LESCore(config)
        self.net = _LESNetGrad(
            attn_hidden_dims=int(config.get("attn_hdim", 8)),
            mlp_hidden_dims=int(config.get("mlp_hdim", 8)),
            gamma=self.core.gamma,
        )
        self.to(device=DEVICE, dtype=DTYPE)

    def reset(self, bounds=None):
        self.core.reset()

    def export_state(self) -> Dict[str, Any]:
        """
        Export *episode state only* (no learnable parameters).

        Used by meta-trainers that interleave multiple tasks/functions (fids) and thus need per-fid
        optimizer state isolation while sharing the same learnable weights.
        """
        s = self.core.state
        return {
            "mean": s.mean.detach().clone(),
            "sigma": s.sigma.detach().clone(),
            "path_c": s.path_c.detach().clone(),
            "path_sigma": s.path_sigma.detach().clone(),
            "best_fitness": None if s.best_fitness is None else s.best_fitness.detach().clone(),
            "gen_counter": int(s.gen_counter),
            "rng_state": self.core._rng.get_state(),
        }

    def import_state(self, state: Optional[Dict[str, Any]]) -> None:
        """
        Import previously exported episode state (see export_state()).
        """
        if state is None:
            self.reset()
            return

        s = self.core.state
        dim = int(self.core.problemdim)
        n_ts = int(self.core.timescales.numel())

        mean = state.get("mean", None)
        sigma = state.get("sigma", None)
        path_c = state.get("path_c", None)
        path_sigma = state.get("path_sigma", None)
        if mean is None or sigma is None or path_c is None or path_sigma is None:
            raise ValueError("Invalid LES state: missing required tensors (mean/sigma/path_c/path_sigma).")

        mean = mean.detach().clone().to(device=DEVICE, dtype=DTYPE).view(-1)
        sigma = sigma.detach().clone().to(device=DEVICE, dtype=DTYPE).view(-1)
        if mean.numel() != dim or sigma.numel() != dim:
            raise ValueError(f"Invalid LES state dim: mean={mean.numel()}, sigma={sigma.numel()}, expected {dim}")

        path_c = path_c.detach().clone().to(device=DEVICE, dtype=DTYPE).view(dim, n_ts)
        path_sigma = path_sigma.detach().clone().to(device=DEVICE, dtype=DTYPE).view(dim, n_ts)

        s.mean = mean
        s.sigma = sigma
        s.path_c = path_c
        s.path_sigma = path_sigma

        best = state.get("best_fitness", None)
        if best is None:
            s.best_fitness = torch.tensor(float("inf") if self.core.minimize else float("-inf"), device=DEVICE, dtype=DTYPE)
        else:
            s.best_fitness = best.detach().clone().to(device=DEVICE, dtype=DTYPE)

        s.gen_counter = int(state.get("gen_counter", 0))

        rng_state = state.get("rng_state", None)
        if rng_state is not None:
            self.core._rng.set_state(rng_state)

    def forward(self, xp: torch.Tensor, problem):
        return self.core.tell_and_ask(xp, problem, self.net)
