# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from typing import Optional, Tuple, Union
import math
import tqdm

import torch
from torch import nn, Tensor

from .groups import Group, ConnectedMatrixLieGroup
from .energies import Energy


class GroupSampler(nn.Module):
    def __init__(self, group: Union[Group, ConnectedMatrixLieGroup]):
        super().__init__()
        self.group = group

    def forward(self, x: Tensor, y: Optional[Tensor]) -> Tensor:
        """
        Sample group elements based on the input tensors.
        Return:
            g: [bsize, ...]
        """
        raise NotImplementedError


class GroupEnergySampler(GroupSampler):  # pylint: disable=abstract-method
    def __init__(self, energy: Energy):
        super().__init__(energy.group)
        self.energy = energy


def clip(x: Tensor, clip_norm: Optional[float]) -> Tensor:
    """
    Arguments:
        x: [bsize, dim]
    """
    assert x.ndim == 2
    if clip_norm is None or clip_norm <= 0:
        return x
    norm = torch.norm(x, dim=1, p=2, keepdim=True)
    clip_mask = (norm > clip_norm).float()
    multiplier = clip_mask * (clip_norm / (norm + 1e-8)) + (1 - clip_mask)
    return x * multiplier


class EnergyKineticLangevinSampler(GroupEnergySampler):
    def __init__(
        self,
        energy: Energy,
        temperature: float,
        step_size: float,
        steps: int,
        friction: float,
        clip_norm: Optional[float],
        num_hypothesis: int,
        init_scale: float,
        dtype: str
    ):
        super().__init__(energy)
        self.temperature = temperature
        self.step_size = step_size
        self.steps = steps
        self.clip_norm = clip_norm
        self.num_hp = num_hypothesis
        self.init_scale = init_scale
        self.dtype = torch.float64 if dtype == "float64" else torch.float32

        self.momentum_scale = math.exp(-friction * step_size)
        self.grad_scale = (1 - math.exp(-friction * step_size)) / friction
        self.noise_scale = math.sqrt(1 - math.exp(-2 * friction * step_size))

    @staticmethod
    def _repeat_and_flatten(x: Tensor, repeats: int) -> Tensor:
        bsize = x.shape[0]
        if repeats == 1:
            return x.clone()
        return x[:, None].expand(bsize, repeats, *x.shape[1:]).flatten(0, 1)
    
    @staticmethod
    def _deflatten(x: Tensor, bsize: int, repeats: int) -> Tensor:
        return x.view(bsize, repeats, *x.shape[1:])

    @torch.enable_grad()
    @torch.inference_mode(False)
    def trivialized_grad(self, g: Tensor, x: Tensor, y: Optional[Tensor]) -> Tensor:
        """
        Compute trivialized gradient
        Args:
            g: [bsize, ...]
            x: [bsize, ...]
            y: [bsize, ...] (optional)
        """
        bsize = g.shape[0]

        # prepare trivialized gradient computation
        zeros = torch.zeros(bsize, self.group.num_generators, device=g.device, dtype=self.dtype)
        zeros.requires_grad_(True)
        id_ = self.group.exp(zeros)
        g_ = self.group.compose(g.clone(), id_.clone())

        # compute energy
        energy = self.energy(self.group.inverse(g_).float(), x, y) / self.temperature

        # compute trivialized gradient
        grad, = torch.autograd.grad(energy.sum(), zeros, create_graph=self.training)
        return grad

    def forward(self, x: Tensor, y: Optional[Tensor]) -> Tensor:
        """
        Sample group elements based on the input tensors.
        Algorithm 1 of https://proceedings.mlr.press/v247/kong24a/kong24a.pdf
        Arguments:
            x: [bsize, ...]
            y: [bsize, ...] (optional)
        Return:
            g: [bsize, ...]
        """
        bsize = x.shape[0]

        g = self.group.exp(self.group.random_coeff(bsize * self.num_hp).to(self.dtype) * self.init_scale)
        m = torch.zeros(bsize * self.num_hp, self.group.num_generators, device=g.device, dtype=self.dtype)

        x = self._repeat_and_flatten(x, self.num_hp)
        y = self._repeat_and_flatten(y, self.num_hp) if y is not None else None

        energy = torch.empty(bsize * self.num_hp, device=x.device)
        best_energy = torch.zeros(bsize * self.num_hp, device=x.device) + float("inf")
        best_g = torch.empty(g.shape, device=g.device, dtype=self.dtype)

        for _ in tqdm.tqdm(range(self.steps), bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}'):
            dg = self.group.exp(self.step_size * m)

            grad = self.trivialized_grad(g, x, y)

            m = self.momentum_scale * m \
                - self.grad_scale * grad \
                + self.noise_scale * torch.randn_like(m)

            g = self.group.compose(g, dg)

            energy = self.energy(self.group.inverse(g).float(), x, y)

            is_best = energy < best_energy
            best_energy[is_best] = energy[is_best]
            best_g[is_best] = g[is_best]

        best_energy = self._deflatten(best_energy, bsize, self.num_hp)
        best_g = self._deflatten(best_g, bsize, self.num_hp)

        hypothesis_index = best_energy.argmin(dim=1)
        best_g = best_g[torch.arange(bsize), hypothesis_index]

        return best_g.float()


class EnergyDiffusionSampler(GroupEnergySampler):
    timesteps: Tensor
    gamma: Tensor

    def __init__(
        self,
        energy: Energy,
        temperature: float,
        steps: int,
        noise_min: float,
        noise_max: float,
        clip_norm: Optional[float],
        num_mc: int,
        num_hypothesis: int,
        dtype: str,
        verbose: bool,
        temperature_start: float = 1.0,
        temperature_end: float = 1.0,
        use_antithetic: bool = False
    ):
        super().__init__(energy)
        self.temperature = temperature
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end
        self.use_antithetic = use_antithetic
        self.steps = steps
        self.step_size = 1 / steps
        self.num_mc = num_mc
        self.num_hp = num_hypothesis
        self.clip_norm = clip_norm
        self.dtype = torch.float64 if dtype == "float64" else torch.float32
        self.verbose = verbose

        # need correction?
        self.adjoint_trace_is_zero = (self.group.adjoint_trace == 0.).all()

        # diffusion parameters
        # gamma: noise_min at t = 0, noise_max at t = 1
        self.register_buffer('timesteps', torch.linspace(0, 1, steps))
        self.register_buffer('gamma', noise_min * ((noise_max / noise_min) ** self.timesteps))

        # Temperature annealing schedule: geometric from start to end
        if temperature_start != temperature_end:
            temp_sched = temperature_start * ((temperature_end / temperature_start) ** self.timesteps)
        else:
            temp_sched = torch.full_like(self.timesteps, self.temperature)
        self.register_buffer('temperature_schedule', temp_sched)

        self.drift_scale = self.step_size
        self.diffusion_scale = self.step_size ** 0.5

    @staticmethod
    def _repeat_and_flatten(x: Tensor, repeats: int) -> Tensor:
        bsize = x.shape[0]
        if repeats == 1:
            return x.clone()
        return x[:, None].expand(bsize, repeats, *x.shape[1:]).flatten(0, 1)

    @staticmethod
    def _deflatten(x: Tensor, bsize: int, repeats: int) -> Tensor:
        return x.view(bsize, repeats, *x.shape[1:])

    @torch.no_grad()
    def forward_diffusion(self, bsize: int, t: int, antithetic: bool = False) -> Tuple[Tensor, Tensor]:
        """
        Forward diffusion process
        Args:
            bsize: number of samples
            t: int in {1, ..., steps}
            antithetic: if True and bsize is even, use antithetic pairs
        Return:
            w: [bsize, ...]
            log_modular_w: [bsize,]
        """
        # steps in Lie algebra
        if antithetic and bsize % 2 == 0:
            half = bsize // 2
            z = self.group.random_coeff(half * t).to(self.dtype)
            dw_coeffs = torch.cat([z, -z], dim=0)  # antithetic pairs
        else:
            dw_coeffs = self.group.random_coeff(bsize * t).to(self.dtype)
        dw_coeffs = self._deflatten(dw_coeffs, bsize, t)
        dw_coeffs *= self.gamma[None, :t].view(1, t, *([1] * (dw_coeffs.ndim - 2)))
        dw_coeffs *= self.diffusion_scale

        # steps in group
        dws = self.group.exp(dw_coeffs.flatten(0, 1))
        dws = self._deflatten(dws, bsize, t)

        # compose steps
        w = dws[:, 0]
        for step in range(1, t):
            w = self.group.compose(w, dws[:, step])

        # log modular function
        if self.adjoint_trace_is_zero:
            log_modular_w = torch.zeros(bsize, device=self.group.device)
        else:
            log_modular_w = torch.einsum(
                'bti,i->bt', -dw_coeffs.float(), self.group.adjoint_trace).cumsum(1)[:, -1]

        return w, log_modular_w

    @torch.enable_grad()
    @torch.inference_mode(False)
    def estimate_score(self, g: Tensor, x: Tensor, y: Optional[Tensor], t: int) -> Tensor:
        """
        Estimate trivialized score at given timesteps
        Args:
            g: [bsize, ...]
            x: [bsize, ...]
            y: [bsize, ...] (optional)
            t: int in {1, ..., steps}
        """
        bsize = g.shape[0]

        # forward diffusion
        w, log_modular_w = self.forward_diffusion(self.num_mc, t, self.use_antithetic)

        # prepare trivialized gradient computation
        zeros = torch.zeros(bsize, self.group.num_generators, device=g.device)
        zeros.requires_grad_(True)
        id_ = self._repeat_and_flatten(self.group.exp(zeros), self.num_mc)
        g_ = self._repeat_and_flatten(g, self.num_mc)
        g_ = self.group.compose(g_, id_)

        # evaluation points
        w_ = w[None].expand(bsize, self.num_mc, *w.shape[1:]).flatten(0, 1)
        w_g_inv = self.group.compose(w_, self.group.inverse(g_))

        # compute energy
        x_ = self._repeat_and_flatten(x, self.num_mc)
        y_ = self._repeat_and_flatten(y, self.num_mc) if y is not None else None
        energy = self.energy(w_g_inv, x_, y_) / self.temperature_schedule[self.steps - t] #- self.group.log_det_jacobian(w_g_inv, x_)
        energy = energy.view(bsize, self.num_mc)

        # logits
        logits = - energy - log_modular_w[None].expand(bsize, self.num_mc)

        # estimate scores
        logsumexp = torch.logsumexp(logits, dim=1)

        score, = torch.autograd.grad(logsumexp.sum(), zeros, create_graph=self.training)
        return score

    def init(self, bsize: int) -> Tensor:
        """
        Initialize group elements using forward process at the last timestep
        """
        w, _ = self.forward_diffusion(bsize, self.steps)
        return w

    def forward(self, x: Tensor, y: Optional[Tensor]) -> Tensor:
        """
        Sample group elements based on the input tensors.
        Arguments:
            x: [bsize, ...]
            y: [bsize, ...] (optional)
        Return:
            g: [bsize, ...]
        """
        bsize = x.shape[0]

        g = self.init(bsize * self.num_hp).to(self.dtype)
        x = self._repeat_and_flatten(x, self.num_hp)
        y = self._repeat_and_flatten(y, self.num_hp) if y is not None else None

        energy = torch.empty(bsize * self.num_hp, device=x.device)
        best_energy = torch.zeros(bsize * self.num_hp, device=x.device) + float("inf")
        best_g = torch.empty(g.shape, dtype=self.dtype, device=x.device)

        for i in range(self.steps):
            t = self.steps - i

            score = self.estimate_score(g, x, y, t)
            clipped_score = clip(score, self.clip_norm)

            noise = self.group.random_coeff(bsize * self.num_hp).to(self.dtype)

            step = self.gamma[t - 1].pow(2) * self.drift_scale * clipped_score \
                + self.gamma[t - 1] * self.diffusion_scale * noise

            dg = self.group.exp(step.to(self.dtype))
            g = self.group.compose(g, dg)

            energy = self.energy(self.group.inverse(g), x, y)

            is_best = energy < best_energy
            best_energy[is_best] = energy[is_best]
            best_g[is_best] = g[is_best]

            if self.verbose:
                energy_ = self._deflatten(energy, bsize, self.num_hp)[:, -1][0]
                norm_noclip = torch.norm(score, dim=1, p=2, keepdim=True)

                print(f"\nstep {i + 1} / {self.steps}")

                print(f"energy mean {energy_.mean().item():.4e} " +
                      f"median {energy_.median().item():.4e} " +
                      f"max {energy_.max().item():.4e} " +
                      f"min {energy_.min().item():.4e}")

                print(f"norm mean {norm_noclip.mean().item():.2e} " +
                      f"median {norm_noclip.median().item():.2e} " +
                      f"max {norm_noclip.max().item():.2e} " +
                      f"min {norm_noclip.min().item():.2e} " +
                      f"zeros {(norm_noclip == 0.).sum().item()} ")

        best_energy = self._deflatten(best_energy, bsize, self.num_hp)
        best_g = self._deflatten(best_g, bsize, self.num_hp)

        hypothesis_index = best_energy.argmin(dim=1)
        best_g = best_g[torch.arange(bsize), hypothesis_index]

        return best_g.float()


class EnergyDiffusionSamplerPDE(EnergyDiffusionSampler):

    @torch.enable_grad()
    @torch.inference_mode(False)
    def estimate_score(self, g: Tensor, x: Tuple[Tensor, Tensor], y: Optional[Tensor], t: int) -> Tensor:  # type: ignore
        """
        Estimate trivialized score at given timesteps
        Args:
            g: [bsize, ...]
            x: tuple([bsize, ...], [bsize, ...])
            y: [bsize, ...] (optional)
            t: int in {1, ..., steps}
        """
        bsize = g.shape[0]

        # forward diffusion
        w, log_modular_w = self.forward_diffusion(self.num_mc, t, self.use_antithetic)

        # prepare trivialized gradient computation
        zeros = torch.zeros(bsize, self.group.num_generators, device=g.device, dtype=self.dtype)
        zeros.requires_grad_(True)
        id_ = self._repeat_and_flatten(self.group.exp(zeros), self.num_mc)
        g_ = self._repeat_and_flatten(g, self.num_mc)
        g_ = self.group.compose(g_, id_)

        # evaluation points
        w = w[None].expand(bsize, self.num_mc, *w.shape[1:]).flatten(0, 1).to(self.dtype)
        w_g_inv = self.group.compose(w, self.group.inverse(g_))

        # compute energy
        jet_0, X_f = x
        jet_0_ = self._repeat_and_flatten(jet_0, self.num_mc)
        X_f_ = self._repeat_and_flatten(X_f, self.num_mc)
        x_ = (jet_0_, X_f_)
        y_ = self._repeat_and_flatten(y, self.num_mc) if y is not None else None
        energy = self.energy(w_g_inv, x_, y_).view(bsize, self.num_mc) / self.temperature

        # logits
        logits = - energy - log_modular_w[None].expand(bsize, self.num_mc)

        # estimate scores
        logsumexp = torch.logsumexp(logits, dim=1)

        score, = torch.autograd.grad(logsumexp.sum(), zeros, create_graph=self.training)
        return score

    def forward(self, x: Tuple[Tensor, Tensor], y: Optional[Tensor]) -> Tensor:  # type: ignore
        """
        Sample group elements based on the input tensors.
        Arguments:
            x: tuple([bsize, ...], [bsize, ...])
            y: [bsize, ...] (optional)
        Return:
            g: [bsize, ...]
        """
        jet_0, X_f = x

        bsize = jet_0.shape[0]
        device = jet_0.device

        g = self.init(bsize * self.num_hp).to(self.dtype)
        jet_0 = self._repeat_and_flatten(jet_0, self.num_hp)
        X_f = self._repeat_and_flatten(X_f, self.num_hp)
        x = (jet_0, X_f)
        y = self._repeat_and_flatten(y, self.num_hp) if y is not None else None

        energy = torch.empty(bsize * self.num_hp, dtype=self.dtype, device=device)
        best_energy = torch.zeros(bsize * self.num_hp, dtype=self.dtype, device=device) + float("inf")
        best_g = torch.empty(g.shape, dtype=self.dtype, device=g.device)

        for i in range(self.steps):
            t = self.steps - i

            score = self.estimate_score(g, x, y, t)
            clipped_score = clip(score, self.clip_norm)

            noise = self.group.random_coeff(bsize * self.num_hp).to(self.dtype)

            step = self.gamma[t - 1].pow(2) * self.drift_scale * clipped_score \
                + self.gamma[t - 1] * self.diffusion_scale * noise

            dg = self.group.exp(step.to(self.dtype))
            g = self.group.compose(g, dg)

            energy = self.energy(self.group.inverse(g), x, y)

            is_best = energy < best_energy
            best_energy[is_best] = energy[is_best]
            best_g[is_best] = g[is_best]

            if self.verbose:
                energy_ = self._deflatten(best_energy, bsize, self.num_hp).min(dim=1)[0]
                norm_noclip = torch.norm(score, dim=1, p=2, keepdim=True)

                print(f"\nstep {i + 1} / {self.steps}")

                print(f"energy mean {energy_.nanmean().item():.4e} " +
                    f"median {energy_.nanmedian().item():.4e} " +
                    f"max {energy_.max().item():.4e} " +
                    f"min {energy_.min().item():.4e}")

                print(f"norm mean {norm_noclip.nanmean().item():.2e} " +
                    f"median {norm_noclip.nanmedian().item():.2e} " +
                    f"max {norm_noclip.max().item():.2e} " +
                    f"min {norm_noclip.min().item():.2e} " +
                    f"zeros {(norm_noclip == 0.).sum().item()} ")

        best_energy = self._deflatten(best_energy, bsize, self.num_hp)
        best_g = self._deflatten(best_g, bsize, self.num_hp)

        hypothesis_index = best_energy.argmin(dim=1)
        best_g = best_g[torch.arange(bsize), hypothesis_index]

        return best_g


class EnergyKineticLangevinSamplerPDE(EnergyKineticLangevinSampler):

    @torch.enable_grad()
    @torch.inference_mode(False)
    def trivialized_grad(self, g: Tensor, x: Tuple[Tensor, Tensor], y: Optional[Tensor]) -> Tensor:  # type: ignore
        """
        Compute trivialized gradient
        Args:
            g: [bsize, ...]
            x: [bsize, ...]
            y: [bsize, ...] (optional)
        """
        bsize = g.shape[0]

        # prepare trivialized gradient computation
        zeros = torch.zeros(bsize, self.group.num_generators, device=g.device, dtype=self.dtype)
        zeros.requires_grad_(True)
        id_ = self.group.exp(zeros)
        g_ = self.group.compose(g.clone(), id_.clone())

        # compute energy
        energy = self.energy(self.group.inverse(g_), x, y) / self.temperature

        # compute trivialized gradient
        grad, = torch.autograd.grad(energy.sum(), zeros, create_graph=self.training)
        return grad

    def forward(self, x: Tuple[Tensor, Tensor], y: Optional[Tensor]) -> Tensor:  # type: ignore
        """
        Sample group elements based on the input tensors.
        Algorithm 1 of https://proceedings.mlr.press/v247/kong24a/kong24a.pdf
        Arguments:
            x: [bsize, ...]
            y: [bsize, ...] (optional)
        Return:
            g: [bsize, ...]
        """
        jet_0, X_f = x

        bsize = jet_0.shape[0]
        device = jet_0.device

        g = self.group.exp(self.group.random_coeff(bsize * self.num_hp).to(self.dtype) * self.init_scale)
        m = torch.zeros(bsize * self.num_hp, self.group.num_generators, device=g.device, dtype=self.dtype)

        jet_0 = self._repeat_and_flatten(jet_0, self.num_hp)
        X_f = self._repeat_and_flatten(X_f, self.num_hp)
        x = (jet_0, X_f)
        y = self._repeat_and_flatten(y, self.num_hp) if y is not None else None

        energy = torch.empty(bsize * self.num_hp, device=device)
        best_energy = torch.zeros(bsize * self.num_hp, device=device) + float("inf")
        best_g = torch.empty(g.shape, dtype=self.dtype, device=g.device)

        for _ in tqdm.tqdm(range(self.steps), bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}'):
            dg = self.group.exp(self.step_size * m)

            grad = self.trivialized_grad(g, x, y)

            m = self.momentum_scale * m \
                - self.grad_scale * grad \
                + self.noise_scale * torch.randn_like(m)

            g = self.group.compose(g, dg)

            energy = self.energy(self.group.inverse(g), x, y).float()

            is_best = energy < best_energy
            best_energy[is_best] = energy[is_best]
            best_g[is_best] = g[is_best]

        best_energy = self._deflatten(best_energy, bsize, self.num_hp)
        best_g = self._deflatten(best_g, bsize, self.num_hp)

        hypothesis_index = best_energy.argmin(dim=1)
        best_g = best_g[torch.arange(bsize), hypothesis_index]

        return best_g
